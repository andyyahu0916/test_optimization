#!/usr/bin/env python3
"""
============================================================================
Fixed-Voltage MD Simulation using ConstantVoltage Plugin
============================================================================
Based on OpenMM-ConstantV(original)/run_openMM.py

This version uses:
- ConstantVoltageForce: Stores electrode atom data
- ConstantVDrudeLangevinIntegrator: Handles SCF charge updates automatically

Key differences from original:
- SCF charge updates are handled by the integrator (no manual Poisson_solver)
- Electrode atoms are registered with ConstantVoltageForce
- The integrator automatically finds ConstantVoltageForce and performs SCF
============================================================================
"""

from __future__ import print_function
import sys
import os
import shutil
import numpy as np
from datetime import datetime

from openmm.app import *
from openmm import *
from openmm.unit import *

# Try to import constantvoltage plugin
# The plugin Python bindings are in the python directory
plugin_python_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python')
if os.path.exists(plugin_python_dir):
    sys.path.insert(0, plugin_python_dir)

try:
    import constantvoltage
    ConstantVoltageForce = constantvoltage.ConstantVoltageForce
    ConstantVDrudeLangevinIntegrator = constantvoltage.ConstantVDrudeLangevinIntegrator
    print("✓ Successfully imported constantvoltage plugin")
except ImportError as e:
    print(f"✗ Error importing constantvoltage plugin: {e}")
    print(f"  Plugin Python directory: {plugin_python_dir}")
    print("  Make sure the plugin is compiled and the Python bindings exist.")
    sys.exit(1)

# Physical Constants (matching original)
conversion_KjmolNm_Au = 0.00719475
small_threshold = 1e-6
VOLTAGE_TO_KJMOL = 96.485

# Simulation Parameters (matching original)
simulation_time_ns = 0.5
freq_charge_update_fs = 200
freq_traj_output_ps = 10
write_charges = False

# Output path
outPath = '1v_0.5ns_plugin'
if os.path.exists(outPath):
    shutil.rmtree(outPath)
os.mkdir(outPath)

# Applied voltage
Voltage = 0.0  # in Volts

# Electrode chain indices (matching original)
# cathode_index = (0, 2) means:
#   - Chain 0: Virtual layer (grpc) - participates in SCF
#   - Chain 2: Real layer (grph) - only used for exclusions
cathode_virtual_chain = 0
cathode_real_chain = 2
anode_virtual_chain = 1
anode_real_chain = 3
exclude_elements = ('H',)

# Force field directory
ffdir = '/home/andy/test_optimization/OpenMM-ConstantV(original)/ffdir/'

# ============================================================================
# Helper Functions
# ============================================================================
def get_chain_atoms(topology, chain_index, exclude_element=('H',)):
    """Get all non-H atoms from a chain"""
    atoms = []
    for chain in topology.chains():
        if chain.index == chain_index:
            for atom in chain.atoms():
                if atom.element is not None and atom.element.symbol not in exclude_element:
                    atoms.append(atom.index)
    return atoms

def add_exclusions_between(list1, list2, customNonbondedForce, nbondedForce):
    """Add exclusions between two atom lists (matching original)"""
    existing = set()
    for i in range(customNonbondedForce.getNumExclusions()):
        p1, p2 = customNonbondedForce.getExclusionParticles(i)
        existing.add((min(p1, p2), max(p1, p2)))
    
    added = 0
    if list1 == list2:
        # Same list - add exclusions within
        for i in range(len(list1)):
            for j in range(i+1, len(list1)):
                pair = (min(list1[i], list1[j]), max(list1[i], list1[j]))
                if pair not in existing:
                    customNonbondedForce.addExclusion(pair[0], pair[1])
                    nbondedForce.addException(pair[0], pair[1], 0, 1, 0, True)
                    existing.add(pair)
                    added += 1
    else:
        # Different lists - add exclusions between
        for i in list1:
            for j in list2:
                pair = (min(i, j), max(i, j))
                if pair not in existing:
                    customNonbondedForce.addExclusion(pair[0], pair[1])
                    nbondedForce.addException(pair[0], pair[1], 0, 1, 0, True)
                    existing.add(pair)
                    added += 1
    return added

# ============================================================================
# System Setup
# ============================================================================
print("=" * 70)
print("Fixed-Voltage MD Simulation (Using ConstantVoltage Plugin)")
print("=" * 70)

# Load force field files
residue_xml_list = [ffdir + 'sapt_residues.xml', ffdir + 'graph_residue_c.xml', ffdir + 'graph_residue_n.xml']
ff_xml_list = [ffdir + 'sapt_noDB_2sheets.xml', ffdir + 'graph_c_freeze.xml', ffdir + 'graph_n_freeze.xml']

for rf in residue_xml_list:
    Topology().loadBondDefinitions(rf)

# Load PDB and create system
pdb = PDBFile('/home/andy/test_optimization/OpenMM-ConstantV(original)/nvt_0V_15ns.pdb')
modeller = Modeller(pdb.topology, pdb.positions)
forcefield = ForceField(*ff_xml_list)
modeller.addExtraParticles(forcefield)

system = forcefield.createSystem(modeller.topology, nonbondedCutoff=1.2*nanometer, constraints=HBonds, rigidWater=True)

# Set force groups
for i in range(system.getNumForces()):
    f = system.getForce(i)
    f.setForceGroup(i)
    if isinstance(f, (HarmonicBondForce, HarmonicAngleForce, PeriodicTorsionForce, RBTorsionForce)):
        f.setUsesPeriodicBoundaryConditions(True)

# Get NonbondedForce and CustomNonbondedForce
nbondedForce = [f for f in [system.getForce(i) for i in range(system.getNumForces())] if isinstance(f, NonbondedForce)][0]
nbondedForce.setNonbondedMethod(NonbondedForce.PME)

customNonbondedForce = [f for f in [system.getForce(i) for i in range(system.getNumForces())] if isinstance(f, CustomNonbondedForce)][0]
customNonbondedForce.setNonbondedMethod(NonbondedForce.CutoffPeriodic)

# ============================================================================
# Initialize Electrodes and Create ConstantVoltageForce
# ============================================================================
print("\nInitializing electrodes...")

# Get electrode atom indices
cathode_virtual = get_chain_atoms(modeller.topology, cathode_virtual_chain, exclude_elements)
cathode_real = get_chain_atoms(modeller.topology, cathode_real_chain, exclude_elements)
anode_virtual = get_chain_atoms(modeller.topology, anode_virtual_chain, exclude_elements)
anode_real = get_chain_atoms(modeller.topology, anode_real_chain, exclude_elements)

print(f"  Cathode: {len(cathode_virtual)} virtual + {len(cathode_real)} real atoms")
print(f"  Anode: {len(anode_virtual)} virtual + {len(anode_real)} real atoms")

# Calculate geometry
positions = modeller.positions
boxVecs = modeller.topology.getPeriodicBoxVectors()
crossBox = np.cross([boxVecs[0][0]._value, boxVecs[0][1]._value, boxVecs[0][2]._value],
                    [boxVecs[1][0]._value, boxVecs[1][1]._value, boxVecs[1][2]._value])
total_area = np.linalg.norm(crossBox)

# Calculate z positions
cathode_z = np.mean([positions[i][2]._value for i in cathode_virtual])
anode_z = np.mean([positions[i][2]._value for i in anode_virtual])

Lcell = abs(anode_z - cathode_z)
box_z = boxVecs[2][2]._value
Lgap = box_z - Lcell

# Calculate area per atom (based on virtual layer)
area_atom = total_area / len(cathode_virtual)

print(f"  Lcell: {Lcell:.3f} nm, Lgap: {Lgap:.3f} nm")
print(f"  Total area: {total_area:.3f} nm², area/atom: {area_atom:.5f} nm²")
print(f"  Cathode z: {cathode_z:.3f} nm, Anode z: {anode_z:.3f} nm")

# Create ConstantVoltageForce
cv_force = ConstantVoltageForce()
cv_force.setVoltage(Voltage)  # in Volts
cv_force.setLgap(Lgap)       # in nm
cv_force.setLcell(Lcell)     # in nm
cv_force.setElectrodeZPositions(cathode_z, anode_z)  # Set both z positions at once
cv_force.setTotalArea(total_area)
cv_force.setSmallThreshold(small_threshold)

# Set SCF parameters (matching original: 4 iterations, every 200 fs)
cv_force.setNumSCFIterations(4)
cv_force.setSCFFrequency(int(freq_charge_update_fs))  # Update every 200 steps

# Add cathode atoms (virtual layer only - these participate in SCF)
for atom_idx in cathode_virtual:
    cv_force.addCathodeAtom(atom_idx, area_atom)

# Add anode atoms (virtual layer only)
for atom_idx in anode_virtual:
    cv_force.addAnodeAtom(atom_idx, area_atom)

# Identify electrolyte atoms (non-electrode)
all_electrode = set(cathode_virtual + cathode_real + anode_virtual + anode_real)
electrolyte_atoms = []
for residue in modeller.topology.residues():
    if len(list(residue.atoms())) < 100:
        for atom in residue.atoms():
            if atom.index not in all_electrode:
                electrolyte_atoms.append(atom.index)

# Add electrolyte atoms to ConstantVoltageForce
for atom_idx in electrolyte_atoms:
    cv_force.addElectrolyteAtom(atom_idx)

print(f"  Electrolyte atoms: {len(electrolyte_atoms)}")
print(f"  Added {cv_force.getNumCathodeAtoms()} cathode atoms to ConstantVoltageForce")
print(f"  Added {cv_force.getNumAnodeAtoms()} anode atoms to ConstantVoltageForce")
print(f"  Added {cv_force.getNumElectrolyteAtoms()} electrolyte atoms to ConstantVoltageForce")

# Add ConstantVoltageForce to system
# NOTE: This requires the Python bindings to be recompiled with the updated SWIG interface
# that includes the asForce() method. See FIX_SWIG_BINDING.md for instructions.

try:
    # Try direct add first (in case SWIG inheritance works)
    system.addForce(cv_force)
    print("✓ Added ConstantVoltageForce to System")
except TypeError:
    # Use the asForce() casting method (added in constantvoltage.i via %extend)
    if hasattr(cv_force, 'asForce'):
        force_ptr = cv_force.asForce()
        system.addForce(force_ptr)
        print("✓ Added ConstantVoltageForce using asForce() cast")
    else:
        print("\n" + "="*70)
        print("ERROR: Cannot add ConstantVoltageForce to System")
        print("="*70)
        print("The Python bindings need to be recompiled with the updated SWIG interface.")
        print("\nTo fix this, run:")
        print("  cd /home/andy/test_optimization/openmm-8.4.0/plugins/constantvoltage/python")
        print("  swig -python -c++ -I../../openmmapi/include constantvoltage.i")
        print("  # Then compile the generated _constantvoltage.so")
        print("\nSee FIX_SWIG_BINDING.md for detailed instructions.")
        print("="*70)
        raise RuntimeError(
            "Cannot add ConstantVoltageForce: Python bindings need to be recompiled.\n"
            "The asForce() method is not available in the current bindings."
        )

# ============================================================================
# Generate Electrode Exclusions (matching original exactly)
# ============================================================================
print("\nGenerating electrode exclusions...")

# Exclusions within primary electrode sheets (virtual-virtual)
n1 = add_exclusions_between(cathode_virtual, cathode_virtual, customNonbondedForce, nbondedForce)
n2 = add_exclusions_between(anode_virtual, anode_virtual, customNonbondedForce, nbondedForce)

# Exclusions between virtual and real layers
n3 = add_exclusions_between(cathode_virtual, cathode_real, customNonbondedForce, nbondedForce)
n4 = add_exclusions_between(anode_virtual, anode_real, customNonbondedForce, nbondedForce)

# Exclusions within real layers
n5 = add_exclusions_between(cathode_real, cathode_real, customNonbondedForce, nbondedForce)
n6 = add_exclusions_between(anode_real, anode_real, customNonbondedForce, nbondedForce)

print(f"  Added exclusions: virtual-virtual={n1+n2}, virtual-real={n3+n4}, real-real={n5+n6}")

# ============================================================================
# Initialize Charges on Virtual Layers
# ============================================================================
print("\nInitializing electrode charges...")

sign_cathode = 1.0
sign_anode = -1.0
flag_small = abs(Voltage) < 0.01

if flag_small:
    print(f"  Adding small value to initial charges (Voltage={Voltage} V)...")

# Initialize cathode charges
for atom_idx in cathode_virtual:
    q_i = sign_cathode / (4.0 * np.pi) * area_atom * (Voltage * VOLTAGE_TO_KJMOL / Lgap + Voltage * VOLTAGE_TO_KJMOL / Lcell) * conversion_KjmolNm_Au
    if flag_small:
        q_i = q_i + sign_cathode * small_threshold
    if abs(q_i) < small_threshold:
        q_i = small_threshold
    nbondedForce.setParticleParameters(atom_idx, q_i, 1.0, 0.0)

# Initialize anode charges
for atom_idx in anode_virtual:
    q_i = sign_anode / (4.0 * np.pi) * area_atom * (Voltage * VOLTAGE_TO_KJMOL / Lgap + Voltage * VOLTAGE_TO_KJMOL / Lcell) * conversion_KjmolNm_Au
    if flag_small:
        q_i = q_i + sign_anode * small_threshold
    if abs(q_i) < small_threshold:
        q_i = -small_threshold
    nbondedForce.setParticleParameters(atom_idx, q_i, 1.0, 0.0)

# ============================================================================
# Create Integrator and Simulation
# ============================================================================
print("\nCreating integrator and simulation...")

# Check for Drude force
has_drude = any(isinstance(system.getForce(i), DrudeForce) for i in range(system.getNumForces()))

if has_drude:
    # Use ConstantVDrudeLangevinIntegrator (handles SCF automatically)
    integrator = ConstantVDrudeLangevinIntegrator(
        300.0,      # temperature (K)
        1.0,        # friction (1/ps)
        1.0,        # drudeTemperature (K)
        40.0,       # drudeFriction (1/ps)
        0.001       # stepSize (ps)
    )
    integrator.setMaxDrudeDistance(0.02)  # 0.02 nm
    print("  Using ConstantVDrudeLangevinIntegrator (with automatic SCF)")
else:
    # Fallback to regular Langevin integrator (but SCF won't work without Drude)
    print("  WARNING: No DrudeForce found. SCF charge updates require DrudeForce.")
    integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.001*picosecond)

platform = Platform.getPlatformByName('CUDA')
simulation = Simulation(modeller.topology, system, integrator, platform, {'Precision': 'mixed'})
simulation.context.setPositions(modeller.positions)

# Reinitialize context with exclusions, then restore charges
print("Reinitializing context with exclusions...")
state = simulation.context.getState(getPositions=True)
positions_snapshot = state.getPositions()
simulation.context.reinitialize()
simulation.context.setPositions(positions_snapshot)
nbondedForce.updateParametersInContext(simulation.context)

# Print initial energies
state = simulation.context.getState(getEnergy=True)
print(f"\nInitial energies:")
print(f"  KE: {state.getKineticEnergy()}")
print(f"  PE: {state.getPotentialEnergy()}")
for j in range(system.getNumForces()):
    f = system.getForce(j)
    print(f"  {type(f).__name__}: {simulation.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()}")

# Setup output
PDBFile.writeFile(simulation.topology, positions_snapshot, open(os.path.join(outPath, 'start_drudes.pdb'), 'w'))
simulation.reporters.append(DCDReporter(os.path.join(outPath, 'FV_NVT.dcd'), int(freq_traj_output_ps * 1000)))

# ============================================================================
# Run Simulation
# ============================================================================
print("\n" + "=" * 70)
print("Starting simulation...")
print("=" * 70)
print(f"  Simulation time: {simulation_time_ns} ns")
print(f"  Charge update frequency: {freq_charge_update_fs} fs")
print(f"  Trajectory output frequency: {freq_traj_output_ps} ps")
print(f"  SCF iterations per update: {cv_force.getNumSCFIterations()}")
print(f"  SCF update frequency: {cv_force.getSCFFrequency()} steps")
print("=" * 70)

num_iterations = int(simulation_time_ns * 1000 / freq_traj_output_ps)
steps_per_output = int(freq_traj_output_ps * 1000)  # steps per output

for i in range(num_iterations):
    state = simulation.context.getState(getEnergy=True)
    print(f"\n{i} iteration: KE={state.getKineticEnergy()}, PE={state.getPotentialEnergy()}")
    
    # The integrator automatically handles SCF charge updates
    # No need to manually call Poisson_solver_fixed_voltage()
    simulation.step(steps_per_output)
    
    if write_charges:
        # TODO: Implement charge output if needed
        pass

print("\ndone!")

