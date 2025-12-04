#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════
Native ConstantV Core Integration - MD Simulation Launcher
═══════════════════════════════════════════════════════════════════════════

This script is the EQUIVALENT of OpenMM-ConstantV(original)/run_openMM.py
but uses the new Native Core Integration instead of the Python plugin.

Key Differences:
----------------
1. Uses ConstantVDrudeLangevinIntegrator (native C++/CUDA) instead of
   Python-based Poisson_solver_fixed_voltage()
2. Uses ConstantVSystemBuilder (factory pattern) instead of MM class
3. SCF iterations are handled internally by the integrator

Workflow Alignment:
-------------------
Original:  Poisson_solver_fixed_voltage(Niterations=4) → simmd.step(freq_charge_update_fs)
New:       integrator.step(freq_charge_update_fs)  (SCF runs internally)

Author: Production Engineering System
License: See OpenMM license
"""

from __future__ import print_function
import sys
import os
from pathlib import Path
from datetime import datetime
import shutil

# ═══════════════════════════════════════════════════════════════════════════
# Import OpenMM and ConstantV Native Integration
# ═══════════════════════════════════════════════════════════════════════════

try:
    import constantv  # Native core integration module
except ImportError as e:
    print("ERROR: constantv module not found!")
    print("Please build and install the native integration:")
    print("  cd openmm_core_integration/build")
    print("  make install")
    sys.exit(1)

from openmm import app
from openmm import unit
from openmm import Platform
from openmm.app import DCDReporter, PDBFile

# ═══════════════════════════════════════════════════════════════════════════
# Simulation Parameters (Matching Original run_openMM.py)
# ═══════════════════════════════════════════════════════════════════════════

# Run control settings (matching original L34)
simulation_time_ns = 0.5
freq_charge_update_fs = 200  # Charge update frequency (fs)
freq_traj_output_ps = 10     # Trajectory output frequency (ps)
write_charges = False        # WARNING: True will generate a lot of data!

# Output path (matching original L37)
outPath = '1v_0.5ns'

if os.path.exists(outPath):
    shutil.rmtree(outPath)
os.mkdir(outPath)

# Charge file (if writing charges)
if write_charges:
    chargeFile = open(os.path.join(outPath, 'charges.dat'), 'w')

# ═══════════════════════════════════════════════════════════════════════════
# System Configuration (Matching Original run_openMM.py)
# ═══════════════════════════════════════════════════════════════════════════

# Force field directory (matching original L52)
ffdir = './ffdir/'

# Applied voltage in Volts (matching original L73)
Voltage = 0.0  # Volts (will be converted internally to kJ/mol)

# Electrode identification (matching original L78)
# Using chain indices (chain=True in original)
cathode_index = (0, 2)  # Chain indices start at 0
anode_index = (1, 3)

# Conductors (matching original L81-85)
# BuckyBalls = [1]  # Uncomment if you have Buckyballs
# NanoTubes = [(1, 4)]  # Uncomment if you have Nanotubes
# nanotube_axis = [(1.0, 0.0, 0.0)]  # Uncomment if you have Nanotubes

# ═══════════════════════════════════════════════════════════════════════════
# Build System Using Native Core Integration
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 75)
print("Building ConstantV System (Native Core Integration)")
print("=" * 75)

# Load PDB and force field files (matching original L89)
pdb_file = 'nvt_0V_15ns.pdb'
residue_xml_files = [
    os.path.join(ffdir, 'sapt_residues.xml'),
    os.path.join(ffdir, 'graph_residue_c.xml'),
    os.path.join(ffdir, 'graph_residue_n.xml'),
]
forcefield_xml_files = [
    os.path.join(ffdir, 'sapt_noDB_2sheets.xml'),
    os.path.join(ffdir, 'graph_c_freeze.xml'),
    os.path.join(ffdir, 'graph_n_freeze.xml'),
]

# Load bond definitions (matching original L66-67)
for residue_file in residue_xml_files:
    if os.path.exists(residue_file):
        app.Topology().loadBondDefinitions(residue_file)

# Load PDB
pdb = app.PDBFile(pdb_file)
modeller = app.Modeller(pdb.topology, pdb.positions)

# Load force field
forcefield = app.ForceField(*forcefield_xml_files)

# Add extra particles (for Drude oscillators, matching original L77)
modeller.addExtraParticles(forcefield)

# Create system (matching original L100)
system = forcefield.createSystem(
    modeller.topology,
    nonbondedCutoff=1.4 * unit.nanometer,  # matching original L49
    constraints=app.HBonds,
    rigidWater=True
)

# Get forces (matching original L102-107)
nonbonded_force = None
custom_nonbonded_force = None
drude_force = None

for i in range(system.getNumForces()):
    f = system.getForce(i)
    if isinstance(f, app.NonbondedForce):
        nonbonded_force = f
    elif isinstance(f, app.CustomNonbondedForce):
        custom_nonbonded_force = f
    elif isinstance(f, app.DrudeForce):
        drude_force = f

# Set PME (matching original L111-112)
if nonbonded_force:
    nonbonded_force.setNonbondedMethod(app.NonbondedForce.PME)
if custom_nonbonded_force:
    custom_nonbonded_force.setNonbondedMethod(
        min(nonbonded_force.getNonbondedMethod(), app.NonbondedForce.CutoffPeriodic)
    )

# Set periodic boundaries (matching original L121-131)
for i in range(system.getNumForces()):
    f = system.getForce(i)
    f.setForceGroup(i)
    if isinstance(f, (app.HarmonicBondForce, app.HarmonicAngleForce,
                      app.PeriodicTorsionForce, app.RBTorsionForce)):
        f.setUsesPeriodicBoundaryConditions(True)

# ═══════════════════════════════════════════════════════════════════════════
# Identify Electrode and Electrolyte Atoms
# ═══════════════════════════════════════════════════════════════════════════

print("\nIdentifying electrodes and electrolyte...")

# Identify cathode atoms (by chain index, matching original L108-109)
cathode_atoms = []
for chain in modeller.topology.chains():
    if chain.index in cathode_index:
        for atom in chain.atoms():
            if atom.element.symbol != 'H':  # exclude_element=("H",)
                cathode_atoms.append(atom.index)

# Identify anode atoms
anode_atoms = []
for chain in modeller.topology.chains():
    if chain.index in anode_index:
        for atom in chain.atoms():
            if atom.element.symbol != 'H':
                anode_atoms.append(atom.index)

# Identify electrolyte atoms (matching original L113)
# Residues with < 100 atoms are electrolyte
electrolyte_atoms = []
natom_cutoff = 100
for residue in modeller.topology.residues():
    num_atoms = sum(1 for _ in residue.atoms())
    if num_atoms < natom_cutoff:
        for atom in residue.atoms():
            electrolyte_atoms.append(atom.index)

print(f"  Cathode atoms: {len(cathode_atoms)}")
print(f"  Anode atoms: {len(anode_atoms)}")
print(f"  Electrolyte atoms: {len(electrolyte_atoms)}")

# ═══════════════════════════════════════════════════════════════════════════
# Compute Cell Geometry (Lgap, Lcell, electrode areas)
# ═══════════════════════════════════════════════════════════════════════════

print("\nComputing cell geometry...")

# Get positions
state = None  # Will be set after creating simulation
positions = modeller.positions

# Compute electrode Z positions
cathode_z = []
anode_z = []

for idx in cathode_atoms:
    if idx < len(positions):
        cathode_z.append(positions[idx][2].value_in_unit(unit.nanometer))

for idx in anode_atoms:
    if idx < len(positions):
        anode_z.append(positions[idx][2].value_in_unit(unit.nanometer))

z_cathode = sum(cathode_z) / len(cathode_z) if cathode_z else 0.0
z_anode = sum(anode_z) / len(anode_z) if anode_z else 0.0

# Compute Lgap and Lcell (approximate from box dimensions)
box_vectors = modeller.topology.getPeriodicBoxVectors()
if box_vectors:
    Lcell = box_vectors[2][2].value_in_unit(unit.nanometer)  # Z dimension
    Lgap = Lcell - (z_anode - z_cathode)  # Vacuum gap
else:
    # Fallback: estimate from electrode positions
    Lcell = abs(z_anode - z_cathode) * 2.0  # Approximate
    Lgap = abs(z_anode - z_cathode) * 0.3  # Approximate

# Compute electrode area (approximate from atom count)
# Assuming ~0.04 nm² per atom for graphene
area_per_atom = 0.04  # nm²
cathode_area = len(cathode_atoms) * area_per_atom
anode_area = len(anode_atoms) * area_per_atom
total_area = (cathode_area + anode_area) / 2.0  # Average

print(f"  Z_cathode: {z_cathode:.3f} nm")
print(f"  Z_anode: {z_anode:.3f} nm")
print(f"  Lgap: {Lgap:.3f} nm")
print(f"  Lcell: {Lcell:.3f} nm")
print(f"  Total area: {total_area:.3f} nm²")

# ═══════════════════════════════════════════════════════════════════════════
# Create ConstantVDrudeLangevinIntegrator
# ═══════════════════════════════════════════════════════════════════════════

print("\nCreating ConstantVDrudeLangevinIntegrator...")

# Check if system is polarizable (matching original L84-96)
is_polarizable = (drude_force is not None)

if is_polarizable:
    # Polarizable: use Drude integrator (matching original L91)
    temperature = 300.0 * unit.kelvin
    temperature_drude = 1.0 * unit.kelvin
    friction = 1.0 / unit.picosecond
    friction_drude = 1.0 / unit.picosecond
    timestep = 0.001 * unit.picoseconds

    integrator = constantv.ConstantVDrudeLangevinIntegrator(
        temperature=temperature.value_in_unit(unit.kelvin),
        frictionCoeff=friction.value_in_unit(1.0 / unit.picosecond),
        drudeTemperature=temperature_drude.value_in_unit(unit.kelvin),
        drudeFrictionCoeff=friction_drude.value_in_unit(1.0 / unit.picosecond),
        stepSize=timestep.value_in_unit(unit.picoseconds),
        voltage=Voltage,  # Volts (converted internally)
        Lgap=Lgap,        # nm
        Lcell=Lcell,      # nm
        scfIterations=4   # matching original Niterations=4
    )
    
    # Set max Drude distance (matching original L93)
    integrator.setMaxDrudeDistance(0.02 * unit.nanometer)
    
    # Set SCF frequency (matching original freq_charge_update_fs=200)
    # Original: Poisson_solver_fixed_voltage() called every 200 fs
    # New: SCF runs every scfFrequency steps
    # timestep = 0.001 ps = 1 fs, so 200 fs = 200 steps
    integrator.setSCFFrequency(int(freq_charge_update_fs / timestep.value_in_unit(unit.picoseconds) / 1000))
else:
    # Non-polarizable: use standard Langevin integrator
    print("WARNING: Non-polarizable system detected. Using standard LangevinIntegrator.")
    from openmm import LangevinIntegrator
    integrator = LangevinIntegrator(
        300.0 * unit.kelvin,
        1.0 / unit.picosecond,
        0.001 * unit.picoseconds
    )

# ═══════════════════════════════════════════════════════════════════════════
# Add Electrodes and Electrolyte to Integrator
# ═══════════════════════════════════════════════════════════════════════════

print("\nAdding electrodes and electrolyte to integrator...")

# Add cathode atoms
for idx in cathode_atoms:
    integrator.addCathodeAtom(idx, area_per_atom)

# Add anode atoms
for idx in anode_atoms:
    integrator.addAnodeAtom(idx, area_per_atom)

# Add electrolyte atoms
for idx in electrolyte_atoms:
    if idx < nonbonded_force.getNumParticles():
        charge, sigma, epsilon = nonbonded_force.getParticleParameters(idx)
        integrator.addElectrolyteAtom(idx, charge.value_in_unit(unit.elementary_charge))

# Set geometry parameters
integrator.setTotalArea(total_area)
integrator.setZCathode(z_cathode)
integrator.setZAnode(z_anode)

print(f"  Added {len(cathode_atoms)} cathode atoms")
print(f"  Added {len(anode_atoms)} anode atoms")
print(f"  Added {len(electrolyte_atoms)} electrolyte atoms")

# ═══════════════════════════════════════════════════════════════════════════
# Create Simulation
# ═══════════════════════════════════════════════════════════════════════════

print("\nCreating simulation...")

# Set platform (matching original L100)
try:
    platform = Platform.getPlatformByName('CUDA')
    properties = {'Precision': 'mixed'}
    simulation = app.Simulation(modeller.topology, system, integrator, platform, properties)
    print("  Using CUDA platform with mixed precision")
except Exception as e:
    print(f"  CUDA not available: {e}")
    platform = Platform.getPlatformByName('Reference')
    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    print("  Using Reference platform")

# Set positions
simulation.context.setPositions(modeller.positions)

# ═══════════════════════════════════════════════════════════════════════════
# Setup Trajectory Output (matching original L142)
# ═══════════════════════════════════════════════════════════════════════════

trajectory_file = os.path.join(outPath, 'FV_NVT.dcd')
simulation.reporters.append(
    DCDReporter(trajectory_file, freq_traj_output_ps * 1000)
)

# Write initial PDB (matching original L132)
initial_pdb = os.path.join(outPath, 'start_drudes.pdb')
PDBFile.writeFile(modeller.topology, modeller.positions, open(initial_pdb, 'w'))

# ═══════════════════════════════════════════════════════════════════════════
# Run Simulation (matching original L144-171)
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 75)
print("Starting ConstantV MD Simulation")
print("=" * 75)
print(f"Simulation time: {simulation_time_ns} ns")
print(f"Charge update frequency: {freq_charge_update_fs} fs")
print(f"Trajectory output frequency: {freq_traj_output_ps} ps")
print(f"SCF iterations per charge update: 4")
print()

t1 = datetime.now()

# Main simulation loop (matching original L144)
num_output_steps = int(simulation_time_ns * 1000 / freq_traj_output_ps)

for i in range(num_output_steps):
    state = simulation.context.getState(
        getEnergy=True, getForces=True, getVelocities=False, getPositions=True
    )
    
    print(f"Iteration {i}")
    print(f"  Kinetic Energy: {state.getKineticEnergy()}")
    print(f"  Potential Energy: {state.getPotentialEnergy()}")
    
    # Print energy by force group (matching original L149-151)
    for j in range(system.getNumForces()):
        f = system.getForce(j)
        energy = simulation.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()
        print(f"  {type(f).__name__}: {energy}")

    # Constant Voltage Simulation (matching original L160-167)
    # Original workflow:
    #   for j in range(num_charge_updates):
    #       Poisson_solver_fixed_voltage(Niterations=4)  # SCF charge update
    #       simmd.step(freq_charge_update_fs)            # MD step
    #
    # New workflow (equivalent):
    #   integrator.step() automatically runs SCF when stepCount % scfFrequency == 0
    #   So we just call step() for the total number of steps
    
    num_charge_updates = int(freq_traj_output_ps * 1000 / freq_charge_update_fs)
    steps_per_charge_update = int(freq_charge_update_fs / timestep.value_in_unit(unit.picoseconds) / 1000)
    
    for j in range(num_charge_updates):
        # Each call to step() will:
        # - Run SCF if stepCount % scfFrequency == 0 (which we set to match freq_charge_update_fs)
        # - Then run MD integration
        # This is equivalent to: Poisson_solver_fixed_voltage(4) + simmd.step(freq_charge_update_fs)
        simulation.step(steps_per_charge_update)
    
    # Write charges if requested (matching original L165-167)
    if write_charges:
        # TODO: Implement charge writing using integrator.getTotalCathodeCharge() etc.
        pass

t2 = datetime.now()
elapsed = (t2 - t1).total_seconds()

print("\n" + "=" * 75)
print("Simulation Complete!")
print(f"Elapsed time: {elapsed:.2f} seconds")
print("=" * 75)

if write_charges:
    chargeFile.close()

print("\nDone!")

