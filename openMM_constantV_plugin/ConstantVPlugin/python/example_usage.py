#!/usr/bin/env python
"""
Complete Example: Constant Voltage Simulation with ConstantV Plugin

This example replicates the Original Python system behavior for flat electrodes.
Follows the exact workflow from run_openMM.py.

Original reference: /home/andy/test_optimization/OpenMM-ConstantV(original)/run_openMM.py

Excludes (out of scope):
- Conductors (Buckyball/Nanotube)
- MC equilibration
- QM/MM interface

⚠️  CRITICAL STEPS:
1. Add electrode exclusions BEFORE creating Context
2. Call context.reinitialize() AFTER creating Context
"""

import sys
import openmm as mm
import openmm.app as app
import openmm.unit as unit

# Import ConstantV plugin
try:
    from constantvplugin import ConstantVIntegrator
    from constantvplugin_helpers import (
        add_electrode_exclusions,
        configure_geometry_from_context,
        add_electrolyte_atoms_auto,
        compute_electrode_area_per_atom,
        validate_setup
    )
except ImportError:
    print("ERROR: ConstantV plugin not found.")
    print("Make sure the plugin is installed:")
    print("  cd ConstantVPlugin/build")
    print("  make install")
    print("  make PythonInstall")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════
# Configuration (modify these for your system)
# ═══════════════════════════════════════════════════════════
PDB_FILE = 'nvt_0V_15ns.pdb'
FORCEFIELD_DIR = './ffdir/'

# Electrode identification (by chain index, like Original)
CATHODE_CHAIN_INDEX = 0  # Chain 0 is cathode
ANODE_CHAIN_INDEX = 1    # Chain 1 is anode
EXCLUDE_ELEMENT = 'H'    # Exclude hydrogens from electrodes

# Simulation parameters (from Original run_openMM.py Lines 34, 73)
VOLTAGE_VOLTS = 0.0              # Applied voltage (Volts)
TIMESTEP_FS = 1.0                # Timestep (femtoseconds)
NUM_SCF_ITERATIONS = 4           # SCF iterations per charge update
SCF_FREQUENCY = 1                # Update charges every N MD steps
SIMULATION_TIME_NS = 0.5         # Total simulation time (nanoseconds)
TRAJ_OUTPUT_FREQ_PS = 10.0       # Trajectory output frequency (picoseconds)

# Output files
OUTPUT_DIR = '1v_0.5ns'
OUTPUT_DCD = f'{OUTPUT_DIR}/output.dcd'
OUTPUT_LOG = f'{OUTPUT_DIR}/output.log'

# ═══════════════════════════════════════════════════════════
# Step 1: Load System (Standard OpenMM)
# ═══════════════════════════════════════════════════════════
print("="*60)
print("STEP 1: Loading system")
print("="*60)

try:
    pdb = app.PDBFile(PDB_FILE)
    print(f"✓ Loaded PDB: {PDB_FILE}")
except FileNotFoundError:
    print(f"ERROR: PDB file not found: {PDB_FILE}")
    sys.exit(1)

# Load force fields (adjust paths for your system)
forcefield_files = [
    f'{FORCEFIELD_DIR}/sapt_residues.xml',
    f'{FORCEFIELD_DIR}/graph_residue_c.xml',
    f'{FORCEFIELD_DIR}/graph_residue_n.xml',
    f'{FORCEFIELD_DIR}/sapt_noDB_2sheets.xml',
    f'{FORCEFIELD_DIR}/graph_c_freeze.xml',
    f'{FORCEFIELD_DIR}/graph_n_freeze.xml'
]

try:
    forcefield = app.ForceField(*forcefield_files)
    print(f"✓ Loaded {len(forcefield_files)} force field files")
except Exception as e:
    print(f"ERROR: Could not load force fields: {e}")
    print("Note: Adjust FORCEFIELD_DIR and file names for your system")
    sys.exit(1)

# Create system
system = forcefield.createSystem(
    pdb.topology,
    nonbondedCutoff=1.4*unit.nanometers,
    constraints=app.HBonds,
    rigidWater=True
)
print(f"✓ Created system with {system.getNumParticles()} particles")

# Get NonbondedForce (required for exclusions)
nonbonded_force = None
custom_nonbonded_force = None

for force in system.getForces():
    if isinstance(force, mm.NonbondedForce):
        nonbonded_force = force
    elif isinstance(force, mm.CustomNonbondedForce):
        custom_nonbonded_force = force

if nonbonded_force is None:
    print("ERROR: NonbondedForce not found in system")
    sys.exit(1)

print(f"✓ Found NonbondedForce")
if custom_nonbonded_force:
    print(f"✓ Found CustomNonbondedForce (e.g., SAPT-FF)")

# ═══════════════════════════════════════════════════════════
# Step 2: Create ConstantVIntegrator
# ═══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 2: Creating ConstantVIntegrator")
print("="*60)

timestep = TIMESTEP_FS * unit.femtoseconds
integrator = ConstantVIntegrator(timestep)

# Set voltage (converted internally from Volts to kJ/mol)
integrator.setVoltage(VOLTAGE_VOLTS)
print(f"✓ Voltage set: {VOLTAGE_VOLTS} V")

# Set SCF parameters (Original defaults: 4 iterations, every MD step)
integrator.setNumSCFIterations(NUM_SCF_ITERATIONS)
integrator.setSCFFrequency(SCF_FREQUENCY)
print(f"✓ SCF iterations: {NUM_SCF_ITERATIONS}")
print(f"✓ SCF frequency: every {SCF_FREQUENCY} MD step(s)")

# ═══════════════════════════════════════════════════════════
# Step 3: Identify and Add Electrode Atoms
# ═══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 3: Identifying electrode atoms")
print("="*60)

# Identify cathode atoms by chain index
cathode_atoms = []
for chain in pdb.topology.chains():
    if chain.index == CATHODE_CHAIN_INDEX:
        for atom in chain.atoms():
            # Exclude specified elements (e.g., hydrogens)
            if atom.element.symbol != EXCLUDE_ELEMENT:
                cathode_atoms.append(atom.index)

print(f"✓ Cathode (chain {CATHODE_CHAIN_INDEX}): {len(cathode_atoms)} atoms")

# Identify anode atoms by chain index
anode_atoms = []
for chain in pdb.topology.chains():
    if chain.index == ANODE_CHAIN_INDEX:
        for atom in chain.atoms():
            if atom.element.symbol != EXCLUDE_ELEMENT:
                anode_atoms.append(atom.index)

print(f"✓ Anode (chain {ANODE_CHAIN_INDEX}): {len(anode_atoms)} atoms")

if len(cathode_atoms) == 0 or len(anode_atoms) == 0:
    print("ERROR: No electrode atoms found. Check CATHODE_CHAIN_INDEX and ANODE_CHAIN_INDEX")
    sys.exit(1)

# Compute area per atom (using helper function)
cathode_area_per_atom, total_area = compute_electrode_area_per_atom(
    pdb.topology, cathode_atoms
)
anode_area_per_atom, _ = compute_electrode_area_per_atom(
    pdb.topology, anode_atoms
)

print(f"✓ Cathode area per atom: {cathode_area_per_atom:.6f} nm²")
print(f"✓ Anode area per atom: {anode_area_per_atom:.6f} nm²")
print(f"✓ Total sheet area: {total_area:.4f} nm²")

# Add to integrator
for atom_idx in cathode_atoms:
    integrator.addCathodeAtom(atom_idx, cathode_area_per_atom)

for atom_idx in anode_atoms:
    integrator.addAnodeAtom(atom_idx, anode_area_per_atom)

# ═══════════════════════════════════════════════════════════
# Step 4: Auto-Identify and Add Electrolyte Atoms
# ═══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 4: Auto-identifying electrolyte atoms")
print("="*60)

# Use helper function to automatically identify electrolyte
# (Residues with < 100 atoms, excluding electrode chains)
electrolyte_atoms = add_electrolyte_atoms_auto(
    pdb.topology,
    integrator,
    nonbonded_force,
    natom_cutoff=100,  # Original default (MM_classes.py:256)
    exclude_chains=[CATHODE_CHAIN_INDEX, ANODE_CHAIN_INDEX]
)

if len(electrolyte_atoms) == 0:
    print("WARNING: No electrolyte atoms found")
    print("This may be intentional for a vacuum simulation")

# ═══════════════════════════════════════════════════════════
# Step 5: Set Geometry Parameters (auto-configure)
# ═══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 5: Configuring electrode geometry")
print("="*60)

# Create a temporary context to get positions
# (We'll recreate it later after adding exclusions)
temp_integrator = mm.VerletIntegrator(timestep)
temp_context = mm.Context(system, temp_integrator)
temp_context.setPositions(pdb.positions)

# Auto-configure geometry using helper function
geometry_params = configure_geometry_from_context(
    temp_context,
    integrator,
    cathode_atoms[0],  # Use first cathode atom for z position
    anode_atoms[0]     # Use first anode atom for z position
)

del temp_context
del temp_integrator

# ═══════════════════════════════════════════════════════════
# Step 6: ⚠️  CRITICAL - Add Electrode Exclusions
# ═══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("⚠️  STEP 6: Adding electrode exclusions (CRITICAL!)")
print("="*60)

# This is the MOST IMPORTANT step!
# Without exclusions, electrode atoms will interact, causing non-physical forces
add_electrode_exclusions(integrator, nonbonded_force, custom_nonbonded_force)

# ═══════════════════════════════════════════════════════════
# Step 7: Create Context and Platform
# ═══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 7: Creating simulation context")
print("="*60)

# Select platform (CUDA preferred, fallback to CPU/Reference)
try:
    platform = mm.Platform.getPlatformByName('CUDA')
    properties = {'Precision': 'mixed'}
    print("✓ Using CUDA platform (GPU)")
except Exception:
    try:
        platform = mm.Platform.getPlatformByName('CPU')
        properties = {}
        print("⚠️  Using CPU platform (slower)")
    except Exception:
        platform = mm.Platform.getPlatformByName('Reference')
        properties = {}
        print("⚠️  Using Reference platform (very slow, for validation only)")

# Create context
context = mm.Context(system, integrator, platform, properties)
context.setPositions(pdb.positions)
print("✓ Context created with positions set")

# ⚠️  CRITICAL: Reinitialize to apply exclusions
print("\n⚠️  CRITICAL: Reinitializing context to apply exclusions...")
context.reinitialize(preserveState=True)
print("✓ Context reinitialized - exclusions are now active")

# Set velocities
context.setVelocitiesToTemperature(300*unit.kelvin)
print("✓ Velocities initialized to 300 K")

# ═══════════════════════════════════════════════════════════
# Step 8: Validate Setup
# ═══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 8: Validating setup")
print("="*60)

valid, messages = validate_setup(context, integrator)

if not valid:
    print("\n❌ Setup validation FAILED. Please fix errors before running.")
    for msg in messages:
        print(f"  {msg}")
    sys.exit(1)

print("✓ Setup validation PASSED")

# ═══════════════════════════════════════════════════════════
# Step 9: Setup Reporters
# ═══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 9: Setting up output reporters")
print("="*60)

import os
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"✓ Created output directory: {OUTPUT_DIR}")

# Create simulation wrapper for reporters
simulation = app.Simulation(pdb.topology, system, integrator, platform, properties)
simulation.context = context  # Use the context we already created

# Trajectory reporter (DCD format)
traj_freq_steps = int(TRAJ_OUTPUT_FREQ_PS * 1000 / TIMESTEP_FS)  # Convert ps to steps
simulation.reporters.append(
    app.DCDReporter(OUTPUT_DCD, traj_freq_steps)
)
print(f"✓ Trajectory output: {OUTPUT_DCD} (every {traj_freq_steps} steps)")

# State data reporter (energies, temperature, etc.)
log_freq_steps = max(100, traj_freq_steps // 10)  # Log more frequently than trajectory
simulation.reporters.append(
    app.StateDataReporter(
        OUTPUT_LOG,
        log_freq_steps,
        step=True,
        time=True,
        potentialEnergy=True,
        kineticEnergy=True,
        totalEnergy=True,
        temperature=True,
        speed=True
    )
)
print(f"✓ Log output: {OUTPUT_LOG} (every {log_freq_steps} steps)")

# Console reporter for progress
simulation.reporters.append(
    app.StateDataReporter(
        sys.stdout,
        traj_freq_steps,
        step=True,
        time=True,
        potentialEnergy=True,
        temperature=True,
        speed=True,
        remainingTime=True,
        totalSteps=int(SIMULATION_TIME_NS * 1e6 / TIMESTEP_FS)
    )
)

# ═══════════════════════════════════════════════════════════
# Step 10: Run Simulation
# ═══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 10: Running constant voltage MD simulation")
print("="*60)
print(f"Simulation parameters:")
print(f"  Voltage: {VOLTAGE_VOLTS} V")
print(f"  Timestep: {TIMESTEP_FS} fs")
print(f"  SCF iterations: {NUM_SCF_ITERATIONS}")
print(f"  SCF frequency: every {SCF_FREQUENCY} MD step(s)")
print(f"  Total time: {SIMULATION_TIME_NS} ns")
print(f"  Total steps: {int(SIMULATION_TIME_NS * 1e6 / TIMESTEP_FS)}")
print("="*60)
print("Starting simulation...\n")

# Calculate total steps
num_steps = int(SIMULATION_TIME_NS * 1e6 / TIMESTEP_FS)

# Run simulation
# Note: The integrator automatically calls SCF solver based on SCF_FREQUENCY
try:
    simulation.step(num_steps)
except Exception as e:
    print(f"\n❌ ERROR during simulation: {e}")
    print("\nTroubleshooting:")
    print("1. If energy explodes, check that exclusions were added correctly")
    print("2. If 'divide by zero', check electrode charges are not all zero")
    print("3. Check that geometry parameters are reasonable")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════
# Step 11: Finalize
# ═══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Simulation complete!")
print("="*60)
print(f"Output files:")
print(f"  Trajectory: {OUTPUT_DCD}")
print(f"  Log: {OUTPUT_LOG}")
print("="*60)

# Get final state for analysis
final_state = context.getState(getEnergy=True, getPositions=True)
print(f"\nFinal state:")
print(f"  Potential energy: {final_state.getPotentialEnergy()}")
print(f"  Kinetic energy: {final_state.getKineticEnergy()}")
print(f"  Total energy: {final_state.getPotentialEnergy() + final_state.getKineticEnergy()}")

print("\n✓ All done!")
