#!/usr/bin/env python3
"""
Fixed-Voltage Molecular Dynamics using ConstantVPlugin

This is a REFACTORED version of the original OpenMM-ConstantV code.

Key differences from original:
1. Uses ConstantVPlugin (C++/CUDA) instead of Python SCF iteration
2. Computes C_inv matrix ONCE at initialization
3. 4× fewer CPU-GPU transfers per timestep
4. Cleaner, more maintainable code

Performance:
  Original: 8 CPU-GPU transfers/timestep (4 SCF iterations × 2 transfers)
  Plugin:   2 CPU-GPU transfers/timestep (1× download + 1× upload via API)
"""

import sys
import os
import numpy as np
import argparse
from datetime import datetime

# OpenMM
from openmm import *
from openmm.app import *
from openmm.unit import *

# Our plugin
import constantvplugin

# Our capacitance matrix calculator
sys.path.append('..')
from compute_capacitance_matrix import compute_inverse_capacitance_matrix

# Constants
COULOMB_CONSTANT = 138.935456  # kJ/mol · nm / e²
CONVERSION_V_TO_KJMOL = 96.485  # 1 V = 96.485 kJ/mol

def setup_system(pdb_file, residue_xml_files, forcefield_xml_files):
    """
    Set up OpenMM system from PDB and force field files.

    This preserves the system setup logic from original code.

    CRITICAL: Must load bond definitions BEFORE loading PDB!
    This is how OpenMM handles residues without CONECT records (like graphene).
    """
    print("Setting up OpenMM system...")

    # Step 1: Load bond definitions from residue XML files BEFORE creating PDB
    # This is CRITICAL for residues like graphene that lack CONECT records
    print(f"Loading bond definitions from {len(residue_xml_files)} residue files...")
    for residue_file in residue_xml_files:
        Topology().loadBondDefinitions(residue_file)

    # Step 2: Load PDB (now it can use the bond definitions we just loaded)
    pdb = PDBFile(pdb_file)

    # Step 3: Create Modeller to handle extra particles (Drude oscillators)
    # This is critical for polarizable force fields
    modeller = Modeller(pdb.topology, pdb.positions)

    # Step 4: Create force field from force field XML files
    forcefield = ForceField(*forcefield_xml_files)

    # Step 5: Add extra particles (virtual sites, Drude oscillators, etc.)
    print(f"Adding extra particles (Drude oscillators for polarizable FF)...")
    modeller.addExtraParticles(forcefield)
    print(f"✓ Topology: {modeller.topology.getNumAtoms()} total atoms (including extra particles)")

    # Step 6: Create system using the modeller's topology (with extra particles)
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=1.4*nanometer,
        constraints=HBonds,
        rigidWater=True
    )

    # Step 7: [CRITICAL] Set periodic boundary conditions for bonded forces
    # This is essential for graphene electrodes that span periodic boundaries
    # Without this, graphene bonds crossing PBC will have incorrect forces
    print("Setting periodic boundary conditions for bonded forces (graphene)...")
    from openmm import HarmonicBondForce, HarmonicAngleForce, PeriodicTorsionForce, RBTorsionForce
    for i in range(system.getNumForces()):
        f = system.getForce(i)
        if (
            isinstance(f, HarmonicBondForce) or
            isinstance(f, HarmonicAngleForce) or
            isinstance(f, PeriodicTorsionForce) or
            isinstance(f, RBTorsionForce)
        ):
            f.setUsesPeriodicBoundaryConditions(True)
    print("✓ Periodic bonded forces enabled.")

    # Find NonbondedForce
    nonbonded = None
    for force in system.getForces():
        if isinstance(force, NonbondedForce):
            nonbonded = force
            break

    if nonbonded is None:
        raise RuntimeError("No NonbondedForce found in system")

    print(f"✓ System created: {system.getNumParticles()} particles")
    print(f"✓ NonbondedForce: {nonbonded.getNumParticles()} particles")

    # Return modeller (not pdb) because it contains the updated topology with extra particles
    return modeller, system, nonbonded


def identify_electrode_atoms(topology, cathode_chain_indices, anode_chain_indices):
    """
    Identify electrode atoms by chain index.

    This preserves the atom identification logic from original code.

    Returns:
    --------
    cathode_atoms : list of int
        Atom indices for cathode
    anode_atoms : list of int
        Atom indices for anode
    """
    cathode_atoms = []
    anode_atoms = []

    for chain in topology.chains():
        if chain.index in cathode_chain_indices:
            for atom in chain.atoms():
                cathode_atoms.append(atom.index)
        elif chain.index in anode_chain_indices:
            for atom in chain.atoms():
                anode_atoms.append(atom.index)

    print(f"✓ Cathode: {len(cathode_atoms)} atoms (chains {cathode_chain_indices})")
    print(f"✓ Anode: {len(anode_atoms)} atoms (chains {anode_chain_indices})")

    return cathode_atoms, anode_atoms


def identify_electrolyte_atoms(system, topology, electrode_atoms, include_drude=True):
    """
    Identify electrolyte atoms (all non-electrode particles).

    CRITICAL FIX: This function now includes ALL non-electrode particles,
    including Drude oscillators, to correctly compute E_f.

    The original SCF algorithm implicitly included Drude particles via
    context.calcForcesAndEnergy(). We must do the same.

    Parameters:
    -----------
    system : System
        OpenMM System object (needed to get total particle count)
    topology : Topology
        OpenMM Topology object
    electrode_atoms : list of int
        All electrode atom indices
    include_drude : bool
        Whether to include Drude particles (default: True, MUST be True for correct physics)

    Returns:
    --------
    electrolyte_atoms : list of int
        Atom indices for ALL non-electrode particles (including Drude oscillators)
    """
    all_electrode_atoms = set(electrode_atoms)

    # Get total particle count (includes Drude particles added by Modeller)
    n_particles = system.getNumParticles()
    n_topology_atoms = topology.getNumAtoms()

    print(f"  System particles: {n_particles}")
    print(f"  Topology atoms: {n_topology_atoms}")
    print(f"  Extra particles (Drude): {n_particles - n_topology_atoms}")

    if include_drude:
        # Include ALL non-electrode particles (correct physics)
        # This includes both regular atoms AND Drude oscillators
        electrolyte_atoms = [i for i in range(n_particles) if i not in all_electrode_atoms]
        print(f"✓ Electrolyte: {len(electrolyte_atoms)} particles (including Drude)")
    else:
        # Legacy mode: only topology atoms (INCORRECT PHYSICS, for debugging only)
        electrolyte_atoms = []
        for chain in topology.chains():
            if chain.index not in electrode_chains:
                for residue in chain.residues():
                    n_atoms = len(list(residue.atoms()))
                    if n_atoms < 100:  # Original heuristic
                        for atom in residue.atoms():
                            if atom.index not in all_electrode_atoms:
                                electrolyte_atoms.append(atom.index)
        print(f"⚠️  Electrolyte: {len(electrolyte_atoms)} atoms (NO Drude - INCORRECT!)")

    return electrolyte_atoms


def compute_electrode_areas(n_atoms, box_vectors):
    """
    Compute area per atom for flat electrodes.

    Assumes electrodes span the xy plane.
    """
    lx = box_vectors[0][0].value_in_unit(nanometer)
    ly = box_vectors[1][1].value_in_unit(nanometer)
    sheet_area = lx * ly
    area_per_atom = sheet_area / n_atoms

    return np.full(n_atoms, area_per_atom)


def freeze_electrode_atoms(system, electrode_atoms):
    """
    Freeze electrode atoms by setting their mass to 0.

    CRITICAL: C_inv matrix is computed ONCE based on initial positions.
    If electrode atoms move, the matrix becomes invalid!

    This preserves the "freeze" logic from graph_c_freeze.xml and graph_n_freeze.xml.
    """
    print(f"\n  Freezing {len(electrode_atoms)} electrode atoms...")

    for atom_idx in electrode_atoms:
        system.setParticleMass(atom_idx, 0.0)

    print(f"  ✓ Electrode atoms frozen (mass = 0)")


def initialize_constantv_plugin(
    system,
    nonbonded,
    cathode_atoms,
    anode_atoms,
    electrolyte_atoms,
    positions,
    box_vectors,
    voltage
):
    """
    Initialize ConstantVPlugin and compute C_inv matrix.

    This is the NEW logic that replaces the old SCF iteration.

    CRITICAL FIX: electrolyte_atoms now includes ALL non-electrode particles
    (including Drude oscillators) for correct physics.
    """
    print("\nInitializing ConstantVPlugin...")

    # Freeze electrode atoms (C_inv assumes static electrodes)
    all_electrode_atoms = cathode_atoms + anode_atoms
    freeze_electrode_atoms(system, all_electrode_atoms)

    # Convert voltage to kJ/mol
    voltage_kjmol = voltage * CONVERSION_V_TO_KJMOL

    # Compute electrode areas
    cathode_areas = compute_electrode_areas(len(cathode_atoms), box_vectors)
    anode_areas = compute_electrode_areas(len(anode_atoms), box_vectors)

    # Get positions for C_inv calculation
    cathode_pos = np.array([positions[i].value_in_unit(nanometer) for i in cathode_atoms])
    anode_pos = np.array([positions[i].value_in_unit(nanometer) for i in anode_atoms])

    # Combine all electrode atoms
    all_electrode_atoms = cathode_atoms + anode_atoms
    all_electrode_pos = np.vstack([cathode_pos, anode_pos])
    all_electrode_areas = np.concatenate([cathode_areas, anode_areas])

    # Compute C_inv matrix (ONCE, at initialization)
    print(f"Computing C_inv matrix for {len(all_electrode_atoms)} electrode atoms...")
    C_inv = compute_inverse_capacitance_matrix(all_electrode_pos, all_electrode_areas)
    print(f"✓ C_inv computed: shape {C_inv.shape}, det = {np.linalg.det(C_inv):.6e}")

    # Create ConstantVForce
    cv_force = constantvplugin.ConstantVForce()

    # Add cathode atoms (positive voltage)
    for atom_idx in cathode_atoms:
        cv_force.addElectrodeAtom(atom_idx, voltage_kjmol)

    # Add anode atoms (negative voltage)
    for atom_idx in anode_atoms:
        cv_force.addElectrodeAtom(atom_idx, -voltage_kjmol)

    # Add electrolyte atoms/particles (including Drude oscillators)
    # CRITICAL FIX: This now includes ALL non-electrode particles
    print(f"Adding {len(electrolyte_atoms)} electrolyte particles (including Drude)...")

    electrolyte_charges = []
    for atom_idx in electrolyte_atoms:
        try:
            charge, sigma, epsilon = nonbonded.getParticleParameters(atom_idx)
            q = charge.value_in_unit(elementary_charge)
        except Exception as e:
            # If NonbondedForce doesn't have this particle (shouldn't happen), use 0
            print(f"  Warning: Could not get charge for particle {atom_idx}: {e}")
            q = 0.0

        cv_force.addElectrolyteAtom(atom_idx, q)
        electrolyte_charges.append(q)

    print(f"  ✓ Electrolyte charges: min={min(electrolyte_charges):.3f}, max={max(electrolyte_charges):.3f}")

    # Set C_inv matrix
    cv_force.setInverseCapacitanceMatrix(C_inv.flatten().tolist())

    # Add force to system
    system.addForce(cv_force)

    print(f"✓ ConstantVForce added to system")
    print(f"  - Electrodes: {len(all_electrode_atoms)} atoms")
    print(f"  - Electrolyte: {len(electrolyte_atoms)} atoms")
    print(f"  - Voltage: ±{voltage} V (±{voltage_kjmol:.3f} kJ/mol)")

    return cv_force, C_inv


def run_simulation(
    modeller,
    system,
    voltage,
    simulation_time_ns,
    output_prefix="fv_md"
):
    """
    Run the FV-MD simulation.

    NO Python loops for charge updates!
    The plugin handles everything automatically.

    Parameters:
    -----------
    modeller : Modeller
        OpenMM Modeller object with topology and positions (including extra particles)
    """
    print(f"\nRunning FV-MD simulation...")
    print(f"  Duration: {simulation_time_ns} ns")
    print(f"  Voltage: ±{voltage} V")

    # Create integrator
    integrator = LangevinIntegrator(
        300*kelvin,
        1.0/picosecond,
        0.001*picosecond
    )

    # Create simulation - detect best available platform
    # Check which forces are in the system
    print(f"  System forces:")
    for i, force in enumerate(system.getForces()):
        force_name = force.__class__.__name__
        print(f"    [{i}] {force_name}")

    # Determine platform based on system compatibility
    platform_name = 'CUDA'  # Default to CUDA

    # Check if system has DrudeForce (may not have CUDA support)
    has_drude = False
    for force in system.getForces():
        if isinstance(force, DrudeForce):
            has_drude = True
            break

    if has_drude:
        print(f"  ⚠️  System has DrudeForce - checking CUDA compatibility...")

    # Try to create simulation with CUDA first
    simulation = None
    tried_platforms = []

    for try_platform_name in ['CUDA', 'Reference']:
        try:
            platform = Platform.getPlatformByName(try_platform_name)
            tried_platforms.append(try_platform_name)
            print(f"  Trying {try_platform_name} platform...")

            simulation = Simulation(modeller.topology, system, integrator, platform)
            print(f"  ✓ Simulation created on {try_platform_name} platform")
            break

        except Exception as e:
            print(f"  ✗ {try_platform_name} failed: {str(e)[:100]}")
            continue

    if simulation is None:
        raise RuntimeError(f"Failed to create simulation on any platform: {tried_platforms}")

    simulation.context.setPositions(modeller.positions)

    # Minimize
    print("Minimizing...")
    simulation.minimizeEnergy(maxIterations=100)

    # Set up reporters
    simulation.reporters.append(
        StateDataReporter(
            f"{output_prefix}.log",
            1000,
            step=True,
            time=True,
            potentialEnergy=True,
            temperature=True,
            speed=True
        )
    )

    simulation.reporters.append(
        DCDReporter(
            f"{output_prefix}.dcd",
            10000
        )
    )

    # Run simulation
    n_steps = int(simulation_time_ns * 1000000 / 0.001)  # ns to steps

    print(f"Running {n_steps} steps...")
    print("(Plugin automatically updates charges every step - no Python overhead!)")

    simulation.step(n_steps)

    print(f"✓ Simulation complete")

    # Save final state
    state = simulation.context.getState(getPositions=True)
    with open(f"{output_prefix}_final.pdb", 'w') as f:
        PDBFile.writeFile(
            simulation.topology,
            state.getPositions(),
            f
        )

    return simulation


def main():
    parser = argparse.ArgumentParser(description="FV-MD with ConstantVPlugin")
    parser.add_argument("pdb", help="Input PDB file")
    parser.add_argument("--voltage", type=float, default=1.0, help="Applied voltage (V)")
    parser.add_argument("--time", type=float, default=0.01, help="Simulation time (ns)")
    parser.add_argument("--residue-xml", nargs="+", required=True,
                        help="Residue XML files (loaded BEFORE PDB)")
    parser.add_argument("--forcefield-xml", nargs="+", required=True,
                        help="Force field XML files")
    parser.add_argument("--cathode-chains", type=int, nargs="+", required=True,
                        help="Chain indices for cathode")
    parser.add_argument("--anode-chains", type=int, nargs="+", required=True,
                        help="Chain indices for anode")
    parser.add_argument("--output", default="fv_md", help="Output prefix")

    args = parser.parse_args()

    print("=" * 70)
    print("Fixed-Voltage MD with ConstantVPlugin")
    print("=" * 70)
    print(f"Started: {datetime.now()}")

    # Setup system (returns Modeller object with topology including extra particles)
    modeller, system, nonbonded = setup_system(args.pdb, args.residue_xml, args.forcefield_xml)

    # Identify atoms
    cathode_atoms, anode_atoms = identify_electrode_atoms(
        modeller.topology,
        args.cathode_chains,
        args.anode_chains
    )

    # Identify electrolyte (ALL non-electrode particles, including Drude)
    all_electrode_atoms = cathode_atoms + anode_atoms
    electrolyte_atoms = identify_electrolyte_atoms(
        system,
        modeller.topology,
        all_electrode_atoms,
        include_drude=True  # CRITICAL: Must be True for correct physics
    )

    # Initialize plugin
    cv_force, C_inv = initialize_constantv_plugin(
        system,
        nonbonded,
        cathode_atoms,
        anode_atoms,
        electrolyte_atoms,
        modeller.positions,
        modeller.topology.getPeriodicBoxVectors(),
        args.voltage
    )

    # Run simulation
    simulation = run_simulation(
        modeller,
        system,
        args.voltage,
        args.time,
        args.output
    )

    print("=" * 70)
    print(f"Completed: {datetime.now()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
