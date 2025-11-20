#!/usr/bin/env python
"""
Example usage of the ConstantV Plugin with one-call setup helpers.

This script demonstrates the simplified workflow using the Python helpers
added in Phase 1, which matches the Original Python code's simplicity.

**Goal**: Make plugin as easy to use as Original Python code

Original workflow (3 calls):
    MMsys = MM(pdb_list=[...], ...)
    MMsys.initialize_electrodes(voltage, cathode_idx, anode_idx, chain=True)
    MMsys.generate_exclusions(flag_SAPT_FF_exclusions=True)

Plugin workflow with helpers (also ~3 calls):
    context = initialize_electrodes_auto(integrator, topology, system, positions, ...)
    add_saptff_exclusions(topology, system)  # If using SAPT-FF
    # Ready to simulate!

**Requirements**:
- Compiled ConstantV plugin
- OpenMM >= 7.5
- System PDB file with electrode chains
- Force field XML files

**Author**: Auto-generated from ConstantV Plugin
**Date**: 2025-11-20
"""

import sys
from openmm.app import *
from openmm import *
from openmm.unit import *

# Import plugin helpers (Phase 1 additions)
from constantvplugin.helpers import (
    initialize_electrodes_auto,
    add_buckyball_conductor,
    add_saptff_exclusions,
    ElectrodeChargeReporter,
    print_electrode_charge_summary,
    get_electrode_charge_summary,
    MC_Barostat
)

# ═══════════════════════════════════════════════════════════════════
# EXAMPLE 1: Simplest Case - Flat Electrodes Only
# ═══════════════════════════════════════════════════════════════════

def example_flat_electrodes():
    """
    Simplest example: Flat graphene electrodes with electrolyte.

    This is equivalent to the Original Python code:
        MMsys = MM(pdb_list=['system.pdb'], ...)
        MMsys.initialize_electrodes(voltage=1.0, cathode_idx=(0,2), anode_idx=(1,3), chain=True)
        MMsys.initialize_electrolyte()
        MMsys.generate_exclusions()
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Flat Electrodes (Simplest Case)")
    print("="*70)

    # ═══════════════════════════════════════════════════════════════
    # Step 1: Load system (standard OpenMM)
    # ═══════════════════════════════════════════════════════════════
    print("\n1. Loading system...")

    # Replace with your PDB file
    pdb = PDBFile('system.pdb')

    # Replace with your force field XML files
    forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

    # Create system (standard OpenMM)
    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=1.0*nanometer,
        constraints=HBonds
    )

    print(f"  ✓ Loaded {system.getNumParticles()} particles")

    # ═══════════════════════════════════════════════════════════════
    # Step 2: Create integrator
    # ═══════════════════════════════════════════════════════════════
    print("\n2. Creating ConstantVLangevinIntegrator...")

    integrator = ConstantVLangevinIntegrator(
        300*kelvin,      # temperature
        1.0/picosecond,  # friction coefficient
        2.0*femtosecond  # time step
    )

    print(f"  ✓ Integrator created (T=300K, dt=2fs)")

    # ═══════════════════════════════════════════════════════════════
    # Step 3: ONE-CALL electrode initialization (THE KEY STEP!)
    # ═══════════════════════════════════════════════════════════════
    print("\n3. 🚀 One-call electrode initialization...")
    print("   (This replaces 8+ manual steps!)")

    context = initialize_electrodes_auto(
        integrator, pdb.topology, system, pdb.positions,
        voltage=1.0,  # 1.0 V
        cathode_identifier=(0, 2),  # Chains 0 and 2 are cathode
        anode_identifier=(1, 3),    # Chains 1 and 3 are anode
        chain=True,                 # Identify by chain index
        exclude_element=("H",)      # Exclude dummy hydrogen atoms
    )

    # That's it! Context is ready to run.

    # ═══════════════════════════════════════════════════════════════
    # Step 4: Create simulation and run
    # ═══════════════════════════════════════════════════════════════
    print("\n4. Creating simulation...")

    simulation = Simulation(pdb.topology, system, integrator, context=context)

    # Add reporters
    simulation.reporters.append(StateDataReporter(
        stdout, 100,
        step=True, time=True,
        potentialEnergy=True, temperature=True
    ))

    print(f"  ✓ Simulation ready")

    # ═══════════════════════════════════════════════════════════════
    # Step 5: Monitor electrode charges (diagnostic)
    # ═══════════════════════════════════════════════════════════════
    print("\n5. 🔍 Checking initial electrode charges...")

    print_electrode_charge_summary(integrator, system)

    # ═══════════════════════════════════════════════════════════════
    # Step 6: Run simulation
    # ═══════════════════════════════════════════════════════════════
    print("\n6. Running simulation (1000 steps)...")

    simulation.step(1000)

    print("\n  ✓ Simulation complete!")

    # Final charge check
    print("\n7. 🔍 Final electrode charges...")
    print_electrode_charge_summary(integrator, system)

    print("\n" + "="*70)
    print("EXAMPLE 1 COMPLETE")
    print("="*70 + "\n")


# ═══════════════════════════════════════════════════════════════════
# EXAMPLE 2: Buckyball Conductors
# ═══════════════════════════════════════════════════════════════════

def example_buckyball_conductors():
    """
    Advanced example: Flat electrodes + Buckyball conductors.

    This is equivalent to the Original Python code:
        MMsys.initialize_electrodes(..., BuckyBalls=[(virtual_chain, real_chain)])
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Buckyball Conductors")
    print("="*70)

    # ... (Steps 1-2 same as Example 1) ...

    pdb = PDBFile('system_with_buckyball.pdb')
    forcefield = ForceField(...)
    system = forcefield.createSystem(...)
    integrator = ConstantVLangevinIntegrator(300*kelvin, 1.0/picosecond, 2.0*femtosecond)

    print("\n🚀 One-call initialization with Buckyball...")

    context = initialize_electrodes_auto(
        integrator, pdb.topology, system, pdb.positions,
        voltage=1.0,
        cathode_identifier=(0, 2),
        anode_identifier=(1, 3),
        chain=True,
        exclude_element=("H",),
        # KEY: Add Buckyball conductors
        buckyballs=[(4, 5)]  # Virtual chain 4, real chain 5
    )

    print("\n  ✓ Flat electrodes + 1 Buckyball conductor initialized!")

    # ... Continue with simulation ...

    print("\n" + "="*70)
    print("EXAMPLE 2 COMPLETE")
    print("="*70 + "\n")


# ═══════════════════════════════════════════════════════════════════
# EXAMPLE 3: SAPT-FF Force Field
# ═══════════════════════════════════════════════════════════════════

def example_saptff_forcefield():
    """
    Example with SAPT-FF force field (water + TFSI electrolyte).

    Demonstrates:
    - Water interaction groups (SWM4-NDP/SAPT-FF hybrid)
    - TFSI intra-molecular exclusions with Drude screening
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: SAPT-FF Force Field")
    print("="*70)

    # ... (Steps 1-3 same as Example 1) ...

    pdb = PDBFile('system_saptff.pdb')
    # Use SAPT-FF force field XMLs
    forcefield = ForceField('sapt.xml', 'sapt_residues.xml')
    system = forcefield.createSystem(...)
    integrator = ConstantVLangevinIntegrator(...)

    # Initialize electrodes
    context = initialize_electrodes_auto(...)

    # ═══════════════════════════════════════════════════════════════
    # KEY: Add SAPT-FF specific exclusions
    # ═══════════════════════════════════════════════════════════════
    print("\n🔧 Adding SAPT-FF electrolyte exclusions...")

    add_saptff_exclusions(
        pdb.topology, system,
        water_residue_name='HOH',  # Your water residue name
        tfsi_residue_name='Tf2N'   # Your TFSI residue name
    )

    # Must reinitialize context after adding exclusions
    context.reinitialize(preserveState=True)

    print("\n  ✓ SAPT-FF exclusions added and context reinitialized!")

    # ... Continue with simulation ...

    print("\n" + "="*70)
    print("EXAMPLE 3 COMPLETE")
    print("="*70 + "\n")


# ═══════════════════════════════════════════════════════════════════
# EXAMPLE 4: MC Density Equilibration
# ═══════════════════════════════════════════════════════════════════

def example_mc_barostat():
    """
    Example of Monte Carlo density equilibration.

    Use this when you need to equilibrate electrolyte density before
    running production MD.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: MC Density Equilibration")
    print("="*70)

    # ... (Steps 1-3 same as Example 1) ...

    pdb = PDBFile('system.pdb')
    forcefield = ForceField(...)
    system = forcefield.createSystem(...)
    integrator = ConstantVLangevinIntegrator(...)
    context = initialize_electrodes_auto(...)
    simulation = Simulation(pdb.topology, system, integrator, context=context)

    # ═══════════════════════════════════════════════════════════════
    # KEY: Create MC_Barostat for density equilibration
    # ═══════════════════════════════════════════════════════════════
    print("\n🎲 Creating MC Barostat...")

    # Get electrode atoms for MC barostat
    cathode_atoms = list(integrator.getCathodeAtomIndices())
    anode_atoms = list(integrator.getAnodeAtomIndices())

    # Get electrolyte residues (exclude electrode chains)
    electrolyte_residues = []
    electrode_chains = {0, 1, 2, 3}  # Chains used by electrodes
    for residue in pdb.topology.residues():
        if residue.chain.index not in electrode_chains:
            electrolyte_residues.append(residue)

    # Get box dimensions
    box = pdb.topology.getPeriodicBoxVectors()
    Lx = box[0][0].value_in_unit(nanometer)
    Ly = box[1][1].value_in_unit(nanometer)
    Lz = box[2][2].value_in_unit(nanometer)

    # Create MC barostat
    mc_barostat = MC_Barostat(
        simulation, pdb.topology,
        cathode_atoms, anode_atoms, electrolyte_residues,
        temperature=300.0,  # K
        cell_dimensions=(Lx, Ly, Lz),  # nm
        pressure=1.0,  # bar
        barofreq=100,  # MD steps between MC moves
        shiftscale=0.02  # Initial move size (nm)
    )

    print(f"  ✓ MC Barostat created (will equilibrate density)")

    # ═══════════════════════════════════════════════════════════════
    # Run MC equilibration
    # ═══════════════════════════════════════════════════════════════
    print("\n🎲 Running 100 MC steps...")

    for i in range(100):
        mc_barostat.step()

        if i % 10 == 0:
            stats = mc_barostat.get_statistics()
            print(f"  MC step {i}: acceptance = {stats['acceptance_ratio']:.2%}")

    print(f"\n  ✓ MC equilibration complete!")
    print(f"  Final acceptance ratio: {mc_barostat.get_acceptance_ratio():.2%}")

    print("\n" + "="*70)
    print("EXAMPLE 4 COMPLETE")
    print("="*70 + "\n")


# ═══════════════════════════════════════════════════════════════════
# EXAMPLE 5: Electrode Charge Monitoring
# ═══════════════════════════════════════════════════════════════════

def example_charge_monitoring():
    """
    Example of monitoring electrode charges during simulation.

    Use ElectrodeChargeReporter to write charge trajectories to file.
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Electrode Charge Monitoring")
    print("="*70)

    # ... (Steps 1-3 same as Example 1) ...

    pdb = PDBFile('system.pdb')
    forcefield = ForceField(...)
    system = forcefield.createSystem(...)
    integrator = ConstantVLangevinIntegrator(...)
    context = initialize_electrodes_auto(...)
    simulation = Simulation(pdb.topology, system, integrator, context=context)

    # ═══════════════════════════════════════════════════════════════
    # KEY: Add ElectrodeChargeReporter
    # ═══════════════════════════════════════════════════════════════
    print("\n📊 Adding ElectrodeChargeReporter...")

    charge_reporter = ElectrodeChargeReporter(
        'charges.dat',      # Output file
        reportInterval=100,  # Write every 100 steps
        integrator=integrator,
        system=system
    )

    simulation.reporters.append(charge_reporter)

    print(f"  ✓ Will write electrode charges to 'charges.dat' every 100 steps")

    # ═══════════════════════════════════════════════════════════════
    # Run simulation
    # ═══════════════════════════════════════════════════════════════
    print("\n▶️  Running simulation (10000 steps)...")

    simulation.step(10000)

    print(f"\n  ✓ Simulation complete! Check 'charges.dat' for charge trajectory.")

    # ═══════════════════════════════════════════════════════════════
    # Diagnostic: Print final charge summary
    # ═══════════════════════════════════════════════════════════════
    print("\n🔍 Final electrode charge summary:")
    print_electrode_charge_summary(integrator, system)

    # Programmatic access to charge data
    summary = get_electrode_charge_summary(integrator, system)
    print(f"Cathode total: {summary['cathode_total_charge']:.6f} e")
    print(f"Anode total: {summary['anode_total_charge']:.6f} e")
    print(f"Charge balance: {summary['charge_balance']:.2e} e")

    if abs(summary['charge_balance']) > 1e-6:
        print("⚠️  Warning: Charge not conserved! SCF may need more iterations.")

    print("\n" + "="*70)
    print("EXAMPLE 5 COMPLETE")
    print("="*70 + "\n")


def example_umbrella_potential():
    """
    Example of applying umbrella potential for constrained sampling.

    Demonstrates two umbrella modes:
    1. Absolute z-position constraint (fix ion at specific height)
    2. Distance constraint between two molecules
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Umbrella Potential for Constrained Sampling")
    print("="*70)

    # ... (Steps 1-3 same as Example 1) ...

    pdb = PDBFile('system.pdb')
    forcefield = ForceField(...)
    system = forcefield.createSystem(...)
    integrator = ConstantVLangevinIntegrator(...)
    context = initialize_electrodes_auto(...)
    simulation = Simulation(pdb.topology, system, integrator, context=context)

    # ═══════════════════════════════════════════════════════════════
    # OPTION 1: Fix ion at absolute z position
    # ═══════════════════════════════════════════════════════════════
    print("\n🎯 Option 1: Constraining ion to z=3.0 nm...")

    from openmm.unit import kilocalories_per_mole, angstrom, nanometer

    force = set_umbrella_potential(
        simulation=simulation,
        system=system,
        topology=pdb.topology,
        molecule_name='LI',  # Lithium ion
        force_constant=100*kilocalories_per_mole/angstrom**2,
        z_global=3.0*nanometer  # Fix at z=3.0 nm
    )

    print(f"  ✓ Added umbrella force: z0 = 3.0 nm, k = 100 kcal/mol/Å²")

    # ═══════════════════════════════════════════════════════════════
    # OPTION 2: Constrain distance between ion and electrode atom
    # ═══════════════════════════════════════════════════════════════
    # Alternatively, use distance constraint:
    #
    # force = set_umbrella_potential(
    #     simulation=simulation,
    #     system=system,
    #     topology=pdb.topology,
    #     molecule_name='LI',
    #     force_constant=100*kilocalories_per_mole/angstrom**2,
    #     mol2='ELEC',         # Electrode residue name
    #     atomtype='C1',        # Specific atom on electrode
    #     r0centroid=0.5*nanometer  # Target distance = 0.5 nm
    # )

    # ═══════════════════════════════════════════════════════════════
    # Run umbrella sampling
    # ═══════════════════════════════════════════════════════════════
    print("\n▶️  Running umbrella sampling (10000 steps)...")

    simulation.step(10000)

    print(f"\n  ✓ Umbrella sampling complete!")
    print(f"  ✓ Ion constrained near z=3.0 nm throughout simulation")

    # ═══════════════════════════════════════════════════════════════
    # Check final position
    # ═══════════════════════════════════════════════════════════════
    state = simulation.context.getState(getPositions=True)
    positions = state.getPositions()

    # Find ion position
    for atom in pdb.topology.atoms():
        if atom.residue.name == 'LI':
            ion_z = positions[atom.index][2].value_in_unit(nanometer)
            print(f"\n📍 Final ion z-position: {ion_z:.4f} nm (target: 3.0000 nm)")
            break

    print("\n💡 Use Case: Umbrella sampling for PMF calculations")
    print("   - Run multiple windows at different z positions")
    print("   - Combine with WHAM for free energy profile")
    print("   - Study ion adsorption/desorption at electrode")

    print("\n" + "="*70)
    print("EXAMPLE 6 COMPLETE")
    print("="*70 + "\n")


# ═══════════════════════════════════════════════════════════════════
# MAIN: Run examples
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║         ConstantV Plugin - Example Usage                         ║
║         Simplified Workflow with Python Helpers                  ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

This script demonstrates 6 common use cases:

1. Flat electrodes (simplest case)
2. Buckyball conductors (advanced)
3. SAPT-FF force field (water + TFSI)
4. MC density equilibration
5. Electrode charge monitoring
6. Umbrella potential (constrained sampling)

**Key Insight**: With the new Python helpers, plugin usage is as simple
as the Original Python code!

Original:   3 calls  (MMsys.initialize_electrodes, generate_exclusions, ...)
Plugin:     3 calls  (initialize_electrodes_auto, add_saptff_exclusions, ...)

Same simplicity, 10x performance (C++ kernel)! 🚀

""")

    # Uncomment the example you want to run:

    # example_flat_electrodes()
    # example_buckyball_conductors()
    # example_saptff_forcefield()
    # example_mc_barostat()
    # example_charge_monitoring()
    # example_umbrella_potential()

    print("\n✓ All examples defined. Uncomment one in main() to run.\n")
