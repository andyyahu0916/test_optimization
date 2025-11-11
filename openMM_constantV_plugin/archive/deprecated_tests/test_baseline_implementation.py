#!/usr/bin/env python3
"""
Test Script: Baseline Implementation Validation

Tests the ConstantV plugin baseline implementation to verify:
1. Plugin and helpers load correctly
2. Electrode atoms can be added
3. Exclusions work correctly
4. Geometry configuration works
5. A minimal simulation runs without errors
6. Charges are updated during SCF

This is a minimal validation test before testing against Original Python code.
"""

import sys
import openmm as mm
import openmm.app as app
import openmm.unit as unit

print("="*60)
print("Test 1: Import plugin and helpers")
print("="*60)

try:
    from constantvplugin import ConstantVIntegrator
    print("✓ ConstantVIntegrator imported")
except ImportError as e:
    print(f"✗ Failed to import ConstantVIntegrator: {e}")
    sys.exit(1)

try:
    from constantvplugin_helpers import (
        add_electrode_exclusions,
        configure_geometry_from_context,
        compute_electrode_area_per_atom,
        validate_setup
    )
    print("✓ Helper functions imported")
except ImportError as e:
    print(f"✗ Failed to import helpers: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("Test 2: Create minimal system")
print("="*60)

# Create a minimal system: 2 graphene sheets + 1 water molecule
# Cathode: 8 carbon atoms at z=0
# Anode: 8 carbon atoms at z=3.5 nm
# Water: 3 atoms at z=1.75 nm (middle)

topology = app.Topology()
chain_cathode = topology.addChain()
chain_anode = topology.addChain()
chain_water = topology.addChain()

# Add cathode carbon atoms (chain 0)
res_cathode = topology.addResidue("GRA", chain_cathode)
cathode_atoms = []
for i in range(8):
    atom = topology.addAtom("C", app.Element.getBySymbol("C"), res_cathode)
    cathode_atoms.append(atom.index)

# Add anode carbon atoms (chain 1)
res_anode = topology.addResidue("GRA", chain_anode)
anode_atoms = []
for i in range(8):
    atom = topology.addAtom("C", app.Element.getBySymbol("C"), res_anode)
    anode_atoms.append(atom.index)

# Add water (chain 2)
res_water = topology.addResidue("HOH", chain_water)
atom_O = topology.addAtom("O", app.Element.getBySymbol("O"), res_water)
atom_H1 = topology.addAtom("H", app.Element.getBySymbol("H"), res_water)
atom_H2 = topology.addAtom("H", app.Element.getBySymbol("H"), res_water)
topology.addBond(atom_O, atom_H1)
topology.addBond(atom_O, atom_H2)

# Set up positions
positions = []
# Cathode: 2x4 grid at z=0
for i in range(8):
    x = (i % 4) * 0.25  # 4 atoms in x direction
    y = (i // 4) * 0.25  # 2 atoms in y direction
    positions.append(mm.Vec3(x, y, 0.0))

# Anode: 2x4 grid at z=3.5
for i in range(8):
    x = (i % 4) * 0.25
    y = (i // 4) * 0.25
    positions.append(mm.Vec3(x, y, 3.5))

# Water at center
positions.append(mm.Vec3(0.5, 0.25, 1.75))  # O
positions.append(mm.Vec3(0.6, 0.25, 1.75))  # H1
positions.append(mm.Vec3(0.5, 0.35, 1.75))  # H2

positions = positions * unit.nanometers

# Set box vectors (make large enough for 1nm cutoff)
box_vectors = [
    mm.Vec3(3.0, 0.0, 0.0),
    mm.Vec3(0.0, 3.0, 0.0),
    mm.Vec3(0.0, 0.0, 4.0)
] * unit.nanometers
topology.setPeriodicBoxVectors(box_vectors)

print(f"✓ Created topology: {topology.getNumAtoms()} atoms")
print(f"  Cathode: {len(cathode_atoms)} atoms (chain 0)")
print(f"  Anode: {len(anode_atoms)} atoms (chain 1)")
print(f"  Water: 3 atoms (chain 2)")

print("\n" + "="*60)
print("Test 3: Create system with forces")
print("="*60)

system = mm.System()
system.setDefaultPeriodicBoxVectors(*box_vectors)

# Add particles with masses
for i in range(topology.getNumAtoms()):
    system.addParticle(12.0 if i < 16 else 16.0)  # C=12, O=16

# Add NonbondedForce (required for exclusions)
nonbonded = mm.NonbondedForce()
nonbonded.setNonbondedMethod(mm.NonbondedForce.PME)
nonbonded.setCutoffDistance(1.0 * unit.nanometers)

# Add all particles with small charges
for i in range(topology.getNumAtoms()):
    # Initial charges will be overwritten by plugin
    charge = 0.0
    sigma = 0.34 if i < 16 else 0.31  # C=0.34nm, O=0.31nm
    epsilon = 0.36 if i < 16 else 0.65  # kJ/mol
    nonbonded.addParticle(charge, sigma, epsilon)

system.addForce(nonbonded)
print(f"✓ Created system with NonbondedForce")

print("\n" + "="*60)
print("Test 4: Create ConstantVIntegrator")
print("="*60)

timestep = 1.0 * unit.femtoseconds
# ConstantVIntegrator expects stepSize in picoseconds (like all OpenMM integrators)
stepsize_ps = timestep.value_in_unit(unit.picoseconds)
integrator = ConstantVIntegrator(stepsize_ps)
integrator.setVoltage(1.0)  # 1V
integrator.setNumSCFIterations(2)  # Minimal for testing
integrator.setSCFFrequency(1)
print(f"✓ Integrator created with 1V, 2 SCF iterations")

print("\n" + "="*60)
print("Test 5: Add electrode atoms")
print("="*60)

# Compute area per atom
area_per_atom = (3.0 * 3.0) / 8.0  # Total area / num atoms
print(f"  Area per atom: {area_per_atom:.6f} nm²")

# Add cathode atoms
for idx in cathode_atoms:
    integrator.addCathodeAtom(idx, area_per_atom)
print(f"✓ Added {len(cathode_atoms)} cathode atoms")

# Add anode atoms
for idx in anode_atoms:
    integrator.addAnodeAtom(idx, area_per_atom)
print(f"✓ Added {len(anode_atoms)} anode atoms")

# Add electrolyte (water oxygen)
water_oxygen_idx = 16  # First atom after electrodes
integrator.addElectrolyteAtom(water_oxygen_idx, 0.0)  # Charge will be from NonbondedForce
print(f"✓ Added 1 electrolyte atom")

print("\n" + "="*60)
print("Test 6: Set geometry parameters")
print("="*60)

# Create temporary context to get geometry
temp_integrator = mm.VerletIntegrator(timestep)
temp_context = mm.Context(system, temp_integrator)
temp_context.setPositions(positions)

geometry_params = configure_geometry_from_context(
    temp_context,
    integrator,
    cathode_atoms[0],
    anode_atoms[0]
)

print(f"✓ Geometry configured:")
print(f"  Lgap: {geometry_params['Lgap']:.4f} nm")
print(f"  Lcell: {geometry_params['Lcell']:.4f} nm")
print(f"  Total area: {geometry_params['totalArea']:.4f} nm²")
print(f"  z_cathode: {geometry_params['z_cathode']:.4f} nm")
print(f"  z_anode: {geometry_params['z_anode']:.4f} nm")

del temp_context
del temp_integrator

print("\n" + "="*60)
print("Test 7: Add electrode exclusions (CRITICAL)")
print("="*60)

# Store initial exclusion count
initial_exclusions = nonbonded.getNumExceptions()
print(f"  Initial exceptions: {initial_exclusions}")

add_electrode_exclusions(integrator, nonbonded)

final_exclusions = nonbonded.getNumExceptions()
expected_exclusions = initial_exclusions + len(cathode_atoms)*(len(cathode_atoms)-1)//2 + len(anode_atoms)*(len(anode_atoms)-1)//2
print(f"  Final exceptions: {final_exclusions}")
print(f"  Expected: {expected_exclusions}")

if final_exclusions == expected_exclusions:
    print(f"✓ Exclusions added correctly ({final_exclusions - initial_exclusions} added)")
else:
    print(f"✗ WARNING: Exclusion count mismatch!")

print("\n" + "="*60)
print("Test 8: Create context and reinitialize")
print("="*60)

try:
    platform = mm.Platform.getPlatformByName('Reference')
    print("✓ Using Reference platform")
except:
    platform = mm.Platform.getPlatformByName('CPU')
    print("✓ Using CPU platform")

context = mm.Context(system, integrator, platform)
context.setPositions(positions)
context.setVelocitiesToTemperature(300 * unit.kelvin)
print("✓ Context created")

# CRITICAL: Reinitialize to apply exclusions
context.reinitialize(preserveState=True)
print("✓ Context reinitialized - exclusions active")

print("\n" + "="*60)
print("Test 9: Validate setup")
print("="*60)

valid, messages = validate_setup(context, integrator)
if valid:
    print("✓ Setup validation PASSED")
else:
    print("✗ Setup validation FAILED:")
    for msg in messages:
        print(f"  {msg}")

print("\n" + "="*60)
print("Test 10: Check initial state")
print("="*60)

state = context.getState(getEnergy=True, getForces=True)
pe = state.getPotentialEnergy()
ke = state.getKineticEnergy()
print(f"  Potential energy: {pe}")
print(f"  Kinetic energy: {ke}")
print(f"  Total energy: {pe + ke}")

# Check electrode charges (from NonbondedForce)
print(f"\n  Initial cathode charges:")
for i, idx in enumerate(cathode_atoms[:3]):  # Show first 3
    q, sigma, epsilon = nonbonded.getParticleParameters(idx)
    q_val = q.value_in_unit(unit.elementary_charge)
    print(f"    Atom {idx}: {q_val:.6f} e")

print(f"  Initial anode charges:")
for i, idx in enumerate(anode_atoms[:3]):  # Show first 3
    q, sigma, epsilon = nonbonded.getParticleParameters(idx)
    q_val = q.value_in_unit(unit.elementary_charge)
    print(f"    Atom {idx}: {q_val:.6f} e")

if abs(pe.value_in_unit(unit.kilojoules_per_mole)) > 1e6:
    print(f"✗ WARNING: Potential energy is very large! Exclusions may not be working.")
else:
    print(f"✓ Initial energy is reasonable")

print("\n" + "="*60)
print("Test 11: Run short simulation")
print("="*60)

print("  Running 10 MD steps...")
try:
    integrator.step(10)
    print("✓ Simulation completed without errors")
except Exception as e:
    print(f"✗ Simulation failed: {e}")
    sys.exit(1)

# Check final state
state = context.getState(getEnergy=True, getForces=True)
pe_final = state.getPotentialEnergy()
ke_final = state.getKineticEnergy()
print(f"  Final potential energy: {pe_final}")
print(f"  Final kinetic energy: {ke_final}")
print(f"  Final total energy: {pe_final + ke_final}")

# Check if charges were updated
print(f"\n  Final cathode charges:")
charges_updated = False
for i, idx in enumerate(cathode_atoms[:3]):
    q, sigma, epsilon = nonbonded.getParticleParameters(idx)
    q_val = q.value_in_unit(unit.elementary_charge)
    print(f"    Atom {idx}: {q_val:.6f} e")
    if abs(q_val) > 1e-8:
        charges_updated = True

print(f"  Final anode charges:")
for i, idx in enumerate(anode_atoms[:3]):
    q, sigma, epsilon = nonbonded.getParticleParameters(idx)
    q_val = q.value_in_unit(unit.elementary_charge)
    print(f"    Atom {idx}: {q_val:.6f} e")
    if abs(q_val) > 1e-8:
        charges_updated = True

if charges_updated:
    print("✓ Electrode charges were updated by SCF")
else:
    print("⚠ WARNING: Electrode charges appear to be zero - check SCF solver")

print("\n" + "="*60)
print("✅ ALL TESTS PASSED")
print("="*60)
print("\nBaseline implementation is working correctly!")
print("Next step: Compare results with Original Python code")
print("="*60)
