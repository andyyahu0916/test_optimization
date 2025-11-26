#!/usr/bin/env python3
"""
Simple test: Verify electrode exclusions are added correctly
"""

import sys
import openmm as mm
import openmm.app as app
import openmm.unit as unit
from constantvplugin import ConstantVIntegrator
from constantvplugin_helpers import add_electrode_exclusions

print("="*60)
print("Test: Electrode Exclusions")
print("="*60)

# Create minimal system: 4 cathode atoms, 4 anode atoms
system = mm.System()
for i in range(8):
    system.addParticle(12.0)  # 8 carbon atoms

# Add NonbondedForce
nonbonded = mm.NonbondedForce()
nonbonded.setNonbondedMethod(mm.NonbondedForce.NoCutoff)
for i in range(8):
    nonbonded.addParticle(0.0, 0.34, 0.36)
system.addForce(nonbonded)

# Create integrator
integrator = ConstantVIntegrator(0.001)
integrator.setVoltage(1.0)
integrator.setLgap(3.5)
integrator.setLcell(4.0)
integrator.setTotalArea(1.0)
integrator.setZCathode(0.0)
integrator.setZAnode(3.5)

# Add cathode atoms (0-3)
cathode_atoms = [0, 1, 2, 3]
for idx in cathode_atoms:
    integrator.addCathodeAtom(idx, 0.25)

# Add anode atoms (4-7)
anode_atoms = [4, 5, 6, 7]
for idx in anode_atoms:
    integrator.addAnodeAtom(idx, 0.25)

print(f"Cathode atoms: {cathode_atoms}")
print(f"Anode atoms: {anode_atoms}")

# Check initial exceptions
initial_exceptions = nonbonded.getNumExceptions()
print(f"\nInitial NonbondedForce exceptions: {initial_exceptions}")

# Add exclusions
print("\nAdding electrode exclusions...")
add_electrode_exclusions(integrator, nonbonded)

# Check final exceptions
final_exceptions = nonbonded.getNumExceptions()
cathode_pairs = len(cathode_atoms) * (len(cathode_atoms) - 1) // 2
anode_pairs = len(anode_atoms) * (len(anode_atoms) - 1) // 2
expected_total = initial_exceptions + cathode_pairs + anode_pairs

print(f"\nFinal NonbondedForce exceptions: {final_exceptions}")
print(f"Expected: {expected_total}")
print(f"  Cathode pairs: {cathode_pairs}")
print(f"  Anode pairs: {anode_pairs}")

# Verify each exception
print(f"\nVerifying exceptions:")
found_cathode = set()
found_anode = set()

for i in range(final_exceptions):
    p1, p2, chargeProd, sigma, epsilon = nonbonded.getExceptionParameters(i)

    # Check if it's a cathode-cathode exception
    if p1 in cathode_atoms and p2 in cathode_atoms:
        found_cathode.add((min(p1,p2), max(p1,p2)))
        print(f"  Cathode-cathode: {p1}-{p2}, ε={epsilon}")
        if epsilon != 0.0:
            print(f"    ⚠ WARNING: epsilon should be 0.0, got {epsilon}")

    # Check if it's an anode-anode exception
    if p1 in anode_atoms and p2 in anode_atoms:
        found_anode.add((min(p1,p2), max(p1,p2)))
        print(f"  Anode-anode: {p1}-{p2}, ε={epsilon}")
        if epsilon != 0.0:
            print(f"    ⚠ WARNING: epsilon should be 0.0, got {epsilon}")

print(f"\nCathode-cathode exclusions found: {len(found_cathode)}/{cathode_pairs}")
print(f"Anode-anode exclusions found: {len(found_anode)}/{anode_pairs}")

if len(found_cathode) == cathode_pairs and len(found_anode) == anode_pairs:
    print("\n✅ All exclusions added correctly!")
else:
    print("\n❌ Exclusions are MISSING!")
    sys.exit(1)
