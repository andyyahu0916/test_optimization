#!/usr/bin/env python3
"""
Minimal debug test - step by step
"""
import sys
import constantvplugin
from openmm import *
from openmm.app import *
from openmm.unit import *

print("Step 1: Create system...")
system = System()
for i in range(30):
    system.addParticle(1.0)
print("  OK")

print("Step 2: Add NonbondedForce...")
nonbonded = NonbondedForce()
nonbonded.setNonbondedMethod(NonbondedForce.CutoffPeriodic)
nonbonded.setCutoffDistance(1.0*nanometer)
for i in range(30):
    nonbonded.addParticle(0.0, 1.0, 0.0)
system.addForce(nonbonded)
print("  OK")

print("Step 3: Create ConstantVForce...")
cv_force = constantvplugin.ConstantVForce()
print("  OK")

print("Step 4: Set parameters...")
cv_force.setVoltage(4.0)
cv_force.setLgap(2.5)
cv_force.setLcell(5.0)
cv_force.setTotalArea(2.0)
cv_force.setNumIterations(4)
print("  OK")

print("Step 5: Add cathode atoms...")
for i in range(10):
    cv_force.addCathodeAtom(i, 0.1)
cv_force.setZCathode(0.0)
print("  OK")

print("Step 6: Add anode atoms...")
for i in range(10, 20):
    cv_force.addAnodeAtom(i, 0.1)
cv_force.setZAnode(2.5)
print("  OK")

print("Step 7: Add electrolyte atoms...")
for i in range(20, 30):
    charge = 0.5 if i % 2 == 0 else -0.5
    cv_force.addElectrolyteAtom(i, charge)
    nonbonded.setParticleParameters(i, charge, 1.0, 0.0)
print("  OK")

print("Step 8: Add force to system...")
system.addForce(cv_force)
print("  OK")

print("Step 9: Create integrator...")
integrator = VerletIntegrator(0.001*picoseconds)
print("  OK")

print("Step 10: Get Reference platform...")
platform = Platform.getPlatformByName('Reference')
print(f"  Platform: {platform.getName()}")

print("Step 11: Create Context...")
sys.stdout.flush()
context = Context(system, integrator, platform)
print("  OK")

print("Step 12: Set positions...")
positions = []
for i in range(10):
    positions.append(Vec3(0, 0, 0)*nanometer)
for i in range(10):
    positions.append(Vec3(0, 0, 2.5)*nanometer)
for i in range(10):
    z = 0.5 + i * 0.15
    positions.append(Vec3(0, 0, z)*nanometer)
context.setPositions(positions)
print("  OK")

print("Step 13: Set box vectors...")
context.setPeriodicBoxVectors(
    Vec3(10, 0, 0)*nanometer,
    Vec3(0, 10, 0)*nanometer,
    Vec3(0, 0, 5.0)*nanometer
)
print("  OK")

print("Step 14: Get State (trigger Force execution)...")
sys.stdout.flush()
state = context.getState(getEnergy=True, getForces=True)
print("  OK")

print(f"Step 15: Check results...")
print(f"  Energy: {state.getPotentialEnergy()}")
print("  SUCCESS!")
