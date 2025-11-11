#!/usr/bin/env python3
"""
Simple CUDA test - just check if CUDA kernel can be initialized
"""
import constantvplugin
from openmm import *
from openmm.app import *
from openmm.unit import *

print("Testing CUDA ConstantV Plugin Initialization...")

# Create minimal system
system = System()
for i in range(30):  # 10 cathode + 10 anode + 10 electrolyte
    system.addParticle(1.0)

# Add NonbondedForce
nonbonded = NonbondedForce()
nonbonded.setNonbondedMethod(NonbondedForce.CutoffPeriodic)
nonbonded.setCutoffDistance(1.0*nanometer)

for i in range(30):
    nonbonded.addParticle(0.0, 1.0, 0.0)

system.addForce(nonbonded)

# Create ConstantVForce (minimal)
cv_force = constantvplugin.ConstantVForce()
cv_force.setVoltage(1.0)  # Higher voltage to avoid small voltage path
cv_force.setLgap(2.5)
cv_force.setLcell(5.0)
cv_force.setTotalArea(2.0)  # 10 * 0.1 * 2

# Add cathode (10 atoms, indices 0-9)
for i in range(10):
    cv_force.addCathodeAtom(i, 0.1)
cv_force.setZCathode(0.0)

# Add anode (10 atoms, indices 10-19)
for i in range(10, 20):
    cv_force.addAnodeAtom(i, 0.1)
cv_force.setZAnode(2.5)

# Add electrolyte atoms (10 atoms, indices 20-29)
for i in range(20, 30):
    charge = 0.5 if i % 2 == 0 else -0.5
    cv_force.addElectrolyteAtom(i, charge)
    nonbonded.setParticleParameters(i, charge, 1.0, 0.0)

system.addForce(cv_force)

print("System created successfully")

# Test Reference platform first
print("\n1. Testing Reference platform...")
integrator_ref = constantvplugin.ConstantVIntegrator(0.001)
platform_ref = Platform.getPlatformByName('Reference')
context_ref = Context(system, integrator_ref, platform_ref)

positions = []
for i in range(10):
    positions.append(Vec3(0, 0, 0)*nanometer)
for i in range(10):
    positions.append(Vec3(0, 0, 2.5)*nanometer)
for i in range(10):
    z = 0.5 + i * 0.15
    positions.append(Vec3(0, 0, z)*nanometer)

context_ref.setPositions(positions)
context_ref.setPeriodicBoxVectors(
    Vec3(10, 0, 0)*nanometer,
    Vec3(0, 10, 0)*nanometer,
    Vec3(0, 0, 5.0)*nanometer
)

print("   Context created")
state = context_ref.getState(getEnergy=True)
print(f"   Energy: {state.getPotentialEnergy()}")
print("   ✅ Reference platform: OK")
del context_ref

# Test CUDA platform
print("\n2. Testing CUDA platform...")
integrator_cuda = constantvplugin.ConstantVIntegrator(0.001)
platform_cuda = Platform.getPlatformByName('CUDA')
print(f"   Platform: {platform_cuda.getName()}")

try:
    context_cuda = Context(system, integrator_cuda, platform_cuda)
    print("   Context created")

    context_cuda.setPositions(positions)
    context_cuda.setPeriodicBoxVectors(
        Vec3(10, 0, 0)*nanometer,
        Vec3(0, 10, 0)*nanometer,
        Vec3(0, 0, 5.0)*nanometer
    )

    print("   Positions set")
    state = context_cuda.getState(getEnergy=True)
    print(f"   Energy: {state.getPotentialEnergy()}")
    print("   ✅ CUDA platform: OK")
    del context_cuda

except Exception as e:
    print(f"   ❌ CUDA platform error: {e}")
    import traceback
    traceback.print_exc()

print("\nTest complete!")
