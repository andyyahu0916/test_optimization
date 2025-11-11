#!/usr/bin/env python3
"""
Test ConstantVForce directly (不用Integrator)
"""
import constantvplugin
from openmm import *
from openmm.app import *
from openmm.unit import *

print("Testing ConstantVForce (not Integrator)...")

# Create minimal system
system = System()
for i in range(30):
    system.addParticle(1.0)

# Add NonbondedForce
nonbonded = NonbondedForce()
nonbonded.setNonbondedMethod(NonbondedForce.CutoffPeriodic)
nonbonded.setCutoffDistance(1.0*nanometer)

for i in range(30):
    nonbonded.addParticle(0.0, 1.0, 0.0)

system.addForce(nonbonded)

# Create ConstantVForce
cv_force = constantvplugin.ConstantVForce()
cv_force.setVoltage(4.0)  # 4.0 V
cv_force.setLgap(2.5)
cv_force.setLcell(5.0)
cv_force.setTotalArea(2.0)
cv_force.setNumIterations(4)

# Add cathode (10 atoms)
for i in range(10):
    cv_force.addCathodeAtom(i, 0.1)
cv_force.setZCathode(0.0)

# Add anode (10 atoms)
for i in range(10, 20):
    cv_force.addAnodeAtom(i, 0.1)
cv_force.setZAnode(2.5)

# Add electrolyte (10 atoms)
for i in range(20, 30):
    charge = 0.5 if i % 2 == 0 else -0.5
    cv_force.addElectrolyteAtom(i, charge)
    nonbonded.setParticleParameters(i, charge, 1.0, 0.0)

system.addForce(cv_force)

print("System created")

# Test Reference platform
print("\n1. Testing Reference platform with ConstantVForce...")
integrator_ref = VerletIntegrator(0.001*picoseconds)  # 用普通integrator
platform_ref = Platform.getPlatformByName('Reference')

try:
    context_ref = Context(system, integrator_ref, platform_ref)
    print("   Context created")

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

    print("   Positions set")

    # Force ConstantVForce to execute
    state = context_ref.getState(getEnergy=True, getForces=True)
    print(f"   Energy: {state.getPotentialEnergy()}")

    # Check electrode charges
    Q_cathode = 0.0
    Q_anode = 0.0
    for i in range(10):
        q, s, e = nonbonded.getParticleParameters(i)
        Q_cathode += q / elementary_charge
    for i in range(10, 20):
        q, s, e = nonbonded.getParticleParameters(i)
        Q_anode += q / elementary_charge

    print(f"   Q_cathode: {Q_cathode:+.6f}e")
    print(f"   Q_anode: {Q_anode:+.6f}e")
    print(f"   Q_total: {Q_cathode + Q_anode:.8f}e")

    print("   ✅ Reference platform: OK")
    del context_ref

except Exception as e:
    print(f"   ❌ Reference platform error: {e}")
    import traceback
    traceback.print_exc()

print("\nTest complete!")
