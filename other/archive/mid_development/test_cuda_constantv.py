#!/usr/bin/env python3
"""
Test CUDA ConstantV Plugin - Verify physics correctness vs Reference platform
"""
import constantvplugin
from openmm import *
from openmm.app import *
from openmm.unit import *
import numpy as np
import time

print("="*70)
print("🚀 CUDA ConstantV Plugin Physics Test")
print("="*70)

# Test parameters
voltage_value = 4.0  # V
Lgap_value = 2.5  # nm
Lcell_value = 5.0  # nm
num_cathode = 10
num_anode = 10
num_electrolyte = 100
area_per_atom = 0.1  # nm²

print(f"\n📋 Test Configuration:")
print(f"  Voltage: {voltage_value} V")
print(f"  Lgap: {Lgap_value} nm")
print(f"  Lcell: {Lcell_value} nm")
print(f"  Cathode atoms: {num_cathode}")
print(f"  Anode atoms: {num_anode}")
print(f"  Electrolyte atoms: {num_electrolyte}")
print(f"  Area per atom: {area_per_atom} nm²")

def create_test_system(platform_name):
    """Create a test system with ConstantVForce"""
    # Create system
    system = System()
    total_atoms = num_cathode + num_anode + num_electrolyte
    for i in range(total_atoms):
        system.addParticle(1.0)  # 1 amu

    # Add NonbondedForce
    nonbonded = NonbondedForce()
    nonbonded.setNonbondedMethod(NonbondedForce.CutoffPeriodic)
    nonbonded.setCutoffDistance(1.0*nanometer)

    # Add all particles
    for i in range(total_atoms):
        nonbonded.addParticle(0.0, 1.0, 0.0)  # Initial charge=0, sigma=1, epsilon=0

    system.addForce(nonbonded)

    # Create ConstantVForce
    cv_force = constantvplugin.ConstantVForce()
    cv_force.setVoltage(voltage_value)
    cv_force.setLgap(Lgap_value)
    cv_force.setLcell(Lcell_value)

    # Calculate total area
    total_area = area_per_atom * (num_cathode + num_anode)
    cv_force.setTotalArea(total_area)

    # Add cathode atoms (indices 0-9)
    z_cathode = 0.0
    for i in range(num_cathode):
        cv_force.addCathodeAtom(i, area_per_atom)
    cv_force.setZCathode(z_cathode)

    # Add anode atoms (indices 10-19)
    z_anode = Lgap_value
    for i in range(num_cathode, num_cathode + num_anode):
        cv_force.addAnodeAtom(i, area_per_atom)
    cv_force.setZAnode(z_anode)

    # Add electrolyte atoms (indices 20-119)
    for i in range(num_cathode + num_anode, total_atoms):
        charge = 0.5 if i % 2 == 0 else -0.5  # Alternating charges
        cv_force.addElectrolyteAtom(i, charge)
        nonbonded.setParticleParameters(i, charge, 1.0, 0.0)

    system.addForce(cv_force)

    # Create integrator (dummy, just for context)
    integrator = VerletIntegrator(0.001*picoseconds)

    # Create platform
    platform = Platform.getPlatformByName(platform_name)

    # Create context
    context = Context(system, integrator, platform)

    # Set positions (simple linear distribution along Z)
    positions = []
    for i in range(num_cathode):
        positions.append(Vec3(0, 0, z_cathode)*nanometer)
    for i in range(num_anode):
        positions.append(Vec3(0, 0, z_anode)*nanometer)
    for i in range(num_electrolyte):
        z = z_cathode + (i+1) * Lgap_value / (num_electrolyte + 1)
        positions.append(Vec3(0, 0, z)*nanometer)

    context.setPositions(positions)
    context.setPeriodicBoxVectors(
        Vec3(10, 0, 0)*nanometer,
        Vec3(0, 10, 0)*nanometer,
        Vec3(0, 0, Lcell_value)*nanometer
    )

    return system, context, cv_force, nonbonded

def get_electrode_charges(context, nonbonded, num_cathode, num_anode):
    """Extract electrode charges from NonbondedForce"""
    cathode_charges = []
    anode_charges = []

    for i in range(num_cathode):
        q, sigma, epsilon = nonbonded.getParticleParameters(i)
        cathode_charges.append(q / elementary_charge)

    for i in range(num_cathode, num_cathode + num_anode):
        q, sigma, epsilon = nonbonded.getParticleParameters(i)
        anode_charges.append(q / elementary_charge)

    return cathode_charges, anode_charges

def test_platform(platform_name):
    """Test ConstantV on a specific platform"""
    print(f"\n{'='*70}")
    print(f"Testing Platform: {platform_name}")
    print(f"{'='*70}")

    # Create system
    system, context, cv_force, nonbonded = create_test_system(platform_name)

    # Run a few MD steps with ConstantVIntegrator
    integrator2 = constantvplugin.ConstantVIntegrator(0.001)  # timestep (ps)

    platform = Platform.getPlatformByName(platform_name)
    context2 = Context(system, integrator2, platform)
    context2.setPositions(context.getState(getPositions=True).getPositions())
    context2.setPeriodicBoxVectors(
        Vec3(10, 0, 0)*nanometer,
        Vec3(0, 10, 0)*nanometer,
        Vec3(0, 0, Lcell_value)*nanometer
    )

    print(f"\n📊 Running 5 MD steps...")
    results = []

    start_time = time.time()
    for step in range(5):
        integrator2.step(1)
        state = context2.getState(getEnergy=True)

        # Get electrode charges
        cathode_q, anode_q = get_electrode_charges(context2, nonbonded, num_cathode, num_anode)

        Q_cathode = sum(cathode_q)
        Q_anode = sum(anode_q)
        Q_total = Q_cathode + Q_anode

        energy = state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)

        print(f"  Step {step}: Q_cathode={Q_cathode:+.6f}e, Q_anode={Q_anode:+.6f}e, "
              f"Q_total={Q_total:.8f}e, E={energy:.2f} kJ/mol")

        results.append({
            'step': step,
            'Q_cathode': Q_cathode,
            'Q_anode': Q_anode,
            'Q_total': Q_total,
            'energy': energy
        })

    elapsed = time.time() - start_time
    print(f"\n⏱️  Time: {elapsed*1000:.2f} ms total, {elapsed*1000/5:.2f} ms/step")

    # Verify charge conservation
    max_total_charge = max(abs(r['Q_total']) for r in results)
    print(f"\n🔬 Physics Verification:")
    print(f"  Max |Q_total|: {max_total_charge:.2e}")
    if max_total_charge < 1e-6:
        print(f"  ✅ Charge conservation: PASSED")
    else:
        print(f"  ❌ Charge conservation: FAILED (threshold 1e-6)")

    del context2
    del context

    return results

# Test Reference platform
print("\n" + "="*70)
print("Test 1: Reference Platform (Baseline)")
print("="*70)
results_ref = test_platform("Reference")

# Test CUDA platform
print("\n" + "="*70)
print("Test 2: CUDA Platform")
print("="*70)
results_cuda = test_platform("CUDA")

# Compare results
print("\n" + "="*70)
print("📊 Comparison: CUDA vs Reference")
print("="*70)

print("\n| Step | Q_cathode Diff | Q_anode Diff | Q_total Diff | Energy Diff |")
print("|------|----------------|--------------|--------------|-------------|")

max_cathode_diff = 0
max_anode_diff = 0
max_total_diff = 0

for i in range(5):
    ref = results_ref[i]
    cuda = results_cuda[i]

    cathode_diff = abs(cuda['Q_cathode'] - ref['Q_cathode'])
    anode_diff = abs(cuda['Q_anode'] - ref['Q_anode'])
    total_diff = abs(cuda['Q_total'] - ref['Q_total'])
    energy_diff = abs(cuda['energy'] - ref['energy'])

    max_cathode_diff = max(max_cathode_diff, cathode_diff)
    max_anode_diff = max(max_anode_diff, anode_diff)
    max_total_diff = max(max_total_diff, total_diff)

    print(f"| {i:4d} | {cathode_diff:.2e} | {anode_diff:.2e} | "
          f"{total_diff:.2e} | {energy_diff:.2f} |")

print("\n🎯 Maximum Differences:")
print(f"  Cathode charge: {max_cathode_diff:.2e}")
print(f"  Anode charge: {max_anode_diff:.2e}")
print(f"  Total charge: {max_total_diff:.2e}")

# Final verdict
threshold = 1e-5
print("\n" + "="*70)
if max_cathode_diff < threshold and max_anode_diff < threshold and max_total_diff < threshold:
    print("🎉 CUDA Implementation: ✅ PASSED")
    print(f"   All differences < {threshold:.0e} (excellent agreement!)")
else:
    print("⚠️  CUDA Implementation: Differences detected")
    print(f"   Some differences > {threshold:.0e}")
print("="*70)
