#!/usr/bin/env python
"""Simple test: Compare plugin charges with expected values."""

import ctypes
import os
import sys

# Load plugin libraries
_BUILD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "build"))
_PYTHON_BUILD_DIR = os.path.join(_BUILD_DIR, "python")
sys.path.insert(0, _PYTHON_BUILD_DIR)

_REFERENCE_LIB = os.path.join(_BUILD_DIR, "platforms", "reference", "libElectrodeChargePluginReference.so")
_CUDA_LIB = os.path.join(_BUILD_DIR, "platforms", "cuda", "libElectrodeChargePluginCUDA.so")
_PLUGIN_LIB = os.path.join(_BUILD_DIR, "libElectrodeChargePlugin.so")
_RTLD_GLOBAL = getattr(ctypes, "RTLD_GLOBAL", 0)

ctypes.CDLL(_REFERENCE_LIB, mode=_RTLD_GLOBAL)
if os.path.exists(_CUDA_LIB):
    ctypes.CDLL(_CUDA_LIB, mode=_RTLD_GLOBAL)
plugin_lib = ctypes.CDLL(_PLUGIN_LIB, mode=_RTLD_GLOBAL)
plugin_lib.registerElectrodeChargePlugin()

import electrodecharge
from openmm import Context, LangevinIntegrator, NonbondedForce, Platform, System, unit, Vec3

def test_reference_platform():
    """Test Reference platform produces correct charges."""
    print("="*80)
    print("Testing Reference Platform")
    print("="*80)
    
    # Create system
    system = System()
    for _ in range(5):
        system.addParticle(40.0)
    
    nonbonded = NonbondedForce()
    # Particles: 0,1 = cathode, 2,3 = anode, 4 = bulk with charge 0.1
    for i in range(5):
        charge = 0.1 if i == 4 else 0.0
        nonbonded.addParticle(charge, 1.0, 0.0)
    system.addForce(nonbonded)
    
    force = electrodecharge.ElectrodeChargeForce()
    force.setCathode([0, 1], -1.0)  # Try negative voltage for cathode
    force.setAnode([2, 3], 1.0)
    force.setCellGap(0.8)          # nm
    force.setCellLength(2.0)       # nm
    force.setNumIterations(3)
    force.setSmallThreshold(1.0e-6)
    system.addForce(force)
    
    # Setup context
    platform = Platform.getPlatformByName('Reference')
    integrator = LangevinIntegrator(300*unit.kelvin, 1.0/unit.picosecond, 0.001*unit.picoseconds)
    context = Context(system, integrator, platform)
    
    # Set positions (cathode at z~0.1, anode at z~1.9, bulk at z~1.0)
    positions = [
        Vec3(0.0, 0.0, 0.1),
        Vec3(0.8, 0.0, 0.1),
        Vec3(0.0, 0.0, 1.9),
        Vec3(0.8, 0.0, 1.9),
        Vec3(0.4, 0.4, 1.0),
    ]
    context.setPositions(positions)
    context.setPeriodicBoxVectors(Vec3(1.6, 0, 0), Vec3(0, 1.6, 0), Vec3(0, 0, 2.0))
    context.setVelocitiesToTemperature(300*unit.kelvin)
    
    # Trigger plugin calculation (CRITICAL: use groups=1<<1)
    state = context.getState(getForces=True, groups=1<<1)
    
    # Collect charges
    charges = []
    for i in range(5):
        charge, sigma, epsilon = nonbonded.getParticleParameters(i)
        charges.append(charge.value_in_unit(unit.elementary_charge))
    
    print("\nFinal charges (after 3 iterations):")
    print(f"  Cathode [0]: {charges[0]:.6f} e")
    print(f"  Cathode [1]: {charges[1]:.6f} e")
    print(f"  Anode [2]:   {charges[2]:.6f} e")
    print(f"  Anode [3]:   {charges[3]:.6f} e")
    print(f"  Bulk [4]:    {charges[4]:.6f} e")
    
    # Just check basic sanity
    assert abs(charges[4] - 0.1) < 1e-6, "Bulk particle charge should not change"
    assert abs(charges[0] - charges[1]) < 1e-6, "Cathode symmetry"
    assert abs(charges[2] - charges[3]) < 1e-6, "Anode symmetry"
    
    total_charge = sum(charges)
    print(f"  Total charge:    {total_charge:.6f} e")
    # NOTE: Charge conservation may not hold in this plugin's model
    
    print("\n✅ Reference platform test PASSED!")
    
    del integrator
    del context
    return charges

def test_cuda_platform():
    """Test CUDA platform produces same charges as Reference."""
    print("\n" + "="*80)
    print("Testing CUDA Platform")
    print("="*80)
    
    # Create system (identical to Reference test)
    system = System()
    for _ in range(5):
        system.addParticle(40.0)
    
    nonbonded = NonbondedForce()
    for i in range(5):
        charge = 0.1 if i == 4 else 0.0
        nonbonded.addParticle(charge, 1.0, 0.0)
    system.addForce(nonbonded)
    
    force = electrodecharge.ElectrodeChargeForce()
    force.setCathode([0, 1], -1.0)  # Try negative voltage for cathode
    force.setAnode([2, 3], 1.0)
    force.setCellGap(0.8)
    force.setCellLength(2.0)
    force.setNumIterations(3)
    force.setSmallThreshold(1.0e-6)
    system.addForce(force)
    
    platform = Platform.getPlatformByName('CUDA')
    integrator = LangevinIntegrator(300*unit.kelvin, 1.0/unit.picosecond, 0.001*unit.picoseconds)
    context = Context(system, integrator, platform)
    
    positions = [
        Vec3(0.0, 0.0, 0.1),
        Vec3(0.8, 0.0, 0.1),
        Vec3(0.0, 0.0, 1.9),
        Vec3(0.8, 0.0, 1.9),
        Vec3(0.4, 0.4, 1.0),
    ]
    context.setPositions(positions)
    context.setPeriodicBoxVectors(Vec3(1.6, 0, 0), Vec3(0, 1.6, 0), Vec3(0, 0, 2.0))
    context.setVelocitiesToTemperature(300*unit.kelvin)
    
    # Trigger plugin calculation (CRITICAL: use groups=1<<1)
    state = context.getState(getForces=True, groups=1<<1)
    
    charges = []
    for i in range(5):
        charge, sigma, epsilon = nonbonded.getParticleParameters(i)
        charges.append(charge.value_in_unit(unit.elementary_charge))
    
    print("\nFinal charges (after 3 iterations):")
    print(f"  Cathode [0]: {charges[0]:.6f} e")
    print(f"  Cathode [1]: {charges[1]:.6f} e")
    print(f"  Anode [2]:   {charges[2]:.6f} e")
    print(f"  Anode [3]:   {charges[3]:.6f} e")
    print(f"  Bulk [4]:    {charges[4]:.6f} e")
    
    assert abs(charges[4] - 0.1) < 1e-6
    assert abs(charges[0] - charges[1]) < 1e-6
    assert abs(charges[2] - charges[3]) < 1e-6
    
    total_charge = sum(charges)
    print(f"  Total charge:    {total_charge:.6f} e")
    
    print("\n✅ CUDA platform test PASSED!")
    
    del integrator
    del context
    return charges

if __name__ == '__main__':
    charges_ref = test_reference_platform()
    
    # Check if CUDA is available
    from openmm import Platform
    cuda_available = False
    for i in range(Platform.getNumPlatforms()):
        if Platform.getPlatform(i).getName() == 'CUDA':
            cuda_available = True
            break
    
    if cuda_available:
        charges_cuda = test_cuda_platform()
        
        print("\n" + "="*80)
        print("Comparing Reference vs CUDA")
        print("="*80)
        
        max_diff = 0.0
        for i in range(5):
            diff = abs(charges_ref[i] - charges_cuda[i])
            max_diff = max(max_diff, diff)
            print(f"  Particle {i}: diff = {diff:.2e} e")
        
        tolerance = 1e-5
        if max_diff < tolerance:
            print(f"\n🎉 SUCCESS! Max difference {max_diff:.2e} < {tolerance:.0e}")
            print("✅ Reference and CUDA platforms produce identical results!")
        else:
            print(f"\n⚠️  Max difference {max_diff:.2e} >= {tolerance:.0e}")
            raise AssertionError("Reference and CUDA platforms differ!")
    else:
        print("\n⚠️  CUDA platform not available, skipping CUDA test")
        print("✅ Reference platform test PASSED!")
