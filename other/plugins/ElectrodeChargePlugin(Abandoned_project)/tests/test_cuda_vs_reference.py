#!/usr/bin/env python
"""Test CUDA vs Reference: numerical accuracy."""

import ctypes
import os
import sys
import numpy as np
import time

_BUILD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "build"))
_PYTHON_BUILD_DIR = os.path.join(_BUILD_DIR, "python")
sys.path.insert(0, _PYTHON_BUILD_DIR)

_REFERENCE_LIB = os.path.join(_BUILD_DIR, "platforms", "reference", "libElectrodeChargePluginReference.so")
_CUDA_LIB = os.path.join(_BUILD_DIR, "platforms", "cuda", "libElectrodeChargePluginCUDA.so")
_PLUGIN_LIB = os.path.join(_BUILD_DIR, "libElectrodeChargePlugin.so")
_RTLD_GLOBAL = getattr(ctypes, "RTLD_GLOBAL", 0)

ctypes.CDLL(_REFERENCE_LIB, mode=_RTLD_GLOBAL)
ctypes.CDLL(_CUDA_LIB, mode=_RTLD_GLOBAL)
plugin_lib = ctypes.CDLL(_PLUGIN_LIB, mode=_RTLD_GLOBAL)

import electrodecharge
from openmm import Context, LangevinIntegrator, NonbondedForce, Platform, System, unit, Vec3

# Load OpenMM plugins
Platform.loadPluginsFromDirectory('/home/andy/miniforge3/envs/cuda/lib/plugins')
plugin_lib.registerElectrodeChargePlugin()

def create_test_system(platform_name):
    """Create system for specified platform."""
    system = System()
    for _ in range(5):
        system.addParticle(40.0)
    
    nonbonded = NonbondedForce()
    nonbonded.setNonbondedMethod(NonbondedForce.NoCutoff)
    for i in range(5):
        # Non-zero init charges for CUDA compatibility
        q_init = 1.0e-10 if i < 4 else 0.1
        nonbonded.addParticle(q_init, 1.0, 0.0)
    system.addForce(nonbonded)
    
    force = electrodecharge.ElectrodeChargeForce()
    force.setCathode([0, 1], 0.5)
    force.setAnode([2, 3], 0.5)
    force.setCellGap(0.8)
    force.setCellLength(2.0)
    force.setNumIterations(3)
    force.setSmallThreshold(1.0e-6)
    system.addForce(force)
    
    platform = Platform.getPlatformByName(platform_name)
    integrator = LangevinIntegrator(300*unit.kelvin, 1.0/unit.picosecond, 0.001*unit.picoseconds)
    context = Context(system, integrator, platform)
    
    positions = [
        Vec3(0.0, 0.0, 0.1),
        Vec3(0.8, 0.0, 0.1),
        Vec3(0.0, 0.0, 1.9),
        Vec3(0.8, 0.0, 1.9),
        Vec3(0.4, 0.4, 1.0),
    ]
    
    box_vectors = [
        Vec3(1.6, 0.0, 0.0),
        Vec3(0.0, 1.6, 0.0),
        Vec3(0.0, 0.0, 2.0)
    ]
    
    context.setPositions(positions)
    context.setPeriodicBoxVectors(*box_vectors)
    
    return context, nonbonded


def get_charges(context, nonbonded, num_particles):
    """Extract charges from context."""
    state = context.getState(getForces=True, groups=1<<1)
    charges = []
    for i in range(num_particles):
        charge, _, _ = nonbonded.getParticleParameters(i)
        charges.append(charge.value_in_unit(unit.elementary_charge))
    return np.array(charges)


def main():
    print("="*80)
    print("CUDA vs Reference Platform Test")
    print("="*80)
    
    # Test Reference
    print("\n--- Reference Platform ---")
    t0 = time.time()
    context_ref, nonbonded_ref = create_test_system('Reference')
    charges_ref = get_charges(context_ref, nonbonded_ref, 5)
    t_ref = time.time() - t0
    
    print("Charges:")
    for i, q in enumerate(charges_ref):
        print(f"  Particle {i}: {q:+.6f} e")
    print(f"Time: {t_ref*1000:.2f} ms")
    
    # Test CUDA
    print("\n--- CUDA Platform ---")
    t0 = time.time()
    context_cuda, nonbonded_cuda = create_test_system('CUDA')
    charges_cuda = get_charges(context_cuda, nonbonded_cuda, 5)
    t_cuda = time.time() - t0
    
    print("Charges:")
    for i, q in enumerate(charges_cuda):
        print(f"  Particle {i}: {q:+.6f} e")
    print(f"Time: {t_cuda*1000:.2f} ms")
    
    # Compare
    print("\n--- Comparison ---")
    max_diff = np.max(np.abs(charges_ref - charges_cuda))
    rms_diff = np.sqrt(np.mean((charges_ref - charges_cuda)**2))
    speedup = t_ref / t_cuda
    
    print(f"Max absolute difference: {max_diff:.2e} e")
    print(f"RMS difference:          {rms_diff:.2e} e")
    print(f"Speedup:                 {speedup:.2f}×")
    
    tolerance = 1e-5
    if max_diff < tolerance:
        print(f"\n✅ SUCCESS! CUDA matches Reference (tolerance < {tolerance})")
        print("✅ CUDA platform validation PASSED!")
        return 0
    else:
        print(f"\n❌ FAILED! Difference {max_diff:.2e} exceeds tolerance {tolerance}")
        print("\nDetailed differences:")
        for i in range(5):
            diff = charges_cuda[i] - charges_ref[i]
            if abs(diff) > tolerance:
                print(f"  Particle {i}: Ref={charges_ref[i]:+.6f}, CUDA={charges_cuda[i]:+.6f}, Δ={diff:+.2e}")
        return 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
