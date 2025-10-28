#!/usr/bin/env python
"""Test CUDA platform loading and basic functionality."""

import ctypes
import os
import sys
import numpy as np

_BUILD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "build"))
_PYTHON_BUILD_DIR = os.path.join(_BUILD_DIR, "python")
sys.path.insert(0, _PYTHON_BUILD_DIR)

_REFERENCE_LIB = os.path.join(_BUILD_DIR, "platforms", "reference", "libElectrodeChargePluginReference.so")
_CUDA_LIB = os.path.join(_BUILD_DIR, "platforms", "cuda", "libElectrodeChargePluginCUDA.so")
_PLUGIN_LIB = os.path.join(_BUILD_DIR, "libElectrodeChargePlugin.so")
_RTLD_GLOBAL = getattr(ctypes, "RTLD_GLOBAL", 0)

print("Loading libraries...")
print(f"  Reference: {os.path.exists(_REFERENCE_LIB)}")
print(f"  CUDA:      {os.path.exists(_CUDA_LIB)}")
print(f"  Plugin:    {os.path.exists(_PLUGIN_LIB)}")

ctypes.CDLL(_REFERENCE_LIB, mode=_RTLD_GLOBAL)
if os.path.exists(_CUDA_LIB):
    print("Loading CUDA library...")
    try:
        ctypes.CDLL(_CUDA_LIB, mode=_RTLD_GLOBAL)
        print("✅ CUDA library loaded successfully")
    except Exception as e:
        print(f"❌ CUDA library loading failed: {e}")
        sys.exit(1)
else:
    print("❌ CUDA library not found!")
    sys.exit(1)

plugin_lib = ctypes.CDLL(_PLUGIN_LIB, mode=_RTLD_GLOBAL)
print("✅ Plugin library loaded")

import electrodecharge
from openmm import Context, LangevinIntegrator, NonbondedForce, Platform, System, unit, Vec3

# Load OpenMM CUDA platform
print("\nLoading OpenMM plugins...")
Platform.loadPluginsFromDirectory('/home/andy/miniforge3/envs/cuda/lib/plugins')

# NOW register our plugin (AFTER OpenMM platforms are loaded)
plugin_lib.registerElectrodeChargePlugin()
print("✅ ElectrodeCharge plugin registered")

print("\nChecking available platforms:")
for i in range(Platform.getNumPlatforms()):
    platform = Platform.getPlatform(i)
    print(f"  {i}: {platform.getName()}")

# Try to get CUDA platform
try:
    cuda_platform = Platform.getPlatformByName('CUDA')
    print(f"\n✅ CUDA platform found: {cuda_platform.getName()}")
    print(f"   Speed: {cuda_platform.getSpeed()}")
except Exception as e:
    print(f"\n❌ CUDA platform not available: {e}")
    sys.exit(1)

# Create simple test system
print("\nCreating test system...")
system = System()
for _ in range(5):
    system.addParticle(40.0)

nonbonded = NonbondedForce()
nonbonded.setNonbondedMethod(NonbondedForce.NoCutoff)
for i in range(5):
    # CRITICAL: Initialize with non-zero charges to enable Coulomb on CUDA
    # If all charges are 0 at Context creation, CUDA kernel skips Coulomb interactions permanently
    q_init = 1.0e-10 if i < 4 else 0.0  # Tiny charge on electrodes
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

# Test CUDA platform
print("\nTesting CUDA platform context creation...")
try:
    integrator = LangevinIntegrator(300*unit.kelvin, 1.0/unit.picosecond, 0.001*unit.picoseconds)
    context = Context(system, integrator, cuda_platform)
    print("✅ CUDA context created successfully")
    
    context.setPositions(positions)
    context.setPeriodicBoxVectors(*box_vectors)
    print("✅ Positions and box vectors set")
    
    # Get state and check charges
    state = context.getState(getForces=True, groups=1<<1)
    print("✅ State obtained with force group isolation")
    
    # Check charges
    for i in range(5):
        charge, _, _ = nonbonded.getParticleParameters(i)
        q = charge.value_in_unit(unit.elementary_charge)
        print(f"  Particle {i}: {q:+.6f} e")
    
    print("\n🎉 CUDA platform test PASSED!")
    
except Exception as e:
    print(f"\n❌ CUDA platform test FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
