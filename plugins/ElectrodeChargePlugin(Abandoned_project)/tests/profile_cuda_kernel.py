#!/usr/bin/env python
"""Profile CUDA kernel execution time breakdown."""

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

def create_test_system():
    """Create test system on CUDA."""
    system = System()
    for _ in range(5):
        system.addParticle(40.0)
    
    nonbonded = NonbondedForce()
    nonbonded.setNonbondedMethod(NonbondedForce.NoCutoff)
    for i in range(5):
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
    
    box_vectors = [
        Vec3(1.6, 0.0, 0.0),
        Vec3(0.0, 1.6, 0.0),
        Vec3(0.0, 0.0, 2.0)
    ]
    
    context.setPositions(positions)
    context.setPeriodicBoxVectors(*box_vectors)
    
    return context, nonbonded


print("="*80)
print("CUDA Kernel Profiling")
print("="*80)

context, nonbonded = create_test_system()

# Warmup
for _ in range(3):
    _ = context.getState(getForces=True, groups=1<<1)

# Profile
n_runs = 100
times = []

print(f"\nRunning {n_runs} iterations...")
for i in range(n_runs):
    t0 = time.perf_counter()
    state = context.getState(getForces=True, groups=1<<1)
    t1 = time.perf_counter()
    times.append((t1 - t0) * 1000)  # ms
    
    if (i+1) % 20 == 0:
        print(f"  {i+1}/{n_runs}... avg={np.mean(times[-20:]):.2f}ms")

times = np.array(times)

print("\n--- Results ---")
print(f"Mean time:   {np.mean(times):.3f} ms")
print(f"Median time: {np.median(times):.3f} ms")
print(f"Std dev:     {np.std(times):.3f} ms")
print(f"Min time:    {np.min(times):.3f} ms")
print(f"Max time:    {np.max(times):.3f} ms")

print("\nCharges:")
for i in range(5):
    charge, _, _ = nonbonded.getParticleParameters(i)
    q = charge.value_in_unit(unit.elementary_charge)
    print(f"  Particle {i}: {q:+.6f} e")
