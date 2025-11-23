#!/usr/bin/env python
"""Test iteration count impact on performance."""
import ctypes, os, sys, time

_BUILD_DIR = "/home/andy/test_optimization/plugins/ElectrodeChargePlugin/build"
_PYTHON_BUILD_DIR = os.path.join(_BUILD_DIR, "python")
sys.path.insert(0, _PYTHON_BUILD_DIR)

_REFERENCE_LIB = os.path.join(_BUILD_DIR, "platforms/reference/libElectrodeChargePluginReference.so")
_CUDA_LIB = os.path.join(_BUILD_DIR, "platforms/cuda/libElectrodeChargePluginCUDA.so")
_PLUGIN_LIB = os.path.join(_BUILD_DIR, "libElectrodeChargePlugin.so")

ctypes.CDLL(_REFERENCE_LIB, mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL(_CUDA_LIB, mode=ctypes.RTLD_GLOBAL)
plugin_lib = ctypes.CDLL(_PLUGIN_LIB, mode=ctypes.RTLD_GLOBAL)

import electrodecharge
from openmm import *

Platform.loadPluginsFromDirectory('/home/andy/miniforge3/envs/cuda/lib/plugins')
plugin_lib.registerElectrodeChargePlugin()

system = System()
for _ in range(5):
    system.addParticle(40.0)

nonbonded = NonbondedForce()
nonbonded.setNonbondedMethod(NonbondedForce.NoCutoff)
for i in range(5):
    nonbonded.addParticle(1e-10 if i<4 else 0.1, 1.0, 0.0)
system.addForce(nonbonded)

force = electrodecharge.ElectrodeChargeForce()
force.setCathode([0, 1], 0.5)
force.setAnode([2, 3], 0.5)
force.setCellGap(0.8)
force.setCellLength(2.0)
force.setNumIterations(1)
force.setSmallThreshold(1.0e-6)
system.addForce(force)

platform = Platform.getPlatformByName('CUDA')
integrator = LangevinIntegrator(300*unit.kelvin, 1.0/unit.picosecond, 0.001*unit.picoseconds)
context = Context(system, integrator, platform)
context.setPositions([Vec3(0,0,0.1), Vec3(0.8,0,0.1), Vec3(0,0,1.9), Vec3(0.8,0,1.9), Vec3(0.4,0.4,1.0)])
context.setPeriodicBoxVectors(Vec3(1.6,0,0), Vec3(0,1.6,0), Vec3(0,0,2.0))

# Test different iteration counts
for n_iter in [1, 3, 5, 10]:
    print(f"\nTesting {n_iter} iterations...")
    
    # Recreate system with new iteration count
    system = System()
    for _ in range(5):
        system.addParticle(40.0)
    
    nonbonded = NonbondedForce()
    nonbonded.setNonbondedMethod(NonbondedForce.NoCutoff)
    for i in range(5):
        nonbonded.addParticle(1e-10 if i<4 else 0.1, 1.0, 0.0)
    system.addForce(nonbonded)
    
    force = electrodecharge.ElectrodeChargeForce()
    force.setCathode([0, 1], 0.5)
    force.setAnode([2, 3], 0.5)
    force.setCellGap(0.8)
    force.setCellLength(2.0)
    force.setNumIterations(n_iter)
    force.setSmallThreshold(1.0e-6)
    system.addForce(force)
    
    integrator = LangevinIntegrator(300*unit.kelvin, 1.0/unit.picosecond, 0.001*unit.picoseconds)
    context = Context(system, integrator, platform)
    context.setPositions([Vec3(0,0,0.1), Vec3(0.8,0,0.1), Vec3(0,0,1.9), Vec3(0.8,0,1.9), Vec3(0.4,0.4,1.0)])
    context.setPeriodicBoxVectors(Vec3(1.6,0,0), Vec3(0,1.6,0), Vec3(0,0,2.0))
    
    # Warmup
    for _ in range(3):
        _ = context.getState(getForces=True, groups=1<<1)
    
    # Measure
    t0 = time.time()
    for _ in range(10):
        state = context.getState(getForces=True, groups=1<<1)
    t1 = time.time()
    
    avg_time = (t1-t0) * 100  # ms per call
    print(f'  Average: {avg_time:.2f} ms/call')
    
    del context
    del integrator

