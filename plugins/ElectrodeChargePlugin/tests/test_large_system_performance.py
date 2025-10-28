#!/usr/bin/env python
"""Test GPU performance on larger system."""
import ctypes, os, sys, time, numpy as np

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

def create_large_system(num_particles, platform_name):
    """Create a larger test system."""
    system = System()
    for _ in range(num_particles):
        system.addParticle(40.0)
    
    nonbonded = NonbondedForce()
    nonbonded.setNonbondedMethod(NonbondedForce.NoCutoff)
    
    # Electrodes: 10% cathode, 10% anode, 80% electrolyte
    num_cathode = max(2, num_particles // 10)
    num_anode = max(2, num_particles // 10)
    
    cathode_indices = list(range(num_cathode))
    anode_indices = list(range(num_cathode, num_cathode + num_anode))
    
    for i in range(num_particles):
        if i in cathode_indices or i in anode_indices:
            q_init = 1e-10
        else:
            q_init = 0.01 * np.random.randn()  # Random bulk charges
        nonbonded.addParticle(q_init, 1.0, 0.0)
    system.addForce(nonbonded)
    
    force = electrodecharge.ElectrodeChargeForce()
    force.setCathode(cathode_indices, 0.5)
    force.setAnode(anode_indices, 0.5)
    force.setCellGap(1.0)
    force.setCellLength(5.0)
    force.setNumIterations(3)
    force.setSmallThreshold(1.0e-6)
    system.addForce(force)
    
    platform = Platform.getPlatformByName(platform_name)
    integrator = LangevinIntegrator(300*unit.kelvin, 1.0/unit.picosecond, 0.001*unit.picoseconds)
    context = Context(system, integrator, platform)
    
    # Create random positions
    box_size = (num_particles / 10.0) ** (1.0/3.0)  # Density ~10 particles/nm³
    positions = []
    for i in range(num_particles):
        if i < num_cathode:
            z = 0.1
        elif i < num_cathode + num_anode:
            z = box_size - 0.1
        else:
            z = 0.2 + np.random.rand() * (box_size - 0.4)
        
        x = np.random.rand() * box_size
        y = np.random.rand() * box_size
        positions.append(Vec3(x, y, z))
    
    context.setPositions(positions)
    context.setPeriodicBoxVectors(
        Vec3(box_size, 0, 0),
        Vec3(0, box_size, 0),
        Vec3(0, 0, box_size)
    )
    
    return context, nonbonded


print("="*80)
print("Large System Performance Test")
print("="*80)

for num_particles in [10, 50, 100, 200]:
    print(f"\n{'='*80}")
    print(f"System size: {num_particles} particles")
    print(f"{'='*80}")
    
    # Reference
    print("\n--- Reference Platform ---")
    ctx_ref, nb_ref = create_large_system(num_particles, 'Reference')
    
    # Warmup
    for _ in range(3):
        _ = ctx_ref.getState(getForces=True, groups=1<<1)
    
    # Measure
    t0 = time.time()
    for _ in range(10):
        _ = ctx_ref.getState(getForces=True, groups=1<<1)
    t_ref = (time.time() - t0) / 10
    
    print(f"  Time: {t_ref*1000:.2f} ms/call")
    
    # CUDA
    print("\n--- CUDA Platform ---")
    ctx_cuda, nb_cuda = create_large_system(num_particles, 'CUDA')
    
    # Warmup
    for _ in range(3):
        _ = ctx_cuda.getState(getForces=True, groups=1<<1)
    
    # Measure
    t0 = time.time()
    for _ in range(10):
        _ = ctx_cuda.getState(getForces=True, groups=1<<1)
    t_cuda = (time.time() - t0) / 10
    
    print(f"  Time: {t_cuda*1000:.2f} ms/call")
    
    speedup = t_ref / t_cuda
    print(f"\n  Speedup: {speedup:.2f}×")
    
    if speedup > 1.0:
        print(f"  🚀 GPU faster!")
    else:
        print(f"  ⚠️  GPU slower (expected for small systems)")
    
    del ctx_ref, ctx_cuda, nb_ref, nb_cuda

print("\n" + "="*80)
print("Test complete!")
