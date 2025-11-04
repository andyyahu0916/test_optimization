#!/usr/bin/env python3
"""
Test CUDA platform vs Reference platform.

This validates that the CUDA implementation produces identical results
to the Reference platform (which was already validated against Numpy).
"""

import sys
import os
import numpy as np
from openmm import *
from openmm.app import *
from openmm.unit import *

# Manually load plugin libraries and register kernels
plugin_dir = "/home/andy/test_optimization/openMM_constantV_plugin/ConstantVPlugin/build"
import ctypes
main_lib = ctypes.CDLL(f"{plugin_dir}/libConstantVPlugin.so", mode=ctypes.RTLD_GLOBAL)
ref_lib = ctypes.CDLL(f"{plugin_dir}/platforms/reference/libConstantVPluginReference.so", mode=ctypes.RTLD_GLOBAL)
cuda_lib = ctypes.CDLL(f"{plugin_dir}/platforms/cuda/libConstantVPluginCUDA.so", mode=ctypes.RTLD_GLOBAL)

# Register kernels
ref_lib.registerKernelFactories()
cuda_lib.registerKernelFactories()

# Import constantvplugin
import constantvplugin

print("="*70)
print("Testing CUDA Platform vs Reference Platform")
print("="*70)

# ========== Define test system ==========
N = 2  # 2 electrode atoms
M = 1  # 1 electrolyte atom

# Positions (nm)
electrode_positions = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 2.0],
])

electrolyte_positions = np.array([
    [0.0, 0.0, 1.0],
])

# Target potentials (kJ/mol)
target_potentials = np.array([10.0, -10.0])

# Fixed charges (elementary charge)
fixed_charges = np.array([1.0])

# Inverse capacitance matrix
inv_cap_matrix = np.array([
    [0.1, 0.0],
    [0.0, 0.1],
])

print(f"\nSystem configuration:")
print(f"  Electrode atoms: {N}")
print(f"  Electrolyte atoms: {M}")
print(f"  Target potentials: {target_potentials} kJ/mol")
print(f"  Fixed charges: {fixed_charges} e")

def create_system_and_run(platform_name):
    """Create system, run simulation, and return electrode charges."""
    print(f"\n" + "-"*70)
    print(f"{platform_name} Platform")
    print("-"*70)

    # Create OpenMM system
    system = System()

    # Add particles (N electrode + M electrolyte)
    for i in range(N + M):
        system.addParticle(1.0)

    # Add NonbondedForce
    nonbonded = NonbondedForce()
    nonbonded.setNonbondedMethod(NonbondedForce.NoCutoff)

    # Add electrode atoms (initially zero charge)
    for i in range(N):
        nonbonded.addParticle(0.0, 1.0, 0.0)

    # Add electrolyte atoms
    for i in range(M):
        nonbonded.addParticle(fixed_charges[i], 1.0, 0.0)

    system.addForce(nonbonded)

    # Create ConstantVForce
    constantv_force = constantvplugin.ConstantVForce()

    # Add electrode atoms with target potentials
    for i in range(N):
        constantv_force.addElectrodeAtom(i, target_potentials[i])

    # Add electrolyte atoms
    for i in range(M):
        constantv_force.addElectrolyteAtom(N + i, fixed_charges[i])

    # Set inverse capacitance matrix
    inv_cap_flat = inv_cap_matrix.flatten().tolist()
    constantv_force.setInverseCapacitanceMatrix(inv_cap_flat)

    system.addForce(constantv_force)

    # Create positions
    positions = []
    for i in range(N):
        positions.append(Vec3(*electrode_positions[i]))
    for i in range(M):
        positions.append(Vec3(*electrolyte_positions[i]))

    # Create context with specified platform
    integrator = VerletIntegrator(0.001*picoseconds)
    context = Context(system, integrator, Platform.getPlatformByName(platform_name))
    context.setPositions(positions)

    # Execute kernel (getState triggers the kernel)
    state = context.getState(getForces=True, getEnergy=True)

    # Read back charges
    q_e = []
    for i in range(N):
        charge, sigma, epsilon = nonbonded.getParticleParameters(i)
        q_e.append(charge.value_in_unit(elementary_charge))

    q_e = np.array(q_e)
    print(f"{platform_name} electrode charges: {q_e}")

    del context
    del integrator

    return q_e

# ========== Run on both platforms ==========
q_e_reference = create_system_and_run("Reference")
q_e_cuda = create_system_and_run("CUDA")

# ========== Comparison ==========
print("\n" + "="*70)
print("RESULTS COMPARISON")
print("="*70)

diff = np.abs(q_e_cuda - q_e_reference)
max_diff = np.max(diff)
rel_diff = max_diff / np.max(np.abs(q_e_reference)) if np.max(np.abs(q_e_reference)) > 0 else 0

print(f"\nReference charges: {q_e_reference}")
print(f"CUDA charges:      {q_e_cuda}")
print(f"\nAbsolute difference: {diff}")
print(f"Maximum difference:  {max_diff}")
print(f"Relative difference: {rel_diff*100:.6f}%")

# Allow for small floating point differences
tolerance = 1e-5
if max_diff < tolerance:
    print("\n" + "="*70)
    print("✓ SUCCESS: CUDA platform matches Reference platform!")
    print("="*70)
else:
    print("\n" + "="*70)
    print(f"✗ FAILURE: Difference {max_diff} exceeds tolerance {tolerance}")
    print("="*70)
    sys.exit(1)
