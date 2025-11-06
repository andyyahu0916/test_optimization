#!/usr/bin/env python3
"""
Test ConstantVPlugin Reference implementation vs Numpy reference.

This validates that the C++ plugin produces identical results to the pure Numpy implementation.
"""

import sys
import os
import ctypes
import numpy as np
from openmm import *
from openmm.app import *
from openmm.unit import *

# Manually load plugin libraries and register kernels
plugin_dir = "/home/andy/test_optimization/openMM_constantV_plugin/ConstantVPlugin/build"
main_lib = ctypes.CDLL(f"{plugin_dir}/libConstantVPlugin.so", mode=ctypes.RTLD_GLOBAL)
ref_lib = ctypes.CDLL(f"{plugin_dir}/platforms/reference/libConstantVPluginReference.so", mode=ctypes.RTLD_GLOBAL)

# Call the registration functions
ref_lib.registerKernelFactories()

# Import constantvplugin
import constantvplugin

print("✓ Plugin libraries loaded and kernels registered")

# Import Numpy reference
sys.path.insert(0, "/home/andy/test_optimization/openMM_constantV_plugin")
from test_numpy_reference import compute_electrode_charges, COULOMB_CONSTANT

print("="*70)
print("Testing ConstantVPlugin C++ vs Numpy Reference")
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

# ========== Numpy reference calculation ==========
print("\n" + "-"*70)
print("Numpy Reference Calculation")
print("-"*70)

q_e_numpy = compute_electrode_charges(
    electrode_positions,
    electrolyte_positions,
    target_potentials,
    fixed_charges,
    inv_cap_matrix
)

print(f"Numpy electrode charges: {q_e_numpy}")

# ========== C++ Plugin calculation ==========
print("\n" + "-"*70)
print("C++ Plugin Calculation")
print("-"*70)

# Create OpenMM system
system = System()

# Add particles (N electrode + M electrolyte)
for i in range(N + M):
    system.addParticle(1.0)  # mass doesn't matter

# Add NonbondedForce (required by plugin)
nonbonded = NonbondedForce()
nonbonded.setNonbondedMethod(NonbondedForce.NoCutoff)

# Add electrode atoms (initially zero charge)
for i in range(N):
    nonbonded.addParticle(0.0, 1.0, 0.0)  # charge=0, sigma=1, epsilon=0

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

# Set inverse capacitance matrix (flatten to 1D)
inv_cap_flat = inv_cap_matrix.flatten().tolist()
constantv_force.setInverseCapacitanceMatrix(inv_cap_flat)

system.addForce(constantv_force)

# Create positions array
positions = []
for i in range(N):
    positions.append(Vec3(*electrode_positions[i]))
for i in range(M):
    positions.append(Vec3(*electrolyte_positions[i]))

# Create context and set positions
integrator = VerletIntegrator(0.001*picoseconds)
context = Context(system, integrator, Platform.getPlatformByName("Reference"))
context.setPositions(positions)

# Get forces and energy (this executes the kernel)
state = context.getState(getForces=True, getEnergy=True)

# Read back the charges from NonbondedForce
q_e_cpp = []
for i in range(N):
    charge, sigma, epsilon = nonbonded.getParticleParameters(i)
    # Extract value in elementary charge units (charge is a Quantity)
    q_e_cpp.append(charge.value_in_unit(elementary_charge))

q_e_cpp = np.array(q_e_cpp)

print(f"C++ electrode charges: {q_e_cpp}")

# ========== Comparison ==========
print("\n" + "="*70)
print("RESULTS COMPARISON")
print("="*70)

diff = np.abs(q_e_cpp - q_e_numpy)
max_diff = np.max(diff)
rel_diff = max_diff / np.max(np.abs(q_e_numpy)) if np.max(np.abs(q_e_numpy)) > 0 else 0

print(f"\nNumpy charges:     {q_e_numpy}")
print(f"C++ plugin charges: {q_e_cpp}")
print(f"\nAbsolute difference: {diff}")
print(f"Maximum difference:  {max_diff}")
print(f"Relative difference: {rel_diff*100:.6f}%")

tolerance = 1e-6
if max_diff < tolerance:
    print("\n" + "="*70)
    print("✓ SUCCESS: C++ plugin matches Numpy reference!")
    print("="*70)
else:
    print("\n" + "="*70)
    print(f"✗ FAILURE: Difference {max_diff} exceeds tolerance {tolerance}")
    print("="*70)
    sys.exit(1)
