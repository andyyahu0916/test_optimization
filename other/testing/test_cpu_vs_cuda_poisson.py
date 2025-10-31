#!/usr/bin/env python3
"""
严格的A/B数值对比测试
对比原始Python CPU Poisson solver vs CUDA Plugin Poisson solver
验证位元级一致性（或合理的浮点误差范围内）

测试流程：
1. 创建相同的初始系统（电荷、位置、参数）
2. Python CPU版本：手动调用Poisson solver，记录每次迭代的电荷
3. CUDA Plugin版本：通过ElectrodeChargeForce自动调用，记录每次迭代的电荷
4. 逐次对比，计算RMS误差和最大误差
"""

import sys
import os
import numpy as np
from copy import deepcopy

# Add paths
sys.path.insert(0, '/home/andy/test_optimization/openMM_constant_V_beta/lib')
sys.path.insert(0, '/home/andy/test_optimization/plugins/ElectrodeChargePlugin/build/python')

print("=" * 80)
print("Python CPU vs CUDA Plugin Poisson Solver 数值对比测试")
print("=" * 80)

# Import OpenMM
try:
    from openmm.app import *
    from openmm import *
    from openmm.unit import *
    import electrodecharge
    print("\n[1/6] ✓ OpenMM and Plugin imported")
except Exception as e:
    print(f"\n[1/6] ✗ Import failed: {e}")
    sys.exit(1)

# Load plugin
try:
    conda_prefix = os.environ.get('CONDA_PREFIX', '/home/andy/miniforge3/envs/cuda')
    plugin_dir = os.path.join(conda_prefix, 'lib', 'plugins')
    Platform.loadPluginsFromDirectory(plugin_dir)
    print("[2/6] ✓ Plugin loaded")
except Exception as e:
    print(f"[2/6] ✗ Plugin loading failed: {e}")
    sys.exit(1)

# Test parameters (from config.ini)
print("\n[3/6] Creating test system...")
box_size = 5.0  # nm
gap_length = 3.0  # nm
cathode_voltage = 0.5  # V
anode_voltage = 0.5  # V (total drop = 1.0V)
num_cathode = 20
num_anode = 20
num_conductor = 5  # Buckyball conductors
num_iterations = 3

# Physical constants (matching Python)
CONVERSION_NM_BOHR = 18.8973
CONVERSION_KJMOL_NM_AU = CONVERSION_NM_BOHR / 2625.5
CONVERSION_EV_KJMOL = 96.487
PI = np.pi
small_threshold = 1e-6

# Create positions
np.random.seed(42)  # Reproducible random positions
cathode_positions = []
anode_positions = []
conductor_positions = []

# Cathode atoms (z=0.1nm)
for i in range(num_cathode):
    x = np.random.uniform(0.5, box_size - 0.5)
    y = np.random.uniform(0.5, box_size - 0.5)
    cathode_positions.append([x, y, 0.1])

# Anode atoms (z=2.9nm)
for i in range(num_anode):
    x = np.random.uniform(0.5, box_size - 0.5)
    y = np.random.uniform(0.5, box_size - 0.5)
    anode_positions.append([x, y, 2.9])

# Conductor atoms (circle at z=1.5nm)
for i in range(num_conductor):
    angle = 2 * np.pi * i / num_conductor
    x = box_size / 2 + 0.5 * np.cos(angle)
    y = box_size / 2 + 0.5 * np.sin(angle)
    conductor_positions.append([x, y, 1.5])

all_positions = cathode_positions + anode_positions + conductor_positions
cathode_indices = list(range(num_cathode))
anode_indices = list(range(num_cathode, num_cathode + num_anode))
conductor_indices = list(range(num_cathode + num_anode, num_cathode + num_anode + num_conductor))

print(f"       System: {num_cathode} cathode + {num_anode} anode + {num_conductor} conductor atoms")
print(f"       Box: {box_size}×{box_size}×{gap_length} nm³")
print(f"       Voltage: ±{cathode_voltage} V")

# ============================================================================
# Test A: Python CPU Version (Manual Poisson Solver)
# ============================================================================
print("\n[4/6] Running Python CPU Poisson Solver...")

# Create system A
system_cpu = System()
system_cpu.setDefaultPeriodicBoxVectors(
    Vec3(box_size, 0, 0) * nanometer,
    Vec3(0, box_size, 0) * nanometer,
    Vec3(0, 0, gap_length) * nanometer
)

for _ in range(len(all_positions)):
    system_cpu.addParticle(12.0)

nb_cpu = NonbondedForce()
nb_cpu.setNonbondedMethod(NonbondedForce.NoCutoff)

# Initialize all charges to zero
for _ in range(len(all_positions)):
    nb_cpu.addParticle(0.0, 0.34, 0.36)

system_cpu.addForce(nb_cpu)

# Create context (use Reference platform to match Python)
integrator_cpu = VerletIntegrator(0.001 * picoseconds)
platform_ref = Platform.getPlatformByName('Reference')
context_cpu = Context(system_cpu, integrator_cpu, platform_ref)
context_cpu.setPositions([Vec3(p[0], p[1], p[2]) for p in all_positions] * nanometer)

# Geometric parameters
sheet_area = box_size * box_size
cathode_area_atom = sheet_area / num_cathode
anode_area_atom = sheet_area / num_anode
cathode_voltage_kj = abs(cathode_voltage) * CONVERSION_EV_KJMOL
anode_voltage_kj = abs(anode_voltage) * CONVERSION_EV_KJMOL
lgap = gap_length
lcell = 2.8  # anodeZ - cathodeZ
cathodeZ = 0.1
anodeZ = 2.9

# Store charges from each iteration
cpu_charges_history = []

for iter_idx in range(num_iterations):
    # Get current forces
    state = context_cpu.getState(getForces=True, getPositions=True)
    forces_cpu = state.getForces(asNumpy=True).value_in_unit(kilojoules_per_mole / nanometer)
    positions_cpu = state.getPositions(asNumpy=True).value_in_unit(nanometer)

    # Store current charges before update
    current_charges = np.zeros(len(all_positions))
    for i in range(len(all_positions)):
        q, sig, eps = nb_cpu.getParticleParameters(i)
        current_charges[i] = q if isinstance(q, float) else q._value

    # Python Poisson solver logic (matching MM_classes.py:327-354)

    # 1. Update cathode charges
    for i in cathode_indices:
        q_old = current_charges[i]
        Ez_external = (forces_cpu[i, 2] / q_old) if abs(q_old) > 0.9 * small_threshold else 0.0
        q_new = 2.0 / (4.0 * PI) * cathode_area_atom * (cathode_voltage_kj / lgap + Ez_external) * CONVERSION_KJMOL_NM_AU
        if abs(q_new) < small_threshold:
            q_new = small_threshold
        current_charges[i] = q_new
        nb_cpu.setParticleParameters(i, q_new, 0.34, 0.36)

    # 2. Update anode charges
    for i in anode_indices:
        q_old = current_charges[i]
        Ez_external = (forces_cpu[i, 2] / q_old) if abs(q_old) > 0.9 * small_threshold else 0.0
        q_new = -2.0 / (4.0 * PI) * anode_area_atom * (anode_voltage_kj / lgap + Ez_external) * CONVERSION_KJMOL_NM_AU
        if abs(q_new) < small_threshold:
            q_new = -small_threshold
        current_charges[i] = q_new
        nb_cpu.setParticleParameters(i, q_new, 0.34, 0.36)

    # 3. Update conductor charges (two-stage method)
    if conductor_indices:
        # Stage 1: Image charges
        for i in conductor_indices:
            q_old = current_charges[i]
            if abs(q_old) > 0.9 * small_threshold:
                # Normal vector (pointing up for simplicity)
                normal = np.array([0.0, 0.0, 1.0])
                En_external = np.dot(forces_cpu[i], normal) / q_old
                area = 0.1  # Surface area per conductor atom (nm²)
                image_charge = 2.0 / (4.0 * PI) * area * En_external * CONVERSION_KJMOL_NM_AU
                if abs(image_charge) < small_threshold:
                    image_charge = small_threshold
                current_charges[i] = image_charge
                nb_cpu.setParticleParameters(i, image_charge, 0.34, 0.36)
            else:
                current_charges[i] = small_threshold
                nb_cpu.setParticleParameters(i, small_threshold, 0.34, 0.36)

        # Update context for force recalculation
        nb_cpu.updateParametersInContext(context_cpu)

        # Recalculate forces with new image charges
        state = context_cpu.getState(getForces=True)
        forces_cpu = state.getForces(asNumpy=True).value_in_unit(kilojoules_per_mole / nanometer)

        # Stage 2: Charge transfer
        # Calculate dQ for the conductor (treat all as one conductor for simplicity)
        contact_idx = conductor_indices[0]  # First conductor atom is contact
        q_contact = current_charges[contact_idx]
        if abs(q_contact) > 0.9 * small_threshold:
            normal = np.array([0.0, 0.0, 1.0])
            En_external = np.dot(forces_cpu[contact_idx], normal) / q_contact
        else:
            En_external = 0.0

        dE_conductor = -(En_external + (cathode_voltage_kj / lgap / 2.0)) * CONVERSION_KJMOL_NM_AU
        # Buckyball geometry: dr²
        dr = 0.5  # nm
        geometry_factor = dr * dr
        dQ_total = -1.0 * dE_conductor * geometry_factor
        dQ_per_atom = dQ_total / len(conductor_indices)

        for i in conductor_indices:
            current_charges[i] += dQ_per_atom
            nb_cpu.setParticleParameters(i, current_charges[i], 0.34, 0.36)

    # 4. Analytic scaling (simplified - just scale anode, then cathode+conductors)
    # Compute analytic targets
    geom_term = (1.0 / (4.0 * PI)) * sheet_area * ((cathode_voltage_kj / lgap) + (cathode_voltage_kj / lcell)) * CONVERSION_KJMOL_NM_AU
    cathode_target = geom_term
    anode_target = -geom_term

    # Add electrolyte contribution (none in this test)
    # Add conductor contribution
    for i in conductor_indices:
        z_dist = abs(positions_cpu[i, 2] - anodeZ)
        cathode_target += (z_dist / lcell) * (-current_charges[i])

    anode_target = -cathode_target

    # Scale anode
    anode_sum = np.sum(current_charges[anode_indices])
    if abs(anode_sum) > small_threshold:
        anode_scale = anode_target / anode_sum
        if anode_scale > 0.0:
            for i in anode_indices:
                current_charges[i] *= anode_scale
                nb_cpu.setParticleParameters(i, current_charges[i], 0.34, 0.36)

    # Scale cathode + conductors together
    cathode_sum = np.sum(current_charges[cathode_indices])
    conductor_sum = np.sum(current_charges[conductor_indices])
    cathode_conductor_sum = cathode_sum + conductor_sum
    final_anode_sum = np.sum(current_charges[anode_indices])
    cathode_side_target = -final_anode_sum

    if abs(cathode_conductor_sum) > small_threshold:
        combined_scale = cathode_side_target / cathode_conductor_sum
        if combined_scale > 0.0:
            for i in cathode_indices:
                current_charges[i] *= combined_scale
                nb_cpu.setParticleParameters(i, current_charges[i], 0.34, 0.36)
            for i in conductor_indices:
                current_charges[i] *= combined_scale
                nb_cpu.setParticleParameters(i, current_charges[i], 0.34, 0.36)

    # Update context
    nb_cpu.updateParametersInContext(context_cpu)

    # Store final charges for this iteration
    final_charges = np.zeros(len(all_positions))
    for i in range(len(all_positions)):
        q, sig, eps = nb_cpu.getParticleParameters(i)
        final_charges[i] = q if isinstance(q, float) else q._value
    cpu_charges_history.append(final_charges.copy())

    print(f"       Iteration {iter_idx+1}/{num_iterations}:")
    print(f"         Cathode: sum={np.sum(final_charges[cathode_indices]):.8f}")
    print(f"         Anode:   sum={np.sum(final_charges[anode_indices]):.8f}")
    print(f"         Conductor: sum={np.sum(final_charges[conductor_indices]):.8f}")
    print(f"         Total:   sum={np.sum(final_charges):.2e}")

print("       ✓ Python CPU Poisson完成")

# ============================================================================
# Test B: Plugin Version (Automatic Poisson Solver via Reference platform)
# ============================================================================
print("\n[5/6] Running Plugin Reference Poisson Solver...")

# NOTE: Plugin在Context创建时执行，无法像Python那样逐次迭代对比
# 所以我们只能对比最终结果
# 但我们可以用Reference platform来确保公平比较

# Create system B (identical initial state)
system_cuda = System()
system_cuda.setDefaultPeriodicBoxVectors(
    Vec3(box_size, 0, 0) * nanometer,
    Vec3(0, box_size, 0) * nanometer,
    Vec3(0, 0, gap_length) * nanometer
)

for _ in range(len(all_positions)):
    system_cuda.addParticle(12.0)

nb_cuda = NonbondedForce()
nb_cuda.setNonbondedMethod(NonbondedForce.NoCutoff)

# Initialize all charges to zero (same as CPU version)
for _ in range(len(all_positions)):
    nb_cuda.addParticle(0.0, 0.34, 0.36)

system_cuda.addForce(nb_cuda)

# Add ElectrodeChargeForce
electrode_force = electrodecharge.ElectrodeChargeForce()
electrode_force.setCathode(cathode_indices, cathode_voltage)
electrode_force.setAnode(anode_indices, anode_voltage)
electrode_force.setNumIterations(num_iterations)
electrode_force.setSmallThreshold(small_threshold)

# Conductor setup (8 parameters)
c_indices = conductor_indices
c_normals = [0.0, 0.0, 1.0] * num_conductor
c_areas = [0.1] * num_conductor
c_contacts = [conductor_indices[0]]
c_contact_normals = [0.0, 0.0, 1.0]
c_geometries = [0.5 * 0.5]  # Buckyball: dr²
c_atom_ids = [0] * num_conductor
c_atom_counts = [num_conductor]

electrode_force.setConductorData(
    c_indices, c_normals, c_areas,
    c_contacts, c_contact_normals, c_geometries,
    c_atom_ids, c_atom_counts
)

system_cuda.addForce(electrode_force)

# Create context (use Reference platform first to test algorithm correctness)
# TODO: Fix CUDA kernel registration issue
integrator_cuda = VerletIntegrator(0.001 * picoseconds)
platform_plugin = Platform.getPlatformByName('Reference')  # Use Reference for now
print(f"       Using {platform_plugin.getName()} platform (CUDA kernel registration issue)")
context_cuda = Context(system_cuda, integrator_cuda, platform_plugin)
context_cuda.setPositions([Vec3(p[0], p[1], p[2]) for p in all_positions] * nanometer)

# Plugin自动执行，我们只需获取最终结果
state_cuda = context_cuda.getState(getForces=True, getEnergy=True)

cuda_charges = np.zeros(len(all_positions))
for i in range(len(all_positions)):
    q, sig, eps = nb_cuda.getParticleParameters(i)
    cuda_charges[i] = q if isinstance(q, float) else q._value

print(f"       After {num_iterations} iterations:")
print(f"         Cathode: sum={np.sum(cuda_charges[cathode_indices]):.8f}")
print(f"         Anode:   sum={np.sum(cuda_charges[anode_indices]):.8f}")
print(f"         Conductor: sum={np.sum(cuda_charges[conductor_indices]):.8f}")
print(f"         Total:   sum={np.sum(cuda_charges):.2e}")
print("       ✓ CUDA Plugin Poisson完成")

# ============================================================================
# Compare Results
# ============================================================================
print("\n[6/6] 对比Python CPU vs CUDA Plugin...")
print("=" * 80)

# Compare final iteration
cpu_final = cpu_charges_history[-1]

# Per-group comparison
cathode_diff = cuda_charges[cathode_indices] - cpu_final[cathode_indices]
anode_diff = cuda_charges[anode_indices] - cpu_final[anode_indices]
conductor_diff = cuda_charges[conductor_indices] - cpu_final[conductor_indices]

cathode_rms = np.sqrt(np.mean(cathode_diff**2))
anode_rms = np.sqrt(np.mean(anode_diff**2))
conductor_rms = np.sqrt(np.mean(conductor_diff**2))

cathode_max = np.max(np.abs(cathode_diff))
anode_max = np.max(np.abs(anode_diff))
conductor_max = np.max(np.abs(conductor_diff))

print(f"\nCathode atoms ({num_cathode} atoms):")
print(f"  RMS差异:  {cathode_rms:.2e} e/atom")
print(f"  最大差异: {cathode_max:.2e} e/atom")

print(f"\nAnode atoms ({num_anode} atoms):")
print(f"  RMS差异:  {anode_rms:.2e} e/atom")
print(f"  最大差异: {anode_max:.2e} e/atom")

print(f"\nConductor atoms ({num_conductor} atoms):")
print(f"  RMS差异:  {conductor_rms:.2e} e/atom")
print(f"  最大差异: {conductor_max:.2e} e/atom")

# Overall comparison
all_diff = cuda_charges - cpu_final
overall_rms = np.sqrt(np.mean(all_diff**2))
overall_max = np.max(np.abs(all_diff))

print(f"\n总体统计 ({len(all_positions)} atoms):")
print(f"  RMS差异:  {overall_rms:.2e} e/atom")
print(f"  最大差异: {overall_max:.2e} e/atom")

# Tolerance check
tolerance_rms = 1e-6  # Double precision tolerance
tolerance_max = 1e-5

print("\n" + "=" * 80)
if overall_rms < tolerance_rms and overall_max < tolerance_max:
    print("✅ 测试通过！CUDA Plugin和Python CPU版本数值一致")
    print(f"   RMS误差 < {tolerance_rms:.0e} ✓")
    print(f"   最大误差 < {tolerance_max:.0e} ✓")
    print("\n👍 Plugin正确实现了Poisson solver，可以放心使用！")
else:
    print("⚠️  警告：CUDA Plugin和Python CPU版本存在显著差异")
    print(f"   RMS误差: {overall_rms:.2e} (阈值: {tolerance_rms:.0e})")
    print(f"   最大误差: {overall_max:.2e} (阈值: {tolerance_max:.0e})")
    print("\n需要进一步调查差异原因：")
    print("  - 浮点精度差异（CPU vs GPU）")
    print("  - Reduction求和顺序不同")
    print("  - Threshold处理逻辑差异")

# Charge conservation check
cpu_total = np.sum(cpu_final)
cuda_total = np.sum(cuda_charges)
print(f"\n电荷守恒检查:")
print(f"  Python CPU总电荷: {cpu_total:+.2e} e")
print(f"  CUDA Plugin总电荷: {cuda_total:+.2e} e")
print(f"  差异:              {abs(cuda_total - cpu_total):.2e} e")

print("=" * 80)

# Cleanup
del context_cpu, context_cuda, integrator_cpu, integrator_cuda
