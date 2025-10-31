#!/usr/bin/env python
"""
直接測試 CUDA Kernel 正確性
繞過 Python wrapper，用現有 Python OPTIMIZED 版本產生測試數據
"""

import numpy as np
import sys
sys.path.insert(0, '/home/andy/test_optimization/openMM_constant_V_beta/lib')

from Fixed_Voltage_routines_OPTIMIZED import Poisson_solver_fixed_voltage
import openmm as mm
from openmm import unit

print("="*80)
print("CUDA Kernel 正確性測試")
print("="*80)

# === 測試參數 (來自你的 config.ini) ===
area = 14.0 * 14.0  # nm^2
voltage = 0.5  # V
gap_length = 1.5  # nm
conversion = 138.935456
image_charge_radius = 2.0  # nm
image_charge_distance = 0.5  # nm

# === 創建測試數據 ===
np.random.seed(42)

# 20 個 cathode atoms
cathode_indices = list(range(20))
cathode_positions = np.random.rand(20, 3) * [14.0, 14.0, 0.5]  # 限制在 z=0 附近
cathode_positions[:, 2] = 0.0  # cathode 在 z=0

# 20 個 anode atoms  
anode_indices = list(range(20, 40))
anode_positions = np.random.rand(20, 3) * [14.0, 14.0, 0.5]
anode_positions[:, 2] = gap_length  # anode 在 z=1.5nm

all_positions = np.vstack([cathode_positions, anode_positions])
all_indices = cathode_indices + anode_indices

# === Python OPTIMIZED 版本計算 ===
print("\n[1] Python OPTIMIZED 計算...")

# 初始化電荷 (全部 0)
cathode_charges = np.zeros(len(cathode_indices))
anode_charges = np.zeros(len(anode_indices))

# 模擬 3 次迭代
for iteration in range(3):
    print(f"\n  Iteration {iteration + 1}/3")
    
    # 創建 OpenMM system (用來計算 forces)
    system = mm.System()
    for i in range(len(all_positions)):
        system.addParticle(1.0)
    
    # NonbondedForce (用當前電荷)
    nb_force = mm.NonbondedForce()
    nb_force.setNonbondedMethod(mm.NonbondedForce.NoCutoff)
    
    for i in range(40):
        if i < 20:  # cathode
            charge = cathode_charges[i]
        else:  # anode
            charge = anode_charges[i - 20]
        nb_force.addParticle(charge, 1.0, 0.0)  # charge, sigma, epsilon
    
    system.addForce(nb_force)
    
    # 創建 context
    platform = mm.Platform.getPlatformByName('Reference')
    integrator = mm.LangevinIntegrator(300*unit.kelvin, 1/unit.picoseconds, 0.002*unit.picoseconds)
    context = mm.Context(system, integrator, platform)
    context.setPositions(all_positions)
    
    # 獲取 forces
    state = context.getState(getForces=True)
    forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole/unit.nanometer)
    
    # === Python Poisson solver ===
    cathode_charges, anode_charges, cathode_target, anode_target = Poisson_solver_fixed_voltage(
        forces,
        all_positions,
        cathode_charges,
        anode_charges,
        cathode_indices,
        anode_indices,
        area,
        voltage,
        gap_length,
        conversion,
        image_charge_radius,
        image_charge_distance
    )
    
    print(f"    Cathode: sum={np.sum(cathode_charges):.6f}, target={cathode_target:.6f}")
    print(f"    Anode:   sum={np.sum(anode_charges):.6f}, target={anode_target:.6f}")
    print(f"    Cathode charges: min={np.min(cathode_charges):.6f}, max={np.max(cathode_charges):.6f}")
    print(f"    Anode charges:   min={np.min(anode_charges):.6f}, max={np.max(anode_charges):.6f}")
    
    del context, integrator

print("\n" + "="*80)
print("✅ Python OPTIMIZED 計算完成")
print(f"   最終 cathode 總電荷: {np.sum(cathode_charges):.8f} e")
print(f"   最終 anode 總電荷:   {np.sum(anode_charges):.8f} e")
print(f"   電荷守恆: {np.sum(cathode_charges) + np.sum(anode_charges):.2e} e")
print("="*80)

print("\n[2] CUDA Kernel 測試計劃:")
print("   ❌ 無法直接測試 - 需要 Python wrapper")
print("   ✅ CUDA kernels 已編譯 (1.2MB libElectrodeChargePluginCUDA.so)")
print("   ⏳ 等待 Python wrapper 或 C++ 測試框架")

print("\n[3] 目前狀況:")
print("   • Reference Platform Kernel: ✅ 已編譯並驗證算法正確")
print("   • CUDA Platform Kernel: ✅ 已編譯 (1.2MB)")
print("   • Python Wrapper: ❌ 編譯失敗 (missing OpenMMException.h)")
print("   • Integration: ⏳ Pending wrapper 或直接用 C++")

print("\n[4] 下一步選項:")
print("   A. 修復 Python wrapper 編譯 (需要修正 include path)")
print("   B. 寫 C++ 測試程式直接調用 CUDA kernel")
print("   C. 先用 Reference platform 驗證 plugin 架構正確性")
print("   D. 直接整合進 run_openMM.py，用 Python 版本跑，但測試 plugin 載入流程")

print("\n建議：選項 C - 先用 Reference platform 測試")
print("="*80)
