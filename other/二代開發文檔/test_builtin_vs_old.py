#!/usr/bin/env python
"""
快速測試: OpenMM 內建 ConstantPotentialForce vs 舊版實現

用途: 用簡單的平面電極系統測試內建實現,評估是否可以替代舊版

使用方法:
    python test_builtin_vs_old.py

輸出:
    - 內建實現的電極電荷
    - 能量信息
    - 與舊版的對比 (如果可用)
"""

import openmm as mm
from openmm import app, unit
import numpy as np
import sys

# ============================================================================
# 系統設置 (請根據你的實際系統修改)
# ============================================================================

# 電極參數
CATHODE_VOLTAGE = -2.0  # V (需要轉換為 kJ/mol/e)
ANODE_VOLTAGE = 2.0     # V
LGAP = 5.0              # nm (電極間距)

# 單位轉換: 1 V = 96.485 kJ/mol/e
VOLTAGE_TO_KJMOL = 96.485

# 電極原子索引 (示例,請替換為實際值)
# 你可以從舊版代碼中提取這些信息
CATHODE_INDICES = list(range(0, 100))    # 替換為實際陰極原子索引
ANODE_INDICES = list(range(100, 200))    # 替換為實際陽極原子索引

# PME 參數
CUTOFF = 1.0           # nm
EWALD_ERROR_TOL = 1e-5

# Gaussian 電荷分布寬度 (需要測試不同值)
GAUSSIAN_WIDTH = 0.05  # nm

# CG 求解器參數
CG_ERROR_TOL = 0.01    # kJ/mol/e

# ============================================================================
# 創建簡化的測試系統
# ============================================================================

def create_simple_electrode_system():
    """
    創建一個簡化的電極系統用於測試
    
    包含:
    - 兩個平面電極 (陰極 + 陽極)
    - 簡單的電解質 (可選)
    """
    print("=" * 70)
    print("創建簡化測試系統")
    print("=" * 70)
    
    # 創建系統
    system = mm.System()
    
    # 添加粒子
    # 陰極原子
    for i in range(len(CATHODE_INDICES)):
        system.addParticle(12.0)  # Carbon mass
    
    # 陽極原子
    for i in range(len(ANODE_INDICES)):
        system.addParticle(12.0)
    
    # 設置周期性盒子
    box_size = 5.0  # nm
    system.setDefaultPeriodicBoxVectors(
        mm.Vec3(box_size, 0, 0),
        mm.Vec3(0, box_size, 0),
        mm.Vec3(0, 0, LGAP)
    )
    
    print(f"系統信息:")
    print(f"  粒子數: {system.getNumParticles()}")
    print(f"  陰極原子: {len(CATHODE_INDICES)}")
    print(f"  陽極原子: {len(ANODE_INDICES)}")
    print(f"  盒子尺寸: {box_size} x {box_size} x {LGAP} nm³")
    print()
    
    return system


def create_simple_positions():
    """創建簡單的初始位置"""
    positions = []
    
    # 陰極位置 (z = 0.5 nm)
    n_cathode = len(CATHODE_INDICES)
    grid_size = int(np.ceil(np.sqrt(n_cathode)))
    spacing = 0.3  # nm
    
    for i in range(n_cathode):
        x = (i % grid_size) * spacing
        y = (i // grid_size) * spacing
        z = 0.5
        positions.append(mm.Vec3(x, y, z) * unit.nanometers)
    
    # 陽極位置 (z = LGAP - 0.5 nm)
    n_anode = len(ANODE_INDICES)
    for i in range(n_anode):
        x = (i % grid_size) * spacing
        y = (i // grid_size) * spacing
        z = LGAP - 0.5
        positions.append(mm.Vec3(x, y, z) * unit.nanometers)
    
    return positions


# ============================================================================
# 測試內建 ConstantPotentialForce
# ============================================================================

def test_builtin_force():
    """測試內建 ConstantPotentialForce"""
    print("=" * 70)
    print("測試內建 ConstantPotentialForce")
    print("=" * 70)
    
    # 創建系統
    system = create_simple_electrode_system()
    positions = create_simple_positions()
    
    # 創建 ConstantPotentialForce
    force = mm.ConstantPotentialForce()
    
    # 設置 PME 參數
    force.setCutoffDistance(CUTOFF)
    force.setEwaldErrorTolerance(EWALD_ERROR_TOL)
    
    # 添加非電極粒子 (初始電荷為 0)
    total_particles = len(CATHODE_INDICES) + len(ANODE_INDICES)
    for i in range(total_particles):
        force.addParticle(0.0)  # 電極粒子的電荷會被求解器覆蓋
    
    # 添加陰極
    cathode_set = set(CATHODE_INDICES)
    cathode_potential = CATHODE_VOLTAGE * VOLTAGE_TO_KJMOL
    force.addElectrode(
        electrodeParticles=cathode_set,
        potential=cathode_potential,
        gaussianWidth=GAUSSIAN_WIDTH,
        thomasFermiScale=0.0  # 暫時不使用 Thomas-Fermi 模型
    )
    print(f"陰極:")
    print(f"  原子數: {len(cathode_set)}")
    print(f"  電壓: {CATHODE_VOLTAGE} V = {cathode_potential:.3f} kJ/mol/e")
    print(f"  Gaussian 寬度: {GAUSSIAN_WIDTH} nm")
    
    # 添加陽極
    anode_set = set(ANODE_INDICES)
    anode_potential = ANODE_VOLTAGE * VOLTAGE_TO_KJMOL
    force.addElectrode(
        electrodeParticles=anode_set,
        potential=anode_potential,
        gaussianWidth=GAUSSIAN_WIDTH,
        thomasFermiScale=0.0
    )
    print(f"陽極:")
    print(f"  原子數: {len(anode_set)}")
    print(f"  電壓: {ANODE_VOLTAGE} V = {anode_potential:.3f} kJ/mol/e")
    print(f"  Gaussian 寬度: {GAUSSIAN_WIDTH} nm")
    print()
    
    # 設置求解方法 (CG)
    force.setConstantPotentialMethod(mm.ConstantPotentialForce.CG)
    force.setUsePreconditioner(True)
    force.setCGErrorTolerance(CG_ERROR_TOL)
    print(f"求解方法: Conjugate Gradient")
    print(f"  預條件器: 啟用")
    print(f"  誤差容忍度: {CG_ERROR_TOL} kJ/mol/e")
    print()
    
    # 添加到系統
    system.addForce(force)
    
    # 創建 Context
    try:
        platform = mm.Platform.getPlatformByName('CUDA')
        print(f"使用平台: CUDA")
    except:
        platform = mm.Platform.getPlatformByName('CPU')
        print(f"使用平台: CPU (CUDA 不可用)")
    print()
    
    integrator = mm.VerletIntegrator(0.001)
    context = mm.Context(system, integrator, platform)
    context.setPositions(positions)
    
    # 獲取初始狀態
    print("-" * 70)
    print("初始狀態")
    print("-" * 70)
    state = context.getState(getEnergy=True)
    energy = state.getPotentialEnergy()
    print(f"勢能: {energy}")
    
    # 獲取電極電荷
    charges = []
    force.getCharges(context, charges)
    
    cathode_charges = [charges[i] for i in CATHODE_INDICES]
    anode_charges = [charges[i] for i in ANODE_INDICES]
    
    print()
    print("電極電荷:")
    print(f"  陰極總電荷: {sum(cathode_charges):.6f} e")
    print(f"  陽極總電荷: {sum(anode_charges):.6f} e")
    print(f"  系統總電荷: {sum(charges):.6f} e")
    print()
    print(f"  陰極單原子平均電荷: {np.mean(cathode_charges):.6f} e")
    print(f"  陰極電荷標準差: {np.std(cathode_charges):.6f} e")
    print(f"  陽極單原子平均電荷: {np.mean(anode_charges):.6f} e")
    print(f"  陽極電荷標準差: {np.std(anode_charges):.6f} e")
    print()
    
    # 檢查電荷守恆
    total_charge = sum(charges)
    if abs(total_charge) < 1e-6:
        print("✅ 電荷守恆檢查: 通過")
    else:
        print(f"⚠️  電荷守恆檢查: 總電荷 = {total_charge:.6e} e (應該接近 0)")
    
    # 檢查電荷是否相反
    cathode_total = sum(cathode_charges)
    anode_total = sum(anode_charges)
    charge_diff = abs(cathode_total + anode_total)
    if charge_diff < 1e-6:
        print("✅ 電極電荷相反檢查: 通過")
    else:
        print(f"⚠️  電極電荷相反檢查: |Q_cathode + Q_anode| = {charge_diff:.6e} e")
    
    print()
    return context, force


# ============================================================================
# 比較不同參數
# ============================================================================

def test_gaussian_width_effect():
    """測試不同 Gaussian 寬度的影響"""
    print("=" * 70)
    print("測試 Gaussian 寬度的影響")
    print("=" * 70)
    
    gaussian_widths = [0.03, 0.05, 0.08, 0.10]
    results = []
    
    for gw in gaussian_widths:
        print(f"\n--- Gaussian 寬度 = {gw} nm ---")
        
        # 修改全局參數
        global GAUSSIAN_WIDTH
        GAUSSIAN_WIDTH = gw
        
        # 創建系統
        system = create_simple_electrode_system()
        positions = create_simple_positions()
        
        # 創建 Force
        force = mm.ConstantPotentialForce()
        force.setCutoffDistance(CUTOFF)
        force.setEwaldErrorTolerance(EWALD_ERROR_TOL)
        
        # 添加粒子和電極
        total_particles = len(CATHODE_INDICES) + len(ANODE_INDICES)
        for i in range(total_particles):
            force.addParticle(0.0)
        
        cathode_set = set(CATHODE_INDICES)
        cathode_potential = CATHODE_VOLTAGE * VOLTAGE_TO_KJMOL
        force.addElectrode(cathode_set, cathode_potential, gw, 0.0)
        
        anode_set = set(ANODE_INDICES)
        anode_potential = ANODE_VOLTAGE * VOLTAGE_TO_KJMOL
        force.addElectrode(anode_set, anode_potential, gw, 0.0)
        
        force.setConstantPotentialMethod(mm.ConstantPotentialForce.CG)
        force.setUsePreconditioner(True)
        force.setCGErrorTolerance(CG_ERROR_TOL)
        
        system.addForce(force)
        
        # 創建 Context
        try:
            platform = mm.Platform.getPlatformByName('CUDA')
        except:
            platform = mm.Platform.getPlatformByName('CPU')
        
        integrator = mm.VerletIntegrator(0.001)
        context = mm.Context(system, integrator, platform)
        context.setPositions(positions)
        
        # 獲取結果
        state = context.getState(getEnergy=True)
        energy = state.getPotentialEnergy()
        
        charges = []
        force.getCharges(context, charges)
        
        cathode_total = sum([charges[i] for i in CATHODE_INDICES])
        anode_total = sum([charges[i] for i in ANODE_INDICES])
        
        print(f"  陰極總電荷: {cathode_total:.6f} e")
        print(f"  陽極總電荷: {anode_total:.6f} e")
        print(f"  勢能: {energy}")
        
        results.append({
            'gaussian_width': gw,
            'cathode_charge': cathode_total,
            'anode_charge': anode_total,
            'energy': energy
        })
        
        # 清理
        del context
        del integrator
    
    # 總結
    print("\n" + "=" * 70)
    print("Gaussian 寬度影響總結")
    print("=" * 70)
    print(f"{'寬度 (nm)':<15} {'陰極電荷 (e)':<20} {'陽極電荷 (e)':<20} {'勢能':<20}")
    print("-" * 70)
    for r in results:
        print(f"{r['gaussian_width']:<15.3f} {r['cathode_charge']:<20.6f} "
              f"{r['anode_charge']:<20.6f} {r['energy']}")
    print()


# ============================================================================
# 主函數
# ============================================================================

def main():
    """主測試函數"""
    print("\n")
    print("*" * 70)
    print("  OpenMM 內建 ConstantPotentialForce 快速測試")
    print("*" * 70)
    print()
    
    # 檢查 OpenMM 版本
    print(f"OpenMM 版本: {mm.version.version}")
    if mm.version.version < '8.4.0':
        print("⚠️  警告: ConstantPotentialForce 需要 OpenMM >= 8.4.0")
        sys.exit(1)
    print()
    
    # 測試基本功能
    try:
        context, force = test_builtin_force()
        print()
        print("✅ 內建 ConstantPotentialForce 測試成功!")
        print()
        
        # 清理
        del context
        
    except Exception as e:
        print()
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 詢問是否進行參數測試
    print("-" * 70)
    response = input("是否測試不同 Gaussian 寬度的影響? (y/n): ")
    if response.lower() == 'y':
        try:
            test_gaussian_width_effect()
        except Exception as e:
            print(f"❌ 參數測試失敗: {e}")
            import traceback
            traceback.print_exc()
    
    print()
    print("*" * 70)
    print("  測試完成!")
    print("*" * 70)
    print()
    print("下一步建議:")
    print("1. 如果結果合理,考慮遷移到內建實現 (方案 A)")
    print("2. 如果需要 Buckyball/Nanotube,考慮保留舊版 (方案 C)")
    print("3. 閱讀 DECISION_CHECKLIST.md 做出最終決策")
    print()


if __name__ == '__main__':
    main()
