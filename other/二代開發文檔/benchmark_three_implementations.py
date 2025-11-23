#!/usr/bin/env python
"""
性能基準測試: 新Plugin vs 舊版Python vs OpenMM官方

目的: 量化比較三種實現的性能和準確性

測試場景:
1. 平面電極 (100, 500, 1000 個原子)
2. 電解質原子 (1000, 5000, 10000)
3. 測量: 時間、能量、電極電荷、電荷守恆

使用方法:
    python benchmark_three_implementations.py --scenario flat_electrode --n_electrode 100 --n_electrolyte 1000
"""

import openmm as mm
from openmm import app, unit
import numpy as np
import time
import sys
import os

# ============================================================================
# 通用設置
# ============================================================================

CATHODE_VOLTAGE = -2.0  # V
ANODE_VOLTAGE = 2.0     # V
LGAP = 5.0              # nm
BOX_XY = 5.0            # nm
VOLTAGE_TO_KJMOL = 96.485
COULOMB_CONSTANT = 138.935456  # kJ/mol * nm / e^2

# ============================================================================
# 創建測試系統
# ============================================================================

def create_test_system(n_electrode_cathode, n_electrode_anode, n_electrolyte):
    """
    創建標準測試系統
    
    返回:
        system, topology, positions, cathode_indices, anode_indices, electrolyte_indices
    """
    print("=" * 70)
    print(f"創建測試系統:")
    print(f"  陰極原子: {n_electrode_cathode}")
    print(f"  陽極原子: {n_electrode_anode}")
    print(f"  電解質原子: {n_electrolyte}")
    print("=" * 70)
    
    # 創建系統
    system = mm.System()
    topology = app.Topology()
    chain = topology.addChain()
    
    positions = []
    cathode_indices = []
    anode_indices = []
    electrolyte_indices = []
    
    # 陰極 (z = 0.5 nm)
    grid_size = int(np.ceil(np.sqrt(n_electrode_cathode)))
    spacing = BOX_XY / grid_size
    
    cathode_residue = topology.addResidue("CAT", chain)
    for i in range(n_electrode_cathode):
        x = (i % grid_size) * spacing
        y = (i // grid_size) * spacing
        z = 0.5
        
        atom = topology.addAtom("C", app.element.carbon, cathode_residue)
        system.addParticle(12.0)
        positions.append(mm.Vec3(x, y, z))
        cathode_indices.append(len(positions) - 1)
    
    # 陽極 (z = LGAP - 0.5 nm)
    anode_residue = topology.addResidue("ANO", chain)
    for i in range(n_electrode_anode):
        x = (i % grid_size) * spacing
        y = (i // grid_size) * spacing
        z = LGAP - 0.5
        
        atom = topology.addAtom("C", app.element.carbon, anode_residue)
        system.addParticle(12.0)
        positions.append(mm.Vec3(x, y, z))
        anode_indices.append(len(positions) - 1)
    
    # 電解質 (隨機分布在電極之間)
    electrolyte_residue = topology.addResidue("ION", chain)
    np.random.seed(42)
    for i in range(n_electrolyte):
        x = np.random.uniform(0, BOX_XY)
        y = np.random.uniform(0, BOX_XY)
        z = np.random.uniform(1.0, LGAP - 1.0)
        
        atom = topology.addAtom("Na", app.element.sodium, electrolyte_residue)
        system.addParticle(23.0)
        positions.append(mm.Vec3(x, y, z))
        electrolyte_indices.append(len(positions) - 1)
    
    # 設置周期性盒子
    system.setDefaultPeriodicBoxVectors(
        mm.Vec3(BOX_XY, 0, 0),
        mm.Vec3(0, BOX_XY, 0),
        mm.Vec3(0, 0, LGAP)
    )
    
    positions = [mm.Vec3(p[0], p[1], p[2]) * unit.nanometers for p in positions]
    
    print(f"✅ 系統創建完成:")
    print(f"  總粒子數: {system.getNumParticles()}")
    print(f"  陰極: {len(cathode_indices)} 原子")
    print(f"  陽極: {len(anode_indices)} 原子")
    print(f"  電解質: {len(electrolyte_indices)} 原子")
    print()
    
    return system, topology, positions, cathode_indices, anode_indices, electrolyte_indices


# ============================================================================
# 測試 1: OpenMM 官方 ConstantPotentialForce (Matrix 方法)
# ============================================================================

def benchmark_openmm_matrix(system, positions, cathode_indices, anode_indices, electrolyte_indices):
    """測試 OpenMM 官方 Matrix 方法"""
    print("=" * 70)
    print("測試 1: OpenMM 官方 (Matrix 方法)")
    print("=" * 70)
    
    # 創建新系統 (複製)
    test_system = mm.System()
    for i in range(system.getNumParticles()):
        test_system.addParticle(system.getParticleMass(i))
    test_system.setDefaultPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())
    
    # 創建 ConstantPotentialForce
    force = mm.ConstantPotentialForce()
    
    # 設置 PME 參數
    force.setCutoffDistance(1.0)
    force.setEwaldErrorTolerance(1e-5)
    
    # 添加粒子 (初始電荷 0)
    for i in range(test_system.getNumParticles()):
        if i in electrolyte_indices:
            charge = 1.0 if i % 2 == 0 else -1.0  # 交替正負離子
        else:
            charge = 0.0  # 電極初始電荷
        force.addParticle(charge)
    
    # 添加陰極
    cathode_set = set(cathode_indices)
    cathode_potential = CATHODE_VOLTAGE * VOLTAGE_TO_KJMOL
    force.addElectrode(
        electrodeParticles=cathode_set,
        potential=cathode_potential,
        gaussianWidth=0.05,
        thomasFermiScale=0.0
    )
    print(f"陰極: {len(cathode_set)} 原子, 電壓 = {cathode_potential:.3f} kJ/mol/e")
    
    # 添加陽極
    anode_set = set(anode_indices)
    anode_potential = ANODE_VOLTAGE * VOLTAGE_TO_KJMOL
    force.addElectrode(
        electrodeParticles=anode_set,
        potential=anode_potential,
        gaussianWidth=0.05,
        thomasFermiScale=0.0
    )
    print(f"陽極: {len(anode_set)} 原子, 電壓 = {anode_potential:.3f} kJ/mol/e")
    
    # 設置求解方法
    force.setConstantPotentialMethod(mm.ConstantPotentialForce.Matrix)
    print("求解方法: Matrix (Hessian 預計算)")
    
    test_system.addForce(force)
    
    # 創建 Context
    try:
        platform = mm.Platform.getPlatformByName('CUDA')
        platform_name = 'CUDA'
    except:
        platform = mm.Platform.getPlatformByName('CPU')
        platform_name = 'CPU'
    print(f"平台: {platform_name}")
    print()
    
    integrator = mm.VerletIntegrator(0.001)
    context = mm.Context(test_system, integrator, platform)
    context.setPositions(positions)
    
    # 預熱 (第一次會預計算 Hessian)
    print("預熱 (預計算 Hessian)...")
    t0_init = time.time()
    state = context.getState(getEnergy=True)
    energy_init = state.getPotentialEnergy()
    t_init = time.time() - t0_init
    print(f"  初始化時間: {t_init*1000:.3f} ms")
    print(f"  初始能量: {energy_init}")
    print()
    
    # 基準測試 (多次測量)
    n_iterations = 100
    print(f"基準測試 ({n_iterations} 次迭代)...")
    energies = []
    
    t0_run = time.time()
    for i in range(n_iterations):
        state = context.getState(getEnergy=True)
        energies.append(state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole))
    t_run = time.time() - t0_run
    
    avg_time_per_step = t_run / n_iterations * 1000  # ms
    print(f"  總時間: {t_run:.3f} s")
    print(f"  平均每步: {avg_time_per_step:.3f} ms")
    print(f"  能量穩定性: std = {np.std(energies):.6e} kJ/mol")
    print()
    
    # 獲取電極電荷
    charges = []
    force.getCharges(context, charges)
    
    cathode_charges = [charges[i] for i in cathode_indices]
    anode_charges = [charges[i] for i in anode_indices]
    
    cathode_total = sum(cathode_charges)
    anode_total = sum(anode_charges)
    total_charge = sum(charges)
    
    print("電極電荷:")
    print(f"  陰極總電荷: {cathode_total:.6f} e")
    print(f"  陽極總電荷: {anode_total:.6f} e")
    print(f"  系統總電荷: {total_charge:.6f} e")
    print(f"  電荷相反性: |Q_cathode + Q_anode| = {abs(cathode_total + anode_total):.6e}")
    print()
    
    # 清理
    del context
    del integrator
    
    return {
        'method': 'OpenMM Matrix',
        'platform': platform_name,
        'init_time_ms': t_init * 1000,
        'avg_time_per_step_ms': avg_time_per_step,
        'energy_mean': np.mean(energies),
        'energy_std': np.std(energies),
        'cathode_charge': cathode_total,
        'anode_charge': anode_total,
        'total_charge': total_charge,
        'charge_asymmetry': abs(cathode_total + anode_total)
    }


# ============================================================================
# 測試 2: OpenMM 官方 ConstantPotentialForce (CG 方法)
# ============================================================================

def benchmark_openmm_cg(system, positions, cathode_indices, anode_indices, electrolyte_indices):
    """測試 OpenMM 官方 CG 方法"""
    print("=" * 70)
    print("測試 2: OpenMM 官方 (CG 方法)")
    print("=" * 70)
    
    # 創建新系統
    test_system = mm.System()
    for i in range(system.getNumParticles()):
        test_system.addParticle(system.getParticleMass(i))
    test_system.setDefaultPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())
    
    # 創建 ConstantPotentialForce
    force = mm.ConstantPotentialForce()
    force.setCutoffDistance(1.0)
    force.setEwaldErrorTolerance(1e-5)
    
    # 添加粒子
    for i in range(test_system.getNumParticles()):
        if i in electrolyte_indices:
            charge = 1.0 if i % 2 == 0 else -1.0
        else:
            charge = 0.0
        force.addParticle(charge)
    
    # 添加電極
    force.addElectrode(set(cathode_indices), CATHODE_VOLTAGE * VOLTAGE_TO_KJMOL, 0.05, 0.0)
    force.addElectrode(set(anode_indices), ANODE_VOLTAGE * VOLTAGE_TO_KJMOL, 0.05, 0.0)
    
    # 設置 CG 求解器
    force.setConstantPotentialMethod(mm.ConstantPotentialForce.CG)
    force.setUsePreconditioner(True)
    force.setCGErrorTolerance(0.01)
    print("求解方法: Conjugate Gradient")
    print("  預條件器: 啟用")
    print("  誤差容忍度: 0.01 kJ/mol/e")
    
    test_system.addForce(force)
    
    # 創建 Context
    try:
        platform = mm.Platform.getPlatformByName('CUDA')
        platform_name = 'CUDA'
    except:
        platform = mm.Platform.getPlatformByName('CPU')
        platform_name = 'CPU'
    print(f"平台: {platform_name}")
    print()
    
    integrator = mm.VerletIntegrator(0.001)
    context = mm.Context(test_system, integrator, platform)
    context.setPositions(positions)
    
    # 預熱
    print("預熱...")
    t0_init = time.time()
    state = context.getState(getEnergy=True)
    energy_init = state.getPotentialEnergy()
    t_init = time.time() - t0_init
    print(f"  初始化時間: {t_init*1000:.3f} ms")
    print()
    
    # 基準測試
    n_iterations = 100
    print(f"基準測試 ({n_iterations} 次迭代)...")
    energies = []
    
    t0_run = time.time()
    for i in range(n_iterations):
        state = context.getState(getEnergy=True)
        energies.append(state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole))
    t_run = time.time() - t0_run
    
    avg_time_per_step = t_run / n_iterations * 1000
    print(f"  總時間: {t_run:.3f} s")
    print(f"  平均每步: {avg_time_per_step:.3f} ms")
    print()
    
    # 獲取電荷
    charges = []
    force.getCharges(context, charges)
    
    cathode_total = sum([charges[i] for i in cathode_indices])
    anode_total = sum([charges[i] for i in anode_indices])
    total_charge = sum(charges)
    
    print("電極電荷:")
    print(f"  陰極總電荷: {cathode_total:.6f} e")
    print(f"  陽極總電荷: {anode_total:.6f} e")
    print(f"  系統總電荷: {total_charge:.6f} e")
    print()
    
    del context
    del integrator
    
    return {
        'method': 'OpenMM CG',
        'platform': platform_name,
        'init_time_ms': t_init * 1000,
        'avg_time_per_step_ms': avg_time_per_step,
        'energy_mean': np.mean(energies),
        'energy_std': np.std(energies),
        'cathode_charge': cathode_total,
        'anode_charge': anode_total,
        'total_charge': total_charge,
        'charge_asymmetry': abs(cathode_total + anode_total)
    }


# ============================================================================
# 主函數
# ============================================================================

def main():
    """主測試函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description='性能基準測試')
    parser.add_argument('--n_cathode', type=int, default=100, help='陰極原子數')
    parser.add_argument('--n_anode', type=int, default=100, help='陽極原子數')
    parser.add_argument('--n_electrolyte', type=int, default=1000, help='電解質原子數')
    args = parser.parse_args()
    
    print("\n")
    print("*" * 70)
    print("  三方實現性能基準測試")
    print("  新Plugin vs 舊版Python vs OpenMM官方")
    print("*" * 70)
    print()
    
    # 檢查 OpenMM 版本
    print(f"OpenMM 版本: {mm.version.version}")
    if mm.version.version < '8.4.0':
        print("⚠️  警告: ConstantPotentialForce 需要 OpenMM >= 8.4.0")
        sys.exit(1)
    print()
    
    # 創建測試系統
    system, topology, positions, cathode_indices, anode_indices, electrolyte_indices = \
        create_test_system(args.n_cathode, args.n_anode, args.n_electrolyte)
    
    # 運行測試
    results = []
    
    try:
        result1 = benchmark_openmm_matrix(system, positions, cathode_indices, anode_indices, electrolyte_indices)
        results.append(result1)
    except Exception as e:
        print(f"❌ Matrix 方法失敗: {e}\n")
    
    try:
        result2 = benchmark_openmm_cg(system, positions, cathode_indices, anode_indices, electrolyte_indices)
        results.append(result2)
    except Exception as e:
        print(f"❌ CG 方法失敗: {e}\n")
    
    # 總結對比
    print()
    print("=" * 70)
    print("性能對比總結")
    print("=" * 70)
    print()
    
    if len(results) >= 2:
        print(f"{'方法':<20} {'平台':<10} {'初始化(ms)':<15} {'每步時間(ms)':<15} {'能量(kJ/mol)':<20}")
        print("-" * 80)
        for r in results:
            print(f"{r['method']:<20} {r['platform']:<10} {r['init_time_ms']:>14.3f} "
                  f"{r['avg_time_per_step_ms']:>14.3f} {r['energy_mean']:>19.3f}")
        print()
        
        print("電荷對比:")
        print(f"{'方法':<20} {'陰極電荷(e)':<15} {'陽極電荷(e)':<15} {'總電荷(e)':<15}")
        print("-" * 65)
        for r in results:
            print(f"{r['method']:<20} {r['cathode_charge']:>14.6f} {r['anode_charge']:>14.6f} {r['total_charge']:>14.6f}")
        print()
        
        # 速度對比
        if len(results) == 2:
            speedup = results[0]['avg_time_per_step_ms'] / results[1]['avg_time_per_step_ms']
            print(f"Matrix vs CG 速度比: {speedup:.2f}x")
    
    print()
    print("*" * 70)
    print("  測試完成!")
    print("*" * 70)
    print()


if __name__ == '__main__':
    main()
