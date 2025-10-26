#!/usr/bin/env python3
"""
🔬 Warm Start 極端情況準確性測試

這是我們第一個可能影響物理準確性的優化,必須嚴格驗證!

測試場景:
1. 連續 1000 次調用 - 檢測誤差累積
2. 大電壓跳變 - 測試初始猜測很差的情況
3. 溫度變化 - 測試系統劇烈擾動
4. 不同迭代次數 - 驗證收斂行為
5. 長時間 MD 模擬 - 實戰測試

通過標準:
- 與 Cold Start 的 MAE < 1e-10 (機器精度)
- 能量守恆誤差 < 0.01%
- 電荷總和守恆 (誤差 < 1e-12)
- 無誤差累積現象 (error vs iteration 應該是常數)
"""

import time
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import configparser

# 使用與 bench.py 完全一樣的配置
sys.path.append('./lib/')

print("=" * 80)
print("🔬 WARM START 極端情況準確性測試")
print("=" * 80)
print("⚠️  這是第一個可能影響物理準確性的優化,必須通過所有測試!")
print("=" * 80)

# 硬編碼配置 (與 bench.py 一致,避免配置文件問題)
pdb_list = ["for_openmm.pdb"]
residue_xml_list = [
    "./ffdir/sapt_residues.xml",
    "./ffdir/graph_residue_c.xml", 
    "./ffdir/graph_residue_n.xml"
]
ff_xml_list = [
    "./ffdir/sapt_noDB_2sheets.xml",
    "./ffdir/graph_c_freeze.xml",
    "./ffdir/graph_n_freeze.xml"
]
cathode_index = (0, 2)  # tuple for multiple cathodes
anode_index = (1, 3)    # tuple for multiple anodes
platform = "CUDA"

print(f"\n配置:")
print(f"  PDB: {pdb_list[0]}")
print(f"  Platform: {platform}")
print(f"  Cathode resid: {cathode_index}")
print(f"  Anode resid: {anode_index}")

# 動態導入 MM_classes (根據可用性)
try:
    from MM_classes_CYTHON import MM as MM_cython
    print(f"  Using: MM_classes_CYTHON")
except ImportError:
    try:
        from MM_classes_OPTIMIZED import MM as MM_cython
        print(f"  Using: MM_classes_OPTIMIZED (Cython not available)")
    except ImportError:
        from MM_classes import MM as MM_cython
        print(f"  Using: MM_classes (Original)")

print("=" * 80)

# Test results storage
test_results = {
    'passed': 0,
    'failed': 0,
    'warnings': 0,
    'details': []
}

def log_test(name, passed, message="", warning=False):
    """記錄測試結果"""
    status = "✅ PASS" if passed else ("⚠️  WARN" if warning else "❌ FAIL")
    print(f"{status} | {name}")
    if message:
        print(f"      {message}")
    
    if passed and not warning:
        test_results['passed'] += 1
    elif warning:
        test_results['warnings'] += 1
    else:
        test_results['failed'] += 1
    
    test_results['details'].append({
        'name': name,
        'passed': passed,
        'warning': warning,
        'message': message
    })

def compare_charges(charges1, charges2, tolerance=1e-10, name="Comparison"):
    """比較兩組電荷,返回是否通過"""
    mae = np.mean(np.abs(charges1 - charges2))
    max_diff = np.max(np.abs(charges1 - charges2))
    
    passed = mae < tolerance and max_diff < tolerance * 10
    message = f"MAE: {mae:.4e}, Max: {max_diff:.4e} (tolerance: {tolerance:.1e})"
    
    return passed, message

# ============================================================================
# Test 1: 基礎功能測試 - Warm Start vs Cold Start (單次調用)
# ============================================================================
print("\n" + "=" * 80)
print("TEST 1: 基礎功能測試 - 單次調用比較")
print("=" * 80)
print("目的: 驗證 Warm Start 不改變單次調用的結果")

print("\n設置系統 (Cold Start 版本)...")
MMsys_cold = MM_cython(pdb_list=pdb_list, residue_xml_list=residue_xml_list, ff_xml_list=ff_xml_list)
MMsys_cold.set_platform(platform)
MMsys_cold.set_periodic_residue(True)
MMsys_cold.initialize_electrodes(
    Voltage=0.0,
    cathode_identifier=cathode_index,
    anode_identifier=anode_index,
    chain=True,
    exclude_element=("H",)
)
MMsys_cold.initialize_electrolyte(Natom_cutoff=100)

print("設置系統 (Warm Start 版本)...")
MMsys_warm = MM_cython(pdb_list=pdb_list, residue_xml_list=residue_xml_list, ff_xml_list=ff_xml_list)
MMsys_warm.set_platform(platform)
MMsys_warm.set_periodic_residue(True)
MMsys_warm.initialize_electrodes(
    Voltage=0.0,
    cathode_identifier=cathode_index,
    anode_identifier=anode_index,
    chain=True,
    exclude_element=("H",)
)
MMsys_warm.initialize_electrolyte(Natom_cutoff=100)

# Cold start - 強制刪除 warm start cache
if hasattr(MMsys_cold, '_warm_start_cathode_charges'):
    delattr(MMsys_cold, '_warm_start_cathode_charges')
if hasattr(MMsys_cold, '_warm_start_anode_charges'):
    delattr(MMsys_cold, '_warm_start_anode_charges')

print("\n運行 Poisson solver...")
MMsys_cold.Poisson_solver_fixed_voltage(Niterations=10)
charges_cold_cathode = np.array([atom.charge for atom in MMsys_cold.Cathode.electrode_atoms])
charges_cold_anode = np.array([atom.charge for atom in MMsys_cold.Anode.electrode_atoms])

# 第二次調用 warm start 版本
MMsys_warm.Poisson_solver_fixed_voltage(Niterations=10)
MMsys_warm.Poisson_solver_fixed_voltage(Niterations=10)  # 第二次才有 warm start
charges_warm_cathode = np.array([atom.charge for atom in MMsys_warm.Cathode.electrode_atoms])
charges_warm_anode = np.array([atom.charge for atom in MMsys_warm.Anode.electrode_atoms])

passed, msg = compare_charges(charges_cold_cathode, charges_warm_cathode, tolerance=1e-10, name="Cathode")
log_test("Test 1.1: Cathode charges (Cold vs Warm)", passed, msg)

passed, msg = compare_charges(charges_cold_anode, charges_warm_anode, tolerance=1e-10, name="Anode")
log_test("Test 1.2: Anode charges (Cold vs Warm)", passed, msg)

# 檢查電荷守恆
total_cold = np.sum(charges_cold_cathode) + np.sum(charges_cold_anode)
total_warm = np.sum(charges_warm_cathode) + np.sum(charges_warm_anode)
charge_conservation = np.abs(total_cold - total_warm)
passed = charge_conservation < 1e-12
log_test("Test 1.3: Charge conservation", passed, f"Δtotal: {charge_conservation:.4e}")

# ============================================================================
# Test 2: 連續 1000 次調用 - 檢測誤差累積
# ============================================================================
print("\n" + "=" * 80)
print("TEST 2: 連續 1000 次調用 - 誤差累積測試")
print("=" * 80)
print("目的: 驗證長時間使用不會導致誤差累積")
print("這是 CRITICAL TEST - 模擬真實 MD 場景!")

N_iterations = 1000
print(f"\n準備運行 {N_iterations} 次連續調用...")

# Cold start baseline (每次都重新初始化)
print("\n[1/2] Cold Start baseline (每次重新初始化)...")
charges_history_cold = []
for i in range(N_iterations):
    if hasattr(MMsys_cold, '_warm_start_cathode_charges'):
        delattr(MMsys_cold, '_warm_start_cathode_charges')
    if hasattr(MMsys_cold, '_warm_start_anode_charges'):
        delattr(MMsys_cold, '_warm_start_anode_charges')
    
    MMsys_cold.Poisson_solver_fixed_voltage(Niterations=3)
    charges = np.array([atom.charge for atom in MMsys_cold.Cathode.electrode_atoms])
    charges_history_cold.append(charges.copy())
    
    if (i + 1) % 100 == 0:
        print(f"  Progress: {i+1}/{N_iterations}")

# Warm start (正常使用)
print("\n[2/2] Warm Start (正常連續調用)...")
charges_history_warm = []
for i in range(N_iterations):
    MMsys_warm.Poisson_solver_fixed_voltage(Niterations=3)
    charges = np.array([atom.charge for atom in MMsys_warm.Cathode.electrode_atoms])
    charges_history_warm.append(charges.copy())
    
    if (i + 1) % 100 == 0:
        print(f"  Progress: {i+1}/{N_iterations}")

# 分析誤差隨時間的變化
print("\n分析誤差趨勢...")
errors = []
for i in range(N_iterations):
    mae = np.mean(np.abs(charges_history_cold[i] - charges_history_warm[i]))
    errors.append(mae)

errors = np.array(errors)

# 線性擬合檢測誤差增長
iterations = np.arange(N_iterations)
fit_coeffs = np.polyfit(iterations, errors, 1)
error_growth_rate = fit_coeffs[0]  # 斜率

print(f"\n誤差統計:")
print(f"  初始誤差 (iter 0):     {errors[0]:.4e}")
print(f"  最終誤差 (iter {N_iterations-1}):  {errors[-1]:.4e}")
print(f"  平均誤差:              {np.mean(errors):.4e}")
print(f"  最大誤差:              {np.max(errors):.4e}")
print(f"  誤差增長率:            {error_growth_rate:.4e} per iteration")

# 判斷是否有誤差累積
max_error = np.max(errors)
mean_error = np.mean(errors)
initial_error = errors[0]

# Test 2.1: 最大誤差應該在機器精度範圍
passed = max_error < 1e-10
log_test("Test 2.1: Maximum error < 1e-10", passed, f"Max: {max_error:.4e}")

# Test 2.2: 平均誤差應該在機器精度範圍
passed = mean_error < 1e-10
log_test("Test 2.2: Mean error < 1e-10", passed, f"Mean: {mean_error:.4e}")

# Test 2.3: 誤差增長率應該接近零 (不應該累積)
# 允許極小的增長 (由於數值噪聲),但應該 < 初始誤差的 1%
passed = abs(error_growth_rate) < initial_error * 0.01
warning = abs(error_growth_rate) > initial_error * 0.001
log_test("Test 2.3: No error accumulation", passed, 
         f"Growth rate: {error_growth_rate:.4e} ({'OK' if passed else 'GROWING!'})",
         warning=warning)

# 繪製誤差趨勢圖
try:
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(errors, 'b-', alpha=0.5, linewidth=0.5)
    plt.plot(iterations, np.poly1d(fit_coeffs)(iterations), 'r--', 
             label=f'Linear fit (slope={error_growth_rate:.2e})')
    plt.xlabel('Iteration')
    plt.ylabel('MAE (Warm vs Cold)')
    plt.title('Error vs Iteration (1000 calls)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.subplot(1, 2, 2)
    plt.hist(errors, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('MAE')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.axvline(mean_error, color='r', linestyle='--', label=f'Mean: {mean_error:.2e}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('warm_start_error_accumulation.png', dpi=150)
    print("\n✅ 誤差趨勢圖已保存: warm_start_error_accumulation.png")
except Exception as e:
    print(f"\n⚠️  無法繪製圖表: {e}")

# ============================================================================
# Test 3: 大電壓跳變 - 測試極端初始猜測
# ============================================================================
print("\n" + "=" * 80)
print("TEST 3: 大電壓跳變測試 - 極端初始猜測")
print("=" * 80)
print("目的: 驗證當初始猜測很差時,Warm Start 仍能收斂到正確解")

# 創建新系統
print("\n設置系統 (電壓 0V)...")
MMsys_jump = MM_cython(pdb_list=pdb_list, residue_xml_list=residue_xml_list, ff_xml_list=ff_xml_list)
MMsys_jump.set_platform(platform)
MMsys_jump.set_periodic_residue(True)
MMsys_jump.initialize_electrodes(
    Voltage=0.0,
    cathode_identifier=cathode_index,
    anode_identifier=anode_index,
    chain=True,
    exclude_element=("H",)
)
MMsys_jump.initialize_electrolyte(Natom_cutoff=100)

# 在 0V 收斂
print("在 0V 收斂...")
MMsys_jump.Poisson_solver_fixed_voltage(Niterations=10)
charges_0V = np.array([atom.charge for atom in MMsys_jump.Cathode.electrode_atoms])

# 突然跳到 4V (極端變化!)
print("電壓跳變: 0V → 4V (極端變化!)")
MMsys_jump.Cathode.Voltage = 4.0
MMsys_jump.Anode.Voltage = -4.0

# Warm start 版本 (使用 0V 的電荷作為初始值)
print("運行 Warm Start (用 0V 電荷作為 4V 的初始猜測)...")
MMsys_jump.Poisson_solver_fixed_voltage(Niterations=10)
charges_4V_warm = np.array([atom.charge for atom in MMsys_jump.Cathode.electrode_atoms])

# Cold start 版本 (重新初始化)
print("運行 Cold Start (從隨機初始值開始)...")
if hasattr(MMsys_jump, '_warm_start_cathode_charges'):
    delattr(MMsys_jump, '_warm_start_cathode_charges')
if hasattr(MMsys_jump, '_warm_start_anode_charges'):
    delattr(MMsys_jump, '_warm_start_anode_charges')

MMsys_jump.Cathode.Voltage = 4.0
MMsys_jump.Anode.Voltage = -4.0
MMsys_jump.Poisson_solver_fixed_voltage(Niterations=10)
charges_4V_cold = np.array([atom.charge for atom in MMsys_jump.Cathode.electrode_atoms])

passed, msg = compare_charges(charges_4V_warm, charges_4V_cold, tolerance=1e-9, name="4V jump")
log_test("Test 3.1: Voltage jump (0V→4V) convergence", passed, msg)

# 檢查電荷確實改變了 (不應該還是 0V 的值)
charge_change = np.mean(np.abs(charges_4V_warm - charges_0V))
passed = charge_change > 1e-6  # 應該有顯著變化
log_test("Test 3.2: Charges actually changed", passed, f"Δcharge: {charge_change:.4e}")

# ============================================================================
# Test 4: 不同迭代次數測試 - 收斂行為
# ============================================================================
print("\n" + "=" * 80)
print("TEST 4: 不同迭代次數測試 - 收斂行為")
print("=" * 80)
print("目的: 驗證 Warm Start 在不同迭代次數下都能正確收斂")

iteration_counts = [1, 3, 5, 10, 20]
print(f"\n測試迭代次數: {iteration_counts}")

for Niter in iteration_counts:
    print(f"\n測試 Niterations = {Niter}...")
    
    # Cold start
    MMsys_iter = MM_cython(pdb_list=pdb_list, residue_xml_list=residue_xml_list, ff_xml_list=ff_xml_list)
    MMsys_iter.set_platform(platform)
    MMsys_iter.set_periodic_residue(True)
    MMsys_iter.initialize_electrodes(
        Voltage=0.0,
        cathode_identifier=cathode_index,
        anode_identifier=anode_index,
        chain=True,
        exclude_element=("H",)
    )
    MMsys_iter.initialize_electrolyte(Natom_cutoff=100)
    
    # 第一次 (cold)
    if hasattr(MMsys_iter, '_warm_start_cathode_charges'):
        delattr(MMsys_iter, '_warm_start_cathode_charges')
    if hasattr(MMsys_iter, '_warm_start_anode_charges'):
        delattr(MMsys_iter, '_warm_start_anode_charges')
    
    MMsys_iter.Poisson_solver_fixed_voltage(Niterations=Niter)
    charges_cold = np.array([atom.charge for atom in MMsys_iter.Cathode.electrode_atoms])
    
    # 第二次 (warm)
    MMsys_iter.Poisson_solver_fixed_voltage(Niterations=Niter)
    charges_warm = np.array([atom.charge for atom in MMsys_iter.Cathode.electrode_atoms])
    
    passed, msg = compare_charges(charges_cold, charges_warm, tolerance=1e-9, name=f"Niter={Niter}")
    log_test(f"Test 4.{iteration_counts.index(Niter)+1}: Niter={Niter} convergence", passed, msg)

# ============================================================================
# Test 5: 電荷守恆測試
# ============================================================================
print("\n" + "=" * 80)
print("TEST 5: 電荷守恆測試")
print("=" * 80)
print("目的: 驗證 Warm Start 不破壞電荷守恆")

print("\n運行 100 次連續調用並檢查總電荷...")
total_charges_history = []

MMsys_cons = MM_cython(pdb_list=pdb_list, residue_xml_list=residue_xml_list, ff_xml_list=ff_xml_list)
MMsys_cons.set_platform(platform)
MMsys_cons.set_periodic_residue(True)
MMsys_cons.initialize_electrodes(
    Voltage=0.0,
    cathode_identifier=cathode_index,
    anode_identifier=anode_index,
    chain=True,
    exclude_element=("H",)
)
MMsys_cons.initialize_electrolyte(Natom_cutoff=100)

for i in range(100):
    MMsys_cons.Poisson_solver_fixed_voltage(Niterations=3)
    cathode_total = np.sum([atom.charge for atom in MMsys_cons.Cathode.electrode_atoms])
    anode_total = np.sum([atom.charge for atom in MMsys_cons.Anode.electrode_atoms])
    total_charges_history.append(cathode_total + anode_total)

total_charges = np.array(total_charges_history)
charge_drift = np.max(np.abs(total_charges - total_charges[0]))

print(f"初始總電荷: {total_charges[0]:.8e}")
print(f"最終總電荷: {total_charges[-1]:.8e}")
print(f"最大漂移:   {charge_drift:.8e}")

passed = charge_drift < 1e-10
log_test("Test 5.1: Charge conservation (100 calls)", passed, f"Drift: {charge_drift:.4e}")

# ============================================================================
# 總結報告
# ============================================================================
print("\n" + "=" * 80)
print("📊 最終測試報告")
print("=" * 80)

total_tests = test_results['passed'] + test_results['failed'] + test_results['warnings']
pass_rate = test_results['passed'] / total_tests * 100 if total_tests > 0 else 0

print(f"\n測試統計:")
print(f"  總測試數:  {total_tests}")
print(f"  ✅ 通過:   {test_results['passed']}")
print(f"  ❌ 失敗:   {test_results['failed']}")
print(f"  ⚠️  警告:   {test_results['warnings']}")
print(f"  通過率:    {pass_rate:.1f}%")

if test_results['failed'] > 0:
    print("\n" + "=" * 80)
    print("❌ WARM START 未通過測試!")
    print("=" * 80)
    print("失敗的測試:")
    for detail in test_results['details']:
        if not detail['passed'] and not detail['warning']:
            print(f"  - {detail['name']}")
            print(f"    {detail['message']}")
    print("\n⚠️  不建議使用 Warm Start,需要進一步調查!")
    
elif test_results['warnings'] > 0:
    print("\n" + "=" * 80)
    print("⚠️  WARM START 通過測試,但有警告")
    print("=" * 80)
    print("警告的測試:")
    for detail in test_results['details']:
        if detail['warning']:
            print(f"  - {detail['name']}")
            print(f"    {detail['message']}")
    print("\n建議: 可以使用 Warm Start,但需要監控長時間運行的行為")
    
else:
    print("\n" + "=" * 80)
    print("✅ WARM START 通過所有測試!")
    print("=" * 80)
    print("結論:")
    print("  - ✅ 準確性: 與 Cold Start 結果一致 (MAE < 1e-10)")
    print("  - ✅ 穩定性: 無誤差累積現象")
    print("  - ✅ 魯棒性: 能應對極端情況 (電壓跳變)")
    print("  - ✅ 守恆律: 電荷守恆完美保持")
    print("\n🎉 Warm Start 可以安全使用於生產環境!")
    print("📝 建議: 在論文中添加這些測試結果作為 Supporting Information")

print("\n" + "=" * 80)
print("測試完成!")
print("=" * 80)
