#!/usr/bin/env python
"""
🔥 Good Taste Version Benchmark
================================

測試目標：
1. 數值正確性：Cython 版 vs 原始版的計算結果一致性
2. 性能穩定性：隨粒子數增加，加速比的穩定性
3. 計算/同步分離：驗證新架構的性能優勢

測試方法：
- 模擬真實的 Poisson solver 計算流程
- 測試多種電極尺寸（100 到 10000 原子）
- 分別測試：純計算部分 vs 完整流程（含同步）
"""

import numpy as np
import time
import sys
import os

# Add lib directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

# Try to import Cython module
try:
    import electrode_charges_cython as ec_cython
    CYTHON_AVAILABLE = True
    print("✅ Cython 模組載入成功")
except ImportError as e:
    CYTHON_AVAILABLE = False
    print(f"❌ Cython 模組載入失敗: {e}")
    sys.exit(1)

# Constants (from original code)
conversion_nmBohr = 18.8973
conversion_KjmolNm_Au = conversion_nmBohr / 2625.5


class MockAtom:
    """模擬 atom_MM 物件"""
    def __init__(self, atom_index, charge):
        self.atom_index = atom_index
        self.charge = charge


class MockNonbondedForce:
    """模擬 OpenMM NonbondedForce（用於測試同步開銷）"""
    def __init__(self, n_atoms):
        self.n_atoms = n_atoms
        self.charges = np.zeros(n_atoms)
    
    def setParticleParameters(self, index, charge, sigma, epsilon):
        """模擬 API 呼叫（故意加入一點開銷）"""
        self.charges[index] = charge
    
    def updateParametersInContext(self, context):
        """模擬 API 呼叫"""
        pass


def create_test_system(n_electrode_atoms):
    """
    創建測試系統
    
    Parameters:
    -----------
    n_electrode_atoms : int
        每個電極的原子數
    
    Returns:
    --------
    dict : 包含測試所需的所有數據
    """
    total_atoms = n_electrode_atoms * 2 + 5000  # 兩個電極 + 電解質
    
    # 創建電極原子（模擬 electrode_atoms 列表）
    cathode_atoms = [
        MockAtom(i, 0.001 * (1 + 0.1 * np.random.randn()))
        for i in range(n_electrode_atoms)
    ]
    
    anode_atoms = [
        MockAtom(n_electrode_atoms + i, -0.001 * (1 + 0.1 * np.random.randn()))
        for i in range(n_electrode_atoms)
    ]
    
    # C 陣列（Good Taste 版本）
    c_indices_cathode = np.array([atom.atom_index for atom in cathode_atoms], dtype=np.int64)
    c_charges_cathode = np.array([atom.charge for atom in cathode_atoms], dtype=np.float64)
    
    c_indices_anode = np.array([atom.atom_index for atom in anode_atoms], dtype=np.int64)
    c_charges_anode = np.array([atom.charge for atom in anode_atoms], dtype=np.float64)
    
    # 模擬 forces（完整系統）
    forces_z = np.random.randn(total_atoms) * 50.0 + 100.0
    
    # 物理參數
    area_atom = 0.05  # nm^2
    voltage = 2.0     # V (已轉換為 kJ/mol)
    Lgap = 10.0       # nm
    small_threshold = 1e-6
    
    # 計算預因子（cathode 正，anode 負）
    coeff = 2.0 / (4.0 * np.pi)
    cathode_prefactor = coeff * area_atom * conversion_KjmolNm_Au
    anode_prefactor = -coeff * area_atom * conversion_KjmolNm_Au
    voltage_term = voltage / Lgap
    threshold_check = 0.9 * small_threshold
    
    # Mock OpenMM force
    nbondedForce = MockNonbondedForce(total_atoms)
    
    return {
        'cathode_atoms': cathode_atoms,
        'anode_atoms': anode_atoms,
        'c_indices_cathode': c_indices_cathode,
        'c_charges_cathode': c_charges_cathode,
        'c_indices_anode': c_indices_anode,
        'c_charges_anode': c_charges_anode,
        'forces_z': forces_z,
        'cathode_prefactor': cathode_prefactor,
        'anode_prefactor': anode_prefactor,
        'voltage_term': voltage_term,
        'threshold_check': threshold_check,
        'small_threshold': small_threshold,
        'nbondedForce': nbondedForce,
        'n_atoms': n_electrode_atoms
    }


#============================================================================
# 原始版本（假優化）
#============================================================================

def compute_original_computation(data):
    """
    原始版本：純計算部分（Python 循環）
    只測試計算，不包含 API 同步
    """
    cathode_atoms = data['cathode_atoms']
    anode_atoms = data['anode_atoms']
    forces_z = data['forces_z']
    cathode_prefactor = data['cathode_prefactor']
    anode_prefactor = data['anode_prefactor']
    voltage_term = data['voltage_term']
    threshold_check = data['threshold_check']
    small_threshold = data['small_threshold']
    
    # Cathode 計算
    cathode_charges_new = np.zeros(len(cathode_atoms))
    for i, atom in enumerate(cathode_atoms):
        q_old = atom.charge
        
        if abs(q_old) > threshold_check:
            Ez_external = forces_z[atom.atom_index] / q_old
        else:
            Ez_external = 0.0
        
        q_new = cathode_prefactor * (voltage_term + Ez_external)
        
        if abs(q_new) < small_threshold:
            q_new = small_threshold
        
        cathode_charges_new[i] = q_new
    
    # Anode 計算
    anode_charges_new = np.zeros(len(anode_atoms))
    for i, atom in enumerate(anode_atoms):
        q_old = atom.charge
        
        if abs(q_old) > threshold_check:
            Ez_external = forces_z[atom.atom_index] / q_old
        else:
            Ez_external = 0.0
        
        q_new = anode_prefactor * (voltage_term + Ez_external)
        
        if abs(q_new) < small_threshold:
            q_new = -small_threshold
        
        anode_charges_new[i] = q_new
    
    return cathode_charges_new, anode_charges_new


def compute_original_full(data):
    """
    原始版本：完整流程（計算 + 同步）
    包含對 atom.charge 和 setParticleParameters 的呼叫
    """
    cathode_charges_new, anode_charges_new = compute_original_computation(data)
    
    # 同步到 atom 物件和 OpenMM
    cathode_atoms = data['cathode_atoms']
    anode_atoms = data['anode_atoms']
    nbondedForce = data['nbondedForce']
    
    for i, atom in enumerate(cathode_atoms):
        atom.charge = cathode_charges_new[i]
        nbondedForce.setParticleParameters(atom.atom_index, cathode_charges_new[i], 1.0, 0.0)
    
    for i, atom in enumerate(anode_atoms):
        atom.charge = anode_charges_new[i]
        nbondedForce.setParticleParameters(atom.atom_index, anode_charges_new[i], 1.0, 0.0)
    
    nbondedForce.updateParametersInContext(None)
    
    return cathode_charges_new, anode_charges_new


#============================================================================
# Good Taste 版本（真優化）
#============================================================================

def compute_goodtaste_computation(data):
    """
    Good Taste 版本：純計算部分（Cython）
    只測試計算，不包含 API 同步
    """
    forces_z = data['forces_z']
    c_charges_cathode = data['c_charges_cathode'].copy()  # 複製避免修改原始數據
    c_charges_anode = data['c_charges_anode'].copy()
    c_indices_cathode = data['c_indices_cathode']
    c_indices_anode = data['c_indices_anode']
    cathode_prefactor = data['cathode_prefactor']
    anode_prefactor = data['anode_prefactor']
    voltage_term = data['voltage_term']
    threshold_check = data['threshold_check']
    small_threshold = data['small_threshold']
    
    # Cathode 計算（Cython）
    cathode_charges_new = ec_cython.compute_electrode_charges_cython(
        forces_z,
        c_charges_cathode,
        c_indices_cathode,
        cathode_prefactor,
        voltage_term,
        threshold_check,
        small_threshold,
        1.0  # sign = +1 for cathode
    )
    
    # Anode 計算（Cython）
    anode_charges_new = ec_cython.compute_electrode_charges_cython(
        forces_z,
        c_charges_anode,
        c_indices_anode,
        anode_prefactor,
        voltage_term,
        threshold_check,
        small_threshold,
        -1.0  # sign = -1 for anode
    )
    
    return cathode_charges_new, anode_charges_new


def compute_goodtaste_full(data):
    """
    Good Taste 版本：完整流程（計算 + 同步）
    分離：Cython 計算 → Python 同步
    """
    cathode_charges_new, anode_charges_new = compute_goodtaste_computation(data)
    
    # 同步到 C 陣列
    data['c_charges_cathode'][:] = cathode_charges_new
    data['c_charges_anode'][:] = anode_charges_new
    
    # 同步到 atom 物件和 OpenMM（Python 層）
    cathode_atoms = data['cathode_atoms']
    anode_atoms = data['anode_atoms']
    c_indices_cathode = data['c_indices_cathode']
    c_indices_anode = data['c_indices_anode']
    nbondedForce = data['nbondedForce']
    
    for i in range(len(cathode_atoms)):
        idx = c_indices_cathode[i]
        q = cathode_charges_new[i]
        cathode_atoms[i].charge = q
        nbondedForce.setParticleParameters(idx, q, 1.0, 0.0)
    
    for i in range(len(anode_atoms)):
        idx = c_indices_anode[i]
        q = anode_charges_new[i]
        anode_atoms[i].charge = q
        nbondedForce.setParticleParameters(idx, q, 1.0, 0.0)
    
    nbondedForce.updateParametersInContext(None)
    
    return cathode_charges_new, anode_charges_new


#============================================================================
# Benchmarking Functions
#============================================================================

def check_correctness(data):
    """
    檢查數值正確性：Cython 版 vs 原始版
    """
    print("\n" + "=" * 70)
    print("📐 數值正確性檢查")
    print("=" * 70)
    
    # 計算兩個版本的結果
    orig_cathode, orig_anode = compute_original_computation(data)
    good_cathode, good_anode = compute_goodtaste_computation(data)
    
    # 計算差異
    diff_cathode = np.abs(orig_cathode - good_cathode)
    diff_anode = np.abs(orig_anode - good_anode)
    
    max_diff_cathode = np.max(diff_cathode)
    max_diff_anode = np.max(diff_anode)
    mean_diff_cathode = np.mean(diff_cathode)
    mean_diff_anode = np.mean(diff_anode)
    
    print(f"\nCathode ({len(orig_cathode)} 原子):")
    print(f"  最大差異: {max_diff_cathode:.2e}")
    print(f"  平均差異: {mean_diff_cathode:.2e}")
    print(f"  相對誤差: {max_diff_cathode / np.mean(np.abs(orig_cathode)):.2e}")
    
    print(f"\nAnode ({len(orig_anode)} 原子):")
    print(f"  最大差異: {max_diff_anode:.2e}")
    print(f"  平均差異: {mean_diff_anode:.2e}")
    print(f"  相對誤差: {max_diff_anode / np.mean(np.abs(orig_anode)):.2e}")
    
    max_diff = max(max_diff_cathode, max_diff_anode)
    tolerance = 1e-10
    
    if max_diff < tolerance:
        print(f"\n✅ 通過：所有數值一致（最大差異 {max_diff:.2e} < {tolerance:.2e}）")
        return True
    else:
        print(f"\n❌ 失敗：數值差異超過容許範圍（{max_diff:.2e} >= {tolerance:.2e}）")
        return False


def benchmark_function(func, data, name, warmup=10, iterations=100):
    """
    Benchmark 單個函數
    """
    # Warmup
    for _ in range(warmup):
        func(data)
    
    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(data)
        end = time.perf_counter()
        times.append((end - start) * 1e6)  # 轉換為微秒
    
    times = np.array(times)
    return {
        'name': name,
        'mean': np.mean(times),
        'std': np.std(times),
        'median': np.median(times),
        'min': np.min(times),
        'max': np.max(times)
    }


def benchmark_single_size(n_atoms, warmup=10, iterations=100):
    """
    對單一尺寸進行 benchmark
    """
    print(f"\n{'=' * 70}")
    print(f"測試尺寸: {n_atoms} 原子/電極")
    print(f"{'=' * 70}")
    
    data = create_test_system(n_atoms)
    
    # Benchmark 計算部分（純數學）
    print("\n📊 純計算部分（不含 API 同步）:")
    result_orig_comp = benchmark_function(
        compute_original_computation, data,
        "原始版本（Python 循環）",
        warmup=warmup, iterations=iterations
    )
    result_good_comp = benchmark_function(
        compute_goodtaste_computation, data,
        "Good Taste（Cython）",
        warmup=warmup, iterations=iterations
    )
    
    speedup_comp = result_orig_comp['mean'] / result_good_comp['mean']
    
    print(f"  原始版本:   {result_orig_comp['mean']:>8.2f} ± {result_orig_comp['std']:>6.2f} μs")
    print(f"  Good Taste: {result_good_comp['mean']:>8.2f} ± {result_good_comp['std']:>6.2f} μs")
    print(f"  加速比:     {speedup_comp:>8.2f}x")
    
    # Benchmark 完整流程（含 API 同步）
    print("\n📊 完整流程（計算 + API 同步）:")
    result_orig_full = benchmark_function(
        compute_original_full, data,
        "原始版本（混雜）",
        warmup=warmup, iterations=iterations
    )
    result_good_full = benchmark_function(
        compute_goodtaste_full, data,
        "Good Taste（分離）",
        warmup=warmup, iterations=iterations
    )
    
    speedup_full = result_orig_full['mean'] / result_good_full['mean']
    
    print(f"  原始版本:   {result_orig_full['mean']:>8.2f} ± {result_orig_full['std']:>6.2f} μs")
    print(f"  Good Taste: {result_good_full['mean']:>8.2f} ± {result_good_full['std']:>6.2f} μs")
    print(f"  加速比:     {speedup_full:>8.2f}x")
    
    # 分析同步開銷
    sync_overhead_orig = result_orig_full['mean'] - result_orig_comp['mean']
    sync_overhead_good = result_good_full['mean'] - result_good_comp['mean']
    
    print(f"\n📊 API 同步開銷分析:")
    print(f"  原始版本:   {sync_overhead_orig:>8.2f} μs ({sync_overhead_orig/result_orig_full['mean']*100:.1f}%)")
    print(f"  Good Taste: {sync_overhead_good:>8.2f} μs ({sync_overhead_good/result_good_full['mean']*100:.1f}%)")
    
    return {
        'n_atoms': n_atoms,
        'speedup_computation': speedup_comp,
        'speedup_full': speedup_full,
        'time_orig_comp': result_orig_comp['mean'],
        'time_good_comp': result_good_comp['mean'],
        'time_orig_full': result_orig_full['mean'],
        'time_good_full': result_good_full['mean'],
        'sync_overhead_orig': sync_overhead_orig,
        'sync_overhead_good': sync_overhead_good
    }


def test_scaling(sizes=[100, 300, 500, 1000, 2000, 5000]):
    """
    測試性能隨粒子數的變化
    """
    print("\n" + "=" * 70)
    print("🚀 性能擴展性測試（Scaling Test）")
    print("=" * 70)
    print("測試不同電極尺寸下的加速比穩定性")
    
    results = []
    
    for n in sizes:
        # 根據尺寸調整迭代次數
        if n <= 500:
            warmup, iterations = 20, 200
        elif n <= 2000:
            warmup, iterations = 10, 100
        else:
            warmup, iterations = 5, 50
        
        result = benchmark_single_size(n, warmup=warmup, iterations=iterations)
        results.append(result)
    
    # 打印總結表格
    print("\n" + "=" * 70)
    print("📊 擴展性測試總結")
    print("=" * 70)
    
    print("\n【純計算部分】")
    print(f"{'尺寸':<10} {'原始版本':<15} {'Good Taste':<15} {'加速比':<10}")
    print("-" * 50)
    for r in results:
        print(f"{r['n_atoms']:<10} {r['time_orig_comp']:>10.2f} μs   {r['time_good_comp']:>10.2f} μs   {r['speedup_computation']:>6.2f}x")
    
    print("\n【完整流程（含同步）】")
    print(f"{'尺寸':<10} {'原始版本':<15} {'Good Taste':<15} {'加速比':<10}")
    print("-" * 50)
    for r in results:
        print(f"{r['n_atoms']:<10} {r['time_orig_full']:>10.2f} μs   {r['time_good_full']:>10.2f} μs   {r['speedup_full']:>6.2f}x")
    
    # 分析加速比穩定性
    speedups_comp = [r['speedup_computation'] for r in results]
    speedups_full = [r['speedup_full'] for r in results]
    
    print("\n【加速比穩定性分析】")
    print(f"純計算部分:")
    print(f"  平均加速比: {np.mean(speedups_comp):.2f}x")
    print(f"  標準差:     {np.std(speedups_comp):.2f}x")
    print(f"  變異係數:   {np.std(speedups_comp)/np.mean(speedups_comp)*100:.1f}%")
    
    print(f"\n完整流程:")
    print(f"  平均加速比: {np.mean(speedups_full):.2f}x")
    print(f"  標準差:     {np.std(speedups_full):.2f}x")
    print(f"  變異係數:   {np.std(speedups_full)/np.mean(speedups_full)*100:.1f}%")
    
    # 判斷穩定性
    cv_comp = np.std(speedups_comp) / np.mean(speedups_comp)
    cv_full = np.std(speedups_full) / np.mean(speedups_full)
    
    print("\n【結論】")
    if cv_comp < 0.1 and cv_full < 0.1:
        print("✅ 加速比非常穩定（變異係數 < 10%）")
    elif cv_comp < 0.2 and cv_full < 0.2:
        print("✅ 加速比穩定（變異係數 < 20%）")
    else:
        print("⚠️  加速比有波動（變異係數 >= 20%）")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Good Taste Benchmark - 測試 Cython 優化的正確性和性能'
    )
    parser.add_argument(
        '-n', '--size',
        type=int,
        default=1000,
        help='單次測試的電極尺寸（原子數，預設 1000）'
    )
    parser.add_argument(
        '--scaling',
        action='store_true',
        help='執行多尺寸擴展性測試'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='快速測試（較少迭代）'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🔥 GOOD TASTE BENCHMARK")
    print("=" * 70)
    print("測試 Cython 優化版本的正確性和性能")
    print("=" * 70)
    
    if not CYTHON_AVAILABLE:
        print("\n❌ Cython 模組未載入，無法執行測試")
        sys.exit(1)
    
    if args.scaling:
        # 擴展性測試
        if args.quick:
            sizes = [100, 500, 1000, 2000]
        else:
            sizes = [100, 300, 500, 1000, 2000, 5000]
        test_scaling(sizes)
    else:
        # 單次測試
        print(f"\n測試尺寸: {args.size} 原子/電極")
        
        data = create_test_system(args.size)
        
        # 檢查正確性
        if not check_correctness(data):
            print("\n⚠️  警告：數值正確性檢查失敗！")
            sys.exit(1)
        
        # 執行 benchmark
        if args.quick:
            warmup, iterations = 5, 50
        else:
            warmup, iterations = 20, 200
        
        benchmark_single_size(args.size, warmup=warmup, iterations=iterations)
    
    print("\n" + "=" * 70)
    print("✅ BENCHMARK 完成")
    print("=" * 70)


if __name__ == '__main__':
    main()
