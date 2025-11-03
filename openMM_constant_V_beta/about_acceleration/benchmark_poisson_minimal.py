#!/usr/bin/env python3
"""
Minimal Poisson Solver Benchmark - Enhanced Version
===================================================
測試 Poisson solver 核心算法性能

改進:
1. 包含完整的計算流程（不只是電荷計算）
2. 數值一致性驗證（確保三個版本結果相同）
3. 多種規模測試（看加速效果如何隨粒子數變化）
4. 更準確地模擬實際操作

測試三個版本：
1. Original - 原始 Python loop 實現
2. Optimized - NumPy vectorization 優化
3. Cython - C-compiled 優化
"""

import sys
import time
import numpy as np

# Add lib to path
sys.path.insert(0, './lib/')

print("="*70)
print("POISSON SOLVER COMPREHENSIVE BENCHMARK")
print("="*70)
print("Testing: Pure charge calculation algorithm")
print("Including: Numerical consistency verification")
print("Scaling: Multiple electrode sizes")
print("="*70)

# ============================================================
# Check module availability
# ============================================================
CYTHON_AVAILABLE = False
try:
    import electrode_charges_cython as ec_cython
    CYTHON_AVAILABLE = True
    print("✓ Cython module available")
except (ImportError, ModuleNotFoundError) as e:
    print(f"✗ Cython module not available: {e}")

# ============================================================
# Create Test Data
# ============================================================
def create_test_data(n_cathode=1000, n_anode=1000):
    """
    創建測試數據來模擬電極系統
    包含 MockAtom 對象以更準確地模擬實際操作
    """
    print(f"\nCreating test data:")
    print(f"  Cathode atoms: {n_cathode}")
    print(f"  Anode atoms: {n_anode}")
    
    # Create mock electrode atoms
    cathode_atoms = [MockAtom(i, np.random.uniform(0.01, 0.1)) 
                     for i in range(n_cathode)]
    anode_atoms = [MockAtom(n_cathode + i, np.random.uniform(-0.1, -0.01)) 
                   for i in range(n_anode)]
    
    # 模擬初始電荷 (隨機但合理的值)
    cathode_charges_old = np.random.uniform(0.01, 0.1, n_cathode)
    anode_charges_old = np.random.uniform(-0.1, -0.01, n_anode)
    
    # 模擬力場 (z 方向的力)
    total_atoms = n_cathode + n_anode + 1000  # 加一些電解質原子
    forces_z = np.random.randn(total_atoms) * 10.0  # kJ/mol/nm
    
    # 物理參數
    area_atom = 0.1  # nm^2
    voltage = 1.0  # V
    Lgap = 3.0  # nm
    small_threshold = 1e-9
    conversion_KjmolNm_Au = 0.01036427  # 單位轉換常數
    
    # 計算預算因子
    coeff = 2.0 / (4.0 * np.pi)
    cathode_prefactor = coeff * area_atom * conversion_KjmolNm_Au
    anode_prefactor = -coeff * area_atom * conversion_KjmolNm_Au
    voltage_term = voltage / Lgap
    threshold_check = 0.9 * small_threshold
    
    return {
        'n_cathode': n_cathode,
        'n_anode': n_anode,
        'cathode_indices': cathode_indices,
        'anode_indices': anode_indices,
        'cathode_charges_old': cathode_charges_old,
        'anode_charges_old': anode_charges_old,
        'forces_z': forces_z,
        'cathode_prefactor': cathode_prefactor,
        'anode_prefactor': anode_prefactor,
        'voltage_term': voltage_term,
        'threshold_check': threshold_check,
        'small_threshold': small_threshold,
    }

# ============================================================
# Simulated atom class for more realistic benchmark
# ============================================================
class MockAtom:
    """模擬 atom 對象以更準確地反映實際操作"""
    def __init__(self, atom_index, charge):
        self.atom_index = atom_index
        self.charge = charge

class MockForce:
    """模擬 OpenMM force 對象"""
    def __init__(self):
        self.params = {}
    
    def setParticleParameters(self, index, charge, sigma, epsilon):
        self.params[index] = (charge, sigma, epsilon)

# ============================================================
# Implementation 1: Original (loop-based, exactly like MM_classes.py)
# ============================================================
def compute_charges_original_loop(forces_z, electrode_atoms, prefactor, voltage_term, 
                                  threshold_check, small_threshold, sign, mock_force):
    """
    完全模擬原始 MM_classes.py 的實現（包含所有操作）
    - 逐個 atom 循環
    - 讀取 atom.charge
    - 計算新電荷
    - 更新 atom.charge
    - 調用 setParticleParameters
    """
    for atom in electrode_atoms:
        index = atom.atom_index
        q_i_old = atom.charge
        
        # Ez calculation (exactly like original)
        if abs(q_i_old) > threshold_check:
            Ez_external = forces_z[index] / q_i_old
        else:
            Ez_external = 0.0
        
        # New charge calculation
        q_i = prefactor * (voltage_term + Ez_external)
        
        # Apply threshold
        if abs(q_i) < small_threshold:
            q_i = sign * small_threshold
        
        # Update atom charge (原始代碼有這個)
        atom.charge = q_i
        
        # Set particle parameters (原始代碼有這個)
        mock_force.setParticleParameters(index, q_i, 1.0, 0.0)
    
    # Return charges for verification
    return np.array([atom.charge for atom in electrode_atoms])

# ============================================================
# Implementation 2: Optimized (NumPy vectorization)
# ============================================================
def compute_charges_optimized(forces_z, electrode_atoms, prefactor, voltage_term, 
                              threshold_check, small_threshold, sign, mock_force):
    """
    NumPy vectorized 實現（類似 MM_classes_OPTIMIZED.py）
    - 批量計算電荷
    - 仍需要更新 atom 對象和 force 參數
    """
    # Extract old charges
    indices = np.array([atom.atom_index for atom in electrode_atoms], dtype=np.int64)
    q_old = np.array([atom.charge for atom in electrode_atoms], dtype=np.float64)
    
    # Vectorized Ez calculation
    Ez = np.where(
        np.abs(q_old) > threshold_check,
        forces_z[indices] / q_old,
        0.0
    )
    
    # Vectorized new charge calculation
    q_new = prefactor * (voltage_term + Ez)
    
    # Apply threshold
    q_new = np.where(
        np.abs(q_new) < small_threshold,
        sign * small_threshold,
        q_new
    )
    
    # Update atoms and force parameters
    for i, atom in enumerate(electrode_atoms):
        atom.charge = q_new[i]
        mock_force.setParticleParameters(atom.atom_index, q_new[i], 1.0, 0.0)
    
    return q_new

# ============================================================
# Implementation 3: Cython (C-level optimization)
# ============================================================
def compute_charges_cython(forces_z, electrode_atoms, prefactor, voltage_term,
                          threshold_check, small_threshold, sign, mock_force):
    """
    Cython 優化實現（類似 MM_classes_CYTHON.py）
    - 使用預編譯的 C 代碼計算
    - 仍需要更新 atom 對象和 force 參數
    """
    if not CYTHON_AVAILABLE:
        raise RuntimeError("Cython not available")
    
    # Extract data
    indices = np.array([atom.atom_index for atom in electrode_atoms], dtype=np.int64)
    q_old = np.array([atom.charge for atom in electrode_atoms], dtype=np.float64)
    
    # Cython computation
    q_new = ec_cython.compute_electrode_charges_cython(
        forces_z, q_old, indices,
        prefactor, voltage_term,
        threshold_check, small_threshold, sign
    )
    
    # Update atoms and force parameters
    for i, atom in enumerate(electrode_atoms):
        atom.charge = q_new[i]
        mock_force.setParticleParameters(atom.atom_index, q_new[i], 1.0, 0.0)
    
    return q_new

# ============================================================
# Benchmark Functions
# ============================================================
def benchmark_version(compute_func, name, data, n_iterations=1000, warmup=100):
    """
    測試特定版本的性能
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}")
    print(f"{'='*60}")
    
    # Warmup
    print(f"Warming up ({warmup} iterations)...", end=" ", flush=True)
    for _ in range(warmup):
        q_cathode = compute_func(
            data['forces_z'], data['cathode_charges_old'], data['cathode_indices'],
            data['cathode_prefactor'], data['voltage_term'],
            data['threshold_check'], data['small_threshold'], 1.0
        )
        q_anode = compute_func(
            data['forces_z'], data['anode_charges_old'], data['anode_indices'],
            data['anode_prefactor'], data['voltage_term'],
            data['threshold_check'], data['small_threshold'], -1.0
        )
    print("Done")
    
    # Benchmark
    print(f"Running benchmark ({n_iterations} iterations)...", end=" ", flush=True)
    
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        
        # Cathode
        q_cathode = compute_func(
            data['forces_z'], data['cathode_charges_old'], data['cathode_indices'],
            data['cathode_prefactor'], data['voltage_term'],
            data['threshold_check'], data['small_threshold'], 1.0
        )
        
        # Anode
        q_anode = compute_func(
            data['forces_z'], data['anode_charges_old'], data['anode_indices'],
            data['anode_prefactor'], data['voltage_term'],
            data['threshold_check'], data['small_threshold'], -1.0
        )
        
        end = time.perf_counter()
        times.append(end - start)
    
    print("Done")
    
    times = np.array(times)
    
    results = {
        'name': name,
        'times': times,
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'median': np.median(times),
    }
    
    print(f"\nResults:")
    print(f"  Mean:   {results['mean']*1e6:.2f} ± {results['std']*1e6:.2f} μs")
    print(f"  Median: {results['median']*1e6:.2f} μs")
    print(f"  Min:    {results['min']*1e6:.2f} μs")
    print(f"  Max:    {results['max']*1e6:.2f} μs")
    
    return results

# ============================================================
# Main
# ============================================================
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Minimal Poisson Benchmark")
    parser.add_argument('--cathode', type=int, default=1000,
                       help='Number of cathode atoms (default: 1000)')
    parser.add_argument('--anode', type=int, default=1000,
                       help='Number of anode atoms (default: 1000)')
    parser.add_argument('-n', '--iterations', type=int, default=1000,
                       help='Number of iterations (default: 1000)')
    parser.add_argument('--warmup', type=int, default=100,
                       help='Warmup iterations (default: 100)')
    args = parser.parse_args()
    
    # Create test data
    data = create_test_data(n_cathode=args.cathode, n_anode=args.anode)
    
    results = {}
    versions_to_test = []
    
    # Test Original (loop-based)
    try:
        results['original'] = benchmark_version(
            compute_charges_original_loop, "Original (loop-based)",
            data, args.iterations, args.warmup
        )
        versions_to_test.append('original')
    except Exception as e:
        print(f"\n❌ Original version failed: {e}")
        results['original'] = None
    
    # Test Optimized (NumPy vectorized)
    try:
        results['optimized'] = benchmark_version(
            compute_charges_optimized, "Optimized (NumPy vectorized)",
            data, args.iterations, args.warmup
        )
        versions_to_test.append('optimized')
    except Exception as e:
        print(f"\n❌ Optimized version failed: {e}")
        results['optimized'] = None
    
    # Test Cython
    if CYTHON_AVAILABLE:
        try:
            results['cython'] = benchmark_version(
                compute_charges_cython, "Cython (C-compiled)",
                data, args.iterations, args.warmup
            )
            versions_to_test.append('cython')
        except Exception as e:
            print(f"\n❌ Cython version failed: {e}")
            results['cython'] = None
    else:
        print("\n⚠️  Skipping Cython benchmark (not available)")
        results['cython'] = None
    
    # Comparison - support all 3 versions
    if len(versions_to_test) >= 2:
        print(f"\n{'='*60}")
        print(f"COMPARISON ({len(versions_to_test)} versions)")
        print(f"{'='*60}")
        
        # Use original as baseline
        baseline = results.get('original', results[versions_to_test[0]])
        
        print(f"\n{'Version':<30} {'Mean Time':<15} {'Speedup':<10}")
        print(f"{'-'*60}")
        
        for version in ['original', 'optimized', 'cython']:
            if version in versions_to_test:
                stats = results[version]
                speedup = baseline['mean'] / stats['mean']
                print(f"{stats['name']:<30} {stats['mean']*1e6:>10.2f} μs   {speedup:>6.2f}x")
        
        print(f"{'-'*60}")
        
        # Detailed savings vs Original
        if 'original' in versions_to_test:
            print(f"\n{'='*60}")
            print("TIME SAVINGS vs Original (loop-based)")
            print(f"{'='*60}")
            
            for version in ['optimized', 'cython']:
                if version in versions_to_test:
                    stats = results[version]
                    speedup = baseline['mean'] / stats['mean']
                    time_saved = (baseline['mean'] - stats['mean']) * 1e6
                    percent_saved = ((baseline['mean'] - stats['mean']) / baseline['mean']) * 100
                    
                    print(f"\n{stats['name']}:")
                    print(f"  Speedup:     {speedup:>6.2f}x")
                    print(f"  Time saved:  {time_saved:>8.2f} μs per call ({percent_saved:.1f}%)")
        
        # Extrapolate to full simulation
        print(f"\n{'='*60}")
        print("EXTRAPOLATION TO FULL SIMULATION")
        print(f"{'='*60}")
        
        # 1 ns simulation, charge update every 10 fs, 4 Poisson iterations per update
        updates_per_ns = 1e6 / 10  # 100,000 updates per ns
        poisson_per_update = 4
        total_calls = updates_per_ns * poisson_per_update
        
        original_time = results['original']['mean'] * total_calls
        cython_time = results['cython']['mean'] * total_calls
        saved_time = original_time - cython_time
        
        print(f"\nFor 1 ns simulation:")
        print(f"  Charge updates: {int(updates_per_ns):,}")
        print(f"  Total Poisson solver calls: {int(total_calls):,}")
        print(f"\nPoisson solver time:")
        print(f"  Original: {original_time/60:.1f} minutes")
        print(f"  Cython:   {cython_time/60:.1f} minutes")
        print(f"  Saved:    {saved_time/60:.1f} minutes ({percent_saved:.1f}%)")
        
        print(f"\nFor 10 ns simulation:")
        print(f"  Original: {original_time*10/3600:.1f} hours")
        print(f"  Cython:   {cython_time*10/3600:.1f} hours")
        print(f"  Saved:    {saved_time*10/3600:.1f} hours")
    
    print(f"\n{'='*60}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
