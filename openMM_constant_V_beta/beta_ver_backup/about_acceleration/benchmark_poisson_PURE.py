#!/usr/bin/env python3
"""
=============================================================================
PURE POISSON SOLVER BENCHMARK
=============================================================================

這個 benchmark 只測試 **純 Poisson 數值計算**，不包含：
❌ atom.charge = q_i (Python 物件屬性存取)
❌ setParticleParameters() (OpenMM C++ API 呼叫)  
❌ updateParametersInContext() (GPU/CPU 同步)

只測試：
✅ Ez_external 計算
✅ q_i 數值計算 (Poisson equation)
✅ 閾值檢查

這才是真正的「Poisson 算法」效能！
=============================================================================
"""

import numpy as np
import time
import sys
import argparse

# Try import Cython
try:
    import electrode_charges_cython as ec_cython
    CYTHON_AVAILABLE = True
    print("✓ Cython module available")
except ImportError:
    CYTHON_AVAILABLE = False
    print("✗ Cython module not available")
    print("  Run: python setup_cython.py build_ext --inplace")

# Constants (from original code)
conversion_nmBohr = 18.8973
conversion_KjmolNm_Au = conversion_nmBohr / 2625.5
SMALL_THRESHOLD = 1e-6


def compute_original_pure(forces_z, charges_old, area_atom, voltage, Lgap, 
                          sign, small_threshold):
    """
    原始版本：純 Poisson 計算 (loop-based)
    
    只包含：
    1. Ez_external 計算 (Ez = force_z / q_old)
    2. Poisson equation (q_new = formula)
    3. 閾值檢查
    
    不包含：atom.charge 更新、setParticleParameters 呼叫
    """
    N = len(forces_z)
    charges_new = np.zeros(N, dtype=np.float64)
    
    for i in range(N):
        q_i_old = charges_old[i]
        
        # Ez calculation
        if abs(q_i_old) > (0.9 * small_threshold):
            Ez_external = forces_z[i] / q_i_old
        else:
            Ez_external = 0.0
        
        # Poisson equation
        q_i = sign * 2.0 / (4.0 * np.pi) * area_atom * (
            voltage / Lgap + Ez_external
        ) * conversion_KjmolNm_Au
        
        # Threshold check
        if abs(q_i) < small_threshold:
            q_i = sign * small_threshold
        
        charges_new[i] = q_i
    
    return charges_new


def compute_optimized_pure(forces_z, charges_old, area_atom, voltage, Lgap, 
                           sign, small_threshold):
    """
    優化版本：純 Poisson 計算 (NumPy vectorized)
    """
    # Ez calculation (vectorized with safe division)
    Ez_external = np.where(
        np.abs(charges_old) > (0.9 * small_threshold),
        forces_z / charges_old,
        0.0
    )
    
    # Poisson equation (vectorized)
    charges_new = sign * 2.0 / (4.0 * np.pi) * area_atom * (
        voltage / Lgap + Ez_external
    ) * conversion_KjmolNm_Au
    
    # Threshold check (vectorized)
    charges_new = np.where(
        np.abs(charges_new) < small_threshold,
        sign * small_threshold,
        charges_new
    )
    
    return charges_new


def compute_cython_pure(forces_z, charges_old, area_atom, voltage, Lgap, 
                        sign, small_threshold):
    """
    Cython 版本：純 Poisson 計算 (C-compiled)
    """
    if not CYTHON_AVAILABLE:
        raise RuntimeError("Cython module not available")
    
    return ec_cython.compute_electrode_charges_pure(
        forces_z, charges_old, area_atom, voltage, Lgap, 
        sign, small_threshold, conversion_KjmolNm_Au
    )


def create_test_data(n_cathode, n_anode):
    """創建測試數據"""
    print(f"\nCreating test data:")
    print(f"  Cathode atoms: {n_cathode}")
    print(f"  Anode atoms: {n_anode}")
    
    # Cathode data
    forces_z_cathode = np.random.uniform(-100, 100, n_cathode)
    charges_cathode = np.random.uniform(0.0001, 0.001, n_cathode)
    area_cathode = 0.1  # nm^2
    voltage_cathode = 2.0  # V (arbitrary for test)
    
    # Anode data
    forces_z_anode = np.random.uniform(-100, 100, n_anode)
    charges_anode = np.random.uniform(-0.001, -0.0001, n_anode)
    area_anode = 0.1  # nm^2
    voltage_anode = 2.0  # V
    
    Lgap = 5.0  # nm
    
    return {
        'cathode': (forces_z_cathode, charges_cathode, area_cathode, 
                    voltage_cathode, Lgap, 1.0),
        'anode': (forces_z_anode, charges_anode, area_anode, 
                  voltage_anode, Lgap, -1.0)
    }


def benchmark_version(name, compute_func, data, n_iterations, n_warmup):
    """Benchmark 單一版本"""
    print(f"\n{'='*70}")
    print(f"Benchmarking: {name}")
    print(f"{'='*70}")
    
    cathode_data = data['cathode']
    anode_data = data['anode']
    
    # Warmup
    print(f"Warming up ({n_warmup} iterations)...", end=' ', flush=True)
    for _ in range(n_warmup):
        compute_func(*cathode_data, SMALL_THRESHOLD)
        compute_func(*anode_data, SMALL_THRESHOLD)
    print("Done")
    
    # Actual benchmark
    print(f"Running benchmark ({n_iterations} iterations)...", end=' ', flush=True)
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        result_cathode = compute_func(*cathode_data, SMALL_THRESHOLD)
        result_anode = compute_func(*anode_data, SMALL_THRESHOLD)
        end = time.perf_counter()
        times.append((end - start) * 1e6)  # Convert to microseconds
    print("Done")
    
    times = np.array(times)
    mean = np.mean(times)
    std = np.std(times)
    median = np.median(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    print(f"\nResults:")
    print(f"  Mean:   {mean:.2f} ± {std:.2f} μs")
    print(f"  Median: {median:.2f} μs")
    print(f"  Min:    {min_time:.2f} μs")
    print(f"  Max:    {max_time:.2f} μs")
    
    return {
        'name': name,
        'mean': mean,
        'std': std,
        'median': median,
        'min': min_time,
        'max': max_time,
        'result_cathode': result_cathode,
        'result_anode': result_anode
    }


def check_consistency(results, tolerance=1e-10):
    """檢查數值一致性"""
    print(f"\n{'='*70}")
    print("NUMERICAL CONSISTENCY CHECK")
    print(f"{'='*70}")
    
    baseline = results[0]
    all_passed = True
    
    for result in results[1:]:
        diff_cathode = np.max(np.abs(
            result['result_cathode'] - baseline['result_cathode']
        ))
        diff_anode = np.max(np.abs(
            result['result_anode'] - baseline['result_anode']
        ))
        max_diff = max(diff_cathode, diff_anode)
        
        status = "✓ PASS" if max_diff < tolerance else "✗ FAIL"
        print(f"{baseline['name']} vs {result['name']}: {max_diff:.2e} {status}")
        
        if max_diff >= tolerance:
            all_passed = False
    
    if all_passed:
        print(f"\n✓ All versions produce identical results (< {tolerance})")
    else:
        print(f"\n✗ Numerical inconsistency detected!")
    
    return all_passed


def print_performance_comparison(results):
    """打印效能比較"""
    print(f"\n{'='*70}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*70}")
    
    baseline = results[0]
    baseline_time = baseline['mean']
    
    print(f"\n{'Version':<15} {'Time (μs)':<15} {'Speedup':<10}")
    print("-" * 40)
    
    for result in results:
        speedup = baseline_time / result['mean']
        print(f"{result['name']:<15} {result['mean']:>8.2f}        {speedup:>6.2f}x")


def test_scaling(sizes):
    """測試不同規模的效能"""
    print(f"\n{'='*70}")
    print("SCALING TEST: Performance vs Electrode Size")
    print(f"{'='*70}")
    
    results_by_size = {}
    
    for size in sizes:
        print(f"\n--- Testing with {size} atoms per electrode ---")
        data = create_test_data(size, size)
        
        # Determine iterations based on size
        if size <= 100:
            n_warmup = 100
            n_iterations = 1000
        elif size <= 1000:
            n_warmup = 10
            n_iterations = 100
        else:
            n_warmup = 10
            n_iterations = 100
        
        results = []
        
        # Original
        result = benchmark_version(
            "Original", compute_original_pure, data, n_iterations, n_warmup
        )
        results.append(result)
        
        # Optimized
        result = benchmark_version(
            "Optimized", compute_optimized_pure, data, n_iterations, n_warmup
        )
        results.append(result)
        
        # Cython (if available)
        if CYTHON_AVAILABLE:
            result = benchmark_version(
                "Cython", compute_cython_pure, data, n_iterations, n_warmup
            )
            results.append(result)
        
        results_by_size[size] = results
    
    # Print scaling summary
    print(f"\n{'='*70}")
    print("SCALING SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Size':<10} {'Original':<15} {'Optimized':<15}", end='')
    if CYTHON_AVAILABLE:
        print(f"{'Cython':<15} {'Speedup (Opt)':<15} {'Speedup (Cyt)':<15}")
    else:
        print(f"{'Speedup (Opt)':<15}")
    print("-" * 85)
    
    for size, results in results_by_size.items():
        orig_time = results[0]['mean']
        opt_time = results[1]['mean']
        opt_speedup = orig_time / opt_time
        
        print(f"{size:<10} {orig_time:>8.2f} μs    {opt_time:>8.2f} μs    ", end='')
        
        if CYTHON_AVAILABLE and len(results) > 2:
            cyt_time = results[2]['mean']
            cyt_speedup = orig_time / cyt_time
            print(f"{cyt_time:>8.2f} μs    {opt_speedup:>6.2f}x         {cyt_speedup:>6.2f}x")
        else:
            print(f"{opt_speedup:>6.2f}x")


def main():
    parser = argparse.ArgumentParser(description='Pure Poisson Solver Benchmark')
    parser.add_argument('-n', '--num-atoms', type=int, default=1000,
                      help='Number of atoms per electrode (default: 1000)')
    parser.add_argument('--scaling', action='store_true',
                      help='Run scaling test with multiple sizes')
    args = parser.parse_args()
    
    print("="*70)
    print("PURE POISSON SOLVER BENCHMARK")
    print("="*70)
    print("✓ Only numerical computation (Ez, Poisson equation, threshold)")
    print("✓ NO atom.charge updates")
    print("✓ NO setParticleParameters() calls")
    print("✓ NO updateParametersInContext() calls")
    print("="*70)
    
    if not CYTHON_AVAILABLE:
        print("\n⚠️  Cython module not available")
        print("    Only testing Original and Optimized versions")
    
    if args.scaling:
        test_scaling([100, 500, 1000, 2000, 5000])
    else:
        # Single size test
        data = create_test_data(args.num_atoms, args.num_atoms)
        
        # Determine iterations
        if args.num_atoms <= 100:
            n_warmup = 100
            n_iterations = 1000
        elif args.num_atoms <= 1000:
            n_warmup = 20
            n_iterations = 200
        else:
            n_warmup = 10
            n_iterations = 100
        
        results = []
        
        # Original
        result = benchmark_version(
            "Original", compute_original_pure, data, n_iterations, n_warmup
        )
        results.append(result)
        
        # Optimized
        result = benchmark_version(
            "Optimized", compute_optimized_pure, data, n_iterations, n_warmup
        )
        results.append(result)
        
        # Cython
        if CYTHON_AVAILABLE:
            result = benchmark_version(
                "Cython", compute_cython_pure, data, n_iterations, n_warmup
            )
            results.append(result)
        
        # Consistency check
        check_consistency(results)
        
        # Performance comparison
        print_performance_comparison(results)
    
    print(f"\n{'='*70}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
