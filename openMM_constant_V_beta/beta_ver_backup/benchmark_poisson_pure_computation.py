#!/usr/bin/env python
"""
Pure Poisson Computation Benchmark
==================================
只測試 Poisson solver 的核心數值計算部分，不包含：
- atom.charge 賦值
- setParticleParameters() 呼叫
- updateParametersInContext() 呼叫

這些都是「應用計算結果」，不是「Poisson 計算」本身！
"""

import numpy as np
import time
import sys
import argparse
import os

# Add lib directory to path for Cython module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

# Try to import Cython module
try:
    from electrode_charges_cython import compute_electrode_charges_cython as compute_cython_raw
    CYTHON_AVAILABLE = True
except ImportError as e:
    CYTHON_AVAILABLE = False
    print(f"⚠️  Cython module not available: {e}")

# Constants (from original code)
conversion_KjmolNm_Au = 0.0143932  # Conversion factor

class MockAtom:
    """模擬 atom 物件，但只用於生成測試數據"""
    def __init__(self, atom_index, charge, position):
        self.atom_index = atom_index
        self.charge = charge
        self.position = position

def create_test_data(n_cathode, n_anode):
    """創建測試數據"""
    total_atoms = n_cathode + n_anode + 1000  # 包含電解質
    
    # Electrode atoms
    cathode_atoms = []
    for i in range(n_cathode):
        atom = MockAtom(
            atom_index=i,
            charge=0.001 * (1 + 0.1 * np.random.randn()),
            position=[np.random.uniform(-5, 5), np.random.uniform(-5, 5), 0.0]
        )
        cathode_atoms.append(atom)
    
    anode_atoms = []
    for i in range(n_anode):
        atom = MockAtom(
            atom_index=n_cathode + i,
            charge=-0.001 * (1 + 0.1 * np.random.randn()),
            position=[np.random.uniform(-5, 5), np.random.uniform(-5, 5), 10.0]
        )
        anode_atoms.append(atom)
    
    # Simulation parameters
    area_atom_cathode = 0.05
    area_atom_anode = 0.05
    V_cathode = 2.0
    V_anode = 2.0
    Lgap = 10.0
    small_threshold = 1e-6
    
    # Mock forces array (complete system)
    forces_z = np.random.randn(total_atoms) * 100.0 + 100.0
    
    return {
        'cathode_atoms': cathode_atoms,
        'anode_atoms': anode_atoms,
        'forces_z': forces_z,
        'area_atom_cathode': area_atom_cathode,
        'area_atom_anode': area_atom_anode,
        'V_cathode': V_cathode,
        'V_anode': V_anode,
        'Lgap': Lgap,
        'small_threshold': small_threshold
    }

#========================================================================
# PURE COMPUTATION VERSIONS - 只計算，不更新任何物件
#========================================================================

def compute_original_pure(data):
    """
    原始版本的純計算部分（包含 Ez 提取）
    包含：
    1. Ez = forces[index][2]._value / q_i_old  (從 force 提取電場)
    2. q_i = formula  (Poisson 公式計算)
    
    不包含：
    - atom.charge = q_i
    - setParticleParameters()
    """
    cathode_atoms = data['cathode_atoms']
    anode_atoms = data['anode_atoms']
    forces_z = data['forces_z']  # Mock forces
    area_atom_cathode = data['area_atom_cathode']
    area_atom_anode = data['area_atom_anode']
    V_cathode = data['V_cathode']
    V_anode = data['V_anode']
    Lgap = data['Lgap']
    small_threshold = data['small_threshold']
    
    charges_cathode = np.zeros(len(cathode_atoms))
    charges_anode = np.zeros(len(anode_atoms))
    
    # Cathode charges (原始 loop-based 計算)
    for i in range(len(cathode_atoms)):
        atom = cathode_atoms[i]
        index = atom.atom_index
        q_i_old = atom.charge
        
        # Extract Ez from forces (這也是計算的一部分！)
        Ez_external = (forces_z[index] / q_i_old) if abs(q_i_old) > (0.9 * small_threshold) else 0.0
        
        # Poisson formula
        q_i = 2.0 / (4.0 * np.pi) * area_atom_cathode * \
              (V_cathode / Lgap + Ez_external) * conversion_KjmolNm_Au
        
        if abs(q_i) < small_threshold:
            q_i = small_threshold
        charges_cathode[i] = q_i
    
    # Anode charges (原始 loop-based 計算)
    for i in range(len(anode_atoms)):
        atom = anode_atoms[i]
        index = atom.atom_index
        q_i_old = atom.charge
        
        # Extract Ez from forces
        Ez_external = (forces_z[index] / q_i_old) if abs(q_i_old) > (0.9 * small_threshold) else 0.0
        
        # Poisson formula
        q_i = -2.0 / (4.0 * np.pi) * area_atom_anode * \
              (V_anode / Lgap + Ez_external) * conversion_KjmolNm_Au
        
        if abs(q_i) < small_threshold:
            q_i = -1.0 * small_threshold
        charges_anode[i] = q_i
    
    return charges_cathode, charges_anode

def compute_optimized_pure(data):
    """
    NumPy 優化版本的純計算部分（包含 Ez 提取）
    包含：
    1. Ez = forces[index] / q_i_old  (vectorized)
    2. q_i = formula  (vectorized)
    
    不包含：
    - atom.charge = q_i
    - setParticleParameters()
    """
    cathode_atoms = data['cathode_atoms']
    anode_atoms = data['anode_atoms']
    forces_z = data['forces_z']
    area_atom_cathode = data['area_atom_cathode']
    area_atom_anode = data['area_atom_anode']
    V_cathode = data['V_cathode']
    V_anode = data['V_anode']
    Lgap = data['Lgap']
    small_threshold = data['small_threshold']
    
    # Extract indices and charges
    cathode_indices = np.array([atom.atom_index for atom in cathode_atoms], dtype=np.int64)
    anode_indices = np.array([atom.atom_index for atom in anode_atoms], dtype=np.int64)
    
    cathode_charges_old = np.array([atom.charge for atom in cathode_atoms])
    anode_charges_old = np.array([atom.charge for atom in anode_atoms])
    
    # Extract Ez from forces (vectorized)
    Ez_cathode = np.where(
        np.abs(cathode_charges_old) > (0.9 * small_threshold),
        forces_z[cathode_indices] / cathode_charges_old,
        0.0
    )
    
    Ez_anode = np.where(
        np.abs(anode_charges_old) > (0.9 * small_threshold),
        forces_z[anode_indices] / anode_charges_old,
        0.0
    )
    
    # Cathode charges (vectorized)
    charges_cathode = 2.0 / (4.0 * np.pi) * area_atom_cathode * \
                      (V_cathode / Lgap + Ez_cathode) * conversion_KjmolNm_Au
    charges_cathode[np.abs(charges_cathode) < small_threshold] = small_threshold
    
    # Anode charges (vectorized)
    charges_anode = -2.0 / (4.0 * np.pi) * area_atom_anode * \
                    (V_anode / Lgap + Ez_anode) * conversion_KjmolNm_Au
    charges_anode[np.abs(charges_anode) < small_threshold] = -small_threshold
    
    return charges_cathode, charges_anode

def compute_cython_pure(data):
    """
    Cython 版本的純計算部分（包含 Ez 提取）
    使用 compute_electrode_charges_pure 函數
    """
    if not CYTHON_AVAILABLE:
        raise RuntimeError("Cython module not available")
    
    # Import the correct function
    from electrode_charges_cython import compute_electrode_charges_pure
    
    cathode_atoms = data['cathode_atoms']
    anode_atoms = data['anode_atoms']
    forces_z = data['forces_z']
    area_atom_cathode = data['area_atom_cathode']
    area_atom_anode = data['area_atom_anode']
    V_cathode = data['V_cathode']
    V_anode = data['V_anode']
    Lgap = data['Lgap']
    small_threshold = data['small_threshold']
    
    # Extract indices and old charges
    cathode_indices = np.array([atom.atom_index for atom in cathode_atoms], dtype=np.int64)
    anode_indices = np.array([atom.atom_index for atom in anode_atoms], dtype=np.int64)
    
    cathode_charges_old = np.array([atom.charge for atom in cathode_atoms])
    anode_charges_old = np.array([atom.charge for atom in anode_atoms])
    
    # Extract forces for electrode atoms
    forces_z_cathode = forces_z[cathode_indices]
    forces_z_anode = forces_z[anode_indices]
    
    # Call Cython function for cathode
    charges_cathode = compute_electrode_charges_pure(
        forces_z_cathode,
        cathode_charges_old,
        area_atom_cathode,
        V_cathode,
        Lgap,
        1.0,  # sign = +1 for cathode
        small_threshold,
        conversion_KjmolNm_Au
    )
    
    # Call Cython function for anode
    charges_anode = compute_electrode_charges_pure(
        forces_z_anode,
        anode_charges_old,
        area_atom_anode,
        V_anode,
        Lgap,
        -1.0,  # sign = -1 for anode
        small_threshold,
        conversion_KjmolNm_Au
    )
    
    return charges_cathode, charges_anode

#========================================================================
# Benchmarking Functions
#========================================================================

def benchmark_version(compute_func, data, name, warmup=100, iterations=1000):
    """Benchmark a specific version"""
    print(f"\n{'='*70}")
    print(f"Benchmarking: {name}")
    print(f"{'='*70}")
    
    # Warmup
    print(f"Warming up ({warmup} iterations)...", end=' ', flush=True)
    for _ in range(warmup):
        compute_func(data)
    print("Done")
    
    # Actual benchmark
    print(f"Running benchmark ({iterations} iterations)...", end=' ', flush=True)
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        compute_func(data)
        end = time.perf_counter()
        times.append((end - start) * 1e6)  # Convert to microseconds
    print("Done")
    
    times = np.array(times)
    print("\nResults:")
    print(f"  Mean:   {np.mean(times):.2f} ± {np.std(times):.2f} μs")
    print(f"  Median: {np.median(times):.2f} μs")
    print(f"  Min:    {np.min(times):.2f} μs")
    print(f"  Max:    {np.max(times):.2f} μs")
    
    return np.mean(times)

def check_consistency(data):
    """Check that all versions produce the same results"""
    print("\n" + "="*70)
    print("NUMERICAL CONSISTENCY CHECK")
    print("="*70)
    
    # Compute with all versions
    charges_orig_c, charges_orig_a = compute_original_pure(data)
    charges_opt_c, charges_opt_a = compute_optimized_pure(data)
    
    # Check Original vs Optimized
    diff_c = np.max(np.abs(charges_orig_c - charges_opt_c))
    diff_a = np.max(np.abs(charges_orig_a - charges_opt_a))
    max_diff_opt = max(diff_c, diff_a)
    
    print(f"Original vs Optimized:")
    print(f"  Max difference (cathode): {diff_c:.2e}")
    print(f"  Max difference (anode):   {diff_a:.2e}")
    
    if CYTHON_AVAILABLE:
        charges_cyt_c, charges_cyt_a = compute_cython_pure(data)
        diff_c = np.max(np.abs(charges_orig_c - charges_cyt_c))
        diff_a = np.max(np.abs(charges_orig_a - charges_cyt_a))
        max_diff_cyt = max(diff_c, diff_a)
        
        print(f"\nOriginal vs Cython:")
        print(f"  Max difference (cathode): {diff_c:.2e}")
        print(f"  Max difference (anode):   {diff_a:.2e}")
        
        max_diff = max(max_diff_opt, max_diff_cyt)
    else:
        max_diff = max_diff_opt
    
    # Pass/Fail
    tolerance = 1e-10
    if max_diff < tolerance:
        print(f"\n✓ PASS - All versions produce identical results ({max_diff:.2e} < {tolerance:.2e})")
        return True
    else:
        print(f"\n✗ FAIL - Differences exceed tolerance ({max_diff:.2e} >= {tolerance:.2e})")
        return False

def test_scaling(sizes=[100, 500, 1000, 2000, 5000]):
    """Test how performance scales with electrode size"""
    print("\n" + "="*70)
    print("SCALING TEST: Performance vs Electrode Size")
    print("="*70)
    
    results = []
    
    for n in sizes:
        print(f"\n--- Testing with {n} atoms per electrode ---\n")
        
        data = create_test_data(n, n)
        
        # Adjust iterations based on size
        if n <= 500:
            warmup, iterations = 100, 1000
        elif n <= 2000:
            warmup, iterations = 20, 200
        else:
            warmup, iterations = 10, 100
        
        time_orig = benchmark_version(compute_original_pure, data, "Original (pure computation)", 
                                     warmup=warmup, iterations=iterations)
        time_opt = benchmark_version(compute_optimized_pure, data, "Optimized (pure computation)", 
                                    warmup=warmup, iterations=iterations)
        
        if CYTHON_AVAILABLE:
            time_cyt = benchmark_version(compute_cython_pure, data, "Cython (pure computation)", 
                                        warmup=warmup, iterations=iterations)
        else:
            time_cyt = None
        
        results.append({
            'size': n,
            'time_orig': time_orig,
            'time_opt': time_opt,
            'time_cyt': time_cyt
        })
    
    # Print summary
    print("\n" + "="*70)
    print("SCALING SUMMARY")
    print("="*70)
    print()
    
    header = f"{'Size':<10} {'Original':<15} {'Optimized':<15}"
    if CYTHON_AVAILABLE:
        header += f" {'Cython':<15} {'Speedup (Opt)':<15} {'Speedup (Cyt)':<15}"
    else:
        header += f" {'Speedup (Opt)':<15}"
    print(header)
    print("-" * len(header))
    
    for r in results:
        speedup_opt = r['time_orig'] / r['time_opt']
        line = f"{r['size']:<10} {r['time_orig']:>10.2f} μs   {r['time_opt']:>10.2f} μs   "
        
        if CYTHON_AVAILABLE and r['time_cyt'] is not None:
            speedup_cyt = r['time_orig'] / r['time_cyt']
            line += f"{r['time_cyt']:>10.2f} μs   {speedup_opt:>10.2f}x      {speedup_cyt:>10.2f}x"
        else:
            line += f"{speedup_opt:>10.2f}x"
        
        print(line)

def main():
    parser = argparse.ArgumentParser(description='Benchmark pure Poisson computation')
    parser.add_argument('-n', '--num-atoms', type=int, default=1000,
                       help='Number of atoms per electrode (default: 1000)')
    parser.add_argument('--scaling', action='store_true',
                       help='Run scaling test with multiple sizes')
    
    args = parser.parse_args()
    
    print("="*70)
    print("PURE POISSON COMPUTATION BENCHMARK")
    print("="*70)
    print("測試範圍：只有數值計算（不含 atom.charge 賦值和 setParticleParameters）")
    print("="*70)
    
    if CYTHON_AVAILABLE:
        print("✓ Cython module available")
    else:
        print("⚠️  Cython module NOT available - will only test Original vs Optimized")
    
    if args.scaling:
        test_scaling()
    else:
        print(f"\nTest data: {args.num_atoms} cathode + {args.num_atoms} anode atoms\n")
        
        data = create_test_data(args.num_atoms, args.num_atoms)
        
        # Check consistency first
        if not check_consistency(data):
            print("\n⚠️  Warning: Versions produce different results!")
            sys.exit(1)
        
        # Benchmark
        time_orig = benchmark_version(compute_original_pure, data, "Original (pure computation)")
        time_opt = benchmark_version(compute_optimized_pure, data, "Optimized (pure computation)")
        
        if CYTHON_AVAILABLE:
            time_cyt = benchmark_version(compute_cython_pure, data, "Cython (pure computation)")
        else:
            time_cyt = None
        
        # Summary
        print("\n" + "="*70)
        print("PERFORMANCE COMPARISON (PURE COMPUTATION ONLY)")
        print("="*70)
        print(f"  Original:  {time_orig:>8.2f} μs  (1.00x)")
        print(f"  Optimized: {time_opt:>8.2f} μs  ({time_orig/time_opt:.2f}x speedup)")
        if CYTHON_AVAILABLE and time_cyt is not None:
            print(f"  Cython:    {time_cyt:>8.2f} μs  ({time_orig/time_cyt:.2f}x speedup)")
    
    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)

if __name__ == '__main__':
    main()
