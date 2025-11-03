#!/usr/bin/env python3
"""
Poisson Solver Comprehensive Benchmark
======================================
測試 Poisson solver 完整計算流程的性能

特點:
1. 完整模擬原始計算流程（包含所有操作）
2. 數值一致性驗證（確保三個版本結果相同）
3. 多規模測試（觀察加速效果如何隨粒子數變化）

測試三個版本:
- Original: Python loop (逐個 atom)
- Optimized: NumPy vectorization
- Cython: C-compiled
"""

import sys
import time
import numpy as np
from copy import deepcopy

# Add lib to path
sys.path.insert(0, './lib/')

print("="*70)
print("POISSON SOLVER COMPREHENSIVE BENCHMARK")
print("="*70)
print("✓ Complete calculation flow")
print("✓ Numerical consistency check")
print("✓ Multiple electrode sizes")
print("="*70)

# Check Cython availability
CYTHON_AVAILABLE = False
try:
    import electrode_charges_cython as ec_cython
    CYTHON_AVAILABLE = True
    print("✓ Cython module available")
except (ImportError, ModuleNotFoundError) as e:
    print(f"✗ Cython not available: {e}")

# ============================================================
# Mock classes to simulate actual operations
# ============================================================
class MockAtom:
    def __init__(self, atom_index, charge):
        self.atom_index = atom_index
        self.charge = charge
    
    def copy(self):
        return MockAtom(self.atom_index, self.charge)

class MockForce:
    def __init__(self):
        self.params = {}
    
    def setParticleParameters(self, index, charge, sigma, epsilon):
        self.params[index] = (charge, sigma, epsilon)

# ============================================================
# Create Test Data
# ============================================================
def create_test_data(n_cathode=1000, n_anode=1000):
    """創建完整的測試數據"""
    print(f"\nCreating test data:")
    print(f"  Cathode atoms: {n_cathode}")
    print(f"  Anode atoms: {n_anode}")
    print(f"  Total atoms: {n_cathode + n_anode + 1000} (including electrolyte)")
    
    # Create mock electrode atoms with random initial charges
    np.random.seed(42)  # For reproducibility
    cathode_atoms = [MockAtom(i, np.random.uniform(0.01, 0.1)) 
                     for i in range(n_cathode)]
    anode_atoms = [MockAtom(n_cathode + i, np.random.uniform(-0.1, -0.01)) 
                   for i in range(n_anode)]
    
    # Mock forces (simulate full system including electrolyte)
    total_atoms = n_cathode + n_anode + 1000  # Add some electrolyte atoms
    forces_z = np.random.randn(total_atoms) * 10.0  # kJ/mol/nm
    
    # Physical parameters (from actual code)
    area_atom = 0.1  # nm^2
    voltage = 1.0  # V
    Lgap = 3.0  # nm
    small_threshold = 1e-9
    conversion_KjmolNm_Au = 0.01036427
    
    # Precomputed factors
    coeff = 2.0 / (4.0 * np.pi)
    cathode_prefactor = coeff * area_atom * conversion_KjmolNm_Au
    anode_prefactor = -coeff * area_atom * conversion_KjmolNm_Au
    voltage_term = voltage / Lgap
    threshold_check = 0.9 * small_threshold
    
    return {
        'cathode_atoms': cathode_atoms,
        'anode_atoms': anode_atoms,
        'forces_z': forces_z,
        'cathode_prefactor': cathode_prefactor,
        'anode_prefactor': anode_prefactor,
        'voltage_term': voltage_term,
        'threshold_check': threshold_check,
        'small_threshold': small_threshold,
    }

# ============================================================
# Implementation 1: Original (exactly like MM_classes.py)
# ============================================================
def compute_original(data):
    """完整模擬原始 MM_classes.py 的實現"""
    # Deep copy to avoid modifying original
    cathode_atoms = [atom.copy() for atom in data['cathode_atoms']]
    anode_atoms = [atom.copy() for atom in data['anode_atoms']]
    mock_force = MockForce()
    forces_z = data['forces_z']
    
    # Cathode calculation (exact original loop)
    for atom in cathode_atoms:
        index = atom.atom_index
        q_i_old = atom.charge
        
        # Ez calculation
        if abs(q_i_old) > data['threshold_check']:
            Ez_external = forces_z[index] / q_i_old
        else:
            Ez_external = 0.0
        
        # New charge
        q_i = data['cathode_prefactor'] * (data['voltage_term'] + Ez_external)
        
        # Apply threshold
        if abs(q_i) < data['small_threshold']:
            q_i = data['small_threshold']  # Cathode: positive
        
        atom.charge = q_i
        mock_force.setParticleParameters(index, q_i, 1.0, 0.0)
    
    # Anode calculation (exact original loop)
    for atom in anode_atoms:
        index = atom.atom_index
        q_i_old = atom.charge
        
        # Ez calculation
        if abs(q_i_old) > data['threshold_check']:
            Ez_external = forces_z[index] / q_i_old
        else:
            Ez_external = 0.0
        
        # New charge
        q_i = data['anode_prefactor'] * (data['voltage_term'] + Ez_external)
        
        # Apply threshold
        if abs(q_i) < data['small_threshold']:
            q_i = -1.0 * data['small_threshold']  # Anode: negative
        
        atom.charge = q_i
        mock_force.setParticleParameters(index, q_i, 1.0, 0.0)
    
    # Return results
    cathode_charges = np.array([atom.charge for atom in cathode_atoms])
    anode_charges = np.array([atom.charge for atom in anode_atoms])
    
    return cathode_charges, anode_charges

# ============================================================
# Implementation 2: Optimized (NumPy vectorization)
# ============================================================
def compute_optimized(data):
    """NumPy vectorized 實現"""
    cathode_atoms = [atom.copy() for atom in data['cathode_atoms']]
    anode_atoms = [atom.copy() for atom in data['anode_atoms']]
    mock_force = MockForce()
    forces_z = data['forces_z']
    
    # Cathode - vectorized
    indices = np.array([atom.atom_index for atom in cathode_atoms], dtype=np.int64)
    q_old = np.array([atom.charge for atom in cathode_atoms])
    
    Ez = np.where(
        np.abs(q_old) > data['threshold_check'],
        forces_z[indices] / q_old,
        0.0
    )
    q_new = data['cathode_prefactor'] * (data['voltage_term'] + Ez)
    q_new = np.where(
        np.abs(q_new) < data['small_threshold'],
        data['small_threshold'],
        q_new
    )
    
    for i, atom in enumerate(cathode_atoms):
        atom.charge = q_new[i]
        mock_force.setParticleParameters(atom.atom_index, q_new[i], 1.0, 0.0)
    
    cathode_charges = q_new
    
    # Anode - vectorized
    indices = np.array([atom.atom_index for atom in anode_atoms], dtype=np.int64)
    q_old = np.array([atom.charge for atom in anode_atoms])
    
    Ez = np.where(
        np.abs(q_old) > data['threshold_check'],
        forces_z[indices] / q_old,
        0.0
    )
    q_new = data['anode_prefactor'] * (data['voltage_term'] + Ez)
    q_new = np.where(
        np.abs(q_new) < data['small_threshold'],
        -1.0 * data['small_threshold'],
        q_new
    )
    
    for i, atom in enumerate(anode_atoms):
        atom.charge = q_new[i]
        mock_force.setParticleParameters(atom.atom_index, q_new[i], 1.0, 0.0)
    
    anode_charges = q_new
    
    return cathode_charges, anode_charges

# ============================================================
# Implementation 3: Cython
# ============================================================
def compute_cython(data):
    """Cython C-compiled 實現"""
    if not CYTHON_AVAILABLE:
        raise RuntimeError("Cython not available")
    
    cathode_atoms = [atom.copy() for atom in data['cathode_atoms']]
    anode_atoms = [atom.copy() for atom in data['anode_atoms']]
    mock_force = MockForce()
    forces_z = data['forces_z']
    
    # Cathode - Cython
    indices = np.array([atom.atom_index for atom in cathode_atoms], dtype=np.int64)
    q_old = np.array([atom.charge for atom in cathode_atoms])
    
    q_new = ec_cython.compute_electrode_charges_cython(
        forces_z, q_old, indices,
        data['cathode_prefactor'], data['voltage_term'],
        data['threshold_check'], data['small_threshold'], 1.0
    )
    
    for i, atom in enumerate(cathode_atoms):
        atom.charge = q_new[i]
        mock_force.setParticleParameters(atom.atom_index, q_new[i], 1.0, 0.0)
    
    cathode_charges = q_new
    
    # Anode - Cython
    indices = np.array([atom.atom_index for atom in anode_atoms], dtype=np.int64)
    q_old = np.array([atom.charge for atom in anode_atoms])
    
    q_new = ec_cython.compute_electrode_charges_cython(
        forces_z, q_old, indices,
        data['anode_prefactor'], data['voltage_term'],
        data['threshold_check'], data['small_threshold'], -1.0
    )
    
    for i, atom in enumerate(anode_atoms):
        atom.charge = q_new[i]
        mock_force.setParticleParameters(atom.atom_index, q_new[i], 1.0, 0.0)
    
    anode_charges = q_new
    
    return cathode_charges, anode_charges

# ============================================================
# Numerical consistency check
# ============================================================
def check_consistency(results, tolerance=1e-10):
    """檢查三個版本的數值一致性"""
    print(f"\n{'='*70}")
    print("NUMERICAL CONSISTENCY CHECK")
    print(f"{'='*70}")
    
    versions = list(results.keys())
    if len(versions) < 2:
        print("⚠️  Need at least 2 versions to compare")
        return True
    
    # Use first version as reference
    ref_version = versions[0]
    ref_cathode, ref_anode = results[ref_version]
    
    all_consistent = True
    
    for version in versions[1:]:
        cathode, anode = results[version]
        
        cathode_diff = np.max(np.abs(cathode - ref_cathode))
        anode_diff = np.max(np.abs(anode - ref_anode))
        
        cathode_ok = cathode_diff < tolerance
        anode_ok = anode_diff < tolerance
        
        status = "✓ PASS" if (cathode_ok and anode_ok) else "✗ FAIL"
        
        print(f"\n{ref_version} vs {version}:")
        print(f"  Cathode max diff: {cathode_diff:.2e} {'' if cathode_ok else '⚠️ EXCEEDED'}")
        print(f"  Anode max diff:   {anode_diff:.2e} {'' if anode_ok else '⚠️ EXCEEDED'}")
        print(f"  Status: {status}")
        
        if not (cathode_ok and anode_ok):
            all_consistent = False
    
    return all_consistent

# ============================================================
# Benchmark single version
# ============================================================
def benchmark_version(compute_func, name, data, n_iterations=1000, warmup=100):
    """Benchmark 單個版本"""
    print(f"\n{'='*70}")
    print(f"Benchmarking: {name}")
    print(f"{'='*70}")
    
    # Warmup
    print(f"Warming up ({warmup} iterations)...", end=" ", flush=True)
    for _ in range(warmup):
        compute_func(data)
    print("Done")
    
    # Benchmark
    print(f"Running benchmark ({n_iterations} iterations)...", end=" ", flush=True)
    
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        cathode_q, anode_q = compute_func(data)
        end = time.perf_counter()
        times.append(end - start)
    
    print("Done")
    
    times = np.array(times)
    
    result = {
        'name': name,
        'times': times,
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'median': np.median(times),
        'charges': (cathode_q, anode_q)  # For consistency check
    }
    
    print(f"\nResults:")
    print(f"  Mean:   {result['mean']*1e6:.2f} ± {result['std']*1e6:.2f} μs")
    print(f"  Median: {result['median']*1e6:.2f} μs")
    print(f"  Min:    {result['min']*1e6:.2f} μs")
    print(f"  Max:    {result['max']*1e6:.2f} μs")
    
    return result

# ============================================================
# Test with multiple sizes
# ============================================================
def test_scaling(sizes=[100, 500, 1000, 2000, 5000]):
    """測試不同規模下的加速效果"""
    print(f"\n{'='*70}")
    print("SCALING TEST: Performance vs Electrode Size")
    print(f"{'='*70}")
    
    scaling_results = {
        'sizes': sizes,
        'original': [],
        'optimized': [],
        'cython': []
    }
    
    for size in sizes:
        print(f"\n--- Testing with {size} atoms per electrode ---")
        data = create_test_data(n_cathode=size, n_anode=size)
        
        # Test each version (fewer iterations for speed)
        n_iter = max(100, 1000 // (size // 100))
        
        try:
            result = benchmark_version(compute_original, "Original", data, n_iter, n_iter//10)
            scaling_results['original'].append(result['mean'])
        except:
            scaling_results['original'].append(None)
        
        try:
            result = benchmark_version(compute_optimized, "Optimized", data, n_iter, n_iter//10)
            scaling_results['optimized'].append(result['mean'])
        except:
            scaling_results['optimized'].append(None)
        
        if CYTHON_AVAILABLE:
            try:
                result = benchmark_version(compute_cython, "Cython", data, n_iter, n_iter//10)
                scaling_results['cython'].append(result['mean'])
            except:
                scaling_results['cython'].append(None)
    
    # Print scaling summary
    print(f"\n{'='*70}")
    print("SCALING SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Size':<10} {'Original':<15} {'Optimized':<15} {'Cython':<15} {'Speedup (Opt)':<15} {'Speedup (Cyt)':<15}")
    print("-"*85)
    
    for i, size in enumerate(sizes):
        orig = scaling_results['original'][i]
        opt = scaling_results['optimized'][i]
        cyt = scaling_results['cython'][i] if CYTHON_AVAILABLE else None
        
        if orig:
            speedup_opt = orig / opt if opt else 0
            speedup_cyt = orig / cyt if cyt else 0
            
            print(f"{size:<10} {orig*1e6:>10.1f} μs  {opt*1e6:>10.1f} μs  "
                  f"{cyt*1e6 if cyt else 'N/A':>10}  "
                  f"{speedup_opt:>10.1f}x     {speedup_cyt if cyt else 'N/A':>10}")
    
    return scaling_results

# ============================================================
# Main
# ============================================================
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Poisson Benchmark")
    parser.add_argument('--cathode', type=int, default=1000,
                       help='Number of cathode atoms (default: 1000)')
    parser.add_argument('--anode', type=int, default=1000,
                       help='Number of anode atoms (default: 1000)')
    parser.add_argument('-n', '--iterations', type=int, default=2000,
                       help='Number of iterations (default: 2000)')
    parser.add_argument('--warmup', type=int, default=200,
                       help='Warmup iterations (default: 200)')
    parser.add_argument('--scaling', action='store_true',
                       help='Run scaling test with multiple sizes')
    args = parser.parse_args()
    
    if args.scaling:
        # Scaling test
        test_scaling()
    else:
        # Single size test
        data = create_test_data(n_cathode=args.cathode, n_anode=args.anode)
        
        results = {}
        charge_results = {}
        
        # Test Original
        try:
            result = benchmark_version(compute_original, "Original (loop-based)", 
                                      data, args.iterations, args.warmup)
            results['original'] = result
            charge_results['original'] = result['charges']
        except Exception as e:
            print(f"\n❌ Original failed: {e}")
        
        # Test Optimized
        try:
            result = benchmark_version(compute_optimized, "Optimized (NumPy)", 
                                      data, args.iterations, args.warmup)
            results['optimized'] = result
            charge_results['optimized'] = result['charges']
        except Exception as e:
            print(f"\n❌ Optimized failed: {e}")
        
        # Test Cython
        if CYTHON_AVAILABLE:
            try:
                result = benchmark_version(compute_cython, "Cython (C-compiled)", 
                                          data, args.iterations, args.warmup)
                results['cython'] = result
                charge_results['cython'] = result['charges']
            except Exception as e:
                print(f"\n❌ Cython failed: {e}")
        
        # Numerical consistency check
        if len(charge_results) >= 2:
            consistent = check_consistency(charge_results)
            if not consistent:
                print("\n⚠️  WARNING: Numerical inconsistency detected!")
        
        # Performance comparison
        if len(results) >= 2:
            print(f"\n{'='*70}")
            print(f"PERFORMANCE COMPARISON")
            print(f"{'='*70}")
            
            baseline = list(results.values())[0]
            
            print(f"\n{'Version':<30} {'Mean Time':<15} {'Speedup':<10}")
            print("-"*60)
            
            for version in ['original', 'optimized', 'cython']:
                if version in results:
                    r = results[version]
                    speedup = baseline['mean'] / r['mean']
                    print(f"{r['name']:<30} {r['mean']*1e6:>10.2f} μs   {speedup:>6.2f}x")
    
    print(f"\n{'='*70}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
