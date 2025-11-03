#!/usr/bin/env python3
"""
Poisson Solver Benchmark - Pure Algorithm Speed Test
====================================================
專門測試 Cython vs Original Poisson 算法的性能差異
不包含完整 OpenMM 模擬，只測試 Poisson solver 核心計算

測試內容:
1. Original Python version (from MM_classes.py)
2. Cython optimized version (from MM_classes_CYTHON.py)

測量指標:
- 單次 Poisson iteration 時間
- 多次 iteration 平均時間
- 總加速比
"""

import sys
import os
import time
import argparse
import configparser
from datetime import datetime
import numpy as np

# Add lib to path
sys.path.insert(0, './lib/')
sys.setrecursionlimit(5000)

# Parse arguments
parser = argparse.ArgumentParser(description="Benchmark Poisson Solver Performance")
parser.add_argument('-c', '--config', default='config_refactored.ini', 
                    help='Config file (for system setup)')
parser.add_argument('-n', '--iterations', type=int, default=10,
                    help='Number of Poisson iterations per test (default: 10)')
parser.add_argument('-r', '--repeats', type=int, default=5,
                    help='Number of times to repeat the benchmark (default: 5)')
parser.add_argument('--warmup', type=int, default=2,
                    help='Warmup iterations (not counted, default: 2)')
args = parser.parse_args()

# Read config
if not os.path.exists(args.config):
    print(f"❌ Error: Config file not found: {args.config}")
    sys.exit(1)

config = configparser.ConfigParser()
config.read(args.config)

# Extract config
sim = config['Simulation']
voltage = sim.getfloat('voltage')
openmm_platform = sim.get('platform', 'CUDA').strip()

files = config['Files']
ffdir = files.get('ffdir')
if not ffdir.endswith('/'):
    ffdir += '/'
pdb_file = files.get('pdb_file')
residue_xml_list = [ffdir + s.strip() for s in files.get('residue_xml_list').split(',')]
ff_xml_list = [ffdir + s.strip() for s in files.get('ff_xml_list').split(',')]

elec = config['Electrodes']
cathode_index = tuple(int(x) for x in elec.get('cathode_index').split(','))
anode_index = tuple(int(x) for x in elec.get('anode_index').split(','))

# Import SAPT exclusions
from sapt_exclusions import *

# ============================================================
# Import different MM versions at module level
# ============================================================
MM_CLASSES = {}
MM_MODULES_AVAILABLE = {}

# Try to import original version
try:
    import MM_classes
    import Fixed_Voltage_routines
    MM_CLASSES['original'] = MM_classes.MM
    MM_MODULES_AVAILABLE['original'] = True
    print("✓ Original version available")
except ImportError as e:
    MM_MODULES_AVAILABLE['original'] = False
    print(f"✗ Original version not available: {e}")

# Try to import Optimized version
try:
    import MM_classes_OPTIMIZED
    import Fixed_Voltage_routines_OPTIMIZED
    MM_CLASSES['optimized'] = MM_classes_OPTIMIZED.MM
    MM_MODULES_AVAILABLE['optimized'] = True
    print("✓ Optimized version available")
except ImportError as e:
    MM_MODULES_AVAILABLE['optimized'] = False
    print(f"✗ Optimized version not available: {e}")

# Try to import Cython version
try:
    import MM_classes_CYTHON
    import Fixed_Voltage_routines_CYTHON
    MM_CLASSES['cython'] = MM_classes_CYTHON.MM
    MM_MODULES_AVAILABLE['cython'] = True
    print("✓ Cython version available")
except ImportError as e:
    MM_MODULES_AVAILABLE['cython'] = False
    print(f"✗ Cython version not available: {e}")

# Import OpenMM at module level
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *

# ============================================================
# Benchmark Helper Functions
# ============================================================

def setup_mm_system(mm_module_name, verbose=True):
    """
    Setup MM system with specified module version
    Returns the initialized MMsys object
    """
    if not MM_MODULES_AVAILABLE.get(mm_module_name, False):
        raise RuntimeError(f"{mm_module_name} version is not available")
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Setting up {mm_module_name} version...")
        print(f"{'='*60}")
    
    # Get the MM class for this version
    MM = MM_CLASSES[mm_module_name]
    
    # Create MM system
    MMsys = MM(
        pdb_list=[pdb_file],
        residue_xml_list=residue_xml_list,
        ff_xml_list=ff_xml_list
    )
    
    MMsys.set_periodic_residue(True)
    MMsys.set_platform(openmm_platform)
    
    # Initialize electrodes
    MMsys.initialize_electrodes(
        voltage,
        cathode_identifier=cathode_index,
        anode_identifier=anode_index,
        chain=True,
        exclude_element=("H",)
    )
    
    # Initialize electrolyte
    MMsys.initialize_electrolyte(Natom_cutoff=100)
    
    # Generate SAPT-FF exclusions
    MMsys.generate_exclusions(flag_SAPT_FF_exclusions=True)
    
    if verbose:
        print(f"✓ {mm_module_name} system initialized")
        print(f"  Cathode atoms: {len(MMsys.Cathode.electrode_atoms)}")
        print(f"  Anode atoms: {len(MMsys.Anode.electrode_atoms)}")
    
    return MMsys


def benchmark_poisson_solver(MMsys, version_name, n_iterations, n_repeats, warmup_iters):
    """
    Benchmark the Poisson solver performance
    
    Returns:
        dict with timing statistics
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking {version_name} Poisson Solver")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  - Poisson iterations per test: {n_iterations}")
    print(f"  - Benchmark repeats: {n_repeats}")
    print(f"  - Warmup iterations: {warmup_iters}")
    print()
    
    # Warmup runs (不計時)
    if warmup_iters > 0:
        print(f"🔥 Warming up ({warmup_iters} iterations)...")
        for i in range(warmup_iters):
            MMsys.Poisson_solver_fixed_voltage(Niterations=1)
        print(f"✓ Warmup complete\n")
    
    # Actual benchmark runs
    times = []
    
    for repeat in range(n_repeats):
        print(f"Run {repeat + 1}/{n_repeats}...", end=" ", flush=True)
        
        start_time = time.perf_counter()
        MMsys.Poisson_solver_fixed_voltage(Niterations=n_iterations)
        end_time = time.perf_counter()
        
        elapsed = end_time - start_time
        times.append(elapsed)
        
        print(f"{elapsed:.6f} s ({elapsed*1000/n_iterations:.3f} ms/iter)")
    
    # Calculate statistics
    times = np.array(times)
    stats = {
        'version': version_name,
        'n_iterations': n_iterations,
        'n_repeats': n_repeats,
        'times': times,
        'total_mean': np.mean(times),
        'total_std': np.std(times),
        'total_min': np.min(times),
        'total_max': np.max(times),
        'per_iter_mean': np.mean(times) / n_iterations,
        'per_iter_std': np.std(times) / n_iterations,
    }
    
    return stats


def print_statistics(stats):
    """Print detailed statistics"""
    print(f"\n{'='*60}")
    print(f"Statistics for {stats['version']}")
    print(f"{'='*60}")
    print(f"Total time ({stats['n_iterations']} iterations):")
    print(f"  Mean:   {stats['total_mean']*1000:.3f} ± {stats['total_std']*1000:.3f} ms")
    print(f"  Min:    {stats['total_min']*1000:.3f} ms")
    print(f"  Max:    {stats['total_max']*1000:.3f} ms")
    print(f"\nPer iteration:")
    print(f"  Mean:   {stats['per_iter_mean']*1000:.3f} ± {stats['per_iter_std']*1000:.3f} ms")
    print(f"  Min:    {stats['total_min']*1000/stats['n_iterations']:.3f} ms")
    print(f"  Max:    {stats['total_max']*1000/stats['n_iterations']:.3f} ms")


def compare_versions(original_stats, cython_stats):
    """Compare and print speedup"""
    print(f"\n{'='*60}")
    print(f"PERFORMANCE COMPARISON")
    print(f"{'='*60}")
    
    speedup = original_stats['total_mean'] / cython_stats['total_mean']
    speedup_per_iter = original_stats['per_iter_mean'] / cython_stats['per_iter_mean']
    
    print(f"\n{'Version':<15} {'Total Time':<20} {'Per Iteration':<20}")
    print(f"{'-'*60}")
    print(f"{'Original':<15} {original_stats['total_mean']*1000:>8.3f} ms       "
          f"{original_stats['per_iter_mean']*1000:>8.3f} ms")
    print(f"{'Cython':<15} {cython_stats['total_mean']*1000:>8.3f} ms       "
          f"{cython_stats['per_iter_mean']*1000:>8.3f} ms")
    print(f"{'-'*60}")
    print(f"{'Speedup':<15} {speedup:>8.2f}x            {speedup_per_iter:>8.2f}x")
    
    time_saved = original_stats['total_mean'] - cython_stats['total_mean']
    percentage_saved = (time_saved / original_stats['total_mean']) * 100
    
    print(f"\n⚡ Time saved per {original_stats['n_iterations']} iterations: "
          f"{time_saved*1000:.3f} ms ({percentage_saved:.1f}%)")
    print(f"⚡ Time saved per iteration: "
          f"{time_saved*1000/original_stats['n_iterations']:.3f} ms")
    
    # Extrapolate to real simulation
    print(f"\n{'='*60}")
    print(f"EXTRAPOLATION TO FULL SIMULATION")
    print(f"{'='*60}")
    
    # Typical simulation parameters
    sim_time_ns = 1.0  # 1 ns simulation
    charge_update_freq_fs = 10  # 每 10 fs 更新一次
    poisson_iters_per_update = 4  # 每次更新 4 iterations
    
    total_updates = int(sim_time_ns * 1e6 / charge_update_freq_fs)
    total_poisson_iters = total_updates * poisson_iters_per_update
    
    original_time_s = original_stats['per_iter_mean'] * total_poisson_iters
    cython_time_s = cython_stats['per_iter_mean'] * total_poisson_iters
    time_saved_s = original_time_s - cython_time_s
    
    print(f"For a typical {sim_time_ns} ns simulation:")
    print(f"  - Charge updates: {total_updates:,}")
    print(f"  - Total Poisson iterations: {total_poisson_iters:,}")
    print(f"\nPoisson solver time:")
    print(f"  Original: {original_time_s/60:.1f} minutes")
    print(f"  Cython:   {cython_time_s/60:.1f} minutes")
    print(f"  Saved:    {time_saved_s/60:.1f} minutes ({percentage_saved:.1f}%)")


# ============================================================
# Main Benchmark
# ============================================================

def main():
    print("\n" + "="*60)
    print("POISSON SOLVER BENCHMARK")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config file: {args.config}")
    print(f"PDB file: {pdb_file}")
    
    results = {}
    
    # Test 1: Original version
    if MM_MODULES_AVAILABLE.get('original', False):
        try:
            print("\n" + "🐍 "*20)
            MMsys_original = setup_mm_system("original", verbose=True)
            stats_original = benchmark_poisson_solver(
                MMsys_original, "Original Python", 
                args.iterations, args.repeats, args.warmup
            )
            print_statistics(stats_original)
            results['original'] = stats_original
        except Exception as e:
            print(f"\n❌ Original version failed: {e}")
            import traceback
            traceback.print_exc()
            results['original'] = None
    else:
        print("\n⚠️  Skipping Original version (not available)")
        results['original'] = None
    
    # Test 2: Optimized version
    if MM_MODULES_AVAILABLE.get('optimized', False):
        try:
            print("\n" + "📊 "*20)
            MMsys_optimized = setup_mm_system("optimized", verbose=True)
            stats_optimized = benchmark_poisson_solver(
                MMsys_optimized, "NumPy Optimized",
                args.iterations, args.repeats, args.warmup
            )
            print_statistics(stats_optimized)
            results['optimized'] = stats_optimized
        except Exception as e:
            print(f"\n❌ Optimized version failed: {e}")
            import traceback
            traceback.print_exc()
            results['optimized'] = None
    else:
        print("\n⚠️  Skipping Optimized version (not available)")
        results['optimized'] = None
    
    # Test 3: Cython version
    if MM_MODULES_AVAILABLE.get('cython', False):
        try:
            print("\n" + "⚡ "*20)
            MMsys_cython = setup_mm_system("cython", verbose=True)
            stats_cython = benchmark_poisson_solver(
                MMsys_cython, "Cython Optimized",
                args.iterations, args.repeats, args.warmup
            )
            print_statistics(stats_cython)
            results['cython'] = stats_cython
        except Exception as e:
            print(f"\n❌ Cython version failed: {e}")
            import traceback
            traceback.print_exc()
            results['cython'] = None
    else:
        print("\n⚠️  Skipping Cython version (not available)")
        results['cython'] = None
    
    # Comparison - now supports 3 versions
    available_versions = [k for k, v in results.items() if v is not None]
    if len(available_versions) >= 2:
        print(f"\n{'='*60}")
        print(f"PERFORMANCE COMPARISON ({len(available_versions)} versions)")
        print(f"{'='*60}")
        
        # Use original as baseline
        baseline_key = 'original' if 'original' in available_versions else available_versions[0]
        baseline_stats = results[baseline_key]
        
        print(f"\n{'Version':<20} {'Total Time':<20} {'Per Iteration':<20} {'Speedup':<10}")
        print(f"{'-'*75}")
        
        for version in ['original', 'optimized', 'cython']:
            if version in available_versions:
                stats = results[version]
                speedup = baseline_stats['total_mean'] / stats['total_mean'] if version != baseline_key else 1.0
                print(f"{stats['version']:<20} {stats['total_mean']*1000:>8.3f} ms       "
                      f"{stats['per_iter_mean']*1000:>8.3f} ms       {speedup:>6.2f}x")
        
        print(f"{'-'*75}")
        
        # Detailed comparison
        if 'original' in available_versions:
            print(f"\n{'='*60}")
            print("TIME SAVINGS vs Original")
            print(f"{'='*60}")
            
            for version in ['optimized', 'cython']:
                if version in available_versions:
                    stats = results[version]
                    time_saved = (baseline_stats['total_mean'] - stats['total_mean']) * 1000
                    percent_saved = ((baseline_stats['total_mean'] - stats['total_mean']) / baseline_stats['total_mean']) * 100
                    speedup = baseline_stats['total_mean'] / stats['total_mean']
                    
                    print(f"\n{stats['version']}:")
                    print(f"  Speedup:     {speedup:.2f}x")
                    print(f"  Time saved:  {time_saved:.3f} ms ({percent_saved:.1f}%)")
    else:
        print("\n⚠️  Not enough versions available for comparison")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"BENCHMARK COMPLETE")
    print(f"{'='*60}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Save results to file
    output_file = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(output_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("POISSON SOLVER BENCHMARK RESULTS\n")
        f.write("="*60 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Config: {args.config}\n")
        f.write(f"Iterations: {args.iterations}\n")
        f.write(f"Repeats: {args.repeats}\n")
        f.write(f"Warmup: {args.warmup}\n\n")
        
        for version in ['original', 'optimized', 'cython']:
            if results.get(version):
                stats = results[version]
                f.write(f"{stats['version']}:\n")
                f.write(f"  Mean time: {stats['total_mean']*1000:.3f} ms\n")
                f.write(f"  Per iter:  {stats['per_iter_mean']*1000:.3f} ms\n\n")
        
        # Calculate speedups
        if results.get('original'):
            f.write("Speedups vs Original:\n")
            baseline = results['original']['total_mean']
            
            if results.get('optimized'):
                speedup = baseline / results['optimized']['total_mean']
                f.write(f"  Optimized: {speedup:.2f}x\n")
            
            if results.get('cython'):
                speedup = baseline / results['cython']['total_mean']
                f.write(f"  Cython:    {speedup:.2f}x\n")
    
    print(f"\n✓ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
