#!/usr/bin/env python3
"""
測試 Warm Start 優化的效果

比較:
1. 無 Warm Start: 每次都從 initialize_Charge 開始
2. 有 Warm Start: 使用上次收斂的電荷作為初始值

預期: Warm Start 版本應該更快 (1.3-1.5x),但結果精度相同
"""

import time
import numpy as np
import os
import sys

# Add lib to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

from MM_classes_CYTHON import MM as MM_cython

# Configuration
pdb_list = ["for_openmm.pdb"]
residue_xml_list = ["ffdir/sapt_residues.xml", "ffdir/dummy_res.xml"]
ff_xml_list = ["ffdir/sapt.xml", "ffdir/dummy.xml"]
cathode_index = 4
anode_index = 5
platform = "CUDA"
Niterations = 10
num_runs = 20  # 測試 20 次連續調用

print("=" * 70)
print("🧪 Testing Warm Start Optimization")
print("=" * 70)
print(f"Configuration:")
print(f"  - Poisson iterations per call: {Niterations}")
print(f"  - Number of consecutive calls: {num_runs}")
print(f"  - Platform: {platform}")
print("=" * 70)

# ============================================
# Test 1: WITH Warm Start (Current version)
# ============================================
print("\n[1/2] Testing WITH Warm Start (current version)...")
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

# First call (no warm start available)
print("  - First call (no warm start yet)...")
t0 = time.time()
MMsys_warm.Poisson_solver_fixed_voltage(Niterations=Niterations)
first_call_time = time.time() - t0
print(f"    Time: {first_call_time:.4f}s")

# Subsequent calls (WITH warm start)
print(f"  - Subsequent {num_runs-1} calls (with warm start)...")
times_warm = []
for i in range(num_runs - 1):
    t0 = time.time()
    MMsys_warm.Poisson_solver_fixed_voltage(Niterations=Niterations)
    times_warm.append(time.time() - t0)

avg_time_warm = np.mean(times_warm)
std_time_warm = np.std(times_warm)

# Save final charges
final_charges_warm = np.array([atom.charge for atom in MMsys_warm.Cathode.electrode_atoms])

print(f"    Average time: {avg_time_warm:.4f}s ± {std_time_warm:.4f}s")

# ============================================
# Test 2: WITHOUT Warm Start (force cold start)
# ============================================
print("\n[2/2] Testing WITHOUT Warm Start (forced cold start)...")
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

times_cold = []
for i in range(num_runs):
    # 🔥 Force cold start by deleting warm start cache
    if hasattr(MMsys_cold, '_warm_start_cathode_charges'):
        delattr(MMsys_cold, '_warm_start_cathode_charges')
    if hasattr(MMsys_cold, '_warm_start_anode_charges'):
        delattr(MMsys_cold, '_warm_start_anode_charges')
    
    t0 = time.time()
    MMsys_cold.Poisson_solver_fixed_voltage(Niterations=Niterations)
    times_cold.append(time.time() - t0)

avg_time_cold = np.mean(times_cold)
std_time_cold = np.std(times_cold)

# Save final charges
final_charges_cold = np.array([atom.charge for atom in MMsys_cold.Cathode.electrode_atoms])

print(f"    Average time: {avg_time_cold:.4f}s ± {std_time_cold:.4f}s")

# ============================================
# Results Comparison
# ============================================
print("\n" + "=" * 70)
print("📊 RESULTS COMPARISON")
print("=" * 70)
print(f"{'Version':<30} {'Avg Time (s)':<20} {'Speedup':<15}")
print("-" * 70)
print(f"{'WITHOUT Warm Start (baseline)':<30} {avg_time_cold:<20.4f} {'1.00x':<15}")
print(f"{'WITH Warm Start':<30} {avg_time_warm:<20.4f} {avg_time_cold/avg_time_warm:<15.2f}x")
print("-" * 70)

speedup = avg_time_cold / avg_time_warm
time_saved_per_call = avg_time_cold - avg_time_warm
print(f"\n🎯 Warm Start Speedup: {speedup:.2f}x")
print(f"⏱️  Time saved per call: {time_saved_per_call*1000:.2f} ms")
print(f"💾 Total time saved ({num_runs} calls): {time_saved_per_call*num_runs:.2f} seconds")

# ============================================
# Accuracy Check
# ============================================
print("\n" + "=" * 70)
print("✅ ACCURACY VERIFICATION")
print("=" * 70)

charge_diff = np.abs(final_charges_warm - final_charges_cold)
mae = np.mean(charge_diff)
max_diff = np.max(charge_diff)

print(f"Mean Absolute Error (MAE): {mae:.4e}")
print(f"Max Difference:            {max_diff:.4e}")

if mae < 1e-10:
    print("✅ PASS: Warm Start does NOT affect final accuracy!")
    print("   (Differences are within numerical precision)")
else:
    print("⚠️  WARNING: Significant differences detected!")
    print("   This should not happen - please investigate")

# ============================================
# Summary for Long Simulations
# ============================================
print("\n" + "=" * 70)
print("💡 IMPACT ON LONG MD SIMULATIONS")
print("=" * 70)

# Example: 100ns simulation with 1fs timestep = 100,000,000 steps
# If Poisson solver called every step: 100M calls
example_calls = 100_000_000
time_saved_total = time_saved_per_call * example_calls

print(f"Example: 100ns MD simulation (1fs timestep)")
print(f"  - Total Poisson solver calls: {example_calls:,}")
print(f"  - Time saved with Warm Start: {time_saved_total/3600:.2f} hours")
print(f"  - Percentage improvement: {(speedup-1)*100:.1f}%")

print("\n" + "=" * 70)
print("✅ Warm Start test completed!")
print("=" * 70)
