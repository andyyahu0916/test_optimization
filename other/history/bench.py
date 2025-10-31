import time
import numpy as np
import sys
import os
import argparse
import configparser
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *

# 設置與 run_openMM.py 一致的 sys.path
sys.path.append('./lib/')  # 從 run_openMM.py 複製，確保能找到 MM_classes 等模組

# 讀取 config.ini，與 run_openMM.py 一致
def load_config(config_path='config.ini'):
    config = configparser.ConfigParser()
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)
    config.read(config_path)
    return config

# 提取 run_openMM.py 的檔案路徑和參數
config = load_config()
sim_config = config['Simulation']
file_config = config['Files']
elec_config = config['Electrodes']

# 路徑設定
outPath = file_config.get('outPath')
ffdir = file_config.get('ffdir')
if not ffdir.endswith('/'):
    ffdir += '/'
pdb_list = [file_config.get('pdb_file')]
residue_xml_list = [ffdir + s.strip() for s in file_config.get('residue_xml_list').split(',')]
ff_xml_list = [ffdir + s.strip() for s in file_config.get('ff_xml_list').split(',')]
cathode_index = tuple(int(s.strip()) for s in elec_config.get('cathode_index').split(','))
anode_index = tuple(int(s.strip()) for s in elec_config.get('anode_index').split(','))
platform = sim_config.get('platform', 'CUDA')
Niterations = 10  # Poisson solver 迭代次數
num_runs = 10    # 跑 10 次取平均，獲得更穩定的結果
simulation_time_ns = 0.001  # 小模擬時間，僅測試效率
freq_traj_output_ps = sim_config.getint('freq_traj_output_ps', 10)

# 載入原始版本
try:
    from MM_classes import MM as MM_original
    from Fixed_Voltage_routines import *
    print("✅ Original module loaded successfully!")
except ImportError as e:
    print(f"Error loading original modules: {e}")
    sys.exit(1)

# 載入 NumPy 優化版本
try:
    from MM_classes_OPTIMIZED import MM as MM_optimized
    from Fixed_Voltage_routines_OPTIMIZED import *
    print("✅ Optimized (NumPy vectorized) module loaded successfully!")
except ImportError as e:
    print(f"⚠️ Warning: Optimized module not found: {e}")
    MM_optimized = None

# 載入 Cython 版本
try:
    from MM_classes_CYTHON import MM as MM_cython
    from Fixed_Voltage_routines_CYTHON import *
    import electrode_charges_cython
    print("✅ Cython module loaded successfully!")
except ImportError as e:
    print(f"⚠️ Warning: Cython module not found: {e}")
    print("Ensure Cython module is compiled: python setup_cython.py build_ext --inplace")
    MM_cython = None

# Benchmark 函數：原始版本
def run_original_poisson():
    MMsys_orig = MM_original(pdb_list=pdb_list, residue_xml_list=residue_xml_list, ff_xml_list=ff_xml_list)
    MMsys_orig.set_platform(platform)
    MMsys_orig.set_periodic_residue(True)
    MMsys_orig.initialize_electrodes(
        Voltage=0.0,  # 測試用，設為 0V 簡化
        cathode_identifier=cathode_index,
        anode_identifier=anode_index,
        chain=True,
        exclude_element=("H",)
    )
    MMsys_orig.initialize_electrolyte(Natom_cutoff=100)
    
    # 設置輸出
    os.makedirs(outPath, exist_ok=True)
    trajectory_file = os.path.join(outPath, 'test_orig.dcd')
    MMsys_orig.set_trajectory_output(trajectory_file, freq_traj_output_ps * 1000)
    
    # 🔥 NEW: 追蹤每次迭代的電荷變化 (檢測誤差累積)
    charge_history = []
    
    start_time = time.time()
    for i in range(num_runs):
        MMsys_orig.Poisson_solver_fixed_voltage(Niterations=Niterations)
        # 記錄每次 Poisson solver 後的電荷
        cathode_charges_iter = np.array([atom.charge for atom in MMsys_orig.Cathode.electrode_atoms])
        charge_history.append(cathode_charges_iter.copy())
    time_taken = (time.time() - start_time) / num_runs
    
    # 收集最終陰極電荷
    cathode_charges = np.array([atom.charge for atom in MMsys_orig.Cathode.electrode_atoms])
    total_charge = np.sum(cathode_charges)
    
    return time_taken, total_charge, cathode_charges, charge_history

# Benchmark 函數：NumPy 優化版本
def run_optimized_poisson():
    MMsys_opt = MM_optimized(pdb_list=pdb_list, residue_xml_list=residue_xml_list, ff_xml_list=ff_xml_list)
    MMsys_opt.set_platform(platform)
    MMsys_opt.set_periodic_residue(True)
    MMsys_opt.initialize_electrodes(
        Voltage=0.0,
        cathode_identifier=cathode_index,
        anode_identifier=anode_index,
        chain=True,
        exclude_element=("H",)
    )
    MMsys_opt.initialize_electrolyte(Natom_cutoff=100)
    
    # 設置輸出
    trajectory_file = os.path.join(outPath, 'test_opt.dcd')
    MMsys_opt.set_trajectory_output(trajectory_file, freq_traj_output_ps * 1000)
    
    # 🔥 NEW: 追蹤每次迭代的電荷變化
    charge_history = []
    
    start_time = time.time()
    for i in range(num_runs):
        MMsys_opt.Poisson_solver_fixed_voltage(Niterations=Niterations)
        cathode_charges_iter = np.array([atom.charge for atom in MMsys_opt.Cathode.electrode_atoms])
        charge_history.append(cathode_charges_iter.copy())
    time_taken = (time.time() - start_time) / num_runs
    
    # 收集陰極電荷
    cathode_charges = np.array([atom.charge for atom in MMsys_opt.Cathode.electrode_atoms])
    total_charge = np.sum(cathode_charges)
    
    return time_taken, total_charge, cathode_charges, charge_history

# Benchmark 函數：Cython 版本
def run_cython_poisson():
    MMsys_cyth = MM_cython(pdb_list=pdb_list, residue_xml_list=residue_xml_list, ff_xml_list=ff_xml_list)
    MMsys_cyth.set_platform(platform)
    MMsys_cyth.set_periodic_residue(True)
    MMsys_cyth.initialize_electrodes(
        Voltage=0.0,
        cathode_identifier=cathode_index,
        anode_identifier=anode_index,
        chain=True,
        exclude_element=("H",)
    )
    MMsys_cyth.initialize_electrolyte(Natom_cutoff=100)
    
    # 設置輸出
    trajectory_file = os.path.join(outPath, 'test_cyth.dcd')
    MMsys_cyth.set_trajectory_output(trajectory_file, freq_traj_output_ps * 1000)
    
    # 🔥 NEW: 追蹤每次迭代的電荷變化
    charge_history = []
    
    start_time = time.time()
    for i in range(num_runs):
        MMsys_cyth.Poisson_solver_fixed_voltage(Niterations=Niterations)
        cathode_charges_iter = np.array([atom.charge for atom in MMsys_cyth.Cathode.electrode_atoms])
        charge_history.append(cathode_charges_iter.copy())
    time_taken = (time.time() - start_time) / num_runs
    
    # 收集陰極電荷
    cathode_charges = np.array([atom.charge for atom in MMsys_cyth.Cathode.electrode_atoms])
    total_charge = np.sum(cathode_charges)
    
    return time_taken, total_charge, cathode_charges, charge_history

# 運行 benchmark
if __name__ == "__main__":
    print("=" * 70)
    print(f"Running benchmark at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"System: {len(pdb_list)} PDB, {platform} platform")
    print(f"Test parameters: {Niterations} iterations, {num_runs} runs per version")
    print("=" * 70)
    
    # 清空輸出目錄
    if os.path.exists(outPath):
        import shutil
        shutil.rmtree(outPath)
    os.makedirs(outPath)
    
    results = {}
    
    # 測試原始版本
    print("\n[1/3] Running ORIGINAL version benchmark...")
    time_orig, total_orig, charges_orig, history_orig = run_original_poisson()
    results['original'] = {'time': time_orig, 'total_charge': total_orig, 'charges': charges_orig, 'history': history_orig}
    print(f"✓ Original completed: {time_orig:.4f} seconds per run")
    
    # 測試 NumPy 優化版本
    if MM_optimized is not None:
        print("\n[2/3] Running OPTIMIZED (NumPy vectorized) version benchmark...")
        time_opt, total_opt, charges_opt, history_opt = run_optimized_poisson()
        results['optimized'] = {'time': time_opt, 'total_charge': total_opt, 'charges': charges_opt, 'history': history_opt}
        print(f"✓ Optimized completed: {time_opt:.4f} seconds per run")
    else:
        print("\n[2/3] ⊘ Optimized version SKIPPED (not available)")
        results['optimized'] = None
    
    # 測試 Cython 版本
    if MM_cython is not None:
        print("\n[3/3] Running CYTHON version benchmark...")
        time_cyth, total_cyth, charges_cyth, history_cyth = run_cython_poisson()
        results['cython'] = {'time': time_cyth, 'total_charge': total_cyth, 'charges': charges_cyth, 'history': history_cyth}
        print(f"✓ Cython completed: {time_cyth:.4f} seconds per run")
    else:
        print("\n[3/3] ⊘ Cython version SKIPPED (not available)")
        results['cython'] = None
    
    # 輸出效率結果
    print("\n" + "=" * 70)
    print("=== PERFORMANCE COMPARISON ===")
    print("=" * 70)
    print(f"{'Version':<20} {'Time (s)':<15} {'Speedup':<15}")
    print("-" * 70)
    
    baseline_time = results['original']['time']
    print(f"{'Original':<20} {baseline_time:<15.4f} {'1.00x (baseline)':<15}")
    
    if results['optimized'] is not None:
        speedup_opt = baseline_time / results['optimized']['time']
        print(f"{'Optimized (NumPy)':<20} {results['optimized']['time']:<15.4f} {speedup_opt:<15.2f}x")
    
    if results['cython'] is not None:
        speedup_cyth = baseline_time / results['cython']['time']
        print(f"{'Cython':<20} {results['cython']['time']:<15.4f} {speedup_cyth:<15.2f}x")
    
    print("-" * 70)
    
    # 檢查計算結果一致性
    print("\n" + "=" * 70)
    print("=== RESULT CONSISTENCY CHECK ===")
    print("=" * 70)
    
    baseline_charge = results['original']['total_charge']
    baseline_charges = results['original']['charges']
    
    print(f"{'Version':<20} {'Total Charge Diff':<20} {'MAE (charges)':<20}")
    print("-" * 70)
    
    if results['optimized'] is not None:
        charge_diff_opt = np.abs(baseline_charge - results['optimized']['total_charge'])
        mae_opt = np.mean(np.abs(baseline_charges - results['optimized']['charges']))
        status_opt = "✓ OK" if charge_diff_opt < 1e-6 and mae_opt < 1e-6 else "⚠ DIFF"
        print(f"{'Optimized vs Orig':<20} {charge_diff_opt:<20.8e} {mae_opt:<20.8e} {status_opt}")
    
    if results['cython'] is not None:
        charge_diff_cyth = np.abs(baseline_charge - results['cython']['total_charge'])
        mae_cyth = np.mean(np.abs(baseline_charges - results['cython']['charges']))
        status_cyth = "✓ OK" if charge_diff_cyth < 1e-6 and mae_cyth < 1e-6 else "⚠ DIFF"
        print(f"{'Cython vs Orig':<20} {charge_diff_cyth:<20.8e} {mae_cyth:<20.8e} {status_cyth}")
    
    print("-" * 70)
    
    # 🔥 NEW: 誤差累積分析
    print("\n" + "=" * 70)
    print("=== ERROR ACCUMULATION ANALYSIS ===")
    print("=" * 70)
    print(f"Analyzing error growth over {num_runs} iterations...")
    print(f"(Each iteration runs Poisson solver with {Niterations} inner steps)")
    print("-" * 70)
    
    baseline_history = results['original']['history']
    
    def analyze_error_accumulation(name, charge_history):
        """分析誤差累積情況"""
        # 計算每次迭代相對於 Original baseline 的誤差
        errors = []
        for i in range(len(charge_history)):
            mae = np.mean(np.abs(charge_history[i] - baseline_history[i]))
            errors.append(mae)
        
        errors = np.array(errors)
        
        # 統計量
        initial_error = errors[0]
        final_error = errors[-1]
        max_error = np.max(errors)
        mean_error = np.mean(errors)
        
        # 誤差增長率 (線性擬合斜率)
        iterations = np.arange(len(errors))
        if len(errors) > 1:
            # 使用最小二乘法擬合: error = a * iteration + b
            fit_coeffs = np.polyfit(iterations, errors, 1)
            growth_rate = fit_coeffs[0]  # 斜率
            
            # 判斷增長類型
            if growth_rate < initial_error * 0.01:  # 增長率 < 1% 初始誤差
                status = "✓ STABLE (constant error)"
            elif growth_rate < max_error / len(errors):  # 次線性增長
                status = "✓ STABLE (sub-linear growth)"
            elif growth_rate < max_error * 2 / len(errors):  # 線性增長
                status = "⚠ LINEAR GROWTH"
            else:  # 超線性增長
                status = "✗ UNSTABLE (super-linear growth)"
        else:
            growth_rate = 0.0
            status = "N/A (single iteration)"
        
        return {
            'initial': initial_error,
            'final': final_error,
            'max': max_error,
            'mean': mean_error,
            'growth_rate': growth_rate,
            'status': status
        }
    
    if results['optimized'] is not None:
        print(f"\n{'OPTIMIZED vs Original:':<30}")
        analysis_opt = analyze_error_accumulation('OPTIMIZED', results['optimized']['history'])
        print(f"  Initial error (iter 0):      {analysis_opt['initial']:.4e}")
        print(f"  Final error (iter {num_runs-1}):       {analysis_opt['final']:.4e}")
        print(f"  Max error:                   {analysis_opt['max']:.4e}")
        print(f"  Mean error:                  {analysis_opt['mean']:.4e}")
        print(f"  Error growth rate:           {analysis_opt['growth_rate']:.4e} per iteration")
        print(f"  Status:                      {analysis_opt['status']}")
    
    if results['cython'] is not None:
        print(f"\n{'CYTHON vs Original:':<30}")
        analysis_cyth = analyze_error_accumulation('CYTHON', results['cython']['history'])
        print(f"  Initial error (iter 0):      {analysis_cyth['initial']:.4e}")
        print(f"  Final error (iter {num_runs-1}):       {analysis_cyth['final']:.4e}")
        print(f"  Max error:                   {analysis_cyth['max']:.4e}")
        print(f"  Mean error:                  {analysis_cyth['mean']:.4e}")
        print(f"  Error growth rate:           {analysis_cyth['growth_rate']:.4e} per iteration")
        print(f"  Status:                      {analysis_cyth['status']}")
    
    print("-" * 70)
    print("\n✓ Benchmark completed successfully!")
    print("=" * 70)