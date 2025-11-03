#!/usr/bin/env python3
"""
benchmark.py
「快又準才是真本事」 的 Benchmark 腳本。

1. "準" (Correctness): 驗證 CYTHON == Original 
                     (並證明 OPTIMIZED 是錯的 - P10 BUG)
2. "快" (Speed):     使用 timeit 精確測量 Poisson_solver_fixed_voltage
3. "大道至簡":     零 I/O、零「垃圾」操作。
"""

import sys
import timeit
import numpy as np
from contextlib import redirect_stdout
import io

# --- 1. Linus 哲學：匯入所有版本 ---
sys.path.insert(0, './lib/')

print("="*60)
print("🔥 正在匯入三個版本的 MM... (1/3)")
print("   (1) 黃金標準 (Original)")
import MM_classes as MM_Orig_Module
import Fixed_Voltage_routines as FV_Orig_Module

print("   (2) 錯誤的快取版 (OPTIMIZED)")
import MM_classes_OPTIMIZED as MM_Opt_Module
import Fixed_Voltage_routines_OPTIMIZED as FV_Opt_Module

print("   (3) 你的最終版 (CYTHON)")
import MM_classes_CYTHON as MM_Cython_Module
import Fixed_Voltage_routines_CYTHON as FV_Cython_Module

from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *

# --- 2. 「實用主義」：定義 Benchmark 參數 ---
# (100% 匹配你的 run_openMM.py 設定)
PDB_FILE = 'for_openmm.pdb' #
FFDIR = './ffdir/'
RESIDUE_XML = [FFDIR + 'sapt_residues.xml', FFDIR + 'graph_residue_c.xml', FFDIR + 'graph_residue_n.xml'] #
FF_XML = [FFDIR + 'sapt_noDB_2sheets.xml', FFDIR + 'graph_c_freeze.xml', FFDIR + 'graph_n_freeze.xml'] #
CATHODE_IDX = (0, 2) #
ANODE_IDX = (1, 3) #
VOLTAGE = 0.0 #
PLATFORM = 'CUDA' #

# Benchmark 控制
N_ITERATIONS = 4      # (Poisson 迭代次數)
N_TIMEIT_RUNS = 100   # (timeit 執行次數)

# --- 3. 「好品味」：Setup 函式 (只在*外面*做 I/O) ---
def setup_simulation(MM_Module, FV_Module):
    """
    建立、初始化並返回一個 MMsys 物件。
    這包含了所有的「慢速」I/O 和 OpenMM 設定。
    """
    # 1. 抑制所有「垃圾」 print 輸出
    f = io.StringIO()
    with redirect_stdout(f):
        # 2. 建立 MMsys 物件 (匹配 run_openMM.py)
        MMsys = MM_Module.MM(
            pdb_list=[PDB_FILE],
            residue_xml_list=RESIDUE_XML,
            ff_xml_list=FF_XML
        )
        MMsys.set_periodic_residue(True)
        MMsys.set_platform(PLATFORM)

        # 3. 初始化電極 (I/O 和設定)
        MMsys.initialize_electrodes(
            VOLTAGE,
            cathode_identifier=CATHODE_IDX,
            anode_identifier=ANODE_IDX,
            chain=True,
            exclude_element=("H",)
        )
        MMsys.initialize_electrolyte(Natom_cutoff=100)
        
        # 4. 載入「黃金標準」 (P13 還原)
        # (我們必須手動把 FV_Module 裡的函式綁定到 MMsys.Cathode 上)
        # (這是 P13 修復 的「手動」版)
        
        # --- P13/P14 修復 (手動) ---
        # 確保 Cathode 呼叫的是*正確*版本的 compute_Electrode_charge_analytic
        MMsys.Cathode.compute_Electrode_charge_analytic = \
            FV_Module.Electrode_Virtual.compute_Electrode_charge_analytic.__get__(MMsys.Cathode, FV_Module.Electrode_Virtual)
        MMsys.Anode.compute_Electrode_charge_analytic = \
            FV_Module.Electrode_Virtual.compute_Electrode_charge_analytic.__get__(MMsys.Anode, FV_Module.Electrode_Virtual)
        
        # 確保 P3/P8 (數學 BUG) 100% 被修復
        MMsys.Scale_charges_analytic_general = \
             MM_Module.MM.Scale_charges_analytic_general.__get__(MMsys, MM_Module.MM)

        # 確保 P1b.3 (熱迴圈優化) 被綁定
        if hasattr(FV_Module, 'scale_electrode_charges_cython'):
             MMsys.Cathode.Scale_charges_analytic = \
                FV_Module.Electrode_Virtual.Scale_charges_analytic.__get__(MMsys.Cathode, FV_Module.Electrode_Virtual)
             MMsys.Anode.Scale_charges_analytic = \
                FV_Module.Electrode_Virtual.Scale_charges_analytic.__get__(MMsys.Anode, FV_Module.Electrode_Virtual)

        MMsys.generate_exclusions(flag_SAPT_FF_exclusions=True)
    
    return MMsys

# --- 4. 「快又準」：驗證函式 ---
def get_charges(MMsys):
    """
    從 MMsys 物件中提取「最終」的陰極電荷。
    """
    # 讀取 Cathode 電荷
    if hasattr(MMsys, 'Cathode'):
        return np.array([atom.charge for atom in MMsys.Cathode.electrode_atoms])
    # (如果 MMsys 是 CYTHON 版，它可能繼承了 OPTIMIZED，我們需要安全地讀取)
    elif hasattr(MMsys, 'Cathode_Virtual'):
         return np.array([atom.charge for atom in MMsys.Cathode_Virtual.electrode_atoms])
    else:
        # Fallback (e.g. Original)
        return np.array([atom.charge for atom in MMsys.Cathode.electrode_atoms])


# --- 5. 「上工了」：主執行緒 ---
def main():
    
    print("="*60)
    print(f"🔥 Benchmark 啟動：{PDB_FILE}")
    print(f"   Poisson 迭代: {N_ITERATIONS} | Timeit 執行: {N_TIMEIT_RUNS}")
    print("="*60)

    # --- Setup (零 I/O 計時) ---
    print("... 正在設定 (1/3) Original MMsys...")
    MM_Orig = setup_simulation(MM_Orig_Module, FV_Orig_Module)
    
    print("... 正在設定 (2/3) OPTIMIZED MMsys...")
    MM_Opt = setup_simulation(MM_Opt_Module, FV_Opt_Module)
    
    print("... 正在設定 (3/3) CYTHON MMsys...")
    MM_Cython = setup_simulation(MM_Cython_Module, FV_Cython_Module)

    # --- 狀態同步 (100% 關鍵！) ---
    print("... 正在同步 OpenMM 狀態 (MD 1 步)...")
    MM_Orig.simmd.step(1)
    saved_state = MM_Orig.simmd.context.getState(
        getPositions=True, getForces=True, getVelocities=True
    )
    
    # 「實用主義」：強制所有版本 100% 從同一個「髒」狀態開始
    MM_Opt.simmd.context.setState(saved_state)
    MM_Cython.simmd.context.setState(saved_state)
    
    print("✅ 狀態 100% 同步。")
    print("="*60)

    # --- 1. 正確性比較 (P10 BUG 獵殺) ---
    print(f"🔥 1. 正確性比較 (N_iter={N_ITERATIONS})")
    
    # 「好品味」：我們必須*呼叫*函式來獲取結果
    MM_Orig.Poisson_solver_fixed_voltage(Niterations=N_ITERATIONS)
    q_orig = get_charges(MM_Orig)

    MM_Opt.Poisson_solver_fixed_voltage(Niterations=N_ITERATIONS)
    q_opt = get_charges(MM_Opt)

    MM_Cython.Poisson_solver_fixed_voltage(Niterations=N_ITERATIONS)
    q_cython = get_charges(MM_Cython)

    # 驗證 P13/P14 修復 (CYTHON 必須 100% 匹配 Original)
    try:
        assert np.allclose(q_orig, q_cython, atol=1e-8)
        print("✅ 【準】 P13/P14 修復成功：CYTHON == Original")
    except AssertionError:
        print("❌ 【準】 P13/P14 修復失敗：CYTHON != Original")
        diff = np.max(np.abs(q_orig - q_cython))
        print(f"   (最大誤差: {diff})")

    # 驗證 P10 BUG (OPTIMIZED 必須 100% *不*匹配 Original)
    try:
        assert not np.allclose(q_orig, q_opt, atol=1e-8)
        print("✅ 【準】 P10 BUG 獵殺成功：OPTIMIZED != Original (這是好事！)")
    except AssertionError:
        print("⚠️ 【準】 P10 BUG 似乎消失了？：OPTIMIZED == Original")

    print("="*60)

    # --- 2. 速度比較 (P1 熱迴圈 壓榨) ---
    print(f"🔥 2. 速度比較 (N_iter={N_ITERATIONS}, N_runs={N_TIMEIT_RUNS})")

    # 「大道至簡」：timeit setup 100% 只做「還原」
    setup_orig = "MM_Orig.simmd.context.setState(saved_state)"
    stmt_orig = f"MM_Orig.Poisson_solver_fixed_voltage(Niterations={N_ITERATIONS})"

    setup_opt = "MM_Opt.simmd.context.setState(saved_state)"
    stmt_opt = f"MM_Opt.Poisson_solver_fixed_voltage(Niterations={N_ITERATIONS})"

    setup_cython = "MM_Cython.simmd.context.setState(saved_state)"
    stmt_cython = f"MM_Cython.Poisson_solver_fixed_voltage(Niterations={N_ITERATIONS})"

    # 執行 Benchmark
    print(f"... 正在執行 Original (x{N_TIMEIT_RUNS})...")
    t_orig = timeit.timeit(stmt_orig, setup=setup_orig, globals=locals(), number=N_TIMEIT_RUNS) / N_TIMEIT_RUNS

    print(f"... 正在執行 OPTIMIZED (x{N_TIMEIT_RUNS})...")
    t_opt = timeit.timeit(stmt_opt, setup=setup_opt, globals=locals(), number=N_TIMEIT_RUNS) / N_TIMEIT_RUNS

    print(f"... 正在執行 CYTHON (x{N_TIMEIT_RUNS})...")
    t_cython = timeit.timeit(stmt_cython, setup=setup_cython, globals=locals(), number=N_TIMEIT_RUNS) / N_TIMEIT_RUNS

    print("="*60)
    print(f"🔥 3. 最終結果 (Poisson Solver Call)")
    print(f"   🐍 Original:  {t_orig * 1000:.4f} ms")
    print(f"   📊 OPTIMIZED: {t_opt * 1000:.4f} ms  (加速: {t_orig / t_opt:.2f}x)")
    print(f"   🔥 CYTHON:    {t_cython * 1000:.4f} ms  (加速: {t_orig / t_cython:.2f}x)")
    print("="*60)

    if (t_cython < t_opt) and (t_opt < t_orig):
        print("✅ 【快】 P1 壓榨成功：CYTHON < OPTIMIZED < Original")
    else:
        print("⚠️ 【快】 警告：CYTHON 並非最快！")
        
    print("\n😎 上工了。")


if __name__ == "__main__":
    # --- 3. 規模變化 ---
    # 這是「好品味」 的做法：
    # 1. 編輯 PDB_FILE 變數
    # 2. 重新執行此腳本
    # 3. 把數據貼到 Excel/Gnuplot
    # (不要把「繪圖」和「I/O 迴圈」 這種「垃圾」 和 benchmark 混在一起)
    main()