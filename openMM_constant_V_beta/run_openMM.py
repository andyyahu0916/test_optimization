from __future__ import print_function
import sys
sys.path.append('./lib/')
#********** OpenMM Drivers
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *

# ⚠️ MM_classes import will be determined by config.ini (see below)
# from MM_classes import *

#*********** Fixed Voltage routines will be imported based on MM version
# from Fixed_Voltage_routines import *
#***************************
import numpy as np
import urllib.request
#from add_customnonbond_xml import add_CustomNonbondedForce_SAPTFF_parameters
# other stuff
from sys import stdout
from time import gmtime, strftime
from datetime import datetime
import shutil
import os  # 確保 os 被 import

# **** 新增的 import ****
import argparse
import configparser

# for electrode sheets, need to up recursion limit for residue atom matching...
sys.setrecursionlimit(2000)

# ============================================================
# 🔥 Linus 重構: 統一 warm-start 激活邏輯
# ============================================================
def should_use_warmstart(
    i_frame: int, 
    current_time_ns: float, 
    warmstart_activated: bool,
    config_enable_warmstart: bool,
    config_warmstart_after_ns: float,
    config_warmstart_after_frames: int
) -> tuple:
    """
    一個函數，統一處理所有 warm-start 激活邏輯。
    
    返回: (use_warmstart_now, new_warmstart_activated_status)
    
    這個函數是為了消除 run_openMM.py 中 'legacy_print' 和 'efficient' 
    兩個模式下重複的 15 行 if/else 垃圾代碼。
    
    不要重複你自己 (Don't Repeat Yourself) 是軟體工程的基本原則。
    違反這個原則的代碼是垃圾。
    """
    
    # 如果全局禁用，直接返回
    if not config_enable_warmstart:
        return False, False
    
    # 如果已經激活，就保持激活
    if warmstart_activated:
        return True, True

    # 尚未激活，檢查是否達到激活條件
    
    # 方式 A: 按時間 (ns)
    if config_warmstart_after_ns > 0:
        if current_time_ns >= config_warmstart_after_ns:
            print(f"\n{'='*80}")
            print(f"🚀 WARM START ACTIVATED at {current_time_ns:.2f} ns (frame {i_frame})")
            print(f"{'='*80}\n")
            return True, True  # 激活
        else:
            return False, False  # 保持 cold start
    
    # 方式 B: 按幀數 (僅在時間控制被禁用的情況下生效)
    elif config_warmstart_after_frames > 0:
        if i_frame >= config_warmstart_after_frames:
            print(f"\n{'='*80}")
            print(f"🚀 WARM START ACTIVATED at frame {i_frame} ({current_time_ns:.2f} ns)")
            print(f"{'='*80}\n")
            return True, True  # 激活
        else:
            return False, False  # 保持 cold start
            
    # 方式 C: 立即激活 (時間=0, 幀數=0)
    else:
        return True, True
# ============================================================


# ============================================================
# 🔥 Linus 重構: 移除 Logger 類 - 土法煉鋼最快
#
# Linus: "Why the fuck do you need a class to print?"
# 
# Legacy print: 直接 print()，然後 `python run.py > log` 重定向
# Efficient: 管理檔案 handles（這個需要，因為要 close）
# ============================================================

# --- 開始：讀取設定檔 ---

# 1. 設定命令列參數解析
parser = argparse.ArgumentParser(description="Run OpenMM Fixed-Voltage MD Simulation")
parser.add_argument('-c', '--config', default='config.ini', help='Path to the configuration file (default: config.ini)')
args = parser.parse_args()

# 2. 讀取 .ini 檔案
config = configparser.ConfigParser()
config_path = args.config
if not os.path.exists(config_path):
    print(f"Error: Configuration file not found at {config_path}")
    sys.exit(1)
    
config.read(config_path)

# 3. 從設定檔中提取參數

# [Simulation] 區塊
sim_config = config['Simulation']
simulation_time_ns = sim_config.getint('simulation_time_ns')
freq_charge_update_fs = sim_config.getint('freq_charge_update_fs')
freq_traj_output_ps = sim_config.getint('freq_traj_output_ps')
simulation_type = sim_config.get('simulation_type')
openmm_platform = sim_config.get('platform')
Voltage = sim_config.getfloat('voltage')

# MM version
mm_version = sim_config.get('mm_version', 'original').lower()

# Warm Start
enable_warmstart = sim_config.getboolean('enable_warmstart', fallback=True)
verify_interval = sim_config.getint('verify_interval', fallback=100)
warmstart_after_ns = sim_config.getfloat('warmstart_after_ns', fallback=0.0)
warmstart_after_frames = sim_config.getint('warmstart_after_frames', fallback=0)

# 🔥 新增：讀取 Logging 模式設定
logging_mode = sim_config.get('logging_mode', 'efficient').lower()
write_charges = sim_config.getboolean('write_charges', fallback=False)
write_components = sim_config.getboolean('write_components', fallback=False)


# Import appropriate MM_classes based on version
if mm_version == 'plugin':
    print("🚀 Loading OpenMM Plugin (C++/CUDA implementation)")
    print("   ⚡ Reference platform: 2-3x speedup (CPU, for testing)")
    print("   🔥 CUDA platform: 10-20x speedup (GPU, production)")
    if enable_warmstart: print("⚠️  Warning: Plugin doesn't use Python warm-start (iteration in kernel)")
    enable_warmstart = False
    # Plugin 不需要 Python 的 MM_classes，但需要基礎類定義
    from MM_classes import *
    from Fixed_Voltage_routines import *
    USE_PLUGIN = True
elif mm_version == 'cython':
    print("🔥 Loading Cython-optimized MM classes (2-5x speedup expected)")
    if enable_warmstart:
        if warmstart_after_ns > 0:
            print(f"🚀 Warm Start will be enabled after {warmstart_after_ns} ns (equilibration period)")
        elif warmstart_after_frames > 0:
            print(f"🚀 Warm Start will be enabled after {warmstart_after_frames} frames")
        else:
            print(f"🚀 Warm Start enabled immediately (verify every {verify_interval} calls, ~1.3-1.5x additional speedup)")
    else:
        print("⚠️  Warm Start disabled (using cold start every time)")
    from MM_classes_CYTHON import *
    from Fixed_Voltage_routines_CYTHON import *
    USE_PLUGIN = False
elif mm_version == 'optimized':
    print("⚡ Loading NumPy-optimized MM classes (1.5-2x speedup expected)")
    if enable_warmstart: print("⚠️  Warning: Warm Start only supported in 'cython' version, ignoring parameter")
    enable_warmstart = False
    from MM_classes_OPTIMIZED import *
    from Fixed_Voltage_routines_OPTIMIZED import *
    USE_PLUGIN = False
else: # 'original'
    print("📦 Loading original MM classes (baseline performance)")
    if enable_warmstart: print("⚠️  Warning: Warm Start only supported in 'cython' version, ignoring parameter")
    enable_warmstart = False
    from MM_classes import *
    from Fixed_Voltage_routines import *
    USE_PLUGIN = False

# [Files] 區塊
file_config = config['Files']
outPath = file_config.get('outPath')
ffdir = file_config.get('ffdir')
if not ffdir.endswith('/'): ffdir += '/'
pdb_list = [file_config.get('pdb_file')]

def parse_xml_list(config_str, prefix_dir):
    return [prefix_dir + s.strip() for s in config_str.split(',')]
residue_xml_list = parse_xml_list(file_config.get('residue_xml_list'), ffdir)
ff_xml_list = parse_xml_list(file_config.get('ff_xml_list'), ffdir)

# [Electrodes] 區塊
elec_config = config['Electrodes']
def parse_index_tuple(config_str):
    return tuple(int(s.strip()) for s in config_str.split(','))
cathode_index = parse_index_tuple(elec_config.get('cathode_index'))
anode_index = parse_index_tuple(elec_config.get('anode_index'))
# --- 結束：讀取設定檔 ---

# [Validation] 區塊（可選）
validation_enabled = False
validation_interval = 50
validation_tol_charge = 5e-4
validation_tol_energy_rel = 5e-4
validation_plugin_dir = ''
if config.has_section('Validation'):
    val_cfg = config['Validation']
    validation_enabled = val_cfg.getboolean('enable', fallback=False)
    validation_interval = val_cfg.getint('interval', fallback=50)
    validation_tol_charge = val_cfg.getfloat('tol_charge', fallback=5e-4)
    validation_tol_energy_rel = val_cfg.getfloat('tol_energy_rel', fallback=5e-4)
    validation_plugin_dir = val_cfg.get('plugin_dir', fallback='').strip()

# [Physics] 區塊（可選）
physics_iterations = 4
physics_enable_convergence = False
physics_conv_tol_energy_rel = 1e-6
physics_conv_tol_charge = 1e-6
physics_enforce_scale_each_iter = True
physics_verify_invariants = False
physics_invariants_tol_charge = 1e-6
if config.has_section('Physics'):
    phy = config['Physics']
    physics_iterations = phy.getint('iterations', fallback=4)
    physics_enable_convergence = phy.getboolean('enable_convergence', fallback=False)
    physics_conv_tol_energy_rel = phy.getfloat('convergence_tol_energy_rel', fallback=1e-6)
    physics_conv_tol_charge = phy.getfloat('convergence_tol_charge', fallback=1e-6)
    physics_enforce_scale_each_iter = phy.getboolean('enforce_analytic_scaling_each_iter', fallback=True)
    physics_verify_invariants = phy.getboolean('verify_invariants', fallback=False)
    physics_invariants_tol_charge = phy.getfloat('invariants_tol_charge', fallback=1e-6)


# set output path
if os.path.exists(outPath):
    shutil.rmtree(outPath)
strdir = outPath
os.mkdir(outPath)


#************************** download SAPT-FF force field files from github
from sapt_exclusions import *

# *********************************************************************
#                     Create MM system object
#**********************************************************************
MMsys=MM( pdb_list = pdb_list , residue_xml_list = residue_xml_list , ff_xml_list = ff_xml_list  )
MMsys.set_periodic_residue(True)
t1=datetime.now()
MMsys.set_platform(openmm_platform)
MMsys.initialize_electrodes( Voltage, cathode_identifier = cathode_index , anode_identifier = anode_index , chain=True , exclude_element=("H",)  )
MMsys.initialize_electrolyte(Natom_cutoff=100)
MMsys.generate_exclusions( flag_SAPT_FF_exclusions = True )

# 🚀 Plugin: 加載並配置 ElectrodeChargeForce
if USE_PLUGIN:
    print("\n" + "="*60)
    print("🔥 Configuring ElectrodeChargePlugin...")
    print("="*60)
    import os
    try:
        plugin_dir_candidates = []
        if validation_plugin_dir:
            plugin_dir_candidates.append(validation_plugin_dir)
        conda_prefix = os.environ.get('CONDA_PREFIX', '/home/andy/miniforge3/envs/cuda')
        plugin_dir_candidates.append(os.path.join(conda_prefix, 'lib', 'plugins'))
        plugin_loaded = False
        for pdir in plugin_dir_candidates:
            if pdir and os.path.exists(pdir):
                Platform.loadPluginsFromDirectory(pdir)
                print(f"✓ Loaded plugins from: {pdir}")
                plugin_loaded = True
                break
        if not plugin_loaded:
            print("⚠️  No plugin directory found. Proceeding without plugin.")
            USE_PLUGIN = False
        else:
            print("✓ Plugin registry updated. Expecting ElectrodeChargeForce to be present in System.")
    except Exception as e:
        print(f"❌ Failed to load plugin: {e}")
        USE_PLUGIN = False
    print("="*60 + "\n")

    # 實例化並加入 ElectrodeChargeForce（使 plugin 真正生效）
    try:
        import electrodecharge as ec
    except Exception as e:
        print(f"❌ Failed to import electrodecharge Python wrapper: {e}")
        print("   Disabling plugin and falling back to Python Poisson solver.")
        USE_PLUGIN = False

    if USE_PLUGIN:
        try:
            force = ec.ElectrodeChargeForce()
            # 收集電極原子索引
            cathode_indices = [atom.atom_index for atom in MMsys.Cathode.electrode_atoms]
            anode_indices = [atom.atom_index for atom in MMsys.Anode.electrode_atoms]
            # 設定電極與電壓（正負由公式決定，這裡傳入幅值）
            force.setCathode(cathode_indices, abs(Voltage))
            force.setAnode(anode_indices, abs(Voltage))
            # 幾何與數值
            force.setSmallThreshold(MMsys.small_threshold)
            force.setCellGap(MMsys.Lgap)
            force.setCellLength(MMsys.Lcell)
            force.setNumIterations(physics_iterations)

            # 傳入導體資料（若有）
            if len(MMsys.Conductor_list) > 0:
                conductor_indices = []
                conductor_normals = []   # flattened nx,ny,nz per atom
                conductor_areas = []
                contact_indices = []     # one per conductor
                contact_normals = []     # flattened nx,ny,nz per conductor
                geometries = []          # per conductor
                types = []               # 0=buckyball, 1=nanotube

                for Conductor in MMsys.Conductor_list:
                    # per-atom data
                    for atom in Conductor.electrode_atoms:
                        conductor_indices.append(atom.atom_index)
                        conductor_normals.extend([atom.nx, atom.ny, atom.nz])
                        conductor_areas.append(Conductor.area_atom)
                    # contact point
                    ci = Conductor.Electrode_contact_atom.atom_index
                    cn = Conductor.Electrode_contact_atom
                    contact_indices.append(ci)
                    contact_normals.extend([cn.nx, cn.ny, cn.nz])
                    # geometry and type
                    cname = type(Conductor).__name__
                    if cname == 'Buckyball_Virtual':
                        geometries.append(Conductor.dr_center_contact**2)
                        types.append(0)
                    elif cname == 'Nanotube_Virtual':
                        geometries.append(Conductor.dr_center_contact * Conductor.length / 2.0)
                        types.append(1)
                    else:
                        continue

                try:
                    force.setConductorData(conductor_indices,
                                           conductor_normals,
                                           conductor_areas,
                                           contact_indices,
                                           contact_normals,
                                           geometries,
                                           types)
                    print(f"✓ Passed {len(conductor_indices)} conductor atoms across {len(contact_indices)} conductors to plugin")
                except Exception as e:
                    print(f"⚠️  Passing conductor data to plugin failed: {e}")
            # 指定 force group 到最後一組，避免覆蓋
            force.setForceGroup(MMsys.system.getNumForces())
            # 加入系統
            MMsys.system.addForce(force)
            # 重新初始化 context 以使新增 Force 生效
            state_tmp = MMsys.simmd.context.getState(getPositions=True, getVelocities=True)
            MMsys.simmd.context.reinitialize()
            MMsys.simmd.context.setPositions(state_tmp.getPositions())
            if state_tmp.getVelocities() is not None:
                MMsys.simmd.context.setVelocities(state_tmp.getVelocities())
            print("✓ ElectrodeChargeForce attached to System and context reinitialized.")
        except Exception as e:
            print(f"❌ Failed to attach ElectrodeChargeForce: {e}")
            print("   Disabling plugin and falling back to Python Poisson solver.")
            USE_PLUGIN = False

# ===== A/B 參考系統（僅在 plugin 且啟用驗證時建立） =====
MMsys_ref = None
if USE_PLUGIN and validation_enabled:
    print("[Validation] 建立 reference 系統以進行 A/B 比對（Python Poisson）...")
    MMsys_ref = MM( pdb_list = pdb_list , residue_xml_list = residue_xml_list , ff_xml_list = ff_xml_list )
    MMsys_ref.set_periodic_residue(True)
    MMsys_ref.set_platform(openmm_platform)
    MMsys_ref.initialize_electrodes( Voltage, cathode_identifier = cathode_index , anode_identifier = anode_index , chain=True , exclude_element=("H",) )
    MMsys_ref.initialize_electrolyte(Natom_cutoff=100)
    MMsys_ref.generate_exclusions( flag_SAPT_FF_exclusions = True )
    # 對齊 positions 與 box（確保與主系統完全一致）
    state0 = MMsys.simmd.context.getState(getPositions=True)
    MMsys_ref.simmd.context.setPositions(state0.getPositions())

    # 收集電極索引順序（Cathode, Conductors, Anode），以便比較每原子電荷
    def collect_electrode_indices(mm_obj):
        idx = []
        for atom in mm_obj.Cathode.electrode_atoms:
            idx.append(atom.atom_index)
        for Conductor in mm_obj.Conductor_list:
            for atom in Conductor.electrode_atoms:
                idx.append(atom.atom_index)
        for atom in mm_obj.Anode.electrode_atoms:
            idx.append(atom.atom_index)
        return idx

    electrode_index_order = collect_electrode_indices(MMsys)

    def read_charges_for_indices(mm_obj, indices):
        vals = []
        for ai in indices:
            (q_i, _, _) = mm_obj.nbondedForce.getParticleParameters(ai)
            vals.append(q_i._value)
        return vals

    def energy_components(mm_obj):
        comps = []
        for j in range(mm_obj.system.getNumForces()):
            e = mm_obj.simmd.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()
            comps.append(e)
        total = mm_obj.simmd.context.getState(getEnergy=True).getPotentialEnergy()
        return total, comps

state = MMsys.simmd.context.getState(getEnergy=True,getForces=True,getVelocities=False,getPositions=True)
positions = state.getPositions()

print("--- Initial State (Before Simulation) ---")
print("Initial Kinetic Energy: " + str(state.getKineticEnergy()))
print("Initial Potential Energy: " + str(state.getPotentialEnergy()))

# 印出所有 force groups 的初始能量
for j in range(MMsys.system.getNumForces()):
    f = MMsys.system.getForce(j)
    force_name = type(f).__name__.replace("Force", "")
    group_energy = MMsys.simmd.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()
    print(f"Force Group {j} ({type(f)}): {group_energy}")

print("------------------------------------------")


# write initial pdb with Drudes, and setup trajectory output
PDBFile.writeFile(MMsys.simmd.topology, positions, open(strdir + 'start_drudes.pdb', 'w'))

if simulation_type == "MC_equil":
    celldim = MMsys.simmd.topology.getUnitCellDimensions()
    MMsys.MC = MC_parameters(  MMsys.temperature , celldim , electrode_move="Anode" , pressure = 1.0*bar , barofreq = 100 , shiftscale = 0.2 )
    trajectory_file_name = strdir + 'equil_MC.dcd'
else :
    trajectory_file_name = strdir + 'FV_NVT.dcd'

# DCD 軌跡檔是一定要寫入的
MMsys.set_trajectory_output( trajectory_file_name , freq_traj_output_ps * 1000 )


# ########################################################################
# ######################  MAIN SIMULATION LOOP ###########################
# ########################################################################
# 🔥 Linus 重構: 統一主循環，土法煉鋼最快
# 
# 好品味原則:
# 1. 一個主循環，不是兩個（消除嵌套循環）
# 2. Warm-start 邏輯只出現一次
# 3. Legacy print: 直接 print()，不需要 class（土法煉鋼）
# 4. Efficient: 只管理檔案 handles（必要的狀態管理）
# ########################################################################

print(f"\nStarting simulation ({simulation_type}) for {simulation_time_ns} ns...")
print(f"🔥 Logging mode set to: {logging_mode}")
t_start = datetime.now()

# 🔥 準備 logging（只有 efficient 需要管理檔案）
legacy_print_interval = 0
legacy_frame_count = 0
chargeFile = None
componentsFile = None

if logging_mode == 'legacy_print':
    print("--- Running in 'legacy_print' mode. This will be VERY SLOW. ---")
    print("    Tip: Use `python run_openMM.py > output.log 2>&1` to redirect output")
    # Legacy 每多少個 charge updates print 一次
    legacy_print_interval = int(freq_traj_output_ps * 1000 / freq_charge_update_fs)
    
elif logging_mode == 'efficient':
    print(f"--- Running in 'efficient' mode. Logging to {strdir}*.log ---")
    
    # StateDataReporter (標準能量項)
    log_freq = freq_traj_output_ps * 1000
    MMsys.simmd.reporters.append(StateDataReporter(
        strdir + 'energy.log', log_freq, step=True,
        potentialEnergy=True, kineticEnergy=True, totalEnergy=True,
        temperature=True, volume=True, density=True, speed=True
    ))
    
    # Charges 和 components 的記錄頻率
    charge_log_interval = int((freq_traj_output_ps * 1000) / freq_charge_update_fs)
    if charge_log_interval == 0: charge_log_interval = 1
    
    # 打開檔案
    if write_charges:
        chargeFile = open(strdir + 'charges.dat', 'w')
        print(f"    Writing charges every {charge_log_interval} updates")
    
    if write_components:
        componentsFile = open(strdir + 'components.log', 'w')
        # 寫入 header
        header = "# Step\t"
        for j in range(MMsys.system.getNumForces()):
            f = MMsys.system.getForce(j)
            force_name = type(f).__name__.replace("Force", "")
            header += f"Group{j}_{force_name}\t"
        componentsFile.write(header + "\n")
        print(f"    Writing energy components every {charge_log_interval} updates (Performance heavy!)")
else:
    print(f"❌ Error: Unknown logging_mode '{logging_mode}' in config.ini")
    print("   Valid options: 'efficient', 'legacy_print'")
    sys.exit(1)


# ########################################################################
# 🔥 統一主循環 (Monte Carlo 或 Constant Voltage)
# ########################################################################

#********** Monte Carlo Simulation ********
if simulation_type == "MC_equil":
    n_total_steps = int((simulation_time_ns * 1000 * 1000) / MMsys.MC.barofreq)
    print(f"Running MC equilibration for {n_total_steps} steps.")
    
    for i in range(n_total_steps):
        MMsys.MC_Barostat_step()

#********** Constant Voltage Simulation ****
elif simulation_type == "Constant_V":
    # 計算總共需要多少次電荷更新
    n_total_updates = int((simulation_time_ns * 1000 * 1000) / freq_charge_update_fs)
    steps_per_charge_update = freq_charge_update_fs
    
    print(f"Running Constant Voltage simulation for {n_total_updates} charge updates.")
    print(f" (MD timestep: {steps_per_charge_update} fs per charge update)")
    
    # 🔥 初始化 warm-start 狀態
    warmstart_activated = False
    
    # 🔥 統一主循環：無嵌套，零複雜度
    for i in range(n_total_updates):
        
        # 1️⃣ Warm-start 判斷 (只出現這一次!)
        current_time_ns = (i * freq_charge_update_fs) / 1e6
        use_warmstart_now, warmstart_activated = should_use_warmstart(
            i,                      # 當前步驟號
            current_time_ns,        # 當前模擬時間 (ns)
            warmstart_activated,    # 當前激活狀態
            enable_warmstart,       # 來自 config.ini
            warmstart_after_ns,     # 來自 config.ini
            warmstart_after_frames  # 來自 config.ini
        )
        
        # 2️⃣ Poisson solver (更新電極電荷)
        if USE_PLUGIN:
            # Plugin 版：Force 內核自動處理；若啟用驗證，對照參考系統
            if validation_enabled and (i % validation_interval == 0) and MMsys_ref is not None:
                # 讓參考系統用 Python Poisson 更新
                MMsys_ref.Poisson_solver_fixed_voltage(
                    Niterations=physics_iterations,
                    enable_convergence=physics_enable_convergence,
                    convergence_tol_energy_rel=physics_conv_tol_energy_rel,
                    convergence_tol_charge=physics_conv_tol_charge,
                    enforce_analytic_scaling_each_iter=physics_enforce_scale_each_iter,
                    verify_invariants=physics_verify_invariants,
                    invariants_tol_charge=physics_invariants_tol_charge
                )
                # 對齊主系統電荷（由 plugin 內核更新），讀出兩邊電荷並比較
                q_ref = read_charges_for_indices(MMsys_ref, electrode_index_order)
                q_plug = read_charges_for_indices(MMsys, electrode_index_order)
                # 計算每原子最大誤差
                max_abs_err = max(abs(a-b) for a,b in zip(q_ref, q_plug)) if q_ref else 0.0
                if max_abs_err > validation_tol_charge:
                    print(f"❌ Validation failed: charge max|Δ|={max_abs_err:.3e} > tol={validation_tol_charge:.3e}")
                    # 比較能量做輔助訊息
                    Et_ref, comps_ref = energy_components(MMsys_ref)
                    Et_plg, comps_plg = energy_components(MMsys)
                    def rel(x,y):
                        xv = x._value if hasattr(x, '_value') else float(x)
                        yv = y._value if hasattr(y, '_value') else float(y)
                        denom = max(1.0, abs(yv))
                        return abs(xv - yv)/denom
                    rel_total = rel(Et_ref, Et_plg)
                    worst_rel = 0.0
                    for a,b in zip(comps_ref, comps_plg):
                        worst_rel = max(worst_rel, rel(a,b))
                    print(f"   totalE relΔ={rel_total:.3e}, worst group relΔ={worst_rel:.3e}")
                    sys.exit(2)
        elif mm_version == 'cython':
            MMsys.Poisson_solver_fixed_voltage(
                Niterations=physics_iterations,
                use_warmstart_this_step=use_warmstart_now,
                verify_interval=verify_interval,
                enable_convergence=physics_enable_convergence,
                convergence_tol_energy_rel=physics_conv_tol_energy_rel,
                convergence_tol_charge=physics_conv_tol_charge,
                enforce_analytic_scaling_each_iter=physics_enforce_scale_each_iter,
                verify_invariants=physics_verify_invariants,
                invariants_tol_charge=physics_invariants_tol_charge
            )
        else:
            MMsys.Poisson_solver_fixed_voltage(
                Niterations=physics_iterations,
                enable_convergence=physics_enable_convergence,
                convergence_tol_energy_rel=physics_conv_tol_energy_rel,
                convergence_tol_charge=physics_conv_tol_charge,
                enforce_analytic_scaling_each_iter=physics_enforce_scale_each_iter,
                verify_invariants=physics_verify_invariants,
                invariants_tol_charge=physics_invariants_tol_charge
            )
        
        # 3️⃣ MD 步驟
        MMsys.simmd.step(steps_per_charge_update)
        
        # 4️⃣ Logging (土法煉鋼：直接 if，不需要 fancy classes)
        
        # Legacy print: 直接 print()，讓 OS 處理重定向
        if logging_mode == 'legacy_print' and i % legacy_print_interval == 0:
            state = MMsys.simmd.context.getState(getEnergy=True, getForces=True, 
                                                  getVelocities=False, getPositions=True)
            print(f"\n--- Legacy Frame {legacy_frame_count} (Time: {current_time_ns:.3f} ns) ---")
            print(f'Iteration: {legacy_frame_count}')
            print(f'Kinetic Energy: {state.getKineticEnergy()}')
            print(f'Potential Energy: {state.getPotentialEnergy()}')
            
            # 印出所有 force groups
            for j in range(MMsys.system.getNumForces()):
                f = MMsys.system.getForce(j)
                group_energy = MMsys.simmd.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()
                print(f'  {type(f)}: {group_energy}')
            
            legacy_frame_count += 1
        
        # Efficient: 寫入檔案
        elif logging_mode == 'efficient' and i % charge_log_interval == 0:
            if chargeFile:
                MMsys.write_electrode_charges(chargeFile)
            
            if componentsFile:
                data_line = f"{i * steps_per_charge_update}\t"
                for j in range(MMsys.system.getNumForces()):
                    group_energy = MMsys.simmd.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()
                    data_line += f"{group_energy._value}\t"
                componentsFile.write(data_line + "\n")

else:
    print('simulation type not recognized ...')
    sys.exit()

# ########################################################################
# ######################   模擬結束  #####################################
# ########################################################################

t_end = datetime.now()
print(f"\nSimulation finished in {t_end - t_start}.")

# 清理資源（只關閉檔案，print 不需要清理）
if chargeFile:
    chargeFile.close()
if componentsFile:
    componentsFile.close()

print('done!')
sys.exit()