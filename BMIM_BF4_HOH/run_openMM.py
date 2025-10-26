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
if mm_version == 'cython':
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
elif mm_version == 'optimized':
    print("⚡ Loading NumPy-optimized MM classes (1.5-2x speedup expected)")
    if enable_warmstart: print("⚠️  Warning: Warm Start only supported in 'cython' version, ignoring parameter")
    enable_warmstart = False
    from MM_classes_OPTIMIZED import *
    from Fixed_Voltage_routines_OPTIMIZED import *
else: # 'original'
    print("📦 Loading original MM classes (baseline performance)")
    if enable_warmstart: print("⚠️  Warning: Warm Start only supported in 'cython' version, ignoring parameter")
    enable_warmstart = False
    from MM_classes import *
    from Fixed_Voltage_routines import *

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


# set output path
if os.path.exists(outPath):
    shutil.rmtree(outPath)
strdir = outPath
os.mkdir(outPath)

# --- 檔案管理器 ---
# 這些檔案主要由 'efficient' 模式使用
chargeFile = open(strdir + 'charges.dat', 'w')
componentsFile = None
if write_components and logging_mode == 'efficient':
    componentsFile = open(strdir + 'components.log', 'w')


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

state = MMsys.simmd.context.getState(getEnergy=True,getForces=True,getVelocities=False,getPositions=True)
positions = state.getPositions()

print("--- Initial State (Before Simulation) ---")
print("Initial Kinetic Energy: " + str(state.getKineticEnergy()))
print("Initial Potential Energy: " + str(state.getPotentialEnergy()))

# 準備標頭和數據 (用於 'efficient' 模式的 components.log)
header = "# Step\t"
data = "0\t"
for j in range(MMsys.system.getNumForces()):
    f = MMsys.system.getForce(j)
    force_name = type(f).__name__.replace("Force", "")
    group_energy = MMsys.simmd.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()
    print(f"Force Group {j} ({type(f)}): " + str(group_energy))
    header += f"Group{j}_{force_name}\t"
    data += f"{group_energy._value}\t"

if componentsFile: # 只有在 'efficient' 且 'write_components' 時才寫入
    componentsFile.write(header + "\n")
    componentsFile.write(data + "\n")
    componentsFile.flush()
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

print(f"\nStarting simulation ({simulation_type}) for {simulation_time_ns} ns...")
print(f"🔥 Logging mode set to: {logging_mode}")
t_start = datetime.now()

# 🔥 NEW: 根據 logging_mode 選擇執行路徑
if logging_mode == 'legacy_print':
    
    # --- 這是您原始的、低效能的「列印到終端機」迴圈 ---
    print("--- WARNING: Running in 'legacy_print' mode. This will be VERY SLOW. ---")
    
    # 🔥 NEW: Track simulation progress for delayed warm start activation
    # (這是從您上傳的 run_openMM.py 複製過來的邏輯)
    current_frame = 0
    warmstart_activated = False

    for i in range( int(simulation_time_ns * 1000 / freq_traj_output_ps ) ):
        
        # 1. 列印所有狀態 (這是效能瓶頸)
        state = MMsys.simmd.context.getState(getEnergy=True,getForces=True,getVelocities=False,getPositions=True)
        print(f"\n--- Legacy Frame {i} (Time: {i * freq_traj_output_ps / 1000.0} ns) ---")
        print('Iteration: ', i)
        print('Kinetic Energy: ' + str(state.getKineticEnergy()))
        print('Potential Energy: ' + str(state.getPotentialEnergy()))
        for j in range(MMsys.system.getNumForces()):
            f = MMsys.system.getForce(j)
            print(f'  {type(f)}: ' + str(MMsys.simmd.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()))

        # 2. 檢查 Warm Start (這是從您上傳的 run_openMM.py 複製過來的邏輯)
        current_time_ns = i * freq_traj_output_ps / 1000.0
        use_warmstart_now = enable_warmstart
        
        if enable_warmstart and not warmstart_activated:
            if warmstart_after_ns > 0:
                if current_time_ns >= warmstart_after_ns:
                    warmstart_activated = True
                    print(f"🚀 WARM START ACTIVATED at {current_time_ns:.2f} ns (frame {i})")
                else:
                    use_warmstart_now = False
            elif warmstart_after_frames > 0:
                if i >= warmstart_after_frames:
                    warmstart_activated = True
                    print(f"🚀 WARM START ACTIVATED at frame {i} ({current_time_ns:.2f} ns)")
                else:
                    use_warmstart_now = False
            else:
                warmstart_activated = True

        # 3. 執行 "一個區塊" (chunk) 的模擬
        #********** Monte Carlo Simulation ********
        if simulation_type == "MC_equil":
            for j in range( int(freq_traj_output_ps * 1000 / MMsys.MC.barofreq) ):
                MMsys.MC_Barostat_step()

        #********** Constant Voltage Simulation ****
        elif simulation_type == "Constant_V":
            # 這是 "一個區塊" 中有多少個電荷更新步驟
            steps_in_this_chunk = int(freq_traj_output_ps * 1000 / freq_charge_update_fs )
            
            for j in range( steps_in_this_chunk ):
                if mm_version == 'cython':
                    MMsys.Poisson_solver_fixed_voltage( 
                        Niterations=4,
                        enable_warmstart=use_warmstart_now,
                        verify_interval=verify_interval
                    )
                else:
                    MMsys.Poisson_solver_fixed_voltage( Niterations=4 )
                MMsys.simmd.step( freq_charge_update_fs )
            
            if write_charges : # Legacy 模式也支援寫入 charge (雖然頻率不同)
                MMsys.write_electrode_charges( chargeFile )

        else:
            print('simulation type not recognized ...')
            sys.exit()
        
        current_frame = i
    
elif logging_mode == 'efficient':

    # --- 這是我們新建的、高效能的「寫入到檔案」迴圈 ---
    print(f"--- Running in 'efficient' mode. Logging to {strdir}*.log ---")

    # 1. 新增 StateDataReporter (負責 energy.log)
    log_freq = freq_traj_output_ps * 1000
    MMsys.simmd.reporters.append(StateDataReporter(strdir + 'energy.log', log_freq, step=True,
                                               potentialEnergy=True, kineticEnergy=True, totalEnergy=True,
                                               temperature=True, volume=True, density=True, speed=True))

    # 2. 建立新的、正確的主模擬迴圈
    #********** Monte Carlo Simulation ********
    if simulation_type == "MC_equil":
        n_total_steps = int((simulation_time_ns * 1000 * 1000) / MMsys.MC.barofreq)
        print(f"Running MC equilibration for {n_total_steps} steps.")
        for _ in range(n_total_steps):
            MMsys.MC_Barostat_step()

    #********** Constant Voltage Simulation ****
    elif simulation_type == "Constant_V":
        n_total_updates = int((simulation_time_ns * 1000 * 1000) / freq_charge_update_fs)
        steps_per_charge_update = freq_charge_update_fs
        
        charge_write_interval = 0
        if write_charges:
            charge_write_interval = int((freq_traj_output_ps * 1000) / freq_charge_update_fs)
            if charge_write_interval == 0: charge_write_interval = 1
        
        components_write_interval = 0
        if write_components and componentsFile:
            components_write_interval = int((freq_traj_output_ps * 1000) / freq_charge_update_fs)
            if components_write_interval == 0: components_write_interval = 1
        
        print(f"Running Constant Voltage simulation for {n_total_updates} charge updates.")
        print(f" (Simulation step size: {steps_per_charge_update} fs per update)")
        if write_charges:
            print(f" Writing charges to {strdir}charges.dat every {charge_write_interval} updates.")
        if write_components and componentsFile:
            print(f" Writing energy components to {strdir}components.log every {components_write_interval} updates (Warning: Performance heavy).")

        # --- 整合 Warm Start 邏輯 ---
        warmstart_activated = False
        warmstart_activation_step = 0
        if enable_warmstart and mm_version == 'cython':
            if warmstart_after_ns > 0:
                steps_per_ns = (1000 * 1000) / freq_charge_update_fs
                warmstart_activation_step = int(warmstart_after_ns * steps_per_ns)
                print(f"🚀 Warm Start will be enabled after {warmstart_activation_step} charge update steps (approx {warmstart_after_ns} ns).")
            elif warmstart_after_frames > 0:
                # 這裡的 frame 是指 'charge update' 步驟
                warmstart_activation_step = warmstart_after_frames
                print(f"🚀 Warm Start will be enabled after {warmstart_activation_step} charge update steps.")
            else:
                print("🚀 Warm Start enabled immediately.")
                warmstart_activated = True
        
        # 這是新的主迴圈
        for i in range(n_total_updates):
            
            use_warmstart_now = enable_warmstart
            if enable_warmstart and not warmstart_activated:
                if i >= warmstart_activation_step:
                    warmstart_activated = True
                    current_time_ns = (i * freq_charge_update_fs) / (1000.0 * 1000.0)
                    print(f"\n{'='*80}")
                    print(f"🚀 WARM START ACTIVATED at step {i} (approx {current_time_ns:.2f} ns)")
                    print(f"{'='*80}\n")
                else:
                    use_warmstart_now = False

            # 執行一次電荷更新 + MD 步驟
            if mm_version == 'cython':
                MMsys.Poisson_solver_fixed_voltage( 
                    Niterations=4,
                    enable_warmstart=use_warmstart_now,
                    verify_interval=verify_interval
                )
            else:
                MMsys.Poisson_solver_fixed_voltage( Niterations=4 )
            
            MMsys.simmd.step( steps_per_charge_update )
            
            # 根據設定的頻率寫入電荷
            if write_charges and (i + 1) % charge_write_interval == 0:
                MMsys.write_electrode_charges( chargeFile )
                
            # 根據設定的頻率，手動寫入所有能量分項
            if componentsFile and (i + 1) % components_write_interval == 0:
                data_line = f"{(i+1) * steps_per_charge_update}\t" # 寫入當前 time step
                for j in range(MMsys.system.getNumForces()):
                    group_energy = MMsys.simmd.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()
                    data_line += f"{group_energy._value}\t"
                componentsFile.write(data_line + "\n")

    else:
        print('simulation type not recognized ...')
        sys.exit()

else:
    print(f"❌ Error: Unknown logging_mode '{logging_mode}' in config.ini")
    print("   Valid options: 'efficient', 'legacy_print'")
    sys.exit(1)

# ########################################################################
# ######################   效能修正結束  #########################
# ########################################################################

t_end = datetime.now()
print(f"Simulation finished in {t_end - t_start}.")

# 關閉檔案
chargeFile.close()
if componentsFile:
    componentsFile.close()

print('done!')
sys.exit()