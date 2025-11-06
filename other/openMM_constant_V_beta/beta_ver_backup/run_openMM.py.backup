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
import copy # For deepcopying force objects

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
    """
    if not config_enable_warmstart:
        return False, False
    if warmstart_activated:
        return True, True
    if config_warmstart_after_ns > 0:
        if current_time_ns >= config_warmstart_after_ns:
            print(f"\n{'='*80}")
            print(f"🚀 WARM START ACTIVATED at {current_time_ns:.2f} ns (frame {i_frame})")
            print(f"{'='*80}\n")
            return True, True
        else:
            return False, False
    elif config_warmstart_after_frames > 0:
        if i_frame >= config_warmstart_after_frames:
            print(f"\n{'='*80}")
            print(f"🚀 WARM START ACTIVATED at frame {i_frame} ({current_time_ns:.2f} ns)")
            print(f"{'='*80}\n")
            return True, True
        else:
            return False, False
    else:
        return True, True

# ============================================================
# 🔥 Linus 重構: 獨立、乾淨的 Python 驗證函式
# ============================================================
def run_python_validation_step(mm_sys, physics_params):
    """
    在一個隔離的環境中執行一次 Python 版本的 Poisson Solver，用於 A/B 驗證。
    它接收主 `MMsys` 物件，但操作一個臨時的 force object，避免污染原始狀態。
    """
    # 1. 創建一個 throw-away 的 MM object 來執行 Python 演算法
    #    這比複製整個 context 更乾淨、更安全
    validation_solver = MM(pdb_list=[], residue_xml_list=[], ff_xml_list=[])
    
    # 2. 將主系統的關鍵狀態複製到這個隔離的 solver
    validation_solver.simmd = mm_sys.simmd # 共享 context, positions, etc.
    validation_solver.nbondedForce = copy.deepcopy(mm_sys.nbondedForce) # 關鍵：操作一個副本！
    validation_solver.Cathode = mm_sys.Cathode
    validation_solver.Anode = mm_sys.Anode
    validation_solver.Conductor_list = mm_sys.Conductor_list
    validation_solver.Lcell = mm_sys.Lcell
    validation_solver.Lgap = mm_sys.Lgap
    validation_solver.electrolyte_atom_indices = mm_sys.electrolyte_atom_indices
    validation_solver.small_threshold = mm_sys.small_threshold
    validation_solver.QMMM = mm_sys.QMMM # 確保 QMMM 旗標一致

    # 3. 執行與原始 Python 版本完全相同的 Poisson Solver
    #    使用字典來安全地傳遞參數
    solver_args = {
        'Niterations': physics_params.getint('iterations', 4),
        'enable_convergence': physics_params.getboolean('enable_convergence', False),
        'convergence_tol_energy_rel': physics_params.getfloat('convergence_tol_energy_rel', 1e-6),
        'convergence_tol_charge': physics_params.getfloat('convergence_tol_charge', 1e-6),
        'enforce_analytic_scaling_each_iter': physics_params.getboolean('enforce_analytic_scaling_each_iter', True),
        'verify_invariants': physics_params.getboolean('verify_invariants', False),
        'invariants_tol_charge': physics_params.getfloat('invariants_tol_charge', 1e-6)
    }
    validation_solver.Poisson_solver_fixed_voltage(**solver_args)

    # 4. 從被修改的副本中讀取電荷，作為黃金標準
    ref_charges = []
    for i in range(validation_solver.nbondedForce.getNumParticles()):
        q, _, _ = validation_solver.nbondedForce.getParticleParameters(i)
        ref_charges.append(q._value)
    
    return ref_charges

# --- 開始：讀取設定檔 ---
parser = argparse.ArgumentParser(description="Run OpenMM Fixed-Voltage MD Simulation")
parser.add_argument('-c', '--config', default='config.ini', help='Path to the configuration file (default: config.ini)')
args = parser.parse_args()

config = configparser.ConfigParser()
config_path = args.config
if not os.path.exists(config_path):
    print(f"Error: Configuration file not found at {config_path}")
    sys.exit(1)
config.read(config_path)

# [Simulation] 區塊
sim_config = config['Simulation']
simulation_time_ns = sim_config.getint('simulation_time_ns')
freq_charge_update_fs = sim_config.getint('freq_charge_update_fs')
freq_traj_output_ps = sim_config.getint('freq_traj_output_ps')
simulation_type = sim_config.get('simulation_type')
openmm_platform = sim_config.get('platform')
Voltage = sim_config.getfloat('voltage')
mm_version = sim_config.get('mm_version', 'original').lower()
enable_warmstart = sim_config.getboolean('enable_warmstart', fallback=True)
verify_interval = sim_config.getint('verify_interval', fallback=100)
warmstart_after_ns = sim_config.getfloat('warmstart_after_ns', fallback=0.0)
warmstart_after_frames = sim_config.getint('warmstart_after_frames', fallback=0)
logging_mode = sim_config.get('logging_mode', 'efficient').lower()
write_charges = sim_config.getboolean('write_charges', fallback=False)
write_components = sim_config.getboolean('write_components', fallback=False)


# Import appropriate MM_classes based on version
if mm_version == 'plugin':
    print("🚀 Loading OpenMM Plugin (C++/CUDA implementation)")
    from MM_classes import *
    from Fixed_Voltage_routines import *
    USE_PLUGIN = True
elif mm_version == 'cython':
    print("🔥 Loading Cython-optimized MM classes (2-5x speedup expected)")
    from MM_classes_CYTHON import *
    from Fixed_Voltage_routines_CYTHON import *
    USE_PLUGIN = False
elif mm_version == 'optimized':
    print("⚡ Loading NumPy-optimized MM classes (1.5-2x speedup expected)")
    from MM_classes_OPTIMIZED import *
    from Fixed_Voltage_routines_OPTIMIZED import *
    USE_PLUGIN = False
else: # 'original'
    print("📦 Loading original MM classes (baseline performance)")
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

# [Validation] 區塊
validation_config = config['Validation'] if config.has_section('Validation') else {}
validation_enabled = validation_config.getboolean('enable', fallback=False)
validation_interval = validation_config.getint('interval', fallback=50)
validation_tol_charge = validation_config.getfloat('tol_charge', fallback=5e-4)
validation_plugin_dir = validation_config.get('plugin_dir', fallback='').strip()

# [Physics] 區塊
physics_config = config['Physics'] if config.has_section('Physics') else {}
physics_iterations = physics_config.getint('iterations', fallback=4)

# set output path
if os.path.exists(outPath):
    shutil.rmtree(outPath)
strdir = outPath
os.mkdir(outPath)

from sapt_exclusions import *

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

    try:
        import electrodecharge as ec
    except Exception as e:
        print(f"❌ Failed to import electrodecharge Python wrapper: {e}")
        print("   Disabling plugin and falling back to Python Poisson solver.")
        USE_PLUGIN = False

    if USE_PLUGIN:
        try:
            force = ec.ElectrodeChargeForce()
            cathode_indices = [atom.atom_index for atom in MMsys.Cathode.electrode_atoms]
            anode_indices = [atom.atom_index for atom in MMsys.Anode.electrode_atoms]
            force.setCathode(cathode_indices, abs(Voltage))
            force.setAnode(anode_indices, abs(Voltage))
            force.setSmallThreshold(MMsys.small_threshold)
            force.setCellGap(MMsys.Lgap)
            force.setCellLength(MMsys.Lcell)
            force.setNumIterations(physics_iterations)

            if len(MMsys.Conductor_list) > 0:
                c_indices, c_normals, c_areas, c_contacts, c_contact_normals, c_geoms, c_atom_ids, c_atom_counts = [],[],[],[],[],[],[],[]
                for i, c in enumerate(MMsys.Conductor_list):
                    for atom in c.electrode_atoms:
                        c_indices.append(atom.atom_index)
                        c_normals.extend([atom.nx, atom.ny, atom.nz])
                        c_areas.append(c.area_atom)
                        c_atom_ids.append(i)
                    c_atom_counts.append(c.Natoms)
                    c_contacts.append(c.Electrode_contact_atom.atom_index)
                    c_contact_normals.extend([c.Electrode_contact_atom.nx, c.Electrode_contact_atom.ny, c.Electrode_contact_atom.nz])
                    # Good taste: geometry factor encodes conductor type
                    # Buckyball: dr^2, Nanotube: dr*L/2 - no type enum needed
                    cname = type(c).__name__
                    if cname == 'Buckyball_Virtual':
                        c_geoms.append(c.dr_center_contact**2)
                    elif cname == 'Nanotube_Virtual':
                        c_geoms.append(c.dr_center_contact * c.length / 2.0)
                force.setConductorData(c_indices, c_normals, c_areas, c_contacts, c_contact_normals, c_geoms, c_atom_ids, c_atom_counts)
                print(f"✓ Passed {len(c_indices)} conductor atoms across {len(c_contacts)} conductors to plugin")

            force.setForceGroup(MMsys.system.getNumForces())
            MMsys.system.addForce(force)
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

state = MMsys.simmd.context.getState(getEnergy=True,getForces=True,getVelocities=False,getPositions=True)
positions = state.getPositions()

print("--- Initial State (Before Simulation) ---")
print("Initial Kinetic Energy: " + str(state.getKineticEnergy()))
print("Initial Potential Energy: " + str(state.getPotentialEnergy()))

for j in range(MMsys.system.getNumForces()):
    f = MMsys.system.getForce(j)
    force_name = type(f).__name__.replace("Force", "")
    group_energy = MMsys.simmd.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()
    print(f"Force Group {j} ({type(f)}): {group_energy}")

print("------------------------------------------")

PDBFile.writeFile(MMsys.simmd.topology, positions, open(strdir + 'start_drudes.pdb', 'w'))

trajectory_file_name = strdir + 'FV_NVT.dcd' if simulation_type != "MC_equil" else strdir + 'equil_MC.dcd'
MMsys.set_trajectory_output( trajectory_file_name , freq_traj_output_ps * 1000 )

print(f"\nStarting simulation ({simulation_type}) for {simulation_time_ns} ns...")
print(f"🔥 Logging mode set to: {logging_mode}")
t_start = datetime.now()

legacy_print_interval = 0
chargeFile = None
componentsFile = None

if logging_mode == 'legacy_print':
    print("--- Running in 'legacy_print' mode. This will be VERY SLOW. ---")
    legacy_print_interval = int(freq_traj_output_ps * 1000 / freq_charge_update_fs)
elif logging_mode == 'efficient':
    print(f"--- Running in 'efficient' mode. Logging to {strdir}*.log ---")
    log_freq = freq_traj_output_ps * 1000
    MMsys.simmd.reporters.append(StateDataReporter(
        strdir + 'energy.log', log_freq, step=True,
        potentialEnergy=True, kineticEnergy=True, totalEnergy=True,
        temperature=True, volume=True, density=True, speed=True
    ))
    charge_log_interval = int((freq_traj_output_ps * 1000) / freq_charge_update_fs) or 1
    if write_charges:
        chargeFile = open(strdir + 'charges.dat', 'w')
        print(f"    Writing charges every {charge_log_interval} updates")
    if write_components:
        componentsFile = open(strdir + 'components.log', 'w')
        header = "# Step\t"
        for j in range(MMsys.system.getNumForces()):
            f = MMsys.system.getForce(j)
            header += f"{type(f).__name__.replace('Force', '')}\t"
        componentsFile.write(header + "\n")
        print(f"    Writing energy components every {charge_log_interval} updates (Performance heavy!)")
else:
    print(f"❌ Error: Unknown logging_mode '{logging_mode}' in config.ini")
    sys.exit(1)

if simulation_type == "MC_equil":
    n_total_steps = int((simulation_time_ns * 1000 * 1000) / MMsys.MC.barofreq)
    print(f"Running MC equilibration for {n_total_steps} steps.")
    for i in range(n_total_steps):
        MMsys.MC_Barostat_step()
elif simulation_type == "Constant_V":
    n_total_updates = int((simulation_time_ns * 1000 * 1000) / freq_charge_update_fs)
    steps_per_charge_update = freq_charge_update_fs
    print(f"Running Constant Voltage simulation for {n_total_updates} charge updates.")
    print(f" (MD timestep: {steps_per_charge_update} fs per charge update)")
    
    warmstart_activated = False
    
    # Use a dictionary for solver arguments to keep the loop clean
    solver_args = {
        'Niterations': physics_iterations,
        'enable_convergence': config.getboolean('Physics', 'enable_convergence', fallback=False),
        'convergence_tol_energy_rel': config.getfloat('Physics', 'convergence_tol_energy_rel', fallback=1e-6),
        'convergence_tol_charge': config.getfloat('Physics', 'convergence_tol_charge', fallback=1e-6),
        'enforce_analytic_scaling_each_iter': config.getboolean('Physics', 'enforce_analytic_scaling_each_iter', fallback=True),
        'verify_invariants': config.getboolean('Physics', 'verify_invariants', fallback=False),
        'invariants_tol_charge': config.getfloat('Physics', 'invariants_tol_charge', fallback=1e-6)
    }

    for i in range(n_total_updates):
        current_time_ns = (i * freq_charge_update_fs) / 1e6
        use_warmstart_now, warmstart_activated = should_use_warmstart(
            i, current_time_ns, warmstart_activated,
            enable_warmstart, warmstart_after_ns, warmstart_after_frames
        )
        
        # 2️⃣ Poisson solver (update charges)
        if not USE_PLUGIN:
            if mm_version == 'cython':
                solver_args['use_warmstart_this_step'] = use_warmstart_now
                solver_args['verify_interval'] = verify_interval
            MMsys.Poisson_solver_fixed_voltage(**solver_args)

        # 3️⃣ MD Step
        # For plugin, charge update is handled by the Force kernel during the step.
        MMsys.simmd.step(steps_per_charge_update)
        
        # 4️⃣ Validation (A/B Comparison)
        if USE_PLUGIN and validation_enabled and (i % validation_interval == 0):
            # The plugin has already run implicitly during the step. 
            # Now, run the Python solver on a temporary state to get the reference charges.
            q_ref = run_python_validation_step(MMsys, config['Physics'])
            
            # Get the actual charges produced by the plugin from the main context
            q_plug = [MMsys.nbondedForce.getParticleParameters(i)[0]._value for i in range(MMsys.nbondedForce.getNumParticles())]
            
            max_abs_err = np.max(np.abs(np.array(q_ref) - np.array(q_plug)))
            if max_abs_err > validation_tol_charge:
                print(f"❌ Validation FAILED at step {i}: charge max|Δ|={max_abs_err:.3e} > tol={validation_tol_charge:.3e}")
                sys.exit(2)
            else:
                print(f"✓ Validation PASSED at step {i}: charge max|Δ|={max_abs_err:.3e}")

        # 5️⃣ Logging
        if logging_mode == 'legacy_print' and i % legacy_print_interval == 0:
            # This mode is slow and should be avoided. Kept for legacy purposes.
            pass 
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

t_end = datetime.now()
print(f"\nSimulation finished in {t_end - t_start}.")

if chargeFile:
    chargeFile.close()
if componentsFile:
    componentsFile.close()

print('done!')
sys.exit()