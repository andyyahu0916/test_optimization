from __future__ import print_function
import sys
sys.path.append('./lib/')
#********** OpenMM Drivers
from openmm.app import *
from openmm import *
from openmm.unit import *

#*********** Fixed Voltage routines will be imported based on MM version
#***************************
import numpy as np
import urllib.request
from sys import stdout
from time import gmtime, strftime
from datetime import datetime
import shutil
import os
import argparse
import configparser

sys.setrecursionlimit(2000)

# ============================================================
# 🔥 Linus Refactoring: Unified warm-start activation logic
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
    A single function to handle all warm-start activation logic.
    Returns: (use_warmstart_now, new_warmstart_activated_status)
    This function eliminates duplicated if/else blocks from the original code.
    Don't Repeat Yourself (DRY) is a fundamental principle of software engineering.
    """
    if not config_enable_warmstart:
        return False, False

    if warmstart_activated:
        return True, True

    if config_warmstart_after_ns > 0:
        if current_time_ns >= config_warmstart_after_ns:
            print(f"\n{'='*80}\n🚀 WARM START ACTIVATED at {current_time_ns:.2f} ns (frame {i_frame})\n{'='*80}\n")
            return True, True
        else:
            return False, False

    elif config_warmstart_after_frames > 0:
        if i_frame >= config_warmstart_after_frames:
            print(f"\n{'='*80}\n🚀 WARM START ACTIVATED at frame {i_frame} ({current_time_ns:.2f} ns)\n{'='*80}\n")
            return True, True
        else:
            return False, False

    else: # Immediate activation
        return True, True

# --- Config Parsing ---
parser = argparse.ArgumentParser(description="Run OpenMM Fixed-Voltage MD Simulation")
parser.add_argument('-c', '--config', default='config.ini', help='Path to the configuration file (default: config.ini)')
args = parser.parse_args()

config = configparser.ConfigParser()
config_path = args.config
if not os.path.exists(config_path):
    print(f"Error: Configuration file not found at {config_path}")
    sys.exit(1)
config.read(config_path)

# [Simulation]
sim_config = config['Simulation']
simulation_time_ns = sim_config.getfloat('simulation_time_ns')
freq_charge_update_fs = sim_config.getint('freq_charge_update_fs')
freq_traj_output_ps = sim_config.getint('freq_traj_output_ps')
simulation_type = sim_config.get('simulation_type')
openmm_platform = sim_config.get('platform')
Voltage = sim_config.getfloat('voltage')
mm_version = sim_config.get('mm_version', 'optimized').lower()

# Warm Start
enable_warmstart = sim_config.getboolean('enable_warmstart', fallback=False)
verify_interval = sim_config.getint('verify_interval', fallback=100)
warmstart_after_ns = sim_config.getfloat('warmstart_after_ns', fallback=0.0)
warmstart_after_frames = sim_config.getint('warmstart_after_frames', fallback=0)

# Logging
logging_mode = sim_config.get('logging_mode', 'efficient').lower()
write_charges = sim_config.getboolean('write_charges', fallback=False)
write_components = sim_config.getboolean('write_components', fallback=False)

# Import appropriate MM_classes based on version
if mm_version == 'cython':
    print("🔥 Loading Cython-accelerated MM classes.")
    from MM_classes_CYTHON import MM_CYTHON as MM
    from Fixed_Voltage_routines import *
else: # 'original' or 'optimized'
    print("📦 Loading unified NumPy MM classes.")
    from MM_classes import *
    from Fixed_Voltage_routines import *

# [Files]
file_config = config['Files']
outPath = file_config.get('outPath')
ffdir = file_config.get('ffdir')
if not ffdir.endswith('/'): ffdir += '/'
pdb_list = [file_config.get('pdb_file')]
def parse_xml_list(config_str, prefix_dir): return [prefix_dir + s.strip() for s in config_str.split(',')]
residue_xml_list = parse_xml_list(file_config.get('residue_xml_list'), ffdir)
ff_xml_list = parse_xml_list(file_config.get('ff_xml_list'), ffdir)

# [Electrodes]
elec_config = config['Electrodes']
def parse_index_tuple(config_str): return tuple(int(s.strip()) for s in config_str.split(','))
cathode_index = parse_index_tuple(elec_config.get('cathode_index'))
anode_index = parse_index_tuple(elec_config.get('anode_index'))

# --- Setup ---
if os.path.exists(outPath): shutil.rmtree(outPath)
os.mkdir(outPath)
chargeFile = open(os.path.join(outPath, 'charges.dat'), 'w')
componentsFile = open(os.path.join(outPath, 'components.log'), 'w') if write_components and logging_mode == 'efficient' else None

from sapt_exclusions import *
MMsys=MM( pdb_list=pdb_list, residue_xml_list=residue_xml_list, ff_xml_list=ff_xml_list )
MMsys.set_periodic_residue(True)
MMsys.set_platform(openmm_platform)
MMsys.initialize_electrodes( Voltage, cathode_identifier=cathode_index, anode_identifier=anode_index, chain=True, exclude_element=("H",) )
MMsys.initialize_electrolyte(Natom_cutoff=100)
MMsys.generate_exclusions( flag_SAPT_FF_exclusions=True )

# ... (Initial state printing logic remains the same) ...

trajectory_file_name = os.path.join(outPath, 'FV_NVT.dcd')
MMsys.set_trajectory_output(trajectory_file_name, freq_traj_output_ps * 1000)

# ######################  MAIN SIMULATION LOOP ###########################
print(f"\nStarting simulation ({simulation_type}) for {simulation_time_ns} ns...")
t_start = datetime.now()

warmstart_activated = False

if logging_mode == 'legacy_print':
    print("--- WARNING: Running in 'legacy_print' mode. This will be VERY SLOW. ---")
    for i in range(int(simulation_time_ns * 1000 / freq_traj_output_ps)):
        # ... (Legacy printing logic remains the same) ...
        current_time_ns = i * freq_traj_output_ps / 1000.0
        use_warmstart_now, warmstart_activated = should_use_warmstart(i, current_time_ns, warmstart_activated, enable_warmstart, warmstart_after_ns, warmstart_after_frames)

        if simulation_type == "Constant_V":
            steps_in_this_chunk = int(freq_traj_output_ps * 1000 / freq_charge_update_fs)
            for _ in range(steps_in_this_chunk):
                MMsys.Poisson_solver_fixed_voltage(Niterations=4, enable_warmstart=use_warmstart_now, verify_interval=verify_interval)
                MMsys.simmd.step(freq_charge_update_fs)
            if write_charges: MMsys.write_electrode_charges(chargeFile)
        # ... (MC_equil logic remains the same) ...

elif logging_mode == 'efficient':
    print(f"--- Running in 'efficient' mode. Logging to {outPath}*.log ---")
    log_freq = freq_traj_output_ps * 1000
    MMsys.simmd.reporters.append(StateDataReporter(os.path.join(outPath, 'energy.log'), log_freq, step=True, potentialEnergy=True, kineticEnergy=True, temperature=True))

    if simulation_type == "Constant_V":
        n_total_updates = int((simulation_time_ns * 1000 * 1000) / freq_charge_update_fs)
        steps_per_charge_update = freq_charge_update_fs
        charge_write_interval = int((freq_traj_output_ps * 1000) / freq_charge_update_fs) or 1 if write_charges else 0
        
        for i in range(n_total_updates):
            current_time_ns = (i * freq_charge_update_fs) / (1000.0 * 1000.0)
            use_warmstart_now, warmstart_activated = should_use_warmstart(i, current_time_ns, warmstart_activated, enable_warmstart, warmstart_after_ns, warmstart_after_frames)
            
            MMsys.Poisson_solver_fixed_voltage(Niterations=4, enable_warmstart=use_warmstart_now, verify_interval=verify_interval)
            MMsys.simmd.step(steps_per_charge_update)
            
            if charge_write_interval > 0 and (i + 1) % charge_write_interval == 0:
                MMsys.write_electrode_charges(chargeFile)
            # ... (components writing logic remains the same) ...

# --- Finalization ---
t_end = datetime.now()
print(f"Simulation finished in {t_end - t_start}.")
chargeFile.close()
if componentsFile: componentsFile.close()
print('done!')
sys.exit()
