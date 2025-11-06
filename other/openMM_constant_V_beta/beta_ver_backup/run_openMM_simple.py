#!/usr/bin/env python3
"""
Linus-style simplified OpenMM Constant Voltage simulation
- No over-engineering
- Direct print output
- Hardcoded sane defaults
- Plugin or Python Poisson solver
"""

import sys, os, shutil, argparse, configparser
from datetime import datetime
import numpy as np

# Set recursion limit for electrode sheets
sys.setrecursionlimit(5000)

# ============================================================
# Config Reading
# ============================================================
parser = argparse.ArgumentParser(description="Run OpenMM Fixed-Voltage MD")
parser.add_argument('-c', '--config', default='config.ini', help='Config file')
args = parser.parse_args()

config = configparser.ConfigParser()
if not os.path.exists(args.config):
    print(f"Error: Config file not found: {args.config}")
    sys.exit(1)
config.read(args.config)

# Read simulation parameters
sim_config = config['Simulation']
simulation_time_ns = sim_config.getint('simulation_time_ns')
freq_charge_update_fs = sim_config.getint('freq_charge_update_fs')
freq_traj_output_ps = sim_config.getint('freq_traj_output_ps')
simulation_type = sim_config.get('simulation_type')
openmm_platform = sim_config.get('platform')
Voltage = sim_config.getfloat('voltage')
mm_version = sim_config.get('mm_version', 'original').lower()

# Read file paths
files_config = config['Files']
outPath = files_config.get('outPath')
ffdir = files_config.get('ffdir')
pdb_file = files_config.get('pdb_file')
residue_xml_list = [x.strip() for x in files_config.get('residue_xml_list').split(',')]
ff_xml_list = [x.strip() for x in files_config.get('ff_xml_list').split(',')]

# Read electrode indices
electrode_config = config['Electrodes']
cathode_index = [int(x) for x in electrode_config.get('cathode_index').split(',')]
anode_index = [int(x) for x in electrode_config.get('anode_index').split(',')]

# ============================================================
# Import MM_classes based on version
# ============================================================
# Add lib directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

if mm_version == 'plugin':
    print("🚀 Using OpenMM Plugin (C++/CUDA)")
    from MM_classes import *
    from Fixed_Voltage_routines import *
    USE_PLUGIN = True
elif mm_version == 'cython':
    print("Using Cython version")
    from MM_classes_CYTHON import *
    from Fixed_Voltage_routines_CYTHON import *
    USE_PLUGIN = False
elif mm_version == 'optimized':
    print("Using NumPy optimized version")
    from MM_classes_OPTIMIZED import *
    from Fixed_Voltage_routines_OPTIMIZED import *
    USE_PLUGIN = False
else:  # original
    print("Using original Python version")
    from MM_classes import *
    from Fixed_Voltage_routines import *
    USE_PLUGIN = False

# ============================================================
# Setup Plugin (if requested)
# ============================================================
if USE_PLUGIN:
    try:
        sys.path.insert(0, os.path.join(os.environ.get('CONDA_PREFIX', ''), 'lib', 'python*/site-packages'))
        import electrodecharge as ec
        print("✓ Plugin imported")
    except Exception as e:
        print(f"✗ Plugin import failed: {e}")
        print("  Falling back to Python Poisson solver")
        USE_PLUGIN = False

# ============================================================
# Setup output directory
# ============================================================
if os.path.exists(outPath):
    shutil.rmtree(outPath)
os.mkdir(outPath)

# Import SAPT exclusions
from lib.electrode_sapt_exclusions import *

# ============================================================
# Create MM System
# ============================================================
pdb_list = [ffdir + pdb_file]
residue_xml_list = [ffdir + f for f in residue_xml_list]
ff_xml_list = [ffdir + f for f in ff_xml_list]

MMsys = MM(pdb_list=pdb_list, residue_xml_list=residue_xml_list, ff_xml_list=ff_xml_list)
MMsys.set_periodic_residue(True)

# Setup electrodes
Cathode_object = Electrode_Virtual(cathode_index, electrode_type='cathode', Voltage=abs(Voltage), MMsys=MMsys)
Anode_object = Electrode_Virtual(anode_index, electrode_type='anode', Voltage=abs(Voltage), MMsys=MMsys)
MMsys.Cathode = Cathode_object
MMsys.Anode = Anode_object

# Initialize system
MMsys.create_openmm_system(residue_xml_list, ff_xml_list)
MMsys.set_platform(openmm_platform)
MMsys.initialize_positions(pdb_list)
MMsys.initialize_velocities(300.0)

# Setup trajectory output
trajectory_file_name = outPath + 'FV_NVT.dcd'
freq_traj = int(freq_traj_output_ps * 1000.0 / MMsys.timestep)
MMsys.add_trajectory(trajectory_file_name, freq_traj)

# ============================================================
# Attach Plugin Force (if using plugin)
# ============================================================
if USE_PLUGIN:
    try:
        force = ec.ElectrodeChargeForce()
        force.setCathode([a.atom_index for a in MMsys.Cathode.electrode_atoms], abs(Voltage))
        force.setAnode([a.atom_index for a in MMsys.Anode.electrode_atoms], abs(Voltage))
        force.setNumIterations(4)  # Hardcoded: sane default
        force.setSmallThreshold(MMsys.small_threshold)

        # Conductors (if any)
        if MMsys.Conductor_list:
            c_indices, c_normals, c_areas = [], [], []
            c_contacts, c_contact_normals, c_geoms = [], [], []
            c_atom_ids, c_atom_counts = [], []
            
            for i, c in enumerate(MMsys.Conductor_list):
                for atom in c.electrode_atoms:
                    c_indices.append(atom.atom_index)
                    c_normals.extend([atom.nx, atom.ny, atom.nz])
                    c_areas.append(c.area_atom)
                    c_atom_ids.append(i)
                c_atom_counts.append(c.Natoms)
                c_contacts.append(c.Electrode_contact_atom.atom_index)
                c_contact_normals.extend([c.Electrode_contact_atom.nx, 
                                         c.Electrode_contact_atom.ny, 
                                         c.Electrode_contact_atom.nz])
                # Geometry factor encodes type
                cname = type(c).__name__
                if cname == 'Buckyball_Virtual':
                    c_geoms.append(c.dr_center_contact**2)
                elif cname == 'Nanotube_Virtual':
                    c_geoms.append(c.dr_center_contact * c.length / 2.0)
            
            force.setConductorData(c_indices, c_normals, c_areas, c_contacts,
                                  c_contact_normals, c_geoms, c_atom_ids, c_atom_counts)

        force.setForceGroup(MMsys.system.getNumForces())
        MMsys.system.addForce(force)
        
        # Reinitialize context
        state_tmp = MMsys.simmd.context.getState(getPositions=True, getVelocities=True)
        MMsys.simmd.context.reinitialize()
        MMsys.simmd.context.setPositions(state_tmp.getPositions())
        if state_tmp.getVelocities() is not None:
            MMsys.simmd.context.setVelocities(state_tmp.getVelocities())
        
        print("✓ Plugin attached to system")
    except Exception as e:
        print(f"✗ Plugin setup failed: {e}")
        USE_PLUGIN = False

# ============================================================
# Print initial state
# ============================================================
state = MMsys.simmd.context.getState(getEnergy=True, getForces=True, getPositions=True)
print("\n--- Initial State ---")
print(f"Kinetic Energy: {state.getKineticEnergy()}")
print(f"Potential Energy: {state.getPotentialEnergy()}")
print(f"Platform: {openmm_platform}")
print(f"MM Version: {mm_version}")
print(f"Plugin Active: {USE_PLUGIN}")

# ============================================================
# Main Simulation Loop (Constant V only)
# ============================================================
if simulation_type != "Constant_V":
    print(f"Error: Only Constant_V supported in simple version")
    sys.exit(1)

print(f"\nStarting Constant_V simulation for {simulation_time_ns} ns...")
t_start = datetime.now()

# Calculate loop parameters
timestep_fs = MMsys.timestep
steps_per_charge_update = int(freq_charge_update_fs / timestep_fs)
total_sim_time_fs = simulation_time_ns * 1e6
n_total_updates = int(total_sim_time_fs / freq_charge_update_fs)

print(f"Total charge updates: {n_total_updates}")
print(f"MD steps per update: {steps_per_charge_update}")

# Main loop
for i in range(n_total_updates):
    # Poisson solver (only if NOT using plugin)
    if not USE_PLUGIN:
        MMsys.Poisson_solver_fixed_voltage(Niterations=4)
    
    # MD step
    MMsys.simmd.step(steps_per_charge_update)
    
    # Print every 10ps
    if i % 50 == 0:
        state = MMsys.simmd.context.getState(getEnergy=True)
        time_ns = i * freq_charge_update_fs / 1e6
        print(f"[{time_ns:.3f} ns] E = {state.getPotentialEnergy()}")

t_end = datetime.now()
print(f"\nSimulation finished in {t_end - t_start}")
print("Done!")
