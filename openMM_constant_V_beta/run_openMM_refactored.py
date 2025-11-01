#!/usr/bin/env python3
"""
OpenMM Fixed-Voltage MD Simulation
Refactored from original with Linus principles:
- Config-driven (no hardcoding)
- Support 4 MM versions (original/optimized/cython/plugin)
- Zero functionality removal
- Clean structure
"""

from __future__ import print_function
import sys
import os
import shutil
import argparse
import configparser
from datetime import datetime

# Add lib to path
sys.path.insert( 0, './lib/')

# Electrode sheets need high recursion limit
sys.setrecursionlimit(5000)

# ============================================================
# Parse Config
# ============================================================
parser = argparse.ArgumentParser(description="Run OpenMM Fixed-Voltage MD")
parser.add_argument('-c', '--config', default='config_refactored.ini', help='Config file path')
args = parser.parse_args()

if not os.path.exists(args.config):
    print(f"Error: Config file not found: {args.config}")
    sys.exit(1)

config = configparser.ConfigParser()
config.read(args.config)

# [Simulation]
sim = config['Simulation']
simulation_time_ns = sim.getfloat('simulation_time_ns')
freq_charge_update_fs = sim.getint('freq_charge_update_fs')
freq_traj_output_ps = sim.getint('freq_traj_output_ps')
write_charges = sim.getboolean('write_charges', fallback=False)
simulation_type = sim.get('simulation_type')  # "Constant_V" or "MC_equil"
voltage = sim.getfloat('voltage')
platform = sim.get('platform', 'CUDA')
mm_version = sim.get('mm_version', 'original').lower()

# [Files]
files = config['Files']
outPath = files.get('outPath')
ffdir = files.get('ffdir')
if not ffdir.endswith('/'):
    ffdir += '/'
pdb_file = files.get('pdb_file')
residue_xml_list = [ffdir + s.strip() for s in files.get('residue_xml_list').split(',')]
ff_xml_list = [ffdir + s.strip() for s in files.get('ff_xml_list').split(',')]

# [Electrodes]
elec = config['Electrodes']
cathode_index = tuple(int(x) for x in elec.get('cathode_index').split(','))
anode_index = tuple(int(x) for x in elec.get('anode_index').split(','))

# [MC_equil] - optional section for MC equilibration
if config.has_section('MC_equil'):
    mc_config = config['MC_equil']
    electrode_move = mc_config.get('electrode_move', 'Anode')
    mc_pressure = mc_config.getfloat('pressure', 1.0)
    mc_barofreq = mc_config.getint('barofreq', 100)
    mc_shiftscale = mc_config.getfloat('shiftscale', 0.2)
else:
    # Defaults
    electrode_move = 'Anode'
    mc_pressure = 1.0
    mc_barofreq = 100
    mc_shiftscale = 0.2

# ============================================================
# Import MM Classes Based on Version (with automatic fallback)
# ============================================================
requested_version = mm_version
actual_version = mm_version
USE_PLUGIN = False

try:
    if mm_version == 'plugin':
        print("🚀 Attempting to use Plugin (C++/CUDA Poisson solver)...")
        from MM_classes import *
        from Fixed_Voltage_routines import *
        USE_PLUGIN = True
        print("✓ Plugin version loaded successfully")

    elif mm_version == 'cython':
        print("⚡ Attempting to use Cython version...")
        from MM_classes_CYTHON import *
        from Fixed_Voltage_routines_CYTHON import *
        print("✓ Cython version loaded successfully")

    elif mm_version == 'optimized':
        print("📊 Attempting to use NumPy optimized version...")
        from MM_classes_OPTIMIZED import *
        from Fixed_Voltage_routines_OPTIMIZED import *
        print("✓ Optimized version loaded successfully")

    else:  # original
        print("🐍 Using original Python version")
        from MM_classes import *
        from Fixed_Voltage_routines import *
        print("✓ Original version loaded successfully")

except (ImportError, ModuleNotFoundError, AttributeError) as e:
    # Fallback to original version
    print("\n" + "="*60)
    print("⚠️  WARNING: Failed to load requested version!")
    print("="*60)
    print(f"Requested version: {requested_version}")
    print(f"Error: {type(e).__name__}: {e}")
    print("\n🔄 Falling back to ORIGINAL Python version...")
    print("="*60 + "\n")

    # Import original version
    from MM_classes import *
    from Fixed_Voltage_routines import *
    actual_version = 'original'
    USE_PLUGIN = False
    print("✓ Original version loaded successfully (fallback)")

    # Log warning
    import warnings
    warnings.warn(
        f"Requested mm_version='{requested_version}' failed to load. "
        f"Using 'original' version instead. "
        f"Check that the module is installed correctly.",
        RuntimeWarning
    )

# Import OpenMM
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *

# ============================================================
# Setup Plugin (if requested)
# ============================================================
if USE_PLUGIN:
    try:
        import electrodecharge as ec
        print("✓ Plugin imported")
    except ImportError as e:
        print(f"✗ Plugin import failed: {e}")
        print("  Falling back to Python Poisson solver")
        USE_PLUGIN = False

# ============================================================
# Setup Output Directory
# ============================================================
if os.path.exists(outPath):
    shutil.rmtree(outPath)

strdir = outPath
os.mkdir(outPath)

# Open charge file if needed
if write_charges:
    chargeFile = open(strdir + 'charges.dat', 'w')

# ============================================================
# Import SAPT Exclusions
# ============================================================
from sapt_exclusions import *

# ============================================================
# Create MM System
# ============================================================
print("\n" + "="*60)
print("Creating MM system")
print("="*60)

MMsys = MM(
    pdb_list=[pdb_file],
    residue_xml_list=residue_xml_list,
    ff_xml_list=ff_xml_list
)

MMsys.set_periodic_residue(True)

t1 = datetime.now()

# Set platform
MMsys.set_platform(platform)

# Initialize electrodes
MMsys.initialize_electrodes(
    voltage,
    cathode_identifier=cathode_index,
    anode_identifier=anode_index,
    chain=True,
    exclude_element=("H",)
)

# Initialize electrolyte (for analytic charge correction)
MMsys.initialize_electrolyte(Natom_cutoff=100)

# Generate SAPT-FF exclusions
MMsys.generate_exclusions(flag_SAPT_FF_exclusions=True)

# ============================================================
# Print Initial State (exactly like original)
# ============================================================
state = MMsys.simmd.context.getState(getEnergy=True, getForces=True, getVelocities=False, getPositions=True)
positions = state.getPositions()

print(str(state.getKineticEnergy()))
print(str(state.getPotentialEnergy()))
for j in range(MMsys.system.getNumForces()):
    f = MMsys.system.getForce(j)
    print(type(f), str(MMsys.simmd.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()))

# Write initial PDB with Drudes
PDBFile.writeFile(MMsys.simmd.topology, positions, open(strdir + 'start_drudes.pdb', 'w'))

# ============================================================
# Attach Plugin Force (if using plugin)
# ============================================================
if USE_PLUGIN:
    print("\n" + "="*60)
    print("🔥 Configuring ElectrodeChargePlugin")
    print("="*60)
    try:
        # Load plugin
        conda_prefix = os.environ.get('CONDA_PREFIX', '')
        plugin_dir = os.path.join(conda_prefix, 'lib', 'plugins')
        if os.path.exists(plugin_dir):
            Platform.loadPluginsFromDirectory(plugin_dir)
            print(f"✓ Loaded plugins from: {plugin_dir}")

        # Create Force object
        force = ec.ElectrodeChargeForce()
        force.setCathode([a.atom_index for a in MMsys.Cathode.electrode_atoms], abs(voltage))
        force.setAnode([a.atom_index for a in MMsys.Anode.electrode_atoms], abs(voltage))
        force.setNumIterations(4)
        force.setSmallThreshold(MMsys.small_threshold)

        # Handle conductors if present
        if hasattr(MMsys, 'Conductor_list') and MMsys.Conductor_list:
            print(f"✓ Found {len(MMsys.Conductor_list)} conductor(s)")
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
                c_contact_normals.extend([
                    c.Electrode_contact_atom.nx,
                    c.Electrode_contact_atom.ny,
                    c.Electrode_contact_atom.nz
                ])
                # Geometry factor encodes conductor type
                cname = type(c).__name__
                if cname == 'Buckyball_Virtual':
                    c_geoms.append(c.dr_center_contact**2)
                elif cname == 'Nanotube_Virtual':
                    c_geoms.append(c.dr_center_contact * c.length / 2.0)
                else:
                    c_geoms.append(0.0)

            force.setConductorData(
                c_indices, c_normals, c_areas,
                c_contacts, c_contact_normals, c_geoms,
                c_atom_ids, c_atom_counts
            )

        # Add force to system
        force.setForceGroup(MMsys.system.getNumForces())
        MMsys.system.addForce(force)

        # Reinitialize context
        state_tmp = MMsys.simmd.context.getState(getPositions=True, getVelocities=True)
        MMsys.simmd.context.reinitialize()
        MMsys.simmd.context.setPositions(state_tmp.getPositions())
        if state_tmp.getVelocities() is not None:
            MMsys.simmd.context.setVelocities(state_tmp.getVelocities())

        print("✓ Plugin force attached to system")
    except Exception as e:
        print(f"✗ Plugin setup failed: {e}")
        print("  Falling back to Python Poisson solver")
        USE_PLUGIN = False

# ============================================================
# Setup Simulation Type
# ============================================================
if simulation_type == "MC_equil":
    # Monte Carlo equilibration
    celldim = MMsys.simmd.topology.getUnitCellDimensions()
    MMsys.MC = MC_parameters(
        MMsys.temperature,
        celldim,
        electrode_move=electrode_move,
        pressure=mc_pressure*bar,
        barofreq=mc_barofreq,
        shiftscale=mc_shiftscale
    )
    trajectory_file_name = strdir + 'equil_MC.dcd'
else:
    trajectory_file_name = strdir + 'FV_NVT.dcd'

MMsys.set_trajectory_output(trajectory_file_name, freq_traj_output_ps * 1000)

# ============================================================
# Main Simulation Loop (EXACTLY like original)
# ============================================================
print("\n" + "="*60)
print(f"Starting {simulation_type} simulation")
print(f"Time: {simulation_time_ns} ns")
print("="*60 + "\n")

for i in range(int(simulation_time_ns * 1000 / freq_traj_output_ps)):
    state = MMsys.simmd.context.getState(getEnergy=True, getForces=True, getVelocities=False, getPositions=True)
    print(i, 'iteration')
    print(str(state.getKineticEnergy()))
    print(str(state.getPotentialEnergy()))
    for j in range(MMsys.system.getNumForces()):
        f = MMsys.system.getForce(j)
        print(type(f), str(MMsys.simmd.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()))

    # Monte Carlo Simulation
    if simulation_type == "MC_equil":
        for j in range(int(freq_traj_output_ps * 1000 / MMsys.MC.barofreq)):
            MMsys.MC_Barostat_step()

    # Constant Voltage Simulation
    elif simulation_type == "Constant_V":
        for j in range(int(freq_traj_output_ps * 1000 / freq_charge_update_fs)):
            # Fixed Voltage Electrostatics (only if NOT using plugin)
            if not USE_PLUGIN:
                MMsys.Poisson_solver_fixed_voltage(Niterations=4)
            # MD step
            MMsys.simmd.step(freq_charge_update_fs)

        if write_charges:
            # Write charges
            MMsys.write_electrode_charges(chargeFile)

    else:
        print('simulation type not recognized ...')
        sys.exit()

print('done!')
sys.exit()
