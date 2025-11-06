#!/usr/bin/env python3
"""
Production FV-MD with ConstantVPlugin - Config File Driven

This is the PRODUCTION version that reads config_refactored.ini
and runs the complete FV-MD simulation.

Key improvements over original:
- 4× reduction in CPU-GPU transfers (8 → 2 per timestep)
- Single-pass algorithm (no SCF iteration)
- Correct Drude polarization handling
- Frozen electrodes (mass = 0)
- Pre-computed C_inv matrix (loaded from file)
"""

import sys
import os
import ctypes
import numpy as np
import argparse
import configparser
from datetime import datetime

# Add plugin to path
plugin_build_dir = os.path.abspath("./ConstantVPlugin/build")
plugin_python_dir = os.path.join(plugin_build_dir, "python/build/lib.linux-x86_64-cpython-313")
sys.path.insert(0, plugin_python_dir)
sys.path.insert(0, './fv_md_plugin')

# Load and register plugin libraries
print("Loading ConstantVPlugin...")
main_lib = ctypes.CDLL(f"{plugin_build_dir}/libConstantVPlugin.so", mode=ctypes.RTLD_GLOBAL)
ref_lib = ctypes.CDLL(f"{plugin_build_dir}/platforms/reference/libConstantVPluginReference.so", mode=ctypes.RTLD_GLOBAL)
cuda_lib = ctypes.CDLL(f"{plugin_build_dir}/platforms/cuda/libConstantVPluginCUDA.so", mode=ctypes.RTLD_GLOBAL)
ref_lib.registerKernelFactories()
cuda_lib.registerKernelFactories()
print("✓ Plugin loaded\n")

# Import OpenMM and our modules
from openmm import *
from openmm.app import *
from openmm.unit import *
import constantvplugin

from run_fv_md_plugin import (
    setup_system,
    identify_electrode_atoms,
    identify_electrolyte_atoms,
    initialize_constantv_plugin
)
from exclusions import apply_all_exclusions

def main():
    parser = argparse.ArgumentParser(description="Production FV-MD with ConstantVPlugin")
    parser.add_argument('-c', '--config', default='config_refactored.ini',
                        help='Config file path')
    parser.add_argument('--load-cinv', default=None,
                        help='Pre-computed C_inv matrix file (.npy)')
    args = parser.parse_args()

    # ========================================================================
    # Parse Config
    # ========================================================================
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    config = configparser.ConfigParser()
    config.read(args.config)

    # [Simulation]
    sim = config['Simulation']
    simulation_time_ns = sim.getfloat('simulation_time_ns')
    freq_traj_output_ps = sim.getint('freq_traj_output_ps')
    write_charges = sim.getboolean('write_charges', fallback=False)
    voltage = sim.getfloat('voltage')
    openmm_platform = sim.get('platform', 'CUDA').strip()

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
    cathode_index = [int(x) for x in elec.get('cathode_index').split(',')]
    anode_index = [int(x) for x in elec.get('anode_index').split(',')]

    print("="*70)
    print("Production FV-MD with ConstantVPlugin")
    print("="*70)
    print(f"Config: {args.config}")
    print(f"PDB: {pdb_file}")
    print(f"Voltage: ±{voltage} V")
    print(f"Duration: {simulation_time_ns} ns")
    print(f"Platform: {openmm_platform}")
    print(f"Output: {outPath}")
    print("="*70)
    print()

    # Create output directory
    if os.path.exists(outPath):
        import shutil
        shutil.rmtree(outPath)
    os.makedirs(outPath)

    # ========================================================================
    # Setup System
    # ========================================================================
    print("Setting up system...")
    modeller, system, nonbonded = setup_system(pdb_file, residue_xml_list, ff_xml_list)

    # ========================================================================
    # Identify Atoms
    # ========================================================================
    print("\nIdentifying electrode and electrolyte atoms...")
    cathode_atoms, anode_atoms = identify_electrode_atoms(
        modeller.topology,
        cathode_index,
        anode_index
    )

    all_electrode_atoms = cathode_atoms + anode_atoms
    electrolyte_atoms = identify_electrolyte_atoms(
        system,
        modeller.topology,
        all_electrode_atoms,
        include_drude=True  # CRITICAL for correct physics
    )

    # ========================================================================
    # Apply Force Field Exclusions (CRITICAL!)
    # ========================================================================
    # This MUST be done after system creation but BEFORE plugin initialization.
    # Without these exclusions, electrode atoms would interact with each other
    # through BOTH NonbondedForce AND ConstantVPlugin, causing double-counting
    # and completely incorrect physics.
    print("\n" + "="*70)
    print("APPLYING FORCE FIELD EXCLUSIONS")
    print("="*70)
    apply_all_exclusions(
        system,
        modeller.topology,
        cathode_atoms,
        anode_atoms,
        apply_sapt=True  # Set to False if not using SAPT-FF
    )

    # ========================================================================
    # Initialize Plugin
    # ========================================================================
    if args.load_cinv:
        print(f"\nLoading pre-computed C_inv from {args.load_cinv}...")
        C_inv = np.load(args.load_cinv)
        print(f"✓ C_inv loaded: shape {C_inv.shape}")

        # Manual initialization without computing C_inv
        from run_fv_md_plugin import freeze_electrode_atoms, CONVERSION_V_TO_KJMOL
        freeze_electrode_atoms(system, all_electrode_atoms)

        voltage_kjmol = voltage * CONVERSION_V_TO_KJMOL
        cv_force = constantvplugin.ConstantVForce()

        for atom_idx in cathode_atoms:
            cv_force.addElectrodeAtom(atom_idx, voltage_kjmol)
        for atom_idx in anode_atoms:
            cv_force.addElectrodeAtom(atom_idx, -voltage_kjmol)

        print(f"Adding {len(electrolyte_atoms)} electrolyte particles...")
        for atom_idx in electrolyte_atoms:
            charge, sigma, epsilon = nonbonded.getParticleParameters(atom_idx)
            cv_force.addElectrolyteAtom(atom_idx, charge.value_in_unit(elementary_charge))

        cv_force.setInverseCapacitanceMatrix(C_inv.flatten().tolist())
        system.addForce(cv_force)
        print("✓ Plugin initialized with pre-computed C_inv")
    else:
        print("\nWARNING: Computing C_inv on-the-fly (this may take >5 minutes)")
        print("For production, use --load-cinv to load pre-computed matrix")
        cv_force, C_inv = initialize_constantv_plugin(
            system,
            nonbonded,
            cathode_atoms,
            anode_atoms,
            electrolyte_atoms,
            modeller.positions,
            modeller.topology.getPeriodicBoxVectors(),
            voltage
        )
        # Save C_inv for future use
        cinv_file = outPath + "C_inv_matrix.npy"
        np.save(cinv_file, C_inv)
        print(f"✓ C_inv saved to {cinv_file} for future use")

    # ========================================================================
    # Run Simulation
    # ========================================================================
    print(f"\nRunning FV-MD simulation ({simulation_time_ns} ns)...")

    # Create integrator
    integrator = LangevinIntegrator(
        300*kelvin,
        1.0/picosecond,
        0.001*picosecond
    )

    # Create simulation with specified platform
    try:
        platform = Platform.getPlatformByName(openmm_platform)
        print(f"Using {openmm_platform} platform")
    except Exception as e:
        print(f"Platform {openmm_platform} not available: {e}")
        print("Falling back to Reference platform")
        platform = Platform.getPlatformByName('Reference')

    simulation = Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)

    # Minimize
    print("Minimizing energy...")
    simulation.minimizeEnergy(maxIterations=100)

    # Set up reporters
    simulation.reporters.append(
        StateDataReporter(
            outPath + 'simulation.log',
            1000,
            step=True,
            time=True,
            potentialEnergy=True,
            temperature=True,
            speed=True
        )
    )

    dcd_freq = int(freq_traj_output_ps / 0.001)  # ps to steps
    simulation.reporters.append(
        DCDReporter(
            outPath + 'trajectory.dcd',
            dcd_freq
        )
    )

    if write_charges:
        charge_file = open(outPath + 'charges.dat', 'w')
        # TODO: Add charge reporter if needed

    # Run simulation
    n_steps = int(simulation_time_ns * 1000000 / 0.001)  # ns to steps
    print(f"Running {n_steps} steps ({simulation_time_ns} ns)...")
    print("(Plugin updates charges automatically - no Python overhead!)")

    simulation.step(n_steps)

    print(f"✓ Simulation complete")

    # Save final state
    state = simulation.context.getState(getPositions=True)
    with open(outPath + 'final.pdb', 'w') as f:
        PDBFile.writeFile(
            simulation.topology,
            state.getPositions(),
            f
        )

    if write_charges:
        charge_file.close()

    print("="*70)
    print(f"Completed: {datetime.now()}")
    print(f"Output saved to: {outPath}")
    print("="*70)


if __name__ == "__main__":
    main()
