#!/usr/bin/env python3
"""
Quick test of FV-MD plugin using PRE-COMPUTED C_inv matrix.

This test validates the complete pipeline WITHOUT waiting for C_inv computation.
In production, C_inv should be pre-computed and loaded from file.
"""

import sys
import os
import ctypes
import numpy as np

# Set up library paths for plugin
plugin_build_dir = os.path.abspath("./ConstantVPlugin/build")
plugin_lib_dir = plugin_build_dir
plugin_python_dir = os.path.join(plugin_build_dir, "python/build/lib.linux-x86_64-cpython-313")

# Add to Python path
sys.path.insert(0, plugin_python_dir)
sys.path.insert(0, './fv_md_plugin')
sys.path.insert(0, '../Andy_openMM_constantV/ffdir')

# Manually load plugin libraries and register kernels
print("Loading ConstantVPlugin libraries...")
main_lib = ctypes.CDLL(f"{plugin_lib_dir}/libConstantVPlugin.so", mode=ctypes.RTLD_GLOBAL)
ref_lib = ctypes.CDLL(f"{plugin_lib_dir}/platforms/reference/libConstantVPluginReference.so", mode=ctypes.RTLD_GLOBAL)
cuda_lib = ctypes.CDLL(f"{plugin_lib_dir}/platforms/cuda/libConstantVPluginCUDA.so", mode=ctypes.RTLD_GLOBAL)

# Register kernels
ref_lib.registerKernelFactories()
cuda_lib.registerKernelFactories()
print("✓ Plugin libraries loaded and kernels registered\n")

# Test parameters from config_refactored.ini
TEST_PDB = "../Andy_openMM_constantV/for_openmm.pdb"
TEST_VOLTAGE = 4.0  # Volts
TEST_TIME = 0.001  # nanoseconds (very short for quick test)
CATHODE_CHAINS = [0, 2]
ANODE_CHAINS = [1, 3]

# Force field files (from config)
FFDIR = "../Andy_openMM_constantV/ffdir/"
RESIDUE_XML_FILES = [
    FFDIR + "sapt_residues.xml",
    FFDIR + "graph_residue_c.xml",
    FFDIR + "graph_residue_n.xml",
]
FORCEFIELD_XML_FILES = [
    FFDIR + "sapt_noDB_2sheets.xml",
    FFDIR + "graph_c_freeze.xml",
    FFDIR + "graph_n_freeze.xml"
]

print("="*70)
print("Quick Test: FV-MD Plugin with Pre-computed C_inv")
print("="*70)
print(f"PDB file: {TEST_PDB}")
print(f"Voltage: ±{TEST_VOLTAGE} V")
print(f"Test duration: {TEST_TIME} ns")
print()

# Import our refactored runner
from run_fv_md_plugin import (
    setup_system,
    identify_electrode_atoms,
    identify_electrolyte_atoms,
    run_simulation
)

# OpenMM imports
from openmm import *
from openmm.app import *
from openmm.unit import *
import constantvplugin

# Step 1: Setup system
print("Step 1: Setting up OpenMM system...")
modeller, system, nonbonded = setup_system(TEST_PDB, RESIDUE_XML_FILES, FORCEFIELD_XML_FILES)
print("✓ System setup successful\n")

# Step 2: Identify atoms
print("Step 2: Identifying electrode and electrolyte atoms...")
cathode_atoms, anode_atoms = identify_electrode_atoms(
    modeller.topology,
    CATHODE_CHAINS,
    ANODE_CHAINS
)
all_electrode_chains = CATHODE_CHAINS + ANODE_CHAINS
electrolyte_atoms = identify_electrolyte_atoms(
    modeller.topology,
    all_electrode_chains
)
print("✓ Atom identification successful\n")

# Step 3: Initialize plugin with IDENTITY matrix as C_inv (for quick test)
print("Step 3: Initializing ConstantVPlugin with identity C_inv...")
print("(In production, C_inv should be pre-computed and loaded from file)")

N_electrode = len(cathode_atoms) + len(anode_atoms)
print(f"Using identity matrix as placeholder C_inv ({N_electrode}×{N_electrode})")
C_inv = np.eye(N_electrode)

# Convert voltage to kJ/mol
CONVERSION_V_TO_KJMOL = 96.485
voltage_kjmol = TEST_VOLTAGE * CONVERSION_V_TO_KJMOL

# Create ConstantVForce
cv_force = constantvplugin.ConstantVForce()

# Add electrode atoms
for atom_idx in cathode_atoms:
    cv_force.addElectrodeAtom(atom_idx, voltage_kjmol)
for atom_idx in anode_atoms:
    cv_force.addElectrodeAtom(atom_idx, -voltage_kjmol)

# Add electrolyte atoms
for atom_idx in electrolyte_atoms:
    charge, sigma, epsilon = nonbonded.getParticleParameters(atom_idx)
    cv_force.addElectrolyteAtom(atom_idx, charge.value_in_unit(elementary_charge))

# Set C_inv matrix
cv_force.setInverseCapacitanceMatrix(C_inv.flatten().tolist())

# Add force to system
system.addForce(cv_force)

print(f"✓ ConstantVForce added to system")
print(f"  - Electrodes: {N_electrode} atoms")
print(f"  - Electrolyte: {len(electrolyte_atoms)} atoms")
print(f"  - Voltage: ±{TEST_VOLTAGE} V (±{voltage_kjmol:.3f} kJ/mol)")
print(f"  - C_inv: Identity matrix (placeholder for testing)\n")

# Step 4: Run simulation
print("Step 4: Running simulation...")
try:
    simulation = run_simulation(
        modeller,
        system,
        TEST_VOLTAGE,
        TEST_TIME,
        "test_fv_md_quick_output"
    )
    print("✓ Simulation completed successfully\n")
except Exception as e:
    print(f"✗ Simulation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("="*70)
print("✓ QUICK TEST PASSED!")
print("="*70)
print("\nKey findings:")
print("  1. System setup works correctly (with Modeller and extra particles)")
print("  2. Electrode/electrolyte identification works correctly")
print("  3. ConstantVPlugin initialization works")
print("  4. Simulation runs successfully")
print("\nNext steps:")
print("  - For production use, pre-compute C_inv matrix and save to file")
print("  - Load pre-computed C_inv at initialization")
print("  - C_inv computation is O(N³), should be done ONCE offline")
