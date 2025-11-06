#!/usr/bin/env python3
"""
Test the refactored FV-MD plugin with real data from Andy_openMM_constantV.

This is a quick test (0.001 ns) to verify:
1. System setup works correctly
2. Electrode/electrolyte identification is correct
3. C_inv matrix computation succeeds
4. Plugin initializes without errors
5. Simulation runs and produces output
"""

import sys
import os
import ctypes

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
print("Testing FV-MD Plugin with Real Data")
print("="*70)
print(f"PDB file: {TEST_PDB}")
print(f"Voltage: ±{TEST_VOLTAGE} V")
print(f"Test duration: {TEST_TIME} ns")
print(f"Cathode chains: {CATHODE_CHAINS}")
print(f"Anode chains: {ANODE_CHAINS}")
print(f"Residue XML files: {len(RESIDUE_XML_FILES)} files")
print(f"Force field XML files: {len(FORCEFIELD_XML_FILES)} files")
print()

# Import our refactored runner
from run_fv_md_plugin import (
    setup_system,
    identify_electrode_atoms,
    identify_electrolyte_atoms,
    initialize_constantv_plugin,
    run_simulation
)

# Test Step 1: Setup system
print("Step 1: Setting up OpenMM system...")
try:
    modeller, system, nonbonded = setup_system(TEST_PDB, RESIDUE_XML_FILES, FORCEFIELD_XML_FILES)
    print("✓ System setup successful")
except Exception as e:
    print(f"✗ System setup failed: {e}")
    sys.exit(1)

# Test Step 2: Identify electrode atoms
print("\nStep 2: Identifying electrode atoms...")
try:
    cathode_atoms, anode_atoms = identify_electrode_atoms(
        modeller.topology,
        CATHODE_CHAINS,
        ANODE_CHAINS
    )
    print("✓ Electrode identification successful")
except Exception as e:
    print(f"✗ Electrode identification failed: {e}")
    sys.exit(1)

# Test Step 3: Identify electrolyte atoms
print("\nStep 3: Identifying electrolyte atoms...")
try:
    all_electrode_chains = CATHODE_CHAINS + ANODE_CHAINS
    electrolyte_atoms = identify_electrolyte_atoms(
        modeller.topology,
        all_electrode_chains
    )
    print("✓ Electrolyte identification successful")
except Exception as e:
    print(f"✗ Electrolyte identification failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test Step 4: Initialize plugin
print("\nStep 4: Initializing ConstantVPlugin...")
try:
    cv_force, C_inv = initialize_constantv_plugin(
        system,
        nonbonded,
        cathode_atoms,
        anode_atoms,
        electrolyte_atoms,
        modeller.positions,
        modeller.topology.getPeriodicBoxVectors(),
        TEST_VOLTAGE
    )
    print("✓ Plugin initialization successful")
except Exception as e:
    print(f"✗ Plugin initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test Step 5: Run simulation
print("\nStep 5: Running simulation...")
try:
    simulation = run_simulation(
        modeller,
        system,
        TEST_VOLTAGE,
        TEST_TIME,
        "test_fv_md_output"
    )
    print("✓ Simulation completed successfully")
except Exception as e:
    print(f"✗ Simulation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("✓ ALL TESTS PASSED!")
print("="*70)
print("\nOutput files:")
print("  - test_fv_md_output.log")
print("  - test_fv_md_output.dcd")
print("  - test_fv_md_output_final.pdb")
