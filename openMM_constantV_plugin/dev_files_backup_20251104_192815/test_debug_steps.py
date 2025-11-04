#!/usr/bin/env python3
"""Debug test to find where the process is hanging."""

import sys
import os
import ctypes

print("[1] Starting imports...")
sys.stdout.flush()

plugin_build_dir = os.path.abspath("./ConstantVPlugin/build")
plugin_python_dir = os.path.join(plugin_build_dir, "python/build/lib.linux-x86_64-cpython-313")
sys.path.insert(0, plugin_python_dir)
sys.path.insert(0, './fv_md_plugin')
sys.path.insert(0, '../Andy_openMM_constantV/ffdir')

print("[2] Loading plugin libraries...")
sys.stdout.flush()

main_lib = ctypes.CDLL(f"{plugin_build_dir}/libConstantVPlugin.so", mode=ctypes.RTLD_GLOBAL)
ref_lib = ctypes.CDLL(f"{plugin_build_dir}/platforms/reference/libConstantVPluginReference.so", mode=ctypes.RTLD_GLOBAL)
cuda_lib = ctypes.CDLL(f"{plugin_build_dir}/platforms/cuda/libConstantVPluginCUDA.so", mode=ctypes.RTLD_GLOBAL)

print("[3] Registering kernel factories...")
sys.stdout.flush()

ref_lib.registerKernelFactories()
cuda_lib.registerKernelFactories()

print("[4] Importing OpenMM...")
sys.stdout.flush()

from openmm import *
from openmm.app import *
from openmm.unit import *

print("[5] Importing constantvplugin...")
sys.stdout.flush()

import constantvplugin

print("[6] All imports successful!")
sys.stdout.flush()

print("[7] Loading bond definitions...")
sys.stdout.flush()

FFDIR = "../Andy_openMM_constantV/ffdir/"
RESIDUE_XML_FILES = [
    FFDIR + "sapt_residues.xml",
    FFDIR + "graph_residue_c.xml",
    FFDIR + "graph_residue_n.xml",
]

for residue_file in RESIDUE_XML_FILES:
    print(f"    Loading {os.path.basename(residue_file)}...")
    sys.stdout.flush()
    Topology().loadBondDefinitions(residue_file)

print("[8] Loading PDB...")
sys.stdout.flush()

TEST_PDB = "../Andy_openMM_constantV/for_openmm.pdb"
pdb = PDBFile(TEST_PDB)

print(f"[9] PDB loaded: {pdb.topology.getNumAtoms()} atoms")
sys.stdout.flush()

print("[10] Creating Modeller...")
sys.stdout.flush()

modeller = Modeller(pdb.topology, pdb.positions)

print("[11] Loading ForceField...")
sys.stdout.flush()

FORCEFIELD_XML_FILES = [
    FFDIR + "sapt_noDB_2sheets.xml",
    FFDIR + "graph_c_freeze.xml",
    FFDIR + "graph_n_freeze.xml"
]

forcefield = ForceField(*FORCEFIELD_XML_FILES)

print("[12] Adding extra particles (Drude oscillators)...")
sys.stdout.flush()

modeller.addExtraParticles(forcefield)

print(f"[13] Topology after extra particles: {modeller.topology.getNumAtoms()} atoms")
sys.stdout.flush()

print("[14] Creating System (this may take a minute)...")
sys.stdout.flush()

system = forcefield.createSystem(
    modeller.topology,
    nonbondedMethod=PME,
    nonbondedCutoff=1.4*nanometer,
    constraints=HBonds,
    rigidWater=True
)

print(f"[15] System created: {system.getNumParticles()} particles")
sys.stdout.flush()

print("\n✓ Test passed - System creation successful!")
sys.stdout.flush()
