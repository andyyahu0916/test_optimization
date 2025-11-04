#!/usr/bin/env python3
"""
Pre-compute C_inv matrix for production FV-MD.

This script should be run ONCE for each electrode geometry.
The resulting C_inv matrix can be loaded instantly in production runs.

Usage:
    python precompute_cinv.py -c config_refactored.ini -o C_inv_matrix.npy
"""

import sys
import os
import numpy as np
import argparse
import configparser
from datetime import datetime

sys.path.insert(0, './fv_md_plugin')

from openmm import *
from openmm.app import *
from openmm.unit import *

from run_fv_md_plugin import (
    setup_system,
    identify_electrode_atoms,
    compute_electrode_areas
)
from compute_capacitance_matrix import compute_inverse_capacitance_matrix

def main():
    parser = argparse.ArgumentParser(description="Pre-compute C_inv matrix")
    parser.add_argument('-c', '--config', required=True,
                        help='Config file path')
    parser.add_argument('-o', '--output', required=True,
                        help='Output C_inv matrix file (.npy)')
    args = parser.parse_args()

    # Parse config
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    config = configparser.ConfigParser()
    config.read(args.config)

    # [Files]
    files = config['Files']
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
    print("Pre-computing C_inv Matrix for ConstantVPlugin")
    print("="*70)
    print(f"Config: {args.config}")
    print(f"PDB: {pdb_file}")
    print(f"Output: {args.output}")
    print("="*70)
    print()

    # Setup system
    print("Setting up system...")
    modeller, system, nonbonded = setup_system(pdb_file, residue_xml_list, ff_xml_list)

    # Identify electrodes
    print("\nIdentifying electrode atoms...")
    cathode_atoms, anode_atoms = identify_electrode_atoms(
        modeller.topology,
        cathode_index,
        anode_index
    )

    # Get electrode positions and areas
    box_vectors = modeller.topology.getPeriodicBoxVectors()
    cathode_areas = compute_electrode_areas(len(cathode_atoms), box_vectors)
    anode_areas = compute_electrode_areas(len(anode_atoms), box_vectors)

    cathode_pos = np.array([modeller.positions[i].value_in_unit(nanometer) for i in cathode_atoms])
    anode_pos = np.array([modeller.positions[i].value_in_unit(nanometer) for i in anode_atoms])

    all_electrode_pos = np.vstack([cathode_pos, anode_pos])
    all_electrode_areas = np.concatenate([cathode_areas, anode_areas])

    N = len(cathode_atoms) + len(anode_atoms)
    print(f"\nComputing C_inv matrix for {N} electrode atoms...")
    print("(This may take several minutes for large systems)")
    print()

    start_time = datetime.now()

    C_inv = compute_inverse_capacitance_matrix(all_electrode_pos, all_electrode_areas)

    elapsed = (datetime.now() - start_time).total_seconds()

    print(f"\n✓ C_inv computed in {elapsed:.1f} seconds")
    print(f"  Shape: {C_inv.shape}")
    print(f"  Determinant: {np.linalg.det(C_inv):.6e}")
    print(f"  Min/Max: {C_inv.min():.6e} / {C_inv.max():.6e}")

    # Save matrix
    np.save(args.output, C_inv)
    print(f"\n✓ C_inv saved to {args.output}")
    print(f"  File size: {os.path.getsize(args.output) / 1024 / 1024:.1f} MB")

    print("\n" + "="*70)
    print("Pre-computation complete!")
    print("="*70)
    print(f"\nUse this matrix in production runs:")
    print(f"  python run_fv_md_production.py -c {args.config} --load-cinv {args.output}")


if __name__ == "__main__":
    main()
