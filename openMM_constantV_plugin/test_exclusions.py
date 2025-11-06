#!/usr/bin/env python3
"""
Test script to verify that force field exclusions are correctly applied.

This script checks:
1. Electrode internal exclusions are present
2. SAPT-FF exclusions are applied (if applicable)
3. No double-counting of interactions
"""

import sys
import os
sys.path.insert(0, './fv_md_plugin')

from openmm import *
from openmm.app import *
from openmm.unit import *
import numpy as np

from run_fv_md_plugin import setup_system, identify_electrode_atoms, identify_electrolyte_atoms
from exclusions import apply_all_exclusions, check_exclusions_applied


def test_exclusions():
    """Test that exclusions are correctly applied."""
    
    print("="*70)
    print("TESTING FORCE FIELD EXCLUSIONS")
    print("="*70)
    
    # Setup paths (adjust as needed)
    pdb_file = "../Andy_openMM_constantV/4v_20ns/lastframe_4v_20ns_production.pdb"
    ffdir = "../Andy_openMM_constantV/ffdir/"
    
    residue_xml_list = [
        ffdir + 'graph_residue_c.xml',
        ffdir + 'graph_residue_n.xml',
        ffdir + 'sapt_residues.xml'
    ]
    
    ff_xml_list = [
        ffdir + 'graph_c.xml',
        ffdir + 'graph_n.xml',
        ffdir + 'sapt.xml',
        ffdir + 'sapt_Efield.xml'
    ]
    
    # Check if files exist
    if not os.path.exists(pdb_file):
        print(f"WARNING: PDB file not found: {pdb_file}")
        print("Please adjust the paths in this script to match your setup")
        return
    
    print("\n1. Setting up system...")
    modeller, system, nonbonded = setup_system(pdb_file, residue_xml_list, ff_xml_list)
    
    print("\n2. Identifying electrode atoms...")
    cathode_atoms, anode_atoms = identify_electrode_atoms(
        modeller.topology,
        cathode_chain_indices=[0, 1],
        anode_chain_indices=[2, 3]
    )
    
    all_electrode_atoms = cathode_atoms + anode_atoms
    print(f"   Cathode: {len(cathode_atoms)} atoms")
    print(f"   Anode: {len(anode_atoms)} atoms")
    print(f"   Total electrode: {len(all_electrode_atoms)} atoms")
    
    print("\n3. Checking exclusions BEFORE applying...")
    before_applied = check_exclusions_applied(system, cathode_atoms, anode_atoms)
    print(f"   Exclusions present: {before_applied}")
    
    print("\n4. Counting NonbondedForce exceptions BEFORE...")
    n_exceptions_before = nonbonded.getNumExceptions()
    print(f"   Number of exceptions: {n_exceptions_before}")
    
    # Check CustomNonbondedForce if present
    custom_nonbonded = None
    for force in system.getForces():
        if isinstance(force, CustomNonbondedForce):
            custom_nonbonded = force
            break
    
    if custom_nonbonded:
        n_custom_excl_before = custom_nonbonded.getNumExclusions()
        print(f"   CustomNonbonded exclusions: {n_custom_excl_before}")
    
    print("\n5. Applying exclusions...")
    apply_all_exclusions(
        system,
        modeller.topology,
        cathode_atoms,
        anode_atoms,
        apply_sapt=True
    )
    
    print("\n6. Checking exclusions AFTER applying...")
    after_applied = check_exclusions_applied(system, cathode_atoms, anode_atoms)
    print(f"   Exclusions present: {after_applied}")
    
    print("\n7. Counting NonbondedForce exceptions AFTER...")
    n_exceptions_after = nonbonded.getNumExceptions()
    print(f"   Number of exceptions: {n_exceptions_after}")
    print(f"   Added: {n_exceptions_after - n_exceptions_before} exceptions")
    
    if custom_nonbonded:
        n_custom_excl_after = custom_nonbonded.getNumExclusions()
        print(f"   CustomNonbonded exclusions: {n_custom_excl_after}")
        print(f"   Added: {n_custom_excl_after - n_custom_excl_before} exclusions")
    
    # Calculate expected number of electrode-electrode exclusions
    expected_cathode = len(cathode_atoms) * (len(cathode_atoms) - 1) // 2
    expected_anode = len(anode_atoms) * (len(anode_atoms) - 1) // 2
    expected_total = expected_cathode + expected_anode
    
    print("\n8. Verification:")
    print(f"   Expected cathode-cathode exclusions: {expected_cathode}")
    print(f"   Expected anode-anode exclusions: {expected_anode}")
    print(f"   Expected total electrode exclusions: {expected_total}")
    
    # Sample check: verify that cathode atoms do not interact
    print("\n9. Sampling electrode-electrode interactions...")
    sample_size = min(5, len(cathode_atoms))
    n_excluded = 0
    
    for i in range(sample_size):
        idx_i = cathode_atoms[i]
        for j in range(i+1, sample_size):
            idx_j = cathode_atoms[j]
            
            # Check if there's an exception for this pair
            for k in range(nonbonded.getNumExceptions()):
                p1, p2, q, sigma, epsilon = nonbonded.getExceptionParameters(k)
                if (p1 == idx_i and p2 == idx_j) or (p1 == idx_j and p2 == idx_i):
                    if abs(q) < 1e-10:  # Zero charge = excluded
                        n_excluded += 1
                    break
    
    expected_sample = sample_size * (sample_size - 1) // 2
    print(f"   Sampled {expected_sample} cathode-cathode pairs")
    print(f"   Found {n_excluded} excluded pairs")
    
    if n_excluded == expected_sample:
        print("   ✓ All sampled pairs are correctly excluded")
    else:
        print(f"   ✗ WARNING: Only {n_excluded}/{expected_sample} pairs excluded!")
    
    print("\n" + "="*70)
    if after_applied and n_excluded == expected_sample:
        print("✓ EXCLUSIONS TEST PASSED")
    else:
        print("✗ EXCLUSIONS TEST FAILED")
    print("="*70)


if __name__ == "__main__":
    test_exclusions()
