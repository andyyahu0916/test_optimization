"""
Force Field Exclusions for FV-MD Simulation

CRITICAL MODULE: This module implements the exclusions necessary for correct
physical modeling of electrode-electrolyte systems.

Without these exclusions, electrode atoms would interact with each other through
both NonbondedForce AND the ConstantVPlugin, leading to double-counting and
completely incorrect physics.

This module handles:
1. Electrode internal exclusions (no electrode-electrode interactions)
2. SAPT-FF exclusions for water and ions
3. Drude screening for polarizable systems
"""

from openmm.app import *
from openmm import *
from openmm.unit import *
import numpy as np


def apply_electrode_exclusions(system, topology, cathode_atoms, anode_atoms):
    """
    Apply exclusions for intra-electrode interactions.
    
    This is CRITICAL: Without this, electrode atoms will interact with each other
    through NonbondedForce, while the ConstantVPlugin also applies potentials to them.
    This would lead to double-counting and incorrect physics.
    
    Args:
        system: OpenMM System object
        topology: OpenMM Topology object
        cathode_atoms: List of cathode atom indices
        anode_atoms: List of anode atom indices
    """
    print("\n" + "="*70)
    print("APPLYING ELECTRODE EXCLUSIONS (CRITICAL FOR CORRECT PHYSICS)")
    print("="*70)
    
    # Get force objects
    nonbonded = None
    custom_nonbonded = None
    
    for i in range(system.getNumForces()):
        force = system.getForce(i)
        if type(force) == NonbondedForce:
            nonbonded = force
        elif type(force) == CustomNonbondedForce:
            custom_nonbonded = force
    
    if nonbonded is None:
        print("ERROR: NonbondedForce not found!")
        return
    
    # Apply exclusions for cathode
    print(f"\n1. Applying cathode internal exclusions ({len(cathode_atoms)} atoms)...")
    n_cath_excl = _exclude_electrode_internal(
        cathode_atoms, cathode_atoms, nonbonded, custom_nonbonded
    )
    print(f"   ✓ Added {n_cath_excl} cathode-cathode exclusions")
    
    # Apply exclusions for anode
    print(f"\n2. Applying anode internal exclusions ({len(anode_atoms)} atoms)...")
    n_anode_excl = _exclude_electrode_internal(
        anode_atoms, anode_atoms, nonbonded, custom_nonbonded
    )
    print(f"   ✓ Added {n_anode_excl} anode-anode exclusions")
    
    print("\n" + "="*70)
    print("✓ ELECTRODE EXCLUSIONS COMPLETE")
    print("="*70)


def _exclude_electrode_internal(electrode1, electrode2, nonbonded, custom_nonbonded):
    """
    Internal function to exclude all interactions between two electrode lists.
    
    This is a direct port from the original electrode_sapt_exclusions.py
    """
    # Track existing exclusions to avoid duplicates
    existing_exclusions = set()
    
    if custom_nonbonded is not None:
        for i in range(custom_nonbonded.getNumExclusions()):
            p1, p2 = custom_nonbonded.getExclusionParticles(i)
            existing_exclusions.add((min(p1, p2), max(p1, p2)))
    
    n_added = 0
    
    # Add exclusions for every atom pair
    if electrode1 is electrode2:
        # Same electrode - exclude i,j pairs where i < j
        for i in range(len(electrode1)):
            idx_i = electrode1[i]
            for j in range(i+1, len(electrode2)):
                idx_j = electrode2[j]
                pair = (min(idx_i, idx_j), max(idx_i, idx_j))
                
                if pair not in existing_exclusions:
                    # Add to CustomNonbondedForce if present
                    if custom_nonbonded is not None:
                        custom_nonbonded.addExclusion(idx_i, idx_j)
                    
                    # Add exception to NonbondedForce (zero charge, zero LJ)
                    # The 'True' flag means replace if exists
                    nonbonded.addException(idx_i, idx_j, 0.0, 1.0, 0.0, True)
                    n_added += 1
    else:
        # Different electrodes - exclude all i,j pairs
        for i in range(len(electrode1)):
            idx_i = electrode1[i]
            for j in range(len(electrode2)):
                idx_j = electrode2[j]
                pair = (min(idx_i, idx_j), max(idx_i, idx_j))
                
                if pair not in existing_exclusions:
                    if custom_nonbonded is not None:
                        custom_nonbonded.addExclusion(idx_i, idx_j)
                    nonbonded.addException(idx_i, idx_j, 0.0, 1.0, 0.0, True)
                    n_added += 1
    
    return n_added


def apply_sapt_exclusions(system, topology):
    """
    Apply SAPT-FF specific exclusions for water and ions.
    
    This handles:
    1. Water-water vs water-other interaction groups (hybrid model)
    2. TFSI (Tf2N) internal exclusions with Drude screening
    
    Args:
        system: OpenMM System object
        topology: OpenMM Topology object
    """
    print("\n" + "="*70)
    print("APPLYING SAPT-FF FORCE FIELD EXCLUSIONS")
    print("="*70)
    
    # Get force objects
    nonbonded = None
    custom_nonbonded = None
    drude_force = None
    
    for i in range(system.getNumForces()):
        force = system.getForce(i)
        if type(force) == NonbondedForce:
            nonbonded = force
        elif type(force) == CustomNonbondedForce:
            custom_nonbonded = force
        elif type(force) == DrudeForce:
            drude_force = force
    
    if nonbonded is None:
        print("ERROR: NonbondedForce not found!")
        return
    
    # Check for water molecules
    has_water = False
    for res in topology.residues():
        if res.name == 'HOH':
            has_water = True
            break
    
    if has_water:
        print("\n1. Applying water exclusions (hybrid SWM4-NDP/SAPT-FF model)...")
        _apply_water_exclusions(custom_nonbonded, topology)
        print("   ✓ Water interaction groups configured")
    
    # Check for TFSI molecules
    has_tfsi = False
    for res in topology.residues():
        if res.name == 'Tf2N':
            has_tfsi = True
            break
    
    if has_tfsi:
        print("\n2. Applying TFSI exclusions with Drude screening...")
        _apply_tfsi_exclusions(system, topology, nonbonded, custom_nonbonded, drude_force)
        print("   ✓ TFSI exclusions and screening complete")
    
    print("\n" + "="*70)
    print("✓ SAPT-FF EXCLUSIONS COMPLETE")
    print("="*70)


def _apply_water_exclusions(custom_nonbonded, topology):
    """
    Create interaction groups for hybrid water model.
    
    Water-water interactions use SWM4-NDP (in NonbondedForce)
    Water-other interactions use SAPT-FF (in CustomNonbondedForce)
    """
    if custom_nonbonded is None:
        print("   WARNING: No CustomNonbondedForce found, skipping water exclusions")
        return
    
    water_atoms = set()
    other_atoms = set()
    
    for res in topology.residues():
        if res.name == 'HOH':
            # Add all atoms in water residue
            for atom in res.atoms():
                water_atoms.add(atom.index)
        else:
            # Add all atoms in non-water residues
            for atom in res.atoms():
                other_atoms.add(atom.index)
    
    print(f"   Found {len(water_atoms)} water atoms, {len(other_atoms)} other atoms")
    
    # Set up interaction groups: water-other and other-other
    # (water-water is handled by NonbondedForce)
    custom_nonbonded.addInteractionGroup(water_atoms, other_atoms)
    custom_nonbonded.addInteractionGroup(other_atoms, other_atoms)


def _apply_tfsi_exclusions(system, topology, nonbonded, custom_nonbonded, drude_force):
    """
    Apply TFSI-specific exclusions and Drude screening.
    
    This creates exclusions for all atom pairs within TFSI molecules,
    and adds screened Thole interactions for Drude pairs.
    """
    if drude_force is None:
        print("   WARNING: No DrudeForce found, TFSI may not have polarization")
        return
    
    # Map from particle index to drude force index
    particle_to_drude = {}
    for i in range(drude_force.getNumParticles()):
        particle_idx = drude_force.getParticleParameters(i)[0]
        particle_to_drude[particle_idx] = i
    
    # Track existing exceptions and exclusions
    existing_exceptions = set()
    for i in range(nonbonded.getNumExceptions()):
        p1, p2, _, _, _ = nonbonded.getExceptionParameters(i)
        existing_exceptions.add((min(p1, p2), max(p1, p2)))
    
    existing_exclusions = set()
    if custom_nonbonded is not None:
        for i in range(custom_nonbonded.getNumExclusions()):
            p1, p2 = custom_nonbonded.getExclusionParticles(i)
            existing_exclusions.add((min(p1, p2), max(p1, p2)))
    
    n_exclusions = 0
    n_screened = 0
    
    # Process each TFSI residue
    for res in topology.residues():
        if res.name != 'Tf2N':
            continue
        
        atoms = list(res.atoms())
        
        # Exclude all atom pairs in this TFSI molecule
        for i in range(len(atoms)):
            idx_i = atoms[i].index
            for j in range(i+1, len(atoms)):
                idx_j = atoms[j].index
                pair = (min(idx_i, idx_j), max(idx_i, idx_j))
                
                # Add exception to NonbondedForce
                nonbonded.addException(idx_i, idx_j, 0.0, 1.0, 0.0, True)
                
                # Add exclusion to CustomNonbondedForce if not already present
                if custom_nonbonded is not None and pair not in existing_exclusions:
                    custom_nonbonded.addExclusion(idx_i, idx_j)
                    n_exclusions += 1
                
                # Add Thole screening if both are Drude particles
                if idx_i in particle_to_drude and idx_j in particle_to_drude:
                    if pair not in existing_exceptions:
                        drude_i = particle_to_drude[idx_i]
                        drude_j = particle_to_drude[idx_j]
                        drude_force.addScreenedPair(drude_i, drude_j, 2.0)
                        n_screened += 1
    
    print(f"   Added {n_exclusions} TFSI exclusions")
    print(f"   Added {n_screened} Drude screened pairs")


def apply_all_exclusions(system, topology, cathode_atoms, anode_atoms, 
                         apply_sapt=True):
    """
    Apply all necessary exclusions for FV-MD simulation.
    
    This is the main entry point that should be called after system creation
    but before creating the Simulation object.
    
    Args:
        system: OpenMM System object
        topology: OpenMM Topology object
        cathode_atoms: List of cathode atom indices
        anode_atoms: List of anode atom indices
        apply_sapt: Whether to apply SAPT-FF specific exclusions
    """
    print("\n" + "="*70)
    print("APPLYING FORCE FIELD EXCLUSIONS")
    print("="*70)
    print("\nWARNING: Without these exclusions, electrode atoms would interact")
    print("         with each other through BOTH NonbondedForce AND ConstantVPlugin,")
    print("         leading to DOUBLE-COUNTING and COMPLETELY INCORRECT PHYSICS!")
    print("="*70)
    
    # Step 1: Apply electrode exclusions (CRITICAL)
    apply_electrode_exclusions(system, topology, cathode_atoms, anode_atoms)
    
    # Step 2: Apply SAPT-FF exclusions (if using SAPT force field)
    if apply_sapt:
        apply_sapt_exclusions(system, topology)
    
    print("\n" + "="*70)
    print("✓ ALL EXCLUSIONS APPLIED SUCCESSFULLY")
    print("="*70)
    print()


# Convenience function to check if exclusions are needed
def check_exclusions_applied(system, cathode_atoms, anode_atoms):
    """
    Check if electrode exclusions have been applied.
    
    Returns True if exclusions appear to be present, False otherwise.
    This is a simple heuristic check, not exhaustive.
    """
    nonbonded = None
    for i in range(system.getNumForces()):
        force = system.getForce(i)
        if type(force) == NonbondedForce:
            nonbonded = force
            break
    
    if nonbonded is None:
        return False
    
    # Check if there are any exceptions between cathode atoms
    n_exceptions = nonbonded.getNumExceptions()
    cathode_set = set(cathode_atoms)
    
    for i in range(n_exceptions):
        p1, p2, q, sigma, epsilon = nonbonded.getExceptionParameters(i)
        # If we find zero-charge exception between cathode atoms, assume exclusions present
        if p1 in cathode_set and p2 in cathode_set and abs(q) < 1e-10:
            return True
    
    return False
