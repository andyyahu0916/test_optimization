"""
Exclusion Rules for ConstantV Electrode Simulations

This module ports the critical exclusion logic from the original
OpenMM-ConstantV plugin to prevent catastrophic Coulomb explosions.

Corresponds to: electrode_sapt_exclusions.py

CRITICAL: These exclusions MUST be added before Context creation.
Without them, electrode atoms will have huge repulsions and the
simulation will explode immediately.
"""

import logging
from typing import List, Set, Optional, Dict
import openmm
from openmm import app

logger = logging.getLogger(__name__)


def _get_existing_exclusions(custom_nonbonded_force: openmm.CustomNonbondedForce) -> Set[str]:
    """
    Get existing exclusions from a CustomNonbondedForce as a set of strings.
    
    Returns set of "i_j" strings for existing exclusion pairs.
    Corresponds to: electrode_sapt_exclusions.py lines 21-28
    """
    existing = set()
    for i in range(custom_nonbonded_force.getNumExclusions()):
        p1, p2 = custom_nonbonded_force.getExclusionParticles(i)
        existing.add(f"{p1}_{p2}")
        existing.add(f"{p2}_{p1}")
    return existing


def _get_existing_exceptions(nonbonded_force: openmm.NonbondedForce) -> Set[str]:
    """
    Get existing exceptions from a NonbondedForce as a set of strings.
    
    Returns set of "i_j" strings for existing exception pairs.
    """
    existing = set()
    for i in range(nonbonded_force.getNumExceptions()):
        p1, p2, charge, sigma, epsilon = nonbonded_force.getExceptionParameters(i)
        existing.add(f"{p1}_{p2}")
        existing.add(f"{p2}_{p1}")
    return existing


def exclusion_Electrode_NonbondedForce(
    system: openmm.System,
    topology: app.Topology,
    cathode_indices: List[int],
    anode_indices: List[int]
) -> None:
    """
    Add exclusions between electrode atoms to prevent Coulomb explosions.

    This function implements the critical exclusion logic from
    electrode_sapt_exclusions.py::exclusion_Electrode_NonbondedForce()

    CRITICAL: This function now handles BOTH NonbondedForce AND CustomNonbondedForce!
    The original Python code always excludes from both force types.

    Physical Reasoning:
        - Electrode atoms are constrained/frozen (not free to move)
        - Their charges are determined by SCF (not by force field)
        - Coulomb interactions between electrode atoms would create
          unphysical forces that cannot be relieved by relaxation
        - Therefore, we EXCLUDE all electrode-electrode interactions

    Args:
        system: OpenMM System object
        topology: OpenMM Topology object
        cathode_indices: List of cathode atom indices
        anode_indices: List of anode atom indices

    Corresponds to: electrode_sapt_exclusions.py lines 15-61
    """
    # Find NonbondedForce and CustomNonbondedForce
    nonbonded_force: Optional[openmm.NonbondedForce] = None
    custom_nonbonded_force: Optional[openmm.CustomNonbondedForce] = None
    
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            nonbonded_force = force
        elif isinstance(force, openmm.CustomNonbondedForce):
            custom_nonbonded_force = force

    if nonbonded_force is None:
        raise RuntimeError("NonbondedForce not found in system")

    # Combine all electrode indices
    all_electrode_indices = set(cathode_indices + anode_indices)

    logger.info(f"Adding electrode exclusions for {len(all_electrode_indices)} atoms")

    # ═══════════════════════════════════════════════════════════════════════
    # Get existing exclusions (don't add duplicates)
    # Corresponds to: electrode_sapt_exclusions.py lines 21-28
    # ═══════════════════════════════════════════════════════════════════════
    
    existing_exceptions = _get_existing_exceptions(nonbonded_force)
    existing_exclusions = set()
    if custom_nonbonded_force is not None:
        existing_exclusions = _get_existing_exclusions(custom_nonbonded_force)

    # ═══════════════════════════════════════════════════════════════════════
    # Add exclusions between ALL pairs of electrode atoms
    # This prevents catastrophic Coulomb repulsions
    # Corresponds to: electrode_sapt_exclusions.py lines 30-45
    # ═══════════════════════════════════════════════════════════════════════
    
    nonbonded_count = 0
    custom_count = 0
    electrode_list = sorted(all_electrode_indices)

    for i in range(len(electrode_list)):
        for j in range(i + 1, len(electrode_list)):
            atom_i = electrode_list[i]
            atom_j = electrode_list[j]
            pair_key = f"{atom_i}_{atom_j}"
            pair_key_rev = f"{atom_j}_{atom_i}"

            # ═══════════════════════════════════════════════════════════════
            # Add exception to NonbondedForce
            # charge=0, sigma=1, epsilon=0 -> zero interaction
            # The "True" flag replaces any existing exception
            # Corresponds to: nbondedForce.addException(indexi,indexj,0,1,0,True)
            # ═══════════════════════════════════════════════════════════════
            try:
                nonbonded_force.addException(atom_i, atom_j, 0.0, 1.0, 0.0, True)
                nonbonded_count += 1
            except Exception as e:
                logger.warning(f"Could not add NonbondedForce exception for {atom_i}-{atom_j}: {e}")

            # ═══════════════════════════════════════════════════════════════
            # Add exclusion to CustomNonbondedForce (if present)
            # CRITICAL FIX: The original code ALWAYS does this!
            # Corresponds to: customNonbondedForce.addExclusion(indexi,indexj)
            # ═══════════════════════════════════════════════════════════════
            if custom_nonbonded_force is not None:
                if pair_key not in existing_exclusions and pair_key_rev not in existing_exclusions:
                    try:
                        custom_nonbonded_force.addExclusion(atom_i, atom_j)
                        custom_count += 1
                        # Update the set to avoid adding again
                        existing_exclusions.add(pair_key)
                        existing_exclusions.add(pair_key_rev)
                    except Exception as e:
                        logger.warning(f"Could not add CustomNonbondedForce exclusion for {atom_i}-{atom_j}: {e}")

    logger.info(f"Added {nonbonded_count} NonbondedForce exceptions")
    if custom_nonbonded_force is not None:
        logger.info(f"Added {custom_count} CustomNonbondedForce exclusions")


def generate_exclusions_TFSI(
    system: openmm.System,
    topology: app.Topology,
    drude_force: Optional[openmm.DrudeForce] = None
) -> None:
    """
    Add exclusions for TFSI (bis(trifluoromethylsulfonyl)imide) ions.

    This function implements the TFSI-specific exclusion logic from
    electrode_sapt_exclusions.py::generate_exclusions_TFSI()

    Physical Reasoning:
        - TFSI is a large, soft anion with polarizable sites
        - We exclude ALL intramolecular nonbonded interactions
        - For Drude pairs, we add ScreenedPair interactions instead

    CRITICAL: Now handles both NonbondedForce AND CustomNonbondedForce!
    
    Args:
        system: OpenMM System object
        topology: OpenMM Topology object
        drude_force: Optional DrudeForce for adding ScreenedPair (Thole damping)

    Corresponds to: electrode_sapt_exclusions.py::generate_exclusions_TFSI() lines 100-145
    """
    # Find NonbondedForce and CustomNonbondedForce
    nonbonded_force: Optional[openmm.NonbondedForce] = None
    custom_nonbonded_force: Optional[openmm.CustomNonbondedForce] = None
    
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            nonbonded_force = force
        elif isinstance(force, openmm.CustomNonbondedForce):
            custom_nonbonded_force = force

    if nonbonded_force is None:
        raise RuntimeError("NonbondedForce not found in system")

    # Find all TFSI residues (various naming conventions)
    tfsi_names = ["TFSI", "NTF2", "Ntf2", "Tf2N", "TF2N"]
    tfsi_residues = []
    for residue in topology.residues():
        if residue.name in tfsi_names:
            tfsi_residues.append(residue)

    if len(tfsi_residues) == 0:
        logger.info("No TFSI residues found, skipping TFSI exclusions")
        return

    logger.info(f"Found {len(tfsi_residues)} TFSI residues")

    # ═══════════════════════════════════════════════════════════════════════
    # Build Drude particle map if DrudeForce is provided
    # Corresponds to: electrode_sapt_exclusions.py lines 107-109
    # ═══════════════════════════════════════════════════════════════════════
    
    drude_particle_map: Dict[int, int] = {}
    if drude_force is not None:
        for i in range(drude_force.getNumParticles()):
            params = drude_force.getParticleParameters(i)
            parent_idx = params[0]  # First parameter is the parent atom index
            drude_particle_map[parent_idx] = i

    # ═══════════════════════════════════════════════════════════════════════
    # Get existing exclusions/exceptions
    # Corresponds to: electrode_sapt_exclusions.py lines 112-126
    # ═══════════════════════════════════════════════════════════════════════
    
    existing_exceptions = _get_existing_exceptions(nonbonded_force)
    existing_exclusions = set()
    if custom_nonbonded_force is not None:
        existing_exclusions = _get_existing_exclusions(custom_nonbonded_force)

    # ═══════════════════════════════════════════════════════════════════════
    # Add exclusions for all atom pairs on TFSI residues
    # Corresponds to: electrode_sapt_exclusions.py lines 128-145
    # ═══════════════════════════════════════════════════════════════════════
    
    nonbonded_count = 0
    custom_count = 0
    screened_pair_count = 0

    for residue in tfsi_residues:
        atom_indices = [atom.index for atom in residue.atoms()]

        for i in range(len(atom_indices)):
            for j in range(i + 1, len(atom_indices)):
                indi = atom_indices[i]
                indj = atom_indices[j]
                pair_key = f"{indi}_{indj}"
                pair_key_rev = f"{indj}_{indi}"

                # ═══════════════════════════════════════════════════════════
                # Add exception to NonbondedForce (True flag allows replace)
                # Corresponds to: self.nbondedForce.addException(indi,indj,0,1,0,True)
                # ═══════════════════════════════════════════════════════════
                try:
                    nonbonded_force.addException(indi, indj, 0.0, 1.0, 0.0, True)
                    nonbonded_count += 1
                except Exception as e:
                    logger.debug(f"NonbondedForce exception already exists for {indi}-{indj}")

                # ═══════════════════════════════════════════════════════════
                # Add exclusion to CustomNonbondedForce (if present)
                # CRITICAL FIX: Original code does this!
                # Corresponds to: self.customNonbondedForce.addExclusion(indi,indj)
                # ═══════════════════════════════════════════════════════════
                if custom_nonbonded_force is not None:
                    if pair_key not in existing_exclusions and pair_key_rev not in existing_exclusions:
                        try:
                            custom_nonbonded_force.addExclusion(indi, indj)
                            custom_count += 1
                            existing_exclusions.add(pair_key)
                            existing_exclusions.add(pair_key_rev)
                        except Exception as e:
                            logger.debug(f"CustomNonbondedForce exclusion already exists for {indi}-{indj}")

                # ═══════════════════════════════════════════════════════════
                # Add ScreenedPair if excluding two Drude particles
                # Corresponds to: self.drudeForce.addScreenedPair(drudei, drudej, 2.0)
                # ═══════════════════════════════════════════════════════════
                if drude_force is not None:
                    if indi in drude_particle_map and indj in drude_particle_map:
                        if pair_key not in existing_exceptions and pair_key_rev not in existing_exceptions:
                            drudei = drude_particle_map[indi]
                            drudej = drude_particle_map[indj]
                            try:
                                drude_force.addScreenedPair(drudei, drudej, 2.0)  # Thole damping = 2.0
                                screened_pair_count += 1
                            except Exception as e:
                                logger.debug(f"ScreenedPair already exists for {indi}-{indj}")

    logger.info(f"TFSI: Added {nonbonded_count} NonbondedForce exceptions")
    if custom_nonbonded_force is not None:
        logger.info(f"TFSI: Added {custom_count} CustomNonbondedForce exclusions")
    if drude_force is not None and screened_pair_count > 0:
        logger.info(f"TFSI: Added {screened_pair_count} Drude ScreenedPairs")


def add_all_exclusions(
    system: openmm.System,
    topology: app.Topology,
    cathode_indices: List[int],
    anode_indices: List[int],
    include_tfsi: bool = True
) -> None:
    """
    Add all necessary exclusions to the system.

    This is a convenience function that calls all exclusion rules
    in the correct order.

    CRITICAL: Call this BEFORE creating the Context!

    This function now properly handles:
    - NonbondedForce (standard Coulomb/LJ)
    - CustomNonbondedForce (SAPT-FF, custom interactions)
    - DrudeForce (for Drude oscillator ScreenedPairs)

    Args:
        system: OpenMM System object
        topology: OpenMM Topology object
        cathode_indices: List of cathode atom indices
        anode_indices: List of anode atom indices
        include_tfsi: Whether to add TFSI exclusions (default: True)
    """
    logger.info("=" * 60)
    logger.info("Adding Exclusions (CRITICAL for Stability)")
    logger.info("=" * 60)

    # Find DrudeForce if present
    drude_force: Optional[openmm.DrudeForce] = None
    for force in system.getForces():
        if isinstance(force, openmm.DrudeForce):
            drude_force = force
            break

    if drude_force is not None:
        logger.info("DrudeForce detected - will add ScreenedPairs for exclusions")

    # Step 1: Electrode exclusions (MANDATORY)
    exclusion_Electrode_NonbondedForce(
        system, topology,
        cathode_indices, anode_indices
    )

    # Step 2: TFSI exclusions (if applicable)
    if include_tfsi:
        generate_exclusions_TFSI(system, topology, drude_force)

    logger.info("=" * 60)
    logger.info("Exclusion setup complete")
    logger.info("=" * 60)
