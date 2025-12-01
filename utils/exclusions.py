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


def generate_exclusions_water(
    system: openmm.System,
    topology: app.Topology,
    water_residue_name: str = 'HOH'
) -> None:
    """
    Configure hybrid water model interaction groups.

    Creates CustomNonbondedForce interaction groups for hybrid water models:
    - Water-water interactions: Use NonbondedForce (SWM4-NDP/TIP4P parameters)
    - Water-other interactions: Use CustomNonbondedForce (SAPT-FF parameters)
    - Other-other interactions: Use CustomNonbondedForce

    Physical Reasoning:
        Water force fields (TIP4P, SWM4-NDP) are optimized for water-water interactions.
        When mixing with SAPT-FF for ions/organic molecules, we need different
        interaction potentials for water-solute vs. water-water.

    Corresponds to: electrode_sapt_exclusions.py::generate_exclusions_water()

    Args:
        system: OpenMM System object
        topology: OpenMM Topology object
        water_residue_name: Residue name for water molecules (default: 'HOH')
    """
    # Find CustomNonbondedForce
    custom_nonbonded_force: Optional[openmm.CustomNonbondedForce] = None
    for force in system.getForces():
        if isinstance(force, openmm.CustomNonbondedForce):
            custom_nonbonded_force = force
            break

    if custom_nonbonded_force is None:
        logger.warning("CustomNonbondedForce not found; cannot configure hybrid water model")
        return

    # Build water and non-water atom sets
    water_atoms: Set[int] = set()
    notwater_atoms: Set[int] = set()

    for residue in topology.residues():
        atom_indices = [atom.index for atom in residue.atoms()]
        if residue.name == water_residue_name:
            water_atoms.update(atom_indices)
        else:
            notwater_atoms.update(atom_indices)

    if len(water_atoms) == 0:
        logger.info(f"No water residues ('{water_residue_name}') found; skipping hybrid water model")
        return

    if len(notwater_atoms) == 0:
        logger.warning("Only water atoms found; hybrid water model not applicable")
        return

    # Add interaction groups
    # Group 1: water × notwater (water-other interactions via CustomNonbonded)
    # Group 2: notwater × notwater (other-other interactions via CustomNonbonded)
    # Note: water × water is handled by NonbondedForce (SWM4-NDP/TIP4P)
    custom_nonbonded_force.addInteractionGroup(water_atoms, notwater_atoms)
    custom_nonbonded_force.addInteractionGroup(notwater_atoms, notwater_atoms)

    logger.info(
        f"Hybrid water model configured: {len(water_atoms)} water atoms, "
        f"{len(notwater_atoms)} non-water atoms"
    )
    logger.info("  Water-water: NonbondedForce (SWM4-NDP/TIP4P)")
    logger.info("  Water-other: CustomNonbondedForce (SAPT-FF)")
    logger.info("  Other-other: CustomNonbondedForce (SAPT-FF)")


def exclusion_Conductor_NonbondedForce(
    system: openmm.System,
    topology: app.Topology,
    conductor_virtual_indices: List[int],
    conductor_real_indices: List[int]
) -> None:
    """
    Add exclusions for conductor (Buckyball/Nanotube) atoms.

    Conductors have two layers:
    - Virtual layer: Used for electrostatics (Maxwell BC)
    - Real layer: Used for VDW/steric interactions

    Exclusion rules:
    - Real × Real: EXCLUDE (no double-counting of VDW)
    - Real × Virtual: EXCLUDE (prevent unphysical forces)
    - Virtual × Virtual: DO NOT EXCLUDE (needed for electrostatics)

    Corresponds to: MM_classes.py::generate_exclusions() lines 592-601

    Args:
        system: OpenMM System object
        topology: OpenMM Topology object
        conductor_virtual_indices: Virtual layer atom indices
        conductor_real_indices: Real layer atom indices
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

    # Get existing exclusions
    existing_exceptions = _get_existing_exceptions(nonbonded_force)
    existing_exclusions = set()
    if custom_nonbonded_force is not None:
        existing_exclusions = _get_existing_exclusions(custom_nonbonded_force)

    # Rule 1: Exclude Real × Real (all pairs within real layer)
    real_real_count = 0
    for i in range(len(conductor_real_indices)):
        for j in range(i + 1, len(conductor_real_indices)):
            idx_i = conductor_real_indices[i]
            idx_j = conductor_real_indices[j]
            pair_key = f"{idx_i}_{idx_j}"
            pair_key_rev = f"{idx_j}_{idx_i}"

            # NonbondedForce exception
            try:
                nonbonded_force.addException(idx_i, idx_j, 0.0, 1.0, 0.0, True)
                real_real_count += 1
            except Exception:
                pass

            # CustomNonbondedForce exclusion
            if custom_nonbonded_force is not None:
                if pair_key not in existing_exclusions and pair_key_rev not in existing_exclusions:
                    try:
                        custom_nonbonded_force.addExclusion(idx_i, idx_j)
                        existing_exclusions.add(pair_key)
                        existing_exclusions.add(pair_key_rev)
                    except Exception:
                        pass

    # Rule 2: Exclude Real × Virtual (all pairs between layers)
    real_virtual_count = 0
    for idx_real in conductor_real_indices:
        for idx_virtual in conductor_virtual_indices:
            pair_key = f"{idx_real}_{idx_virtual}"
            pair_key_rev = f"{idx_virtual}_{idx_real}"

            # NonbondedForce exception
            try:
                nonbonded_force.addException(idx_real, idx_virtual, 0.0, 1.0, 0.0, True)
                real_virtual_count += 1
            except Exception:
                pass

            # CustomNonbondedForce exclusion
            if custom_nonbonded_force is not None:
                if pair_key not in existing_exclusions and pair_key_rev not in existing_exclusions:
                    try:
                        custom_nonbonded_force.addExclusion(idx_real, idx_virtual)
                        existing_exclusions.add(pair_key)
                        existing_exclusions.add(pair_key_rev)
                    except Exception:
                        pass

    # Rule 3: DO NOT exclude Virtual × Virtual (needed for electrostatics)
    logger.debug(
        f"Conductor exclusions: Real×Real={real_real_count}, Real×Virtual={real_virtual_count}, "
        f"Virtual×Virtual=0 (not excluded)"
    )


def add_all_exclusions(
    system: openmm.System,
    topology: app.Topology,
    cathode_indices: List[int],
    anode_indices: List[int],
    include_tfsi: bool = True,
    include_water: bool = False,
    water_residue_name: str = 'HOH',
    conductor_configs: Optional[List[Dict]] = None
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
    - Hybrid water model (interaction groups)
    - Conductor exclusions (Buckyball/Nanotube virtual/real layers)

    Args:
        system: OpenMM System object
        topology: OpenMM Topology object
        cathode_indices: List of cathode atom indices
        anode_indices: List of anode atom indices
        include_tfsi: Whether to add TFSI exclusions (default: True)
        include_water: Whether to configure hybrid water model (default: False)
        water_residue_name: Residue name for water (default: 'HOH')
        conductor_configs: List of conductor configurations with 'virtual_indices' and 'real_indices'
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

    # Step 2: Conductor exclusions (if conductors are present)
    if conductor_configs is not None and len(conductor_configs) > 0:
        logger.info(f"Adding exclusions for {len(conductor_configs)} conductor(s)")
        for config in conductor_configs:
            exclusion_Conductor_NonbondedForce(
                system, topology,
                config['virtual_indices'],
                config['real_indices']
            )

    # Step 3: Hybrid water model (if applicable)
    if include_water:
        generate_exclusions_water(system, topology, water_residue_name)

    # Step 4: TFSI exclusions (if applicable)
    if include_tfsi:
        generate_exclusions_TFSI(system, topology, drude_force)

    logger.info("=" * 60)
    logger.info("Exclusion setup complete")
    logger.info("=" * 60)
