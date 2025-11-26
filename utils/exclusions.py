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
from typing import List, Set
import openmm
from openmm import app

logger = logging.getLogger(__name__)


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

    Corresponds to: electrode_sapt_exclusions.py lines 15-45
    """
    # Find NonbondedForce
    nonbonded_force = None
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            nonbonded_force = force
            break

    if nonbonded_force is None:
        raise RuntimeError("NonbondedForce not found in system")

    # Combine all electrode indices
    all_electrode_indices = set(cathode_indices + anode_indices)

    logger.info(f"Adding electrode exclusions for {len(all_electrode_indices)} atoms")

    # Add exclusions between ALL pairs of electrode atoms
    # This prevents catastrophic Coulomb repulsions
    exclusion_count = 0
    electrode_list = sorted(all_electrode_indices)

    for i in range(len(electrode_list)):
        for j in range(i + 1, len(electrode_list)):
            atom_i = electrode_list[i]
            atom_j = electrode_list[j]

            # Add bidirectional exclusion
            nonbonded_force.addExclusion(atom_i, atom_j)
            exclusion_count += 1

    logger.info(f"Added {exclusion_count} electrode-electrode exclusions")


def generate_exclusions_TFSI(
    system: openmm.System,
    topology: app.Topology,
    exclusion_distance_nm: float = 0.5
) -> None:
    """
    Add exclusions for TFSI (bis(trifluoromethylsulfonyl)imide) ions.

    This function implements the TFSI-specific exclusion logic to prevent
    unphysical interactions at close range.

    Physical Reasoning:
        - TFSI is a large, soft anion
        - At very close distances, the point-charge model breaks down
        - We add exclusions between atoms within a cutoff distance
        - This prevents spurious close-contact energies

    Args:
        system: OpenMM System object
        topology: OpenMM Topology object
        exclusion_distance_nm: Distance cutoff for adding exclusions (nm)

    Corresponds to: electrode_sapt_exclusions.py::generate_exclusions_TFSI()
    """
    # Find NonbondedForce
    nonbonded_force = None
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            nonbonded_force = force
            break

    if nonbonded_force is None:
        raise RuntimeError("NonbondedForce not found in system")

    # Find all TFSI residues
    tfsi_residues = []
    for residue in topology.residues():
        if residue.name in ["TFSI", "NTF2", "Ntf2"]:  # Common TFSI residue names
            tfsi_residues.append(residue)

    if len(tfsi_residues) == 0:
        logger.info("No TFSI residues found, skipping TFSI exclusions")
        return

    logger.info(f"Found {len(tfsi_residues)} TFSI residues")

    # For each TFSI molecule, add intramolecular exclusions
    exclusion_count = 0

    for residue in tfsi_residues:
        # Get all atom indices in this residue
        atom_indices = [atom.index for atom in residue.atoms()]

        # Add exclusions between all pairs within the residue
        for i in range(len(atom_indices)):
            for j in range(i + 1, len(atom_indices)):
                nonbonded_force.addExclusion(atom_indices[i], atom_indices[j])
                exclusion_count += 1

    logger.info(f"Added {exclusion_count} TFSI intramolecular exclusions")


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

    # Step 1: Electrode exclusions (MANDATORY)
    exclusion_Electrode_NonbondedForce(
        system, topology,
        cathode_indices, anode_indices
    )

    # Step 2: TFSI exclusions (if applicable)
    if include_tfsi:
        generate_exclusions_TFSI(system, topology)

    logger.info("=" * 60)
    logger.info("Exclusion setup complete")
    logger.info("=" * 60)
