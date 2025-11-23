"""
ConstantV System Builder - Factory Pattern

This module provides a factory class for building OpenMM systems with
the ConstantV plugin. It encapsulates all the complexity of:
    - Loading PDB/force fields
    - Adding extra particles (Drude oscillators)
    - Configuring PME
    - Setting up ConstantVForce
    - Identifying electrode/electrolyte atoms

Design Philosophy:
    - Automatic: addExtraParticles() is called automatically for polarizable systems
    - Forced PME: NonbondedMethod.PME is enforced (required for ConstantV physics)
    - Strict Validation: All inputs are validated before system creation

Corresponds to: MM_classes.py::__init__() and initialize_electrodes()
"""

from typing import List, Dict, Tuple, Set
import logging
from pathlib import Path

import openmm
from openmm import app
from openmm import unit

from ..models.config import SystemConfig, ElectrodeConfig
from ..constants import (
    DEFAULT_CUTOFF_NM,
    DEFAULT_PME_ERROR_TOLERANCE,
    CONSTANTV_FORCE_GROUP,
)

logger = logging.getLogger(__name__)


class ConstantVSystemBuilder:
    """
    Factory class for building OpenMM systems with ConstantV plugin.

    This class implements the "Factory" design pattern to encapsulate
    the complex system initialization logic.

    Attributes:
        config: System configuration (validated Pydantic model)
        pdb: Loaded PDB file
        modeller: OpenMM Modeller object
        forcefield: OpenMM ForceField object
        system: OpenMM System object
        topology: OpenMM Topology object
        is_polarizable: Whether system uses polarizable force field
        cathode_indices: List of cathode atom indices
        anode_indices: List of anode atom indices
        electrolyte_indices: List of electrolyte atom indices
    """

    def __init__(self, config: SystemConfig):
        """
        Initialize system builder with validated configuration.

        Args:
            config: System configuration (Pydantic model)

        Raises:
            FileNotFoundError: If any input file is missing
            ValueError: If configuration is invalid
        """
        self.config = config

        # Validate input files exist
        self._validate_input_files()

        # Core OpenMM objects (initialized in build())
        self.pdb: app.PDBFile | None = None
        self.modeller: app.Modeller | None = None
        self.forcefield: app.ForceField | None = None
        self.system: openmm.System | None = None
        self.topology: app.Topology | None = None

        # System properties
        self.is_polarizable: bool = False

        # Atom indices (populated during build())
        self.cathode_indices: List[int] = []
        self.anode_indices: List[int] = []
        self.electrolyte_indices: List[int] = []
        self.buckyball_virtual_indices: List[List[int]] = []
        self.buckyball_real_indices: List[List[int]] = []
        self.nanotube_virtual_indices: List[List[int]] = []
        self.nanotube_real_indices: List[List[int]] = []

        logger.info("ConstantVSystemBuilder initialized with config")

    def _validate_input_files(self) -> None:
        """
        Validate that all input files exist.

        Raises:
            FileNotFoundError: If any file is missing
        """
        all_files: List[str] = [
            *self.config.pdb_files,
            *self.config.residue_xml_files,
            *self.config.forcefield_xml_files,
        ]

        for filepath in all_files:
            if not Path(filepath).exists():
                raise FileNotFoundError(f"Input file not found: {filepath}")

    def build(self) -> Tuple[openmm.System, app.Topology, app.Modeller]:
        """
        Build complete OpenMM system with ConstantV force.

        This method orchestrates the entire system building process:
            1. Load PDB and force field
            2. Add extra particles (if polarizable)
            3. Create OpenMM System
            4. Configure PME
            5. Identify electrode/electrolyte atoms
            6. Add ConstantVForce
            7. Assign force groups

        Returns:
            (system, topology, modeller): Complete OpenMM system ready for simulation

        Corresponds to:
            - MM_classes.py::__init__() (Lines 64-112)
            - MM_classes.py::initialize_electrodes() (Lines 183-220)
        """
        logger.info("Building ConstantV system...")

        # Step 1: Load PDB and force field (Lines 64-75)
        self._load_pdb_and_forcefield()

        # Step 2: Add extra particles (Line 77)
        # MANDATORY: This is called automatically for polarizable force fields
        self._add_extra_particles()

        # Step 3: Create OpenMM System (Line 100)
        self._create_system()

        # Step 4: Configure PME (Lines 111-112)
        # MANDATORY: ConstantV requires PME for long-range electrostatics
        self._configure_pme()

        # Step 5: Identify electrode and electrolyte atoms
        self._identify_electrodes()
        self._identify_electrolytes()

        # Step 6: Add ConstantVForce
        self._add_constantv_force()

        # Step 7: Assign force groups (防止recursion)
        self._assign_force_groups()

        logger.info("System build complete")
        return self.system, self.topology, self.modeller

    def _load_pdb_and_forcefield(self) -> None:
        """
        Load PDB file and force field.

        Corresponds to: MM_classes.py::__init__() Lines 64-75
        """
        # Line 66-67: Load bond definitions BEFORE creating PDB object
        # This ensures bonds are defined when PDBFile calls createStandardBonds()
        for residue_file in self.config.residue_xml_files:
            app.Topology().loadBondDefinitions(residue_file)
            logger.debug(f"Loaded bond definitions from {residue_file}")

        # Line 70: Create PDB object
        self.pdb = app.PDBFile(self.config.pdb_files[0])
        logger.info(f"Loaded PDB: {self.config.pdb_files[0]}")

        # Line 73: Create Modeller
        self.modeller = app.Modeller(self.pdb.topology, self.pdb.positions)

        # Line 75: Create ForceField
        self.forcefield = app.ForceField(*self.config.forcefield_xml_files)
        logger.info(f"Loaded {len(self.config.forcefield_xml_files)} force field files")

    def _add_extra_particles(self) -> None:
        """
        Add extra particles (Drude oscillators) for polarizable force fields.

        MANDATORY: This is called automatically (Line 77).
        User does not need to manually add Drude particles.

        Corresponds to: MM_classes.py::__init__() Line 77
        """
        natoms_before = self.modeller.topology.getNumAtoms()

        # Line 77: modeller.addExtraParticles(self.forcefield)
        self.modeller.addExtraParticles(self.forcefield)

        natoms_after = self.modeller.topology.getNumAtoms()

        # Line 85-87: Detect if system is polarizable
        self.is_polarizable = (natoms_after > natoms_before)

        if self.is_polarizable:
            logger.info(
                f"Polarizable force field detected: "
                f"added {natoms_after - natoms_before} Drude particles"
            )
        else:
            logger.info("Non-polarizable force field detected")

        self.topology = self.modeller.topology

    def _create_system(self) -> None:
        """
        Create OpenMM System object with proper settings.

        Corresponds to: MM_classes.py::__init__() Line 100
        """
        # Line 100: self.system = self.forcefield.createSystem(...)
        self.system = self.forcefield.createSystem(
            self.modeller.topology,
            nonbondedCutoff=self.config.cutoff_nm * unit.nanometer,
            constraints=app.HBonds,
            rigidWater=True,
        )

        logger.info(
            f"Created System with {self.system.getNumParticles()} particles, "
            f"cutoff={self.config.cutoff_nm} nm"
        )

    def _configure_pme(self) -> None:
        """
        Configure PME (Particle Mesh Ewald) for long-range electrostatics.

        MANDATORY: ConstantV requires PME. This method FORCES PME if not set.

        Corresponds to: MM_classes.py::__init__() Lines 111-112
        """
        # Line 102: Get NonbondedForce
        nonbonded_force = None
        for force in self.system.getForces():
            if isinstance(force, openmm.NonbondedForce):
                nonbonded_force = force
                break

        if nonbonded_force is None:
            raise RuntimeError("NonbondedForce not found in system")

        # Line 111: Force PME method
        # MANDATORY: self.nbondedForce.setNonbondedMethod(NonbondedForce.PME)
        nonbonded_force.setNonbondedMethod(openmm.NonbondedForce.PME)
        logger.info("Forced NonbondedMethod to PME (required for ConstantV)")

        # Set PME error tolerance
        nonbonded_force.setEwaldErrorTolerance(DEFAULT_PME_ERROR_TOLERANCE)

    def _identify_electrodes(self) -> None:
        """
        Identify cathode and anode atom indices.

        Corresponds to: Fixed_Voltage_routines.py::Conductor_Virtual.__init__()
        """
        self.cathode_indices = self._identify_electrode_atoms(self.config.cathode)
        self.anode_indices = self._identify_electrode_atoms(self.config.anode)

        logger.info(
            f"Identified {len(self.cathode_indices)} cathode atoms, "
            f"{len(self.anode_indices)} anode atoms"
        )

    def _identify_electrode_atoms(self, electrode_config: ElectrodeConfig) -> List[int]:
        """
        Identify electrode atom indices based on configuration.

        Corresponds to: Fixed_Voltage_routines.py::Conductor_Virtual.__init__()
        Lines 111-156

        Args:
            electrode_config: Electrode configuration

        Returns:
            List of atom indices for this electrode
        """
        atom_indices: List[int] = []

        if electrode_config.by_chain:
            # Line 112-136: Initialize by chain
            for chain in self.topology.chains():
                if chain.index == electrode_config.identifier:
                    for atom in chain.atoms():
                        if atom.element.symbol not in electrode_config.exclude_elements:
                            atom_indices.append(atom.index)
        else:
            # Line 138-151: Initialize by residue
            for residue in self.topology.residues():
                if residue.name == electrode_config.identifier:
                    for atom in residue.atoms():
                        if atom.element.symbol not in electrode_config.exclude_elements:
                            atom_indices.append(atom.index)

        # Line 153-156: Validate
        if len(atom_indices) == 0:
            raise ValueError(
                f"Could not find electrode atoms for identifier '{electrode_config.identifier}'. "
                f"Please check electrode configuration."
            )

        return atom_indices

    def _identify_electrolytes(self) -> None:
        """
        Identify electrolyte atom indices.

        Uses residue size heuristic: residues with < natom_cutoff atoms are electrolyte.

        Corresponds to: MM_classes.py::initialize_electrolyte() Lines 256-279
        """
        # Line 258: Create set of electrolyte residue names (for caching)
        electrolyte_names: Set[str] = set()

        # Line 259-279: Loop over residues
        for residue in self.topology.residues():
            if residue.name in electrolyte_names:
                # Line 263-267: Already know this is electrolyte
                for atom in residue.atoms():
                    self.electrolyte_indices.append(atom.index)
            else:
                # Line 269-272: Count atoms in residue
                natoms = sum(1 for _ in residue.atoms())

                # Line 273: Check if electrolyte (size heuristic)
                if natoms < self.config.natom_cutoff:
                    # Line 274-279: Add to electrolyte list
                    electrolyte_names.add(residue.name)
                    for atom in residue.atoms():
                        self.electrolyte_indices.append(atom.index)

        logger.info(f"Identified {len(self.electrolyte_indices)} electrolyte atoms")

    def _add_constantv_force(self) -> None:
        """
        Add ConstantVForce to the system.

        This will be implemented using the ConstantVForceWrapper class.
        """
        # This method will use the Force Wrapper (to be implemented next)
        # For now, this is a placeholder
        pass

    def _assign_force_groups(self) -> None:
        """
        Assign force groups to prevent recursion during SCF iterations.

        ConstantVForce is assigned to CONSTANTV_FORCE_GROUP (31) to prevent
        it from being included when calculating forces during charge updates.

        Corresponds to: CudaConstantVKernels.cu::execute() Line 1053
        """
        for i, force in enumerate(self.system.getForces()):
            if type(force).__name__ == "ConstantVForce":
                force.setForceGroup(CONSTANTV_FORCE_GROUP)
                logger.debug(f"Assigned ConstantVForce to force group {CONSTANTV_FORCE_GROUP}")
            else:
                force.setForceGroup(i)
