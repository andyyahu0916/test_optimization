"""
ConstantV System Builder - Factory Pattern

This module provides a factory class for building OpenMM systems with
the ConstantV Native Core Integration. It encapsulates all the complexity of:
    - Loading PDB/force fields
    - Adding extra particles (Drude oscillators)
    - Configuring PME
    - Creating ConstantVDrudeLangevinIntegrator
    - Identifying electrode/electrolyte atoms
    - Computing conductor geometries (Buckyball/Nanotube)

Design Philosophy:
    - Automatic: addExtraParticles() is called automatically for polarizable systems
    - Forced PME: NonbondedMethod.PME is enforced (required for ConstantV physics)
    - Strict Validation: All inputs are validated before system creation
    - Native Core: Uses ConstantVDrudeLangevinIntegrator (NOT Force-based)

Corresponds to: MM_classes.py::__init__() and initialize_electrodes()
"""

from typing import List, Dict, Tuple, Set, Optional
import json
import logging
from pathlib import Path
import numpy as np

import openmm
from openmm import app
from openmm import unit

try:  # pragma: no cover - optional native module
    import constantv
except ImportError:  # pragma: no cover - handled at runtime
    constantv = None

from ..models.config import SystemConfig, ElectrodeConfig, BuckyballConfig, NanotubeConfig
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
        self.constantv_force: "constantv.ConstantVForce" | None = None

        # System properties
        self.is_polarizable: bool = False
        self._nonbonded_force: openmm.NonbondedForce | None = None
        self._custom_nonbonded_force: openmm.CustomNonbondedForce | None = None
        self._drude_force: openmm.DrudeForce | None = None
        self._custom_bond_force: openmm.CustomBondForce | None = None
        self._particle_charges: List[float] = []
        self._water_groups_configured: bool = False
        self.planar_area_nm2: Optional[float] = None
        self.z_cathode_nm: Optional[float] = None
        self.z_anode_nm: Optional[float] = None
        self.Lcell_nm: Optional[float] = None
        self.Lgap_nm: Optional[float] = None

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

    def build(
        self,
        attach_constantv_force: bool = True,
    ) -> Tuple[openmm.System, app.Topology, app.Modeller]:
        """
        Build complete OpenMM system with ConstantV force.

        This method orchestrates the entire system building process:
            1. Load PDB and force field
            2. Add extra particles (if polarizable)
            3. Create OpenMM System
            4. Configure PME
            5. Identify electrode/electrolyte atoms
            6. Compute electrode geometry metadata
            7. (Optional) Add ConstantVForce
            8. Assign force groups

        Returns:
            (system, topology, modeller): Complete OpenMM system ready for simulation

        Args:
            attach_constantv_force: If True (default), attach constantv.ConstantVForce
                to the system immediately. Set to False when you only need the raw
                system object (e.g., to export kernel configs without the plugin
                installed).

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
        self._collect_conductors()

        # Step 6: Compute cell geometry (Lgap, Lcell, electrode areas)
        self._compute_cell_geometry()

        # Step 7: Optionally attach ConstantVForce immediately
        if attach_constantv_force:
            self.create_constantv_force()

        # Step 7b: Apply electrode/SAPT exclusions prior to force-group assignment
        self._apply_exclusion_workflow()

        # Step 8: Assign force groups
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

        self._cache_force_handles()

    def _cache_force_handles(self) -> None:
        """Locate and store references to frequently used OpenMM forces."""
        if self.system is None:
            raise RuntimeError("System must be created before caching forces")

        self._nonbonded_force = None
        self._custom_nonbonded_force = None
        self._drude_force = None
        self._custom_bond_force = None

        for force in self.system.getForces():
            if isinstance(force, openmm.NonbondedForce):
                self._nonbonded_force = force
            elif isinstance(force, openmm.CustomNonbondedForce) and self._custom_nonbonded_force is None:
                self._custom_nonbonded_force = force
            elif isinstance(force, openmm.DrudeForce):
                self._drude_force = force
            elif isinstance(force, openmm.CustomBondForce) and self._custom_bond_force is None:
                self._custom_bond_force = force

        if self._nonbonded_force is None:
            raise RuntimeError("NonbondedForce not found in system")

    def _configure_pme(self) -> None:
        """
        Configure PME (Particle Mesh Ewald) for long-range electrostatics.

        MANDATORY: ConstantV requires PME. This method FORCES PME if not set.

        Corresponds to: MM_classes.py::__init__() Lines 111-112
        """
        if self._nonbonded_force is None:
            raise RuntimeError("NonbondedForce not cached before PME configuration")
        nonbonded_force = self._nonbonded_force

        # Line 111: Force PME method
        # MANDATORY: self.nbondedForce.setNonbondedMethod(NonbondedForce.PME)
        nonbonded_force.setNonbondedMethod(openmm.NonbondedForce.PME)
        logger.info("Forced NonbondedMethod to PME (required for ConstantV)")

        # Set PME error tolerance and cache charges for ConstantV metadata
        nonbonded_force.setEwaldErrorTolerance(DEFAULT_PME_ERROR_TOLERANCE)
        self._particle_charges = self._extract_particle_charges(nonbonded_force)

    def _extract_particle_charges(
        self,
        nonbonded_force: openmm.NonbondedForce
    ) -> List[float]:
        """Cache particle charges (elementary charge units)."""
        charges: List[float] = []
        for idx in range(nonbonded_force.getNumParticles()):
            charge, _, _ = nonbonded_force.getParticleParameters(idx)
            charges.append(charge.value_in_unit(unit.elementary_charge))
        return charges

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

    def _collect_conductors(self) -> None:
        """Resolve conductor atom indices for later force wiring and exclusions."""
        self.buckyball_virtual_indices = []
        self.buckyball_real_indices = []
        self.nanotube_virtual_indices = []
        self.nanotube_real_indices = []

        if self.topology is None:
            raise RuntimeError("Topology required before collecting conductor indices")

        for config in self.config.buckyballs:
            virtual_indices = self._identify_conductor_atoms(
                config.virtual_chain_index,
                config.exclude_elements,
            )
            real_indices = self._identify_conductor_atoms(
                config.real_chain_index,
                config.exclude_elements,
            )
            self.buckyball_virtual_indices.append(virtual_indices)
            self.buckyball_real_indices.append(real_indices)

        for config in self.config.nanotubes:
            virtual_indices = self._identify_conductor_atoms(
                config.virtual_chain_index,
                config.exclude_elements,
            )
            real_indices = self._identify_conductor_atoms(
                config.real_chain_index,
                config.exclude_elements,
            )
            self.nanotube_virtual_indices.append(virtual_indices)
            self.nanotube_real_indices.append(real_indices)

    def _iter_conductors(self):
        """Yield (config, virtual_indices, real_indices) for each conductor."""
        for idx, config in enumerate(self.config.buckyballs):
            yield config, self.buckyball_virtual_indices[idx], self.buckyball_real_indices[idx]
        for idx, config in enumerate(self.config.nanotubes):
            yield config, self.nanotube_virtual_indices[idx], self.nanotube_real_indices[idx]

    # ═══════════════════════════════════════════════════════════════════════════
    # Geometry Calculations (matching ConstantVGeometry.h)
    # ═══════════════════════════════════════════════════════════════════════════

    def _compute_sphere_center(self, positions: List[openmm.Vec3]) -> openmm.Vec3:
        """Compute geometric center of sphere (average of all positions)."""
        center = np.array([0.0, 0.0, 0.0])
        for pos in positions:
            center += np.array([pos.x, pos.y, pos.z])
        center /= len(positions)
        return openmm.Vec3(center[0], center[1], center[2])

    def _compute_sphere_radius(self, positions: List[openmm.Vec3], center: openmm.Vec3) -> float:
        """Compute average radius from center."""
        radius_sum = 0.0
        for pos in positions:
            dx = pos.x - center.x
            dy = pos.y - center.y
            dz = pos.z - center.z
            radius_sum += np.sqrt(dx*dx + dy*dy + dz*dz)
        return radius_sum / len(positions)

    def _compute_sphere_normals(self, positions: List[openmm.Vec3], center: openmm.Vec3) -> List[openmm.Vec3]:
        """Compute outward normal vectors (position - center, normalized)."""
        normals = []
        for pos in positions:
            dx = pos.x - center.x
            dy = pos.y - center.y
            dz = pos.z - center.z
            r = np.sqrt(dx*dx + dy*dy + dz*dz)
            normals.append(openmm.Vec3(dx/r, dy/r, dz/r))
        return normals

    def _compute_sphere_area_per_atom(self, radius: float, num_atoms: int) -> float:
        """Compute surface area per atom: 4πr² / N."""
        return 4.0 * np.pi * radius * radius / num_atoms

    def _find_contact_electrode_atom(
        self,
        conductor_center: openmm.Vec3,
        electrode_indices: List[int],
        positions: List[openmm.Vec3]
    ) -> Tuple[int, float]:
        """Find closest electrode atom to conductor center."""
        min_distance = float('inf')
        contact_atom = electrode_indices[0]

        for idx in electrode_indices:
            pos = positions[idx]
            dx = pos.x - conductor_center.x
            dy = pos.y - conductor_center.y
            dz = pos.z - conductor_center.z
            distance = np.sqrt(dx*dx + dy*dy + dz*dz)

            if distance < min_distance:
                min_distance = distance
                contact_atom = idx

        return contact_atom, min_distance

    def _add_conductors_to_force(self, force) -> None:
        """Register Buckyball/Nanotube conductors with ConstantVForce."""
        positions = self.modeller.getPositions()

        if (self.config.buckyballs and not self.buckyball_virtual_indices) or (
            self.config.nanotubes and not self.nanotube_virtual_indices
        ):
            self._collect_conductors()

        for i, bucky_config in enumerate(self.config.buckyballs):
            logger.info("Adding Buckyball conductor %d", i + 1)
            virtual_indices = self.buckyball_virtual_indices[i]
            real_indices = self.buckyball_real_indices[i]

            virtual_positions = [positions[idx] for idx in virtual_indices]
            center = self._compute_sphere_center(virtual_positions)
            radius = self._compute_sphere_radius(virtual_positions, center)
            normals = self._compute_sphere_normals(virtual_positions, center)
            area_per_atom = self._compute_sphere_area_per_atom(radius, len(virtual_indices))

            electrode_indices = (
                self.cathode_indices if bucky_config.electrode_type == "cathode" else self.anode_indices
            )
            contact_atom, contact_distance = self._find_contact_electrode_atom(
                center,
                electrode_indices,
                positions,
            )

            force.addBuckyballConductor(
                virtual_indices,
                real_indices,
                bucky_config.electrode_type,
                self.config.voltage_volts,
            )

            logger.debug(
                "  Buckyball center=%s radius=%.3f nm contact_atom=%d contact_distance=%.3f nm",
                center,
                radius,
                contact_atom,
                contact_distance,
            )

        for i, tube_config in enumerate(self.config.nanotubes):
            logger.info("Adding Nanotube conductor %d", i + 1)
            virtual_indices = self.nanotube_virtual_indices[i]
            real_indices = self.nanotube_real_indices[i]

            virtual_positions = [positions[idx] for idx in virtual_indices]
            center = self._compute_sphere_center(virtual_positions)
            axis = openmm.Vec3(*tube_config.axis)
            electrode_indices = (
                self.cathode_indices if tube_config.electrode_type == "cathode" else self.anode_indices
            )
            contact_atom, contact_distance = self._find_contact_electrode_atom(
                center,
                electrode_indices,
                positions,
            )

            force.addNanotubeConductor(
                virtual_indices,
                real_indices,
                tube_config.electrode_type,
                self.config.voltage_volts,
                axis,
            )

            logger.debug(
                "  Nanotube center=%s axis=%s contact_atom=%d contact_distance=%.3f nm",
                center,
                axis,
                contact_atom,
                contact_distance,
            )

    # ═══════════════════════════════════════════════════════════════════════════
    # Exclusion Handling (Delegated to utils/exclusions.py)
    # ═══════════════════════════════════════════════════════════════════════════

    def _apply_exclusion_workflow(self) -> None:
        """Apply electrode, conductor, and SAPT-FF exclusions using utils/exclusions.py.

        NOTE: This delegates to utils/exclusions.py (SINGLE SOURCE OF TRUTH).
        Do NOT duplicate exclusion logic here.
        """
        if self.system is None:
            raise RuntimeError("System not initialized before exclusion workflow")

        # Import from utils package (single source of truth)
        # NOTE: Requires PYTHONPATH to include project root:
        #   export PYTHONPATH=/home/andy/test_optimization:$PYTHONPATH
        from utils import add_all_exclusions

        # Prepare conductor configs for exclusions
        conductor_configs = []
        for config, virtual_indices, real_indices in self._iter_conductors():
            conductor_configs.append({
                'virtual_indices': virtual_indices,
                'real_indices': real_indices
            })

        # Delegate to utils/exclusions.py
        add_all_exclusions(
            self.system,
            self.topology,
            self.cathode_indices,
            self.anode_indices,
            include_tfsi=self.config.sapt_ff_exclusions,
            include_water=self.config.hybrid_water_model or self.config.sapt_ff_exclusions,
            water_residue_name=self.config.water_residue_name,
            conductor_configs=conductor_configs if len(conductor_configs) > 0 else None
        )

    def _compute_electrode_area(self, electrode_indices: List[int]) -> float:
        """Compute electrode area with fallback when box vectors are missing."""
        if self.topology.getPeriodicBoxVectors() is None:
            logger.warning(
                "No periodic box vectors found; falling back to 0.1 nm² per atom estimate."
            )
            return max(len(electrode_indices), 1) * 0.1
        return self._compute_planar_area_nm2()

    def _compute_planar_area_nm2(self) -> float:
        """Return planar electrode area from the A×B cross product (nm²)."""
        box_vectors = self.topology.getPeriodicBoxVectors()
        if box_vectors is None:
            raise RuntimeError("Topology does not define periodic box vectors.")

        a = np.array([box_vectors[0].x, box_vectors[0].y, box_vectors[0].z])
        b = np.array([box_vectors[1].x, box_vectors[1].y, box_vectors[1].z])
        cross = np.cross(a, b)
        area_nm2 = float(np.sqrt(np.dot(cross, cross)))
        logger.debug("Planar electrode area = %.4f nm²", area_nm2)
        return area_nm2

    def _average_z(self, positions: List[openmm.Vec3], atom_indices: List[int]) -> float:
        """Average z-position (nm) for provided atom indices."""
        if not atom_indices:
            raise ValueError("Cannot compute average z with an empty atom list.")
        return float(np.mean([positions[idx].z for idx in atom_indices]))

    def _compute_cell_geometry(self) -> None:
        """Compute planar area, electrode z positions, and Lgap/Lcell values."""
        if self.topology is None or self.modeller is None:
            raise RuntimeError("Topology/modeller not initialized before geometry computation.")

        positions = self.modeller.getPositions()
        box_vectors = self.topology.getPeriodicBoxVectors()
        if box_vectors is None:
            raise RuntimeError("Periodic box vectors are required for ConstantV simulations.")

        self.planar_area_nm2 = self._compute_planar_area_nm2()
        self.z_cathode_nm = self._average_z(positions, self.cathode_indices)
        self.z_anode_nm = self._average_z(positions, self.anode_indices)

        cell_vector = np.array([box_vectors[2].x, box_vectors[2].y, box_vectors[2].z])
        box_length_nm = float(np.linalg.norm(cell_vector))
        self.Lcell_nm = abs(self.z_anode_nm - self.z_cathode_nm)
        self.Lgap_nm = max(box_length_nm - self.Lcell_nm, 1e-6)

        logger.info(
            "Cell geometry: Lcell=%.3f nm, Lgap=%.3f nm, area=%.2f nm²",
            self.Lcell_nm,
            self.Lgap_nm,
            self.planar_area_nm2,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # ConstantVForce Attachment + Kernel Metadata
    # ═══════════════════════════════════════════════════════════════════════════

    def create_constantv_force(self) -> "constantv.ConstantVForce":
        """Attach and return a configured ConstantVForce instance."""
        if self.system is None or self.topology is None:
            raise RuntimeError("build() must be called before creating ConstantVForce.")

        if self.constantv_force is not None:
            return self.constantv_force

        if constantv is None:
            raise RuntimeError(
                "constantv module is not available. Install the ConstantV plugin "
                "or call build(attach_constantv_force=False) to skip attachment."
            )

        if self.planar_area_nm2 is None or self.Lgap_nm is None or self.Lcell_nm is None:
            self._compute_cell_geometry()

        logger.info("Attaching ConstantVForce (voltage=%.2f V)...", self.config.voltage_volts)

        cathode_count = len(self.cathode_indices)
        anode_count = len(self.anode_indices)
        if cathode_count == 0 or anode_count == 0:
            raise ValueError("Electrode atom lists cannot be empty when building ConstantVForce.")

        cathode_area_per_atom = self.planar_area_nm2 / cathode_count
        anode_area_per_atom = self.planar_area_nm2 / anode_count

        force = constantv.ConstantVForce()
        force.setVoltage(self.config.voltage_volts)
        force.setLgap(self.Lgap_nm)
        force.setLcell(self.Lcell_nm)
        force.setTotalArea(self.planar_area_nm2)
        force.setZCathode(self.z_cathode_nm)
        force.setZAnode(self.z_anode_nm)
        force.setNumIterations(self.config.scf_iterations)

        for idx in self.cathode_indices:
            force.addCathodeAtom(idx, cathode_area_per_atom)
        for idx in self.anode_indices:
            force.addAnodeAtom(idx, anode_area_per_atom)

        if not self._particle_charges:
            if self._nonbonded_force is None:
                raise RuntimeError("NonbondedForce not cached; cannot determine particle charges.")
            self._particle_charges = self._extract_particle_charges(self._nonbonded_force)

        for idx in self.electrolyte_indices:
            force.addElectrolyteAtom(idx, self._particle_charges[idx])

        self._add_conductors_to_force(force)

        self.system.addForce(force)
        self.constantv_force = force
        logger.info("ConstantVForce attached successfully")
        return force

    def build_kernel_config(
        self,
        gpu_architecture: str,
        output_path: Optional[str] = None,
    ) -> Dict[str, object]:
        """Return (and optionally persist) a kernel_compiler config dict."""
        if self.planar_area_nm2 is None or self.Lgap_nm is None or self.Lcell_nm is None:
            self._compute_cell_geometry()

        cathode_area_per_atom = self.planar_area_nm2 / max(len(self.cathode_indices), 1)
        anode_area_per_atom = self.planar_area_nm2 / max(len(self.anode_indices), 1)

        config_dict = {
            "_source_file": str(self.config.pdb_files[0]),
            "voltage_volts": self.config.voltage_volts,
            "Lgap_nm": self.Lgap_nm,
            "Lcell_nm": self.Lcell_nm,
            "total_area_nm2": self.planar_area_nm2,
            "z_cathode_nm": self.z_cathode_nm,
            "z_anode_nm": self.z_anode_nm,
            "num_cathodes": len(self.cathode_indices),
            "num_anodes": len(self.anode_indices),
            "num_electrolytes": len(self.electrolyte_indices),
            "num_buckyballs": len(self.config.buckyballs),
            "num_nanotubes": len(self.config.nanotubes),
            "cathode_indices": self.cathode_indices,
            "cathode_areas": [cathode_area_per_atom] * len(self.cathode_indices),
            "anode_indices": self.anode_indices,
            "anode_areas": [anode_area_per_atom] * len(self.anode_indices),
            "electrolyte_indices": self.electrolyte_indices,
            "gpu_architecture": gpu_architecture,
        }

        if output_path is not None:
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump(config_dict, handle, indent=2)
            logger.info("Kernel compiler config written to %s", output_path)

        return config_dict

    def _identify_conductor_atoms(self, chain_index: int, exclude_elements: Tuple[str, ...]) -> List[int]:
        """Identify conductor atoms by chain index."""
        atom_indices = []
        for chain in self.topology.chains():
            if chain.index == chain_index:
                for atom in chain.atoms():
                    if atom.element.symbol not in exclude_elements:
                        atom_indices.append(atom.index)

        if len(atom_indices) == 0:
            raise ValueError(f"No atoms found for chain index {chain_index}")

        return atom_indices

    def _assign_force_groups(self) -> None:
        """
        Assign force groups, preserving ConstantVForce in group 31.

        FIX C2: ConstantVForce must remain in CONSTANTV_FORCE_GROUP (31)
        to prevent SCF timing issues. Other forces use groups 0-30.
        """
        from ..constants import CONSTANTV_FORCE_GROUP
        
        other_force_idx = 0
        for force in self.system.getForces():
            # Check if this is ConstantVForce by class name (avoid import issues)
            force_class_name = force.__class__.__name__
            if 'ConstantV' in force_class_name:
                # FIX C2: Preserve ConstantVForce in its dedicated group
                force.setForceGroup(CONSTANTV_FORCE_GROUP)
            else:
                # Assign groups 0-30 to other forces
                force.setForceGroup(other_force_idx % 31)
                other_force_idx += 1

    def seed_electrode_charges(self, context: "openmm.Context", threshold: float = 1e-6) -> None:
        """
        Initialize electrode charges to small non-zero values.

        This improves SCF convergence by avoiding division by zero in the first
        iteration when computing Ez_external = F_z / q_old.

        Corresponds to: Original implementation's pre-simulation SCF warmup

        Args:
            context: OpenMM context (must be created with the system from this builder)
            threshold: Small charge value to seed (default: 1e-6 elementary charge)

        Example:
            >>> builder = ConstantVSystemBuilder(config)
            >>> system, topology, modeller = builder.build()
            >>> context = openmm.Context(system, integrator)
            >>> builder.seed_electrode_charges(context)  # Seed before simulation
            >>> integrator.step(1000000)
        """
        if self._nonbonded_force is None:
            raise RuntimeError("NonbondedForce not available for charge seeding")

        logger.info(
            "Seeding electrode charges to ±%.1e for better SCF convergence",
            threshold
        )

        # Set cathode atoms to +threshold
        for idx in self.cathode_indices:
            charge, sigma, epsilon = self._nonbonded_force.getParticleParameters(idx)
            self._nonbonded_force.setParticleParameters(idx, threshold, sigma, epsilon)

        # Set anode atoms to -threshold
        for idx in self.anode_indices:
            charge, sigma, epsilon = self._nonbonded_force.getParticleParameters(idx)
            self._nonbonded_force.setParticleParameters(idx, -threshold, sigma, epsilon)

        # Update context with new charges
        self._nonbonded_force.updateParametersInContext(context)

        logger.info(
            "Seeded %d cathode atoms (+%.1e) and %d anode atoms (-%.1e)",
            len(self.cathode_indices),
            threshold,
            len(self.anode_indices),
            threshold,
        )
