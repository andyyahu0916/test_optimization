#!/usr/bin/env python3
"""
Production Simulation Script for ConstantV Native Core

This script implements the complete workflow for running production
simulations with the ConstantV Native C++ Integrator.

Workflow (matching original run_openMM.py):
    1. Load PDB and Force Field
    2. Add Drude Particles (MANDATORY for polarizable FF)
    3. Create System with PME (MANDATORY for ConstantV)
    4. Identify Electrode and Electrolyte Atoms
    5. Add Exclusions (CRITICAL - prevents Coulomb explosions)
    6. Create ConstantVDrudeLangevinIntegrator (Native Core)
    7. Configure Integrator (electrodes, conductors, parameters)
    8. Create Simulation and Add Reporters
    9. Run Equilibration and Production

Author: Claude (Anthropic) + User
Date: 2025-11-25
Status: PRODUCTION READY
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Set
import numpy as np

import openmm
from openmm import app, unit

try:
    import constantv
except ImportError:  # pragma: no cover - load error handled at runtime
    constantv = None

# Import exclusion utilities
sys.path.insert(0, str(Path(__file__).parent))
from utils.exclusions import add_all_exclusions
from openmm_constantv.reporters.electrode_charge_reporter import ElectrodeChargeReporter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('production.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ConstantVProductionSimulation:
    """
    Production simulation manager for ConstantV Native Core.

    This class encapsulates the complete workflow for setting up and
    running a ConstantV simulation with the native C++ integrator.
    """

    def __init__(self, config_file: str):
        """
        Initialize simulation from JSON configuration.

        Args:
            config_file: Path to JSON configuration file
        """
        logger.info("=" * 80)
        logger.info("ConstantV Production Simulation - Native Core Integration")
        logger.info("=" * 80)

        # Load configuration
        with open(config_file, 'r') as f:
            self.config = json.load(f)

        logger.info(f"Loaded configuration from {config_file}")

        # Core OpenMM objects
        self.pdb: app.PDBFile = None
        self.modeller: app.Modeller = None
        self.forcefield: app.ForceField = None
        self.system: openmm.System = None
        self.topology: app.Topology = None
        self.integrator: "constantv.ConstantVDrudeLangevinIntegrator" | None = None
        self.simulation: app.Simulation | None = None

        # Electrode/electrolyte atom indices
        self.cathode_indices: List[int] = []
        self.anode_indices: List[int] = []
        self.electrolyte_indices: List[int] = []
        self.conductor_charge_indices: List[List[int]] = []

        # Conductor configurations (for exclusions)
        self.conductor_exclusion_configs: List[Dict] = []

        # System properties
        self.is_polarizable: bool = False
        self._particle_charges: List[float] = []

        # Geometry parameters (computed during integrator creation)
        self._Lgap: float = 0.0
        self._Lcell: float = 0.0
        self._total_area: float = 0.0

    # ═══════════════════════════════════════════════════════════════════════
    # Step 1: Load PDB and Force Field
    # ═══════════════════════════════════════════════════════════════════════

    def load_system(self) -> None:
        """
        Load PDB file and force field.

        Corresponds to: MM_classes.py::__init__() Lines 64-75
        """
        logger.info("Step 1: Loading PDB and Force Field...")

        # Load bond definitions FIRST (before creating PDB object)
        for residue_file in self.config['system']['residue_xml_files']:
            app.Topology().loadBondDefinitions(residue_file)
            logger.info(f"  Loaded bond definitions: {residue_file}")

        # Load PDB
        pdb_file = self.config['system']['pdb_files'][0]
        self.pdb = app.PDBFile(pdb_file)
        logger.info(f"  Loaded PDB: {pdb_file} ({self.pdb.topology.getNumAtoms()} atoms)")

        # Create Modeller
        self.modeller = app.Modeller(self.pdb.topology, self.pdb.positions)

        # Load Force Field
        self.forcefield = app.ForceField(*self.config['system']['forcefield_xml_files'])
        logger.info(f"  Loaded {len(self.config['system']['forcefield_xml_files'])} force field files")

    # ═══════════════════════════════════════════════════════════════════════
    # Step 2: Add Drude Particles (MANDATORY)
    # ═══════════════════════════════════════════════════════════════════════

    def add_drude_particles(self) -> None:
        """
        Add Drude oscillators for polarizable force fields.

        CRITICAL: This is MANDATORY. The native integrator expects Drude
        particles to exist in the system. Without this call, the simulation
        will fail.

        Corresponds to: MM_classes.py::__init__() Line 77
        """
        logger.info("Step 2: Adding Drude Particles...")

        natoms_before = self.modeller.topology.getNumAtoms()

        # MANDATORY: Add Drude particles
        self.modeller.addExtraParticles(self.forcefield)

        natoms_after = self.modeller.topology.getNumAtoms()
        self.is_polarizable = (natoms_after > natoms_before)

        if self.is_polarizable:
            logger.info(
                f"  ✓ Polarizable force field detected: "
                f"added {natoms_after - natoms_before} Drude particles"
            )
        else:
            logger.info("  ✓ Non-polarizable force field detected (no Drudes added)")

        self.topology = self.modeller.topology

    # ═══════════════════════════════════════════════════════════════════════
    # Step 3: Create System with PME (MANDATORY)
    # ═══════════════════════════════════════════════════════════════════════

    def create_system(self) -> None:
        """
        Create OpenMM System with PME electrostatics.

        CRITICAL: PME is MANDATORY for ConstantV. The physics requires
        long-range electrostatics to be handled correctly.

        Corresponds to: MM_classes.py::__init__() Lines 100-112
        """
        logger.info("Step 3: Creating System with PME...")

        cutoff_nm = self.config['simulation_parameters']['cutoff_nm']

        # Create system
        self.system = self.forcefield.createSystem(
            self.modeller.topology,
            nonbondedMethod=app.PME,  # MANDATORY: Must use PME
            nonbondedCutoff=cutoff_nm * unit.nanometer,
            constraints=app.HBonds,
            rigidWater=True
        )

        logger.info(
            f"  ✓ System created: {self.system.getNumParticles()} particles, "
            f"cutoff={cutoff_nm} nm"
        )

        # Force PME if not already set (defensive programming)
        for force in self.system.getForces():
            if isinstance(force, openmm.NonbondedForce):
                force.setNonbondedMethod(openmm.NonbondedForce.PME)
                force.setEwaldErrorTolerance(1e-5)
                logger.info("  ✓ Forced NonbondedMethod to PME (required for ConstantV)")
                self._particle_charges = self._extract_particle_charges(force)

    def _extract_particle_charges(self, nonbonded_force: openmm.NonbondedForce) -> List[float]:
        """Return particle charges (in elementary charge units)."""
        charges: List[float] = []
        for idx in range(nonbonded_force.getNumParticles()):
            charge, _, _ = nonbonded_force.getParticleParameters(idx)
            charges.append(charge.value_in_unit(unit.elementary_charge))
        return charges

    # ═══════════════════════════════════════════════════════════════════════
    # Step 4: Identify Electrodes and Electrolytes
    # ═══════════════════════════════════════════════════════════════════════

    def identify_electrodes(self) -> None:
        """
        Identify cathode and anode atom indices from topology.

        Uses residue name matching from configuration.
        """
        logger.info("Step 4a: Identifying Electrode Atoms...")

        cathode_config = self.config['electrodes']['cathode']
        anode_config = self.config['electrodes']['anode']

        # Identify cathode
        self.cathode_indices = self._identify_electrode_atoms(cathode_config)
        logger.info(f"  ✓ Cathode: {len(self.cathode_indices)} atoms (residue '{cathode_config['identifier']}')")

        # Identify anode
        self.anode_indices = self._identify_electrode_atoms(anode_config)
        logger.info(f"  ✓ Anode: {len(self.anode_indices)} atoms (residue '{anode_config['identifier']}')")

    def _identify_electrode_atoms(self, electrode_config: Dict) -> List[int]:
        """Helper method to identify electrode atoms from config."""
        atom_indices = []
        identifier = electrode_config['identifier']
        method = electrode_config.get('identification_method', 'residue_name')
        exclude_elements = set(electrode_config.get('exclude_elements', []))

        if method == 'residue_name':
            for residue in self.topology.residues():
                if residue.name == identifier:
                    for atom in residue.atoms():
                        if atom.element.symbol not in exclude_elements:
                            atom_indices.append(atom.index)
        else:
            raise ValueError(f"Unsupported identification method: {method}")

        if len(atom_indices) == 0:
            raise ValueError(f"Could not find electrode atoms for identifier '{identifier}'")

        return atom_indices

    def identify_electrolytes(self) -> None:
        """
        Identify electrolyte atom indices using residue size heuristic.

        Residues with < natom_cutoff atoms are classified as electrolyte.

        Corresponds to: MM_classes.py::initialize_electrolyte() Lines 256-279
        """
        logger.info("Step 4b: Identifying Electrolyte Atoms...")

        natom_cutoff = self.config['advanced']['natom_cutoff_for_electrolyte']
        electrolyte_names: Set[str] = set()

        for residue in self.topology.residues():
            if residue.name in electrolyte_names:
                # Already know this is electrolyte
                for atom in residue.atoms():
                    self.electrolyte_indices.append(atom.index)
            else:
                # Count atoms in residue
                natoms = sum(1 for _ in residue.atoms())

                # Check if electrolyte (size heuristic)
                if natoms < natom_cutoff:
                    electrolyte_names.add(residue.name)
                    for atom in residue.atoms():
                        self.electrolyte_indices.append(atom.index)

        logger.info(
            f"  ✓ Electrolyte: {len(self.electrolyte_indices)} atoms "
            f"({len(electrolyte_names)} unique residue types)"
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Step 5: Add Exclusions (CRITICAL!)
    # ═══════════════════════════════════════════════════════════════════════

    def add_exclusions(self) -> None:
        """
        Add exclusions to prevent Coulomb explosions.

        CRITICAL: Without these exclusions, electrode atoms will have
        huge Coulomb repulsions and the simulation will explode.

        This MUST be called BEFORE creating the Context.

        NEW: Now supports:
        - Hybrid water model configuration
        - Conductor (Buckyball/Nanotube) exclusions

        Corresponds to: electrode_sapt_exclusions.py + MM_classes.py::generate_exclusions()
        """
        logger.info("Step 5: Adding Exclusions (CRITICAL for Stability)...")

        # Prepare conductor exclusion configs if needed
        self._prepare_conductor_exclusion_configs()

        add_all_exclusions(
            self.system,
            self.topology,
            self.cathode_indices,
            self.anode_indices,
            include_tfsi=self.config['advanced'].get('add_tfsi_exclusions', True),
            include_water=self.config['advanced'].get('hybrid_water_model', False),
            water_residue_name=self.config['advanced'].get('water_residue_name', 'HOH'),
            conductor_configs=self.conductor_exclusion_configs if len(self.conductor_exclusion_configs) > 0 else None
        )

        logger.info("  ✓ Exclusions added successfully")

    # ═══════════════════════════════════════════════════════════════════════
    # Step 6-7: Create and Configure ConstantV Integrator (Integrator-Based Architecture)
    # ═══════════════════════════════════════════════════════════════════════

    def create_integrator(self) -> None:
        """Create the ConstantVDrudeLangevinIntegrator with electrode configuration.

        CRITICAL CHANGE: This now uses the Integrator-based architecture instead of
        Force-based architecture. The integrator handles SCF charge updates internally,
        with configurable frequency to match the original implementation (200 steps).
        """
        if constantv is None:
            raise RuntimeError(
                "constantv module is not available. Build/install the native ConstantV "
                "integration before running production."
            )

        logger.info("Step 6-7: Creating ConstantVDrudeLangevinIntegrator...")

        params = self.config['simulation_parameters']

        # Compute geometry parameters needed for integrator initialization
        positions = self.modeller.getPositions()
        total_area = self._compute_planar_area_nm2()
        z_cathode = self._average_z(positions, self.cathode_indices)
        z_anode = self._average_z(positions, self.anode_indices)

        # Calculate Lgap and Lcell from box vectors
        box_vectors = self.topology.getPeriodicBoxVectors()
        if box_vectors is None:
            raise RuntimeError("Periodic box vectors required for ConstantV")

        cell_vector = np.array([box_vectors[2].x, box_vectors[2].y, box_vectors[2].z])
        box_length_nm = float(np.linalg.norm(cell_vector))
        Lcell = abs(z_anode - z_cathode)
        Lgap = box_length_nm - Lcell

        if not self.is_polarizable:
            raise RuntimeError(
                "ConstantVDrudeLangevinIntegrator requires a polarizable force field. "
                "Please use a Drude force field (e.g., forcefield_drude.xml)."
            )

        # Create ConstantVDrudeLangevinIntegrator with all required parameters
        # This replaces the separate DrudeLangevinIntegrator + ConstantVForce approach
        self.integrator = constantv.ConstantVDrudeLangevinIntegrator(
            params['temperature_kelvin'],
            params['friction_coeff'],
            params['temperature_drude_kelvin'],
            params['drude_friction_coeff'],
            params['timestep_ps'],
            params['voltage_volts'],           # Voltage (V)
            Lgap,                               # Vacuum gap (nm)
            Lcell,                              # Electrode spacing (nm)
            params['scf_iterations']            # SCF iterations per update
        )

        # Set Drude distance constraint
        self.integrator.setMaxDrudeDistance(params['max_drude_distance_nm'])

        # Set geometry parameters
        self.integrator.setTotalArea(total_area)
        self.integrator.setZCathode(z_cathode)
        self.integrator.setZAnode(z_anode)

        # CRITICAL: Set SCF frequency to match original implementation (200 steps)
        # This prevents the 200x performance degradation from running SCF every step
        self.integrator.setSCFFrequency(params['scf_frequency'])

        logger.info("  ✓ Using ConstantVDrudeLangevinIntegrator (Integrator-based architecture)")
        logger.info(f"    Voltage: {params['voltage_volts']:.2f} V")
        logger.info(f"    Lgap: {Lgap:.3f} nm, Lcell: {Lcell:.3f} nm")
        logger.info(f"    SCF iterations: {params['scf_iterations']}")
        logger.info(f"    SCF frequency: every {params['scf_frequency']} steps")

        # Store for later use in physics summary
        self._Lgap = Lgap
        self._Lcell = Lcell
        self._total_area = total_area

    def configure_electrodes_in_integrator(self) -> None:
        """Configure electrode atoms, electrolyte atoms, and conductors in the integrator.

        ARCHITECTURE CHANGE: With Integrator-based approach, we configure atoms
        directly in the integrator instead of creating a separate Force object.
        """
        if self.integrator is None:
            raise RuntimeError("Integrator must be created before configuring electrodes")

        logger.info("Step 7b: Configuring electrodes and electrolytes in integrator...")

        # Calculate area per atom for electrodes
        cathode_area_per_atom = self._total_area / len(self.cathode_indices)
        anode_area_per_atom = self._total_area / len(self.anode_indices)

        # Add cathode atoms
        for idx in self.cathode_indices:
            self.integrator.addCathodeAtom(idx, cathode_area_per_atom)

        # Add anode atoms
        for idx in self.anode_indices:
            self.integrator.addAnodeAtom(idx, anode_area_per_atom)

        # Add electrolyte atoms (for Green's Reciprocity correction)
        for idx in self.electrolyte_indices:
            self.integrator.addElectrolyteAtom(idx, self._particle_charges[idx])

        logger.info(f"  ✓ Added {len(self.cathode_indices)} cathode atoms")
        logger.info(f"  ✓ Added {len(self.anode_indices)} anode atoms")
        logger.info(f"  ✓ Added {len(self.electrolyte_indices)} electrolyte atoms")

        # Add conductors (Buckyballs and Nanotubes)
        self._add_conductors_to_integrator()

        logger.info("  ✓ Electrode configuration complete")

    def _add_conductors_to_integrator(self) -> None:
        """Register Buckyball and Nanotube conductors with the integrator.

        IMPORTANT: This method is called AFTER add_exclusions(), so it only
        configures the integrator. Conductor exclusions must be added separately
        in add_exclusions().
        """
        buckyballs = self.config['conductors'].get('buckyballs', [])
        nanotubes = self.config['conductors'].get('nanotubes', [])

        if not buckyballs and not nanotubes:
            logger.info("  No conductors configured")
            return

        logger.info(
            f"  Adding {len(buckyballs)} Buckyball(s) and {len(nanotubes)} Nanotube(s)"
        )

        for config in buckyballs:
            virtual_indices = self._get_chain_atoms(
                config['virtual_chain_index'],
                set(config.get('exclude_elements', []))
            )
            real_indices = self._get_chain_atoms(
                config['real_chain_index'],
                set(config.get('exclude_elements', []))
            )
            self.integrator.addBuckyballConductor(
                virtual_indices,
                real_indices,
                config['electrode_type'],
                config['voltage']
            )
            self.conductor_charge_indices.append(real_indices)
            logger.info(f"    ✓ Buckyball: {len(real_indices)} atoms, {config['electrode_type']}, {config['voltage']} V")

        for config in nanotubes:
            virtual_indices = self._get_chain_atoms(
                config['virtual_chain_index'],
                set(config.get('exclude_elements', []))
            )
            real_indices = self._get_chain_atoms(
                config['real_chain_index'],
                set(config.get('exclude_elements', []))
            )
            axis = openmm.Vec3(*config['axis'])
            self.integrator.addNanotubeConductor(
                virtual_indices,
                real_indices,
                config['electrode_type'],
                config['voltage'],
                axis
            )
            self.conductor_charge_indices.append(real_indices)
            logger.info(f"    ✓ Nanotube: {len(real_indices)} atoms, {config['electrode_type']}, {config['voltage']} V")

    def _prepare_conductor_exclusion_configs(self) -> None:
        """Prepare conductor configurations for exclusion handling.

        This must be called BEFORE add_exclusions() to ensure conductor
        exclusions are added correctly.
        """
        buckyballs = self.config['conductors'].get('buckyballs', [])
        nanotubes = self.config['conductors'].get('nanotubes', [])

        self.conductor_exclusion_configs = []

        for config in buckyballs:
            virtual_indices = self._get_chain_atoms(
                config['virtual_chain_index'],
                set(config.get('exclude_elements', []))
            )
            real_indices = self._get_chain_atoms(
                config['real_chain_index'],
                set(config.get('exclude_elements', []))
            )
            self.conductor_exclusion_configs.append({
                'virtual_indices': virtual_indices,
                'real_indices': real_indices
            })

        for config in nanotubes:
            virtual_indices = self._get_chain_atoms(
                config['virtual_chain_index'],
                set(config.get('exclude_elements', []))
            )
            real_indices = self._get_chain_atoms(
                config['real_chain_index'],
                set(config.get('exclude_elements', []))
            )
            self.conductor_exclusion_configs.append({
                'virtual_indices': virtual_indices,
                'real_indices': real_indices
            })

    def _get_chain_atoms(self, chain_index: int, exclude_elements: Set[str]) -> List[int]:
        """Helper to get atom indices from chain index."""
        atom_indices = []
        for chain in self.topology.chains():
            if chain.index == chain_index:
                for atom in chain.atoms():
                    if atom.element.symbol not in exclude_elements:
                        atom_indices.append(atom.index)

        if len(atom_indices) == 0:
            raise ValueError(f"No atoms found for chain index {chain_index}")

        return atom_indices

    def _compute_planar_area_nm2(self) -> float:
        """Compute the planar area from periodic box vectors (nm^2)."""
        box_vectors = self.topology.getPeriodicBoxVectors()
        if box_vectors is None:
            raise RuntimeError("Topology is missing periodic box vectors.")

        a = np.array([box_vectors[0].x, box_vectors[0].y, box_vectors[0].z])
        b = np.array([box_vectors[1].x, box_vectors[1].y, box_vectors[1].z])
        cross = np.cross(a, b)
        return float(np.sqrt(np.dot(cross, cross)))

    def _average_z(self, positions, atom_indices: List[int]) -> float:
        """Average z-position in nanometers for the provided indices."""
        if not atom_indices:
            raise ValueError("Atom index list cannot be empty when computing averages.")

        z_values = []
        for idx in atom_indices:
            pos = positions[idx]
            if hasattr(pos, 'value_in_unit'):
                z_values.append(pos.value_in_unit(unit.nanometer)[2])
            else:
                z_values.append(pos.z)
        return float(np.mean(z_values))

    # ═══════════════════════════════════════════════════════════════════════
    # Physics Summary and Validation (Optimization D)
    # ═══════════════════════════════════════════════════════════════════════

    def print_physics_summary(self) -> None:
        """
        Print physics summary and validation warnings.

        This helps users verify their simulation parameters before running
        long simulations, preventing "ran for 3 days with wrong parameters"
        scenarios.
        """
        logger.info("=" * 80)
        logger.info("PHYSICS SUMMARY & VALIDATION")
        logger.info("=" * 80)

        params = self.config['simulation_parameters']

        # Get parameters (now computed from geometry, not from config)
        voltage = params['voltage_volts']
        Lgap = self._Lgap
        Lcell = self._Lcell
        temperature = params['temperature_kelvin']

        if self.integrator is None:
            logger.warning("Integrator is not configured; skipping physics summary.")
            return

        # Get total electrode area (computed during integrator initialization)
        total_area = self._total_area

        # Calculate theoretical capacitance (C = ε₀ * A / d)
        # ε₀ = 8.854e-12 F/m = 8.854e-3 F/nm
        # C (F) = 8.854e-3 (F/nm) * A (nm²) / d (nm)
        epsilon_0 = 8.854e-3  # F/nm (vacuum permittivity)
        capacitance_F = epsilon_0 * total_area / Lgap
        capacitance_pF = capacitance_F * 1e12  # Convert to pF

        # Calculate expected charge magnitude (Q = C * V)
        # Q (C) = C (F) * V (V)
        expected_charge_C = capacitance_F * voltage
        expected_charge_e = expected_charge_C / 1.602e-19  # Convert to elementary charges

        # Print summary
        logger.info(f"Total Electrode Area:        {total_area:.2f} nm²")
        logger.info(f"Electrode Spacing (Lgap):    {Lgap:.3f} nm")
        logger.info(f"Periodic Cell Length (Lcell): {Lcell:.3f} nm")
        logger.info(f"Applied Voltage:             {voltage:.2f} V")
        logger.info(f"Temperature:                 {temperature:.1f} K")
        logger.info(f"SCF Iterations:              {params['scf_iterations']}")
        logger.info("")
        logger.info(f"Theoretical Capacitance:     {capacitance_pF:.4e} pF")
        logger.info(f"Expected Charge (per electrode): ±{expected_charge_e:.2e} e⁻")
        logger.info(f"                            (±{expected_charge_C:.2e} C)")

        # Validation warnings
        logger.info("")
        logger.info("VALIDATION CHECKS:")

        warnings_found = False

        # Check Lgap
        if Lgap < 0.1:
            logger.warning(f"  ⚠️  WARNING: Lgap = {Lgap:.3f} nm is very small (< 0.1 nm)")
            logger.warning("      This may cause numerical instability or unphysical results.")
            logger.warning("      Recommendation: Increase Lgap to at least 1.0 nm.")
            warnings_found = True

        # Check Voltage
        if voltage > 10.0:
            logger.warning(f"  ⚠️  WARNING: Voltage = {voltage:.2f} V is very large (> 10 V)")
            logger.warning("      This may cause excessive charging and simulation instability.")
            logger.warning("      Recommendation: Use voltage < 5 V for typical systems.")
            warnings_found = True

        # Check if Lgap < Lcell/10 (electrode too close relative to cell size)
        if Lgap < Lcell / 10:
            logger.warning(f"  ⚠️  WARNING: Lgap ({Lgap:.3f} nm) is very small compared to Lcell ({Lcell:.3f} nm)")
            logger.warning("      Ratio Lgap/Lcell = {:.2%}".format(Lgap / Lcell))
            logger.warning("      This may cause image charge artifacts.")
            logger.warning("      Recommendation: Ensure Lgap > Lcell/5 for reliable results.")
            warnings_found = True

        # Check timestep relative to temperature (Drude oscillator stability)
        timestep_ps = params['timestep_ps']
        if self.is_polarizable:
            # For Drude oscillators, timestep should be < 0.5 fs typically
            if timestep_ps > 0.001:  # 1 fs
                logger.warning(f"  ⚠️  WARNING: Timestep = {timestep_ps*1000:.2f} fs may be too large for Drude oscillators")
                logger.warning("      Drude oscillators typically require timestep < 0.5 fs (0.0005 ps)")
                logger.warning("      Recommendation: Reduce timestep to 0.0005 ps or less.")
                warnings_found = True

        if not warnings_found:
            logger.info("  ✓ All validation checks passed")

        logger.info("=" * 80)

    # ═══════════════════════════════════════════════════════════════════════
    # Step 8: Create Simulation and Add Reporters
    # ═══════════════════════════════════════════════════════════════════════

    def create_simulation(self) -> None:
        """Create OpenMM Simulation with configured integrator."""
        logger.info("Step 8: Creating Simulation...")

        # Get platform
        platform_name = self.config['run_parameters']['platform']
        platform = openmm.Platform.getPlatformByName(platform_name)

        # Set platform properties (CUDA precision, etc.)
        properties = {}
        if platform_name == "CUDA":
            properties['Precision'] = self.config['run_parameters']['cuda_precision']

        # Create simulation
        self.simulation = app.Simulation(
            self.topology,
            self.system,
            self.integrator,
            platform,
            properties
        )

        # Set positions
        self.simulation.context.setPositions(self.modeller.positions)

        logger.info(f"  ✓ Simulation created on {platform.getName()} platform")
        if properties:
            logger.info(f"    Properties: {properties}")

        # Seed electrode charges for better SCF convergence (optional but recommended)
        logger.info("  Seeding electrode charges...")
        self._seed_electrode_charges()

        # Minimize energy
        logger.info("  Minimizing energy...")
        self.simulation.minimizeEnergy()
        logger.info("  ✓ Energy minimization complete")

    def _seed_electrode_charges(self, threshold: float = 1e-6) -> None:
        """
        Initialize electrode charges to small non-zero values.

        This avoids division by zero in the first SCF iteration when
        computing Ez_external = F_z / q_old, improving convergence.

        Args:
            threshold: Small charge value (elementary charge units)
        """
        # Get NonbondedForce
        nonbonded_force = None
        for force in self.system.getForces():
            if isinstance(force, openmm.NonbondedForce):
                nonbonded_force = force
                break

        if nonbonded_force is None:
            logger.warning("  ⚠️  NonbondedForce not found, skipping charge seeding")
            return

        # Seed cathode atoms (+threshold)
        for idx in self.cathode_indices:
            charge, sigma, epsilon = nonbonded_force.getParticleParameters(idx)
            nonbonded_force.setParticleParameters(idx, threshold, sigma, epsilon)

        # Seed anode atoms (-threshold)
        for idx in self.anode_indices:
            charge, sigma, epsilon = nonbonded_force.getParticleParameters(idx)
            nonbonded_force.setParticleParameters(idx, -threshold, sigma, epsilon)

        # Update context
        nonbonded_force.updateParametersInContext(self.simulation.context)

        logger.info(
            f"  ✓ Seeded {len(self.cathode_indices)} cathode (+{threshold:.1e}) "
            f"and {len(self.anode_indices)} anode (-{threshold:.1e}) atoms"
        )

    def add_reporters(self) -> None:
        """Add trajectory and charge reporters."""
        logger.info("Step 9: Adding Reporters...")

        output = self.config['output']
        freq = output['reporter_frequency_steps']

        # DCD trajectory
        if output.get('output_dcd'):
            self.simulation.reporters.append(
                app.DCDReporter(output['output_dcd'], freq)
            )
            logger.info(f"  ✓ DCD Reporter: {output['output_dcd']} (freq={freq})")

        # State reporter (console output)
        self.simulation.reporters.append(
            app.StateDataReporter(
                sys.stdout,
                freq,
                step=True,
                time=True,
                potentialEnergy=True,
                kineticEnergy=True,
                temperature=True,
                speed=True
            )
        )
        logger.info(f"  ✓ State Reporter: stdout (freq={freq})")

        # Checkpoint
        if output.get('checkpoint_file'):
            checkpoint_freq = output.get('checkpoint_frequency_steps', 10000)
            self.simulation.reporters.append(
                app.CheckpointReporter(output['checkpoint_file'], checkpoint_freq)
            )
            logger.info(f"  ✓ Checkpoint Reporter: {output['checkpoint_file']} (freq={checkpoint_freq})")

        # Electrode charge reporter
        if output.get('output_charges'):
            charge_reporter = ElectrodeChargeReporter(
                output['output_charges'],
                freq,
                self.cathode_indices,
                self.anode_indices,
                self.conductor_charge_indices,
            )
            self.simulation.reporters.append(charge_reporter)
            logger.info(f"  ✓ Electrode Charge Reporter: {output['output_charges']} (freq={freq})")

    # ═══════════════════════════════════════════════════════════════════════
    # Step 10: Run Simulation
    # ═══════════════════════════════════════════════════════════════════════

    def run(self) -> None:
        """Run equilibration and production simulation."""
        run_params = self.config['run_parameters']

        # Equilibration
        if run_params['equilibration_steps'] > 0:
            logger.info("=" * 80)
            logger.info(f"EQUILIBRATION: {run_params['equilibration_steps']} steps")
            logger.info("=" * 80)
            self.simulation.step(run_params['equilibration_steps'])
            logger.info("  ✓ Equilibration complete")

        # Production
        logger.info("=" * 80)
        logger.info(f"PRODUCTION: {run_params['production_steps']} steps")
        logger.info("=" * 80)
        self.simulation.step(run_params['production_steps'])

        logger.info("=" * 80)
        logger.info("✓ SIMULATION COMPLETE")
        logger.info("=" * 80)

    # ═══════════════════════════════════════════════════════════════════════
    # Main Workflow
    # ═══════════════════════════════════════════════════════════════════════

    def execute(self) -> None:
        """Execute complete simulation workflow.

        ARCHITECTURE: Now uses Integrator-based approach instead of Force-based.
        """
        try:
            self.load_system()
            self.add_drude_particles()
            self.create_system()
            self.identify_electrodes()
            self.identify_electrolytes()
            self.add_exclusions()
            # ARCHITECTURE CHANGE: create_integrator now creates ConstantVDrudeLangevinIntegrator
            # with all ConstantV parameters built-in (instead of separate integrator + force)
            self.create_integrator()
            # Configure electrodes, electrolytes, and conductors in the integrator
            self.configure_electrodes_in_integrator()
            self.print_physics_summary()  # Optimization D: Validate before running
            self.create_simulation()
            self.add_reporters()
            self.run()

        except Exception as e:
            logger.error("=" * 80)
            logger.error("✗ SIMULATION FAILED")
            logger.error("=" * 80)
            logger.error(f"Error: {e}", exc_info=True)
            raise


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python run_production.py <config.json>")
        print("Example: python run_production.py production_config.json")
        sys.exit(1)

    config_file = sys.argv[1]

    if not Path(config_file).exists():
        print(f"Error: Configuration file not found: {config_file}")
        sys.exit(1)

    # Create and run simulation
    sim = ConstantVProductionSimulation(config_file)
    sim.execute()


if __name__ == "__main__":
    main()
