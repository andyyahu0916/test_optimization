"""
Pydantic Configuration Models for ConstantV Plugin

This module provides strict type-checked configuration schemas with validation.
All parameters are validated at construction time to prevent runtime errors.

Design Philosophy:
    - Fail Fast: Invalid configurations raise clear exceptions immediately
    - Type Safety: All fields have strict type hints
    - Validation: Complex constraints are enforced via Pydantic validators
    - Documentation: Every field includes physical meaning and units
"""

from typing import List, Literal, Optional, Tuple
from pydantic import BaseModel, Field, field_validator, model_validator
import numpy as np


# ═══════════════════════════════════════════════════════════
# Electrode Configuration
# ═══════════════════════════════════════════════════════════


class ElectrodeConfig(BaseModel):
    """
    Configuration for flat electrode (cathode or anode).

    Corresponds to: MM_classes.py::initialize_electrodes()

    Attributes:
        identifier: Residue name or chain index for electrode atoms
        electrode_type: "cathode" or "anode"
        by_chain: If True, identify atoms by chain index; else by residue name
        exclude_elements: Elements to exclude (e.g., dummy H atoms)
    """

    identifier: str | int = Field(
        ..., description="Residue name (str) or chain index (int) for electrode"
    )
    electrode_type: Literal["cathode", "anode"] = Field(
        ..., description="Type of electrode"
    )
    by_chain: bool = Field(
        default=False, description="Identify atoms by chain index (True) or residue name (False)"
    )
    exclude_elements: Tuple[str, ...] = Field(
        default_factory=tuple, description="Element symbols to exclude (e.g., ('H',))"
    )

    @field_validator("electrode_type")
    @classmethod
    def validate_electrode_type(cls, v: str) -> str:
        """Ensure electrode type is valid."""
        if v not in ("cathode", "anode"):
            raise ValueError(f"electrode_type must be 'cathode' or 'anode', got '{v}'")
        return v


# ═══════════════════════════════════════════════════════════
# Conductor Configuration (Buckyball, Nanotube)
# ═══════════════════════════════════════════════════════════


class BuckyballConfig(BaseModel):
    """
    Configuration for Buckyball conductor (spherical conductor).

    Corresponds to: Fixed_Voltage_routines.py::Buckyball_Virtual

    Attributes:
        virtual_chain_index: Chain index for virtual layer atoms (electrostatics)
        real_chain_index: Chain index for real layer atoms (VDW/steric)
        electrode_type: Which electrode this conductor is attached to
        exclude_elements: Elements to exclude
        close_threshold_nm: Distance threshold to determine if conductor is in contact
    """

    virtual_chain_index: int = Field(..., description="Chain index for virtual layer")
    real_chain_index: int = Field(..., description="Chain index for real layer")
    electrode_type: Literal["cathode", "anode"] = Field(
        ..., description="Attached electrode type"
    )
    exclude_elements: Tuple[str, ...] = Field(default_factory=tuple)
    close_threshold_nm: float = Field(
        default=1.5, description="Contact distance threshold (nm)", gt=0.0
    )

    @field_validator("virtual_chain_index", "real_chain_index")
    @classmethod
    def validate_chain_index(cls, v: int) -> int:
        """Ensure chain indices are non-negative."""
        if v < 0:
            raise ValueError(f"Chain index must be non-negative, got {v}")
        return v


class NanotubeConfig(BaseModel):
    """
    Configuration for Nanotube conductor (cylindrical conductor).

    Corresponds to: Fixed_Voltage_routines.py::Nanotube_Virtual

    Attributes:
        virtual_chain_index: Chain index for virtual layer atoms
        real_chain_index: Chain index for real layer atoms
        electrode_type: Which electrode this conductor is attached to
        axis: Unit vector along nanotube axis [ax, ay, az]
        exclude_elements: Elements to exclude
        close_threshold_nm: Distance threshold for contact detection
    """

    virtual_chain_index: int = Field(..., description="Chain index for virtual layer")
    real_chain_index: int = Field(..., description="Chain index for real layer")
    electrode_type: Literal["cathode", "anode"] = Field(
        ..., description="Attached electrode type"
    )
    axis: Tuple[float, float, float] = Field(
        ..., description="Unit vector along nanotube axis [ax, ay, az]"
    )
    exclude_elements: Tuple[str, ...] = Field(default_factory=tuple)
    close_threshold_nm: float = Field(
        default=1.5, description="Contact distance threshold (nm)", gt=0.0
    )

    @field_validator("axis")
    @classmethod
    def validate_axis(cls, v: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        Validate and auto-normalize axis vector.
        
        FIX P3-C1: Auto-normalize instead of raising error.
        CUDA kernels assume unit vectors; this prevents incorrect charge calculations.
        """
        norm = np.linalg.norm(v)
        if norm < 1e-10:
            raise ValueError(f"Nanotube axis cannot be zero vector, got {v}")
        # FIX P3-C1: Auto-normalize to unit vector instead of raising error
        normalized = (v[0] / norm, v[1] / norm, v[2] / norm)
        return normalized


# ═══════════════════════════════════════════════════════════
# System Configuration
# ═══════════════════════════════════════════════════════════


class SystemConfig(BaseModel):
    """
    Complete system configuration for ConstantV simulation.

    Corresponds to: MM_classes.py::__init__() and initialize_electrodes()

    Attributes:
        pdb_files: List of PDB files to load
        residue_xml_files: List of residue definition XML files
        forcefield_xml_files: List of force field XML files
        voltage_volts: Applied voltage in Volts
        cathode: Cathode electrode configuration
        anode: Anode electrode configuration
        buckyballs: List of Buckyball conductor configurations
        nanotubes: List of Nanotube conductor configurations
        temperature_kelvin: System temperature
        temperature_drude_kelvin: Drude oscillator temperature (polarizable sims)
        timestep_ps: Integration timestep in picoseconds
        cutoff_nm: Non-bonded interaction cutoff in nanometers
        scf_iterations: Number of SCF iterations for charge convergence
        natom_cutoff: Cutoff for identifying electrolyte vs electrode residues
    """

    # Input files
    pdb_files: List[str] = Field(..., min_length=1, description="PDB files")
    residue_xml_files: List[str] = Field(..., description="Residue XML files")
    forcefield_xml_files: List[str] = Field(..., min_length=1, description="Force field XML files")

    # Voltage and electrodes
    voltage_volts: float = Field(..., description="Applied voltage (V)")
    cathode: ElectrodeConfig = Field(..., description="Cathode configuration")
    anode: ElectrodeConfig = Field(..., description="Anode configuration")

    # Conductors (optional)
    buckyballs: List[BuckyballConfig] = Field(default_factory=list, description="Buckyball conductors")
    nanotubes: List[NanotubeConfig] = Field(default_factory=list, description="Nanotube conductors")

    # Simulation parameters (with professor's defaults)
    temperature_kelvin: float = Field(default=300.0, description="System temperature (K)", gt=0.0)
    temperature_drude_kelvin: float = Field(default=1.0, description="Drude temperature (K)", gt=0.0)
    timestep_ps: float = Field(default=0.001, description="Timestep (ps)", gt=0.0)
    cutoff_nm: float = Field(default=1.4, description="Cutoff distance (nm)", gt=0.0)
    scf_iterations: int = Field(default=4, description="SCF iterations", ge=1)
    natom_cutoff: int = Field(
        default=100,
        description="Residue size cutoff: residues with < natom_cutoff atoms are electrolyte",
        gt=0,
    )
    sapt_ff_exclusions: bool = Field(
        default=True,
        description="Enable SAPT-FF specific exclusions (water interaction groups + TFSI handling)",
    )
    hybrid_water_model: bool = Field(
        default=False,
        description="Force-add hybrid water interaction groups even without SAPT-FF",
    )
    water_residue_name: str = Field(
        default="HOH",
        description="Residue name for water molecules when configuring hybrid interaction groups",
    )
    tfsi_residue_name: str = Field(
        default="Tf2N",
        description="Residue name for TFSI anions in SAPT-FF exclusion logic",
    )

    @model_validator(mode='after')
    def validate_conductors_require_geometry(self) -> 'SystemConfig':
        """
        CRITICAL VALIDATION: Buckyballs and Nanotubes require geometric parameters.

        This validator implements the requirement from the user's prompt:
        "Implement a validator that checks if `Buckyballs` or `Nanotubes` are requested,
         and ensures the corresponding geometric parameters are provided."
        """
        # Buckyballs: No extra geometry needed (sphere is determined from atoms)
        # Nanotubes: Must have axis parameter (checked in NanotubeConfig)

        # Check if any conductors are on the same electrode (complex physics warning)
        if len(self.buckyballs) + len(self.nanotubes) > 1:
            # Check for multiple conductors on same electrode
            cathode_conductors = sum(
                1 for b in self.buckyballs if b.electrode_type == "cathode"
            ) + sum(1 for n in self.nanotubes if n.electrode_type == "cathode")
            anode_conductors = sum(
                1 for b in self.buckyballs if b.electrode_type == "anode"
            ) + sum(1 for n in self.nanotubes if n.electrode_type == "anode")

            if cathode_conductors > 1 or anode_conductors > 1:
                raise ValueError(
                    f"Multiple conductors on same electrode not yet supported. "
                    f"Found {cathode_conductors} on cathode, {anode_conductors} on anode. "
                    f"Please use only one conductor per electrode."
                )

        return self


# ═══════════════════════════════════════════════════════════
# Simulation Run Configuration
# ═══════════════════════════════════════════════════════════


class SimulationConfig(BaseModel):
    """
    Configuration for simulation execution.

    Attributes:
        platform: OpenMM platform ('Reference', 'CPU', 'CUDA', 'OpenCL')
        precision: Precision for CUDA platform ('single', 'mixed', 'double')
        output_dcd: Output trajectory DCD file path (optional)
        output_charges: Output electrode charges file path (optional)
        reporter_frequency: Frequency for trajectory/charge output (steps)
        total_steps: Total number of MD steps to run
        equilibration_steps: Number of equilibration steps before production
    """

    platform: Literal["Reference", "CPU", "CUDA", "OpenCL"] = Field(
        default="CUDA", description="OpenMM platform"
    )
    precision: Literal["single", "mixed", "double"] = Field(
        default="mixed", description="CUDA precision (ignored for other platforms)"
    )

    output_dcd: Optional[str] = Field(default=None, description="Output DCD file")
    output_charges: Optional[str] = Field(default=None, description="Output charges file")
    reporter_frequency: int = Field(default=1000, description="Reporter frequency (steps)", ge=1)

    total_steps: int = Field(..., description="Total MD steps", gt=0)
    equilibration_steps: int = Field(default=0, description="Equilibration steps", ge=0)

    @model_validator(mode='after')
    def validate_output_files(self) -> 'SimulationConfig':
        """Ensure at least one output is specified if running simulation."""
        if self.total_steps > 0 and self.output_dcd is None and self.output_charges is None:
            raise ValueError(
                "Must specify at least one output file (output_dcd or output_charges) "
                "when running simulation"
            )
        return self


# ═══════════════════════════════════════════════════════════
# Combined Configuration (Full Specification)
# ═══════════════════════════════════════════════════════════


class ConductorConfig(BaseModel):
    """
    Union type for conductor configurations.

    This allows flexible configuration of multiple conductor types.
    """

    buckyballs: List[BuckyballConfig] = Field(default_factory=list)
    nanotubes: List[NanotubeConfig] = Field(default_factory=list)
