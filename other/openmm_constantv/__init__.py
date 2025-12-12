"""
OpenMM ConstantV Plugin - Production-Grade Python SDK

A modern, type-safe, defensive Python SDK for the OpenMM ConstantV plugin.

Features:
    - Factory pattern for system building
    - Pydantic configuration validation
    - Automatic PME enforcement
    - Automatic Drude particle addition
    - Type-safe electrode/conductor configuration
    - Custom reporters for electrode charges

Example:
    >>> from openmm_constantv import SystemConfig, ConstantVSystemBuilder, ElectrodeConfig
    >>> config = SystemConfig(
    ...     pdb_files=["system.pdb"],
    ...     forcefield_xml_files=["ff.xml"],
    ...     voltage_volts=1.0,
    ...     cathode=ElectrodeConfig(identifier="GRA", electrode_type="cathode"),
    ...     anode=ElectrodeConfig(identifier="GRA", electrode_type="anode"),
    ... )
    >>> builder = ConstantVSystemBuilder(config)
    >>> system, topology, modeller = builder.build()
"""

__version__ = "1.0.0"
__author__ = "Production Engineering System"

from .constants import (
    CONVERSION_NM_TO_BOHR,
    CONVERSION_KJMOL_NM_TO_AU,
    CONVERSION_EV_TO_KJMOL,
    SMALL_THRESHOLD,
    DEFAULT_SCF_ITERATIONS,
    DEFAULT_CUTOFF_NM,
    CONSTANTV_FORCE_GROUP,
)

from .models import (
    SystemConfig,
    SimulationConfig,
    ElectrodeConfig,
    BuckyballConfig,
    NanotubeConfig,
    ConductorConfig,
)

from .core import (
    ConstantVSystemBuilder,
)

from .reporters import (
    ElectrodeChargeReporter,
)

__all__ = [
    # Constants
    "CONVERSION_NM_TO_BOHR",
    "CONVERSION_KJMOL_NM_TO_AU",
    "CONVERSION_EV_TO_KJMOL",
    "SMALL_THRESHOLD",
    "DEFAULT_SCF_ITERATIONS",
    "DEFAULT_CUTOFF_NM",
    "CONSTANTV_FORCE_GROUP",
    # Configuration Models
    "SystemConfig",
    "SimulationConfig",
    "ElectrodeConfig",
    "BuckyballConfig",
    "NanotubeConfig",
    "ConductorConfig",
    # Core Classes
    "ConstantVSystemBuilder",
    # Reporters
    "ElectrodeChargeReporter",
]
