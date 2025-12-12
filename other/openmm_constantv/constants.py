"""
Physical Constants for OpenMM ConstantV Plugin

This module defines all physical constants used in the Constant Voltage simulation.
These constants are copied from the professor's original code to ensure
100% physical accuracy.

References:
    - MM_classes.py:48  (SMALL_THRESHOLD)
    - Fixed_Voltage_routines.py:36-38 (conversions)
"""

from typing import Final

# ═══════════════════════════════════════════════════════════
# Unit Conversions (Fixed_Voltage_routines.py:36-38)
# ═══════════════════════════════════════════════════════════

# Line 36: Nanometer to Bohr radius conversion
# 1 nm = 18.8973 Bohr radii
CONVERSION_NM_TO_BOHR: Final[float] = 18.8973

# Line 37: kJ/mol·nm to Atomic Units conversion
# Used for converting energy gradients (electric fields) from OpenMM to quantum units
# conversion_KjmolNm_Au = conversion_nmBohr / 2625.5
CONVERSION_KJMOL_NM_TO_AU: Final[float] = CONVERSION_NM_TO_BOHR / 2625.5  # ≈ 0.00719924

# Line 38: Electronvolt to kJ/mol conversion
# 1 eV = 96.487 kJ/mol (NIST CODATA 2018)
# This is used to convert voltage from Volts to kJ/mol
CONVERSION_EV_TO_KJMOL: Final[float] = 96.487

# ═══════════════════════════════════════════════════════════
# Numerical Thresholds (MM_classes.py:48)
# ═══════════════════════════════════════════════════════════

# Line 48: self.small_threshold = 1e-6
# Threshold for charge magnitude to prevent division by zero in field calculations
# WARNING: Do NOT change this value without consulting the original physics!
# The professor uses 1e-6 (not 1e-10) for good numerical stability reasons.
SMALL_THRESHOLD: Final[float] = 1e-6  # electrons (elementary charge units)

# ═══════════════════════════════════════════════════════════
# Physical Constants
# ═══════════════════════════════════════════════════════════

# Coulomb's constant in vacuum: k_e = 1/(4πε₀)
# In Gaussian units (atomic units), 4πε₀ = 1, so this appears in many formulas
FOUR_PI: Final[float] = 4.0 * 3.141592653589793

# ═══════════════════════════════════════════════════════════
# Default Simulation Parameters
# ═══════════════════════════════════════════════════════════

# Default number of SCF iterations for charge convergence (MM_classes.py:287)
DEFAULT_SCF_ITERATIONS: Final[int] = 4

# Default cutoff distance for non-bonded interactions (MM_classes.py:49)
DEFAULT_CUTOFF_NM: Final[float] = 1.4  # nanometers

# Default PME error tolerance
DEFAULT_PME_ERROR_TOLERANCE: Final[float] = 0.0005

# ═══════════════════════════════════════════════════════════
# Conductor Parameters
# ═══════════════════════════════════════════════════════════

# Default close contact threshold for Buckyball/Nanotube conductors
# Fixed_Voltage_routines.py:100
CONDUCTOR_CLOSE_THRESHOLD_NM: Final[float] = 1.5  # nanometers

# ═══════════════════════════════════════════════════════════
# OpenMM Force Group Assignment
# ═══════════════════════════════════════════════════════════

# ConstantVForce should be assigned to this force group to prevent recursion
# during SCF iterations (see CudaConstantVKernels.cu:45)
CONSTANTV_FORCE_GROUP: Final[int] = 31
