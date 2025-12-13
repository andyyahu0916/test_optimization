"""
Shared utilities for OpenMM ConstantV.

This package provides common utility functions that can be used across
different layers of the ConstantV architecture.

Public API:
-----------
- add_all_exclusions: Unified exclusion workflow for electrodes, conductors,
                       water, and SAPT-FF force fields.

Version: 1.0.0
"""

from .exclusions import add_all_exclusions

__version__ = '1.0.0'
__all__ = ['add_all_exclusions']
