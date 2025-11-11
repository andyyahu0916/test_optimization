"""
ConstantV Plugin for OpenMM - Constant Voltage Molecular Dynamics

This package provides constant voltage MD simulation capabilities for OpenMM,
replicating the algorithm from the Original Python implementation.

Main components:
- ConstantVIntegrator: Verlet integrator with SCF charge updates
- helpers: Helper functions for electrode exclusions, geometry setup, etc.
"""

# Import SWIG-generated classes
# These will be imported from the parent level after installation
try:
    from constantvplugin import ConstantVIntegrator, ConstantVForce
    __all__ = ['ConstantVIntegrator', 'ConstantVForce', 'helpers']
except ImportError:
    # During build, the module may not be available yet
    pass
