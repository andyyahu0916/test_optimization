#!/usr/bin/env python3
"""
Test if all required modules can be imported
"""
import sys
sys.path.insert(0, './lib/')

print("Testing imports...")
print("="*60)

# Test Original version
try:
    import MM_classes
    import Fixed_Voltage_routines
    print("✓ Original version: MM_classes and Fixed_Voltage_routines")
except ImportError as e:
    print(f"✗ Original version failed: {e}")

# Test Cython version
try:
    import MM_classes_CYTHON
    import Fixed_Voltage_routines_CYTHON
    print("✓ Cython version: MM_classes_CYTHON and Fixed_Voltage_routines_CYTHON")
except ImportError as e:
    print(f"✗ Cython version failed: {e}")

# Test Optimized version
try:
    import MM_classes_OPTIMIZED
    import Fixed_Voltage_routines_OPTIMIZED
    print("✓ Optimized version: MM_classes_OPTIMIZED and Fixed_Voltage_routines_OPTIMIZED")
except ImportError as e:
    print(f"✗ Optimized version failed: {e}")

# Test Cython module
try:
    import electrode_charges_cython
    print("✓ Cython module: electrode_charges_cython")
    print(f"  Available functions: {[x for x in dir(electrode_charges_cython) if not x.startswith('_')]}")
except ImportError as e:
    print(f"✗ Cython module failed: {e}")

# Test OpenMM
try:
    from simtk.openmm.app import *
    from simtk.openmm import *
    from simtk.unit import *
    print("✓ OpenMM imported successfully")
except ImportError as e:
    print(f"✗ OpenMM failed: {e}")

print("="*60)
print("Import test complete!")
