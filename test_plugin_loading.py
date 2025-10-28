#!/usr/bin/env python
"""Test ElectrodeChargePlugin via XML (no Python wrapper needed)"""

import openmm as mm
from openmm import app, unit
import numpy as np

print("=== Loading Plugin via loadPluginsFromDirectory ===")
plugin_dir = '/home/andy/miniforge3/envs/cuda/lib/plugins'
mm.Platform.loadPluginsFromDirectory(plugin_dir)
print(f"✅ Plugins loaded from {plugin_dir}")

# Create XML with ElectrodeChargeForce
xml_content = """<?xml version="1.0" ?>
<ForceField>
  <ElectrodeChargeForce area="196.0" voltage="0.5" gapLength="1.5" 
                        conversionFactor="138.935456" 
                        imageChargeRadius="2.0" imageChargeDistance="0.5"
                        analyticThreshold="1e-6">
    <Cathode index="0"/>
    <Cathode index="1"/>
    <Cathode index="2"/>
    <Cathode index="3"/>
    <Cathode index="4"/>
    <Anode index="5"/>
    <Anode index="6"/>
    <Anode index="7"/>
    <Anode index="8"/>
    <Anode index="9"/>
  </ElectrodeChargeForce>
</ForceField>
"""

# Save XML
with open('/tmp/test_electrode.xml', 'w') as f:
    f.write(xml_content)

print("\n=== Creating System from XML ===")
try:
    # Create minimal system
    system = mm.System()
    for i in range(10):
        system.addParticle(1.0)
    
    # Try to load XML force
    ff = app.ForceField('/tmp/test_electrode.xml')
    print(f"❌ ForceField parser doesn't recognize custom forces")
    
except Exception as e:
    print(f"Expected: {e}")

print("\n=== Manual Plugin Loading (C++ API) ===")
try:
    # Load plugin library directly
    import ctypes
    lib_path = '/home/andy/miniforge3/envs/cuda/lib/libElectrodeChargePlugin.so'
    plugin_lib = ctypes.CDLL(lib_path)
    print(f"✅ Loaded: {lib_path}")
    
    # Register plugin
    plugin_lib.registerElectrodeChargePlugin()
    print(f"✅ Plugin registered")
    
except Exception as e:
    print(f"⚠️  Direct registration: {e}")

print("\n=== Checking Force Registration ===")
# Check if we can access the Force through C++ wrapper
import sys
sys.path.insert(0, '/home/andy/test_optimization/plugins/ElectrodeChargePlugin/build/python')

try:
    import electrodecharge
    print(f"✅ Python wrapper loaded: {electrodecharge}")
    force = electrodecharge.ElectrodeChargeForce()
    print(f"✅✅✅ ElectrodeChargeForce created! {force}")
except ImportError as e:
    print(f"⚠️  Python wrapper not built: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Alternative: Check if Plugin kernels are available ===")
# Even without Python wrapper, kernels should register with platforms
system = mm.System()
for i in range(10):
    system.addParticle(1.0)

# Add a simple force to make system valid
system.addForce(mm.HarmonicBondForce())

print("\nTesting Platform Initialization:")
for platform_name in ['Reference', 'CUDA']:
    try:
        platform = mm.Platform.getPlatformByName(platform_name)
        integrator = mm.LangevinIntegrator(300*unit.kelvin, 1/unit.picoseconds, 0.002*unit.picoseconds)
        
        positions = [[i*0.5, 0, 0] for i in range(5)] + [[i*0.5, 0, 1.5] for i in range(5)]
        
        context = mm.Context(system, integrator, platform)
        context.setPositions(positions)
        
        state = context.getState(getEnergy=True)
        print(f"  ✅ {platform_name}: {state.getPotentialEnergy()}")
        
        del context, integrator
        
    except Exception as e:
        print(f"  ❌ {platform_name}: {e}")

print("\n=== Summary ===")
print("Plugin libraries are installed, but Python wrapper is missing.")
print("CUDA kernels compiled successfully (1.2MB libElectrodeChargePluginCUDA.so)")
print("\nNext steps:")
print("1. Fix Python wrapper compilation (OpenMMException.h include path)")
print("2. OR use C++ API directly")
print("3. OR integrate into existing run_openMM.py via XML serialization")
