#!/usr/bin/env python
"""Test ElectrodeChargePlugin with CUDA platform"""

import openmm as mm
from openmm import app, unit
import numpy as np
import electrodecharge  # Plugin Python binding

# Load plugin
plugin_dir = '/home/andy/miniforge3/envs/cuda/lib/plugins'
mm.Platform.loadPluginsFromDirectory(plugin_dir)

# Create minimal test system
system = mm.System()

# Add 10 atoms (particles)
for i in range(10):
    system.addParticle(1.0)  # 1 amu each

# Create ElectrodeChargeForce
try:
    force = electrodecharge.ElectrodeChargeForce()
    
    # Add 5 cathode atoms (indices 0-4)
    for i in range(5):
        force.addCathodeAtom(i)
    
    # Add 5 anode atoms (indices 5-9)
    for i in range(5):
        force.addAnodeAtom(i)
    
    # Set parameters (matching your config)
    force.setArea(14.0 * 14.0)  # nm^2
    force.setVoltage(0.5)  # V
    force.setGapLength(1.5)  # nm
    force.setConversionFactor(138.935456)  # OpenMM units
    force.setImageChargeRadius(2.0)  # nm
    force.setImageChargeDistance(0.5)  # nm
    force.setAnalyticThreshold(1e-6)
    
    system.addForce(force)
    
    print("✅ ElectrodeChargeForce created successfully")
    print(f"   Cathode atoms: {force.getNumCathodeAtoms()}")
    print(f"   Anode atoms: {force.getNumAnodeAtoms()}")
    print(f"   Area: {force.getArea()} nm²")
    print(f"   Voltage: {force.getVoltage()} V")
    
except Exception as e:
    print(f"❌ Error creating ElectrodeChargeForce: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test with Reference platform first
print("\n=== Testing Reference Platform ===")
try:
    platform_ref = mm.Platform.getPlatformByName('Reference')
    integrator_ref = mm.LangevinIntegrator(300*unit.kelvin, 1.0/unit.picoseconds, 0.002*unit.picoseconds)
    
    # Create initial positions (2D sheet structure)
    positions = []
    for i in range(5):  # Cathode
        positions.append([i*0.5, 0.0, 0.0])
    for i in range(5):  # Anode
        positions.append([i*0.5, 0.0, 1.5])
    
    context_ref = mm.Context(system, integrator_ref, platform_ref)
    context_ref.setPositions(positions)
    
    # Get initial state
    state_ref = context_ref.getState(getForces=True, getEnergy=True)
    energy_ref = state_ref.getPotentialEnergy()
    forces_ref = state_ref.getForces(asNumpy=True)
    
    print(f"✅ Reference Platform OK")
    print(f"   Energy: {energy_ref}")
    print(f"   Force on atom 0: {forces_ref[0]}")
    
    del context_ref
    del integrator_ref
    
except Exception as e:
    print(f"❌ Reference Platform error: {e}")
    import traceback
    traceback.print_exc()

# Test with CUDA platform
print("\n=== Testing CUDA Platform ===")
try:
    platform_cuda = mm.Platform.getPlatformByName('CUDA')
    integrator_cuda = mm.LangevinIntegrator(300*unit.kelvin, 1.0/unit.picoseconds, 0.002*unit.picoseconds)
    
    context_cuda = mm.Context(system, integrator_cuda, platform_cuda)
    context_cuda.setPositions(positions)
    
    # Get initial state
    state_cuda = context_cuda.getState(getForces=True, getEnergy=True)
    energy_cuda = state_cuda.getPotentialEnergy()
    forces_cuda = state_cuda.getForces(asNumpy=True)
    
    print(f"✅ CUDA Platform OK")
    print(f"   Energy: {energy_cuda}")
    print(f"   Force on atom 0: {forces_cuda[0]}")
    
    # Compare with Reference
    energy_diff = abs(energy_cuda - energy_ref).value_in_unit(unit.kilojoules_per_mole)
    force_diff = np.linalg.norm(forces_cuda[0] - forces_ref[0])
    
    print(f"\n=== Comparison ===")
    print(f"   Energy difference: {energy_diff:.6e} kJ/mol")
    print(f"   Force difference (atom 0): {force_diff:.6e} kJ/(mol·nm)")
    
    if energy_diff < 1e-3 and force_diff < 1e-3:
        print("\n✅✅✅ CUDA Plugin CORRECT! ✅✅✅")
    else:
        print("\n⚠️  Large differences detected - may need investigation")
    
    del context_cuda
    del integrator_cuda
    
except Exception as e:
    print(f"❌ CUDA Platform error: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Test Complete ===")
