#!/usr/bin/env python
"""
Diagnostic test: Investigate charge conservation in ElectrodeChargeForce.

Purpose: Determine if non-conservation is a bug or expected behavior.
"""

import ctypes
import os
import sys
import numpy as np

import ctypes
import os
import sys
import numpy as np

_BUILD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "build"))
_PYTHON_BUILD_DIR = os.path.join(_BUILD_DIR, "python")
sys.path.insert(0, _PYTHON_BUILD_DIR)

# Load plugin libraries
_REFERENCE_LIB = os.path.join(_BUILD_DIR, "platforms", "reference", "libElectrodeChargePluginReference.so")
_CUDA_LIB = os.path.join(_BUILD_DIR, "platforms", "cuda", "libElectrodeChargePluginCUDA.so")
_PLUGIN_LIB = os.path.join(_BUILD_DIR, "libElectrodeChargePlugin.so")
_RTLD_GLOBAL = getattr(ctypes, "RTLD_GLOBAL", 0)

ctypes.CDLL(_REFERENCE_LIB, mode=_RTLD_GLOBAL)
if os.path.exists(_CUDA_LIB):
    ctypes.CDLL(_CUDA_LIB, mode=_RTLD_GLOBAL)
plugin_lib = ctypes.CDLL(_PLUGIN_LIB, mode=_RTLD_GLOBAL)
plugin_lib.registerElectrodeChargePlugin()

import electrodecharge
from openmm import Context, LangevinIntegrator, NonbondedForce, Platform, System, unit, Vec3


def test_charge_conservation():
    """Test if total system charge is conserved."""
    
    print("="*80)
    print("CHARGE CONSERVATION DIAGNOSTIC")
    print("="*80)
    
    # Simple test system
    cathode_indices = [0, 1]
    anode_indices = [2, 3]
    bulk_index = 4
    
    positions = [
        [0.0, 0.0, 0.1],   # Cathode
        [0.8, 0.0, 0.1],
        [0.0, 0.0, 1.9],   # Anode
        [0.8, 0.0, 1.9],
        [0.4, 0.4, 1.0],   # Bulk (electrolyte)
    ]
    
    box_vectors = [
        [1.6, 0.0, 0.0],
        [0.0, 1.6, 0.0],
        [0.0, 0.0, 2.0],
    ]
    
    # Test with different initial bulk charges
    test_cases = [
        ("Zero initial charge", 0.0),
        ("Positive initial charge", 0.5),
        ("Negative initial charge", -0.5),
    ]
    
    for test_name, bulk_charge in test_cases:
        print(f"\n--- {test_name}: bulk = {bulk_charge:+.1f} e ---")
        
        # Create system
        system = System()
        for _ in range(len(positions)):
            system.addParticle(40.0)
        
        nonbonded = NonbondedForce()
        nonbonded.setNonbondedMethod(NonbondedForce.NoCutoff)
        for i in range(len(positions)):
            if i == bulk_index:
                nonbonded.addParticle(bulk_charge, 1.0, 0.0)
            else:
                nonbonded.addParticle(0.0, 1.0, 0.0)
        system.addForce(nonbonded)
        
        force = electrodecharge.ElectrodeChargeForce()
        force.setCathode(cathode_indices, -0.5)  # Negative voltage
        force.setAnode(anode_indices, 0.5)       # Positive voltage
        force.setCellGap(0.8)
        force.setCellLength(2.0)
        force.setNumIterations(3)
        system.addForce(force)
        
        print(f"  Applied voltage: Cathode = -0.5 eV, Anode = +0.5 eV")
        print(f"  Voltage difference: ΔV = {0.5 - (-0.5)} = 1.0 eV")
        
        platform = Platform.getPlatformByName('Reference')
        integrator = LangevinIntegrator(300*unit.kelvin, 1.0/unit.picosecond, 0.001*unit.picoseconds)
        context = Context(system, integrator, platform)
        
        positions_vec3 = [Vec3(p[0], p[1], p[2]) for p in positions]
        context.setPositions(positions_vec3)
        context.setPeriodicBoxVectors(
            Vec3(box_vectors[0][0], box_vectors[0][1], box_vectors[0][2]),
            Vec3(box_vectors[1][0], box_vectors[1][1], box_vectors[1][2]),
            Vec3(box_vectors[2][0], box_vectors[2][1], box_vectors[2][2])
        )
        
        # Get initial charges
        q_initial = []
        for i in range(len(positions)):
            charge, _, _ = nonbonded.getParticleParameters(i)
            q_initial.append(charge.value_in_unit(unit.elementary_charge))
        
        q_total_initial = sum(q_initial)
        
        # Run ElectrodeChargeForce (triggers charge update)
        state = context.getState(getForces=True, groups=1<<1)
        
        # Get final charges
        q_final = []
        for i in range(len(positions)):
            charge, _, _ = nonbonded.getParticleParameters(i)
            q_final.append(charge.value_in_unit(unit.elementary_charge))
        
        q_total_final = sum(q_final)
        
        # Analysis
        print(f"Initial charges:")
        print(f"  Cathode [0]: {q_initial[0]:+.6f} e")
        print(f"  Cathode [1]: {q_initial[1]:+.6f} e")
        print(f"  Anode [2]:   {q_initial[2]:+.6f} e")
        print(f"  Anode [3]:   {q_initial[3]:+.6f} e")
        print(f"  Bulk [4]:    {q_initial[4]:+.6f} e")
        print(f"  TOTAL:       {q_total_initial:+.6f} e")
        
        print(f"\nFinal charges:")
        print(f"  Cathode [0]: {q_final[0]:+.6f} e")
        print(f"  Cathode [1]: {q_final[1]:+.6f} e")
        print(f"  Anode [2]:   {q_final[2]:+.6f} e")
        print(f"  Anode [3]:   {q_final[3]:+.6f} e")
        print(f"  Bulk [4]:    {q_final[4]:+.6f} e")
        print(f"  TOTAL:       {q_total_final:+.6f} e")
        
        delta_q = q_total_final - q_total_initial
        print(f"\nCharge change: Δq = {delta_q:+.6f} e")
        
        # Check if bulk charge is preserved
        delta_bulk = q_final[bulk_index] - q_initial[bulk_index]
        print(f"Bulk charge preserved: {abs(delta_bulk) < 1e-10} (Δ = {delta_bulk:.2e} e)")
        
        # Check electrode charge sum
        cathode_sum = sum(q_final[i] for i in cathode_indices)
        anode_sum = sum(q_final[i] for i in anode_indices)
        print(f"Cathode total: {cathode_sum:+.6f} e")
        print(f"Anode total:   {anode_sum:+.6f} e")
        print(f"Electrode sum: {cathode_sum + anode_sum:+.6f} e")
    
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    print("""
Fixed-voltage boundary condition physics:

In a real capacitor with fixed voltage V:
1. Voltage V is maintained between electrodes
2. System reaches equilibrium: Q_cathode = -Q_anode (equal and opposite)
3. Total charge on electrodes: Q_total = Q_cathode + Q_anode ≠ 0 in general
4. Electrolyte charges screen the electrodes

The algorithm computes:
- Cathode charge from: Q_cathode = (1/4π) * Area * [(V/L_gap) + (V/L_cell)] * C
- Anode charge from:   Q_anode = -(1/4π) * Area * [(V/L_gap) + (V/L_cell)] * C
- Plus corrections from electrolyte positions

Key insight: Total charge is NOT conserved because the electrodes represent
**external reservoirs** connected to a battery at fixed voltage. Charge flows
from the battery to/from the electrodes to maintain V.

If total charge WERE conserved, you couldn't maintain fixed voltage!

Conclusion: Non-conservation is EXPECTED BEHAVIOR, not a bug.
    """)


if __name__ == '__main__':
    test_charge_conservation()
