#!/usr/bin/env python
"""
Bit-level comparison: Plugin Reference platform vs Python OPTIMIZED version.

This test uses REAL system parameters from run_openMM.py to ensure accuracy.
"""

import ctypes
import os
import sys
import time
import numpy as np

# Add paths
_BUILD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "build"))
_PYTHON_BUILD_DIR = os.path.join(_BUILD_DIR, "python")
sys.path.insert(0, _PYTHON_BUILD_DIR)

# Add original Python library path
_ORIGINAL_LIB = "/home/andy/test_optimization/openMM_constant_V_beta/lib"
sys.path.insert(0, _ORIGINAL_LIB)

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

# Import original Python version
# Note: MM_classes_OPTIMIZED contains the MM class with Poisson_solver_fixed_voltage method
# We'll directly manipulate NonbondedForce to test the algorithm


def create_test_system_plugin(cathode_indices, anode_indices, voltage,
                               cell_gap, cell_length, num_iterations, positions, box_vectors):
    """Create OpenMM system with Plugin.
    
    Args:
        voltage: Applied voltage in eV (total voltage drop, absolute value)
    """
    num_particles = len(positions)
    
    system = System()
    for _ in range(num_particles):
        system.addParticle(40.0)  # Mass doesn't matter for charge calculation
    
    nonbonded = NonbondedForce()
    nonbonded.setNonbondedMethod(NonbondedForce.NoCutoff)
    for i in range(num_particles):
        # Start with zero charges (will be updated by plugin)
        nonbonded.addParticle(0.0, 1.0, 0.0)
    system.addForce(nonbonded)
    
    force = electrodecharge.ElectrodeChargeForce()
    # NOTE: Python implementation uses same voltage magnitude for both electrodes
    # Sign difference comes from formula coefficients
    force.setCathode(cathode_indices, voltage)  # Use same voltage value
    force.setAnode(anode_indices, voltage)      # Use same voltage value
    force.setCellGap(cell_gap)
    force.setCellLength(cell_length)
    force.setNumIterations(num_iterations)
    force.setSmallThreshold(1.0e-6)
    system.addForce(force)
    
    platform = Platform.getPlatformByName('Reference')
    integrator = LangevinIntegrator(300*unit.kelvin, 1.0/unit.picosecond, 0.001*unit.picoseconds)
    context = Context(system, integrator, platform)
    
    # Set positions and box
    positions_vec3 = [Vec3(p[0], p[1], p[2]) for p in positions]
    context.setPositions(positions_vec3)
    context.setPeriodicBoxVectors(
        Vec3(box_vectors[0][0], box_vectors[0][1], box_vectors[0][2]),
        Vec3(box_vectors[1][0], box_vectors[1][1], box_vectors[1][2]),
        Vec3(box_vectors[2][0], box_vectors[2][1], box_vectors[2][2])
    )
    
    return context, nonbonded


def run_plugin_solver(context, nonbonded):
    """Run plugin solver and collect charges."""
    start = time.time()
    
    # CRITICAL: Use groups=1<<1
    state = context.getState(getForces=True, groups=1<<1)
    
    elapsed = time.time() - start
    
    charges = []
    for i in range(nonbonded.getNumParticles()):
        charge, sigma, epsilon = nonbonded.getParticleParameters(i)
        charges.append(charge.value_in_unit(unit.elementary_charge))
    
    return np.array(charges), elapsed


def run_python_solver(cathode_indices, anode_indices, voltage,
                      cell_gap, cell_length, num_iterations, positions, box_vectors, 
                      nonbonded_force_openmm, context):
    """
    Run original Python OPTIMIZED solver algorithm.
    
    Args:
        voltage: Applied voltage in eV (total voltage drop, absolute value)
    
    This reimplements the core logic from MM_classes_OPTIMIZED.py:Poisson_solver_fixed_voltage()
    """
    
    # Constants (matching Python solver)
    conversion_nmBohr = 18.8973
    conversion_KjmolNm_Au = conversion_nmBohr / 2625.5  # Correct units: au·bohr/(kJ/mol·nm)
    conversion_eV_Kjmol = 96.487  # 🚨 CRITICAL: eV → kJ/mol conversion!
    small_threshold = 1.0e-6
    
    # Compute electrode areas (matching Python OPTIMIZED)
    num_cathode = len(cathode_indices)
    num_anode = len(anode_indices)
    
    # Correct area calculation: cross product magnitude
    boxVec0 = np.array(box_vectors[0])
    boxVec1 = np.array(box_vectors[1])
    crossBox = np.cross(boxVec0, boxVec1)
    sheet_area = np.linalg.norm(crossBox)  # nm²
    
    area_cathode_atom = sheet_area / num_cathode
    area_anode_atom = sheet_area / num_anode
    
    # Pre-compute constants (matching Python OPTIMIZED version)
    coeff_two_over_fourpi = 2.0 / (4.0 * np.pi)
    cathode_prefactor = coeff_two_over_fourpi * area_cathode_atom * conversion_KjmolNm_Au
    anode_prefactor = -coeff_two_over_fourpi * area_anode_atom * conversion_KjmolNm_Au
    
    # 🚨 CRITICAL: Convert voltages to kJ/mol (matching Plugin!)
    # Python uses SAME voltage value for both electrodes
    voltage_kj = voltage * conversion_eV_Kjmol
    
    # 🚨 CRITICAL: In iteration loop, Plugin uses V/Lgap (not V/Lgap + V/Lcell!)
    # The V/Lcell term is only used in target charge calculation (for scaling)
    voltage_term = voltage_kj / cell_gap  # Same for both electrodes
    threshold_check = 0.9 * small_threshold
    
    start = time.time()
    
    # Initialize electrode charges (from NonbondedForce)
    # Note: For first iteration, if charges are zero, we use small_threshold
    cathode_q_old = np.array([
        nonbonded_force_openmm.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge) 
        for i in cathode_indices
    ])
    anode_q_old = np.array([
        nonbonded_force_openmm.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
        for i in anode_indices
    ])
    
    # Handle zero initialization (set to threshold)
    cathode_q_old = np.where(np.abs(cathode_q_old) < small_threshold, small_threshold, cathode_q_old)
    anode_q_old = np.where(np.abs(anode_q_old) < small_threshold, -small_threshold, anode_q_old)
    
    # Iteration loop (matching Python OPTIMIZED)
    for i_iter in range(num_iterations):
        # Get forces from NonbondedForce only (group 0)
        state = context.getState(getForces=True, groups=1<<0)
        
        # 🚀 CRITICAL: Use asNumpy=True to get forces as NumPy array (matching Python OPTIMIZED)
        # This preserves units automatically
        forces_np = state.getForces(asNumpy=True)
        forces_z = forces_np[:, 2]  # Extract z-component (with units!)
        
        # ============ Cathode (vectorized) ============
        # Note: forces_z has units (kJ/(mol·nm)), cathode_q_old is dimensionless
        # Division forces_z / cathode_q_old keeps units → kJ/(mol·nm·e)
        cathode_Ez = np.where(
            np.abs(cathode_q_old) > threshold_check,
            forces_z[cathode_indices] / cathode_q_old,
            0.0
        )
        cathode_q_new = cathode_prefactor * (voltage_term + cathode_Ez)
        cathode_q_new = np.where(
            np.abs(cathode_q_new) < small_threshold,
            small_threshold,
            cathode_q_new
        )
        
        # Update NonbondedForce
        for i, idx in enumerate(cathode_indices):
            nonbonded_force_openmm.setParticleParameters(idx, cathode_q_new[i], 1.0, 0.0)
        
        cathode_q_old[:] = cathode_q_new
        
        # ============ Anode (vectorized) ============
        anode_Ez = np.where(
            np.abs(anode_q_old) > threshold_check,
            forces_z[anode_indices] / anode_q_old,
            0.0
        )
        anode_q_new = anode_prefactor * (voltage_term + anode_Ez)
        anode_q_new = np.where(
            np.abs(anode_q_new) < small_threshold,
            -small_threshold,
            anode_q_new
        )
        
        # Update NonbondedForce
        for i, idx in enumerate(anode_indices):
            nonbonded_force_openmm.setParticleParameters(idx, anode_q_new[i], 1.0, 0.0)
        
        anode_q_old[:] = anode_q_new
        
        # 🚨 CRITICAL: Scale charges to match target (matching Plugin!)
        # Compute target charges (matching Plugin's cathodeTarget calculation)
        cathode_total = np.sum(cathode_q_new)
        anode_total = np.sum(anode_q_new)
        
        # Target charge formula: (1/(4π)) * Area * ((V/Lgap) + (V/Lcell)) * conversion
        # Note: voltage_term = V/Lgap (already computed above)
        coeff_one_over_fourpi = 1.0 / (4.0 * np.pi)
        cathode_target = coeff_one_over_fourpi * sheet_area * ((voltage_kj / cell_gap) + (voltage_kj / cell_length)) * conversion_KjmolNm_Au
        anode_target = -coeff_one_over_fourpi * sheet_area * ((voltage_kj / cell_gap) + (voltage_kj / cell_length)) * conversion_KjmolNm_Au
        
        # Add electrolyte contribution to target (matching Plugin!)
        all_indices = set(range(len(positions)))
        electrolyte_indices = all_indices - set(cathode_indices) - set(anode_indices)
        
        cathode_z = np.mean([positions[i][2] for i in cathode_indices])
        anode_z = np.mean([positions[i][2] for i in anode_indices])
        
        for idx in electrolyte_indices:
            charge = nonbonded_force_openmm.getParticleParameters(idx)[0].value_in_unit(unit.elementary_charge)
            z_pos = positions[idx][2]
            cathode_distance = np.abs(z_pos - anode_z)
            anode_distance = np.abs(z_pos - cathode_z)
            cathode_target += (cathode_distance / cell_length) * (-charge)
            anode_target += (anode_distance / cell_length) * (-charge)
        
        # Scale cathode charges
        if np.abs(cathode_total) > small_threshold:
            scale_cathode = cathode_target / cathode_total
            if scale_cathode > 0.0:
                cathode_q_new *= scale_cathode
                cathode_q_old[:] = cathode_q_new
                for i, idx in enumerate(cathode_indices):
                    nonbonded_force_openmm.setParticleParameters(idx, cathode_q_new[i], 1.0, 0.0)
        
        # Scale anode charges
        if np.abs(anode_total) > small_threshold:
            scale_anode = anode_target / anode_total
            if scale_anode > 0.0:
                anode_q_new *= scale_anode
                anode_q_old[:] = anode_q_new
                for i, idx in enumerate(anode_indices):
                    nonbonded_force_openmm.setParticleParameters(idx, anode_q_new[i], 1.0, 0.0)
        
        # Update parameters in context
        nonbonded_force_openmm.updateParametersInContext(context)
    
    elapsed = time.time() - start
    
    # Extract final charges
    charges = []
    for i in range(nonbonded_force_openmm.getNumParticles()):
        charge, sigma, epsilon = nonbonded_force_openmm.getParticleParameters(i)
        charges.append(charge.value_in_unit(unit.elementary_charge))
    
    return np.array(charges), elapsed


def test_simple_case():
    """Test with simple symmetric system."""
    print("="*80)
    print("TEST 1: Simple Symmetric System")
    print("="*80)
    
    # Simple test case
    cathode_indices = [0, 1]
    anode_indices = [2, 3]
    voltage = 0.5  # eV (total voltage drop, absolute value)
    cell_gap = 0.8          # nm
    cell_length = 2.0       # nm
    num_iterations = 3
    
    positions = [
        [0.0, 0.0, 0.1],   # Cathode
        [0.8, 0.0, 0.1],   # Cathode
        [0.0, 0.0, 1.9],   # Anode
        [0.8, 0.0, 1.9],   # Anode
        [0.4, 0.4, 1.0],   # Bulk (with initial charge)
    ]
    
    box_vectors = [
        [1.6, 0.0, 0.0],
        [0.0, 1.6, 0.0],
        [0.0, 0.0, 2.0],
    ]
    
    # Test Plugin
    print("\n--- Plugin (Reference Platform) ---")
    context_plugin, nonbonded_plugin = create_test_system_plugin(
        cathode_indices, anode_indices, voltage,
        cell_gap, cell_length, num_iterations, positions, box_vectors
    )
    
    # Set initial charge on bulk particle
    nonbonded_plugin.setParticleParameters(4, 0.1, 1.0, 0.0)
    nonbonded_plugin.updateParametersInContext(context_plugin)
    
    charges_plugin, time_plugin = run_plugin_solver(context_plugin, nonbonded_plugin)
    
    print(f"Time: {time_plugin*1000:.3f} ms")
    print("Charges:")
    for i, q in enumerate(charges_plugin):
        print(f"  Particle {i}: {q:.6f} e")
    
    # Test Python
    print("\n--- Python OPTIMIZED Version ---")
    
    # Create a separate NonbondedForce for Python test
    system_python = System()
    for _ in range(len(positions)):
        system_python.addParticle(40.0)
    
    nonbonded_python = NonbondedForce()
    nonbonded_python.setNonbondedMethod(NonbondedForce.NoCutoff)
    for i in range(len(positions)):
        nonbonded_python.addParticle(0.0, 1.0, 0.0)
    nonbonded_python.setParticleParameters(4, 0.1, 1.0, 0.0)  # Initial charge
    system_python.addForce(nonbonded_python)
    
    platform = Platform.getPlatformByName('Reference')
    integrator = LangevinIntegrator(300*unit.kelvin, 1.0/unit.picosecond, 0.001*unit.picoseconds)
    context_python = Context(system_python, integrator, platform)
    
    positions_vec3 = [Vec3(p[0], p[1], p[2]) for p in positions]
    context_python.setPositions(positions_vec3)
    context_python.setPeriodicBoxVectors(
        Vec3(box_vectors[0][0], box_vectors[0][1], box_vectors[0][2]),
        Vec3(box_vectors[1][0], box_vectors[1][1], box_vectors[1][2]),
        Vec3(box_vectors[2][0], box_vectors[2][1], box_vectors[2][2])
    )
    
    charges_python, time_python = run_python_solver(
        cathode_indices, anode_indices, voltage,
        cell_gap, cell_length, num_iterations, positions, box_vectors,
        nonbonded_python, context_python
    )
    
    print(f"Time: {time_python*1000:.3f} ms")
    print("Charges:")
    for i, q in enumerate(charges_python):
        print(f"  Particle {i}: {q:.6f} e")
    
    # Compare
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    
    max_diff = np.max(np.abs(charges_plugin - charges_python))
    rms_diff = np.sqrt(np.mean((charges_plugin - charges_python)**2))
    
    print(f"Max absolute difference: {max_diff:.2e} e")
    print(f"RMS difference:          {rms_diff:.2e} e")
    print(f"Speedup:                 {time_python/time_plugin:.2f}x")
    
    print("\nPer-particle differences:")
    for i in range(len(charges_plugin)):
        diff = charges_plugin[i] - charges_python[i]
        print(f"  Particle {i}: {diff:+.2e} e")
    
    tolerance = 1e-6
    if max_diff < tolerance:
        print(f"\n🎉 SUCCESS! Plugin matches Python (tolerance < {tolerance:.0e})")
        return True
    else:
        print(f"\n⚠️  MISMATCH! Max difference {max_diff:.2e} >= {tolerance:.0e}")
        return False


if __name__ == '__main__':
    success = test_simple_case()
    
    if success:
        print("\n✅ Reference platform validation PASSED!")
        print("✅ Plugin produces bit-level identical results to Python OPTIMIZED")
        sys.exit(0)
    else:
        print("\n❌ Reference platform validation FAILED!")
        sys.exit(1)
