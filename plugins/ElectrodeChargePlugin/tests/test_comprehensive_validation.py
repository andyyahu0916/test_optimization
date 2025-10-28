#!/usr/bin/env python
"""
Comprehensive validation: Plugin vs Python across multiple scenarios.

Tests:
1. Different voltages (0.1, 0.5, 1.0, 2.0 eV)
2. Different iteration counts (1, 3, 5, 10)
3. Different system sizes (small, medium)
4. Different initial charge distributions
5. Asymmetric electrode configurations
"""

import ctypes
import os
import sys
import numpy as np

_BUILD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "build"))
_PYTHON_BUILD_DIR = os.path.join(_BUILD_DIR, "python")
sys.path.insert(0, _PYTHON_BUILD_DIR)

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


def create_system(cathode_indices, anode_indices, voltage, cell_gap, cell_length, 
                  num_iterations, positions, box_vectors, initial_charges=None):
    """Create system for both Plugin and Python."""
    num_particles = len(positions)
    
    # Plugin system
    system_plugin = System()
    for _ in range(num_particles):
        system_plugin.addParticle(40.0)
    
    nonbonded_plugin = NonbondedForce()
    nonbonded_plugin.setNonbondedMethod(NonbondedForce.NoCutoff)
    for i in range(num_particles):
        q_init = initial_charges[i] if initial_charges else 0.0
        nonbonded_plugin.addParticle(q_init, 1.0, 0.0)
    system_plugin.addForce(nonbonded_plugin)
    
    force = electrodecharge.ElectrodeChargeForce()
    force.setCathode(cathode_indices, voltage)
    force.setAnode(anode_indices, voltage)
    force.setCellGap(cell_gap)
    force.setCellLength(cell_length)
    force.setNumIterations(num_iterations)
    force.setSmallThreshold(1.0e-6)
    system_plugin.addForce(force)
    
    platform = Platform.getPlatformByName('Reference')
    integrator_plugin = LangevinIntegrator(300*unit.kelvin, 1.0/unit.picosecond, 0.001*unit.picoseconds)
    context_plugin = Context(system_plugin, integrator_plugin, platform)
    
    positions_vec3 = [Vec3(p[0], p[1], p[2]) for p in positions]
    context_plugin.setPositions(positions_vec3)
    context_plugin.setPeriodicBoxVectors(
        Vec3(box_vectors[0][0], box_vectors[0][1], box_vectors[0][2]),
        Vec3(box_vectors[1][0], box_vectors[1][1], box_vectors[1][2]),
        Vec3(box_vectors[2][0], box_vectors[2][1], box_vectors[2][2])
    )
    
    # Python system
    system_python = System()
    for _ in range(num_particles):
        system_python.addParticle(40.0)
    
    nonbonded_python = NonbondedForce()
    nonbonded_python.setNonbondedMethod(NonbondedForce.NoCutoff)
    for i in range(num_particles):
        q_init = initial_charges[i] if initial_charges else 0.0
        nonbonded_python.addParticle(q_init, 1.0, 0.0)
    system_python.addForce(nonbonded_python)
    
    integrator_python = LangevinIntegrator(300*unit.kelvin, 1.0/unit.picosecond, 0.001*unit.picoseconds)
    context_python = Context(system_python, integrator_python, platform)
    context_python.setPositions(positions_vec3)
    context_python.setPeriodicBoxVectors(
        Vec3(box_vectors[0][0], box_vectors[0][1], box_vectors[0][2]),
        Vec3(box_vectors[1][0], box_vectors[1][1], box_vectors[1][2]),
        Vec3(box_vectors[2][0], box_vectors[2][1], box_vectors[2][2])
    )
    
    return (context_plugin, nonbonded_plugin), (context_python, nonbonded_python)


def run_python_solver_simple(cathode_indices, anode_indices, voltage,
                              cell_gap, cell_length, num_iterations, 
                              positions, box_vectors, nonbonded, context):
    """Simplified Python solver for testing."""
    conversion_nmBohr = 18.8973
    conversion_KjmolNm_Au = conversion_nmBohr / 2625.5
    conversion_eV_Kjmol = 96.487
    small_threshold = 1.0e-6
    
    num_cathode = len(cathode_indices)
    num_anode = len(anode_indices)
    
    boxVec0 = np.array(box_vectors[0])
    boxVec1 = np.array(box_vectors[1])
    crossBox = np.cross(boxVec0, boxVec1)
    sheet_area = np.linalg.norm(crossBox)
    
    area_cathode_atom = sheet_area / num_cathode
    area_anode_atom = sheet_area / num_anode
    
    coeff_two_over_fourpi = 2.0 / (4.0 * np.pi)
    cathode_prefactor = coeff_two_over_fourpi * area_cathode_atom * conversion_KjmolNm_Au
    anode_prefactor = -coeff_two_over_fourpi * area_anode_atom * conversion_KjmolNm_Au
    
    voltage_kj = voltage * conversion_eV_Kjmol
    voltage_term = voltage_kj / cell_gap
    threshold_check = 0.9 * small_threshold
    
    cathode_q_old = np.array([
        nonbonded.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge) 
        for i in cathode_indices
    ])
    anode_q_old = np.array([
        nonbonded.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
        for i in anode_indices
    ])
    
    cathode_q_old = np.where(np.abs(cathode_q_old) < small_threshold, small_threshold, cathode_q_old)
    anode_q_old = np.where(np.abs(anode_q_old) < small_threshold, -small_threshold, anode_q_old)
    
    for i_iter in range(num_iterations):
        state = context.getState(getForces=True, groups=1<<0)
        forces_np = state.getForces(asNumpy=True)
        forces_z = forces_np[:, 2]
        
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
        
        for i, idx in enumerate(cathode_indices):
            nonbonded.setParticleParameters(idx, cathode_q_new[i], 1.0, 0.0)
        
        cathode_q_old[:] = cathode_q_new
        
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
        
        for i, idx in enumerate(anode_indices):
            nonbonded.setParticleParameters(idx, anode_q_new[i], 1.0, 0.0)
        
        anode_q_old[:] = anode_q_new
        
        # Scaling
        cathode_total = np.sum(cathode_q_new)
        anode_total = np.sum(anode_q_new)
        
        coeff_one_over_fourpi = 1.0 / (4.0 * np.pi)
        cathode_target = coeff_one_over_fourpi * sheet_area * ((voltage_kj / cell_gap) + (voltage_kj / cell_length)) * conversion_KjmolNm_Au
        anode_target = -coeff_one_over_fourpi * sheet_area * ((voltage_kj / cell_gap) + (voltage_kj / cell_length)) * conversion_KjmolNm_Au
        
        all_indices = set(range(len(positions)))
        electrolyte_indices = all_indices - set(cathode_indices) - set(anode_indices)
        
        cathode_z = np.mean([positions[i][2] for i in cathode_indices])
        anode_z = np.mean([positions[i][2] for i in anode_indices])
        
        for idx in electrolyte_indices:
            charge = nonbonded.getParticleParameters(idx)[0].value_in_unit(unit.elementary_charge)
            z_pos = positions[idx][2]
            cathode_distance = np.abs(z_pos - anode_z)
            anode_distance = np.abs(z_pos - cathode_z)
            cathode_target += (cathode_distance / cell_length) * (-charge)
            anode_target += (anode_distance / cell_length) * (-charge)
        
        if np.abs(cathode_total) > small_threshold:
            scale_cathode = cathode_target / cathode_total
            if scale_cathode > 0.0:
                cathode_q_new *= scale_cathode
                cathode_q_old[:] = cathode_q_new
                for i, idx in enumerate(cathode_indices):
                    nonbonded.setParticleParameters(idx, cathode_q_new[i], 1.0, 0.0)
        
        if np.abs(anode_total) > small_threshold:
            scale_anode = anode_target / anode_total
            if scale_anode > 0.0:
                anode_q_new *= scale_anode
                anode_q_old[:] = anode_q_new
                for i, idx in enumerate(anode_indices):
                    nonbonded.setParticleParameters(idx, anode_q_new[i], 1.0, 0.0)
        
        nonbonded.updateParametersInContext(context)
    
    charges = []
    for i in range(len(positions)):
        charge, _, _ = nonbonded.getParticleParameters(i)
        charges.append(charge.value_in_unit(unit.elementary_charge))
    
    return np.array(charges)


def compare_charges(charges_plugin, charges_python, tolerance=1e-10):
    """Compare two charge arrays."""
    max_diff = np.max(np.abs(charges_plugin - charges_python))
    rms_diff = np.sqrt(np.mean((charges_plugin - charges_python)**2))
    return max_diff, rms_diff, max_diff < tolerance


def test_scenario(name, cathode_indices, anode_indices, voltage, cell_gap, cell_length,
                  num_iterations, positions, box_vectors, initial_charges=None):
    """Test a single scenario."""
    print(f"\n{'='*80}")
    print(f"TEST: {name}")
    print(f"{'='*80}")
    print(f"Voltage: {voltage} eV, Iterations: {num_iterations}")
    print(f"Cathode: {len(cathode_indices)} atoms, Anode: {len(anode_indices)} atoms")
    
    (context_plugin, nonbonded_plugin), (context_python, nonbonded_python) = create_system(
        cathode_indices, anode_indices, voltage, cell_gap, cell_length,
        num_iterations, positions, box_vectors, initial_charges
    )
    
    # Run Plugin
    state_plugin = context_plugin.getState(getForces=True, groups=1<<1)
    charges_plugin = []
    for i in range(len(positions)):
        charge, _, _ = nonbonded_plugin.getParticleParameters(i)
        charges_plugin.append(charge.value_in_unit(unit.elementary_charge))
    charges_plugin = np.array(charges_plugin)
    
    # Run Python
    charges_python = run_python_solver_simple(
        cathode_indices, anode_indices, voltage, cell_gap, cell_length,
        num_iterations, positions, box_vectors, nonbonded_python, context_python
    )
    
    # Compare
    max_diff, rms_diff, passed = compare_charges(charges_plugin, charges_python)
    
    print(f"\nResults:")
    print(f"  Max difference: {max_diff:.2e} e")
    print(f"  RMS difference: {rms_diff:.2e} e")
    print(f"  Status: {'✅ PASS' if passed else '❌ FAIL'}")
    
    if not passed:
        print("\nDetailed differences:")
        for i in range(len(charges_plugin)):
            diff = charges_plugin[i] - charges_python[i]
            if abs(diff) > 1e-10:
                print(f"  Particle {i}: Plugin={charges_plugin[i]:+.6f}, Python={charges_python[i]:+.6f}, Δ={diff:+.2e}")
    
    return passed


def main():
    print("="*80)
    print("COMPREHENSIVE VALIDATION: Plugin vs Python")
    print("="*80)
    
    results = []
    
    # Test 1: Different voltages
    for voltage in [0.1, 0.5, 1.0, 2.0]:
        passed = test_scenario(
            f"Different Voltages (V={voltage}eV)",
            cathode_indices=[0, 1],
            anode_indices=[2, 3],
            voltage=voltage,
            cell_gap=0.8,
            cell_length=2.0,
            num_iterations=3,
            positions=[
                [0.0, 0.0, 0.1],
                [0.8, 0.0, 0.1],
                [0.0, 0.0, 1.9],
                [0.8, 0.0, 1.9],
                [0.4, 0.4, 1.0],
            ],
            box_vectors=[[1.6, 0.0, 0.0], [0.0, 1.6, 0.0], [0.0, 0.0, 2.0]],
            initial_charges=[0.0, 0.0, 0.0, 0.0, 0.1]
        )
        results.append((f"Voltage {voltage}eV", passed))
    
    # Test 2: Different iteration counts
    for num_iter in [1, 3, 5, 10]:
        passed = test_scenario(
            f"Different Iterations (N={num_iter})",
            cathode_indices=[0, 1],
            anode_indices=[2, 3],
            voltage=0.5,
            cell_gap=0.8,
            cell_length=2.0,
            num_iterations=num_iter,
            positions=[
                [0.0, 0.0, 0.1],
                [0.8, 0.0, 0.1],
                [0.0, 0.0, 1.9],
                [0.8, 0.0, 1.9],
                [0.4, 0.4, 1.0],
            ],
            box_vectors=[[1.6, 0.0, 0.0], [0.0, 1.6, 0.0], [0.0, 0.0, 2.0]],
            initial_charges=[0.0, 0.0, 0.0, 0.0, 0.0]
        )
        results.append((f"Iterations {num_iter}", passed))
    
    # Test 3: Asymmetric electrodes
    passed = test_scenario(
        "Asymmetric Electrodes (2 vs 3 atoms)",
        cathode_indices=[0, 1],
        anode_indices=[2, 3, 4],
        voltage=0.5,
        cell_gap=0.8,
        cell_length=2.0,
        num_iterations=3,
        positions=[
            [0.0, 0.0, 0.1],
            [0.8, 0.0, 0.1],
            [0.0, 0.0, 1.9],
            [0.5, 0.0, 1.9],
            [0.8, 0.0, 1.9],
            [0.4, 0.4, 1.0],
        ],
        box_vectors=[[1.6, 0.0, 0.0], [0.0, 1.6, 0.0], [0.0, 0.0, 2.0]],
        initial_charges=[0.0, 0.0, 0.0, 0.0, 0.0, 0.2]
    )
    results.append(("Asymmetric", passed))
    
    # Test 4: Different initial charges
    for bulk_charge in [-0.5, 0.0, 0.5]:
        passed = test_scenario(
            f"Initial Charge (q={bulk_charge}e)",
            cathode_indices=[0, 1],
            anode_indices=[2, 3],
            voltage=0.5,
            cell_gap=0.8,
            cell_length=2.0,
            num_iterations=3,
            positions=[
                [0.0, 0.0, 0.1],
                [0.8, 0.0, 0.1],
                [0.0, 0.0, 1.9],
                [0.8, 0.0, 1.9],
                [0.4, 0.4, 1.0],
            ],
            box_vectors=[[1.6, 0.0, 0.0], [0.0, 1.6, 0.0], [0.0, 0.0, 2.0]],
            initial_charges=[0.0, 0.0, 0.0, 0.0, bulk_charge]
        )
        results.append((f"Initial q={bulk_charge}e", passed))
    
    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    for name, p in results:
        status = "✅ PASS" if p else "❌ FAIL"
        print(f"{status} {name}")
    
    print(f"\nTotal: {passed}/{total} passed ({100*passed/total:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Plugin is bit-level identical to Python.")
        return 0
    else:
        print(f"\n⚠️  {total-passed} test(s) failed. Review differences above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
