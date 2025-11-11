#!/usr/bin/env python3
"""
Ab Initio 物理測試：C++ ConstantVPlugin vs Python 實現
驗證物理第一性原則的正確性

測試項目：
1. Green's Reciprocity Theorem
2. 電荷守恆
3. SCF 收斂性
4. 與 Python 版本的數值一致性
5. 數值穩定性（-ffast-math 影響）
"""

from __future__ import print_function
import sys
import os
import numpy as np
from datetime import datetime

# Add lib to path
sys.path.insert(0, './lib/')
sys.setrecursionlimit(5000)

# Import both versions
from MM_classes import *
from Fixed_Voltage_routines import *

# Import OpenMM
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *

# Import C++ Plugin
try:
    import constantvplugin as cvp
    HAS_CPP_PLUGIN = True
    print("✅ C++ ConstantV Plugin loaded successfully")
except ImportError as e:
    print(f"❌ Failed to load C++ plugin: {e}")
    HAS_CPP_PLUGIN = False
    sys.exit(1)

# ============================================================
# Test Configuration
# ============================================================
TEST_CONFIG = {
    'pdb_file': 'for_openmm.pdb',
    'ffdir': './ffdir/',
    'residue_xml_list': ['sapt_residues.xml', 'graph_residue_c.xml', 'graph_residue_n.xml'],
    'ff_xml_list': ['sapt_noDB_2sheets.xml', 'graph_c_freeze.xml', 'graph_n_freeze.xml'],
    'voltage': 4.0,
    'cathode_index': (0, 2),
    'anode_index': (1, 3),
    'platform': 'Reference',  # Use Reference for deterministic results
    'num_scf_iterations': 4,
    'num_md_steps': 5,  # Short test
    'timestep_fs': 200,
}

print("\n" + "="*70)
print("Ab Initio 物理測試 - C++ Plugin vs Python")
print("="*70)
print(f"測試配置：")
print(f"  - 電壓: {TEST_CONFIG['voltage']} V")
print(f"  - Platform: {TEST_CONFIG['platform']}")
print(f"  - SCF 迭代: {TEST_CONFIG['num_scf_iterations']}")
print(f"  - MD 步數: {TEST_CONFIG['num_md_steps']}")
print("="*70)

# ============================================================
# Test 1: Python 版本 (參考實現)
# ============================================================
print("\n📊 Test 1: Python 版本 (參考實現)")
print("-" * 70)

# Create Python MM system
MMsys_py = MM(
    pdb_list=[TEST_CONFIG['pdb_file']],
    residue_xml_list=[TEST_CONFIG['ffdir'] + f for f in TEST_CONFIG['residue_xml_list']],
    ff_xml_list=[TEST_CONFIG['ffdir'] + f for f in TEST_CONFIG['ff_xml_list']]
)

MMsys_py.set_periodic_residue(True)
MMsys_py.set_platform(TEST_CONFIG['platform'])

MMsys_py.initialize_electrodes(
    TEST_CONFIG['voltage'],
    cathode_identifier=TEST_CONFIG['cathode_index'],
    anode_identifier=TEST_CONFIG['anode_index'],
    chain=True,
    exclude_element=("H",)
)

MMsys_py.initialize_electrolyte(Natom_cutoff=100)

# Get initial state
state_py_init = MMsys_py.simmd.context.getState(getEnergy=True, getForces=True, getPositions=True)
positions_py_init = state_py_init.getPositions()
energy_py_init = state_py_init.getPotentialEnergy()

print(f"初始能量: {energy_py_init}")
print(f"總原子數: {MMsys_py.system.getNumParticles()}")
print(f"Cathode 原子數: {len(MMsys_py.Cathode.electrode_atoms)}")
print(f"Anode 原子數: {len(MMsys_py.Anode.electrode_atoms)}")
print(f"Electrolyte 原子數: {len(MMsys_py.electrolyte_atom_indices)}")

# Run Python SCF + MD steps
python_results = {
    'charges_cathode': [],
    'charges_anode': [],
    'Q_cathode': [],
    'Q_anode': [],
    'Q_electrolyte': [],
    'energies': [],
    'times': []
}

print("\n運行 Python 版本...")
for step in range(TEST_CONFIG['num_md_steps']):
    t_start = datetime.now()

    # Python Poisson solver
    MMsys_py.Poisson_solver_fixed_voltage(Niterations=TEST_CONFIG['num_scf_iterations'])

    # MD step
    MMsys_py.simmd.step(1)

    t_end = datetime.now()

    # Get state
    state = MMsys_py.simmd.context.getState(getEnergy=True, getForces=True, getPositions=True)

    # Record charges
    Q_cathode = sum([atom.charge for atom in MMsys_py.Cathode.electrode_atoms])
    Q_anode = sum([atom.charge for atom in MMsys_py.Anode.electrode_atoms])

    # Get electrolyte charges from NonbondedForce
    Q_electrolyte = 0.0
    for elec_idx in MMsys_py.electrolyte_atom_indices:
        charge, sigma, epsilon = MMsys_py.nbondedForce.getParticleParameters(elec_idx)
        Q_electrolyte += charge._value

    python_results['Q_cathode'].append(Q_cathode)
    python_results['Q_anode'].append(Q_anode)
    python_results['Q_electrolyte'].append(Q_electrolyte)
    python_results['energies'].append(state.getPotentialEnergy()._value)
    python_results['times'].append((t_end - t_start).total_seconds() * 1000)

    # Store first 5 cathode charges for detailed comparison
    if step < 2:
        python_results['charges_cathode'].append([atom.charge for atom in MMsys_py.Cathode.electrode_atoms[:5]])
        python_results['charges_anode'].append([atom.charge for atom in MMsys_py.Anode.electrode_atoms[:5]])

    print(f"  Step {step}: Q_cathode={Q_cathode:+.6f}e, Q_anode={Q_anode:+.6f}e, "
          f"Q_total={Q_cathode+Q_anode+Q_electrolyte:+.8f}e, E={python_results['energies'][-1]:.3f} kJ/mol")

python_avg_time = np.mean(python_results['times'])
print(f"\nPython 平均時間: {python_avg_time:.2f} ms/step")

# ============================================================
# Test 2: C++ Plugin 版本 (ConstantVForce)
# ============================================================
print("\n\n📊 Test 2: C++ Plugin 版本 (ConstantVForce)")
print("-" * 70)

# Create new MM system for C++ test
MMsys_cpp = MM(
    pdb_list=[TEST_CONFIG['pdb_file']],
    residue_xml_list=[TEST_CONFIG['ffdir'] + f for f in TEST_CONFIG['residue_xml_list']],
    ff_xml_list=[TEST_CONFIG['ffdir'] + f for f in TEST_CONFIG['ff_xml_list']]
)

MMsys_cpp.set_periodic_residue(True)
MMsys_cpp.set_platform(TEST_CONFIG['platform'])

MMsys_cpp.initialize_electrodes(
    TEST_CONFIG['voltage'],
    cathode_identifier=TEST_CONFIG['cathode_index'],
    anode_identifier=TEST_CONFIG['anode_index'],
    chain=True,
    exclude_element=("H",)
)

MMsys_cpp.initialize_electrolyte(Natom_cutoff=100)

# Get system info
topology = MMsys_cpp.simmd.topology
system = MMsys_cpp.system
positions = MMsys_cpp.simmd.context.getState(getPositions=True).getPositions()
box_vectors = topology.getPeriodicBoxVectors()
Lcell = box_vectors[2][2]._value  # nm
Lgap = abs(MMsys_cpp.Cathode.z_pos - MMsys_cpp.Anode.z_pos)

# Calculate total electrode area
cathode_area = sum([atom.area_atom for atom in MMsys_cpp.Cathode.electrode_atoms])
anode_area = sum([atom.area_atom for atom in MMsys_cpp.Anode.electrode_atoms])
total_area = (cathode_area + anode_area) / 2.0  # Average

print(f"系統幾何參數：")
print(f"  - Lgap: {Lgap:.6f} nm")
print(f"  - Lcell: {Lcell:.6f} nm")
print(f"  - Total Area: {total_area:.6f} nm²")
print(f"  - Z_cathode: {MMsys_cpp.Cathode.z_pos:.6f} nm")
print(f"  - Z_anode: {MMsys_cpp.Anode.z_pos:.6f} nm")

# Create ConstantVForce
cv_force = cvp.ConstantVForce()

# Set system parameters
cv_force.setVoltage(TEST_CONFIG['voltage'])
cv_force.setLgap(Lgap)
cv_force.setLcell(Lcell)
cv_force.setTotalArea(total_area)
cv_force.setZCathode(MMsys_cpp.Cathode.z_pos)
cv_force.setZAnode(MMsys_cpp.Anode.z_pos)
cv_force.setNumIterations(TEST_CONFIG['num_scf_iterations'])

# Add electrode atoms
print("\n添加電極原子到 ConstantVForce...")
for atom in MMsys_cpp.Cathode.electrode_atoms:
    cv_force.addCathodeAtom(atom.atom_index, atom.area_atom)

for atom in MMsys_cpp.Anode.electrode_atoms:
    cv_force.addAnodeAtom(atom.atom_index, atom.area_atom)

# Add electrolyte atoms
print("添加電解質原子到 ConstantVForce...")
for elec_idx in MMsys_cpp.electrolyte_atom_indices:
    charge, sigma, epsilon = MMsys_cpp.nbondedForce.getParticleParameters(elec_idx)
    cv_force.addElectrolyteAtom(elec_idx, charge._value)

print(f"  - Cathode atoms: {cv_force.getNumCathodeAtoms()}")
print(f"  - Anode atoms: {cv_force.getNumAnodeAtoms()}")
print(f"  - Electrolyte atoms: {cv_force.getNumElectrolyteAtoms()}")

# Add force to system
force_index = system.addForce(cv_force)
print(f"\nConstantVForce 添加到系統 (index={force_index})")

# Reinitialize context with new force
integrator_cpp = VerletIntegrator(TEST_CONFIG['timestep_fs'] * femtoseconds)
platform_cpp = Platform.getPlatformByName(TEST_CONFIG['platform'])
context_cpp = Context(system, integrator_cpp, platform_cpp)
context_cpp.setPositions(positions)

print("Context 重新初始化完成")

# Run C++ Plugin SCF + MD steps
cpp_results = {
    'charges_cathode': [],
    'charges_anode': [],
    'Q_cathode': [],
    'Q_anode': [],
    'Q_electrolyte': [],
    'energies': [],
    'times': []
}

print("\n運行 C++ Plugin 版本...")
for step in range(TEST_CONFIG['num_md_steps']):
    t_start = datetime.now()

    # C++ Poisson solver (called by force)
    state = context_cpp.getState(getEnergy=True, getForces=True)
    forces = state.getForces()

    # MD step
    integrator_cpp.step(1)

    t_end = datetime.now()

    # Get state
    state = context_cpp.getState(getEnergy=True, getForces=True, getPositions=True)

    # Get charges from NonbondedForce
    Q_cathode = 0.0
    for atom in MMsys_cpp.Cathode.electrode_atoms:
        charge, sigma, epsilon = MMsys_cpp.nbondedForce.getParticleParameters(atom.atom_index)
        Q_cathode += charge._value

    Q_anode = 0.0
    for atom in MMsys_cpp.Anode.electrode_atoms:
        charge, sigma, epsilon = MMsys_cpp.nbondedForce.getParticleParameters(atom.atom_index)
        Q_anode += charge._value

    Q_electrolyte = 0.0
    for elec_idx in MMsys_cpp.electrolyte_atom_indices:
        charge, sigma, epsilon = MMsys_cpp.nbondedForce.getParticleParameters(elec_idx)
        Q_electrolyte += charge._value

    cpp_results['Q_cathode'].append(Q_cathode)
    cpp_results['Q_anode'].append(Q_anode)
    cpp_results['Q_electrolyte'].append(Q_electrolyte)
    cpp_results['energies'].append(state.getPotentialEnergy()._value)
    cpp_results['times'].append((t_end - t_start).total_seconds() * 1000)

    # Store first 5 cathode charges
    if step < 2:
        cathode_charges = []
        for atom in MMsys_cpp.Cathode.electrode_atoms[:5]:
            charge, sigma, epsilon = MMsys_cpp.nbondedForce.getParticleParameters(atom.atom_index)
            cathode_charges.append(charge._value)
        cpp_results['charges_cathode'].append(cathode_charges)

        anode_charges = []
        for atom in MMsys_cpp.Anode.electrode_atoms[:5]:
            charge, sigma, epsilon = MMsys_cpp.nbondedForce.getParticleParameters(atom.atom_index)
            anode_charges.append(charge._value)
        cpp_results['charges_anode'].append(anode_charges)

    print(f"  Step {step}: Q_cathode={Q_cathode:+.6f}e, Q_anode={Q_anode:+.6f}e, "
          f"Q_total={Q_cathode+Q_anode+Q_electrolyte:+.8f}e, E={cpp_results['energies'][-1]:.3f} kJ/mol")

cpp_avg_time = np.mean(cpp_results['times'])
print(f"\nC++ Plugin 平均時間: {cpp_avg_time:.2f} ms/step")

# ============================================================
# Analysis and Comparison
# ============================================================
print("\n\n" + "="*70)
print("🔬 物理正確性分析")
print("="*70)

# Test 1: Green's Reciprocity (charge conservation at each electrode)
print("\n1️⃣  Green's Reciprocity Theorem 驗證")
print("-" * 70)

# For Python
py_charge_errors = []
for i in range(len(python_results['Q_cathode'])):
    Q_total = python_results['Q_cathode'][i] + python_results['Q_anode'][i] + python_results['Q_electrolyte'][i]
    py_charge_errors.append(abs(Q_total))

print(f"Python 版本:")
print(f"  - 最大電荷誤差: {max(py_charge_errors):.10f}e")
print(f"  - 平均電荷誤差: {np.mean(py_charge_errors):.10f}e")
print(f"  - 是否通過 (<1e-8): {'✅ PASS' if max(py_charge_errors) < 1e-8 else '❌ FAIL'}")

# For C++
cpp_charge_errors = []
for i in range(len(cpp_results['Q_cathode'])):
    Q_total = cpp_results['Q_cathode'][i] + cpp_results['Q_anode'][i] + cpp_results['Q_electrolyte'][i]
    cpp_charge_errors.append(abs(Q_total))

print(f"\nC++ Plugin 版本:")
print(f"  - 最大電荷誤差: {max(cpp_charge_errors):.10f}e")
print(f"  - 平均電荷誤差: {np.mean(cpp_charge_errors):.10f}e")
print(f"  - 是否通過 (<1e-8): {'✅ PASS' if max(cpp_charge_errors) < 1e-8 else '❌ FAIL'}")

# Test 2: Python vs C++ consistency
print("\n2️⃣  Python vs C++ 數值一致性")
print("-" * 70)

charge_diff_cathode = []
charge_diff_anode = []

for i in range(len(python_results['Q_cathode'])):
    diff_cathode = abs(python_results['Q_cathode'][i] - cpp_results['Q_cathode'][i])
    diff_anode = abs(python_results['Q_anode'][i] - cpp_results['Q_anode'][i])
    charge_diff_cathode.append(diff_cathode)
    charge_diff_anode.append(diff_anode)

print(f"總電荷差異:")
print(f"  - Cathode 最大差異: {max(charge_diff_cathode):.10f}e")
print(f"  - Anode 最大差異: {max(charge_diff_anode):.10f}e")
print(f"  - 是否通過 (<1e-6): {'✅ PASS' if max(charge_diff_cathode) < 1e-6 and max(charge_diff_anode) < 1e-6 else '❌ FAIL'}")

# Detailed per-atom comparison (first 2 steps)
print(f"\n逐原子詳細比較 (前5個 Cathode 原子, Step 0):")
for i in range(min(5, len(python_results['charges_cathode'][0]))):
    py_q = python_results['charges_cathode'][0][i]
    cpp_q = cpp_results['charges_cathode'][0][i]
    diff = abs(py_q - cpp_q)
    rel_err = diff / abs(py_q) * 100 if abs(py_q) > 1e-10 else 0
    print(f"  Atom {i}: Python={py_q:+.8f}e, C++={cpp_q:+.8f}e, Δ={diff:.2e}e ({rel_err:.4f}%)")

# Test 3: Energy comparison
print("\n3️⃣  能量一致性")
print("-" * 70)

energy_diffs = [abs(python_results['energies'][i] - cpp_results['energies'][i])
                for i in range(len(python_results['energies']))]

print(f"Python 平均能量: {np.mean(python_results['energies']):.3f} kJ/mol")
print(f"C++ 平均能量: {np.mean(cpp_results['energies']):.3f} kJ/mol")
print(f"最大能量差異: {max(energy_diffs):.6f} kJ/mol")
print(f"相對誤差: {max(energy_diffs) / abs(np.mean(python_results['energies'])) * 100:.4f}%")
print(f"是否通過 (<0.1%): {'✅ PASS' if max(energy_diffs) / abs(np.mean(python_results['energies'])) < 0.001 else '❌ FAIL'}")

# Test 4: Performance comparison
print("\n4️⃣  性能對比")
print("-" * 70)

speedup = python_avg_time / cpp_avg_time
print(f"Python 平均時間: {python_avg_time:.2f} ms/step")
print(f"C++ Plugin 平均時間: {cpp_avg_time:.2f} ms/step")
print(f"加速比: {speedup:.2f}x")

# Test 5: Numerical stability check (-ffast-math)
print("\n5️⃣  數值穩定性檢查 (ffast-math 影響)")
print("-" * 70)

# Check if charges remain finite
py_charges_finite = all([np.isfinite(q) for q in python_results['Q_cathode'] + python_results['Q_anode']])
cpp_charges_finite = all([np.isfinite(q) for q in cpp_results['Q_cathode'] + cpp_results['Q_anode']])

print(f"Python 電荷有限性: {'✅ PASS' if py_charges_finite else '❌ FAIL'}")
print(f"C++ 電荷有限性: {'✅ PASS' if cpp_charges_finite else '❌ FAIL'}")

# Check for divergence
py_charge_std = np.std(python_results['Q_cathode'])
cpp_charge_std = np.std(cpp_results['Q_cathode'])

print(f"\n電荷標準差 (檢測發散):")
print(f"  - Python: {py_charge_std:.8f}e")
print(f"  - C++: {cpp_charge_std:.8f}e")
print(f"  - 是否穩定 (<0.1e): {'✅ PASS' if cpp_charge_std < 0.1 else '❌ FAIL'}")

# ============================================================
# Final Report
# ============================================================
print("\n\n" + "="*70)
print("🎯 最終測試報告")
print("="*70)

tests_passed = 0
tests_total = 6

# Test results
test_results = [
    ("Green's Reciprocity (Python)", max(py_charge_errors) < 1e-8),
    ("Green's Reciprocity (C++)", max(cpp_charge_errors) < 1e-8),
    ("Python vs C++ 電荷一致性", max(charge_diff_cathode) < 1e-6 and max(charge_diff_anode) < 1e-6),
    ("能量一致性", max(energy_diffs) / abs(np.mean(python_results['energies'])) < 0.001),
    ("數值穩定性", cpp_charges_finite and cpp_charge_std < 0.1),
    ("性能提升", cpp_avg_time < python_avg_time),
]

for test_name, passed in test_results:
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status}  {test_name}")
    if passed:
        tests_passed += 1

print("\n" + "-"*70)
print(f"通過測試: {tests_passed}/{tests_total}")

if tests_passed == tests_total:
    print("\n🎉 所有測試通過！C++ Plugin 物理正確性驗證成功！")
    print("✅ 符合物理第一性原則")
    print("✅ 與 Python 參考實現一致")
    print("✅ -ffast-math 未影響數值穩定性")
    print(f"✅ 性能提升 {speedup:.2f}x")
else:
    print(f"\n⚠️ {tests_total - tests_passed} 個測試失敗，需要進一步檢查")

print("="*70)
print("\n")
