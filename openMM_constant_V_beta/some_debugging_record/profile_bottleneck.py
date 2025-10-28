#!/usr/bin/env python
"""
深度 Profile - 找出 Poisson solver 的真正瓶頸
"""
import sys
import time
import numpy as np
sys.path.append('./lib/')

from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
import configparser

# 讀取配置
config = configparser.ConfigParser()
config.read('config.ini')
ffdir = config['Files'].get('ffdir')
if not ffdir.endswith('/'):
    ffdir += '/'
pdb_list = [config['Files'].get('pdb_file')]
residue_xml_list = [ffdir + s.strip() for s in config['Files'].get('residue_xml_list').split(',')]
ff_xml_list = [ffdir + s.strip() for s in config['Files'].get('ff_xml_list').split(',')]

print("=" * 80)
print("深度 Profiling - Poisson Solver 瓶頸分析")
print("=" * 80)

# 測試 CYTHON 版本
from MM_classes_CYTHON import MM as MM_cython

print("\n創建系統...")
MMsys = MM_cython(pdb_list=pdb_list, residue_xml_list=residue_xml_list, ff_xml_list=ff_xml_list)
MMsys.set_platform('CUDA')
MMsys.set_periodic_residue(True)
MMsys.initialize_electrodes(
    Voltage=0.0,
    cathode_identifier=(0,2),
    anode_identifier=(1,3),
    chain=True,
    exclude_element=('H',)
)
MMsys.initialize_electrolyte(Natom_cutoff=100)

print(f"系統資訊:")
print(f"  陰極原子數: {len(MMsys.Cathode.electrode_atoms)}")
print(f"  陽極原子數: {len(MMsys.Anode.electrode_atoms)}")
print(f"  電解質原子數: {len(MMsys.electrolyte_atom_indices)}")

# 手動分解 Poisson solver 的每個步驟
print("\n" + "=" * 80)
print("逐步 Profiling (10 次迭代平均)")
print("=" * 80)

Niterations = 3
num_runs = 10

# Step 1: getState(getPositions=True)
times = []
for _ in range(num_runs):
    start = time.perf_counter()
    state = MMsys.simmd.context.getState(getEnergy=False, getForces=False, getVelocities=False, getPositions=True)
    times.append(time.perf_counter() - start)
print(f"\n1. getState(getPositions=True): {np.mean(times)*1000:.3f} ms")

# Step 2: Extract positions as NumPy
times = []
for _ in range(num_runs):
    state = MMsys.simmd.context.getState(getEnergy=False, getForces=False, getVelocities=False, getPositions=True)
    start = time.perf_counter()
    positions_np = state.getPositions(asNumpy=True)
    z_positions_array = positions_np[:, 2]._value if hasattr(positions_np[:, 2], '_value') else positions_np[:, 2]
    times.append(time.perf_counter() - start)
print(f"2. Extract z-positions (NumPy): {np.mean(times)*1000:.3f} ms")

# Step 3: Compute analytic charges (cathode + anode)
z_positions_array = state.getPositions(asNumpy=True)[:, 2]._value
times = []
for _ in range(num_runs):
    start = time.perf_counter()
    MMsys.Cathode.compute_Electrode_charge_analytic(MMsys, z_positions_array, MMsys.Conductor_list, z_opposite=MMsys.Anode.z_pos)
    MMsys.Anode.compute_Electrode_charge_analytic(MMsys, z_positions_array, MMsys.Conductor_list, z_opposite=MMsys.Cathode.z_pos)
    times.append(time.perf_counter() - start)
print(f"3. Compute analytic charges (both): {np.mean(times)*1000:.3f} ms")

print(f"\n{'─' * 80}")
print("迭代循環內部 (每次迭代):")
print(f"{'─' * 80}")

# Step 4: getState(getForces=True)
times = []
for _ in range(num_runs):
    start = time.perf_counter()
    state = MMsys.simmd.context.getState(getEnergy=False, getForces=True, getVelocities=False, getPositions=False)
    times.append(time.perf_counter() - start)
print(f"4. getState(getForces=True): {np.mean(times)*1000:.3f} ms")

# Step 5: Extract forces as NumPy
times = []
for _ in range(num_runs):
    state = MMsys.simmd.context.getState(getEnergy=False, getForces=True, getVelocities=False, getPositions=False)
    start = time.perf_counter()
    forces_np = state.getForces(asNumpy=True)
    forces_z = forces_np[:, 2]._value if hasattr(forces_np[:, 2], '_value') else forces_np[:, 2]
    times.append(time.perf_counter() - start)
print(f"5. Extract forces_z (NumPy): {np.mean(times)*1000:.3f} ms")

# Step 6: Collect old charges (Cython)
import electrode_charges_cython as ec_cython
times = []
for _ in range(num_runs):
    start = time.perf_counter()
    cathode_q_old = ec_cython.collect_electrode_charges_cython(
        MMsys.Cathode.electrode_atoms,
        MMsys.nbondedForce
    )
    times.append(time.perf_counter() - start)
print(f"6. Collect electrode charges (Cython): {np.mean(times)*1000:.3f} ms")

# Step 7: Compute new charges (Cython)
forces_np = state.getForces(asNumpy=True)
forces_z = forces_np[:, 2]._value
cathode_q_old = ec_cython.collect_electrode_charges_cython(MMsys.Cathode.electrode_atoms, MMsys.nbondedForce)
coeff_two_over_fourpi = 2.0 / (4.0 * np.pi)
cathode_prefactor = coeff_two_over_fourpi * MMsys.Cathode.area_atom * (18.8973 / 2625.5)
voltage_term_cathode = MMsys.Cathode.Voltage / MMsys.Lgap
threshold_check = 0.9 * MMsys.small_threshold

times = []
for _ in range(num_runs):
    start = time.perf_counter()
    cathode_q_new = ec_cython.compute_electrode_charges_cython(
        forces_z,
        cathode_q_old,
        MMsys._cathode_indices,
        cathode_prefactor,
        voltage_term_cathode,
        threshold_check,
        MMsys.small_threshold,
        1.0
    )
    times.append(time.perf_counter() - start)
print(f"7. Compute electrode charges (Cython): {np.mean(times)*1000:.3f} ms")

# Step 8: Update OpenMM charges (Cython batch)
cathode_q_new = ec_cython.compute_electrode_charges_cython(
    forces_z, cathode_q_old, MMsys._cathode_indices,
    cathode_prefactor, voltage_term_cathode, threshold_check,
    MMsys.small_threshold, 1.0
)
times = []
for _ in range(num_runs):
    start = time.perf_counter()
    ec_cython.update_openmm_charges_batch(
        MMsys.nbondedForce,
        MMsys.Cathode.electrode_atoms,
        cathode_q_new
    )
    times.append(time.perf_counter() - start)
print(f"8. Update OpenMM charges (Cython): {np.mean(times)*1000:.3f} ms")

# Step 9: Scale charges analytic
times = []
for _ in range(num_runs):
    start = time.perf_counter()
    MMsys.Scale_charges_analytic_general()
    times.append(time.perf_counter() - start)
print(f"9. Scale_charges_analytic_general: {np.mean(times)*1000:.3f} ms")

# Step 10: updateParametersInContext
times = []
for _ in range(num_runs):
    start = time.perf_counter()
    MMsys.nbondedForce.updateParametersInContext(MMsys.simmd.context)
    times.append(time.perf_counter() - start)
print(f"10. updateParametersInContext (GPU sync): {np.mean(times)*1000:.3f} ms")

# 完整測試
print(f"\n{'=' * 80}")
print("完整 Poisson Solver (3 次迭代):")
print(f"{'=' * 80}")
times = []
for _ in range(num_runs):
    start = time.perf_counter()
    MMsys.Poisson_solver_fixed_voltage(Niterations=3)
    times.append(time.perf_counter() - start)
print(f"總時間: {np.mean(times)*1000:.3f} ms")
print(f"理論時間 (步驟 1-3 + 3×(步驟 4-10)):")

# 計算理論時間
print("\n時間分解:")
print("  初始化部分 (一次):")
print("    - getState(positions): ?")
print("    - Extract positions: ?")
print("    - Compute analytic: ?")
print("  迭代部分 (×3):")
print("    - getState(forces): ?")
print("    - Extract forces: ?")
print("    - Collect charges (×2): ?")
print("    - Compute charges (×2): ?")
print("    - Update charges (×2): ?")
print("    - Scale analytic: ?")
print("    - GPU sync: ?")

print("\n✅ Profiling 完成！")
print("=" * 80)
