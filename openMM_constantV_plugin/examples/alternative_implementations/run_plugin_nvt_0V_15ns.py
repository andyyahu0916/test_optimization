#!/usr/bin/env python3
"""
Plugin版本的run_openMM.py - 完全复现Original的模拟

Original参考: /home/andy/test_optimization/OpenMM-ConstantV(original)/run_openMM.py

这个脚本将Original的模拟转换成Plugin版本，保持所有参数一致。
"""

import sys
import os
import shutil
from datetime import datetime

import openmm as mm
import openmm.app as app
import openmm.unit as unit

# Import plugin
from constantvplugin import ConstantVIntegrator
from constantvplugin_helpers import (
    add_electrode_exclusions,
    configure_geometry_from_context,
    add_electrolyte_atoms_auto,
    compute_electrode_area_per_atom,
    validate_setup
)

print("="*70)
print("Plugin版本 - 复现Original的nvt_0V_15ns.pdb模拟")
print("="*70)

# ═══════════════════════════════════════════════════════════
# 配置参数（从Original的run_openMM.py复制）
# ═══════════════════════════════════════════════════════════

# Original Line 34: 运行参数
simulation_time_ns = 0.5
freq_charge_update_fs = 200  # 每200 fs更新一次电荷
freq_traj_output_ps = 10     # 每10 ps输出一次轨迹
write_charges = False        # 是否写出电荷数据

# Original Line 37: 输出路径
outPath = '1v_0.5ns'

# Original Line 73: 电压
Voltage = 0.0  # Volts

# Original Line 78: 电极chain indices（注意Original用的是tuple）
cathode_index = (0, 2)  # chains 0 and 2
anode_index = (1, 3)    # chains 1 and 3

# Original Line 109: 排除元素
exclude_element = ("H",)  # 排除H原子

# Original Line 52: force field目录
# 修改成你的实际路径
original_dir = '/home/andy/test_optimization/OpenMM-ConstantV(original)'
ffdir = f'{original_dir}/ffdir/'
pdb_file = f'{original_dir}/nvt_0V_15ns.pdb'

# Original Line 163: SCF迭代次数
Niterations = 4

# Original Line 113: Electrolyte cutoff
Natom_cutoff = 100

# ═══════════════════════════════════════════════════════════
# 设置输出目录
# ═══════════════════════════════════════════════════════════
if os.path.exists(outPath):
    shutil.rmtree(outPath)
os.mkdir(outPath)

print(f"\n输出目录: {outPath}")

if write_charges:
    chargeFile = open(f'{outPath}/charges.dat', 'w')

# ═══════════════════════════════════════════════════════════
# 步骤1: 加载PDB和Force Fields（对应Original Line 88-89）
# ═══════════════════════════════════════════════════════════
print("\n" + "="*70)
print("步骤1: 加载系统")
print("="*70)

try:
    pdb = app.PDBFile(pdb_file)
    print(f"✓ 加载PDB: {pdb_file}")
    print(f"  原子数: {pdb.topology.getNumAtoms()}")
    print(f"  残基数: {pdb.topology.getNumResidues()}")
except FileNotFoundError:
    print(f"✗ 错误: 找不到PDB文件: {pdb_file}")
    print(f"  请修改脚本中的pdb_file路径")
    sys.exit(1)

# 对应Original的residue_xml_list和ff_xml_list
forcefield_files = [
    f'{ffdir}/sapt_residues.xml',
    f'{ffdir}/graph_residue_c.xml',
    f'{ffdir}/graph_residue_n.xml',
    f'{ffdir}/sapt_noDB_2sheets.xml',
    f'{ffdir}/graph_c_freeze.xml',
    f'{ffdir}/graph_n_freeze.xml'
]

try:
    forcefield = app.ForceField(*forcefield_files)
    print(f"✓ 加载force fields: {len(forcefield_files)}个文件")
except Exception as e:
    print(f"✗ 错误: 无法加载force fields: {e}")
    sys.exit(1)

# 对应Original Line 100: set_platform
# 后面会用到，这里先定义platform name
platform_name = 'CUDA'  # 可以改成'CPU'或'Reference'

# ═══════════════════════════════════════════════════════════
# 步骤2: 创建System（对应Original的MM.__init__）
# ═══════════════════════════════════════════════════════════
print("\n" + "="*70)
print("步骤2: 创建OpenMM System")
print("="*70)

# 对应Original Line 49: cutoff
cutoff = 1.4 * unit.nanometers

system = forcefield.createSystem(
    pdb.topology,
    nonbondedCutoff=cutoff,
    constraints=app.HBonds,
    rigidWater=True
)

print(f"✓ 创建system: {system.getNumParticles()}个粒子")

# 获取NonbondedForce和CustomNonbondedForce
nonbonded_force = None
custom_nonbonded_force = None

for force in system.getForces():
    if isinstance(force, mm.NonbondedForce):
        nonbonded_force = force
        print(f"✓ 找到NonbondedForce")
    elif isinstance(force, mm.CustomNonbondedForce):
        custom_nonbonded_force = force
        print(f"✓ 找到CustomNonbondedForce (SAPT-FF)")

if nonbonded_force is None:
    print("✗ 错误: 找不到NonbondedForce")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════
# 步骤3: 创建ConstantVIntegrator（对应Original的Integrator创建）
# ═══════════════════════════════════════════════════════════
print("\n" + "="*70)
print("步骤3: 创建ConstantVIntegrator")
print("="*70)

# Original Line 47: timestep = 0.001 ps
timestep = 0.001 * unit.picoseconds
integrator = ConstantVIntegrator(timestep.value_in_unit(unit.picoseconds))

# 设置电压
integrator.setVoltage(Voltage)
print(f"✓ 电压: {Voltage} V")

# 设置SCF参数
# Plugin的setSCFFrequency对应Original的freq_charge_update_fs
# 因为plugin内部自动处理SCF循环
integrator.setNumSCFIterations(Niterations)
integrator.setSCFFrequency(int(freq_charge_update_fs / timestep.value_in_unit(unit.femtoseconds)))
print(f"✓ SCF迭代次数: {Niterations}")
print(f"✓ SCF频率: 每{freq_charge_update_fs} fs更新一次电荷")

# ═══════════════════════════════════════════════════════════
# 步骤4: 识别并添加电极atoms（对应Original Line 109）
# ═══════════════════════════════════════════════════════════
print("\n" + "="*70)
print("步骤4: 识别电极atoms")
print("="*70)

# 识别cathode atoms（来自多个chains）
cathode_atoms = []
for chain in pdb.topology.chains():
    if chain.index in cathode_index:
        for atom in chain.atoms():
            # 排除指定元素
            if atom.element.symbol not in exclude_element:
                cathode_atoms.append(atom.index)

print(f"✓ Cathode atoms (chains {cathode_index}): {len(cathode_atoms)}")

# 识别anode atoms
anode_atoms = []
for chain in pdb.topology.chains():
    if chain.index in anode_index:
        for atom in chain.atoms():
            if atom.element.symbol not in exclude_element:
                anode_atoms.append(atom.index)

print(f"✓ Anode atoms (chains {anode_index}): {len(anode_atoms)}")

if len(cathode_atoms) == 0 or len(anode_atoms) == 0:
    print("✗ 错误: 找不到电极atoms，请检查chain indices")
    sys.exit(1)

# 计算每个atom的面积
cathode_area_per_atom, total_area = compute_electrode_area_per_atom(
    pdb.topology, cathode_atoms
)
anode_area_per_atom, _ = compute_electrode_area_per_atom(
    pdb.topology, anode_atoms
)

print(f"✓ Cathode面积/atom: {cathode_area_per_atom:.6f} nm²")
print(f"✓ Anode面积/atom: {anode_area_per_atom:.6f} nm²")
print(f"✓ 总面积: {total_area:.4f} nm²")

# 添加到integrator
for atom_idx in cathode_atoms:
    integrator.addCathodeAtom(atom_idx, cathode_area_per_atom)

for atom_idx in anode_atoms:
    integrator.addAnodeAtom(atom_idx, anode_area_per_atom)

print(f"✓ 已添加 {len(cathode_atoms)} 个cathode atoms")
print(f"✓ 已添加 {len(anode_atoms)} 个anode atoms")

# ═══════════════════════════════════════════════════════════
# 步骤5: 识别并添加Electrolyte（对应Original Line 113）
# ═══════════════════════════════════════════════════════════
print("\n" + "="*70)
print("步骤5: 识别electrolyte atoms")
print("="*70)

electrolyte_atoms = add_electrolyte_atoms_auto(
    pdb.topology,
    integrator,
    nonbonded_force,
    natom_cutoff=Natom_cutoff,
    exclude_chains=list(cathode_index) + list(anode_index)
)

if len(electrolyte_atoms) == 0:
    print("⚠ 警告: 没有找到electrolyte atoms")
    print("  (如果是真空模拟，这是正常的)")

# ═══════════════════════════════════════════════════════════
# 步骤6: 配置Geometry参数（对应Original Line 216）
# ═══════════════════════════════════════════════════════════
print("\n" + "="*70)
print("步骤6: 配置电极geometry")
print("="*70)

# 创建临时context来获取positions
temp_integrator = mm.VerletIntegrator(timestep)
temp_context = mm.Context(system, temp_integrator)
temp_context.setPositions(pdb.positions)

geometry_params = configure_geometry_from_context(
    temp_context,
    integrator,
    cathode_atoms[0],
    anode_atoms[0]
)

del temp_context
del temp_integrator

# ═══════════════════════════════════════════════════════════
# ⚠️ 步骤7: 添加Electrode Exclusions（对应Original Line 118）
# ═══════════════════════════════════════════════════════════
print("\n" + "="*70)
print("⚠️  步骤7: 添加electrode exclusions (CRITICAL!)")
print("="*70)

initial_exceptions = nonbonded_force.getNumExceptions()
print(f"  初始exceptions: {initial_exceptions}")

add_electrode_exclusions(integrator, nonbonded_force, custom_nonbonded_force)

final_exceptions = nonbonded_force.getNumExceptions()
print(f"  最终exceptions: {final_exceptions}")
print(f"  新增: {final_exceptions - initial_exceptions}")

# ═══════════════════════════════════════════════════════════
# 步骤8: 创建Context（对应Original Line 100 + Line 176）
# ═══════════════════════════════════════════════════════════
print("\n" + "="*70)
print("步骤8: 创建Context")
print("="*70)

try:
    platform = mm.Platform.getPlatformByName(platform_name)
    if platform_name == 'CUDA':
        properties = {'Precision': 'mixed'}
        print(f"✓ 使用{platform_name}平台 (mixed precision)")
    else:
        properties = {}
        print(f"✓ 使用{platform_name}平台")
except:
    print(f"⚠ {platform_name}平台不可用，切换到Reference")
    platform = mm.Platform.getPlatformByName('Reference')
    properties = {}

# 创建context
context = mm.Context(system, integrator, platform, properties)
context.setPositions(pdb.positions)
context.setVelocitiesToTemperature(300*unit.kelvin)

print("✓ Context创建成功")

# ⚠️ 关键: Reinitialize（对应Original Line 621）
print("\n⚠️  CRITICAL: Reinitializing context to apply exclusions...")
context.reinitialize(preserveState=True)
print("✓ Context reinitialized")

# ═══════════════════════════════════════════════════════════
# 步骤9: 验证Setup
# ═══════════════════════════════════════════════════════════
print("\n" + "="*70)
print("步骤9: 验证setup")
print("="*70)

valid, messages = validate_setup(context, integrator)
if not valid:
    print("\n✗ Setup验证失败:")
    for msg in messages:
        print(f"  {msg}")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════
# 步骤10: 检查初始能量（对应Original Line 120-127）
# ═══════════════════════════════════════════════════════════
print("\n" + "="*70)
print("步骤10: 检查初始状态")
print("="*70)

state = context.getState(getEnergy=True, getForces=True, getPositions=True)
print(f"动能: {state.getKineticEnergy()}")
print(f"势能: {state.getPotentialEnergy()}")

# 分别输出各个force的能量（对应Original Line 125-127）
print("\n各个force的贡献:")
for j in range(system.getNumForces()):
    f = system.getForce(j)
    force_energy = context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()
    print(f"  {type(f).__name__}: {force_energy}")

# 写入初始PDB（对应Original Line 132）
positions = state.getPositions()
app.PDBFile.writeFile(pdb.topology, positions, open(f'{outPath}/start.pdb', 'w'))
print(f"\n✓ 写入初始结构: {outPath}/start.pdb")

# ═══════════════════════════════════════════════════════════
# 步骤11: 设置Reporters（对应Original Line 142）
# ═══════════════════════════════════════════════════════════
print("\n" + "="*70)
print("步骤11: 设置输出")
print("="*70)

simulation = app.Simulation(pdb.topology, system, integrator, platform, properties)
simulation.context = context

trajectory_file_name = f'{outPath}/FV_NVT.dcd'
traj_freq_steps = int(freq_traj_output_ps * 1000 / timestep.value_in_unit(unit.femtoseconds))

simulation.reporters.append(app.DCDReporter(trajectory_file_name, traj_freq_steps))
print(f"✓ 轨迹输出: {trajectory_file_name} (每{traj_freq_steps}步)")

log_file_name = f'{outPath}/output.log'
simulation.reporters.append(app.StateDataReporter(
    log_file_name,
    max(100, traj_freq_steps // 10),
    step=True,
    time=True,
    potentialEnergy=True,
    kineticEnergy=True,
    totalEnergy=True,
    temperature=True,
    speed=True
))
print(f"✓ Log输出: {log_file_name}")

# Console reporter
simulation.reporters.append(app.StateDataReporter(
    sys.stdout,
    traj_freq_steps,
    step=True,
    time=True,
    potentialEnergy=True,
    temperature=True,
    speed=True,
    remainingTime=True,
    totalSteps=int(simulation_time_ns * 1e6 / timestep.value_in_unit(unit.femtoseconds))
))

# ═══════════════════════════════════════════════════════════
# 步骤12: 运行模拟（对应Original Line 144-171）
# ═══════════════════════════════════════════════════════════
print("\n" + "="*70)
print("步骤12: 运行Constant Voltage MD模拟")
print("="*70)
print(f"模拟参数:")
print(f"  总时间: {simulation_time_ns} ns")
print(f"  电压: {Voltage} V")
print(f"  SCF迭代: {Niterations}次")
print(f"  SCF频率: 每{freq_charge_update_fs} fs")
print(f"  轨迹输出: 每{freq_traj_output_ps} ps")
print("="*70)

t1 = datetime.now()

# Original的循环结构 (Line 144-171):
# for i in range(int(simulation_time_ns * 1000 / freq_traj_output_ps)):
#     for j in range(int(freq_traj_output_ps * 1000 / freq_charge_update_fs)):
#         MMsys.Poisson_solver_fixed_voltage(Niterations=4)
#         MMsys.simmd.step(freq_charge_update_fs)

# Plugin版本: integrator内部自动处理SCF循环，只需要一个step调用
num_steps = int(simulation_time_ns * 1e6 / timestep.value_in_unit(unit.femtoseconds))

print(f"开始模拟 ({num_steps}步)...\n")

try:
    simulation.step(num_steps)
except Exception as e:
    print(f"\n✗ 模拟出错: {e}")
    print("\n可能的原因:")
    print("1. 如果能量爆炸，检查exclusions是否正确添加")
    print("2. 如果'divide by zero'，检查geometry参数")
    print("3. 检查force field文件是否正确")
    sys.exit(1)

t2 = datetime.now()
print(f"\n✓ 模拟完成!")
print(f"  耗时: {t2 - t1}")

# ═══════════════════════════════════════════════════════════
# 步骤13: 输出最终状态
# ═══════════════════════════════════════════════════════════
print("\n" + "="*70)
print("最终状态")
print("="*70)

final_state = context.getState(getEnergy=True, getPositions=True, getVelocities=True)
print(f"势能: {final_state.getPotentialEnergy()}")
print(f"动能: {final_state.getKineticEnergy()}")
print(f"总能: {final_state.getPotentialEnergy() + final_state.getKineticEnergy()}")

# 写入最终PDB
final_positions = final_state.getPositions()
app.PDBFile.writeFile(pdb.topology, final_positions, open(f'{outPath}/final.pdb', 'w'))
print(f"\n✓ 写入最终结构: {outPath}/final.pdb")

print("\n" + "="*70)
print("✅ 全部完成!")
print("="*70)
print(f"输出文件:")
print(f"  轨迹: {trajectory_file_name}")
print(f"  日志: {log_file_name}")
print(f"  初始PDB: {outPath}/start.pdb")
print(f"  最终PDB: {outPath}/final.pdb")
print("="*70)
