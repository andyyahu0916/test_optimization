#!/usr/bin/env python3
"""
Run ConstantV Plugin Simulation from Configuration File

使用方法:
    python3 run_from_config.py [config_file.ini]

默认使用 simulation_config.ini
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

# Import config parser
from config_parser import load_config


def main():
    """主函数"""

    # ═══════════════════════════════════════════════════════════
    # 步骤1: 加载配置文件
    # ═══════════════════════════════════════════════════════════
    print("="*70)
    print("ConstantV Plugin - 从配置文件运行模拟")
    print("="*70)

    # 获取配置文件路径
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = 'simulation_config.ini'

    print(f"\n加载配置文件: {config_file}")
    config = load_config(config_file)
    print("✓ 配置加载成功\n")

    # 打印配置摘要
    config.print_summary()

    # 设置递归限制（用于大残基）
    sys.setrecursionlimit(config.recursion_limit)

    # ═══════════════════════════════════════════════════════════
    # 步骤2: 设置输出目录
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("设置输出目录")
    print("="*70)

    if os.path.exists(config.output_dir):
        if config.overwrite_output:
            shutil.rmtree(config.output_dir)
            print(f"✓ 删除旧输出目录: {config.output_dir}")
        else:
            print(f"✗ 错误: 输出目录已存在: {config.output_dir}")
            print(f"  请修改配置文件中的output_dir或设置overwrite_output=True")
            sys.exit(1)

    os.mkdir(config.output_dir)
    print(f"✓ 创建输出目录: {config.output_dir}")

    if config.write_charges:
        chargeFile = open(f'{config.output_dir}/charges.dat', 'w')

    # ═══════════════════════════════════════════════════════════
    # 步骤3: 加载PDB和Force Fields
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("加载系统")
    print("="*70)

    pdb = app.PDBFile(config.pdb_file)
    print(f"✓ 加载PDB: {config.pdb_file}")
    print(f"  原子数: {pdb.topology.getNumAtoms()}")
    print(f"  残基数: {pdb.topology.getNumResidues()}")

    forcefield = app.ForceField(*config.forcefield_files)
    print(f"✓ 加载force fields: {len(config.forcefield_files)}个文件")

    # ═══════════════════════════════════════════════════════════
    # 步骤4: 创建System
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("创建OpenMM System")
    print("="*70)

    system = forcefield.createSystem(
        pdb.topology,
        nonbondedCutoff=config.nonbonded_cutoff * unit.nanometers,
        constraints=config.get_constraints_enum(),
        rigidWater=config.rigid_water
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
    # 步骤5: 创建ConstantVIntegrator
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("创建ConstantVIntegrator")
    print("="*70)

    integrator = ConstantVIntegrator(config.timestep_ps)
    integrator.setVoltage(config.voltage)
    print(f"✓ 电压: {config.voltage} V")

    integrator.setNumSCFIterations(config.num_scf_iterations)
    integrator.setSCFFrequency(config.calculate_scf_frequency_steps())
    print(f"✓ SCF迭代次数: {config.num_scf_iterations}")
    print(f"✓ SCF频率: 每{config.scf_frequency_fs} fs ({config.calculate_scf_frequency_steps()}步)")

    # ═══════════════════════════════════════════════════════════
    # 步骤6: 识别并添加电极atoms
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("识别电极atoms")
    print("="*70)

    # 识别cathode atoms
    cathode_atoms = []
    for chain in pdb.topology.chains():
        if chain.index in config.cathode_chains:
            for atom in chain.atoms():
                if atom.element.symbol not in config.exclude_elements:
                    cathode_atoms.append(atom.index)

    print(f"✓ Cathode atoms (chains {config.cathode_chains}): {len(cathode_atoms)}")

    # 识别anode atoms
    anode_atoms = []
    for chain in pdb.topology.chains():
        if chain.index in config.anode_chains:
            for atom in chain.atoms():
                if atom.element.symbol not in config.exclude_elements:
                    anode_atoms.append(atom.index)

    print(f"✓ Anode atoms (chains {config.anode_chains}): {len(anode_atoms)}")

    if len(cathode_atoms) == 0 or len(anode_atoms) == 0:
        print("✗ 错误: 找不到电极atoms，请检查配置文件中的chain indices")
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
    # 步骤7: 识别并添加Electrolyte
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("识别electrolyte atoms")
    print("="*70)

    electrolyte_atoms = add_electrolyte_atoms_auto(
        pdb.topology,
        integrator,
        nonbonded_force,
        natom_cutoff=config.natom_cutoff,
        exclude_chains=list(config.cathode_chains) + list(config.anode_chains)
    )

    if len(electrolyte_atoms) == 0:
        print("⚠ 警告: 没有找到electrolyte atoms")
        print("  (如果是真空模拟，这是正常的)")

    # ═══════════════════════════════════════════════════════════
    # 步骤8: 配置Geometry参数
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("配置电极geometry")
    print("="*70)

    # 创建临时context来获取positions
    temp_integrator = mm.VerletIntegrator(config.timestep_ps * unit.picoseconds)
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
    # ⚠️ 步骤9: 添加Electrode Exclusions
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("⚠️  添加electrode exclusions (CRITICAL!)")
    print("="*70)

    initial_exceptions = nonbonded_force.getNumExceptions()
    print(f"  初始exceptions: {initial_exceptions}")

    # 根据配置决定是否添加CustomNonbondedForce exclusions
    if config.sapt_ff_exclusions and custom_nonbonded_force is not None:
        add_electrode_exclusions(integrator, nonbonded_force, custom_nonbonded_force)
    else:
        add_electrode_exclusions(integrator, nonbonded_force, None)

    final_exceptions = nonbonded_force.getNumExceptions()
    print(f"  最终exceptions: {final_exceptions}")
    print(f"  新增: {final_exceptions - initial_exceptions}")

    # ═══════════════════════════════════════════════════════════
    # 步骤10: 创建Context
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("创建Context")
    print("="*70)

    try:
        platform = mm.Platform.getPlatformByName(config.platform_name)
        if config.platform_name == 'CUDA':
            properties = {'Precision': config.cuda_precision}
            print(f"✓ 使用{config.platform_name}平台 ({config.cuda_precision} precision)")
        else:
            properties = {}
            print(f"✓ 使用{config.platform_name}平台")
    except:
        print(f"⚠ {config.platform_name}平台不可用，切换到Reference")
        platform = mm.Platform.getPlatformByName('Reference')
        properties = {}

    # 创建context
    context = mm.Context(system, integrator, platform, properties)
    context.setPositions(pdb.positions)
    context.setVelocitiesToTemperature(config.temperature * unit.kelvin)

    print("✓ Context创建成功")

    # ⚠️ 关键: Reinitialize
    print("\n⚠️  CRITICAL: Reinitializing context to apply exclusions...")
    context.reinitialize(preserveState=True)
    print("✓ Context reinitialized")

    # ═══════════════════════════════════════════════════════════
    # 步骤11: 验证Setup
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("验证setup")
    print("="*70)

    valid, messages = validate_setup(context, integrator)
    if not valid:
        print("\n✗ Setup验证失败:")
        for msg in messages:
            print(f"  {msg}")
        sys.exit(1)

    # ═══════════════════════════════════════════════════════════
    # 步骤12: 检查初始能量
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("检查初始状态")
    print("="*70)

    state = context.getState(getEnergy=True, getForces=True, getPositions=True)
    print(f"动能: {state.getKineticEnergy()}")
    print(f"势能: {state.getPotentialEnergy()}")

    # 分别输出各个force的能量
    print("\n各个force的贡献:")
    for j in range(system.getNumForces()):
        f = system.getForce(j)
        force_energy = context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()
        print(f"  {type(f).__name__}: {force_energy}")

    # 写入初始PDB
    positions = state.getPositions()
    app.PDBFile.writeFile(pdb.topology, positions,
                          open(f'{config.output_dir}/start.pdb', 'w'))
    print(f"\n✓ 写入初始结构: {config.output_dir}/start.pdb")

    # ═══════════════════════════════════════════════════════════
    # 步骤13: 设置Reporters
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("设置输出")
    print("="*70)

    simulation = app.Simulation(pdb.topology, system, integrator, platform, properties)
    simulation.context = context

    trajectory_file = f'{config.output_dir}/trajectory.dcd'
    traj_freq_steps = config.calculate_trajectory_output_steps()

    simulation.reporters.append(app.DCDReporter(trajectory_file, traj_freq_steps))
    print(f"✓ 轨迹输出: {trajectory_file} (每{traj_freq_steps}步)")

    log_file = f'{config.output_dir}/output.log'
    simulation.reporters.append(app.StateDataReporter(
        log_file,
        config.log_output_steps,
        step=True,
        time=True,
        potentialEnergy=True,
        kineticEnergy=True,
        totalEnergy=True,
        temperature=True,
        speed=True
    ))
    print(f"✓ Log输出: {log_file} (每{config.log_output_steps}步)")

    # Console reporter
    console_freq_steps = config.calculate_console_output_steps()
    simulation.reporters.append(app.StateDataReporter(
        sys.stdout,
        console_freq_steps,
        step=True,
        time=True,
        potentialEnergy=True,
        temperature=True,
        speed=True,
        remainingTime=True,
        totalSteps=config.calculate_total_steps()
    ))
    print(f"✓ Console输出: 每{console_freq_steps}步")

    # ═══════════════════════════════════════════════════════════
    # 步骤14: 运行模拟
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("运行Constant Voltage MD模拟")
    print("="*70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总步数: {config.calculate_total_steps()}")
    print("="*70 + "\n")

    t1 = datetime.now()

    try:
        simulation.step(config.calculate_total_steps())
    except Exception as e:
        print(f"\n✗ 模拟出错: {e}")
        print("\n可能的原因:")
        print("1. 如果能量爆炸，检查exclusions是否正确添加")
        print("2. 如果'divide by zero'，检查geometry参数")
        print("3. 检查force field文件是否正确")
        sys.exit(1)

    t2 = datetime.now()

    # ═══════════════════════════════════════════════════════════
    # 步骤15: 输出最终状态
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
    app.PDBFile.writeFile(pdb.topology, final_positions,
                          open(f'{config.output_dir}/final.pdb', 'w'))
    print(f"\n✓ 写入最终结构: {config.output_dir}/final.pdb")

    print("\n" + "="*70)
    print("✅ 模拟完成!")
    print("="*70)
    print(f"结束时间: {t2.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {t2 - t1}")
    print(f"\n输出文件:")
    print(f"  轨迹: {trajectory_file}")
    print(f"  日志: {log_file}")
    print(f"  初始PDB: {config.output_dir}/start.pdb")
    print(f"  最终PDB: {config.output_dir}/final.pdb")
    print("="*70)

    if config.write_charges:
        chargeFile.close()


if __name__ == '__main__':
    main()
