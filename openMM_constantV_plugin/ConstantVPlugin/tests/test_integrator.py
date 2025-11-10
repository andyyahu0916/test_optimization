#!/usr/bin/env python
"""
测试ConstantVIntegrator实现 - 验证C++翻译的教授算法

使用ConstantVIntegrator自动执行SCF迭代
系统设置与test_minimal.py相同，用于对比验证
"""

import sys
import numpy as np
from openmm.app import *
from openmm import *
from openmm.unit import *

# 加载插件
try:
    from constantvplugin import ConstantVIntegrator
    print("✓ 成功加载ConstantVIntegrator")
except ImportError as e:
    print(f"✗ 加载ConstantVIntegrator失败: {e}")
    print("请确保插件已正确编译和安装")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════
# 常数（与教授代码一致）
# ═══════════════════════════════════════════════════════════
CONVERSION_NMBOHR = 18.8973
CONVERSION_KJMOLNM_AU = CONVERSION_NMBOHR / 2625.5  # 0.00719924...
CONVERSION_EV_KJMOL = 96.487
SMALL_THRESHOLD = 1e-6

# ═══════════════════════════════════════════════════════════
# 创建最小系统（与test_minimal.py相同）
# ═══════════════════════════════════════════════════════════

def create_minimal_system():
    """创建超级简单的测试系统"""

    # 创建拓扑
    topology = Topology()
    chain_cathode = topology.addChain()
    chain_anode = topology.addChain()
    chain_electrolyte = topology.addChain()

    # 阴极：2个碳原子
    res_cathode = topology.addResidue("CATH", chain_cathode)
    atom_c1 = topology.addAtom("C1", element.carbon, res_cathode)
    atom_c2 = topology.addAtom("C2", element.carbon, res_cathode)

    # 阳极：2个碳原子
    res_anode = topology.addResidue("ANOD", chain_anode)
    atom_a1 = topology.addAtom("A1", element.carbon, res_anode)
    atom_a2 = topology.addAtom("A2", element.carbon, res_anode)

    # 电解质：1个钠离子
    res_electrolyte = topology.addResidue("NA", chain_electrolyte)
    atom_na = topology.addAtom("Na", element.sodium, res_electrolyte)

    # 设置周期盒子（z方向5nm，xy方向2nm）
    topology.setPeriodicBoxVectors([
        Vec3(2.0, 0.0, 0.0),  # a
        Vec3(0.0, 2.0, 0.0),  # b
        Vec3(0.0, 0.0, 5.0)   # c
    ])

    # 位置：沿z轴排列
    # 阴极在z=1.0nm，阳极在z=4.0nm，电解质在中间z=2.5nm
    positions = [
        Vec3(1.0, 1.0, 1.0),  # C1 (阴极)
        Vec3(1.1, 1.0, 1.0),  # C2 (阴极)
        Vec3(1.0, 1.0, 4.0),  # A1 (阳极)
        Vec3(1.1, 1.0, 4.0),  # A2 (阳极)
        Vec3(1.0, 1.0, 2.5),  # Na (电解质)
    ]

    # 创建System
    system = System()
    for i in range(5):
        system.addParticle(12.0)  # 质量（对电荷计算无影响）

    # 设置周期边界
    system.setDefaultPeriodicBoxVectors(
        Vec3(2.0, 0.0, 0.0),
        Vec3(0.0, 2.0, 0.0),
        Vec3(0.0, 0.0, 5.0)
    )

    # NonbondedForce（简单LJ + Coulomb）
    nonbonded = NonbondedForce()
    nonbonded.setNonbondedMethod(NonbondedForce.PME)
    nonbonded.setCutoffDistance(1.0*nanometer)

    # 阴极原子：初始电荷0.0，LJ参数（sigma=0.34nm, epsilon=0.1kJ/mol）
    nonbonded.addParticle(0.0, 0.34, 0.1)  # C1
    nonbonded.addParticle(0.0, 0.34, 0.1)  # C2

    # 阳极原子：初始电荷0.0，LJ参数
    nonbonded.addParticle(0.0, 0.34, 0.1)  # A1
    nonbonded.addParticle(0.0, 0.34, 0.1)  # A2

    # 电解质：钠离子，固定电荷+1e
    nonbonded.addParticle(1.0, 0.24, 0.05)  # Na+

    system.addForce(nonbonded)

    return topology, system, positions

# ═══════════════════════════════════════════════════════════
# 主测试函数
# ═══════════════════════════════════════════════════════════

def main():
    print("="*60)
    print("ConstantVIntegrator测试 - 验证C++实现")
    print("="*60)

    # 创建系统
    topology, system, positions = create_minimal_system()

    # 系统参数
    voltage_volts = 1.0  # 1V
    voltage_kjmol = voltage_volts * CONVERSION_EV_KJMOL

    # 几何参数（从positions计算）
    z_cathode = 1.0  # nm
    z_anode = 4.0    # nm
    Lcell = abs(z_cathode - z_anode)  # 3.0 nm
    box_z = 5.0  # nm
    Lgap = box_z - Lcell  # 2.0 nm

    # 电极面积（2nm × 2nm = 4 nm^2）
    sheet_area = 2.0 * 2.0  # 4.0 nm^2

    # 每个原子的面积（2个阴极原子，每个2nm^2）
    cathode_areas = [sheet_area / 2.0, sheet_area / 2.0]
    anode_areas = [sheet_area / 2.0, sheet_area / 2.0]

    # 电极索引
    cathode_indices = [0, 1]
    anode_indices = [2, 3]
    electrolyte_indices = [4]
    electrolyte_charges = [1.0]  # Na+

    print(f"\n系统参数:")
    print(f"  电压: {voltage_volts} V = {voltage_kjmol:.3f} kJ/mol")
    print(f"  Lcell: {Lcell} nm")
    print(f"  Lgap: {Lgap} nm")
    print(f"  电极面积: {sheet_area} nm^2")
    print(f"  阴极原子: {cathode_indices}")
    print(f"  阳极原子: {anode_indices}")
    print(f"  电解质原子: {electrolyte_indices}")

    # ═══════════════════════════════════════════════════════════
    # 创建ConstantVIntegrator
    # ═══════════════════════════════════════════════════════════

    print("\n" + "="*60)
    print("配置ConstantVIntegrator")
    print("="*60)

    integrator = ConstantVIntegrator(0.001*picoseconds)

    # 设置物理参数
    integrator.setVoltage(voltage_volts)
    integrator.setLgap(Lgap)
    integrator.setLcell(Lcell)
    integrator.setTotalArea(sheet_area)
    integrator.setZCathode(z_cathode)
    integrator.setZAnode(z_anode)

    # 设置SCF参数
    integrator.setNumSCFIterations(4)  # 与test_minimal.py相同
    integrator.setSCFFrequency(1)      # 每步都执行SCF

    # 添加阴极原子
    for idx, area in zip(cathode_indices, cathode_areas):
        integrator.addCathodeAtom(idx, area)
        print(f"  添加阴极原子 {idx}: area = {area:.2f} nm^2")

    # 添加阳极原子
    for idx, area in zip(anode_indices, anode_areas):
        integrator.addAnodeAtom(idx, area)
        print(f"  添加阳极原子 {idx}: area = {area:.2f} nm^2")

    # 添加电解质原子
    for idx, charge in zip(electrolyte_indices, electrolyte_charges):
        integrator.addElectrolyteAtom(idx, charge)
        print(f"  添加电解质原子 {idx}: charge = {charge:.2f}")

    print(f"\n  SCF迭代次数: {integrator.getNumSCFIterations()}")
    print(f"  SCF更新频率: {integrator.getSCFFrequency()}")

    # ═══════════════════════════════════════════════════════════
    # 创建Context并运行
    # ═══════════════════════════════════════════════════════════

    print("\n" + "="*60)
    print("创建Context并运行SCF")
    print("="*60)

    platform = Platform.getPlatformByName('Reference')
    context = Context(system, integrator, platform)
    context.setPositions(positions)

    print("  初始化完成")

    # 运行1步（自动执行SCF迭代）
    print("  执行 integrator.step(1)...")
    integrator.step(1)

    print("  ✓ SCF执行完成")

    # ═══════════════════════════════════════════════════════════
    # 检查结果
    # ═══════════════════════════════════════════════════════════

    print("\n" + "="*60)
    print("检查结果")
    print("="*60)

    # 获取NonbondedForce
    nonbonded = None
    for force in system.getForces():
        if isinstance(force, NonbondedForce):
            nonbonded = force
            break

    # 读取最终电荷
    final_charges = np.zeros(5)
    for i in range(5):
        charge, sigma, epsilon = nonbonded.getParticleParameters(i)
        final_charges[i] = charge

    print("\n最终电荷:")
    for i, idx in enumerate(cathode_indices):
        print(f"  阴极原子{idx}: q = {final_charges[idx]:.10e}")
    for i, idx in enumerate(anode_indices):
        print(f"  阳极原子{idx}: q = {final_charges[idx]:.10e}")

    Q_cathode = sum([final_charges[idx] for idx in cathode_indices])
    Q_anode = sum([final_charges[idx] for idx in anode_indices])

    print(f"\n总电荷:")
    print(f"  Q_cathode = {Q_cathode:.10e}")
    print(f"  Q_anode   = {Q_anode:.10e}")
    print(f"  Q_total   = {Q_cathode + Q_anode:.10e}")

    # ═══════════════════════════════════════════════════════════
    # 与test_minimal.py对比
    # ═══════════════════════════════════════════════════════════

    print("\n" + "="*60)
    print("与test_minimal.py对比")
    print("="*60)

    # test_minimal.py的参考结果（从test_minimal_reference.npz读取）
    try:
        import os
        ref_file = os.path.join(os.path.dirname(__file__), "test_minimal_reference.npz")
        if os.path.exists(ref_file):
            ref_data = np.load(ref_file)
            ref_charges = ref_data['final_charges']
            ref_Q_cathode = ref_data['Q_cathode']
            ref_Q_anode = ref_data['Q_anode']

            print("\ntest_minimal.py参考结果:")
            print(f"  Q_cathode = {ref_Q_cathode:.10e}")
            print(f"  Q_anode   = {ref_Q_anode:.10e}")

            print("\nConstantVIntegrator结果:")
            print(f"  Q_cathode = {Q_cathode:.10e}")
            print(f"  Q_anode   = {Q_anode:.10e}")

            print("\n差异:")
            diff_cathode = abs(Q_cathode - ref_Q_cathode)
            diff_anode = abs(Q_anode - ref_Q_anode)
            print(f"  ΔQ_cathode = {diff_cathode:.10e}")
            print(f"  ΔQ_anode   = {diff_anode:.10e}")

            # 验证精度（应该< 1e-6）
            if diff_cathode < 1e-6 and diff_anode < 1e-6:
                print("\n✓✓✓ 通过验证！C++实现与Python参考完全一致 ✓✓✓")
            else:
                print("\n✗✗✗ 验证失败！存在数值差异 ✗✗✗")
                sys.exit(1)
        else:
            print(f"\n参考文件 {ref_file} 不存在")
            print("请先运行 test_minimal.py 生成参考结果")
    except Exception as e:
        print(f"\n读取参考数据出错: {e}")

    print("\n" + "="*60)
    print("测试完成")
    print("="*60)

if __name__ == "__main__":
    main()
