#!/usr/bin/env python
"""
Plugin测试 - 使用C++翻译的教授算法

对比验证：与test_minimal.py的手工实现对比
"""

import sys
import os
import numpy as np
from openmm.app import *
from openmm import *
from openmm.unit import *

# 尝试加载plugin（使用已安装的模块）
try:
    import constantvplugin
    print("✅ constantvplugin loaded successfully")
except ImportError as e:
    print(f"❌ Failed to load constantvplugin: {e}")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════
# 创建与test_minimal.py完全相同的系统
# ═══════════════════════════════════════════════════════════

def create_minimal_system():
    """创建与test_minimal.py完全相同的测试系统"""

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
        system.addParticle(12.0)

    # 设置周期边界
    system.setDefaultPeriodicBoxVectors(
        Vec3(2.0, 0.0, 0.0),
        Vec3(0.0, 2.0, 0.0),
        Vec3(0.0, 0.0, 5.0)
    )

    # NonbondedForce
    nonbonded = NonbondedForce()
    nonbonded.setNonbondedMethod(NonbondedForce.PME)
    nonbonded.setCutoffDistance(1.0*nanometer)

    # 阴极原子：初始电荷0.0
    nonbonded.addParticle(0.0, 0.34, 0.1)  # C1
    nonbonded.addParticle(0.0, 0.34, 0.1)  # C2

    # 阳极原子：初始电荷0.0
    nonbonded.addParticle(0.0, 0.34, 0.1)  # A1
    nonbonded.addParticle(0.0, 0.34, 0.1)  # A2

    # 电解质：钠离子，固定电荷+1e
    nonbonded.addParticle(1.0, 0.24, 0.05)  # Na+

    system.addForce(nonbonded)

    # ═══════════════════════════════════════════════════════════
    # 添加ConstantVForce（我们的Plugin）
    # ═══════════════════════════════════════════════════════════

    constantV = constantvplugin.ConstantVForce()

    # 系统参数
    voltage_volts = 1.0  # 1V
    z_cathode = 1.0  # nm
    z_anode = 4.0    # nm
    Lcell = abs(z_cathode - z_anode)  # 3.0 nm
    box_z = 5.0  # nm
    Lgap = box_z - Lcell  # 2.0 nm
    sheet_area = 2.0 * 2.0  # 4.0 nm^2

    # 设置系统参数
    constantV.setVoltage(voltage_volts)
    constantV.setLgap(Lgap)
    constantV.setLcell(Lcell)
    constantV.setTotalArea(sheet_area)
    constantV.setZCathode(z_cathode)
    constantV.setZAnode(z_anode)
    constantV.setNumIterations(4)

    # 添加阴极原子（每个原子面积 = 总面积/2）
    area_per_atom = sheet_area / 2.0
    constantV.addCathodeAtom(0, area_per_atom)  # C1
    constantV.addCathodeAtom(1, area_per_atom)  # C2

    # 添加阳极原子
    constantV.addAnodeAtom(2, area_per_atom)  # A1
    constantV.addAnodeAtom(3, area_per_atom)  # A2

    # 添加电解质原子
    constantV.addElectrolyteAtom(4, 1.0)  # Na+, charge=1.0

    system.addForce(constantV)

    return topology, system, positions

# ═══════════════════════════════════════════════════════════
# 主测试函数
# ═══════════════════════════════════════════════════════════

def main():
    print("="*60)
    print("Plugin测试 - C++翻译版本")
    print("="*60)

    # 创建系统
    print("\n[DEBUG] Creating system...")
    topology, system, positions = create_minimal_system()
    print("[DEBUG] System created successfully")

    # 创建Integrator和Context
    print("[DEBUG] Creating integrator and context...")
    integrator = VerletIntegrator(0.001*picoseconds)
    platform = Platform.getPlatformByName('Reference')
    print("[DEBUG] Creating Context...")
    context = Context(system, integrator, platform)
    print("[DEBUG] Setting positions...")
    context.setPositions(positions)
    print("[DEBUG] Context created successfully")

    print(f"\n系统信息:")
    print(f"  原子数: {system.getNumParticles()}")
    print(f"  力场数: {system.getNumForces()}")

    # 获取NonbondedForce和ConstantVForce
    nonbonded = None
    constantV = None
    for force in system.getForces():
        if isinstance(force, NonbondedForce):
            nonbonded = force
        elif isinstance(force, constantvplugin.ConstantVForce):
            constantV = force

    print(f"  NonbondedForce: {'✅' if nonbonded else '❌'}")
    print(f"  ConstantVForce: {'✅' if constantV else '❌'}")

    if constantV:
        print(f"\nConstantVForce参数:")
        print(f"  电压: {constantV.getVoltage()} V")
        print(f"  Lgap: {constantV.getLgap()} nm")
        print(f"  Lcell: {constantV.getLcell()} nm")
        print(f"  总面积: {constantV.getTotalArea()} nm^2")
        print(f"  阴极原子数: {constantV.getNumCathodeAtoms()}")
        print(f"  阳极原子数: {constantV.getNumAnodeAtoms()}")
        print(f"  电解质原子数: {constantV.getNumElectrolyteAtoms()}")
        print(f"  SCF迭代次数: {constantV.getNumIterations()}")

    # 运行一步（会触发ConstantVForce::execute()）
    print("\n" + "="*60)
    print("运行Plugin（触发SCF迭代）")
    print("="*60)

    print("[DEBUG] About to call integrator.step(1)...")
    integrator.step(1)
    print("[DEBUG] integrator.step(1) completed successfully")

    # 获取最终电荷
    print("\n获取最终电荷:")
    final_charges = []
    for i in range(5):
        charge, sigma, epsilon = nonbonded.getParticleParameters(i)
        final_charges.append(charge)
        print(f"  原子{i}: q = {charge:.10e}")

    # 验证
    Q_cathode = final_charges[0] + final_charges[1]
    Q_anode = final_charges[2] + final_charges[3]

    print(f"\n总电荷:")
    print(f"  阴极: {Q_cathode:.10e}")
    print(f"  阳极: {Q_anode:.10e}")
    print(f"  总和: {Q_cathode + Q_anode:.10e}")

    # 与参考结果对比
    if os.path.exists('test_minimal_reference.npz'):
        print("\n" + "="*60)
        print("与参考结果对比")
        print("="*60)

        ref = np.load('test_minimal_reference.npz')
        ref_charges = ref['final_charges']
        ref_Q_cathode = ref['Q_cathode']
        ref_Q_anode = ref['Q_anode']

        print(f"\n参考值（手工Python实现）:")
        for i in range(5):
            print(f"  原子{i}: q = {ref_charges[i]:.10e}")

        print(f"\n总电荷对比:")
        print(f"  阴极: Plugin={Q_cathode:.10e}, 参考={ref_Q_cathode:.10e}")
        print(f"  阳极: Plugin={Q_anode:.10e}, 参考={ref_Q_anode:.10e}")

        # 计算误差
        max_error = max(abs(final_charges[i] - ref_charges[i]) for i in range(5))
        rel_error_cathode = abs(Q_cathode - ref_Q_cathode) / abs(ref_Q_cathode) if ref_Q_cathode != 0 else 0
        rel_error_anode = abs(Q_anode - ref_Q_anode) / abs(ref_Q_anode) if ref_Q_anode != 0 else 0

        print(f"\n误差分析:")
        print(f"  最大绝对误差: {max_error:.10e}")
        print(f"  阴极相对误差: {rel_error_cathode:.10e}")
        print(f"  阳极相对误差: {rel_error_anode:.10e}")

        # 判断
        if max_error < 1e-6:
            print("\n✅ Plugin结果与参考结果一致！")
        else:
            print("\n❌ Plugin结果与参考结果不一致！")
    else:
        print("\n⚠️  参考结果文件不存在，请先运行 test_minimal.py")

    print("\n完成！")

if __name__ == "__main__":
    main()
