#!/usr/bin/env python
"""
最小测试案例 - 验证教授算法的C++翻译

系统设置：
- 2个阴极原子（chain 0）
- 2个阳极原子（chain 1）
- 1个电解质原子（Na+离子）
- 简单的Lennard-Jones + Coulomb力场
"""

import sys
import numpy as np
from openmm.app import *
from openmm import *
from openmm.unit import *

# ═══════════════════════════════════════════════════════════
# 常数（与教授代码一致）
# ═══════════════════════════════════════════════════════════
CONVERSION_NMBOHR = 18.8973
CONVERSION_KJMOLNM_AU = CONVERSION_NMBOHR / 2625.5  # 0.00719924...
CONVERSION_EV_KJMOL = 96.487
SMALL_THRESHOLD = 1e-6

# ═══════════════════════════════════════════════════════════
# 创建最小系统
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
# 手工实现教授的算法（用于验证）
# ═══════════════════════════════════════════════════════════

def compute_analytic_charge_manual(
    voltage_kjmol, Lgap, Lcell, sheet_area, z_opposite, sign,
    electrolyte_indices, electrolyte_charges, positions
):
    """
    手工计算解析总电荷
    对应：Fixed_Voltage_routines.py::compute_Electrode_charge_analytic
    """
    # Line 324-325: 几何贡献
    Q_analytic = sign / (4.0 * np.pi) * sheet_area * \
                 (voltage_kjmol / Lgap + voltage_kjmol / Lcell) * \
                 CONVERSION_KJMOLNM_AU

    # Line 327-333: 镜像电荷贡献
    for i, idx in enumerate(electrolyte_indices):
        q_i = electrolyte_charges[i]
        # 提取z坐标的数值（nm单位）
        z_atom = positions[idx][2]._value if hasattr(positions[idx][2], '_value') else positions[idx][2]
        z_distance = abs(z_atom - z_opposite)
        Q_analytic += (z_distance / Lcell) * (-q_i)

    print(f"  手工计算 Q_analytic = {Q_analytic:.10f}")
    return Q_analytic

def scf_iteration_manual(
    context, nonbonded,
    cathode_indices, anode_indices,
    cathode_areas, anode_areas,
    voltage_kjmol, Lgap, Lcell, sheet_area,
    z_cathode, z_anode,
    electrolyte_indices, electrolyte_charges,
    nIterations=4
):
    """
    手工实现SCF迭代
    对应：MM_classes.py::Poisson_solver_fixed_voltage
    """

    print("\n" + "="*60)
    print("开始SCF迭代（手工实现教授算法）")
    print("="*60)

    # 阶段0：计算解析总电荷
    state = context.getState(getPositions=True)
    positions = state.getPositions(asNumpy=True)

    Q_analytic_cathode = compute_analytic_charge_manual(
        voltage_kjmol, Lgap, Lcell, sheet_area, z_anode, +1.0,
        electrolyte_indices, electrolyte_charges, positions
    )

    Q_analytic_anode = compute_analytic_charge_manual(
        voltage_kjmol, Lgap, Lcell, sheet_area, z_cathode, -1.0,
        electrolyte_indices, electrolyte_charges, positions
    )

    print(f"\n解析总电荷:")
    print(f"  Q_analytic_cathode = {Q_analytic_cathode:.10f}")
    print(f"  Q_analytic_anode   = {Q_analytic_anode:.10f}")

    # 当前电荷
    current_charges = np.zeros(5)

    # 阶段1：SCF迭代
    for iter_num in range(nIterations):
        print(f"\n{'─'*60}")
        print(f"SCF迭代 {iter_num + 1}/{nIterations}")
        print(f"{'─'*60}")

        # 获取力
        state = context.getState(getForces=True, getPositions=True)
        forces = state.getForces(asNumpy=True)

        # 更新阴极电荷
        print("\n更新阴极电荷:")
        for i, idx in enumerate(cathode_indices):
            q_old = current_charges[idx]

            # Ez从力计算
            if abs(q_old) > (0.9 * SMALL_THRESHOLD):
                # 提取力的数值（kJ/(mol·nm)单位）
                force_z = forces[idx][2]._value if hasattr(forces[idx][2], '_value') else forces[idx][2]
                Ez = force_z / q_old
            else:
                Ez = 0.0

            # 边界条件
            q_new = 2.0 / (4.0 * np.pi) * cathode_areas[i] * \
                    (voltage_kjmol / Lgap + Ez) * CONVERSION_KJMOLNM_AU

            # 防归零
            if abs(q_new) < SMALL_THRESHOLD:
                q_new = SMALL_THRESHOLD

            current_charges[idx] = q_new
            nonbonded.setParticleParameters(idx, q_new, 0.34, 0.1)

            print(f"  原子{idx}: q_old={q_old:.6e}, Ez={Ez:.6e}, q_new={q_new:.6e}")

        # 更新阳极电荷
        print("\n更新阳极电荷:")
        for i, idx in enumerate(anode_indices):
            q_old = current_charges[idx]

            # Ez从力计算
            if abs(q_old) > (0.9 * SMALL_THRESHOLD):
                # 提取力的数值（kJ/(mol·nm)单位）
                force_z = forces[idx][2]._value if hasattr(forces[idx][2], '_value') else forces[idx][2]
                Ez = force_z / q_old
            else:
                Ez = 0.0

            # 边界条件（注意负号）
            q_new = -2.0 / (4.0 * np.pi) * anode_areas[i] * \
                    (voltage_kjmol / Lgap + Ez) * CONVERSION_KJMOLNM_AU

            # 防归零
            if abs(q_new) < SMALL_THRESHOLD:
                q_new = -1.0 * SMALL_THRESHOLD

            current_charges[idx] = q_new
            nonbonded.setParticleParameters(idx, q_new, 0.34, 0.1)

            print(f"  原子{idx}: q_old={q_old:.6e}, Ez={Ez:.6e}, q_new={q_new:.6e}")

        # Green's校正 - 阴极
        Q_numeric_cathode = sum([current_charges[idx] for idx in cathode_indices])
        scale_cathode = Q_analytic_cathode / Q_numeric_cathode if abs(Q_numeric_cathode) > SMALL_THRESHOLD else -1.0

        print(f"\nGreen's校正（阴极）:")
        print(f"  Q_numeric = {Q_numeric_cathode:.10f}")
        print(f"  Q_analytic = {Q_analytic_cathode:.10f}")
        print(f"  scale = {scale_cathode:.10f}")

        if scale_cathode > 0.0:
            for idx in cathode_indices:
                current_charges[idx] *= scale_cathode
                nonbonded.setParticleParameters(idx, current_charges[idx], 0.34, 0.1)

        # Green's校正 - 阳极
        Q_numeric_anode = sum([current_charges[idx] for idx in anode_indices])
        scale_anode = Q_analytic_anode / Q_numeric_anode if abs(Q_numeric_anode) > SMALL_THRESHOLD else -1.0

        print(f"\nGreen's校正（阳极）:")
        print(f"  Q_numeric = {Q_numeric_anode:.10f}")
        print(f"  Q_analytic = {Q_analytic_anode:.10f}")
        print(f"  scale = {scale_anode:.10f}")

        if scale_anode > 0.0:
            for idx in anode_indices:
                current_charges[idx] *= scale_anode
                nonbonded.setParticleParameters(idx, current_charges[idx], 0.34, 0.1)

        # 更新context
        nonbonded.updateParametersInContext(context)

        # 打印当前电荷
        print(f"\n迭代{iter_num+1}后的电荷:")
        for i, idx in enumerate(cathode_indices):
            print(f"  阴极原子{idx}: q = {current_charges[idx]:.10e}")
        for i, idx in enumerate(anode_indices):
            print(f"  阳极原子{idx}: q = {current_charges[idx]:.10e}")

    print("\n" + "="*60)
    print("SCF迭代完成")
    print("="*60)

    return current_charges

# ═══════════════════════════════════════════════════════════
# 主测试函数
# ═══════════════════════════════════════════════════════════

def main():
    print("="*60)
    print("最小测试案例 - 教授算法验证")
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

    # 创建Integrator和Context
    integrator = VerletIntegrator(0.001*picoseconds)
    platform = Platform.getPlatformByName('Reference')
    context = Context(system, integrator, platform)
    context.setPositions(positions)

    # 获取NonbondedForce
    nonbonded = None
    for force in system.getForces():
        if isinstance(force, NonbondedForce):
            nonbonded = force
            break

    # 运行SCF迭代
    final_charges = scf_iteration_manual(
        context, nonbonded,
        cathode_indices, anode_indices,
        cathode_areas, anode_areas,
        voltage_kjmol, Lgap, Lcell, sheet_area,
        z_cathode, z_anode,
        electrolyte_indices, electrolyte_charges,
        nIterations=4
    )

    # 验证
    print("\n" + "="*60)
    print("验证结果")
    print("="*60)

    Q_cathode = sum([final_charges[idx] for idx in cathode_indices])
    Q_anode = sum([final_charges[idx] for idx in anode_indices])

    print(f"\n总电荷:")
    print(f"  阴极: {Q_cathode:.10e}")
    print(f"  阳极: {Q_anode:.10e}")
    print(f"  总和: {Q_cathode + Q_anode:.10e}")

    if abs(Q_cathode + Q_anode) < 1e-6:
        print("\n✅ 电荷守恒：通过")
    else:
        print("\n❌ 电荷守恒：失败")

    # 保存结果供Plugin测试对比
    np.savez('test_minimal_reference.npz',
             final_charges=final_charges,
             Q_cathode=Q_cathode,
             Q_anode=Q_anode,
             voltage_volts=voltage_volts,
             Lcell=Lcell,
             Lgap=Lgap,
             sheet_area=sheet_area,
             z_cathode=z_cathode,
             z_anode=z_anode)

    print("\n结果已保存到 test_minimal_reference.npz")
    print("\n完成！")

if __name__ == "__main__":
    main()
