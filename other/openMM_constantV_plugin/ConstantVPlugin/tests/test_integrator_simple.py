#!/usr/bin/env python
"""
简单测试 - 验证ConstantVIntegrator能否加载和运行
"""

import sys
from openmm.app import *
from openmm import *
from openmm.unit import *

# 加载插件
try:
    from constantvplugin import ConstantVIntegrator
    print("✓ 成功加载ConstantVIntegrator")
except ImportError as e:
    print(f"✗ 加载ConstantVIntegrator失败: {e}")
    sys.exit(1)

# 创建简单系统
system = System()
system.addParticle(12.0)  # 1个粒子
system.addParticle(12.0)
system.addParticle(12.0)
system.addParticle(12.0)
system.addParticle(23.0)  # Na+

# 位置
positions = [
    Vec3(1.0, 1.0, 1.0),
    Vec3(1.1, 1.0, 1.0),
    Vec3(1.0, 1.0, 4.0),
    Vec3(1.1, 1.0, 4.0),
    Vec3(1.0, 1.0, 2.5),
]

# NonbondedForce
nonbonded = NonbondedForce()
nonbonded.setNonbondedMethod(NonbondedForce.NoCutoff)
nonbonded.addParticle(0.0, 0.34, 0.1)  # C1
nonbonded.addParticle(0.0, 0.34, 0.1)  # C2
nonbonded.addParticle(0.0, 0.34, 0.1)  # A1
nonbonded.addParticle(0.0, 0.34, 0.1)  # A2
nonbonded.addParticle(1.0, 0.24, 0.05)  # Na+
system.addForce(nonbonded)

# 创建ConstantVIntegrator
print("\n创建ConstantVIntegrator...")
integrator = ConstantVIntegrator(0.001*picoseconds)

# 设置参数
integrator.setVoltage(1.0)  # 1V
integrator.setLgap(2.0)     # 2 nm
integrator.setLcell(3.0)    # 3 nm
integrator.setTotalArea(4.0)  # 4 nm^2
integrator.setZCathode(1.0)
integrator.setZAnode(4.0)
integrator.setNumSCFIterations(4)
integrator.setSCFFrequency(1)

# 添加电极原子
integrator.addCathodeAtom(0, 2.0)
integrator.addCathodeAtom(1, 2.0)
integrator.addAnodeAtom(2, 2.0)
integrator.addAnodeAtom(3, 2.0)
integrator.addElectrolyteAtom(4, 1.0)

print("✓ 参数设置完成")

# 创建Context
print("\n创建Context...")
try:
    platform = Platform.getPlatformByName('Reference')
    context = Context(system, integrator, platform)
    context.setPositions(positions)
    print("✓ Context创建成功")
except Exception as e:
    print(f"✗ Context创建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 运行1步
print("\n运行1步...")
try:
    integrator.step(1)
    print("✓ 运行成功！")
except Exception as e:
    print(f"✗ 运行失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 检查电荷
print("\n最终电荷:")
for i in range(5):
    charge, sigma, epsilon = nonbonded.getParticleParameters(i)
    print(f"  粒子{i}: q = {charge:.6e}")

print("\n✓✓✓ 测试通过！✓✓✓")
