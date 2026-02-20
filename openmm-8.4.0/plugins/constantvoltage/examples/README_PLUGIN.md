# ConstantVoltage Plugin Usage

## 概述

`run_openMM_plugin.py` 是基于 ConstantVoltage 插件的新版本模拟脚本，对应原始的 `OpenMM-ConstantV(original)/run_openMM.py`。

## 主要区别

### 原始版本 (run_openMM.py)
- 手动调用 `Poisson_solver_fixed_voltage()` 进行 SCF 电荷更新
- 每次迭代都需要：
  1. `getState(getForces=True)` - 获取力
  2. 更新电极电荷
  3. `updateParametersInContext()` - 同步电荷

### 插件版本 (run_openMM_plugin.py)
- 使用 `ConstantVoltageForce` 存储电极原子数据
- 使用 `ConstantVDrudeLangevinIntegrator` 自动处理 SCF 电荷更新
- 集成器自动：
  1. 检测 `ConstantVoltageForce`
  2. 在每个 SCF 频率步骤执行 SCF 迭代
  3. 更新电极电荷并同步到 context

## 使用方法

```bash
cd /home/andy/test_optimization/openmm-8.4.0/plugins/constantvoltage/examples
python run_openMM_plugin.py
```

## 关键配置

### ConstantVoltageForce 设置
```python
cv_force = ConstantVoltageForce()
cv_force.setVoltage(Voltage)        # 电压 (V)
cv_force.setLgap(Lgap)              # 间隙长度 (nm)
cv_force.setLcell(Lcell)            # 电池长度 (nm)
cv_force.setZCathode(cathode_z)     # 阴极 z 位置 (nm)
cv_force.setZAnode(anode_z)         # 阳极 z 位置 (nm)
cv_force.setTotalArea(total_area)   # 总面积 (nm²)
cv_force.setSmallThreshold(1e-6)    # 小电荷阈值

# SCF 参数
cv_force.setNumSCFIterations(4)           # 每次更新的 SCF 迭代次数
cv_force.setSCFFrequency(200)              # SCF 更新频率 (步数)

# 添加电极原子
for atom_idx in cathode_virtual:
    cv_force.addCathodeAtom(atom_idx, area_atom)
for atom_idx in anode_virtual:
    cv_force.addAnodeAtom(atom_idx, area_atom)
for atom_idx in electrolyte_atoms:
    cv_force.addElectrolyteAtom(atom_idx)
```

### ConstantVDrudeLangevinIntegrator 设置
```python
integrator = ConstantVDrudeLangevinIntegrator(
    300.0,      # temperature (K)
    1.0,        # friction (1/ps)
    1.0,        # drudeTemperature (K)
    40.0,       # drudeFriction (1/ps)
    0.001       # stepSize (ps)
)
integrator.setMaxDrudeDistance(0.02)  # 0.02 nm
```

## 修复的问题

### SCF 迭代中的电荷同步

**问题**：在 SCF 迭代中，直接修改 `posq` buffer 后，PME 等力计算可能仍使用缓存的旧电荷值。

**修复**：在 `CudaConstantVoltageKernels.cpp` 中添加了 `invalidateMolecules()` 调用：

1. **每次 SCF 迭代开始时**：确保使用最新电荷计算力
2. **每次 SCF 迭代结束后**：同步电荷变化，确保下次迭代使用更新后的电荷

```cpp
for (int i = 0; i < numSCFIterations; i++) {
    cu.invalidateMolecules();  // 确保电荷是最新的
    context.calcForcesAndEnergy(...);
    forceKernel->updateElectrodeCharges(context);
    cu.invalidateMolecules();  // 同步电荷变化
}
```

这确保了与原始 Python 实现的一致性，其中每次迭代都调用 `updateParametersInContext()`。

## 输出文件

- `1v_0.5ns_plugin/start_drudes.pdb` - 初始结构（包含 Drude 粒子）
- `1v_0.5ns_plugin/FV_NVT.dcd` - 轨迹文件

## 注意事项

1. **需要 DrudeForce**：`ConstantVDrudeLangevinIntegrator` 需要系统包含 `DrudeForce`
2. **电极结构**：只使用虚拟层（virtual chain）的原子参与 SCF 电荷更新
3. **排除元素**：默认排除氢原子（H）从电极原子中

## 与原始版本的兼容性

- 使用相同的力场文件
- 使用相同的电极结构（chain indices）
- 使用相同的 SCF 参数（4 次迭代，每 200 fs 更新）
- 输出格式相同

