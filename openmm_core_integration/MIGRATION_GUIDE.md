# 🔄 从原始实现迁移到 Native Core Integration 指南

本文档说明如何使用新的 Native Core Integration 运行与原始 `run_openMM.py` 完全相同的计算。

---

## 📋 快速对比

| 原始实现 | Native Core Integration |
|---------|------------------------|
| `MM` 类 | `ConstantVSystemBuilder` + 手动设置 |
| `Poisson_solver_fixed_voltage(Niterations=4)` | `integrator.step()` (SCF 内部处理) |
| `simmd.step(freq_charge_update_fs)` | `simulation.step(steps)` |
| Python 插件 | C++/CUDA 原生实现 |

---

## 🚀 使用方法

### 方法 1: 使用等价启动脚本（推荐）

```bash
cd /home/andy/test_optimization/openmm_core_integration
python run_native_constantv.py
```

这个脚本完全对齐原始 `run_openMM.py` 的工作流程。

### 方法 2: 使用 SystemBuilder（高级）

```python
from openmm_constantv import SystemConfig, ConstantVSystemBuilder, ElectrodeConfig

config = SystemConfig(
    pdb_files=['nvt_0V_15ns.pdb'],
    residue_xml_files=[
        'ffdir/sapt_residues.xml',
        'ffdir/graph_residue_c.xml',
        'ffdir/graph_residue_n.xml',
    ],
    forcefield_xml_files=[
        'ffdir/sapt_noDB_2sheets.xml',
        'ffdir/graph_c_freeze.xml',
        'ffdir/graph_n_freeze.xml',
    ],
    voltage_volts=0.0,
    cathode=ElectrodeConfig(identifier=(0, 2), electrode_type="cathode", by_chain=True),
    anode=ElectrodeConfig(identifier=(1, 3), electrode_type="anode", by_chain=True),
    temperature_kelvin=300.0,
    temperature_drude_kelvin=1.0,
    timestep_ps=0.001,
    cutoff_nm=1.4,
    scf_iterations=4,
)

builder = ConstantVSystemBuilder(config)
system, topology, modeller = builder.build()

# 创建 integrator
import constantv
integrator = constantv.ConstantVDrudeLangevinIntegrator(
    temperature=300.0,
    frictionCoeff=1.0,
    drudeTemperature=1.0,
    drudeFrictionCoeff=1.0,
    stepSize=0.001,
    voltage=0.0,
    Lgap=builder.Lgap_nm,
    Lcell=builder.Lcell_nm,
    scfIterations=4
)

# 设置 SCF 频率（每 200 fs 更新一次电荷）
integrator.setSCFFrequency(200)  # 200 fs / 1 fs timestep = 200 steps

# 添加电极和电解质
for idx in builder.cathode_indices:
    integrator.addCathodeAtom(idx, builder.planar_area_nm2 / len(builder.cathode_indices))
for idx in builder.anode_indices:
    integrator.addAnodeAtom(idx, builder.planar_area_nm2 / len(builder.anode_indices))
for idx in builder.electrolyte_indices:
    charge, _, _ = system.getForce(0).getParticleParameters(idx)
    integrator.addElectrolyteAtom(idx, charge.value_in_unit(unit.elementary_charge))

# 创建模拟
from openmm import app, Platform
platform = Platform.getPlatformByName('CUDA')
simulation = app.Simulation(topology, system, integrator, platform, {'Precision': 'mixed'})
simulation.context.setPositions(modeller.positions)

# 运行模拟
simulation.step(1000000)
```

---

## 🔑 关键差异说明

### 1. SCF 更新频率

**原始实现**:
```python
for j in range(num_charge_updates):
    MMsys.Poisson_solver_fixed_voltage(Niterations=4)  # 每 200 fs 调用一次
    MMsys.simmd.step(freq_charge_update_fs)            # 运行 200 fs MD
```

**Native Core Integration**:
```python
integrator.setSCFFrequency(200)  # 每 200 步运行一次 SCF
simulation.step(200)             # 运行 200 步（SCF 自动在第一步运行）
```

### 2. 电极识别

**原始实现**:
```python
MMsys.initialize_electrodes(
    Voltage,
    cathode_identifier=cathode_index,  # (0, 2)
    anode_identifier=anode_index,      # (1, 3)
    chain=True,
    exclude_element=("H",)
)
```

**Native Core Integration**:
```python
# 方法 1: 手动识别（run_native_constantv.py 使用）
for chain in topology.chains():
    if chain.index in cathode_index:
        for atom in chain.atoms():
            if atom.element.symbol != 'H':
                integrator.addCathodeAtom(atom.index, area_per_atom)

# 方法 2: 使用 SystemBuilder（自动识别）
cathode = ElectrodeConfig(identifier=(0, 2), electrode_type="cathode", by_chain=True)
```

### 3. 电解质识别

**原始实现**:
```python
MMsys.initialize_electrolyte(Natom_cutoff=100)
```

**Native Core Integration**:
```python
# 手动识别
for residue in topology.residues():
    if sum(1 for _ in residue.atoms()) < 100:
        for atom in residue.atoms():
            integrator.addElectrolyteAtom(atom.index, charge)
```

---

## ⚙️ 参数对应表

| 原始参数 | Native Core Integration | 说明 |
|---------|------------------------|------|
| `Voltage` (Volts) | `voltage` (Volts) | 相同 |
| `Niterations=4` | `scfIterations=4` | 相同 |
| `freq_charge_update_fs=200` | `setSCFFrequency(200)` | 每 200 步更新一次 |
| `freq_traj_output_ps=10` | `DCDReporter(..., 10000)` | 每 10 ps = 10000 步输出 |
| `temperature=300*kelvin` | `temperature=300.0` | 相同 |
| `temperature_drude=1*kelvin` | `drudeTemperature=1.0` | 相同 |
| `friction=1/picosecond` | `frictionCoeff=1.0` | 相同 |
| `friction_drude=1/picosecond` | `drudeFrictionCoeff=1.0` | 相同 |
| `timestep=0.001*picoseconds` | `stepSize=0.001` | 相同 |

---

## 📝 完整工作流程对比

### 原始实现 (`run_openMM.py`)

```python
# 1. 创建 MM 系统
MMsys = MM(pdb_list=[...], residue_xml_list=[...], ff_xml_list=[...])

# 2. 设置平台
MMsys.set_platform('CUDA')

# 3. 初始化电极
MMsys.initialize_electrodes(Voltage, cathode_identifier=..., anode_identifier=..., chain=True)

# 4. 初始化电解质
MMsys.initialize_electrolyte(Natom_cutoff=100)

# 5. 生成排除项
MMsys.generate_exclusions(flag_SAPT_FF_exclusions=True)

# 6. 运行模拟
for i in range(num_output_steps):
    for j in range(num_charge_updates):
        MMsys.Poisson_solver_fixed_voltage(Niterations=4)
        MMsys.simmd.step(freq_charge_update_fs)
```

### Native Core Integration (`run_native_constantv.py`)

```python
# 1. 加载 PDB 和力场
pdb = app.PDBFile('nvt_0V_15ns.pdb')
forcefield = app.ForceField(...)
modeller = app.Modeller(pdb.topology, pdb.positions)
system = forcefield.createSystem(...)

# 2. 识别电极和电解质
cathode_atoms = [...]  # 通过 chain index
anode_atoms = [...]
electrolyte_atoms = [...]

# 3. 创建 integrator
integrator = constantv.ConstantVDrudeLangevinIntegrator(...)
integrator.setSCFFrequency(200)  # 每 200 步更新电荷

# 4. 添加电极和电解质
for idx in cathode_atoms:
    integrator.addCathodeAtom(idx, area_per_atom)
# ... 类似地添加 anode 和 electrolyte

# 5. 创建模拟
simulation = app.Simulation(topology, system, integrator, platform, properties)

# 6. 运行模拟
for i in range(num_output_steps):
    for j in range(num_charge_updates):
        simulation.step(200)  # SCF 自动在第一步运行
```

---

## ✅ 验证对齐

运行 `run_native_constantv.py` 应该产生与原始实现相同的结果：

1. ✅ **物理正确性**: SCF 迭代逻辑完全对齐
2. ✅ **算法正确性**: 电荷更新频率和 MD 步数相同
3. ✅ **数值精度**: 使用相同的单位转换常数和公式

---

## 🐛 故障排除

### 问题 1: `constantv` 模块未找到

```bash
cd openmm_core_integration/build
make install
```

### 问题 2: SCF 更新频率不正确

确保 `setSCFFrequency()` 的值正确：
```python
# timestep = 0.001 ps = 1 fs
# freq_charge_update_fs = 200 fs
# scfFrequency = 200 fs / 1 fs = 200 steps
integrator.setSCFFrequency(200)
```

### 问题 3: 电极原子未识别

检查 chain index 或 residue name 是否正确：
```python
# 打印识别的原子
print(f"Cathode atoms: {cathode_atoms}")
print(f"Anode atoms: {anode_atoms}")
```

---

## 📚 更多信息

- 详细 API 文档: `openmm_core_integration/README.md`
- 测试示例: `openmm_core_integration/test_native_integration.py`
- 深度审核报告: `openmm_core_integration/ULTIMATE_DEEP_AUDIT.md`

