# Original Python vs Plugin - 详细使用对比

## 核心区别总结

### Original Python (自动化)
- **一个类包办所有**: `MMsys = MM(...)` 自动处理system, integrator, platform, simulation
- **自动识别电极**: `initialize_electrodes()` 自动识别atoms, 计算geometry, 初始化charges
- **自动添加exclusions**: `generate_exclusions()` 自动添加并reinitialize
- **手动控制循环**: 需要写循环调用 `Poisson_solver_fixed_voltage()` + `step()`

### Plugin (手动化)
- **手动创建每个组件**: 需要分别创建 system, integrator, context
- **手动添加电极atoms**: 需要循环调用 `integrator.addCathodeAtom()`
- **手动配置geometry**: 需要调用helper function或手动set参数
- **手动添加exclusions**: **必须调用helper** 和 **手动reinitialize**
- **自动SCF循环**: Integrator内部自动处理SCF + MD

---

## 以Original的run_openMM.py为例的完整对比

### Original代码结构 (run_openMM.py)

```python
# ═══════════════════════════════════════════════════════════
# 步骤1: 创建MM对象（自动处理一切）
# ═══════════════════════════════════════════════════════════
MMsys = MM(
    pdb_list = ['nvt_0V_15ns.pdb'],
    residue_xml_list = [
        ffdir + 'sapt_residues.xml',
        ffdir + 'graph_residue_c.xml',
        ffdir + 'graph_residue_n.xml'
    ],
    ff_xml_list = [
        ffdir + 'sapt_noDB_2sheets.xml',
        ffdir + 'graph_c_freeze.xml',
        ffdir + 'graph_n_freeze.xml'
    ]
)

# 步骤2: 设置周期性residue
MMsys.set_periodic_residue(True)

# 步骤3: 选择平台
MMsys.set_platform('CUDA')

# 步骤4: 初始化电极（自动识别atoms, 计算geometry, 初始化charges）
Voltage = 0.0  # Volts
cathode_index = (0, 2)  # chain indices
anode_index = (1, 3)
MMsys.initialize_electrodes(
    Voltage,
    cathode_identifier=cathode_index,
    anode_identifier=anode_index,
    chain=True,  # 使用chain index而不是residue name
    exclude_element=("H",)  # 排除H原子
)

# 步骤5: 初始化电解质（自动识别）
MMsys.initialize_electrolyte(Natom_cutoff=100)

# 步骤6: 生成exclusions（自动添加AND自动reinitialize）
MMsys.generate_exclusions(flag_SAPT_FF_exclusions=True)

# 步骤7: 设置轨迹输出
MMsys.set_trajectory_output('FV_NVT.dcd', 10 * 1000)  # freq_traj_output_ps * 1000

# 步骤8: 运行模拟（手动循环）
simulation_time_ns = 0.5
freq_charge_update_fs = 200
freq_traj_output_ps = 10

for i in range(int(simulation_time_ns * 1000 / freq_traj_output_ps)):
    for j in range(int(freq_traj_output_ps * 1000 / freq_charge_update_fs)):
        MMsys.Poisson_solver_fixed_voltage(Niterations=4)
        MMsys.simmd.step(freq_charge_update_fs)
```

---

### Plugin对应代码（完整转换）

```python
import openmm as mm
import openmm.app as app
import openmm.unit as unit
from constantvplugin import ConstantVIntegrator
from constantvplugin_helpers import (
    add_electrode_exclusions,
    configure_geometry_from_context,
    add_electrolyte_atoms_auto,
    compute_electrode_area_per_atom
)

# ═══════════════════════════════════════════════════════════
# 步骤1: 手动加载PDB和force fields
# ═══════════════════════════════════════════════════════════
pdb = app.PDBFile('nvt_0V_15ns.pdb')

forcefield = app.ForceField(
    ffdir + 'sapt_residues.xml',
    ffdir + 'graph_residue_c.xml',
    ffdir + 'graph_residue_n.xml',
    ffdir + 'sapt_noDB_2sheets.xml',
    ffdir + 'graph_c_freeze.xml',
    ffdir + 'graph_n_freeze.xml'
)

# 步骤2: 手动创建system
system = forcefield.createSystem(
    pdb.topology,
    nonbondedCutoff=1.4*unit.nanometers,
    constraints=app.HBonds,
    rigidWater=True
)

# 步骤3: 获取NonbondedForce和CustomNonbondedForce（用于exclusions）
nonbonded_force = None
custom_nonbonded_force = None
for force in system.getForces():
    if isinstance(force, mm.NonbondedForce):
        nonbonded_force = force
    elif isinstance(force, mm.CustomNonbondedForce):
        custom_nonbonded_force = force

# ═══════════════════════════════════════════════════════════
# 步骤4: 创建ConstantVIntegrator
# ═══════════════════════════════════════════════════════════
timestep = 1.0 * unit.femtoseconds
integrator = ConstantVIntegrator(timestep.value_in_unit(unit.picoseconds))

# 设置电压
Voltage = 0.0  # Volts
integrator.setVoltage(Voltage)

# 设置SCF参数
integrator.setNumSCFIterations(4)  # 对应Original的Niterations=4
integrator.setSCFFrequency(1)  # 每1步MD做一次SCF（因为plugin内部处理循环）

# ═══════════════════════════════════════════════════════════
# 步骤5: 手动识别并添加电极atoms
# ═══════════════════════════════════════════════════════════
cathode_chain_indices = (0, 2)  # 对应Original的cathode_index=(0,2)
anode_chain_indices = (1, 3)    # 对应Original的anode_index=(1,3)
exclude_element = 'H'

# 识别cathode atoms（来自chain 0和2）
cathode_atoms = []
for chain in pdb.topology.chains():
    if chain.index in cathode_chain_indices:
        for atom in chain.atoms():
            if atom.element.symbol != exclude_element:
                cathode_atoms.append(atom.index)

# 识别anode atoms（来自chain 1和3）
anode_atoms = []
for chain in pdb.topology.chains():
    if chain.index in anode_chain_indices:
        for atom in chain.atoms():
            if atom.element.symbol != exclude_element:
                anode_atoms.append(atom.index)

# 计算每个atom的面积
cathode_area_per_atom, total_area = compute_electrode_area_per_atom(
    pdb.topology, cathode_atoms
)
anode_area_per_atom, _ = compute_electrode_area_per_atom(
    pdb.topology, anode_atoms
)

# 添加到integrator
for atom_idx in cathode_atoms:
    integrator.addCathodeAtom(atom_idx, cathode_area_per_atom)

for atom_idx in anode_atoms:
    integrator.addAnodeAtom(atom_idx, anode_area_per_atom)

# ═══════════════════════════════════════════════════════════
# 步骤6: 自动识别并添加electrolyte atoms
# ═══════════════════════════════════════════════════════════
electrolyte_atoms = add_electrolyte_atoms_auto(
    pdb.topology,
    integrator,
    nonbonded_force,
    natom_cutoff=100,  # 对应Original的Natom_cutoff=100
    exclude_chains=list(cathode_chain_indices) + list(anode_chain_indices)
)

# ═══════════════════════════════════════════════════════════
# 步骤7: 自动配置geometry参数
# ═══════════════════════════════════════════════════════════
# 创建临时context来获取positions
temp_integrator = mm.VerletIntegrator(timestep)
temp_context = mm.Context(system, temp_integrator)
temp_context.setPositions(pdb.positions)

geometry_params = configure_geometry_from_context(
    temp_context,
    integrator,
    cathode_atoms[0],  # 第一个cathode atom
    anode_atoms[0]     # 第一个anode atom
)

del temp_context
del temp_integrator

# ═══════════════════════════════════════════════════════════
# ⚠️ 步骤8: 添加electrode exclusions（最关键！）
# ═══════════════════════════════════════════════════════════
add_electrode_exclusions(integrator, nonbonded_force, custom_nonbonded_force)

# ═══════════════════════════════════════════════════════════
# 步骤9: 创建context和platform
# ═══════════════════════════════════════════════════════════
platform = mm.Platform.getPlatformByName('CUDA')
properties = {'Precision': 'mixed'}
context = mm.Context(system, integrator, platform, properties)
context.setPositions(pdb.positions)
context.setVelocitiesToTemperature(300*unit.kelvin)

# ⚠️ 关键: Reinitialize来应用exclusions
context.reinitialize(preserveState=True)

# ═══════════════════════════════════════════════════════════
# 步骤10: 设置reporters（对应Original的set_trajectory_output）
# ═══════════════════════════════════════════════════════════
simulation = app.Simulation(pdb.topology, system, integrator, platform, properties)
simulation.context = context

freq_traj_output_ps = 10.0
traj_freq_steps = int(freq_traj_output_ps * 1000 / 1.0)  # ps to steps

simulation.reporters.append(app.DCDReporter('output.dcd', traj_freq_steps))
simulation.reporters.append(app.StateDataReporter(
    'output.log',
    100,
    step=True, time=True,
    potentialEnergy=True, temperature=True
))

# ═══════════════════════════════════════════════════════════
# 步骤11: 运行模拟（plugin自动处理SCF循环）
# ═══════════════════════════════════════════════════════════
simulation_time_ns = 0.5
num_steps = int(simulation_time_ns * 1e6 / 1.0)  # ns to steps

# Plugin的integrator.step()内部自动处理：
# - 每scf_frequency步做一次SCF（包含nIterations次迭代）
# - 然后做MD step
# 所以不需要手动循环！
simulation.step(num_steps)
```

---

## 关键区别详解

### 1. SCF循环控制

**Original (手动外层循环):**
```python
# 在run_openMM.py中，用户手动控制循环：
for j in range(int(freq_traj_output_ps * 1000 / freq_charge_update_fs)):
    MMsys.Poisson_solver_fixed_voltage(Niterations=4)  # SCF 4次迭代
    MMsys.simmd.step(freq_charge_update_fs)            # 然后MD 200步
```

**Plugin (自动内层循环):**
```python
# Plugin的integrator内部自动处理，用户只需：
integrator.setNumSCFIterations(4)   # 每次SCF做4次迭代
integrator.setSCFFrequency(200)     # 每200步MD做一次SCF

# 然后直接运行，integrator自动处理：
simulation.step(500000)  # 内部自动每200步做一次SCF
```

**重要**: 如果要完全复现Original的行为：
- Original: `freq_charge_update_fs = 200` → 每200 fs做一次SCF
- Plugin需要设置: `setSCFFrequency(200)` → 每200步做一次SCF（因为timestep=1fs）

### 2. Chain Index处理

**Original:**
```python
cathode_index = (0, 2)  # tuple, 多个chains
anode_index = (1, 3)

MMsys.initialize_electrodes(
    Voltage,
    cathode_identifier=cathode_index,
    anode_identifier=anode_index,
    chain=True,  # ← 关键: 告诉Original用chain index
    exclude_element=("H",)
)
```

**Plugin:**
```python
# 需要手动循环处理多个chains:
cathode_chain_indices = (0, 2)
cathode_atoms = []
for chain in pdb.topology.chains():
    if chain.index in cathode_chain_indices:  # ← 手动检查
        for atom in chain.atoms():
            if atom.element.symbol != 'H':
                cathode_atoms.append(atom.index)

# 然后添加到integrator:
for atom_idx in cathode_atoms:
    integrator.addCathodeAtom(atom_idx, area_per_atom)
```

### 3. Exclusions处理

**Original (完全自动):**
```python
# 一行搞定，自动添加所有exclusions AND reinitialize:
MMsys.generate_exclusions(flag_SAPT_FF_exclusions=True)

# 内部实现 (MM_classes.py Line 570-622):
# 1. 添加cathode-cathode exclusions
# 2. 添加anode-anode exclusions
# 3. 添加SAPT-FF exclusions（如果需要）
# 4. 自动reinitialize context
```

**Plugin (必须手动):**
```python
# ⚠️ 必须在创建context之前:
add_electrode_exclusions(integrator, nonbonded_force, custom_nonbonded_force)

# ⚠️ 必须在创建context之后:
context = mm.Context(system, integrator, platform)
context.reinitialize(preserveState=True)  # ← 不能忘记！
```

### 4. Geometry配置

**Original (自动):**
```python
# initialize_electrodes()内部自动调用 (MM_classes.py Line 216):
self.set_electrochemical_cell_parameters(positions, boxVecs)

# 自动计算:
# - Lcell = |z_cathode - z_anode|
# - Lgap = box_z - Lcell
# - totalArea = box_x * box_y
```

**Plugin (手动或helper):**
```python
# 选项1: 使用helper (推荐)
geometry_params = configure_geometry_from_context(
    context, integrator,
    cathode_atoms[0], anode_atoms[0]
)

# 选项2: 手动设置
integrator.setLgap(0.5)      # nm
integrator.setLcell(3.5)     # nm
integrator.setTotalArea(9.0)  # nm^2
integrator.setZCathode(0.0)  # nm
integrator.setZAnode(3.5)    # nm
```

---

## 完整的从Original转换到Plugin的Checklist

假设你有Original的模拟，要转换成Plugin:

### 第1步: 识别Original的配置参数

从 `run_openMM.py` 中找到:
- [ ] `Voltage = ?` (Line 73)
- [ ] `cathode_index = ?` (Line 78)
- [ ] `anode_index = ?` (Line 78)
- [ ] `exclude_element = ?` (Line 109)
- [ ] `freq_charge_update_fs = ?` (Line 34)
- [ ] `Niterations = ?` (Line 163, 通常是4)
- [ ] PDB file path
- [ ] Force field XML files

### 第2步: 创建对应的Plugin setup

```python
# 1. 加载PDB
pdb = app.PDBFile('nvt_0V_15ns.pdb')

# 2. 加载force fields
forcefield = app.ForceField(
    # ... 所有Original用的XML files
)

# 3. 创建system
system = forcefield.createSystem(
    pdb.topology,
    nonbondedCutoff=1.4*unit.nanometers,  # 检查Original的cutoff
    constraints=app.HBonds,
    rigidWater=True
)

# 4. 获取forces
nonbonded_force = None
custom_nonbonded_force = None
for force in system.getForces():
    if isinstance(force, mm.NonbondedForce):
        nonbonded_force = force
    elif isinstance(force, mm.CustomNonbondedForce):
        custom_nonbonded_force = force
```

### 第3步: 创建integrator并设置参数

```python
# 1. 创建integrator
timestep = 1.0 * unit.femtoseconds  # 检查Original的timestep
integrator = ConstantVIntegrator(timestep.value_in_unit(unit.picoseconds))

# 2. 设置电压 (从Original的Voltage)
integrator.setVoltage(0.0)  # Volts

# 3. 设置SCF参数
integrator.setNumSCFIterations(4)  # 从Original的Niterations
integrator.setSCFFrequency(200)    # 从Original的freq_charge_update_fs
```

### 第4步: 添加电极atoms

```python
# 从Original的cathode_index和anode_index
cathode_chain_indices = (0, 2)
anode_chain_indices = (1, 3)
exclude_element = 'H'

# 识别cathode atoms
cathode_atoms = []
for chain in pdb.topology.chains():
    if chain.index in cathode_chain_indices:
        for atom in chain.atoms():
            if atom.element.symbol != exclude_element:
                cathode_atoms.append(atom.index)

# 识别anode atoms
anode_atoms = []
for chain in pdb.topology.chains():
    if chain.index in anode_chain_indices:
        for atom in chain.atoms():
            if atom.element.symbol != exclude_element:
                anode_atoms.append(atom.index)

# 计算面积
from constantvplugin_helpers import compute_electrode_area_per_atom
cathode_area_per_atom, _ = compute_electrode_area_per_atom(
    pdb.topology, cathode_atoms
)
anode_area_per_atom, _ = compute_electrode_area_per_atom(
    pdb.topology, anode_atoms
)

# 添加到integrator
for idx in cathode_atoms:
    integrator.addCathodeAtom(idx, cathode_area_per_atom)
for idx in anode_atoms:
    integrator.addAnodeAtom(idx, anode_area_per_atom)
```

### 第5步: 添加electrolyte

```python
from constantvplugin_helpers import add_electrolyte_atoms_auto

electrolyte_atoms = add_electrolyte_atoms_auto(
    pdb.topology,
    integrator,
    nonbonded_force,
    natom_cutoff=100,  # 从Original的Natom_cutoff
    exclude_chains=list(cathode_chain_indices) + list(anode_chain_indices)
)
```

### 第6步: 配置geometry

```python
from constantvplugin_helpers import configure_geometry_from_context

# 创建临时context
temp_integrator = mm.VerletIntegrator(timestep)
temp_context = mm.Context(system, temp_integrator)
temp_context.setPositions(pdb.positions)

# 自动配置
geometry_params = configure_geometry_from_context(
    temp_context,
    integrator,
    cathode_atoms[0],
    anode_atoms[0]
)

del temp_context
del temp_integrator
```

### 第7步: ⚠️ 添加exclusions（最关键！）

```python
from constantvplugin_helpers import add_electrode_exclusions

# ⚠️ 必须在创建context之前
add_electrode_exclusions(integrator, nonbonded_force, custom_nonbonded_force)
```

### 第8步: 创建context

```python
# 选择平台（从Original的set_platform）
platform = mm.Platform.getPlatformByName('CUDA')
properties = {'Precision': 'mixed'}

# 创建context
context = mm.Context(system, integrator, platform, properties)
context.setPositions(pdb.positions)
context.setVelocitiesToTemperature(300*unit.kelvin)

# ⚠️ 必须reinitialize
context.reinitialize(preserveState=True)
```

### 第9步: 设置reporters

```python
simulation = app.Simulation(pdb.topology, system, integrator, platform, properties)
simulation.context = context

# 从Original的freq_traj_output_ps
freq_traj_output_ps = 10.0
traj_freq_steps = int(freq_traj_output_ps * 1000 / 1.0)

simulation.reporters.append(app.DCDReporter('output.dcd', traj_freq_steps))
simulation.reporters.append(app.StateDataReporter(
    'output.log', 100,
    step=True, time=True, potentialEnergy=True, temperature=True
))
```

### 第10步: 运行

```python
# 从Original的simulation_time_ns
simulation_time_ns = 0.5
num_steps = int(simulation_time_ns * 1e6 / 1.0)

simulation.step(num_steps)
```

---

## 常见错误和解决方法

### 错误1: 忘记添加exclusions
**症状**: 能量爆炸，simulation crash，或非常大的forces

**解决**:
```python
# 必须在创建context之前添加:
add_electrode_exclusions(integrator, nonbonded_force, custom_nonbonded_force)

# 必须在创建context之后reinitialize:
context.reinitialize(preserveState=True)
```

### 错误2: SCF频率设置错误
**症状**: 结果与Original不一致

**检查**:
- Original: `freq_charge_update_fs = 200`
- Plugin应设置: `integrator.setSCFFrequency(200)` (假设timestep=1fs)

### 错误3: Chain index识别错误
**症状**: 找不到电极atoms或数量不对

**检查**:
- Original可能用tuple: `cathode_index=(0,2)` 表示chain 0和2
- Plugin需要循环检查: `if chain.index in (0, 2)`

### 错误4: 忘记设置geometry
**症状**: `Lgap = 0` 或除零错误

**解决**:
```python
# 必须在创建context之前设置，使用helper:
geometry_params = configure_geometry_from_context(
    temp_context, integrator,
    cathode_atoms[0], anode_atoms[0]
)
```

---

## 验证转换正确性

运行后对比以下内容与Original:

1. **电极atom数量**:
   ```python
   print(f"Cathode atoms: {integrator.getNumCathodeAtoms()}")
   print(f"Anode atoms: {integrator.getNumAnodeAtoms()}")
   ```

2. **Geometry参数**:
   ```python
   print(f"Lgap: {integrator.getLgap()}")
   print(f"Lcell: {integrator.getLcell()}")
   print(f"Total area: {integrator.getTotalArea()}")
   ```

3. **Exclusions数量**:
   ```python
   print(f"NonbondedForce exceptions: {nonbonded_force.getNumExceptions()}")
   ```

4. **初始能量**:
   ```python
   state = context.getState(getEnergy=True)
   print(f"Potential energy: {state.getPotentialEnergy()}")
   ```

5. **运行几步后的能量变化**: 应该稳定，不应该爆炸

---

## 总结

### Original的优势
- 高度自动化
- 一个MM类包办所有
- 适合快速prototyping

### Plugin的优势
- 完全兼容标准OpenMM workflow
- 更灵活，可以集成到复杂的OpenMM代码中
- 性能更好（C++实现，支持CUDA）
- 可以与其他OpenMM plugins组合使用

### 转换的核心要点
1. **Exclusions**: 必须手动添加 + reinitialize
2. **SCF循环**: Plugin自动处理，只需设置frequency
3. **Geometry**: 使用helper或手动设置
4. **Chain indices**: 手动循环处理多个chains

**记住: 读完这个文档后，最好的参考是 `python/example_usage.py`，它包含了完整的工作示例！**
