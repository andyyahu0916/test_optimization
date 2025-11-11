# Plugin版本使用指南 - 针对Original用户

## 快速开始

如果你已经有Original版本的模拟代码，想转换成Plugin版本：

### 1. 直接运行转换好的脚本

我已经为你的`nvt_0V_15ns.pdb`系统创建了完整的Plugin版本脚本：

```bash
cd /home/andy/test_optimization/openMM_constantV_plugin
python3 run_plugin_nvt_0V_15ns.py
```

这个脚本完全复现了`/home/andy/test_optimization/OpenMM-ConstantV(original)/run_openMM.py`的行为。

### 2. 验证结果一致性

运行完成后，对比两个版本的输出：

```bash
# Plugin输出
ls 1v_0.5ns/
# 应该看到: FV_NVT.dcd, output.log, start.pdb, final.pdb

# Original输出（如果已运行）
ls /home/andy/test_optimization/OpenMM-ConstantV\(original\)/1v_0.5ns/
```

可以用VMD或其他工具对比轨迹文件。

---

## 核心使用区别

### Original版本（你熟悉的）

```python
# 1行创建MM对象，自动处理一切
MMsys = MM(pdb_list=['nvt_0V_15ns.pdb'], ...)

# 1行初始化电极
MMsys.initialize_electrodes(Voltage, cathode_index=(0,2), anode_index=(1,3),
                            chain=True, exclude_element=("H",))

# 1行初始化电解质
MMsys.initialize_electrolyte(Natom_cutoff=100)

# 1行生成exclusions（自动reinitialize）
MMsys.generate_exclusions(flag_SAPT_FF_exclusions=True)

# 手动循环SCF + MD
for i in range(...):
    for j in range(...):
        MMsys.Poisson_solver_fixed_voltage(Niterations=4)
        MMsys.simmd.step(200)
```

### Plugin版本（新方式）

```python
# 手动创建system
pdb = app.PDBFile('nvt_0V_15ns.pdb')
forcefield = app.ForceField(...)
system = forcefield.createSystem(...)

# 手动创建integrator
integrator = ConstantVIntegrator(0.001)  # timestep in ps
integrator.setVoltage(0.0)
integrator.setNumSCFIterations(4)
integrator.setSCFFrequency(200)  # 每200步一次SCF

# 手动添加电极atoms
for chain in pdb.topology.chains():
    if chain.index in (0, 2):  # cathode
        for atom in chain.atoms():
            if atom.element.symbol != 'H':
                integrator.addCathodeAtom(atom.index, area_per_atom)

# 使用helper添加exclusions
add_electrode_exclusions(integrator, nonbonded_force, custom_nonbonded_force)

# 创建context并reinitialize
context = mm.Context(system, integrator, platform)
context.setPositions(pdb.positions)
context.reinitialize(preserveState=True)  # ← 必须！

# 自动处理SCF循环，只需一个step
simulation.step(500000)  # integrator内部自动每200步做SCF
```

---

## 最关键的3个区别

### 1. ⚠️ Exclusions必须手动添加和reinitialize

**Original（自动）:**
```python
MMsys.generate_exclusions(flag_SAPT_FF_exclusions=True)
# ↑ 内部自动调用context.reinitialize()
```

**Plugin（手动）:**
```python
# 在创建context之前:
add_electrode_exclusions(integrator, nonbonded_force, custom_nonbonded_force)

# 创建context
context = mm.Context(system, integrator, platform)
context.setPositions(positions)

# 在创建context之后:
context.reinitialize(preserveState=True)  # ← 不能忘记！
```

**如果忘记**: 能量会爆炸，模拟会crash

### 2. SCF循环控制方式不同

**Original（外层手动循环）:**
```python
freq_charge_update_fs = 200

# 你写循环:
for j in range(int(10000 / 200)):  # 10 ps / 200 fs
    MMsys.Poisson_solver_fixed_voltage(Niterations=4)
    MMsys.simmd.step(200)  # 200步 = 200 fs（假设timestep=1fs）
```

**Plugin（内层自动循环）:**
```python
# Plugin内部处理循环，你只需设置频率:
integrator.setNumSCFIterations(4)
integrator.setSCFFrequency(200)  # 每200步做一次SCF

# 然后直接运行:
simulation.step(10000)  # integrator自动每200步调用SCF
```

### 3. Chain Index处理

**Original（支持tuple）:**
```python
cathode_index = (0, 2)  # tuple: chain 0和2
anode_index = (1, 3)

MMsys.initialize_electrodes(
    Voltage,
    cathode_identifier=cathode_index,
    anode_identifier=anode_index,
    chain=True  # ← 告诉它用chain
)
```

**Plugin（需要手动循环）:**
```python
cathode_index = (0, 2)
cathode_atoms = []

# 手动循环所有chains:
for chain in pdb.topology.chains():
    if chain.index in cathode_index:  # ← 手动检查
        for atom in chain.atoms():
            if atom.element.symbol != 'H':
                cathode_atoms.append(atom.index)

# 然后添加:
for idx in cathode_atoms:
    integrator.addCathodeAtom(idx, area_per_atom)
```

---

## 参数对应表

| Original参数 | Original值 | Plugin参数 | Plugin值 |
|-------------|-----------|-----------|---------|
| `simulation_time_ns` | 0.5 | `num_steps` | `0.5 * 1e6 / 1.0` |
| `freq_charge_update_fs` | 200 | `setSCFFrequency(...)` | 200 |
| `Niterations` | 4 | `setNumSCFIterations(...)` | 4 |
| `freq_traj_output_ps` | 10 | `DCDReporter(file, steps)` | `10 * 1000` |
| `Voltage` | 0.0 V | `setVoltage(...)` | 0.0 |
| `cathode_index` | (0, 2) | loop chains | `if chain.index in (0,2)` |
| `anode_index` | (1, 3) | loop chains | `if chain.index in (1,3)` |
| `exclude_element` | ("H",) | check symbol | `if symbol != 'H'` |
| `Natom_cutoff` | 100 | `natom_cutoff=...` | 100 |
| `cutoff` | 1.4 nm | `nonbondedCutoff=...` | 1.4 nm |
| `timestep` | 0.001 ps | `ConstantVIntegrator(...)` | 0.001 |

---

## 完整转换Checklist

### 第0步: 确保plugin已安装

```bash
cd ConstantVPlugin/build
make PythonInstall

# 验证:
python3 -c "from constantvplugin import ConstantVIntegrator; from constantvplugin_helpers import add_electrode_exclusions; print('✓ OK')"
```

### 第1步: 从Original提取参数

打开你的`run_openMM.py`，找到以下内容：

- [ ] Line 34: `simulation_time_ns`, `freq_charge_update_fs`, `freq_traj_output_ps`
- [ ] Line 73: `Voltage`
- [ ] Line 78: `cathode_index`, `anode_index`
- [ ] Line 109: `exclude_element`
- [ ] Line 163: `Niterations`
- [ ] Line 113: `Natom_cutoff`
- [ ] PDB文件路径
- [ ] Force field XML文件列表

### 第2步: 使用我提供的脚本

两个选择：

**选项A**: 直接运行我为你准备的脚本
```bash
python3 run_plugin_nvt_0V_15ns.py
```

**选项B**: 复制`example_usage.py`并修改参数
```bash
cp ConstantVPlugin/python/example_usage.py my_simulation.py
# 然后修改里面的配置参数
```

### 第3步: 对比结果

```python
# 简单的能量对比脚本
import mdtraj as md

# 加载轨迹
traj_original = md.load_dcd('OpenMM-ConstantV(original)/1v_0.5ns/FV_NVT.dcd',
                             top='nvt_0V_15ns.pdb')
traj_plugin = md.load_dcd('1v_0.5ns/FV_NVT.dcd',
                          top='nvt_0V_15ns.pdb')

# 对比RMSD
import numpy as np
rmsd = md.rmsd(traj_plugin, traj_original, frame=0)
print(f"RMSD: mean={np.mean(rmsd):.4f} nm, std={np.std(rmsd):.4f} nm")
```

---

## 常见问题

### Q1: 能量爆炸，模拟crash

**原因**: 99%是忘记添加exclusions或忘记reinitialize

**解决**:
```python
# 检查是否有这两行:
add_electrode_exclusions(integrator, nonbonded_force, custom_nonbonded_force)
# ... create context ...
context.reinitialize(preserveState=True)
```

### Q2: 结果与Original不一致

**检查列表**:
1. SCF频率是否一致？
   - Original: `freq_charge_update_fs = 200`
   - Plugin: `integrator.setSCFFrequency(200)`

2. SCF迭代次数是否一致？
   - Original: `Niterations = 4`
   - Plugin: `integrator.setNumSCFIterations(4)`

3. 电极atoms数量是否一致？
   ```python
   print(f"Cathode: {integrator.getNumCathodeAtoms()}")
   print(f"Anode: {integrator.getNumAnodeAtoms()}")
   ```

4. Geometry参数是否一致？
   ```python
   print(f"Lgap: {integrator.getLgap()}")
   print(f"Lcell: {integrator.getLcell()}")
   ```

### Q3: 找不到电极atoms

**原因**: Chain index错误

**检查**:
```python
# 打印所有chain信息:
for chain in pdb.topology.chains():
    natoms = sum(1 for _ in chain.atoms())
    print(f"Chain {chain.index}: {natoms} atoms")
```

然后确认cathode和anode的chain indices。

### Q4: 模拟太慢

**优化**:
1. 确保使用CUDA平台:
   ```python
   platform = mm.Platform.getPlatformByName('CUDA')
   ```

2. 减少SCF频率（如果可以）:
   ```python
   integrator.setSCFFrequency(400)  # 从200改成400
   ```

3. 减少SCF迭代次数（慎重！可能影响准确性）:
   ```python
   integrator.setNumSCFIterations(2)  # 从4改成2
   ```

---

## 推荐工作流程

### 初次使用（测试）

1. 先用短时间测试:
   ```python
   simulation_time_ns = 0.01  # 10 ps
   ```

2. 检查输出:
   ```bash
   tail -20 1v_0.5ns/output.log
   ```

3. 验证能量稳定

4. 检查电极charges是否更新（如果有输出）

### 生产运行

1. 恢复正常时间:
   ```python
   simulation_time_ns = 0.5  # 或更长
   ```

2. 启用checkpoint:
   ```python
   simulation.reporters.append(
       app.CheckpointReporter('checkpoint.chk', 10000)
   )
   ```

3. 考虑写出电荷数据（如果需要分析）:
   ```python
   # TODO: 等电荷输出功能实现
   ```

---

## 获取帮助

1. **查看示例**: `ConstantVPlugin/python/example_usage.py`
2. **查看文档**: `README_USAGE.md`
3. **查看对比**: `USAGE_COMPARISON.md`
4. **查看审计**: `IMPLEMENTATION_AUDIT.md`

---

## 总结

### Plugin的优势
- ✅ 完全兼容标准OpenMM workflow
- ✅ 可以与其他OpenMM plugins组合
- ✅ C++/CUDA实现，性能更好
- ✅ 更灵活，可以集成到复杂代码中

### Plugin的劣势
- ❌ 需要更多手动setup代码
- ❌ 必须记得添加exclusions和reinitialize
- ❌ 多个chain需要手动循环

### 什么时候用Plugin？
- 需要集成到现有OpenMM项目
- 需要最佳性能（CUDA）
- 需要与其他OpenMM功能组合
- 需要灵活的控制

### 什么时候继续用Original？
- 快速原型开发
- 需要QM/MM接口
- 需要MC equilibration
- 需要Conductor support (Buckyball/Nanotube)

---

**记住: 最重要的是添加exclusions和reinitialize！**

```python
# 这两行是Plugin版本的灵魂:
add_electrode_exclusions(integrator, nonbonded_force, custom_nonbonded_force)
context.reinitialize(preserveState=True)
```
