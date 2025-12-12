# 配置文件使用指南

## 快速开始

### 1. 使用默认配置运行

```bash
python3 run_from_config.py
```

这会使用默认的`simulation_config.ini`配置文件。

### 2. 使用自定义配置文件

```bash
python3 run_from_config.py my_config.ini
```

### 3. 测试配置文件

在运行模拟之前，可以先测试配置文件是否正确：

```bash
python3 config_parser.py my_config.ini
```

这会显示配置摘要，不会运行模拟。

---

## 配置文件结构

配置文件使用标准的`.ini`格式，包含以下sections:

### [System] - 系统文件

```ini
[System]
# PDB文件（绝对路径或相对路径）
pdb_file = /path/to/system.pdb

# Force field目录
forcefield_dir = /path/to/ffdir/

# Force field XML文件（相对于forcefield_dir，逗号分隔）
forcefield_files = sapt_residues.xml,
                   graph_residue_c.xml,
                   sapt.xml

# 非键cutoff (nm)
nonbonded_cutoff = 1.4
```

### [Electrodes] - 电极配置

```ini
[Electrodes]
# 电压 (Volts)
voltage = 1.0

# Cathode chain indices（逗号分隔）
# 对应Original的 cathode_index = (0, 2)
cathode_chains = 0, 2

# Anode chain indices（逗号分隔）
anode_chains = 1, 3

# 排除的元素（逗号分隔）
exclude_elements = H
```

**重要**:
- `cathode_chains`和`anode_chains`对应Original中的`cathode_index`和`anode_index`
- 如果Original用的是单个chain，写`cathode_chains = 0`
- 如果是多个chains，写`cathode_chains = 0, 2`

### [SCF] - SCF参数

```ini
[SCF]
# SCF迭代次数（每次SCF做几次迭代）
num_iterations = 4

# SCF频率 (femtoseconds)
# 每隔多少fs更新一次电荷
scf_frequency_fs = 200
```

**对应关系**:
- `num_iterations` ↔ Original的`Niterations`
- `scf_frequency_fs` ↔ Original的`freq_charge_update_fs`

### [Electrolyte] - 电解质配置

```ini
[Electrolyte]
# Residue atom数cutoff
# atom数 < cutoff 的residue被识别为electrolyte
natom_cutoff = 100
```

### [Simulation] - 模拟参数

```ini
[Simulation]
# 总模拟时间 (nanoseconds)
total_time_ns = 0.5

# 时间步长 (picoseconds)
timestep_ps = 0.001

# 初始温度 (Kelvin)
temperature = 300.0
```

### [Output] - 输出设置

```ini
[Output]
# 输出目录
output_dir = output

# 轨迹输出频率 (picoseconds)
trajectory_output_ps = 10.0

# Log输出频率 (steps)
log_output_steps = 100

# 是否写出电荷数据
write_charges = False

# 是否覆盖已存在的输出目录
overwrite_output = True
```

### [Platform] - 计算平台

```ini
[Platform]
# 平台名称: CUDA, OpenCL, CPU, Reference
platform_name = CUDA

# CUDA精度: mixed, single, double
cuda_precision = mixed
```

### [Advanced] - 高级选项（可选）

```ini
[Advanced]
# 是否使用SAPT-FF exclusions
sapt_ff_exclusions = True

# 约束: HBonds, AllBonds, None
constraints = HBonds

# 是否使用rigid water
rigid_water = True

# 递归深度限制
recursion_limit = 2000

# Console输出频率 (picoseconds)
console_output_frequency_ps = 10.0
```

---

## 示例配置文件

### 示例1: 短时间测试 (10 ps, 1V)

保存为`test_short.ini`:

```ini
[System]
pdb_file = system.pdb
forcefield_dir = ./ffdir/
forcefield_files = sapt_residues.xml, sapt.xml
nonbonded_cutoff = 1.4

[Electrodes]
voltage = 1.0
cathode_chains = 0
anode_chains = 1
exclude_elements = H

[SCF]
num_iterations = 4
scf_frequency_fs = 200

[Electrolyte]
natom_cutoff = 100

[Simulation]
total_time_ns = 0.01
timestep_ps = 0.001
temperature = 300.0

[Output]
output_dir = test_10ps
trajectory_output_ps = 1.0
log_output_steps = 100
write_charges = False
overwrite_output = True

[Platform]
platform_name = CUDA
cuda_precision = mixed
```

运行:
```bash
python3 run_from_config.py test_short.ini
```

### 示例2: 生产运行 (5 ns, 2V)

保存为`production.ini`:

```ini
[System]
pdb_file = equilibrated.pdb
forcefield_dir = ./ffdir/
forcefield_files = sapt_residues.xml, sapt.xml
nonbonded_cutoff = 1.4

[Electrodes]
voltage = 2.0
cathode_chains = 0, 2
anode_chains = 1, 3
exclude_elements = H

[SCF]
num_iterations = 4
scf_frequency_fs = 200

[Electrolyte]
natom_cutoff = 100

[Simulation]
total_time_ns = 5.0
timestep_ps = 0.001
temperature = 300.0

[Output]
output_dir = production_2V_5ns
trajectory_output_ps = 20.0
log_output_steps = 1000
write_charges = True
overwrite_output = False

[Platform]
platform_name = CUDA
cuda_precision = mixed
```

### 示例3: CPU调试

保存为`debug.ini`:

```ini
[System]
pdb_file = system.pdb
forcefield_dir = ./ffdir/
forcefield_files = sapt_residues.xml, sapt.xml
nonbonded_cutoff = 1.4

[Electrodes]
voltage = 0.5
cathode_chains = 0
anode_chains = 1
exclude_elements = H

[SCF]
num_iterations = 2
scf_frequency_fs = 100

[Electrolyte]
natom_cutoff = 100

[Simulation]
total_time_ns = 0.001
timestep_ps = 0.001
temperature = 300.0

[Output]
output_dir = debug
trajectory_output_ps = 0.1
log_output_steps = 10
write_charges = False
overwrite_output = True

[Platform]
platform_name = CPU
cuda_precision = mixed

[Advanced]
console_output_frequency_ps = 0.1
```

---

## 从Original转换到配置文件

如果你有Original的`run_openMM.py`，按照以下步骤转换成配置文件：

### 步骤1: 创建新的.ini文件

```bash
cp simulation_config.ini my_simulation.ini
```

### 步骤2: 修改参数

打开Original的`run_openMM.py`，找到以下内容并填入配置文件：

| Original参数 | 位置 | 配置文件参数 | Section |
|-------------|------|-------------|---------|
| PDB file | Line 89 | `pdb_file` | [System] |
| Force field XMLs | Line 89 | `forcefield_files` | [System] |
| `Voltage` | Line 73 | `voltage` | [Electrodes] |
| `cathode_index` | Line 78 | `cathode_chains` | [Electrodes] |
| `anode_index` | Line 78 | `anode_chains` | [Electrodes] |
| `exclude_element` | Line 109 | `exclude_elements` | [Electrodes] |
| `Niterations` | Line 163 | `num_iterations` | [SCF] |
| `freq_charge_update_fs` | Line 34 | `scf_frequency_fs` | [SCF] |
| `simulation_time_ns` | Line 34 | `total_time_ns` | [Simulation] |
| `freq_traj_output_ps` | Line 34 | `trajectory_output_ps` | [Output] |
| `outPath` | Line 37 | `output_dir` | [Output] |
| `set_platform('CUDA')` | Line 100 | `platform_name` | [Platform] |

### 步骤3: 运行

```bash
python3 run_from_config.py my_simulation.ini
```

---

## 参数扫描（Voltage Scan）

要运行不同电压的模拟，可以创建多个配置文件：

```bash
# 创建配置文件
cp simulation_config.ini voltage_0V.ini
cp simulation_config.ini voltage_1V.ini
cp simulation_config.ini voltage_2V.ini

# 修改每个文件的voltage和output_dir

# 批量运行
python3 run_from_config.py voltage_0V.ini
python3 run_from_config.py voltage_1V.ini
python3 run_from_config.py voltage_2V.ini
```

或者写一个简单的bash脚本：

```bash
#!/bin/bash
for voltage in 0.0 0.5 1.0 1.5 2.0; do
    # 创建临时配置文件
    sed "s/^voltage = .*/voltage = $voltage/" simulation_config.ini > temp.ini
    sed -i "s/^output_dir = .*/output_dir = voltage_${voltage}V/" temp.ini

    # 运行模拟
    python3 run_from_config.py temp.ini
done
```

---

## 常见问题

### Q1: 如何验证配置文件是否正确？

```bash
python3 config_parser.py my_config.ini
```

这会显示所有参数的摘要，不运行模拟。

### Q2: 配置文件中的路径是相对路径还是绝对路径？

两者都可以。如果使用相对路径，是相对于**运行脚本时的当前目录**。

推荐使用绝对路径以避免混淆。

### Q3: 如何修改多个chains的cathode？

```ini
# 如果cathode是chain 0和2:
cathode_chains = 0, 2

# 如果只有一个chain 0:
cathode_chains = 0
```

### Q4: 如何排除多个元素？

```ini
# 排除H和He:
exclude_elements = H, He
```

### Q5: 如何快速改变电压进行测试？

只需要修改配置文件的一行：

```ini
[Electrodes]
voltage = 1.5  # 从0.0改成1.5
```

然后重新运行：
```bash
python3 run_from_config.py simulation_config.ini
```

---

## 提供的示例配置文件

在`configs/`目录下有几个示例配置文件：

1. **`config_1V_short.ini`** - 1V电压，10 ps短测试
2. **`config_2V_long.ini`** - 2V电压，5 ns生产运行
3. **`config_CPU_debug.ini`** - CPU平台调试配置

使用方法：
```bash
python3 run_from_config.py configs/config_1V_short.ini
```

---

## 配置文件的优势

### vs 直接修改Python代码

| 方面 | 修改Python代码 | 使用配置文件 |
|------|---------------|-------------|
| 修改参数 | 需要编辑代码 | 只需编辑.ini文件 |
| 参数扫描 | 需要写循环或脚本 | 创建多个.ini文件 |
| 可读性 | 混杂在代码中 | 集中在配置文件 |
| 版本控制 | 代码和参数混在一起 | 可以分别管理 |
| 出错风险 | 可能改坏代码 | 只改参数，代码安全 |
| 分享参数 | 需要分享整个代码 | 只分享小的.ini文件 |

### vs Original的方式

| 方面 | Original | Plugin + 配置文件 |
|------|---------|-----------------|
| 参数位置 | 分散在代码多处 | 集中在.ini文件 |
| 修改参数 | 需要找到代码中的行 | 直接看section名称 |
| 批量运行 | 需要修改代码 | 创建多个.ini文件 |
| 新用户学习 | 需要理解代码 | 只需看配置文件 |

---

## 总结

使用配置文件的工作流程：

1. **准备配置文件**
   ```bash
   cp simulation_config.ini my_run.ini
   # 编辑my_run.ini
   ```

2. **验证配置**
   ```bash
   python3 config_parser.py my_run.ini
   ```

3. **运行模拟**
   ```bash
   python3 run_from_config.py my_run.ini
   ```

4. **参数扫描**
   ```bash
   # 创建多个配置文件，每个不同的参数
   python3 run_from_config.py config1.ini
   python3 run_from_config.py config2.ini
   python3 run_from_config.py config3.ini
   ```

**核心优势**: 修改参数不需要改代码，只需要编辑文本文件！
