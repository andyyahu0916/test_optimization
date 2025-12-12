# 快速开始 - 使用配置文件运行模拟

## 3种运行方式

### 方式1: 使用默认配置（最简单）

```bash
# 1. 编辑配置文件
nano simulation_config.ini

# 2. 运行
python3 run_from_config.py
```

### 方式2: 使用自定义配置

```bash
# 1. 复制并编辑配置
cp simulation_config.ini my_run.ini
nano my_run.ini

# 2. 运行
python3 run_from_config.py my_run.ini
```

### 方式3: Voltage Scan（批量运行多个电压）

```bash
# 1. 编辑基础配置
nano simulation_config.ini

# 2. 运行voltage scan脚本
./voltage_scan.sh
```

---

## 完整示例：从Original转换

假设你有Original的模拟，想转换成Plugin+配置文件：

### 步骤1: 找到Original的参数

打开Original的`run_openMM.py`，记下以下参数：

```python
# Line 73
Voltage = 0.0

# Line 78
cathode_index = (0, 2)
anode_index = (1, 3)

# Line 109
exclude_element = ("H",)

# Line 34
simulation_time_ns = 0.5
freq_charge_update_fs = 200
freq_traj_output_ps = 10

# Line 163
Niterations = 4

# Line 89
pdb_list = ['nvt_0V_15ns.pdb']
ff_xml_list = ['sapt_noDB_2sheets.xml', 'graph_c_freeze.xml', ...]
```

### 步骤2: 创建配置文件

```bash
cp simulation_config.ini my_simulation.ini
```

编辑`my_simulation.ini`：

```ini
[System]
pdb_file = /path/to/nvt_0V_15ns.pdb
forcefield_dir = /path/to/ffdir/
forcefield_files = sapt_noDB_2sheets.xml,
                   graph_c_freeze.xml,
                   ...
nonbonded_cutoff = 1.4

[Electrodes]
voltage = 0.0
cathode_chains = 0, 2
anode_chains = 1, 3
exclude_elements = H

[SCF]
num_iterations = 4
scf_frequency_fs = 200

[Simulation]
total_time_ns = 0.5
timestep_ps = 0.001
temperature = 300.0

[Output]
output_dir = output
trajectory_output_ps = 10.0
...
```

### 步骤3: 测试配置

```bash
python3 config_parser.py my_simulation.ini
```

应该看到配置摘要，确认所有参数正确。

### 步骤4: 运行模拟

```bash
python3 run_from_config.py my_simulation.ini
```

---

## 修改参数的最快方式

### 修改电压

```bash
nano simulation_config.ini
```

找到`[Electrodes]` section，修改：

```ini
voltage = 1.5  # 从0.0改成1.5
```

保存并运行：

```bash
python3 run_from_config.py
```

### 修改模拟时间

```bash
nano simulation_config.ini
```

找到`[Simulation]` section：

```ini
total_time_ns = 2.0  # 从0.5改成2.0
```

### 修改SCF频率

```bash
nano simulation_config.ini
```

找到`[SCF]` section：

```ini
scf_frequency_fs = 400  # 从200改成400（减少一半的SCF调用）
```

---

## Voltage Scan示例

要运行0V, 0.5V, 1V, 1.5V, 2V的模拟：

### 方法1: 使用提供的脚本

```bash
# 编辑脚本中的电压列表（如果需要）
nano voltage_scan.sh

# 运行
./voltage_scan.sh
```

### 方法2: 手动运行每个电压

```bash
# 创建多个配置文件
cp simulation_config.ini voltage_0V.ini
cp simulation_config.ini voltage_1V.ini
cp simulation_config.ini voltage_2V.ini

# 编辑每个文件的voltage和output_dir
nano voltage_0V.ini    # voltage = 0.0, output_dir = voltage_0V
nano voltage_1V.ini    # voltage = 1.0, output_dir = voltage_1V
nano voltage_2V.ini    # voltage = 2.0, output_dir = voltage_2V

# 依次运行
python3 run_from_config.py voltage_0V.ini
python3 run_from_config.py voltage_1V.ini
python3 run_from_config.py voltage_2V.ini
```

### 方法3: 使用简单的bash循环

```bash
for v in 0.0 0.5 1.0 1.5 2.0; do
    sed "s/^voltage = .*/voltage = $v/" simulation_config.ini > temp.ini
    sed -i "s/^output_dir = .*/output_dir = voltage_${v}V/" temp.ini
    python3 run_from_config.py temp.ini
done
rm temp.ini
```

---

## 使用示例配置文件

在`configs/`目录有3个示例配置：

### 1. 短时间测试 (1V, 10 ps)

```bash
python3 run_from_config.py configs/config_1V_short.ini
```

用于快速测试setup是否正确。

### 2. 长时间生产 (2V, 5 ns)

```bash
python3 run_from_config.py configs/config_2V_long.ini
```

用于生产运行，输出电荷数据。

### 3. CPU调试 (0.5V, 1 ps)

```bash
python3 run_from_config.py configs/config_CPU_debug.ini
```

用于在CPU上快速调试，不需要GPU。

---

## 常见工作流程

### 工作流程1: 新系统测试

```bash
# 1. 短时间测试（确保setup正确）
python3 run_from_config.py configs/config_1V_short.ini

# 2. 检查输出
ls test_1V_10ps/
tail test_1V_10ps/output.log

# 3. 如果正常，运行完整模拟
nano simulation_config.ini  # 设置想要的参数
python3 run_from_config.py
```

### 工作流程2: 参数优化

```bash
# 1. 测试不同SCF频率
# 编辑simulation_config.ini，设置scf_frequency_fs = 100
python3 run_from_config.py

# 2. 编辑simulation_config.ini，设置scf_frequency_fs = 200
python3 run_from_config.py

# 3. 编辑simulation_config.ini，设置scf_frequency_fs = 400
python3 run_from_config.py

# 4. 比较结果和性能
```

### 工作流程3: 生产运行

```bash
# 1. Equilibration (0V, 2 ns)
cp simulation_config.ini equilibration.ini
# 编辑: voltage=0.0, total_time_ns=2.0, output_dir=equilibration
python3 run_from_config.py equilibration.ini

# 2. Production runs (不同电压)
./voltage_scan.sh  # 自动运行0V, 0.5V, 1V, 1.5V, 2V

# 3. 分析结果
# 每个电压的输出在voltage_XXV_0.5ns/目录
```

---

## 故障排除

### 问题1: 配置文件格式错误

```bash
# 测试配置文件
python3 config_parser.py my_config.ini

# 如果有错误，会显示具体的问题
```

### 问题2: 路径错误

```ini
# 使用绝对路径最安全:
pdb_file = /home/andy/path/to/file.pdb

# 相对路径是相对于运行脚本时的当前目录
pdb_file = ../data/file.pdb
```

### 问题3: Chain index错误

```bash
# 查看PDB文件的chain信息
python3 -c "
import openmm.app as app
pdb = app.PDBFile('system.pdb')
for chain in pdb.topology.chains():
    natoms = sum(1 for _ in chain.atoms())
    print(f'Chain {chain.index}: {natoms} atoms')
"
```

然后在配置文件中设置正确的chain indices。

---

## 总结

### 核心优势

1. **不需要修改代码** - 只改配置文件
2. **参数集中管理** - 所有参数在一个文件
3. **易于分享** - 发送小的.ini文件即可
4. **批量运行** - 创建多个配置文件，依次运行
5. **版本控制友好** - 可以track配置文件的变化

### 推荐使用方式

**新用户**:
1. 从示例配置开始: `configs/config_1V_short.ini`
2. 修改成你的参数
3. 测试运行

**熟练用户**:
1. 准备一个基础配置文件
2. 根据需要复制并修改参数
3. 批量运行或使用voltage_scan.sh

**从Original转换**:
1. 阅读`USAGE_COMPARISON.md`了解对应关系
2. 填写配置文件
3. 用`config_parser.py`验证
4. 运行

---

## 下一步

- 阅读`CONFIG_FILE_GUIDE.md`了解所有参数细节
- 查看`USAGE_COMPARISON.md`了解与Original的区别
- 参考`README_USAGE.md`了解完整的API文档

**开始运行你的第一个模拟**:

```bash
python3 run_from_config.py configs/config_1V_short.ini
```
