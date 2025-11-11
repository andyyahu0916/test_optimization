# 从这里开始 - ConstantV Plugin配置文件系统

## 👋 欢迎！

你现在有了一个完整的**配置文件驱动**的模拟系统。

**核心理念**: 修改模拟参数**不需要改代码**，只需要编辑配置文件（`.ini`文件）。

---

## 🚀 最快开始（3步）

### 1️⃣ 编辑配置文件

```bash
nano simulation_config.ini
```

修改你需要的参数（比如电压、模拟时间等）。

### 2️⃣ 验证配置

```bash
python3 config_parser.py simulation_config.ini
```

会显示配置摘要，确认所有参数正确。

### 3️⃣ 运行模拟

```bash
python3 run_from_config.py
```

就这么简单！

---

## 📚 文档导航

### 新用户必读

1. **`QUICK_START.md`** ⭐ 最重要！
   - 3种运行方式
   - 完整示例
   - 从Original转换的详细步骤

2. **`CONFIG_FILE_GUIDE.md`**
   - 所有参数的详细说明
   - 配置文件结构
   - 大量示例

### 从Original转换

3. **`USAGE_COMPARISON.md`**
   - Original vs Plugin详细对比
   - 参数对应表
   - 转换checklist

4. **`HOW_TO_USE_PLUGIN.md`**
   - 针对Original用户的快速指南
   - 关键区别
   - 常见错误

### 快速参考

5. **`README_CONFIG_FILES.md`**
   - 配置文件系统总结
   - 所有文件说明
   - 使用方式对比

---

## 📂 重要文件

### 运行脚本

- **`run_from_config.py`** - 主运行脚本
  ```bash
  python3 run_from_config.py [config_file.ini]
  ```

- **`config_parser.py`** - 配置文件验证工具
  ```bash
  python3 config_parser.py [config_file.ini]
  ```

- **`voltage_scan.sh`** - 批量运行多个电压
  ```bash
  ./voltage_scan.sh
  ```

### 配置文件

- **`simulation_config.ini`** - 默认配置文件
  - 修改这个文件来设置你的参数

- **`configs/config_1V_short.ini`** - 快速测试 (1V, 10 ps)
- **`configs/config_2V_long.ini`** - 生产运行 (2V, 5 ns)
- **`configs/config_CPU_debug.ini`** - CPU调试配置

---

## 💡 3个典型使用场景

### 场景A: 单次运行

```bash
# 1. 编辑默认配置
nano simulation_config.ini

# 2. 运行
python3 run_from_config.py
```

### 场景B: 测试不同参数

```bash
# 1. 创建多个配置
cp simulation_config.ini test1.ini
cp simulation_config.ini test2.ini
cp simulation_config.ini test3.ini

# 2. 编辑每个配置文件的参数

# 3. 依次运行
python3 run_from_config.py test1.ini
python3 run_from_config.py test2.ini
python3 run_from_config.py test3.ini
```

### 场景C: Voltage Scan

```bash
# 1. （可选）编辑voltage_scan.sh中的电压列表
nano voltage_scan.sh

# 2. 运行
./voltage_scan.sh

# 自动运行0V, 0.5V, 1V, 1.5V, 2V
```

---

## 🎯 快速修改参数

### 修改电压

```bash
nano simulation_config.ini
```

找到：
```ini
[Electrodes]
voltage = 0.0  # ← 改成你想要的电压
```

### 修改模拟时间

```ini
[Simulation]
total_time_ns = 0.5  # ← 改成你想要的时间（纳秒）
```

### 修改输出目录

```ini
[Output]
output_dir = 1v_0.5ns  # ← 改成你想要的目录名
```

---

## ✅ 与Original对比

### Original方式

```python
# run_openMM.py Line 73
Voltage = 0.0  # 要改电压需要找到这一行

# Line 34
simulation_time_ns = 0.5  # 要改时间需要找到这一行

# Line 37
outPath = '1v_0.5ns'  # 要改输出需要找到这一行

# ...参数分散在代码各处
```

### Plugin + 配置文件方式

```ini
# simulation_config.ini - 所有参数在一个文件！
[Electrodes]
voltage = 0.0

[Simulation]
total_time_ns = 0.5

[Output]
output_dir = 1v_0.5ns
```

**优势**: 不需要在代码中找参数，所有参数集中管理！

---

## 🔧 故障排除

### 问题1: 不知道配置文件格式对不对

```bash
python3 config_parser.py simulation_config.ini
```

会显示配置摘要。如果有错误，会提示具体问题。

### 问题2: 不知道chain index是什么

查看PDB文件的chain信息：
```bash
python3 -c "
import openmm.app as app
pdb = app.PDBFile('your_file.pdb')
for chain in pdb.topology.chains():
    print(f'Chain {chain.index}: {sum(1 for _ in chain.atoms())} atoms')
"
```

然后在配置文件中设置正确的`cathode_chains`和`anode_chains`。

### 问题3: 找不到PDB或force field文件

在配置文件中使用**绝对路径**：
```ini
[System]
pdb_file = /home/andy/path/to/file.pdb
forcefield_dir = /home/andy/path/to/ffdir/
```

---

## 📖 学习路径

### 初学者

1. ✅ 阅读本文件 (`START_HERE.md`)
2. ✅ 阅读`QUICK_START.md`
3. ✅ 运行测试: `python3 run_from_config.py configs/config_1V_short.ini`
4. ✅ 修改`simulation_config.ini`为你的系统
5. ✅ 运行你的模拟

### 中级用户

1. ✅ 理解配置文件的所有section（阅读`CONFIG_FILE_GUIDE.md`）
2. ✅ 学习批量运行（使用`voltage_scan.sh`）
3. ✅ 创建你自己的配置文件模板

### 从Original转换

1. ✅ 阅读`USAGE_COMPARISON.md`了解差异
2. ✅ 按照`HOW_TO_USE_PLUGIN.md`的checklist转换
3. ✅ 验证转换结果（对比能量、轨迹）

---

## 🎓 最佳实践

### 1. 保留多个配置文件

```bash
# 为不同项目/参数创建不同配置
production.ini
test_short.ini
voltage_scan_base.ini
my_experiment_2025.ini
```

### 2. 使用描述性的文件名

```bash
# 好的命名:
2V_5ns_CUDA_mixed.ini
0V_equilibration_2ns.ini

# 不好的命名:
config1.ini
test.ini
```

### 3. 备份你的配置文件

```bash
# 配置文件很小，容易备份
cp simulation_config.ini backup_$(date +%Y%m%d).ini
```

### 4. 用Git管理配置文件

```bash
git add *.ini
git commit -m "Updated voltage to 1.5V"
```

---

## 💬 常见问题

**Q: 配置文件在哪里？**
A: 默认是`simulation_config.ini`，你也可以创建自己的。

**Q: 如何批量运行多个电压？**
A: 使用`./voltage_scan.sh`或创建多个配置文件手动运行。

**Q: 配置文件会不会破坏代码？**
A: 不会！配置文件只是参数，不会影响代码。

**Q: 如何知道参数是否正确？**
A: 运行`python3 config_parser.py your_config.ini`查看摘要。

**Q: 从Original转换难吗？**
A: 不难，阅读`USAGE_COMPARISON.md`中的对照表，一个个填入即可。

---

## 🚀 现在开始！

### 选项1: 快速测试

```bash
python3 run_from_config.py configs/config_1V_short.ini
```

10 ps的短测试，验证setup正确。

### 选项2: 运行你的模拟

```bash
# 1. 编辑配置
nano simulation_config.ini

# 2. 验证
python3 config_parser.py simulation_config.ini

# 3. 运行
python3 run_from_config.py
```

### 选项3: Voltage Scan

```bash
./voltage_scan.sh
```

自动运行多个电压。

---

## 📞 需要帮助？

1. **快速参考**: 查看`README_CONFIG_FILES.md`
2. **详细指南**: 查看`CONFIG_FILE_GUIDE.md`
3. **对比Original**: 查看`USAGE_COMPARISON.md`
4. **故障排除**: 每个文档都有故障排除section

---

**祝模拟顺利！🎉**

记住：**修改参数不需要改代码，只需要编辑配置文件！**
