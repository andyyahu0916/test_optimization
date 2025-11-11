# ConstantV Plugin - Constant Voltage Molecular Dynamics for OpenMM

**基于教授的穩健算法，完全照抄Original實現**

## 🚀 快速开始

**新用户从这里开始**:

```bash
# 1. 阅读文档
cat docs/user_guides/START_HERE.md

# 2. 编辑配置
nano simulation_config.ini

# 3. 运行模拟
python3 run_from_config.py
```

或者运行快速测试:
```bash
python3 run_from_config.py configs/config_1V_short.ini
```

## 📚 文档

### 用户指南 (`docs/user_guides/`)

1. **`START_HERE.md`** ⭐ - 从这里开始
2. **`QUICK_START.md`** - 快速开始指南
3. **`CONFIG_FILE_GUIDE.md`** - 配置文件完整参考
4. **`HOW_TO_USE_PLUGIN.md`** - 从Original迁移指南
5. **`USAGE_COMPARISON.md`** - 与Original的详细对比

### 技术文档 (`docs/technical_references/`)

- **`README_USAGE.md`** - 完整API文档

## 📂 项目结构

```
openMM_constantV_plugin/
├── README.md                    ← 你在这里
├── simulation_config.ini        ← 默认配置文件
├── run_from_config.py          ← 主运行脚本
├── config_parser.py            ← 配置解析器
├── voltage_scan.sh             ← 电压扫描脚本
│
├── ConstantVPlugin/            ← C++插件源代码
│   ├── openmmapi/
│   ├── platforms/
│   └── python/
│
├── configs/                    ← 示例配置
│   ├── config_1V_short.ini    ← 快速测试
│   ├── config_2V_long.ini     ← 生产运行
│   └── config_CPU_debug.ini   ← CPU调试
│
├── docs/                       ← 📖 文档
│   ├── user_guides/           ← 用户指南
│   └── technical_references/  ← 技术文档
│
├── examples/                   ← 示例和替代实现
│
└── archive/                    ← 🗄️ 历史归档
    ├── failed_algorithms/      ← ❌ 失败的算法尝试
    ├── formula_verification_breakthrough/ ← 🔬 转折点！
    ├── early_development/      ← 📁 早期开发
    ├── mid_development/        ← 📁 中期开发
    ├── successful_implementation_docs/ ← ✅ 成功实现
    ├── cuda_development/       ← 🎮 CUDA开发
    ├── exclusions_development/ ← 🔒 Exclusions开发
    └── deprecated_tests/       ← 🗑️ 已废弃测试
```

## 🎯 核心理念

**完全照抄教授的Original算法，不要自己乱搞！**

### 开发历程

1. **11月4日**: ❌ 尝试自创算法（电容矩阵等）→ 失败
2. **11月6日**: ❌ 各种尝试 → 继续失败
3. **11月11日凌晨**: 🔬 **转折点！** 开始逐行验证教授的公式
4. **11月11日晚上**: ✅ 实现完全照抄教授的版本 → 成功！
5. **11月12日**: ✅ 添加配置文件系统，完善用户体验

### 为什么成功？

**因为完全照抄，不优化！**

- ✅ 每个公式都有Original代码的行号引用
- ✅ 所有常数完全一致（`2.0`, `0.9`, `1e-6`等）
- ✅ 执行顺序完全一致（SCF → MD step）
- ✅ 所有逻辑完全一致（`0.9*threshold`, not `1.0*threshold`）

详见: `archive/formula_verification_breakthrough/`

## ✨ 特点

- ✅ **100%复现** Original Python的物理算法
- ✅ **配置文件驱动** - 修改参数不需要改代码
- ✅ **多平台支持** - CUDA (GPU), OpenCL, CPU, Reference
- ✅ **自动化exclusions** - helper函数处理电极exclusions
- ✅ **批量运行** - voltage scan脚本用于参数扫描

## ⚠️ 关键步骤

使用plugin时，**必须**执行以下关键步骤：

1. **添加exclusions** (创建context之前):
   ```python
   add_electrode_exclusions(integrator, nonbonded_force, custom_nonbonded_force)
   ```

2. **Reinitialize** (创建context之后):
   ```python
   context = mm.Context(system, integrator, platform)
   context.reinitialize(preserveState=True)  # ← 必须！
   ```

详见: `docs/user_guides/START_HERE.md`

## 🔧 安装

```bash
cd ConstantVPlugin/build
cmake ..
make
sudo make install
make PythonInstall
```

详见: `ConstantVPlugin/README.md`

## 📖 快速参考

### 修改参数

编辑 `simulation_config.ini`:

```ini
[Electrodes]
voltage = 1.0              # 电压

[Simulation]
total_time_ns = 2.0        # 模拟时间

[Output]
output_dir = my_output     # 输出目录
```

### 运行模拟

```bash
python3 run_from_config.py
```

### 验证配置

```bash
python3 config_parser.py simulation_config.ini
```

### Voltage扫描

```bash
./voltage_scan.sh
```

## 🗄️ 历史归档

`archive/`目录包含开发过程中的所有文档，按时间线分类：

- **`failed_algorithms/`** - 失败的算法尝试（电容矩阵等）
- **`formula_verification_breakthrough/`** - 🔬 **转折点！** 公式验证文档
- **`successful_implementation_docs/`** - 基于正确公式的实现

详见: `archive/` 下各子目录的README.md

## 🙏 致谢

基于教授团队用于期刊发表的Original Python实现。

---

**开始使用**: [`docs/user_guides/START_HERE.md`](docs/user_guides/START_HERE.md)
