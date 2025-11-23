# ElectrodeChargePlugin 編譯指南

## 🔧 環境要求

### **已確認的環境**
```bash
Python: 3.13
OpenMM: 8.3.1.dev-32603cc
Conda環境: cuda
OpenMM位置: /home/andy/miniforge3/envs/cuda
```

### **需要的工具**
- CMake >= 3.17
- C++ compiler (g++ >= 7.0)
- CUDA Toolkit (如果編譯 CUDA platform)
- SWIG (如果編譯 Python bindings)

---

## 📋 編譯前準備

### **Step 1: 激活 Conda 環境**

```bash
conda activate cuda
```

### **Step 2: 設置環境變量**

```bash
# OpenMM 安裝路徑
export OPENMM_DIR=/home/andy/miniforge3/envs/cuda

# 添加 OpenMM 庫路徑到 LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$OPENMM_DIR/lib:$LD_LIBRARY_PATH

# 驗證 OpenMM 可用
python3 -c "import openmm; print('OpenMM version:', openmm.version.version)"
```

### **Step 3: 檢查 CMake**

```bash
cmake --version  # 需要 >= 3.17
```

如果沒有或版本太舊：
```bash
conda install cmake
```

---

## 🏗️ 編譯 Plugin

### **完整編譯流程**

```bash
cd /home/andy/test_optimization/plugins/ElectrodeChargePlugin

# 清理舊的 build
rm -rf build
mkdir build
cd build

# 配置 CMake
cmake .. \
    -DOPENMM_DIR=$OPENMM_DIR \
    -DCMAKE_INSTALL_PREFIX=$OPENMM_DIR \
    -DCMAKE_BUILD_TYPE=Release

# 編譯
make -j$(nproc)

# 安裝（會安裝到 OpenMM 目錄）
make install

# 測試 Python binding
make PythonInstall
```

### **分步驟編譯（用於 debug）**

#### **1. 只編譯 API 層和 Reference platform**

```bash
cmake .. \
    -DOPENMM_DIR=$OPENMM_DIR \
    -DCMAKE_INSTALL_PREFIX=$OPENMM_DIR \
    -DCMAKE_BUILD_TYPE=Debug

make OpenMMElectrodeChargePlugin -j$(nproc)
make OpenMMElectrodeChargePluginReference -j$(nproc)
```

#### **2. 編譯 CUDA platform（如果 CUDA 可用）**

```bash
# CMake 會自動檢測 CUDA
# 如果找到 CUDA，會自動編譯 CUDA platform
make OpenMMElectrodeChargePluginCUDA -j$(nproc)
```

#### **3. 編譯 Python bindings**

```bash
cd python
make -j$(nproc)
make install
```

---

## 🧪 測試 Plugin

### **Step 1: 驗證 Plugin 載入**

```python
#!/usr/bin/env python3
import sys
import ctypes

# 載入 Plugin 庫
plugin_dir = "/home/andy/test_optimization/plugins/ElectrodeChargePlugin/build"
ctypes.CDLL(f"{plugin_dir}/libElectrodeChargePlugin.so", mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL(f"{plugin_dir}/platforms/reference/libElectrodeChargePluginReference.so", mode=ctypes.RTLD_GLOBAL)

# 導入 Python binding
sys.path.insert(0, f"{plugin_dir}/python")
import electrodecharge

print("✅ Plugin loaded successfully!")
print("ElectrodeChargeForce available:", hasattr(electrodecharge, "ElectrodeChargeForce"))
```

### **Step 2: 運行單元測試**

```bash
cd /home/andy/test_optimization/plugins/ElectrodeChargePlugin
python tests/test_reference_kernel.py
```

### **Step 3: 整合測試**

```python
#!/usr/bin/env python3
from openmm import *
from openmm.app import *
from openmm.unit import *
import electrodecharge

# 創建簡單系統
system = System()
for i in range(5):
    system.addParticle(40.0)

# 添加 NonbondedForce
nonbonded = NonbondedForce()
for i in range(5):
    charge = 0.1 if i == 4 else 0.0
    nonbonded.addParticle(charge, 1.0, 0.0)
system.addForce(nonbonded)

# 添加 ElectrodeChargeForce
electrode_force = electrodecharge.ElectrodeChargeForce()
electrode_force.setCathode([0, 1], 1.0)  # indices, voltage
electrode_force.setAnode([2, 3], 1.0)
electrode_force.setNumIterations(3)
electrode_force.setSmallThreshold(1e-6)
electrode_force.setCellGap(0.8)
electrode_force.setCellLength(2.0)
system.addForce(electrode_force)

# 創建 Context
integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.001*picoseconds)
context = Context(system, integrator)

# 設置初始狀態
positions = [Vec3(0, 0, 0.1), Vec3(0.8, 0, 0.1), Vec3(0, 0, 1.9), Vec3(0.8, 0, 1.9), Vec3(0.4, 0.4, 1.0)]
context.setPositions(positions)
context.setPeriodicBoxVectors(Vec3(1.6, 0, 0), Vec3(0, 1.6, 0), Vec3(0, 0, 2.0))

# 觸發 Plugin
state = context.getState(getForces=True)
print("✅ Plugin executed successfully!")

# 檢查電荷是否更新
charges = []
for i in range(5):
    charge, sigma, epsilon = nonbonded.getParticleParameters(i)
    charges.append(float(charge))
print(f"Updated charges: {charges}")
```

---

## 🐛 常見問題排除

### **問題 1: CMake 找不到 OpenMM**

```
CMake Error: Could not find OpenMM
```

**解決方案：**
```bash
# 確認 OpenMM 路徑
python3 -c "import openmm; import os; print(os.path.dirname(os.path.dirname(openmm.__file__)))"

# 設置 OPENMM_DIR
export OPENMM_DIR=/home/andy/miniforge3/envs/cuda
cmake .. -DOPENMM_DIR=$OPENMM_DIR
```

### **問題 2: 找不到 OpenMM headers**

```
fatal error: openmm/Force.h: No such file or directory
```

**解決方案：**
```bash
# 檢查 headers 是否存在
ls $OPENMM_DIR/include/openmm/Force.h

# 如果不存在，可能需要安裝 dev headers
conda install openmm-dev
```

### **問題 3: SWIG 錯誤**

```
Error: SWIG not found
```

**解決方案：**
```bash
conda install swig
```

### **問題 4: 運行時找不到 .so 文件**

```
ImportError: libOpenMMElectrodeChargePlugin.so: cannot open shared object file
```

**解決方案：**
```bash
# 添加到 LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$OPENMM_DIR/lib:$LD_LIBRARY_PATH

# 或者在 Python 中設置
import os
os.environ['LD_LIBRARY_PATH'] = f"{os.environ['OPENMM_DIR']}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"
```

### **問題 5: Python 找不到 electrodecharge 模塊**

```
ModuleNotFoundError: No module named 'electrodecharge'
```

**解決方案：**
```bash
# 確認 Python binding 已編譯
ls build/python/electrodecharge*.so

# 添加到 Python path
export PYTHONPATH=$PWD/build/python:$PYTHONPATH

# 或在 Python 中
import sys
sys.path.insert(0, "/path/to/build/python")
```

---

## 📊 編譯驗證清單

編譯完成後，檢查以下文件是否存在：

```bash
cd /home/andy/test_optimization/plugins/ElectrodeChargePlugin/build

# 核心庫
✅ libElectrodeChargePlugin.so

# Reference platform
✅ platforms/reference/libElectrodeChargePluginReference.so

# CUDA platform (如果編譯了)
✅ platforms/cuda/libElectrodeChargePluginCUDA.so

# Python binding
✅ python/electrodecharge.py
✅ python/_electrodecharge.so  (或 .cpython-*.so)
```

---

## 🚀 下一步：修復代碼

編譯成功後，我們需要修復代碼架構（參考 `LINUS_PLUGIN_IMPLEMENTATION.md`）：

1. 修改 `ElectrodeChargeForceImpl` 使用 `updateContextState`
2. 簡化 kernel 接口
3. 實現正確的 Reference kernel
4. 實現 CUDA kernel（可選）

準備好了嗎？
