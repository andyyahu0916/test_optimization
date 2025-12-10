# ConstantV 整合到 OpenMM 源码树 - 执行计划

## 目标
将 ConstantVDrudeLangevinIntegrator 作为 Drude Plugin 的扩展整合到 OpenMM 8.4.0。

## ⚠️ 重要前提

1. **Drude 依赖**: `ConstantVDrudeLangevinIntegrator` 继承自 `DrudeLangevinIntegrator`
2. **推荐方案**: 将代码整合到 `plugins/drude/` 目录（避免依赖问题）
3. **文件重复**: `CudaConstantVKernels.cpp` 和 `.cu` 重复，只保留 `.cu` 版本

---

## 整合步骤

### 1. API 层整合 (`plugins/drude/openmmapi/`)

#### 1.1 复制头文件
| 源文件 | 目标位置 |
|--------|---------|
| `openmmapi/include/openmm/ConstantVDrudeLangevinIntegrator.h` | `plugins/drude/openmmapi/include/openmm/` |
| `openmmapi/include/openmm/ConstantVKernels.h` | `plugins/drude/openmmapi/include/openmm/` |
| `openmmapi/include/openmm/internal/ConstantVGeometry.h` | `plugins/drude/openmmapi/include/openmm/internal/` |

#### 1.2 复制实现文件
| 源文件 | 目标位置 |
|--------|---------|
| `openmmapi/src/ConstantVDrudeLangevinIntegrator.cpp` | `plugins/drude/openmmapi/src/` |

#### 1.3 Kernel 接口定义（在 ConstantVKernels.h 中，已包含）
- **不要**修改 OpenMM 核心的 `kernels.h`
- Kernel 定义保留在 `ConstantVKernels.h` 中

### 2. CUDA 平台实现 (`plugins/drude/platforms/cuda/`)

#### 2.1 复制 CUDA Kernel 实现
| 源文件 | 目标位置 |
|--------|---------|
| `platforms/cuda/src/kernels/constantVDrudeLangevin.cu` | `plugins/drude/platforms/cuda/src/kernels/` |

#### 2.2 复制 CUDA Kernel 包装
| 源文件 | 目标位置 | 备注 |
|--------|---------|------|
| `platforms/cuda/src/CudaConstantVKernels.cu` | `plugins/drude/platforms/cuda/src/` | ⚠️ **只复制 .cu，不要复制 .cpp** |
| `platforms/cuda/include/CudaConstantVKernels.h` | `plugins/drude/platforms/cuda/include/` | |

#### 2.3 注册 Kernel
在 `plugins/drude/platforms/cuda/src/CudaDrudeKernelFactory.cpp` 中添加:
```cpp
#include "CudaConstantVKernels.h"

// 在 createKernelImpl() 中添加:
if (name == "IntegrateConstantVDrudeLangevinStep")
    return new CudaIntegrateConstantVDrudeLangevinStepKernel(name, platform, cu);
```

#### 2.4 更新 CMakeLists.txt
在 `plugins/drude/platforms/cuda/CMakeLists.txt` 中:
- 添加 `src/CudaConstantVKernels.cu` 到源文件列表
- 添加 `src/kernels/constantVDrudeLangevin.cu` 到 kernel 文件列表

### 3. Reference 平台实现 (`plugins/drude/platforms/reference/`)

#### 3.1 复制头文件
| 源文件 | 目标位置 |
|--------|---------|
| `platforms/reference/include/ReferenceConstantVKernels.h` | `plugins/drude/platforms/reference/include/` |
| `platforms/reference/include/ReferenceConstantVDrudeLangevinDynamics.h` | `plugins/drude/platforms/reference/include/` |

#### 3.2 复制实现文件
| 源文件 | 目标位置 |
|--------|---------|
| `platforms/reference/src/ReferenceConstantVKernels.cpp` | `plugins/drude/platforms/reference/src/` |
| `platforms/reference/src/ReferenceConstantVDrudeLangevinDynamics.cpp` | `plugins/drude/platforms/reference/src/` |

#### 3.3 注册 Kernel
在 `plugins/drude/platforms/reference/src/ReferenceDrudeKernelFactory.cpp` 中添加:
```cpp
#include "ReferenceConstantVKernels.h"

// 在 createKernelImpl() 中添加:
if (name == "IntegrateConstantVDrudeLangevinStep")
    return new ReferenceIntegrateConstantVDrudeLangevinStepKernel(name, platform);
```

#### 3.4 更新 CMakeLists.txt
在 `plugins/drude/platforms/reference/CMakeLists.txt` 中添加新源文件

### 4. 更新 CMakeLists.txt

#### 4.1 Drude Plugin CMakeLists.txt
文件自动通过 `FILE(GLOB ...)` 收集源文件，无需手动修改（如果遵循正确的目录结构）

#### 4.2 CUDA 平台 CMakeLists.txt (`plugins/drude/platforms/cuda/CMakeLists.txt`)
检查是否使用 GLOB，如果是手动列表则添加：
```cmake
SET(CUDA_DRUDE_SOURCE_FILES
    # ... existing files ...
    src/CudaConstantVKernels.cu
)
```

#### 4.3 Reference 平台 CMakeLists.txt (`plugins/drude/platforms/reference/CMakeLists.txt`)
同上，添加新源文件

### 5. Python 绑定 (SWIG)

#### 5.1 更新 Drude Plugin SWIG 文件
在 `plugins/drude/wrappers/drudePluginWrapper.i` 中添加:
```swig
%include "openmm/ConstantVDrudeLangevinIntegrator.h"

namespace OpenMM {
    %extend ConstantVDrudeLangevinIntegrator {
        // Python-specific methods if needed
    }
}
```

#### 5.2 或创建独立 SWIG 文件
创建 `plugins/drude/wrappers/constantVWrapper.i`（如果需要单独的模块）

### 6. 依赖关系

#### 6.1 Drude 插件内部依赖
✅ 由于代码整合到 Drude 插件内部，依赖问题自动解决

#### 6.2 头文件包含路径
需要检查并更新以下包含路径:
```cpp
// 原来的路径
#include "openmm/ConstantVDrudeLangevinIntegrator.h"

// 可能需要改为（取决于目录结构）
#include "openmm/ConstantVDrudeLangevinIntegrator.h"  // 通常不需要改
```

### 7. 测试

#### 7.1 单元测试
- 复制测试文件到 `plugins/drude/tests/`
- 更新测试的包含路径

#### 7.2 编译验证
```bash
cd openmm-8.4.0/build
cmake .. -DOPENMM_BUILD_CUDA_LIB=ON -DOPENMM_BUILD_DRUDE_CUDA_LIB=ON
make -j8
```

#### 7.3 运行测试
```bash
ctest -R ConstantV
```

---

## 📋 完整文件清单

### 必须复制的文件

| 类别 | 源文件 | 目标目录 |
|------|--------|---------|
| **API Header** | `openmmapi/include/openmm/ConstantVDrudeLangevinIntegrator.h` | `plugins/drude/openmmapi/include/openmm/` |
| **API Header** | `openmmapi/include/openmm/ConstantVKernels.h` | `plugins/drude/openmmapi/include/openmm/` |
| **API Header** | `openmmapi/include/openmm/internal/ConstantVGeometry.h` | `plugins/drude/openmmapi/include/openmm/internal/` |
| **API Impl** | `openmmapi/src/ConstantVDrudeLangevinIntegrator.cpp` | `plugins/drude/openmmapi/src/` |
| **Ref Header** | `platforms/reference/include/ReferenceConstantVKernels.h` | `plugins/drude/platforms/reference/include/` |
| **Ref Header** | `platforms/reference/include/ReferenceConstantVDrudeLangevinDynamics.h` | `plugins/drude/platforms/reference/include/` |
| **Ref Impl** | `platforms/reference/src/ReferenceConstantVKernels.cpp` | `plugins/drude/platforms/reference/src/` |
| **Ref Impl** | `platforms/reference/src/ReferenceConstantVDrudeLangevinDynamics.cpp` | `plugins/drude/platforms/reference/src/` |
| **CUDA Header** | `platforms/cuda/include/CudaConstantVKernels.h` | `plugins/drude/platforms/cuda/include/` |
| **CUDA Impl** | `platforms/cuda/src/CudaConstantVKernels.cu` | `plugins/drude/platforms/cuda/src/` |
| **CUDA Kernel** | `platforms/cuda/src/kernels/constantVDrudeLangevin.cu` | `plugins/drude/platforms/cuda/src/kernels/` |

### ⚠️ 不要复制的文件

| 文件 | 原因 |
|------|------|
| `CudaConstantVKernels.cpp` | 与 `.cu` 重复，会导致链接错误 |
| `ConstantVKernelFactory.cpp` | 功能已整合到 Drude 的 KernelFactory |
| `registerConstantV.cpp` | 功能已整合到 Drude 的注册逻辑 |
| `kernel_compiler.py` | JIT 优化工具，非必需 |

---

## 注意事项

1. **⚠️ 不要复制 .cpp 版本的 CudaConstantVKernels**: 只保留 `.cu` 版本
2. **命名空间**: 确保所有代码在 `OpenMM` 命名空间中
3. **包含路径**: 整合后可能需要调整 `#include` 路径
4. **Kernel 注册**: 在 Drude 的 KernelFactory 中添加创建逻辑
5. **SWIG 绑定**: 添加到 Drude 的 SWIG 文件或创建新文件

---

## 执行顺序（带检查点）

1. ✅ 创建整合计划
2. ⏳ 复制 API 文件到 `plugins/drude/openmmapi/`
3. ⏳ 复制 Reference 平台文件
4. ⏳ 复制 CUDA 平台文件（**只复制 .cu，不要 .cpp**）
5. ⏳ 修改 Drude KernelFactory 注册 ConstantV kernels
6. ⏳ 检查 CMakeLists.txt（确认 GLOB 会自动收集新文件）
7. ⏳ 更新 SWIG 绑定
8. ⏳ 编译测试：`cmake .. && make -j8`
9. ⏳ 修复编译错误（预期会有包含路径问题）
10. ⏳ 运行单元测试

---

## 快速复制脚本

```bash
#!/bin/bash
SRC=/home/andy/test_optimization/openmm_core_integration
DST=/home/andy/test_optimization/openmm-8.4.0/plugins/drude

# API 层
cp $SRC/openmmapi/include/openmm/ConstantVDrudeLangevinIntegrator.h $DST/openmmapi/include/openmm/
cp $SRC/openmmapi/include/openmm/ConstantVKernels.h $DST/openmmapi/include/openmm/
mkdir -p $DST/openmmapi/include/openmm/internal
cp $SRC/openmmapi/include/openmm/internal/ConstantVGeometry.h $DST/openmmapi/include/openmm/internal/
cp $SRC/openmmapi/src/ConstantVDrudeLangevinIntegrator.cpp $DST/openmmapi/src/

# Reference 平台
cp $SRC/platforms/reference/include/*.h $DST/platforms/reference/include/
cp $SRC/platforms/reference/src/*.cpp $DST/platforms/reference/src/

# CUDA 平台 (⚠️ 只复制 .cu，不要 .cpp)
cp $SRC/platforms/cuda/include/CudaConstantVKernels.h $DST/platforms/cuda/include/
cp $SRC/platforms/cuda/src/CudaConstantVKernels.cu $DST/platforms/cuda/src/
mkdir -p $DST/platforms/cuda/src/kernels
cp $SRC/platforms/cuda/src/kernels/constantVDrudeLangevin.cu $DST/platforms/cuda/src/kernels/

echo "✅ 文件复制完成！"
echo "⚠️ 请手动修改 KernelFactory 注册 ConstantV kernels"
```

