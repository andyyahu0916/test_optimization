# ConstantV 整合到 OpenMM 核心源码树 - 执行计划

## 目标
将 ConstantVDrudeLangevinIntegrator 整合到 OpenMM 8.4.0 源码树，使其成为官方级别的核心扩展。

## 整合步骤

### 1. API 层整合 (`openmmapi/`)

#### 1.1 复制头文件
- 源: `openmm_core_integration/openmmapi/include/openmm/ConstantVDrudeLangevinIntegrator.h`
- 目标: `openmm-8.4.0/openmmapi/include/openmm/ConstantVDrudeLangevinIntegrator.h`
- 修改: 更新版权信息为 OpenMM 标准格式

#### 1.2 复制实现文件
- 源: `openmm_core_integration/openmmapi/src/ConstantVDrudeLangevinIntegrator.cpp`
- 目标: `openmm-8.4.0/openmmapi/src/ConstantVDrudeLangevinIntegrator.cpp`
- 修改: 确保包含路径正确

#### 1.3 添加 Kernel 接口定义
- 在 `openmm-8.4.0/openmmapi/include/openmm/kernels.h` 中添加:
  ```cpp
  class IntegrateConstantVDrudeLangevinStepKernel : public KernelImpl {
  public:
      static std::string Name() { return "IntegrateConstantVDrudeLangevinStep"; }
      virtual void initialize(...) = 0;
      virtual void execute(ContextImpl& context, const ConstantVDrudeLangevinIntegrator& integrator) = 0;
  };
  ```

### 2. CUDA 平台实现 (`platforms/cuda/`)

#### 2.1 复制 CUDA Kernel 实现
- 源: `openmm_core_integration/platforms/cuda/src/kernels/constantVDrudeLangevin.cu`
- 目标: `openmm-8.4.0/platforms/cuda/src/kernels/constantVDrudeLangevin.cu`

#### 2.2 复制 CUDA Kernel 包装
- 源: `openmm_core_integration/platforms/cuda/src/CudaConstantVKernels.cpp`
- 目标: `openmm-8.4.0/platforms/cuda/src/CudaConstantVKernels.cpp`
- 重命名: 可能需要重命名为 `CudaIntegrateConstantVDrudeLangevinStepKernel.cpp`

#### 2.3 复制头文件
- 源: `openmm_core_integration/platforms/cuda/include/CudaConstantVKernels.h`
- 目标: `openmm-8.4.0/platforms/cuda/include/CudaIntegrateConstantVDrudeLangevinStepKernel.h`

#### 2.4 注册 Kernel
- 在 `openmm-8.4.0/platforms/cuda/src/CudaPlatform.cpp` 中添加:
  ```cpp
  registerKernelFactory(IntegrateConstantVDrudeLangevinStepKernel::Name(), factory);
  ```

#### 2.5 创建 Kernel 实例
- 在 `openmm-8.4.0/platforms/cuda/src/CudaKernelFactory.cpp` 中添加:
  ```cpp
  if (name == IntegrateConstantVDrudeLangevinStepKernel::Name())
      return new CudaIntegrateConstantVDrudeLangevinStepKernel(name, platform, cu);
  ```

### 3. Reference 平台实现 (`platforms/reference/`)

#### 3.1 复制 Reference 实现
- 源: `openmm_core_integration/platforms/reference/src/ReferenceConstantVKernels.cpp`
- 目标: `openmm-8.4.0/platforms/reference/src/ReferenceIntegrateConstantVDrudeLangevinStepKernel.cpp`

#### 3.2 复制头文件
- 源: `openmm_core_integration/platforms/reference/include/ReferenceConstantVKernels.h`
- 目标: `openmm-8.4.0/platforms/reference/include/ReferenceIntegrateConstantVDrudeLangevinStepKernel.h`

#### 3.3 注册 Kernel
- 在 `openmm-8.4.0/platforms/reference/src/ReferencePlatform.cpp` 中添加注册

#### 3.4 创建 Kernel 实例
- 在 `openmm-8.4.0/platforms/reference/src/ReferenceKernelFactory.cpp` 中添加创建逻辑

### 4. 更新 CMakeLists.txt

#### 4.1 主 CMakeLists.txt
- 确保 `openmmapi` 子目录包含新的源文件（自动通过 GLOB 收集）

#### 4.2 CUDA 平台 CMakeLists.txt
- 添加 `constantVDrudeLangevin.cu` 和 `CudaIntegrateConstantVDrudeLangevinStepKernel.cpp` 到编译列表

#### 4.3 Reference 平台 CMakeLists.txt
- 添加 `ReferenceIntegrateConstantVDrudeLangevinStepKernel.cpp` 到编译列表

### 5. SWIG Python 绑定

#### 5.1 复制 SWIG 接口
- 源: `openmm_core_integration/python/ConstantVPlugin.i`
- 目标: `openmm-8.4.0/wrappers/python/src/ConstantVDrudeLangevinIntegrator.i`
- 或者: 整合到现有的 `openmmapi.i` 中

#### 5.2 更新 Python 包装生成
- 确保 SWIG 能正确生成 Python 绑定

### 6. 依赖关系

#### 6.1 Drude 插件依赖
- ConstantVDrudeLangevinIntegrator 继承自 DrudeLangevinIntegrator
- 需要确保 Drude 插件已编译并可用
- 或者: 将 DrudeLangevinIntegrator 的依赖改为可选

### 7. 测试

#### 7.1 单元测试
- 复制测试文件到 `openmm-8.4.0/tests/`

#### 7.2 集成测试
- 确保与 OpenMM 测试框架兼容

## 注意事项

1. **版权信息**: 所有文件需要更新为 OpenMM 标准版权格式
2. **命名空间**: 确保所有代码在 `OpenMM` 命名空间中
3. **包含路径**: 更新所有 `#include` 路径以匹配 OpenMM 源码树结构
4. **Drude 依赖**: 需要处理 Drude 插件的依赖关系
5. **向后兼容**: 确保不影响现有 OpenMM 功能

## 执行顺序

1. ✅ 创建整合计划
2. ⏳ 复制 API 文件
3. ⏳ 复制平台实现
4. ⏳ 注册 Kernels
5. ⏳ 更新 CMakeLists.txt
6. ⏳ 更新 SWIG 绑定
7. ⏳ 测试编译
8. ⏳ 运行测试

