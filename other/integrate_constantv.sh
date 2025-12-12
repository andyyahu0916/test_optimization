#!/bin/bash
# ConstantV 整合到 OpenMM 核心源码树脚本

set -e

OPENMM_ROOT="/home/andy/test_optimization/openmm-8.4.0"
CONSTANTV_SRC="/home/andy/test_optimization/openmm_core_integration"

echo "=== 开始整合 ConstantV 到 OpenMM 核心 ==="

# 1. 复制 API 头文件
echo "1. 复制 API 头文件..."
cp "$CONSTANTV_SRC/openmmapi/include/openmm/ConstantVDrudeLangevinIntegrator.h" \
   "$OPENMM_ROOT/openmmapi/include/openmm/ConstantVDrudeLangevinIntegrator.h"

# 2. 复制 API 实现文件
echo "2. 复制 API 实现文件..."
cp "$CONSTANTV_SRC/openmmapi/src/ConstantVDrudeLangevinIntegrator.cpp" \
   "$OPENMM_ROOT/openmmapi/src/ConstantVDrudeLangevinIntegrator.cpp"

# 3. 复制 CUDA Kernel 源文件
echo "3. 复制 CUDA Kernel 源文件..."
mkdir -p "$OPENMM_ROOT/platforms/cuda/src/kernels"
cp "$CONSTANTV_SRC/platforms/cuda/src/kernels/constantVDrudeLangevin.cu" \
   "$OPENMM_ROOT/platforms/cuda/src/kernels/constantVDrudeLangevin.cu"

# 4. 复制 CUDA Kernel 包装
echo "4. 复制 CUDA Kernel 包装..."
cp "$CONSTANTV_SRC/platforms/cuda/src/CudaConstantVKernels.cpp" \
   "$OPENMM_ROOT/platforms/cuda/src/CudaIntegrateConstantVDrudeLangevinStepKernel.cpp"

# 5. 复制 CUDA 头文件
echo "5. 复制 CUDA 头文件..."
mkdir -p "$OPENMM_ROOT/platforms/cuda/include"
cp "$CONSTANTV_SRC/platforms/cuda/include/CudaConstantVKernels.h" \
   "$OPENMM_ROOT/platforms/cuda/include/CudaIntegrateConstantVDrudeLangevinStepKernel.h"

# 6. 复制 Reference 实现
echo "6. 复制 Reference 实现..."
if [ -f "$CONSTANTV_SRC/platforms/reference/src/ReferenceConstantVKernels.cpp" ]; then
    cp "$CONSTANTV_SRC/platforms/reference/src/ReferenceConstantVKernels.cpp" \
       "$OPENMM_ROOT/platforms/reference/src/ReferenceIntegrateConstantVDrudeLangevinStepKernel.cpp"
fi

# 7. 复制 Reference 头文件
echo "7. 复制 Reference 头文件..."
if [ -f "$CONSTANTV_SRC/platforms/reference/include/ReferenceConstantVKernels.h" ]; then
    mkdir -p "$OPENMM_ROOT/platforms/reference/include"
    cp "$CONSTANTV_SRC/platforms/reference/include/ReferenceConstantVKernels.h" \
       "$OPENMM_ROOT/platforms/reference/include/ReferenceIntegrateConstantVDrudeLangevinStepKernel.h"
fi

echo "=== 文件复制完成 ==="
echo "下一步: 需要手动更新以下文件:"
echo "  1. openmmapi/include/openmm/kernels.h - 添加 Kernel 接口"
echo "  2. platforms/cuda/src/CudaPlatform.cpp - 注册 Kernel"
echo "  3. platforms/cuda/src/CudaKernelFactory.cpp - 创建 Kernel 实例"
echo "  4. platforms/reference/src/ReferencePlatform.cpp - 注册 Kernel"
echo "  5. platforms/reference/src/ReferenceKernelFactory.cpp - 创建 Kernel 实例"
echo "  6. CMakeLists.txt - 确保源文件被包含"
