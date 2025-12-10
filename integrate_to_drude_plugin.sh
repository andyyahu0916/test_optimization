#!/bin/bash
# ConstantV 整合到 Drude Plugin 脚本

set -e

SRC="/home/andy/test_optimization/openmm_core_integration"
DST="/home/andy/test_optimization/openmm-8.4.0/plugins/drude"

echo "=== 开始整合 ConstantV 到 Drude Plugin ==="

# 1. API 层
echo "1. 复制 API 文件..."
cp "$SRC/openmmapi/include/openmm/ConstantVDrudeLangevinIntegrator.h" \
   "$DST/openmmapi/include/openmm/" 2>/dev/null || echo "  ⚠️ 文件可能已存在或路径不存在"

# 检查 ConstantVKernels.h 是否存在
if [ -f "$SRC/openmmapi/include/openmm/ConstantVKernels.h" ]; then
    cp "$SRC/openmmapi/include/openmm/ConstantVKernels.h" \
       "$DST/openmmapi/include/openmm/"
    echo "  ✅ ConstantVKernels.h"
else
    echo "  ⚠️ ConstantVKernels.h 不存在，可能需要创建"
fi

# 检查 ConstantVGeometry.h
if [ -f "$SRC/openmmapi/include/openmm/internal/ConstantVGeometry.h" ]; then
    mkdir -p "$DST/openmmapi/include/openmm/internal"
    cp "$SRC/openmmapi/include/openmm/internal/ConstantVGeometry.h" \
       "$DST/openmmapi/include/openmm/internal/"
    echo "  ✅ ConstantVGeometry.h"
fi

cp "$SRC/openmmapi/src/ConstantVDrudeLangevinIntegrator.cpp" \
   "$DST/openmmapi/src/" 2>/dev/null || echo "  ⚠️ .cpp 文件可能已存在"

# 2. Reference 平台
echo "2. 复制 Reference 平台文件..."
if [ -d "$SRC/platforms/reference/include" ]; then
    cp "$SRC/platforms/reference/include/ReferenceConstantVKernels.h" \
       "$DST/platforms/reference/include/" 2>/dev/null || echo "  ⚠️ ReferenceConstantVKernels.h"
    if [ -f "$SRC/platforms/reference/include/ReferenceConstantVDrudeLangevinDynamics.h" ]; then
        cp "$SRC/platforms/reference/include/ReferenceConstantVDrudeLangevinDynamics.h" \
           "$DST/platforms/reference/include/"
        echo "  ✅ ReferenceConstantVDrudeLangevinDynamics.h"
    fi
fi

if [ -d "$SRC/platforms/reference/src" ]; then
    cp "$SRC/platforms/reference/src/ReferenceConstantVKernels.cpp" \
       "$DST/platforms/reference/src/" 2>/dev/null || echo "  ⚠️ ReferenceConstantVKernels.cpp"
    if [ -f "$SRC/platforms/reference/src/ReferenceConstantVDrudeLangevinDynamics.cpp" ]; then
        cp "$SRC/platforms/reference/src/ReferenceConstantVDrudeLangevinDynamics.cpp" \
           "$DST/platforms/reference/src/"
        echo "  ✅ ReferenceConstantVDrudeLangevinDynamics.cpp"
    fi
fi

# 3. CUDA 平台 (⚠️ 只复制 .cu，不要 .cpp)
echo "3. 复制 CUDA 平台文件（只复制 .cu）..."
if [ -f "$SRC/platforms/cuda/include/CudaConstantVKernels.h" ]; then
    cp "$SRC/platforms/cuda/include/CudaConstantVKernels.h" \
       "$DST/platforms/cuda/include/"
    echo "  ✅ CudaConstantVKernels.h"
fi

# 检查是否有 .cu 版本
if [ -f "$SRC/platforms/cuda/src/CudaConstantVKernels.cu" ]; then
    cp "$SRC/platforms/cuda/src/CudaConstantVKernels.cu" \
       "$DST/platforms/cuda/src/"
    echo "  ✅ CudaConstantVKernels.cu"
elif [ -f "$SRC/platforms/cuda/src/CudaIntegrateConstantVDrudeLangevinStepKernel.cpp" ]; then
    # 如果只有 .cpp，需要重命名为 .cu（因为包含 CUDA kernel 调用）
    echo "  ⚠️ 发现 .cpp 文件，需要检查是否应重命名为 .cu"
fi

mkdir -p "$DST/platforms/cuda/src/kernels"
if [ -f "$SRC/platforms/cuda/src/kernels/constantVDrudeLangevin.cu" ]; then
    cp "$SRC/platforms/cuda/src/kernels/constantVDrudeLangevin.cu" \
       "$DST/platforms/cuda/src/kernels/"
    echo "  ✅ constantVDrudeLangevin.cu"
fi

echo ""
echo "=== 文件复制完成 ==="
echo "下一步需要："
echo "  1. 修改 plugins/drude/platforms/cuda/src/CudaDrudeKernelFactory.cpp"
echo "  2. 修改 plugins/drude/platforms/reference/src/ReferenceDrudeKernelFactory.cpp"
echo "  3. 检查 CMakeLists.txt"
echo "  4. 更新 SWIG 绑定"
