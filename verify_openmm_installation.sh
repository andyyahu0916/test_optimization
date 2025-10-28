#!/bin/bash
# OpenMM 安裝驗證腳本

echo "======================================"
echo "OpenMM 安裝驗證腳本"
echo "======================================"
echo ""

INSTALL_PREFIX="/home/andy/miniforge3/envs/cuda"

echo "1. 檢查 OpenMM 共享庫..."
echo "--------------------------------------"
if ls ${INSTALL_PREFIX}/lib/libOpenMM*.so 1> /dev/null 2>&1; then
    echo "✓ 找到 OpenMM 共享庫:"
    ls -lh ${INSTALL_PREFIX}/lib/libOpenMM*.so | head -5
    echo "   ... (可能還有更多)"
else
    echo "✗ 未找到 OpenMM 共享庫"
fi
echo ""

echo "2. 檢查 OpenMM 頭文件..."
echo "--------------------------------------"
if [ -d "${INSTALL_PREFIX}/include/openmm" ]; then
    echo "✓ 找到 OpenMM 頭文件目錄:"
    echo "   ${INSTALL_PREFIX}/include/openmm"
    echo "   包含 $(ls ${INSTALL_PREFIX}/include/openmm/*.h 2>/dev/null | wc -l) 個頭文件"
else
    echo "✗ 未找到 OpenMM 頭文件目錄"
fi
echo ""

echo "3. 檢查 Python openmm 模塊..."
echo "--------------------------------------"
if python -c "import openmm; print('OpenMM version:', openmm.version.version)" 2>/dev/null; then
    echo "✓ Python 模塊安裝成功"
    python -c "import openmm; print('   安裝路徑:', openmm.__file__)"
else
    echo "✗ Python 模塊導入失敗"
fi
echo ""

echo "4. 檢查可用的 Platform..."
echo "--------------------------------------"
if python -c "import openmm; print('可用平台:', ', '.join([openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())]))" 2>/dev/null; then
    echo "✓ Platform 檢測成功"
else
    echo "✗ Platform 檢測失敗"
fi
echo ""

echo "5. 運行 OpenMM 安裝測試..."
echo "--------------------------------------"
if python -m openmm.testInstallation 2>&1 | tail -10; then
    echo "✓ 安裝測試執行完畢(請檢查上面輸出)"
else
    echo "✗ 安裝測試失敗"
fi
echo ""

echo "6. 檢查環境變量設置..."
echo "--------------------------------------"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
if echo "$LD_LIBRARY_PATH" | grep -q "${INSTALL_PREFIX}/lib"; then
    echo "✓ LD_LIBRARY_PATH 包含 OpenMM 庫路徑"
else
    echo "⚠ 建議將 OpenMM 庫路徑添加到 LD_LIBRARY_PATH:"
    echo "   export LD_LIBRARY_PATH=${INSTALL_PREFIX}/lib:\$LD_LIBRARY_PATH"
fi
echo ""

echo "7. 檢查 CMake 配置 (可選)..."
echo "--------------------------------------"
if [ -d "${INSTALL_PREFIX}/lib/cmake" ]; then
    echo "✓ 找到 cmake 目錄:"
    ls -d ${INSTALL_PREFIX}/lib/cmake/*/ | head -5
else
    echo "⚠ cmake 目錄不存在 (OpenMM 不提供標準 CMake config)"
fi
echo ""

echo "======================================"
echo "驗證完成!"
echo "======================================"
