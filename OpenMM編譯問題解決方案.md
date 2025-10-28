# OpenMM 編譯問題診斷與解決方案

## 問題描述
按照 OpenMM 官方文檔步驟編譯後,**CMake 配置文件(如 `OpenMMConfig.cmake`)沒有生成**。

## 根本原因分析

### 1. **源目錄路徑不匹配**
你的 build 目錄中的 CMakeCache.txt 記錄的源目錄是:
```
/home/andy/test_optimization/plugins/openmm-8.3.1
```

但實際目錄名是:
```
/home/andy/test_optimization/plugins/openmm
```

這導致 `make install` 無法執行,報錯:
```
CMake Error: The source directory "/home/andy/test_optimization/plugins/openmm-8.3.1" does not exist.
```

### 2. **OpenMM 的 CMake 配置文件生成機制**
重要提醒:**OpenMM 不會自動生成標準的 `OpenMMConfig.cmake` 文件**。

OpenMM 使用較舊的 CMake 安裝方式:
- 只有執行 `make install` 後,才會將庫文件、頭文件等安裝到指定位置
- OpenMM 主要通過設置環境變量來讓其他項目找到它
- 不提供標準的 CMake package config 文件

## 完整解決方案

### 步驟 1: 清理並重新配置

```bash
# 進入 build 目錄
cd /home/andy/test_optimization/plugins/openmm/build

# 清空 build 目錄(重要!)
rm -rf *

# 重新運行 CMake 配置
cmake .. -DCMAKE_INSTALL_PREFIX=/home/andy/miniforge3/envs/cuda
```

**關鍵 CMake 選項:**
- `CMAKE_INSTALL_PREFIX`: 安裝位置(你的 conda 環境)
- `PYTHON_EXECUTABLE`: Python 解釋器路徑(通常自動檢測)
- `OPENMM_BUILD_*`: 控制要編譯的功能

### 步驟 2: 編譯

```bash
# 使用所有 CPU 核心編譯(更快)
make -j$(nproc)

# 或使用指定核心數
make -j8
```

### 步驟 3: 測試(可選但推薦)

```bash
make test
```

### 步驟 4: 安裝

```bash
# 安裝 C++ 庫和頭文件
make install

# 安裝 Python 模塊
make PythonInstall
```

### 步驟 5: 驗證安裝

```bash
# 測試 Python 模塊
python -m openmm.testInstallation

# 檢查安裝的文件
ls /home/andy/miniforge3/envs/cuda/lib/libOpenMM*
ls /home/andy/miniforge3/envs/cuda/include/openmm/
```

## 如何在其他項目中使用本地編譯的 OpenMM

### 方法 1: 環境變量(推薦)

```bash
# 設置環境變量
export OPENMM_INCLUDE_PATH=/home/andy/miniforge3/envs/cuda/include
export OPENMM_LIB_PATH=/home/andy/miniforge3/envs/cuda/lib
export LD_LIBRARY_PATH=/home/andy/miniforge3/envs/cuda/lib:$LD_LIBRARY_PATH
```

### 方法 2: CMake 手動指定

在你的自定義 plugin CMakeLists.txt 中:

```cmake
# 設置 OpenMM 路徑
set(OPENMM_DIR "/home/andy/miniforge3/envs/cuda")

# 包含 OpenMM 頭文件
include_directories("${OPENMM_DIR}/include")

# 鏈接 OpenMM 庫
link_directories("${OPENMM_DIR}/lib")
target_link_libraries(YourPlugin OpenMM)
```

### 方法 3: 創建自己的 OpenMMConfig.cmake

如果確實需要標準的 CMake package config 文件,可以創建一個:

```cmake
# OpenMMConfig.cmake
set(OpenMM_INCLUDE_DIR "/home/andy/miniforge3/envs/cuda/include")
set(OpenMM_LIBRARY_DIR "/home/andy/miniforge3/envs/cuda/lib")
set(OpenMM_LIBRARIES OpenMM)

add_library(OpenMM SHARED IMPORTED)
set_target_properties(OpenMM PROPERTIES
    IMPORTED_LOCATION "${OpenMM_LIBRARY_DIR}/libOpenMM.so"
    INTERFACE_INCLUDE_DIRECTORIES "${OpenMM_INCLUDE_DIR}"
)
```

將此文件放在:
```
/home/andy/miniforge3/envs/cuda/lib/cmake/OpenMM/OpenMMConfig.cmake
```

## 與 conda 安裝的 OpenMM 共存

如果你想保留本地編譯版本,同時使用依賴 OpenMM 的其他工具:

```bash
# 1. 安裝依賴工具(會自動安裝 conda OpenMM)
conda install -c conda-forge openmmtools

# 2. 強制刪除 conda 的 OpenMM(不刪除依賴)
conda remove --force openmm

# 3. 安裝你本地編譯的 OpenMM
cd /home/andy/test_optimization/plugins/openmm/build
make install
make PythonInstall
```

## 常見問題排查

### Q: `make install` 報錯找不到源目錄
**A:** build 目錄中的 CMakeCache.txt 記錄的源路徑錯誤。清空 build 目錄重新配置。

### Q: 找不到 `OpenMMConfig.cmake`
**A:** OpenMM 不提供此文件。使用環境變量或手動指定路徑。

### Q: Python 無法導入 openmm
**A:** 確保執行了 `make PythonInstall`,並且 Python 環境正確。

### Q: 其他程序找不到 libOpenMM.so
**A:** 設置 `LD_LIBRARY_PATH`:
```bash
export LD_LIBRARY_PATH=/home/andy/miniforge3/envs/cuda/lib:$LD_LIBRARY_PATH
```

或添加到系統配置:
```bash
echo "/home/andy/miniforge3/envs/cuda/lib" | sudo tee /etc/ld.so.conf.d/openmm.conf
sudo ldconfig
```

## 當前編譯狀態

你的編譯正在進行中。完成後需要執行:

```bash
cd /home/andy/test_optimization/plugins/openmm/build
make install
make PythonInstall
```

然後 OpenMM 的所有文件才會被安裝到 `/home/andy/miniforge3/envs/cuda` 目錄下。

## 參考資料

- [OpenMM 官方文檔 - 從源碼編譯](https://openmm.github.io/openmm-org/documentation/index.html)
- CMake find_package 文檔
- OpenMM GitHub Issues
