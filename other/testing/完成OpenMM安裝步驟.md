# 完成 OpenMM 安裝的後續步驟

## 當前狀態
- CMake 已正確配置 ✅
- 編譯進行到 95% 後被中斷
- 核心庫已經編譯完成

## 立即執行的步驟

### 1. 完成編譯

```bash
cd /home/andy/test_optimization/plugins/openmm/build
make -j$(nproc)
```

如果遇到問題或只想編譯核心部分:
```bash
make -j$(nproc) OpenMM
```

### 2. 安裝到 conda 環境

```bash
cd /home/andy/test_optimization/plugins/openmm/build

# 安裝 C++ 庫和頭文件
make install

# 安裝 Python 模塊
make PythonInstall
```

### 3. 驗證安裝

```bash
# 運行驗證腳本
/home/andy/test_optimization/verify_openmm_installation.sh

# 或手動測試
python -m openmm.testInstallation
```

## 關於 OpenMMConfig.cmake 的重要說明

### ⚠️ OpenMM 不提供標準的 CMake Config 文件

OpenMM **沒有** `OpenMMConfig.cmake` 或 `openmm-config.cmake`!這不是安裝錯誤,而是 OpenMM 的設計方式。

### 在你的自定義 Plugin 中使用 OpenMM

#### 方式 1: 直接指定路徑 (推薦)

在你的 `ElectrodeChargePlugin/CMakeLists.txt` 中:

```cmake
# 設置 OpenMM 安裝路徑
set(OPENMM_DIR "/home/andy/miniforge3/envs/cuda")

# 包含 OpenMM 頭文件
include_directories("${OPENMM_DIR}/include")

# 鏈接 OpenMM 庫
link_directories("${OPENMM_DIR}/lib")

# 在你的 target 中鏈接
target_link_libraries(YourPluginLibrary OpenMM)
```

#### 方式 2: 使用 find_library

```cmake
set(OPENMM_DIR "/home/andy/miniforge3/envs/cuda")

# 尋找 OpenMM 庫
find_library(OPENMM_LIBRARY OpenMM HINTS "${OPENMM_DIR}/lib")
find_path(OPENMM_INCLUDE_DIR OpenMM.h HINTS "${OPENMM_DIR}/include")

if(NOT OPENMM_LIBRARY OR NOT OPENMM_INCLUDE_DIR)
    message(FATAL_ERROR "找不到 OpenMM! 請檢查 OPENMM_DIR 設置")
endif()

include_directories(${OPENMM_INCLUDE_DIR})
target_link_libraries(YourPluginLibrary ${OPENMM_LIBRARY})
```

#### 方式 3: 創建自定義的 FindOpenMM.cmake

創建文件 `/home/andy/test_optimization/plugins/ElectrodeChargePlugin/cmake/FindOpenMM.cmake`:

```cmake
# FindOpenMM.cmake
# 查找 OpenMM 安裝

set(OPENMM_ROOT_DIR "/home/andy/miniforge3/envs/cuda" CACHE PATH "OpenMM 安裝根目錄")

# 查找頭文件
find_path(OPENMM_INCLUDE_DIR 
    NAMES OpenMM.h
    PATHS ${OPENMM_ROOT_DIR}/include
    DOC "OpenMM 頭文件目錄"
)

# 查找庫文件
find_library(OPENMM_LIBRARY 
    NAMES OpenMM
    PATHS ${OPENMM_ROOT_DIR}/lib
    DOC "OpenMM 庫文件"
)

# 標準的 CMake 查找模塊結尾
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenMM
    DEFAULT_MSG
    OPENMM_LIBRARY
    OPENMM_INCLUDE_DIR
)

if(OpenMM_FOUND)
    set(OPENMM_LIBRARIES ${OPENMM_LIBRARY})
    set(OPENMM_INCLUDE_DIRS ${OPENMM_INCLUDE_DIR})
    
    # 創建 imported target
    if(NOT TARGET OpenMM::OpenMM)
        add_library(OpenMM::OpenMM SHARED IMPORTED)
        set_target_properties(OpenMM::OpenMM PROPERTIES
            IMPORTED_LOCATION "${OPENMM_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${OPENMM_INCLUDE_DIR}"
        )
    endif()
endif()

mark_as_advanced(OPENMM_INCLUDE_DIR OPENMM_LIBRARY)
```

然後在你的主 CMakeLists.txt 中:

```cmake
# 添加 cmake 模塊搜索路徑
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# 查找 OpenMM
find_package(OpenMM REQUIRED)

# 使用
target_link_libraries(YourPlugin OpenMM::OpenMM)
```

## 環境變量設置

為了確保運行時能找到 OpenMM 庫,添加到你的 `~/.bashrc`:

```bash
# OpenMM 環境設置
export OPENMM_ROOT=/home/andy/miniforge3/envs/cuda
export LD_LIBRARY_PATH=$OPENMM_ROOT/lib:$LD_LIBRARY_PATH
export CPATH=$OPENMM_ROOT/include:$CPATH
export LIBRARY_PATH=$OPENMM_ROOT/lib:$LIBRARY_PATH
```

然後執行:
```bash
source ~/.bashrc
```

## 參考其他使用 OpenMM 的項目

可以參考你已有的 `openmmexampleplugin`:

```bash
cd /home/andy/test_optimization/plugins/openmmexampleplugin
cat CMakeLists.txt | grep -A10 -B10 OpenMM
```

這個示例插件展示了如何正確鏈接 OpenMM。

## 故障排除

### Q: 編譯插件時提示找不到 OpenMM.h
**A:** 檢查 `OPENMM_DIR` 是否正確,確認該目錄下有 `include/OpenMM.h`

### Q: 鏈接時提示 undefined reference to OpenMM
**A:** 確認已正確鏈接 OpenMM 庫,並且 `LD_LIBRARY_PATH` 包含 OpenMM 庫路徑

### Q: 運行時提示找不到 libOpenMM.so
**A:** 
```bash
export LD_LIBRARY_PATH=/home/andy/miniforge3/envs/cuda/lib:$LD_LIBRARY_PATH
# 或
sudo ldconfig /home/andy/miniforge3/envs/cuda/lib
```

## 快速驗證清單

- [ ] `make` 編譯完成無錯誤
- [ ] `make install` 執行完成
- [ ] `make PythonInstall` 執行完成  
- [ ] `python -m openmm.testInstallation` 通過
- [ ] `ls /home/andy/miniforge3/envs/cuda/lib/libOpenMM.so` 存在
- [ ] `ls /home/andy/miniforge3/envs/cuda/include/OpenMM.h` 存在
