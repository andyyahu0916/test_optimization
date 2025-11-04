#---------------------------------------------------
# OpenMM Setup for Plugin Development
# 此文件設置 OpenMM 的包含路徑和庫鏈接
#
# OpenMM 不提供標準的 CMake Config 文件,
# 需要手動指定安裝路徑
#---------------------------------------------------

# 設置 OpenMM 安裝路徑
# 如果使用 conda 環境,指向 conda 環境根目錄
IF(NOT DEFINED OPENMM_DIR)
    SET(OPENMM_DIR "/home/andy/miniforge3/envs/cuda" CACHE PATH "Where OpenMM is installed")
ENDIF()

MESSAGE(STATUS "Looking for OpenMM in: ${OPENMM_DIR}")

# 檢查 OpenMM 是否存在
IF(NOT EXISTS "${OPENMM_DIR}/include/OpenMM.h")
    MESSAGE(FATAL_ERROR "找不到 OpenMM! 請確認 OPENMM_DIR 設置正確: ${OPENMM_DIR}")
ENDIF()

# 添加 OpenMM 包含目錄
INCLUDE_DIRECTORIES("${OPENMM_DIR}/include")

# 添加 OpenMM 庫目錄
LINK_DIRECTORIES("${OPENMM_DIR}/lib" "${OPENMM_DIR}/lib/plugins")

# 查找 OpenMM 主庫
FIND_LIBRARY(OPENMM_LIBRARY 
    NAMES OpenMM
    PATHS "${OPENMM_DIR}/lib"
    NO_DEFAULT_PATH
)

IF(NOT OPENMM_LIBRARY)
    MESSAGE(FATAL_ERROR "找不到 libOpenMM.so 在 ${OPENMM_DIR}/lib")
ENDIF()

MESSAGE(STATUS "Found OpenMM library: ${OPENMM_LIBRARY}")

# 設置變量供其他 CMakeLists.txt 使用
SET(OPENMM_INCLUDE_DIR "${OPENMM_DIR}/include")
SET(OPENMM_LIBRARY_DIR "${OPENMM_DIR}/lib")
SET(OPENMM_LIBRARIES OpenMM)

# 顯示配置信息
MESSAGE(STATUS "OpenMM Configuration:")
MESSAGE(STATUS "  Include Dir: ${OPENMM_INCLUDE_DIR}")
MESSAGE(STATUS "  Library Dir: ${OPENMM_LIBRARY_DIR}")
MESSAGE(STATUS "  Libraries: ${OPENMM_LIBRARIES}")
