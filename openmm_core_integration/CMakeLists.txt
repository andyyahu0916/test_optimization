# ═══════════════════════════════════════════════════════════════════════════
# CMake Build Configuration for ConstantV Native Core Integration
# ═══════════════════════════════════════════════════════════════════════════

cmake_minimum_required(VERSION 3.18)
project(ConstantVIntegration VERSION 1.0.0 LANGUAGES CXX CUDA)

# ═══════════════════════════════════════════════════════════════════════════
# Build Options
# ═══════════════════════════════════════════════════════════════════════════

option(BUILD_CUDA_LIB "Build CUDA platform library" ON)
option(BUILD_REFERENCE_LIB "Build Reference platform library" ON)
option(BUILD_PYTHON_WRAPPERS "Build Python SWIG bindings" ON)

# ═══════════════════════════════════════════════════════════════════════════
# Find Dependencies (mamba/conda compatible)
# ═══════════════════════════════════════════════════════════════════════════

# Try to find OpenMM in conda environment first
if(DEFINED ENV{CONDA_PREFIX})
    set(CONDA_PREFIX $ENV{CONDA_PREFIX})
    message(STATUS "Detected mamba/conda environment: ${CONDA_PREFIX}")
    
    # Set OpenMM paths for conda
    set(OpenMM_INCLUDE_DIR "${CONDA_PREFIX}/include" CACHE PATH "OpenMM include directory")
    set(OpenMM_LIBRARY "${CONDA_PREFIX}/lib/libOpenMM.so" CACHE FILEPATH "OpenMM library")
    set(OpenMM_LIBRARIES "${OpenMM_LIBRARY}" CACHE STRING "OpenMM libraries")
    
    # Set CUDA paths for conda
    if(EXISTS "${CONDA_PREFIX}/bin/nvcc")
        set(CMAKE_CUDA_COMPILER "${CONDA_PREFIX}/bin/nvcc" CACHE FILEPATH "CUDA compiler")
        set(CUDAToolkit_ROOT "${CONDA_PREFIX}" CACHE PATH "CUDA toolkit root")
    endif()
else()
    # OpenMM (required) - standard find_package
    find_package(OpenMM REQUIRED)
endif()

include_directories(${OpenMM_INCLUDE_DIR})

# CUDA (for CUDA platform)
if(BUILD_CUDA_LIB)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    # Set CUDA architecture (adjust for your GPU)
    # sm_70: V100, sm_75: T4, sm_80: A100, sm_86: RTX 30xx, sm_89: RTX 40xx, sm_90: H100
    set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90" CACHE STRING "CUDA architectures to compile for")
endif()

# SWIG (for Python bindings)
if(BUILD_PYTHON_WRAPPERS)
    find_package(SWIG 4.0 REQUIRED)
    include(${SWIG_USE_FILE})

    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    include_directories(${Python3_INCLUDE_DIRS})
endif()

# ═══════════════════════════════════════════════════════════════════════════
# Compiler Settings
# ═══════════════════════════════════════════════════════════════════════════

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA Standard
if(BUILD_CUDA_LIB)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Warning flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wno-unused-parameter)
endif()

# Optimization
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG --use_fast_math")

# ═══════════════════════════════════════════════════════════════════════════
# Include Directories
# ═══════════════════════════════════════════════════════════════════════════

include_directories(
    ${CMAKE_SOURCE_DIR}/openmmapi/include
    ${CMAKE_SOURCE_DIR}/platforms/cuda/include
    ${CMAKE_SOURCE_DIR}/platforms/reference/include
)

# ═══════════════════════════════════════════════════════════════════════════
# Core API Library (Platform-Independent)
# ═══════════════════════════════════════════════════════════════════════════

set(CORE_API_SOURCES
    openmmapi/src/ConstantVForce.cpp
    openmmapi/src/ConstantVForceImpl.cpp
    # openmmapi/src/ConstantVIntegrator.cpp  # TODO: Fix Verlet kernel integration
    openmmapi/src/ConstantVDrudeLangevinIntegrator.cpp
    openmmapi/src/ConstantVKernelFactory.cpp
    openmmapi/src/registerConstantV.cpp
)

add_library(ConstantVAPI SHARED ${CORE_API_SOURCES})

# Add compile definitions for conditional compilation
if(BUILD_CUDA_LIB)
    target_compile_definitions(ConstantVAPI PRIVATE OPENMM_BUILD_CUDA_LIB)
endif()

if(BUILD_REFERENCE_LIB)
    target_compile_definitions(ConstantVAPI PRIVATE OPENMM_BUILD_REFERENCE_LIB)
endif()

target_link_libraries(ConstantVAPI
    ${OpenMM_LIBRARIES}
)

target_include_directories(ConstantVAPI PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/openmmapi/include>
    $<INSTALL_INTERFACE:include>
)

set_target_properties(ConstantVAPI PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    PUBLIC_HEADER "openmmapi/include/openmm/ConstantVDrudeLangevinIntegrator.h"
)

# ═══════════════════════════════════════════════════════════════════════════
# CUDA Platform Library
# ═══════════════════════════════════════════════════════════════════════════

if(BUILD_CUDA_LIB)
    set(CUDA_SOURCES
        platforms/cuda/src/kernels/constantVDrudeLangevin.cu
        platforms/cuda/src/CudaConstantVKernels.cpp
    )

    # Mark C++ file as CUDA source (needed for kernel calls with <<<>>> syntax)
    set_source_files_properties(platforms/cuda/src/CudaConstantVKernels.cpp PROPERTIES LANGUAGE CUDA)

    add_library(ConstantVCUDA SHARED ${CUDA_SOURCES})

    target_link_libraries(ConstantVCUDA
        ConstantVAPI
        ${OpenMM_LIBRARIES}
        CUDA::cudart
        CUDA::cuda_driver
    )

    # CUDA compilation options
    target_compile_options(ConstantVCUDA PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            --expt-relaxed-constexpr
            --use_fast_math
            -Xptxas=-v
            --generate-line-info
        >
    )

    set_target_properties(ConstantVCUDA PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        POSITION_INDEPENDENT_CODE ON
    )
endif()

# ═══════════════════════════════════════════════════════════════════════════
# Reference Platform Library
# ═══════════════════════════════════════════════════════════════════════════

if(BUILD_REFERENCE_LIB)
    set(REFERENCE_SOURCES
        platforms/reference/src/ReferenceConstantVDrudeLangevinDynamics.cpp
        platforms/reference/src/ReferenceConstantVKernels.cpp
    )

    add_library(ConstantVReference SHARED ${REFERENCE_SOURCES})

    target_include_directories(ConstantVReference PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/platforms/reference/include>
        $<INSTALL_INTERFACE:include/openmm/constantv/reference>
    )

    target_link_libraries(ConstantVReference
        ConstantVAPI
        ${OpenMM_LIBRARIES}
    )

    set_target_properties(ConstantVReference PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )
endif()

# ═══════════════════════════════════════════════════════════════════════════
# Python SWIG Bindings
# ═══════════════════════════════════════════════════════════════════════════

if(BUILD_PYTHON_WRAPPERS)
    set(CMAKE_SWIG_FLAGS -c++)

    # SWIG interface file
    set_property(SOURCE python/ConstantVPlugin.i PROPERTY CPLUSPLUS ON)
    set_property(SOURCE python/ConstantVPlugin.i PROPERTY SWIG_MODULE_NAME constantv)

    swig_add_library(constantv
        TYPE SHARED
        LANGUAGE python
        SOURCES python/ConstantVPlugin.i
    )

    target_link_libraries(constantv
        ConstantVAPI
        ${OpenMM_LIBRARIES}
        Python3::Python
    )

    if(BUILD_CUDA_LIB)
        target_link_libraries(constantv ConstantVCUDA)
    endif()

    if(BUILD_REFERENCE_LIB)
        target_link_libraries(constantv ConstantVReference)
    endif()

    # Set Python module properties
    set_target_properties(constantv PROPERTIES
        PREFIX ""  # No 'lib' prefix for Python modules
        OUTPUT_NAME "_constantv"
        SUFFIX "${Python3_SOABI}.so"
    )

    # Install Python module
    # FIX H2/P4-C4: Use sysconfig.get_path('platlib') for correct venv/conda path
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('platlib'))"
        OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    install(TARGETS constantv
        LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES}
    )

    install(FILES ${CMAKE_BINARY_DIR}/constantv.py
        DESTINATION ${PYTHON_SITE_PACKAGES}
    )
endif()

# ═══════════════════════════════════════════════════════════════════════════
# Installation Rules
# ═══════════════════════════════════════════════════════════════════════════

install(TARGETS ConstantVAPI
    EXPORT ConstantVTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    PUBLIC_HEADER DESTINATION include/openmm
)

if(BUILD_CUDA_LIB)
    install(TARGETS ConstantVCUDA
        LIBRARY DESTINATION lib/plugins
    )
endif()

if(BUILD_REFERENCE_LIB)
    install(TARGETS ConstantVReference
        LIBRARY DESTINATION lib/plugins
    )

    install(FILES
        platforms/reference/include/ReferenceConstantVDrudeLangevinDynamics.h
        platforms/reference/include/ReferenceConstantVKernels.h
        DESTINATION include/openmm/constantv/reference
    )
endif()

# Export targets for other CMake projects
install(EXPORT ConstantVTargets
    FILE ConstantVTargets.cmake
    NAMESPACE ConstantV::
    DESTINATION lib/cmake/ConstantV
)

# Generate package config file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/ConstantVConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_file(cmake/ConstantVConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/ConstantVConfig.cmake"
    @ONLY
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/ConstantVConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/ConstantVConfigVersion.cmake"
    DESTINATION lib/cmake/ConstantV
)

# ═══════════════════════════════════════════════════════════════════════════
# Testing
# ═══════════════════════════════════════════════════════════════════════════

enable_testing()

add_test(NAME import_test
    COMMAND ${Python3_EXECUTABLE} -c "import constantv; print('Import successful')"
)

# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════

message(STATUS "")
message(STATUS "ConstantV Native Integration Configuration Summary")
message(STATUS "==================================================")
message(STATUS "Build type:           ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Compiler:         ${CMAKE_CXX_COMPILER}")
message(STATUS "CUDA Compiler:        ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA Architectures:   ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "OpenMM Include:       ${OpenMM_INCLUDE_DIR}")
message(STATUS "OpenMM Libraries:     ${OpenMM_LIBRARIES}")
message(STATUS "Python Executable:    ${Python3_EXECUTABLE}")
message(STATUS "Python Site-Packages: ${PYTHON_SITE_PACKAGES}")
message(STATUS "")
message(STATUS "Build Options:")
message(STATUS "  CUDA Library:       ${BUILD_CUDA_LIB}")
message(STATUS "  Reference Library:  ${BUILD_REFERENCE_LIB}")
message(STATUS "  Python Wrappers:    ${BUILD_PYTHON_WRAPPERS}")
message(STATUS "")
