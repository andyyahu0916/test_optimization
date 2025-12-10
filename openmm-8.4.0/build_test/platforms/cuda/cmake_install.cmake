# Install script for directory: /home/andy/test_optimization/openmm-8.4.0/platforms/cuda

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/andy/miniforge3/envs/cuda")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/andy/test_optimization/openmm-8.4.0/build_test/platforms/cuda/tests/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/openmm/cuda" TYPE FILE FILES
    "/home/andy/test_optimization/openmm-8.4.0/build_test/platforms/cuda/src/CudaKernelSources.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/cuda/include/CudaArray.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/cuda/include/CudaBondedUtilities.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/cuda/include/CudaContext.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/cuda/include/CudaEvent.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/cuda/include/CudaExpressionUtilities.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/cuda/include/CudaFFT3D.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/cuda/include/CudaForceInfo.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/cuda/include/CudaIntegrationUtilities.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/cuda/include/CudaKernel.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/cuda/include/CudaKernelFactory.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/cuda/include/CudaKernels.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/cuda/include/CudaNonbondedUtilities.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/cuda/include/CudaParallelKernels.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/cuda/include/CudaPlatform.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/cuda/include/CudaProgram.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/cuda/include/CudaQueue.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/cuda/include/CudaSort.h"
    )
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/andy/test_optimization/openmm-8.4.0/build_test/platforms/cuda/sharedTarget/cmake_install.cmake")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/home/andy/test_optimization/openmm-8.4.0/build_test/platforms/cuda/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
