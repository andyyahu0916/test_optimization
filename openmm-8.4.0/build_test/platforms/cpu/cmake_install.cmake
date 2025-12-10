# Install script for directory: /home/andy/test_optimization/openmm-8.4.0/platforms/cpu

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
  include("/home/andy/test_optimization/openmm-8.4.0/build_test/platforms/cpu/tests/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/openmm/cpu" TYPE FILE FILES
    "/home/andy/test_optimization/openmm-8.4.0/platforms/cpu/include/AlignedArray.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/cpu/include/CpuBondForce.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/cpu/include/CpuConstantPotentialForce.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/cpu/include/CpuConstantPotentialForceFvec.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/cpu/include/CpuCustomGBForce.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/cpu/include/CpuCustomManyParticleForce.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/cpu/include/CpuCustomNonbondedForce.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/cpu/include/CpuCustomNonbondedForceFvec.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/cpu/include/CpuGBSAOBCForce.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/cpu/include/CpuGayBerneForce.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/cpu/include/CpuKernelFactory.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/cpu/include/CpuKernels.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/cpu/include/CpuLangevinMiddleDynamics.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/cpu/include/CpuNeighborList.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/cpu/include/CpuNonbondedForce.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/cpu/include/CpuNonbondedForceFvec.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/cpu/include/CpuPlatform.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/cpu/include/CpuRandom.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/cpu/include/CpuSETTLE.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/cpu/include/windowsExportCpu.h"
    )
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/andy/test_optimization/openmm-8.4.0/build_test/platforms/cpu/sharedTarget/cmake_install.cmake")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/home/andy/test_optimization/openmm-8.4.0/build_test/platforms/cpu/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
