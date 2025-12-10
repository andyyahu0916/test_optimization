# Install script for directory: /home/andy/test_optimization/openmm-8.4.0/platforms/common

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

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/openmm/common" TYPE FILE FILES
    "/home/andy/test_optimization/openmm-8.4.0/build_test/platforms/common/src/CommonKernelSources.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/common/include/openmm/common/ArrayInterface.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/common/include/openmm/common/BondedUtilities.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/common/include/openmm/common/CommonCalcConstantPotentialForce.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/common/include/openmm/common/CommonCalcCustomGBForceKernel.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/common/include/openmm/common/CommonCalcCustomHbondForceKernel.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/common/include/openmm/common/CommonCalcCustomManyParticleForceKernel.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/common/include/openmm/common/CommonCalcCustomNonbondedForceKernel.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/common/include/openmm/common/CommonCalcNonbondedForce.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/common/include/openmm/common/CommonIntegrateCustomStepKernel.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/common/include/openmm/common/CommonIntegrateNoseHooverStepKernel.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/common/include/openmm/common/CommonIntegrateQTBStepKernel.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/common/include/openmm/common/CommonKernelUtilities.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/common/include/openmm/common/CommonKernels.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/common/include/openmm/common/CommonParallelKernels.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/common/include/openmm/common/ComputeArray.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/common/include/openmm/common/ComputeContext.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/common/include/openmm/common/ComputeEvent.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/common/include/openmm/common/ComputeForceInfo.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/common/include/openmm/common/ComputeKernel.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/common/include/openmm/common/ComputeParameterInfo.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/common/include/openmm/common/ComputeParameterSet.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/common/include/openmm/common/ComputeProgram.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/common/include/openmm/common/ComputeQueue.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/common/include/openmm/common/ComputeSort.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/common/include/openmm/common/ComputeVectorTypes.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/common/include/openmm/common/ContextSelector.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/common/include/openmm/common/ExpressionUtilities.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/common/include/openmm/common/FFT3D.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/common/include/openmm/common/IntegrationUtilities.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/common/include/openmm/common/NonbondedUtilities.h"
    "/home/andy/test_optimization/openmm-8.4.0/platforms/common/include/openmm/common/windowsExportCommon.h"
    )
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/home/andy/test_optimization/openmm-8.4.0/build_test/platforms/common/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
