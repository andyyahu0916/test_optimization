# Install script for directory: /home/andy/test_optimization/openmm-8.4.0/plugins/drude

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
  include("/home/andy/test_optimization/openmm-8.4.0/build_test/plugins/drude/platforms/reference/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/andy/test_optimization/openmm-8.4.0/build_test/plugins/drude/platforms/common/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/andy/test_optimization/openmm-8.4.0/build_test/plugins/drude/platforms/cuda/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/andy/test_optimization/openmm-8.4.0/plugins/drude/openmmapi/include/OpenMMDrude.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/openmm" TYPE FILE FILES
    "/home/andy/test_optimization/openmm-8.4.0/plugins/drude/openmmapi/include/openmm/DrudeForce.h"
    "/home/andy/test_optimization/openmm-8.4.0/plugins/drude/openmmapi/include/openmm/DrudeIntegrator.h"
    "/home/andy/test_optimization/openmm-8.4.0/plugins/drude/openmmapi/include/openmm/DrudeKernels.h"
    "/home/andy/test_optimization/openmm-8.4.0/plugins/drude/openmmapi/include/openmm/DrudeLangevinIntegrator.h"
    "/home/andy/test_optimization/openmm-8.4.0/plugins/drude/openmmapi/include/openmm/DrudeNoseHooverIntegrator.h"
    "/home/andy/test_optimization/openmm-8.4.0/plugins/drude/openmmapi/include/openmm/DrudeSCFIntegrator.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/openmm/internal" TYPE FILE FILES
    "/home/andy/test_optimization/openmm-8.4.0/plugins/drude/openmmapi/include/openmm/internal/DrudeForceImpl.h"
    "/home/andy/test_optimization/openmm-8.4.0/plugins/drude/openmmapi/include/openmm/internal/DrudeHelpers.h"
    "/home/andy/test_optimization/openmm-8.4.0/plugins/drude/openmmapi/include/openmm/internal/windowsExportDrude.h"
    )
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/andy/test_optimization/openmm-8.4.0/build_test/plugins/drude/serialization/tests/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libOpenMMDrude.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libOpenMMDrude.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libOpenMMDrude.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/andy/test_optimization/openmm-8.4.0/build_test/libOpenMMDrude.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libOpenMMDrude.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libOpenMMDrude.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libOpenMMDrude.so"
         OLD_RPATH "/home/andy/test_optimization/openmm-8.4.0/build_test:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libOpenMMDrude.so")
    endif()
  endif()
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/home/andy/test_optimization/openmm-8.4.0/build_test/plugins/drude/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
