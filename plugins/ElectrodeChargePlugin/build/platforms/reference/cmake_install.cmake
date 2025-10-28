# Install script for directory: /home/andy/test_optimization/plugins/ElectrodeChargePlugin/platforms/reference

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local/openmm")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
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
  set(CMAKE_OBJDUMP "/home/andy/miniforge3/envs/cuda/bin/x86_64-conda-linux-gnu-objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/usr/local/openmm/lib/plugins/libElectrodeChargePluginReference.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/usr/local/openmm/lib/plugins/libElectrodeChargePluginReference.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/usr/local/openmm/lib/plugins/libElectrodeChargePluginReference.so"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/usr/local/openmm/lib/plugins/libElectrodeChargePluginReference.so")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/usr/local/openmm/lib/plugins" TYPE SHARED_LIBRARY FILES "/home/andy/test_optimization/plugins/ElectrodeChargePlugin/build/platforms/reference/libElectrodeChargePluginReference.so")
  if(EXISTS "$ENV{DESTDIR}/usr/local/openmm/lib/plugins/libElectrodeChargePluginReference.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/usr/local/openmm/lib/plugins/libElectrodeChargePluginReference.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/usr/local/openmm/lib/plugins/libElectrodeChargePluginReference.so"
         OLD_RPATH "/home/andy/test_optimization/plugins/ElectrodeChargePlugin/build:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/home/andy/miniforge3/envs/cuda/bin/x86_64-conda-linux-gnu-strip" "$ENV{DESTDIR}/usr/local/openmm/lib/plugins/libElectrodeChargePluginReference.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/andy/test_optimization/plugins/ElectrodeChargePlugin/build/platforms/reference/CMakeFiles/ElectrodeChargePluginReference.dir/install-cxx-module-bmi-noconfig.cmake" OPTIONAL)
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/home/andy/test_optimization/plugins/ElectrodeChargePlugin/build/platforms/reference/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
