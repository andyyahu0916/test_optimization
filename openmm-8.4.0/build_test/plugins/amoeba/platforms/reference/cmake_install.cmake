# Install script for directory: /home/andy/test_optimization/openmm-8.4.0/plugins/amoeba/platforms/reference

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
  if(EXISTS "$ENV{DESTDIR}/home/andy/miniforge3/envs/cuda/lib/plugins/libOpenMMAmoebaReference.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/andy/miniforge3/envs/cuda/lib/plugins/libOpenMMAmoebaReference.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/home/andy/miniforge3/envs/cuda/lib/plugins/libOpenMMAmoebaReference.so"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/andy/miniforge3/envs/cuda/lib/plugins/libOpenMMAmoebaReference.so")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/andy/miniforge3/envs/cuda/lib/plugins" TYPE SHARED_LIBRARY FILES "/home/andy/test_optimization/openmm-8.4.0/build_test/libOpenMMAmoebaReference.so")
  if(EXISTS "$ENV{DESTDIR}/home/andy/miniforge3/envs/cuda/lib/plugins/libOpenMMAmoebaReference.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/andy/miniforge3/envs/cuda/lib/plugins/libOpenMMAmoebaReference.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/home/andy/miniforge3/envs/cuda/lib/plugins/libOpenMMAmoebaReference.so"
         OLD_RPATH "/home/andy/test_optimization/openmm-8.4.0/build_test:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/home/andy/miniforge3/envs/cuda/lib/plugins/libOpenMMAmoebaReference.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/andy/test_optimization/openmm-8.4.0/build_test/plugins/amoeba/platforms/reference/CMakeFiles/OpenMMAmoebaReference.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/andy/test_optimization/openmm-8.4.0/build_test/plugins/amoeba/platforms/reference/tests/cmake_install.cmake")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/home/andy/test_optimization/openmm-8.4.0/build_test/plugins/amoeba/platforms/reference/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
