# Install script for directory: /home/andy/test_optimization/openmm-8.4.0/examples/python-examples

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
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/examples/python-examples" TYPE FILE FILES
    "/home/andy/test_optimization/openmm-8.4.0/examples/python-examples/simulateAmber.py"
    "/home/andy/test_optimization/openmm-8.4.0/examples/python-examples/input.inpcrd"
    "/home/andy/test_optimization/openmm-8.4.0/examples/python-examples/input.prmtop"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/examples/python-examples" TYPE FILE FILES
    "/home/andy/test_optimization/openmm-8.4.0/examples/python-examples/simulateCharmm.py"
    "/home/andy/test_optimization/openmm-8.4.0/examples/python-examples/ala_ala_ala.pdb"
    "/home/andy/test_optimization/openmm-8.4.0/examples/python-examples/ala_ala_ala.psf"
    "/home/andy/test_optimization/openmm-8.4.0/examples/python-examples/charmm22.par"
    "/home/andy/test_optimization/openmm-8.4.0/examples/python-examples/charmm22.rtf"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/examples/python-examples" TYPE FILE FILES
    "/home/andy/test_optimization/openmm-8.4.0/examples/python-examples/simulateGromacs.py"
    "/home/andy/test_optimization/openmm-8.4.0/examples/python-examples/input.gro"
    "/home/andy/test_optimization/openmm-8.4.0/examples/python-examples/input.top"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/examples/python-examples" TYPE FILE FILES
    "/home/andy/test_optimization/openmm-8.4.0/examples/python-examples/simulatePdb.py"
    "/home/andy/test_optimization/openmm-8.4.0/examples/python-examples/input.pdb"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/examples/python-examples" TYPE FILE FILES "/home/andy/test_optimization/openmm-8.4.0/examples/python-examples/argon-chemical-potential.py")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/home/andy/test_optimization/openmm-8.4.0/build_test/examples/python-examples/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
