# CMake generated Testfile for 
# Source directory: /home/andy/test_optimization/openmm_core_integration
# Build directory: /home/andy/test_optimization/openmm_core_integration/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(import_test "/home/andy/miniforge3/envs/cuda/bin/python3.13" "-c" "import constantv; print('Import successful')")
set_tests_properties(import_test PROPERTIES  _BACKTRACE_TRIPLES "/home/andy/test_optimization/openmm_core_integration/CMakeLists.txt;326;add_test;/home/andy/test_optimization/openmm_core_integration/CMakeLists.txt;0;")
