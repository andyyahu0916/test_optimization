# CMake generated Testfile for 
# Source directory: /home/andy/test_optimization/openmm-8.4.0/plugins/rpmd/platforms/cuda/tests
# Build directory: /home/andy/test_optimization/openmm-8.4.0/build_test/plugins/rpmd/platforms/cuda/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(TestCudaRpmdSingle "/home/andy/test_optimization/openmm-8.4.0/build_test/TestCudaRpmd" "single")
set_tests_properties(TestCudaRpmdSingle PROPERTIES  _BACKTRACE_TRIPLES "/home/andy/test_optimization/openmm-8.4.0/plugins/rpmd/platforms/cuda/tests/CMakeLists.txt;24;ADD_TEST;/home/andy/test_optimization/openmm-8.4.0/plugins/rpmd/platforms/cuda/tests/CMakeLists.txt;0;")
add_test(TestCudaRpmdMixed "/home/andy/test_optimization/openmm-8.4.0/build_test/TestCudaRpmd" "mixed")
set_tests_properties(TestCudaRpmdMixed PROPERTIES  _BACKTRACE_TRIPLES "/home/andy/test_optimization/openmm-8.4.0/plugins/rpmd/platforms/cuda/tests/CMakeLists.txt;26;ADD_TEST;/home/andy/test_optimization/openmm-8.4.0/plugins/rpmd/platforms/cuda/tests/CMakeLists.txt;0;")
add_test(TestCudaRpmdDouble "/home/andy/test_optimization/openmm-8.4.0/build_test/TestCudaRpmd" "double")
set_tests_properties(TestCudaRpmdDouble PROPERTIES  _BACKTRACE_TRIPLES "/home/andy/test_optimization/openmm-8.4.0/plugins/rpmd/platforms/cuda/tests/CMakeLists.txt;27;ADD_TEST;/home/andy/test_optimization/openmm-8.4.0/plugins/rpmd/platforms/cuda/tests/CMakeLists.txt;0;")
