# CMake generated Testfile for 
# Source directory: /home/vincent/Documents/LeidenUniv/Vakken/Thesis/cutlass/test/unit/gemm/thread
# Build directory: /home/vincent/Documents/LeidenUniv/Vakken/Thesis/cutlass/build/test/unit/gemm/thread
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(ctest_unit_gemm_thread "cutlass_test_unit_gemm_thread")
set_tests_properties(ctest_unit_gemm_thread PROPERTIES  _BACKTRACE_TRIPLES "/home/vincent/Documents/LeidenUniv/Vakken/Thesis/cutlass/test/unit/CMakeLists.txt;75;add_test;/home/vincent/Documents/LeidenUniv/Vakken/Thesis/cutlass/test/unit/gemm/thread/CMakeLists.txt;23;cutlass_test_unit_add_executable;/home/vincent/Documents/LeidenUniv/Vakken/Thesis/cutlass/test/unit/gemm/thread/CMakeLists.txt;0;")
subdirs("host")