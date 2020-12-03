# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/vincent/Documents/LeidenUniv/Vakken/Thesis/cutlass

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/vincent/Documents/LeidenUniv/Vakken/Thesis/cutlass/build

# Include any dependencies generated for this target.
include test/unit/transform/threadblock/CMakeFiles/cutlass_test_unit_transform_threadblock.dir/depend.make

# Include the progress variables for this target.
include test/unit/transform/threadblock/CMakeFiles/cutlass_test_unit_transform_threadblock.dir/progress.make

# Include the compile flags for this target's objects.
include test/unit/transform/threadblock/CMakeFiles/cutlass_test_unit_transform_threadblock.dir/flags.make

test/unit/transform/threadblock/CMakeFiles/cutlass_test_unit_transform_threadblock.dir/regular_tile_iterator_tensor_op.cu.o: test/unit/transform/threadblock/CMakeFiles/cutlass_test_unit_transform_threadblock.dir/flags.make
test/unit/transform/threadblock/CMakeFiles/cutlass_test_unit_transform_threadblock.dir/regular_tile_iterator_tensor_op.cu.o: ../test/unit/transform/threadblock/regular_tile_iterator_tensor_op.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/vincent/Documents/LeidenUniv/Vakken/Thesis/cutlass/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object test/unit/transform/threadblock/CMakeFiles/cutlass_test_unit_transform_threadblock.dir/regular_tile_iterator_tensor_op.cu.o"
	cd /home/vincent/Documents/LeidenUniv/Vakken/Thesis/cutlass/build/test/unit/transform/threadblock && /usr/local/cuda/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/vincent/Documents/LeidenUniv/Vakken/Thesis/cutlass/test/unit/transform/threadblock/regular_tile_iterator_tensor_op.cu -o CMakeFiles/cutlass_test_unit_transform_threadblock.dir/regular_tile_iterator_tensor_op.cu.o

test/unit/transform/threadblock/CMakeFiles/cutlass_test_unit_transform_threadblock.dir/regular_tile_iterator_tensor_op.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/cutlass_test_unit_transform_threadblock.dir/regular_tile_iterator_tensor_op.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

test/unit/transform/threadblock/CMakeFiles/cutlass_test_unit_transform_threadblock.dir/regular_tile_iterator_tensor_op.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/cutlass_test_unit_transform_threadblock.dir/regular_tile_iterator_tensor_op.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

test/unit/transform/threadblock/CMakeFiles/cutlass_test_unit_transform_threadblock.dir/predicated_tile_iterator.cu.o: test/unit/transform/threadblock/CMakeFiles/cutlass_test_unit_transform_threadblock.dir/flags.make
test/unit/transform/threadblock/CMakeFiles/cutlass_test_unit_transform_threadblock.dir/predicated_tile_iterator.cu.o: ../test/unit/transform/threadblock/predicated_tile_iterator.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/vincent/Documents/LeidenUniv/Vakken/Thesis/cutlass/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CUDA object test/unit/transform/threadblock/CMakeFiles/cutlass_test_unit_transform_threadblock.dir/predicated_tile_iterator.cu.o"
	cd /home/vincent/Documents/LeidenUniv/Vakken/Thesis/cutlass/build/test/unit/transform/threadblock && /usr/local/cuda/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/vincent/Documents/LeidenUniv/Vakken/Thesis/cutlass/test/unit/transform/threadblock/predicated_tile_iterator.cu -o CMakeFiles/cutlass_test_unit_transform_threadblock.dir/predicated_tile_iterator.cu.o

test/unit/transform/threadblock/CMakeFiles/cutlass_test_unit_transform_threadblock.dir/predicated_tile_iterator.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/cutlass_test_unit_transform_threadblock.dir/predicated_tile_iterator.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

test/unit/transform/threadblock/CMakeFiles/cutlass_test_unit_transform_threadblock.dir/predicated_tile_iterator.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/cutlass_test_unit_transform_threadblock.dir/predicated_tile_iterator.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target cutlass_test_unit_transform_threadblock
cutlass_test_unit_transform_threadblock_OBJECTS = \
"CMakeFiles/cutlass_test_unit_transform_threadblock.dir/regular_tile_iterator_tensor_op.cu.o" \
"CMakeFiles/cutlass_test_unit_transform_threadblock.dir/predicated_tile_iterator.cu.o"

# External object files for target cutlass_test_unit_transform_threadblock
cutlass_test_unit_transform_threadblock_EXTERNAL_OBJECTS = \
"/home/vincent/Documents/LeidenUniv/Vakken/Thesis/cutlass/build/test/unit/CMakeFiles/cutlass_test_unit_infra.dir/common/filter_architecture.cpp.o" \
"/home/vincent/Documents/LeidenUniv/Vakken/Thesis/cutlass/build/test/unit/CMakeFiles/cutlass_test_unit_infra_lib.dir/test_unit.cpp.o"

test/unit/transform/threadblock/cutlass_test_unit_transform_threadblock: test/unit/transform/threadblock/CMakeFiles/cutlass_test_unit_transform_threadblock.dir/regular_tile_iterator_tensor_op.cu.o
test/unit/transform/threadblock/cutlass_test_unit_transform_threadblock: test/unit/transform/threadblock/CMakeFiles/cutlass_test_unit_transform_threadblock.dir/predicated_tile_iterator.cu.o
test/unit/transform/threadblock/cutlass_test_unit_transform_threadblock: test/unit/CMakeFiles/cutlass_test_unit_infra.dir/common/filter_architecture.cpp.o
test/unit/transform/threadblock/cutlass_test_unit_transform_threadblock: test/unit/CMakeFiles/cutlass_test_unit_infra_lib.dir/test_unit.cpp.o
test/unit/transform/threadblock/cutlass_test_unit_transform_threadblock: test/unit/transform/threadblock/CMakeFiles/cutlass_test_unit_transform_threadblock.dir/build.make
test/unit/transform/threadblock/cutlass_test_unit_transform_threadblock: /usr/local/cuda/lib64/libcublas.so
test/unit/transform/threadblock/cutlass_test_unit_transform_threadblock: /usr/local/cuda/lib64/libcublasLt.so
test/unit/transform/threadblock/cutlass_test_unit_transform_threadblock: _deps/googletest-build/googlemock/gtest/libgtest.a
test/unit/transform/threadblock/cutlass_test_unit_transform_threadblock: test/unit/transform/threadblock/CMakeFiles/cutlass_test_unit_transform_threadblock.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/vincent/Documents/LeidenUniv/Vakken/Thesis/cutlass/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable cutlass_test_unit_transform_threadblock"
	cd /home/vincent/Documents/LeidenUniv/Vakken/Thesis/cutlass/build/test/unit/transform/threadblock && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cutlass_test_unit_transform_threadblock.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/unit/transform/threadblock/CMakeFiles/cutlass_test_unit_transform_threadblock.dir/build: test/unit/transform/threadblock/cutlass_test_unit_transform_threadblock

.PHONY : test/unit/transform/threadblock/CMakeFiles/cutlass_test_unit_transform_threadblock.dir/build

test/unit/transform/threadblock/CMakeFiles/cutlass_test_unit_transform_threadblock.dir/clean:
	cd /home/vincent/Documents/LeidenUniv/Vakken/Thesis/cutlass/build/test/unit/transform/threadblock && $(CMAKE_COMMAND) -P CMakeFiles/cutlass_test_unit_transform_threadblock.dir/cmake_clean.cmake
.PHONY : test/unit/transform/threadblock/CMakeFiles/cutlass_test_unit_transform_threadblock.dir/clean

test/unit/transform/threadblock/CMakeFiles/cutlass_test_unit_transform_threadblock.dir/depend:
	cd /home/vincent/Documents/LeidenUniv/Vakken/Thesis/cutlass/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/vincent/Documents/LeidenUniv/Vakken/Thesis/cutlass /home/vincent/Documents/LeidenUniv/Vakken/Thesis/cutlass/test/unit/transform/threadblock /home/vincent/Documents/LeidenUniv/Vakken/Thesis/cutlass/build /home/vincent/Documents/LeidenUniv/Vakken/Thesis/cutlass/build/test/unit/transform/threadblock /home/vincent/Documents/LeidenUniv/Vakken/Thesis/cutlass/build/test/unit/transform/threadblock/CMakeFiles/cutlass_test_unit_transform_threadblock.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/unit/transform/threadblock/CMakeFiles/cutlass_test_unit_transform_threadblock.dir/depend
