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

# Utility rule file for cutlass_test_unit_transform.

# Include the progress variables for this target.
include test/unit/transform/CMakeFiles/cutlass_test_unit_transform.dir/progress.make

test/unit/transform/CMakeFiles/cutlass_test_unit_transform: test/unit/transform/threadblock/cutlass_test_unit_transform_threadblock


cutlass_test_unit_transform: test/unit/transform/CMakeFiles/cutlass_test_unit_transform
cutlass_test_unit_transform: test/unit/transform/CMakeFiles/cutlass_test_unit_transform.dir/build.make

.PHONY : cutlass_test_unit_transform

# Rule to build all files generated by this target.
test/unit/transform/CMakeFiles/cutlass_test_unit_transform.dir/build: cutlass_test_unit_transform

.PHONY : test/unit/transform/CMakeFiles/cutlass_test_unit_transform.dir/build

test/unit/transform/CMakeFiles/cutlass_test_unit_transform.dir/clean:
	cd /home/vincent/Documents/LeidenUniv/Vakken/Thesis/cutlass/build/test/unit/transform && $(CMAKE_COMMAND) -P CMakeFiles/cutlass_test_unit_transform.dir/cmake_clean.cmake
.PHONY : test/unit/transform/CMakeFiles/cutlass_test_unit_transform.dir/clean

test/unit/transform/CMakeFiles/cutlass_test_unit_transform.dir/depend:
	cd /home/vincent/Documents/LeidenUniv/Vakken/Thesis/cutlass/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/vincent/Documents/LeidenUniv/Vakken/Thesis/cutlass /home/vincent/Documents/LeidenUniv/Vakken/Thesis/cutlass/test/unit/transform /home/vincent/Documents/LeidenUniv/Vakken/Thesis/cutlass/build /home/vincent/Documents/LeidenUniv/Vakken/Thesis/cutlass/build/test/unit/transform /home/vincent/Documents/LeidenUniv/Vakken/Thesis/cutlass/build/test/unit/transform/CMakeFiles/cutlass_test_unit_transform.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/unit/transform/CMakeFiles/cutlass_test_unit_transform.dir/depend

