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
include examples/01_cutlass_utilities/CMakeFiles/01_cutlass_utilities.dir/depend.make

# Include the progress variables for this target.
include examples/01_cutlass_utilities/CMakeFiles/01_cutlass_utilities.dir/progress.make

# Include the compile flags for this target's objects.
include examples/01_cutlass_utilities/CMakeFiles/01_cutlass_utilities.dir/flags.make

examples/01_cutlass_utilities/CMakeFiles/01_cutlass_utilities.dir/cutlass_utilities.cu.o: examples/01_cutlass_utilities/CMakeFiles/01_cutlass_utilities.dir/flags.make
examples/01_cutlass_utilities/CMakeFiles/01_cutlass_utilities.dir/cutlass_utilities.cu.o: ../examples/01_cutlass_utilities/cutlass_utilities.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/vincent/Documents/LeidenUniv/Vakken/Thesis/cutlass/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object examples/01_cutlass_utilities/CMakeFiles/01_cutlass_utilities.dir/cutlass_utilities.cu.o"
	cd /home/vincent/Documents/LeidenUniv/Vakken/Thesis/cutlass/build/examples/01_cutlass_utilities && /usr/local/cuda/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/vincent/Documents/LeidenUniv/Vakken/Thesis/cutlass/examples/01_cutlass_utilities/cutlass_utilities.cu -o CMakeFiles/01_cutlass_utilities.dir/cutlass_utilities.cu.o

examples/01_cutlass_utilities/CMakeFiles/01_cutlass_utilities.dir/cutlass_utilities.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/01_cutlass_utilities.dir/cutlass_utilities.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

examples/01_cutlass_utilities/CMakeFiles/01_cutlass_utilities.dir/cutlass_utilities.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/01_cutlass_utilities.dir/cutlass_utilities.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target 01_cutlass_utilities
01_cutlass_utilities_OBJECTS = \
"CMakeFiles/01_cutlass_utilities.dir/cutlass_utilities.cu.o"

# External object files for target 01_cutlass_utilities
01_cutlass_utilities_EXTERNAL_OBJECTS =

examples/01_cutlass_utilities/01_cutlass_utilities: examples/01_cutlass_utilities/CMakeFiles/01_cutlass_utilities.dir/cutlass_utilities.cu.o
examples/01_cutlass_utilities/01_cutlass_utilities: examples/01_cutlass_utilities/CMakeFiles/01_cutlass_utilities.dir/build.make
examples/01_cutlass_utilities/01_cutlass_utilities: /usr/local/cuda/lib64/libcublas.so
examples/01_cutlass_utilities/01_cutlass_utilities: /usr/local/cuda/lib64/libcublasLt.so
examples/01_cutlass_utilities/01_cutlass_utilities: examples/01_cutlass_utilities/CMakeFiles/01_cutlass_utilities.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/vincent/Documents/LeidenUniv/Vakken/Thesis/cutlass/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable 01_cutlass_utilities"
	cd /home/vincent/Documents/LeidenUniv/Vakken/Thesis/cutlass/build/examples/01_cutlass_utilities && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/01_cutlass_utilities.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/01_cutlass_utilities/CMakeFiles/01_cutlass_utilities.dir/build: examples/01_cutlass_utilities/01_cutlass_utilities

.PHONY : examples/01_cutlass_utilities/CMakeFiles/01_cutlass_utilities.dir/build

examples/01_cutlass_utilities/CMakeFiles/01_cutlass_utilities.dir/clean:
	cd /home/vincent/Documents/LeidenUniv/Vakken/Thesis/cutlass/build/examples/01_cutlass_utilities && $(CMAKE_COMMAND) -P CMakeFiles/01_cutlass_utilities.dir/cmake_clean.cmake
.PHONY : examples/01_cutlass_utilities/CMakeFiles/01_cutlass_utilities.dir/clean

examples/01_cutlass_utilities/CMakeFiles/01_cutlass_utilities.dir/depend:
	cd /home/vincent/Documents/LeidenUniv/Vakken/Thesis/cutlass/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/vincent/Documents/LeidenUniv/Vakken/Thesis/cutlass /home/vincent/Documents/LeidenUniv/Vakken/Thesis/cutlass/examples/01_cutlass_utilities /home/vincent/Documents/LeidenUniv/Vakken/Thesis/cutlass/build /home/vincent/Documents/LeidenUniv/Vakken/Thesis/cutlass/build/examples/01_cutlass_utilities /home/vincent/Documents/LeidenUniv/Vakken/Thesis/cutlass/build/examples/01_cutlass_utilities/CMakeFiles/01_cutlass_utilities.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/01_cutlass_utilities/CMakeFiles/01_cutlass_utilities.dir/depend

