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
CMAKE_SOURCE_DIR = /home/ubuntu/git/ML-Framework-CPP

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ubuntu/git/ML-Framework-CPP/build

# Utility rule file for NightlyUpdate.

# Include the progress variables for this target.
include _deps/yaml-cpp-build/CMakeFiles/NightlyUpdate.dir/progress.make

_deps/yaml-cpp-build/CMakeFiles/NightlyUpdate:
	cd /home/ubuntu/git/ML-Framework-CPP/build/_deps/yaml-cpp-build && /usr/bin/ctest -D NightlyUpdate

NightlyUpdate: _deps/yaml-cpp-build/CMakeFiles/NightlyUpdate
NightlyUpdate: _deps/yaml-cpp-build/CMakeFiles/NightlyUpdate.dir/build.make

.PHONY : NightlyUpdate

# Rule to build all files generated by this target.
_deps/yaml-cpp-build/CMakeFiles/NightlyUpdate.dir/build: NightlyUpdate

.PHONY : _deps/yaml-cpp-build/CMakeFiles/NightlyUpdate.dir/build

_deps/yaml-cpp-build/CMakeFiles/NightlyUpdate.dir/clean:
	cd /home/ubuntu/git/ML-Framework-CPP/build/_deps/yaml-cpp-build && $(CMAKE_COMMAND) -P CMakeFiles/NightlyUpdate.dir/cmake_clean.cmake
.PHONY : _deps/yaml-cpp-build/CMakeFiles/NightlyUpdate.dir/clean

_deps/yaml-cpp-build/CMakeFiles/NightlyUpdate.dir/depend:
	cd /home/ubuntu/git/ML-Framework-CPP/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ubuntu/git/ML-Framework-CPP /home/ubuntu/git/ML-Framework-CPP/.cpmcache/yaml-cpp/905940a232aea3437e2fb77334461b1fe343318b /home/ubuntu/git/ML-Framework-CPP/build /home/ubuntu/git/ML-Framework-CPP/build/_deps/yaml-cpp-build /home/ubuntu/git/ML-Framework-CPP/build/_deps/yaml-cpp-build/CMakeFiles/NightlyUpdate.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : _deps/yaml-cpp-build/CMakeFiles/NightlyUpdate.dir/depend

