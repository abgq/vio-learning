# ==============================================================================
# Visual-Inertial Odometry (VIO) Pipeline
# Core CMake Configuration
# ==============================================================================
cmake_minimum_required(VERSION 3.25)
project(vio CXX)

# Enforce modern C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include configuration scripts
include(cmake/Warnings.cmake)
include(cmake/Dependencies.cmake)

# ---------------------------------------------------------
# VIO Core Library Target
# ---------------------------------------------------------
add_library(vio_core STATIC src/dummy.cpp)

target_include_directories(vio_core PUBLIC include)

# Propagate Eigen headers downstream, but keep strict warnings private
target_link_libraries(vio_core
    PUBLIC Eigen3::Eigen
    PRIVATE vio_warnings
)

# ---------------------------------------------------------
# Testing Infrastructure
# ---------------------------------------------------------
enable_testing()

# Milestone 1 Sanity Test executable
add_executable(test_milestone1 tests/test_milestone1.cpp)

target_link_libraries(test_milestone1 
    PRIVATE Catch2::Catch2WithMain
    PRIVATE vio_core
    PRIVATE vio_warnings
)

# Register tests dynamically with CTest
include(Catch)
catch_discover_tests(test_milestone1)

add_executable(test_milestone2 tests/test_milestone2.cpp)
target_link_libraries(test_milestone2 
    PRIVATE Catch2::Catch2WithMain
    PRIVATE vio_core
    PRIVATE vio_warnings
)
catch_discover_tests(test_milestone2)