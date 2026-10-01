# 03: The Root CMake and Testing Harness

This document breaks down every single line of our root `CMakeLists.txt` file. For each command, we will first explore what the function does generally, look at its most common options, and then analyze exactly why we used it the way we did.

---

## 1. `cmake_minimum_required`
**What it is:** A built-in CMake command that establishes the minimum version of CMake required to build the project. Crucially, it also sets CMake's internal "policies" to match the behavior of that specific version, ensuring backward compatibility doesn't break modern features.
**Common Options:**
*   `VERSION <min>`: The standard way to set the floor version.
*   `FATAL_ERROR`: An older option to force CMake to crash if the version is too low (now the default behavior).

**Our Code:**
```cmake
cmake_minimum_required(VERSION 3.25)
```
**Why we did it:** We enforce version 3.25 specifically because it is the exact version that introduced native support for the `SYSTEM` keyword in `FetchContent`. If a user with CMake 3.24 tries to build our project, the `SYSTEM` keyword would be ignored, and Eigen's warnings would crash our `-Werror` build.

---

## 2. `project`
**What it is:** Initializes the CMake project. It sets the project name, version, and configures the internal toolchains for the programming languages you specify.
**Common Options:**
*   `<PROJECT-NAME>`: The name of the project.
*   `LANGUAGES <lang...>`: Specifies which compilers to look for (e.g., `CXX` for C++, `C` for C).

**Our Code:**
```cmake
project(vio CXX)
```
**Why we did it:** We named our project `vio`. By explicitly specifying `CXX`, we tell CMake that this is a pure C++ project. If we didn't specify `CXX`, CMake would default to searching for both C and C++ compilers, wasting time during configuration.

---

## 3. `set`
**What it is:** Defines or modifies a CMake variable. These variables dictate how CMake behaves globally or within specific directories.
**Common Options:**
*   `<variable> <value>`: Sets a standard variable in the current scope.
*   `CACHE <type> <docstring>`: Sets a cache variable that is saved to `CMakeCache.txt` and can be overridden by the user via the terminal or a GUI.

**Our Code:**
```cmake
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
```
**Why we did it:** 
*   `CMAKE_CXX_STANDARD 20` tells CMake to automatically pass the `-std=c++20` flag to the Clang compiler.
*   `CMAKE_CXX_STANDARD_REQUIRED ON` is a strict safety net. Without it, if a user's compiler doesn't support C++20, CMake will silently fall back to compiling in C++17 or C++14. This flag forces CMake to halt and throw a fatal error instead of compromising our modern C++ idioms.

---

## 4. `include`
**What it is:** Loads and executes CMake code from another file or module. It behaves exactly as if you copy-pasted the contents of that external file into the current spot.
**Common Options:**
*   `OPTIONAL`: If the file doesn't exist, CMake will just ignore it instead of throwing a fatal error.

**Our Code:**
```cmake
include(cmake/Warnings.cmake)
include(cmake/Dependencies.cmake)
```
**Why we did it:** We use this purely for modularity. By moving the verbose dependency fetching and warning configurations into their own files, the root `CMakeLists.txt` remains a clean, high-level map of our actual C++ targets.

---

## 5. `add_library`
**What it is:** Instructs CMake to take a list of source files and compile them into a library target.
**Common Options:**
*   `STATIC`: Compiles the code into an archive (`.a` on Unix/MinGW, `.lib` on MSVC). The code is baked directly into the final executable at link time.
*   `SHARED`: Compiles the code into a dynamic library (`.so`, `.dll`). It is loaded by the operating system at runtime, making executables smaller but deployment harder.
*   `INTERFACE`: Creates a "virtual" library that produces no compiled output. It is used solely to bundle headers or compiler flags together.

**Our Code:**
```cmake
# ---------------------------------------------------------
# VIO Core Library Target
# ---------------------------------------------------------
add_library(vio_core STATIC src/dummy.cpp)
```
**Why we did it:** We chose `STATIC` because baking our math directly into our final executables makes testing and deploying the drone software much simpler. 
*Why the dummy file?* CMake needs to determine a "link language" to know which linker to use. If you provide an empty target, CMake crashes with `Cannot determine link language`. Providing a blank `src/dummy.cpp` gives CMake exactly what it needs to successfully configure the target until we write our real Lie group math in Milestone 2.

---

## 6. `target_link_libraries`
**What it is:** The most powerful command in modern CMake. It specifies which targets depend on which other targets. Crucially, it dictates how properties (include directories, compiler flags, linked libraries) propagate through your project graph.
**Common Options:**
*   `PRIVATE`: The dependency is used to build the target, but is **not** exposed to anyone who links against this target.
*   `PUBLIC`: The dependency is used to build the target, **and** is exposed to anyone who links against this target.
*   `INTERFACE`: The dependency is **not** used to build the target itself, but **is** exposed to anyone who links against it.

**Our Code:**
```cmake
target_link_libraries(vio_core
    PUBLIC Eigen3::Eigen
    PRIVATE vio_warnings
)
```
**Why we did it:** 
*   `PUBLIC Eigen3::Eigen`: `vio_core` needs Eigen's math headers to compile. Furthermore, any target that uses `vio_core` (like our test executable) will *also* need Eigen's headers. Making it `PUBLIC` ensures CMake automatically hands Eigen's include paths to everything downstream.
*   `PRIVATE vio_warnings`: We want `vio_core` to be compiled using our strict `-Werror` flags. However, we made it `PRIVATE` so that downstream consumers of our library don't forcefully inherit our strict warning rules against their will.

---

## 7. `enable_testing` and `add_executable`
**What it is:** `enable_testing()` tells CMake to generate the underlying infrastructure required by the `ctest` test runner. `add_executable` compiles source files into a standalone runnable program (an `.exe`), unlike `add_library`.

**Our Code:**
```cmake
# ---------------------------------------------------------
# Testing Infrastructure
# ---------------------------------------------------------
enable_testing()

# Milestone 1 Sanity Test executable
add_executable(test_milestone1 tests/test_milestone1.cpp)
```
**Why we did it:** Without `enable_testing()`, typing `ctest` in your terminal would do absolutely nothing. We created a standalone executable for our Milestone 1 test runner. 

```cmake
target_link_libraries(test_milestone1 
    PRIVATE Catch2::Catch2WithMain
    PRIVATE vio_core
    PRIVATE vio_warnings
)
```
We used `PRIVATE` for all links here because an executable is the end of the line—nothing links against an executable, so visibility modifiers don't really matter. Note that we link against `Catch2::Catch2WithMain`, which is a special Catch2 target that automatically injects a `main()` function into our executable so we don't have to write one ourselves.

---

## 8. Catch2 Test Discovery
**What it is:** Traditionally, you had to manually register every single test with CMake using the `add_test()` command. `catch_discover_tests()` is a magical CMake function provided by Catch2.
**Our Code:**
```cmake
include(Catch)
catch_discover_tests(test_milestone1)
```
**Why we did it:** `include(Catch)` loads Catch2's custom CMake scripts. `catch_discover_tests()` tells CMake to run your compiled executable in the background as a pre-build step, ask it to list all the `TEST_CASE` macros it contains, and dynamically register them with CTest. This allows you to just write C++ code, and CTest instantly knows about it without you ever updating `CMakeLists.txt`.

---

## Appendix: Eigen Math & Floating-Point Equality
In your initial sanity test, you wrote:
```cpp
REQUIRE(x == Eigen::Vector3d(2.6, 3.0, 4.2));
```
In this *exact* scenario (`2.1 * 2`), this assertion passes. Because `2` is a perfect power of two, binary floating-point multiplication just increments the exponent without altering the mantissa (the decimal precision). 

However, using the `==` operator on floating-point numbers is a cardinal sin in robotics. Because computers represent decimals in base-2, numbers like `0.1` cannot be stored perfectly. `0.1 + 0.2` will often equal `0.30000000000000004` in memory, causing a strict `==` check to fail and crashing your Kalman Filter.

**The Solution:** Eigen provides a built-in method for safe, tolerance-based floating-point comparison:
```cpp
REQUIRE( x.isApprox(Eigen::Vector3d(2.6, 3.0, 4.2)) );
```
This checks if the values are identical within a microscopic threshold (machine epsilon), ensuring your math tests don't randomly fail due to CPU architecture quirks.
