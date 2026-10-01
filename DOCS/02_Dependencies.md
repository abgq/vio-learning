# 02: Dependency Management (`cmake/Dependencies.cmake`)

This document explains how we handle third-party libraries (Eigen and Catch2) in our project using CMake's `FetchContent` module.

## 1. The `FetchContent` Module
```cmake
include(FetchContent)
```
Before we can use any fetching commands, we must include the built-in `FetchContent` module. This module allows CMake to download source code from the internet (e.g., GitHub, GitLab) during the configure step, and then build it directly alongside our own code. This guarantees that anyone building our project gets the exact same version of the dependencies, without having to manually install them on their operating system.

## 2. Declaring the Dependency
```cmake
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    SYSTEM
)
```
`FetchContent_Declare` tells CMake *where* to get the code and *what version* to get.
*   **`GIT_REPOSITORY`**: The URL of the library's repository.
*   **`GIT_TAG`**: We strictly pin this to a specific release (e.g., `3.4.0` for Eigen, `v3.5.2` for Catch2) rather than a moving target like `main` or `master`. This ensures our build remains deterministic and won't suddenly break if the library authors push an incompatible update.
*   **`SYSTEM`**: This is a critical keyword in modern CMake (≥ 3.25). By default, CMake includes dependency headers using the `-I` compiler flag. If the compiler finds anything in those headers that violates our strict `-Werror` warnings, it will throw an error and crash our build. Adding `SYSTEM` tells CMake to include the headers using the `-isystem` flag instead. Compilers have a hardcoded rule to **never emit warnings for system headers**, effectively silencing any complaints about code we didn't write.

## 3. Configuring and Making Available
```cmake
set(EIGEN_BUILD_DOC OFF)
set(EIGEN_BUILD_PKGCONFIG OFF)
set(BUILD_TESTING OFF) 

FetchContent_MakeAvailable(eigen)
```
Once declared, we must actually bring the code into our build using `FetchContent_MakeAvailable`. Under the hood, this downloads the code and essentially executes the dependency's own `CMakeLists.txt` file as a subdirectory of our project.

*   **Variable Creation in CMake**: CMake is a dynamically typed scripting language. You do not need to "declare" a variable before using it. When you call `set(EIGEN_BUILD_DOC OFF)`, CMake simply creates that variable in the current scope if it doesn't already exist.
*   **Overriding Defaults**: Dependencies often come with their own test suites, documentation builders, and examples (which are often turned `ON` by default). We don't want to waste time compiling Eigen's or Catch2's internal tests. By calling `set()` *before* we call `MakeAvailable`, we inject our preferences into the CMake scope. 
*   **How Dependencies Read Variables**: Inside Eigen's `CMakeLists.txt`, they use CMake's `option()` command (e.g., `option(EIGEN_BUILD_DOC "Enable creation of Eigen documentation" ON)`). This command means: *"If the user hasn't already provided a value for this variable, create it and set it to `ON`."* Because we set it to `OFF` earlier in our file, the `option()` command respects our choice and does nothing.
*   **Discovering Default Values**: If you omit the `set()` lines entirely, the variables will simply fall back to whatever default the library author chose. If you want to find out what variables a library exposes and what their defaults are, you have two options:
    1. Look at the library's source code directly (search for `option(` inside their `CMakeLists.txt`).
    2. Check the **CMake Cache**. After you configure your project for the first time, CMake generates a `CMakeCache.txt` file in your `build/` directory. You can open this file in a text editor to see a list of every configurable variable in your entire project, including those from fetched dependencies. You can also view these easily using interactive terminal tools like `ccmake` or GUI tools like `cmake-gui`.
