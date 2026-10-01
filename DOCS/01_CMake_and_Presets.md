# 01: CMake Infrastructure and Presets

This document explains the foundational build files we set up in Milestone 1: `cmake/Warnings.cmake` and `CMakePresets.json`.

## 1. Target-Scoped Warnings (`cmake/Warnings.cmake`)

Instead of forcing compiler warnings globally across the entire project (which would infect third-party code like Eigen and Catch2), we use **target-scoped warnings**.

```cmake
add_library(vio_warnings INTERFACE)
target_compile_options(vio_warnings INTERFACE
    -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion 
    -Wold-style-cast -Wnon-virtual-dtor -Wdouble-promotion -Werror -Wno-c2y-extensions
)
```

*   **`INTERFACE` Target**: We created a "virtual" library target called `vio_warnings`. It doesn't compile any source code of its own (`INTERFACE`). It purely exists to hold properties (like compiler flags) and propagate them to any other target that links against it.
*   **`-Werror`**: This turns all warnings into hard compiler errors. In robotics, a warning is just a bug waiting to happen.

### The Compiler Flags Explained
Here is exactly what each flag in our strict warning set does:
*   **`-Wall -Wextra`**: The standard baseline. These catch common logical errors (like uninitialized variables, unused parameters, or misleading indentation).
*   **`-Wpedantic`**: Enforces strict ISO C++ compliance, rejecting compiler-specific extensions so our code remains portable.
*   **`-Wshadow`**: Triggers if a local variable has the same name as a variable in an outer scope (or a class member). Shadowing causes subtle bugs where you think you are updating the class state, but you are actually updating a temporary local variable.
*   **`-Wconversion`**: Warns about implicit conversions that may alter a value (e.g., converting a `double` to an `int` or a `float`, losing precision or truncating the value).
*   **`-Wsign-conversion`**: Warns about implicit conversions between signed and unsigned integers (e.g., `int` to `size_t`). Mixing these in loops or array indexing is a classic source of memory bugs.
*   **`-Wold-style-cast`**: Bans C-style casts like `(int)x`. In modern C++, you must use `static_cast`, `reinterpret_cast`, etc., which are explicit and searchable.
*   **`-Wnon-virtual-dtor`**: Warns if a class has virtual functions but a non-virtual destructor. If you delete a derived class through a base class pointer without a virtual destructor, you will get a memory leak because the derived destructor will not execute.
*   **`-Wdouble-promotion`**: Warns if a `float` is implicitly promoted to a `double`. This is critical in high-performance or embedded code where you want strict control over precision and execution speed.
*   **`-Wno-c2y-extensions`**: Disables the warning for C2y (C++26) extensions. We specifically had to add this because Catch2's internal macros use `__COUNTER__`, which triggered this warning under our strict `-Wpedantic` flag. (See Note 04 for a deep dive into this).

## 2. Deterministic Build Environments (`CMakePresets.json`)

CMake Presets define standardized configurations so you don't have to type long strings of compiler flags into the terminal. Everyone who clones the repository gets the exact same build environments.

### The `base` Configure Preset
```json
{
    "name": "base",
    "hidden": true,
    "generator": "Ninja",
    "cacheVariables": {
        "CMAKE_CXX_COMPILER": "clang++",
        "CMAKE_EXPORT_COMPILE_COMMANDS": "ON"
    }
}
```
*   **`hidden: true`**: This preset isn't meant to be used directly. It acts as a base class that other presets inherit from.
*   **`Ninja` & `clang++`**: We explicitly mandate the Ninja build system and the Clang compiler (matching our MSYS2 CLANG64 toolchain requirement).

### The `debug-sanitize` Preset
This is our primary development preset. It inherits from `base` and activates the **Address Sanitizer (ASan)** and **Undefined Behavior Sanitizer (UBSan)**.

> **Background: What are Sanitizers and Profilers?**
> *   **Sanitizers** are specialized tools built into the compiler (like Clang or GCC) that hunt for bugs. When enabled, the compiler injects extra checks into your program to monitor execution at runtime. For example, the Address Sanitizer (ASan) tracks memory allocations and will instantly crash your program if you try to read an array out of bounds. The Undefined Behavior Sanitizer (UBSan) watches for mathematically illegal operations, like dividing by zero or overflowing an integer. 
> *   **Profilers** are performance analysis tools. They watch your program run and measure where it spends the most time (e.g., "Function A took 40% of the CPU time"). They help you find bottlenecks to optimize.

```json
{
    "name": "debug-sanitize",
    "inherits": "base",
    "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Debug",
        "CMAKE_CXX_FLAGS": "-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all -DEIGEN_RUNTIME_NO_MALLOC",
        "CMAKE_EXE_LINKER_FLAGS": "-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all"
    }
}
```
*   **Sanitizers**: `-fsanitize=address,undefined` turns on both ASan and UBSan. (Note that we apply these flags to both compile *and* link steps).

> **Background: What is a Frame Pointer?**
> When a program runs, it uses a block of memory called the "call stack" to keep track of which function called which. Every time a function is called, it gets a chunk of memory on the stack called a **stack frame** (which holds its local variables). 
> The **frame pointer** is a dedicated CPU register that points to the start of the current function's stack frame. By following the frame pointers backwards in memory, a debugging tool can easily reconstruct the call stack and tell you the exact sequence of function calls (e.g., `main()` called `update()` called `math_logic()`).

*   **`-fno-omit-frame-pointer`**: Because maintaining the frame pointer costs a tiny bit of performance (it takes up a CPU register), compilers often try to "omit" or optimize it away. However, sanitizers and profilers rely heavily on the frame pointer to quickly walk up the call stack when a bug happens. By forcing the compiler to keep it (`-fno-omit-frame-pointer`), we guarantee that when the sanitizer crashes our program, it can print a readable, accurate stack trace showing exactly where the crash originated.
*   **`-fno-sanitize-recover=all`**: By default, UBSan just prints a warning and lets the program continue running. This flag forces the program to crash immediately when Undefined Behavior occurs. 
*   **`EIGEN_RUNTIME_NO_MALLOC`**: A custom Eigen flag. If Eigen tries to allocate memory dynamically (on the heap) at runtime, it will crash. We use this to strictly enforce our Tier 2 Memory Discipline (Zero heap allocations in the Core hot-path).

### The Test Presets
```json
"testPresets": [
    {
        "name": "debug-sanitize",
        "configurePreset": "debug-sanitize",
        "environment": {
            "UBSAN_OPTIONS": "print_stacktrace=1:halt_on_error=1"
        }
    }
]
```
*   **`testPresets`** map directly to our build presets and are invoked using `ctest --preset debug-sanitize`.
*   **`UBSAN_OPTIONS`**: This environment variable configures the runtime behavior of the Undefined Behavior Sanitizer during testing. 
    *   **`halt_on_error=1`**: Redundantly enforces that UBSan should kill the test runner immediately upon finding a bug. Without this, CTest might see the program finish with a 0 exit code and mark the test as "Passed", even if a UBSan violation was printed in the background. 
    *   **`print_stacktrace=1`**: Instructs UBSan to explicitly print the function call hierarchy leading up to the error. Combined with `-fno-omit-frame-pointer` (which makes the stack readable), this guarantees you will see the exact file and line number where the undefined behavior occurred.
