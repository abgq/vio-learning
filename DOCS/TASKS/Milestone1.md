# Milestone 1 Tasks: Project Skeleton & Strict Environment

This document tracks all the actionable steps you performed to satisfy the Exit Criteria for Milestone 1.

- [x] **Task 1: Build the CMake Architecture**
  - Create `CMakePresets.json` with `debug-sanitize` and `release` profiles.
  - Create `cmake/Warnings.cmake` to enforce strict `-Werror` and `-Wpedantic` rules.
  - Create `cmake/Dependencies.cmake` using `FetchContent` to download Eigen 3.4.0 and Catch2 v3.5.2 (using the `SYSTEM` keyword).
  - Create the root `CMakeLists.txt` to wire everything together into a `vio_core` library and `test_milestone1` executable.

- [x] **Task 2: The Eigen Sanity Test**
  - Create `tests/test_milestone1.cpp`.
  - Perform basic fixed-size matrix math to prove Eigen compiles successfully under C++20.
  - Learn to use `.isApprox()` instead of `==` for floating-point comparisons.
  - Fix the `-Wc2y-extensions` error caused by Catch2's `__COUNTER__` macro.

- [x] **Task 3: The Manual Sanitizer Proof**
  - Deliberately introduce an out-of-bounds array read (ASan) and a signed integer overflow (UBSan) into the test.
  - Use the `volatile` keyword to trick the compiler's static analysis, forcing the bugs to be evaluated at runtime.
  - Suppress warnings using the modern `[[maybe_unused]]` C++17 attribute.
  - Confirm that `ctest --preset debug-sanitize` successfully catches the memory corruption and prints a stack trace, proving the environment is secure.

- [ ] **Task 4: The Allocation Guard (CURRENT TASK)**
  - Open `tests/test_milestone1.cpp` and include `tests/support/alloc_guard.hpp` (which intercepts global `operator new` and `operator delete`).
  - Write a second Catch2 `TEST_CASE` named `"Allocation guard check"`.
  - Inside the test:
    1. Call `vio::AllocStats::reset()`.
    2. Deliberately trigger a heap allocation (e.g., `std::vector<int> v(100);`).
    3. Assert that it was caught: `REQUIRE(vio::AllocStats::allocations > 0);`.
    4. Call `vio::AllocStats::reset()` again.
    5. Perform a fixed-size Eigen matrix multiplication (e.g., `Eigen::Matrix3d m = Eigen::Matrix3d::Identity() * 2.0;`).
    6. Assert that Eigen did **not** touch the heap: `REQUIRE(vio::AllocStats::allocations == 0);`.

- [ ] **Task 5: Wrap up Milestone 1**
  - Verify all tests pass with `ctest --preset debug-sanitize`.
  - Open `PROGRESS.md` and change the status of Milestone 1 to `Completed`.
  - Run `git add .` and `git commit -m "Complete Milestone 1: CMake, Catch2, Eigen, and memory discipline setup"` to commit the milestone.
