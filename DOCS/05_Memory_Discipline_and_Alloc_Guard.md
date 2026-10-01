# 05: Memory Discipline and The Allocation Guard

This document breaks down the fundamental differences between Stack and Heap memory in C++, the mechanics of global operator overloading, and how our `alloc_guard.hpp` utility enforces our strict robotics memory discipline.

---

## 1. The Stack vs. The Heap
Before understanding the allocation guard, we must understand how C++ manages memory under the hood.

**What they are:**
*   **The Stack:** Fast, deterministic memory managed automatically by the CPU. When a function is called, the CPU simply moves a "stack pointer" down to reserve space for local variables (like `int x = 5;` or `Eigen::Vector3d v;`). When the function exits, the pointer moves back up, instantly freeing the memory. It takes a fraction of a nanosecond.
*   **The Heap:** Dynamic memory managed by the Operating System. When you use the `new` keyword or a `std::vector`, C++ has to ask the OS for memory. The OS must search its memory tables for a contiguous block of free space, lock it to prevent other threads from taking it, and return a pointer to it.

**Why we care (Tier 2 Memory Discipline):**
Searching the Heap for free space takes thousands of times longer than moving the Stack pointer, and the time it takes is non-deterministic (it varies wildly depending on memory fragmentation). 
In high-frequency robotics, a sudden microsecond delay from the OS heap manager can cause the Kalman Filter to miss a sensor reading, leading to mathematical divergence and a drone crash. Therefore, we strictly forbid Heap allocations in our Core math pipeline.

---

## 2. The One Definition Rule and `inline` Variables
To enforce our zero-heap rule, we need a global counter to track allocations.

**What the ODR is:**
The C++ "One Definition Rule" states that a variable or function can only be defined *once* across your entire compiled program. 
If you simply write `static int allocations = 0;` inside a header file, every single `.cpp` file that `#include`s that header gets its own completely separate copy of the variable. If File A increments it, File B won't see the change.

**The Modern C++ Solution:**
```cpp
namespace vio {

struct AllocStats {
    // C++17 inline variables make this easy to share across translation units
    static inline int allocations = 0;
    static inline int deallocations = 0;
    
    static void reset() {
        allocations = 0;
        deallocations = 0;
    }
};

} // namespace vio
```
**Why we did it:** 
In C++17, the language introduced `inline` variables. Putting `inline` on a static variable tells the Linker: *"You are going to see this variable defined in multiple `.cpp` files. Do not throw an error; instead, magically merge all of them into one single, universally shared global variable."* 
This allows our `alloc_guard.hpp` to act as a lightweight, header-only utility without needing its own `.cpp` file.

---

## 3. Overriding the Global Memory Operators
In C++, when you write `std::vector<int> v(100);` or `new MyClass();`, the language performs two distinct steps:
1. It requests raw, uninitialized bytes from the system by calling a global function named `operator new`.
2. It runs the object's constructor inside those bytes.

**What Global Overriding is:**
C++ allows us to completely replace step 1. By writing our own global `operator new`, we hijack every single heap allocation in the entire program—even the ones hidden deep inside the Standard Library or Eigen.

**Our Code:**
```cpp
void* operator new(std::size_t size) {
    vio::AllocStats::allocations++;
    void* p = std::malloc(size);
    if (!p) throw std::bad_alloc();
    return p;
}
```
*(Note: The C++ standard strictly forbids declaring global replacement operators as `inline`. Doing so makes the program "ill-formed". Because we use strict `-Werror` flags, Clang correctly caught this rule violation earlier and threw a `-Winline-new-delete` error, which is why we removed it!)*
**Why we did it:** 
*   We intercept the memory request and increment our `allocations` counter.
*   Because we hijacked the memory allocator, it is now our responsibility to actually find memory. We do this by dropping down to C and calling `std::malloc(size)`, asking the underlying C library for raw bytes.
*   The C++ Standard dictates that if the system is completely out of memory (i.e., `malloc` returns a null pointer), `operator new` must throw a `std::bad_alloc` exception rather than silently failing. We adhere strictly to that standard.

---

## 4. Unsized vs. Sized Deallocation
Just as we hijacked `new`, we must hijack `delete` to track when memory is freed.

**How does unsized delete know how much to free?**
You might look at the traditional `operator delete(void* p)` and wonder: *If `p` is just a pointer to the first byte, how does the system know how large the memory block is when freeing it?* 
The secret lies in how `std::malloc` works under the hood. When you ask `malloc` for memory, it actually allocates slightly more than you asked for. Just *before* the memory address it returns to you (at a negative offset), it stores a hidden "bookkeeping" header containing the exact size of the allocation. When you call `std::free(p)`, the C library looks just behind the pointer you gave it, reads that hidden size, and safely releases the block. 

**What Sized Deallocation is:**
Relying on `std::free` to read hidden headers causes cache misses and takes extra CPU cycles. Because the C++ compiler often knows *exactly* how big an object is at compile time (e.g., `delete myVector3d`), modern C++ (C++14 onwards) introduced "Sized Deallocation": `operator delete(void* p, std::size_t size)`. This allows the compiler to pass the size directly to the allocator, skipping the hidden header lookup for better performance.

**Our Code:**
```cpp
void operator delete(void* p) noexcept {
    vio::AllocStats::deallocations++;
    std::free(p);
}

void operator delete(void* p, std::size_t size) noexcept {
    vio::AllocStats::deallocations++;
    std::free(p);
}
```
**Why we did it:** 
We must provide overrides for *both* the unsized and sized versions of `operator delete`. If we only provided the unsized version, modern compilers would quietly bypass our tracker when doing sized deallocations, causing our `deallocations` count to be wrong. We increment our counter and pass the pointer back to `std::free()` to safely release it to the OS.

---

## 5. Why `std::vector` Uses the Heap (And How to Avoid It)
You might wonder: *If I write `std::vector<int> v(100);` as a local variable, why does it trigger a heap allocation? Can I allocate a vector on the stack?*

A `std::vector` actually consists of two distinct parts:
1. **The Control Block:** A small struct containing three pointers (start of data, end of data, and end of capacity). This 24-byte block *does* live on the stack.
2. **The Data Buffer:** The actual 100 integers. Because a `vector` is designed to dynamically grow at runtime (e.g., if you call `.push_back()`), the compiler cannot guarantee how much stack space it will ultimately need. Therefore, the vector's constructor reaches out to the OS (via `operator new`) to allocate the data buffer on the **heap**.

**The Stack Alternative:**
If you want to allocate exactly 100 integers entirely on the stack, you must use a container whose size is strictly fixed at compile time so the CPU knows exactly how far to move the stack pointer. In modern C++, you use `std::array`:
```cpp
std::array<int, 100> v; // 100% Stack memory. Zero heap allocations.
```
This is the exact same reason we use fixed-size `Eigen::Matrix3d` (stack) instead of dynamically sized `Eigen::MatrixXd` (heap) for our fast robotics math!
