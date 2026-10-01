/**
 * @file alloc_guard.hpp
 * @brief Tier 2 Memory Discipline Enforcement.
 * 
 * Overrides global operator new and operator delete to intercept and count
 * all heap memory allocations across the program. Used in test cases to 
 * verify that the core Kalman Filter hot-paths operate with exactly zero 
 * dynamic allocations.
 */
#pragma once
#include <new>
#include <cstdlib>

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

// Override global new/delete to intercept all dynamic memory usage
void* operator new(std::size_t size) {
    vio::AllocStats::allocations++;
    void* p = std::malloc(size);
    if (!p) throw std::bad_alloc();
    return p;
}

void operator delete(void* p) noexcept {
    vio::AllocStats::deallocations++;
    std::free(p);
}

void operator delete(void* p, std::size_t) noexcept {
    vio::AllocStats::deallocations++;
    std::free(p);
}
