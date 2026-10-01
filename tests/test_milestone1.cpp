/**
 * @file test_milestone1.cpp
 * @brief Sanity checks for the VIO project environment.
 * 
 * Verifies that Catch2 is running, Eigen is compiling under C++20 without
 * warnings, and that the Tier 2 memory discipline (zero heap allocations)
 * is strictly enforced.
 */

#include <catch2/catch_test_macros.hpp>
#include <Eigen/Core>
#include "support/alloc_guard.hpp"

TEST_CASE("Eigen sanity check", "[milestone1]")
{
    Eigen::Vector3d x;
    
    x(0) = 1.3;
    x(1) = 1.5;
    x(2) = 2.1;

    x = 2 * x;

    // Use isApprox for safe floating-point comparison
    REQUIRE(x.isApprox(Eigen::Vector3d(2.6, 3.0, 4.2)));
}

TEST_CASE("Allocation guard check", "[milestone1]")
{
    // 1. Prove the tracker catches dynamic allocations (std::vector)
    vio::AllocStats::reset();
    std::vector<int> v(100);
    REQUIRE(vio::AllocStats::allocations > 0);

    // 2. Prove fixed-size Eigen math obeys Tier 2 Memory Discipline (zero heap)
    vio::AllocStats::reset();
    Eigen::Matrix3d m = Eigen::Matrix3d::Identity() * 2.0;
    REQUIRE(vio::AllocStats::allocations == 0);
}