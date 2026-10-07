/**
 * @file test_milestone2.cpp
 * @brief Tests for 3D Lie Group Math on SO(3).
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "vio/so3.hpp"
#include "support/alloc_guard.hpp"
#include "support/numerical_jacobian.hpp"

using namespace vio;

TEST_CASE("SO(3) Skew-Symmetric", "[milestone2]")
{
    // TODO: Implement skew(a) * b == a.cross(b)
}

// TODO: Add more test cases for Exp, Log, Exp_q, Log_q, J_r, J_r_inv
