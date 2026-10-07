/**
 * @file numerical_jacobian.hpp
 * @brief Finite-difference Jacobian computation using local (right) perturbations.
 */

#pragma once

#include <Eigen/Core>
#include <functional>

namespace vio {

/**
 * @brief Computes the numerical Jacobian of f(x) with respect to x.
 * 
 * Uses central differences:
 * J_i = (f(x boxplus (eps * e_i)) boxminus f(x boxplus (-eps * e_i))) / (2 * eps)
 * 
 * For simple vector states, boxplus is addition and boxminus is subtraction.
 * For SO(3) rotations, boxplus is q * Exp_q(delta), and boxminus is Log_q(q1^-1 * q2).
 */
template <typename T_out, typename T_in, typename BoxPlus, typename BoxMinus>
[[nodiscard]] auto numerical_jacobian(
    const std::function<T_out(const T_in&)>& f,
    const T_in& x,
    const BoxPlus& boxplus,
    const BoxMinus& boxminus,
    int dim_in,
    int dim_out,
    double eps = 1e-6)
{
    Eigen::MatrixXd J(dim_out, dim_in);
    
    // TODO: Implement central finite difference
    
    return J;
}

} // namespace vio
