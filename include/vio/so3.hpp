/**
 * @file so3.hpp
 * @brief 3D Lie Group Math on SO(3)
 */

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace vio {
namespace so3 {

[[nodiscard]] constexpr Eigen::Matrix3d skew(const Eigen::Vector3d& w) noexcept {
    // TODO: Implement skew-symmetric matrix
    return Eigen::Matrix3d::Zero();
}

[[nodiscard]] Eigen::Matrix3d Exp(const Eigen::Vector3d& phi) noexcept {
    // TODO: Implement SO(3) Exponential map (with Taylor expansion for small angles)
    return Eigen::Matrix3d::Identity();
}

[[nodiscard]] Eigen::Vector3d Log(const Eigen::Matrix3d& R) noexcept {
    // TODO: Implement SO(3) Logarithm map (with Taylor expansion for small angles)
    return Eigen::Vector3d::Zero();
}

[[nodiscard]] Eigen::Quaterniond Exp_q(const Eigen::Vector3d& phi) noexcept {
    // TODO: Implement quaternion Exponential map (with Taylor expansion for small angles)
    return Eigen::Quaterniond::Identity();
}

[[nodiscard]] Eigen::Vector3d Log_q(const Eigen::Quaterniond& q) noexcept {
    // TODO: Implement quaternion Logarithm map (handle double cover and small angles)
    return Eigen::Vector3d::Zero();
}

[[nodiscard]] Eigen::Matrix3d J_r(const Eigen::Vector3d& phi) noexcept {
    // TODO: Implement right Jacobian of SO(3)
    return Eigen::Matrix3d::Identity();
}

[[nodiscard]] Eigen::Matrix3d J_r_inv(const Eigen::Vector3d& phi) noexcept {
    // TODO: Implement inverse right Jacobian of SO(3)
    return Eigen::Matrix3d::Identity();
}

} // namespace so3
} // namespace vio
