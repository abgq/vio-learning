# Milestone 2 Tasks: 3D Lie Group Math on SO(3)

This document breaks down the actionable steps required to complete Milestone 2. 
You will be implementing the core math for the Error-State Kalman Filter in `include/vio/so3.hpp` and its tests in `tests/test_milestone2.cpp`.

## Requirements & Math

### 1. Skew-Symmetric Matrix
- [ ] Implement `skew(const Eigen::Vector3d& w) -> Eigen::Matrix3d`.
- [ ] Test: `skew(a) * b == a.cross(b)`.

### 2. SO(3) Exponential and Logarithm
- [ ] Implement `Exp(const Eigen::Vector3d& phi) -> Eigen::Matrix3d`.
- [ ] Implement `Log(const Eigen::Matrix3d& R) -> Eigen::Vector3d`.
- **Note:** Use the Taylor expansion near `||phi|| == 0` for `Exp` and near `theta == 0` for `Log`.
- [ ] Test `Exp`: Output must be a valid rotation matrix (`R^T R = I` and `det(R) = 1`).
- [ ] Test `Log(Exp(phi)) == phi` for random `phi`, zero, near-zero, and near `pi`.
- [ ] Test `Exp(Log(R)) == R` for random `R` and near `pi`.

### 3. Quaternion Exponential and Logarithm
- [ ] Implement `Exp_q(const Eigen::Vector3d& phi) -> Eigen::Quaterniond`.
- [ ] Implement `Log_q(const Eigen::Quaterniond& q) -> Eigen::Vector3d`.
- **Note:** `Eigen::Quaterniond` coefficients are ordered `(w, x, y, z)` in constructor but `(x, y, z, w)` in `.coeffs()`.
- **Note:** Handle the double cover `q` and `-q`. Before `Log_q`, canonicalize `w >= 0`.
- [ ] Test: `Exp_q(phi)` matches `Exp(phi)` as a rotation matrix.
- [ ] Test: `Log_q(q) == Log_q(-q)`.

### 4. Right Jacobian and its Inverse
- [ ] Implement `J_r(const Eigen::Vector3d& phi) -> Eigen::Matrix3d`.
- [ ] Implement `J_r_inv(const Eigen::Vector3d& phi) -> Eigen::Matrix3d`.
- **Note:** Implement small-angle approximations.
- [ ] Test: `Exp(phi + delta) ≈ Exp(phi) * Exp(J_r(phi) * delta)`.
- [ ] Test: `J_r_inv(phi) * J_r(phi) == I`.

### 5. Numerical Differentiation
- [ ] Implement `numerical_jacobian` in `tests/support/numerical_jacobian.hpp` using the right/local perturbation operator `boxplus`.

### 6. Memory Discipline
- [ ] Wrap the math tests with `vio::AllocStats` to guarantee zero allocations.

### Instructions
1. Review the boilerplates provided in `include/vio/so3.hpp` and `tests/support/numerical_jacobian.hpp`.
2. Write the implementations in `so3.hpp`.
3. Add Catch2 test cases in `tests/test_milestone2.cpp`.
4. Modify `CMakeLists.txt` to compile `test_milestone2`.
5. Run tests with `ctest --preset debug-sanitize`.
6. Once everything passes and you feel you own the concepts, ask for a **checkpoint review**.
