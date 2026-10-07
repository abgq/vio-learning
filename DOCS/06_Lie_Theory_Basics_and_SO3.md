# 06: Lie Theory Basics and SO(3) Math

This document explains the mathematical foundations of 3D rotations as they are used in our Visual-Inertial Odometry (VIO) pipeline.

## 1. The Core Problem: Rotations are not "Flat"

Imagine you are standing on the Earth (a sphere) and you want to describe a movement. If you walk 1 meter North, then 1 meter East, you don't end up in the exact same spot as walking 1 meter East, then 1 meter North, due to the curvature of the Earth. 

Similarly, **3D rotations do not commute**. Rotating a drone 90 degrees around X, then 90 degrees around Y, results in a completely different orientation than doing Y then X. 

Because of this curvature:
* You cannot just add two rotation matrices together: $R_1 + R_2$ is not a valid rotation.
* You cannot just add two quaternions together: $q_1 + q_2$ won't have a length of 1.
* Using Euler Angles (Roll, Pitch, Yaw) seems like you can just add them, but they suffer from **Gimbal Lock** (singularities where you lose a degree of freedom) and their math behaves wildly near 90-degree pitches.

**The Lie Theory Solution:** 
We store the "global" orientation of the drone using a robust representation (a Quaternion $q$), which lives on a curved mathematical surface called a **Manifold** (specifically, the $SO(3)$ manifold). 
However, when our Kalman Filter wants to estimate a tiny error or make a tiny update, it calculates a 3D vector $\delta\theta = [\delta x, \delta y, \delta z]$ in a **flat tangent space** (like a flat map of a small town on Earth). We then "wrap" that flat 3D vector around the curved manifold to apply the update.

---

## 2. The Functions in `so3.hpp`

### `skew(w)`: The Cross-Product Matrix
If you have two 3D vectors $a$ and $b$, their cross product is $a \times b$. 
We can rewrite this operation as a matrix multiplication: $a \times b = [a]_\times b$.
The `skew(w)` function takes a 3D vector $w = [x, y, z]^T$ and returns the 3x3 **skew-symmetric matrix** $[w]_\times$:
```math
[w]_\times = \begin{bmatrix} 0 & -z & y \\ z & 0 & -x \\ -y & x & 0 \end{bmatrix}
```
This matrix belongs to the "Lie Algebra" $\mathfrak{so}(3)$. It is the mathematical bridge that allows us to turn 3D vectors into rotation rates.

### `Exp(phi)` and `Exp_q(phi)`: The Exponential Map
The **Exponential Map** takes a 3D vector $\phi$ from the flat tangent space and wraps it onto the curved manifold, returning a valid rotation matrix (`Exp`) or Quaternion (`Exp_q`). 
*   **What is $\phi$?** It is an "axis-angle" vector. Its direction is the axis of rotation, and its length (magnitude) is the angle of rotation in radians.
*   **How it works (Rodrigues' Formula):** For matrices, `Exp` computes the infinite Taylor series of $e^{[w]_\times}$. Fortunately, Rodrigues' rotation formula gives us a closed-form solution so we don't have to compute infinity!
*   **How we use it:** To update our drone's orientation with an error $\delta\theta$, we do **not** do $q = q + \delta\theta$. We do $q = q \otimes \mathrm{Exp}_q(\delta\theta)$. (This is the **Local Right Perturbation** rule defined in `AGENTS.md` C4).

### `Log(R)` and `Log_q(q)`: The Logarithm Map
The **Logarithm Map** is the exact inverse of the Exponential map. It takes a rotation on the manifold (a Matrix or Quaternion) and unwraps it, returning the shortest 3D axis-angle vector $\phi$ that represents that rotation.
*   **How we use it:** If we have a measured quaternion from a sensor ($q_{meas}$) and our estimated quaternion ($\hat{q}$), we can't just subtract them to find the error. Instead, we find the rotation between them ($\hat{q}^{-1} \otimes q_{meas}$) and then use `Log_q` to flatten that difference into a 3D vector that our Kalman Filter can actually digest.

### `J_r(phi)` and `J_r_inv(phi)`: The Right Jacobian
In standard algebra, $e^{a+b} = e^a e^b$. 
Because rotations don't commute, $\mathrm{Exp}(\phi_1 + \phi_2) \neq \mathrm{Exp}(\phi_1)\mathrm{Exp}(\phi_2)$.
If we have a rotation $\phi$ and we add a tiny perturbation $\delta\phi$ to it, how does that translate to the manifold?
It turns out:
```math
\mathrm{Exp}(\phi + \delta\phi) \approx \mathrm{Exp}(\phi) \otimes \mathrm{Exp}(J_r(\phi) \delta\phi)
```
The **Right Jacobian** $J_r(\phi)$ is a 3x3 matrix that maps a perturbation in the flat vector space into a local multiplicative perturbation on the manifold. We need this heavily when deriving the Jacobians $F_c$ and $H$ for our EKF later, so we build it now.

---

## 3. The "Small Angle" Problem (Taylor Expansions)
If you look at the formulas for `Exp` and `Log` (which you'll find online or I can provide), they divide by the angle $\theta = \|\phi\|$. 
If $\theta = 0$ (no rotation), you get a division by zero (`NaN`), which instantly crashes the filter. Even if $\theta$ is just very small ($10^{-8}$), floating-point division becomes extremely inaccurate.
To fix this, whenever $\theta$ is close to zero (e.g., $< 10^{-6}$), your code must branch to an `if` statement and use the **Taylor series expansion** of the function, which safely bypasses the division by zero.
