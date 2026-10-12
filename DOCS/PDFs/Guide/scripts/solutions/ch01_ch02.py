"""Exercise solutions, Chapters 1 and 2: prints the numbers quoted in the worked solutions.

Run from DOCS/PDFs/Guide:  .venv/Scripts/python.exe scripts/solutions/<this file>
"""
import numpy as np
from scipy.optimize import brentq

g = 9.81
# 1.1
ba = 5 * 9.81e-3
t = np.sqrt(2 / ba)
bg = 6 / (g * t**3)
print(f"1.1 ba={ba:.5f} m/s2, t={t:.3f} s, bg={bg:.4e} rad/s = {np.degrees(bg):.4f} deg/s = {np.degrees(bg)*3600:.1f} deg/h; check 3ba/(g t)={3*ba/(g*t):.4e}")
# 1.2  first-order g*th vs exact horizontal g*sin(th): 1% error
th = brentq(lambda x: x / np.sin(x) - 1.01, 0.01, 1.0)
print(f"1.2 tilt for 1% error (horizontal component): {th:.4f} rad = {np.degrees(th):.2f} deg")
th2 = brentq(lambda x: x / (2 * np.sin(x / 2)) - 1.01, 0.01, 2.0)
print(f"    vs magnitude 2g sin(th/2): {th2:.4f} rad = {np.degrees(th2):.2f} deg")
# 2.2
a, b, c = np.array([1, 2, 3.]), np.array([4, 5, 6.]), np.array([7, 8, 9.])
print("2.2 det", np.linalg.det(np.column_stack([a, b, c])), "combo", a - 2 * b + c)
# 2.3 Gram-Schmidt
A = np.array([[1, 1, 0.], [1, 0, 1.], [0, 1, 1.]])
Q = []
for v in A:
    w = v - sum((q @ v) * q for q in Q)
    Q.append(w / np.linalg.norm(w))
Q = np.array(Q)
print("2.3", np.round(Q * np.array([[np.sqrt(2)], [np.sqrt(6)], [np.sqrt(3)]]), 6), "\n orthonormal:", np.allclose(Q @ Q.T, np.eye(3)))
# 2.5 Jacobi
rng = np.random.default_rng(0)
a, b, c = rng.normal(size=(3, 3))
print("2.5 Jacobi residual", np.linalg.norm(np.cross(a, np.cross(b, c)) + np.cross(b, np.cross(c, a)) + np.cross(c, np.cross(a, b))))
# 2.8
w, V = np.linalg.eigh(np.array([[2, 1], [1, 2.]]))
print("2.8", w, V)
