"""Exercise solutions, Chapter 3: prints the numbers quoted in the worked solutions.

Run from DOCS/PDFs/Guide:  .venv/Scripts/python.exe scripts/solutions/<this file>
"""
import numpy as np
from scipy.linalg import expm

# 3.1 elimination with and without pivoting, plain floats
a11, a12, a21, a22, b1, b2 = 1e-20, 1.0, 1.0, 1.0, 1.0, 2.0
l = a21 / a11
u22 = a22 - l * a12
c2 = b2 - l * b1
x2 = c2 / u22
x1 = (b1 - a12 * x2) / a11
print("3.1 no pivot:", x1, x2, " u22 =", u22, " c2 =", c2)
l = a11 / a21
u22 = a12 - l * a22
c2 = b1 - l * b2
x2 = c2 / u22
x1 = (b2 - a22 * x2) / a21
print("3.1 pivot   :", x1, x2, " exact:", 1 / (1 - 1e-20), (1 - 2e-20) / (1 - 1e-20))

# 3.2
S = np.array([[4, 2, 2], [2, 5, 3], [2, 3, 6.]])
L = np.linalg.cholesky(S)
print("3.2 L=\n", L, "\n x =", np.linalg.solve(S, [8, 10, 11]))


# 3.4 Householder QR
def householder_qr(A):
    A = A.astype(float).copy()
    m, n = A.shape
    vs = []
    for k in range(n):
        x = A[k:, k].copy()
        alpha = -np.copysign(np.linalg.norm(x), x[0])
        v = x.copy()
        v[0] -= alpha
        v /= np.linalg.norm(v)
        A[k:, k:] -= 2.0 * np.outer(v, v @ A[k:, k:])
        vs.append(v)
    Q = np.eye(m)
    for k in reversed(range(n)):  # Q = H1 H2 ... Hn applied to I
        v = vs[k]
        Q[k:, :] -= 2.0 * np.outer(v, v @ Q[k:, :])
    return Q, np.triu(A[:n, :])


rng = np.random.default_rng(1)
A = rng.normal(size=(10, 3))
Q, R1 = householder_qr(A)
Q1, Q2 = Q[:, :3], Q[:, 3:]
print("3.4 ||Q^T Q - I|| =", np.linalg.norm(Q.T @ Q - np.eye(10)), " ||Q1 R1 - A|| =", np.linalg.norm(Q1 @ R1 - A),
      " ||Q2^T A|| =", np.linalg.norm(Q2.T @ A))
# ill-conditioned LS, kappa = 1e8, consistent system
U, _ = np.linalg.qr(rng.normal(size=(10, 3)))
V, _ = np.linalg.qr(rng.normal(size=(3, 3)))
A = U @ np.diag([1.0, 1e-4, 1e-8]) @ V.T
x_true = np.array([1.0, 2.0, 3.0])
b = A @ x_true
print("    cond(A) =", np.linalg.cond(A))
x_ne = np.linalg.solve(A.T @ A, A.T @ b)
Q, R1 = householder_qr(A)
x_qr = np.linalg.solve(R1, (Q[:, :3].T @ b))
print("    rel. error normal eq.:", np.linalg.norm(x_ne - x_true) / np.linalg.norm(x_true))
print("    rel. error QR       :", np.linalg.norm(x_qr - x_true) / np.linalg.norm(x_true))
# with a residual orthogonal to range(A)
r_perp = Q[:, 3:] @ rng.normal(size=7) * 1e-3
b2 = b + r_perp
x_ne = np.linalg.solve(A.T @ A, A.T @ b2)
x_qr = np.linalg.solve(R1, (Q[:, :3].T @ b2))
print("    with residual 1e-3: NE", np.linalg.norm(x_ne - x_true) / np.linalg.norm(x_true), " QR", np.linalg.norm(x_qr - x_true) / np.linalg.norm(x_true))

# 3.6 flop estimates
n = 75
for m in (2, 500):
    info = n**3 / 3 * 2 + 2 * m * n**2 + 2 * n**3  # invert P (chol+inverse ~) + H^T R^-1 H (diag R) + invert sum
    cov = 2 * m * n**2 + 2 * n * m**2 + m**3 / 3 + 2 * m * n**2 + 2 * n**2 * m  # PH^T, H(PH^T), chol(S), solve, outer
    print(f"3.6 m={m}: information form ~{info:.2e} flops, covariance form ~{cov:.2e} flops")

# 3.7, 3.8
th = 0.7
print("3.7", np.allclose(expm(np.array([[0, -th], [th, 0]])), [[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]))
A = np.array([[0, 1], [0, 0.]])
B = np.array([[0, 0], [1, 0.]])
print("3.8 exp(A+B)=\n", expm(A + B), "\n exp(A)exp(B)=\n", expm(A) @ expm(B))

# 3.9
sp = np.spacing(2.0**30)
print("3.9 spacing near 2^30 s:", sp, "s =", sp * 1e9, "ns;  epoch now ~", 1.79e9, "s: spacing", np.spacing(1.79e9) * 1e9, "ns")
t0 = 1.79e9
dt_errs = []
rng = np.random.default_rng(0)
for _ in range(10000):
    tk = t0 + rng.integers(0, 10**6) * 0.005 + rng.uniform(0, 1e-3)
    dt = (tk + 0.005) - tk
    dt_errs.append(dt - 0.005)
print("    max |dt error| over samples:", np.max(np.abs(dt_errs)), "s; relative", np.max(np.abs(dt_errs)) / 0.005)
# accumulate time by adding dt for one hour at 200 Hz
t = t0
for _ in range(200 * 3600):
    t += 0.005
print("    drift after 1 h of t += 0.005:", t - (t0 + 3600.0), "s")

# 3.10
x = 1e8
print("3.10 naive sqrt:", np.sqrt(x * x + 1) - x, " rewritten:", 1 / (np.sqrt(x * x + 1) + x))
x = 1e-8
print("     naive (1-cos)/x^2:", (1 - np.cos(x)) / x**2, " rewritten:", 0.5 * (np.sin(x / 2) / (x / 2)) ** 2)
x = 1e-5
print("     at 1e-5 naive:", (1 - np.cos(x)) / x**2, " rewritten:", 0.5 * (np.sin(x / 2) / (x / 2)) ** 2)
