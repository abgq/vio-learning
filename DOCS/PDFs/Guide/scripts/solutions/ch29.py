"""Exercise solutions, Chapter 29 (exercises 1, 3, 4): prints the numbers quoted in the worked solutions.

Run from DOCS/PDFs/Guide:  .venv/Scripts/python.exe scripts/solutions/ch29.py
Exercise 5 is ch29_nullspace_leak.py; the ideal filter's augmentation Jacobian is studied in ch29_ideal_augmentation.py.
"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot

rng = np.random.default_rng(29)


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def Log(R):
    return Rot.from_matrix(R).as_rotvec()


# 29.1: the VIO nullspace N, checked against x' boxminus x for a small global yaw and translation
p, v = rng.normal(0, 3, 3), rng.normal(0, 1, 3)
R = Rot.from_rotvec(rng.normal(0, 1, 3)).as_matrix()
ba, bg = rng.normal(0, 0.1, 3), rng.normal(0, 0.01, 3)   # biases: unchanged by the transformation (zero rows of N)
pc = rng.normal(0, 3, 3)
Rc = Rot.from_rotvec(rng.normal(0, 1, 3)).as_matrix()
pf = rng.normal(0, 5, 3)
ez = np.array([0, 0, 1.0])
for scale in (1e-3, 1e-5):
    t = rng.normal(0, 1, 3) * scale
    a = rng.normal() * scale
    Rz = Rot.from_rotvec(a * ez).as_matrix()
    p2, v2, R2, pc2, Rc2, pf2 = Rz @ p + t, Rz @ v, Rz @ R, Rz @ pc + t, Rz @ Rc, Rz @ pf + t
    dx = np.concatenate([p2 - p, v2 - v, Log(R.T @ R2), np.zeros(6), pc2 - pc, Log(Rc.T @ Rc2), pf2 - pf])
    N = np.zeros((24, 4))
    N[0:3, 0:3] = np.eye(3)
    N[0:3, 3] = -skew(p) @ ez
    N[3:6, 3] = -skew(v) @ ez
    N[6:9, 3] = R.T @ ez
    N[15:18, 0:3] = np.eye(3)
    N[15:18, 3] = -skew(pc) @ ez
    N[18:21, 3] = Rc.T @ ez
    N[21:24, 0:3] = np.eye(3)
    N[21:24, 3] = -skew(pf) @ ez
    pred = N @ np.concatenate([t, [a]])
    print(f"29.1 scale {scale:.0e}: |x' boxminus x - N[t; a]| = {np.abs(dx - pred).max():.2e}, entries of N[t; a] up to {np.abs(pred).max():.2e}")

# 29.3: two unknown landmarks; gradients of h and of its Lie derivatives
Om = np.array([[0, -1.0], [1, 0]])


def rot(psi):
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s], [s, c]])


def h(x):
    pr, ps_, l1, l2 = x[0:2], x[2], x[3:5], x[5:7]
    return np.concatenate([rot(ps_).T @ (l1 - pr), rot(ps_).T @ (l2 - pr)])


def jac(fun, x, e=1e-6):
    return np.column_stack([(fun(x + e * u) - fun(x - e * u)) / (2 * e) for u in np.eye(len(x))])


x0 = np.array([1.3, -0.4, 0.7, 4.0, 2.0, -1.0, 5.0])
f_v = np.array([np.cos(x0[2]), np.sin(x0[2]), 0, 0, 0, 0, 0])
f_w = np.array([0, 0, 1.0, 0, 0, 0, 0])
print("29.3 L_fv h =", (jac(h, x0) @ f_v).round(6), " (expect -e1 for each landmark)")
L_w = lambda x: -np.concatenate([Om @ h(x)[:2], Om @ h(x)[2:]])     # closed form of L_fw h, checked next
print("     max |L_fw h - (-Omega h)| =", np.abs(jac(h, x0) @ f_w - L_w(x0)).max())
L_ww = lambda x: -h(x)                                                # L_fw L_fw h = Omega^2 h = -h
G = np.vstack([jac(h, x0), jac(L_w, x0), jac(L_ww, x0)])
s = np.linalg.svd(G, compute_uv=False)
print("     singular values of the stacked gradients:", s.round(6))
null = np.linalg.svd(G)[2][4:]
Nt = np.array([[1, 0, 0, 1, 0, 1, 0], [0, 1, 0, 0, 1, 0, 1.0]]).T
Nr = np.concatenate([Om @ x0[0:2], [1.0], Om @ x0[3:5], Om @ x0[5:7]])
basis = np.column_stack([Nt, Nr])
proj = basis @ np.linalg.lstsq(basis, null.T, rcond=None)[0]
print("     nullspace = span(translations, rotation): residual", np.abs(proj - null.T).max())

# 29.4: Jacobian of (range, bearing) with respect to the body-frame position d
d = np.array([3.0, -1.2])
rho = np.linalg.norm(d)
Jz = np.vstack([d / rho, (Om @ d) / rho**2])
print("29.4 det J_z =", np.linalg.det(Jz), " 1/rho =", 1 / rho)
