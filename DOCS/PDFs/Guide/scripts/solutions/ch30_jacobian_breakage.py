"""Exercise solutions, Chapter 30, exercise 4 (which Jacobian breakages a finite-difference test detects): prints the numbers quoted in the worked solutions.

Run from DOCS/PDFs/Guide:  .venv/Scripts/python.exe scripts/solutions/<this file>
"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def Exp(v):
    return Rot.from_rotvec(v).as_matrix()


def Log(R):
    return Rot.from_matrix(R).as_rotvec()


def Jr(phi, branch=1e-8, drop2=False, flip=False):
    th = np.linalg.norm(phi)
    S = skew(phi)
    if th < branch:
        return np.eye(3) - 0.5 * S * (-1 if flip else 1)
    a = (1 - np.cos(th)) / th**2
    b = (th - np.sin(th)) / th**3
    return np.eye(3) - (-a if flip else a) * S + (0 if drop2 else b) * S @ S


def Jr_numeric(phi, h=1e-6):
    R = Exp(phi)
    cols = []
    for i in range(3):
        d = np.zeros(3); d[i] = h
        cols.append((Log(R.T @ Exp(phi + d)) - Log(R.T @ Exp(phi - d))) / (2 * h))
    return np.column_stack(cols)


rng = np.random.default_rng(30)
u = rng.normal(size=3); u /= np.linalg.norm(u)
points = {"phi = 0": np.zeros(3), "|phi| = 0.05": 0.05 * u, "|phi| = 0.5": 0.5 * u, "|phi| = 2.5": 2.5 * u}
variants = {
    "correct": dict(),
    "sign of [phi]x term flipped (J_l)": dict(flip=True),
    "second-order term dropped": dict(drop2=True),
    "Taylor branch used below 0.1 rad": dict(branch=0.1),
}
print(f"{'variant':38s}" + "".join(f"{k:>14s}" for k in points))
for name, kw in variants.items():
    row = []
    for k, phi in points.items():
        err = np.abs(Jr(phi, **kw) - Jr_numeric(phi)).max()
        row.append(err)
    print(f"{name:38s}" + "".join(f"{e:14.1e}" for e in row))
