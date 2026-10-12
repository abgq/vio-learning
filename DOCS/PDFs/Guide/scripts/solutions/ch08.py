"""Exercise solutions, Chapter 8: prints the numbers quoted in the worked solutions.

Run from DOCS/PDFs/Guide:  .venv/Scripts/python.exe scripts/solutions/<this file>
"""
import numpy as np
from mpmath import mp, mpf, cos as mcos, sin as msin

mp.dps = 50


def skew(a):
    return np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])


def Exp(p):
    t = np.linalg.norm(p)
    K = skew(p)
    if t < 1e-8:
        return np.eye(3) + K + 0.5 * K @ K
    return np.eye(3) + np.sin(t) / t * K + (1 - np.cos(t)) / t**2 * K @ K


def Log(R):
    w = 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = np.linalg.norm(w)
    c = 0.5 * (np.trace(R) - 1)
    th = np.arctan2(s, c)
    if th < 1e-8:
        return w
    if np.pi - th < 1e-3:  # near-pi branch: symmetric part
        S = 0.5 * (R + R.T) - c * np.eye(3)
        S /= (1 - c)
        i = np.argmax(np.diag(S))
        u = S[:, i] / np.sqrt(S[i, i])
        if u @ w < 0:
            u = -u
        return th * u
    return th / s * w


def Jr(p):
    t = np.linalg.norm(p)
    K = skew(p)
    if t < 1e-8:
        return np.eye(3) - 0.5 * K + K @ K / 6
    return np.eye(3) - (1 - np.cos(t)) / t**2 * K + (t - np.sin(t)) / t**3 * K @ K


def Jrinv(p):
    t = np.linalg.norm(p)
    K = skew(p)
    if t < 1e-8:
        return np.eye(3) + 0.5 * K + K @ K / 12
    return np.eye(3) + 0.5 * K + (1 / t**2 - (1 + np.cos(t)) / (2 * t * np.sin(t))) * K @ K


# 8.3: arccos vs atan2, and near pi
u = np.array([1.0, 2.0, -2.0]) / 3
print("8.3 theta, err_acos, err_atan2")
for th in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10):
    R = Exp(th * u)
    a = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    b = np.arctan2(0.5 * np.linalg.norm([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]), 0.5 * (np.trace(R) - 1))
    print(f"    {th:.0e}: {abs(a-th)/th:.1e} {abs(b-th)/th:.1e}")
for th in (np.pi - 1e-6, np.pi):
    R = Exp(th * u)
    phi = Log(R)
    err = min(np.linalg.norm(phi - th * u), np.linalg.norm(phi + th * u))
    w = 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    with np.errstate(all="ignore"):
        naive = th / (2 * np.sin(th)) * 2 * w if np.sin(th) != 0 else np.full(3, np.nan)
    errn = np.linalg.norm(naive - th * u)
    print(f"8.3 near pi theta=pi-{np.pi-th:.0e}: near-pi branch err {err:.1e}, naive skew-part err {errn:.1e}")

# 8.4: Jr with threshold 0.02 and three-term series vs 50-digit reference
def coeffs_ref(t):
    t = mpf(t)
    return float((1 - mcos(t)) / t**2), float((t - msin(t)) / t**3)


def Jr_scheme(p, t0=0.02):
    t = np.linalg.norm(p)
    K = skew(p)
    if t < t0:
        t2 = t * t
        B = 0.5 - t2 / 24 + t2 * t2 / 720
        C = 1 / 6 - t2 / 120 + t2 * t2 / 5040
    else:
        B = (1 - np.cos(t)) / t2_ if False else (1 - np.cos(t)) / t**2
        C = (t - np.sin(t)) / t**3
    return np.eye(3) - B * K + C * K @ K


worst = 0
worst_naive = 0
for t in np.concatenate([np.logspace(-12, -1, 400), [0.0]]):
    p = t * u
    K = skew(p)
    if t == 0:
        ref = np.eye(3)
    else:
        B, C = coeffs_ref(t)
        ref = np.eye(3) - B * K + C * K @ K
    worst = max(worst, np.abs(Jr_scheme(p) - ref).max())
    if t > 0:
        with np.errstate(all="ignore"):
            nv = np.eye(3) - (1 - np.cos(t)) / t**2 * K + (t - np.sin(t)) / t**3 * K @ K
        worst_naive = max(worst_naive, np.nanmax(np.abs(nv - ref)))
print(f"8.4 scheme max abs error on [0,0.1]: {worst:.1e}; naive closed form max error: {worst_naive:.1e}")
for t in (0.019, 0.02, 0.05, 0.1):
    p = t * u
    B, C = coeffs_ref(t)
    K = skew(p)
    ref = np.eye(3) - B * K + C * K @ K
    nv = np.eye(3) - (1 - np.cos(t)) / t**2 * K + (t - np.sin(t)) / t**3 * K @ K
    print(f"    t={t}: closed-form error {np.abs(nv-ref).max():.1e}, u/t = {2**-53/t:.1e}")

# 8.7
Ry = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0.]])
print("8.7", Ry.T @ np.diag([1, 1, 9.]) @ Ry)

# 8.8 numerical check of the table
rng = np.random.default_rng(4)
R = Exp(rng.normal(size=3))
R2 = Exp(rng.normal(size=3))
v = rng.normal(size=3)
phi = rng.normal(size=3)
h = 1e-6


def num(f, x0_plus, diff, dim=3):
    cols = []
    for e in np.eye(dim):
        cols.append(diff(f(x0_plus(h * e)), f(x0_plus(-h * e))) / (2 * h))
    return np.column_stack(cols)


vec = lambda a, b: a - b
rot = lambda a, b: Log(b.T @ a)
checks = {
    "Rv": (num(lambda X: X @ v, lambda d: R @ Exp(d), vec), -R @ skew(v)),
    "R^T v": (num(lambda X: X.T @ v, lambda d: R @ Exp(d), vec), skew(R.T @ v)),
    "R^-1": (num(lambda X: X.T, lambda d: R @ Exp(d), rot), -R),
    "R1R2 (R1)": (num(lambda X: X @ R2, lambda d: R @ Exp(d), rot), R2.T),
    "R1R2 (R2)": (num(lambda X: R @ X, lambda d: R2 @ Exp(d), rot), np.eye(3)),
    "Exp": (num(Exp, lambda d: phi + d, rot), Jr(phi)),
    "Log": (num(Log, lambda d: Exp(phi) @ Exp(d), vec), Jrinv(phi)),
    "Exp(phi)v": (num(lambda p: Exp(p) @ v, lambda d: phi + d, vec), -Exp(phi) @ skew(v) @ Jr(phi)),
}
for k, (n, a) in checks.items():
    print(f"8.8 {k:10s} max error {np.abs(n-a).max():.1e}")
print("    flipped sign of Rv row:", np.abs(checks["Rv"][0] - (R @ skew(v))).max())
print("    |phi| =", np.linalg.norm(phi))
