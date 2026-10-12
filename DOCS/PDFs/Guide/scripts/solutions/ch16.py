"""Exercise solutions, Chapter 16: prints the numbers quoted in the worked solutions.

Run from DOCS/PDFs/Guide:  .venv/Scripts/python.exe scripts/solutions/<this file>
"""
from pathlib import Path
import sys
import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # DOCS/PDFs/Guide/scripts
import make_figures as mf  # noqa: E402
from make_figures import Exp, skew  # noqa: E402

GW = np.array([0, 0, -9.81])


def Jl(p):
    t = np.linalg.norm(p)
    K = skew(p)
    return np.eye(3) + (1 - np.cos(t)) / t**2 * K + (t - np.sin(t)) / t**3 * K @ K


def N(p):
    t = np.linalg.norm(p)
    K = skew(p)
    return 0.5 * np.eye(3) + (t - np.sin(t)) / t**3 * K + (t * t / 2 + np.cos(t) - 1) / t**4 * K @ K


# 16.1
w = np.array([0.3, -0.5, 0.8])
a = np.array([1.0, -2.0, 9.0])
dt = 0.5
R0 = Exp(np.array([0.2, -0.1, 0.4]))
v0 = np.array([1.0, 0.5, -0.2])
p0 = np.array([0.1, 0.2, 0.3])
phi = w * dt
R1 = R0 @ Exp(phi)
v1 = v0 + GW * dt + R0 @ (dt * Jl(phi) @ a)
p1 = p0 + v0 * dt + 0.5 * GW * dt**2 + R0 @ (dt**2 * N(phi) @ a)


def f(t, y):
    R = y[:9].reshape(3, 3)
    v = y[9:12]
    return np.concatenate([(R @ skew(w)).ravel(), R @ a + GW, v])


sol = solve_ivp(f, [0, dt], np.concatenate([R0.ravel(), v0, p0]), rtol=1e-13, atol=1e-13, method="DOP853")
y = sol.y[:, -1]
print("16.1 R err", np.abs(y[:9].reshape(3, 3) - R1).max(), " v err", np.abs(y[9:12] - v1).max(), " p err", np.abs(y[12:] - p1).max())

# 16.2 Euler velocity error vs exact per step at 200 Hz, 1 rad/s
dt = 0.005
w = np.array([0, 0, 1.0])
a = np.array([1.0, 0, 9.81])
exact = dt * Jl(w * dt) @ a
euler = dt * a
print("16.2 relative velocity-increment error:", np.linalg.norm(exact - euler) / np.linalg.norm(dt * np.array([1, 0, 0])),
      " predicted 0.5|w|dt =", 0.5 * dt)

# 16.3 integration error vs noise-induced error after 10 s on the figure trajectory
T = 10.0
p_true = mf.traj(T)[0]
rates = [25, 50, 100, 200, 400, 800, 1600]
res = {}
for scheme in ("euler", "midpoint", "rk4lin"):
    res[scheme] = [np.linalg.norm(mf.integrate(scheme, 1 / r, T) - p_true) for r in rates]
    print("16.3", scheme, ["%.1e" % e for e in res[scheme]])


def integrate_noisy(scheme, dt, T, rng, sg, sa):
    n = int(round(T / dt))
    p, v, R, _, _ = mf.traj(0.0)
    for k in range(n):
        t0, t1 = k * dt, (k + 1) * dt
        _, _, _, w0, a0 = mf.traj(t0)
        _, _, _, w1, a1 = mf.traj(t1)
        w0 = w0 + rng.normal(0, sg / np.sqrt(dt), 3)
        a0 = a0 + rng.normal(0, sa / np.sqrt(dt), 3)
        aw = R @ a0 + GW
        p = p + v * dt + 0.5 * aw * dt**2
        v = v + aw * dt
        R = R @ Exp(w0 * dt)
    return p


rng = np.random.default_rng(1)
for r in (100, 200, 800):
    ref = mf.integrate("euler", 1 / r, T)
    errs = [np.linalg.norm(integrate_noisy("euler", 1 / r, T, rng, mf.SIG_G, mf.SIG_A) - ref) for _ in range(40)]
    print(f"16.3 noise-induced rms position error after 10 s at {r} Hz: {np.sqrt(np.mean(np.square(errs))):.3f} m")

# 16.4 static initializer
def init(am):
    u = am / np.linalg.norm(am)
    ez = np.array([0, 0, 1.0])
    c = np.cross(u, ez)
    s = np.linalg.norm(c)
    if s < 1e-12:
        return np.eye(3)
    ang = np.arctan2(s, u @ ez)
    return Exp(ang * c / s)


def Rx(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


R_true = Ry(np.radians(-3)) @ Rx(np.radians(5))
rng = np.random.default_rng(0)
dt = 0.005
n = 200
am = (R_true.T @ np.array([0, 0, 9.81]))[None, :] + rng.normal(0, mf.SIG_A / np.sqrt(dt), (n, 3))
Rh = init(am.mean(axis=0))
tilt_err = lambda Rh: np.degrees(np.arccos(np.clip((Rh @ R_true.T @ np.array([0, 0, 1]))[2], -1, 1)))  # angle between estimated and true up
up_true = R_true.T @ np.array([0, 0, 1.0])
up_est = Rh.T @ np.array([0, 0, 1.0])
print("16.4 tilt error without bias [mrad]:", 1e3 * np.arccos(np.clip(up_true @ up_est, -1, 1)))
b = np.array([0.05, 0.0, 0.0])  # body-frame bias, roughly horizontal for this small tilt
Rh = init(am.mean(axis=0) + b)
up_est = Rh.T @ np.array([0, 0, 1.0])
print("16.4 tilt error with |b|=0.05 [mrad]:", 1e3 * np.arccos(np.clip(up_true @ up_est, -1, 1)), " predicted |b_horiz|/g =",
      1e3 * np.linalg.norm(b - (b @ up_true) * up_true) / 9.81)
