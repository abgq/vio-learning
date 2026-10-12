"""Exercise solutions, Chapter 4: prints the numbers quoted in the worked solutions.

Run from DOCS/PDFs/Guide:  .venv/Scripts/python.exe scripts/solutions/<this file>
"""
import numpy as np
from scipy.linalg import expm


def f(v):
    x, y = v
    return np.array([x * x * y, np.sin(x * y), np.exp(x)])


def J(v):
    x, y = v
    return np.array([[2 * x * y, x * x], [y * np.cos(x * y), x * np.cos(x * y)], [np.exp(x), 0.0]])


v0 = np.array([0.7, -1.3])
h = 1e-6
Jn = np.column_stack([(f(v0 + h * e) - f(v0 - h * e)) / (2 * h) for e in np.eye(2)])
print("4.1 analytic\n", J(v0), "\n max diff", np.abs(Jn - J(v0)).max())
print("4.2 actual error at 0.1:", np.cos(0.1) - (1 - 0.005), " bound", 0.1**4 / 24)


def osc(y):
    return np.array([y[1], -y[0]])


def step(m, y, h):
    if m == "Euler":
        return y + h * osc(y)
    if m == "Heun":
        k1 = osc(y)
        k2 = osc(y + h * k1)
        return y + h / 2 * (k1 + k2)
    k1 = osc(y); k2 = osc(y + h / 2 * k1); k3 = osc(y + h / 2 * k2); k4 = osc(y + h * k3)
    return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


for m in ("Euler", "Heun", "RK4"):
    errs = []
    for N in (32, 64, 128, 256):
        y = np.array([1.0, 0.0])
        for _ in range(N):
            y = step(m, y, 2 * np.pi / N)
        errs.append(np.linalg.norm(y - [1, 0]))
    orders = [np.log2(errs[i] / errs[i + 1]) for i in range(3)]
    print(f"4.8 {m}: errors {['%.2e' % e for e in errs]}, measured orders {np.round(orders, 2)}")

w = np.array([0.3, -0.5, 0.8])
hh = 0.01
K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
print("4.9 det(I+hK) =", np.linalg.det(np.eye(3) + hh * K), " 1+h^2|w|^2 =", 1 + hh**2 * (w @ w))

u = 2.0**-53
x = 1.0
hs = np.logspace(-14, -1, 4000)
fwd = np.abs((np.sin(x + hs) - np.sin(x)) / hs - np.cos(x))
cen = np.abs((np.sin(x + hs) - np.sin(x - hs)) / (2 * hs) - np.cos(x))
print("4.10 forward: theory h*", 2 * np.sqrt(u * np.sin(x) / np.sin(x)), " err*", 2 * np.sqrt(u * np.sin(x) * np.sin(x)))
# smoothed observed minimum: median in log-windows
def obs_min(err):
    le = np.log10(np.maximum(err, 1e-18))
    k = 61
    sm = np.convolve(le, np.ones(k) / k, mode="same")
    i = np.argmin(sm[k:-k]) + k
    return hs[i], 10 ** sm[i]
print("     observed (smoothed) forward min at h=%.2e, err=%.2e" % obs_min(fwd))
hstar = (3 * u * np.sin(x) / np.cos(x)) ** (1 / 3)
print("4.10 central: theory h*", hstar, " err*", hstar**2 / 6 * np.cos(x) + u * np.sin(x) / hstar)
print("     observed (smoothed) central min at h=%.2e, err=%.2e" % obs_min(cen))

# 4.7 composition order
def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def skew(a):
    return np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])


q = np.pi / 2
RA = Rx(q) @ Ry(q)
RB = Ry(q) @ Rx(q)
RC = expm(skew([q, q, 0]))
print("4.7 R_A=\n", np.round(RA, 3), "\nR_B=\n", np.round(RB, 3), "\nExp(int w)=\n", np.round(RC, 3))
ang = lambda R: np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
print("    angles A,B,C:", ang(RA), ang(RB), ang(RC), " angle between A and B:", ang(RA.T @ RB))
