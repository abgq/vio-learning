"""Exercise solutions, Chapter 13: prints the numbers quoted in the worked solutions.

Run from DOCS/PDFs/Guide:  .venv/Scripts/python.exe scripts/solutions/<this file>
"""
import numpy as np
from scipy.optimize import brentq

sr = 0.02


def d_of(sp):
    my = np.exp(-sp**2 / 2)
    vy = (1 + sr**2) * (1 + np.exp(-2 * sp**2)) / 2 - np.exp(-sp**2)
    return (1 - my) / np.sqrt(vy)


for deg in (15, 30, 45, 60, 75, 90):
    print(f"13.2 sigma_phi={deg} deg: Mahalanobis distance of linearized mean {d_of(np.radians(deg)):.3f}")
sp = brentq(lambda s: d_of(s) - 1, 0.1, 3)
print("13.2 threshold", np.degrees(sp), "deg")
# Monte Carlo check at threshold
rng = np.random.default_rng(0)
r = 1 + sr * rng.normal(size=400000)
ph = np.pi / 2 + sp * rng.normal(size=400000)
Y = np.column_stack([r * np.cos(ph), r * np.sin(ph)])
m, C = Y.mean(0), np.cov(Y.T)
dvec = np.array([0, 1.0]) - m
print("     MC Mahalanobis at threshold:", np.sqrt(dvec @ np.linalg.solve(C, dvec)))

# 13.4 check
mu, s = 0.7, 0.3
n, kappa, alpha = 1, 2, 1
lam = alpha**2 * (n + kappa) - n
pts = [mu, mu + np.sqrt(n + lam) * s, mu - np.sqrt(n + lam) * s]
W = [lam / (n + lam), 1 / (2 * (n + lam)), 1 / (2 * (n + lam))]
print("13.4 UT mean", sum(w * p**2 for w, p in zip(W, pts)), " exact", mu**2 + s**2)

# 13.5 particle filter: 1D robot, landmarks at 0 and 10, sensor returns range to nearest landmark (identity unknown)
rng = np.random.default_rng(3)
L = np.array([0.0, 10.0])
Np = 2000
sz, sd = 0.1, 0.05
x_true = 2.0
parts = rng.uniform(0, 10, Np)
w = np.ones(Np) / Np


def meas(x):
    return np.min(np.abs(x[..., None] - L), axis=-1)


for k in range(6):
    if k > 0:
        d = 0.5
        x_true += d
        parts = parts + d + rng.normal(0, sd, Np)
    z = meas(np.array([x_true]))[0] + rng.normal(0, sz)
    w = w * np.exp(-(meas(parts) - z) ** 2 / (2 * sz**2))
    w /= w.sum()
    left = w[parts < 5].sum()
    print(f"13.5 step {k}: true x={x_true:.2f}, z={z:.2f}, weight of x<5: {left:.3f}, weight near mirror (x>5): {1-left:.3f}, "
          f"mean {np.sum(w*parts):.2f}")
    ess = 1 / np.sum(w**2)
    if ess < Np / 2:
        idx = rng.choice(Np, Np, p=w)
        parts = parts[idx]
        w = np.ones(Np) / Np
