"""Checks of numbers stated in the main text of chapters 20 and 23 (distortion shifts, RANSAC table, triangulation tails).

Run from DOCS/PDFs/Guide:  .venv/Scripts/python.exe scripts/checks/ch20_ch23_numbers.py
"""
import numpy as np

# ---- chapter 23: triangulation statements (reproduces fig_triangulation's Monte Carlo)
rng = np.random.default_rng(17)
f, z = 458.0, 5.0
sig = 1.0 / f
pf = np.array([0.2, -0.1, z])
for par in (0.25, 0.5, 1.0, 2.0, 4.0):
    b = 2 * z * np.tan(np.radians(par) / 2)
    est = []
    for _ in range(20000):
        m1 = pf[:2] / pf[2] + sig * rng.normal(size=2)
        m2 = (pf - [b, 0, 0])[:2] / pf[2] + sig * rng.normal(size=2)
        A = np.array([[1, 0, -m1[0], 0.0], [0, 1, -m1[1], 0.0], [1, 0, -m2[0], -b], [0, 1, -m2[1], 0.0]])
        X = np.linalg.svd(A)[2][-1]
        est.append(X[2] / X[3])
    est = np.array(est)
    robust = 1.4826 * np.median(np.abs(est - np.median(est)))
    first = z**2 * np.sqrt(2) * sig / b
    neg = np.mean(est < 0)
    far = np.mean(np.abs(est - z) > 5 * first)
    print(f"parallax {par:4.2f} deg (b = {b*100:5.1f} cm): robust rel. std {robust/z*100:6.1f}%, first order {first/z*100:6.1f}%, "
          f"negative depths {neg*100:.2f}%, beyond 5 sigma {far*100:.2f}% (Gaussian: 0.00006%)")

# ---- chapter 23: RANSAC table
for s in (8, 5, 2):
    row = [int(np.ceil(np.log(0.01) / np.log(1 - w**s))) for w in (0.9, 0.7, 0.5)]
    print("RANSAC s =", s, row)

# ---- chapter 20: radial factor and pixel shift at the edge midpoints and the corners
fu, fv, cu, cv = 458.654, 457.296, 367.215, 248.375
k1, k2, p1, p2 = -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05


def distort(x, y):
    r2 = x * x + y * y
    rad = 1 + k1 * r2 + k2 * r2 * r2
    return x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x), y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y


def undistort(xd, yd):
    x, y = xd, yd
    for _ in range(200):
        r2 = x * x + y * y
        rad = 1 + k1 * r2 + k2 * r2 * r2
        x, y = (xd - (2 * p1 * x * y + p2 * (r2 + 2 * x * x))) / rad, (yd - (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)) / rad
    return x, y


for lab, (u, v) in (("left edge middle", (0, cv)), ("right edge middle", (751, cv)), ("top edge middle", (cu, 0)),
                    ("corner (0,0)", (0, 0)), ("corner (751,479)", (751, 479))):
    xd, yd = (u - cu) / fu, (v - cv) / fv
    x, y = undistort(xd, yd)
    r2 = x * x + y * y
    rad = 1 + k1 * r2 + k2 * r2 * r2
    shift = np.hypot((x - xd) * fu, (y - yd) * fv)
    print(f"{lab:18s}: radial factor {rad:.3f}, ideal point lies {shift:.0f} px further out")

# ---- chapter 20: blur and time offset
print("motion blur f w T =", 458 * 1.0 * 0.005, "px; time offset 1 ms at 0.5/s =", 0.5e-3, "normalized =", 0.5e-3 * fu, "px")
