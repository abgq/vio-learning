"""Exercise solutions, Chapter 5: prints the numbers quoted in the worked solutions.

Run from DOCS/PDFs/Guide:  .venv/Scripts/python.exe scripts/solutions/<this file>
"""
import numpy as np
from scipy.stats import chi2
from scipy.integrate import quad

pf = 0.01 * 0.98 / (0.01 * 0.98 + 0.99 * 0.03)
print("5.1", pf, 0.01 * 0.98 + 0.99 * 0.03)
c, s = np.cos(np.radians(30)), np.sin(np.radians(30))
A = np.array([[c, -s], [s, c]])
C = A @ np.diag([1, 4.]) @ A.T
print("5.2", C, np.linalg.eigh(C))
for k in (1, 2, 3):
    print(f"5.8 k={k}: 2D {chi2.cdf(k*k, 2):.4f}  1D {chi2.cdf(k*k, 1):.4f}  3D {chi2.cdf(k*k, 3):.4f}")
print("5.9", quad(lambda x: x**4 * np.exp(-x * x / 2) / np.sqrt(2 * np.pi), -np.inf, np.inf))
N = 100
lo, hi = chi2.ppf(0.025, N - 1) / (N - 1), chi2.ppf(0.975, N - 1) / (N - 1)
p_one = chi2.sf(1.25 * (N - 1), N - 1)
print(f"5.10 interval [{lo:.3f}, {hi:.3f}], one-sided p = {p_one:.3f}, two-sided p = {2*p_one:.3f}")
# how many runs to make a 25% excess significant at 5% (expected ratio exactly 1.25)
for n in range(100, 400, 10):
    if chi2.ppf(0.975, n - 1) / (n - 1) < 1.25:
        print("     a ratio of 1.25 is outside the interval from N =", n)
        break
