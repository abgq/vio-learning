"""Exercise solutions, Chapter 17 (exercises 4-6): prints the numbers quoted in the worked solutions.

Run from DOCS/PDFs/Guide:  .venv/Scripts/python.exe scripts/solutions/ch17.py
(Exercises 1-3 are the project's derivation exercises; their check is the finite-difference test of Milestone 5.)
"""
import numpy as np
from scipy.stats import chi2, norm


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


# 17.4 Monte Carlo scatter of a sample standard deviation from M = 400 runs
M = 400
print(f"17.4 relative std of a sample std from {M} runs: {1 / np.sqrt(2 * (M - 1)):.3f} (two-sigma band +-{2 / np.sqrt(2 * (M - 1)) * 100:.0f}%)")

# 17.5 body-frame attitude covariance after a rotation without information
P0 = np.diag([1e-4, 1e-6, 1e-6])
for deg in (45, 90):
    R = Rz(np.radians(deg))
    P = R.T @ P0 @ R
    print(f"17.5 after {deg} deg: diag {np.diag(P)}, off-diagonal (0,1) {P[0, 1]:.3e}, eigenvalues {np.linalg.eigvalsh(P)}")

# 17.6 power of the variance test against a 10% error in sigma_a
print("17.6 power (M runs):")
for ratio in (1.1**2, 0.9**2):
    row = []
    for Mr in (100, 200, 500, 700):
        df = Mr - 1
        lo, hi = chi2.ppf(0.025, df), chi2.ppf(0.975, df)
        # under the alternative (M-1) s^2 / sigma_pred^2 ~ ratio * chi2(df)
        power = chi2.cdf(lo / ratio, df) + chi2.sf(hi / ratio, df)
        row.append(round(power, 2))
    print(f"     variance ratio {ratio:.2f}: {row}")
# runs needed for 95% power (exact search)
for ratio in (1.1**2, 0.9**2):
    for Mr in range(50, 2000, 10):
        df = Mr - 1
        lo, hi = chi2.ppf(0.025, df), chi2.ppf(0.975, df)
        if chi2.cdf(lo / ratio, df) + chi2.sf(hi / ratio, df) >= 0.95:
            print(f"     ratio {ratio:.2f}: 95% power first reached at M = {Mr}")
            break
print("     normal approximation: z-values", norm.ppf(0.975).round(3), norm.ppf(0.95).round(3))
