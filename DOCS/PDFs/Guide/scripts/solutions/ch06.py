"""Exercise solutions, Chapter 6: prints the numbers quoted in the worked solutions.

Run from DOCS/PDFs/Guide:  .venv/Scripts/python.exe scripts/solutions/<this file>
"""
import numpy as np
from scipy.optimize import least_squares

a = 0.99
print("6.2 std", np.sqrt(1 / (1 - a * a)), " lag", 1 / -np.log(a))
s = 2e-3
print("6.3 1kHz sample", s * np.sqrt(1000), " avg1000", s * np.sqrt(1000) / np.sqrt(1000), " 100Hz", s * np.sqrt(100))

# 6.5 integrated noise Monte Carlo
rng = np.random.default_rng(5)
dt, T, R, sig = 1 / 200, 10.0, 1000, 1.0
n = int(T / dt)
w = rng.normal(0, sig / np.sqrt(dt), (R, n))
I1 = np.cumsum(w, axis=1) * dt
I2 = np.cumsum(I1, axis=1) * dt
I3 = np.cumsum(I2, axis=1) * dt
for m, I, th in ((1, I1, T), (2, I2, T**3 / 3), (3, I3, T**5 / 20)):
    v = I[:, -1].var(ddof=1)
    print(f"6.5 m={m}: MC var {v:.4g}, theory {th:.4g}, ratio {v/th:.3f}")
print("    95% band for ratio with 1000 runs: +-", 1.96 * np.sqrt(2 / 999))

# 6.8 Allan deviation
SIG_G, SIG_BG = 1.6968e-4, 1.9393e-5


def sim(T, fs, rng):
    n = int(T * fs)
    dt = 1 / fs
    white = rng.normal(0, SIG_G / np.sqrt(dt), n)
    bias = np.cumsum(rng.normal(0, SIG_BG * np.sqrt(dt), n))
    return white + bias, dt


def oavar(y, dt, taus):
    theta = np.concatenate([[0.0], np.cumsum(y) * dt])
    out = []
    for tau in taus:
        m = int(round(tau / dt))
        d = theta[2 * m:] - 2 * theta[m:-m] + theta[:-2 * m]
        out.append(np.mean(d * d) / (2 * (m * dt) ** 2))
    return np.array(out)


def fit(taus, av):
    def res(p):
        return np.log(np.exp(p[0]) / taus + np.exp(p[1]) * taus) - np.log(av)
    p = least_squares(res, [np.log(SIG_G**2), np.log(SIG_BG**2 / 3)]).x
    return np.sqrt(np.exp(p[0])), np.sqrt(3 * np.exp(p[1]))


rng = np.random.default_rng(11)
y, dt = sim(2 * 3600, 200, rng)
taus = np.logspace(np.log10(0.05), np.log10(2 * 3600 / 8), 40)
av = oavar(y, dt, taus)
sg, sb = fit(taus, av)
i1 = np.argmin(abs(taus - 1))
print(f"6.8 2h @200Hz: fitted sigma_g {sg:.4e} (true {SIG_G:.4e}), sigma_bg {sb:.4e} (true {SIG_BG:.4e})")
print(f"    min of Allan dev at tau={taus[np.argmin(av)]:.1f}s, value {np.sqrt(av.min()):.3e}; crossover sqrt(3) sg/sb = {np.sqrt(3)*SIG_G/SIG_BG:.1f}s")
print(f"    reading at 1 s: sigma_A(1) = {np.sqrt(av[i1]):.4e} (white alone would be {SIG_G:.4e})")
# spread of sigma_bg estimates versus record length (10 Hz is enough for tau >= 1 s)
for hours in (0.5, 1, 2, 4, 8, 16):
    T = hours * 3600
    ests = []
    for k in range(60):
        y, dt = sim(T, 10, rng)
        taus = np.logspace(0, np.log10(T / 8), 30)
        av = oavar(y, dt, taus)
        ests.append(fit(taus, av)[1])
    ests = np.array(ests)
    print(f"    T={hours:5.1f} h: rel. std of sigma_bg {np.std(ests)/SIG_BG:.3f}, mean ratio {np.mean(ests)/SIG_BG:.3f}")

# ---- weighted Allan-variance fit
import numpy as np
from scipy.optimize import least_squares
SIG_G, SIG_BG = 1.6968e-4, 1.9393e-5
def sim(T, fs, rng):
    n = int(T * fs); dt = 1 / fs
    return rng.normal(0, SIG_G / np.sqrt(dt), n) + np.cumsum(rng.normal(0, SIG_BG * np.sqrt(dt), n)), dt
def oavar(y, dt, taus):
    theta = np.concatenate([[0.0], np.cumsum(y) * dt]); out = []
    for tau in taus:
        m = int(round(tau / dt)); d = theta[2*m:] - 2*theta[m:-m] + theta[:-2*m]
        out.append(np.mean(d * d) / (2 * (m * dt) ** 2))
    return np.array(out)
def fit(taus, av, T):
    wts = np.sqrt(T / taus)          # each point's log-error shrinks like sqrt(clusters)
    res = lambda p: wts * (np.log(np.exp(p[0]) / taus + np.exp(p[1]) * taus) - np.log(av))
    p = least_squares(res, [np.log(SIG_G**2), np.log(SIG_BG**2 / 3)]).x
    return np.sqrt(np.exp(p[0])), np.sqrt(3 * np.exp(p[1]))
rng = np.random.default_rng(3)
for hours in (0.5, 1, 2, 4, 8):
    T = hours * 3600; ests = []
    for k in range(80):
        y, dt = sim(T, 10, rng); taus = np.logspace(0, np.log10(T / 4), 30)
        ests.append(fit(taus, oavar(y, dt, taus), T)[1])
    ests = np.array(ests)
    print(f"T={hours:4.1f} h: rel std {np.std(ests)/SIG_BG:.3f}, mean ratio {np.mean(ests)/SIG_BG:.3f}")
