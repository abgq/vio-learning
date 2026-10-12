"""Checks of the simulation results quoted in chapters 27 and 29 (planar MSCKF and planar EKF-SLAM). Takes a few minutes.

Run from DOCS/PDFs/Guide:  .venv/Scripts/python.exe scripts/checks/ch27_ch29_planar_numbers.py
"""
from pathlib import Path
import sys
import numpy as np
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # DOCS/PDFs/Guide/scripts
import planar_sims as ps  # noqa: E402

# Chapter 27 figure statistics, computed exactly as fig_planar_msckf does
landmarks = ps.planar_scenario(0)
runs = [ps.run_planar_msckf(s, landmarks) for s in range(30)]
t = runs[0]["t"]
rms_dr = np.sqrt(np.mean([np.sum((r["pt"] - r["drp"]) ** 2, axis=1) for r in runs], axis=0))
rms_f = np.sqrt(np.mean([np.sum((r["pt"] - r["p"]) ** 2, axis=1) for r in runs], axis=0))
hd = np.sqrt(np.mean([ps.wrap(r["psit"][-1] - r["psi"][-1]) ** 2 for r in runs]))
sd0 = np.sqrt(runs[0]["P"][-1][2, 2])
print(f"27: final RMS position DR {rms_dr[-1]:.2f} m, MSCKF {rms_f[-1]:.2f} m, ratio {rms_dr[-1]/rms_f[-1]:.2f}")
print(f"27: final RMS heading error MSCKF {hd:.4f} rad; run-0 final heading sigma {sd0:.4f} rad; odometry sigma_w sqrt(90) = {0.02*np.sqrt(90):.3f}")
nees = np.mean([r["nees"] for r in runs], axis=0)
lo, hi = chi2.ppf(0.025, 90) / 30, chi2.ppf(0.975, 90) / 30
print(f"27: NEES bounds [{lo:.2f}, {hi:.2f}]")
for a, b in ((0, 10), (10, 20), (20, 30), (30, 45), (45, 60), (60, 75), (75, 90)):
    m = (t > a) & (t <= b)
    print(f"    NEES mean in ({a},{b}] s: {nees[m].mean():.2f}, min {nees[m].min():.2f}, max {nees[m].max():.2f}")
below = t[nees < lo]
above = t[nees > hi]
print(f"    times below lower bound: {below.min() if len(below) else None} .. {below.max() if len(below) else None} s ({len(below)} samples); above upper: {len(above)} samples")
used = sum(r["used"] for r in runs)
rej = sum(r["rejected"] for r in runs)
print(f"27: gate rejection rate {rej/(used+rej):.3f}")

# Chapter 29: when does the standard EKF leave the 95% interval?
N, steps = 50, 1200
kw = dict(sig_v=0.05, sig_w=0.01, sig_m=0.1, rmax=5.0, sig_p0=0.1, sig_psi0=0.05, full=True)
tt = np.arange(1, steps + 1) * 0.1
lo, hi = chi2.ppf(0.025, 3 * N) / N, chi2.ppf(0.975, 3 * N) / N
for variant in ("standard", "fej", "ideal"):
    arr = np.array([ps.run_ekf_slam(100 + s, variant, steps, **kw) for s in range(N)])
    m = arr.mean(axis=0)[:, 0]
    out = np.where(m > hi)[0]
    # first time after which it never comes back inside
    stay = None
    for i in range(len(m)):
        if np.all(m[i:] > hi):
            stay = tt[i]
            break
    print(f"29 {variant}: first exceed {tt[out[0]] if len(out) else None} s, permanently above from {stay} s, final {m[-1]:.2f}, max {m.max():.2f}, bounds [{lo:.2f},{hi:.2f}]")
