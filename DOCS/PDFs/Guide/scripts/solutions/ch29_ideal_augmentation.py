"""Exercise 29.5, remark on the ideal filter: what happens if its landmark-initialization Jacobian uses the noisy measurement
instead of the true relative position (the fig_fej settings, 50 runs). Takes a few minutes.

Run from DOCS/PDFs/Guide:  .venv/Scripts/python.exe scripts/solutions/ch29_ideal_augmentation.py
"""
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]  # DOCS/PDFs/Guide/scripts
sys.path.insert(0, str(SCRIPTS))
import planar_sims as ps  # noqa: E402

# Build a variant of run_ekf_slam whose ideal filter evaluates the augmentation Jacobian with the noisy measurement z.
src = (SCRIPTS / "planar_sims.py").read_text(encoding="utf-8")
fn = "def run_ekf_slam" + src.split("def run_ekf_slam")[1].split("def fig_fej")[0]
true_aug = '            Jx[:, 2] = OM @ (lms[j] - p) if variant == "ideal" else OM @ rot(ps) @ z\n'
assert fn.count(true_aug) == 1, "planar_sims.run_ekf_slam changed; update this script"
noisy = fn.replace(true_aug, "            Jx[:, 2] = OM @ rot(ps) @ z\n").replace("def run_ekf_slam", "def run_ekf_slam_noisy_aug")
ns = dict(vars(ps))
exec(noisy, ns)

N, steps = 50, 1200
kw = dict(sig_v=0.05, sig_w=0.01, sig_m=0.1, rmax=5.0, sig_p0=0.1, sig_psi0=0.05, full=True)
for label, fun, variant in (("ideal, true-state augmentation", ps.run_ekf_slam, "ideal"),
                            ("ideal, noisy augmentation", ns["run_ekf_slam_noisy_aug"], "ideal"),
                            ("FEJ", ps.run_ekf_slam, "fej"), ("standard", ps.run_ekf_slam, "standard")):
    m = np.array([fun(100 + s, variant, steps, **kw) for s in range(N)]).mean(axis=0)
    print(f"{label:32s}: final NEES {m[-1, 0]:.2f}, RMS heading {np.degrees(np.sqrt(m[-1, 1])):.2f} deg, "
          f"filter sigma {np.degrees(np.sqrt(m[-1, 2])):.2f} deg, variance / MSE {m[-1, 2] / m[-1, 1]:.2f}", flush=True)
