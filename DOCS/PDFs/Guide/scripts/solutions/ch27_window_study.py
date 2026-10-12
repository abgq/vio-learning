"""Exercise 27.7: final RMS errors of the planar MSCKF for different window sizes, and how long landmarks stay in view.

Run from DOCS/PDFs/Guide:  .venv/Scripts/python.exe scripts/solutions/ch27_window_study.py   (about six minutes)
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # DOCS/PDFs/Guide/scripts
import planar_sims as ps  # noqa: E402

landmarks = ps.planar_scenario(0)
NRUNS = 30
for n_max in (3, 5, 10, 20, 40, 80):
    t0 = time.time()
    errs, herrs, nees_end, used, rej = [], [], [], 0, 0
    for s in range(NRUNS):
        log = ps.run_planar_msckf(s, landmarks, n_max=n_max)
        e = log["pt"][-1] - log["p"][-1]
        errs.append(e @ e)
        herrs.append(ps.wrap(log["psit"][-1] - log["psi"][-1]) ** 2)
        nees_end.append(np.mean(log["nees"][-90:]))
        used += log["used"]
        rej += log["rejected"]
    print(f"N_max={n_max:2d}: final RMS position {np.sqrt(np.mean(errs)):.3f} m, final RMS heading {np.sqrt(np.mean(herrs)):.4f} rad, "
          f"mean NEES last 9 s {np.mean(nees_end):.2f}, rejection rate {rej / (used + rej):.3f}, time {time.time() - t0:.0f} s", flush=True)

# how long does a landmark stay in the field of view on this trajectory?
dt, cam_every, v, w = 0.01, 10, 1.0, 0.1
fov = np.tan(np.radians(50.0))
p, psi = np.array([10.0, 0.0]), np.pi / 2
vis = []
for k in range(1, 9001):
    p = p + ps.rot(psi + 0.5 * w * dt) @ np.array([v * dt * np.sinc(w * dt / 2 / np.pi), 0.0])
    psi = psi + w * dt
    if k % cam_every == 0:
        pb = (landmarks - p) @ ps.rot(psi)
        vis.append((pb[:, 0] > 1.0) & (pb[:, 0] < 15.0) & (np.abs(pb[:, 1] / pb[:, 0]) < fov))
V = np.array(vis)
lengths = []
for j in range(V.shape[1]):
    run = 0
    for f in V[:, j]:
        if f:
            run += 1
        elif run:
            lengths.append(run)
            run = 0
    if run:
        lengths.append(run)
lengths = np.array(lengths)
print(f"visibility runs: median {np.median(lengths):.0f} frames ({np.median(lengths) * 0.1:.1f} s), mean {lengths.mean():.1f} frames; "
      f"landmarks visible per frame {V.sum(1).mean():.1f}")
