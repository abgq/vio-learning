"""Exercise solutions, Chapter 28 (exercise 2: instrumenting the planar filter): prints the numbers quoted in the worked solutions.

Run from DOCS/PDFs/Guide:  .venv/Scripts/python.exe scripts/solutions/<this file>
"""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # DOCS/PDFs/Guide/scripts
import planar_sims as ps  # noqa: E402

records = []
orig_update = ps.PlanarMSCKF.update


def update(self, tracks):
    Ms = [len(o) for o in tracks if len(o) >= 3]
    records.append((len(Ms), Ms, sum(m - 2 for m in Ms)))
    return orig_update(self, tracks)


ps.PlanarMSCKF.update = update
landmarks = ps.planar_scenario(0)
for s in range(5):
    ps.run_planar_msckf(s, landmarks, n_max=10)
nfeat = np.array([r[0] for r in records])
d = np.array([r[2] for r in records])
Ms = np.concatenate([r[1] for r in records if r[1]])
print(f"updates: {len(records)}; features per update: mean {nfeat.mean():.1f}, 99th pct {np.percentile(nfeat, 99)}, max {nfeat.max()}")
print(f"observation counts M: {np.bincount(Ms)[3:]} (for M = 3..)  mean {Ms.mean():.2f}")
print(f"rows d per update: mean {d.mean():.1f}, median {np.median(d)}, 99th pct {np.percentile(d, 99)}, 99.9th {np.percentile(d, 99.9):.0f}, max {d.max()}")
print("bound from visible features x (N_max - 2):", 40 * (10 - 2))
