"""Exercise 22.6: corner selection with and without grid bucketing.

Uses a EuRoC image if one is available, otherwise a synthetic 752x480 test image whose texture is concentrated in one region:
  .venv/Scripts/python.exe scripts/solutions/ch22_bucketing.py                 # data/MH_01_easy/mav0/cam0/data/<first image>
  .venv/Scripts/python.exe scripts/solutions/ch22_bucketing.py --index 500     # another frame of MH_01_easy
  .venv/Scripts/python.exe scripts/solutions/ch22_bucketing.py path/to/img.png # any image
  .venv/Scripts/python.exe scripts/solutions/ch22_bucketing.py --synthetic
Writes output/ch22_bucketing.png next to this script. Needs opencv-python.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[5]
EUROC = REPO / "data" / "MH_01_easy" / "mav0" / "cam0" / "data"

# EuRoC cam0 calibration (cam0/sensor.yaml)
K = np.array([[458.654, 0, 367.215], [0, 457.296, 248.375], [0, 0, 1.0]])
D = np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05])

N_FEATURES, MIN_DIST, GRID = 150, 10, (8, 6)


def synthetic_image(seed=3):
    """Weakly textured background (walls, floor); one strongly textured 'machine' region; a few objects elsewhere."""
    rng = np.random.default_rng(seed)
    h, w = 480, 752
    yy, xx = np.mgrid[0:h, 0:w]
    tex = cv2.GaussianBlur(rng.standard_normal((h, w)), (0, 0), 1.5)
    tex /= tex.std()
    img = 90 + 40 * (xx / w) + 12 * tex                   # weak texture everywhere, 1/5 of the machine's contrast
    machine = (xx > 40) & (xx < 300) & (yy > 60) & (yy < 420)
    img[machine] += 48 * tex[machine]
    for _ in range(40):                                   # bright and dark boxes on the machine
        x0, y0 = rng.integers(40, 280), rng.integers(60, 400)
        cv2.rectangle(img, (int(x0), int(y0)), (int(x0 + rng.integers(8, 30)), int(y0 + rng.integers(8, 30))), float(rng.uniform(20, 230)), -1)
    for _ in range(12):                                   # a few large low-contrast objects in the rest of the hall
        x0, y0 = rng.integers(320, 700), rng.integers(20, 430)
        cv2.rectangle(img, (int(x0), int(y0)), (int(x0 + rng.integers(20, 60)), int(y0 + rng.integers(20, 50))), float(rng.uniform(80, 150)), -1)
    img += rng.normal(0, 2, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def candidates(img):
    """Shi-Tomasi response, 3x3 non-maximum suppression, global threshold 1% of the maximum (as goodFeaturesToTrack)."""
    eig = cv2.cornerMinEigenVal(img, blockSize=3, ksize=3)
    peaks = (eig == cv2.dilate(eig, np.ones((3, 3), np.uint8))) & (eig > 0.01 * eig.max())
    vs, us = np.nonzero(peaks)
    order = np.argsort(-eig[vs, us])
    return np.column_stack([us[order], vs[order]]).astype(float)


def select(cands, shape, quota=None):
    """Greedy strongest-first selection with a minimum distance; optional per-cell quota (grid bucketing)."""
    h, w = shape
    cw, ch = w / GRID[0], h / GRID[1]
    counts = np.zeros(GRID[::-1], int)
    chosen = []
    for u, v in cands:
        if len(chosen) == N_FEATURES:
            break
        cx, cy = min(int(u // cw), GRID[0] - 1), min(int(v // ch), GRID[1] - 1)
        if quota is not None and counts[cy, cx] >= quota:
            continue
        if chosen and np.min(np.hypot(*(np.array(chosen) - (u, v)).T)) < MIN_DIST:
            continue
        chosen.append((u, v))
        counts[cy, cx] += 1
    return np.array(chosen), counts


def report(name, pts, counts):
    norm = cv2.undistortPoints(pts.reshape(-1, 1, 2), K, D, criteria=(cv2.TERM_CRITERIA_COUNT, 50, 0)).reshape(-1, 2)
    ev = np.linalg.eigvalsh(np.cov(norm.T))
    occupied = np.count_nonzero(counts) / counts.size
    print(f"{name:20s}: {len(pts)} corners, occupied cells {occupied:.0%}, bearing std along principal axes "
          f"{np.sqrt(ev[1]):.3f} / {np.sqrt(ev[0]):.3f} (ratio {np.sqrt(ev[0] / ev[1]):.2f}), max per cell {counts.max()}")


def main(argv):
    if "--synthetic" in argv:
        img, src = synthetic_image(), "synthetic test image"
    elif argv and not argv[0].startswith("--"):
        img, src = cv2.imread(argv[0], cv2.IMREAD_GRAYSCALE), argv[0]
    elif EUROC.exists():
        files = sorted(EUROC.glob("*.png"))
        idx = int(argv[argv.index("--index") + 1]) if "--index" in argv else 0
        img, src = cv2.imread(str(files[idx]), cv2.IMREAD_GRAYSCALE), f"EuRoC MH_01_easy cam0 frame {idx} ({files[idx].name})"
    else:
        img, src = synthetic_image(), "synthetic test image (data/MH_01_easy not found)"
    print("image:", src)
    cands = candidates(img)
    quota = int(np.ceil(N_FEATURES / (GRID[0] * GRID[1])))
    plain, c_plain = select(cands, img.shape)
    bucketed, c_buck = select(cands, img.shape, quota=quota)
    report("strongest 150", plain, c_plain)
    report(f"bucketed (quota {quota})", bucketed, c_buck)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    vis = np.hstack([vis, vis.copy()])
    for k, pts in enumerate((plain, bucketed)):
        for u, v in pts:
            cv2.circle(vis, (int(u + k * img.shape[1]), int(v)), 4, (0, 120, 255) if k == 0 else (255, 120, 0), 1)
    (HERE / "output").mkdir(exist_ok=True)
    cv2.imwrite(str(HERE / "output" / "ch22_bucketing.png"), vis)
    print("wrote scripts/solutions/output/ch22_bucketing.png (left: strongest corners, right: bucketed)")


if __name__ == "__main__":
    main(sys.argv[1:])
