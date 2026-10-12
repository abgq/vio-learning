# Scripts behind the worked solutions

Every number quoted in a worked solution of the guide is printed by one of these scripts. The scripts in
`../checks/` re-derive the numerical claims of the main text that rest on simulations or on OpenCV.

Run them from `DOCS/PDFs/Guide` with the guide's virtual environment:

```bash
.venv/Scripts/python.exe scripts/solutions/ch20.py
.venv/Scripts/python.exe scripts/solutions/run_all.py           # everything, about 20 minutes
.venv/Scripts/python.exe scripts/solutions/run_all.py --quick   # skips the long Monte Carlo studies
```

`run_all.py` stores each script's output in `output/<script>.txt`; the text files committed there are the reference
outputs of guide version 1.0 (Python 3.14, NumPy 2.5, SciPy 1.18, OpenCV 5.0). Images written to `output/` are not
committed. Random numbers come from seeded
generators, so a rerun on the same library versions reproduces them exactly; other versions may differ in the
last digits of Monte Carlo results.

Requirements: `numpy`, `scipy`, `matplotlib` (for the figure modules some scripts import) and, for the
OpenCV-based scripts (`ch22_bucketing.py`, `ch23_opencv.py`, `../checks/opencv_claims.py`), `opencv-python`.

| Script | Exercises |
|---|---|
| `ch01_ch02.py` ... `ch21.py` | the numerical exercises of chapters 1-21 (one file per chapter) |
| `ch22.py` | 22.1-22.5 (structure tensors, Lucas-Kanade, forward-backward error, gyro prediction) |
| `ch22_bucketing.py` | 22.6 (grid bucketing); uses `data/MH_01_easy` when present, otherwise a synthetic image; writes `output/ch22_bucketing.png` |
| `ch23.py` | 23.1-23.5, 23.8 (epipolar geometry, eight-point, decomposition, two-point RANSAC, triangulation) |
| `ch23_opencv.py` | 23.6 (`cv::findEssentialMat` under pure rotation, the parallax test) |
| `ch24.py` ... `ch26.py` | chapters 24-26 |
| `ch27.py` | 27.1-27.6, 27.8, 27.9, and the heading-ramp analysis of 27.7 |
| `ch27_window_study.py` | 27.7 (window sizes 3-80 on the planar MSCKF, about six minutes) |
| `ch28.py` | 28.2 (stacked rows per update in the planar filter) |
| `ch29.py` | 29.1, 29.3, 29.4 |
| `ch29_nullspace_leak.py` | 29.5 (the nullspace condition for the standard, FEJ and ideal planar EKF-SLAM filters) |
| `ch29_ideal_augmentation.py` | 29.5, remark on the ideal filter's landmark-initialization Jacobian |
| `ch30_jacobian_breakage.py` | 30.4 |
| `cpp/` | 30.1-30.3 and 30.5-30.7: run `cpp/build_and_run.ps1` (needs the MSYS2 CLANG64 toolchain and the Eigen sources fetched by the project's `release` preset) |

Exercises whose solutions contain no computed numbers are not covered. The derivation exercises of the project
Jacobians are checked by the project's own finite-difference tests, not here.
