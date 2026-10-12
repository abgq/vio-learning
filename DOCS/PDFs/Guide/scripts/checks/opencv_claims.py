"""Checks of the OpenCV statements in chapters 20-22: undistortPoints stops after five iterations by default, and the accuracy of
pyramidal Lucas-Kanade on a noise-free 8-bit image. Needs opencv-python.

Run from DOCS/PDFs/Guide:  .venv/Scripts/python.exe scripts/checks/opencv_claims.py
"""
import cv2
import numpy as np
from scipy import ndimage

print("OpenCV", cv2.__version__)
fu, fv, cu, cv_ = 458.654, 457.296, 367.215, 248.375
K = np.array([[fu, 0, cu], [0, fv, cv_], [0, 0, 1.0]])
D = np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05])

pts = np.array([[[0.0, cv_]], [[0.0, 0.0]], [[751.0, 479.0]]], dtype=np.float64)
ref = cv2.undistortPoints(pts, K, D, criteria=(cv2.TERM_CRITERIA_COUNT, 200, 0)) if True else None
for name, kw in (("default", {}), ("5 iterations explicit", {"criteria": (cv2.TERM_CRITERIA_COUNT, 5, 0)}),
                 ("200 iterations", {"criteria": (cv2.TERM_CRITERIA_COUNT, 200, 0)})):
    try:
        out = cv2.undistortPoints(pts, K, D, **kw)
    except Exception as e:  # older bindings: undistortPointsIter
        out = cv2.undistortPointsIter(pts, K, D, None, None, kw.get("criteria", (cv2.TERM_CRITERIA_COUNT, 5, 0)))
    err = np.linalg.norm((out - ref).reshape(-1, 2), axis=1)
    print(f"undistortPoints {name}: error vs converged (normalized) {err}, in px {err * fu}")

# Lucas-Kanade accuracy on a smooth texture with a known sub-pixel shift
rng = np.random.default_rng(8)
n = 320
base = ndimage.gaussian_filter(rng.standard_normal((n, n)), 2.0)
base = (base - base.min()) / (base.max() - base.min())
shift = np.array([0.3, -0.7])
I1 = (base * 255).round().astype(np.uint8)
I2f = ndimage.shift(base, (shift[1], shift[0]), order=3, mode="nearest")
I2 = (I2f * 255).round().astype(np.uint8)
grid = np.array([[x, y] for y in range(60, 261, 20) for x in range(60, 261, 20)], np.float32).reshape(-1, 1, 2)
for eps in (0.01, 1e-3, 1e-5):
    p2, st, err = cv2.calcOpticalFlowPyrLK(I1, I2, grid, None, winSize=(21, 21), maxLevel=3,
                                           criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, eps))
    e = np.linalg.norm((p2 - grid).reshape(-1, 2) - shift, axis=1)
    print(f"LK eps={eps:g}: median error {np.median(e):.4f} px, 90th pct {np.percentile(e, 90):.4f} px (8-bit images, no noise)")
# the same with float images (no 8-bit quantization) is not supported by calcOpticalFlowPyrLK (8-bit only); quantization effect:
I1q = base * 255
print("8-bit quantization step relative to typical gradient:", 1.0 / np.median(np.abs(np.gradient(I1q)[1])))
