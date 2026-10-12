"""Exercise solutions, Chapter 23, exercise 6 (cv::findEssentialMat under pure rotation; needs opencv-python): prints the numbers quoted in the worked solutions.

Run from DOCS/PDFs/Guide:  .venv/Scripts/python.exe scripts/solutions/<this file>
"""
# What cv::findEssentialMat returns under pure rotation, and the rotation-compensated parallax test.
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot

F_PX = 458.0
SIGMA = 1.0 / F_PX            # 1 px noise in normalized coordinates


def make_tracks(rng, t, n=150, outlier_frac=0.2):
    R = Rot.from_rotvec([0.0, 0.05, 0.0]).as_matrix()          # 0.05 rad yaw of the camera between the frames
    X1 = np.column_stack([rng.uniform(-3, 3, n), rng.uniform(-2, 2, n), rng.uniform(3, 8, n)])
    X2 = X1 @ R.T + t
    x1 = X1[:, :2] / X1[:, 2:] + rng.normal(0, SIGMA, (n, 2))
    x2 = X2[:, :2] / X2[:, 2:] + rng.normal(0, SIGMA, (n, 2))
    n_out = int(outlier_frac * n)
    bad = rng.choice(n, n_out, replace=False)
    x2[bad] += rng.normal(0, 30 * SIGMA, (n_out, 2))          # wrong tracks, about 40 px off
    good = np.ones(n, bool)
    good[bad] = False
    return R, x1, x2, good


def run(t, seed):
    rng = np.random.default_rng(seed)
    R_true, x1, x2, good = make_tracks(rng, t)
    cv2.setRNGSeed(seed)
    E, mask = cv2.findEssentialMat(x1, x2, np.eye(3), method=cv2.RANSAC, prob=0.999, threshold=2 * SIGMA)
    E = E[:3]                                                   # several solutions may be stacked
    _, R, tt, _ = cv2.recoverPose(E, x1, x2, np.eye(3), mask=mask.copy())
    inl = mask.ravel().astype(bool)
    rot_err = np.degrees(np.linalg.norm(Rot.from_matrix(R @ R_true.T).as_rotvec()))
    # rotation-compensated parallax with the true (gyro) rotation
    h = np.column_stack([x1, np.ones(len(x1))]) @ R_true.T
    par = np.linalg.norm(h[:, :2] / h[:, 2:] - x2, axis=1) * F_PX
    return dict(t_dir=tt.ravel(), inliers=inl.sum(), wrong_in=np.sum(inl & ~good), good_out=np.sum(~inl & good),
                rot_err=rot_err, parallax=np.median(par[good]))


for label, t in (("pure rotation", np.zeros(3)), ("10 cm sideways", np.array([-0.1, 0.0, 0.0]))):
    print(f"== {label}")
    for seed in range(4):
        r = run(t, seed)
        print(f"  seed {seed}: t direction {np.round(r['t_dir'], 2)}, inliers {r['inliers']}/150 "
              f"(wrong tracks accepted {r['wrong_in']}/30, good tracks rejected {r['good_out']}/120), "
              f"rotation error {r['rot_err']:.2f} deg, median parallax {r['parallax']:.1f} px")
