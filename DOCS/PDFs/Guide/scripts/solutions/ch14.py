"""Exercise solutions, Chapter 14: prints the numbers quoted in the worked solutions.

Run from DOCS/PDFs/Guide:  .venv/Scripts/python.exe scripts/solutions/<this file>
"""
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # DOCS/PDFs/Guide/scripts
from make_figures import Exp, skew  # noqa: E402


def Jr(p):
    t = np.linalg.norm(p)
    K = skew(p)
    return np.eye(3) - (1 - np.cos(t)) / t**2 * K + (t - np.sin(t)) / t**3 * K @ K


d = np.array([0, 0, 0.1])
P = np.diag([1e-4, 1e-4, 1e-2])
G = np.eye(3) - 0.5 * skew(d)
J = Jr(d)
print("14.2 GPG^T diag", np.diag(G @ P @ G.T), " JPJ^T diag", np.diag(J @ P @ J.T))
print("     max offdiag", np.abs(G @ P @ G.T - np.diag(np.diag(G @ P @ G.T))).max(), np.abs(J @ P @ J.T - np.diag(np.diag(J @ P @ J.T))).max())
print("     relative diff xx:", (G @ P @ G.T)[0, 0] / (J @ P @ J.T)[0, 0] - 1)
# anisotropic example to show first-order agreement
P2 = np.diag([1e-4, 1e-2, 1e-4])
print("     anisotropic: GPG", np.round(G @ P2 @ G.T, 8), "\n JPJ", np.round(J @ P2 @ J.T, 8))
for th in (0.003, 0.03, 0.3, 1.0):
    dd = np.array([th, 0, 0]) / 1.0
    dd = np.array([0, 0, th])
    G = np.eye(3) - 0.5 * skew(dd)
    J = Jr(dd)
    M1, M2 = G @ P2 @ G.T, J @ P2 @ J.T
    print(f"     |dtheta|={th}: max rel diff {np.abs(M1-M2).max()/np.abs(M2).max():.1e}")


# 14.4 AHRS still
def ahrs(still, T=60.0, seed=21):
    rng = np.random.default_rng(seed)
    dt = 0.01
    n = int(round(T / dt))
    sig_g, sig_b, sig_a = 5e-3, 1e-5, 0.05
    v = np.array([0.0, 0.0, 9.81])
    b_true = np.array([0.008, -0.006, 0.005])
    R = np.eye(3)
    P = np.diag([0.05**2, 0.05**2, 0.01**2, 0.01**2, 0.01**2, 0.01**2])
    dth0 = rng.normal(0, 1, 3) * np.sqrt(np.diag(P)[:3])
    Rh = Exp(-dth0) @ R
    bh = np.zeros(3)
    I3 = np.eye(3)
    for k in range(n):
        t = k * dt
        w = np.zeros(3) if still else np.array([0.6 * np.sin(0.31 * t), 0.5 * np.cos(0.23 * t + 0.4), 0.4 * np.sin(0.17 * t + 1.0)])
        wm = w + b_true + rng.normal(0, sig_g / np.sqrt(dt), 3)
        R = R @ Exp(w * dt)
        F = np.zeros((6, 6)); F[0:3, 3:6] = -Rh
        Phi = np.eye(6) + F * dt
        Gn = np.zeros((6, 6)); Gn[0:3, 0:3] = -Rh; Gn[3:6, 3:6] = I3
        P = Phi @ P @ Phi.T + Gn @ np.diag([sig_g**2] * 3 + [sig_b**2] * 3) @ Gn.T * dt
        Rh = Rh @ Exp((wm - bh) * dt)
        if (k + 1) % 10 == 0:
            am = R.T @ v + rng.normal(0, sig_a, 3)
            H = np.zeros((3, 6)); H[:, 0:3] = Rh.T @ skew(v)
            r = am - Rh.T @ v
            S = H @ P @ H.T + sig_a**2 * I3
            K = np.linalg.solve(S, H @ P).T
            dx = K @ r
            IKH = np.eye(6) - K @ H
            P = IKH @ P @ IKH.T + K @ (sig_a**2 * I3) @ K.T
            Rh = Exp(dx[0:3]) @ Rh
            bh = bh + dx[3:6]
            Gr = np.eye(6); Gr[0:3, 0:3] = I3 + 0.5 * skew(dx[0:3])
            P = Gr @ P @ Gr.T
            P = 0.5 * (P + P.T)
    return np.sqrt(np.diag(P))[3:], b_true - bh


for still in (False, True):
    s, e = ahrs(still)
    print(f"14.4 still={still}: final bias sigma {np.round(s, 5)}, final bias error {np.round(e, 5)}")
