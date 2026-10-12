"""Exercise solutions, Chapter 29, exercise 5 (||H Phi...Phi N|| for the three planar EKF-SLAM filters): prints the numbers quoted in the worked solutions.

Run from DOCS/PDFs/Guide:  .venv/Scripts/python.exe scripts/solutions/<this file>
"""
from pathlib import Path
# An instrumented copy of planar_sims.run_ekf_slam that tracks ||H_k Phi_{k-1}...Phi_1 N||.
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # DOCS/PDFs/Guide/scripts
from planar_sims import rot, wrap, OM  # noqa: E402


def run(seed, variant, steps=1200, sig_v=0.05, sig_w=0.01, sig_m=0.1, rmax=5.0, sig_p0=0.1, sig_psi0=0.05, ideal_true_aug=False):
    rng = np.random.default_rng(seed)
    dt, v, w = 0.1, 1.0, 0.1
    rng_lm = np.random.default_rng(7)
    ang = np.linspace(0, 2 * np.pi, 21)[:-1]
    lms = np.column_stack([(10 + 2.0 * np.sign(np.sin(3 * ang))) * np.cos(ang), (10 + 2.0 * np.sign(np.sin(3 * ang))) * np.sin(ang)])
    lms += rng_lm.normal(0, 0.5, lms.shape)
    p, psi = np.array([10.0, 0.0]), np.pi / 2
    P = np.diag([sig_p0**2, sig_p0**2, sig_psi0**2])
    x = np.array([p[0], p[1], psi]) + rng.normal(0, 1, 3) * np.sqrt(np.diag(P))
    lm_index, first = {}, {}
    xprop_prev = x[:3].copy()
    # cumulative nullspace image: columns (t_x, t_y, rotation), rows = error state
    if variant == "ideal":
        Ncum = np.array([[1, 0, -p[1]], [0, 1, p[0]], [0, 0, 1.0]])
    else:
        Ncum = np.array([[1, 0, -x[1]], [0, 1, x[0]], [0, 0, 1.0]])
    leak_t, leak_r = [], []
    for k in range(steps):
        vm = v + rng.normal(0, sig_v)
        wm = w + rng.normal(0, sig_w)
        p_new = p + rot(psi) @ np.array([v * dt, 0.0])
        psi_new = psi + w * dt
        xr = x[:3]
        xr_new = np.array([*(xr[:2] + rot(xr[2]) @ np.array([vm * dt, 0.0])), xr[2] + wm * dt])
        if variant == "standard":
            dp = xr_new[:2] - xr[:2]
        elif variant == "fej":
            dp = xr_new[:2] - xprop_prev[:2]
        else:
            dp = p_new - p
        Phi = np.eye(3)
        Phi[0:2, 2] = OM @ dp
        psi_lin = {"standard": xr[2], "fej": xprop_prev[2], "ideal": psi}[variant]
        G = np.zeros((3, 2))
        G[0:2, 0] = rot(psi_lin) @ np.array([dt, 0.0])
        G[2, 1] = dt
        n = len(x)
        F = np.eye(n)
        F[:3, :3] = Phi
        Q = np.zeros((n, n))
        Q[:3, :3] = G @ np.diag([sig_v**2, sig_w**2]) @ G.T
        P = F @ P @ F.T + Q
        Ncum = F @ Ncum
        x[:3] = xr_new
        p, psi = p_new, psi_new
        xprop_prev = x[:3].copy()
        rel = (lms - p) @ rot(psi)
        vis = np.where(np.linalg.norm(rel, axis=1) < rmax)[0]
        zs = {int(j): rel[j] + rng.normal(0, sig_m, 2) for j in vis}
        new = [j for j in zs if j not in lm_index]
        for j in new:
            z = zs[j]
            ps = psi if variant == "ideal" else x[2]
            pl = x[:2] + rot(x[2]) @ z
            n = len(x)
            Jx = np.zeros((2, n))
            Jx[:, 0:2] = np.eye(2)
            if variant == "ideal" and ideal_true_aug:
                Jx[:, 2] = OM @ (lms[j] - p)          # true-state augmentation Jacobian (noise-free z)
            else:
                Jx[:, 2] = OM @ rot(ps) @ z
            Jz = rot(ps)
            Pnew = np.zeros((n + 2, n + 2))
            Pnew[:n, :n] = P
            Pnew[n:, :n] = Jx @ P
            Pnew[:n, n:] = (Jx @ P).T
            Pnew[n:, n:] = Jx @ P @ Jx.T + Jz @ (sig_m**2 * np.eye(2)) @ Jz.T
            P = Pnew
            Ncum = np.vstack([Ncum, Jx @ Ncum])
            x = np.concatenate([x, pl])
            lm_index[j] = n
            first[j] = pl.copy()
        for j, z in zs.items():
            if j in new:
                continue
            o = lm_index[j]
            if variant == "ideal":
                pr, ps, pl = p, psi, lms[j]
            elif variant == "fej":
                pr, ps, pl = xprop_prev[:2], xprop_prev[2], first[j]
            else:
                pr, ps, pl = x[:2], x[2], x[o:o + 2]
            n = len(x)
            H = np.zeros((2, n))
            H[:, 0:2] = -rot(ps).T
            H[:, 2] = -OM @ rot(ps).T @ (pl - pr)
            H[:, o:o + 2] = rot(ps).T
            HN = H @ Ncum
            leak_t.append(np.linalg.norm(HN[:, :2]))
            leak_r.append(np.linalg.norm(HN[:, 2]))
            r = z - rot(x[2]).T @ (x[o:o + 2] - x[:2])
            S = H @ P @ H.T + sig_m**2 * np.eye(2)
            K = np.linalg.solve(S, H @ P).T
            x = x + K @ r
            x[2] = wrap(x[2])
            IKH = np.eye(n) - K @ H
            P = IKH @ P @ IKH.T + K @ (sig_m**2 * np.eye(2)) @ K.T
            P = 0.5 * (P + P.T)
    return np.array(leak_t), np.array(leak_r)


for variant, kw, tag in (("standard", {}, "standard"), ("fej", {}, "FEJ"),
                         ("ideal", {"ideal_true_aug": True}, "ideal (as in planar_sims.py)"),
                         ("ideal", {}, "ideal, noisy augmentation Jacobian")):
    lt, lr = [], []
    for s in range(5):
        a, b = run(100 + s, variant, **kw)
        lt.append(a.max())
        lr.append(b)
    lr_all = np.concatenate(lr)
    print(f"{tag:38s}: max ||H N_t|| {max(lt):.1e}; ||H N_r||: median {np.median(lr_all):.1e}, max {lr_all.max():.1e}; "
          f"median over last 200 updates {np.median(np.concatenate([x[-200:] for x in lr])):.1e}")
