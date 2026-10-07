"""Planar simulations for the lecture notes.

* A complete planar MSCKF (Chapter 27): a robot with noisy speed/turn-rate odometry and a
  forward-looking 1D pinhole camera observing point landmarks.
* Planar EKF-SLAM with standard, first-estimates and ideal Jacobians (Chapter 29), the classic
  demonstration of EKF inconsistency along an unobservable direction.

Both write PDF figures into ../figures. Every random generator is seeded.
"""
from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.linalg import qr  # noqa: E402
from scipy.stats import chi2  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(exist_ok=True)
BLUE, ORANGE, GREEN, PURPLE, GREY = "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#555555"

OM = np.array([[0.0, -1.0], [1.0, 0.0]])  # Omega, the planar skew matrix: rot(psi + d) ~ rot(psi) (I + d OM)


def rot(psi):
    """Planar rotation R(psi) = R_wb: maps body-frame vectors to the world frame."""
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s], [s, c]])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# =============================================================================
# Planar MSCKF
# =============================================================================
class PlanarMSCKF:
    """State: robot (p in R^2, psi) followed by N clones (p_i, psi_i). Error state is additive
    (planar rotations commute), so the reset Jacobian is the identity."""

    def __init__(self, p0, psi0, P0, sig_v, sig_w, sig_z, n_max):
        self.p = np.array(p0, float)
        self.psi = float(psi0)
        self.P = np.array(P0, float)
        self.sig_v, self.sig_w, self.sig_z, self.n_max = sig_v, sig_w, sig_z, n_max
        self.clones = []  # list of (clone_id, p, psi)
        self.next_id = 0
        self.rejected = 0
        self.used = 0

    @property
    def n(self):
        return 3 + 3 * len(self.clones)

    def propagate(self, v_m, w_m, dt):
        psi = self.psi
        # covariance first, linearized at the start of the step
        F = np.zeros((3, 3))
        F[0:2, 2] = v_m * (rot(psi) @ np.array([0.0, 1.0]))
        G = np.zeros((3, 2))
        G[0:2, 0] = -rot(psi) @ np.array([1.0, 0.0])
        G[2, 1] = -1.0
        Phi = np.eye(3) + F * dt
        Qd = G @ np.diag([self.sig_v**2, self.sig_w**2]) @ G.T * dt
        P = self.P
        P[:3, :3] = Phi @ P[:3, :3] @ Phi.T + Qd
        P[:3, 3:] = Phi @ P[:3, 3:]
        P[3:, :3] = P[:3, 3:].T
        self.P = 0.5 * (P + P.T)
        # nominal (midpoint in heading)
        self.p = self.p + rot(psi + 0.5 * w_m * dt) @ np.array([v_m * dt, 0.0])
        self.psi = wrap(psi + w_m * dt)

    def clone(self):
        n = self.n
        J = np.zeros((n + 3, n))
        J[:n, :n] = np.eye(n)
        J[n:, :3] = np.eye(3)
        self.P = J @ self.P @ J.T
        cid = self.next_id
        self.next_id += 1
        self.clones.append((cid, self.p.copy(), self.psi))
        return cid

    def marginalize_oldest(self):
        keep = [i for i in range(self.n) if not (3 <= i < 6)]
        self.P = self.P[np.ix_(keep, keep)]
        self.clones.pop(0)

    def clone_index(self, cid):
        for k, (c, _, _) in enumerate(self.clones):
            if c == cid:
                return k
        return None

    def triangulate(self, obs):
        A, b = [], []
        poses = []
        for cid, z in obs:
            k = self.clone_index(cid)
            _, p_i, psi_i = self.clones[k]
            d = rot(psi_i) @ np.array([1.0, z])
            nrm = OM @ d
            A.append(nrm)
            b.append(nrm @ p_i)
            poses.append((p_i, psi_i, z))
        A, b = np.array(A), np.array(b)
        if np.linalg.cond(A) > 1e4:
            return None
        pf = np.linalg.lstsq(A, b, rcond=None)[0]
        for _ in range(5):  # Gauss-Newton on the reprojection error
            Hs, rs = [], []
            for p_i, psi_i, z in poses:
                pb = rot(psi_i).T @ (pf - p_i)
                if pb[0] < 0.2:
                    return None
                Jpi = np.array([-pb[1] / pb[0] ** 2, 1.0 / pb[0]])
                Hs.append(Jpi @ rot(psi_i).T)
                rs.append(z - pb[1] / pb[0])
            H, r = np.array(Hs), np.array(rs)
            pf = pf + np.linalg.solve(H.T @ H, H.T @ r)
        return pf

    def feature_residual(self, obs):
        pf = self.triangulate(obs)
        if pf is None:
            return None
        m = len(obs)
        Hx = np.zeros((m, self.n))
        Hf = np.zeros((m, 2))
        r = np.zeros(m)
        for row, (cid, z) in enumerate(obs):
            k = self.clone_index(cid)
            _, p_i, psi_i = self.clones[k]
            Rt = rot(psi_i).T
            pb = Rt @ (pf - p_i)
            if pb[0] < 0.2:
                return None
            Jpi = np.array([-pb[1] / pb[0] ** 2, 1.0 / pb[0]])  # d(y/x)/d(x, y)
            col = 3 + 3 * k
            Hx[row, col:col + 2] = -Jpi @ Rt          # w.r.t. clone position error
            Hx[row, col + 2] = -Jpi @ (OM @ pb)       # w.r.t. clone heading error
            Hf[row] = Jpi @ Rt                         # w.r.t. feature position error
            r[row] = z - pb[1] / pb[0]
        Q, _ = qr(Hf, mode="full")
        A = Q[:, 2:]                                   # left nullspace of Hf
        Ho, ro = A.T @ Hx, A.T @ r
        S = Ho @ self.P @ Ho.T + self.sig_z**2 * np.eye(m - 2)
        gamma = ro @ np.linalg.solve(S, ro)
        if gamma > chi2.ppf(0.95, m - 2):
            self.rejected += 1
            return None
        self.used += 1
        return Ho, ro

    def update(self, tracks):
        Hs, rs = [], []
        for obs in tracks:
            if len(obs) < 3:
                continue
            res = self.feature_residual(obs)
            if res is not None:
                Hs.append(res[0])
                rs.append(res[1])
        if not Hs:
            return
        H, r = np.vstack(Hs), np.concatenate(rs)
        Rz = self.sig_z**2 * np.eye(len(r))
        S = H @ self.P @ H.T + Rz
        K = np.linalg.solve(S, H @ self.P).T
        dx = K @ r
        IKH = np.eye(self.n) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ Rz @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self.p = self.p + dx[0:2]
        self.psi = wrap(self.psi + dx[2])
        for k in range(len(self.clones)):
            cid, p_i, psi_i = self.clones[k]
            o = 3 + 3 * k
            self.clones[k] = (cid, p_i + dx[o:o + 2], wrap(psi_i + dx[o + 2]))


def planar_scenario(seed):
    rng = np.random.default_rng(seed)
    n_lm = 260
    ang = rng.uniform(0, 2 * np.pi, n_lm)
    rad = np.concatenate([rng.uniform(3.0, 7.0, n_lm // 2), rng.uniform(13.0, 19.0, n_lm - n_lm // 2)])
    landmarks = np.column_stack([rad * np.cos(ang), rad * np.sin(ang)])
    return landmarks


def run_planar_msckf(seed, landmarks, T=90.0, record=False, sig_v=0.01, sig_w=0.02, sig_z=0.003, n_max=10, use_camera=True):
    rng = np.random.default_rng(1000 + seed)
    dt, cam_every = 0.01, 10            # 100 Hz odometry, 10 Hz camera
    R0, v, w = 10.0, 1.0, 0.1           # circle of radius 10 m
    fov = np.tan(np.radians(50.0))
    p_true, psi_true = np.array([R0, 0.0]), np.pi / 2
    P0 = np.diag([1e-4, 1e-4, 1e-5])
    f = PlanarMSCKF(p_true + rng.normal(0, 1e-2, 2), psi_true + rng.normal(0, 3e-3), P0, sig_v, sig_w, sig_z, n_max=n_max)
    dr_p, dr_psi = f.p.copy(), f.psi
    tracks = {}  # landmark id -> list of (clone id, z)
    n = int(T / dt)
    log = {"t": [], "p": [], "psi": [], "pt": [], "psit": [], "P": [], "drp": [], "nees": []}
    for k in range(1, n + 1):
        v_m = v + rng.normal(0, sig_v / np.sqrt(dt))
        w_m = w + rng.normal(0, sig_w / np.sqrt(dt))
        # truth (exact circular motion) and filter
        p_true = p_true + rot(psi_true + 0.5 * w * dt) @ np.array([v * dt * np.sinc(w * dt / 2 / np.pi), 0.0])
        psi_true = wrap(psi_true + w * dt)
        f.propagate(v_m, w_m, dt)
        dr_p = dr_p + rot(dr_psi + 0.5 * w_m * dt) @ np.array([v_m * dt, 0.0])
        dr_psi = wrap(dr_psi + w_m * dt)
        if use_camera and k % cam_every == 0:
            # simulate the image
            pb = (landmarks - p_true) @ rot(psi_true)
            vis = np.where((pb[:, 0] > 1.0) & (pb[:, 0] < 15.0) & (np.abs(pb[:, 1] / pb[:, 0]) < fov))[0]
            meas = {int(j): pb[j, 1] / pb[j, 0] + rng.normal(0, sig_z) for j in vis}
            to_process = [tracks.pop(j) for j in list(tracks) if j not in meas]
            if len(f.clones) == f.n_max:
                oldest = f.clones[0][0]
                for j in list(tracks):
                    if tracks[j][0][0] == oldest:
                        to_process.append(tracks.pop(j))
            f.update(to_process)
            if len(f.clones) == f.n_max:
                f.marginalize_oldest()
            cid = f.clone()
            for j, z in meas.items():
                tracks.setdefault(j, []).append((cid, z))
        if record or k % 10 == 0:
            e = np.concatenate([p_true - f.p, [wrap(psi_true - f.psi)]])
            log["t"].append(k * dt)
            log["p"].append(f.p.copy())
            log["psi"].append(f.psi)
            log["pt"].append(p_true.copy())
            log["psit"].append(psi_true)
            log["P"].append(f.P[:3, :3].copy())
            log["drp"].append(dr_p.copy())
            log["nees"].append(e @ np.linalg.solve(f.P[:3, :3], e))
    for key in log:
        log[key] = np.array(log[key])
    log["used"], log["rejected"] = f.used, f.rejected
    return log


def fig_planar_msckf():
    landmarks = planar_scenario(0)
    runs = [run_planar_msckf(s_, landmarks) for s_ in range(30)]
    log = runs[0]
    t = log["t"]
    # --- trajectory of one run, and RMS position error over all runs
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.2), gridspec_kw={"width_ratios": [1.0, 1.15]})
    ax = axs[0]
    ax.scatter(landmarks[:, 0], landmarks[:, 1], s=3, color=GREY, alpha=0.45, label="landmarks")
    ax.plot(log["pt"][:, 0], log["pt"][:, 1], color="k", lw=1.5, label="truth")
    ax.plot(log["drp"][:, 0], log["drp"][:, 1], color=ORANGE, lw=1.1, ls="--", label="dead reckoning")
    ax.plot(log["p"][:, 0], log["p"][:, 1], color=BLUE, lw=1.1, label="planar MSCKF")
    ax.set_aspect("equal")
    ax.set_xlabel("$x$ [m]")
    ax.set_ylabel("$y$ [m]")
    ax.set_title("(a) one run", fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax = axs[1]
    rms_dr = np.sqrt(np.mean([np.sum((r["pt"] - r["drp"]) ** 2, axis=1) for r in runs], axis=0))
    rms_f = np.sqrt(np.mean([np.sum((r["pt"] - r["p"]) ** 2, axis=1) for r in runs], axis=0))
    ax.plot(t, rms_dr, color=ORANGE, lw=1.2, ls="--", label="dead reckoning")
    ax.plot(t, rms_f, color=BLUE, lw=1.2, label="planar MSCKF")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("RMS position error [m]")
    ax.set_title(f"(b) RMS over {len(runs)} runs", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_planar_traj.pdf")
    plt.close(fig)
    # --- errors of one run with 3 sigma envelopes
    e = np.column_stack([log["pt"] - log["p"], (log["psit"] - log["psi"] + np.pi) % (2 * np.pi) - np.pi])
    sd = np.sqrt(np.array([np.diag(P) for P in log["P"]]))
    fig, axs = plt.subplots(1, 3, figsize=(6.8, 2.4), sharex=True)
    labels = [r"$x$ error [m]", r"$y$ error [m]", r"heading error [rad]"]
    for i in range(3):
        axs[i].plot(t, e[:, i], color=BLUE, lw=0.9)
        axs[i].fill_between(t, -3 * sd[:, i], 3 * sd[:, i], color=BLUE, alpha=0.15, lw=0)
        axs[i].set_title(labels[i], fontsize=9)
        axs[i].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(OUT / "fig_planar_err.pdf")
    plt.close(fig)
    # --- Monte Carlo NEES
    nees = np.mean([r["nees"] for r in runs], axis=0)
    N = len(runs)
    lo, hi = chi2.ppf(0.025, 3 * N) / N, chi2.ppf(0.975, 3 * N) / N
    fig, ax = plt.subplots(figsize=(5.4, 2.4))
    ax.axhspan(lo, hi, color=GREY, alpha=0.2, lw=0, label="95% interval")
    ax.plot(t, nees, color=BLUE, lw=1.0, label=f"average NEES, {N} runs")
    ax.axhline(3, color=GREY, ls="--", lw=0.8)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("NEES (3 DOF)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.savefig(OUT / "fig_planar_nees.pdf")
    plt.close(fig)
    used = sum(r["used"] for r in runs)
    rej = sum(r["rejected"] for r in runs)
    print(f"wrote fig_planar_traj/err/nees.pdf; final RMS: DR {rms_dr[-1]:.2f} m, MSCKF {rms_f[-1]:.2f} m; "
          f"gate rejections {rej}/{used + rej} = {rej / (used + rej):.3f}; final heading sigma {sd[-1, 2]:.4f} rad")


# =============================================================================
# Planar EKF-SLAM: standard vs first-estimates vs ideal Jacobians
# =============================================================================
def run_ekf_slam(seed, variant, steps=600, sig_v=0.05, sig_w=0.004, sig_m=0.05, rmax=5.0, sig_p0=0.0, sig_psi0=0.0, full=False):
    rng = np.random.default_rng(seed)
    dt, v, w = 0.1, 1.0, 0.1
    # sig_v, sig_w: per-step input noise (std); sig_m: measurement noise (std)
    rng_lm = np.random.default_rng(7)
    ang = np.linspace(0, 2 * np.pi, 21)[:-1]
    lms = np.column_stack([(10 + 2.0 * np.sign(np.sin(3 * ang))) * np.cos(ang), (10 + 2.0 * np.sign(np.sin(3 * ang))) * np.sin(ang)])
    lms += rng_lm.normal(0, 0.5, lms.shape)
    p, psi = np.array([10.0, 0.0]), np.pi / 2
    P = np.diag([sig_p0**2, sig_p0**2, sig_psi0**2])
    x = np.array([p[0], p[1], psi]) + rng.normal(0, 1, 3) * np.sqrt(np.diag(P))  # estimate: robot, then landmarks
    lm_index = {}                            # landmark id -> state offset
    first = {}                               # first estimates of landmarks
    xprop_prev = x[:3].copy()                # propagated robot estimate at previous step (for FEJ)
    nees = []
    for k in range(steps):
        vm = v + rng.normal(0, sig_v)
        wm = w + rng.normal(0, sig_w)
        # truth
        p_new = p + rot(psi) @ np.array([v * dt, 0.0])
        psi_new = psi + w * dt
        # propagation
        xr = x[:3]
        xr_new = np.array([*(xr[:2] + rot(xr[2]) @ np.array([vm * dt, 0.0])), xr[2] + wm * dt])
        if variant == "standard":
            dp = xr_new[:2] - xr[:2]
        elif variant == "fej":
            dp = xr_new[:2] - xprop_prev[:2]
        else:  # ideal: true states
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
        x[:3] = xr_new
        p, psi = p_new, psi_new
        xprop_prev = x[:3].copy()
        # measurements: relative position of landmarks within 5 m
        rel = (lms - p) @ rot(psi)
        vis = np.where(np.linalg.norm(rel, axis=1) < rmax)[0]
        zs = {int(j): rel[j] + rng.normal(0, sig_m, 2) for j in vis}
        # 1) new landmarks enter the state first, at the propagated estimate x_{k+1|k}
        new = [j for j in zs if j not in lm_index]
        for j in new:
            z = zs[j]
            if variant == "ideal":
                pr, ps = p, psi
            else:
                pr, ps = x[:2], x[2]
            pl = x[:2] + rot(x[2]) @ z
            n = len(x)
            Jx = np.zeros((2, n))
            Jx[:, 0:2] = np.eye(2)
            Jx[:, 2] = OM @ rot(ps) @ z
            Jz = rot(ps)
            Pnew = np.zeros((n + 2, n + 2))
            Pnew[:n, :n] = P
            Pnew[n:, :n] = Jx @ P
            Pnew[:n, n:] = (Jx @ P).T
            Pnew[n:, n:] = Jx @ P @ Jx.T + Jz @ (sig_m**2 * np.eye(2)) @ Jz.T
            P = Pnew
            x = np.concatenate([x, pl])
            lm_index[j] = n
            first[j] = pl.copy()
        # 2) sequential updates with the re-observed landmarks
        for j, z in zs.items():
            if j in new:
                continue
            o = lm_index[j]
            # linearization point
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
            r = z - rot(x[2]).T @ (x[o:o + 2] - x[:2])
            S = H @ P @ H.T + sig_m**2 * np.eye(2)
            K = np.linalg.solve(S, H @ P).T
            x = x + K @ r
            x[2] = wrap(x[2])
            IKH = np.eye(n) - K @ H
            P = IKH @ P @ IKH.T + K @ (sig_m**2 * np.eye(2)) @ K.T
            P = 0.5 * (P + P.T)
        e = np.array([*(p - x[:2]), wrap(psi - x[2])])
        nees.append([e @ np.linalg.solve(P[:3, :3] + 1e-12 * np.eye(3), e), e[2] ** 2, P[2, 2]])
    nees = np.array(nees)
    return nees if full else nees[:, 0]


def fig_fej():
    N, steps = 50, 1200
    kw = dict(sig_v=0.05, sig_w=0.01, sig_m=0.1, rmax=5.0, sig_p0=0.1, sig_psi0=0.05, full=True)
    t = np.arange(1, steps + 1) * 0.1
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.9))
    lo, hi = chi2.ppf(0.025, 3 * N) / N, chi2.ppf(0.975, 3 * N) / N
    axs[0].axhspan(lo, hi, color=GREY, alpha=0.2, lw=0, label="95% interval")
    for variant, c, lab in [("standard", ORANGE, "standard EKF"), ("fej", BLUE, "FEJ-EKF"), ("ideal", GREEN, "ideal EKF")]:
        arr = np.array([run_ekf_slam(100 + s_, variant, steps, **kw) for s_ in range(N)])
        m = arr.mean(axis=0)
        axs[0].plot(t, m[:, 0], color=c, lw=1.0, label=lab)
        axs[1].plot(t, np.degrees(np.sqrt(m[:, 1])), color=c, lw=1.0)
        axs[1].plot(t, np.degrees(np.sqrt(m[:, 2])), color=c, lw=1.0, ls="--")
        print(f"  {variant}: final NEES {m[-1, 0]:.2f}, rms heading {np.degrees(np.sqrt(m[-1, 1])):.2f} deg, "
              f"filter sigma {np.degrees(np.sqrt(m[-1, 2])):.2f} deg")
    axs[0].axhline(3, color=GREY, ls="--", lw=0.8)
    axs[0].set_xlabel("time [s]")
    axs[0].set_ylabel("robot-pose NEES (3 DOF)")
    axs[0].set_title("(a) average NEES, %d runs" % N, fontsize=9)
    axs[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    axs[1].set_xlabel("time [s]")
    axs[1].set_ylabel("heading [deg]")
    axs[1].set_title(r"(b) RMS error (solid) vs filter $\sigma$ (dashed)", fontsize=9)
    axs[1].set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(OUT / "fig_fej.pdf")
    plt.close(fig)
    print("wrote fig_fej.pdf")


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
                         "font.size": 9.5, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.03})
    import sys
    which = sys.argv[1:] or ["msckf", "fej"]
    if "msckf" in which:
        fig_planar_msckf()
    if "fej" in which:
        fig_fej()
