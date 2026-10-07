"""Generate every data-driven figure of the VIO lecture notes.

Run from the Guide folder with the local venv:
    .venv\\Scripts\\python.exe scripts\\make_figures.py

All randomness uses fixed seeds, so the figures are reproducible.
"""
from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402
from mpmath import mp, mpf  # noqa: E402
from mpmath import cos as mcos  # noqa: E402
from mpmath import sin as msin  # noqa: E402
from mpmath import tan as mtan  # noqa: E402
from scipy.linalg import expm, polar  # noqa: E402
from scipy.stats import chi2  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(exist_ok=True)

BLUE, ORANGE, GREEN, PURPLE = "#0072B2", "#D55E00", "#009E73", "#CC79A7"
YELLOW, SKY, GREY = "#E69F00", "#56B4E9", "#555555"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9.5,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.6,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})

G = 9.81
# EuRoC imu0/sensor.yaml (ADIS16448), continuous-time densities
SIG_G, SIG_BG = 1.6968e-04, 1.9393e-05
SIG_A, SIG_BA = 2.0e-3, 3.0e-3


def skew(a):
    return np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])


def Exp(phi):
    th = np.linalg.norm(phi)
    K = skew(phi)
    if th < 1e-8:
        return np.eye(3) + K + 0.5 * K @ K
    return np.eye(3) + np.sin(th) / th * K + (1 - np.cos(th)) / th**2 * K @ K


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", name)


# ----------------------------------------------------------------------------
# Fig 1: deterministic dead-reckoning drift
# ----------------------------------------------------------------------------
def fig_dr_drift():
    t = np.logspace(-1, 2, 300)
    b_a, b_g, tilt = 0.02, 1e-3, 1e-3
    fig, ax = plt.subplots(figsize=(5.6, 3.1))
    ax.loglog(t, 0.5 * b_a * t**2, color=BLUE,
              label=r"accel. bias $b_a=0.02\,\mathrm{m/s^2}$: $\frac{1}{2}b_a t^2$")
    ax.loglog(t, 0.5 * G * tilt * t**2, color=GREEN,
              label=r"initial tilt $1\,\mathrm{mrad}$: $\frac{1}{2}g\,\delta\theta_0\, t^2$")
    ax.loglog(t, G * b_g * t**3 / 6, color=ORANGE,
              label=r"gyro bias $b_g=10^{-3}\,\mathrm{rad/s}$: $\frac{1}{6}g\,b_g t^3$")
    for y, lab in [(0.01, "1 cm"), (1.0, "1 m"), (100.0, "100 m")]:
        ax.axhline(y, color=GREY, lw=0.7, ls=":")
        ax.text(0.105, y * 1.15, lab, color=GREY, fontsize=8)
    ax.set_xlabel("time since last correction $t$ [s]")
    ax.set_ylabel("horizontal position error [m]")
    ax.set_ylim(1e-5, 1e4)
    ax.legend(loc="upper left", frameon=False)
    save(fig, "fig_dr_drift.pdf")


# ----------------------------------------------------------------------------
# Fig 2: cancellation in coefficient functions, Taylor truncation, acos vs atan2
# ----------------------------------------------------------------------------
def fig_taylor():
    mp.dps = 60
    th = np.logspace(-9, 0, 400)

    def exact(f):
        return np.array([float(f(mpf(float(x)))) for x in th])

    B_ex = exact(lambda x: (1 - mcos(x)) / x**2)
    C_ex = exact(lambda x: (x - msin(x)) / x**3)
    D_ex = exact(lambda x: 1 / x**2 - 1 / (2 * x * mtan(x / 2)))
    with np.errstate(divide="ignore", invalid="ignore"):
        B_nv = (1 - np.cos(th)) / th**2
        C_nv = (th - np.sin(th)) / th**3
        D_nv = 1 / th**2 - 1 / (2 * th * np.tan(th / 2))
    B_t2 = 0.5 - th**2 / 24
    C_t2 = 1 / 6 - th**2 / 120

    def rel(a, b):
        e = np.abs(a - b) / np.abs(b)
        return np.clip(e, 1e-17, 10)

    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.9))
    ax = axs[0]
    ax.loglog(th, rel(B_nv, B_ex), color=BLUE, lw=1.0, label=r"naive $(1-\cos\theta)/\theta^2$")
    ax.loglog(th, rel(C_nv, C_ex), color=ORANGE, lw=1.0, label=r"naive $(\theta-\sin\theta)/\theta^3$")
    ax.loglog(th, rel(D_nv, D_ex), color=PURPLE, lw=1.0,
              label=r"naive $\frac{1}{\theta^2}-\frac{\mathrm{cot}(\theta/2)}{2\theta}$")
    ax.loglog(th, rel(B_t2, B_ex), color=BLUE, ls="--", lw=1.2, label=r"Taylor $\frac{1}{2}-\frac{\theta^2}{24}$")
    ax.loglog(th, rel(C_t2, C_ex), color=ORANGE, ls="--", lw=1.2, label=r"Taylor $\frac{1}{6}-\frac{\theta^2}{120}$")
    ax.axhline(2.2e-16, color=GREY, lw=0.8, ls=":")
    ax.text(1.5e-9, 4e-16, r"$\varepsilon_{\mathrm{mach}}$", color=GREY, fontsize=8)
    ax.set_xlabel(r"rotation angle $\theta$ [rad]")
    ax.set_ylabel("relative error of coefficient")
    ax.set_ylim(1e-17, 20)
    ax.set_title("(a) coefficient functions in double precision")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.1), frameon=False, fontsize=6.6)

    ax = axs[1]
    rng = np.random.default_rng(3)
    thetas = np.logspace(-10, -0.2, 120)
    e_acos, e_atan = [], []
    for t in thetas:
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        R = Exp(t * u)
        c = np.clip((np.trace(R) - 1) / 2, -1, 1)
        s = np.linalg.norm(np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])) / 2
        e_acos.append(abs(np.arccos(c) - t) / t)
        e_atan.append(abs(np.arctan2(s, c) - t) / t)
    ax.loglog(thetas, np.clip(e_acos, 1e-17, 10), "o", ms=2.2, color=ORANGE,
              label=r"$\theta=\arccos\frac{\mathrm{tr}R-1}{2}$")
    ax.loglog(thetas, np.clip(e_atan, 1e-17, 10), "s", ms=2.0, color=BLUE,
              label=r"$\theta=\mathrm{atan2}(\|\mathrm{vee}(R-R^{\top})\|/2,\ \frac{\mathrm{tr}R-1}{2})$")
    ax.axhline(2.2e-16, color=GREY, lw=0.8, ls=":")
    ax.set_xlabel(r"true angle $\theta$ [rad]")
    ax.set_ylabel(r"relative error of recovered $\theta$")
    ax.set_title(r"(b) recovering $\theta$ from $R$")
    ax.set_ylim(1e-17, 20)
    ax.legend(loc="upper right", frameon=False, fontsize=6.6)
    fig.tight_layout()
    save(fig, "fig_taylor.pdf")


# ----------------------------------------------------------------------------
# Fig 3: banana distribution (uncertainty on a manifold)
# ----------------------------------------------------------------------------
def fig_banana():
    rng = np.random.default_rng(7)
    n = 2500
    L = 10.0
    sig_psi, sig_fwd = 0.25, 0.15
    psi = rng.normal(0, sig_psi, n)
    d = L + rng.normal(0, sig_fwd, n)
    x, y = d * np.cos(psi), d * np.sin(psi)
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.scatter(x, y, s=1.2, color=BLUE, alpha=0.35, label="samples of the true pose", rasterized=True)
    # linearised Gaussian: mean (L,0), cov diag(sig_fwd^2, (L sig_psi)^2)
    for k, ls in [(1, "-"), (3, "--")]:
        e = Ellipse((L, 0), 2 * k * sig_fwd, 2 * k * L * sig_psi, fill=False, color=ORANGE, ls=ls, lw=1.4)
        ax.add_patch(e)
    ax.plot([], [], color=ORANGE, label=r"linearised Gaussian ($1\sigma$, $3\sigma$)")
    ax.plot(0, 0, "k^", ms=7)
    ax.annotate("start", (0, 0), (0.4, -1.6), fontsize=8)
    ax.plot(L, 0, "o", color=ORANGE, ms=4)
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(-7.5, 7.5)
    ax.set_xlabel("$x$ [m]")
    ax.set_ylabel("$y$ [m]")
    ax.legend(loc="upper left", frameon=False, markerscale=5)
    save(fig, "fig_banana.pdf")


# ----------------------------------------------------------------------------
# Fig 4: chi-square densities and 95% quantiles
# ----------------------------------------------------------------------------
def fig_chi2():
    x = np.linspace(0.01, 20, 600)
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    for k, c in [(1, BLUE), (2, ORANGE), (3, GREEN), (6, PURPLE)]:
        ax.plot(x, chi2.pdf(x, k), color=c, label=rf"$k={k}$")
        q = chi2.ppf(0.95, k)
        ax.axvline(q, color=c, lw=0.9, ls="--")
        ax.text(q + 0.15, 0.42 - 0.05 * [1, 2, 3, 6].index(k), rf"$\chi^2_{{0.95}}({k})={q:.2f}$",
                color=c, fontsize=7.5)
    ax.set_ylim(0, 0.5)
    ax.set_xlim(0, 20)
    ax.set_xlabel(r"$\gamma = r^{\top}S^{-1}r$")
    ax.set_ylabel("probability density")
    ax.legend(frameon=False, loc="upper right")
    save(fig, "fig_chi2.pdf")


# ----------------------------------------------------------------------------
# Fig 5: 1D Kalman filter
# ----------------------------------------------------------------------------
def fig_kf1d():
    rng = np.random.default_rng(11)
    dt, T = 0.1, 20.0
    n = int(T / dt)
    q, r = 0.05, 1.0
    F = np.array([[1, dt], [0, 1]])
    Qd = q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
    H = np.array([[1.0, 0.0]])
    x = np.array([0.0, 1.0])
    xh = np.array([0.0, 0.0])
    P = np.diag([4.0, 4.0])
    Lq = np.linalg.cholesky(Qd)
    ts, xs, zs, xhs, sp, sv, vs, vhs = [], [], [], [], [], [], [], []
    for k in range(n):
        x = F @ x + Lq @ rng.normal(size=2)
        xh = F @ xh
        P = F @ P @ F.T + Qd
        z = x[0] + np.sqrt(r) * rng.normal()
        S = H @ P @ H.T + r
        K = P @ H.T / S
        xh = xh + (K * (z - H @ xh)).ravel()
        IKH = np.eye(2) - K @ H
        P = IKH @ P @ IKH.T + r * K @ K.T
        ts.append((k + 1) * dt)
        xs.append(x[0])
        zs.append(z)
        xhs.append(xh[0])
        vs.append(x[1])
        vhs.append(xh[1])
        sp.append(np.sqrt(P[0, 0]))
        sv.append(np.sqrt(P[1, 1]))
    ts, xs, zs, xhs, sp = map(np.array, (ts, xs, zs, xhs, sp))
    vs, vhs, sv = map(np.array, (vs, vhs, sv))
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.7))
    ax = axs[0]
    ax.plot(ts, zs, ".", ms=2.5, color=GREY, label="measurements $z_k$")
    ax.plot(ts, xs, color="k", lw=1.0, label="true position")
    ax.plot(ts, xhs, color=BLUE, lw=1.2, label="KF estimate")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("position [m]")
    ax.legend(frameon=False, fontsize=7.5)
    ax.set_title("(a) estimate")
    ax = axs[1]
    ax.plot(ts, xhs - xs, color=BLUE, lw=1.0, label=r"position error $\hat p-p$")
    ax.fill_between(ts, -3 * sp, 3 * sp, color=BLUE, alpha=0.15, label=r"$\pm3\sigma_p$")
    ax.plot(ts, vhs - vs, color=ORANGE, lw=1.0, label=r"velocity error $\hat v-v$")
    ax.plot(ts, 3 * sv, color=ORANGE, lw=0.8, ls="--", label=r"$\pm3\sigma_v$")
    ax.plot(ts, -3 * sv, color=ORANGE, lw=0.8, ls="--")
    ax.set_ylim(-3, 3)
    ax.set_xlabel("time [s]")
    ax.set_title(r"(b) errors and $3\sigma$ envelopes")
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout()
    save(fig, "fig_kf1d.pdf")


# ----------------------------------------------------------------------------
# Fig 6: Allan deviation with EuRoC parameters
# ----------------------------------------------------------------------------
def allan_dev(y, fs, taus_m):
    """Overlapping Allan deviation of rate samples y (fs Hz) for cluster sizes m."""
    theta = np.cumsum(y) / fs
    out = []
    n = len(theta)
    for m in taus_m:
        tau = m / fs
        d = theta[2 * m:] - 2 * theta[m:n - m] + theta[:n - 2 * m]
        out.append(np.sqrt(np.mean(d**2) / (2 * tau**2)))
    return np.array(out)


def fig_allan():
    rng = np.random.default_rng(5)
    fs = 200.0
    hours = 4.0
    n = int(hours * 3600 * fs)
    dt = 1 / fs
    m = np.unique(np.logspace(0, np.log10(n / 5), 70).astype(int))
    taus = m / fs
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.9))
    for ax, sig_w, sig_b, unit, name in [
        (axs[0], SIG_G, SIG_BG, r"rad/s", "gyroscope"),
        (axs[1], SIG_A, SIG_BA, r"m/s$^2$", "accelerometer"),
    ]:
        white = sig_w / np.sqrt(dt) * rng.normal(size=n)
        bias = np.cumsum(sig_b * np.sqrt(dt) * rng.normal(size=n))
        ad = allan_dev(white + bias, fs, m)
        ax.loglog(taus, ad, color=BLUE, lw=1.4, label="simulated sensor")
        ax.loglog(taus, sig_w / np.sqrt(taus), color=ORANGE, ls="--", lw=1.0,
                  label=r"white noise: $\sigma/\sqrt{\tau}$ (slope $-\frac{1}{2}$)")
        ax.loglog(taus, sig_b * np.sqrt(taus / 3), color=GREEN, ls="--", lw=1.0,
                  label=r"bias random walk: $\sigma_b\sqrt{\tau/3}$ (slope $+\frac{1}{2}$)")
        ax.plot([1.0], [sig_w], "o", color=ORANGE, ms=4)
        ax.annotate(rf"$\sigma$ read at $\tau=1$ s", (1.0, sig_w), (0.012, sig_w * 0.25), fontsize=7,
                    arrowprops=dict(arrowstyle="-", color=GREY, lw=0.6))
        ax.plot([3.0], [sig_b], "o", color=GREEN, ms=4)
        ax.annotate(rf"$\sigma_b$ read at $\tau=3$ s", (3.0, sig_b), (12, sig_b * 0.35), fontsize=7,
                    arrowprops=dict(arrowstyle="-", color=GREY, lw=0.6))
        ax.set_xlabel(r"averaging time $\tau$ [s]")
        ax.set_ylabel(rf"Allan deviation [{unit}]")
        ax.set_title(f"({'a' if name == 'gyroscope' else 'b'}) {name}")
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=7.5, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    save(fig, "fig_allan.pdf")


# ----------------------------------------------------------------------------
# Fig 7: convergence order of integration schemes
# ----------------------------------------------------------------------------
OM, RAD, HZ, NU, BETA, MU = 0.8, 2.0, 0.5, 1.3, 0.4, 2.1
GW = np.array([0.0, 0.0, -G])


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def traj(t):
    p = np.array([RAD * np.cos(OM * t), RAD * np.sin(OM * t), HZ * np.sin(NU * t)])
    v = np.array([-RAD * OM * np.sin(OM * t), RAD * OM * np.cos(OM * t), HZ * NU * np.cos(NU * t)])
    a = np.array([-RAD * OM**2 * np.cos(OM * t), -RAD * OM**2 * np.sin(OM * t), -HZ * NU**2 * np.sin(NU * t)])
    psi, psid = OM * t + np.pi / 2, OM
    phi, phid = BETA * np.sin(MU * t), BETA * MU * np.cos(MU * t)
    R = Rz(psi) @ Rx(phi)
    w_b = psid * Rx(phi).T @ np.array([0, 0, 1.0]) + phid * np.array([1.0, 0, 0])
    a_m = R.T @ (a - GW)
    return p, v, R, w_b, a_m


def integrate(scheme, dt, T):
    n = int(round(T / dt))
    p, v, R, _, _ = traj(0.0)
    for k in range(n):
        t0, t1 = k * dt, (k + 1) * dt
        _, _, _, w0, a0 = traj(t0)
        _, _, _, w1, a1 = traj(t1)
        if scheme == "euler":
            aw = R @ a0 + GW
            p = p + v * dt + 0.5 * aw * dt**2
            v = v + aw * dt
            R = R @ Exp(w0 * dt)
        elif scheme == "midpoint":
            R1 = R @ Exp(0.5 * (w0 + w1) * dt)
            aw = 0.5 * (R @ a0 + R1 @ a1) + GW
            p = p + v * dt + 0.5 * aw * dt**2
            v = v + aw * dt
            R = R1
        else:  # rk4 (exact mid-samples) or rk4lin (linearly interpolated mid-samples)
            if scheme == "rk4":
                _, _, _, wm, am = traj(t0 + 0.5 * dt)
            else:
                wm, am = 0.5 * (w0 + w1), 0.5 * (a0 + a1)

            def f(Rk, vk, w, a):
                return Rk @ skew(w), vk, Rk @ a + GW

            k1 = f(R, v, w0, a0)
            k2 = f(R + 0.5 * dt * k1[0], v + 0.5 * dt * k1[2], wm, am)
            k3 = f(R + 0.5 * dt * k2[0], v + 0.5 * dt * k2[2], wm, am)
            k4 = f(R + dt * k3[0], v + dt * k3[2], w1, a1)
            R = R + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            p = p + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            v = v + dt / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
            R, _ = polar(R)
    return p


def fig_integration_order():
    T = 10.0
    p_true = traj(T)[0]
    dts = np.array([1 / 25, 1 / 50, 1 / 100, 1 / 200, 1 / 400, 1 / 800])
    fig, ax = plt.subplots(figsize=(5.6, 3.1))
    for scheme, c, lab in [
        ("euler", ORANGE, "Euler (zero-order hold)"),
        ("midpoint", BLUE, "midpoint / trapezoidal"),
        ("rk4lin", PURPLE, "RK4, interpolated samples"),
        ("rk4", GREEN, "RK4, exact mid-samples"),
    ]:
        err = np.array([np.linalg.norm(integrate(scheme, dt, T) - p_true) for dt in dts])
        slope = np.polyfit(np.log(dts[2:]), np.log(err[2:]), 1)[0]
        ax.loglog(dts, err, "o-", color=c, ms=3.5, label=f"{lab} (slope {slope:.2f})")
    ax.axvline(1 / 200, color=GREY, ls=":", lw=0.8)
    ax.text(1 / 200 * 1.05, 2e-11, "EuRoC\n200 Hz", color=GREY, fontsize=7.5)
    ax.set_xticks(dts)
    ax.set_xticklabels([f"1/{int(round(1 / d))}" for d in dts])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(r"IMU period $\Delta t$ [s]")
    ax.set_ylabel(r"position error after $10$ s [m]")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    save(fig, "fig_integration_order.pdf")


# ----------------------------------------------------------------------------
# Fig 8: covariance growth during dead reckoning (analytic + Monte Carlo)
# ----------------------------------------------------------------------------
def fig_cov_growth():
    t = np.logspace(-1, 2, 300)
    s_aw = SIG_A * np.sqrt(t**3 / 3)
    s_gw = G * SIG_G * np.sqrt(t**5 / 20)
    s_ab = SIG_BA * np.sqrt(t**5 / 20)
    s_gb = G * SIG_BG * np.sqrt(t**7 / 252)
    total = np.sqrt(s_aw**2 + s_gw**2 + s_ab**2 + s_gb**2)
    # Monte Carlo of the scalar horizontal chain: dth' = -n_bg_int..., level static IMU
    rng = np.random.default_rng(21)
    dt, T, M = 0.005, 100.0, 400
    n = int(T / dt)
    dth = np.zeros(M)
    dv = np.zeros(M)
    dp = np.zeros(M)
    dba = np.zeros(M)
    dbg = np.zeros(M)
    sd = np.sqrt(dt)
    rec_t, rec_s = [], []
    checkpoints = set(np.unique(np.logspace(np.log10(20), np.log10(n), 25).astype(int)))
    for k in range(1, n + 1):
        na = SIG_A / sd * rng.normal(size=M)
        ng = SIG_G / sd * rng.normal(size=M)
        dp += dv * dt
        dv += (G * dth - dba - na) * dt
        dth += (-dbg - ng) * dt
        dba += SIG_BA * sd * rng.normal(size=M)
        dbg += SIG_BG * sd * rng.normal(size=M)
        if k in checkpoints:
            rec_t.append(k * dt)
            rec_s.append(dp.std())
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.loglog(t, s_aw, color=BLUE, lw=1.1, label=r"accel. white noise: $\sigma_a\sqrt{t^3/3}$")
    ax.loglog(t, s_gw, color=ORANGE, lw=1.1, label=r"gyro white noise: $g\sigma_g\sqrt{t^5/20}$")
    ax.loglog(t, s_ab, color=GREEN, lw=1.1, label=r"accel. bias walk: $\sigma_{b_a}\sqrt{t^5/20}$")
    ax.loglog(t, s_gb, color=PURPLE, lw=1.1, label=r"gyro bias walk: $g\sigma_{b_g}\sqrt{t^7/252}$")
    ax.loglog(t, total, color="k", lw=1.8, label="total (analytic)")
    ax.loglog(rec_t, rec_s, "o", color="k", mfc="white", ms=3.5, label=f"Monte Carlo ({M} runs)")
    ax.set_xlabel("dead-reckoning time $t$ [s]")
    ax.set_ylabel(r"$1\sigma$ horizontal position [m]")
    ax.legend(frameon=False, fontsize=7.3, loc="upper left")
    ax.set_ylim(1e-5, 1e3)
    save(fig, "fig_cov_growth.pdf")


# ----------------------------------------------------------------------------
# Fig 9: NEES / NIS consistency example
# ----------------------------------------------------------------------------
def fig_nees():
    rng = np.random.default_rng(13)
    dt, n, N = 0.1, 200, 50
    q_true, r = 0.5, 0.5
    F = np.array([[1, dt], [0, 1]])

    def Qd(q):
        return q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])

    H = np.array([[1.0, 0.0]])
    Lq = np.linalg.cholesky(Qd(q_true))
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.8))
    cases = [(q_true, BLUE, "consistent ($q=q_{\\mathrm{true}}$)"),
             (q_true / 20, ORANGE, "overconfident ($q=q_{\\mathrm{true}}/20$)"),
             (q_true * 20, GREEN, "underconfident ($q=20\\,q_{\\mathrm{true}}$)")]
    for q_f, c, lab in cases:
        nees = np.zeros((N, n))
        nis = np.zeros((N, n))
        for j in range(N):
            x = np.array([0.0, 1.0])
            xh = x + rng.normal(size=2)
            P = np.eye(2)
            for k in range(n):
                x = F @ x + Lq @ rng.normal(size=2)
                xh = F @ xh
                P = F @ P @ F.T + Qd(q_f)
                z = x[0] + np.sqrt(r) * rng.normal()
                S = (H @ P @ H.T).item() + r
                nu = z - xh[0]
                nis[j, k] = nu**2 / S
                K = (P @ H.T / S).ravel()
                xh = xh + K * nu
                IKH = np.eye(2) - np.outer(K, H)
                P = IKH @ P @ IKH.T + r * np.outer(K, K)
                e = x - xh
                nees[j, k] = e @ np.linalg.solve(P, e)
        tt = np.arange(1, n + 1) * dt
        axs[0].plot(tt, nees.mean(0), color=c, lw=1.0, label=lab)
        axs[1].plot(tt, nis.mean(0), color=c, lw=1.0)
    for ax, dof, title in [(axs[0], 2, "(a) average NEES, $n=2$"), (axs[1], 1, "(b) average NIS, $m=1$")]:
        lo, hi = chi2.ppf(0.025, dof * N) / N, chi2.ppf(0.975, dof * N) / N
        ax.axhspan(lo, hi, color=GREY, alpha=0.18, lw=0)
        ax.axhline(dof, color=GREY, lw=0.8, ls="--")
        ax.set_yscale("log")
        ax.set_xlabel("time [s]")
        ax.set_title(title)
    h, lab = axs[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(h, lab, frameon=False, fontsize=7.5, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    save(fig, "fig_nees.pdf")


# ----------------------------------------------------------------------------
# Fig 10: EuRoC cam0 radial-tangential distortion
# ----------------------------------------------------------------------------
def fig_distortion():
    fu, fv, cu, cv = 458.654, 457.296, 367.215, 248.375
    k1, k2, p1, p2 = -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05
    W, Hh = 752, 480

    def distort(x, y):
        r2 = x * x + y * y
        rad = 1 + k1 * r2 + k2 * r2 * r2
        xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        return xd, yd

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    xs = np.linspace(-1.05, 1.05, 15)
    ys = np.linspace(-0.68, 0.68, 11)
    s = np.linspace(-1.2, 1.2, 200)
    for i, x0 in enumerate(xs):
        y = np.linspace(-0.68, 0.68, 200)
        ax.plot(fu * x0 + cu + 0 * y, fv * y + cv, color=GREY, lw=0.5, alpha=0.6,
                label="ideal pinhole" if i == 0 else None)
        xd, yd = distort(x0 + 0 * y, y)
        ax.plot(fu * xd + cu, fv * yd + cv, color=BLUE, lw=0.9, label="with distortion" if i == 0 else None)
    for y0 in ys:
        x = np.linspace(-1.05, 1.05, 200)
        ax.plot(fu * x + cu, fv * y0 + cv + 0 * x, color=GREY, lw=0.5, alpha=0.6)
        xd, yd = distort(x, y0 + 0 * x)
        ax.plot(fu * xd + cu, fv * yd + cv, color=BLUE, lw=0.9)
    ax.add_patch(plt.Rectangle((0, 0), W, Hh, fill=False, color=ORANGE, lw=1.6))
    ax.text(5, -14, r"sensor $752\times480$", color=ORANGE, fontsize=8)
    ax.set_xlim(-140, W + 140)
    ax.set_ylim(Hh + 90, -90)
    ax.set_aspect("equal")
    ax.set_xlabel("$u$ [px]")
    ax.set_ylabel("$v$ [px]")
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    save(fig, "fig_distortion.pdf")
    _ = s


# ----------------------------------------------------------------------------
# Fig 11: triangulation depth uncertainty vs parallax
# ----------------------------------------------------------------------------
def fig_triangulation():
    rng = np.random.default_rng(17)
    f = 458.0
    sig_uv = 1.0 / f
    z = 5.0
    baselines = np.logspace(-2.3, 0.6, 18)
    mc = []
    for b in baselines:
        pf = np.array([0.2, -0.1, z])
        c1, c2 = np.zeros(3), np.array([b, 0.0, 0.0])
        est = []
        for _ in range(3000):
            m1 = (pf - c1)[:2] / (pf - c1)[2] + sig_uv * rng.normal(size=2)
            m2 = (pf - c2)[:2] / (pf - c2)[2] + sig_uv * rng.normal(size=2)
            # linear triangulation (DLT on normalized coords, identity rotations)
            A = np.array([
                [1, 0, -m1[0], -0.0],
                [0, 1, -m1[1], -0.0],
                [1, 0, -m2[0], -b],
                [0, 1, -m2[1], 0.0],
            ])
            # unknown X=[x,y,z,1]: rows encode x - u z - c = 0
            A[2, 3] = -c2[0]
            _, _, Vt = np.linalg.svd(A)
            X = Vt[-1]
            X = X[:3] / X[3]
            est.append(X[2])
        est = np.array(est)
        # robust spread (1.4826 * MAD): at low parallax the error is heavy-tailed and the RMS is meaningless
        mc.append(1.4826 * np.median(np.abs(est - np.median(est))))
    mc = np.array(mc)
    parallax = np.degrees(2 * np.arctan(baselines / (2 * z)))
    analytic = z**2 * np.sqrt(2) * sig_uv / baselines
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    ax.loglog(parallax, analytic / z * 100, color=ORANGE, label=r"first order: $\sigma_z \approx \sqrt{2}\,z^2\sigma_{uv}/b$")
    ax.loglog(parallax, mc / z * 100, "o", color=BLUE, ms=3.5, label="Monte Carlo, linear triangulation (robust std)")
    ax.axhline(10, color=GREY, ls=":", lw=0.8)
    ax.text(parallax[0] * 1.05, 11.5, "10 % depth error", color=GREY, fontsize=7.5)
    ax.set_xlabel("parallax angle [deg]  ($z=5$ m, $\\sigma_{px}=1$ px, $f=458$ px)")
    ax.set_ylabel(r"depth error $\sigma_z/z$ [%]")
    ax.legend(frameon=False, fontsize=7.8)
    save(fig, "fig_triangulation.pdf")


# ----------------------------------------------------------------------------
# Fig 12: structure tensor eigenvalue plane
# ----------------------------------------------------------------------------
def fig_structure_tensor():
    l1, l2 = np.meshgrid(np.linspace(0, 10, 400), np.linspace(0, 10, 400))
    k = 0.05
    Rh = l1 * l2 - k * (l1 + l2) ** 2
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.9))
    for ax, field, title, thr in [
        (axs[0], Rh, r"(a) Harris: $\det M - k\,(\mathrm{tr}\,M)^2$", 2.0),
        (axs[1], np.minimum(l1, l2), r"(b) Shi–Tomasi: $\min(\lambda_1,\lambda_2)$", 2.0),
    ]:
        cs = ax.contourf(l1, l2, field, levels=20, cmap="Blues")
        ax.contour(l1, l2, field, levels=[thr], colors=[ORANGE], linewidths=1.6)
        ax.set_xlabel(r"$\lambda_1$")
        ax.set_ylabel(r"$\lambda_2$")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.text(0.4, 0.4, "flat", fontsize=8.5, color="k")
        ax.text(6.5, 0.4, "edge", fontsize=8.5, color="k")
        ax.text(0.3, 8.8, "edge", fontsize=8.5, color="k")
        ax.text(6.5, 7.5, "corner", fontsize=9, color="white")
        ax.grid(False)
        _ = cs
    fig.tight_layout()
    save(fig, "fig_structure_tensor.pdf")


if __name__ == "__main__":
    fig_dr_drift()
    fig_taylor()
    fig_banana()
    fig_chi2()
    fig_kf1d()
    fig_allan()
    fig_integration_order()
    fig_cov_growth()
    fig_nees()
    fig_distortion()
    fig_triangulation()
    fig_structure_tensor()
