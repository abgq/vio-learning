"""Figures for the foundations, filtering-example and computer-vision chapters.

Called by make_figures.py; can also be run alone, optionally with figure names:
    .venv\\Scripts\\python.exe scripts\\figures_extra.py [fig_ut fig_lk ...]

All randomness uses fixed seeds, so the figures are reproducible.
"""
from __future__ import annotations

import sys
from math import factorial

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from scipy import ndimage
from scipy.stats import norm

from make_figures import BLUE, GREEN, GREY, ORANGE, PURPLE, SKY, YELLOW, Exp, save, skew

# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------


def cov_ellipse(ax, mu, P, k=1.0, **kw):
    """Draw the k-sigma ellipse of a 2x2 covariance."""
    w, V = np.linalg.eigh(P)
    ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
    ax.add_patch(Ellipse(mu, 2 * k * np.sqrt(w[1]), 2 * k * np.sqrt(w[0]), angle=ang, fill=False, **kw))


def Log(R):
    """Rotation vector of a rotation matrix (adequate away from pi)."""
    c = np.clip(0.5 * (np.trace(R) - 1.0), -1.0, 1.0)
    th = np.arccos(c)
    w = 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th < 1e-8:
        return w
    return th / np.sin(th) * w


def qmul(p, q):
    pw, pv = p[0], p[1:]
    qw, qv = q[0], q[1:]
    return np.concatenate([[pw * qw - pv @ qv], pw * qv + qw * pv + np.cross(pv, qv)])


def qexp(phi):
    th = np.linalg.norm(phi)
    if th < 1e-12:
        return np.concatenate([[1.0], 0.5 * phi])
    return np.concatenate([[np.cos(th / 2)], np.sin(th / 2) * phi / th])


def qangle(p, q):
    """Angle of the relative rotation between two unit quaternions."""
    d = qmul(np.concatenate([[p[0]], -p[1:]]), q)
    return 2.0 * np.arctan2(np.linalg.norm(d[1:]), abs(d[0]))


def normalize(img):
    img = img - img.min()
    return img / img.max()


def show(ax, img, title=None, cmap="gray", **kw):
    ax.imshow(img, cmap=cmap, interpolation="nearest", **kw)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.5)
        sp.set_color(GREY)
    if title:
        ax.set_title(title, fontsize=8.5)


# ----------------------------------------------------------------------------
# Chapter 3: SVD geometry
# ----------------------------------------------------------------------------
def fig_svd_geometry():
    A = np.array([[1.5, 0.8], [0.4, 1.0]])
    U, S, Vt = np.linalg.svd(A)
    V = Vt.T
    for i in range(2):  # orient the vectors into the upper half plane for readability
        if V[1, i] < 0 or (abs(V[1, i]) < 1e-12 and V[0, i] < 0):
            V[:, i] *= -1
            U[:, i] *= -1
    th = np.linspace(0, 2 * np.pi, 400)
    circ = np.vstack([np.cos(th), np.sin(th)])
    ell = A @ circ
    fig, axs = plt.subplots(1, 2, figsize=(6.2, 3.0))
    lim = 2.0
    for ax in axs:
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.axhline(0, color=GREY, lw=0.5)
        ax.axvline(0, color=GREY, lw=0.5)
    ax = axs[0]
    ax.fill(circ[0], circ[1], color=SKY, alpha=0.15, lw=0)
    ax.plot(circ[0], circ[1], color=GREY, lw=1.1)
    for i, c in enumerate([BLUE, ORANGE]):
        ax.annotate("", xy=V[:, i], xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=c, lw=1.8))
        ax.text(*(V[:, i] * 1.25), f"$v_{i + 1}$", color=c, ha="center", va="center", fontsize=11)
    ax.set_title("(a) unit circle and right singular vectors")
    ax = axs[1]
    ax.fill(ell[0], ell[1], color=SKY, alpha=0.15, lw=0)
    ax.plot(ell[0], ell[1], color=GREY, lw=1.1)
    for i, c in enumerate([BLUE, ORANGE]):
        tip = S[i] * U[:, i]
        ax.annotate("", xy=tip, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=c, lw=1.8))
        off = 0.3 * U[:, i] / np.linalg.norm(U[:, i])
        ax.text(*(tip + off), rf"$\sigma_{i + 1}u_{i + 1}$", color=c, ha="center", va="center", fontsize=11)
    ax.set_title(r"(b) its image under $A$")
    fig.tight_layout()
    save(fig, "fig_svd_geometry.pdf")


# ----------------------------------------------------------------------------
# Chapter 4: Taylor series, ODE integrators, finite differences
# ----------------------------------------------------------------------------
def _sin_taylor(x, n):
    return sum((-1) ** k * x ** (2 * k + 1) / factorial(2 * k + 1) for k in range((n + 1) // 2))


def _sin_taylor_error(x, n, terms=40):
    """sin x - T_n(x), computed from the tail of the series (accurate for tiny x)."""
    k0 = (n + 1) // 2
    return sum((-1) ** k * x ** (2 * k + 1) / factorial(2 * k + 1) for k in range(k0, k0 + terms))


def fig_taylor_sin():
    x = np.linspace(-2 * np.pi, 2 * np.pi, 800)
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.8))
    ax = axs[0]
    ax.plot(x, np.sin(x), color="k", lw=1.8, label=r"$\sin x$")
    cols = [(1, BLUE), (3, ORANGE), (5, GREEN)]
    for n, c in cols:
        ax.plot(x, _sin_taylor(x, n), color=c, lw=1.2, label=f"degree {n}")
    ax.set_ylim(-2.2, 2.2)
    ax.set_xlabel("$x$")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left", ncol=2)
    ax.set_title("(a) Taylor polynomials about 0")
    ax = axs[1]
    xs = np.logspace(-2, np.log10(4.0), 300)
    for n, c in cols:
        ax.loglog(xs, np.abs(_sin_taylor_error(xs, n)), color=c, lw=1.3, label=f"degree {n}: slope {n + 2}")
    ax.set_xlabel("$|x|$")
    ax.set_ylabel("absolute error")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax.set_title("(b) truncation error")
    fig.tight_layout()
    save(fig, "fig_taylor_sin.pdf")


def _osc(y):
    return np.array([y[1], -y[0]])


def _ode_step(method, y, h):
    if method == "Euler":
        return y + h * _osc(y)
    if method == "Heun":
        k1 = _osc(y)
        k2 = _osc(y + h * k1)
        return y + 0.5 * h * (k1 + k2)
    k1 = _osc(y)
    k2 = _osc(y + 0.5 * h * k1)
    k3 = _osc(y + 0.5 * h * k2)
    k4 = _osc(y + h * k3)
    return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def fig_ode_methods():
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.9))
    ax = axs[0]
    Ns = 2 ** np.arange(3, 13)
    for method, c in [("Euler", ORANGE), ("Heun", GREEN), ("RK4", BLUE)]:
        errs, hs = [], []
        for N in Ns:
            h = 2 * np.pi / N
            y = np.array([1.0, 0.0])
            for _ in range(N):
                y = _ode_step(method, y, h)
            errs.append(np.linalg.norm(y - np.array([1.0, 0.0])))
            hs.append(h)
        ax.loglog(hs, errs, "o-", color=c, ms=3, lw=1.2, label=method)
    hs = np.array(hs)
    for p, y0 in [(1, 3.0), (2, 1.2), (4, 0.08)]:
        ax.loglog(hs, y0 * (hs / hs[0]) ** p, color=GREY, ls=":", lw=0.9)
    ax.text(0.03, 0.05, "slope 1", fontsize=7.5, color=GREY)
    ax.text(0.03, 1.2e-4, "slope 2", fontsize=7.5, color=GREY)
    ax.text(0.06, 2e-9, "slope 4", fontsize=7.5, color=GREY)
    ax.set_xlabel("step size $h$")
    ax.set_ylabel("error after one period")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("(a) convergence order")
    ax = axs[1]
    th = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(th), -np.sin(th), color="k", lw=1.0, label="exact")
    for method, c in [("Euler", ORANGE), ("RK4", BLUE)]:
        y = np.array([1.0, 0.0])
        pts = [y]
        for _ in range(int(4 * np.pi / 0.1)):
            y = _ode_step(method, y, 0.1)
            pts.append(y)
        pts = np.array(pts)
        ax.plot(pts[:, 0], pts[:, 1], color=c, lw=1.1, ls="--" if method == "RK4" else "-", label=f"{method}, $h = 0.1$")
    ax.set_aspect("equal")
    ax.set_xlabel("$y$")
    ax.set_ylabel(r"$\dot y$")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.set_title("(b) two periods in the phase plane")
    fig.tight_layout()
    save(fig, "fig_ode_methods.pdf")


def fig_fd_error():
    x = 1.0
    h = np.logspace(-16, -0.3, 600)
    fwd = np.abs((np.sin(x + h) - np.sin(x)) / h - np.cos(x))
    cen = np.abs((np.sin(x + h) - np.sin(x - h)) / (2 * h) - np.cos(x))
    fwd[fwd == 0] = np.nan
    cen[cen == 0] = np.nan
    u = 2.0**-53
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.loglog(h, fwd, color=ORANGE, lw=0.9, label="forward difference")
    ax.loglog(h, cen, color=BLUE, lw=0.9, label="central difference")
    ax.loglog(h, 0.5 * np.sin(x) * h, color=ORANGE, ls=":", lw=1.0)
    ax.loglog(h, np.cos(x) / 6 * h**2, color=BLUE, ls=":", lw=1.0)
    ax.loglog(h, u / h, color=GREY, ls="--", lw=1.0)
    ax.text(3e-15, 0.5, r"rounding $\propto u/h$", color=GREY, fontsize=8)
    ax.text(2e-3, 2.5e-3, "slope 1", color=ORANGE, fontsize=8)
    ax.text(2e-3, 2e-8, "slope 2", color=BLUE, fontsize=8)
    ax.set_ylim(1e-13, 10)
    ax.set_xlabel("step $h$")
    ax.set_ylabel(r"error in $f'(1)$, $f = \sin$")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    save(fig, "fig_fd_error.pdf")


# ----------------------------------------------------------------------------
# Chapter 5: Gaussians and the central limit theorem
# ----------------------------------------------------------------------------
def fig_gauss2d():
    rng = np.random.default_rng(3)
    Sigma = np.array([[1.0, 0.7], [0.7, 0.8]])
    L = np.linalg.cholesky(Sigma)
    X = (L @ rng.standard_normal((2, 3000))).T
    Si = np.linalg.inv(Sigma)
    Xb = (L @ rng.standard_normal((2, 200000))).T
    d2 = np.einsum("ij,jk,ik->i", Xb, Si, Xb)
    fig = plt.figure(figsize=(4.9, 4.5))
    gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4), wspace=0.05, hspace=0.05)
    ax = fig.add_subplot(gs[1, 0])
    axt = fig.add_subplot(gs[0, 0], sharex=ax)
    axr = fig.add_subplot(gs[1, 1], sharey=ax)
    ax.scatter(X[:, 0], X[:, 1], s=2, color=GREY, alpha=0.35, lw=0)
    w, V = np.linalg.eigh(Sigma)
    for k, c in [(1, BLUE), (2, GREEN), (3, ORANGE)]:
        cov_ellipse(ax, (0, 0), Sigma, k, color=c, lw=1.5)
        frac = np.mean(d2 <= k * k)
        ax.plot([], [], color=c, lw=1.5, label=rf"${k}\sigma$ ellipse: {100 * frac:.0f}% of samples")
    ax.set_xlim(-3.8, 3.8)
    ax.set_ylim(-3.6, 3.6)
    ax.legend(frameon=False, fontsize=7.2, loc="upper left")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    xs = np.linspace(-3.8, 3.8, 300)
    axt.hist(X[:, 0], bins=40, density=True, color=SKY, alpha=0.5)
    axt.plot(xs, norm.pdf(xs, 0, np.sqrt(Sigma[0, 0])), color=BLUE, lw=1.3)
    axr.hist(X[:, 1], bins=40, density=True, color=SKY, alpha=0.5, orientation="horizontal")
    axr.plot(norm.pdf(xs, 0, np.sqrt(Sigma[1, 1])), xs, color=BLUE, lw=1.3)
    for a in (axt, axr):
        a.tick_params(labelbottom=False, labelleft=False)
        a.grid(False)
    axt.set_yticks([])
    axr.set_xticks([])
    axt.set_title(r"samples, Mahalanobis ellipses and marginals", fontsize=9)
    save(fig, "fig_gauss2d.pdf")


def fig_clt():
    rng = np.random.default_rng(7)
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.8))
    ax = axs[0]
    xs = np.linspace(-3.6, 3.6, 300)
    bins = np.linspace(-3.6, 3.6, 49)
    for n, c in [(1, ORANGE), (2, GREEN), (4, BLUE)]:
        s = rng.uniform(0, 1, (200000, n)).sum(axis=1)
        z = (s - n / 2) / np.sqrt(n / 12)
        ax.hist(z, bins=bins, density=True, histtype="step", color=c, lw=1.3, label=f"$n = {n}$")
    ax.plot(xs, norm.pdf(xs), color="k", ls="--", lw=1.1, label=r"$\mathcal{N}(0, 1)$")
    ax.set_xlabel("standardized sum")
    ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.set_title("(a) sums of $n$ uniform variables")
    ax = axs[1]
    N = np.arange(1, 10001)
    sd = 1 / np.sqrt(12)
    ax.fill_between(N, -1.96 * sd / np.sqrt(N), 1.96 * sd / np.sqrt(N), color=GREY, alpha=0.2, lw=0, label=r"$\pm1.96\,\sigma/\sqrt{N}$")
    for r, c in zip(range(4), [BLUE, ORANGE, GREEN, PURPLE]):
        x = rng.uniform(0, 1, N[-1])
        ax.plot(N, np.cumsum(x) / N - 0.5, color=c, lw=0.8)
    ax.set_xscale("log")
    ax.set_ylim(-0.4, 0.4)
    ax.set_xlabel("number of samples $N$")
    ax.set_ylabel("error of the sample mean")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.set_title("(b) Monte Carlo estimate of a mean")
    fig.tight_layout()
    save(fig, "fig_clt.pdf")


# ----------------------------------------------------------------------------
# Chapter 6: random processes and aliasing
# ----------------------------------------------------------------------------
def fig_processes():
    rng = np.random.default_rng(12)
    dt, T, tau = 0.05, 20.0, 2.0
    n = int(T / dt)
    t = np.arange(n) * dt
    phi = np.exp(-dt / tau)
    w = rng.standard_normal(n)
    gm = np.zeros(n)
    for k in range(1, n):
        gm[k] = phi * gm[k - 1] + np.sqrt(1 - phi**2) * w[k]
    rw = np.concatenate([[0.0], np.cumsum(np.sqrt(dt) * w[1:])])
    fig, axs = plt.subplots(2, 3, figsize=(6.8, 3.9))
    axs[0, 0].plot(t, w, color=BLUE, lw=0.6)
    axs[0, 0].set_title("white noise")
    axs[0, 1].plot(t, gm, color=GREEN, lw=0.9)
    axs[0, 1].set_title(rf"Gauss–Markov, $\tau = {tau:.0f}$ s")
    ens = np.cumsum(np.sqrt(dt) * rng.standard_normal((30, n)), axis=1)
    for e in ens:
        axs[0, 2].plot(t, e, color=GREY, lw=0.4, alpha=0.35)
    axs[0, 2].plot(t, rw, color=ORANGE, lw=1.0)
    axs[0, 2].plot(t, np.sqrt(t), color="k", ls="--", lw=0.9)
    axs[0, 2].plot(t, -np.sqrt(t), color="k", ls="--", lw=0.9)
    axs[0, 2].set_title(r"random walk ($\pm\sqrt{t}$ dashed)")
    for a in axs[0]:
        a.set_xlabel("time [s]")
    # autocorrelations
    lags = np.arange(-60, 61)
    long = rng.standard_normal(400000)
    ac_w = np.array([np.mean(long[: len(long) - abs(L)] * long[abs(L):]) for L in lags])
    g = np.zeros_like(long)
    for k in range(1, len(long)):
        g[k] = phi * g[k - 1] + np.sqrt(1 - phi**2) * long[k]
    ac_g = np.array([np.mean(g[: len(g) - abs(L)] * g[abs(L):]) for L in lags])
    k0 = int(10.0 / dt)
    walks = np.cumsum(np.sqrt(dt) * rng.standard_normal((4000, k0 + 61)), axis=1)
    var0 = np.var(walks[:, k0])
    ac_r = np.array([np.mean(walks[:, k0] * walks[:, k0 + L]) / var0 for L in lags])
    tl = lags * dt
    axs[1, 0].plot(tl, ac_w, color=BLUE, lw=1.1)
    axs[1, 1].plot(tl, ac_g, color=GREEN, lw=1.1)
    axs[1, 1].plot(tl, np.exp(-np.abs(tl) / tau), color="k", ls="--", lw=0.9)
    axs[1, 2].plot(tl, ac_r, color=ORANGE, lw=1.1)
    axs[1, 2].plot(tl, np.minimum(1.0, (10.0 + tl) / 10.0), color="k", ls="--", lw=0.9)
    axs[1, 0].set_ylabel("autocorrelation")
    axs[1, 2].set_title(r"corr. with $x(10\,\mathrm{s})$", fontsize=9)
    for a in axs[1]:
        a.set_xlabel("lag [s]")
        a.set_ylim(-0.2, 1.1)
    fig.tight_layout()
    save(fig, "fig_processes.pdf")


def fig_aliasing():
    t = np.linspace(0, 1, 3000)
    ts = np.arange(0, 1.0001, 0.1)
    fig, ax = plt.subplots(figsize=(5.6, 2.2))
    ax.plot(t, np.cos(2 * np.pi * 9 * t), color=BLUE, lw=0.8, label="9 Hz")
    ax.plot(t, np.cos(2 * np.pi * 1 * t), color=ORANGE, ls="--", lw=1.4, label="1 Hz")
    ax.plot(ts, np.cos(2 * np.pi * 9 * ts), "o", color="k", ms=5, label="samples at 10 Hz")
    ax.set_xlabel("time [s]")
    ax.set_ylim(-1.25, 1.6)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center")
    save(fig, "fig_aliasing.pdf")


# ----------------------------------------------------------------------------
# Chapter 7: SLERP vs NLERP
# ----------------------------------------------------------------------------
def fig_slerp():
    axis = np.array([1.0, 2.0, 2.0]) / 3.0
    ang = np.radians(150.0)
    q0 = np.array([1.0, 0, 0, 0])
    q1 = np.concatenate([[np.cos(ang / 2)], np.sin(ang / 2) * axis])
    tau = np.linspace(0, 1, 401)
    slerp = np.degrees(ang * tau)
    nl = []
    for s in tau:
        q = (1 - s) * q0 + s * q1
        q /= np.linalg.norm(q)
        nl.append(np.degrees(qangle(q0, q)))
    nl = np.array(nl)
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.6))
    axs[0].plot(tau, slerp, color=BLUE, label="SLERP")
    axs[0].plot(tau, nl, color=ORANGE, ls="--", label="NLERP")
    axs[0].set_xlabel(r"interpolation parameter $\tau$")
    axs[0].set_ylabel("angle from start [deg]")
    axs[0].legend(frameon=False, fontsize=8)
    axs[0].set_title("(a) angle swept")
    axs[1].plot(tau, np.gradient(slerp, tau), color=BLUE, label="SLERP")
    axs[1].plot(tau, np.gradient(nl, tau), color=ORANGE, ls="--", label="NLERP")
    axs[1].set_xlabel(r"interpolation parameter $\tau$")
    axs[1].set_ylabel(r"angular rate [deg per unit $\tau$]")
    axs[1].set_ylim(0, None)
    axs[1].set_title("(b) speed along the arc")
    fig.tight_layout()
    save(fig, "fig_slerp.pdf")


# ----------------------------------------------------------------------------
# Chapter 9: least squares and recursive least squares
# ----------------------------------------------------------------------------
def fig_lsfit():
    rng = np.random.default_rng(11)
    p0, v = 1.0, 0.8
    t = np.arange(10.0)
    sig = np.full(10, 0.5)
    sig[[2, 6]] = 1.5
    z = p0 + v * t + rng.normal(0, sig)
    H = np.column_stack([np.ones_like(t), t])
    W = np.diag(1 / sig**2)
    Pxx = np.linalg.inv(H.T @ W @ H)
    xh = Pxx @ H.T @ W @ z
    tt = np.linspace(-0.5, 9.5, 200)
    Ht = np.column_stack([np.ones_like(tt), tt])
    sd = np.sqrt(np.einsum("ij,jk,ik->i", Ht, Pxx, Ht))
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.9))
    ax = axs[0]
    ax.fill_between(tt, Ht @ xh - 2 * sd, Ht @ xh + 2 * sd, color=BLUE, alpha=0.15, lw=0, label=r"fit $\pm2\sigma$")
    ax.plot(tt, Ht @ xh, color=BLUE, lw=1.4, label="weighted LS fit")
    ax.plot(tt, p0 + v * tt, color="k", ls=":", lw=1.0, label="truth")
    ax.errorbar(t, z, yerr=sig, fmt="o", color=ORANGE, ms=3.5, lw=0.9, capsize=2, label=r"measurements $\pm\sigma$")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("position [m]")
    ax.legend(frameon=False, fontsize=7.2, loc="upper left")
    ax.set_title("(a) weighted least squares")
    # recursive least squares
    ax = axs[1]
    m_max, s = 40, 0.5
    tm = np.arange(m_max, dtype=float)
    zm = p0 + v * tm + rng.normal(0, s, m_max)
    x = np.zeros(2)
    P = np.diag([1e4, 1e4])
    vh, vs = [], []
    for k in range(m_max):
        h = np.array([1.0, tm[k]])
        S = h @ P @ h + s**2
        K = P @ h / S
        x = x + K * (zm[k] - h @ x)
        P = P - np.outer(K, h @ P)
        vh.append(x[1])
        vs.append(np.sqrt(P[1, 1]))
    vh, vs = np.array(vh), np.array(vs)
    m = np.arange(1, m_max + 1)
    sel = m >= 3
    ax.fill_between(m[sel], vh[sel] - 2 * vs[sel], vh[sel] + 2 * vs[sel], color=BLUE, alpha=0.15, lw=0, label=r"$\pm2\sigma$")
    ax.plot(m[sel], vh[sel], "o-", color=BLUE, ms=2.5, lw=1.0, label="estimate")
    ax.axhline(v, color="k", ls=":", lw=1.0, label="truth")
    ax.set_xlabel("number of measurements $m$")
    ax.set_ylabel("velocity [m/s]")
    ax.legend(frameon=False, fontsize=7.2, loc="upper right")
    ax.set_title("(b) recursive least squares")
    ins = ax.inset_axes([0.42, 0.08, 0.32, 0.36])
    ins.loglog(m[sel], vs[sel], color=BLUE, lw=1.0)
    ins.loglog(m[sel], vs[sel][0] * (m[sel] / m[sel][0]) ** -1.5, color=GREY, ls=":", lw=0.9)
    ins.set_title(r"$\sigma_v \propto m^{-3/2}$", fontsize=7)
    ins.set_xticks([3, 10, 30])
    ins.set_xticklabels(["3", "10", "30"])
    ins.xaxis.set_minor_formatter(plt.NullFormatter())
    ins.tick_params(labelsize=6)
    ins.grid(False)
    fig.tight_layout()
    save(fig, "fig_lsfit.pdf")


# ----------------------------------------------------------------------------
# Chapter 10: Gauss-Newton vs LM vs gradient descent; robust kernels; BA sparsity
# ----------------------------------------------------------------------------
def fig_gn_lm():
    rng = np.random.default_rng(2)
    t = np.linspace(0, 1.5, 25)
    k_true, A_true = 1.3, 2.0
    y = A_true * np.exp(-k_true * t) + rng.normal(0, 0.04, t.size)

    def resid(th):
        return th[1] * np.exp(-th[0] * t) - y

    def jac(th):
        e = np.exp(-th[0] * t)
        return np.column_stack([-th[1] * t * e, e])

    def cost(th):
        r = resid(th)
        return 0.5 * r @ r

    start = np.array([0.2, 0.8])
    # Gradient descent with a fixed, stable step
    th_opt = np.array([k_true, A_true])
    for _ in range(50):
        J, r = jac(th_opt), resid(th_opt)
        th_opt = th_opt - np.linalg.solve(J.T @ J, J.T @ r)
    lmax = np.linalg.eigvalsh(jac(th_opt).T @ jac(th_opt)).max()
    paths = {}
    th = start.copy()
    p = [th.copy()]
    for _ in range(400):
        J, r = jac(th), resid(th)
        th = th - (0.9 / lmax) * (J.T @ r)
        p.append(th.copy())
    paths["gradient descent"] = np.array(p)
    th = start.copy()
    p = [th.copy()]
    for _ in range(60):
        J, r = jac(th), resid(th)
        th = th - np.linalg.solve(J.T @ J, J.T @ r)
        p.append(th.copy())
    paths["Gauss–Newton"] = np.array(p)
    th = start.copy()
    p = [th.copy()]
    lam = 1.0
    for _ in range(60):
        J, r = jac(th), resid(th)
        Hm = J.T @ J
        step = -np.linalg.solve(Hm + lam * np.diag(np.diag(Hm)), J.T @ r)
        if cost(th + step) < cost(th):
            th = th + step
            lam /= 10
        else:
            lam *= 10
        p.append(th.copy())
    paths["Levenberg–Marquardt"] = np.array(p)
    K, A = np.meshgrid(np.linspace(0.0, 2.6, 300), np.linspace(0.0, 3.0, 300))
    C = np.zeros_like(K)
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            C[i, j] = cost(np.array([K[i, j], A[i, j]]))
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.0))
    ax = axs[0]
    ax.contour(K, A, np.log10(C), levels=18, colors=GREY, linewidths=0.5, linestyles="solid")
    cols = {"gradient descent": ORANGE, "Gauss–Newton": BLUE, "Levenberg–Marquardt": GREEN}
    for name, pth in paths.items():
        ax.plot(pth[:, 0], pth[:, 1], "o-", color=cols[name], ms=2.5, lw=1.0, label=name)
    ax.plot(*start, "ks", ms=5)
    ax.plot(*th_opt, "k*", ms=9)
    ax.set_xlim(0, 2.6)
    ax.set_ylim(0, 3.0)
    ax.set_xlabel("decay rate $k$")
    ax.set_ylabel("amplitude $A$")
    ax.legend(frameon=False, fontsize=7.2, loc="upper right")
    ax.set_title("(a) paths on the cost contours")
    ax = axs[1]
    for name, pth in paths.items():
        c = np.array([cost(th) for th in pth])
        ax.semilogy(np.arange(len(c)), c, "o-", color=cols[name], ms=2, lw=1.0, label=name)
    ax.axhline(cost(th_opt), color="k", ls=":", lw=0.9)
    ax.set_xlabel("iteration")
    ax.set_ylabel("cost")
    ax.set_xlim(0, 40)
    ax.set_title("(b) cost per iteration")
    fig.tight_layout()
    save(fig, "fig_gn_lm.pdf")


def fig_robust_kernels():
    s = np.linspace(0, 6, 600)
    ch, cc, ct = 1.345, 2.385, 4.685
    rho = {
        "least squares": 0.5 * s**2,
        "Huber": np.where(s <= ch, 0.5 * s**2, ch * (s - ch / 2)),
        "Cauchy": 0.5 * cc**2 * np.log1p(s**2 / cc**2),
        "Tukey": np.where(s <= ct, ct**2 / 6 * (1 - (1 - s**2 / ct**2) ** 3), ct**2 / 6),
    }
    wgt = {
        "least squares": np.ones_like(s),
        "Huber": np.where(s <= ch, 1.0, ch / np.maximum(s, 1e-12)),
        "Cauchy": 1 / (1 + s**2 / cc**2),
        "Tukey": np.where(s <= ct, (1 - s**2 / ct**2) ** 2, 0.0),
    }
    cols = {"least squares": GREY, "Huber": BLUE, "Cauchy": GREEN, "Tukey": ORANGE}
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.7))
    for name in rho:
        axs[0].plot(s, rho[name], color=cols[name], lw=1.4, label=name)
        axs[1].plot(s, wgt[name], color=cols[name], lw=1.4, label=name)
    axs[0].set_ylim(0, 6)
    axs[0].set_xlabel("normalized residual $s$")
    axs[0].set_ylabel(r"$\rho(s)$")
    axs[0].legend(frameon=False, fontsize=7.5)
    axs[0].set_title("(a) kernels")
    axs[1].set_xlabel("normalized residual $s$")
    axs[1].set_ylabel(r"weight $w(s)$")
    axs[1].set_ylim(-0.03, 1.08)
    axs[1].set_title("(b) weights")
    fig.tight_layout()
    save(fig, "fig_robust_kernels.pdf")


def fig_ba_sparsity():
    rng = np.random.default_rng(5)
    nc, nl = 6, 40
    obs = []
    for j in range(nl):
        start = int(rng.integers(0, nc - 1))
        length = int(rng.integers(2, 5))
        for c in range(start, min(nc, start + length)):
            obs.append((c, j))
    n = 6 * nc + 3 * nl
    M = np.zeros((n, n), bool)
    for c, j in obs:
        ci = slice(6 * c, 6 * c + 6)
        lj = slice(6 * nc + 3 * j, 6 * nc + 3 * j + 3)
        M[ci, ci] = M[lj, lj] = M[ci, lj] = M[lj, ci] = True
    R = np.zeros((6 * nc, 6 * nc), bool)
    for j in range(nl):
        cams = [c for c, jj in obs if jj == j]
        for a in cams:
            for b in cams:
                R[6 * a:6 * a + 6, 6 * b:6 * b + 6] = True
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.3), gridspec_kw={"width_ratios": [1.5, 1.0]})
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["white", BLUE])
    axs[0].imshow(M, cmap=cmap, interpolation="nearest")
    axs[0].axhline(6 * nc - 0.5, color=ORANGE, lw=0.8)
    axs[0].axvline(6 * nc - 0.5, color=ORANGE, lw=0.8)
    axs[0].set_title(r"(a) $J^{\mathsf{T}}WJ$: 6 cameras, 40 landmarks", fontsize=9)
    axs[1].imshow(R, cmap=cmap, interpolation="nearest")
    axs[1].set_title("(b) reduced camera system", fontsize=9)
    for a in axs:
        a.set_xticks([])
        a.set_yticks([])
        a.grid(False)
    axs[0].set_xticks([6 * nc / 2 - 0.5, 6 * nc + 3 * nl / 2 - 0.5])
    axs[0].set_xticklabels(["cameras", "landmarks"], fontsize=8)
    axs[0].tick_params(length=0)
    axs[1].set_xticks([6 * nc / 2 - 0.5])
    axs[1].set_xticklabels(["cameras"], fontsize=8)
    axs[1].tick_params(length=0)
    fig.tight_layout()
    save(fig, "fig_ba_sparsity.pdf")


# ----------------------------------------------------------------------------
# Chapter 13: unscented transform
# ----------------------------------------------------------------------------
def fig_ut():
    rng = np.random.default_rng(9)
    mu = np.array([1.0, np.pi / 2])
    P = np.diag([0.02**2, np.radians(15.0) ** 2])

    def g(x):
        return np.array([x[0] * np.cos(x[1]), x[0] * np.sin(x[1])])

    smp = rng.multivariate_normal(mu, P, 40000)
    Y = np.column_stack([smp[:, 0] * np.cos(smp[:, 1]), smp[:, 0] * np.sin(smp[:, 1])])
    m_true, P_true = Y.mean(axis=0), np.cov(Y.T)
    J = np.array([[np.cos(mu[1]), -mu[0] * np.sin(mu[1])], [np.sin(mu[1]), mu[0] * np.cos(mu[1])]])
    m_lin, P_lin = g(mu), J @ P @ J.T
    n, alpha, beta, kappa = 2, 1.0, 0.0, 1.0
    lam = alpha**2 * (n + kappa) - n
    L = np.linalg.cholesky(P)
    chi = [mu] + [mu + np.sqrt(n + lam) * L[:, i] for i in range(n)] + [mu - np.sqrt(n + lam) * L[:, i] for i in range(n)]
    Wm = np.array([lam / (n + lam)] + [1 / (2 * (n + lam))] * (2 * n))
    Wc = Wm.copy()
    Wc[0] += 1 - alpha**2 + beta
    Ysp = np.array([g(c) for c in chi])
    m_ut = Wm @ Ysp
    P_ut = sum(Wc[i] * np.outer(Ysp[i] - m_ut, Ysp[i] - m_ut) for i in range(2 * n + 1))
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    ax.scatter(Y[:3000, 0], Y[:3000, 1], s=2, color=GREY, alpha=0.3, lw=0, label="samples")
    for m, Pm, c, ls, mk, lab in [(m_true, P_true, "k", "-", "o", "true (Monte Carlo)"), (m_lin, P_lin, ORANGE, "-", "o", "linearized"),
                                  (m_ut, P_ut, BLUE, "--", "x", "unscented")]:
        cov_ellipse(ax, m, Pm, 2.0, color=c, lw=1.6 if c == "k" else 1.3, ls=ls)
        ax.plot(*m, mk, color=c, ms=7, mew=1.8, label=lab, zorder=6)
    ax.plot(Ysp[:, 0], Ysp[:, 1], "s", color=SKY, ms=4, mfc="none", mew=1.0, label="sigma points", zorder=4)
    ax.set_aspect("equal")
    ax.set_xlim(-0.75, 0.75)
    ax.set_ylim(0.62, 1.1)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.legend(frameon=False, fontsize=7.2, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    save(fig, "fig_ut.pdf")
    print(f"  UT: true mean y {m_true[1]:.4f}, linearized {m_lin[1]:.4f}, unscented {m_ut[1]:.4f}")


# ----------------------------------------------------------------------------
# Chapter 14: attitude ESKF (global perturbation)
# ----------------------------------------------------------------------------
def fig_ahrs():
    rng = np.random.default_rng(21)
    dt, T = 0.01, 60.0
    n = int(round(T / dt))
    sig_g, sig_b, sig_a = 5e-3, 1e-5, 0.05
    v = np.array([0.0, 0.0, 9.81])
    b_true = np.array([0.008, -0.006, 0.005])
    R = np.eye(3)
    P = np.diag([0.05**2, 0.05**2, 0.01**2, 0.01**2, 0.01**2, 0.01**2])
    dth0 = rng.normal(0, 1, 3) * np.sqrt(np.diag(P)[:3])
    Rh = Exp(-dth0) @ R                      # R = Exp(dtheta) Rh
    bh = np.zeros(3)
    I3 = np.eye(3)
    log_t, log_e, log_s = [], [], []
    for k in range(n):
        t = k * dt
        w = np.array([0.6 * np.sin(0.31 * t), 0.5 * np.cos(0.23 * t + 0.4), 0.4 * np.sin(0.17 * t + 1.0)])
        wm = w + b_true + rng.normal(0, sig_g / np.sqrt(dt), 3)
        R = R @ Exp(w * dt)
        F = np.zeros((6, 6))
        F[0:3, 3:6] = -Rh
        Phi = np.eye(6) + F * dt
        G = np.zeros((6, 6))
        G[0:3, 0:3] = -Rh
        G[3:6, 3:6] = I3
        Q = G @ np.diag([sig_g**2] * 3 + [sig_b**2] * 3) @ G.T * dt
        P = Phi @ P @ Phi.T + Q
        Rh = Rh @ Exp((wm - bh) * dt)
        if (k + 1) % 10 == 0:
            am = R.T @ v + rng.normal(0, sig_a, 3)
            H = np.zeros((3, 6))
            H[:, 0:3] = Rh.T @ skew(v)
            r = am - Rh.T @ v
            S = H @ P @ H.T + sig_a**2 * I3
            K = np.linalg.solve(S, H @ P).T
            dx = K @ r
            IKH = np.eye(6) - K @ H
            P = IKH @ P @ IKH.T + K @ (sig_a**2 * I3) @ K.T
            Rh = Exp(dx[0:3]) @ Rh
            bh = bh + dx[3:6]
            Gr = np.eye(6)
            Gr[0:3, 0:3] = I3 + 0.5 * skew(dx[0:3])
            P = Gr @ P @ Gr.T
            P = 0.5 * (P + P.T)
        if k % 5 == 0:
            log_t.append(t + dt)
            log_e.append(np.concatenate([Log(R @ Rh.T), b_true - bh]))
            log_s.append(np.sqrt(np.diag(P)))
    t = np.array(log_t)
    e = np.degrees(np.array(log_e))
    s = np.degrees(np.array(log_s))
    fig, axs = plt.subplots(2, 3, figsize=(6.8, 3.9), sharex=True)
    names = [r"about $x_w$", r"about $y_w$", r"about $z_w$ (yaw)", "$x$", "$y$", "$z$"]
    for i in range(6):
        ax = axs[i // 3, i % 3]
        c = BLUE if i < 3 else GREEN
        ax.fill_between(t, -3 * s[:, i], 3 * s[:, i], color=c, alpha=0.15, lw=0)
        ax.plot(t, e[:, i], color=c, lw=0.8)
        ax.set_title(names[i], fontsize=8.5)
        if i >= 3:
            ax.set_xlabel("time [s]")
    axs[0, 0].set_ylabel("(a) attitude\nerror [deg]")
    axs[1, 0].set_ylabel("(b) gyro bias\nerror [deg/s]")
    for i in range(3):
        axs[0, i].set_ylim(-7, 7)
        axs[1, i].set_ylim(-0.8, 0.8)
    fig.tight_layout()
    save(fig, "fig_ahrs.pdf")


# ----------------------------------------------------------------------------
# Chapter 15-16: accelerometer frequency response, coning
# ----------------------------------------------------------------------------
def fig_accel_bode():
    r = np.logspace(-2, 1, 800)
    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    for zeta, c in [(0.1, ORANGE), (0.3, YELLOW), (0.7, BLUE), (1.0, GREEN)]:
        Hm = np.abs(1 / (1 - r**2 + 2j * zeta * r))
        ax.loglog(r, Hm, color=c, lw=1.4, label=rf"$\zeta = {zeta}$")
    ax.set_xlabel(r"frequency ratio $\omega/\omega_n$")
    ax.set_ylabel("normalized amplitude")
    ax.set_ylim(1e-2, 8)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    save(fig, "fig_accel_bode.pdf")


def fig_coning():
    a, Om = 0.5, 2 * np.pi * 10.0

    def omega(t):
        return a * np.array([np.cos(Om * t), np.sin(Om * t), 0.0])

    def alpha(t0, t1):
        return a / Om * np.array([np.sin(Om * t1) - np.sin(Om * t0), -np.cos(Om * t1) + np.cos(Om * t0), 0.0])

    def exact(h):
        nsub = max(200, int(h / 2e-6))
        dt = h / nsub
        q = np.array([1.0, 0, 0, 0])

        def f(t, q):
            return 0.5 * qmul(q, np.concatenate([[0.0], omega(t)]))

        for i in range(nsub):
            t = i * dt
            k1 = f(t, q)
            k2 = f(t + dt / 2, q + dt / 2 * k1)
            k3 = f(t + dt / 2, q + dt / 2 * k2)
            k4 = f(t + dt, q + dt * k3)
            q = q + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return q / np.linalg.norm(q)

    hs = np.logspace(np.log10(2e-3), np.log10(2.5e-2), 10)
    e1, e2 = [], []
    for h in hs:
        qe = exact(h)
        a1, a2 = alpha(0, h / 2), alpha(h / 2, h)
        e1.append(qangle(qe, qexp(a1 + a2)))
        e2.append(qangle(qe, qexp(a1 + a2 + 2.0 / 3.0 * np.cross(a1, a2))))
    e1, e2 = np.array(e1), np.array(e2)
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    ax.loglog(hs, e1, "o-", color=ORANGE, ms=3.5, lw=1.2, label=r"$\mathrm{Exp}(\alpha_1 + \alpha_2)$")
    ax.loglog(hs, e2, "s-", color=BLUE, ms=3.5, lw=1.2, label=r"two-sample coning correction")
    ax.loglog(hs, e1[-1] * (hs / hs[-1]) ** 3, color=GREY, ls=":", lw=0.9)
    ax.loglog(hs, e2[-1] * (hs / hs[-1]) ** 5, color=GREY, ls=":", lw=0.9)
    ax.text(hs[5], e1[5] * 0.12, "slope 3", color=ORANGE, fontsize=8)
    ax.text(hs[5], e2[5] * 0.06, "slope 5", color=BLUE, fontsize=8)
    ax.set_xlabel("step size [s]")
    ax.set_ylabel("attitude error per step [rad]")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, "fig_coning.pdf")
    print(f"  coning: h={hs[-1]:.4f}: {e1[-1]:.3e} -> {e2[-1]:.3e}; h={hs[0]:.4f}: {e1[0]:.3e} -> {e2[0]:.3e}")


# ----------------------------------------------------------------------------
# Chapter 21: image processing
# ----------------------------------------------------------------------------
def _shapes_image(n=128, noise=0.05, seed=1):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    img = np.full((n, n), 0.2)
    img[18:58, 14:56] = 0.85
    img[(xx - 92) ** 2 + (yy - 38) ** 2 < 19**2] = 0.6
    tri = (yy > 74) & (yy < 116) & (xx > 20) & (xx - 20 < (yy - 74) * 1.3)
    img[tri] = 0.7
    img[80:112, 76:110] = 0.45
    img[88:104, 84:102] = 0.95
    return img + rng.normal(0, noise, img.shape)


def fig_image_filters():
    img = _shapes_image()
    sm = ndimage.gaussian_filter(img, 1.5)
    Iu = ndimage.sobel(sm, axis=1) / 8
    Iv = ndimage.sobel(sm, axis=0) / 8
    mag = np.hypot(Iu, Iv)
    fig, axs = plt.subplots(1, 5, figsize=(6.9, 1.75))
    show(axs[0], img, "noisy image", vmin=0, vmax=1)
    show(axs[1], sm, r"Gaussian, $\sigma = 1.5$", vmin=0, vmax=1)
    lim = np.abs(np.concatenate([Iu.ravel(), Iv.ravel()])).max()
    show(axs[2], Iu, r"$I_u$ (Sobel)", cmap="RdBu_r", vmin=-lim, vmax=lim)
    show(axs[3], Iv, r"$I_v$ (Sobel)", cmap="RdBu_r", vmin=-lim, vmax=lim)
    show(axs[4], mag, "gradient magnitude", cmap="magma")
    fig.tight_layout(pad=0.3)
    save(fig, "fig_image_filters.pdf")


def fig_aliasing_image():
    n = 256
    yy, xx = np.mgrid[0:n, 0:n] - n / 2 + 0.5
    zp = 0.5 + 0.5 * np.cos(np.pi * (xx**2 + yy**2) / n)
    naive = zp[::4, ::4]
    blurred = ndimage.gaussian_filter(zp, 2.0)[::4, ::4]
    fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.35))
    ext = [0, n, n, 0]
    show(axs[0], zp, "zone plate, 256 px", vmin=0, vmax=1, extent=ext)
    show(axs[1], naive, "every 4th pixel, 64 px", vmin=0, vmax=1, extent=ext)
    show(axs[2], blurred, "blur, then every 4th pixel", vmin=0, vmax=1, extent=ext)
    fig.tight_layout(pad=0.3)
    save(fig, "fig_aliasing_image.pdf")


def _texture_image(n=256, seed=4):
    rng = np.random.default_rng(seed)
    coarse = ndimage.gaussian_filter(rng.standard_normal((n, n)), 12)
    mid = ndimage.gaussian_filter(rng.standard_normal((n, n)), 3)
    fine = ndimage.gaussian_filter(rng.standard_normal((n, n)), 0.8)
    img = normalize(coarse) + 0.6 * normalize(mid) + 0.45 * normalize(fine)
    yy, xx = np.mgrid[0:n, 0:n]
    img[(xx - 70) ** 2 + (yy - 80) ** 2 < 30**2] += 0.8
    img[150:210, 140:220] -= 0.6
    img += 0.35 * (np.sin(2 * np.pi * xx / 5.0) > 0) * ((yy > 30) & (yy < 70) & (xx > 140) & (xx < 230))
    return normalize(img)


def fig_pyramid():
    img = _texture_image()
    levels = [img]
    for _ in range(3):
        levels.append(ndimage.gaussian_filter(levels[-1], 1.0)[::2, ::2])
    fig, axs = plt.subplots(1, 4, figsize=(6.8, 1.95))
    n = img.shape[0]
    for lvl, (ax, L) in enumerate(zip(axs, levels)):
        show(ax, L, f"level {lvl}: {L.shape[0]} px", vmin=0, vmax=1, extent=[0, n, n, 0])
        # the same 24 px (level-0) displacement, drawn as a scale bar and labelled in this level's pixels
        x0, y0 = 16, 238
        ax.add_patch(plt.Rectangle((x0 - 6, y0 - 13), 80, 26, fc="white", ec="none", alpha=0.85))
        ax.plot([x0, x0 + 24], [y0, y0], color=ORANGE, lw=2.2, solid_capstyle="butt")
        for xe in (x0, x0 + 24):
            ax.plot([xe, xe], [y0 - 7, y0 + 7], color=ORANGE, lw=1.2)
        ax.text(x0 + 30, y0, f"{24 // 2 ** lvl} px", color=ORANGE, fontsize=7.5, ha="left", va="center")
    fig.tight_layout(pad=0.3)
    save(fig, "fig_pyramid.pdf")


# ----------------------------------------------------------------------------
# Chapter 22: corners and Lucas-Kanade
# ----------------------------------------------------------------------------
def _corner_image(n=160, seed=6):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    img = np.full((n, n), 0.25)
    cb = ((xx // 12 + yy // 12) % 2 == 0) & (xx > 10) & (xx < 70) & (yy > 10) & (yy < 70)
    img[cb] = 0.85
    img[(xx > 90) & (xx < 145) & (yy > 15) & (yy < 55)] = 0.7
    img[(xx - 118) ** 2 + (yy - 110) ** 2 < 22**2] = 0.9
    tri = (yy > 90) & (yy < 145) & (xx > 15) & (xx - 15 < (yy - 90) * 1.2)
    img[tri] = 0.6
    img[(xx > 60) & (xx < 75) & (yy > 95)] = 0.05
    img = ndimage.gaussian_filter(img, 0.7) + rng.normal(0, 0.02, img.shape)
    return img


def _shi_tomasi(img, sigma=1.5):
    Iu = ndimage.sobel(img, axis=1) / 8
    Iv = ndimage.sobel(img, axis=0) / 8
    A = ndimage.gaussian_filter(Iu * Iu, sigma)
    B = ndimage.gaussian_filter(Iu * Iv, sigma)
    C = ndimage.gaussian_filter(Iv * Iv, sigma)
    return 0.5 * (A + C - np.sqrt((A - C) ** 2 + 4 * B**2))


def fig_harris():
    img = _corner_image()
    R = _shi_tomasi(img)
    order = np.argsort(R.ravel())[::-1]
    picked = []
    thr = 0.05 * R.max()
    for idx in order:
        if R.flat[idx] < thr or len(picked) >= 60:
            break
        v, u = divmod(idx, R.shape[1])
        if u < 4 or v < 4 or u > R.shape[1] - 5 or v > R.shape[0] - 5:
            continue
        if all((u - pu) ** 2 + (v - pv) ** 2 >= 8**2 for pu, pv in picked):
            picked.append((u, v))
    fig, axs = plt.subplots(1, 2, figsize=(6.0, 3.0))
    show(axs[0], np.sqrt(np.maximum(R, 0)), r"Shi–Tomasi response $\sqrt{\lambda_{\min}}$", cmap="magma")
    show(axs[1], img, f"{len(picked)} strongest corners, spacing 8 px", vmin=0, vmax=1)
    pk = np.array(picked)
    axs[1].plot(pk[:, 0], pk[:, 1], "o", mfc="none", mec=ORANGE, ms=6, mew=1.3)
    fig.tight_layout(pad=0.4)
    save(fig, "fig_harris.pdf")


def _lk_track(I1, I2, p, d0, iters, half=10):
    ys, xs = np.mgrid[-half:half + 1, -half:half + 1]
    u, v = p[0] + xs, p[1] + ys
    T = ndimage.map_coordinates(I1, [v, u], order=1)
    gu = ndimage.map_coordinates(ndimage.sobel(I1, axis=1) / 8, [v, u], order=1)
    gv = ndimage.map_coordinates(ndimage.sobel(I1, axis=0) / 8, [v, u], order=1)
    M = np.array([[np.sum(gu * gu), np.sum(gu * gv)], [np.sum(gu * gv), np.sum(gv * gv)]])
    d = np.array(d0, float)
    hist = [d.copy()]
    for _ in range(iters):
        W = ndimage.map_coordinates(I2, [v + d[1], u + d[0]], order=3, mode="nearest")
        e = T - W
        d = d + np.linalg.solve(M, np.array([np.sum(gu * e), np.sum(gv * e)]))
        hist.append(d.copy())
    return d, hist


def fig_lk():
    rng = np.random.default_rng(8)
    n = 160
    I1 = normalize(ndimage.gaussian_filter(rng.standard_normal((n, n)), 2.0))
    p = np.array([80.0, 80.0])

    def moved(d):
        return ndimage.shift(I1, (d[1], d[0]), order=3, mode="nearest")

    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.8))
    d_small = np.array([0.6, -0.4])
    _, hist = _lk_track(I1, moved(d_small), p, [0.0, 0.0], 8)
    err = [np.linalg.norm(h - d_small) for h in hist]
    axs[0].semilogy(range(len(err)), err, "o-", color=BLUE, ms=3.5, lw=1.2)
    axs[0].set_xlabel("iteration")
    axs[0].set_ylabel("tracking error [px]")
    axs[0].set_title(r"(a) shift $(0.6, -0.4)$ px")
    d_big = np.array([9.0, -6.0])
    I2 = moved(d_big)
    _, hist1 = _lk_track(I1, I2, p, [0.0, 0.0], 18)
    err1 = [np.linalg.norm(h - d_big) for h in hist1]
    # three-level pyramid
    pyr1, pyr2 = [I1], [I2]
    for _ in range(2):
        pyr1.append(ndimage.gaussian_filter(pyr1[-1], 1.0)[::2, ::2])
        pyr2.append(ndimage.gaussian_filter(pyr2[-1], 1.0)[::2, ::2])
    d = np.zeros(2)
    errp, switches = [np.linalg.norm(d_big)], []
    for lvl in (2, 1, 0):
        s = 2.0**lvl
        d_l, hist_l = _lk_track(pyr1[lvl], pyr2[lvl], p / s, d / s, 6, half=10 if lvl == 0 else 7)
        errp.extend(np.linalg.norm(h * s - d_big) for h in hist_l[1:])
        d = d_l * s
        switches.append(len(errp) - 1)
    axs[1].semilogy(range(len(err1)), err1, "o-", color=ORANGE, ms=3, lw=1.1, label="single level")
    axs[1].semilogy(range(len(errp)), errp, "s-", color=BLUE, ms=3, lw=1.1, label="3-level pyramid")
    for sw in switches[:-1]:
        axs[1].axvline(sw + 0.5, color=GREY, ls=":", lw=0.8)
    axs[1].set_xlabel("iteration")
    axs[1].set_ylabel("tracking error [px]")
    axs[1].set_title(r"(b) shift $(9, -6)$ px")
    axs[1].legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    save(fig, "fig_lk.pdf")
    print(f"  LK: small final {err[-1]:.2e}; big single {err1[-1]:.2f} px, pyramid {errp[-1]:.2e} px")


# ----------------------------------------------------------------------------
# Chapter 23: RANSAC line fit
# ----------------------------------------------------------------------------
def fig_ransac():
    rng = np.random.default_rng(4)
    n_in, n_out = 60, 40
    a_true, b_true = 0.5, 1.0
    x_in = rng.uniform(0, 10, n_in)
    y_in = a_true * x_in + b_true + rng.normal(0, 0.2, n_in)
    x_out = rng.uniform(0, 10, n_out)
    y_out = rng.uniform(-1, 10, n_out)
    x = np.concatenate([x_in, x_out])
    y = np.concatenate([y_in, y_out])
    a_ls, b_ls = np.polyfit(x, y, 1)
    best = None
    thr = 0.6
    for _ in range(100):
        i, j = rng.choice(len(x), 2, replace=False)
        if abs(x[i] - x[j]) < 1e-9:
            continue
        a = (y[j] - y[i]) / (x[j] - x[i])
        b = y[i] - a * x[i]
        d = np.abs(a * x - y + b) / np.sqrt(a * a + 1)
        inl = d < thr
        if best is None or inl.sum() > best.sum():
            best = inl
    a_r, b_r = np.polyfit(x[best], y[best], 1)
    xx = np.linspace(0, 10, 50)
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.plot(x[~best], y[~best], "o", color=GREY, ms=3.5, alpha=0.7, label="data")
    ax.plot(x[best], y[best], "o", color=GREY, ms=3.5, alpha=0.7)
    ax.plot(x[best], y[best], "o", mfc="none", mec=BLUE, ms=7, mew=1.0, label=f"RANSAC inliers ({best.sum()})")
    ax.plot(xx, a_ls * xx + b_ls, color=ORANGE, ls="--", lw=1.6, label="least squares")
    ax.plot(xx, a_r * xx + b_r, color=BLUE, lw=1.6, label="RANSAC + refit")
    ax.plot(xx, a_true * xx + b_true, color="k", ls=":", lw=1.0, label="truth")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_ylim(-1.5, 12.5)
    save(fig, "fig_ransac.pdf")


ALL = [fig_svd_geometry, fig_taylor_sin, fig_ode_methods, fig_fd_error, fig_gauss2d, fig_clt, fig_processes,
       fig_aliasing, fig_slerp, fig_lsfit, fig_gn_lm, fig_robust_kernels, fig_ba_sparsity, fig_ut, fig_ahrs,
       fig_accel_bode, fig_coning, fig_image_filters, fig_aliasing_image, fig_pyramid, fig_harris, fig_lk, fig_ransac]


def main(names=None):
    for f in ALL:
        if not names or f.__name__ in names:
            f()


if __name__ == "__main__":
    main(sys.argv[1:])
