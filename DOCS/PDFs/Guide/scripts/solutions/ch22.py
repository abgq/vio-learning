"""Exercise solutions, Chapter 22 (exercises 1-5): prints the numbers quoted in the worked solutions.

Run from DOCS/PDFs/Guide:  .venv/Scripts/python.exe scripts/solutions/<this file>
"""
import numpy as np
from scipy import ndimage


def normalize(a):
    a = a - a.min()
    return a / a.max()


# 22.1 structure tensors of an ideal edge and corner (central differences, uniform 15x15 window centered on the feature)
n = 41
yy, xx = np.mgrid[0:n, 0:n]
c = n // 2
h = 1.0
edge = np.where(xx >= c, h, 0.0)
corner = np.where((xx >= c) & (yy >= c), h, 0.0)


def tensor(img, half=7):
    Iu = ndimage.correlate1d(img, [-0.5, 0, 0.5], axis=1)
    Iv = ndimage.correlate1d(img, [-0.5, 0, 0.5], axis=0)
    sl = (slice(c - half, c + half + 1), slice(c - half, c + half + 1))
    a, b, d = (Iu * Iu)[sl].sum(), (Iu * Iv)[sl].sum(), (Iv * Iv)[sl].sum()
    M = np.array([[a, b], [b, d]])
    return M, np.linalg.eigvalsh(M)


for lab, img in (("edge", edge), ("corner", corner)):
    M, ev = tensor(img)
    print(f"22.1 {lab}: M = {M.tolist()}, eigenvalues {ev}")
# corner with angle alpha between normals: eigenvalues c(1 +- cos alpha)
for alpha_deg in (90, 45, 20):
    a = np.radians(alpha_deg)
    n1 = np.array([1.0, 0.0])
    n2 = np.array([np.cos(a), np.sin(a)])
    print(f"     normals at {alpha_deg} deg: eigenvalues of n1n1T+n2n2T", np.linalg.eigvalsh(np.outer(n1, n1) + np.outer(n2, n2)), "1-cos =", 1 - np.cos(a))

# 22.2 cancellation check of tr^2-4det form
M = np.array([[1.0, 1e-9], [1e-9, 1.0 + 1e-9]])
tr, det = np.trace(M), np.linalg.det(M)
print("22.2 tr^2-4det =", tr * tr - 4 * det, " (a-c)^2+4b^2 =", (M[0, 0] - M[1, 1]) ** 2 + 4 * M[0, 1] ** 2)


# 22.3 LK
def lk_track(I1, I2, p, d0, iters, half=10, tol=None):
    ys, xs = np.mgrid[-half:half + 1, -half:half + 1]
    u, v = p[0] + xs, p[1] + ys
    T = ndimage.map_coordinates(I1, [v, u], order=1)
    gu = ndimage.map_coordinates(ndimage.sobel(I1, axis=1) / 8, [v, u], order=1)
    gv = ndimage.map_coordinates(ndimage.sobel(I1, axis=0) / 8, [v, u], order=1)
    M = np.array([[np.sum(gu * gu), np.sum(gu * gv)], [np.sum(gu * gv), np.sum(gv * gv)]])
    d = np.array(d0, float)
    hist = [d.copy()]
    for _ in range(iters):
        W = ndimage.map_coordinates(I2, [v + d[1], u + d[0]], order=1, mode="nearest")
        e = T - W
        dd = np.linalg.solve(M, np.array([np.sum(gu * e), np.sum(gv * e)]))
        d = d + dd
        hist.append(d.copy())
        if tol is not None and np.linalg.norm(dd) < tol:
            break
    return d, hist, M


rng = np.random.default_rng(8)
N = 160
I1 = normalize(ndimage.gaussian_filter(rng.standard_normal((N, N)), 2.0))
p = np.array([80.0, 80.0])


def moved(img, d):
    return ndimage.shift(img, (d[1], d[0]), order=3, mode="nearest")


d_small = np.array([0.3, -0.7])
_, hist, M = lk_track(I1, moved(I1, d_small), p, [0, 0], 8)
print("22.3 shift (0.3,-0.7): errors per iteration", ["%.1e" % np.linalg.norm(h - d_small) for h in hist])


def pyr_track(I1, I2, p, levels, iters=10, half=7):
    pyr1, pyr2 = [I1], [I2]
    for _ in range(levels):
        pyr1.append(ndimage.gaussian_filter(pyr1[-1], 1.0)[::2, ::2])
        pyr2.append(ndimage.gaussian_filter(pyr2[-1], 1.0)[::2, ::2])
    d = np.zeros(2)
    for lvl in range(levels, -1, -1):
        s = 2.0**lvl
        d_l, _, _ = lk_track(pyr1[lvl], pyr2[lvl], p / s, d / s, iters, half=half)
        d = d_l * s
    return d


for shift in ((8.0, 0.0), (5.66, -5.66), (0.0, 8.0)):
    d_big = np.array(shift)
    I2 = moved(I1, d_big)
    res = []
    for L in range(0, 4):
        # many feature positions to get a success rate
        ok = 0
        tot = 0
        for py in range(40, 121, 10):
            for px in range(40, 121, 10):
                d = pyr_track(I1, I2, np.array([px, py], float), L)
                ok += np.linalg.norm(d - d_big) < 0.1
                tot += 1
        res.append(ok / tot)
    print(f"22.3 shift {shift}: success rate (err<0.1px) with 0..3 pyramid levels above the base: {np.round(res, 2)}")

# texture correlation length: autocorrelation of I1
f = np.fft.fft2(I1 - I1.mean())
ac = np.real(np.fft.ifft2(f * np.conj(f)))
ac /= ac[0, 0]
print("     autocorrelation at lag 1..8 px:", np.round(ac[0, 1:9], 3))

# 22.4 forward-backward error statistics with noise
sig_I = 0.01  # noise std in units of the normalized [0,1] image
fb = []
fwd_err = []
for trial in range(300):
    r = np.random.default_rng(100 + trial)
    d = r.uniform(-1.5, 1.5, 2)
    J1 = I1 + r.normal(0, sig_I, I1.shape)
    J2 = moved(I1, d) + r.normal(0, sig_I, I1.shape)
    pp = np.array([r.uniform(40, 120), r.uniform(40, 120)])
    dfw, _, M = lk_track(J1, J2, pp, [0, 0], 20, half=7, tol=0.001)
    dbw, _, _ = lk_track(J2, J1, pp + dfw, [0, 0], 20, half=7, tol=0.001)
    fb.append(np.linalg.norm(dfw + dbw))
    fwd_err.append(np.linalg.norm(dfw - d))
fb = np.array(fb)
lam2 = np.linalg.eigvalsh(M)[0]
print(f"22.4 sigma_I={sig_I}: FB error median {np.median(fb):.4f} px, 95% {np.percentile(fb,95):.4f}, 99% {np.percentile(fb,99):.4f}, max {fb.max():.4f}; forward error median {np.median(fwd_err):.4f}")
print(f"     predicted per-axis std of one track sigma/sqrt(lambda) ~ {sig_I/np.sqrt(np.linalg.eigvalsh(M)).round(5)} (last window)")
# noise-free: interpolation only
fb0 = []
for trial in range(100):
    r = np.random.default_rng(500 + trial)
    d = r.uniform(-1.5, 1.5, 2)
    J2 = moved(I1, d)
    pp = np.array([r.uniform(40, 120), r.uniform(40, 120)])
    dfw, _, _ = lk_track(I1, J2, pp, [0, 0], 30, half=7, tol=1e-5)
    dbw, _, _ = lk_track(J2, I1, pp + dfw, [0, 0], 30, half=7, tol=1e-5)
    fb0.append(np.linalg.norm(dfw + dbw))
print(f"     noise-free FB error (interpolation only): median {np.median(fb0):.4f}, max {np.max(fb0):.4f} px")

# a wrong track: occluding patch appears in second image near the feature
fbw = []
for trial in range(100):
    r = np.random.default_rng(900 + trial)
    d = r.uniform(-1.5, 1.5, 2)
    J2 = moved(I1, d).copy()
    pp = np.array([r.uniform(40, 120), r.uniform(40, 120)])
    # occluder: replace half of the window region by another texture moving differently
    other = normalize(ndimage.gaussian_filter(r.standard_normal((N, N)), 2.0))
    x0 = int(pp[0])
    J2[:, x0:] = other[:, x0:]
    dfw, _, _ = lk_track(I1, J2, pp, [0, 0], 20, half=7, tol=0.001)
    dbw, _, _ = lk_track(J2, I1, pp + dfw, [0, 0], 20, half=7, tol=0.001)
    fbw.append(np.linalg.norm(dfw + dbw))
fbw = np.array(fbw)
print(f"     occluded (half window replaced) FB error: median {np.median(fbw):.3f} px, fraction > 0.5 px: {(fbw>0.5).mean():.2f}, > 0.1 px: {(fbw>0.1).mean():.2f}")

# 22.5 gyro prediction
fu, cu, cv = 458.654, 367.215, 248.375
th = 0.05
K = np.array([[fu, 0, cu], [0, 457.296, cv], [0, 0, 1.0]])
Ry = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
R_c2c1 = Ry.T
u2 = K @ R_c2c1 @ np.linalg.inv(K) @ np.array([cu, cv, 1.0])
u2 = u2 / u2[2]
print("22.5 predicted pixel:", u2, " displacement", u2[0] - cu, u2[1] - cv, " fu*tan:", fu * np.tan(th))
# translation effect for comparison: 1 m/s, 20 Hz, depth 5 m
print("     translation 1 m/s over 50 ms at 5 m depth:", fu * 0.05 / 5, "px")
