"""Exercise solutions, Chapter 10: prints the numbers quoted in the worked solutions.

Run from DOCS/PDFs/Guide:  .venv/Scripts/python.exe scripts/solutions/<this file>
"""
import numpy as np

rng = np.random.default_rng(2)
t = np.linspace(0, 1.5, 25)
a_true, b_true = 2.0, 1.3
y = a_true * np.exp(-b_true * t) + rng.normal(0, 0.04, t.size)


def r(th):
    return th[0] * np.exp(-th[1] * t) - y


def J(th):
    e = np.exp(-th[1] * t)
    return np.column_stack([e, -th[0] * t * e])


def cost(th):
    v = r(th)
    return 0.5 * v @ v


def gn(th, it=50):
    for _ in range(it):
        Jm, rv = J(th), r(th)
        try:
            th = th - np.linalg.solve(Jm.T @ Jm, Jm.T @ rv)
        except np.linalg.LinAlgError:
            return th, False
        if not np.all(np.isfinite(th)) or abs(th[1]) > 1e3:
            return th, False
    return th, True


def lm(th, it=300):
    lam = 1e-2
    import warnings; warnings.filterwarnings('ignore')
    for _ in range(it):
        Jm, rv = J(th), r(th)
        Hm = Jm.T @ Jm
        step = -np.linalg.solve(Hm + lam * np.diag(np.maximum(np.diag(Hm), 1e-9)), Jm.T @ rv)
        if cost(th + step) < cost(th):
            th = th + step
            lam = max(lam / 10, 1e-12)
        else:
            lam *= 10
    return th, True


ref, _ = gn(np.array([2.0, 1.3]))
print("10.2 optimum", ref)
ok_gn = ok_lm = tot = 0
for a0 in np.linspace(-1.75, 5.75, 16):
    for b0 in np.linspace(-2, 8, 21):
        tot += 1
        with np.errstate(all="ignore"):
            th, fine = gn(np.array([a0, b0]))
        ok_gn += fine and np.allclose(th, ref, atol=1e-6)
        th, _ = lm(np.array([a0, b0]))
        ok_lm += np.allclose(th, ref, atol=1e-6)
print(f"10.2 starts converging: GN {ok_gn}/{tot}, LM {ok_lm}/{tot}")
for start in ([0.5, 6.0], [4.0, -1.0], [1.0, 4.0]):
    with np.errstate(all="ignore"):
        th1, f1 = gn(np.array(start, float))
    th2, _ = lm(np.array(start, float))
    print("     start", start, "GN ->", np.round(th1, 3), f1, " LM ->", np.round(th2, 3))

# 10.3 quadratic convergence demo (zero residual)
y0 = a_true * np.exp(-b_true * t)
def r0(th):
    return th[0] * np.exp(-th[1] * t) - y0
th = np.array([2.3, 1.0])
errs = []
for _ in range(6):
    Jm = J(th)
    th = th - np.linalg.solve(Jm.T @ Jm, Jm.T @ r0(th))
    errs.append(np.linalg.norm(th - [a_true, b_true]))
print("10.3 zero-residual GN errors:", ["%.1e" % e for e in errs])

# 10.4 robust line fit
rng = np.random.default_rng(7)
n = 100
x = rng.uniform(0, 10, n)
ytrue = 0.5 * x + 1.0
yy = ytrue + rng.normal(0, 0.2, n)
out = rng.choice(n, 20, replace=False)
yy[out] = rng.uniform(-2, 12, 20)
A = np.column_stack([x, np.ones(n)])


def irls(kind, c, it=50):
    th = np.linalg.lstsq(A, yy, rcond=None)[0]
    for _ in range(it):
        res = yy - A @ th
        s = 1.4826 * np.median(np.abs(res - np.median(res)))  # robust scale
        u = np.abs(res) / s
        if kind == "huber":
            w = np.where(u <= c, 1.0, c / u)
        else:
            w = 1 / (1 + (u / c) ** 2)
        th = np.linalg.solve(A.T @ (A * w[:, None]), A.T @ (w * yy))
    return th


ls = np.linalg.lstsq(A, yy, rcond=None)[0]
hub = irls("huber", 1.345)
cau = irls("cauchy", 2.385)
inl = np.setdiff1d(np.arange(n), out)
oracle = np.linalg.lstsq(A[inl], yy[inl], rcond=None)[0]
for name, th in (("LS", ls), ("Huber", hub), ("Cauchy", cau), ("inliers only", oracle)):
    print(f"10.4 {name:12s} slope {th[0]:.3f} intercept {th[1]:.3f}")

# 10.5 Schur complement
rng = np.random.default_rng(3)
nc, nl = 8, 60
rows = []
for j in range(nl):
    cams = rng.choice(nc, 3, replace=False)
    for c in cams:
        Jc = np.zeros((2, 6 * nc + 3 * nl))
        Jc[:, 6 * c:6 * c + 6] = rng.normal(size=(2, 6))
        Jc[:, 6 * nc + 3 * j:6 * nc + 3 * j + 3] = rng.normal(size=(2, 3))
        rows.append(Jc)
Jm = np.vstack(rows)
res = rng.normal(size=Jm.shape[0])
M = Jm.T @ Jm + 1e-6 * np.eye(Jm.shape[1])
g = -Jm.T @ res
full = np.linalg.solve(M, g)
Am, Bm, Cm = M[:6 * nc, :6 * nc], M[:6 * nc, 6 * nc:], M[6 * nc:, 6 * nc:]
gc, gl = g[:6 * nc], g[6 * nc:]
Cinv = np.zeros_like(Cm)
for j in range(nl):  # block-diagonal inverse
    s = slice(3 * j, 3 * j + 3)
    Cinv[s, s] = np.linalg.inv(Cm[s, s])
dc = np.linalg.solve(Am - Bm @ Cinv @ Bm.T, gc - Bm @ Cinv @ gl)
dl = Cinv @ (gl - Bm.T @ dc)
print("10.5 max |schur - full| =", np.abs(np.concatenate([dc, dl]) - full).max(), " off-diag of C zero:",
      np.abs(Cm - np.where(Cinv != 0, Cm, 0)).max())
Minv = np.linalg.inv(M)
print("10.6 reduced system vs inv of camera block:", np.abs(np.linalg.inv(Minv[:6 * nc, :6 * nc]) - (Am - Bm @ Cinv @ Bm.T)).max())
