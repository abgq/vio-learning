"""Exercise solutions, Chapter 23: prints the numbers quoted in the worked solutions.

Run from DOCS/PDFs/Guide:  .venv/Scripts/python.exe scripts/solutions/<this file>
"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot

rng = np.random.default_rng(23)
fu = fv = 458.0
cu, cv = 367.0, 248.0
K = np.array([[fu, 0, cu], [0, fv, cv], [0, 0, 1.0]])


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def scene(n, rng, t=None, R=None, depth=(3, 8)):
    if R is None:
        R = Rot.from_rotvec(rng.normal(0, 0.1, 3)).as_matrix()
    if t is None:
        t = rng.normal(0, 1, 3)
        t = 0.5 * t / np.linalg.norm(t)
    X1 = np.column_stack([rng.uniform(-3, 3, n), rng.uniform(-2, 2, n), rng.uniform(*depth, n)])
    X2 = X1 @ R.T + t
    keep = X2[:, 2] > 0.5
    return R, t, X1[keep], X2[keep]


# 23.1 epipolar constraint
R, t, X1, X2 = scene(20, rng)
x1 = X1 / X1[:, 2:]
x2 = X2 / X2[:, 2:]
E = skew(t) @ R
res = np.einsum("ij,jk,ik->i", x2, E, x1)
print("23.1 max |x2^T E x1| for inliers:", np.abs(res).max())


def sampson(E, x1, x2):
    Ex1 = x1 @ E.T
    Etx2 = x2 @ E
    num = np.einsum("ij,ij->i", x2, Ex1) ** 2
    den = Ex1[:, 0] ** 2 + Ex1[:, 1] ** 2 + Etx2[:, 0] ** 2 + Etx2[:, 1] ** 2
    return num / den


x2o = x2.copy()
x2o[0, :2] += np.array([0.05, -0.03])  # outlier: 0.058 normalized = 27 px
print("     outlier algebraic residual", abs(x2o[0] @ E @ x1[0]), " sqrt Sampson", np.sqrt(sampson(E, x1[:1], x2o[:1]))[0], "=", np.sqrt(sampson(E, x1[:1], x2o[:1]))[0] * fu, "px")
# distance of displaced point to epipolar line in image 2
l = E @ x1[0]
print("     point-line distance in image 2:", abs(l @ x2o[0]) / np.hypot(l[0], l[1]) * fu, "px")
print("     inlier sqrt Sampson max:", np.sqrt(sampson(E, x1, x2)).max())

# 23.2 left null vector
print("23.2 t^T E =", t @ E, " E R^T t =", E @ R.T @ t, " singular values", np.linalg.svd(E, compute_uv=False), "|t| =", np.linalg.norm(t))


# 23.3 eight-point with/without Hartley normalization on pixels
def eight_point(p1, p2):
    A = np.column_stack([p2[:, 0] * p1[:, 0], p2[:, 0] * p1[:, 1], p2[:, 0], p2[:, 1] * p1[:, 0], p2[:, 1] * p1[:, 1], p2[:, 1], p1[:, 0], p1[:, 1], np.ones(len(p1))])
    _, s, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)
    U, S, Vt2 = np.linalg.svd(F)
    F = U @ np.diag([S[0], S[1], 0]) @ Vt2
    return F, s


def normalize_pts(p):
    m = p[:, :2].mean(0)
    d = np.mean(np.linalg.norm(p[:, :2] - m, axis=1))
    s = np.sqrt(2) / d
    T = np.array([[s, 0, -s * m[0]], [0, s, -s * m[1]], [0, 0, 1]])
    return p @ T.T, T


def rot_err_from_F(F, R_true, t_true, x1n, x2n):
    E = K.T @ F @ K
    sols = decompose(E)
    best = pick(sols, x1n, x2n)
    Rb, tb = best
    ang = np.degrees(np.linalg.norm(Rot.from_matrix(Rb @ R_true.T).as_rotvec()))
    tang = np.degrees(np.arccos(np.clip(abs(tb @ t_true) / np.linalg.norm(t_true), -1, 1)))
    return ang, tang


W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])


def decompose(E):
    U, S, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(Vt) < 0:
        Vt = -Vt
    u3 = U[:, 2]
    return [(U @ W @ Vt, u3), (U @ W @ Vt, -u3), (U @ W.T @ Vt, u3), (U @ W.T @ Vt, -u3)]


def triangulate2(R, t, x1, x2):
    # point in frame 1: X2 = R X1 + t ; solve [x1, -R^T x2] depths
    out = []
    for a, b in zip(x1, x2):
        # X1 = l1 a ; X2 = l2 b = R l1 a + t -> l1 R a - l2 b = -t
        Am = np.column_stack([R @ a, -b])
        l, *_ = np.linalg.lstsq(Am, -t, rcond=None)
        out.append(l)
    return np.array(out)


def pick(sols, x1, x2):
    best, bestn = None, -1
    for R, t in sols:
        l = triangulate2(R, t, x1, x2)
        n = np.sum((l[:, 0] > 0) & (l[:, 1] > 0))
        if n > bestn:
            best, bestn = (R, t), n
    return best


errs_raw, errs_norm, conds = [], [], []
for trial in range(200):
    r = np.random.default_rng(1000 + trial)
    R, t, X1, X2 = scene(60, r)
    x1n = X1 / X1[:, 2:]
    x2n = X2 / X2[:, 2:]
    p1 = x1n @ K.T + np.column_stack([r.normal(0, 0.5, (len(x1n), 2)), np.zeros(len(x1n))])
    p2 = x2n @ K.T + np.column_stack([r.normal(0, 0.5, (len(x2n), 2)), np.zeros(len(x2n))])
    F_raw, s_raw = eight_point(p1, p2)
    q1, T1 = normalize_pts(p1)
    q2, T2 = normalize_pts(p2)
    F_n, s_n = eight_point(q1, q2)
    F_n = T2.T @ F_n @ T1
    xx1 = p1 @ np.linalg.inv(K).T
    xx2 = p2 @ np.linalg.inv(K).T
    errs_raw.append(rot_err_from_F(F_raw, R, t, xx1, xx2))
    errs_norm.append(rot_err_from_F(F_n, R, t, xx1, xx2))
    conds.append((s_raw[0] / s_raw[-2], s_n[0] / s_n[-2]))
errs_raw, errs_norm, conds = np.array(errs_raw), np.array(errs_norm), np.array(conds)
print("23.3 median rotation error deg raw / normalized:", np.median(errs_raw[:, 0]), np.median(errs_norm[:, 0]))
print("     median translation direction error deg raw / normalized:", np.median(errs_raw[:, 1]), np.median(errs_norm[:, 1]))
print("     90th pct rotation error raw / norm:", np.percentile(errs_raw[:, 0], 90), np.percentile(errs_norm[:, 0], 90))
print("     90th pct translation error raw / norm:", np.percentile(errs_raw[:, 1], 90), np.percentile(errs_norm[:, 1], 90))
print("     median condition sigma1/sigma8 raw / normalized:", np.median(conds[:, 0]), np.median(conds[:, 1]))

# 23.4 decomposition + cheirality
ok = 0
skipped = 0
for trial in range(500):
    r = np.random.default_rng(5000 + trial)
    R, t, X1, X2 = scene(30, r, R=Rot.from_rotvec(r.normal(0, 0.3, 3)).as_matrix())
    if len(X1) < 5:
        skipped += 1
        continue
    x1n = X1 / X1[:, 2:]
    x2n = X2 / X2[:, 2:]
    E = skew(t) @ R
    Rb, tb = pick(decompose(E), x1n, x2n)
    ok += np.allclose(Rb, R, atol=1e-8) and np.allclose(tb, t / np.linalg.norm(t), atol=1e-8)
print("23.4 correct decomposition selected:", ok, "/", 500 - skipped)
# how many points pass for each of the four solutions in one example
R, t, X1, X2 = scene(30, np.random.default_rng(7))
x1n = X1 / X1[:, 2:]
x2n = X2 / X2[:, 2:]
for i, (Rc, tc) in enumerate(decompose(skew(t) @ R)):
    l = triangulate2(Rc, tc, x1n, x2n)
    print("     solution", i, "points in front of both:", np.sum((l[:, 0] > 0) & (l[:, 1] > 0)), "of", len(l), "| front of 1 only:", np.sum((l[:, 0] > 0) & (l[:, 1] < 0)), "| behind 1:", np.sum(l[:, 0] < 0))


# 23.5 two-point RANSAC with known rotation
def ransac_n(w, s, p=0.99):
    return int(np.ceil(np.log(1 - p) / np.log(1 - w**s)))


print("23.5 iterations at w=0.7: 2-pt", ransac_n(0.7, 2), " 5-pt", ransac_n(0.7, 5), " 8-pt", ransac_n(0.7, 8))
succ = 0
iters_used = []
tdir_err = []
for trial in range(300):
    r = np.random.default_rng(9000 + trial)
    R, t, X1, X2 = scene(100, r)
    n = len(X1)
    x1n = X1 / X1[:, 2:]
    x2n = X2 / X2[:, 2:]
    sig = 1.0 / fu
    x1m = x1n.copy(); x2m = x2n.copy()
    x1m[:, :2] += r.normal(0, sig, (n, 2))
    x2m[:, :2] += r.normal(0, sig, (n, 2))
    nout = int(0.3 * n)
    out = r.choice(n, nout, replace=False)
    x2m[out, :2] = np.column_stack([r.uniform(-0.8, 0.8, nout), r.uniform(-0.55, 0.55, nout)])
    inl_true = np.ones(n, bool); inl_true[out] = False
    c = np.cross(x1m @ R.T, x2m)  # c_i = R x1 x x2
    best_inl, best_t = None, None
    N, k = 1000, 0
    thr = (2.0 / fu) ** 2
    while k < N:
        i, j = r.choice(n, 2, replace=False)
        th = np.cross(c[i], c[j])
        if np.linalg.norm(th) < 1e-12:
            k += 1; continue
        th /= np.linalg.norm(th)
        Eh = skew(th) @ R
        d = sampson(Eh, x1m, x2m)
        inl = d < thr
        if best_inl is None or inl.sum() > best_inl.sum():
            best_inl, best_t = inl, th
            w = inl.sum() / n
            N = min(N, ransac_n(max(w, 1e-3), 2))
        k += 1
    iters_used.append(k)
    succ += np.mean(best_inl == inl_true)
    tdir_err.append(np.degrees(np.arccos(min(1, abs(best_t @ t) / np.linalg.norm(t)))))
print(f"     2-pt RANSAC: mean iterations {np.mean(iters_used):.1f}, classification accuracy {succ/300:.3f}, median t-direction error {np.median(tdir_err):.2f} deg")
print("     expected draws until first clean sample: 2-pt", 1 / 0.7**2, " 5-pt", 1 / 0.7**5)

# 23.6 pure rotation: 8-point singular values, and the parallax statistic
for lab, tt in (("pure rotation", np.zeros(3)), ("10 cm sideways", np.array([0.1, 0, 0]))):
    r = np.random.default_rng(77)
    R = Rot.from_rotvec([0.0, 0.05, 0.0]).as_matrix()
    _, _, X1, X2 = scene(100, r, t=tt, R=R, depth=(3, 8))
    x1n = X1 / X1[:, 2:]
    x2n = X2 / X2[:, 2:]
    sig = 1.0 / fu
    x1m = x1n.copy(); x2m = x2n.copy()
    x1m[:, :2] += r.normal(0, sig, x1m[:, :2].shape)
    x2m[:, :2] += r.normal(0, sig, x2m[:, :2].shape)
    A = np.column_stack([x2m[:, 0] * x1m[:, 0], x2m[:, 0] * x1m[:, 1], x2m[:, 0], x2m[:, 1] * x1m[:, 0], x2m[:, 1] * x1m[:, 1], x2m[:, 1], x1m[:, 0], x1m[:, 1], np.ones(len(x1m))])
    s = np.linalg.svd(A, compute_uv=False)
    Rx1 = x1m @ R.T
    par = np.linalg.norm(Rx1[:, :2] / Rx1[:, 2:] - x2m[:, :2], axis=1) * fu
    print(f"23.6 {lab}: smallest four singular values of A {np.round(s[-4:], 5)}, median rotation-compensated parallax {np.median(par):.2f} px")
    # with pure rotation, any t works: residuals of E=[t]x R for random t
    if lab == "pure rotation":
        for k in range(3):
            tr = r.normal(size=3); tr /= np.linalg.norm(tr)
            d = np.sqrt(sampson(skew(tr) @ R, x1m, x2m)) * fu
            print(f"     random t #{k}: fraction of tracks with Sampson distance < 1 px: {(d < 1).mean():.2f}")

# 23.8 triangulation accuracy: two views and three views over the same baseline
Z = 5.0
sig = 1.0 / fu
for par_deg in (1.0, 2.0, 5.0, 10.0):
    b = Z * np.tan(np.radians(par_deg))
    pred = Z**2 / b * np.sqrt(2) * sig
    for nviews in (2, 3, 5):
        s_pos = np.linspace(0, b, nviews)
        r = np.random.default_rng(42)
        Zs = []
        for trial in range(20000):
            X = b / 2
            xs = (X - s_pos) / Z + r.normal(0, sig, nviews)
            # linear regression x = a - s w
            A = np.column_stack([np.ones(nviews), -s_pos])
            a_hat, w_hat = np.linalg.lstsq(A, xs, rcond=None)[0]
            Zs.append(1 / w_hat if w_hat > 0 else np.inf)
        Zs = np.array(Zs)
        fin = np.isfinite(Zs) & (Zs < 1e3)
        rel = np.std(Zs[fin] - Z) / Z
        med = np.median(np.abs(Zs - Z)) / 0.6745 / Z
        lin_pred = Z**2 * sig / np.sqrt(np.sum((s_pos - s_pos.mean()) ** 2)) / Z
        print(f"23.8 parallax {par_deg} deg (b={b:.3f} m), {nviews} views: robust rel std {med:.3f}, linear prediction {lin_pred:.3f}; two-view formula {pred/Z:.3f}; negative depth {np.mean(~fin):.4f}")

# ---- 23.5 two-point RANSAC with a least-squares refit


def sampson(E, x1, x2):
    Ex1 = x1 @ E.T
    Etx2 = x2 @ E
    num = np.einsum("ij,ij->i", x2, Ex1) ** 2
    den = Ex1[:, 0] ** 2 + Ex1[:, 1] ** 2 + Etx2[:, 0] ** 2 + Etx2[:, 1] ** 2
    return num / den


def ransac_n(w, s, p=0.99):
    return int(np.ceil(np.log(1 - p) / np.log(1 - w**s)))


stats = []
for trial in range(300):
    r = np.random.default_rng(9000 + trial)
    R, t, X1, X2 = scene(100, r)
    n = len(X1)
    x1n = X1 / X1[:, 2:]
    x2n = X2 / X2[:, 2:]
    sig = 1.0 / fu
    x1m = x1n.copy(); x2m = x2n.copy()
    x1m[:, :2] += r.normal(0, sig, (n, 2))
    x2m[:, :2] += r.normal(0, sig, (n, 2))
    nout = int(0.3 * n)
    out = r.choice(n, nout, replace=False)
    x2m[out, :2] = np.column_stack([r.uniform(-0.8, 0.8, nout), r.uniform(-0.55, 0.55, nout)])
    inl_true = np.ones(n, bool); inl_true[out] = False
    c = np.cross(x1m @ R.T, x2m)
    best_inl, best_t = None, None
    N, k = 1000, 0
    thr = (2.0 / fu) ** 2
    while k < N:
        i, j = r.choice(n, 2, replace=False)
        th = np.cross(c[i], c[j])
        k += 1
        if np.linalg.norm(th) < 1e-12:
            continue
        th /= np.linalg.norm(th)
        inl = sampson(skew(th) @ R, x1m, x2m) < thr
        if best_inl is None or inl.sum() > best_inl.sum():
            best_inl, best_t = inl, th
            N = min(N, ransac_n(max(inl.sum() / n, 1e-3), 2))
    # refit on inliers: unit t minimizing sum (t.c_i)^2 over inliers, then reclassify (twice)
    t_fit = best_t
    for _ in range(2):
        C = c[best_inl]
        w_, V = np.linalg.eigh(C.T @ C)
        t_fit = V[:, 0]
        best_inl = sampson(skew(t_fit) @ R, x1m, x2m) < thr
    fr = np.mean(~best_inl[inl_true])   # inliers rejected
    fa = np.mean(best_inl[~inl_true])   # outliers accepted
    terr = np.degrees(np.arccos(min(1, abs(t_fit @ t) / np.linalg.norm(t))))
    terr0 = np.degrees(np.arccos(min(1, abs(best_t @ t) / np.linalg.norm(t))))
    stats.append((k, fr, fa, terr, terr0))
stats = np.array(stats)
print(f"23.5 iterations mean {stats[:,0].mean():.1f}; inliers rejected {stats[:,1].mean():.3f}; outliers accepted {stats[:,2].mean():.3f};"
      f" t-direction error median: best hypothesis {np.median(stats[:,4]):.2f} deg, after refit {np.median(stats[:,3]):.2f} deg")
from scipy.stats import norm
print("     expected inlier rejection at 2 sigma:", 2 * norm.sf(2))
