"""Exercise solutions, Chapter 26: prints the numbers quoted in the worked solutions.

Run from DOCS/PDFs/Guide:  .venv/Scripts/python.exe scripts/solutions/<this file>
"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot

rng = np.random.default_rng(26)
M, n = 4, 39  # 4 observations, state 15+6*4
m = 2 * M
Hx = rng.normal(size=(m, n))
Hf = rng.normal(size=(m, 3))
L = rng.normal(size=(n, n))
P = L @ L.T / n + 0.1 * np.eye(n)
sig2 = 0.3
r = rng.normal(size=m)

Q, _ = np.linalg.qr(Hf, mode="complete")
A = Q[:, 3:]
Ho, ro = A.T @ Hx, A.T @ r


def update(P, H, R, r):
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    return K @ r, P - K @ H @ P


dx_p, P_p = update(P, Ho, sig2 * np.eye(m - 3), ro)
print("26.1 projected vs augmented-state update with prior lambda on p_f:")
for lam in (1e2, 1e4, 1e6, 1e8, 1e10):
    Paug = np.block([[P, np.zeros((n, 3))], [np.zeros((3, n)), lam * np.eye(3)]])
    Haug = np.hstack([Hx, Hf])
    dxa, Pa = update(Paug, Haug, sig2 * np.eye(m), r)
    print(f"     lambda={lam:.0e}: max |dP| {np.abs(Pa[:n,:n]-P_p).max():.2e}, max |d dx| {np.abs(dxa[:n]-dx_p).max():.2e}")

# 26.2 recovering the feature correction from the discarded rows
Q1 = Q[:, :3]
Rf = Q1.T @ Hf
dpf = np.linalg.solve(Rf, Q1.T @ (r - Hx @ dx_p))
Paug = np.block([[P, np.zeros((n, 3))], [np.zeros((3, n)), 1e10 * np.eye(3)]])
dxa, Pa = update(Paug, np.hstack([Hx, Hf]), sig2 * np.eye(m), r)
print("26.2 feature correction from discarded rows vs augmented update:", np.abs(dpf - dxa[n:]).max())
# covariance of the recovered feature
Pff = np.linalg.solve(Rf, (Q1.T @ (Hx @ P_p @ Hx.T + sig2 * np.eye(m)) @ Q1)) @ np.linalg.inv(Rf).T
Pfx = -np.linalg.solve(Rf, Q1.T @ Hx @ P_p)
print("     feature covariance vs augmented:", np.abs(Pff - Pa[n:, n:]).max() / np.abs(Pa[n:, n:]).max(), " cross-cov:", np.abs(Pfx - Pa[n:, :n]).max())


# 26.3 two views: gradient of the epipolar function is the left null vector of a numerically computed H_f
def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


R_wc1 = Rot.from_rotvec(rng.normal(0, 0.3, 3)).as_matrix()
R_wc2 = Rot.from_rotvec(rng.normal(0, 0.3, 3)).as_matrix()
p_wc1 = rng.normal(0, 1, 3)
p_wc2 = p_wc1 + rng.normal(0, 0.3, 3)
pf = p_wc1 + R_wc1 @ np.array([0.3, -0.2, 5.0])


def proj(pf):
    out = []
    for R, p in ((R_wc1, p_wc1), (R_wc2, p_wc2)):
        X = R.T @ (pf - p)
        out.extend(X[:2] / X[2])
    return np.array(out)


e = 1e-6
Hf_num = np.column_stack([(proj(pf + e * ei) - proj(pf - e * ei)) / (2 * e) for ei in np.eye(3)])
# relative pose c1 -> c2: X2 = R X1 + t
Rr = R_wc2.T @ R_wc1
t = R_wc2.T @ (p_wc1 - p_wc2)
E = skew(t) @ Rr
z = proj(pf)
x1 = np.array([z[0], z[1], 1.0])
x2 = np.array([z[2], z[3], 1.0])
g = np.concatenate([(E.T @ x2)[:2], (E @ x1)[:2]])
print("26.3 |g^T H_f| / (|g| |H_f|) =", np.linalg.norm(g @ Hf_num) / np.linalg.norm(g) / np.linalg.norm(Hf_num), " (finite-difference precision)")
Qf, _ = np.linalg.qr(Hf_num, mode="complete")
a = Qf[:, 3]
print("     QR null vector vs g/|g|: |cos| =", abs(a @ g) / np.linalg.norm(g))

# 26.4 Householder vs explicit A
def householder_project(Hf, Hx, r):
    W = np.hstack([Hf, Hx, r[:, None]]).copy()
    mm = W.shape[0]
    for c in range(3):
        x = W[c:, c].copy()
        v = x.copy()
        v[0] += np.copysign(np.linalg.norm(x), x[0])
        beta = 2.0 / (v @ v)
        W[c:, :] -= beta * np.outer(v, v @ W[c:, :])
    return W[3:, 3:-1], W[3:, -1], W[:, :3]


for MM in (3, 6, 10):
    mm, k = 2 * MM, 6 * MM
    Hf2 = rng.normal(size=(mm, 3))
    Hx2 = rng.normal(size=(mm, k))
    r2 = rng.normal(size=mm)
    Ho_h, ro_h, Wf = householder_project(Hf2, Hx2, r2)
    Qc, _ = np.linalg.qr(Hf2, mode="complete")
    Ac = Qc[:, 3:]
    Ho_q, ro_q = Ac.T @ Hx2, Ac.T @ r2
    # SVD-based nullspace basis: different basis
    U, s, Vt = np.linalg.svd(Hf2)
    As = U[:, 3:]
    Ho_s, ro_s = As.T @ Hx2, As.T @ r2
    Pk = np.eye(k)
    gam = lambda H, rr: rr @ np.linalg.solve(H @ Pk @ H.T + np.eye(len(rr)), rr)
    hh = 3 * 4 * mm * (k + 1)
    ex = 2 * mm * mm * 3 + 2 * (mm - 3) * mm * (k + 1)
    print(f"26.4 M={MM}: |Ho_house - Ho_QR| = {np.abs(np.abs(Ho_h) - np.abs(Ho_q)).max():.1e} (up to row signs); info equal (SVD basis): "
          f"{np.allclose(Ho_s.T @ Ho_s, Ho_h.T @ Ho_h)}, gate stats {gam(Ho_h, ro_h):.6f} {gam(Ho_s, ro_s):.6f}; flops house ~{hh}, explicit ~{ex}; rows 4.. of Hf after reflections max {np.abs(Wf[3:]).max():.1e}")

# 26.5 projected noise independence and residual correlation
Mj, Mk = 3, 4
Hf_j, Hf_k = rng.normal(size=(2 * Mj, 3)), rng.normal(size=(2 * Mk, 3))
Aj = np.linalg.qr(Hf_j, mode="complete")[0][:, 3:]
Ak = np.linalg.qr(Hf_k, mode="complete")[0][:, 3:]
Ncov = np.eye(2 * Mj + 2 * Mk) * sig2
T = np.block([[Aj.T, np.zeros((2 * Mj - 3, 2 * Mk))], [np.zeros((2 * Mk - 3, 2 * Mj)), Ak.T]])
C = T @ Ncov @ T.T
print("26.5 projected noise covariance is sigma^2 I:", np.allclose(C, sig2 * np.eye(len(C))))

# ---- 26.2 covariance of the recovered feature, against stable augmented updates

Q1 = Q[:, :3]
Rf = Q1.T @ Hf
Rfi = np.linalg.inv(Rf)
Pff = Rfi @ (Q1.T @ Hx @ P_p @ Hx.T @ Q1 + sig2 * np.eye(3)) @ Rfi.T
for lam in (1e4, 1e6, 1e8):
    Paug = np.block([[P, np.zeros((n, 3))], [np.zeros((3, n)), lam * np.eye(3)]])
    Haug = np.hstack([Hx, Hf])
    S = Haug @ Paug @ Haug.T + sig2 * np.eye(m)
    K = Paug @ Haug.T @ np.linalg.inv(S)
    I = np.eye(n + 3)
    Pj = (I - K @ Haug) @ Paug @ (I - K @ Haug).T + sig2 * K @ K.T
    # information form (exact for any lambda)
    Info = np.linalg.inv(Paug) + Haug.T @ Haug / sig2
    Pinfo = np.linalg.inv(Info)
    print(f"lambda={lam:.0e}: Joseph Pff rel diff {np.abs(Pj[n:,n:]-Pff).max()/np.abs(Pff).max():.2e}; info-form Pff rel diff {np.abs(Pinfo[n:,n:]-Pff).max()/np.abs(Pff).max():.2e}")
print("Pff diag", np.diag(Pff))
