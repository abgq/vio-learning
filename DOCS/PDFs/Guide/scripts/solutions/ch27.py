"""Exercise solutions, Chapter 27 (exercises 1-6, 8, 9; the window study is ch27_window_study.py): prints the numbers quoted in the worked solutions.

Run from DOCS/PDFs/Guide:  .venv/Scripts/python.exe scripts/solutions/<this file>
"""
import numpy as np

OM = np.array([[0.0, -1.0], [1.0, 0.0]])


def rot(psi):
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s], [s, c]])


rng = np.random.default_rng(27)

# 27.1 finite differences
p_i = rng.normal(0, 2, 2)
psi_i = rng.uniform(-np.pi, np.pi)
pf = p_i + rot(psi_i) @ np.array([5.0, 1.5])


def meas(p, psi, f):
    pb = rot(psi).T @ (f - p)
    return pb[1] / pb[0]


pb = rot(psi_i).T @ (pf - p_i)
z = pb[1] / pb[0]
Jpi = np.array([-z, 1.0]) / pb[0]
H_p = -Jpi @ rot(psi_i).T
H_psi = -(1 + z * z)
H_f = Jpi @ rot(psi_i).T
e = 1e-6
num_p = np.array([(meas(p_i + e * u, psi_i, pf) - meas(p_i - e * u, psi_i, pf)) / (2 * e) for u in np.eye(2)])
num_psi = (meas(p_i, psi_i + e, pf) - meas(p_i, psi_i - e, pf)) / (2 * e)
num_f = np.array([(meas(p_i, psi_i, pf + e * u) - meas(p_i, psi_i, pf - e * u)) / (2 * e) for u in np.eye(2)])
print("27.1 max FD errors:", np.abs(num_p - H_p).max(), abs(num_psi - H_psi), np.abs(num_f - H_f).max(), " z =", z)

# 27.2 I + Fc dt vs exact Phi
v, w, dt, psi = 1.0, 0.1, 0.01, 0.3
s = rot(psi + 0.5 * w * dt) @ np.array([v * dt, 0.0])
exact = OM @ s
approx = OM @ rot(psi) @ np.array([1.0, 0.0]) * v * dt
term = -0.5 * v * w * dt**2 * rot(psi) @ np.array([1.0, 0.0])
print("27.2 exact - approx:", exact - approx, " predicted -1/2 v w dt^2 R e1:", term, " norm", np.linalg.norm(exact - approx))


# 27.3 bias observability on the circle trajectory
def obs_matrix(with_features=True, n_clones=10, dt=0.01, cam_every=10, v=1.0, w=0.1):
    # nominal circle
    p, psi = np.array([10.0, 0.0]), np.pi / 2
    Phi_tot = np.eye(4)  # maps (dp0, dpsi0, db) at t=0 to the current robot error
    clones = []
    for k in range(1, n_clones * cam_every + 1):
        sk = rot(psi + 0.5 * w * dt) @ np.array([v * dt, 0.0])
        Phik = np.eye(4)
        Phik[0:2, 2] = OM @ sk
        Phik[0:2, 3] = -0.5 * dt * (OM @ sk)
        Phik[2, 3] = -dt
        Phi_tot = Phik @ Phi_tot
        p = p + sk
        psi = psi + w * dt
        if k % cam_every == 0:
            clones.append((p.copy(), psi, Phi_tot[:3].copy()))
    if not with_features:
        return np.zeros((0, 4))
    lms = np.array([[0.0, 5.0], [-4.0, 8.0], [3.0, 12.0], [-2.0, 3.0], [6.0, 9.0], [0.5, 15.0]])
    rows = []
    for f in lms:
        Hx, Hf, Mats = [], [], []
        for (pc, psc, Mc) in clones:
            pb = rot(psc).T @ (f - pc)
            if pb[0] < 0.5 or abs(pb[1] / pb[0]) > np.tan(np.radians(50)):
                continue
            zz = pb[1] / pb[0]
            J = np.array([-zz, 1.0]) / pb[0]
            Hx.append(np.concatenate([-J @ rot(psc).T, [-(1 + zz * zz)]]) @ Mc)
            Hf.append(J @ rot(psc).T)
        if len(Hx) < 3:
            continue
        Hx, Hf = np.array(Hx), np.array(Hf)
        Q, _ = np.linalg.qr(Hf, mode="complete")
        rows.append(Q[:, 2:].T @ Hx)
    return np.vstack(rows)


O = obs_matrix()
U, sv, Vt = np.linalg.svd(O)
print("27.3 observability matrix rows", O.shape[0], "singular values", sv.round(6))
null = Vt[np.sum(sv > 1e-9 * sv[0]):]
print("     nullspace basis (dp_x, dp_y, dpsi, db):\n", null.round(4))
print("     bias component of nullspace vectors:", np.abs(null[:, 3]).max())

# 27.4 unequal baselines
P = [np.array([0.0, -1.0]), np.array([0.0, 0.0]), np.array([0.0, 2.0])]
f = np.array([4.0, 0.0])
Hf, Hx = [], []
for k, pc in enumerate(P):
    pb = f - pc
    zz = pb[1] / pb[0]
    J = np.array([-zz, 1.0]) / pb[0]
    Hf.append(J)
    row = np.zeros(9)
    row[3 * k:3 * k + 2] = -J
    row[3 * k + 2] = -(1 + zz * zz)
    Hx.append(row)
Hf, Hx = np.array(Hf), np.array(Hx)
a = np.cross(Hf[:, 0], Hf[:, 1])
print("27.4 z =", [(f - pc)[1] / (f - pc)[0] for pc in P], " Hf =", Hf.tolist(), " a =", a, "normalized", a / np.linalg.norm(a), "a*sqrt14 =", a / np.linalg.norm(a) * np.sqrt(14))
Ho = (a / np.linalg.norm(a)) @ Hx
print("     Ho*sqrt(14) =", (Ho * np.sqrt(14)).round(4))
# ratio check: measurements from a configuration where the middle clone is at y2 = d instead of 0
for d in (0.0, 0.1):
    Pm = [np.array([0.0, -1.0]), np.array([0.0, d]), np.array([0.0, 2.0])]
    zz = np.array([(f - pc)[1] / (f - pc)[0] for pc in Pm])
    zh = np.array([(f - pc)[1] / (f - pc)[0] for pc in P])
    r = zz - zh
    print(f"     middle clone truly at y={d}: r_o = {(a/np.linalg.norm(a)) @ r:.5f}; image-displacement ratio (z1-z2)/(z2-z3) = {(zz[0]-zz[1])/(zz[1]-zz[2]):.4f}; baseline ratio estimate 1/2")

# 27.5 from the chapter's worked example
P = [np.array([0.0, -1.0]), np.array([0.0, 0.0]), np.array([0.0, 1.0])]
Hf, Hx = [], []
for k, pc in enumerate(P):
    pb = f - pc
    zz = pb[1] / pb[0]
    J = np.array([-zz, 1.0]) / pb[0]
    Hf.append(J)
    row = np.zeros(9)
    row[3 * k:3 * k + 2] = -J
    row[3 * k + 2] = -(1 + zz * zz)
    Hx.append(row)
Hf, Hx = np.array(Hf), np.array(Hx)
a = np.cross(Hf[:, 0], Hf[:, 1])
a = a / np.linalg.norm(a)
Ho = a @ Hx
print("27.5 Ho*sqrt6 =", (Ho * np.sqrt(6)).round(4), " middle along x:", Ho[3], " first along x:", Ho[0])

from pathlib import Path
# ---- 27.7 how well one window determines a heading ramp

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # DOCS/PDFs/Guide/scripts
import planar_sims as ps  # noqa: E402

landmarks = ps.planar_scenario(0)


def obs_matrix_lm(n_clones, lms, dt=0.01, cam_every=10, v=1.0, w=0.1):
    p, psi = np.array([10.0, 0.0]), np.pi / 2
    Phi_tot = np.eye(4)
    clones = []
    for k in range(1, n_clones * cam_every + 1):
        sk = rot(psi + 0.5 * w * dt) @ np.array([v * dt, 0.0])
        Phik = np.eye(4)
        Phik[0:2, 2] = OM @ sk
        Phik[0:2, 3] = -0.5 * dt * (OM @ sk)
        Phik[2, 3] = -dt
        Phi_tot = Phik @ Phi_tot
        p = p + sk
        psi = psi + w * dt
        if k % cam_every == 0:
            clones.append((p.copy(), psi, Phi_tot[:3].copy()))
    rows = []
    nfeat = 0
    for f in lms:
        Hx, Hf = [], []
        for (pc, psc, Mc) in clones:
            pb = rot(psc).T @ (f - pc)
            if pb[0] < 1.0 or pb[0] > 15.0 or abs(pb[1] / pb[0]) > np.tan(np.radians(50)):
                continue
            zz = pb[1] / pb[0]
            J = np.array([-zz, 1.0]) / pb[0]
            Hx.append(np.concatenate([-J @ rot(psc).T, [-(1 + zz * zz)]]) @ Mc)
            Hf.append(J @ rot(psc).T)
        if len(Hx) < 3:
            continue
        nfeat += 1
        Hx, Hf = np.array(Hx), np.array(Hf)
        Q, _ = np.linalg.qr(Hf, mode="complete")
        rows.append(Q[:, 2:].T @ Hx)
    O = np.vstack(rows)
    return O, nfeat


sig_z = 0.003
for N in (3, 5, 10, 20):
    O, nf = obs_matrix_lm(N, landmarks)
    info_b = (O.T @ O)[3, 3] / sig_z**2
    # information about the bias after removing the unobservable directions = only nonzero singular value^2 / sigma^2
    sv = np.linalg.svd(O, compute_uv=False)
    T = (N - 1) * 0.1
    print(f"N={N}: features used {nf}, sigma(bias) from one window {sig_z/sv[0]:.4f} rad/s; heading-ramp over window {T:.1f} s -> "
          f"equivalent heading-difference std {sig_z/sv[0]*T:.5f} rad; odometry heading-difference std over window {0.02*np.sqrt(T):.4f} rad")
