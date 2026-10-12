"""Exercise solutions, Chapter 25: prints the numbers quoted in the worked solutions.

Run from DOCS/PDFs/Guide:  .venv/Scripts/python.exe scripts/solutions/<this file>
"""
import numpy as np


def kf_update(x, P, H, R, z):
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    I = np.eye(len(x))
    return x + K @ (z - H @ x), (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T


# 25.1
P0, Q, R = 1.0, 1.0, 1.0
Paug = np.array([[P0 + Q, P0], [P0, P0]])
H = np.array([[1.0, -1.0]])
_, Pp = kf_update(np.zeros(2), Paug, H, np.array([[R]]), np.zeros(1))
print("25.1 posterior (x2, xc):", Pp.round(4).tolist(), " formula P+Q-Q^2/(Q+R) =", P0 + Q - Q**2 / (Q + R))
# absolute measurement of x2 after cloning, then relative
_, Pa = kf_update(np.zeros(2), Paug, np.array([[1.0, 0.0]]), np.array([[1.0]]), np.zeros(1))
S = H @ Pa @ H.T + R
K = Pa @ H.T / S
print("     after absolute fix of x2: P =", Pa.round(4).tolist(), " relative-measurement gain", K.ravel().round(4))
_, Pb = kf_update(np.zeros(2), Pa, H, np.array([[R]]), np.zeros(1))
print("     clone variance after relative update:", Pb[1, 1].round(4))

# 25.2 two clones: state (x3, xc1, xc2); xc1 cloned at t1, xc2 at t2, current x3 at t3
# build covariance: at t1 var P0; clone; propagate to t2 (+Q); clone; propagate to t3 (+Q)
P1 = np.array([[P0]])
# state order (x, xc1)
Pt = np.array([[P0, P0], [P0, P0]])
Pt[0, 0] += Q
# clone x -> xc2 : order (x, xc1, xc2)
J = np.array([[1, 0], [0, 1], [1, 0.0]])
Pt = J @ Pt @ J.T
Pt[0, 0] += Q
print("25.2 prior covariance (x3, xc1, xc2):\n", Pt)
Ha = np.array([[0, -1, 1.0]])   # xc2 - xc1
Hb = np.array([[1, 0, -1.0]])   # x3 - xc2
Hs = np.vstack([Ha, Hb])
Rs = np.eye(2) * R
_, P_batch = kf_update(np.zeros(3), Pt, Hs, Rs, np.zeros(2))
_, P_s1 = kf_update(np.zeros(3), Pt, Ha, np.array([[R]]), np.zeros(1))
_, P_seq = kf_update(np.zeros(3), P_s1, Hb, np.array([[R]]), np.zeros(1))
print("     chain, batch == sequential:", np.allclose(P_batch, P_seq))
print("     posterior (chain):\n", P_batch.round(4))
# anchored re-expression: z1' = x3 - xc1 = za + zb, z2' = xc2 - xc1 = za ; T = [[1,1],[1,0]]
T = np.array([[1, 1], [1, 0.0]])
Hp = T @ Hs
Rp = T @ Rs @ T.T
_, P_anch = kf_update(np.zeros(3), Pt, Hp, Rp, np.zeros(2))
print("     anchored with correlated noise == chain:", np.allclose(P_anch, P_batch), " Rp =", Rp.tolist())
# sequential anchored with correlation: process z1', then z2' conditioned on z1' (decorrelate)
_, P_a1 = kf_update(np.zeros(3), Pt, Hp[:1], Rp[:1, :1], np.zeros(1))
# decorrelated second measurement: z2'' = z2' - (Rp21/Rp11) z1'  -> H'' = H2 - c H1, R'' = R22 - c R12
c = Rp[1, 0] / Rp[0, 0]
H2 = Hp[1:] - c * Hp[:1]
# but z1' already used: after decorrelation the second is independent of first's noise; however the state correlation is in P_a1
R2 = np.array([[Rp[1, 1] - c * Rp[0, 1]]])
_, P_a2 = kf_update(np.zeros(3), P_a1, H2, R2, np.zeros(1))
print("     anchored sequential after decorrelation == chain:", np.allclose(P_a2, P_batch), " H''=", H2.round(3), "R''=", R2)
# dropping the noise correlation
_, P_drop = kf_update(np.zeros(3), Pt, Hp, np.diag(np.diag(Rp)), np.zeros(2))
print("     anchored, correlation dropped: posterior variances", np.diag(P_drop).round(4), "vs correct", np.diag(P_batch).round(4))
# info matrices
print("     info about relative displacements (H^T R^-1 H) chain vs dropped:\n", (Hs.T @ np.linalg.inv(Rs) @ Hs).round(3), "\n", (Hp.T @ np.linalg.inv(np.diag(np.diag(Rp))) @ Hp).round(3))

# 25.3 verify in-place deletion order on column-major storage
rng = np.random.default_rng(3)
nmax, n, o = 90, 75, 15 + 6 * 4
A = rng.normal(size=(n, n))
A = A @ A.T
buf = np.zeros(nmax * nmax)  # column-major: element (i,j) at j*nmax + i
for j in range(n):
    buf[j * nmax:j * nmax + n] = A[:, j]
# shift columns left by 6 (ascending)
for j in range(o, n - 6):
    buf[j * nmax:j * nmax + n] = buf[(j + 6) * nmax:(j + 6) * nmax + n]
# shift rows up by 6 within each column, element by element ascending
for j in range(n - 6):
    base = j * nmax
    for i in range(o, n - 6):
        buf[base + i] = buf[base + i + 6]
m = n - 6
B = np.array([[buf[j * nmax + i] for j in range(m)] for i in range(m)])
keep = [i for i in range(n) if not (o <= i < o + 6)]
print("25.3 in-place result equals reference:", np.array_equal(B, A[np.ix_(keep, keep)]))
# descending order would corrupt
buf2 = np.zeros(nmax * nmax)
for j in range(n):
    buf2[j * nmax:j * nmax + n] = A[:, j]
for j in range(o, n - 6):
    buf2[j * nmax:j * nmax + n] = buf2[(j + 6) * nmax:(j + 6) * nmax + n]
for j in range(n - 6):
    base = j * nmax
    for i in reversed(range(o, n - 6)):
        buf2[base + i] = buf2[base + i + 6]
B2 = np.array([[buf2[j * nmax + i] for j in range(m)] for i in range(m)])
print("     descending row shift equals reference?", np.array_equal(B2, A[np.ix_(keep, keep)]))


# 25.4 flops
def flops(n, d, compress):
    out = {}
    if compress and d > n:
        out["QR"] = 2 * (n + 1) ** 2 * (d - (n + 1) / 3)
        d = n
    out["HP"] = 2 * d * n * n
    out["HPHt"] = 2 * d * d * n
    out["chol"] = d**3 / 3
    out["K"] = 2 * d * d * n
    out["KH"] = 2 * n * n * d
    out["joseph"] = 4 * n**3
    out["KRKt"] = 2 * n * n * d
    return sum(out.values()), out


d = 50 * (12 - 3)
proj = 50 * (3 * 2 * 12 * (36 + 1) * 2)
for N in (10, 20, 30):
    n = 15 + 6 * N
    a, _ = flops(n, d, False)
    b, _ = flops(n, d, True)
    print(f"25.4 N={N} n={n} d={d}: without QR {a/1e6:.1f} Mflop, with QR {b/1e6:.1f} Mflop (projection {proj/1e6:.2f} Mflop)")
