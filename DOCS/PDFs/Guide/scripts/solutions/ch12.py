"""Exercise solutions, Chapter 12: prints the numbers quoted in the worked solutions.

Run from DOCS/PDFs/Guide:  .venv/Scripts/python.exe scripts/solutions/<this file>
"""
import numpy as np

# 12.3 search for float32 example
rng = np.random.default_rng(0)
found = None
for trial in range(200000):
    s1 = 10 ** rng.uniform(-2, 3)
    s2 = 10 ** rng.uniform(-2, 3)
    rho = 1 - 10 ** rng.uniform(-6, -1)
    R = 10 ** rng.uniform(-8, 0)
    P = np.array([[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]], dtype=np.float32)
    H = np.array([[1.0, 0.0]], dtype=np.float32)
    S = (H @ P @ H.T + np.float32(R)).astype(np.float32)
    K = (P @ H.T / S).astype(np.float32)
    I = np.eye(2, dtype=np.float32)
    simple = ((I - K @ H) @ P).astype(np.float32)
    simple_sym = 0.5 * (simple + simple.T)
    jos = ((I - K @ H) @ P @ (I - K @ H).T + K @ K.T * np.float32(R)).astype(np.float32)
    jos = 0.5 * (jos + jos.T)
    e1 = np.linalg.eigvalsh(simple_sym.astype(np.float64)).min()
    e2 = np.linalg.eigvalsh(jos.astype(np.float64)).min()
    if e1 < 0 and e2 >= 0:
        found = (s1, s2, rho, R, P, simple_sym, jos, e1, e2)
        break
print("12.3 found:", found is not None)
if found:
    s1, s2, rho, R, P, simple, jos, e1, e2 = found
    print(" P=", P, "\n R=", R, "\n simple(sym)=", simple, "min eig", e1, "\n joseph=", jos, "min eig", e2)
# a clean hand-made example
P = np.array([[1.0, 0.9999], [0.9999, 1.0]], dtype=np.float32)
R = np.float32(1e-7)
H = np.array([[1.0, 0.0]], dtype=np.float32)
S = (H @ P @ H.T + R).astype(np.float32)
K = (P @ H.T / S).astype(np.float32)
I = np.eye(2, dtype=np.float32)
simple = ((I - K @ H) @ P).astype(np.float32)
jos = ((I - K @ H) @ P @ (I - K @ H).T + K @ K.T * R).astype(np.float32)
P64 = P.astype(np.float64)
S64 = P64[0, 0] + 1e-7
K64 = P64[:, :1] / S64
exact = P64 - K64 @ K64.T * S64
print("12.3 hand example simple:\n", simple, np.linalg.eigvalsh(0.5 * (simple + simple.T).astype(float)),
      "\n joseph:\n", jos, np.linalg.eigvalsh(0.5 * (jos + jos.T).astype(float)), "\n exact:\n", exact, np.linalg.eigvalsh(exact))

# 12.4
Pm = 2.429
for k in range(3, 8):
    Pp = Pm * 2 / (Pm + 2)
    Pm = Pp + 1
    print(f"12.4 k={k}: P+_{k-1}={Pp:.4f}, P-_{k}={Pm:.5f}")
# exact from start
Pm = 5.0
for k in range(1, 8):
    Pp = Pm * 2 / (Pm + 2)
    print(f"   exact k={k}: P-={Pm:.5f}  K={Pm/(Pm+2):.4f}  P+={Pp:.5f}")
    Pm = Pp + 1


# 12.6 and 12.7
def run_kf(seed, qscale=1.0, n=200, mc=False):
    rng = np.random.default_rng(seed)
    dt = 0.1
    q, r = 0.05, 1.0
    F = np.array([[1, dt], [0, 1]])
    Qd = q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
    H = np.array([[1.0, 0.0]])
    x = np.array([0.0, 1.0])
    xh = np.zeros(2)
    P = np.diag([4.0, 4.0])
    Lq = np.linalg.cholesky(Qd)
    out, innov = 0, []
    for k in range(n):
        x = F @ x + Lq @ rng.normal(size=2)
        xh = F @ xh
        P = F @ P @ F.T + Qd * qscale
        z = x[0] + np.sqrt(r) * rng.normal()
        S = (H @ P @ H.T)[0, 0] + r
        rr = z - xh[0]
        innov.append(rr / np.sqrt(S))
        K = P[:, 0] / S
        xh = xh + K * rr
        IKH = np.eye(2) - np.outer(K, H[0])
        P = IKH @ P @ IKH.T + r * np.outer(K, K)
        out += abs(xh[0] - x[0]) > 3 * np.sqrt(P[0, 0])
    return out / n, np.array(innov)


frac, _ = run_kf(11)
print("12.6 fraction outside 3 sigma (seed 11):", frac)
fr = [run_kf(s, n=200)[0] for s in range(1000)]
print("     Monte Carlo mean fraction over 1000 runs:", np.mean(fr))


def acf(x, L):
    x = x - x.mean()
    return np.array([np.sum(x[:-l] * x[l:]) / np.sum(x * x) for l in range(1, L + 1)])


for qs in (1.0, 0.01, 100.0):
    _, inn = run_kf(5, qscale=qs, n=5000)
    a = acf(inn[50:], 5)
    print(f"12.7 Q scale {qs}: acf lags 1-5 {np.round(a, 3)}, bound {1.96/np.sqrt(len(inn)-50):.3f}, NIS mean {np.mean(inn[50:]**2):.2f}")
