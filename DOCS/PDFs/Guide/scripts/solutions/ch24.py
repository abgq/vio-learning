"""Exercise solutions, Chapter 24: prints the numbers quoted in the worked solutions.

Run from DOCS/PDFs/Guide:  .venv/Scripts/python.exe scripts/solutions/<this file>
"""
import numpy as np

# 24.1 memory and flops
n_slam = 15 + 3 * 200
print("24.1 EKF-SLAM n =", n_slam, " P entries", n_slam**2, " memory", n_slam**2 * 8 / 1e6, "MB")
# per scalar-pair observation, exploiting sparsity of H (18 non-zero columns): P H^T costs 2*n*18*2 flops; covariance update P -= K S K^T costs ~ 2 * n^2 * 2 (rank-2)
m_obs = 100
per_obs = 2 * n_slam * 18 * 2 + 4 * n_slam**2
print("     per observation (rank-2 update, sparse H):", per_obs / 1e6, "Mflop; 100 observations per frame:", m_obs * per_obs / 1e6, "Mflop;",
      "at 20 Hz:", 20 * m_obs * per_obs / 1e9, "Gflop/s")
print("     dense Joseph form per update: ~", 4 * n_slam**3 / 1e6, "Mflop")

n = 15 + 6 * 10
print("     MSCKF n =", n, " P entries", n * n, " memory", n * n * 8 / 1e3, "kB")
d_raw = 50 * (2 * 6 - 3)
print("     stacked rows d =", d_raw)
qr = 2 * d_raw * n**2 - 2 * n**3 / 3   # Householder QR of d x n
d = n
HP = 2 * d * n * n
S = 2 * d * d * n
chol = d**3 / 3
K = 2 * d * d * n
joseph = 4 * n**3 + 2 * n * n * d + 2 * n * d * d
total = qr + HP + S + chol + K + joseph
print(f"     MSCKF: QR compression {qr/1e6:.2f} Mflop, update with d=n={n}: {(HP+S+chol+K+joseph)/1e6:.2f} Mflop, total {total/1e6:.1f} Mflop")
# per-feature nullspace projection: 12 x 3 Householder on 12 x (6*6) clone columns
per_feat = 3 * 2 * 12 * (6 * 6 + 1) * 2
print(f"     per-feature projection ~{per_feat/1e3:.0f} kflop, 50 features {50*per_feat/1e6:.2f} Mflop")

# 24.3 scalar example
s1, q, r = 1.0, 1.0, 1.0
P = np.array([[s1, s1], [s1, s1 + q]])  # (x1, x2)
H = np.array([[-1.0, 1.0]])
S = H @ P @ H.T + r
Kc = P @ H.T / S
Pc = P - Kc @ H @ P
print("24.3 correct posterior covariance (x1,x2):", Pc.tolist())
Kw = (s1 + q) / (s1 + q + r)
claimed = (1 - Kw) * (s1 + q)
actual = s1 + (1 - Kw) ** 2 * q + Kw**2 * r
print(f"     wrong filter: claimed var {claimed:.4f}, actual var {actual:.4f}, optimal {Pc[1,1]:.4f}")
# Monte Carlo
rng = np.random.default_rng(1)
Nmc = 200000
x1 = rng.normal(0, np.sqrt(s1), Nmc)
x1h = np.zeros(Nmc)
w = rng.normal(0, np.sqrt(q), Nmc)
x2 = x1 + w
z = x2 - x1 + rng.normal(0, np.sqrt(r), Nmc)
x2h_wrong = x1h + Kw * (z + x1h - x1h)
print("     MC actual var of wrong filter:", np.var(x2h_wrong - x2))
# repeated: chain of K relative measurements, each anchored to the previous estimate
for steps in (1, 5, 20):
    x = rng.normal(0, 1, Nmc)
    xh = np.zeros(Nmc)
    Pw = 1.0
    for k in range(steps):
        xprev, xhprev = x, xh
        x = xprev + rng.normal(0, np.sqrt(q), Nmc)
        z = x - xprev + rng.normal(0, np.sqrt(r), Nmc)
        Pp = Pw + q
        Kk = Pp / (Pp + r)
        xh = xhprev + Kk * (z + xhprev - xhprev)
        Pw = (1 - Kk) * Pp
    print(f"     after {steps} anchored relative updates: claimed var {Pw:.3f}, actual var {np.var(xh - x):.3f}")
