"""Exercise solutions, Chapter 9: prints the numbers quoted in the worked solutions.

Run from DOCS/PDFs/Guide:  .venv/Scripts/python.exe scripts/solutions/<this file>
"""
import numpy as np

# 9.2
for ts in (np.arange(-5, 6.0), np.arange(0, 11.0)):
    H = np.column_stack([np.ones_like(ts), ts])
    C = np.linalg.inv(H.T @ H)
    print("9.2 t from", ts[0], ":", np.round(C, 5), " corr", C[0, 1] / np.sqrt(C[0, 0] * C[1, 1]))

# 9.5 RLS equals batch
rng = np.random.default_rng(2)
n, m = 3, 30
x_true = rng.normal(size=n)
Hs = rng.normal(size=(m, n))
Rv = rng.uniform(0.1, 2.0, m)
z = Hs @ x_true + rng.normal(size=m) * np.sqrt(Rv)
P0 = np.eye(n) * 100.0
x0 = np.zeros(n)
x, P = x0.copy(), P0.copy()
maxdiff = 0
for k in range(m):
    h = Hs[k]
    S = h @ P @ h + Rv[k]
    K = P @ h / S
    x = x + K * (z[k] - h @ x)
    P = P - np.outer(K, h @ P)
    # batch with the same prior
    Hk, zk, Rk = Hs[: k + 1], z[: k + 1], Rv[: k + 1]
    I = np.linalg.inv(P0) + Hk.T @ (Hk / Rk[:, None])
    xb = np.linalg.solve(I, np.linalg.inv(P0) @ x0 + Hk.T @ (zk / Rk))
    maxdiff = max(maxdiff, np.abs(xb - x).max(), np.abs(np.linalg.inv(I) - P).max())
print("9.5 max difference RLS vs batch:", maxdiff)

# 9.6
for sig in (0.01, 0.1):
    K = (1 / 5) / (1 / 7 + sig**2)
    mse_l = 1 / 3 - (1 / 25) / (1 / 7 + sig**2)
    # conditional mean by numerical integration on a grid
    xs = np.linspace(-1, 1, 4001)
    zs = np.linspace(-1.5, 1.5, 3001)
    lik = np.exp(-(zs[:, None] - xs[None, :] ** 3) ** 2 / (2 * sig**2))
    post = lik / lik.sum(axis=1, keepdims=True)
    cm = post @ xs
    var_post = post @ xs**2 - cm**2
    pz = lik.sum(axis=1)
    mse_c = (var_post * pz).sum() / pz.sum()
    print(f"9.6 sigma={sig}: LMMSE gain {K:.3f}, LMMSE rms {np.sqrt(mse_l):.3f}, conditional-mean rms {np.sqrt(mse_c):.3f}")
    for x0_ in (0.1, 0.5, 0.9):
        zz = x0_**3
        i = np.argmin(abs(zs - zz))
        print(f"      x={x0_}: z={zz:.3f}, LMMSE {K*zz:.3f}, cond. mean {cm[i]:.3f}")
