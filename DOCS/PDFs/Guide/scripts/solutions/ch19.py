"""Exercise solutions, Chapter 19: prints the numbers quoted in the worked solutions.

Run from DOCS/PDFs/Guide:  .venv/Scripts/python.exe scripts/solutions/<this file>
"""
import numpy as np
from scipy.stats import chi2

print("19.1 single-run 95% quantile chi2(15):", chi2.ppf(0.95, 15), " two-sided single-run interval:",
      chi2.ppf(0.025, 15), chi2.ppf(0.975, 15), " 50-run average interval:", chi2.ppf(0.025, 750) / 50, chi2.ppf(0.975, 750) / 50)


# 19.2 NEES / NIS with wrong R
def run(seed, qs=1.0, rs=1.0, T=20.0):
    rng = np.random.default_rng(seed)
    dt = 0.1
    n = int(T / dt)
    q, r = 1.0, 1.0
    F = np.array([[1, dt], [0, 1]])
    Qd = q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
    Lq = np.linalg.cholesky(Qd)
    x = np.array([0.0, 1.0])
    xh = x + rng.normal(size=2)
    P = np.eye(2)
    nees, nis = [], []
    for k in range(n):
        x = F @ x + Lq @ rng.normal(size=2)
        xh = F @ xh
        P = F @ P @ F.T + Qd * qs
        z = x[0] + np.sqrt(r) * rng.normal()
        S = P[0, 0] + r * rs
        rr = z - xh[0]
        nis.append(rr * rr / S)
        K = P[:, 0] / S
        xh = xh + K * rr
        IKH = np.eye(2) - np.outer(K, [1, 0])
        P = IKH @ P @ IKH.T + r * rs * np.outer(K, K)
        e = x - xh
        nees.append(e @ np.linalg.solve(P, e))
    return np.array(nees), np.array(nis)


for qs, rs, lab in ((1, 1, "consistent"), (1 / 20, 1, "Q/20"), (20, 1, "20Q"), (1, 1 / 20, "R/20"), (1, 20, "20R")):
    ne, ni = zip(*[run(s, qs, rs) for s in range(50)])
    ne, ni = np.mean(ne, axis=0), np.mean(ni, axis=0)
    print(f"19.2 {lab:10s}: mean NEES (2 DOF) {ne[20:].mean():6.2f}   mean NIS (1 DOF) {ni[20:].mean():5.2f}")
print("     50-run bands: NEES", chi2.ppf(0.025, 100) / 50, chi2.ppf(0.975, 100) / 50, " NIS", chi2.ppf(0.025, 50) / 50, chi2.ppf(0.975, 50) / 50)


# 19.5 RPE vs ATE for a straight 10-minute flight with constant yaw drift
wd = np.radians(1.0) / 60.0
dt = 0.1
t = np.arange(0, 600 + dt, dt)
p_true = np.column_stack([t, 0 * t])
psi_err = wd * t
v_est = np.column_stack([np.cos(psi_err), np.sin(psi_err)])
p_est = np.vstack([[0, 0], np.cumsum(v_est[:-1] * dt, axis=0)])
# RPE over 10 m (10 s) segments: relative translation expressed in the start frame
k = int(10 / dt)
errs = []
for i in range(0, len(t) - k, k):
    d_true = p_true[i + k] - p_true[i]
    Ri = np.array([[np.cos(psi_err[i]), -np.sin(psi_err[i])], [np.sin(psi_err[i]), np.cos(psi_err[i])]])
    d_est = Ri.T @ (p_est[i + k] - p_est[i])
    errs.append(np.linalg.norm(d_est - d_true))
print("19.5 RPE translation per 10 m segment:", np.mean(errs), "m; rotation per segment", np.degrees(wd * 10), "deg")
end_err = np.linalg.norm(p_est[-1] - p_true[-1])
ate_noalign = np.sqrt(np.mean(np.sum((p_est - p_true) ** 2, axis=1)))
# rigid (2D rotation + translation) alignment
mu, mh = p_true.mean(0), p_est.mean(0)
A, B = p_true - mu, p_est - mh
U, D, Vt = np.linalg.svd(A.T @ B / len(t))
S = np.diag([1, np.sign(np.linalg.det(U @ Vt))])
R = U @ S @ Vt
tt = mu - R @ mh
ate_aligned = np.sqrt(np.mean(np.sum((p_true - (p_est @ R.T + tt)) ** 2, axis=1)))
print(f"19.5 final error {end_err:.1f} m, ATE without alignment {ate_noalign:.1f} m, ATE after rigid alignment {ate_aligned:.2f} m")
