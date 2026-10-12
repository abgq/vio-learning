"""Exercise solutions, Chapter 18: prints the numbers quoted in the worked solutions.

Run from DOCS/PDFs/Guide:  .venv/Scripts/python.exe scripts/solutions/<this file>
"""
import numpy as np
from scipy.stats import chi2


def run(qscale, gate=True, T=300.0, seed=4, q=0.5, r=1.0):
    rng = np.random.default_rng(seed)
    dt = 0.1
    n = int(T / dt)
    F = np.array([[1, dt], [0, 1]])
    Qd = q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
    Lq = np.linalg.cholesky(Qd)
    x = np.array([0.0, 1.0])
    xh = np.zeros(2)
    P = np.diag([1.0, 1.0])
    thr = chi2.ppf(0.95, 1)
    rej, err, sig = [], [], []
    for k in range(n):
        x = F @ x + Lq @ rng.normal(size=2)
        xh = F @ xh
        P = F @ P @ F.T + Qd * qscale
        z = x[0] + np.sqrt(r) * rng.normal()
        S = P[0, 0] + r
        rr = z - xh[0]
        reject = gate and rr * rr / S > thr
        rej.append(reject)
        if not reject:
            K = P[:, 0] / S
            xh = xh + K * rr
            IKH = np.eye(2) - np.outer(K, [1, 0])
            P = IKH @ P @ IKH.T + r * np.outer(K, K)
        err.append(x[0] - xh[0])
        sig.append(np.sqrt(P[0, 0]))
    rej = np.array(rej)
    err = np.array(err)
    return rej, err, np.array(sig)


for qs, gate in ((1.0, True), (0.01, True), (0.01, False)):
    rej, err, sig = run(qs, gate)
    w = 300  # 30 s windows
    rates = [rej[i:i + w].mean() for i in range(0, len(rej), w)]
    rms = [np.sqrt(np.mean(err[i:i + w] ** 2)) for i in range(0, len(err), w)]
    print(f"18.3 Q scale {qs}, gate {gate}: rejection rate per 30 s {np.round(rates, 2)}")
    print(f"     rms error per 30 s {np.round(rms, 2)}; final sigma {sig[-1]:.2f}")
