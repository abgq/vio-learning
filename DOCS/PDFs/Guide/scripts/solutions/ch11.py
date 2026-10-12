"""Exercise solutions, Chapter 11: prints the numbers quoted in the worked solutions.

Run from DOCS/PDFs/Guide:  .venv/Scripts/python.exe scripts/solutions/ch11.py
"""
import numpy as np
from scipy.linalg import expm

# 11.2 zero-order-hold input matrix of the double integrator, from the block exponential
dt = 0.1
F = np.array([[0, 1.0], [0, 0]])
B = np.array([[0.0], [1.0]])
M = np.zeros((3, 3))
M[:2, :2], M[:2, 2:] = F, B
E = expm(M * dt)
print("11.2 Gamma =", E[:2, 2], " expected (dt^2/2, dt) =", (dt**2 / 2, dt))

# 11.3 Van Loan
sigma = 1.0
G = np.array([[0.0], [1.0]])
Qc = np.array([[sigma**2]])
A = np.zeros((4, 4))
A[:2, :2], A[:2, 2:], A[2:, 2:] = -F, G @ Qc @ G.T, F.T
C = expm(A * dt)
Phi = C[2:, 2:].T
Qd = Phi @ C[:2, 2:]
exact = sigma**2 * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
print("11.3 Phi =", Phi.tolist())
print("     Qd =", Qd.tolist(), " max |Qd - exact| =", np.abs(Qd - exact).max())

# 11.4 two targets, relative position measurement
Phi2 = np.kron(np.eye(2), np.array([[1, dt], [0, 1.0]]))
H = np.array([[1.0, 0, -1, 0]])
O = np.vstack([H @ np.linalg.matrix_power(Phi2, k) for k in range(4)])
print("11.4 rank O =", np.linalg.matrix_rank(O), " O @ (1,0,1,0) =", O @ [1, 0, 1, 0], " O @ (0,1,0,1) =", O @ [0, 1, 0, 1])

# 11.6 gyro + inclinometer: observability of the bias
F6 = np.array([[0, -1.0], [0, 0]])
H6 = np.array([[1.0, 0]])
print("11.6 O =", np.vstack([H6, H6 @ F6]).tolist(), " rank", np.linalg.matrix_rank(np.vstack([H6, H6 @ F6])))
