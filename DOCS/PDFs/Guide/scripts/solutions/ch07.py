"""Exercise solutions, Chapter 7: prints the numbers quoted in the worked solutions.

Run from DOCS/PDFs/Guide:  .venv/Scripts/python.exe scripts/solutions/<this file>
"""
import numpy as np


def quat_to_rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def shepperd(R):
    t = np.trace(R)
    cand = np.array([1 + t, 1 + R[0, 0] - R[1, 1] - R[2, 2], 1 - R[0, 0] + R[1, 1] - R[2, 2], 1 - R[0, 0] - R[1, 1] + R[2, 2]])
    i = int(np.argmax(cand))
    s = 2.0 * np.sqrt(cand[i])  # = 4 * (largest component)
    if i == 0:
        q = [s / 4, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif i == 1:
        q = [(R[2, 1] - R[1, 2]) / s, s / 4, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif i == 2:
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, s / 4, (R[1, 2] + R[2, 1]) / s]
    else:
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, s / 4]
    q = np.array(q)
    return q if q[0] >= 0 else -q


def naive(R):
    w = 0.5 * np.sqrt(max(1 + np.trace(R), 0.0))
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (4 * w) if w > 0 else np.full(3, np.nan)
    return np.concatenate([[w], v])


def qdist(a, b):
    return min(np.linalg.norm(a - b), np.linalg.norm(a + b))


axis = np.array([1.0, -2.0, 0.5])
axis /= np.linalg.norm(axis)
for deg in (0, 90, 179.9999, 180):
    th = np.radians(deg)
    q = np.concatenate([[np.cos(th / 2)], np.sin(th / 2) * axis])
    R = quat_to_rot(q)
    with np.errstate(all="ignore"):
        print(f"7.7 {deg:>9}: Shepperd err {qdist(shepperd(R), q):.1e}   naive err {qdist(naive(R), q):.1e}")

# 7.9 optical axis
T_BS = np.array([[0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975],
                 [0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768],
                 [-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949]])
z = T_BS[:3, 2]
print("7.9 optical axis in body:", z, " angle to body z:", np.degrees(np.arccos(z[2])), "deg")
print("    camera x in body:", T_BS[:3, 0], " camera y in body:", T_BS[:3, 1])
