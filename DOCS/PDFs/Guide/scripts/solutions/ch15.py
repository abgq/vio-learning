"""Exercise solutions, Chapter 15: prints the numbers quoted in the worked solutions.

Run from DOCS/PDFs/Guide:  .venv/Scripts/python.exe scripts/solutions/ch15.py
"""
import numpy as np

g = 9.81
# 15.1 accelerometer at rest, tilted 10 deg about x
th = np.radians(10.0)
Rx = np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]])
print("15.1 f = R^T (0, 0, g) =", (Rx.T @ np.array([0, 0, g])).round(3))

# 15.2 static sensitivity of a 5 kHz MEMS accelerometer
wn = 2 * np.pi * 5000
print(f"15.2 1/wn^2 = {1 / wn**2:.2e} m per m/s^2; 1 g deflects {g / wn**2 * 1e9:.1f} nm; "
      f"noise 2e-3 * sqrt(100 Hz) = {2e-3 * 10:.3f} m/s^2 -> {2e-3 * 10 / wn**2:.1e} m")

# 15.4 six-position test
up, down = 9.83, -9.79
print(f"15.4 bias {(up + down) / 2:.3f} m/s^2, 1 + s = {(up - down) / (2 * g):.4f}")

# 15.5 angle random walk conversion
arw = np.radians(1.0) / np.sqrt(3600.0)
sg = 0.2 * arw
print(f"15.5 1 deg/sqrt(h) = {arw:.4e} rad/s/sqrt(Hz); 0.2 deg/sqrt(h) = {sg:.3e}; per-sample std at 400 Hz {sg * np.sqrt(400):.3e} rad/s; "
      f"EuRoC 1.6968e-4 is {1.6968e-4 / sg:.2f} times larger")

# 15.6 scale-factor error 1e-3
s = 1e-3
print(f"15.6 equivalent bias at 1 rad/s: {s:.0e} rad/s = {np.degrees(s):.3f} deg/s; attitude error per full turn 2*pi*s = {2 * np.pi * s * 1e3:.1f} mrad")
