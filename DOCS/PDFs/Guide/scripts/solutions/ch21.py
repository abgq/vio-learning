"""Exercise solutions, Chapter 21: prints the numbers quoted in the worked solutions.

Run from DOCS/PDFs/Guide:  .venv/Scripts/python.exe scripts/solutions/<this file>
"""
import numpy as np
from scipy import ndimage

# 21.2 binomial kernels and their variances
k = np.array([0.5, 0.5])
b = np.array([1.0])
for n in range(1, 7):
    b = np.convolve(b, k)
    x = np.arange(len(b))
    m = (x * b).sum()
    var = ((x - m) ** 2 * b).sum()
    print(f"21.2 n={n}: kernel*2^n = {np.round(b * 2**n).astype(int)}, variance {var:.3f} (n/4 = {n/4})")

# 21.3 Sobel on a ramp
c = 0.7
I = c * np.tile(np.arange(10.0), (10, 1))  # I[row, col] = c*col
Su = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)
corr = ndimage.correlate(I, Su)[5, 5]
conv = ndimage.convolve(I, Su)[5, 5]
print("21.3 correlation response", corr, "=", corr / c, "c ; convolution response", conv / c, "c")

# 21.4 zone plate aliasing vs blur sigma, shrink by 4
n = 256
yy, xx = np.mgrid[0:n, 0:n] - n / 2 + 0.5
zp = 0.5 + 0.5 * np.cos(np.pi * (xx**2 + yy**2) / n)
r = np.hypot(xx, yy)
# local frequency r/n cycles/px; after shrinking by 4 representable up to 1/8 -> r < 32
mask = (r[::4, ::4] > 48) & (r[::4, ::4] < 128)  # region where the true pattern cannot be represented
for s in (0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
    img = ndimage.gaussian_filter(zp, s) if s > 0 else zp
    sub = img[::4, ::4]
    # false pattern amplitude: deviation from mean grey in the unrepresentable region
    rms = np.sqrt(np.mean((sub[mask] - 0.5) ** 2))
    # amplitude kept at r=24 (f = 0.094 cycles/px < 1/8), compare to unblurred amplitude 0.5
    resp8 = np.exp(-2 * np.pi**2 * s**2 * (1 / 8) ** 2)
    resp4 = np.exp(-2 * np.pi**2 * s**2 * (1 / 4) ** 2)
    print(f"21.4 sigma={s}: RMS false pattern in r in (48,128): {rms:.3f} (no blur: {np.sqrt(np.mean((zp[::4,::4][mask]-0.5)**2)):.3f});"
          f" Gaussian response at 1/8: {resp8:.3f}, at 1/4: {resp4:.4f}")

# 21.6 parabola refinement bias on a Gaussian peak
def para(rm, r0, rp):
    return (rm - rp) / (2 * (rm - 2 * r0 + rp))

for sig in (0.7, 1.0, 1.5):
    errs = []
    for d in np.linspace(-0.5, 0.5, 101):
        f = lambda x: np.exp(-(x - d) ** 2 / (2 * sig**2))
        errs.append(para(f(-1), f(0), f(1)) - d)
    errs = np.array(errs)
    print(f"21.6 Gaussian peak sigma={sig}: max |bias| {np.abs(errs).max():.3f} px; at d=0.3: {para(*(np.exp(-(np.array([-1,0,1]) - 0.3)**2/(2*sig**2))))-0.3:.3f}")
