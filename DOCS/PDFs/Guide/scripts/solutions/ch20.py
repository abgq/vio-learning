"""Exercise solutions, Chapter 20: prints the numbers quoted in the worked solutions.

Run from DOCS/PDFs/Guide:  .venv/Scripts/python.exe scripts/solutions/<this file>
"""
import numpy as np

# 20.1 thin lens
f = 4e-3
zo = 0.5
zi = 1 / (1 / f - 1 / zo)
print(f"20.1 z_i = {zi*1e3:.5f} mm, shift = {(zi-f)*1e6:.2f} um, approx f^2/z_o = {f*f/zo*1e6:.2f} um")

# EuRoC cam0
fu, fv, cu, cv = 458.654, 457.296, 367.215, 248.375
k1, k2, p1, p2 = -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05
W, H = 752, 480


def distort(x, y):
    r2 = x * x + y * y
    rad = 1 + k1 * r2 + k2 * r2 * r2
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return xd, yd


def jac(x, y):
    r2 = x * x + y * y
    rad = 1 + k1 * r2 + k2 * r2 * r2
    drad = 2 * k1 + 4 * k2 * r2  # d rad / d(r2) * 2 -> d rad/dx = drad * x
    J = np.array([
        [rad + x * drad * x + 2 * p1 * y + p2 * (2 * x + 4 * x), x * drad * y + 2 * p1 * x + p2 * 2 * y],
        [y * drad * x + p1 * 2 * x + 2 * p2 * y, rad + y * drad * y + p1 * (2 * y + 4 * y) + 2 * p2 * x],
    ])
    return J


# check jac numerically
x0, y0 = 0.4, -0.3
e = 1e-7
Jn = np.column_stack([(np.array(distort(x0 + e, y0)) - np.array(distort(x0 - e, y0))) / (2 * e),
                      (np.array(distort(x0, y0 + e)) - np.array(distort(x0, y0 - e))) / (2 * e)])
print("jac check", np.abs(Jn - jac(x0, y0)).max())


def undistort_newton(xd, yd, iters=50):
    x, y = xd, yd
    for _ in range(iters):
        fx, fy = distort(x, y)
        r = np.array([fx - xd, fy - yd])
        dx = np.linalg.solve(jac(x, y), r)
        x, y = x - dx[0], y - dx[1]
    return x, y


def fixed_point_iters(xd, yd, xt, yt, tol=1e-6, maxit=200):
    x, y = xd, yd
    hist = []
    for j in range(1, maxit + 1):
        r2 = x * x + y * y
        rad = 1 + k1 * r2 + k2 * r2 * r2
        dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        x, y = (xd - dx) / rad, (yd - dy) / rad
        err = np.hypot(x - xt, y - yt)
        hist.append(err)
        if err < tol:
            return j, hist
    return None, hist


def newton_iters(xd, yd, xt, yt, tol=1e-6, maxit=50):
    x, y = xd, yd
    hist = []
    for j in range(1, maxit + 1):
        fx, fy = distort(x, y)
        r = np.array([fx - xd, fy - yd])
        dx = np.linalg.solve(jac(x, y), r)
        x, y = x - dx[0], y - dx[1]
        err = np.hypot(x - xt, y - yt)
        hist.append(err)
        if err < tol:
            return j, hist
    return None, hist


def pix2norm_d(u, v):
    return (u - cu) / fu, (v - cv) / fv


# 20.2 fields of view
def angle_x(x):
    return np.degrees(np.arctan(x))


for edge_lab, uL, uR, vT, vB in (("pixel centers 0..W-1", 0.0, W - 1.0, 0.0, H - 1.0), ("sensor edges -0.5..W-0.5", -0.5, W - 0.5, -0.5, H - 0.5)):
    xl, _ = pix2norm_d(uL, cv)
    xr, _ = pix2norm_d(uR, cv)
    _, yt = pix2norm_d(cu, vT)
    _, yb = pix2norm_d(cu, vB)
    hfov = angle_x(-xl) + angle_x(xr)
    vfov = angle_x(-yt) + angle_x(yb)
    # with distortion: undistort the edge points
    ul, _ = undistort_newton(xl, 0.0)
    ur, _ = undistort_newton(xr, 0.0)
    _, vt = undistort_newton(0.0, yt)
    _, vb = undistort_newton(0.0, yb)
    hfov_d = angle_x(-ul) + angle_x(ur)
    vfov_d = angle_x(-vt) + angle_x(vb)
    print(f"20.2 [{edge_lab}] no distortion: H {hfov:.2f} deg, V {vfov:.2f} deg; with distortion: H {hfov_d:.2f}, V {vfov_d:.2f}")
    print(f"     left/right half-angles no-dist {angle_x(-xl):.2f} {angle_x(xr):.2f}, dist {angle_x(-ul):.2f} {angle_x(ur):.2f}")
    print(f"     top/bottom half-angles no-dist {angle_x(-yt):.2f} {angle_x(yb):.2f}, dist {angle_x(-vt):.2f} {angle_x(vb):.2f}")
print("20.2 symmetric formula 2 atan(W/2fu) =", 2 * angle_x(W / 2 / fu), " 2 atan(H/2fv) =", 2 * angle_x(H / 2 / fv))
# exact undistorted positions at the middle of the edges, along the row/col through the principal point
# diagonal FOV
xd0, yd0 = pix2norm_d(0, 0)
xd1, yd1 = pix2norm_d(W - 1, H - 1)
for (xd, yd) in ((xd0, yd0), (xd1, yd1)):
    x, y = undistort_newton(xd, yd)
    print(f"     corner distorted r={np.hypot(xd,yd):.4f} ({np.degrees(np.arctan(np.hypot(xd,yd))):.2f} deg) undistorted r={np.hypot(x,y):.4f} ({np.degrees(np.arctan(np.hypot(x,y))):.2f} deg)")

# 20.5 iterations at the corners
print("20.5 iteration counts to 1e-6 accuracy (normalized units)")
for (u, v) in ((0, 0), (W - 1, 0), (0, H - 1), (W - 1, H - 1), (cu, 0), (0, cv)):
    xd, yd = pix2norm_d(u, v)
    xt, yt = undistort_newton(xd, yd, 100)
    # check
    chk = np.hypot(*(np.array(distort(xt, yt)) - np.array([xd, yd])))
    nf, hf = fixed_point_iters(xd, yd, xt, yt)
    nn, hn = newton_iters(xd, yd, xt, yt)
    rate = [hf[i + 1] / hf[i] for i in range(min(len(hf) - 1, 6))]
    print(f"  pixel ({u:.0f},{v:.0f}): r_d={np.hypot(xd,yd):.4f}, r={np.hypot(xt,yt):.4f}, residual {chk:.1e}; fixed-point {nf} iters (ratio ~{np.round(rate,3)}), Gauss-Newton {nn} iters, GN errors {['%.1e'%e for e in hn]}")
    print(f"     first fixed-point errors {['%.1e'%e for e in hf[:5]]}")

# Also: what does OpenCV's default (5 iterations) leave at the corner?
xd, yd = pix2norm_d(0, 0)
xt, yt = undistort_newton(xd, yd, 100)
nf, hf = fixed_point_iters(xd, yd, xt, yt, tol=0, maxit=30)
print("  fixed-point error after 5 iterations at corner (0,0):", hf[4], " = ", hf[4] * fu, "px; after 10:", hf[9], "=", hf[9]*fu, "px")

# 20.7 image velocities
Z, v, w = 10.0, 1.0, 1.0
print(f"20.7 translational image velocity v/Z = {v/Z} rad/s = {v/Z*fu:.0f} px/s; rotational ~ w = {w} rad/s = {w*fu:.0f} px/s")

# ---- error left by five fixed-point iterations; worst contraction factor
for (u,v) in ((0,248.375),(751,248.375),(0,0)):
    xd, yd = pix2norm_d(u, v)
    xt, yt = undistort_newton(xd, yd, 100)
    nf, hf = fixed_point_iters(xd, yd, xt, yt, tol=0, maxit=30)
    print(u,v,"after 5:", hf[4], hf[4]*fu, "px")
k1v, k2v = k1, k2
r = np.linspace(0, 1.4, 14001)
rho = np.abs(2*k1v*r**2 + 4*k2v*r**4)/(1 + k1v*r**2 + k2v*r**4)
print("max contraction factor", rho.max(), "at r =", r[rho.argmax()])

# ---- diagonal field of view and defocus blur
def ray(u, v, dist=True):
    xd, yd = pix2norm_d(u, v)
    x, y = undistort_newton(xd, yd) if dist else (xd, yd)
    r = np.array([x, y, 1.0]); return r / np.linalg.norm(r)
for dist in (False, True):
    a = np.degrees(np.arccos(ray(-0.5, -0.5, dist) @ ray(W - 0.5, H - 0.5, dist)))
    b = np.degrees(np.arccos(ray(W - 0.5, -0.5, dist) @ ray(-0.5, H - 0.5, dist)))
    print("diagonal FOV, distortion" if dist else "diagonal FOV, pinhole", a, b)
# blur spot for f/2 at 0.5 m when focused at infinity
fL, zo = 4e-3, 0.5
zi = 1/(1/fL - 1/zo); D = fL/2
print("blur spot diameter", D*(zi - fL)/zi*1e6, "um =", D*(zi-fL)/zi/6e-6, "pixels of 6 um")
