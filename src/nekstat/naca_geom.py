"""
Python port of the interpolation-mesh generator:
    00_GenProfile/wrt_int_mesh_wing.m  (+ create_profiles_wing_testbudget.m,
     naca_prof2.m, rot.m, save_inter_mesh.m)

Builds the NACA-4412 wall-normal interpolation mesh and writes the Fortran-binary
x.fort / y.fort files that Nek5000 reads, together with the arrays stored in
02_MatLab/int_mesh.mat (x_pts, y_pts, alphau, alphal, yn, xc).
"""

import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import CubicSpline


def _colon(a, step, b):
    """Replicate MATLAB `a:step:b` (inclusive endpoint when it lands on the grid)."""
    n = int(np.floor((b - a) / step + 1e-10)) + 1
    return a + np.arange(n) * step


def naca_prof2(m, p, xx, inputx, tol):
    """Port of naca_prof2.m: surface points (xu,yu),(xl,yl) and wall-normal angles
    alphau, alphal sampled (spline) at the chord stations `inputx`."""
    M = m / 100.0
    P = p / 10.0
    T = xx / 100.0

    n = int(np.floor(np.pi / tol)) + 1          # MATLAB beta = 0:tol:pi
    beta = np.arange(n) * tol
    x = 0.5 * (1 - np.cos(beta))
    del beta

    maskf = x < P
    yc = np.empty_like(x)
    dycdx = np.empty_like(x)
    xf = x[maskf]
    yc[maskf] = M / P ** 2 * (2 * P * xf - xf ** 2)
    dycdx[maskf] = 2 * M / P ** 2 * (P - xf)
    xb = x[~maskf]
    yc[~maskf] = M / (1 - P) ** 2 * (1 - 2 * P + 2 * P * xb - xb ** 2)
    dycdx[~maskf] = 2 * M / (1 - P) ** 2 * (P - xb)
    del xf, xb, maskf
    theta = np.arctan(dycdx)
    del dycdx

    a0, a1, a2, a3, a4 = 0.2969, -0.126, -0.3516, 0.2843, -0.1036
    yt = T / 0.2 * (a0 * np.sqrt(x) + a1 * x + a2 * x ** 2 + a3 * x ** 3 + a4 * x ** 4)

    st, ct = np.sin(theta), np.cos(theta)
    del theta
    xu = x - yt * st
    yu = yc + yt * ct
    xl = x + yt * st
    yl = yc - yt * ct
    del yc, yt, st, ct

    def normal_angle(xs, ys):
        # central-difference surface normal (naca_prof2.m lines 76-90)
        d1y = np.diff(np.concatenate([ys, ys[:1]]))
        d2y = np.diff(np.concatenate([ys[-1:], ys]))
        d1x = np.diff(np.concatenate([xs, xs[:1]]))
        d2x = np.diff(np.concatenate([xs[-1:], xs]))
        nx = (d1y + d2y) / 2.0
        ny = -(d1x + d2x) / 2.0
        return np.arctan(-nx / ny)

    alphau = normal_angle(xu, yu)
    alphal = normal_angle(xl, yl)
    alphau[-1] = alphau[-2]                     # MATLAB line 92 (upper only)

    def sp(y):
        return CubicSpline(x, y, bc_type="not-a-knot")(inputx)

    oxu, oyu = sp(xu), sp(yu)
    oalphau = sp(alphau)
    oxl, oyl = sp(xl), sp(yl)
    oalphal = sp(alphal)
    oalphau[inputx == 0] = np.pi / 2            # LE normal is vertical
    oalphal[inputx == 0] = np.pi / 2
    return oxu, oyu, oalphau, oxl, oyl, oalphal


def rot(xu, yu, xl, yl, yn, alphau, alphal):
    """Port of rot.m — place wall-normal lines along the surface normals."""
    yn = np.asarray(yn).reshape(-1, 1)
    su, cu = np.sin(alphau)[None, :], np.cos(alphau)[None, :]
    sl, cl = np.sin(alphal)[None, :], np.cos(alphal)[None, :]
    xprofu = -su * yn + xu[None, :]
    yprofu = cu * yn + yu[None, :]
    xprofl = sl * yn + xl[None, :]
    yprofl = -cl * yn + yl[None, :]
    return xprofu, yprofu, xprofl, yprofl


def create_profiles_wing(m, p, xx, dx, nle, tol, dy1, yn_thick, ry, xn=0.295):
    """Port of create_profiles_wing_testbudget.m.

    `xn` is the chord location where the leading-edge clustering ends and the uniform
    spacing `dx` begins.  The committed 02_MatLab/int_mesh.mat was generated with
    xn=0.295, dx=0.01625 (the current wrt_int_mesh_wing.m has since been edited to
    xn=0.395, dx=0.015, which yields a different 80-station mesh)."""
    c = 1.0
    x1 = 0.0
    xn1 = xn - dx

    def find_r(r):
        return r ** (1 - nle) * (xn - xn1) * (r ** nle - r) / (r - 1) - xn + x1

    r = brentq(find_r, 1.0001, 2.0)             # matches fzero(...,1.15)
    dx1 = r ** (2 - nle) * (xn - xn1)
    x = np.empty(nle)
    x[0] = x1
    for i in range(2, nle + 1):
        x[i - 1] = x[0] + dx1 * (1 - r ** (i - 1)) / (1 - r)
    x = np.concatenate([x[:-1], _colon(x[-1], dx, c)])

    xu, yu, alphau, xl, yl, alphal = naca_prof2(m, p, xx, x, tol)

    # wall-normal distribution yn
    y1 = 0.0
    n = int(np.ceil(np.log(-(y1 * (ry - 1) + yn_thick * (1 - ry) - dy1) / dy1)
                    / np.log(ry) + 1))
    yn = np.empty(n)
    yn[0] = y1
    for i in range(2, n + 1):
        yn[i - 1] = yn[0] + dy1 * (1 - ry ** (i - 1)) / (1 - ry)

    xprofu, yprofu, xprofl, yprofl = rot(xu, yu, xl, yl, yn, alphau, alphal)

    # rearrange (upper all stations, lower stations 2..end, then wall-normal marker)
    xu_flat = xprofu.ravel(order="F")
    yu_flat = yprofu.ravel(order="F")
    xl_flat = xprofl[:, 1:].ravel(order="F")
    yl_flat = yprofl[:, 1:].ravel(order="F")
    x_pts = np.concatenate([xu_flat, xl_flat, -np.ones(len(yn))])
    y_pts = np.concatenate([yu_flat, yl_flat, yn])
    npoints = len(x_pts)
    return x_pts, y_pts, npoints, yn, alphau, alphal, x


def save_inter_mesh(path, npoints, x_pts, y_pts):
    """Port of save_inter_mesh.m — little-endian Fortran-record x.fort / y.fort."""
    for coord, fn in ((x_pts, "x.fort"), (y_pts, "y.fort")):
        with open(f"{path}/ZSTAT/{fn}", "wb") as f:
            np.array([4], "<i4").tofile(f)                 # eor = len([npoints])*4
            np.array([npoints], "<i4").tofile(f)
            np.array([4], "<i4").tofile(f)
            eor = np.array([npoints * 8], "<i4")
            eor.tofile(f)
            np.asarray(coord, "<f8").tofile(f)
            eor.tofile(f)


def build_mesh(profile_name="NACA4412", tol=0.5e-7):
    """Replicate wrt_int_mesh_wing.m parameter set and build the mesh arrays."""
    if profile_name == "NACA4412":
        m, p, xx = 4, 4, 12
    elif profile_name == "NACA0012":
        m, p, xx = 0, 0, 12
    else:
        raise ValueError(profile_name)

    Rec = 200e3
    nu = 1.0 / Rec
    utz = 0.0424
    dy1 = 0.5 * nu / utz
    yn_thick = 0.12
    ry = 1.02
    nle = 40
    dx = 0.01625          # committed int_mesh.mat spacing (= 0.65/nle)
    xn = 0.295            # committed int_mesh.mat clustering end
    return create_profiles_wing(m, p, xx, dx, nle, tol, dy1, yn_thick, ry, xn)
