"""
Ports of 02_MatLab/matlab_func/diagnostic.m and trapezoidal2.m,
plus a MATLAB-style scalar linear interpolation (interp1, 'linear').
"""

import numpy as np


def interp1_lin(x, v, xq):
    """MATLAB interp1(x, v, xq, 'linear'): x monotonic (increasing or decreasing);
    returns NaN if xq is outside the sample range."""
    x = np.asarray(x, float)
    v = np.asarray(v, float)
    if x[0] > x[-1]:                      # decreasing -> flip to increasing
        x = x[::-1]
        v = v[::-1]
    if xq < x[0] or xq > x[-1]:
        return np.nan
    k = int(np.searchsorted(x, xq))
    if k <= 0:
        k = 1
    if k >= len(x):
        k = len(x) - 1
    x0, x1, v0, v1 = x[k - 1], x[k], v[k - 1], v[k]
    if x1 == x0:
        return v0
    return v0 + (v1 - v0) * (xq - x0) / (x1 - x0)


def diagnostic(y, U, u):
    """Boundary-layer diagnostic (returns d99, Uinf, i99, H12, d1, d2).

    `i99` is the MATLAB 1-based index into the *wall-trimmed* profile, exactly as
    returned by diagnostic.m; load_prof_data then indexes the full-length arrays
    with it (reproducing the original one-point offset in Ue etc.)."""
    y = np.asarray(y, float).ravel()
    U = np.asarray(U, float).ravel()
    u = np.asarray(u, float).ravel()
    if y[0] == 0:
        y = y[1:]
        U = U[1:]
        u = u[1:]
    uU = np.abs(u / U)
    lev = 0.02

    # h == 1 : first estimate of H12 from full profile
    Uinf = U.max()
    y0 = np.concatenate(([0.0], y))
    U0 = np.concatenate(([0.0], U))
    d1 = np.trapz(1 - U0 / Uinf, y0)
    d2 = np.trapz((U0 / Uinf) * (1 - U0 / Uinf), y0)
    H12 = d1 / d2

    # h == 2 : locate edge and recompute integral quantities
    ratio = uU / np.sqrt(H12)
    below = np.where(ratio < lev)[0]           # 0-based
    i99_0 = below[0]
    if (i99_0 + 1) < 5:                         # MATLAB: if i99 < 5 -> buff(2)
        i99_0 = below[1]
    i99 = i99_0 + 1                             # 1-based

    xs = ratio[:i99 + 1]                        # uU(1:i99+1)/sqrt(H12)
    Uinf = interp1_lin(xs, U[:i99 + 1], lev) / 0.99
    d99 = interp1_lin(xs, y[:i99 + 1], lev)

    yin = np.concatenate(([0.0], y[:i99]))     # [0; y(1:i99)]
    Uin = np.concatenate(([0.0], U[:i99]))     # [0; U(1:i99)]
    d1 = np.trapz(1 - Uin / Uinf, yin)
    d2 = np.trapz((Uin / Uinf) * (1 - Uin / Uinf), yin)
    H12 = d1 / d2
    return d99, Uinf, i99, H12, d1, d2


def trapezoidal2(x, y, v, ang, ind):
    """Port of trapezoidal2.m (surface-integration quadrature used for Cl/Cd)."""
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    v = np.asarray(v, float).ravel()
    ang = np.asarray(ang, float).ravel()
    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    dv = (v[1:] + v[:-1]) / 2.0
    dang = (ang[1:] + ang[:-1]) / 2.0
    if ind == 1:
        return np.sum(dx * dv)
    elif ind == 2:
        return np.sum(dx * dv * np.tan(dang))
    elif ind == 3:
        return np.sum(dy * dv)
    elif ind == 4:
        return np.sum(dy * dv / np.tan(dang))
    elif ind == 5:
        return np.sum(dx * dv * (dx / 2 + x[:-1]))
    elif ind == 6:
        return np.sum(dx * dv * (dy / 2 + y[:-1]))
    elif ind == 7:
        return np.sum(dx * dv * np.tan(dang) * (dx / 2 + x[:-1]))
    else:
        return np.sum(dx * dv * np.tan(dang) * (dy / 2 + y[:-1]))
