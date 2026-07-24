"""
Python port of 02_MatLab/matlab_func/read_data_profiles.m

Reads the interpolated Nek5000 statistics binary ZSTAT/int_fld together with the
interpolation-mesh definition (int_mesh.mat) and produces two lists of per-profile
dictionaries (`top`, `bottom`), one entry per wall-normal profile, with every field
rotated into the wall-tangent / wall-normal frame exactly as the MATLAB code does.

Indexing note: the MATLAB source is 1-based; here `top[p]`/`bottom[p]` use 0-based p
(p=0 <-> MATLAB profile 1).  The field arrays are the (ln, 2*np) matrices F1..F66,
D1..D68 exactly as in read_data_profiles.m.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Symmetric-tensor component map: struct-suffix -> (i, j) 0-based
# ---------------------------------------------------------------------------
SYM = {"xx": (0, 0), "yy": (1, 1), "zz": (2, 2),
       "xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}


def read_header(fid):
    """Replicate the header read in read_data_profiles.m; leaves fid positioned
    exactly at the first field record."""
    hdr = np.fromfile(fid, np.int32, 1)[0]
    F = fid.read(hdr).decode("latin1")
    _dum5 = np.fromfile(fid, np.float64, 1)[0]
    Rer = np.fromfile(fid, np.float64, 1)[0]
    Domain = np.fromfile(fid, np.float64, 3)
    nel = np.fromfile(fid, np.int32, 3)
    Poly = np.fromfile(fid, np.int32, 3)
    nstat = int(np.fromfile(fid, np.int32, 1)[0])
    nderiv = int(np.fromfile(fid, np.int32, 1)[0])
    _times = np.fromfile(fid, np.float64, 1)[0]
    _timee = np.fromfile(fid, np.float64, 1)[0]
    _atime = np.fromfile(fid, np.float64, 1)[0]
    _DT = np.fromfile(fid, np.float64, 1)[0]
    _nrec = int(np.fromfile(fid, np.int32, 1)[0])
    _Tint = np.fromfile(fid, np.float64, 1)[0]
    npoints = int(np.fromfile(fid, np.int32, 1)[0])
    _dum6 = np.fromfile(fid, np.float64, 1)[0]
    return dict(F=F, Rer=Rer, Domain=Domain, nel=nel, Poly=Poly,
                nstat=nstat, nderiv=nderiv, npoints=npoints)


def read_fields(path, x_pts):
    """Read int_fld and return dicts F (1..nstat) and D (1..nderiv), each value an
    (ln, 2*np) float64 array, following the exact column-major reshape and the
    8-byte inter-record skip of read_data_profiles.m."""
    npoints_total = len(x_pts)
    inw = np.where(x_pts == -1)[0]
    ln = len(inw)
    two_np = npoints_total // ln          # = 2*np
    fname = f"{path}/ZSTAT/int_fld"
    with open(fname, "rb") as fid:
        hd = read_header(fid)
        nstat, nderiv = hd["nstat"], hd["nderiv"]
        npoints = hd["npoints"]
        assert npoints == npoints_total, (npoints, npoints_total)

        F, D = {}, {}
        for ii in range(1, nstat + nderiv + 1):
            if ii != 1:
                fid.seek(8, 1)            # skip 8-byte record separator
            matr = np.fromfile(fid, np.float64, npoints)
            # MATLAB: Mat(:,i) = MatR(ln*(i-1)+1 : ln*(i-1)+ln)  (column-major)
            mat = matr.reshape(two_np, ln).T          # (ln, 2*np)
            if ii <= nstat:
                F[ii] = mat
            else:
                D[ii - nstat] = mat
    return F, D, hd, ln, two_np // 2


def _R_top(alpha):
    """Suction-side rotation matrices, shape (P,3,3)."""
    c, s = np.cos(alpha), np.sin(alpha)
    z, o = np.zeros_like(c), np.ones_like(c)
    return np.stack([np.stack([c, s, z], -1),
                     np.stack([-s, c, z], -1),
                     np.stack([z, z, o], -1)], -2)


def _R_bot(alpha):
    """Pressure-side rotation matrices (a reflection), shape (P,3,3)."""
    c, s = np.cos(alpha), np.sin(alpha)
    z, o = np.zeros_like(c), np.ones_like(c)
    return np.stack([np.stack([c, s, z], -1),
                     np.stack([s, -c, z], -1),
                     np.stack([z, z, o], -1)], -2)


def _rot_vec(R, comps):
    """R:(P,3,3), comps:(3,ln,P) -> (3,ln,P)."""
    return np.einsum("pac,crp->arp", R, comps, optimize=True)


def _rot_ten(R, M):
    """R:(P,3,3), M:(3,3,ln,P) -> (3,3,ln,P)  (R M R^T)."""
    return np.einsum("pac,pbd,cdrp->abrp", R, R, M, optimize=True)


def _rot_ten3(R, A):
    """R:(P,3,3), A:(3,3,3,ln,P) -> (3,3,3,ln,P)."""
    return np.einsum("pai,pbj,pck,ijkrp->abcrp", R, R, R, A, optimize=True)


def read_data_profiles(path, mesh, long=1):
    """Full port of read_data_profiles.m.

    Parameters
    ----------
    path : str        directory containing ZSTAT/int_fld  (e.g. '../01_NEK')
    mesh : dict        with keys x_pts,y_pts,alphau,alphal,yn,xc (from int_mesh.mat)
    long : int         1 -> also compute the turbulent-budget tensors

    Returns
    -------
    top, bottom : list[dict]   length np each (0-based)
    """
    x_pts = np.asarray(mesh["x_pts"], float).ravel()
    y_pts = np.asarray(mesh["y_pts"], float).ravel()
    alphau = np.asarray(mesh["alphau"], float).ravel()
    alphal = np.asarray(mesh["alphal"], float).ravel()
    xc = np.asarray(mesh["xc"], float).ravel()

    F, D, hd, ln, npv = read_fields(path, x_pts)
    two_np = 2 * npv
    npf = npv                                     # 'np' in MATLAB (# profiles/side)

    inw = np.where(x_pts == -1)[0]
    yn = y_pts[inw].reshape(-1, 1)                # (ln,1) wall-normal coordinate

    Reb = hd["Rer"]
    rho = 1.0
    nu = 1.0 / Reb
    mu = rho * nu

    # ---- geometry / coordinates (read_data_profiles.m lines 24-57) ----------
    top = [dict() for _ in range(npf)]
    bottom = [dict() for _ in range(npf)]
    for i in range(npf):                          # MATLAB i=1..np
        blk = slice(ln * i, ln * i + ln)
        top[i]["xa"] = x_pts[ln * i]
        top[i]["ya"] = y_pts[ln * i]
        top[i]["xc"] = xc[i]
        top[i]["x"] = x_pts[blk].reshape(-1, 1)
        top[i]["y"] = y_pts[blk].reshape(-1, 1)
    bottom[0]["xa"] = x_pts[0]
    bottom[0]["ya"] = y_pts[0]
    bottom[0]["x"] = top[0]["x"].copy()
    bottom[0]["y"] = top[0]["y"].copy()
    for i in range(npf - 1):                      # MATLAB i=1..np-1 -> bottom(i+1)
        base = ln * i + npf * ln
        blk = slice(base, base + ln)
        bottom[i + 1]["xa"] = x_pts[base]
        bottom[i + 1]["ya"] = y_pts[base]
        bottom[i + 1]["xc"] = xc[i]
        bottom[i + 1]["x"] = x_pts[blk].reshape(-1, 1)
        bottom[i + 1]["y"] = y_pts[blk].reshape(-1, 1)
    for i in range(npf):
        top[i]["alpha"] = alphau[i]
        bottom[i]["alpha"] = alphal[i]
        top[i]["yn"] = yn.copy()
        bottom[i]["yn"] = yn.copy()
        for s in (top[i], bottom[i]):
            s["mu"], s["nu"], s["rho"] = mu, nu, rho

    # ---- column -> profile maps (0-based) -----------------------------------
    top_cols = np.arange(0, npf)                          # cols 0..np-1
    bot_cols = np.concatenate(([0], np.arange(npf, 2 * npf - 1)))  # [0, np..2np-2]
    R_top = _R_top(alphau)
    R_bot = _R_bot(alphal)

    def col(fld, side):
        return fld[:, top_cols] if side == "t" else fld[:, bot_cols]

    def scatter_scalar(name, arr_top, arr_bot):
        for p in range(npf):
            top[p][name] = arr_top[:, p].reshape(-1, 1)
            bottom[p][name] = arr_bot[:, p].reshape(-1, 1)

    def scatter_vec(names, vr_top, vr_bot):
        for k, name in enumerate(names):
            for p in range(npf):
                top[p][name] = vr_top[k, :, p].reshape(-1, 1)
                bottom[p][name] = vr_bot[k, :, p].reshape(-1, 1)

    def scatter_sym(prefix, tr_top, tr_bot, factor=1.0):
        for suf, (a, b) in SYM.items():
            for p in range(npf):
                top[p][prefix + suf] = (factor * tr_top[a, b, :, p]).reshape(-1, 1)
                bottom[p][prefix + suf] = (factor * tr_bot[a, b, :, p]).reshape(-1, 1)

    # ---- raw (unrotated) full fields ---------------------------------------
    U, V, W = F[1], F[2], F[3]
    P = F[4]
    dUdx, dVdx, dWdx = D[1], D[3], D[5]
    dUdy, dVdy, dWdy = D[2], D[4], D[6]
    Z = np.zeros_like(U)
    dUdz = dVdz = dWdz = Z

    uu = F[5] - U * U
    vv = F[6] - V * V
    ww = F[7] - W * W
    uv = F[9] - U * V
    uw = F[11] - U * W
    vw = F[10] - V * W

    # ---- mean pressure & fluctuation stats (lines 320-542) ------------------
    pinf = P.max() - 0.5
    pp = F[8] - P * P
    prms = np.sqrt(pp)
    scatter_scalar("P", col(P, "t"), col(P, "b"))
    scatter_scalar("pp", col(pp, "t"), col(pp, "b"))
    scatter_scalar("prms", col(prms, "t"), col(prms, "b"))
    for p in range(npf):
        top[p]["pinf"] = pinf
        bottom[p]["pinf"] = pinf

    # ---- mean velocity (rank-1) --------------------------------------------
    Vc = np.stack([U, V, W])
    scatter_vec(["U", "V", "W"], _rot_vec(R_top, Vc[:, :, top_cols]),
                _rot_vec(R_bot, Vc[:, :, bot_cols]))

    # ---- pressure gradient (rank-1): [D7, D8, 0] ---------------------------
    Pg = np.stack([D[7], D[8], Z])
    scatter_vec(["dPdx", "dPdy", "dPdz"], _rot_vec(R_top, Pg[:, :, top_cols]),
                _rot_vec(R_bot, Pg[:, :, bot_cols]))

    # ---- Reynolds-stress tensor (rank-2) -----------------------------------
    Rey = np.array([[uu, uv, uw], [uv, vv, vw], [uw, vw, ww]])
    scatter_sym("", _rot_ten(R_top, Rey[:, :, :, top_cols]),
                _rot_ten(R_bot, Rey[:, :, :, bot_cols]))
    # rename to uu/vv/... : SYM prefix "" already gives keys 'xx'.. so fix names
    for p in range(npf):
        for src, dst in (("xx", "uu"), ("yy", "vv"), ("zz", "ww"),
                         ("xy", "uv"), ("xz", "uw"), ("yz", "vw")):
            top[p][dst] = top[p].pop(src)
            bottom[p][dst] = bottom[p].pop(src)

    # ---- velocity-gradient tensor (rank-2, asymmetric) ---------------------
    G = np.array([[dUdx, dUdy, dUdz],
                  [dVdx, dVdy, dVdz],
                  [dWdx, dWdy, dWdz]])
    Gt = _rot_ten(R_top, G[:, :, :, top_cols])
    Gb = _rot_ten(R_bot, G[:, :, :, bot_cols])
    grad_map = {"dUdx": (0, 0), "dUdy": (0, 1), "dVdx": (1, 0),
                "dVdy": (1, 1), "dWdx": (2, 0), "dWdy": (2, 1)}
    for name, (a, b) in grad_map.items():
        for p in range(npf):
            top[p][name] = Gt[a, b, :, p].reshape(-1, 1)
            bottom[p][name] = Gb[a, b, :, p].reshape(-1, 1)
    scatter_scalar("dUdxnotR", col(dUdx, "t"), col(dUdx, "b"))
    scatter_scalar("dVdynotR", col(dVdy, "t"), col(dVdy, "b"))

    # ---- raw stress derivatives (lines 630-649) ----------------------------
    duudx = D[9] - 2 * U * dUdx
    dvvdx = D[11] - 2 * V * dVdx
    dwwdx = D[13] - 2 * W * dWdx
    duvdx = D[17] - U * dVdx - V * dUdx
    duwdx = D[21] - U * dWdx - W * dUdx
    dvwdx = D[19] - V * dWdx - W * dVdx
    duudy = D[10] - 2 * U * dUdy
    dvvdy = D[12] - 2 * V * dVdy
    dwwdy = D[14] - 2 * W * dWdy
    duvdy = D[18] - U * dVdy - V * dUdy
    duwdy = D[22] - U * dWdy - W * dUdy
    dvwdy = D[20] - V * dWdy - W * dVdy
    duudz = dvvdz = dwwdz = duvdz = duwdz = dvwdz = Z

    if long == 1:
        _budget(top, bottom, npf, top_cols, bot_cols, R_top, R_bot,
                F, D, U, V, W, P, uu, vv, ww, uv, uw, vw,
                dUdx, dUdy, dVdx, dVdy, dWdx, dWdy, dUdz, dVdz, dWdz,
                duudx, dvvdx, dwwdx, duvdx, duwdx, dvwdx,
                duudy, dvvdy, dwwdy, duvdy, duwdy, dvwdy,
                nu, rho, ln)

    return top, bottom


def _budget(top, bottom, npf, top_cols, bot_cols, R_top, R_bot,
            F, D, U, V, W, P, uu, vv, ww, uv, uw, vw,
            dUdx, dUdy, dVdx, dVdy, dWdx, dWdy, dUdz, dVdz, dWdz,
            duudx, dvvdx, dwwdx, duvdx, duwdx, dvwdx,
            duudy, dvvdy, dwwdy, duvdy, duwdy, dvwdy,
            nu, rho, ln):
    """Turbulent-budget tensors (read_data_profiles.m long==1 block)."""
    Z = np.zeros_like(U)
    duudz = dvvdz = dwwdz = duvdz = duwdz = dvwdz = Z

    def scatter_sym(prefix, tr_t, tr_b, factor=1.0):
        for suf, (a, b) in SYM.items():
            for p in range(npf):
                top[p][prefix + suf] = (factor * tr_t[a, b, :, p]).reshape(-1, 1)
                bottom[p][prefix + suf] = (factor * tr_b[a, b, :, p]).reshape(-1, 1)

    # Production  P_ij = -(<uiuk>dUj/dxk + <ujuk>dUi/dxk)
    Pxx = -2 * (uu * dUdx + uv * dUdy)
    Pxy = -(uu * dVdx + uv * dVdy + uv * dUdx + vv * dUdy)
    Pxz = -(uu * dWdx + uv * dWdy + uw * dUdx + vw * dUdy)
    Pyy = -2 * (uv * dVdx + vv * dVdy)
    Pyz = -(uv * dWdx + vv * dWdy + uw * dVdx + vw * dVdy)
    Pzz = -2 * (uw * dWdx + vw * dWdy)
    Prod = np.array([[Pxx, Pxy, Pxz], [Pxy, Pyy, Pyz], [Pxz, Pyz, Pzz]])
    scatter_sym("P", _rot_ten(R_top, Prod[:, :, :, top_cols]),
                _rot_ten(R_bot, Prod[:, :, :, bot_cols]))

    # Dissipation  D_ij = -2 nu <dui/dxk duj/dxk>
    dU = np.array([[dUdx, dUdy, dUdz], [dVdx, dVdy, dVdz], [dWdx, dWdy, dWdz]])
    e11 = F[42] - dU[0, 0] ** 2 - dU[0, 1] ** 2
    e22 = F[43] - dU[1, 0] ** 2 - dU[1, 1] ** 2
    e33 = F[44] - dU[2, 0] ** 2 - dU[2, 1] ** 2
    e12 = F[45] - dU[0, 0] * dU[1, 0] - dU[0, 1] * dU[1, 1]
    e13 = F[46] - dU[0, 0] * dU[2, 0] - dU[0, 1] * dU[2, 1]
    e23 = F[47] - dU[1, 0] * dU[2, 0] - dU[1, 1] * dU[2, 1]
    E = np.array([[e11, e12, e13], [e12, e22, e23], [e13, e23, e33]])
    scatter_sym("D", _rot_ten(R_top, E[:, :, :, top_cols]),
                _rot_ten(R_bot, E[:, :, :, bot_cols]), factor=-2 * nu)

    # Forcing  (rank-1)
    Fc = np.array([F[61], F[62], F[63]])
    ft = _rot_vec(R_top, Fc[:, :, top_cols]); fb = _rot_vec(R_bot, Fc[:, :, bot_cols])
    for k, name in enumerate(["Forcingx", "Forcingy", "Forcingz"]):
        for p in range(npf):
            top[p][name] = ft[k, :, p].reshape(-1, 1)
            bottom[p][name] = fb[k, :, p].reshape(-1, 1)

    # RT-filter dissipation (rank-1)
    Dr = np.array([F[64], F[65], F[66]])
    dt = _rot_vec(R_top, Dr[:, :, top_cols]); db = _rot_vec(R_bot, Dr[:, :, bot_cols])
    for k, name in enumerate(["DRTx", "DRTy", "DRTz"]):
        for p in range(npf):
            top[p][name] = dt[k, :, p].reshape(-1, 1)
            bottom[p][name] = db[k, :, p].reshape(-1, 1)

    # Mean convection  C_ij = U d<uiuj>/dx + V d<uiuj>/dy
    Cxx = U * duudx + V * duudy
    Cyy = U * dvvdx + V * dvvdy
    Czz = U * dwwdx + V * dwwdy
    Cxy = U * duvdx + V * duvdy
    Cxz = U * duwdx + V * duwdy
    Cyz = U * dvwdx + V * dvwdy
    Cm = np.array([[Cxx, Cxy, Cxz], [Cxy, Cyy, Cyz], [Cxz, Cyz, Czz]])
    scatter_sym("C", _rot_ten(R_top, Cm[:, :, :, top_cols]),
                _rot_ten(R_bot, Cm[:, :, :, bot_cols]))

    # Turbulent transport  T_ij = -d<uiujuk>/dxk
    duuudx = D[23] - 3 * U * U * dUdx - 3 * (U * duudx + uu * dUdx)
    dvvudx = D[35] - 2 * (V * duvdx + uv * dVdx) - (U * dvvdx + vv * dUdx) - (V * V * dUdx + 2 * U * V * dVdx)
    dwwudx = D[39] - 2 * (W * duwdx + uw * dWdx) - (U * dwwdx + ww * dUdx) - (W * W * dUdx + 2 * U * W * dWdx)
    duvudx = D[31] - 2 * (U * duvdx + uv * dUdx) - (V * duudx + uu * dVdx) - (U * U * dVdx + 2 * U * V * dUdx)
    duwudx = D[33] - 2 * (U * duwdx + uw * dUdx) - (W * duudx + uu * dWdx) - (U * U * dWdx + 2 * U * W * dUdx)
    dvwudx = D[43] - (U * dvwdx + vw * dUdx) - (V * duwdx + uw * dVdx) - (W * duvdx + uv * dWdx) - (U * V * dWdx + U * W * dVdx + V * W * dUdx)
    duuvdy = D[32] - 2 * (U * duvdy + uv * dUdy) - (V * duudy + uu * dVdy) - (U * U * dVdy + 2 * U * V * dUdy)
    dvvvdy = D[26] - 3 * (V * dvvdy + vv * dVdy) - 3 * V * V * dVdy
    dwwvdy = D[42] - 2 * (W * dvwdy + vw * dWdy) - (V * dwwdy + ww * dVdy) - (W * W * dVdy + 2 * V * W * dWdy)
    duvvdy = D[36] - 2 * (V * duvdy + uv * dVdy) - (U * dvvdy + vv * dUdy) - (V * V * dUdy + 2 * U * V * dVdy)
    duwvdy = D[44] - (U * dvwdy + vw * dUdy) - (V * duwdy + uw * dVdy) - (W * duvdy + uv * dWdy) - (U * V * dWdy + U * W * dVdy + V * W * dUdy)
    dvwvdy = D[38] - 2 * (V * dvwdy + vw * dVdy) - (W * dvvdy + vv * dWdy) - (V * V * dWdy + 2 * V * W * dVdy)
    duuwdz = dvvwdz = dwwwdz = duvwdz = duwwdz = dvwwdz = Z
    Txx = -(duuudx + duuvdy + duuwdz)
    Txy = -(duvudx + duvvdy + duvwdz)
    Txz = -(duwudx + duwvdy + duwwdz)
    Tyy = -(dvvudx + dvvvdy + dvvwdz)
    Tyz = -(dvwudx + dvwvdy + dvwwdz)
    Tzz = -(dwwudx + dwwvdy + dwwwdz)
    Tm = np.array([[Txx, Txy, Txz], [Txy, Tyy, Tyz], [Txz, Tyz, Tzz]])
    scatter_sym("T", _rot_ten(R_top, Tm[:, :, :, top_cols]),
                _rot_ten(R_bot, Tm[:, :, :, bot_cols]))

    # Viscous diffusion  VD_ij = nu d2<uiuj>/dxk2
    d2uudx2 = D[51] - 2 * (U * D[45] + dUdx * dUdx)
    d2vvdx2 = D[53] - 2 * (V * D[47] + dVdx * dVdx)
    d2wwdx2 = D[55] - 2 * (W * D[49] + dWdx * dWdx)
    d2uvdx2 = D[57] - (V * D[45] + U * D[47] + 2 * dUdx * dVdx)
    d2uwdx2 = D[59] - (U * D[49] + W * D[45] + 2 * dUdx * dWdx)
    d2vwdx2 = D[61] - (V * D[49] + W * D[47] + 2 * dVdx * dWdx)
    d2uudy2 = D[52] - 2 * (U * D[46] + dUdy * dUdy)
    d2vvdy2 = D[54] - 2 * (V * D[48] + dVdy * dVdy)
    d2wwdy2 = D[56] - 2 * (W * D[50] + dWdy * dWdy)
    d2uvdy2 = D[58] - (V * D[46] + U * D[48] + 2 * dUdy * dVdy)
    d2uwdy2 = D[60] - (U * D[50] + W * D[46] + 2 * dUdy * dWdy)
    d2vwdy2 = D[62] - (V * D[50] + W * D[48] + 2 * dVdy * dWdy)
    VDxx = nu * (d2uudx2 + d2uudy2)
    VDyy = nu * (d2vvdx2 + d2vvdy2)
    VDzz = nu * (d2wwdx2 + d2wwdy2)
    VDxy = nu * (d2uvdx2 + d2uvdy2)
    VDxz = nu * (d2uwdx2 + d2uwdy2)
    VDyz = nu * (d2vwdx2 + d2vwdy2)
    VDm = np.array([[VDxx, VDxy, VDxz], [VDxy, VDyy, VDyz], [VDxz, VDyz, VDzz]])
    scatter_sym("VD", _rot_ten(R_top, VDm[:, :, :, top_cols]),
                _rot_ten(R_bot, VDm[:, :, :, bot_cols]))

    # Pressure transport  PT_ij (built from dpuidxj) then symmetric combos
    dpudx = D[63] - P * dUdx - U * D[7]
    dpvdx = D[65] - P * dVdx - V * D[7]
    dpwdx = D[67] - P * dWdx - W * D[7]
    dpudy = D[64] - P * dUdy - U * D[8]
    dpvdy = D[66] - P * dVdy - V * D[8]
    dpwdy = D[68] - P * dWdy - W * D[8]
    dpudz = dpvdz = dpwdz = Z
    PTxx = dpudx
    PTyy = dpvdy
    PTzz = dpwdz
    PTxy = dpudy + dpvdx
    PTxz = dpudz + dpwdx
    PTyz = dpvdz + dpwdy
    PTm = np.array([[PTxx, PTxy, PTxz], [PTxy, PTyy, PTyz], [PTxz, PTyz, PTzz]])
    scatter_sym("PT", _rot_ten(R_top, PTm[:, :, :, top_cols]),
                _rot_ten(R_bot, PTm[:, :, :, bot_cols]))

    # Pressure strain  PS_ij
    pdudx = F[15] - P * dUdx; pdudy = F[16] - P * dUdy; pdudz = F[17] - P * dUdz
    pdvdx = F[18] - P * dVdx; pdvdy = F[19] - P * dVdy; pdvdz = F[20] - P * dVdz
    pdwdx = F[21] - P * dWdx; pdwdy = F[22] - P * dWdy; pdwdz = F[23] - P * dWdz
    PSxx = pdudx
    PSyy = pdvdy
    PSzz = pdwdz
    PSxy = pdudy + pdvdx
    PSxz = pdudz + pdwdx
    PSyz = pdvdz + pdwdy
    PSm = np.array([[PSxx, PSxy, PSxz], [PSxy, PSyy, PSyz], [PSxz, PSyz, PSzz]])
    scatter_sym("PS", _rot_ten(R_top, PSm[:, :, :, top_cols]),
                _rot_ten(R_bot, PSm[:, :, :, bot_cols]))

    # velocity-pressure-gradient tensor Pi
    for s in (top, bottom):
        for p in range(npf):
            e = s[p]
            e["Pixx"] = -2 / rho * (e["PTxx"] - e["PSxx"])
            e["Piyy"] = -2 / rho * (e["PTyy"] - e["PSyy"])
            e["Pizz"] = -2 / rho * (e["PTzz"] - e["PSzz"])
            e["Pixy"] = -1 / rho * (e["PTxy"] - e["PSxy"])
            e["Pixz"] = -1 / rho * (e["PTxz"] - e["PSxz"])
            e["Piyz"] = -1 / rho * (e["PTyz"] - e["PSyz"])

    # Reynolds-stress budget sums (with/without mean convection)
    for s in (top, bottom):
        for p in range(npf):
            e = s[p]
            for suf in ("xx", "yy", "zz", "xy", "xz", "yz"):
                e["S" + suf] = (e["P" + suf] + e["D" + suf] + e["T" + suf]
                                + e["VD" + suf] + e["Pi" + suf])
                e["Sc" + suf] = e["S" + suf] - e["C" + suf]

    # Skewness tensor (rank-3, central moments) then rotate
    Rtot = np.empty((3, 3, 3) + U.shape)
    Rtot[:, :, 0] = np.array([[F[24], F[28], F[29]],
                              [F[28], F[30], F[34]],
                              [F[29], F[34], F[32]]])
    Rtot[:, :, 1] = np.array([[F[28], F[30], F[34]],
                              [F[30], F[25], F[31]],
                              [F[34], F[31], F[33]]])
    Rtot[:, :, 2] = np.array([[F[29], F[34], F[32]],
                              [F[34], F[31], F[33]],
                              [F[32], F[33], F[26]]])
    R3 = np.empty_like(Rtot)
    # central third moments (read_data_profiles.m lines 1568-1597)
    R3[0, 0, 0] = Rtot[0, 0, 0] - 3 * U * uu - U * U * U
    R3[0, 1, 0] = Rtot[0, 1, 0] - 2 * U * uv - V * uu - U * U * V
    R3[0, 2, 0] = Rtot[0, 2, 0] - 2 * U * uw - W * uu - U * U * W
    R3[1, 1, 0] = Rtot[1, 1, 0] - 2 * V * uv - U * vv - V * V * U
    R3[1, 2, 0] = Rtot[1, 2, 0] - U * vw - V * uw - W * uv - U * V * W
    R3[2, 2, 0] = Rtot[2, 2, 0] - 2 * W * uw - U * ww - W * W * U
    R3[1, 0, 0] = R3[0, 1, 0]; R3[2, 0, 0] = R3[0, 2, 0]; R3[2, 1, 0] = R3[1, 2, 0]
    R3[0, 0, 1] = R3[0, 1, 0]
    R3[0, 1, 1] = R3[1, 1, 0]
    R3[0, 2, 1] = R3[1, 2, 0]
    R3[1, 1, 1] = Rtot[1, 1, 1] - 3 * V * vv - V * V * V
    R3[1, 2, 1] = Rtot[1, 2, 1] - 2 * V * vw - W * vv - V * V * W
    R3[2, 2, 1] = Rtot[2, 2, 1] - 2 * W * vw - V * ww - W * W * V
    R3[1, 0, 1] = R3[0, 1, 1]; R3[2, 0, 1] = R3[0, 2, 1]; R3[2, 1, 1] = R3[1, 2, 1]
    R3[0, 0, 2] = R3[0, 2, 0]
    R3[0, 1, 2] = R3[1, 2, 0]
    R3[0, 2, 2] = R3[2, 2, 0]
    R3[1, 1, 2] = R3[1, 2, 1]      # MATLAB slice-3 (2,2): Rtot(2,2,3)-2V vw-W vv-V^2 W
    R3[1, 2, 2] = Rtot[1, 2, 2] - 2 * W * vw - V * ww - W * W * V
    R3[2, 2, 2] = Rtot[2, 2, 2] - 3 * W * ww - W * W * W
    R3[1, 0, 2] = R3[0, 1, 2]; R3[2, 0, 2] = R3[0, 2, 2]; R3[2, 1, 2] = R3[1, 2, 2]
    S3t = _rot_ten3(R_top, R3[:, :, :, :, top_cols])
    S3b = _rot_ten3(R_bot, R3[:, :, :, :, bot_cols])
    skew_map = {"uuu": (0, 0, 0), "vvv": (1, 1, 1), "www": (2, 2, 2),
                "uuv": (0, 1, 0), "uuw": (0, 2, 0), "uvv": (1, 1, 0),
                "vvw": (1, 2, 1), "uww": (2, 2, 0), "vww": (2, 2, 1),
                "uvw": (1, 2, 0)}
    for name, (a, b, c) in skew_map.items():
        for p in range(npf):
            top[p][name] = S3t[a, b, c, :, p].reshape(-1, 1)
            bottom[p][name] = S3b[a, b, c, :, p].reshape(-1, 1)

    # D45 (second derivative, stored unrotated)
    for p in range(npf):
        top[p]["D45"] = D[45][:, top_cols[p]].reshape(-1, 1)
        bottom[p]["D45"] = D[45][:, bot_cols[p]].reshape(-1, 1)

    # Derivatives of Reynolds-stress tensor (rank-3) rotated
    RS = np.empty((3, 3, 3) + U.shape)
    RS[:, :, 0] = np.array([[duudx, duvdx, duwdx],
                            [duvdx, dvvdx, dvwdx],
                            [duwdx, dvwdx, dwwdx]])
    RS[:, :, 1] = np.array([[duudy, duvdy, duwdy],
                            [duvdy, dvvdy, dvwdy],
                            [duwdy, dvwdy, dwwdy]])
    RS[:, :, 2] = np.array([[duudz, duvdz, duwdz],
                            [duvdz, dvvdz, dvwdz],
                            [duwdz, dvwdz, dwwdz]])
    DSt = _rot_ten3(R_top, RS[:, :, :, :, top_cols])
    DSb = _rot_ten3(R_bot, RS[:, :, :, :, bot_cols])
    sd_map = {"duudx": (0, 0, 0), "duudy": (1, 0, 0), "dvvdx": (0, 1, 1),
              "dvvdy": (1, 1, 1), "duvdx": (0, 1, 0), "duvdy": (1, 1, 0),
              "duwdx": (0, 2, 0), "duwdy": (1, 2, 0), "dvwdx": (0, 2, 1),
              "dvwdy": (1, 2, 1)}
    for name, (a, b, c) in sd_map.items():
        for p in range(npf):
            top[p][name] = DSt[a, b, c, :, p].reshape(-1, 1)
            bottom[p][name] = DSb[a, b, c, :, p].reshape(-1, 1)

    # continuity-check raw derivatives D1..D4
    for name, k in (("D1", 1), ("D2", 2), ("D3", 3), ("D4", 4)):
        for p in range(npf):
            top[p][name] = D[k][:, top_cols[p]].reshape(-1, 1)
            bottom[p][name] = D[k][:, bot_cols[p]].reshape(-1, 1)

    # energy flux pu,pv,pw  (note: pw uses F[14]-F[2]*F[4], as in MATLAB)
    PU = np.array([F[12] - F[1] * F[4], F[13] - F[2] * F[4], F[14] - F[2] * F[4]])
    put = _rot_vec(R_top, PU[:, :, top_cols]); pub = _rot_vec(R_bot, PU[:, :, bot_cols])
    for k, name in enumerate(["pu", "pv", "pw"]):
        for p in range(npf):
            top[p][name] = put[k, :, p].reshape(-1, 1)
            bottom[p][name] = pub[k, :, p].reshape(-1, 1)
