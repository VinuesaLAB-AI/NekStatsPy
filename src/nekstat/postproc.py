"""
Port of 02_MatLab/matlab_func/load_prof_data.m

Consumes the `top`/`bottom` profile lists from read_intfld.read_data_profiles and
adds inner-scaling, boundary-layer integral quantities, Cf/Cp, the turbulent-budget
scalars, and the airfoil Cl/Cd decomposition.  MATLAB 1-based istart/iend are kept
(profiles top[i-1] for i in istart..iend); Cl/Cd is integrated over all profiles.
"""

import numpy as np
from .diagnostic import diagnostic, trapezoidal2

_BUDGET_SUF = ("xx", "yy", "zz", "xy", "xz", "yz")


def load_prof_data(top, bottom, istart, iend, budget_flag=1):
    bot = bottom
    Uinf = 1.0
    rho = float(top[0]["rho"])
    mu = float(top[0]["mu"])
    nu = float(top[0]["nu"])
    npts = len(top)                 # 'np' in MATLAB
    nyn = top[0]["U"].shape[0]      # 'ln'

    def rng():                      # MATLAB for i=istart:iend  (1-based)
        return range(istart, iend + 1)

    # --- inner scaling ------------------------------------------------------
    for side in (top, bot):
        for i in rng():
            e = side[i - 1]
            e["tauw"] = mu * e["dUdy"][0, 0]
            e["dUdyw"] = e["dUdy"][0, 0]
            e["ut"] = np.sqrt(abs(e["tauw"]) / rho)
            e["Up"] = e["U"] / e["ut"]
            e["Vp"] = e["V"] / e["ut"]
            e["yp"] = e["yn"] * e["ut"] / nu
            e["dUpdyp"] = nu / e["ut"] ** 2 * e["dUdy"]

    # --- inner-scaled stresses ---------------------------------------------
    for side in (top, bot):
        for i in rng():
            e = side[i - 1]
            e["uup"] = e["uu"] / e["ut"] ** 2
            e["vvp"] = e["vv"] / e["ut"] ** 2
            e["wwp"] = e["ww"] / e["ut"] ** 2
            e["uvp"] = e["uv"] / e["ut"] ** 2

    # --- BL edge + integral quantities (via diagnostic) ---------------------
    for side in (top, bot):
        for i in rng():
            e = side[i - 1]
            u_diag = np.sqrt(np.abs(e["uup"]))          # == abs(sqrt(uup))
            d99p, Uinfp, i99, H, dsp, thp = diagnostic(e["yp"], e["Up"], u_diag)
            e["deltap99"], e["Uinfp"], e["i99"] = d99p, Uinfp, i99
            e["H"], e["deltasp"], e["thetap"] = H, dsp, thp
            e["delta99"] = d99p * nu / e["ut"]
            e["deltas"] = dsp * nu / e["ut"]
            e["dsoverd99"] = e["deltas"] / e["delta99"]
            e["theta"] = thp * nu / e["ut"]
            e["Uinf"] = Uinfp * e["ut"]
            e["Ret"] = d99p
            e["Reds"] = e["Uinf"] * e["deltas"] / nu
            e["Reth"] = e["Uinf"] * e["theta"] / nu

    # --- edge velocities (MATLAB loops over ALL profiles, i99 set only in range) --
    for side in (top, bot):
        for e in side:
            if "i99" in e:
                k = e["i99"] - 1                        # MATLAB U(i99) on full array
                e["Ue"] = e["U"][k, 0]
                e["Ve"] = e["V"][k, 0]

    # --- pressure-gradient parameter ---------------------------------------
    for side in (top, bot):
        for i in rng():
            e = side[i - 1]
            k = e["i99"] - 1
            e["Delta"] = e["Uinfp"] * e["deltas"]
            e["dUinfdx"] = e["dUdx"][k, 0]
            e["beta"] = e["deltas"] / e["tauw"] * e["dPdx"][k, 0]
            e["dPdxe"] = e["dPdx"][k, 0]
            e["dtauw"] = e["deltas"] / e["tauw"]

    # --- skin friction ------------------------------------------------------
    for i in rng():
        e = top[i - 1]
        cf = 2 * (e["ut"] / e["Ue"]) ** 2
        e["Cf"] = cf if e["dUdyw"] >= 0 else -cf
    for i in rng():
        e = bot[i - 1]
        e["Cf"] = 2 * (e["ut"] / e["Ue"]) ** 2

    # --- wall pressure / Cp -------------------------------------------------
    for i in rng():
        et, eb = top[i - 1], bot[i - 1]
        et["Pw"] = et["P"][0, 0]
        et["Cp"] = 2 * (et["P"][0, 0] - et["pinf"]) / (rho * Uinf ** 2)
        eb["Pw"] = eb["P"][0, 0]
        eb["Cp"] = 2 * (eb["P"][0, 0] - et["pinf"]) / (rho * Uinf ** 2)

    # --- Cf based on Uinf ---------------------------------------------------
    for i in rng():
        e = top[i - 1]
        cfi = 2 * (e["ut"] / Uinf) ** 2
        e["Cfinf"] = cfi if e["dUdyw"] >= 0 else -cfi
    for i in rng():
        bot[i - 1]["Cfinf"] = 2 * (bot[i - 1]["ut"] / Uinf) ** 2

    # --- FIK integrand helpers ---------------------------------------------
    for i in rng():
        et, eb = top[i - 1], bot[i - 1]
        Uinf_i = et["Uinf"]
        d99 = et["delta99"]
        eta = 1 - et["yn"] / d99
        eta[et["i99"] - 1:] = 0
        et["fiketa"] = eta
        et["fiketa2"] = eta ** 2
        et["profcpf"] = et["dPdx"] / Uinf_i ** 2 * d99 * et["fiketa2"]
        eta = 1 - eb["yn"] / d99                       # (uses top's Uinf/delta99)
        eta[eb["i99"] - 1:] = 0
        eb["fiketa"] = eta
        eb["fiketa2"] = eta ** 2
        eb["profcpf"] = eb["dPdx"] / Uinf_i ** 2 * d99 * eb["fiketa2"]

    # --- Stevenson-scaled velocity -----------------------------------------
    for side in (top, bot):
        for i in rng():
            e = side[i - 1]
            up = e["Up"]
            vwp = e["V"][0, 0] / e["ut"]
            e["stev"] = 2 * up / (np.sqrt(1 + vwp * up) + 1)

    # --- turbulent-kinetic-energy budget scalars ---------------------------
    if budget_flag == 1:
        _budget_scalars(top, bot, rng, nu)

    # --- Cl / Cd decomposition (all profiles) ------------------------------
    _cl_cd(top, bot, npts, nyn, mu, rho)

    return top, bot


def _S(e, prefix):
    return e[prefix + "xx"] + e[prefix + "yy"] + e[prefix + "zz"]


def _budget_scalars(top, bot, rng, nu):
    for i in rng():
        e = top[i - 1]
        ut4 = e["ut"] ** 4
        e["Pk"] = _S(e, "P")
        e["Dk"] = _S(e, "D")
        e["DRTk"] = e["DRTx"] + e["DRTy"] + e["DRTz"]
        e["Tk"] = _S(e, "T")
        e["VDk"] = _S(e, "VD")
        e["VPGk"] = _S(e, "Pi")
        e["Ck"] = -_S(e, "C")
        e["Pkp"] = _S(e, "P") * nu / ut4 / 2
        e["Dkp"] = _S(e, "D") * nu / ut4 / 2
        e["DRTkp"] = (e["DRTx"] + e["DRTy"] + e["DRTz"]) * nu / ut4 / 2
        e["Tkp"] = _S(e, "T") * nu / ut4 / 2
        e["VDkp"] = _S(e, "VD") * nu / ut4 / 2
        e["VPGkp"] = _S(e, "Pi") * nu / ut4 / 2
        e["Ckp"] = -_S(e, "C") * nu / ut4 / 2
    for i in rng():
        e = bot[i - 1]
        ut4 = e["ut"] ** 4
        e["Pk"] = _S(e, "P")
        e["Dk"] = _S(e, "D")
        e["DRTk"] = e["DRTx"] + e["DRTy"] + e["DRTz"]
        e["Tk"] = _S(e, "T")
        e["VDk"] = _S(e, "VD")
        e["VPGk"] = _S(e, "Pi")
        e["Ck"] = -_S(e, "C")
        e["Pkp"] = _S(e, "P") * nu / ut4 / 2
        e["Dkp"] = _S(e, "D") * nu / ut4 / 2
        # NB: MATLAB overwrites bot.DRTk here (bug); bot.DRTkp is never set.
        e["DRTk"] = (e["DRTx"] + e["DRTy"] + e["DRTz"]) * nu / ut4 / 2
        e["Tkp"] = _S(e, "T") * nu / ut4 / 2
        e["VDkp"] = _S(e, "VD") * nu / ut4 / 2
        e["VPGkp"] = _S(e, "Pi") * nu / ut4 / 2
        e["Ckp"] = -_S(e, "C") * nu / ut4 / 2


def _cl_cd(top, bot, npts, nyn, mu, rho):
    rhoinf = 1.0
    pinf = top[0]["pinf"]

    Pt = np.empty(npts); Pb = np.empty(npts)
    tau_sn_t = np.empty(npts); tau_sn_b = np.empty(npts)
    angle_t = np.empty(npts); angle_b = np.empty(npts)
    xt = np.empty(npts); xb = np.empty(npts)
    yt = np.empty(npts); yb = np.empty(npts)
    for i in range(npts):
        et, eb = top[i], bot[i]
        # viscous total-stress at the wall (second MATLAB block: no Reynolds term)
        taosn_t = mu * (et["dUdy"][0, 0] + et["dVdx"][0, 0])
        taosn_b = mu * (eb["dUdy"][0, 0] + eb["dVdx"][0, 0])
        Pt[i] = et["P"][0, 0] - pinf
        Pb[i] = eb["P"][0, 0] - pinf
        tau_sn_t[i] = taosn_t
        tau_sn_b[i] = taosn_b
        angle_t[i] = et["alpha"]
        angle_b[i] = eb["alpha"]
        xt[i] = et["xa"]; xb[i] = eb["xa"]
        yt[i] = et["ya"]; yb[i] = eb["ya"]

    N_p_top = -trapezoidal2(xt, yt, Pt, angle_t, 1)
    N_p_bot = +trapezoidal2(xb, yb, Pb, angle_b, 1)
    N_tw_top = trapezoidal2(xt, yt, tau_sn_t, angle_t, 2)
    N_tw_bot = trapezoidal2(xb, yb, tau_sn_b, angle_b, 2)
    A_p_top = trapezoidal2(xt, yt, Pt, angle_t, 3)
    A_p_bot = -trapezoidal2(xb, yb, Pb, angle_b, 3)
    A_tw_top = trapezoidal2(xt, yt, tau_sn_t, angle_t, 4)
    A_tw_bot = trapezoidal2(xb, yb, tau_sn_b, angle_b, 4)

    N2_p = N_p_top + N_p_bot
    N2_s = N_tw_top + N_tw_bot
    A2_p = A_p_top + A_p_bot
    A2_s = A_tw_top + A_tw_bot
    N2 = N2_p + N2_s
    A2 = A2_p + A2_s

    c5, s5 = np.cos(np.deg2rad(5)), np.sin(np.deg2rad(5))
    cl2 = 2 * (N2 * c5 - A2 * s5)
    cd2 = 2 * (N2 * s5 + A2 * c5)
    cd_p = 2 * (N2_p * s5 + A2_p * c5)
    cd_f = 2 * (N2_s * s5 + A2_s * c5)
    LD = cl2 / cd2

    for i in range(npts):
        for e in (top[i], bot[i]):
            e["cl2"], e["cd2"] = cl2, cd2
            e["cd_p"], e["cd_f"], e["LD"] = cd_p, cd_f, LD
    print("Cl,Cd_tot,Cd_p,Cd_f,L/D")
    print([cl2, cd2, cd_p, cd_f, LD])
