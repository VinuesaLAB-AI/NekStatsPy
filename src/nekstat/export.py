"""
Port of 02_MatLab/matlab_func/export_drag_decomposition.m plus the assembly of the
final `Re200k` structure and its MATLAB-compatible .mat serialisation.
"""

import numpy as np
import scipy.io as sio

# (out_name, source_key) — order follows export_drag_decomposition.m
_EXPORT_FIELDS = [
    ("xa", "xa"), ("ya", "ya"), ("alpha", "alpha"),
    ("x", "x"), ("y", "y"), ("yn", "yn"),
    ("mu", "mu"), ("nu", "nu"), ("rho", "rho"), ("pinf", "pinf"),
    ("U", "U"), ("V", "V"), ("W", "W"),
    ("P", "P"), ("pp", "pp"),
    ("dPdx", "dPdx"), ("dPdy", "dPdy"), ("dPdz", "dPdz"),
    ("uu", "uu"), ("vv", "vv"), ("ww", "ww"),
    ("uv", "uv"), ("uw", "uw"), ("vw", "vw"),
    ("dUdx", "dUdx"), ("dUdy", "dUdy"), ("dVdx", "dVdx"), ("dVdy", "dVdy"),
    ("Uinf", "Uinf"), ("Ue", "Ue"),
    ("tauw", "tauw"), ("ut", "ut"), ("delta99", "delta99"),
    ("theta", "theta"), ("deltas", "deltas"), ("Reds", "Reds"),
    ("dsoverd99", "dsoverd99"),
    ("beta", "beta"), ("Reth", "Reth"), ("Ret", "Ret"),
    ("Cf", "Cf"), ("Cfinf", "Cfinf"), ("Cp", "Cp"),
    ("H12", "_H12"),
    ("yp", "yp"), ("Up", "Up"), ("Vp", "Vp"),
    ("uup", "uup"), ("vvp", "vvp"), ("wwp", "wwp"), ("uvp", "uvp"),
    ("Pkp", "Pkp"), ("Pxx", "Pxx"), ("Pyy", "Pyy"),
    ("Dkp", "Dkp"), ("Tkp", "Tkp"), ("VDkp", "VDkp"),
    ("VPGkp", "VPGkp"), ("Ckp", "Ckp"),
    ("vvv", "vvv"), ("pu", "pu"), ("pv", "pv"), ("pw", "pw"),
    ("stev", "stev"),
    ("Cl", "cl2"), ("Cd", "cd2"), ("Cd_p", "cd_p"), ("Cd_f", "cd_f"), ("LD", "LD"),
]


def export_drag_decomposition(buffdata, istart, iend):
    """Select the profiles istart+1..iend (MATLAB 1-based) into a list of export
    dicts.  Adds the derived H12 = deltas/theta."""
    out = []
    for iprof in range(1, iend - istart + 1):          # MATLAB iprof=1..nprof
        e = buffdata[iprof + istart - 1]               # buffdata(iprof+istart)
        e["_H12"] = e["deltas"] / e["theta"]
        rec = {}
        for out_name, key in _EXPORT_FIELDS:
            rec[out_name] = e[key]
        out.append(rec)
    return out


def _as_field(v):
    """Wrap a profile value the way MATLAB stores it: profile arrays -> (n,1)
    column, scalars -> (1,1)."""
    a = np.asarray(v, dtype=np.float64)
    if a.ndim == 0 or a.size == 1:
        return a.reshape(1, 1)
    return a.reshape(-1, 1)


def _struct_array(records):
    """Build a (1, nprof) numpy structured array that scipy.io.savemat serialises
    as a MATLAB 1xN struct array."""
    nprof = len(records)
    names = [n for n, _ in _EXPORT_FIELDS]
    dt = np.dtype([(n, "O") for n in names])
    arr = np.empty((1, nprof), dtype=dt)
    for p, rec in enumerate(records):
        for n in names:
            arr[n][0, p] = _as_field(rec[n])
    return arr


def save_Re200k(top_ref, bot_ref, filename):
    """Assemble Re200k.top.ref / Re200k.bot.ref and write the .mat file."""
    Re200k = {"top": {"ref": _struct_array(top_ref)},
              "bot": {"ref": _struct_array(bot_ref)}}
    sio.savemat(filename, {"Re200k": Re200k}, do_compression=True)
    return filename
