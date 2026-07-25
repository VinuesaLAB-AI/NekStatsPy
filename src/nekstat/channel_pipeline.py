"""Pure Python channel-flow post-processing equivalent to ``post_STAT.m``."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import numpy as np
import scipy.io as sio

from .channel_intfld import ChannelIntFld, channel_database, read_channel_int_fld

Pathish = Union[str, os.PathLike]


@dataclass(frozen=True)
class ChannelPostResult:
    """Output paths and scalar diagnostics from channel post-processing."""

    int_fld: Path
    mat_file: Optional[Path]
    figdir: Optional[Path]
    data: Dict[str, Any]

    @property
    def retau(self) -> float:
        return float(np.asarray(self.data["Ret"]).squeeze())

    @property
    def re_theta(self) -> float:
        return float(np.asarray(self.data["Reth"]).squeeze())

    @property
    def h12(self) -> float:
        return float(np.asarray(self.data["H12"]).squeeze())


def process_channel(
    int_fld: Pathish,
    *,
    nx: int = 100,
    ny: int = 1000,
    reb: float = 4600.0,
    nu: Optional[float] = None,
    rho: float = 1.0,
) -> ChannelPostResult:
    """Process a KTH channel ``int_fld`` and return the MATLAB-style data struct."""

    int_fld = Path(int_fld)
    viscosity = (1.0 / float(reb)) if nu is None else float(nu)
    raw = read_channel_int_fld(int_fld)
    db = channel_database(raw, nx, ny)
    fields = compute_channel_fields(db)
    data = matlab_channel_data(fields, nx, ny, viscosity, float(rho))
    data.update(
        {
            "Reb": float(reb),
            "int_fld_start_time": raw.start_time,
            "int_fld_end_time": raw.end_time,
            "int_fld_effective_average_time": raw.effective_average_time,
            "int_fld_time_interval": raw.time_interval,
            "int_fld_nrec": raw.nrec,
            "int_fld_npoints": raw.npoints,
            "int_fld_re": raw.reynolds_number,
        }
    )
    return ChannelPostResult(int_fld=int_fld, mat_file=None, figdir=None, data=data)


def run_channel_post(
    int_fld: Pathish,
    *,
    nx: int = 100,
    ny: int = 1000,
    reb: float = 4600.0,
    nu: Optional[float] = None,
    rho: float = 1.0,
    out_file: Optional[Pathish] = None,
    figdir: Optional[Pathish] = None,
    save_figures: bool = True,
) -> ChannelPostResult:
    """Process channel data, optionally saving a MATLAB ``data`` struct and plots."""

    result = process_channel(int_fld, nx=nx, ny=ny, reb=reb, nu=nu, rho=rho)
    data = result.data

    mat_file = Path(out_file) if out_file else None
    if mat_file is not None:
        mat_file.parent.mkdir(parents=True, exist_ok=True)
        sio.savemat(mat_file, {"data": data}, do_compression=False, oned_as="column")

    resolved_figdir = Path(figdir) if figdir else None
    if resolved_figdir is not None and save_figures:
        save_channel_figures(data, resolved_figdir)

    return ChannelPostResult(
        int_fld=result.int_fld,
        mat_file=mat_file,
        figdir=resolved_figdir,
        data=data,
    )


def run_channel_post_from_config(
    config: Mapping[str, Any],
    *,
    base_dir: Optional[Path] = None,
    run_dir: Optional[Path] = None,
    int_fld_override: Optional[Pathish] = None,
    out_override: Optional[Pathish] = None,
    figdir_override: Optional[Pathish] = None,
    save_figures_override: Optional[bool] = None,
) -> ChannelPostResult:
    """Run the channel post stage from a resolved config mapping."""

    base_dir = Path.cwd() if base_dir is None else Path(base_dir)
    root = _repo_root(base_dir) if run_dir is None else run_dir
    conf = dict(config.get("channel_post", {}) or {})

    int_fld = int_fld_override or conf.get("int_fld", "int_fld")
    int_fld_path = _resolve_path(int_fld, root, base_dir)
    out_value = out_override if out_override is not None else conf.get("out")
    out_path = _resolve_optional_output_path(out_value, root)
    figdir_value = figdir_override if figdir_override is not None else conf.get("figdir")
    figdir = _resolve_optional_output_path(figdir_value, root)
    save_figures = bool(conf.get("save_figures", True))
    if save_figures_override is not None:
        save_figures = save_figures_override

    nu_value = conf.get("nu", conf.get("viscosity"))
    nu = None if nu_value in (None, "", "auto") else float(nu_value)
    return run_channel_post(
        int_fld_path,
        nx=int(conf.get("nx", 100)),
        ny=int(conf.get("ny", 1000)),
        reb=float(conf.get("reb", 4600.0)),
        nu=nu,
        rho=float(conf.get("rho", 1.0)),
        out_file=out_path,
        figdir=figdir,
        save_figures=save_figures,
    )


def compute_channel_fields(db: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Compute the subset of KTH turbulence fields used by ``post_STAT.m``."""

    required = ("F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F25", "D2")
    missing = [name for name in required if name not in db]
    if missing:
        raise ValueError(f"channel int_fld is missing required fields: {', '.join(missing)}")

    u = db["F1"]
    v = db["F2"]
    w = db["F3"]
    p = db["F4"]
    uu = db["F5"] - u * u
    vv = db["F6"] - v * v
    ww = db["F7"] - w * w
    pp = db["F8"] - p * p
    uv = db["F9"] - u * v
    vvv = db["F25"] - 3.0 * v * vv - v * v * v

    return {
        "x": db["x"],
        "y": db["y"],
        "U": u,
        "V": v,
        "W": w,
        "P": p,
        "dUdy": db["D2"],
        "uu": uu,
        "vv": vv,
        "ww": ww,
        "uv": uv,
        "pp": pp,
        "vvv": vvv,
    }


def matlab_channel_data(
    fields: Mapping[str, np.ndarray],
    nx: int,
    ny: int,
    nu: float,
    rho: float,
) -> Dict[str, Any]:
    """Mirror the reshape, averaging, and wall-unit calculations in ``post_STAT.m``."""

    matrices = {name: _matlab_matrix(value, nx, ny) for name, value in fields.items()}
    data: Dict[str, Any] = {
        "x": matrices["x"],
        "yy": matrices["y"],
        "Ur": matrices["U"],
        "dUdyr": matrices["dUdy"],
        "uur": matrices["uu"],
        "vvr": matrices["vv"],
        "wwr": matrices["ww"],
        "uvr": matrices["uv"],
        "Vr": matrices["V"],
        "Pr": matrices["P"],
        "pp": matrices["pp"],
        "nu": float(nu),
        "rho": float(rho),
    }

    data["y"] = matrices["y"][:, 0]
    data["U"] = np.mean(matrices["U"], axis=1)
    data["V"] = np.mean(matrices["V"], axis=1)
    data["dUdy"] = np.mean(matrices["dUdy"], axis=1)
    data["uu"] = np.mean(matrices["uu"], axis=1)
    data["vv"] = np.mean(matrices["vv"], axis=1)
    data["ww"] = np.mean(matrices["ww"], axis=1)
    data["uv"] = np.mean(matrices["uv"], axis=1)
    data["urms"] = np.sqrt(data["uu"])
    data["prms"] = np.mean(np.sqrt(matrices["pp"]), axis=1)
    data["vvv"] = np.mean(np.abs(matrices["vvv"]), axis=1)

    data["twall"] = float(data["rho"] * data["nu"] * data["dUdy"][0])
    data["utau"] = float(np.sqrt(data["twall"] / data["rho"]))
    data["Ret"] = float(data["utau"] / data["nu"])
    data["lstar"] = float(data["nu"] / data["utau"])

    data["yp"] = data["y"] / data["lstar"]
    data["Up"] = data["U"] / data["utau"]
    data["Vp"] = data["V"] / data["utau"]
    data["uup"] = data["uu"] / data["utau"] ** 2
    data["vvp"] = data["vv"] / data["utau"] ** 2
    data["wwp"] = data["ww"] / data["utau"] ** 2
    data["uvp"] = data["uv"] / data["utau"] ** 2
    data["urmsp"] = data["urms"] / data["utau"]

    data["Uinf"] = float(np.max(data["U"]))
    y_with_wall = np.concatenate(([0.0], data["y"]))
    u_with_wall = np.concatenate(([0.0], data["U"]))
    u_outer = u_with_wall / data["Uinf"]
    data["theta"] = float(_trapz(u_outer * (1.0 - u_outer), y_with_wall))
    data["deltas"] = float(_trapz(1.0 - u_outer, y_with_wall))
    data["Reth"] = float(data["Uinf"] * data["theta"] / data["nu"])
    data["H12"] = float(data["deltas"] / data["theta"])
    return data


def save_channel_figures(data: Mapping[str, Any], figdir: Pathish) -> None:
    """Write the same figure set produced by ``post_STAT.m``."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figdir = Path(figdir)
    figdir.mkdir(parents=True, exist_ok=True)

    _semilogx(plt, data["y"], data["U"], figdir / "U_prof.jpg")
    _semilogx(plt, data["yp"], data["Up"], figdir / "U+_prof.jpg", xmax=2 * data["Ret"])
    _semilogx(plt, data["yp"], -data["uvp"], figdir / "uv+_prof.jpg", xmax=2 * data["Ret"])
    _semilogx(plt, data["yp"], data["Vp"], figdir / "V+_prof.jpg", xmax=2 * data["Ret"])
    _semilogx(plt, data["yp"], data["vvp"], figdir / "vv+_prof.jpg", xmax=2 * data["Ret"])
    _semilogx(plt, data["yp"], data["vv"], figdir / "vv+_prof.jpg", xmax=2 * data["Ret"])
    _semilogx(plt, data["yp"], data["prms"], figdir / "prms+_prof.jpg", xmax=2 * data["Ret"])


def compare_channel_mat_files(
    mine: Pathish,
    ref: Pathish,
    *,
    fields: Optional[tuple] = None,
) -> Dict[str, float]:
    """Return max absolute differences for common MATLAB ``data`` fields."""

    mine_data = _load_data_struct(mine)
    ref_data = _load_data_struct(ref)
    names = fields or tuple(sorted(set(mine_data) & set(ref_data)))
    diffs: Dict[str, float] = {}
    for name in names:
        if name.startswith("__") or name not in mine_data or name not in ref_data:
            continue
        a = np.asarray(mine_data[name])
        b = np.asarray(ref_data[name])
        if a.shape != b.shape or not np.issubdtype(a.dtype, np.number):
            continue
        delta = np.abs(a.astype(float) - b.astype(float))
        finite = np.isfinite(delta)
        if finite.any():
            diffs[name] = float(np.max(delta[finite]))
        else:
            same_nan = np.array_equal(np.isnan(a), np.isnan(b))
            diffs[name] = 0.0 if same_nan else float("nan")
    return diffs


def _matlab_matrix(value: np.ndarray, nx: int, ny: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (nx, ny):
        array = array.reshape(nx, ny)
    return array.T.copy()


def _semilogx(plt, x, y, path: Path, xmax: Optional[float] = None) -> None:
    plt.figure()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    positive = x > 0
    if positive.any():
        plt.semilogx(x[positive], y[positive])
    else:
        plt.plot(x, y)
    if xmax is not None:
        left = float(np.min(x[positive])) if positive.any() else 0.0
        plt.xlim([left, xmax])
    plt.savefig(path, format="jpg")
    plt.close()


def _load_data_struct(path: Pathish) -> Dict[str, np.ndarray]:
    mat = sio.loadmat(path, squeeze_me=False, struct_as_record=False)
    data = mat.get("data")
    if data is None:
        raise ValueError(f"{path}: missing MATLAB variable 'data'")
    if isinstance(data, np.ndarray) and data.dtype.names:
        item = data[0, 0]
        return {name: item[name] for name in data.dtype.names}
    if isinstance(data, np.ndarray) and data.size == 1 and hasattr(data.item(), "_fieldnames"):
        item = data.item()
        return {name: getattr(item, name) for name in item._fieldnames}
    raise ValueError(f"{path}: unsupported 'data' struct layout")


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return float(trapezoid(y, x))


def _resolve_path(value: Any, root: Path, base_dir: Path) -> Path:
    path = Path(os.path.expanduser(str(value)))
    if path.is_absolute():
        return path
    candidate = (root / path).resolve()
    if candidate.exists():
        return candidate
    return (base_dir / path).resolve()


def _resolve_optional_path(value: Any, root: Path, base_dir: Path) -> Optional[Path]:
    if value in (None, "", False):
        return None
    return _resolve_path(value, root, base_dir)


def _resolve_optional_output_path(value: Any, root: Path) -> Optional[Path]:
    if value in (None, "", False):
        return None
    path = Path(os.path.expanduser(str(value)))
    if path.is_absolute():
        return path
    return (root / path).resolve()


def _repo_root(start: Path) -> Path:
    start = start.resolve()
    candidates = [start] + list(start.parents)
    for path in candidates:
        if (path / "pyproject.toml").exists() and (path / "src" / "nekstat").exists():
            return path
    return Path.cwd().resolve()
