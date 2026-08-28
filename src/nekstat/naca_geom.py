"""Configurable interpolation meshes for Nek5000 NACA-airfoil statistics.

The legacy MATLAB generators emit two related artifacts: Fortran-record
``ZSTAT/x.fort`` and ``ZSTAT/y.fort`` for Nek5000, and an ``int_mesh.mat`` file
with the profile metadata consumed by the Python post-processing pipeline.  This
module keeps that contract explicit and supports both wall-normal airfoil meshes
and coordinate-only Cartesian meshes.

The NACA surface and tangent angles are evaluated analytically.  This is the
limiting form of the very dense sampled spline used by ``naca_prof2.m`` and
avoids allocating its roughly 63-million-point default intermediate grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union
import hashlib
import json

import numpy as np
import scipy.io as sio
import yaml
from scipy.optimize import brentq


Pathish = Union[str, Path]


@dataclass(frozen=True)
class MeshCapabilities:
    """Operations supported by one interpolation mesh layout."""

    nek_interpolation: bool = True
    profile_post: bool = False
    clcd: bool = False

    def as_dict(self) -> Dict[str, bool]:
        return {
            "nek_interpolation": self.nek_interpolation,
            "profile_post": self.profile_post,
            "clcd": self.clcd,
        }


@dataclass(frozen=True)
class InterpolationMesh:
    """Interpolation coordinates plus metadata required by downstream stages."""

    x_pts: np.ndarray
    y_pts: np.ndarray
    kind: str
    capabilities: MeshCapabilities
    alphau: Optional[np.ndarray] = None
    alphal: Optional[np.ndarray] = None
    yn: Optional[np.ndarray] = None
    xc: Optional[np.ndarray] = None
    preset: Optional[str] = None

    @property
    def npoints(self) -> int:
        return int(np.asarray(self.x_pts).size)

    @property
    def is_profile_mesh(self) -> bool:
        return self.capabilities.profile_post

    def mat_dict(self) -> Dict[str, np.ndarray]:
        """Return MATLAB-compatible arrays as column vectors."""

        values: Dict[str, np.ndarray] = {
            "x_pts": np.asarray(self.x_pts, dtype=float).reshape(-1, 1),
            "y_pts": np.asarray(self.y_pts, dtype=float).reshape(-1, 1),
        }
        for name in ("alphau", "alphal", "yn", "xc"):
            value = getattr(self, name)
            if value is not None:
                values[name] = np.asarray(value, dtype=float).reshape(-1, 1)
        return values


@dataclass(frozen=True)
class MeshArtifacts:
    """Paths written by :func:`write_mesh_artifacts`."""

    root: Path
    x_fort: Path
    y_fort: Path
    mat_file: Path
    manifest: Path
    figure: Optional[Path]


_PRESETS: Dict[str, Dict[str, Any]] = {
    # This is the committed data/int_mesh.mat compatibility target, not the
    # currently uncommented wrt_int_mesh_wing.m parameter set.
    "naca4412_reference": {
        "kind": "wall_normal_profile",
        "profile": {"naca": "4412"},
        "chordwise": {
            "transition_x": 0.295,
            "leading_edge_count": 40,
            "uniform_dx": 0.01625,
        },
        "wall_normal": {
            "first_spacing": 0.5 * (1.0 / 200000.0) / 0.0424,
            "target_extent": 0.12,
            "growth_ratio": 1.02,
        },
        "extrusion": "surface_normal",
    },
    # Settings exposed by the current wrt_int_mesh_wing.m and its helper.
    "naca4412_script_current": {
        "kind": "wall_normal_profile",
        "profile": {"naca": "4412"},
        "chordwise": {
            "transition_x": 0.395,
            "leading_edge_count": 40,
            "uniform_dx": 0.015,
        },
        "wall_normal": {
            "first_spacing": 0.5 * (1.0 / 200000.0) / 0.0424,
            "target_extent": 0.12,
            "growth_ratio": 1.02,
        },
        "extrusion": "surface_normal",
    },
    # Matches the supplied 0012 reference asset while providing a safe
    # symmetric-airfoil branch instead of inheriting legacy naca_prof2's 0/0.
    "naca0012_reference": {
        "kind": "wall_normal_profile",
        "profile": {"naca": "0012"},
        "chordwise": {
            "transition_x": 0.395,
            "leading_edge_count": 40,
            "uniform_dx": 0.015,
        },
        "wall_normal": {
            "first_spacing": 0.5 * (1.0 / 200000.0) / 0.0471,
            "target_extent": 0.12,
            "growth_ratio": 1.002,
        },
        "extrusion": "surface_normal",
    },
}


def available_presets() -> Tuple[str, ...]:
    """Names of compatibility presets accepted by ``mesh.preset``."""

    return tuple(sorted(_PRESETS))


def _colon(a: float, step: float, b: float) -> np.ndarray:
    """Replicate MATLAB ``a:step:b`` endpoint semantics for positive steps."""

    if step <= 0:
        raise ValueError("spacing must be positive")
    n = int(np.floor((b - a) / step + 1e-10)) + 1
    return a + np.arange(n, dtype=float) * step


def _merged_spec(spec: Mapping[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    """Merge a named preset with explicit configuration overrides."""

    raw = dict(spec)
    preset = raw.pop("preset", None)
    if not preset:
        return raw, None
    try:
        base = _PRESETS[str(preset)]
    except KeyError as exc:
        raise ValueError(
            f"unknown mesh preset {preset!r}; choose one of {', '.join(available_presets())}"
        ) from exc
    merged: Dict[str, Any] = {}
    for key, value in base.items():
        merged[key] = dict(value) if isinstance(value, Mapping) else value
    for key, value in raw.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            child = dict(merged[key])
            child.update(value)
            merged[key] = child
        else:
            merged[key] = value
    return merged, str(preset)


def _naca_parameters(profile: Union[str, Mapping[str, Any]]) -> Tuple[float, float, float]:
    """Return NACA four-digit camber, camber location, and thickness fractions."""

    if isinstance(profile, Mapping):
        if "naca" in profile:
            profile = str(profile["naca"])
        else:
            try:
                return (
                    float(profile["max_camber"]),
                    float(profile["camber_location"]),
                    float(profile["thickness"]),
                )
            except KeyError as exc:
                raise ValueError(
                    "profile requires naca or max_camber, camber_location, and thickness"
                ) from exc
    digits = str(profile).upper().replace("NACA", "").strip()
    if len(digits) != 4 or not digits.isdigit():
        raise ValueError(f"NACA profile must be four digits, got {profile!r}")
    return int(digits[0]) / 100.0, int(digits[1]) / 10.0, int(digits[2:]) / 100.0


def _naca4_surface(
    max_camber: float,
    camber_location: float,
    thickness: float,
    x: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate closed-trailing-edge NACA-4 surfaces and tangent angles."""

    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) == 0 or np.any(x < 0) or np.any(x > 1):
        raise ValueError("surface stations must be a non-empty 1-D array in [0, 1]")
    if max_camber == 0.0:
        yc = np.zeros_like(x)
        dyc = np.zeros_like(x)
        d2yc = np.zeros_like(x)
    else:
        if not 0.0 < camber_location < 1.0:
            raise ValueError("camber_location must be in (0, 1) for a cambered NACA profile")
        front = x < camber_location
        yc = np.empty_like(x)
        dyc = np.empty_like(x)
        d2yc = np.empty_like(x)
        xf, xb = x[front], x[~front]
        p = camber_location
        m = max_camber
        yc[front] = m / p**2 * (2 * p * xf - xf**2)
        dyc[front] = 2 * m / p**2 * (p - xf)
        d2yc[front] = -2 * m / p**2
        yc[~front] = m / (1 - p) ** 2 * (1 - 2 * p + 2 * p * xb - xb**2)
        dyc[~front] = 2 * m / (1 - p) ** 2 * (p - xb)
        d2yc[~front] = -2 * m / (1 - p) ** 2
    theta = np.arctan(dyc)
    dtheta = d2yc / (1.0 + dyc**2)
    a0, a1, a2, a3, a4 = 0.2969, -0.126, -0.3516, 0.2843, -0.1036
    yt = thickness / 0.2 * (a0 * np.sqrt(x) + a1 * x + a2 * x**2 + a3 * x**3 + a4 * x**4)
    dyt = np.empty_like(x)
    nonzero = x > 0
    dyt[nonzero] = thickness / 0.2 * (
        a0 / (2 * np.sqrt(x[nonzero])) + a1 + 2 * a2 * x[nonzero]
        + 3 * a3 * x[nonzero] ** 2 + 4 * a4 * x[nonzero] ** 3
    )
    # The exact thickness slope is singular at the leading edge.  Its tangent
    # angle is prescribed below to match naca_prof2.m, so use a finite
    # placeholder while evaluating the coordinate derivatives.
    dyt[~nonzero] = 0.0
    st, ct = np.sin(theta), np.cos(theta)
    xu, yu = x - yt * st, yc + yt * ct
    xl, yl = x + yt * st, yc - yt * ct
    dxu = 1.0 - dyt * st - yt * ct * dtheta
    dyu = dyc + dyt * ct - yt * st * dtheta
    dxl = 1.0 + dyt * st + yt * ct * dtheta
    dyl = dyc - dyt * ct + yt * st * dtheta
    alphau = np.arctan(dyu / dxu)
    alphal = np.arctan(dyl / dxl)
    # Legacy naca_prof2.m assigns both leading-edge angles explicitly.
    alphau[x == 0] = np.pi / 2
    alphal[x == 0] = np.pi / 2
    return xu, yu, alphau, xl, yl, alphal


def naca_prof2(m: float, p: float, xx: float, inputx: Sequence[float], tol: float = 0.5e-7):
    """Compatibility API for ``naca_prof2.m`` without its sampled beta grid."""

    del tol
    return _naca4_surface(m / 100.0, p / 10.0, xx / 100.0, inputx)


def rot(
    xu: Sequence[float], yu: Sequence[float], xl: Sequence[float], yl: Sequence[float],
    yn: Sequence[float], alphau: Sequence[float], alphal: Sequence[float],
):
    """Port of ``rot.m``: extrude surface samples along outward normals."""

    yn2 = np.asarray(yn, dtype=float).reshape(-1, 1)
    xu, yu, xl, yl = (np.asarray(v, dtype=float) for v in (xu, yu, xl, yl))
    alphau, alphal = np.asarray(alphau, dtype=float), np.asarray(alphal, dtype=float)
    xprofu = -np.sin(alphau)[None, :] * yn2 + xu[None, :]
    yprofu = np.cos(alphau)[None, :] * yn2 + yu[None, :]
    xprofl = np.sin(alphal)[None, :] * yn2 + xl[None, :]
    yprofl = -np.cos(alphal)[None, :] * yn2 + yl[None, :]
    return xprofu, yprofu, xprofl, yprofl


def _wall_normal_grid(first_spacing: float, target_extent: float, growth_ratio: float) -> np.ndarray:
    if first_spacing <= 0 or target_extent <= 0:
        raise ValueError("wall_normal.first_spacing and target_extent must be positive")
    if growth_ratio <= 1.0:
        raise ValueError("wall_normal.growth_ratio must be greater than 1")
    n = int(np.ceil(np.log((target_extent * (growth_ratio - 1) / first_spacing) + 1)
                    / np.log(growth_ratio) + 1))
    index = np.arange(n, dtype=float)
    return first_spacing * (growth_ratio**index - 1.0) / (growth_ratio - 1.0)


def _chordwise_grid(
    transition_x: float,
    leading_edge_count: int,
    uniform_dx: float,
    include_trailing_edge: bool = False,
) -> np.ndarray:
    if not 0.0 < transition_x < 1.0:
        raise ValueError("chordwise.transition_x must be in (0, 1)")
    if leading_edge_count < 2:
        raise ValueError("chordwise.leading_edge_count must be at least 2")
    if uniform_dx <= 0 or uniform_dx >= transition_x:
        raise ValueError("chordwise.uniform_dx must be positive and smaller than transition_x")
    xn1 = transition_x - uniform_dx

    def find_r(r: float) -> float:
        return r ** (1 - leading_edge_count) * (transition_x - xn1) * (
            r ** leading_edge_count - r
        ) / (r - 1) - transition_x

    ratio = brentq(find_r, 1.0000001, 5.0)
    dx1 = ratio ** (2 - leading_edge_count) * (transition_x - xn1)
    index = np.arange(leading_edge_count, dtype=float)
    clustered = dx1 * (1.0 - ratio**index) / (1.0 - ratio)
    stations = np.concatenate((clustered[:-1], _colon(clustered[-1], uniform_dx, 1.0)))
    if include_trailing_edge and not np.isclose(stations[-1], 1.0, rtol=0.0, atol=1e-13):
        stations = np.concatenate((stations, np.array([1.0])))
    return stations


def create_profiles_wing(
    m: float, p: float, xx: float, dx: float, nle: int, tol: float,
    dy1: float, yn_thick: float, ry: float, xn: float = 0.295,
):
    """Build the legacy wall-normal profile layout and return its tuple API."""

    xc = _chordwise_grid(xn, int(nle), dx)
    xu, yu, alphau, xl, yl, alphal = naca_prof2(m, p, xx, xc, tol)
    yn = _wall_normal_grid(dy1, yn_thick, ry)
    xprofu, yprofu, xprofl, yprofl = rot(xu, yu, xl, yl, yn, alphau, alphal)
    x_pts = np.concatenate((
        xprofu.ravel(order="F"), xprofl[:, 1:].ravel(order="F"), -np.ones(len(yn)),
    ))
    y_pts = np.concatenate((
        yprofu.ravel(order="F"), yprofl[:, 1:].ravel(order="F"), yn,
    ))
    return x_pts, y_pts, len(x_pts), yn, alphau, alphal, xc


def _build_profile_mesh(spec: Mapping[str, Any], preset: Optional[str]) -> InterpolationMesh:
    m, p, thickness = _naca_parameters(spec.get("profile", {"naca": "4412"}))
    chordwise = dict(spec.get("chordwise") or {})
    wall_normal = dict(spec.get("wall_normal") or {})
    transition_x = float(chordwise.get("transition_x", 0.295))
    nle = int(chordwise.get("leading_edge_count", 40))
    dx = float(chordwise.get("uniform_dx", 0.01625))
    first_spacing = wall_normal.get("first_spacing")
    if first_spacing is None:
        rec = float(wall_normal.get("reynolds_number", 200000.0))
        utau = float(wall_normal.get("friction_velocity", 0.0424))
        yplus = float(wall_normal.get("first_y_plus", 0.5))
        first_spacing = yplus / rec / utau
    yn_extent = float(wall_normal.get("target_extent", 0.12))
    ratio = float(wall_normal.get("growth_ratio", 1.02))
    extrusion = str(spec.get("extrusion", "surface_normal")).lower()
    include_te = bool(chordwise.get("include_trailing_edge", False))
    xc = _chordwise_grid(transition_x, nle, dx, include_trailing_edge=include_te)
    xu, yu, alphau, xl, yl, alphal = _naca4_surface(m, p, thickness, xc)
    yn = _wall_normal_grid(float(first_spacing), yn_extent, ratio)
    if extrusion == "surface_normal":
        xprofu, yprofu, xprofl, yprofl = rot(xu, yu, xl, yl, yn, alphau, alphal)
    elif extrusion == "vertical":
        xprofu = np.broadcast_to(xu[None, :], (len(yn), len(xc))).copy()
        yprofu = yu[None, :] + yn[:, None]
        xprofl = np.broadcast_to(xl[None, :], (len(yn), len(xc))).copy()
        yprofl = yl[None, :] - yn[:, None]
    else:
        raise ValueError("mesh.extrusion must be 'surface_normal' or 'vertical'")
    x_pts = np.concatenate((
        xprofu.ravel(order="F"), xprofl[:, 1:].ravel(order="F"), -np.ones(len(yn)),
    ))
    y_pts = np.concatenate((
        yprofu.ravel(order="F"), yprofl[:, 1:].ravel(order="F"), yn,
    ))
    mesh = InterpolationMesh(
        x_pts=x_pts, y_pts=y_pts, kind="wall_normal_profile",
        capabilities=MeshCapabilities(profile_post=True, clcd=True),
        alphau=alphau, alphal=alphal, yn=yn, xc=xc, preset=preset,
    )
    validate_mesh(mesh)
    return mesh


def _axis_from_spec(axis: Mapping[str, Any], name: str) -> np.ndarray:
    try:
        return _colon(float(axis["start"]), float(axis["spacing"]), float(axis["stop"]))
    except KeyError as exc:
        raise ValueError(f"cartesian.{name} requires start, stop, and spacing") from exc


def _build_cartesian_mesh(spec: Mapping[str, Any], preset: Optional[str]) -> InterpolationMesh:
    cartesian = dict(spec.get("cartesian") or {})
    x = _axis_from_spec(dict(cartesian.get("x") or {}), "x")
    y = _axis_from_spec(dict(cartesian.get("y") or {}), "y")
    mesh = InterpolationMesh(
        x_pts=np.repeat(x, len(y)), y_pts=np.tile(y, len(x)), kind="cartesian",
        capabilities=MeshCapabilities(), preset=preset,
    )
    validate_mesh(mesh)
    return mesh


def build_mesh_from_spec(spec: Mapping[str, Any]) -> InterpolationMesh:
    """Build an :class:`InterpolationMesh` from the ``mesh`` YAML mapping."""

    merged, preset = _merged_spec(spec)
    kind = str(merged.get("kind", "wall_normal_profile")).lower().replace("-", "_")
    if kind in {"wall_normal_profile", "profile", "wing_profile"}:
        return _build_profile_mesh(merged, preset)
    if kind in {"cartesian", "farfield", "control_volume"}:
        return _build_cartesian_mesh(merged, preset)
    raise ValueError("mesh.kind must be wall_normal_profile or cartesian")


def build_mesh(profile_name: str = "NACA4412", tol: float = 0.5e-7):
    """Backward-compatible tuple API for the bundled compatibility meshes."""

    del tol
    name = str(profile_name).upper().replace("NACA", "")
    if name == "4412":
        mesh = build_mesh_from_spec({"preset": "naca4412_reference"})
    elif name == "0012":
        mesh = build_mesh_from_spec({"preset": "naca0012_reference"})
    else:
        raise ValueError(f"unsupported legacy profile_name {profile_name!r}")
    assert mesh.yn is not None and mesh.alphau is not None and mesh.alphal is not None and mesh.xc is not None
    return mesh.x_pts, mesh.y_pts, mesh.npoints, mesh.yn, mesh.alphau, mesh.alphal, mesh.xc


def _write_fortran_coordinate(path: Path, npoints: int, coordinate: Sequence[float]) -> None:
    data = np.asarray(coordinate, dtype="<f8").reshape(-1)
    if len(data) != npoints:
        raise ValueError("npoints does not match coordinate length")
    with path.open("wb") as handle:
        np.array([4], dtype="<i4").tofile(handle)
        np.array([npoints], dtype="<i4").tofile(handle)
        np.array([4], dtype="<i4").tofile(handle)
        marker = np.array([npoints * 8], dtype="<i4")
        marker.tofile(handle)
        data.tofile(handle)
        marker.tofile(handle)


def save_inter_mesh(path: Pathish, npoints: int, x_pts: Sequence[float], y_pts: Sequence[float]) -> None:
    """Write legacy ``<path>/ZSTAT/x.fort`` and ``y.fort`` files."""

    zstat = Path(path) / "ZSTAT"
    zstat.mkdir(parents=True, exist_ok=True)
    _write_fortran_coordinate(zstat / "x.fort", int(npoints), x_pts)
    _write_fortran_coordinate(zstat / "y.fort", int(npoints), y_pts)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_path(root: Path, name: str, label: str) -> Path:
    """Resolve an artifact name while keeping it below its output root."""

    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must be a relative path below the mesh output directory")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must be a path below the mesh output directory") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _plot_mesh(mesh: InterpolationMesh, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    x, y = np.asarray(mesh.x_pts), np.asarray(mesh.y_pts)
    if mesh.is_profile_mesh and mesh.yn is not None:
        x, y = x[:-len(mesh.yn)], y[:-len(mesh.yn)]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, ".", color="black", markersize=1)
    ax.set_aspect("equal", adjustable="box")
    ax.set(xlabel="X", ylabel="Y", title="Interpolation mesh")
    ax.grid(True, linewidth=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def write_mesh_artifacts(
    mesh: InterpolationMesh, output_dir: Pathish, *, spec: Optional[Mapping[str, Any]] = None,
    mat_file: str = "int_mesh.mat", write_figure: bool = True,
    figure_file: str = "mesh.png", manifest_file: str = "mesh_manifest.yml",
) -> MeshArtifacts:
    """Write a self-consistent ``.fort``, ``.mat``, figure, and manifest set."""

    validate_mesh(mesh)
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    save_inter_mesh(root, mesh.npoints, mesh.x_pts, mesh.y_pts)
    mat_path = _artifact_path(root, mat_file, "mat_file")
    sio.savemat(mat_path, mesh.mat_dict())
    figure_path = _artifact_path(root, figure_file, "figure_file") if write_figure else None
    if figure_path is not None:
        _plot_mesh(mesh, figure_path)
    x_path, y_path = root / "ZSTAT" / "x.fort", root / "ZSTAT" / "y.fort"
    non_marker = np.asarray(mesh.x_pts)[np.asarray(mesh.x_pts) != -1]
    encoded_spec = json.dumps(spec or {}, sort_keys=True, separators=(",", ":"), default=str)
    manifest = {
        "schema_version": 1,
        "kind": mesh.kind,
        "preset": mesh.preset,
        "capabilities": mesh.capabilities.as_dict(),
        "npoints": mesh.npoints,
        "bounds": {
            "x": [float(np.min(non_marker)), float(np.max(non_marker))],
            "y": [float(np.min(mesh.y_pts)), float(np.max(mesh.y_pts))],
        },
        "profile": None if not mesh.is_profile_mesh else {
            "stations": int(len(mesh.xc)), "wall_normal_points": int(len(mesh.yn)),
        },
        "spec_sha256": hashlib.sha256(encoded_spec.encode("utf-8")).hexdigest(),
        "artifacts": {
            "x_fort": str(x_path.relative_to(root)), "y_fort": str(y_path.relative_to(root)),
            "mat_file": str(mat_path.relative_to(root)),
            "figure": None if figure_path is None else str(figure_path.relative_to(root)),
            "sha256": {"x_fort": _sha256(x_path), "y_fort": _sha256(y_path), "mat_file": _sha256(mat_path)},
        },
    }
    manifest_path = _artifact_path(root, manifest_file, "manifest_file")
    with manifest_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False)
    return MeshArtifacts(root, x_path, y_path, mat_path, manifest_path, figure_path)


def validate_mesh(mesh: InterpolationMesh, *, require: Optional[str] = None) -> None:
    """Check coordinate layout and optional downstream capability requirements."""

    x = np.asarray(mesh.x_pts, dtype=float).reshape(-1)
    y = np.asarray(mesh.y_pts, dtype=float).reshape(-1)
    if len(x) == 0 or len(x) != len(y):
        raise ValueError("mesh coordinates must be non-empty arrays with equal lengths")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(x[x != -1])):
        raise ValueError("mesh contains non-finite coordinates")
    if require == "profile_post" and not mesh.capabilities.profile_post:
        raise ValueError(f"mesh kind {mesh.kind!r} does not support profile post-processing")
    if require == "clcd" and not mesh.capabilities.clcd:
        raise ValueError(f"mesh kind {mesh.kind!r} does not support Cl/Cd integration")
    if not mesh.is_profile_mesh:
        return
    if any(getattr(mesh, key) is None for key in ("alphau", "alphal", "yn", "xc")):
        raise ValueError("profile mesh requires alphau, alphal, yn, and xc metadata")
    yn, xc = np.asarray(mesh.yn, dtype=float).reshape(-1), np.asarray(mesh.xc, dtype=float).reshape(-1)
    alphau, alphal = np.asarray(mesh.alphau, dtype=float).reshape(-1), np.asarray(mesh.alphal, dtype=float).reshape(-1)
    if len(yn) < 2 or len(xc) < 2 or len(alphau) != len(xc) or len(alphal) != len(xc):
        raise ValueError("profile metadata dimensions are inconsistent")
    if not np.all(np.diff(yn) > 0) or not np.all(np.diff(xc) > 0):
        raise ValueError("profile yn and xc coordinates must be strictly increasing")
    if len(x) != 2 * len(yn) * len(xc):
        raise ValueError("profile coordinate count does not match the expected packed layout")
    if not np.all(x[-len(yn):] == -1):
        raise ValueError("profile mesh must end with the legacy x_pts=-1 wall-normal marker")
