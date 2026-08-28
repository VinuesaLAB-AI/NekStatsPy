"""Blockwise Cl/Cd backward-convergence workflow.

This module divides selected Nek ``sts`` files into chronological averaging
blocks, reruns the Nek interpolation stage for each block, writes one
Python-computed Cl/Cd row per block, then constructs duration-weighted backward
averages over the latest block, latest two blocks, and so on.
"""

from __future__ import annotations

import csv
import copy
import math
import os
import re
import shutil
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.io as sio

from .diagnostic import trapezoidal2
from .nek5000 import (
    RunPaths,
    StsFile,
    build_solver,
    load_config,
    patch_rea_params,
    prepare_run,
    run_nek,
)
from .nek5000 import _coerce_float, _repo_root, _resolve_path, _section
from .read_intfld import _R_bot, _R_top, _rot_ten, read_header

Pathish = Union[str, os.PathLike]
INT_FLD_HEADER_LIMIT = 1_000_000

__all__ = [
    "BackwardWindow",
    "ClCdBackwardPaths",
    "ClCdCoefficients",
    "IntFldHeader",
    "build_backward_windows",
    "compute_clcd_from_int_fld",
    "parse_int_fld_header",
    "run_clcd_backward_from_config",
]


@dataclass(frozen=True)
class BackwardWindow:
    """One short block-averaging Nek window."""

    window_index: int
    first_file: int
    last_file: int
    number_of_files: int
    start_time: float
    last_file_end_time: float
    first_sample_count: Optional[int] = None
    last_sample_count: Optional[int] = None

    @property
    def duration(self) -> float:
        return self.last_file_end_time - self.start_time


@dataclass(frozen=True)
class IntFldHeader:
    """Validated metadata from the binary ``int_fld`` header."""

    reynolds_number: float
    nstat: int
    nderiv: int
    start_time: float
    end_time: float
    effective_average_time: float
    time_step: float
    nrec: int
    time_interval: float
    npoints: int
    byte_order: str


@dataclass(frozen=True)
class ClCdCoefficients:
    """Aerodynamic coefficients extracted from one ``int_fld`` file."""

    reynolds_number: float
    aoa_deg: float
    pinf: float
    cl: float
    cd: float
    cd_p: float
    cd_f: float
    l_over_d: float


@dataclass(frozen=True)
class ClCdBackwardPaths:
    """Important files from a backward Cl/Cd workflow run."""

    root: Path
    config: Path
    manifest: Path
    csv: Path
    block_csv: Path
    int_fld_dir: Path
    log_dir: Path
    windows: Tuple[BackwardWindow, ...]


@dataclass(frozen=True)
class _BlockAverage:
    window: BackwardWindow
    header: IntFldHeader
    coeffs: ClCdCoefficients
    status: str
    int_fld_path: Path

    @property
    def duration(self) -> float:
        return self.header.end_time - self.header.start_time


def run_clcd_backward_from_config(
    config_path: Pathish,
    *,
    build: Optional[bool] = None,
    rebuild: bool = False,
    prepare_only: bool = False,
    skip_nek: bool = False,
    force: Optional[bool] = None,
    discard_int_fld: Optional[bool] = None,
) -> ClCdBackwardPaths:
    """Run the Python-only backward Cl/Cd workflow described by a YAML config."""

    config_path = Path(config_path).resolve()
    conf = load_config(config_path)
    base_dir = config_path.parent
    build_conf = _section(conf, "build")
    do_build = bool(build_conf.get("enabled", False)) if build is None else build
    if do_build:
        built = build_solver(conf, base_dir=base_dir, rebuild=rebuild)
        conf = copy.deepcopy(dict(conf))
        case_conf = _section(conf, "case")
        case_conf["executable"] = str(built.installed_executable)
        conf["case"] = case_conf

    clcd_conf = _section(conf, "clcd_backward")
    if not clcd_conf:
        raise ValueError("clcd_backward section is required for this workflow")
    csv_name = _clcd_csv_name(conf, clcd_conf, key="backward_csv", suffix="_clcd_backward")
    block_csv_name = _clcd_csv_name(conf, clcd_conf, key="block_csv", suffix="_clcd_blocks")
    work_conf = _backward_run_config(conf, clcd_conf, base_dir)
    run_paths = prepare_run(work_conf, base_dir=base_dir)

    block_size_files = _block_size_from_config(clcd_conf)
    include_incomplete_block = bool(clcd_conf.get("include_incomplete_block", False))
    block_mode = str(clcd_conf.get("block_mode", "fixed_files"))
    windows = build_backward_windows(
        run_paths.staged_stats,
        block_size_files=block_size_files,
        include_incomplete_block=include_incomplete_block,
        block_mode=block_mode,
        nek_conf=_section(work_conf, "nek"),
    )

    paths = _backward_paths(run_paths, windows, csv_name, block_csv_name)
    paths.int_fld_dir.mkdir(parents=True, exist_ok=True)
    paths.log_dir.mkdir(parents=True, exist_ok=True)

    requested_force = bool(clcd_conf.get("force", False)) if force is None else force
    requested_discard = (
        bool(clcd_conf.get("discard_int_fld", False))
        if discard_int_fld is None
        else discard_int_fld
    )

    if prepare_only:
        write_manifest(paths.manifest, windows, paths.int_fld_dir, paths.log_dir, {}, {})
        return paths

    mesh_file, aoa_deg = _clcd_mesh_settings(work_conf, paths)
    block_rows: List[Dict[str, Any]] = []
    block_averages: List[_BlockAverage] = []

    # Publish a readable report before Nek starts.  Each replacement below is
    # atomic, so an observer sees either the prior complete report or the next
    # complete report, never a partially written CSV.
    _atomic_write_csv(paths.block_csv, _block_csv_fields(), block_rows)

    def record_completed_block(
        window: BackwardWindow,
        status: str,
        header: IntFldHeader,
        int_fld_path: Path,
    ) -> None:
        block = _block_average_from_int_fld(
            window,
            header,
            status,
            int_fld_path,
            mesh_file,
            aoa_deg,
        )
        block_averages.append(block)
        block_rows.append(_block_row(block))
        _atomic_write_csv(paths.block_csv, _block_csv_fields(), block_rows)

    if skip_nek:
        statuses, headers = validate_existing_windows(windows, paths.int_fld_dir)
        for window in windows:
            int_fld_path = paths.int_fld_dir / f"int_fld_{window_stem(window)}"
            record_completed_block(
                window,
                statuses[window.window_index],
                headers[window.window_index],
                int_fld_path,
            )
    else:
        statuses, headers = run_nek_windows(
            work_conf,
            run_paths,
            windows,
            paths.int_fld_dir,
            paths.log_dir,
            force=requested_force,
            on_completed=record_completed_block,
        )
    write_manifest(paths.manifest, windows, paths.int_fld_dir, paths.log_dir, statuses, headers)
    write_clcd_tables(
        work_conf,
        paths,
        windows,
        statuses,
        headers,
        block_rows=block_rows,
        block_averages=block_averages,
    )

    if requested_discard:
        discard_int_fld_files(windows, paths.int_fld_dir)

    return paths


def build_backward_windows(
    staged_stats: Sequence[StsFile],
    *,
    block_size_files: Optional[int] = None,
    include_incomplete_block: bool = False,
    block_mode: str = "fixed_files",
    stride: Optional[int] = None,
    nek_conf: Mapping[str, Any],
) -> Tuple[BackwardWindow, ...]:
    """Create chronological block windows from staged, contiguous ``sts`` metadata.

    ``block_mode="accumulator_reset"`` starts a new block whenever the ``sts``
    header sample count drops. ``stride`` is accepted as a compatibility alias
    for the old config name, but the value now means block size in fixed mode.
    """

    if not staged_stats:
        raise ValueError("Cannot build backward windows without staged sts files")
    mode = _normalize_block_mode(block_mode)
    if block_size_files is None:
        if stride is not None:
            block_size_files = stride
        elif mode == "fixed_files":
            raise ValueError("clcd_backward.block_size_files must be a positive integer")
        else:
            block_size_files = 1
    elif stride is not None and int(stride) != int(block_size_files):
        raise ValueError(
            "clcd_backward.block_size_files and legacy clcd_backward.stride disagree"
        )
    block_size_files = _positive_int(
        block_size_files,
        "clcd_backward.block_size_files",
    )

    stats = sorted(staged_stats, key=lambda item: item.index)
    indices = [item.index for item in stats]
    expected = list(range(indices[0], indices[-1] + 1))
    if indices != expected:
        missing = sorted(set(expected) - set(indices))
        preview = ", ".join(str(index) for index in missing[:12])
        suffix = " ..." if len(missing) > 12 else ""
        raise ValueError(f"staged sts files are not contiguous; missing {preview}{suffix}")

    if mode == "accumulator_reset":
        stat_blocks = _accumulator_reset_blocks(stats)
    else:
        stat_blocks = _fixed_file_blocks(
            stats,
            block_size_files=block_size_files,
            include_incomplete_block=include_incomplete_block,
        )

    windows: List[BackwardWindow] = []
    previous_sample_time_step: Optional[float] = None
    for block in stat_blocks:
        _validate_block_time_order(block)
        first_stat = block[0]
        last_stat = block[-1]
        start_time = _block_start_time(
            block,
            nek_conf,
            reset_aware=mode == "accumulator_reset",
            fallback_sample_time_step=previous_sample_time_step,
        )
        if start_time >= last_stat.end_time:
            raise ValueError(
                f"invalid start time {start_time:.16g} for block starting at "
                f"file {first_stat.index}; its last header end time is "
                f"{last_stat.end_time:.16g}"
            )
        windows.append(
            BackwardWindow(
                window_index=len(windows) + 1,
                first_file=first_stat.index,
                last_file=last_stat.index,
                number_of_files=len(block),
                start_time=start_time,
                last_file_end_time=last_stat.end_time,
                first_sample_count=first_stat.sample_count,
                last_sample_count=last_stat.sample_count,
            )
        )
        block_sample_time_step = _sample_time_step_from_block(block)
        if block_sample_time_step is not None:
            previous_sample_time_step = block_sample_time_step
    if not windows:
        raise ValueError(
            "No complete Cl/Cd averaging blocks were created. Reduce "
            "clcd_backward.block_size_files or set "
            "clcd_backward.include_incomplete_block: true."
        )
    return tuple(windows)


def _normalize_block_mode(value: str) -> str:
    mode = str(value).strip().lower().replace("-", "_")
    aliases = {
        "fixed": "fixed_files",
        "fixed_file": "fixed_files",
        "fixed_files": "fixed_files",
        "files": "fixed_files",
        "accumulator": "accumulator_reset",
        "accumulator_reset": "accumulator_reset",
        "sample_reset": "accumulator_reset",
        "nrec_reset": "accumulator_reset",
        "reset": "accumulator_reset",
        "adaptive": "accumulator_reset",
        "auto": "accumulator_reset",
    }
    try:
        return aliases[mode]
    except KeyError as exc:
        raise ValueError(
            "clcd_backward.block_mode must be 'fixed_files' or 'accumulator_reset'"
        ) from exc


def _fixed_file_blocks(
    stats: Sequence[StsFile],
    *,
    block_size_files: int,
    include_incomplete_block: bool,
) -> List[Sequence[StsFile]]:
    blocks: List[Sequence[StsFile]] = []
    for offset in range(0, len(stats), block_size_files):
        block = stats[offset:offset + block_size_files]
        if len(block) < block_size_files and not include_incomplete_block:
            break
        blocks.append(block)
    return blocks


def _accumulator_reset_blocks(stats: Sequence[StsFile]) -> List[Sequence[StsFile]]:
    missing = [stat.index for stat in stats if stat.sample_count is None]
    if missing:
        preview = ", ".join(str(index) for index in missing[:12])
        suffix = " ..." if len(missing) > 12 else ""
        raise ValueError(
            "clcd_backward.block_mode: accumulator_reset requires the sts header "
            f"sample count; missing for file(s) {preview}{suffix}"
        )

    blocks: List[List[StsFile]] = []
    current: List[StsFile] = []
    for stat in stats:
        if current and stat.sample_count < current[-1].sample_count:
            blocks.append(current)
            current = [stat]
        else:
            current.append(stat)
    if current:
        blocks.append(current)
    return blocks


def _validate_block_time_order(block: Sequence[StsFile]) -> None:
    for previous, current in zip(block, block[1:]):
        if current.end_time <= previous.end_time:
            raise ValueError(
                "sts end times must be strictly increasing within each Cl/Cd "
                f"block; file {previous.index} ends at {previous.end_time:.16g}, "
                f"file {current.index} at {current.end_time:.16g}"
            )


def _block_start_time(
    block: Sequence[StsFile],
    nek_conf: Mapping[str, Any],
    *,
    reset_aware: bool,
    fallback_sample_time_step: Optional[float],
) -> float:
    first = block[0]
    if not reset_aware:
        return _window_start_time(first, nek_conf)

    if _predecessor_is_same_accumulator(first):
        return float(first.predecessor_end_time)

    if first.predecessor_end_time is None:
        initial = nek_conf.get("initial_start_time")
        if initial not in (None, ""):
            return _coerce_float(initial, "nek.initial_start_time")

    inferred = _infer_accumulator_start_time(
        block,
        fallback_sample_time_step=fallback_sample_time_step,
    )
    if inferred is not None:
        return inferred

    if first.predecessor_end_time is None:
        return _window_start_time(first, nek_conf)

    raise ValueError(
        "Cannot infer the start time for reset-detected Cl/Cd block "
        f"{first.index}-{block[-1].index}. Add another sts file from the same "
        "accumulator cycle, set clcd_backward.allow_time_overlap_between_blocks: "
        "false to require strict continuity, or provide a larger source window."
    )


def _predecessor_is_same_accumulator(stat: StsFile) -> bool:
    if stat.predecessor_end_time is None:
        return False
    if stat.sample_count is None or stat.predecessor_sample_count is None:
        return True
    return stat.sample_count > stat.predecessor_sample_count


def _infer_accumulator_start_time(
    block: Sequence[StsFile],
    *,
    fallback_sample_time_step: Optional[float],
) -> Optional[float]:
    first = block[0]
    if first.sample_count is None or first.sample_count <= 0:
        return None
    sample_time_step = _sample_time_step_from_block(block) or fallback_sample_time_step
    if sample_time_step is None or sample_time_step <= 0:
        return None
    return first.end_time - first.sample_count * sample_time_step


def _sample_time_step_from_block(block: Sequence[StsFile]) -> Optional[float]:
    for previous, current in zip(block, block[1:]):
        if previous.sample_count is None or current.sample_count is None:
            continue
        sample_delta = current.sample_count - previous.sample_count
        time_delta = current.end_time - previous.end_time
        if sample_delta > 0 and time_delta > 0:
            return time_delta / sample_delta
    return None


def run_nek_windows(
    conf: Mapping[str, Any],
    paths: RunPaths,
    windows: Sequence[BackwardWindow],
    int_fld_dir: Path,
    log_dir: Path,
    *,
    force: bool = False,
    on_completed: Optional[Callable[[BackwardWindow, str, IntFldHeader, Path], None]] = None,
) -> Tuple[Dict[int, str], Dict[int, IntFldHeader]]:
    """Generate or reuse one ``int_fld`` per backward window.

    ``on_completed`` runs after each validated result is available at its final
    destination, including reused results.  It is intentionally called before
    the next Nek block starts so callers can publish incremental diagnostics.
    """

    case_conf = _section(conf, "case")
    session = str(case_conf.get("session", "case"))
    rea_path = paths.root / str(case_conf.get("rea", f"{session}.rea"))
    if not rea_path.is_file():
        raise FileNotFoundError(f"Nek .rea file is missing: {rea_path}")

    statuses: Dict[int, str] = {}
    headers: Dict[int, IntFldHeader] = {}
    for window in windows:
        stem = window_stem(window)
        destination = int_fld_dir / f"int_fld_{stem}"
        destination_log = log_dir / f"nek_{stem}.log"

        if destination.exists() and not force:
            headers[window.window_index] = validate_int_fld_for_window(destination, window)
            statuses[window.window_index] = "reused"
            if on_completed:
                on_completed(
                    window,
                    statuses[window.window_index],
                    headers[window.window_index],
                    destination,
                )
            continue

        patch_rea_params(
            rea_path,
            {
                68: window.last_file,
                69: window.first_file,
                70: window.start_time,
            },
        )
        try:
            run_nek(conf, paths)
        finally:
            if paths.logfile.is_file():
                shutil.copy2(paths.logfile, destination_log)

        generated = paths.zstat / "int_fld"
        header = validate_int_fld_for_window(generated, window)
        os.replace(generated, destination)
        headers[window.window_index] = header
        statuses[window.window_index] = "generated"
        if on_completed:
            on_completed(window, statuses[window.window_index], header, destination)

    return statuses, headers


def validate_existing_windows(
    windows: Sequence[BackwardWindow],
    int_fld_dir: Path,
) -> Tuple[Dict[int, str], Dict[int, IntFldHeader]]:
    """Validate previously generated window files for a Nek-skipped run."""

    statuses: Dict[int, str] = {}
    headers: Dict[int, IntFldHeader] = {}
    for window in windows:
        path = int_fld_dir / f"int_fld_{window_stem(window)}"
        if not path.is_file():
            raise FileNotFoundError(f"Missing reusable int_fld for window: {path}")
        headers[window.window_index] = validate_int_fld_for_window(path, window)
        statuses[window.window_index] = "reused"
    return statuses, headers


def parse_int_fld_header(path: Pathish) -> IntFldHeader:
    """Read and validate the binary metadata header from an ``int_fld`` file."""

    path = Path(path)
    failures: List[str] = []
    for byte_order in ("<", ">"):
        try:
            with path.open("rb") as handle:
                header_text = _read_record_from_handle(handle, byte_order)
                if not header_text.startswith(b"(Re ="):
                    raise ValueError("unexpected int_fld text header")
                values = _read_record_from_handle(handle, byte_order)
            header = _parse_binary_header(values, byte_order)
            return header
        except (OSError, struct.error, ValueError) as exc:
            failures.append(f"{byte_order}: {exc}")
    raise ValueError(f"{path}: invalid int_fld header ({'; '.join(failures)})")


def validate_int_fld_for_window(path: Pathish, window: BackwardWindow) -> IntFldHeader:
    """Validate one ``int_fld`` header against the requested time window."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing int_fld file: {path}")
    header = parse_int_fld_header(path)
    tolerance = max(1.0e-10, abs(window.start_time) * 1.0e-10)
    if abs(header.start_time - window.start_time) > tolerance:
        raise ValueError(
            f"{path}: header start time {header.start_time:.16g} does not match "
            f"window start {window.start_time:.16g}"
        )
    if abs(header.end_time - window.last_file_end_time) > tolerance:
        raise ValueError(
            f"{path}: header end time {header.end_time:.16g} does not match "
            f"last sts end {window.last_file_end_time:.16g}"
        )
    if header.nstat < 4 or header.nderiv < 4:
        raise ValueError(
            f"{path}: Cl/Cd requires F4 and D1-D4, but header reports "
            f"nstat={header.nstat}, nderiv={header.nderiv}"
        )
    return header


def compute_clcd_from_int_fld(
    int_fld_path: Pathish,
    mesh_file: Pathish,
    *,
    aoa_deg: float = 5.0,
) -> ClCdCoefficients:
    """Compute Cl/Cd from only ``F4`` and ``D1``-``D4`` records."""

    mesh = sio.loadmat(mesh_file)
    fields, header = _read_minimal_fields(Path(int_fld_path), mesh)
    x_pts = np.asarray(mesh["x_pts"], float).ravel()
    y_pts = np.asarray(mesh["y_pts"], float).ravel()
    alphau = np.asarray(mesh["alphau"], float).ravel()
    alphal = np.asarray(mesh["alphal"], float).ravel()

    npoints_total = len(x_pts)
    ln = int(np.count_nonzero(x_pts == -1))
    if ln <= 0 or npoints_total % ln != 0:
        raise ValueError("mesh x_pts does not define a valid wall-normal layout")
    npf = npoints_total // ln // 2
    if len(alphau) != npf or len(alphal) != npf:
        raise ValueError("mesh angle arrays do not match int_fld profile count")

    top_cols = np.arange(0, npf)
    bot_cols = np.concatenate(([0], np.arange(npf, 2 * npf - 1)))

    pressure = fields["F4"]
    d_udx = fields["D1"]
    d_udy = fields["D2"]
    d_vdx = fields["D3"]
    d_vdy = fields["D4"]
    zeros = np.zeros_like(pressure)
    gradients = np.array(
        [
            [d_udx, d_udy, zeros],
            [d_vdx, d_vdy, zeros],
            [zeros, zeros, zeros],
        ]
    )
    top_grad = _rot_ten(_R_top(alphau), gradients[:, :, :, top_cols])
    bot_grad = _rot_ten(_R_bot(alphal), gradients[:, :, :, bot_cols])

    rho = 1.0
    mu = rho / header.reynolds_number
    pinf = float(np.max(pressure) - 0.5)

    tau_sn_t = mu * (top_grad[0, 1, 0, :] + top_grad[1, 0, 0, :])
    tau_sn_b = mu * (bot_grad[0, 1, 0, :] + bot_grad[1, 0, 0, :])
    pressure_top = pressure[0, top_cols] - pinf
    pressure_bot = pressure[0, bot_cols] - pinf

    top_surface = ln * np.arange(npf)
    bottom_surface = np.empty(npf, dtype=int)
    bottom_surface[0] = 0
    for index in range(npf - 1):
        bottom_surface[index + 1] = ln * index + npf * ln

    normal_p_top = -trapezoidal2(x_pts[top_surface], y_pts[top_surface], pressure_top, alphau, 1)
    normal_p_bot = +trapezoidal2(x_pts[bottom_surface], y_pts[bottom_surface], pressure_bot, alphal, 1)
    normal_tw_top = trapezoidal2(x_pts[top_surface], y_pts[top_surface], tau_sn_t, alphau, 2)
    normal_tw_bot = trapezoidal2(x_pts[bottom_surface], y_pts[bottom_surface], tau_sn_b, alphal, 2)
    axial_p_top = trapezoidal2(x_pts[top_surface], y_pts[top_surface], pressure_top, alphau, 3)
    axial_p_bot = -trapezoidal2(x_pts[bottom_surface], y_pts[bottom_surface], pressure_bot, alphal, 3)
    axial_tw_top = trapezoidal2(x_pts[top_surface], y_pts[top_surface], tau_sn_t, alphau, 4)
    axial_tw_bot = trapezoidal2(x_pts[bottom_surface], y_pts[bottom_surface], tau_sn_b, alphal, 4)

    normal_pressure = normal_p_top + normal_p_bot
    normal_skin = normal_tw_top + normal_tw_bot
    axial_pressure = axial_p_top + axial_p_bot
    axial_skin = axial_tw_top + axial_tw_bot
    normal_total = normal_pressure + normal_skin
    axial_total = axial_pressure + axial_skin

    ca, sa = math.cos(math.radians(aoa_deg)), math.sin(math.radians(aoa_deg))
    cl = 2 * (normal_total * ca - axial_total * sa)
    cd = 2 * (normal_total * sa + axial_total * ca)
    cd_p = 2 * (normal_pressure * sa + axial_pressure * ca)
    cd_f = 2 * (normal_skin * sa + axial_skin * ca)
    return ClCdCoefficients(
        reynolds_number=header.reynolds_number,
        aoa_deg=aoa_deg,
        pinf=pinf,
        cl=float(cl),
        cd=float(cd),
        cd_p=float(cd_p),
        cd_f=float(cd_f),
        l_over_d=float(cl / cd),
    )


def write_manifest(
    manifest_path: Path,
    windows: Sequence[BackwardWindow],
    int_fld_dir: Path,
    log_dir: Path,
    statuses: Mapping[int, str],
    headers: Mapping[int, IntFldHeader],
) -> None:
    """Write the per-block Nek manifest."""

    fieldnames = (
        "block_index",
        "first_file",
        "last_file",
        "number_of_files",
        "first_sts_sample_count",
        "last_sts_sample_count",
        "start_time",
        "end_time",
        "duration",
        "int_fld_path",
        "nek_log_path",
        "status",
        "time_interval",
        "effective_average_time",
        "nrec",
        "npoints",
    )
    rows = []
    for window in windows:
        stem = window_stem(window)
        header = headers.get(window.window_index)
        rows.append(
            {
                "block_index": window.window_index,
                "first_file": window.first_file,
                "last_file": window.last_file,
                "number_of_files": window.number_of_files,
                "first_sts_sample_count": _optional_int(window.first_sample_count),
                "last_sts_sample_count": _optional_int(window.last_sample_count),
                "start_time": format(window.start_time, ".16g"),
                "end_time": format(window.last_file_end_time, ".16g"),
                "duration": format(window.duration, ".16g"),
                "int_fld_path": str((int_fld_dir / f"int_fld_{stem}").resolve()),
                "nek_log_path": str((log_dir / f"nek_{stem}.log").resolve()),
                "status": statuses.get(window.window_index, "planned"),
                "time_interval": format(header.time_interval, ".16g") if header else "",
                "effective_average_time": (
                    format(header.effective_average_time, ".16g") if header else ""
                ),
                "nrec": header.nrec if header else "",
                "npoints": header.npoints if header else "",
            }
        )
    _atomic_write_csv(manifest_path, fieldnames, rows)


def write_clcd_tables(
    conf: Mapping[str, Any],
    paths: ClCdBackwardPaths,
    windows: Sequence[BackwardWindow],
    statuses: Mapping[int, str],
    headers: Mapping[int, IntFldHeader],
    *,
    block_rows: Optional[Sequence[Mapping[str, Any]]] = None,
    block_averages: Optional[Sequence[_BlockAverage]] = None,
) -> Path:
    """Compute block Cl/Cd rows and duration-weighted backward averages."""

    if (block_rows is None) != (block_averages is None):
        raise ValueError("block_rows and block_averages must be provided together")
    if block_rows is None:
        mesh_file, aoa_deg = _clcd_mesh_settings(conf, paths)
        block_rows, block_averages = _compute_block_rows(
            windows,
            paths,
            statuses,
            headers,
            mesh_file,
            aoa_deg,
        )
    _atomic_write_csv(paths.block_csv, _block_csv_fields(), block_rows)
    backward_rows = _compute_backward_rows(block_averages)
    _atomic_write_csv(paths.csv, _backward_csv_fields(), backward_rows)
    return paths.csv


def write_clcd_csv(
    conf: Mapping[str, Any],
    paths: ClCdBackwardPaths,
    windows: Sequence[BackwardWindow],
    statuses: Mapping[int, str],
    headers: Mapping[int, IntFldHeader],
) -> Path:
    """Compatibility wrapper for the blockwise Cl/Cd CSV writer."""

    return write_clcd_tables(conf, paths, windows, statuses, headers)


def _compute_block_rows(
    windows: Sequence[BackwardWindow],
    paths: ClCdBackwardPaths,
    statuses: Mapping[int, str],
    headers: Mapping[int, IntFldHeader],
    mesh_file: Path,
    aoa_deg: float,
) -> Tuple[List[Dict[str, Any]], List[_BlockAverage]]:
    rows: List[Dict[str, Any]] = []
    averages: List[_BlockAverage] = []
    for window in windows:
        int_fld_path = paths.int_fld_dir / f"int_fld_{window_stem(window)}"
        header = headers.get(window.window_index) or validate_int_fld_for_window(
            int_fld_path,
            window,
        )
        block = _block_average_from_int_fld(
            window,
            header,
            statuses.get(window.window_index, ""),
            int_fld_path,
            mesh_file,
            aoa_deg,
        )
        averages.append(block)
        rows.append(_block_row(block))
    return rows, averages


def _clcd_mesh_settings(
    conf: Mapping[str, Any],
    paths: ClCdBackwardPaths,
) -> Tuple[Path, float]:
    clcd_conf = _section(conf, "clcd_backward")
    post_conf = _section(conf, "post")
    mesh = clcd_conf.get("mesh") or post_conf.get("mesh")
    if not mesh:
        raise ValueError("clcd_backward.mesh or post.mesh is required")
    mesh_file = _resolve_path(mesh, paths.root, _repo_root(paths.root))
    aoa_deg = float(clcd_conf.get("aoa_deg", post_conf.get("aoa_deg", 5.0)))
    return mesh_file, aoa_deg


def _block_average_from_int_fld(
    window: BackwardWindow,
    header: IntFldHeader,
    status: str,
    int_fld_path: Path,
    mesh_file: Path,
    aoa_deg: float,
) -> _BlockAverage:
    coeffs = compute_clcd_from_int_fld(int_fld_path, mesh_file, aoa_deg=aoa_deg)
    block = _BlockAverage(
        window=window,
        header=header,
        coeffs=coeffs,
        status=status,
        int_fld_path=int_fld_path,
    )
    if block.duration <= 0:
        raise ValueError(
            f"{int_fld_path}: non-positive block duration {block.duration:.16g}"
        )
    return block


def _block_csv_fields() -> Tuple[str, ...]:
    return (
        "block_index",
        "first_file",
        "last_file",
        "number_of_files",
        "first_sts_sample_count",
        "last_sts_sample_count",
        "start_time",
        "end_time",
        "duration",
        "time_interval",
        "effective_average_time",
        "nrec",
        "npoints",
        "status",
        "reynolds_number",
        "aoa_deg",
        "pinf",
        "Cl",
        "Cd",
        "Cd_p",
        "Cd_f",
        "L_over_D",
        "int_fld_path",
    )


def _backward_csv_fields() -> Tuple[str, ...]:
    return (
        "backward_index",
        "first_block",
        "last_block",
        "number_of_blocks",
        "first_file",
        "last_file",
        "number_of_files",
        "first_sts_sample_count",
        "last_sts_sample_count",
        "start_time",
        "end_time",
        "total_duration",
        "reynolds_number",
        "aoa_deg",
        "pinf",
        "Cl",
        "Cd",
        "Cd_p",
        "Cd_f",
        "L_over_D",
    )


def _block_row(block: _BlockAverage) -> Dict[str, Any]:
    window = block.window
    header = block.header
    coeffs = block.coeffs
    return {
        "block_index": window.window_index,
        "first_file": window.first_file,
        "last_file": window.last_file,
        "number_of_files": window.number_of_files,
        "first_sts_sample_count": _optional_int(window.first_sample_count),
        "last_sts_sample_count": _optional_int(window.last_sample_count),
        "start_time": format(window.start_time, ".16g"),
        "end_time": format(window.last_file_end_time, ".16g"),
        "duration": format(block.duration, ".16g"),
        "time_interval": format(header.time_interval, ".16g"),
        "effective_average_time": format(header.effective_average_time, ".16g"),
        "nrec": header.nrec,
        "npoints": header.npoints,
        "status": block.status,
        "reynolds_number": format(coeffs.reynolds_number, ".16g"),
        "aoa_deg": format(coeffs.aoa_deg, ".16g"),
        "pinf": format(coeffs.pinf, ".16g"),
        "Cl": format(coeffs.cl, ".16g"),
        "Cd": format(coeffs.cd, ".16g"),
        "Cd_p": format(coeffs.cd_p, ".16g"),
        "Cd_f": format(coeffs.cd_f, ".16g"),
        "L_over_D": format(coeffs.l_over_d, ".16g"),
        "int_fld_path": str(block.int_fld_path.resolve()),
    }


def _compute_backward_rows(blocks: Sequence[_BlockAverage]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for backward_index in range(1, len(blocks) + 1):
        selected = list(blocks[-backward_index:])
        rows.append(_backward_row(backward_index, selected))
    return rows


def _backward_row(backward_index: int, blocks: Sequence[_BlockAverage]) -> Dict[str, Any]:
    weights = np.asarray([block.duration for block in blocks], dtype=float)
    total_duration = float(weights.sum())
    if total_duration <= 0:
        raise ValueError("Cannot compute backward Cl/Cd average with non-positive duration")

    def avg(attr: str) -> float:
        values = np.asarray([getattr(block.coeffs, attr) for block in blocks], dtype=float)
        return float(np.dot(values, weights) / total_duration)

    first = blocks[0].window
    last = blocks[-1].window
    cl = avg("cl")
    cd = avg("cd")
    return {
        "backward_index": backward_index,
        "first_block": first.window_index,
        "last_block": last.window_index,
        "number_of_blocks": len(blocks),
        "first_file": first.first_file,
        "last_file": last.last_file,
        "number_of_files": sum(block.window.number_of_files for block in blocks),
        "first_sts_sample_count": _optional_int(first.first_sample_count),
        "last_sts_sample_count": _optional_int(last.last_sample_count),
        "start_time": format(first.start_time, ".16g"),
        "end_time": format(last.last_file_end_time, ".16g"),
        "total_duration": format(total_duration, ".16g"),
        "reynolds_number": format(avg("reynolds_number"), ".16g"),
        "aoa_deg": format(avg("aoa_deg"), ".16g"),
        "pinf": format(avg("pinf"), ".16g"),
        "Cl": format(cl, ".16g"),
        "Cd": format(cd, ".16g"),
        "Cd_p": format(avg("cd_p"), ".16g"),
        "Cd_f": format(avg("cd_f"), ".16g"),
        "L_over_D": format(cl / cd, ".16g"),
    }


def discard_int_fld_files(windows: Iterable[BackwardWindow], int_fld_dir: Path) -> None:
    """Remove stored per-block ``int_fld`` files after successful CSV export."""

    for window in windows:
        path = int_fld_dir / f"int_fld_{window_stem(window)}"
        if path.is_file():
            path.unlink()


def window_stem(window: BackwardWindow) -> str:
    return f"first{window.first_file:05d}_last{window.last_file:05d}"


def _backward_run_config(
    conf: Mapping[str, Any],
    clcd_conf: Mapping[str, Any],
    base_dir: Path,
) -> Dict[str, Any]:
    work_conf: Dict[str, Any] = copy.deepcopy(dict(conf))
    repo_root = _repo_root(base_dir)
    source_conf = _section(work_conf, "source")
    if "first_file" in clcd_conf:
        source_conf["first_file"] = clcd_conf["first_file"]
    if "last_file" in clcd_conf:
        source_conf["last_file"] = clcd_conf["last_file"]
    source_conf["allow_time_overlap"] = bool(
        clcd_conf.get("allow_time_overlap_between_blocks", True)
    )
    work_conf["source"] = source_conf

    nek_conf = _section(work_conf, "nek")
    nek_conf["sampling_start_time"] = "auto"
    if "initial_start_time" in clcd_conf:
        nek_conf["initial_start_time"] = clcd_conf["initial_start_time"]
    work_conf["nek"] = nek_conf

    run_conf = _section(work_conf, "run")
    case_conf = _section(work_conf, "case")
    session = str(case_conf.get("session", "case"))
    if clcd_conf.get("output_dir"):
        output_dir = _resolve_path(clcd_conf["output_dir"], base_dir, repo_root)
        run_conf["root"] = str(output_dir.parent)
        run_conf["name"] = output_dir.name
    else:
        base_name = str(run_conf.get("name") or f"{session}_run")
        run_conf["name"] = str(clcd_conf.get("run_name") or f"{base_name}_clcd_backward")
    work_conf["run"] = run_conf
    return work_conf


def _backward_paths(
    run_paths: RunPaths,
    windows: Sequence[BackwardWindow],
    csv_name: str,
    block_csv_name: str,
) -> ClCdBackwardPaths:
    return ClCdBackwardPaths(
        root=run_paths.root,
        config=run_paths.config,
        manifest=run_paths.root / "nek_windows.csv",
        csv=run_paths.root / csv_name,
        block_csv=run_paths.root / block_csv_name,
        int_fld_dir=run_paths.root / "int_fld",
        log_dir=run_paths.root / "logs",
        windows=tuple(windows),
    )


def _clcd_csv_name(
    conf: Mapping[str, Any],
    clcd_conf: Mapping[str, Any],
    *,
    key: str,
    suffix: str,
) -> str:
    explicit = clcd_conf.get(key)
    if explicit not in (None, ""):
        return _safe_filename_part(explicit)
    run_conf = _section(conf, "run")
    stem_source = run_conf.get("name")
    if stem_source in (None, ""):
        case_conf = _section(conf, "case")
        stem_source = (
            case_conf.get("id") or case_conf.get("name") or case_conf.get("session") or "case"
        )
    stem = _safe_filename_part(stem_source)
    stem_lower = stem.lower()
    for known_suffix in ("_clcd_backward", "_clcd_blocks"):
        if stem_lower.endswith(known_suffix):
            stem = stem[: -len(known_suffix)]
            break
    return f"{stem}{suffix}.csv"


def _safe_filename_part(value: Any) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("._-")
    return safe or "case"


def _optional_int(value: Optional[int]) -> Union[int, str]:
    return "" if value is None else int(value)


def _block_size_from_config(clcd_conf: Mapping[str, Any]) -> int:
    has_block_size = clcd_conf.get("block_size_files") not in (None, "")
    has_legacy_stride = clcd_conf.get("stride") not in (None, "")
    if has_block_size and has_legacy_stride:
        block_size = int(clcd_conf["block_size_files"])
        stride = int(clcd_conf["stride"])
        if block_size != stride:
            raise ValueError(
                "clcd_backward.block_size_files and legacy clcd_backward.stride disagree"
            )
        return _positive_int(block_size, "clcd_backward.block_size_files")
    if has_block_size:
        return _positive_int(clcd_conf["block_size_files"], "clcd_backward.block_size_files")
    if has_legacy_stride:
        return _positive_int(clcd_conf["stride"], "clcd_backward.stride")
    return 1


def _window_start_time(stat: StsFile, nek_conf: Mapping[str, Any]) -> float:
    raw = nek_conf.get("sampling_start_time", "auto")
    if raw not in (None, "", "auto"):
        raise ValueError(
            "Backward Cl/Cd block windows require nek.sampling_start_time: auto; "
            "use nek.initial_start_time only when the first predecessor sts file "
            "is unavailable."
        )
    if stat.predecessor_end_time is not None:
        return stat.predecessor_end_time
    initial = nek_conf.get("initial_start_time")
    if initial not in (None, ""):
        return _coerce_float(initial, "nek.initial_start_time")
    if bool(nek_conf.get("allow_first_file_end_time_as_start", False)):
        return stat.end_time
    raise ValueError(
        "Cannot infer backward Cl/Cd start time for first_file=1 because no "
        "predecessor sts file is available. Set nek.initial_start_time, start "
        "from clcd_backward.first_file > 1, or enable "
        "nek.allow_first_file_end_time_as_start."
    )


def _read_minimal_fields(path: Path, mesh: Mapping[str, Any]) -> Tuple[Dict[str, np.ndarray], IntFldHeader]:
    x_pts = np.asarray(mesh["x_pts"], float).ravel()
    npoints_total = len(x_pts)
    ln = int(np.count_nonzero(x_pts == -1))
    if ln <= 0 or npoints_total % ln != 0:
        raise ValueError("mesh x_pts does not define a valid wall-normal layout")
    two_np = npoints_total // ln
    header = parse_int_fld_header(path)
    wanted_positions = {
        4: "F4",
        header.nstat + 1: "D1",
        header.nstat + 2: "D2",
        header.nstat + 3: "D3",
        header.nstat + 4: "D4",
    }
    fields: Dict[str, np.ndarray] = {}
    with path.open("rb") as handle:
        read_header(handle)
        for position in range(1, header.nstat + header.nderiv + 1):
            if position != 1:
                handle.seek(8, os.SEEK_CUR)
            name = wanted_positions.get(position)
            if name:
                raw = np.fromfile(handle, np.float64, npoints_total)
                if raw.size != npoints_total:
                    raise ValueError(f"{path}: truncated field record {name}")
                fields[name] = raw.reshape(two_np, ln).T
            else:
                handle.seek(npoints_total * 8, os.SEEK_CUR)
    missing = sorted(set(wanted_positions.values()) - set(fields))
    if missing:
        raise ValueError(f"{path}: missing required field record(s): {', '.join(missing)}")
    return fields, header


def _read_record_from_handle(handle, byte_order: str) -> bytes:
    marker = handle.read(4)
    if len(marker) != 4:
        raise ValueError("missing Fortran record marker")
    (record_size,) = struct.unpack(f"{byte_order}I", marker)
    if not 0 < record_size <= INT_FLD_HEADER_LIMIT:
        raise ValueError(f"implausible record size {record_size}")
    payload = handle.read(record_size)
    trailing = handle.read(4)
    if len(payload) != record_size or len(trailing) != 4:
        raise ValueError("truncated Fortran record")
    (trailing_size,) = struct.unpack(f"{byte_order}I", trailing)
    if trailing_size != record_size:
        raise ValueError("mismatched Fortran record markers")
    return payload


def _parse_binary_header(values: bytes, byte_order: str) -> IntFldHeader:
    offset = 0

    def unpack(code: str) -> Tuple[Union[float, int], ...]:
        nonlocal offset
        size = struct.calcsize(f"{byte_order}{code}")
        if offset + size > len(values):
            raise ValueError("truncated int_fld binary header")
        result = struct.unpack_from(f"{byte_order}{code}", values, offset)
        offset += size
        return result

    reynolds_number = float(unpack("d")[0])
    unpack("3d")
    unpack("3i")
    unpack("3i")
    nstat = int(unpack("i")[0])
    nderiv = int(unpack("i")[0])
    start_time = float(unpack("d")[0])
    end_time = float(unpack("d")[0])
    effective_average_time = float(unpack("d")[0])
    time_step = float(unpack("d")[0])
    nrec = int(unpack("i")[0])
    time_interval = float(unpack("d")[0])
    npoints = int(unpack("i")[0])
    if offset != len(values):
        raise ValueError(f"unexpected {len(values) - offset} extra header byte(s)")
    if (
        reynolds_number <= 0
        or nstat <= 0
        or nderiv < 0
        or npoints <= 0
        or end_time <= start_time
        or time_interval <= 0
    ):
        raise ValueError("implausible int_fld header values")
    return IntFldHeader(
        reynolds_number=reynolds_number,
        nstat=nstat,
        nderiv=nderiv,
        start_time=start_time,
        end_time=end_time,
        effective_average_time=effective_average_time,
        time_step=time_step,
        nrec=nrec,
        time_interval=time_interval,
        npoints=npoints,
        byte_order="little" if byte_order == "<" else "big",
    )


def _positive_int(value: Any, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if result <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return result


def _atomic_write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        delete=False,
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)
