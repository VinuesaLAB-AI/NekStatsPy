"""Config-driven runner for KTH Framework/PSTAT2D channel cases."""

from __future__ import annotations

import copy
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import yaml

from .channel_pipeline import ChannelPostResult, run_channel_post_from_config
from .nek5000 import (
    BuildPaths,
    build_solver,
    load_config,
    read_sts_header,
)
from .nek5000 import (
    _build_command,
    _coerce_float,
    _discover_stats,
    _repo_root,
    _resolve_path,
    _resolve_under,
    _section,
    _stage_file,
)

Pathish = Union[str, os.PathLike]

_DEFAULT_CHANNEL_FILES = (
    "SIZE",
    "makefile_usr.inc",
    "tcf.usr",
    "tcf.par",
    "tcf.re2",
    "tcf.ma2",
    "inputs_phill_pp.in",
)
_DEFAULT_CHANNEL_DATA_FILES = ("DATA/int_pos", "DATA/c2Dtcf0.f00001")
_PAR_SECTION_RE = re.compile(r"^\s*\[(?P<section>[^\]]+)\]")
_PAR_VALUE_RE = re.compile(
    r"^(?P<prefix>\s*(?P<key>[A-Za-z0-9_]+)\s*=\s*)"
    r"(?P<value>[^#\n]*?)"
    r"(?P<suffix>\s*(?:#.*)?)$"
)


@dataclass(frozen=True)
class ChannelStsFile:
    """One source ``sts`` file staged into KTH ``DATA`` numbering."""

    source_index: int
    staged_index: int
    source: Path
    staged: Path
    end_time: float
    header: str
    predecessor_end_time: Optional[float]


@dataclass(frozen=True)
class ChannelRunPaths:
    """Important paths created for one KTH channel post-processing run."""

    root: Path
    data_dir: Path
    config: Path
    logfile: Path
    int_fld: Path
    mat_file: Optional[Path]
    figdir: Optional[Path]
    staged_stats: Tuple[ChannelStsFile, ...] = ()


def run_channel_from_config(
    config_path: Pathish,
    *,
    build: Optional[bool] = None,
    rebuild: bool = False,
    prepare_only: bool = False,
    skip_nek: bool = False,
    skip_post: bool = False,
) -> ChannelRunPaths:
    """Prepare and optionally run the KTH channel pipeline from YAML config."""

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

    paths = prepare_channel_run(conf, base_dir=base_dir)
    if prepare_only:
        return paths

    if not skip_nek:
        run_channel_nek(conf, paths)

    if not skip_post:
        run_channel_python_post(conf, paths)

    return paths


def build_channel_from_config(config_path: Pathish, *, rebuild: bool = False) -> BuildPaths:
    """Build the KTH channel executable described by a config."""

    config_path = Path(config_path).resolve()
    conf = load_config(config_path)
    return build_solver(conf, base_dir=config_path.parent, rebuild=rebuild)


def prepare_channel_run(
    conf: Mapping[str, Any],
    *,
    base_dir: Optional[Path] = None,
) -> ChannelRunPaths:
    """Create a run folder for a KTH PSTAT2D channel case."""

    base_dir = Path.cwd() if base_dir is None else Path(base_dir)
    repo_root = _repo_root(base_dir)
    run_conf = _section(conf, "run")
    case_conf = _section(conf, "case")
    pstat_conf = _section(conf, "pstat2d")
    post_conf = _section(conf, "channel_post")

    case_id = str(case_conf.get("id", "channel"))
    session = str(case_conf.get("session", "tcf"))
    run_name = str(run_conf.get("name") or f"{session}_channel")
    run_root = _resolve_path(run_conf.get("root", "runs"), base_dir, repo_root)
    run_dir = _resolve_under(run_name, run_root)
    data_dir = _resolve_under(pstat_conf.get("data_dir", "DATA"), run_dir)

    run_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    deps_dir = _resolve_path(
        case_conf.get("dependencies", f"data/dependencies/{case_id}"),
        base_dir,
        repo_root,
    )
    _stage_channel_case_files(conf, deps_dir, run_dir, session)
    _stage_channel_data_files(conf, deps_dir, run_dir)
    staged_stats = stage_channel_stats_files(conf, data_dir, session, base_dir, repo_root)
    _write_session(run_dir, session)
    _patch_channel_par_if_present(conf, run_dir, session, staged_stats)

    int_fld = _resolve_under(post_conf.get("int_fld", "int_fld"), run_dir)
    out_file = post_conf.get("out")
    mat_file = _resolve_under(out_file, run_dir) if out_file else None
    figdir = _resolve_under(post_conf["figdir"], run_dir) if post_conf.get("figdir") else None
    paths = ChannelRunPaths(
        root=run_dir,
        data_dir=data_dir,
        config=run_dir / "current_conf.yml",
        logfile=run_dir / str(_section(conf, "nek").get("logfile", "log01.txt")),
        int_fld=int_fld,
        mat_file=mat_file,
        figdir=figdir,
        staged_stats=tuple(staged_stats),
    )
    _write_resolved_config(conf, paths.config)
    return paths


def stage_channel_stats_files(
    conf: Mapping[str, Any],
    data_dir: Path,
    session: str,
    base_dir: Optional[Path] = None,
    repo_root: Optional[Path] = None,
) -> List[ChannelStsFile]:
    """Stage source ``sts`` files into contiguous KTH ``DATA/sts<session>0.fNNNNN``."""

    source_conf = _section(conf, "source")
    source_path = source_conf.get("path")
    if not source_path:
        return []

    base_dir = Path.cwd() if base_dir is None else Path(base_dir)
    repo_root = _repo_root(base_dir) if repo_root is None else repo_root
    source_root = _resolve_path(source_path, base_dir, repo_root)
    pattern = str(source_conf.get("pattern") or f"res_*/sts_data/sts{session}0.f*")
    first_index = int(source_conf.get("first_file", 1))
    last_index = source_conf.get("last_file", "auto")
    if first_index < 1:
        raise ValueError("source.first_file must be >= 1")

    files = _discover_stats(source_root, pattern)
    if first_index > len(files):
        raise FileNotFoundError(
            f"source.first_file={first_index} exceeds the {len(files)} matched "
            f"stats file(s) from {source_root / pattern}"
        )
    if last_index not in (None, "", "auto"):
        last_index = int(last_index)
        if last_index < first_index:
            raise ValueError("source.last_file must be >= source.first_file")
        if last_index > len(files):
            raise FileNotFoundError(
                f"source.last_file={last_index} exceeds the {len(files)} matched "
                f"stats file(s) from {source_root / pattern}"
            )
        selected = files[first_index - 1:last_index]
    else:
        selected = files[first_index - 1:]

    if not selected:
        raise FileNotFoundError(
            f"No stats files matched {source_root / pattern}. "
            "Set source.path/source.pattern, or omit source.path for prepare-only runs."
        )

    prefix = str(source_conf.get("target_prefix") or f"sts{session}0.f")
    stage_mode = str(source_conf.get("stage_mode", "symlink")).lower()
    overwrite = bool(source_conf.get("overwrite", False))
    previous_end_time = read_sts_header(files[first_index - 2])[1] if first_index > 1 else None

    staged: List[ChannelStsFile] = []
    for staged_index, (source_index, src) in enumerate(
        enumerate(selected, start=first_index),
        start=1,
    ):
        header, end_time = read_sts_header(src)
        if previous_end_time is not None and end_time <= previous_end_time:
            raise ValueError(
                "sts end times are not strictly increasing: "
                f"file {source_index - 1} ends at {previous_end_time:.16g}, "
                f"file {source_index} at {end_time:.16g}"
            )
        dst = data_dir / f"{prefix}{staged_index:05d}"
        _stage_file(src, dst, mode=stage_mode, overwrite=overwrite)
        staged.append(
            ChannelStsFile(
                source_index=source_index,
                staged_index=staged_index,
                source=src,
                staged=dst,
                end_time=end_time,
                header=header,
                predecessor_end_time=previous_end_time,
            )
        )
        previous_end_time = end_time
    return staged


def run_channel_nek(conf: Mapping[str, Any], paths: ChannelRunPaths) -> None:
    """Run the staged KTH channel Nek executable inside the run directory."""

    nek_conf = _section(conf, "nek")
    session = str(_section(conf, "case").get("session", "tcf"))
    _write_session(paths.root, session)
    _prepare_channel_run_state(nek_conf, paths)

    np = int(nek_conf.get("np", 1))
    executable = str(nek_conf.get("executable_name", "./nek5000"))
    mpirun = str(nek_conf.get("mpirun", "mpirun"))
    cmd = _build_command([mpirun, "-n", str(np), executable], nek_conf.get("env_script"))
    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in _section(nek_conf, "env").items()})

    with open(paths.logfile, "w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            cwd=str(paths.root),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"KTH channel Nek failed with exit code {proc.returncode}; see {paths.logfile}")
    _validate_channel_nek_run(nek_conf, paths)


def run_channel_python_post(
    conf: Mapping[str, Any],
    paths: ChannelRunPaths,
) -> ChannelPostResult:
    """Run the Python channel ``int_fld -> data .mat`` post stage."""

    return run_channel_post_from_config(
        conf,
        base_dir=_repo_root(paths.root),
        run_dir=paths.root,
    )


def patch_channel_par(path: Pathish, section: str, params: Mapping[str, Any]) -> None:
    """Patch a Nek/KTH ``.par`` section while preserving comments."""

    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    target = section.strip().lower()
    remaining = {str(k).upper(): v for k, v in params.items()}
    patched: List[str] = []
    active = False
    inserted = False

    for line in lines:
        section_match = _PAR_SECTION_RE.match(line)
        if section_match:
            if active and remaining and not inserted:
                patched.extend(_format_par_insertions(remaining))
                remaining.clear()
                inserted = True
            active = section_match.group("section").strip().lower() == target
            patched.append(line)
            continue

        if active:
            value_match = _PAR_VALUE_RE.match(line.rstrip("\n"))
            if value_match:
                key = value_match.group("key").upper()
                if key in remaining:
                    newline = "\n" if line.endswith("\n") else ""
                    patched.append(
                        f"{value_match.group('prefix')}"
                        f"{_format_par_value(remaining.pop(key))}"
                        f"{value_match.group('suffix')}{newline}"
                    )
                    continue
        patched.append(line)

    if remaining:
        if not active and not inserted and f"[{section}]" not in "".join(lines):
            patched.append(f"\n[{section}]\n")
        patched.extend(_format_par_insertions(remaining))

    path.write_text("".join(patched), encoding="utf-8")


def _stage_channel_case_files(
    conf: Mapping[str, Any],
    deps_dir: Path,
    run_dir: Path,
    session: str,
) -> None:
    case_conf = _section(conf, "case")
    stage_mode = str(case_conf.get("stage_mode", "copy")).lower()
    overwrite = bool(case_conf.get("overwrite", False))
    files = list(case_conf.get("files") or _DEFAULT_CHANNEL_FILES)
    for name in files:
        _stage_file(deps_dir / str(name), run_dir / str(name), mode=stage_mode, overwrite=overwrite)

    executable = case_conf.get("executable")
    src = _resolve_path(executable, deps_dir, _repo_root(run_dir)) if executable else deps_dir / "nek5000"
    if not src.exists():
        src = deps_dir / f"nek5000_{case_conf.get('id', session)}"
    dst = run_dir / str(case_conf.get("run_executable", "nek5000"))
    _stage_file(
        src,
        dst,
        mode=str(case_conf.get("executable_stage_mode", stage_mode)),
        overwrite=overwrite,
    )
    try:
        dst.chmod(dst.stat().st_mode | 0o111)
    except FileNotFoundError:
        pass


def _stage_channel_data_files(conf: Mapping[str, Any], deps_dir: Path, run_dir: Path) -> None:
    pstat_conf = _section(conf, "pstat2d")
    stage_mode = str(pstat_conf.get("stage_mode", "copy")).lower()
    overwrite = bool(pstat_conf.get("overwrite", True))
    files = list(pstat_conf.get("static_files") or _DEFAULT_CHANNEL_DATA_FILES)
    for name in files:
        _stage_file(deps_dir / str(name), run_dir / str(name), mode=stage_mode, overwrite=overwrite)


def _patch_channel_par_if_present(
    conf: Mapping[str, Any],
    run_dir: Path,
    session: str,
    staged_stats: Sequence[ChannelStsFile],
) -> None:
    pstat_conf = _section(conf, "pstat2d")
    params: Dict[str, Any] = {}
    if staged_stats:
        params["STS_NFILE"] = len(staged_stats)
        params["STS_STIME"] = _channel_sampling_start_time(pstat_conf, staged_stats)
    elif pstat_conf.get("sts_nfile") not in (None, ""):
        params["STS_NFILE"] = pstat_conf["sts_nfile"]
    elif pstat_conf.get("sts_stime") not in (None, ""):
        params["STS_STIME"] = pstat_conf["sts_stime"]

    if pstat_conf.get("sts_nstep") not in (None, ""):
        params["STS_NSTEP"] = pstat_conf["sts_nstep"]
    if pstat_conf.get("c2d_fnum") not in (None, ""):
        params["C2D_FNUM"] = pstat_conf["c2d_fnum"]
    if pstat_conf.get("amr_nref") not in (None, ""):
        params["AMR_NREF"] = pstat_conf["amr_nref"]
    params.update({str(k).upper(): v for k, v in _section(pstat_conf, "par_params").items()})

    if not params:
        return
    par_name = str(_section(conf, "case").get("par", f"{session}.par"))
    par_path = run_dir / par_name
    if par_path.exists():
        patch_channel_par(par_path, str(pstat_conf.get("section", "_PSTAT2D")), params)


def _channel_sampling_start_time(
    pstat_conf: Mapping[str, Any],
    staged_stats: Sequence[ChannelStsFile],
) -> float:
    raw = pstat_conf.get("sts_stime", pstat_conf.get("sampling_start_time", "auto"))
    if raw not in (None, "", "auto"):
        return _coerce_float(raw, "pstat2d.sts_stime")

    predecessor = staged_stats[0].predecessor_end_time
    if predecessor is not None:
        return predecessor

    initial = pstat_conf.get("initial_start_time")
    if initial not in (None, ""):
        return _coerce_float(initial, "pstat2d.initial_start_time")

    if bool(pstat_conf.get("allow_first_file_end_time_as_start", False)):
        return staged_stats[0].end_time

    raise ValueError(
        "Cannot infer pstat2d.sts_stime for first_file=1 because no predecessor "
        "sts file is available. Set pstat2d.initial_start_time or explicitly "
        "enable pstat2d.allow_first_file_end_time_as_start."
    )


def _prepare_channel_run_state(nek_conf: Mapping[str, Any], paths: ChannelRunPaths) -> None:
    if bool(nek_conf.get("remove_existing_int_fld", True)):
        _safe_unlink(paths.int_fld)
    if bool(nek_conf.get("clear_sch", True)):
        for entry in paths.root.glob("*.sch*"):
            _safe_unlink(entry)
    if bool(nek_conf.get("clear_pstat_outputs", True)):
        for pattern in ("st1*.f*", "st2*.f*"):
            for entry in paths.root.glob(pattern):
                _safe_unlink(entry)
    if bool(nek_conf.get("remove_logfile", True)):
        _safe_unlink(paths.logfile)


def _validate_channel_nek_run(nek_conf: Mapping[str, Any], paths: ChannelRunPaths) -> None:
    log_text = ""
    if paths.logfile.exists():
        log_text = paths.logfile.read_text(encoding="utf-8", errors="replace")

    abort_marker = nek_conf.get("abort_marker", "ABORT")
    if abort_marker and abort_marker in log_text:
        raise RuntimeError(f"KTH channel Nek reported {abort_marker!r}; see {paths.logfile}")

    marker = nek_conf.get("completion_marker", "run successful")
    if marker and marker not in log_text:
        raise RuntimeError(
            f"KTH channel Nek log is missing completion marker {marker!r}; see {paths.logfile}"
        )
    if bool(nek_conf.get("require_int_fld", True)) and not paths.int_fld.is_file():
        raise RuntimeError(f"KTH channel Nek did not create {paths.int_fld}")


def _safe_unlink(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        raise ValueError(f"Refusing to remove directory or special file: {path}")


def _write_session(run_dir: Path, session: str) -> None:
    (run_dir / "SESSION.NAME").write_text(
        f"{session}\n{run_dir.resolve()}/\n",
        encoding="utf-8",
    )


def _write_resolved_config(conf: Mapping[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(conf), handle, sort_keys=False)


def _format_par_insertions(params: Mapping[str, Any]) -> List[str]:
    return [f"{key:<22}= {_format_par_value(value)}\n" for key, value in params.items()]


def _format_par_value(value: Any) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.12E}"
    text = str(value)
    try:
        return f"{float(text):.12E}"
    except ValueError:
        return text
