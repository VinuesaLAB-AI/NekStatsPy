"""Run-folder orchestration for the Nek5000 interpolation stage.

The numerical post-processing in :mod:`nekstat.pipeline` starts from
``ZSTAT/int_fld``.  This module prepares and optionally runs the small Nek5000
post-processing case that converts ordered ``sts`` files plus ``x.fort`` /
``y.fort`` interpolation points into that ``int_fld`` file.
"""

from __future__ import annotations

import glob
import math
import os
import re
import shlex
import shutil
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import yaml

from .naca_geom import build_mesh, save_inter_mesh
from .pipeline import run as run_post

__all__ = [
    "BuildPaths",
    "RunPaths",
    "StsFile",
    "build_from_config",
    "build_solver",
    "load_config",
    "prepare_run",
    "patch_rea_params",
    "read_sts_header",
    "run_nek",
    "run_from_config",
    "stage_stats_files",
]


_DEFAULT_CASE_FILES = ("SIZE", "makefile_usr.inc")
_REA_PARAM_RE = re.compile(r"^(\s*)([-+0-9.EeDd]+)(\s+p(?P<num>\d{3})\b.*)$")
Pathish = Union[str, os.PathLike]


@dataclass(frozen=True)
class RunPaths:
    """Important paths created for one Nek post-processing run."""

    root: Path
    zstat: Path
    config: Path
    logfile: Path
    stat_file: Optional[Path]


@dataclass(frozen=True)
class BuildPaths:
    """Important paths from a Nek5000 case build."""

    source_dir: Path
    work_dir: Path
    executable: Path
    installed_executable: Path
    logfile: Path


@dataclass(frozen=True)
class StsFile:
    """One source ``sts`` file staged into the run-local ``ZSTAT`` directory."""

    index: int
    source: Path
    staged: Path
    end_time: float
    header: str
    predecessor_end_time: Optional[float]


def load_config(path: Pathish) -> Dict[str, Any]:
    """Load a YAML run config and normalize empty files to an empty dict."""

    with open(path, "r", encoding="utf-8") as handle:
        conf = yaml.safe_load(handle) or {}
    if not isinstance(conf, dict):
        raise ValueError(f"Run config must be a mapping: {path}")
    return conf


def build_from_config(config_path: Pathish, *, rebuild: bool = False) -> BuildPaths:
    """Extract the solver archive if needed and build the configured Nek case."""

    config_path = Path(config_path).resolve()
    conf = load_config(config_path)
    return build_solver(conf, base_dir=config_path.parent, rebuild=rebuild)


def run_from_config(
    config_path: Pathish,
    *,
    build: Optional[bool] = None,
    rebuild: bool = False,
    prepare_only: bool = False,
    skip_nek: bool = False,
    skip_post: bool = False,
) -> RunPaths:
    """Prepare a run folder and optionally execute Nek + Python post-processing."""

    config_path = Path(config_path).resolve()
    conf = load_config(config_path)
    base_dir = config_path.parent
    build_conf = _section(conf, "build")
    do_build = bool(build_conf.get("enabled", False)) if build is None else build
    if do_build:
        built = build_solver(conf, base_dir=base_dir, rebuild=rebuild)
        conf = dict(conf)
        case_conf = _section(conf, "case")
        case_conf["executable"] = str(built.installed_executable)
        conf["case"] = case_conf
    paths = prepare_run(conf, base_dir=base_dir)

    if prepare_only:
        return paths

    if not skip_nek:
        run_nek(conf, paths)

    if not skip_post:
        run_python_post(conf, paths)

    return paths


def build_solver(
    conf: Mapping[str, Any],
    *,
    base_dir: Optional[Path] = None,
    rebuild: bool = False,
) -> BuildPaths:
    """Build the Nek5000 executable described by a run config.

    The solver archive is preserved and only extracted when ``build.source_dir``
    does not already exist.  The compile command runs inside ``build.work_dir``;
    if ``build.env_script`` is set, it is sourced in a Bash shell before the
    compile command so the local MPI wrappers are used.
    """

    base_dir = Path.cwd() if base_dir is None else Path(base_dir)
    repo_root = _repo_root(base_dir)
    case_conf = _section(conf, "case")
    build_conf = _section(conf, "build")

    archive = _resolve_path(build_conf.get("archive", "tmp/Nek1093_dong.tar.gz"), base_dir, repo_root)
    source_dir = _resolve_under(build_conf.get("source_dir", "tmp/Nek1093_dong"), repo_root)
    work_dir = _resolve_path(build_conf.get("work_dir", "tmp/01_NEK"), base_dir, repo_root)
    executable_name = str(build_conf.get("executable_name", "nek5000"))
    built_executable = work_dir / executable_name
    output = build_conf.get("output") or case_conf.get("executable")
    if not output:
        case_id = str(case_conf.get("id", "case"))
        output = f"data/dependencies/{case_id}/nek5000_{case_id}"
    installed_executable = _resolve_path(output, base_dir, repo_root)
    logfile = _resolve_under(build_conf.get("logfile", "build.log"), work_dir)

    if installed_executable.exists() and not rebuild and build_conf.get("skip_if_exists", True):
        return BuildPaths(source_dir, work_dir, built_executable, installed_executable, logfile)

    if not source_dir.exists():
        _extract_solver_archive(archive, source_dir.parent)
    if not source_dir.exists():
        raise FileNotFoundError(f"Solver source directory was not created: {source_dir}")

    _stage_build_template(conf, build_conf, work_dir, base_dir, repo_root)
    if not work_dir.exists():
        raise FileNotFoundError(f"Nek build work directory not found: {work_dir}")

    command = build_conf.get("command", "./01-compile-command")
    env_script = build_conf.get("env_script")
    cmd = _build_command(command, env_script)
    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in _section(build_conf, "env").items()})

    logfile.parent.mkdir(parents=True, exist_ok=True)
    with open(logfile, "w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            cwd=str(work_dir),
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"Nek5000 build failed with exit code {proc.returncode}; see {logfile}")
    if not built_executable.exists():
        raise FileNotFoundError(f"Build completed but executable was not created: {built_executable}")

    installed_executable.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(built_executable, installed_executable)
    installed_executable.chmod(installed_executable.stat().st_mode | 0o111)
    return BuildPaths(source_dir, work_dir, built_executable, installed_executable, logfile)


def prepare_run(conf: Mapping[str, Any], *, base_dir: Optional[Path] = None) -> RunPaths:
    """Create and populate ``runs/<run_name>`` from a resolved configuration."""

    base_dir = Path.cwd() if base_dir is None else Path(base_dir)
    repo_root = _repo_root(base_dir)
    run_conf = _section(conf, "run")
    case_conf = _section(conf, "case")
    post_conf = _section(conf, "post")

    case_id = _required_str(case_conf, "id", "case")
    session = _required_str(case_conf, "session", "case")
    run_name = str(run_conf.get("name") or f"{session}_run")
    run_root = _resolve_path(run_conf.get("root", "runs"), base_dir, repo_root)
    run_dir = _resolve_under(run_name, run_root)
    zstat_dir = run_dir / "ZSTAT"

    run_dir.mkdir(parents=True, exist_ok=True)
    zstat_dir.mkdir(parents=True, exist_ok=True)
    _runtime_output_dir(conf, run_dir).mkdir(parents=True, exist_ok=True)

    deps_dir = _resolve_path(
        case_conf.get("dependencies", f"data/dependencies/{case_id}"),
        base_dir,
        repo_root,
    )
    _stage_case_files(conf, deps_dir, run_dir, session)
    _stage_interpolation_mesh(conf, run_dir, base_dir, repo_root)
    staged_stats = stage_stats_files(conf, zstat_dir, session, base_dir, repo_root)
    _write_session(run_dir, session)
    _patch_rea_if_present(conf, run_dir, session, staged_stats)

    stat_file = None
    out_file = post_conf.get("out")
    if out_file:
        stat_file = _resolve_under(out_file, run_dir)

    paths = RunPaths(
        root=run_dir,
        zstat=zstat_dir,
        config=run_dir / "current_conf.yml",
        logfile=run_dir / str(_section(conf, "nek").get("logfile", "logfile")),
        stat_file=stat_file,
    )
    _write_resolved_config(conf, paths.config)
    return paths


def stage_stats_files(
    conf: Mapping[str, Any],
    zstat_dir: Path,
    session: str,
    base_dir: Optional[Path] = None,
    repo_root: Optional[Path] = None,
) -> List[StsFile]:
    """Discover, numerically sort, and stage ``sts`` files into ``ZSTAT``."""

    source_conf = _section(conf, "source")
    source_path = source_conf.get("path")
    if not source_path:
        return []

    base_dir = Path.cwd() if base_dir is None else Path(base_dir)
    repo_root = _repo_root(base_dir) if repo_root is None else repo_root
    source_root = _resolve_path(source_path, base_dir, repo_root)
    pattern = str(source_conf.get("pattern") or f"res_*/sts_data/sts{session}0.f*")
    stage_mode = str(source_conf.get("stage_mode", "symlink")).lower()
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

    staged: List[StsFile] = []
    prefix = str(source_conf.get("target_prefix") or f"sts{session}0.f")
    overwrite = bool(source_conf.get("overwrite", False))
    previous_end_time = (
        read_sts_header(files[first_index - 2])[1]
        if first_index > 1
        else None
    )
    for src_pos, src in enumerate(selected, start=first_index):
        header, end_time = read_sts_header(src)
        if previous_end_time is not None and end_time <= previous_end_time:
            raise ValueError(
                "sts end times are not strictly increasing: "
                f"file {src_pos - 1} ends at {previous_end_time:.16g}, "
                f"file {src_pos} at {end_time:.16g}"
            )
        dst = zstat_dir / f"{prefix}{src_pos:05d}"
        _stage_file(src, dst, mode=stage_mode, overwrite=overwrite)
        staged.append(
            StsFile(
                index=src_pos,
                source=src,
                staged=dst,
                end_time=end_time,
                header=header,
                predecessor_end_time=previous_end_time,
            )
        )
        previous_end_time = end_time
    return staged


def run_nek(conf: Mapping[str, Any], paths: RunPaths) -> None:
    """Run the staged Nek5000 executable inside the run directory."""

    nek_conf = _section(conf, "nek")
    session = _required_str(_section(conf, "case"), "session", "case")
    _write_session(paths.root, session)
    _prepare_nek_run_state(nek_conf, paths)

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
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Nek5000 failed with exit code {proc.returncode}; see {paths.logfile}"
        )
    _validate_nek_run(nek_conf, paths)


def run_python_post(conf: Mapping[str, Any], paths: RunPaths):
    """Run the validated Python ``int_fld -> stat_*.mat`` stage."""

    post_conf = _section(conf, "post")
    mesh = post_conf.get("mesh")
    if not mesh:
        raise ValueError("post.mesh is required for Python post-processing")

    mesh_file = _resolve_path(mesh, paths.root, _repo_root(paths.root))
    out_file = post_conf.get("out", "stat.mat")
    out_path = _resolve_under(out_file, paths.root)
    return run_post(
        str(paths.root),
        str(mesh_file),
        str(out_path),
        istart=int(post_conf.get("istart", 25)),
        iend=int(post_conf.get("iend", 80)),
        long=int(post_conf.get("long", 1)),
    )


def _prepare_nek_run_state(nek_conf: Mapping[str, Any], paths: RunPaths) -> None:
    paths.zstat.mkdir(parents=True, exist_ok=True)
    a0_dir = _runtime_output_dir_from_nek(nek_conf, paths.root)
    a0_dir.mkdir(parents=True, exist_ok=True)

    if bool(nek_conf.get("clear_a0", True)):
        _clear_generated_entries(a0_dir)
    if bool(nek_conf.get("remove_existing_int_fld", True)):
        _safe_unlink(paths.zstat / "int_fld")
    if bool(nek_conf.get("clear_sch", True)):
        for entry in paths.root.glob("*.sch*"):
            _safe_unlink(entry)
    if bool(nek_conf.get("remove_logfile", True)):
        _safe_unlink(paths.logfile)


def _validate_nek_run(nek_conf: Mapping[str, Any], paths: RunPaths) -> None:
    log_text = ""
    if paths.logfile.exists():
        log_text = paths.logfile.read_text(encoding="utf-8", errors="replace")

    abort_marker = nek_conf.get("abort_marker", "ABORT:")
    if abort_marker and abort_marker in log_text:
        raise RuntimeError(f"Nek5000 reported {abort_marker!r}; see {paths.logfile}")

    marker = nek_conf.get("completion_marker", "Output interporlated field in int_fld")
    if marker and marker not in log_text:
        raise RuntimeError(
            f"Nek5000 log is missing completion marker {marker!r}; see {paths.logfile}"
        )

    if bool(nek_conf.get("require_int_fld", True)) and not (paths.zstat / "int_fld").is_file():
        raise RuntimeError(f"Nek5000 did not create {paths.zstat / 'int_fld'}")


def _stage_case_files(
    conf: Mapping[str, Any],
    deps_dir: Path,
    run_dir: Path,
    session: str,
) -> None:
    case_conf = _section(conf, "case")
    stage_mode = str(case_conf.get("stage_mode", "copy")).lower()
    overwrite = bool(case_conf.get("overwrite", False))
    files = list(case_conf.get("files") or _case_files(session))

    for name in files:
        src = deps_dir / str(name)
        dst = run_dir / str(name)
        _stage_file(src, dst, mode=stage_mode, overwrite=overwrite)

    executable = case_conf.get("executable")
    if executable:
        src = _resolve_path(executable, deps_dir, _repo_root(run_dir))
    else:
        src = deps_dir / "nek5000"
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


def _runtime_output_dir(conf: Mapping[str, Any], run_dir: Path) -> Path:
    return _runtime_output_dir_from_nek(_section(conf, "nek"), run_dir)


def _runtime_output_dir_from_nek(nek_conf: Mapping[str, Any], run_dir: Path) -> Path:
    return _resolve_under(nek_conf.get("output_dir", "A0"), run_dir)


def _clear_generated_entries(directory: Path) -> None:
    for entry in directory.iterdir():
        _safe_unlink(entry)


def _safe_unlink(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        raise ValueError(f"Refusing to remove directory or special file: {path}")


def _stage_build_template(
    conf: Mapping[str, Any],
    build_conf: Mapping[str, Any],
    work_dir: Path,
    base_dir: Path,
    repo_root: Path,
) -> None:
    template_dir_value = build_conf.get("template_dir")
    if template_dir_value in (None, "", False):
        return

    case_conf = _section(conf, "case")
    session = _required_str(case_conf, "session", "case")
    template_dir = _resolve_path(template_dir_value, base_dir, repo_root)
    files = list(build_conf.get("template_files") or _build_template_files(session))
    stage_mode = str(build_conf.get("template_stage_mode", "copy")).lower()
    overwrite = bool(build_conf.get("template_overwrite", True))

    work_dir.mkdir(parents=True, exist_ok=True)
    for name in files:
        _stage_file(template_dir / str(name), work_dir / str(name), mode=stage_mode, overwrite=overwrite)


def _build_template_files(session: str) -> Tuple[str, ...]:
    return _case_files(session) + (
        "compile-script",
        "01-compile-command",
        "compile.sh",
        "misc.f",
    )


def _stage_interpolation_mesh(
    conf: Mapping[str, Any],
    run_dir: Path,
    base_dir: Path,
    repo_root: Path,
) -> None:
    mesh_conf = _section(conf, "mesh")
    mode = str(mesh_conf.get("mode", "copy")).lower()
    overwrite = bool(mesh_conf.get("overwrite", True))
    zstat_dir = run_dir / "ZSTAT"

    if mode == "generate":
        x_pts, y_pts, npoints, *_ = build_mesh(tol=float(mesh_conf.get("tol", 0.5e-7)))
        save_inter_mesh(str(run_dir), npoints, x_pts, y_pts)
        return

    if mode != "copy":
        raise ValueError("mesh.mode must be 'copy' or 'generate'")

    source_dir = mesh_conf.get("source_dir", "data/ZSTAT")
    source_dir = _resolve_path(source_dir, base_dir, repo_root)
    stage_mode = str(mesh_conf.get("stage_mode", "copy")).lower()
    for filename in ("x.fort", "y.fort"):
        _stage_file(
            source_dir / filename,
            zstat_dir / filename,
            mode=stage_mode,
            overwrite=overwrite,
        )


def _patch_rea_if_present(
    conf: Mapping[str, Any],
    run_dir: Path,
    session: str,
    staged_stats: Sequence[StsFile],
) -> None:
    nek_conf = _section(conf, "nek")
    params: Dict[int, Any] = {}
    if staged_stats:
        params[68] = staged_stats[-1].index
        params[69] = staged_stats[0].index
        params[70] = _sampling_start_time(nek_conf, staged_stats)
    elif nek_conf.get("sampling_start_time") not in (None, "", "auto"):
        params[70] = _coerce_float(
            nek_conf["sampling_start_time"],
            "nek.sampling_start_time",
        )
    params.update({int(k): v for k, v in _section(nek_conf, "rea_params").items()})

    if not params:
        return

    rea_name = str(_section(conf, "case").get("rea", f"{session}.rea"))
    rea_path = run_dir / rea_name
    if not rea_path.exists():
        return
    patch_rea_params(rea_path, params)


def patch_rea_params(path: Pathish, params: Mapping[int, Any]) -> None:
    """Patch Nek ``pNNN`` values while preserving the comment text."""

    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    remaining = {int(k): v for k, v in params.items()}
    patched: List[str] = []

    for line in lines:
        match = _REA_PARAM_RE.match(line.rstrip("\n"))
        if not match:
            patched.append(line)
            continue

        num = int(match.group("num"))
        if num not in remaining:
            patched.append(line)
            continue

        value = _format_rea_value(remaining.pop(num))
        newline = "\n" if line.endswith("\n") else ""
        patched.append(f"{match.group(1)}{value:<18}{match.group(3)}{newline}")

    if remaining:
        missing = ", ".join(f"p{k:03d}" for k in sorted(remaining))
        raise ValueError(f"Could not find {missing} in {path}")

    path.write_text("".join(patched), encoding="utf-8")


def read_sts_header(path: Pathish) -> Tuple[str, float]:
    """Return ``(header_text, end_time)`` from a Nek ``sts`` file header."""

    path = Path(path)
    try:
        payload = path.read_bytes()[:136]
    except OSError as exc:
        raise ValueError(f"{path}: cannot read sts header ({exc})") from exc

    failures: List[str] = []
    for offset in (0, 4):
        candidate = payload[offset:offset + 132]
        if len(candidate) < 8:
            failures.append(f"offset {offset}: truncated header")
            continue
        try:
            header = candidate.decode("ascii", errors="strict").strip()
            fields = header.split()
            if not fields or fields[0] != "#std" or len(fields) < 8:
                raise ValueError(f"unexpected header {header!r}")
            end_time = _coerce_float(fields[7], f"{path}: sts header end time")
            return header, end_time
        except (UnicodeDecodeError, ValueError) as exc:
            failures.append(f"offset {offset}: {exc}")

    raise ValueError(f"{path}: cannot parse sts header ({'; '.join(failures)})")


def _discover_stats(source_root: Path, pattern: str) -> List[Path]:
    if source_root.is_file():
        files = [source_root]
    else:
        files = [Path(p) for p in glob.glob(str(source_root / pattern))]
    files = [p for p in files if p.is_file()]
    files.sort(key=_natural_key)
    return files


def _sampling_start_time(
    nek_conf: Mapping[str, Any],
    staged_stats: Sequence[StsFile],
) -> float:
    raw = nek_conf.get("sampling_start_time", "auto")
    if raw not in (None, "", "auto"):
        return _coerce_float(raw, "nek.sampling_start_time")

    predecessor = staged_stats[0].predecessor_end_time
    if predecessor is not None:
        return predecessor

    initial = nek_conf.get("initial_start_time")
    if initial not in (None, ""):
        return _coerce_float(initial, "nek.initial_start_time")

    if bool(nek_conf.get("allow_first_file_end_time_as_start", False)):
        return staged_stats[0].end_time

    raise ValueError(
        "Cannot infer nek.sampling_start_time=auto for first_file=1 because no "
        "predecessor sts file is available. Set nek.initial_start_time, start "
        "from source.first_file > 1, or explicitly enable "
        "nek.allow_first_file_end_time_as_start."
    )


def _extract_solver_archive(archive: Path, target_dir: Path) -> None:
    if not archive.exists():
        raise FileNotFoundError(f"Nek solver archive not found: {archive}")
    target_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as tar:
        members = tar.getmembers()
        target = target_dir.resolve()
        for member in members:
            member_path = (target_dir / member.name).resolve()
            if target not in [member_path] + list(member_path.parents):
                raise ValueError(f"Unsafe path in solver archive: {member.name}")
        try:
            tar.extractall(target_dir, members=members, filter="fully_trusted")
        except TypeError:
            tar.extractall(target_dir, members=members)


def _build_command(command: Any, env_script: Any) -> List[str]:
    if isinstance(command, (list, tuple)):
        direct = [str(part) for part in command]
        command_str = " ".join(shlex.quote(part) for part in direct)
    else:
        command_str = str(command)
        direct = shlex.split(command_str)

    if env_script:
        env_path = os.path.expanduser(str(env_script))
        script = f"set -e; source {shlex.quote(env_path)}; {command_str}"
        return ["bash", "-lc", script]
    return direct


def _stage_file(src: Path, dst: Path, *, mode: str, overwrite: bool) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Required file not found: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        dst.unlink()

    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        raise ValueError("stage mode must be 'copy' or 'symlink'")


def _write_session(run_dir: Path, session: str) -> None:
    (run_dir / "SESSION.NAME").write_text(
        f"{session}\n{run_dir.resolve()}\n",
        encoding="utf-8",
    )


def _write_resolved_config(conf: Mapping[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(conf), handle, sort_keys=False)


def _case_files(session: str) -> Tuple[str, ...]:
    return _DEFAULT_CASE_FILES + (
        f"{session}.usr",
        f"{session}.rea",
        f"{session}.re2",
        f"{session}.map",
    )


def _format_rea_value(value: Any) -> str:
    if isinstance(value, int):
        return f"{float(value):.5E}"
    if isinstance(value, float):
        return f"{value:.12E}"
    text = str(value)
    try:
        return f"{float(text):.12E}"
    except ValueError:
        return text


def _coerce_float(value: Any, label: str) -> float:
    try:
        result = float(str(value).replace("D", "E").replace("d", "e"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite float, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return result


def _natural_key(path: Path) -> Tuple[Any, ...]:
    rel = str(path)
    parts = re.split(r"(\d+)", rel)
    return tuple(int(part) if part.isdigit() else part for part in parts)


def _section(mapping: Mapping[str, Any], name: str) -> Dict[str, Any]:
    value = mapping.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _required_str(mapping: Mapping[str, Any], key: str, section: str) -> str:
    value = mapping.get(key)
    if not value:
        raise ValueError(f"{section}.{key} is required")
    return str(value)


def _resolve_path(value: Any, base_dir: Path, repo_root: Path) -> Path:
    path = Path(os.path.expanduser(str(value)))
    if path.is_absolute():
        return path
    base_candidate = (base_dir / path).resolve()
    if base_candidate.exists():
        return base_candidate
    return (repo_root / path).resolve()


def _resolve_under(value: Any, base_dir: Path) -> Path:
    path = Path(os.path.expanduser(str(value)))
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _repo_root(start: Path) -> Path:
    start = start.resolve()
    candidates = [start] + list(start.parents)
    for path in candidates:
        if (path / "pyproject.toml").exists() and (path / "src" / "nekstat").exists():
            return path
    return Path.cwd().resolve()
