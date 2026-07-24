#!/usr/bin/env python3
"""Automate backward-window Nek statistics and minimal MATLAB Cl/Cd extraction."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_NEK_DIR = SCRIPT_DIR / "01-NEK-Pstat"
DEFAULT_MATLAB_DIR = SCRIPT_DIR / "02-MatLab-Read-Stat"
STS_NAME_RE_TEMPLATE = r"^sts{case}0\.f(?P<index>\d{{5}})$"
REA_PARAMETERS = (68, 69, 70)
INT_FLD_HEADER_LIMIT = 1_000_000


class AutomationError(RuntimeError):
    """An expected, user-actionable automation failure."""


@dataclass(frozen=True)
class StsFile:
    index: int
    path: Path
    end_time: float
    header: str


@dataclass(frozen=True)
class Window:
    window_index: int
    first_file: int
    last_file: int
    number_of_files: int
    start_time: float
    last_file_end_time: float


@dataclass(frozen=True)
class IntFldHeader:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate fixed-end backward-averaged int_fld files with Nek5000 and "
            "extract Cl/Cd from only F4 and D1-D4 in one MATLAB batch."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--first-file", type=positive_int, required=True)
    parser.add_argument("--last-file", type=positive_int, required=True)
    parser.add_argument("--stride", type=positive_int, required=True)
    parser.add_argument(
        "--aoa-deg",
        type=finite_float,
        required=True,
        help="Angle of attack in degrees used to resolve lift and drag.",
    )
    parser.add_argument(
        "--initial-start-time",
        type=finite_float,
        help=(
            "Sampling start time immediately before --first-file. Required only "
            "when the preceding sts file is unavailable."
        ),
    )
    parser.add_argument("--case-name", default="naca_wing")
    parser.add_argument("--nek-dir", type=Path, default=DEFAULT_NEK_DIR)
    parser.add_argument(
        "--zstat-dir",
        type=Path,
        help="Source ZSTAT directory; defaults to <nek-dir>/ZSTAT.",
    )
    parser.add_argument("--matlab-dir", type=Path, default=DEFAULT_MATLAB_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Run output directory. The default includes case, final file, stride, "
            "and AOA to avoid cross-case collisions."
        ),
    )
    parser.add_argument(
        "--stage",
        choices=("all", "nek", "matlab"),
        default="all",
        help="Run both stages, only generate int_fld files, or only extract Cl/Cd.",
    )
    parser.add_argument(
        "--matlab-command",
        default="matlab",
        help='Command prefix used to start MATLAB, for example "matlab" or "srun matlab".',
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate int_fld files that already pass header validation.",
    )
    parser.add_argument(
        "--discard-int-fld",
        action="store_true",
        help="Delete generated int_fld files only after MATLAB succeeds.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print planned windows without creating or running anything.",
    )
    return parser


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("must be finite")
    return parsed


def resolve_from(base: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (base / value).resolve()


def read_sts_header(path: Path) -> tuple[str, float]:
    try:
        with path.open("rb") as handle:
            payload = handle.read(132)
        header = payload.decode("ascii", errors="strict").strip()
        fields = header.split()
        if not fields or fields[0] != "#std" or len(fields) < 8:
            raise AutomationError(f"unexpected Nek header {header!r}")
        end_time = float(fields[7].replace("D", "E").replace("d", "e"))
        if not math.isfinite(end_time):
            raise AutomationError("non-finite end time")
        return header, end_time
    except (OSError, UnicodeDecodeError, ValueError, AutomationError) as exc:
        raise AutomationError(f"{path}: cannot parse 132-byte sts header ({exc})") from exc


def discover_sts_files(zstat_dir: Path, case_name: str) -> dict[int, StsFile]:
    if not zstat_dir.is_dir():
        raise AutomationError(f"ZSTAT directory does not exist: {zstat_dir}")
    name_re = re.compile(STS_NAME_RE_TEMPLATE.format(case=re.escape(case_name)))
    discovered: dict[int, StsFile] = {}
    for path in zstat_dir.iterdir():
        match = name_re.match(path.name)
        if not match or not path.is_file():
            continue
        index = int(match.group("index"))
        if index in discovered:
            raise AutomationError(f"duplicate sts index {index}: {path}")
        header, end_time = read_sts_header(path)
        discovered[index] = StsFile(index, path.resolve(), end_time, header)
    if not discovered:
        raise AutomationError(
            f"no files matching sts{case_name}0.fNNNNN in {zstat_dir}"
        )
    return discovered


def require_contiguous(files: dict[int, StsFile], first: int, last: int) -> None:
    missing = [index for index in range(first, last + 1) if index not in files]
    if missing:
        preview = ", ".join(str(index) for index in missing[:12])
        suffix = " ..." if len(missing) > 12 else ""
        raise AutomationError(
            f"missing {len(missing)} required sts file(s) between {first} and "
            f"{last}: {preview}{suffix}"
        )
    previous_time = files[first].end_time
    for index in range(first + 1, last + 1):
        current_time = files[index].end_time
        if current_time <= previous_time:
            raise AutomationError(
                f"sts end times are not strictly increasing: file {index - 1} "
                f"ends at {previous_time:.16g}, file {index} at {current_time:.16g}"
            )
        previous_time = current_time


def build_windows(
    files: dict[int, StsFile],
    first_file: int,
    last_file: int,
    stride: int,
    initial_start_time: float | None,
) -> list[Window]:
    if first_file > last_file:
        raise AutomationError("--first-file must not exceed --last-file")
    predecessor = first_file - 1
    validation_first = predecessor if predecessor in files else first_file
    require_contiguous(files, validation_first, last_file)

    windows: list[Window] = []
    for window_index, current_first in enumerate(
        range(first_file, last_file + 1, stride), start=1
    ):
        prior_index = current_first - 1
        if prior_index in files:
            start_time = files[prior_index].end_time
        elif current_first == first_file and initial_start_time is not None:
            start_time = initial_start_time
        else:
            raise AutomationError(
                f"cannot derive p070 for first file {current_first}: "
                f"sts file {prior_index} is unavailable. Supply "
                "--initial-start-time for the first requested window."
            )
        first_end_time = files[current_first].end_time
        if start_time >= first_end_time:
            raise AutomationError(
                f"invalid start time {start_time:.16g} for file {current_first}; "
                f"its header end time is {first_end_time:.16g}"
            )
        windows.append(
            Window(
                window_index=window_index,
                first_file=current_first,
                last_file=last_file,
                number_of_files=last_file - current_first + 1,
                start_time=start_time,
                last_file_end_time=files[last_file].end_time,
            )
        )
    return windows


def format_aoa_for_path(aoa_deg: float) -> str:
    return format(aoa_deg, ".12g").replace("-", "m").replace("+", "p").replace(".", "p")


def default_output_dir(case_name: str, last_file: int, stride: int, aoa_deg: float) -> Path:
    name = (
        f"{case_name}_last{last_file:05d}_stride{stride:05d}_"
        f"aoa{format_aoa_for_path(aoa_deg)}"
    )
    return SCRIPT_DIR / "clcd_backward_results" / name


def update_rea_text(
    template: str, last_file: int, first_file: int, start_time: float
) -> str:
    values = {68: float(last_file), 69: float(first_file), 70: start_time}
    lines = template.splitlines(keepends=True)
    for parameter in REA_PARAMETERS:
        tag = f"p{parameter:03d}"
        matches = [
            line_index
            for line_index, line in enumerate(lines)
            if re.search(rf"\b{re.escape(tag)}\b", line, flags=re.IGNORECASE)
        ]
        if len(matches) != 1:
            raise AutomationError(
                f"expected exactly one {tag} entry in .rea template; found {len(matches)}"
            )
        line_index = matches[0]
        match = re.match(
            rf"^(?P<indent>\s*)(?P<value>\S+)(?P<suffix>\s+{tag}\b.*)$",
            lines[line_index],
            flags=re.IGNORECASE,
        )
        if not match:
            raise AutomationError(f"cannot rewrite {tag} line: {lines[line_index]!r}")
        newline = "\n" if lines[line_index].endswith("\n") else ""
        suffix = match.group("suffix").rstrip("\r\n")
        lines[line_index] = (
            f"{match.group('indent')}{values[parameter]:.13E}{suffix}{newline}"
        )
    return "".join(lines)


def ensure_symlink(target: Path, link: Path) -> None:
    target = target.resolve()
    if link.is_symlink():
        if link.resolve() == target:
            return
        link.unlink()
    elif link.exists():
        raise AutomationError(
            f"refusing to replace non-symlink staging entry: {link}"
        )
    link.symlink_to(target)


def prepare_workdir(
    work_dir: Path,
    nek_dir: Path,
    zstat_dir: Path,
    files: dict[int, StsFile],
    case_name: str,
) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    work_zstat = work_dir / "ZSTAT"
    work_zstat.mkdir(exist_ok=True)
    (work_dir / "A0").mkdir(exist_ok=True)

    runtime_names = (
        "nek5000",
        "02-run-postNek",
        f"{case_name}.re2",
        f"{case_name}.map",
    )
    for name in runtime_names:
        source = nek_dir / name
        if not source.exists():
            raise AutomationError(f"required Nek runtime input is missing: {source}")
        ensure_symlink(source, work_dir / name)

    for mesh_name in ("x.fort", "y.fort"):
        source = zstat_dir / mesh_name
        if not source.is_file():
            raise AutomationError(f"required interpolation mesh is missing: {source}")
        ensure_symlink(source, work_zstat / mesh_name)

    for sts_file in files.values():
        ensure_symlink(sts_file.path, work_zstat / sts_file.path.name)


def clear_staged_a0(work_dir: Path) -> None:
    a0_dir = work_dir / "A0"
    for entry in a0_dir.iterdir():
        if entry.is_file() or entry.is_symlink():
            entry.unlink()
        else:
            raise AutomationError(
                f"refusing to remove unexpected directory in staging A0: {entry}"
            )


def parse_int_fld_header(path: Path) -> IntFldHeader:
    failures: list[str] = []
    for byte_order in ("<", ">"):
        try:
            with path.open("rb") as handle:
                header_text = read_record_from_handle(handle, byte_order)
                if not header_text.startswith(b"(Re ="):
                    raise AutomationError("unexpected int_fld text header")
                values = read_record_from_handle(handle, byte_order)
                offset = 0

                def unpack(code: str) -> tuple[float | int, ...]:
                    nonlocal offset
                    size = struct.calcsize(f"{byte_order}{code}")
                    if offset + size > len(values):
                        raise AutomationError("truncated int_fld binary header")
                    result = struct.unpack_from(f"{byte_order}{code}", values, offset)
                    offset += size
                    return result

                reynolds_number = float(unpack("d")[0])
                unpack("3d")  # Domain
                unpack("3i")  # Element counts
                unpack("3i")  # Polynomial order
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
                    raise AutomationError(
                        f"unexpected {len(values) - offset} extra header byte(s)"
                    )
                if (
                    reynolds_number <= 0
                    or nstat <= 0
                    or nderiv < 0
                    or npoints <= 0
                    or end_time <= start_time
                    or time_interval <= 0
                ):
                    raise AutomationError("implausible int_fld header values")
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
        except (AutomationError, OSError, struct.error) as exc:
            failures.append(f"{byte_order}: {exc}")
    raise AutomationError(f"{path}: invalid int_fld header ({'; '.join(failures)})")


def read_record_from_handle(handle, byte_order: str) -> bytes:
    marker = handle.read(4)
    if len(marker) != 4:
        raise AutomationError("missing Fortran record marker")
    (record_size,) = struct.unpack(f"{byte_order}I", marker)
    if not 0 < record_size <= INT_FLD_HEADER_LIMIT:
        raise AutomationError(f"implausible record size {record_size}")
    payload = handle.read(record_size)
    trailing = handle.read(4)
    if len(payload) != record_size or len(trailing) != 4:
        raise AutomationError("truncated Fortran record")
    (trailing_size,) = struct.unpack(f"{byte_order}I", trailing)
    if trailing_size != record_size:
        raise AutomationError("mismatched Fortran record markers")
    return payload


def validate_int_fld_for_window(path: Path, window: Window) -> IntFldHeader:
    header = parse_int_fld_header(path)
    tolerance = max(1.0e-10, abs(window.start_time) * 1.0e-10)
    if abs(header.start_time - window.start_time) > tolerance:
        raise AutomationError(
            f"{path}: header start time {header.start_time:.16g} does not match "
            f"window start {window.start_time:.16g}"
        )
    if abs(header.end_time - window.last_file_end_time) > tolerance:
        raise AutomationError(
            f"{path}: header end time {header.end_time:.16g} does not match "
            f"last sts end {window.last_file_end_time:.16g}"
        )
    if header.nstat < 4 or header.nderiv < 4:
        raise AutomationError(
            f"{path}: Cl/Cd requires F4 and D1-D4, but header reports "
            f"nstat={header.nstat}, nderiv={header.nderiv}"
        )
    return header


def window_stem(window: Window) -> str:
    return f"first{window.first_file:05d}_last{window.last_file:05d}"


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_write_json(path: Path, data: object) -> None:
    atomic_write_text(path, json.dumps(data, indent=2, sort_keys=True) + "\n")


def write_manifest(
    manifest_path: Path,
    windows: Sequence[Window],
    int_fld_dir: Path,
    log_dir: Path,
    statuses: dict[int, str],
    headers: dict[int, IntFldHeader],
) -> None:
    fieldnames = (
        "window_index",
        "first_file",
        "last_file",
        "number_of_files",
        "start_time",
        "last_file_end_time",
        "int_fld_path",
        "nek_log_path",
        "status",
        "time_interval",
        "effective_average_time",
        "nrec",
        "npoints",
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=manifest_path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for window in windows:
            stem = window_stem(window)
            header = headers.get(window.window_index)
            writer.writerow(
                {
                    "window_index": window.window_index,
                    "first_file": window.first_file,
                    "last_file": window.last_file,
                    "number_of_files": window.number_of_files,
                    "start_time": format(window.start_time, ".16g"),
                    "last_file_end_time": format(
                        window.last_file_end_time, ".16g"
                    ),
                    "int_fld_path": str((int_fld_dir / f"int_fld_{stem}").resolve()),
                    "nek_log_path": str((log_dir / f"nek_{stem}.log").resolve()),
                    "status": statuses.get(window.window_index, "planned"),
                    "time_interval": (
                        format(header.time_interval, ".16g") if header else ""
                    ),
                    "effective_average_time": (
                        format(header.effective_average_time, ".16g")
                        if header
                        else ""
                    ),
                    "nrec": header.nrec if header else "",
                    "npoints": header.npoints if header else "",
                }
            )
        temporary = Path(handle.name)
    os.replace(temporary, manifest_path)


def run_nek_windows(
    args: argparse.Namespace,
    windows: Sequence[Window],
    files: dict[int, StsFile],
    nek_dir: Path,
    zstat_dir: Path,
    output_dir: Path,
) -> tuple[dict[int, str], dict[int, IntFldHeader]]:
    work_dir = output_dir / "work"
    int_fld_dir = output_dir / "int_fld"
    log_dir = output_dir / "logs"
    int_fld_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    prepare_workdir(work_dir, nek_dir, zstat_dir, files, args.case_name)

    rea_template_path = nek_dir / f"{args.case_name}.rea"
    if not rea_template_path.is_file():
        raise AutomationError(f".rea template does not exist: {rea_template_path}")
    rea_template = rea_template_path.read_text(encoding="utf-8")
    statuses: dict[int, str] = {}
    headers: dict[int, IntFldHeader] = {}

    for window in windows:
        stem = window_stem(window)
        destination = int_fld_dir / f"int_fld_{stem}"
        destination_log = log_dir / f"nek_{stem}.log"
        driver_log = log_dir / f"driver_{stem}.log"

        if destination.exists() and not args.force:
            try:
                headers[window.window_index] = validate_int_fld_for_window(
                    destination, window
                )
                statuses[window.window_index] = "reused"
                print(
                    f"[{window.window_index}/{len(windows)}] reuse {destination.name}",
                    flush=True,
                )
                continue
            except AutomationError as exc:
                raise AutomationError(
                    f"existing output failed validation: {exc}. "
                    "Use --force only if it is safe to regenerate it."
                ) from exc

        generated = work_dir / "ZSTAT" / "int_fld"
        if generated.exists() or generated.is_symlink():
            generated.unlink()
        clear_staged_a0(work_dir)

        rea_text = update_rea_text(
            rea_template,
            last_file=window.last_file,
            first_file=window.first_file,
            start_time=window.start_time,
        )
        atomic_write_text(work_dir / f"{args.case_name}.rea", rea_text)
        print(
            f"[{window.window_index}/{len(windows)}] Nek files "
            f"{window.first_file}-{window.last_file}, "
            f"p070={window.start_time:.13g}",
            flush=True,
        )
        completed = subprocess.run(
            ["bash", "./02-run-postNek", "--m", args.case_name],
            cwd=work_dir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        atomic_write_text(driver_log, completed.stdout)
        work_log = work_dir / "logfile"
        nek_log_text = ""
        if work_log.is_file():
            nek_log_text = work_log.read_text(encoding="utf-8", errors="replace")
            shutil.copy2(work_log, destination_log)
        if completed.returncode != 0:
            raise AutomationError(
                f"Nek wrapper failed for files {window.first_file}-"
                f"{window.last_file} with exit code {completed.returncode}; "
                f"see {driver_log}"
            )
        if "Output interporlated field in int_fld" not in nek_log_text:
            raise AutomationError(
                f"Nek log has no int_fld completion marker for files "
                f"{window.first_file}-{window.last_file}; see {destination_log}"
            )
        if "ABORT:" in nek_log_text:
            raise AutomationError(
                f"Nek reported an abort for files {window.first_file}-"
                f"{window.last_file}; see {destination_log}"
            )
        if not generated.is_file():
            raise AutomationError(
                f"Nek did not create {generated} for files {window.first_file}-"
                f"{window.last_file}; see {driver_log} and {destination_log}"
            )
        header = validate_int_fld_for_window(generated, window)
        os.replace(generated, destination)
        headers[window.window_index] = header
        statuses[window.window_index] = "generated"

    return statuses, headers


def validate_existing_windows(
    windows: Sequence[Window], output_dir: Path
) -> tuple[dict[int, str], dict[int, IntFldHeader]]:
    statuses: dict[int, str] = {}
    headers: dict[int, IntFldHeader] = {}
    for window in windows:
        path = output_dir / "int_fld" / f"int_fld_{window_stem(window)}"
        if not path.is_file():
            raise AutomationError(
                f"MATLAB-only stage requires an existing int_fld: {path}"
            )
        headers[window.window_index] = validate_int_fld_for_window(path, window)
        statuses[window.window_index] = "reused"
    return statuses, headers


def command_is_available(command: Sequence[str]) -> bool:
    executable = command[0]
    if "/" in executable:
        return Path(executable).is_file() and os.access(executable, os.X_OK)
    return shutil.which(executable) is not None


def matlab_preflight(args: argparse.Namespace, matlab_dir: Path) -> list[str]:
    command = shlex.split(args.matlab_command)
    if not command:
        raise AutomationError("--matlab-command must not be empty")
    if not command_is_available(command):
        raise AutomationError(
            f"MATLAB command is unavailable: {command[0]!r}. Load the MATLAB "
            "environment or pass --matlab-command."
        )
    batch_function = matlab_dir / "matlab_func" / "compute_clcd_batch.m"
    mesh_file = matlab_dir / "int_mesh.mat"
    if not batch_function.is_file():
        raise AutomationError(f"MATLAB batch function is missing: {batch_function}")
    if not mesh_file.is_file():
        raise AutomationError(f"MATLAB interpolation mesh is missing: {mesh_file}")
    return command


def matlab_quote(value: Path) -> str:
    return str(value.resolve()).replace("'", "''")


def run_matlab(
    args: argparse.Namespace,
    matlab_dir: Path,
    output_dir: Path,
    manifest_path: Path,
) -> Path:
    command = matlab_preflight(args, matlab_dir)
    matlab_func_dir = matlab_dir / "matlab_func"
    mesh_file = matlab_dir / "int_mesh.mat"
    output_csv = output_dir / "clcd_backward.csv"
    expression = (
        f"addpath('{matlab_quote(matlab_func_dir)}');"
        f"compute_clcd_batch('{matlab_quote(manifest_path)}',"
        f"'{matlab_quote(mesh_file)}','{matlab_quote(output_csv)}',"
        f"{args.aoa_deg:.17g});"
    )
    matlab_log = output_dir / "logs" / "matlab.log"
    print(f"MATLAB: extracting Cl/Cd into {output_csv}", flush=True)
    completed = subprocess.run(
        [*command, "-batch", expression],
        cwd=matlab_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    atomic_write_text(matlab_log, completed.stdout)
    if completed.returncode != 0:
        raise AutomationError(
            f"MATLAB failed with exit code {completed.returncode}; see {matlab_log}"
        )
    if not output_csv.is_file():
        raise AutomationError(
            f"MATLAB exited successfully but did not create {output_csv}; "
            f"see {matlab_log}"
        )
    return output_csv


def discard_int_fld_files(windows: Iterable[Window], output_dir: Path) -> None:
    for window in windows:
        path = output_dir / "int_fld" / f"int_fld_{window_stem(window)}"
        if path.is_file():
            path.unlink()


def print_plan(
    windows: Sequence[Window], zstat_dir: Path, output_dir: Path, aoa_deg: float
) -> None:
    print(f"ZSTAT:     {zstat_dir}")
    print(f"Output:    {output_dir}")
    print(f"AOA:       {aoa_deg:.16g} deg")
    print(f"Windows:   {len(windows)}")
    print(" index  first   last  files             p070/start          final end")
    for window in windows:
        print(
            f"{window.window_index:6d}  {window.first_file:5d}  "
            f"{window.last_file:5d}  {window.number_of_files:5d}  "
            f"{window.start_time:21.13g}  "
            f"{window.last_file_end_time:17.13g}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    nek_dir = resolve_from(SCRIPT_DIR, args.nek_dir)
    zstat_dir = (
        resolve_from(SCRIPT_DIR, args.zstat_dir)
        if args.zstat_dir is not None
        else nek_dir / "ZSTAT"
    )
    matlab_dir = resolve_from(SCRIPT_DIR, args.matlab_dir)
    output_dir = (
        resolve_from(SCRIPT_DIR, args.output_dir)
        if args.output_dir is not None
        else default_output_dir(
            args.case_name, args.last_file, args.stride, args.aoa_deg
        ).resolve()
    )

    files = discover_sts_files(zstat_dir, args.case_name)
    windows = build_windows(
        files,
        first_file=args.first_file,
        last_file=args.last_file,
        stride=args.stride,
        initial_start_time=args.initial_start_time,
    )
    print_plan(windows, zstat_dir, output_dir, args.aoa_deg)
    if args.dry_run:
        return 0
    if args.discard_int_fld and args.stage == "nek":
        raise AutomationError("--discard-int-fld requires --stage all or matlab")
    if args.stage in ("all", "matlab"):
        matlab_preflight(args, matlab_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "nek_windows.csv"
    config = {
        "case_name": args.case_name,
        "first_file": args.first_file,
        "last_file": args.last_file,
        "stride": args.stride,
        "aoa_deg": args.aoa_deg,
        "initial_start_time": args.initial_start_time,
        "nek_dir": str(nek_dir),
        "zstat_dir": str(zstat_dir),
        "matlab_dir": str(matlab_dir),
        "output_dir": str(output_dir),
        "stage": args.stage,
        "windows": [asdict(window) for window in windows],
    }
    atomic_write_json(output_dir / "run_config.json", config)

    if args.stage in ("all", "nek"):
        statuses, headers = run_nek_windows(
            args, windows, files, nek_dir, zstat_dir, output_dir
        )
    else:
        statuses, headers = validate_existing_windows(windows, output_dir)
    write_manifest(
        manifest_path,
        windows,
        output_dir / "int_fld",
        output_dir / "logs",
        statuses,
        headers,
    )

    output_csv: Path | None = None
    if args.stage in ("all", "matlab"):
        output_csv = run_matlab(args, matlab_dir, output_dir, manifest_path)
        if args.discard_int_fld:
            discard_int_fld_files(windows, output_dir)
            print("Removed generated int_fld files after successful MATLAB export.")
    print("Complete.")
    print(f"Manifest: {manifest_path}")
    if output_csv is not None:
        print(f"Cl/Cd:    {output_csv}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AutomationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
