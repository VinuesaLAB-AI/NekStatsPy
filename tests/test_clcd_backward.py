import csv
import math
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.io as sio
import yaml

from nekstat.clcd_backward import (
    build_backward_windows,
    compute_clcd_from_int_fld,
    parse_int_fld_header,
    run_clcd_backward_from_config,
)
from nekstat.nek5000 import StsFile


def write_sts(path: Path, end_time: float, sample_count: int = 500) -> None:
    header = (
        f"#std 8 12 12 1 4692 12345 {end_time:20.13E} "
        f"{sample_count} 0 1 aa 66"
    )
    path.write_bytes(header.encode("ascii").ljust(132, b" "))


def write_record(handle, payload: bytes) -> None:
    marker = struct.pack("<I", len(payload))
    handle.write(marker)
    handle.write(payload)
    handle.write(marker)


def write_int_fld(
    path: Path,
    mesh: dict,
    start_time: float,
    end_time: float,
    *,
    pressure_scale: float = 1.0,
    gradient_scale: float = 1.0,
) -> None:
    x_pts = np.asarray(mesh["x_pts"], float).ravel()
    ln = int(np.count_nonzero(x_pts == -1))
    two_np = len(x_pts) // ln
    npoints = len(x_pts)
    nstat = 4
    nderiv = 4

    def field_record(matrix):
        raw = np.asarray(matrix.T.reshape(-1), dtype="<f8")
        return raw.tobytes()

    zeros = np.zeros((ln, two_np))
    pressure = pressure_scale * np.array(
        [
            [1.0, 1.1, 1.2, 0.9, 0.8, 0.7],
            [0.95, 1.05, 1.15, 0.85, 0.75, 0.65],
        ]
    )
    d1 = np.full((ln, two_np), 0.10 * gradient_scale)
    d2 = np.full((ln, two_np), 0.20 * gradient_scale)
    d3 = np.full((ln, two_np), 0.30 * gradient_scale)
    d4 = np.full((ln, two_np), 0.40 * gradient_scale)

    values = b"".join(
        [
            struct.pack("<d", 200000.0),
            struct.pack("<3d", 1.0, 1.0, 1.0),
            struct.pack("<3i", 1, 1, 1),
            struct.pack("<3i", 1, 1, 1),
            struct.pack("<i", nstat),
            struct.pack("<i", nderiv),
            struct.pack("<d", start_time),
            struct.pack("<d", end_time),
            struct.pack("<d", end_time - start_time),
            struct.pack("<d", 0.01),
            struct.pack("<i", 10),
            struct.pack("<d", end_time - start_time),
            struct.pack("<i", npoints),
        ]
    )
    fields = (zeros, zeros, zeros, pressure, d1, d2, d3, d4)
    with path.open("wb") as handle:
        write_record(handle, b"(Re = synthetic)")
        write_record(handle, values)
        for matrix in fields:
            write_record(handle, field_record(matrix))


def write_minimal_case(root: Path) -> dict:
    deps = root / "deps"
    mesh = root / "mesh"
    source = root / "source" / "res_1" / "sts_data"
    for path in (deps, mesh, source):
        path.mkdir(parents=True, exist_ok=True)
    for name in (
        "SIZE",
        "makefile_usr.inc",
        "naca_wing.usr",
        "naca_wing.re2",
        "naca_wing.map",
    ):
        (deps / name).write_text(name, encoding="utf-8")
    (deps / "naca_wing.rea").write_text(
        "  4.00000E+00      p068 Number of stat files\n"
        "  2.00000E+00      p069 First file\n"
        "  1.00000E+01      p070 Sampling start\n",
        encoding="utf-8",
    )
    exe = deps / "nek5000_4412"
    exe.write_text("#!/bin/sh\n", encoding="utf-8")
    exe.chmod(0o755)
    (mesh / "x.fort").write_text("x", encoding="utf-8")
    (mesh / "y.fort").write_text("y", encoding="utf-8")
    for index, (end_time, sample_count) in enumerate(
        ((10.0, 5000), (20.0, 7500), (30.0, 10000), (40.0, 5000)),
        start=1,
    ):
        write_sts(source / f"stsnaca_wing0.f{index:05d}", end_time, sample_count)
    return {
        "run": {"name": "naca_wing_test", "root": str(root / "runs")},
        "case": {
            "id": "naca4412",
            "session": "naca_wing",
            "dependencies": str(deps),
            "executable": str(exe),
            "overwrite": True,
        },
        "mesh": {"source_dir": str(mesh), "overwrite": True},
        "source": {
            "path": str(root / "source"),
            "pattern": "res_*/sts_data/stsnaca_wing0.f*",
            "stage_mode": "copy",
            "overwrite": True,
        },
        "nek": {"sampling_start_time": "auto"},
        "post": {"mesh": "data/int_mesh.mat", "out": "stat.mat", "aoa_deg": 5.0},
        "clcd_backward": {
            "output_dir": str(root / "runs" / "clcd"),
            "first_file": 2,
            "last_file": 4,
            "block_size_files": 2,
            "aoa_deg": 5.0,
        },
    }


class ClCdBackwardTests(unittest.TestCase):
    def test_build_backward_windows_makes_complete_blocks_by_default(self):
        stats = tuple(
            StsFile(
                index=index,
                source=Path(f"src{index}"),
                staged=Path(f"dst{index}"),
                end_time=float(index * 10),
                header="#std",
                predecessor_end_time=float((index - 1) * 10),
                sample_count=(index - 1) * 2500,
            )
            for index in range(2, 7)
        )

        windows = build_backward_windows(
            stats,
            block_size_files=2,
            nek_conf={"sampling_start_time": "auto"},
        )

        self.assertEqual([(w.first_file, w.last_file) for w in windows], [(2, 3), (4, 5)])
        self.assertEqual([w.start_time for w in windows], [10.0, 30.0])
        self.assertEqual([w.number_of_files for w in windows], [2, 2])

    def test_build_backward_windows_can_keep_short_final_block(self):
        stats = tuple(
            StsFile(
                index=index,
                source=Path(f"src{index}"),
                staged=Path(f"dst{index}"),
                end_time=float(index * 10),
                header="#std",
                predecessor_end_time=float((index - 1) * 10),
                sample_count=(index - 1) * 2500,
            )
            for index in range(2, 7)
        )

        windows = build_backward_windows(
            stats,
            block_size_files=2,
            include_incomplete_block=True,
            nek_conf={"sampling_start_time": "auto"},
        )

        self.assertEqual(
            [(w.first_file, w.last_file, w.number_of_files) for w in windows],
            [(2, 3, 2), (4, 5, 2), (6, 6, 1)],
        )
        self.assertEqual([w.start_time for w in windows], [10.0, 30.0, 50.0])

    def test_build_backward_windows_can_follow_accumulator_resets(self):
        sample_counts = (7500, 10000, 12500, 15000, 17500, 20000, 22500, 25000, 27500, 30000, 5000, 7500, 10000)
        stats = tuple(
            StsFile(
                index=index,
                source=Path(f"src{index}"),
                staged=Path(f"dst{index}"),
                end_time=40.0 + 0.0375 * (index - 2),
                header="#std",
                predecessor_end_time=40.0 + 0.0375 * (index - 3),
                sample_count=sample_count,
                predecessor_sample_count=5000 if index == 2 else sample_counts[index - 3],
            )
            for index, sample_count in zip(range(2, 15), sample_counts)
        )

        windows = build_backward_windows(
            stats,
            block_mode="accumulator_reset",
            nek_conf={"sampling_start_time": "auto"},
        )

        self.assertEqual(
            [(w.first_file, w.last_file, w.number_of_files) for w in windows],
            [(2, 11, 10), (12, 14, 3)],
        )
        self.assertEqual(
            [(w.first_sample_count, w.last_sample_count) for w in windows],
            [(7500, 30000), (5000, 10000)],
        )
        self.assertAlmostEqual(windows[1].start_time, 40.3)

    def test_prepare_only_writes_planned_backward_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            conf = write_minimal_case(root)
            config_path = root / "config.yml"
            config_path.write_text(yaml.safe_dump(conf), encoding="utf-8")

            paths = run_clcd_backward_from_config(config_path, prepare_only=True)
            manifest = paths.manifest.read_text(encoding="utf-8")

        self.assertEqual(paths.root.name, "clcd")
        self.assertEqual(paths.csv.name, "naca_wing_test_clcd_backward.csv")
        self.assertEqual(paths.block_csv.name, "naca_wing_test_clcd_blocks.csv")
        self.assertEqual([(w.first_file, w.last_file) for w in paths.windows], [(2, 3)])
        self.assertIn("planned", manifest)
        self.assertIn("int_fld_first00002_last00003", manifest)

    def test_prepare_only_can_write_adaptive_reset_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            conf = write_minimal_case(root)
            conf["clcd_backward"].pop("block_size_files")
            conf["clcd_backward"]["block_mode"] = "accumulator_reset"
            conf["clcd_backward"]["include_incomplete_block"] = False
            config_path = root / "config.yml"
            config_path.write_text(yaml.safe_dump(conf), encoding="utf-8")

            paths = run_clcd_backward_from_config(config_path, prepare_only=True)
            with paths.manifest.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual([(w.first_file, w.last_file) for w in paths.windows], [(2, 3), (4, 4)])
        self.assertEqual(
            [(row["first_sts_sample_count"], row["last_sts_sample_count"]) for row in rows],
            [("7500", "10000"), ("5000", "5000")],
        )

    def test_prepare_only_allows_time_overlap_between_reset_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            conf = write_minimal_case(root)
            source = root / "source" / "res_1" / "sts_data"
            for index, (end_time, sample_count) in enumerate(
                (
                    (10.0, 5000),
                    (20.0, 7500),
                    (30.0, 10000),
                    (25.0, 5000),
                    (35.0, 10000),
                ),
                start=1,
            ):
                write_sts(source / f"stsnaca_wing0.f{index:05d}", end_time, sample_count)
            conf["clcd_backward"].pop("block_size_files")
            conf["clcd_backward"]["block_mode"] = "accumulator_reset"
            conf["clcd_backward"]["allow_time_overlap_between_blocks"] = True
            conf["clcd_backward"]["last_file"] = 5
            config_path = root / "config.yml"
            config_path.write_text(yaml.safe_dump(conf), encoding="utf-8")

            paths = run_clcd_backward_from_config(config_path, prepare_only=True)

        self.assertEqual(
            [(w.first_file, w.last_file) for w in paths.windows],
            [(2, 3), (4, 5)],
        )
        self.assertLess(paths.windows[1].start_time, paths.windows[0].last_file_end_time)

    def test_skip_nek_writes_block_and_weighted_backward_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mesh_path = root / "int_mesh.mat"
            mesh = {
                "x_pts": np.array(
                    [0.0, 0.1, 0.5, 0.6, 1.0, 1.1, 0.4, 0.5, 0.9, 1.0, -1.0, -1.0]
                ),
                "y_pts": np.array(
                    [
                        0.0,
                        0.02,
                        0.06,
                        0.08,
                        0.01,
                        0.03,
                        -0.04,
                        -0.06,
                        -0.01,
                        -0.03,
                        0.0,
                        0.1,
                    ]
                ),
                "alphau": np.array([0.20, 0.25, 0.30]),
                "alphal": np.array([-0.20, -0.25, -0.30]),
                "xc": np.array([0.0, 0.5, 1.0]),
            }
            sio.savemat(mesh_path, {name: value.reshape(-1, 1) for name, value in mesh.items()})
            conf = write_minimal_case(root)
            conf["post"]["mesh"] = str(mesh_path)
            conf["clcd_backward"]["include_incomplete_block"] = True
            config_path = root / "config.yml"
            config_path.write_text(yaml.safe_dump(conf), encoding="utf-8")

            paths = run_clcd_backward_from_config(config_path, prepare_only=True)
            paths.int_fld_dir.mkdir(parents=True, exist_ok=True)
            for window, pressure_scale in zip(paths.windows, (1.0, 1.6)):
                int_fld = (
                    paths.int_fld_dir
                    / f"int_fld_first{window.first_file:05d}_last{window.last_file:05d}"
                )
                write_int_fld(
                    int_fld,
                    mesh,
                    start_time=window.start_time,
                    end_time=window.last_file_end_time,
                    pressure_scale=pressure_scale,
                )

            paths = run_clcd_backward_from_config(config_path, skip_nek=True)
            with paths.block_csv.open(newline="", encoding="utf-8") as handle:
                block_rows = list(csv.DictReader(handle))
            with paths.csv.open(newline="", encoding="utf-8") as handle:
                backward_rows = list(csv.DictReader(handle))

        self.assertEqual(
            [(r["first_file"], r["last_file"]) for r in block_rows],
            [("2", "3"), ("4", "4")],
        )
        self.assertEqual(len(backward_rows), 2)
        block_cl = [float(row["Cl"]) for row in block_rows]
        latest_cl = float(backward_rows[0]["Cl"])
        combined_cl = float(backward_rows[1]["Cl"])
        expected_combined = (block_cl[0] * 20.0 + block_cl[1] * 10.0) / 30.0
        self.assertAlmostEqual(latest_cl, block_cl[1])
        self.assertAlmostEqual(combined_cl, expected_combined)
        self.assertEqual(backward_rows[0]["number_of_blocks"], "1")
        self.assertEqual(backward_rows[1]["number_of_blocks"], "2")

    def test_parse_and_compute_lightweight_clcd(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mesh_path = root / "int_mesh.mat"
            int_fld = root / "int_fld"
            mesh = {
                "x_pts": np.array([0.0, 0.1, 0.5, 0.6, 1.0, 1.1, 0.4, 0.5, 0.9, 1.0, -1.0, -1.0]),
                "y_pts": np.array([0.0, 0.02, 0.06, 0.08, 0.01, 0.03, -0.04, -0.06, -0.01, -0.03, 0.0, 0.1]),
                "alphau": np.array([0.20, 0.25, 0.30]),
                "alphal": np.array([-0.20, -0.25, -0.30]),
                "xc": np.array([0.0, 0.5, 1.0]),
            }
            sio.savemat(mesh_path, {name: value.reshape(-1, 1) for name, value in mesh.items()})
            write_int_fld(int_fld, mesh, start_time=10.0, end_time=20.0)

            header = parse_int_fld_header(int_fld)
            coeffs = compute_clcd_from_int_fld(int_fld, mesh_path, aoa_deg=5.0)

        self.assertEqual(header.nstat, 4)
        self.assertEqual(header.nderiv, 4)
        self.assertEqual(header.npoints, 12)
        self.assertAlmostEqual(header.start_time, 10.0)
        self.assertTrue(math.isfinite(coeffs.cl))
        self.assertTrue(math.isfinite(coeffs.cd))
        self.assertTrue(math.isfinite(coeffs.l_over_d))


if __name__ == "__main__":
    unittest.main()
