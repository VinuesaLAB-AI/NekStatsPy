import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.io as sio

from nekstat.naca_geom import (
    build_mesh_from_spec,
    validate_mesh,
    write_mesh_artifacts,
)
from nekstat.nek5000 import prepare_run


REPO_ROOT = Path(__file__).resolve().parents[1]


def read_fortran_coordinate(path: Path):
    with path.open("rb") as handle:
        first = struct.unpack("<i", handle.read(4))[0]
        npoints = struct.unpack("<i", handle.read(first))[0]
        assert struct.unpack("<i", handle.read(4))[0] == first
        second = struct.unpack("<i", handle.read(4))[0]
        values = np.frombuffer(handle.read(second), dtype="<f8").copy()
        assert struct.unpack("<i", handle.read(4))[0] == second
    return npoints, values


def write_sts(path: Path, end_time: float) -> None:
    header = (
        f"#std 8 12 12 1 4692 12345 {end_time:20.13E} "
        "500 0 1 aa 66"
    )
    path.write_bytes(header.encode("ascii").ljust(132, b" "))


class MeshGenerationTests(unittest.TestCase):
    def test_reference_presets_reproduce_bundled_meshes(self):
        cases = (
            ("naca4412_reference", REPO_ROOT / "data" / "int_mesh.mat", 1e-8),
            ("naca0012_reference", REPO_ROOT / "data" / "meshes" / "naca0012" / "int_mesh.mat", 1e-8),
        )
        for preset, reference_path, tolerance in cases:
            with self.subTest(preset=preset):
                mesh = build_mesh_from_spec({"preset": preset})
                reference = sio.loadmat(reference_path)
                for field, generated in mesh.mat_dict().items():
                    baseline = np.asarray(reference[field]).ravel()
                    self.assertEqual(generated.size, baseline.size)
                    self.assertLessEqual(
                        np.max(np.abs(generated.ravel() - baseline)), tolerance,
                        field,
                    )

    def test_artifacts_preserve_matlab_and_fortran_coordinates(self):
        mesh = build_mesh_from_spec({"preset": "naca4412_reference"})
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = write_mesh_artifacts(mesh, tmp, write_figure=False)
            saved = sio.loadmat(artifacts.mat_file)
            nx, x = read_fortran_coordinate(artifacts.x_fort)
            ny, y = read_fortran_coordinate(artifacts.y_fort)
        self.assertEqual(nx, mesh.npoints)
        self.assertEqual(ny, mesh.npoints)
        self.assertTrue(np.array_equal(x, saved["x_pts"].ravel()))
        self.assertTrue(np.array_equal(y, saved["y_pts"].ravel()))

    def test_cartesian_mesh_uses_matlab_loop_order_and_rejects_profile_post(self):
        mesh = build_mesh_from_spec({
            "kind": "cartesian",
            "cartesian": {
                "x": {"start": -1.0, "stop": 1.0, "spacing": 1.0},
                "y": {"start": 0.0, "stop": 1.0, "spacing": 0.5},
            },
        })
        self.assertTrue(np.array_equal(mesh.x_pts, [-1, -1, -1, 0, 0, 0, 1, 1, 1]))
        self.assertTrue(np.array_equal(mesh.y_pts, [0, .5, 1, 0, .5, 1, 0, .5, 1]))
        with self.assertRaisesRegex(ValueError, "does not support profile"):
            validate_mesh(mesh, require="profile_post")

    def test_profile_can_explicitly_include_the_trailing_edge(self):
        mesh = build_mesh_from_spec({
            "preset": "naca4412_reference",
            "chordwise": {"include_trailing_edge": True},
        })
        self.assertEqual(mesh.xc[-1], 1.0)
        self.assertEqual(mesh.npoints, 2 * len(mesh.xc) * len(mesh.yn))

    def test_artifacts_cannot_escape_the_output_directory(self):
        mesh = build_mesh_from_spec({"preset": "naca4412_reference"})
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "below the mesh output"):
                write_mesh_artifacts(mesh, tmp, mat_file="../outside.mat", write_figure=False)

    def test_generated_mesh_is_staged_with_matching_auto_post_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            deps = root / "deps"
            source = root / "source" / "res_1" / "sts_data"
            deps.mkdir()
            source.mkdir(parents=True)
            for name in ("SIZE", "makefile_usr.inc", "naca_wing.usr", "naca_wing.re2", "naca_wing.map"):
                (deps / name).write_text(name, encoding="utf-8")
            (deps / "naca_wing.rea").write_text(
                "  3.00000E+00      p068 Last file\n"
                "  1.00000E+00      p069 First file\n"
                "  0.00000E+00      p070 Start time\n",
                encoding="utf-8",
            )
            executable = deps / "nek5000_4412"
            executable.write_text("#!/bin/sh\n", encoding="utf-8")
            executable.chmod(0o755)
            write_sts(source / "stsnaca_wing0.f00001", 1.0)
            conf = {
                "run": {"name": "generated_mesh", "root": str(root / "runs")},
                "case": {
                    "id": "naca4412", "session": "naca_wing",
                    "dependencies": str(deps), "executable": str(executable),
                },
                "mesh": {"mode": "generate", "preset": "naca4412_reference"},
                "source": {
                    "path": str(root / "source"), "pattern": "res_*/sts_data/stsnaca_wing0.f*",
                    "first_file": 1, "last_file": 1,
                },
                "nek": {"sampling_start_time": "auto", "initial_start_time": 0.0},
                "post": {"mesh": "auto", "out": "stat.mat"},
            }
            paths = prepare_run(conf, base_dir=root)
            staged = sio.loadmat(paths.root / "int_mesh.mat")
            self.assertTrue((paths.zstat / "x.fort").is_file())
            self.assertTrue((paths.zstat / "y.fort").is_file())
            self.assertEqual(np.asarray(staged["x_pts"]).size, 31540)
            self.assertIn("alphau", staged)


if __name__ == "__main__":
    unittest.main()
