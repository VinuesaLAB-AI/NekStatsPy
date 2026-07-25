import re
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.io as sio

from nekstat.nek5000 import load_config, prepare_run


REPO_ROOT = Path(__file__).resolve().parents[1]


def read_fortran_mesh(path: Path):
    with path.open("rb") as handle:
        marker = struct.unpack("<i", handle.read(4))[0]
        payload = handle.read(marker)
        closing_marker = struct.unpack("<i", handle.read(4))[0]
        if marker != closing_marker:
            raise ValueError(f"{path}: mismatched point-count record")
        npoints = struct.unpack("<i", payload)[0]

        marker = struct.unpack("<i", handle.read(4))[0]
        payload = handle.read(marker)
        closing_marker = struct.unpack("<i", handle.read(4))[0]
        if marker != closing_marker:
            raise ValueError(f"{path}: mismatched coordinate record")
    values = np.frombuffer(payload, dtype="<f8").copy()
    return npoints, values


def write_sts(path: Path, end_time: float) -> None:
    header = (
        f"#std 8 12 12 1 4692 12345 {end_time:20.13E} "
        "500 0 1 aa 66"
    )
    path.write_bytes(header.encode("ascii").ljust(132, b" "))


class Naca0012AssetTests(unittest.TestCase):
    def test_committed_fortran_mesh_matches_matlab_mesh(self):
        mesh_dir = REPO_ROOT / "data" / "meshes" / "naca0012"
        mesh = sio.loadmat(mesh_dir / "int_mesh.mat")
        x_mat = np.asarray(mesh["x_pts"], dtype="<f8").ravel()
        y_mat = np.asarray(mesh["y_pts"], dtype="<f8").ravel()

        x_npoints, x_fort = read_fortran_mesh(mesh_dir / "ZSTAT" / "x.fort")
        y_npoints, y_fort = read_fortran_mesh(mesh_dir / "ZSTAT" / "y.fort")

        self.assertEqual(x_npoints, 137120)
        self.assertEqual(y_npoints, 137120)
        self.assertEqual(len(np.asarray(mesh["xc"]).ravel()), 80)
        self.assertEqual(len(np.asarray(mesh["yn"]).ravel()), 857)
        self.assertTrue(np.array_equal(x_fort, x_mat))
        self.assertTrue(np.array_equal(y_fort, y_mat))

    def test_naca0012_config_stages_case_mesh_and_rea_window(self):
        conf = load_config(REPO_ROOT / "configs" / "naca0012_post.yml")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source" / "res_001" / "sts_data"
            source.mkdir(parents=True)
            write_sts(source / "stsnaca_wing0.f00001", 7.0)
            write_sts(source / "stsnaca_wing0.f00002", 7.2)

            conf["run"]["root"] = str(root / "runs")
            conf["run"]["name"] = "naca0012_prepare_check"
            conf["source"]["path"] = str(root / "source")
            conf["source"]["last_file"] = 2
            paths = prepare_run(conf, base_dir=REPO_ROOT / "configs")

            self.assertEqual(len(paths.staged_stats), 2)
            self.assertTrue((paths.root / "A0").is_dir())
            self.assertTrue((paths.zstat / "x.fort").is_file())
            self.assertTrue((paths.zstat / "y.fort").is_file())
            self.assertTrue((paths.root / "nek5000").is_file())

            rea = (paths.root / "naca_wing.rea").read_text(encoding="utf-8")
            self.assertRegex(rea, re.compile(r"2\.00000E\+00\s+p068"))
            self.assertRegex(rea, re.compile(r"1\.00000E\+00\s+p069"))
            self.assertRegex(rea, re.compile(r"7\.000000000028E\+00\s+p070"))


if __name__ == "__main__":
    unittest.main()
