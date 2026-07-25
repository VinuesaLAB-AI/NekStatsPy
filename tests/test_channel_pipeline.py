import math
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.io as sio

from nekstat.channel_intfld import channel_database, read_channel_int_fld
from nekstat.channel_pipeline import (
    compare_channel_mat_files,
    process_channel,
    run_channel_post,
)
from nekstat.kth_channel import prepare_channel_run, run_channel_nek
from nekstat.nek5000 import read_sts_header


def write_channel_int_fld(path: Path, nx: int = 2, ny: int = 3) -> dict:
    npoints = nx * ny
    nstat = 25
    nderiv = 2
    coords = []
    for ix in range(nx):
        for iy in range(ny):
            coords.append((float(ix), 0.1 * iy))
    coords = np.asarray(coords, dtype="<f8")
    x = coords[:, 0]
    y = coords[:, 1]
    u = 1.0 + 0.2 * x + y
    v = 0.01 + 0.001 * x
    w = np.zeros(npoints)
    p = 2.0 + 0.05 * y
    uu = np.full(npoints, 0.04)
    vv = np.full(npoints, 0.01)
    ww = np.full(npoints, 0.02)
    pp = np.full(npoints, 0.09)
    uv = np.full(npoints, -0.003)
    vvv = 0.001 + 0.0001 * y
    dudy = np.full(npoints, 12.0)

    stats = np.zeros((nstat, npoints))
    stats[0] = u
    stats[1] = v
    stats[2] = w
    stats[3] = p
    stats[4] = u * u + uu
    stats[5] = v * v + vv
    stats[6] = w * w + ww
    stats[7] = p * p + pp
    stats[8] = u * v + uv
    stats[24] = vvv + 3.0 * v * vv + v * v * v

    derivs = np.zeros((nderiv, npoints))
    derivs[1] = dudy

    header = (
        "#iv1 8 (Re =  4.600000000E+03) (Nstat =       25) "
        "(Nderiv =        2) (npoints =        6)"
    )
    with path.open("wb") as handle:
        handle.write(header.encode("ascii").ljust(460, b" "))
        handle.write(struct.pack("<f", 6.54321))
        handle.write(np.asarray([4600.0], dtype="<f8").tobytes())
        handle.write(np.asarray([2.0, 1.0, 0.0], dtype="<f8").tobytes())
        handle.write(np.asarray([2, 1, 1], dtype="<i4").tobytes())
        handle.write(np.asarray([12, 12, 1], dtype="<i4").tobytes())
        handle.write(np.asarray([nstat], dtype="<i4").tobytes())
        handle.write(np.asarray([nderiv], dtype="<i4").tobytes())
        handle.write(np.asarray([10.0, 20.0, 5.0, 0.01], dtype="<f8").tobytes())
        handle.write(np.asarray([500], dtype="<i4").tobytes())
        handle.write(np.asarray([10.0], dtype="<f8").tobytes())
        handle.write(np.asarray([npoints], dtype="<i4").tobytes())
        handle.write(coords.astype("<f8").tobytes())
        handle.write(stats.astype("<f8").tobytes())
        handle.write(derivs.astype("<f8").tobytes())
    return {
        "u": u.reshape(nx, ny).T,
        "v": v.reshape(nx, ny).T,
        "uu": uu.reshape(nx, ny).T,
        "uv": uv.reshape(nx, ny).T,
        "dudy": dudy.reshape(nx, ny).T,
    }


def write_sts(path: Path, end_time: float, payload: bytes = b"") -> None:
    header = (
        f"#std 8 12 12 1 4692 12345 {end_time:20.13E} "
        "500 0 1 XS44 0.0 F"
    )
    path.write_bytes(header.encode("ascii").ljust(132, b" ") + payload)


class ChannelPipelineTests(unittest.TestCase):
    def test_read_and_process_synthetic_channel_int_fld(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "int_fld"
            expected = write_channel_int_fld(path)

            raw = read_channel_int_fld(path)
            db = channel_database(raw, 2, 3)
            result = process_channel(path, nx=2, ny=3, reb=4600)

        self.assertEqual(raw.nstat, 25)
        self.assertEqual(raw.nderiv, 2)
        self.assertEqual(raw.npoints, 6)
        self.assertEqual(db["x"].shape, (2, 3))
        self.assertTrue(np.allclose(result.data["Ur"], expected["u"]))
        self.assertTrue(np.allclose(result.data["uvr"], expected["uv"]))
        self.assertTrue(np.allclose(result.data["dUdyr"], expected["dudy"]))
        self.assertAlmostEqual(result.data["twall"], (1.0 / 4600.0) * 12.0)
        self.assertTrue(math.isfinite(result.data["Ret"]))
        self.assertTrue(math.isfinite(result.data["H12"]))

    def test_channel_post_saves_matlab_struct_and_self_validates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            int_fld = root / "int_fld"
            out = root / "channel.mat"
            write_channel_int_fld(int_fld)

            result = run_channel_post(int_fld, nx=2, ny=3, reb=4600, out_file=out, figdir=None)
            mat = sio.loadmat(out)
            diffs = compare_channel_mat_files(out, out)

        self.assertEqual(result.mat_file, out)
        self.assertIn("data", mat)
        self.assertTrue(diffs)
        self.assertTrue(all(value == 0.0 for value in diffs.values()))

    def test_prepare_channel_run_renumbers_stats_and_patches_par(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            deps = root / "deps"
            source = root / "source" / "res_1" / "sts_data"
            for path in (deps / "DATA", source):
                path.mkdir(parents=True)
            for name in ("SIZE", "makefile_usr.inc", "tcf.usr", "tcf.re2", "tcf.ma2", "inputs_phill_pp.in"):
                (deps / name).write_text(name, encoding="utf-8")
            (deps / "tcf.par").write_text(
                "[_PSTAT2D]\n"
                "STS_NFILE            = 5                      # Number of stat files\n"
                "STS_STIME            = 1.0                    # Statistics starting time\n"
                "STS_NSTEP            = 10                     # Number of steps\n",
                encoding="utf-8",
            )
            (deps / "DATA" / "int_pos").write_text("pos", encoding="utf-8")
            (deps / "DATA" / "c2Dtcf0.f00001").write_text("c2d", encoding="utf-8")
            exe = deps / "nek5000_channel"
            exe.write_text("#!/bin/sh\n", encoding="utf-8")
            exe.chmod(0o755)

            write_sts(source / "ststcf0.f00001", 10.0, b"first")
            write_sts(source / "ststcf0.f00002", 20.0, b"second")
            write_sts(source / "ststcf0.f00003", 30.0, b"third")
            conf = {
                "run": {"name": "channel_test", "root": str(root / "runs")},
                "case": {
                    "id": "no_resize_lc",
                    "session": "tcf",
                    "dependencies": str(deps),
                    "executable": str(exe),
                    "overwrite": True,
                },
                "source": {
                    "path": str(root / "source"),
                    "pattern": "res_*/sts_data/ststcf0.f*",
                    "stage_mode": "copy",
                    "first_file": 2,
                    "last_file": 3,
                    "overwrite": True,
                },
                "pstat2d": {"sts_nstep": 20},
                "channel_post": {"out": "channel.mat"},
            }

            paths = prepare_channel_run(conf, base_dir=root)
            staged = paths.data_dir / "ststcf0.f00001"
            par_text = (paths.root / "tcf.par").read_text(encoding="utf-8")

            self.assertEqual(len(paths.staged_stats), 2)
            self.assertEqual(read_sts_header(staged)[1], 20.0)
            self.assertFalse((paths.data_dir / "ststcf0.f00003").exists())
            self.assertIn("STS_NFILE            = 2", par_text)
            self.assertIn("STS_STIME            = 1.000000000000E+01", par_text)
            self.assertIn("STS_NSTEP            = 20", par_text)

    def test_run_channel_nek_validates_fake_completion(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            data_dir = root / "DATA"
            data_dir.mkdir()
            (root / "tcf.sch").write_text("old", encoding="utf-8")
            (root / "st1tcf0.f00001").write_text("old", encoding="utf-8")
            (root / "int_fld").write_text("old", encoding="utf-8")
            (root / "nek5000").write_text(
                "#!/bin/sh\n"
                "printf new > int_fld\n"
                "echo 'run successful: dying ...'\n",
                encoding="utf-8",
            )
            (root / "nek5000").chmod(0o755)
            fake_mpirun = root / "mpirun"
            fake_mpirun.write_text("#!/bin/sh\nshift 2\nexec \"$@\"\n", encoding="utf-8")
            fake_mpirun.chmod(0o755)
            paths = type(
                "Paths",
                (),
                {
                    "root": root,
                    "data_dir": data_dir,
                    "config": root / "current_conf.yml",
                    "logfile": root / "log01.txt",
                    "int_fld": root / "int_fld",
                },
            )()

            run_channel_nek(
                {"case": {"session": "tcf"}, "nek": {"mpirun": str(fake_mpirun), "np": 1}},
                paths,
            )

            self.assertEqual((root / "int_fld").read_text(encoding="utf-8"), "new")
            self.assertFalse((root / "tcf.sch").exists())
            self.assertFalse((root / "st1tcf0.f00001").exists())


if __name__ == "__main__":
    unittest.main()
