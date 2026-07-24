import stat
import tarfile
import tempfile
import unittest
from pathlib import Path

from nekstat.nek5000 import (
    RunPaths,
    build_solver,
    patch_rea_params,
    prepare_run,
    read_sts_header,
    run_nek,
)


def write_sts(path: Path, end_time: float, payload: bytes = b"") -> None:
    header = (
        f"#std 8 12 12 1 4692 12345 {end_time:20.13E} "
        "500 0 1 aa 66"
    )
    path.write_bytes(header.encode("ascii").ljust(132, b" ") + payload)


class Nek5000RunTests(unittest.TestCase):
    def test_build_solver_extracts_archive_and_installs_executable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_src = root / "archive_src" / "Nek1093_dong" / "trunk" / "nek"
            archive_src.mkdir(parents=True)
            (archive_src / "marker").write_text("solver", encoding="utf-8")
            archive = root / "Nek1093_dong.tar.gz"
            with tarfile.open(archive, "w") as tar:
                tar.add(root / "archive_src" / "Nek1093_dong", arcname="Nek1093_dong")

            work = root / "case"
            work.mkdir()
            (work / "build.sh").write_text(
                "printf '#!/bin/sh\\n' > nek5000\nchmod +x nek5000\n",
                encoding="utf-8",
            )
            output = root / "deps" / "nek5000_4412"
            conf = {
                "case": {"id": "naca4412", "session": "naca_wing"},
                "build": {
                    "archive": str(archive),
                    "source_dir": str(root / "Nek1093_dong"),
                    "work_dir": str(work),
                    "command": ["sh", "build.sh"],
                    "output": str(output),
                    "logfile": "build.log",
                    "skip_if_exists": False,
                },
            }

            paths = build_solver(conf, base_dir=root)

            self.assertTrue((root / "Nek1093_dong" / "trunk" / "nek" / "marker").exists())
            self.assertEqual(paths.installed_executable, output)
            self.assertTrue(output.exists())
            self.assertTrue(output.stat().st_mode & stat.S_IXUSR)
            self.assertTrue((work / "build.log").exists())

    def test_read_sts_header_extracts_end_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            sts = Path(tmp) / "stsnaca_wing0.f00001"
            write_sts(sts, 15.525)

            header, end_time = read_sts_header(sts)

        self.assertTrue(header.startswith("#std"))
        self.assertAlmostEqual(end_time, 15.525)

    def test_patch_rea_params_preserves_comments(self):
        with tempfile.TemporaryDirectory() as tmp:
            rea = Path(tmp) / "case.rea"
            rea.write_text(
                "  3.00000E+00      p068 Number of stat files\n"
                "  1.00000E+00      p069 First file\n"
                "  0.15450E+02      p070 Sampling start\n",
                encoding="utf-8",
            )

            patch_rea_params(rea, {68: 12, 69: 2, 70: 25.5})
            text = rea.read_text(encoding="utf-8")

        self.assertIn("1.20000E+01", text)
        self.assertIn("p068 Number of stat files", text)
        self.assertIn("2.00000E+00", text)
        self.assertIn("2.550000000000E+01", text)

    def test_prepare_run_stages_workspace_and_infers_start_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            deps = root / "deps"
            mesh = root / "mesh"
            source = root / "source"
            for path in (deps, mesh, source / "res_2" / "sts_data", source / "res_1" / "sts_data"):
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
                "  3.00000E+00      p068 Number of stat files (LAST FILE, NOW!)\n"
                "  1.00000E+00      p069 (FIRST FILE!)\n"
                "  0.15450E+02      p070 Start Time of time-series sampling.\n",
                encoding="utf-8",
            )
            exe = deps / "nek5000_4412"
            exe.write_text("#!/bin/sh\n", encoding="utf-8")
            exe.chmod(0o755)
            (mesh / "x.fort").write_text("x", encoding="utf-8")
            (mesh / "y.fort").write_text("y", encoding="utf-8")

            write_sts(
                source / "res_2" / "sts_data" / "stsnaca_wing0.f00001",
                30.0,
                b"third",
            )
            write_sts(
                source / "res_1" / "sts_data" / "stsnaca_wing0.f00002",
                20.0,
                b"second",
            )
            write_sts(
                source / "res_1" / "sts_data" / "stsnaca_wing0.f00001",
                10.0,
                b"first",
            )

            conf = {
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
                    "path": str(source),
                    "pattern": "res_*/sts_data/stsnaca_wing0.f*",
                    "stage_mode": "copy",
                    "first_file": 2,
                    "last_file": 3,
                    "overwrite": True,
                },
                "nek": {"sampling_start_time": "auto"},
                "post": {"mesh": "data/int_mesh.mat", "out": "stat.mat"},
            }

            paths = prepare_run(conf, base_dir=root)
            staged = paths.zstat / "stsnaca_wing0.f00002"
            rea_text = (paths.root / "naca_wing.rea").read_text(encoding="utf-8")
            session_text = (paths.root / "SESSION.NAME").read_text(encoding="utf-8")

            self.assertEqual(read_sts_header(staged)[1], 20.0)
            self.assertFalse((paths.zstat / "stsnaca_wing0.f00001").exists())
            self.assertTrue((paths.zstat / "stsnaca_wing0.f00003").exists())
            self.assertTrue((paths.root / "nek5000").exists())
            self.assertTrue((paths.root / "nek5000").stat().st_mode & stat.S_IXUSR)
            self.assertTrue((paths.root / "A0").is_dir())
            self.assertTrue((paths.root / "current_conf.yml").exists())
            self.assertEqual(session_text.splitlines()[0], "naca_wing")
            self.assertIn("3.00000E+00", rea_text)
            self.assertIn("2.00000E+00", rea_text)
            self.assertIn("1.000000000000E+01", rea_text)

    def test_prepare_run_requires_initial_time_without_predecessor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
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
                "  1.00000E+00      p068 Number of stat files\n"
                "  1.00000E+00      p069 First file\n"
                "  0.00000E+00      p070 Sampling start\n",
                encoding="utf-8",
            )
            exe = deps / "nek5000_4412"
            exe.write_text("#!/bin/sh\n", encoding="utf-8")
            exe.chmod(0o755)
            (mesh / "x.fort").write_text("x", encoding="utf-8")
            (mesh / "y.fort").write_text("y", encoding="utf-8")
            write_sts(source / "stsnaca_wing0.f00001", 10.0)

            conf = {
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
                    "first_file": 1,
                    "last_file": "auto",
                    "overwrite": True,
                },
                "nek": {"sampling_start_time": "auto"},
            }

            with self.assertRaisesRegex(ValueError, "Cannot infer"):
                prepare_run(conf, base_dir=root)

    def test_run_nek_prepares_a0_and_validates_output(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            zstat = root / "ZSTAT"
            a0 = root / "A0"
            zstat.mkdir()
            a0.mkdir()
            (a0 / "stale").write_text("old", encoding="utf-8")
            (zstat / "int_fld").write_text("old", encoding="utf-8")
            (root / "naca_wing.sch").write_text("old", encoding="utf-8")
            (root / "nek5000").write_text(
                "#!/bin/sh\n"
                "printf new > ZSTAT/int_fld\n"
                "printf field > A0/new_field\n"
                "echo 'Output interporlated field in int_fld'\n",
                encoding="utf-8",
            )
            (root / "nek5000").chmod(0o755)
            fake_mpirun = root / "mpirun"
            fake_mpirun.write_text("#!/bin/sh\nshift 2\nexec \"$@\"\n", encoding="utf-8")
            fake_mpirun.chmod(0o755)
            paths = RunPaths(
                root=root,
                zstat=zstat,
                config=root / "current_conf.yml",
                logfile=root / "logfile",
                stat_file=None,
            )
            conf = {
                "case": {"id": "naca4412", "session": "naca_wing"},
                "nek": {"mpirun": str(fake_mpirun), "np": 1},
            }

            run_nek(conf, paths)

            self.assertFalse((a0 / "stale").exists())
            self.assertFalse((root / "naca_wing.sch").exists())
            self.assertEqual((zstat / "int_fld").read_text(encoding="utf-8"), "new")
            self.assertTrue((a0 / "new_field").exists())


if __name__ == "__main__":
    unittest.main()
