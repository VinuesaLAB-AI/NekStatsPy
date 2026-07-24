"""
nekstat -- post-processing of Nek5000 on-the-fly wing statistics.

Python port of the MATLAB pipeline for NACA-4412 (Re=200k, AoA=5 deg): read the
interpolated ``ZSTAT/int_fld`` field, rotate every statistic into the wall frame
(mean flow, Reynolds stresses, full turbulent-KE budget), compute boundary-layer
integral quantities and the Cl/Cd decomposition, and write a ``Re200k`` .mat.

Quick start
-----------
>>> import nekstat
>>> res = nekstat.run("path/to/nek", "path/to/int_mesh.mat", "out.mat")
>>> res.scal("ss", "Cf")            # suction-side skin friction vs profile
>>> from nekstat import viz
>>> viz.plot_single(res, "figs/")
"""

from .pipeline import load_mesh, process, run, load_result
from .result import Result
from .read_intfld import read_data_profiles
from .postproc import load_prof_data
from .export import export_drag_decomposition, save_Re200k
from .naca_geom import build_mesh, save_inter_mesh
from ._data import data_path, sample_paths

__version__ = "0.1.0"

__all__ = [
    "load_mesh", "process", "run", "load_result", "Result",
    "read_data_profiles", "load_prof_data",
    "export_drag_decomposition", "save_Re200k",
    "build_mesh", "save_inter_mesh",
    "data_path", "sample_paths", "viz",
]

from . import viz  # noqa: E402
