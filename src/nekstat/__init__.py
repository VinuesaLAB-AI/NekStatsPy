"""
nekstat -- post-processing workflows for Nek5000 statistics.

Python port of the MATLAB pipeline for NACA-4412 (Re=200k, AoA=5 deg): read the
interpolated ``ZSTAT/int_fld`` field, rotate every statistic into the wall frame
(mean flow, Reynolds stresses, full turbulent-KE budget), compute boundary-layer
integral quantities and the Cl/Cd decomposition, and write a ``Re200k`` .mat.
Also includes a KTH Framework/PSTAT2D channel-flow workflow that replaces the
MATLAB ``post_STAT.m`` wrapper with pure Python.

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
from .nek5000 import (
    BuildPaths,
    RunPaths,
    StsFile,
    build_from_config,
    build_solver,
    load_config,
    prepare_run,
    read_sts_header,
    run_from_config,
)
from .clcd_backward import (
    BackwardWindow,
    ClCdBackwardPaths,
    ClCdCoefficients,
    IntFldHeader,
    build_backward_windows,
    compute_clcd_from_int_fld,
    parse_int_fld_header,
    run_clcd_backward_from_config,
)
from .channel_intfld import ChannelIntFld, channel_database, read_channel_int_fld
from .channel_pipeline import (
    ChannelPostResult,
    compare_channel_mat_files,
    process_channel,
    run_channel_post,
    run_channel_post_from_config,
)
from .kth_channel import (
    ChannelRunPaths,
    ChannelStsFile,
    build_channel_from_config,
    prepare_channel_run,
    run_channel_from_config,
)

__version__ = "0.1.0"

__all__ = [
    "load_mesh", "process", "run", "load_result", "Result",
    "read_data_profiles", "load_prof_data",
    "export_drag_decomposition", "save_Re200k",
    "build_mesh", "save_inter_mesh",
    "BuildPaths", "RunPaths", "StsFile",
    "build_from_config", "build_solver",
    "load_config", "prepare_run", "read_sts_header", "run_from_config",
    "BackwardWindow", "ClCdBackwardPaths", "ClCdCoefficients", "IntFldHeader",
    "build_backward_windows", "compute_clcd_from_int_fld",
    "parse_int_fld_header", "run_clcd_backward_from_config",
    "ChannelIntFld", "channel_database", "read_channel_int_fld",
    "ChannelPostResult", "compare_channel_mat_files",
    "process_channel", "run_channel_post", "run_channel_post_from_config",
    "ChannelRunPaths", "ChannelStsFile", "build_channel_from_config",
    "prepare_channel_run", "run_channel_from_config",
    "data_path", "sample_paths", "viz",
]

from . import viz  # noqa: E402
