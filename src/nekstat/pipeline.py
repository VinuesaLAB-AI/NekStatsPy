"""High-level pipeline: int_fld -> processed Re200k result.

This is the Python equivalent of 02_MatLab/main_post_proc_prof.m, exposed as a
small functional API.
"""

import scipy.io as sio

from .read_intfld import read_data_profiles
from .postproc import load_prof_data
from .export import export_drag_decomposition, save_Re200k
from .result import Result

__all__ = ["load_mesh", "process", "run", "load_result"]


def load_mesh(mesh_file):
    """Load an int_mesh.mat with keys x_pts, y_pts, alphau, alphal, yn, xc."""
    return sio.loadmat(mesh_file)


def process(nek_dir, mesh_file, istart=25, iend=80, long=1):
    """Read + rotate + integrate; return (top_ref, bot_ref) export lists.

    Parameters
    ----------
    nek_dir : str   directory containing ``ZSTAT/int_fld`` (+ x.fort, y.fort).
    mesh_file : str path to int_mesh.mat.
    istart, iend : int  MATLAB 1-based profile window to export (default 25..80).
    long : int      1 -> also compute the turbulent-budget tensors.
    """
    mesh = load_mesh(mesh_file)
    top, bottom = read_data_profiles(nek_dir, mesh, long=long)
    top, bot = load_prof_data(top, bottom, istart, iend, budget_flag=long)
    top_ref = export_drag_decomposition(top, istart, iend)
    bot_ref = export_drag_decomposition(bot, istart, iend)
    return top_ref, bot_ref


def run(nek_dir, mesh_file, out_file=None, istart=25, iend=80, long=1):
    """Full pipeline; optionally save `out_file` (.mat) and return a Result."""
    top_ref, bot_ref = process(nek_dir, mesh_file, istart, iend, long)
    if out_file:
        save_Re200k(top_ref, bot_ref, out_file)
    return Result.from_refs(top_ref, bot_ref, name=out_file or "in-memory")


def load_result(mat_file):
    """Load a previously-saved stat_*.mat as a Result."""
    return Result.load(mat_file)
