"""Locate the bundled sample dataset.

The sample data (int_fld ~33 MB, mesh, reference output) lives in the repository
``data/`` directory, not inside the installed wheel.  These helpers find it when
running from a source checkout / editable install; for a plain wheel install the
user supplies their own paths.
"""

import os

_ENV = "NEKSTAT_DATA"


def data_path():
    """Return the repository ``data/`` directory, or raise a helpful error."""
    env = os.environ.get(_ENV)
    if env and os.path.isdir(env):
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    # walk up looking for a sibling data/ (works for src/ layout checkouts)
    d = here
    for _ in range(5):
        d = os.path.dirname(d)
        cand = os.path.join(d, "data")
        if os.path.isdir(os.path.join(cand, "ZSTAT")):
            return cand
    raise FileNotFoundError(
        "Sample data not found. Set the NEKSTAT_DATA environment variable to the "
        "directory containing ZSTAT/int_fld and int_mesh.mat, or pass explicit "
        "paths to nekstat.run().")


def sample_paths():
    """Return dict of paths for the bundled sample case."""
    d = data_path()
    return {
        "nek_dir": d,                                     # contains ZSTAT/int_fld
        "mesh": os.path.join(d, "int_mesh.mat"),
        "reference": os.path.join(
            d, "reference", "stat_benchmark_N11_naca4412_200k.mat"),
    }
