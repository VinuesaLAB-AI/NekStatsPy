"""Convenience wrapper around a processed ``Re200k`` result (.mat or in-memory)."""

import numpy as np
import scipy.io as sio

_SIDE = {"ss": "top", "top": "top", "suction": "top",
         "ps": "bot", "bot": "bot", "pressure": "bot"}


class Result:
    """Uniform accessor for the nested ``Re200k`` struct produced by the pipeline.

    Parameters
    ----------
    ss, ps : numpy structured arrays of shape (1, nprof) with the exported fields.
    """

    def __init__(self, ss, ps, name=""):
        self.ss = ss
        self.ps = ps
        self.name = name

    # ---- constructors -----------------------------------------------------
    @classmethod
    def load(cls, mat_file, name=None):
        R = sio.loadmat(mat_file)["Re200k"][0][0]
        return cls(R["top"][0][0]["ref"], R["bot"][0][0]["ref"],
                   name=name or mat_file)

    @classmethod
    def from_refs(cls, top_ref, bot_ref, name=""):
        from .export import _struct_array
        return cls(_struct_array(top_ref), _struct_array(bot_ref), name=name)

    # ---- access -----------------------------------------------------------
    def _arr(self, side):
        return self.ss if _SIDE[side] == "top" else self.ps

    @property
    def n(self):
        return self.ss.shape[1]

    def fields(self):
        return self.ss.dtype.names

    def scal(self, side, field):
        """1-D array of a scalar field over all profiles."""
        a = self._arr(side)
        return np.array([float(np.ravel(a[field][0, p])[0]) for p in range(a.shape[1])])

    def prof(self, side, p, field):
        """Wall-normal profile of `field` at profile index `p`."""
        return np.asarray(self._arr(side)[field][0, p], float).ravel()

    def station(self, side, xc):
        """Index of the first profile with xa >= xc."""
        xa = self.scal(side, "xa")
        return int(np.where(xa >= xc)[0][0])

    def yplus(self, side, p):
        a = self._arr(side)
        ut = float(a["ut"][0, p]); nu = float(a["nu"][0, p])
        return self.prof(side, p, "yn") * ut / nu
