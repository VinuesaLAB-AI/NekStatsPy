"""Reader for KTH Framework/PSTAT2D channel ``int_fld`` files."""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union

import numpy as np

Pathish = Union[str, os.PathLike]
HEADER_BYTES = 460
ENDIAN_TAG = 6.54321


@dataclass(frozen=True)
class ChannelIntFld:
    """Vectorized contents of a KTH PSTAT2D interpolation file."""

    path: Path
    word_size: int
    byte_order: str
    reynolds_number: float
    box_size: np.ndarray
    box_elements: np.ndarray
    polynomial_order: np.ndarray
    nstat: int
    nderiv: int
    start_time: float
    end_time: float
    effective_average_time: float
    time_step: float
    nrec: int
    time_interval: float
    npoints: int
    coordinates: np.ndarray
    stats: np.ndarray
    derivatives: np.ndarray

    @property
    def x(self) -> np.ndarray:
        return self.coordinates[:, 0]

    @property
    def y(self) -> np.ndarray:
        return self.coordinates[:, 1]


def read_channel_int_fld(path: Pathish) -> ChannelIntFld:
    """Read a KTH PSTAT2D ``int_fld`` file into dense NumPy arrays."""

    path = Path(path)
    with path.open("rb") as handle:
        header = handle.read(HEADER_BYTES).split()
        if len(header) < 2 or header[0] != b"#iv1":
            raise ValueError(f"{path}: not a KTH PSTAT2D int_fld file")
        word_size = int(header[1])
        if word_size not in (4, 8):
            raise ValueError(f"{path}: unsupported real word size {word_size}")

        tag_bytes = handle.read(4)
        byte_order = _byte_order_from_tag(tag_bytes, path)
        real_dtype = np.dtype(f"{byte_order}f{word_size}")
        int_dtype = np.dtype(f"{byte_order}i4")

        reynolds_number = _read_real(handle, real_dtype, 1, path)[0]
        box_size = _read_real(handle, real_dtype, 3, path)
        box_elements = _read_int(handle, int_dtype, 3, path)
        polynomial_order = _read_int(handle, int_dtype, 3, path)
        nstat = int(_read_int(handle, int_dtype, 1, path)[0])
        nderiv = int(_read_int(handle, int_dtype, 1, path)[0])
        times = _read_real(handle, real_dtype, 4, path)
        nrec = int(_read_int(handle, int_dtype, 1, path)[0])
        time_interval = float(_read_real(handle, real_dtype, 1, path)[0])
        npoints = int(_read_int(handle, int_dtype, 1, path)[0])

        ldim = 2
        coordinates = _read_real(handle, real_dtype, npoints * ldim, path).reshape(
            npoints,
            ldim,
        )
        stats_raw = _read_real(handle, real_dtype, nstat * npoints, path)
        deriv_raw = _read_real(handle, real_dtype, nderiv * npoints, path)

    stats = stats_raw.reshape(nstat, npoints).T
    derivatives = deriv_raw.reshape(nderiv, npoints).T
    return ChannelIntFld(
        path=path,
        word_size=word_size,
        byte_order=byte_order,
        reynolds_number=float(reynolds_number),
        box_size=box_size.astype(float, copy=False),
        box_elements=box_elements.astype(int, copy=False),
        polynomial_order=polynomial_order.astype(int, copy=False),
        nstat=nstat,
        nderiv=nderiv,
        start_time=float(times[0]),
        end_time=float(times[1]),
        effective_average_time=float(times[2]),
        time_step=float(times[3]),
        nrec=nrec,
        time_interval=time_interval,
        npoints=npoints,
        coordinates=coordinates.astype(float, copy=False),
        stats=stats.astype(float, copy=False),
        derivatives=derivatives.astype(float, copy=False),
    )


def channel_database(data: ChannelIntFld, nx: int, ny: int) -> Dict[str, np.ndarray]:
    """Create ``x``, ``y``, ``F*``, and ``D*`` fields shaped as ``(nx, ny)``."""

    expected = nx * ny
    if expected > data.npoints:
        raise ValueError(
            f"nx*ny={expected} exceeds int_fld point count {data.npoints}"
        )

    db: Dict[str, np.ndarray] = {
        "x": data.x[:expected].reshape(nx, ny),
        "y": data.y[:expected].reshape(nx, ny),
    }
    for index in range(data.nstat):
        db[f"F{index + 1}"] = data.stats[:expected, index].reshape(nx, ny)
    for index in range(data.nderiv):
        db[f"D{index + 1}"] = data.derivatives[:expected, index].reshape(nx, ny)
    return db


def _byte_order_from_tag(tag_bytes: bytes, path: Path) -> str:
    if len(tag_bytes) != 4:
        raise ValueError(f"{path}: truncated endian tag")
    for byte_order in ("<", ">"):
        tag = struct.unpack(f"{byte_order}f", tag_bytes)[0]
        if abs(tag - ENDIAN_TAG) < 1.0e-5:
            return byte_order
    raise ValueError(f"{path}: invalid endian tag")


def _read_real(handle, dtype: np.dtype, count: int, path: Path) -> np.ndarray:
    return _read_array(handle, dtype, count, path)


def _read_int(handle, dtype: np.dtype, count: int, path: Path) -> np.ndarray:
    return _read_array(handle, dtype, count, path)


def _read_array(handle, dtype: np.dtype, count: int, path: Path) -> np.ndarray:
    payload = handle.read(dtype.itemsize * count)
    if len(payload) != dtype.itemsize * count:
        raise ValueError(f"{path}: truncated binary payload")
    return np.frombuffer(payload, dtype=dtype, count=count).copy()
