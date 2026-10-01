"""The packaged phantom slab: the toy tier's object.

The phantom aslscan simulates is piecewise constant: each of gray matter, white matter, and
CSF has one perfusion, transit time, T1, T2, T2* and M0 (:mod:`aslbook.presets`). Everything
the toy tier needs is therefore the fraction of each tissue in every acquisition voxel, which
is what ``aslbook/data/phantom_slab.npz`` holds: the 1 mm ASLDRO phantom box-averaged onto the
book's acquisition grid (64 x 68 x 20 at 3.5 x 3.5 x 5 mm) by exactly the rule aslscan uses
(:mod:`aslbook.grid`), plus one 1 mm slice of the label map for figures that need fine detail.
The file is written by ``pipelines/scripts/prepare_phantom.py`` from the same source the
pipeline crops for the simulator, so the toy and pipeline tiers share one brain.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from . import presets

_DATA = Path(__file__).with_name("data")

#: Axial slice of the slab shown throughout the book (through the lateral ventricles).
DISPLAY_SLICE = 9


@lru_cache(maxsize=1)
def slab() -> dict:
    """Tissue fractions of the phantom on the acquisition grid.

    Keys: ``gm``, ``wm``, ``csf`` (fractions, ``(x, y, z)`` float32), ``mask`` (any tissue),
    ``dseg`` (majority label: 1 GM, 2 WM, 3 CSF, 0 background), ``fieldmap`` (the synthetic
    B0 field of the ``slab-fieldmap`` phantom in Hz, box-averaged to the acquisition grid),
    ``voxel_mm``, ``affine``, ``provenance``. Axes are the phantom's: x left-right, y
    posterior-anterior, z inferior-superior, so ``take_slice`` in :mod:`aslbook.plotting`
    shows anterior at the top.
    """
    with np.load(_DATA / "phantom_slab.npz") as f:
        out = {k: f[k].astype(np.float32) for k in ("gm", "wm", "csf")}
        out["dseg"] = f["dseg"].astype(np.int16)
        out["fieldmap"] = f["fieldmap"].astype(np.float32) if "fieldmap" in f else np.zeros(out["gm"].shape, np.float32)
        out["voxel_mm"] = tuple(float(v) for v in f["voxel_mm"])
        out["affine"] = f["affine"].astype(float)
        out["provenance"] = str(f["provenance"])
    out["mask"] = (out["gm"] + out["wm"] + out["csf"]) > 0.5
    return out


@lru_cache(maxsize=1)
def fine_slice() -> dict:
    """One axial slice of the 1 mm label map (``dseg``, int16) at the level of
    :data:`DISPLAY_SLICE`, with its z index in the 1 mm slab and the voxel size."""
    with np.load(_DATA / "phantom_slice_1mm.npz") as f:
        return {"dseg": f["dseg"].astype(np.int16), "z_mm": float(f["z_mm"]), "voxel_mm": 1.0,
                "acq_slice": int(f["acq_slice"])}


def fractions(ph: dict | None = None) -> dict[str, np.ndarray]:
    """``{"GM": ..., "WM": ..., "CSF": ...}`` fraction maps."""
    ph = ph or slab()
    return {"GM": ph["gm"], "WM": ph["wm"], "CSF": ph["csf"]}


def maps(ph: dict | None = None) -> dict[str, np.ndarray]:
    """The ground-truth maps on the acquisition grid, by aslscan's resampling rules: the
    volume-weighted mean of each constant (``perfusion``, ``t1``, ``t2``, ``t2star``, ``m0``)
    and, for ``att``, the mean over the perfused tissues only (CSF's sentinel never leaks)."""
    fr = fractions(ph)
    out = {}
    for key in ("perfusion", "t1", "t2", "t2star", "m0"):
        out[key] = sum(fr[n] * getattr(t, key) for n, t in presets.TISSUES.items())
    perfused = fr["GM"] + fr["WM"]
    num = fr["GM"] * presets.TISSUES["GM"].att + fr["WM"] * presets.TISSUES["WM"].att
    out["att"] = np.divide(num, perfused, out=np.zeros_like(num), where=perfused > 0)
    return out


def class_maps_from_labels(dseg: np.ndarray, key: str) -> np.ndarray:
    """A constant-per-class map from a label image (for the 1 mm slice)."""
    out = np.zeros(dseg.shape, np.float32)
    for t in presets.TISSUES.values():
        out[dseg == t.label] = getattr(t, key)
    return out
