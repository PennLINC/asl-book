"""The ground-truth maps of a simulated run, and what each one is.

aslscan writes the answer key next to every series (``sub-01/perf/ground-truth``), resampled
from the 1 mm phantom to the acquisition grid by the rules in :mod:`aslbook.grid`; the
pipeline adds the tissue fractions of every acquisition voxel (``derivatives/aslbook``). This
module names them for Appendix E and the chapters.
"""

from __future__ import annotations

import numpy as np

#: name -> (units, definition, chapters that use it)
TRUTH_MAPS: dict[str, tuple[str, str, str]] = {
    "perfusion": ("ml/100 g/min", "cerebral blood flow: the volume-weighted mean of the phantom's per-tissue values (GM 60, WM 20, CSF 0) over the phantom voxels each acquisition voxel overlaps", "12, 14, 15, 16, 17"),
    "att": ("s", "arterial transit time: the mean over the perfused tissues only (GM 0.8, WM 1.2 s), so CSF's sentinel never enters", "5, 15"),
    "T1map": ("s", "tissue T1, volume-weighted mean (GM 1.33, WM 0.83, CSF 3.0 s)", "1, 9, 16"),
    "T2map": ("s", "tissue T2, volume-weighted mean (GM 0.08, WM 0.11, CSF 0.30 s)", "1, 2"),
    "M0map": ("arbitrary", "equilibrium magnetization, volume-weighted mean; the M0 scan measures it times the saturation and T2 factors of its readout", "6, 16"),
    "dseg": ("label", "majority tissue label of each acquisition voxel (1 GM, 2 WM, 3 CSF, 0 background); ties go to the lower label", "12"),
    "deltam": ("M0 units", "the noise-free label-control magnetization difference of every label (and deltam) volume at its own timing, box-averaged like the data; zero on other rows; the moved truth in motion runs", "6, 8, 9, 10, 11, 14, 15"),
    "deltamStatic": ("M0 units", "the unmoved deltam of a motion run", "10"),
    "motion": ("mm, rad", "the applied per-volume pose (translations and rotations) and any within-volume events, as a TSV", "10"),
    "label-GM_probseg": ("fraction", "gray matter fraction of each acquisition voxel, the box average of the 1 mm label map (pipeline derivative)", "12, 14"),
    "label-WM_probseg": ("fraction", "white matter fraction", "12, 14"),
    "label-CSF_probseg": ("fraction", "CSF fraction", "12"),
}


def truth_slice(vol: np.ndarray, k: int) -> np.ndarray:
    """One axial slice of a 3-D truth map, anterior up."""
    return np.rot90(vol[:, :, k])
