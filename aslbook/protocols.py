"""The book's ASL protocols as BIDS sidecars and ``aslcontext`` tables.

aslscan reads a real BIDS ASL sidecar (``*_asl.json``) and ``*_aslcontext.tsv``, so a protocol
in this book is exactly those two things. The functions below build them for the labeling
schemes the book uses; the pipeline writes them to disk for the simulator, and the toy tier
reads the same dictionaries to evaluate the kinetic model in the page. Times in seconds,
voxels in mm, as BIDS requires.
"""

from __future__ import annotations

from typing import Any

from . import presets

Protocol = dict[str, Any]


def _base(pe: str = "j-") -> Protocol:
    r = presets.REFERENCE
    return {
        "Manufacturer": "aslscan",
        "MagneticFieldStrength": presets.B0_T,
        "MRAcquisitionType": "2D",
        "PulseSequenceType": "EPI",
        "PhaseEncodingDirection": pe,
        "BackgroundSuppression": False,
        "VascularCrushing": False,
        "FlipAngle": 90,
        "RepetitionTimePreparation": r.repetition_time,
        "EchoTime": r.echo_time,
        "TotalReadoutTime": r.total_readout_time,
        "AcquisitionVoxelSize": list(r.voxel_mm),
        "SliceTiming": r.slice_timing,
    }


def slice_timing(n_slices: int, spacing: float) -> list[float]:
    """Sequential 2D slice timing: slice ``z`` is excited ``z * spacing`` after the first."""
    return [round(i * spacing, 6) for i in range(n_slices)]


def pcasl(
    pld: float | list[float] = presets.REFERENCE.post_labeling_delay,
    ld: float = presets.REFERENCE.labeling_duration,
    n_pairs: int = presets.REFERENCE.n_pairs,
    tr: float = presets.REFERENCE.repetition_time,
    m0_type: str = "Separate",
    label_type: str = "PCASL",
    pe: str = "j-",
    voxel_mm: tuple[float, float, float] | None = None,
    n_slices: int | None = None,
    slice_spacing: float | None = None,
    te: float | None = None,
    background_suppression: list[float] | None = None,
    extra: dict[str, Any] | None = None,
) -> Protocol:
    """A (P)CASL sidecar. ``pld`` may be a list with one entry per acquired volume (multi-delay).

    ``background_suppression`` is a list of pulse times in seconds from the start of labeling;
    when given, the sidecar carries the BIDS suppression fields. ``m0_type`` is ``Separate``
    (an M0 scan of its own), ``Included`` (``m0scan`` rows in the context), ``Estimate`` or
    ``Absent``.
    """
    p = _base(pe)
    p.update({
        "ArterialSpinLabelingType": label_type,
        "LabelingDuration": ld,
        "PostLabelingDelay": pld,
        "M0Type": m0_type,
        "TotalAcquiredPairs": n_pairs,
        "RepetitionTimePreparation": tr,
    })
    _finish(p, voxel_mm, n_slices, slice_spacing, te, background_suppression, extra)
    return p


def pasl(
    ti: float | list[float] = presets.REFERENCE.post_labeling_delay,
    cutoff: float = 0.7,
    technique: str = "Q2TIPS",
    n_pairs: int = presets.REFERENCE.n_pairs,
    tr: float = presets.REFERENCE.repetition_time,
    m0_type: str = "Separate",
    pe: str = "j-",
    voxel_mm: tuple[float, float, float] | None = None,
    n_slices: int | None = None,
    slice_spacing: float | None = None,
    te: float | None = None,
    background_suppression: list[float] | None = None,
    extra: dict[str, Any] | None = None,
) -> Protocol:
    """A PASL (FAIR) sidecar with a bolus cut-off, which fixes the bolus duration at ``cutoff``.

    BIDS records the PASL inversion time as ``PostLabelingDelay``, measured from the middle of
    the labeling pulse, which is also the kinetic model's clock.
    """
    p = _base(pe)
    p.update({
        "ArterialSpinLabelingType": "PASL",
        "PASLType": "FAIR",
        "PostLabelingDelay": ti,
        "BolusCutOffFlag": True,
        "BolusCutOffDelayTime": cutoff,
        "BolusCutOffTechnique": technique,
        "M0Type": m0_type,
        "TotalAcquiredPairs": n_pairs,
        "RepetitionTimePreparation": tr,
    })
    _finish(p, voxel_mm, n_slices, slice_spacing, te, background_suppression, extra)
    return p


def _finish(p, voxel_mm, n_slices, slice_spacing, te, background_suppression, extra):
    r = presets.REFERENCE
    if voxel_mm is not None:
        p["AcquisitionVoxelSize"] = list(voxel_mm)
    if n_slices is not None or slice_spacing is not None:
        p["SliceTiming"] = slice_timing(n_slices or r.n_slices, slice_spacing or r.slice_spacing)
    if te is not None:
        p["EchoTime"] = te
    if background_suppression:
        p["BackgroundSuppression"] = True
        p["BackgroundSuppressionNumberPulses"] = len(background_suppression)
        p["BackgroundSuppressionPulseTime"] = list(background_suppression)
    if extra:
        p.update(extra)


def multi_pld(
    plds: list[float],
    pairs_per_pld: int,
    label_type: str = "PCASL",
    **kw,
) -> Protocol:
    """A multi-delay (P)CASL sidecar: ``pairs_per_pld`` control-label pairs at each delay, in
    ascending order of delay, with ``PostLabelingDelay`` given per volume."""
    per_volume = [pld for pld in plds for _ in range(2 * pairs_per_pld)]
    return pcasl(pld=per_volume, n_pairs=pairs_per_pld * len(plds), label_type=label_type, **kw)


def aslcontext(p: Protocol, m0_rows: int = 1, order: str = "control-first") -> list[str]:
    """The ``volume_type`` column for a protocol: alternating control and label rows, with
    ``m0scan`` rows first when ``M0Type`` is ``Included``."""
    n = int(p["TotalAcquiredPairs"])
    pair = ["control", "label"] if order == "control-first" else ["label", "control"]
    rows = pair * n
    if p.get("M0Type") == "Included":
        rows = ["m0scan"] * m0_rows + rows
    return rows


def label_type(p: Protocol) -> str:
    return p["ArterialSpinLabelingType"]


def bolus_duration(p: Protocol) -> float:
    """``tau`` of the kinetic model: the labeling duration for (P)CASL, the cut-off for PASL."""
    if label_type(p) == "PASL":
        c = p["BolusCutOffDelayTime"]
        return float(c[0] if isinstance(c, list) else c)
    return float(p["LabelingDuration"])


def signal_times(p: Protocol, ctx: list[str]) -> list[float]:
    """The kinetic signal time ``t`` of every volume (seconds from the start of labeling,
    before the per-slice offset): PLD + tau for (P)CASL, the inversion time for PASL, zero for
    ``m0scan`` rows. A scalar ``PostLabelingDelay`` applies to every non-M0 row; a list gives
    one entry per non-M0 row, in order."""
    pld = p["PostLabelingDelay"]
    n_asl = sum(1 for r in ctx if r != "m0scan")
    plds = list(pld) if isinstance(pld, list) else [float(pld)] * n_asl
    if len(plds) != n_asl:
        raise ValueError(f"{len(plds)} PostLabelingDelay values for {n_asl} control/label rows")
    tau = 0.0 if label_type(p) == "PASL" else bolus_duration(p)
    out, i = [], 0
    for r in ctx:
        if r == "m0scan":
            out.append(0.0)
        else:
            out.append(plds[i] + tau)
            i += 1
    return out


def slice_offsets(p: Protocol) -> list[float]:
    """Per-slice readout offsets: ``SliceTiming`` minus its minimum, in acquired order."""
    st = p["SliceTiming"]
    m = min(st)
    return [t - m for t in st]


def readout_end(p: Protocol, ctx: list[str]) -> float:
    """The latest slice readout time of any row, which aslscan requires to fit within TR."""
    return max(signal_times(p, ctx)) + max(slice_offsets(p))


def describe(p: Protocol) -> str:
    """One line for figure titles and tables."""
    lt = label_type(p)
    pld = p["PostLabelingDelay"]
    pld_s = f"{min(pld):g}-{max(pld):g} s ({len(set(pld))} delays)" if isinstance(pld, list) else f"{pld:g} s"
    if lt == "PASL":
        return f"PASL, TI {pld_s}, bolus cut-off {bolus_duration(p):g} s, TR {p['RepetitionTimePreparation']:g} s"
    return f"{lt}, LD {p['LabelingDuration']:g} s, PLD {pld_s}, TR {p['RepetitionTimePreparation']:g} s"
