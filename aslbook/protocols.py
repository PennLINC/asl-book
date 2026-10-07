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


# ----------------------------------------------------------------------------- 3D readouts
#: The book's 3D readout timing (Chapter 2). The EPI line spacing of the 2D reference is kept
#: (0.3 ms), so a GRASE echo reads ``ny / ky_segments`` lines in one block centred on its spin
#: echo; the echo spacing follows from the sidecar's ``EchoTime`` (the k-space-centre time of a
#: centric train, so TE = ESP for the first echo). The spiral's interleaves, readout time and
#: dwell time are not BIDS fields and go in the overlay's ``[readout]`` table.
GRASE_LINE_SPACING_MS = 0.3
SPIRAL_READOUT_MS = 5.0
SPIRAL_DWELL_S = 4e-6
SPIRAL_INTERLEAVES = 8


def readout_3d(p: Protocol, kind: str, n_shots: int, refocusing_deg: float = 180.0, te: float = 0.010) -> Protocol:
    """Turn a 2D sidecar into a 3D one: ``MRAcquisitionType`` 3D with ``PulseSequenceType``
    naming the readout (``grase`` or ``spiral``), the slab's ``NumberShots``, ``FlipAngle`` read
    by aslscan as the echo train's refocusing angle (the excitation is 90 degrees), and
    ``EchoTime`` as the time to the k-space centre. ``SliceTiming`` and ``TotalReadoutTime``
    are 2D fields and are dropped (BIDS says the first must not be defined for 3D; the second
    is written by the simulator as it resolves the train); a spiral has no phase-encode axis,
    so ``PhaseEncodingDirection`` goes too."""
    kind = kind.lower()
    if kind not in ("grase", "spiral"):
        raise ValueError(f"3D readout {kind!r}: grase or spiral")
    p = dict(p)
    for k in ("SliceTiming", "TotalReadoutTime"):
        p.pop(k, None)
    if kind == "spiral":
        p.pop("PhaseEncodingDirection", None)
    p["MRAcquisitionType"] = "3D"
    p["PulseSequenceType"] = "3Dgrase" if kind == "grase" else "spiral"
    p["NumberShots"] = int(n_shots)
    p["FlipAngle"] = refocusing_deg
    p["EchoTime"] = te
    return p


def _labeling(label: str, kw: dict) -> Protocol:
    """The 2D sidecar of a labeling builder (``pcasl``, ``pasl``, ``multi_pld``) by name."""
    builders = {"pcasl": pcasl, "pasl": pasl, "multi_pld": multi_pld}
    return builders[label](**kw)


def grase(n_shots: int = 4, refocusing_deg: float = 180.0, te: float = 0.010, label: str = "pcasl", **kw) -> Protocol:
    """A 3D GRASE sidecar on the book's labeling (``label`` names the builder, ``pcasl``,
    ``pasl`` or ``multi_pld``; the remaining keywords are its arguments). The in-plane
    segmentation (``ky_segments``, which must multiply with ``kz_segments`` to ``n_shots``) and
    the actual line spacing are overlay ``[readout]`` keys."""
    return readout_3d(_labeling(label, kw), "grase", n_shots, refocusing_deg, te)


def spiral(interleaves: int = SPIRAL_INTERLEAVES, refocusing_deg: float = 180.0, te: float = 0.012, label: str = "pcasl", **kw) -> Protocol:
    """A 3D stack-of-spirals sidecar: one interleaf per shot (``NumberShots`` = interleaves with
    one kz segment). The trajectory itself (``interleaves``, ``spiral_readout_time``,
    ``dwell_time``) is described in the overlay's ``[readout]`` table, since BIDS has no field
    for it; a spiral needs a square in-plane matrix (overlay ``acquisition.matrix``)."""
    return readout_3d(_labeling(label, kw), "spiral", interleaves, refocusing_deg, te)


# ----------------------------------------------------------------------------- encodings
def hadamard(
    order: int = 8,
    sub_bolus: float = 0.4,
    pld: float = 0.3,
    n_cycles: int = 7,
    tr: float = presets.REFERENCE.repetition_time,
    m0_type: str = "Separate",
    readout: str | None = None,
    n_shots: int = 4,
    refocusing_deg: float = 180.0,
    te: float | None = None,
    **kw,
) -> Protocol:
    """A Hadamard time-encoded PCASL sidecar, written as BIDS and aslscan read it: the arrays
    describe the *decoded* volumes. Each encoding cycle has ``order - 1`` sub-boli of
    ``sub_bolus`` seconds, labeled back to back, and one readout ``pld`` seconds after the last;
    decoded row ``j`` (1-based, in labeling order) carries ``LabelingDuration = sub_bolus`` and
    its own effective ``PostLabelingDelay``, from the end of sub-bolus ``j`` to the excitation:
    ``pld + sub_bolus * (order - 1 - j)``. ``TotalAcquiredPairs`` is the number of cycles (the
    simulator's convention, stated in its sidecar). With ``readout`` ``grase`` or ``spiral``
    the readout is 3D (``n_shots`` shots per raw volume); otherwise the book's 2D EPI. The
    overlay's ``[hadamard] order`` must match."""
    n_sub = order - 1
    plds = [round(pld + sub_bolus * (n_sub - j), 6) for j in range(1, n_sub + 1)] * n_cycles
    lds = [sub_bolus] * (n_sub * n_cycles)
    if m0_type == "Included":
        plds, lds = [0.0] + plds, [0.0] + lds
    p = pcasl(pld=plds, ld=lds, n_pairs=n_cycles, tr=tr, m0_type=m0_type, **kw)
    if readout is not None:
        p = readout_3d(p, readout, n_shots, refocusing_deg, te or (0.010 if readout == "grase" else 0.012))
    return p


def look_locker(
    tis: list[float] | None = None,
    flip_deg: float = 35.0,
    n_pairs: int = 8,
    tr: float = 5.0,
    cutoff: float = 0.7,
    technique: str = "Q2TIPS",
    m0_type: str = "Separate",
    slice_spacing: float = 0.024,
    **kw,
) -> Protocol:
    """A Look-Locker PASL sidecar: after each labeling (FAIR with a bolus cut-off, which also
    bounds the first readout, since aslscan refuses a PASL readout before the cut-off), the slab
    is read ``len(tis)`` times with a low-flip gradient-echo excitation of ``flip_deg`` degrees,
    at the inversion times ``tis``. Each cycle of readouts is one row run in the arrays: a
    control cycle, then a label cycle, ``n_pairs`` times. The slices follow each other
    ``slice_spacing`` apart, so a whole readout must fit between consecutive inversion times.
    The overlay must set ``[signal] acq_contrast = "ge"``."""
    tis = tis or [round(0.8 + 0.5 * i, 3) for i in range(8)]
    per_volume = list(tis) * (2 * n_pairs)
    p = pasl(ti=per_volume, cutoff=cutoff, technique=technique, n_pairs=n_pairs, tr=tr, m0_type=m0_type,
             slice_spacing=slice_spacing, **kw)
    p["LookLocker"] = True
    p["FlipAngle"] = flip_deg
    return p


def multi_te(echo_times: list[float], contrast: str = "ge", flip_deg: float = 60.0, **kw) -> list[Protocol]:
    """One sidecar per echo for a multi-echo series (aslscan takes them as repeated
    ``--asl-json``, writes the BIDS ``echo-N`` layout, and checks that every other field
    agrees). ``contrast`` is the overlay's ``acq_contrast`` (``ge`` with the excitation
    ``flip_deg``, written as the sidecar's ``FlipAngle``; or ``se``, whose refocusing pulses
    must fit between the echoes' EPI blocks). The remaining keywords build the base PCASL
    sidecar; its ``slice_spacing`` must hold every echo's block."""
    base = pcasl(te=echo_times[0], **kw)
    if contrast == "ge":
        base["FlipAngle"] = flip_deg
    return [dict(base, EchoTime=te) for te in echo_times]


def aslcontext(p: Protocol, m0_rows: int = 1, order: str = "control-first") -> list[str]:
    """The ``volume_type`` column for a protocol: alternating control and label rows, with
    ``m0scan`` rows first when ``M0Type`` is ``Included``. A time-encoded sidecar (one
    ``LabelingDuration`` per volume) lists the decoded ``deltam`` volumes, cycle by cycle; a
    Look-Locker sidecar lists each cycle's readouts as a run of one volume type."""
    n = int(p["TotalAcquiredPairs"])
    if isinstance(p.get("LabelingDuration"), list):
        rows = ["deltam"] * sum(1 for ld in p["LabelingDuration"] if ld > 0)
        if p.get("M0Type") == "Included":
            rows = ["m0scan"] * sum(1 for ld in p["LabelingDuration"] if ld == 0) + rows
        return rows
    if p.get("LookLocker"):
        m = readouts_per_cycle(p)
        pair = ["control"] * m + ["label"] * m
        if order != "control-first":
            pair = pair[m:] + pair[:m]
        rows = pair * n
    else:
        pair = ["control", "label"] if order == "control-first" else ["label", "control"]
        rows = pair * n
    if p.get("M0Type") == "Included":
        rows = ["m0scan"] * m0_rows + rows
    return rows


def readouts_per_cycle(p: Protocol) -> int:
    """A Look-Locker series' readouts per labeling: the length of the first strictly increasing
    run of ``PostLabelingDelay``."""
    pld = p["PostLabelingDelay"]
    if not isinstance(pld, list):
        return 1
    m = 1
    while m < len(pld) and pld[m] > pld[m - 1]:
        m += 1
    return m


def is_3d(p: Protocol) -> bool:
    return str(p.get("MRAcquisitionType", "2D")).upper() == "3D"


def is_time_encoded(p: Protocol) -> bool:
    return isinstance(p.get("LabelingDuration"), list)


def label_type(p: Protocol) -> str:
    return p["ArterialSpinLabelingType"]


def bolus_duration(p: Protocol) -> float:
    """``tau`` of the kinetic model: the labeling duration for (P)CASL, the cut-off for PASL;
    for a time-encoded sidecar the whole labeling of one cycle (the sum of its sub-boli)."""
    if label_type(p) == "PASL":
        c = p["BolusCutOffDelayTime"]
        return float(c[0] if isinstance(c, list) else c)
    ld = p["LabelingDuration"]
    if isinstance(ld, list):
        return float(sum(sub_boli(p)))
    return float(ld)


def sub_boli(p: Protocol) -> list[float]:
    """The sub-bolus durations of one encoding cycle of a time-encoded sidecar, in labeling
    order (the first cycle's non-zero ``LabelingDuration`` entries, up to the first repeat of
    the cycle)."""
    lds = [ld for ld in p["LabelingDuration"] if ld > 0]
    plds = [pld for ld, pld in zip(p["LabelingDuration"], p["PostLabelingDelay"]) if ld > 0]
    n = 1
    while n < len(plds) and plds[n] < plds[n - 1]:
        n += 1
    return [float(x) for x in lds[:n]]


def signal_times(p: Protocol, ctx: list[str]) -> list[float]:
    """The kinetic signal time ``t`` of every volume (seconds from the start of labeling,
    before the per-slice offset): PLD + tau for (P)CASL, the inversion time for PASL, zero for
    ``m0scan`` rows. A scalar ``PostLabelingDelay`` applies to every non-M0 row; a list gives
    one entry per non-M0 row, in order. For a time-encoded sidecar every decoded row of a cycle
    shares the cycle's excitation: the last sub-bolus's delay plus the whole labeling."""
    pld = p["PostLabelingDelay"]
    n_asl = sum(1 for r in ctx if r != "m0scan")
    if is_time_encoded(p):
        pairs = [(ld, d) for ld, d in zip(p["LabelingDuration"], pld) if ld > 0]
        plds = [d for _, d in pairs]
    else:
        plds = list(pld) if isinstance(pld, list) else [float(pld)] * n_asl
    if len(plds) != n_asl:
        raise ValueError(f"{len(plds)} PostLabelingDelay values for {n_asl} control/label rows")
    tau = 0.0 if label_type(p) == "PASL" else bolus_duration(p)
    if is_time_encoded(p):
        n_sub = len(sub_boli(p))
        t_exc = [min(plds[c * n_sub:(c + 1) * n_sub]) + tau for c in range(len(plds) // n_sub)]
        plds = [t_exc[i // n_sub] - tau for i in range(len(plds))]
    out, i = [], 0
    for r in ctx:
        if r == "m0scan":
            out.append(0.0)
        else:
            out.append(plds[i] + tau)
            i += 1
    return out


def slice_offsets(p: Protocol) -> list[float]:
    """Per-slice readout offsets: ``SliceTiming`` minus its minimum, in acquired order. A 3D
    readout excites the whole slab at once, so every partition's offset is zero."""
    if is_3d(p) or "SliceTiming" not in p:
        return [0.0]
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
    ro = f", 3D {p['PulseSequenceType']} x {p['NumberShots']} shots" if is_3d(p) else ""
    if lt == "PASL":
        ll = f", Look-Locker x {readouts_per_cycle(p)} at {p['FlipAngle']:g} deg" if p.get("LookLocker") else ""
        return f"PASL, TI {pld_s}, bolus cut-off {bolus_duration(p):g} s, TR {p['RepetitionTimePreparation']:g} s{ll}{ro}"
    if is_time_encoded(p):
        sb = sub_boli(p)
        return f"{lt} time-encoded, {len(sb)} sub-boli of {sb[0]:g} s, PLD {min(pld):g} s, TR {p['RepetitionTimePreparation']:g} s{ro}"
    return f"{lt}, LD {p['LabelingDuration']:g} s, PLD {pld_s}, TR {p['RepetitionTimePreparation']:g} s{ro}"
