---
title: "17. What your data allow"
kernelspec:
  name: python3
  display_name: Python 3
---

:::{admonition} Simulated datasets in this chapter
:class: note
- **Built in this page:** a small rules engine that reads an ASL sidecar and says which quantities the acquisition supports, applied to the book's protocols and to five public example sidecars ([Appendix B](../appendices/b-data-manifest.md#app-b-package-data)).
- **`ref-pcasl`**: the reference series, quantified with and without its M0 scan ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-ref-pcasl)).
- **`bgsup`**: the `on` run, whose control images hold no static signal ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-bgsup)).
- **`multi-pld`**: the six-delay series with its 2D readout, fitted with and without slice timing ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-multi-pld)).
- **`pld-sweep`**: single-delay runs at 0.5 to 3.0 s, for the delay-versus-transit-time question ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-pld-sweep)).

Pipeline-tier datasets are simulated offline by aslscan ([Chapter 0.2](../00-frontmatter/the-simulated-datasets.md)).
:::

## Learning goals

After this chapter you can:

- read an ASL sidecar and say which analyses it supports, which are marginal and why, and
  which are not possible
- work through a dataset you did not design: calibrate without an M0 scan, fit transit
  times from a 2D multi-delay series, and judge a delay against a population's transit times
- say what complex data add to ASL, and which sidecar fields a quantification needs

```{code-cell} python
:tags: [hide-cell]
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch, Rectangle

from aslbook import data, kinetic, phantom, presets, protocols, quant
from aslbook.plotting import INK, PALETTE, TISSUE_COLORS, set_style, show_slice, take_slice

set_style()
K = phantom.DISPLAY_SLICE
GM, WM = presets.TISSUES["GM"], presets.TISSUES["WM"]
R = presets.REFERENCE
TAU, PLD, TE = R.labeling_duration, R.post_labeling_delay, R.echo_time


def summarize(run):
    """Mean difference and control images, sidecar, per-slice PLDs, offsets and pure-tissue masks."""
    mag, ctx, p = run.mag(), run.context(), run.sidecar()
    dm = quant.subtract(mag, ctx).mean(-1)
    ctrl = mag[..., [i for i, r in enumerate(ctx) if r == "control"]].mean(-1)
    fr = run.fractions()
    masks = {t: fr[t.lower()] >= 0.9 for t in ("GM", "WM", "CSF")}
    pld = p["PostLabelingDelay"]
    offsets = np.asarray(protocols.slice_offsets(p))
    plds = None if isinstance(pld, list) else quant.slice_plds(pld, offsets)[None, None, :]
    return dict(mag=mag, ctx=ctx, p=p, dm=dm, ctrl=ctrl, fr=fr, masks=masks, plds=plds, offsets=offsets)
```

## The decision table as a function

Every verdict in Parts II through IV reduces to a few fields of the sidecar: whether
`PostLabelingDelay` is one number or a list, whether `BackgroundSuppression` is on, whether
`MRAcquisitionType` is 2D or 3D, what `M0Type` says, how many pairs were acquired, and how
the delay compares with the transit times of the population. The function below is the
book's decision table written as code. It takes a sidecar dictionary and the range of
arterial transit times expected in the population (the shortest matters for sampling the
inflow, the longest for a single delay) and returns a verdict and a reason for each of
seven analyses. It is deliberately simple, and the code is shown so that the rules can be
argued with.

```{code-cell} python
def features(p, n_pairs=None):
    """The acquisition features the rules read, from a BIDS ASL sidecar dictionary."""
    pld = p["PostLabelingDelay"]
    plds = sorted(set(pld)) if isinstance(pld, list) else [float(pld)]
    pasl = p["ArterialSpinLabelingType"] == "PASL"
    offsets = protocols.slice_offsets(p) if p.get("MRAcquisitionType", "2D") == "2D" and "SliceTiming" in p else [0.0]
    return dict(
        plds=plds, multi=len(plds) > 1, pasl=pasl, bs=bool(p.get("BackgroundSuppression")),
        readout=p.get("MRAcquisitionType", "2D"), m0=p.get("M0Type", "Absent"),
        n_pairs=int(n_pairs or p.get("TotalAcquiredPairs", 0)), offsets=offsets,
        bolus=(p.get("BolusCutOffDelayTime") if pasl and p.get("BolusCutOffFlag") else p.get("LabelingDuration")),
        alpha_recorded="LabelingEfficiency" in p, voxel=max(p.get("AcquisitionVoxelSize", [3, 3])[:2]),
    )


def verdicts(p, att_range=(0.8, 1.5), n_pairs=None):
    """(analysis, verdict, reason) for seven analyses; verdicts are 'yes', 'marginal' or 'no'."""
    f = features(p, n_pairs)
    att_lo, att_hi = att_range
    bolus = f["bolus"] if isinstance(f["bolus"], (int, float)) else None
    out = []

    def worst(*items):  # combine (verdict, reason) pairs: the worst verdict wins, with its reason
        rank = {"yes": 0, "marginal": 1, "no": 2}
        return max(items, key=lambda v: rank[v[0]])

    # arrival: (P)CASL needs PLD >= ATT; PASL needs TI >= ATT + bolus (the bolus tail must have arrived)
    arrival_ok = (f["plds"][-1] >= att_hi + (bolus or 0)) if f["pasl"] else (f["plds"][-1] >= att_hi)
    arrival = ("yes", "") if arrival_ok or f["multi"] else ("marginal", f"longest delay {f['plds'][-1]:g} s is below the expected ATT of {att_hi:g} s{' + bolus' if f['pasl'] else ''}: underestimates where the bolus has not arrived")
    pairs = ("yes", "") if f["n_pairs"] >= 10 else ("marginal", f"only {f['n_pairs']} pairs: noisy maps")
    out.append(("relative CBF map",) + worst(pairs, arrival))

    if f["m0"] in ("Separate", "Included"):
        calib = ("yes", "")
    elif f["m0"] == "Estimate":
        calib = ("marginal", "M0Estimate: one number for the whole brain, no per-voxel calibration")
    elif not f["bs"]:
        calib = ("marginal", "no M0 image: calibrate on the mean control image, corrected for its TR and TE")
    else:
        calib = ("no", "no M0 image and background suppression: no static signal to calibrate against")
    if bolus is None:
        calib = worst(calib, ("no", "PASL without a bolus cut-off: the bolus duration is unknown"))
    timing = ("yes", "") if f["readout"] == "3D" or "SliceTiming" in p else ("marginal", "2D readout without SliceTiming: per-slice delays unknown")
    absolute = worst(calib, arrival, timing)
    out.append(("absolute CBF",) + absolute)

    if not f["multi"]:
        att = ("no", "one delay cannot separate transit time from CBF")
    else:
        t0 = 0.0 if f["pasl"] else (bolus or 0.0)  # signal time = delay + bolus for (P)CASL, = TI for PASL
        inflow_end = att_lo + (bolus or 0.0)
        n_inflow = sum(1 for o in f["offsets"] if f["plds"][0] + t0 + o < inflow_end)
        if n_inflow == len(f["offsets"]):
            att = ("yes", "")
        elif n_inflow == 0:
            att = ("no", f"no delay samples the inflow: shortest delay {f['plds'][0]:g} s reaches every slice after ATT {att_lo:g} s + bolus")
        else:
            att = ("marginal", f"2D readout: only {n_inflow} of {len(f['offsets'])} slices sample the inflow of the fastest tissue (ATT {att_lo:g} s)")
    out.append(("arterial transit time",) + att)

    pv = ("yes", "") if f["voxel"] <= 4.0 else ("marginal", f"{f['voxel']:g} mm in-plane voxels: few pure-tissue voxels to anchor the regression")
    out.append(("PV-corrected GM CBF",) + worst(absolute if absolute[0] != "no" else ("marginal", "relative maps only: " + absolute[1]), pv))

    region = worst(pairs, ("yes", "") if absolute[0] != "no" else ("marginal", "relative (normalized) CBF only"),
                   ("yes", "") if f["multi"] or arrival_ok else ("marginal", "single delay near the expected ATT: transit-time differences between groups look like CBF differences"))
    out.append(("region-level group comparison",) + region)

    if f["n_pairs"] < 10:
        vox = ("no", f"{f['n_pairs']} pairs: voxel noise exceeds any plausible effect")
    elif f["n_pairs"] < 30:
        vox = ("marginal", f"{f['n_pairs']} pairs: smooth or pool voxels")
    else:
        vox = ("yes", "")
    vox = worst(vox, region, ("yes", "") if f["bs"] else ("marginal", "no background suppression: motion and physiological noise scale with the static signal"))
    out.append(("voxelwise group comparison",) + vox)

    longi = worst(("yes", "") if f["m0"] in ("Separate", "Included") else ("marginal", "calibration from the control image drifts with scanner gain and tissue T1 between sessions"),
                  region, ("yes", "") if absolute[0] != "no" else absolute)
    out.append(("longitudinal change",) + longi)
    return out
```

The columns below are the book's three protocols, evaluated against the phantom's transit
times (0.8 to 1.2 s), and five real acquisitions from the ASL-BIDS example datasets
{cite:p}`clement2022`, evaluated against a healthy adult range of 0.8 to 1.5 s. The real
sidecars are reconstructed from their published descriptions: only the fields the rules
read are filled in, and the 2D readouts are given the book's slice timing where the
example does not state one.

```{code-cell} python
:tags: [hide-input]
def sidecar(label_type, pld, ld=None, readout="3D", m0="Separate", n_pairs=30, bs=True, cutoff=None, alpha=None, slices=None, voxel=(3.5, 3.5, 5.0)):
    p = {"ArterialSpinLabelingType": label_type, "PostLabelingDelay": pld, "MRAcquisitionType": readout, "M0Type": m0,
         "TotalAcquiredPairs": n_pairs, "BackgroundSuppression": bs, "AcquisitionVoxelSize": list(voxel)}
    if ld is not None:
        p["LabelingDuration"] = ld
    if cutoff is not None:
        p.update({"BolusCutOffFlag": True, "BolusCutOffDelayTime": cutoff, "BolusCutOffTechnique": "Q2TIPS"})
    if alpha is not None:
        p["LabelingEfficiency"] = alpha
    if readout == "2D":
        p["SliceTiming"] = R.slice_timing if slices is None else slices
    return p


PHANTOM_ATT, ADULT_ATT = (0.8, 1.2), (0.8, 1.5)
COLUMNS = [
    ("Reference\n2D PCASL\nPLD 1.8 s\nsep. M0, 30 pairs", protocols.pcasl(), PHANTOM_ATT),
    ("Multi-PLD\n2D PCASL\nPLD 0.5–3.0 s\nsep. M0, 6 × 5 pairs", protocols.multi_pld([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], 5, tr=6.0), PHANTOM_ATT),
    ("PASL\n2D, TI 1.8 s\nbolus 0.7 s\nsep. M0, 30 pairs", protocols.pasl(ti=1.8, cutoff=0.7), PHANTOM_ATT),
    ("asl001 (GE)\n3D spiral PCASL\nPLD 2.025 s, BS\nincl. M0, 3 pairs", sidecar("PCASL", 2.025, ld=1.45, m0="Included", n_pairs=3), ADULT_ATT),
    ("asl002 (Philips)\n2D EPI PCASL\nPLD 2.0 s, BS\nsep. M0, 35 pairs", sidecar("PCASL", 2.0, ld=1.8, readout="2D", n_pairs=35), ADULT_ATT),
    ("asl003 (Siemens)\n3D GRASE PASL\nTI 0.3–3.0 s, BS\nsep. M0, 10 pairs", sidecar("PASL", list(np.round(np.linspace(0.3, 3.0, 10), 2)), cutoff=0.7, n_pairs=10), ADULT_ATT),
    ("asl004 (Siemens)\n2D EPI PCASL\nPLD 0.25–1.5 s, BS\nsep. M0, 48 pairs, α 0.88", sidecar("PCASL", [0.25, 0.5, 0.75, 1.0, 1.25, 1.5], ld=1.5, readout="2D", n_pairs=48, alpha=0.88), ADULT_ATT),
    ("asl005 (Siemens)\n3D GRASE PCASL\nPLD 2.0 s, BS\nsep. M0, 8 pairs", sidecar("PCASL", 2.0, ld=1.8, n_pairs=8), ADULT_ATT),
]
N_BOOK = 3
matrix = [verdicts(p, att_range=a) for _, p, a in COLUMNS]
analyses = [row[0] for row in matrix[0]]
ROW_NOTES = {
    "relative CBF map": "≥ 10 pairs; delay reaches the population's ATT · Ch 6, 14",
    "absolute CBF": "M0 image or unsuppressed control; known bolus; slice timing · Ch 14, 16",
    "arterial transit time": "several delays, some on the inflow in every slice · Ch 15",
    "PV-corrected GM CBF": "absolute CBF; voxels with pure-tissue anchors · Ch 12",
    "region-level group comparison": "calibrated maps; delay above both groups' ATT · Ch 14, 16",
    "voxelwise group comparison": "≥ 30 pairs and background suppression · Ch 8, 9",
    "longitudinal change": "same M0 route every session; delay above the ATT · Ch 16",
}


def tint(color, amount):
    return tuple(1 - amount * (1 - v) for v in to_rgb(color))


STYLE = {"yes": (PALETTE[5], "white"), "marginal": (PALETTE[3], INK["primary"]), "no": (tint(PALETTE[7], 0.22), INK["secondary"])}
notes = {}
for j, rows in enumerate(matrix):
    for analysis, verdict, reason in rows:
        if verdict != "yes":
            notes.setdefault((analysis, reason), []).append(COLUMNS[j][0].split("\n")[0].split(" (")[0])
note_id = {key: k + 1 for k, key in enumerate(notes)}
SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")

label_w, cell_w, cell_h, gap, head_h, foot_h = 3.4, 1.35, 0.42, 0.25, 1.05, 0.42
n_rows, n_cols = len(analyses), len(COLUMNS)
W, H = label_w + n_cols * cell_w + gap + 0.05, head_h + n_rows * cell_h + foot_h
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W), ax.set_ylim(H, 0), ax.set_axis_off()
col_x = lambda j: label_w + j * cell_w + (gap if j >= N_BOOK else 0)
for i, analysis in enumerate(analyses):
    y = head_h + i * cell_h
    ax.text(label_w - 0.08, y + 0.15, analysis, ha="right", va="center", fontsize=8.5, color=INK["primary"])
    ax.text(label_w - 0.08, y + 0.31, ROW_NOTES[analysis], ha="right", va="center", fontsize=6.4, color=INK["secondary"])
    for j, rows in enumerate(matrix):
        _, verdict, reason = rows[i]
        face, ink = STYLE[verdict]
        x = col_x(j)
        ax.add_patch(Rectangle((x + 0.02, y + 0.02), cell_w - 0.04, cell_h - 0.04, facecolor=face, edgecolor="none"))
        text = verdict + (str(note_id[(analysis, reason)]).translate(SUP) if verdict != "yes" else "")
        ax.text(x + cell_w / 2, y + cell_h / 2, text, ha="center", va="center", fontsize=7.5, color=ink, fontweight="bold" if verdict == "yes" else "normal")
for j, (header, _, _) in enumerate(COLUMNS):
    ax.text(col_x(j) + cell_w / 2, head_h - 0.06, header, ha="center", va="bottom", fontsize=6.6, color=INK["primary"], linespacing=1.25)
for lo, hi, title in [(0, N_BOOK, "This book's protocols (phantom ATT 0.8–1.2 s)"), (N_BOOK, n_cols, "ASL-BIDS example datasets (adult ATT 0.8–1.5 s)")]:
    x0, x1 = col_x(lo) + 0.04, col_x(hi - 1) + cell_w - 0.04
    ax.plot([x0, x1], [0.24, 0.24], color=INK["secondary"], lw=0.8)
    ax.text((x0 + x1) / 2, 0.2, title, ha="center", va="bottom", fontsize=7.5, color=INK["secondary"])
ax.legend(handles=[Patch(facecolor=STYLE[v][0], label=v) for v in ("yes", "marginal", "no")], loc="lower right",
          bbox_to_anchor=(W - 0.05, H - 0.02), bbox_transform=ax.transData, ncol=3, fontsize=8, handlelength=1.4, frameon=False)
plt.show()

for (analysis, reason), cols in notes.items():
    print(f"{str(note_id[(analysis, reason)]).translate(SUP)} {analysis} — {', '.join(cols)}: {reason}")
```

Read a row across to see which acquisitions support that analysis, or a column down to
see what one dataset supports. Green is "yes", amber "marginal", pale red "no", and the
numbered notes under the figure give the reason for every cell that is not green. The
transit-time row separates the columns most sharply: only the three multi-delay columns
are not red, and of those only the 3D PASL series (asl003) is green, because a 3D readout
samples every voxel at the same time and its shortest inversion time, 0.3 s, is well
before any bolus has arrived. The two 2D multi-delay columns are marginal for the same
reason, which case 2 below measures: the later slices of a 2D readout are sampled up to
0.76 s after the first, and the shortest delay no longer catches the inflow there.

The PASL column is marginal for absolute CBF at the phantom's transit times even though
its inversion time of 1.8 s exceeds the longest ATT: for pulsed labeling the condition is
that the *end* of the bolus has arrived, TI ≥ ATT + bolus, and 1.2 + 0.7 = 1.9 s is
longer than 1.8 s. The three-pair GE series (asl001) is the product default of a
clinical scanner: absolute CBF is possible from its included M0 volumes, but every row
that depends on the noise level is marginal or red. The eight-pair 3D GRASE series
(asl005) is the same case with better maps: marginal for regions, red for voxels. The
reference protocol itself is marginal for voxelwise comparison because it has no
background suppression: its 30 pairs are enough, but the static signal is 200 times the
difference ([Chapter 9](../03-preprocessing/09-background-suppression.md)).

The verdicts are rules of thumb. A "yes" means the quantity is determined and the
acquisition is inside the range where the method was developed; "marginal" means it can
be computed but a stated assumption is strained, and the chapter for that method shows
what the strain costs; "no" means the quantity is not identifiable from these data. The
same verdicts as text, with the reason for every cell, are in the collapsed output below,
and the function runs on any sidecar: `verdicts(json.load(open("sub-01_asl.json")))`.

```{code-cell} python
:tags: [hide-input, hide-output]
for header, p, att_range in COLUMNS:
    print(f"\n{header.replace(chr(10), ': ')}  (ATT {att_range[0]:g}–{att_range[1]:g} s)")
    for analysis, verdict, reason in verdicts(p, att_range=att_range):
        print(f"  {analysis:<32} {verdict:<9} {reason}")
```

## Worked case 1: a single-delay PCASL series without an M0 scan

The commonest inherited dataset: single-delay PCASL, one series, no `m0scan` file, and
the sidecar says `M0Type: Absent` (or says nothing). If the series was acquired without
background suppression, the control images are a calibration image at the series'
repetition time. [Chapter 16](./16-calibration.md) showed that the mean control, corrected
for the saturation at TR 4.5 s and for the blood-tissue $T_2$ difference at TE 12 ms,
gives the same gray-matter CBF as the separate M0 scan. Here the whole map is compared,
voxel by voxel, and the run with background suppression shows what is left to calibrate
against when the series was suppressed.

```{code-cell} python
:tags: [hide-input]
ref = data.load_dataset("ref-pcasl").run()
s = summarize(ref)
m0_scan = quant.m0_correction(ref.m0scan(), tr=ref.m0scan_sidecar()["RepetitionTimePreparation"])
m0_ctrl = quant.m0_correction(s["ctrl"], tr=s["p"]["RepetitionTimePreparation"])
cbf_scan = quant.cbf_pcasl(s["dm"], m0_scan, s["plds"])
cbf_ctrl = quant.cbf_pcasl(s["dm"], m0_ctrl, s["plds"])
mask = ref.mask()
truth = ref.truth("perfusion")

fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
show_slice(axes[0], np.where(mask, cbf_scan, np.nan), K, title="CBF from the separate M0 scan", kind="cbf")
show_slice(axes[1], np.where(mask, cbf_ctrl, np.nan), K, title="CBF from the mean control image", kind="cbf")
im = show_slice(axes[2], np.where(mask, cbf_ctrl - cbf_scan, np.nan), K, title="control route − M0-scan route", kind="diff", vmin=-5, vmax=5)
fig.colorbar(im, ax=axes[2], shrink=0.8, label="ml/100 g/min")
fig.tight_layout()

sc = quant.score(cbf_ctrl, cbf_scan, mask)
gm, wm = s["masks"]["GM"], s["masks"]["WM"]
print(f"GM CBF: M0 scan {cbf_scan[gm].mean():.1f}, control mean {cbf_ctrl[gm].mean():.1f};  WM: {cbf_scan[wm].mean():.1f} vs {cbf_ctrl[wm].mean():.1f}")
print(f"control route vs M0-scan route over the brain: bias {sc['bias']:+.2f}, RMSE {sc['rmse']:.2f} ml/100 g/min, r = {sc['r']:.3f}")
print(f"both vs the truth in pure GM ({truth[gm].mean():.1f}): white-paper formula {cbf_scan[gm].mean():.1f}; see Chapter 14 for the formula's T1' assumption")
bs_on = data.load_dataset("bgsup").run("on")
sb = summarize(bs_on)
print(f"with background suppression on, the mean control image in GM is {sb['ctrl'][sb['masks']['GM']].mean():.0f} image units against {s['ctrl'][gm].mean():.0f} without")
```

The two maps are the same map. Over the whole brain the control-mean route differs from
the M0-scan route by a bias of −0.02 and an RMSE of 0.73 ml/100 g/min, with a correlation
of 0.999; in gray matter both give 43.8. Most of the difference image is noise: the
control mean averages 30 volumes and the M0 scan is one, so the control route is in fact
the less noisy calibration here, and the pipelines that use it when no M0 scan exists
(ExploreASL {cite:p}`mutsaerts2020`, ASLPrep {cite:p}`adebimpe2022`) are not making a
compromise on precision. The one structure in it is the ring around the ventricles. The
saturation correction uses the gray-matter $T_1$, and at TR 4.5 s CSF, with its 3 s $T_1$,
is far more saturated (0.78) than gray matter (0.97), so the control route under-corrects
voxels with CSF and their CBF comes out a few units higher; the TR 8 s scan is nearly
immune to the same assumption. That is the compromise of the route: the two constants
the correction needs, a tissue $T_1$ that is one number for a brain of several, and the
blood $T_2$. The series with background suppression has no such fallback: its mean
control image in gray matter is 791 image units against 6246 without, the residual of
two inversion pulses at 95 percent efficiency, and it calibrates nothing. Decision:
quantify with the control mean, record `M0Type: Absent` and the route in the
derivatives, and treat the absolute scale as uncertain by the few percent those
constants carry.

## Worked case 2: a multi-delay series with a 2D readout

The `multi-pld` dataset acquires six delays from 0.5 to 3.0 s with the book's 20-slice 2D
readout, in which slice $z$ is excited $0.04\,z$ s after the first. Every slice therefore
samples the kinetic curve at its own six times, $t_z = \mathrm{PLD} + \tau + 0.04\,z$,
and the first question is which of those times fall on the inflow, between ATT and
ATT + $\tau$, where the curve carries the transit-time information.

```{code-cell} python
:tags: [hide-input]
mp = data.load_dataset("multi-pld").run()
sm = summarize(mp)
d_all = quant.subtract(sm["mag"], sm["ctx"])
plds_all = quant.pld_of_pairs(sm["p"], sm["ctx"])
delays = np.unique(plds_all)
d_mean = np.stack([d_all[..., plds_all == v].mean(-1) for v in delays], -1)
m0c_mp = quant.m0_correction(mp.m0scan(), tr=mp.m0scan_sidecar()["RepetitionTimePreparation"])
mask_mp = mp.mask()
offsets = sm["offsets"]
nz = len(offsets)

fig, ax = plt.subplots(figsize=(9, 3.4))
for t_name, t in (("GM", GM), ("WM", WM)):
    ax.axvspan(t.att, t.att + TAU, color=TISSUE_COLORS[t_name], alpha=0.18, lw=0, label=f"{t_name} inflow: ATT {t.att:g} to {t.att + TAU:g} s")
for z in range(nz):
    ax.plot(delays + TAU + offsets[z], np.full(len(delays), z), "o", color=INK["primary"], ms=3, label="signal times of the slice" if z == 0 else None)
ax.set(xlabel="signal time from the start of labeling (s)", ylabel="slice", yticks=range(0, nz, 5), xlim=(0, 6))
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
fig.tight_layout()

n_gm = [(delays[0] + TAU + o < GM.att + TAU) for o in offsets].count(True)
n_wm = [(delays[0] + TAU + o < WM.att + TAU) for o in offsets].count(True)
print(f"slices with a sample on the GM inflow (t < {GM.att + TAU:g} s): {n_gm} of {nz};  on the WM inflow (t < {WM.att + TAU:g} s): {n_wm} of {nz}")
```

Each row is one slice, each dot one of its six signal times; the shaded bands are the
inflow windows of gray matter (0.8 to 2.6 s) and white matter (1.2 to 3.0 s). Only the
shortest delay lands inside either band, and the readout's 0.76 s spread moves it out:
8 of 20 slices have a sample on the gray-matter inflow, 18 of 20 on the white-matter
inflow. In the other slices every sample lies on the decaying tail, where the transit time
enters only through a weak amplitude factor and the fit cannot separate it from CBF. A
3D readout would have all 20 rows on the first line, and a shortest delay of 0.25 s
would put the first sample well inside the band; this protocol was designed for its first
slice.

```{code-cell} python
:tags: [hide-input]
cbf_with, att_with = quant.fit_multi_pld(d_mean, delays, m0c_mp, mask=mask_mp, slice_offsets=offsets)
cbf_wo, att_wo = quant.fit_multi_pld(d_mean, delays, m0c_mp, mask=mask_mp)
gm_mp = sm["masks"]["GM"]
zs = np.arange(nz)
per_slice = lambda vol: np.array([vol[:, :, z][gm_mp[:, :, z]].mean() if gm_mp[:, :, z].any() else np.nan for z in zs])

fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
axes[0].plot(zs, per_slice(att_with), "o-", color=PALETTE[0], label="fit with slice timing")
axes[0].plot(zs, per_slice(att_wo), "o-", color=PALETTE[7], label="fit without slice timing")
axes[0].axhline(GM.att, color="k", lw=0.8, ls="--", label="truth 0.8 s")
axes[0].axvspan(n_gm - 0.5, nz - 0.5, color=INK["grid"], alpha=0.6, lw=0, label="no sample on the GM inflow")
axes[0].set(xlabel="slice", ylabel="fitted ATT in pure GM (s)", ylim=(0.4, 1.6))
axes[0].legend(loc="upper left", fontsize=7)
axes[1].plot(zs, per_slice(cbf_with), "o-", color=PALETTE[0], label="with slice timing")
axes[1].plot(zs, per_slice(cbf_wo), "o-", color=PALETTE[7], label="without")
axes[1].axhline(GM.perfusion, color="k", lw=0.8, ls="--", label="truth 60")
axes[1].axvspan(n_gm - 0.5, nz - 0.5, color=INK["grid"], alpha=0.6, lw=0)
axes[1].set(xlabel="slice", ylabel="fitted CBF in pure GM (ml/100 g/min)", ylim=(30, 100))
axes[1].legend(loc="upper left", fontsize=7)
fig.tight_layout()

a_w, a_o = per_slice(att_with), per_slice(att_wo)
print(f"GM ATT over all slices: with slice timing {att_with[gm_mp].mean():.2f} s, without {att_wo[gm_mp].mean():.2f} s (truth {GM.att:g})")
print(f"GM ATT in slices 0-{n_gm - 1} (inflow sampled): with {np.nanmean(a_w[:n_gm]):.2f} s, without {np.nanmean(a_o[:n_gm]):.2f} s;  slice 0: {a_w[0]:.2f} / {a_o[0]:.2f}, slice {n_gm - 1}: {a_w[n_gm - 1]:.2f} / {a_o[n_gm - 1]:.2f}")
print(f"GM ATT in slices {n_gm}-{nz - 1} (tail only): with {np.nanmean(a_w[n_gm:]):.2f} s, without {np.nanmean(a_o[n_gm:]):.2f} s;  slice {nz - 1}: {a_w[-1]:.2f} / {a_o[-1]:.2f}")
print(f"GM CBF over all slices: with {cbf_with[gm_mp].mean():.1f}, without {cbf_wo[gm_mp].mean():.1f} (truth {GM.perfusion:g})")
```

Left, the fitted gray-matter transit time slice by slice. In the first slice the two
fits coincide at 0.80 s, because its offset is zero. Through the slices that still sample
the inflow (the unshaded region) the fit without slice timing drifts down by the slice's
offset, to 0.62 s at slice 7, while the fit with slice timing stays at 0.80 to 0.87 s
(0.82 s on average). In the shaded slices neither fit is anchored: the values with slice
timing wander upward, 1.04 s on average and 1.35 s in the last slice, the values without
stay low at 0.68 s, and neither number means anything, because no sample was taken
while the bolus was arriving. Right, the CBF
follows the transit time through the coupling of [Chapter 15](./15-multi-delay.md): the
fit without slice timing gives 56.2 against 67.1 with, and it is closer to 60 by
accident, an ATT bias cancelling the noise bias. Decision: fit with the slice timing
(the `SliceTiming` field, which is why the rules engine asks for it), report ATT only in
the slices that sample the inflow, and treat the CBF from the other slices as
single-delay estimates whose transit time was assumed rather than measured. For a 2D
multi-delay protocol, choose the shortest delay so that the *last* slice still samples
the inflow of the fastest tissue: here that is 0.5 − 0.76 s, which is not possible, so the
readout has to be shorter, or 3D.

## Worked case 3: the delay against the population's transit times

The white paper recommends a delay of 1.5 s for children, 1.8 s for healthy adults
under 70, and 2.0 s for older adults and patients, tracking the transit times of each
population. The phantom cannot age, but its two tissues stand in for two populations:
gray matter arrives at 0.8 s and white matter at 1.2 s, so a delay that is comfortable
for one can be short for the other. The `pld-sweep` dataset quantifies the same brain at
six delays with the single-delay formula, corrected for slice timing.

```{code-cell} python
:tags: [hide-input]
sweep = data.load_dataset("pld-sweep")
names = ["pld05", "pld10", "pld15", "pld20", "pld25", "pld30"]
rows = []
per_slice_wm = {}
for name in names:
    r = sweep.run(name)
    sr = summarize(r)
    m0c = quant.m0_correction(r.m0scan(), tr=r.m0scan_sidecar()["RepetitionTimePreparation"])
    cbf = quant.cbf_pcasl(sr["dm"], m0c, sr["plds"])
    pld = sr["p"]["PostLabelingDelay"]
    rows.append((pld, cbf[sr["masks"]["GM"]].mean(), cbf[sr["masks"]["WM"]].mean()))
    wm_z = sr["masks"]["WM"]
    per_slice_wm[pld] = (pld + sr["offsets"], np.array([cbf[:, :, z][wm_z[:, :, z]].mean() if wm_z[:, :, z].any() else np.nan for z in range(nz)]))
rows = np.array(rows)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
axes[0].plot(rows[:, 0], rows[:, 1] / GM.perfusion, "o-", color=TISSUE_COLORS["GM"], label="GM (ATT 0.8 s)")
axes[0].plot(rows[:, 0], rows[:, 2] / WM.perfusion, "o-", color=TISSUE_COLORS["WM"], label="WM (ATT 1.2 s)")
axes[0].axvline(GM.att, color=TISSUE_COLORS["GM"], lw=0.8, ls=":")
axes[0].axvline(WM.att, color=TISSUE_COLORS["WM"], lw=0.8, ls=":")
axes[0].set(xlabel="post-labeling delay (s)", ylabel="estimated / true CBF", ylim=(0, 1.0))
axes[0].legend(loc="upper right")
for pld, (x, y) in per_slice_wm.items():
    if pld in (1.0, 1.5, 2.0):
        axes[1].plot(x, y / WM.perfusion, "o-", ms=3, label=f"PLD {pld:g} s, slices 0–19")
axes[1].axvline(WM.att, color=TISSUE_COLORS["WM"], lw=0.8, ls=":")
axes[1].text(WM.att + 0.02, 0.05, "WM ATT", fontsize=7, color=TISSUE_COLORS["WM"])
axes[1].set(xlabel="effective delay of the slice (s)", ylabel="estimated / true WM CBF", ylim=(0, 0.9))
axes[1].legend(loc="upper right")
fig.tight_layout()

for pld, g, w in rows:
    print(f"PLD {pld:.1f} s: GM {g:5.1f} ({g / GM.perfusion:.0%} of 60)   WM {w:5.1f} ({w / WM.perfusion:.0%} of 20)   GM/WM ratio {g / w:.1f} (truth 3.0)")
```

Left, the estimate as a fraction of the truth against the delay, with the two transit
times marked. Neither tissue reaches 1, for the reason [Chapter 14](./14-cbf-quantification.md)
and [Chapter 16](./16-calibration.md) give: the formula lets the label decay with the
blood's $T_1$ after arrival, the simulator's label decays with the tissue's, and the
shortfall grows with PLD − ATT. White matter, with its short $T_1$ of 0.83 s, pays the
most: 44 percent of the truth at PLD 1.5 s and 34 percent at 2.0 s, so the gray-to-white
ratio, 3 in the phantom, reads 5.2 at 1.5 s and 6.3 at 2.0 s. Below the transit time the
other error adds to it: at 0.5 s, before the bolus has arrived in white matter, its
estimate falls from 58 percent at 1.0 s to 51 percent, and gray matter from 83 to 81. The
drop is modest because the 1.8 s labeling duration means that most of the bolus has
still arrived by the time of the readout; a shorter bolus, or a longer transit time,
would make it steep. Right, the white-matter estimate of each slice against that slice's
effective delay for the 1.0, 1.5, and 2.0 s runs: the three runs join into one curve,
which is the point. What matters is the delay each slice actually has, and a 2D protocol
at a nominal 1.5 s spans 1.5 to 2.26 s.

The decision for a population depends on which error one fears. For a pediatric
cohort with short transit times a delay of 1.5 s loses nothing to arrival and keeps 7
percent more gray-matter signal than 2.0 s (76 against 71 percent of the truth here).
For an older cohort whose slowest regions arrive after 1.5 s, a 1.5 s delay produces the
left edge of this figure in those regions: a regional deficit that is a transit-time
deficit, indistinguishable from low perfusion without a second delay. The white paper's
2.0 s is the price of not knowing the ATT; the multi-delay acquisition of case 2 is the
alternative that measures it.

## What complex data add

Less than in diffusion imaging. The phase of an ASL volume is dominated by the field
offset and the coil, both static between control and label, and the difference signal
is a 1 percent change in magnitude, not in phase. Complex subtraction (subtracting
control and label as complex numbers before taking the magnitude) removes the Rician
floor in the lowest-signal voxels and lets the noise average without bias
([Chapter 8](../03-preprocessing/08-noise.md)); with background suppression, where the
control's magnitude is near the noise floor, this is the case in which the phase matters.
Saving the phase costs no scan time and doubles the storage, and every simulated series
in this book carries a `part-phase` file for that reason. No ASL model in this book uses
the phase itself.

## The reproducibility checklist

A quantification needs fields the images do not carry. The list below is what the rules
engine and the formulas of Chapters [14](./14-cbf-quantification.md) through 16 read
from the sidecar, all of them defined by ASL-BIDS {cite:p}`clement2022`.

```{code-cell} python
:tags: [hide-input]
REQUIRED = {
    "any": ["ArterialSpinLabelingType", "PostLabelingDelay", "MagneticFieldStrength", "RepetitionTimePreparation", "EchoTime",
            "MRAcquisitionType", "BackgroundSuppression", "M0Type", "TotalAcquiredPairs", "AcquisitionVoxelSize"],
    "PCASL/CASL": ["LabelingDuration"],
    "PASL": ["BolusCutOffFlag", "BolusCutOffDelayTime", "BolusCutOffTechnique"],
    "2D": ["SliceTiming"],
    "BackgroundSuppression": ["BackgroundSuppressionNumberPulses", "BackgroundSuppressionPulseTime"],
    "M0Type Estimate": ["M0Estimate"],
    "recommended": ["LabelingEfficiency", "VascularCrushing", "PhaseEncodingDirection", "TotalReadoutTime"],
}


def missing(p):
    need = list(REQUIRED["any"])
    need += REQUIRED["PASL"] if p.get("ArterialSpinLabelingType") == "PASL" else REQUIRED["PCASL/CASL"]
    if p.get("MRAcquisitionType") == "2D":
        need += REQUIRED["2D"]
    if p.get("BackgroundSuppression"):
        need += REQUIRED["BackgroundSuppression"]
    if p.get("M0Type") == "Estimate":
        need += REQUIRED["M0Type Estimate"]
    return [k for k in need if k not in p], [k for k in REQUIRED["recommended"] if k not in p]


for label, p in [("ref-pcasl sidecar (aslscan)", ref.sidecar()), ("ref-pcasl M0 scan sidecar", ref.m0scan_sidecar() | {"ArterialSpinLabelingType": "n/a", "PostLabelingDelay": 0, "BackgroundSuppression": False, "M0Type": "n/a", "TotalAcquiredPairs": 0, "LabelingDuration": 0})]:
    req, rec = missing(p)
    print(f"{label}: missing required {req or 'none'}; missing recommended {rec or 'none'}")
```

Beside the sidecar, the `aslcontext.tsv` that names every volume, the M0 scan's own
`RepetitionTimePreparation` and `EchoTime`, and, from the analysis side, the values of
$\lambda$, $T_{1b}$, $\alpha$, and the tissue $T_1$ used, with the calibration route. The
book's own sidecars carry every required field and the recommended ones; a real
dataset that lacks `SliceTiming` on a 2D readout, or `BolusCutOffDelayTime` on a PASL
series, cannot be quantified without guessing, and the rules engine says so.

## What this implies for acquisition

- **Decide the analyses, then read the table along their rows.** The cheapest column
  that is green for all of them is the protocol.
- **For absolute CBF in a single-delay study:** an M0 scan, background suppression, 30
  pairs or more, and a delay above the longest transit time you expect, which for a
  mixed or older population is 2.0 s.
- **For transit times:** a 3D readout or a short 2D readout, and a shortest delay that
  samples the inflow in every slice; with the book's 0.76 s readout that is a delay
  shorter than the fastest ATT minus 0.76 s, which is not available, so it is 3D.
- **For an inherited series without M0:** if it was not suppressed, calibrate on the mean
  control image with the TR and TE corrections and say so; if it was suppressed, report
  relative CBF.
- **Record `SliceTiming`, `M0Type`, the bolus fields, and `LabelingEfficiency`.** They
  cost nothing and every verdict above reads them.

## Further reading

The consensus recommendations for single-delay {cite:p}`alsop2015` and multi-delay
{cite:p}`woods2024` ASL; the ASL-BIDS specification {cite:p}`clement2022`; and how two
pipelines make these decisions automatically {cite:p}`adebimpe2022,mutsaerts2020`.
