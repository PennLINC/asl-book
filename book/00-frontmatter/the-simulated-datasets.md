---
title: "0.2 The simulated datasets"
subtitle: One digital brain, simulated end to end, with its answer key
kernelspec:
  name: python3
  display_name: Python 3
---

The simulated data in this book come from one digital phantom, a brain in which every
voxel's perfusion, transit time, and relaxation times are known exactly, passed through one
simulator, aslscan, under protocols that change one thing at a time. This page describes
what the phantom is made of, what the simulator does to it, which datasets are made from it,
what the ground truth is, and what the ground truth is not.

:::{tip} Skim this page on a first read
This page is a reference for the whole book, and it uses terms that the chapters introduce
later. On a first read, look at the diagram below and the phantom maps, and read the first
paragraph of each section. Come back when a chapter points here.
:::

```{code-cell} python
:tags: [hide-cell]
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from aslbook import cookbook, phantom, presets
from aslbook.plotting import INK, PALETTE, TISSUE_COLORS, set_style, show_slice

set_style()
```

## The simulator in one picture

aslscan starts from a description of a brain and of an ASL protocol and produces the files
a scanner would have produced. Because it also knows the brain it started from, it writes
the correct answer for every quantity the book later estimates from the data. Read the
diagram from left to right: the gray boxes are the inputs, the blue boxes the three stages
of the simulation, the green box the data a reader would receive from a real scanner, and
the orange box the answer key that no real scanner can provide.

```{code-cell} python
:tags: [hide-input]
def box(ax, x0, y0, x1, y1, title, body, color):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=color + "22", ec=color, lw=1.4))
    ax.text((x0 + x1) / 2, y1 - 0.14, title, ha="center", va="top", fontsize=9, weight="bold", color=INK["primary"])
    ax.text((x0 + x1) / 2, y1 - 0.45, body, ha="center", va="top", fontsize=7.6, color=INK["secondary"], linespacing=1.3)

def arrow(ax, xy_from, xy_to, dashed=False):
    ax.annotate("", xy=xy_to, xytext=xy_from, arrowprops=dict(arrowstyle="-|>", color=INK["secondary"], lw=1.3,
                                                             ls="--" if dashed else "-", shrinkA=0, shrinkB=0))

fig, ax = plt.subplots(figsize=(12.5, 5.0))
ax.set(xlim=(0, 12.5), ylim=(0, 5.0))
ax.set_axis_off()
inp, stage, out, truth = "#8a8a8a", PALETTE[0], PALETTE[2], PALETTE[1]
box(ax, 0.1, 3.95, 2.3, 4.85, "protocol", "BIDS sidecar +\naslcontext.tsv", inp)
box(ax, 0.1, 2.75, 2.3, 3.65, "phantom", "perfusion, transit time,\nT1, T2, T2*, M0, labels\nper 1 mm voxel", inp)
box(ax, 0.1, 1.55, 2.3, 2.45, "overlay", "noise, coils, suppression\nefficiency, motion, ...", inp)
box(ax, 3.0, 2.15, 5.1, 3.85, "1. kinetics", "the general kinetic model:\nthe label-control difference\nin every voxel at every\nvolume's and slice's timing", stage)
box(ax, 5.5, 2.15, 7.6, 3.85, "2. signal", "tissue: saturation recovery\nat TR (or the suppression\ntimeline); blood: the\ndifference, signed per row", stage)
box(ax, 8.0, 2.15, 10.1, 3.85, "3. acquisition", "2D spin-echo EPI: k-space,\nT2/T2* decay, field map,\ncoils, GRAPPA, ringing,\nghosts, noise, motion", stage)
box(ax, 10.5, 2.15, 12.4, 3.85, "output data", "BIDS ASL series:\nmagnitude + phase,\naslcontext, sidecars,\nM0 scan", out)
box(ax, 3.0, 0.15, 9.0, 1.35, "ground truth (the answer key)", "perfusion, transit time, T1, T2, M0, tissue labels and fractions, and the noise-free\ndifference of every volume, all on the acquisition grid, with no scanner in between", truth)
arrow(ax, (2.3, 4.4), (4.05, 4.4)); arrow(ax, (4.05, 4.4), (4.05, 3.85))
arrow(ax, (2.3, 3.2), (3.0, 3.2))
arrow(ax, (2.3, 2.0), (9.05, 2.0)); arrow(ax, (9.05, 2.0), (9.05, 2.15))
arrow(ax, (5.1, 3.0), (5.5, 3.0)); arrow(ax, (7.6, 3.0), (8.0, 3.0)); arrow(ax, (10.1, 3.0), (10.5, 3.0))
arrow(ax, (4.0, 2.15), (4.0, 1.35))
ax.plot([11.45, 11.45], [2.15, 0.75], color=INK["secondary"], lw=1.3, ls="--")
arrow(ax, (11.45, 0.75), (9.0, 0.75), dashed=True)
ax.text(10.2, 0.62, "quantify the data,\nscore against the truth", ha="center", va="top", fontsize=7.6, color=INK["secondary"])
fig.tight_layout()
```

The protocol enters the kinetics (it sets every volume's timing) and the acquisition (it
sets the readout); the overlay carries what BIDS does not record. The ground truth branches
off after the kinetics: it describes the tissue and the label, not the images.

## The phantom

The phantom is ASLDRO's `hrgt_icbm_2009a_nls_3t` digital reference object
{cite:p}`olivertaylor2021`: the ICBM 2009a nonlinear symmetric template segmented into gray
matter, white matter, and CSF at 1 mm, with one value of each quantity per tissue class.
That is the important property. A real brain has a distribution of perfusion within gray
matter; this one has exactly 60 ml/100 g/min in every gray matter voxel, so every estimate
can be scored against a number with no uncertainty of its own.

```{code-cell} python
:tags: [hide-input]
print(f"{'tissue':>6} {'CBF':>6} {'ATT (s)':>8} {'T1 (s)':>7} {'T2 (ms)':>8} {'T2* (ms)':>9} {'M0':>6}")
for t in presets.TISSUES.values():
    att = "–" if t.perfusion == 0 else f"{t.att:.1f}"
    print(f"{t.name:>6} {t.perfusion:6.0f} {att:>8} {t.t1:7.2f} {1000 * t.t2:8.0f} {1000 * t.t2star:9.0f} {t.m0:6.1f}")
print(f"\narterial blood: T1 {presets.T1_BLOOD} s, T2 {1000 * presets.T2_BLOOD:.0f} ms; partition coefficient {presets.LAMBDA} ml/g; "
      f"labeling efficiency PCASL {presets.ALPHA['PCASL']}, PASL {presets.ALPHA['PASL']}")
```

CBF is in ml/100 g/min and the transit time (ATT) in seconds; CSF is not perfused (its
transit time is a sentinel the simulator never uses). M0 is the equilibrium magnetization in
arbitrary units, which set the units of every image.

The book uses a 100 mm axial slab of the phantom, from the temporal lobes to just below the
vertex, which 20 slices of 5 mm tile exactly. The simulator evaluates its physics on the
1 mm grid and averages the magnetization onto the acquisition grid, so a 3.5 × 3.5 × 5 mm
voxel at a tissue boundary holds the mixture a real voxel would. The pipeline records the
exact fraction of each tissue in every acquisition voxel, and the packaged slab that the toy
tier uses is those fractions:

```{code-cell} python
:tags: [hide-input]
ph = phantom.slab()
m = phantom.maps(ph)
k = phantom.DISPLAY_SLICE
fig, axes = plt.subplots(1, 5, figsize=(14, 3.1))
for ax, key, name in zip(axes, ("gm", "wm", "csf"), ("gray matter", "white matter", "CSF")):
    show_slice(ax, ph[key], k, f"{name} fraction", kind="fraction")
show_slice(axes[3], m["perfusion"], k, "true CBF (ml/100 g/min)", kind="cbf", colorbar=True)
show_slice(axes[4], np.where(ph["mask"], m["att"], np.nan), k, "true transit time (s)", kind="att", colorbar=True)
fig.tight_layout()
print(f"slab {ph['gm'].shape} at {ph['voxel_mm']} mm; {int(ph['mask'].sum())} brain voxels, "
      f"{int((ph['gm'] > 0.9).sum())} of them at least 90 % gray matter")
```

The first three panels are the tissue fractions of the display slice, which passes through
the lateral ventricles; brighter means more of that tissue in the voxel. The fourth is the
true CBF of the same slice: 60 where the voxel is pure gray matter, 20 in pure white matter,
and in between at every boundary. The fifth is the true transit time, longer in white
matter. Only a minority of voxels are nearly pure gray matter, which is the partial volume
problem of [Chapter 12](../03-preprocessing/12-partial-volume.md).

```{code-cell} python
:tags: [hide-input]
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
show_slice(axes[0], m["perfusion"], k, "axial (display slice)", kind="cbf")
axes[1].imshow(np.rot90(m["perfusion"][:, 34, :]), cmap="inferno", vmin=0, vmax=90, aspect=5 / 3.5)
axes[1].set_title("coronal"); axes[1].axis("off")
axes[2].imshow(np.rot90(m["perfusion"][32, :, :]), cmap="inferno", vmin=0, vmax=90, aspect=5 / 3.5)
axes[2].set_title("sagittal (anterior to the left)"); axes[2].axis("off")
fig.tight_layout()
```

The slab in three directions, as true CBF. The slices are thicker than they are wide, which
is the usual ASL choice and is drawn to scale here.

## What aslscan simulates

aslscan is a headless ASL simulator, written to reproduce ASLDRO's kinetic and signal models
{cite:p}`olivertaylor2021` and to add a physical model of the acquisition. It works in the
three stages of the diagram.

:::{dropdown} The kinetics (Chapter 5 explains the terms)
For every phantom voxel and every volume of the series, the general kinetic model of
{cite:t}`buxton1998` gives the label-control difference in longitudinal magnetization from
the voxel's perfusion, transit time, T1, and M0, and from the protocol's labeling type,
bolus duration, and post-labeling delay, evaluated at the time each 2D slice is actually
read out (the delay plus the slice's offset in the readout). The result is exactly what the
toy tier computes with `aslbook.kinetic.delta_m`.
:::

:::{dropdown} The signal (Chapters 1, 6, and 9)
Each tissue class is one compartment carrying the longitudinal magnetization of static
tissue before the 90° excitation: the saturation-recovery steady state at the repetition
time, or, under background suppression, the signed value after the inversion pulses. The
labeled blood is a second compartment per class carrying the difference, negative on label
rows and positive on `deltam` rows; under suppression it is multiplied by the pulses' factor.
An included or separate M0 scan takes the plain steady state at its own repetition time.
:::

:::{dropdown} The acquisition (Chapters 2, 8, 10, 11, and 13)
The compartment images are simulated on a grid twice as fine as the acquisition in-plane
and acquired slice by slice as a single-shot spin-echo EPI: each compartment decays with its
own T2 and T2' along the readout (the labeled blood with the T2 of blood), the field map
displaces the image along the phase-encode axis, k-space is truncated to the acquisition
matrix (so edges ring), partial Fourier, Nyquist ghosts, spikes, receive coils, GRAPPA, and
Gaussian noise are applied in k-space, and the image is reconstructed as magnitude and
phase. Head motion moves the compartment images by a rigid pose before each volume is
acquired. This stage is the shared `mrsim-acq` library, also used by the diffusion MRI
simulator TRXScan.
:::

The output is a BIDS ASL dataset {cite:p}`clement2022`: `part-mag` and `part-phase` images,
`aslcontext.tsv`, the separate M0 scan when the protocol has one, and JSON sidecars that
repeat the protocol and add an `AslscanSimulation` block recording every value the simulator
resolved and where it came from. [Appendix A](../appendices/a-aslscan-cookbook.md) shows the
inputs behind every dataset.

## The reference protocol

Every dataset changes one thing from one reference acquisition, so that a reader learns one
series and recognizes it everywhere. The reference follows the ASL white paper's
recommendations {cite:p}`alsop2015` for a 2D pseudo-continuous acquisition, within what the
simulator models:

```{code-cell} python
:tags: [hide-input]
r = presets.REFERENCE
print(f"2D PCASL: labeling duration {r.labeling_duration} s, post-labeling delay {r.post_labeling_delay} s, TR {r.repetition_time} s, TE {1000 * r.echo_time:.0f} ms")
print(f"voxels {r.voxel_mm[0]} x {r.voxel_mm[1]} x {r.voxel_mm[2]} mm, matrix {r.matrix[0]} x {r.matrix[1]}, {r.n_slices} slices {1000 * r.slice_spacing:.0f} ms apart (readout {r.readout_duration:.2f} s)")
print(f"{r.n_pairs} control-label pairs ({2 * r.n_pairs * r.repetition_time / 60:.1f} min), separate M0 scan at TR {r.m0_repetition_time} s, no background suppression")
print(f"noise: variance {r.noise_variance:.0f} per component in image units (sigma {r.noise_variance ** 0.5:.0f}); gray matter control SNR about {presets.REFERENCE_GM_SNR:.0f}")
```

The white paper prefers a 3D readout with background suppression; the simulator reads 2D
slices, so the book treats the 2D-specific effects (slice timing in
[Chapter 6](../02-labeling/06-the-asl-signal.md), the slice dependence of suppression in
[Chapter 9](../03-preprocessing/09-background-suppression.md)) explicitly and discusses 3D
readouts in [Chapters 7](../02-labeling/07-acquisition-parameters.md) and
[19](../05-advanced/19-frontiers.md).

## The datasets

Two kinds of simulated data appear in the book, and every chapter opens with a box that
lists which it uses.

**Toy tier.** The kinetic model and the signal equations evaluated in the page on the
packaged slab (`aslbook.synth`), which is the simulator's first two stages without the
third: every voxel's tissue and blood signals from its tissue fractions and the class
constants, the T2 decay to the echo time per compartment, Gaussian noise, and the
magnitude. There is no k-space, so no ringing, distortion, ghosts, or coils. Its noise-free
difference reproduces the simulator's `deltam` ground truth to floating-point precision
([Chapter 5](../02-labeling/05-kinetic-model.md) shows it), which is what makes the two tiers
one simulator rather than two.

**Pipeline tier.** Full aslscan runs on the slab, made offline by the repository's Snakemake
pipeline, each a BIDS dataset with its ground truth and a provenance record.
[Appendix A](../appendices/a-aslscan-cookbook.md#app-a-datasets) describes every dataset and
shows the protocol, overlay, and command behind each run.

```{code-cell} python
:tags: [hide-input]
cfg = cookbook.load_config()
print(f"{'dataset':<14} {'runs':<44} chapters")
for ds, entry in cfg["datasets"].items():
    runs = ", ".join(r.name for r in cookbook.expand(cfg, ds))
    print(f"{ds:<14} {runs:<44} {', '.join(str(c) for c in entry['chapters'])}")
```

| Dataset | What varies |
|---|---|
| [`ref-pcasl`](../appendices/a-aslscan-cookbook.md#ds-ref-pcasl) | nothing: the reference series |
| [`ref-clean`](../appendices/a-aslscan-cookbook.md#ds-ref-clean) | the noise is off |
| [`label-types`](../appendices/a-aslscan-cookbook.md#ds-label-types) | the labeling scheme: PCASL, CASL, PASL |
| [`pld-sweep`](../appendices/a-aslscan-cookbook.md#ds-pld-sweep) | the post-labeling delay, 0.5 to 3.0 s |
| [`multi-pld`](../appendices/a-aslscan-cookbook.md#ds-multi-pld) | six delays in one series |
| [`noise-sweep`](../appendices/a-aslscan-cookbook.md#ds-noise-sweep) | the noise level; eight coils with GRAPPA |
| [`bgsup`](../appendices/a-aslscan-cookbook.md#ds-bgsup) | background suppression off, on, perfect, with presaturation |
| [`motion`](../appendices/a-aslscan-cookbook.md#ds-motion) | random head motion, a drift, motion under suppression |
| [`sdc`](../appendices/a-aslscan-cookbook.md#ds-sdc) | a field map with both phase-encode polarities, and none |
| [`voxel-sweep`](../appendices/a-aslscan-cookbook.md#ds-voxel-sweep) | the in-plane voxel size |
| [`m0-types`](../appendices/a-aslscan-cookbook.md#ds-m0-types) | how the calibration image is acquired |
| [`te-sweep`](../appendices/a-aslscan-cookbook.md#ds-te-sweep) | the echo time |
| [`readout`](../appendices/a-aslscan-cookbook.md#ds-readout) | partial Fourier, a ghost, spikes |
| [`kitchen-sink`](../appendices/a-aslscan-cookbook.md#ds-kitchen-sink) | everything at once |

## The ground truth

Because the simulator knows each voxel's tissue mixture exactly, it writes the correct value
of the quantities that Part IV estimates from the data, on the acquisition grid, next to
every series: the perfusion and transit time maps, the T1, T2, and M0 maps, the tissue
labels, and the noise-free label-control difference of every volume at its own timing
([Appendix E](../appendices/e-truth-map-catalogue.md)). The pipeline adds the fraction of
each tissue in every acquisition voxel. An estimate can then be scored on the quantity it
claims to measure, which is what every *Measure it* section does.

## What the ground truth is not

The phantom is a model, and its answer key is exact for that model, not for real tissue. A
method that recovers the truth here has shown that it works on data like these; it has not
shown that a real brain is built the way the phantom assumes.

:::{dropdown} The limits, and where the book returns to each
- **One value per tissue.** Real gray matter has a distribution of perfusion and transit
  time, and the boundary between tissues is not the only source of variation. A voxelwise
  score against the truth here measures noise and partial volume, not physiological
  heterogeneity ([Chapter 12](../03-preprocessing/12-partial-volume.md)).
- **No arterial compartment and no dispersion.** The kinetic model delivers the label
  straight into the tissue compartment as a sharp bolus. Real data show labeled blood still
  in arteries at short delays, and a bolus whose edges have blurred in transit
  ([Chapter 5](../02-labeling/05-kinetic-model.md), [Chapter 19](../05-advanced/19-frontiers.md)).
- **The label decays with the blood's T1 until it arrives, then with the tissue's.** There
  is no exchange time and no restricted exchange
  ([Chapter 19](../05-advanced/19-frontiers.md)).
- **Background suppression inverts the whole bolus.** Every pulse acts on all labeled
  blood wherever it is; real pulses cover the imaging region, so the label still in transit
  is inverted fewer times. The retained label in a suppressed dataset is an upper bound, and
  the sidecar names the approximation ([Chapter 9](../03-preprocessing/09-background-suppression.md)).
- **Motion moves finished images.** A rigid pose is applied to the simulated compartment
  images of a volume; the field map does not move with the head, and the kinetics are not
  re-evaluated after through-plane motion ([Chapter 10](../03-preprocessing/10-motion.md)).
- **A 2D single-shot spin-echo EPI scanner.** No 3D readouts, no physiological noise, no
  pulsatility, no vessel-selective or velocity-selective labeling
  ([Chapter 19](../05-advanced/19-frontiers.md)).
- **The readout is not neutral at edges.** The truth maps are box averages of the phantom,
  but the images pass through a truncated k-space, so even the noise-free difference rings
  by about a unit around tissue boundaries and sits about 2 % above the truth in pure gray
  matter ([Chapter 5](../02-labeling/05-kinetic-model.md)); whole-brain sums agree to a
  fraction of a percent.
:::

Within those limits the simulated datasets do what no real dataset can: they let every
step from acquisition to perfusion map be scored against a known answer, under the
artifacts a real scanner produces, at the parameters of a specific protocol.

## Further reading

The ASLDRO digital reference object whose phantom and kinetic model aslscan reproduces
{cite:p}`olivertaylor2021`, the general kinetic model {cite:p}`buxton1998`, the ASL white
paper whose recommendations the reference protocol follows {cite:p}`alsop2015`, and the
ASL-BIDS specification the datasets are written in {cite:p}`clement2022`.
