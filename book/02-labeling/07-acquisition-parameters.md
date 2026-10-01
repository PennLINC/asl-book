---
title: "7. Acquisition parameter choices"
kernelspec:
  name: python3
  display_name: Python 3
---

:::{admonition} Simulated datasets in this chapter
:class: note
- **Built in this page:** kinetic-model curves for the phantom's tissues under varied timing, and the SNR arithmetic of a protocol ([Appendix B](../appendices/b-data-manifest.md#app-b-package-data)).
- **`pld-sweep`**: single-delay PCASL at six post-labeling delays, 15 pairs each at TR 6 s ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-pld-sweep)).
- **`voxel-sweep`**: the reference protocol at 2.5, 3.5, and 5 mm in-plane voxels with the noise scaled inversely with the voxel volume (σ 78, 40, 20), 15 pairs ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-voxel-sweep)).
- **`te-sweep`**: the reference protocol at echo times of 12, 30, and 60 ms, 15 pairs ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-te-sweep)).
- **`noise-sweep`**: the reference protocol at noise σ of 10, 40, and 80 image units, 30 pairs ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-noise-sweep)).

Pipeline-tier datasets are simulated offline by aslscan ([Chapter 0.2](../00-frontmatter/the-simulated-datasets.md)).
:::

## Learning goals

After this chapter you can:

- state what each acquisition parameter of an ASL protocol controls physically, what it
  costs, and which later chapter deals with its consequences: post-labeling delay, labeling
  duration, TR, number of pairs, voxel size, TE, 2D or 3D readout, background suppression,
  the M0 scan, and field strength
- estimate the difference signal, its SNR, and the scan time of a proposed protocol from
  the kinetic model before it is run
- design a single-delay PCASL protocol under a scan-time budget for a given population and
  justify each number

```{code-cell} python
:tags: [hide-cell]
import numpy as np
import matplotlib.pyplot as plt

from aslbook import data, kinetic, phantom, presets, protocols, quant
from aslbook.plotting import INK, PALETTE, TISSUE_COLORS, sequence_diagram, set_style, show_image, take_slice

set_style()
T = presets.TISSUES
REF = presets.REFERENCE
TAU = REF.labeling_duration
GM = T["GM"]
SCALE = REF.signal_scale * np.exp(-REF.echo_time / presets.T2_BLOOD)   # M0 units -> image units at TE 12 ms
OVERHEAD = REF.repetition_time - TAU - REF.post_labeling_delay          # what TR holds beyond LD + PLD: the 2D readout and a margin

def dm_gm(pld, *, att=GM.att, tau=TAU, t1b=presets.T1_BLOOD):
    """Gray matter difference signal at the first slice, as a fraction of M0."""
    return kinetic.delta_m(np.asarray(pld) + tau, GM.perfusion, att, GM.t1, GM.m0, tau=tau, t1b=t1b) / GM.m0

print(f"TR of the reference protocol: LD {TAU:.1f} + PLD {REF.post_labeling_delay:.1f} + {OVERHEAD:.1f} s for the readout and margin = {REF.repetition_time:.1f} s")
```

Every parameter below trades signal, time, and robustness against each other, and the
kinetic model of [Chapter 5](./05-kinetic-model.md) makes every trade computable before the
scan. The white paper {cite:p}`alsop2015` turned those computations into a set of
recommendations for a 3 T clinical protocol; this chapter reproduces the reasoning behind
them on the phantom, parameter by parameter, and ends by assembling a protocol.

## Post-labeling delay and labeling duration

The post-labeling delay $w$ sets where on the kinetic curve the image is read. It must be
longer than the transit time of every voxel you care about, or those voxels are read during
the arriving phase and their CBF is underestimated
([Chapter 5](./05-kinetic-model.md)); once it is longer, every additional second costs a
factor $e^{-1/T_1'}$, about 0.47 in gray matter, of signal for no gain in accuracy. The
white paper's recommendations for PCASL at 3 T are a delay of 1.8 s for healthy adults
under 70, 2.0 s for older adults, 1.5 s for children, and 2.0 s or more for patients with
vascular disease, whose transit times are the longest and least predictable.

The labeling duration $\tau$ sets how much label is delivered. The signal in the arrived
phase grows as $1 - e^{-\tau/T_1'}$, so 1.8 s of labeling captures three quarters of the
signal an infinitely long label would give, and 3 s about 90 %. The white paper recommends
1.8 s. Longer labeling raises the signal but also the TR, since the label and the delay
are both dead time before the readout, and it heats the tissue at the labeling plane (the
specific absorption rate, SAR, of the pulse train).

The figure shows the gray matter signal at readout against the delay for three labeling
durations, for the phantom's transit time (0.8 s, left) and for a slow-arriving territory
(1.5 s, right, typical of an older adult or a watershed region). The dotted lines are the
white paper's three delays.

```{code-cell} python
:tags: [hide-input]
w = np.linspace(0.2, 3.5, 331)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
for ax, att, title in zip(axes, (GM.att, 1.5), ("GM, ATT 0.8 s (the phantom)", "GM with ATT 1.5 s (slow arrival)")):
    for tau, color in zip((1.0, 1.8, 3.0), PALETTE):
        ax.plot(w, dm_gm(w, att=att, tau=tau) * 100, color=color, label=f"LD {tau:.1f} s")
    for pld, lab in ((1.5, "children"), (1.8, "adults"), (2.0, "elderly")):
        ax.axvline(pld, color=INK["secondary"], lw=0.8, ls=":")
        ax.text(pld, 1.62, lab, rotation=90, ha="right", va="top", fontsize=7, color=INK["secondary"])
    ax.set(title=title, xlabel="post-labeling delay (s)", xlim=(0.2, 3.5), ylim=(0, 1.7))
axes[0].set_ylabel("ΔM / M0 at readout (%)")
axes[0].legend(loc="upper right")
fig.tight_layout()
ref = dm_gm(1.8)
print("GM at PLD 1.8 s, relative to LD 1.8 s: " + ", ".join(f"LD {tau:.1f} s -> {dm_gm(1.8, tau=tau) / ref:.2f}" for tau in (1.0, 1.8, 3.0)))
print(f"GM at PLD 1.8 s: ATT 1.5 s gives {dm_gm(1.8, att=1.5) / ref:.2f} of the ATT 0.8 s signal; at PLD 2.0 s, {dm_gm(2.0, att=1.5) / ref:.2f}")
```

With the phantom's short transit time (left), every delay from 1.0 s up is on the decaying
side of the curve and the shortest delay wins on signal. With a 1.5 s transit time (right),
a 1.5 s delay reads the curve at its peak and any shorter delay would fall into the arriving
phase; the elderly recommendation of 2.0 s sits safely past it. Longer labeling raises every
curve: 3.0 s gives 1.2 times the signal of 1.8 s at PLD 1.8 s, and 1.0 s gives 0.71 times.
The next figure asks what a wrong delay does to the number that comes out.

## See it: how robust is one delay to the transit time?

The single-delay formula ([Chapter 5](./05-kinetic-model.md)) assumes the bolus has fully
arrived. The figure evaluates the model's gray matter signal for transit times from 0.3 to
2.5 s, reads it at four delays, and converts each with the white-paper formula using
`quant.cbf_pcasl`; the vertical axis is the estimate relative to the true 60 ml/100 g/min.
The left panel gives the label the blood's $T_1$ after arrival, which is the formula's own
assumption, so that the transit-time effect stands alone; the right panel uses the phantom's
gray matter $T_1$, which adds the systematic offset measured in [Chapter 5](./05-kinetic-model.md).

```{code-cell} python
:tags: [hide-input]
att_axis = np.linspace(0.3, 2.5, 221)
plds = (1.5, 1.8, 2.0, 2.5)
t1_as_blood = 1 / (1 / presets.T1_BLOOD - GM.perfusion / 6000 / presets.LAMBDA)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
for ax, t1, title in zip(axes, (t1_as_blood, GM.t1), ("label decays with T1b after arrival (the formula's assumption)", "label decays with the tissue T1 (the phantom)")):
    for pld, color in zip(plds, PALETTE):
        dm = kinetic.delta_m(pld + TAU, GM.perfusion, att_axis, t1, GM.m0)
        est = quant.cbf_pcasl(dm, GM.m0, pld, tau=TAU) / GM.perfusion
        ax.plot(att_axis, est, color=color, label=f"PLD {pld:.1f} s")
        world = "formula's world" if t1 == t1_as_blood else "phantom        "
        print(f"PLD {pld:.1f} s, {world}: estimate/true at ATT " + ", ".join(f"{a:.1f} s {np.interp(a, att_axis, est):.2f}" for a in (0.8, 1.5, 2.0, 2.2)))
    ax.axhline(1, color=INK["secondary"], lw=0.8, ls="--")
    ax.set(title=title, xlabel="true arterial transit time (s)", xlim=(0.3, 2.5), ylim=(0, 1.1))
axes[0].set_ylabel("estimated / true CBF")
axes[0].legend(loc="lower left")
fig.tight_layout()
```

On the left each curve is flat at 1.0 for every transit time shorter than its delay and
falls off beyond it: that is the whole argument for a long delay, in one picture. A 1.5 s
delay is exact up to a 1.5 s transit time and reports half the true CBF at 2.2 s; a 2.5 s
delay is exact throughout the range. On the right the plateaus sit below one (the $T_1'$
effect of [Chapter 5](./05-kinetic-model.md), 0.76 at PLD 1.8 s for this phantom) and tilt
slightly upward, but the shape of the failure is the same: the estimate is stable until the
transit time reaches the delay, then collapses. The printed numbers give the estimate at
three transit times for each delay; the comparison to make is between the columns, since
the offset within a row is the model's, not the delay's.

## See it: the delay that gives the most SNR in a fixed time

Robustness costs signal, but the accounting is not as simple as the curve suggests, because
a longer delay also lengthens the TR, so fewer pairs fit in the same scan. Let the TR be
$\tau + w$ plus what the readout needs (the reference protocol's 0.9 s), the number of pairs
in a time budget be $\mathrm{time}/(2\,\mathrm{TR})$, and the SNR of the averaged difference
be $\Delta M(w)\sqrt{\mathrm{pairs}}$. The figure plots that SNR against the delay, for two
transit times and two labeling durations, relative to the reference protocol.

```{code-cell} python
:tags: [hide-input]
budget = 300.0                                   # 5 minutes of control-label pairs
def snr_per_budget(pld, att, tau):
    tr = tau + pld + OVERHEAD
    return dm_gm(pld, att=att, tau=tau) * np.sqrt(budget / (2 * tr))
ref_snr = snr_per_budget(1.8, GM.att, TAU)
fig, ax = plt.subplots(figsize=(7, 3.4))
for att, color in zip((GM.att, 1.5), (TISSUE_COLORS["GM"], PALETTE[4])):
    for tau, ls in ((1.8, "-"), (3.0, "--")):
        rel = snr_per_budget(w, att, tau) / ref_snr
        ax.plot(w, rel, color=color, ls=ls, label=f"ATT {att:.1f} s, LD {tau:.1f} s")
        print(f"ATT {att:.1f} s, LD {tau:.1f} s: best PLD {w[rel.argmax()]:.1f} s (SNR {rel.max():.2f} x reference); at PLD 1.8 s {np.interp(1.8, w, rel):.2f}, at 2.0 s {np.interp(2.0, w, rel):.2f}")
for pld in (1.5, 1.8, 2.0):
    ax.axvline(pld, color=INK["secondary"], lw=0.8, ls=":")
ax.set(xlabel="post-labeling delay (s)", ylabel="SNR in 5 min, relative to the reference", xlim=(0.2, 3.5), ylim=(0, 2.8),
       title="gray matter SNR per fixed scan time (TR grows with LD + PLD)")
ax.legend(fontsize=8)
fig.tight_layout()
```

For the phantom's fast-arriving gray matter (orange) the best delay is the transit time
itself, and it would give 2.4 times the SNR of the reference delay of 1.8 s in the same
five minutes. The price of caution is real: a delay chosen to be safe for every subject
throws away more than half of the SNR available for the fastest ones. For the slow territory (pink) the peak moves out to 1.5 s and the curve
falls steeply on its short side, where the bolus has not arrived, so a delay chosen for
speed on one population is a delay that fails on another. Longer labeling (dashed) raises
every curve despite the longer TR, and in this model 3 s of label beats 1.8 s at every
delay. The white paper's 1.8 s is a compromise with SAR and with scanner implementations,
not an optimum of this curve.

## Repetition time

The TR of an ASL series is not a free parameter. It must hold the labeling, the delay, and
the whole readout: for a 2D acquisition, every slice, since each slice is excited in turn
after the delay. `protocols.readout_end` computes that minimum from a sidecar: the latest
signal time of any volume plus the last slice's offset. The reference protocol's 20 slices
40 ms apart put the last slice 0.76 s after the first, so its images occupy the interval
from 3.6 to 4.4 s after labeling begins, and its TR of 4.5 s holds them with a margin. Any
extra TR beyond that is wasted time: the tissue signal has long since reached its steady
state (recovery over 4.5 s at a $T_1$ of 1.33 s is 97 % complete), and the difference signal
does not depend on TR at all. Scan time is $2 \times \mathrm{pairs} \times \mathrm{TR}$.

```{code-cell} python
:tags: [hide-input]
p_ref = protocols.pcasl()
ctx_ref = protocols.aslcontext(p_ref)
end = protocols.readout_end(p_ref, ctx_ref)
fig, ax = plt.subplots(figsize=(8, 2.2))
blocks = [("labeling", 0, TAU, PALETTE[3]), ("delay", TAU, TAU + REF.post_labeling_delay, INK["grid"])]
blocks += [("2D readout", TAU + REF.post_labeling_delay + off, TAU + REF.post_labeling_delay + off + 0.03, PALETTE[0]) for off in protocols.slice_offsets(p_ref)]
sequence_diagram(ax, blocks, REF.repetition_time)
ax.axvline(REF.repetition_time, color=INK["primary"], lw=1, ls="--")
ax.text(REF.repetition_time - 0.05, 2.35, f"TR {REF.repetition_time:.1f} s", ha="right", fontsize=8)
ax.set_title("one repetition of the reference protocol: 20 slices, 40 ms apart")
fig.tight_layout()
print(f"reference protocol: last slice read at {end:.2f} s; TR {REF.repetition_time:.1f} s; "
      f"{REF.n_pairs} pairs take {2 * REF.n_pairs * REF.repetition_time / 60:.1f} min")
p_30 = protocols.pcasl(pld=3.0, n_pairs=15, tr=6.0)
print(f"pld-sweep at PLD 3.0 s: last slice read at {protocols.readout_end(p_30, protocols.aslcontext(p_30)):.2f} s, hence TR 6 s for the whole sweep")
```

The `pld-sweep` dataset used a TR of 6 s for all six delays so that its longest delay would
fit; at that TR its 15 pairs take 3 min per run. In practice the scanner sets the minimum TR
from the protocol and the operator rounds it up, which is what the reference protocol's
4.5 s is.

## Noise level and number of pairs

The difference image of one pair has a noise standard deviation of $\sqrt{2}\sigma$, where
$\sigma$ is the noise of one image, against a gray matter signal of 30 image units at the
reference timing. Averaging $N$ pairs reduces the noise by $\sqrt{N}$ and costs $2N$ TRs.
The `noise-sweep` dataset holds the reference protocol at three noise levels, and the
figure shows the 30-pair mean difference at each on the display slice; the reference's
σ of 40 is the middle column.

```{code-cell} python
:tags: [hide-input]
noise = data.load_dataset("noise-sweep")
k = phantom.DISPLAY_SLICE
fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.7))
diffs = {}
for ax, (name, sigma) in zip(axes, (("sigma10", 10), ("sigma40", 40), ("sigma80", 80))):
    run = noise.run(name)
    d = quant.subtract(run.mag(), run.context())
    diffs[name] = (d, run)
    gm_roi = run.fractions()["gm"] > 0.9
    m, sd = d.mean(-1), d.std(-1, ddof=1)
    show_image(ax, take_slice(m, k), f"σ = {sigma}: mean of 30 pairs", vmin=-10, vmax=50)
    print(f"{name}: GM ΔM {m[gm_roi].mean():.1f}, per-pair SD {sd[gm_roi].mean():.1f} (√2 σ = {np.sqrt(2) * sigma:.1f}), "
          f"per-pair SNR {m[gm_roi].mean() / sd[gm_roi].mean():.2f}, after 30 pairs about {m[gm_roi].mean() / sd[gm_roi].mean() * np.sqrt(30):.1f}")
fig.tight_layout()
```

The per-pair SD in gray matter is $\sqrt{2}\sigma$ to within a percent at every level, and
the per-pair SNR is 2.1, 0.54, and 0.27: at the reference noise a single pair does not show
the cortex at all, and 30 pairs bring the gray matter to an SNR near 3, which is the
grainy but legible middle image. At σ = 10 (a well-tuned coil at 3 T with a large voxel) the
same 30 pairs give an SNR near 12. The noise level is set by the hardware, the voxel, and
the readout; the protocol's own lever is $N$.

## Measure it: SNR against the number of pairs

The claim SNR $\propto \sqrt{N}$ can be tested on the σ = 40 run by averaging its first $N$
pairs and measuring the noise of that mean as the standard deviation, over the pure gray
matter voxels, of its deviation from the `deltam` truth (scaled to image units as in
[Chapter 5](./05-kinetic-model.md)). The truth removes the anatomy from the residual, so
what is left is noise plus the readout's small edge error.

```{code-cell} python
:tags: [hide-input]
d40, run40 = diffs["sigma40"]
ctx40 = run40.context()
lab40 = [i for i, r in enumerate(ctx40) if r == "label"]
gt40 = run40.truth("deltam")[..., lab40[0]] * SCALE
gm40 = run40.fractions()["gm"] > 0.9
ns = np.array([1, 2, 3, 5, 8, 12, 20, 30])
snr = []
for n in ns:
    m = d40[..., :n].mean(-1)
    snr.append(m[gm40].mean() / (m - gt40)[gm40].std())
snr = np.array(snr)
slope = (np.sqrt(ns) @ snr) / (np.sqrt(ns) @ np.sqrt(ns))   # least-squares SNR = slope * sqrt(N)
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.3))
n_fine = np.linspace(1, 32, 100)
axes[0].plot(n_fine, slope * np.sqrt(n_fine), color=INK["secondary"], lw=1, ls="--", label=f"{slope:.2f} √N")
axes[0].plot(ns, snr, "o", color=TISSUE_COLORS["GM"], label="measured, pure GM")
axes[0].set(xlabel="pairs averaged N", ylabel="SNR of the mean difference", title="σ = 40: SNR grows as √N")
axes[0].legend()
axes[1].plot(n_fine, 2 * n_fine * REF.repetition_time / 60, color=PALETTE[0])
axes[1].set(xlabel="pairs N", ylabel="scan time (min)", title=f"scan time at TR {REF.repetition_time:.1f} s")
fig.tight_layout()
for n, s in zip(ns, snr):
    print(f"N = {n:2d}: SNR {s:.2f} (fit {slope * np.sqrt(n):.2f}), {2 * n * REF.repetition_time / 60:.1f} min")
print(f"fitted per-pair SNR {slope:.3f}; expected from 30 / (√2 · 40) = {30 / (np.sqrt(2) * 40):.3f}")
```

The points follow the $\sqrt{N}$ line: the fitted per-pair SNR is 0.53, the value the
signal and noise levels predict, and 30 pairs reach an SNR near 3 in 4.5 min. Doubling the
SNR from there would take 120 pairs and 18 min, which is the reason ASL protocols run
4 to 6 min and the reason background suppression, larger voxels, and 3D readouts, which
raise the SNR per pair, are worth their complications.

## Voxel size

The signal of a voxel is proportional to the water it holds, so at a fixed receive chain
the SNR scales with the voxel volume: a 2.5 mm in-plane voxel has 0.51 times the volume of
a 3.5 mm one and half its SNR, and a 5 mm voxel has twice. The trade is partial volume:
a 3.5 × 3.5 × 5 mm voxel at the cortex mixes gray matter, white matter, and CSF, and its
measured CBF is the mixture's ([Chapter 12](../03-preprocessing/12-partial-volume.md)).
The white paper recommends 3 to 4 mm in-plane and 4 to 8 mm slices for this reason: ASL is
signal-starved, and the anatomy it resolves is coarse.

The `voxel-sweep` dataset holds the reference protocol at 2.5, 3.5, and 5 mm in-plane
voxels (matrices 80 × 96, 64 × 68, and 40 × 48, slices 5 mm throughout), with the image
noise scaled inversely with the voxel volume (σ 78, 40, and 20 image units), as thermal
noise per voxel behaves at a fixed receiver bandwidth. The signal per voxel is the same in
all three because the simulator's image units are per voxel, so the SNR alone follows the
volume. The figure shows the 15-pair mean difference of each run on the display slice.

```{code-cell} python
:tags: [hide-input]
vox = data.load_dataset("voxel-sweep")
fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.7))
for ax, name in zip(axes, ("vox25", "vox35", "vox50")):
    run = vox.run(name)
    p = run.sidecar()
    d = quant.subtract(run.mag(), run.context())
    fr = run.fractions()
    gm_roi = fr["gm"] > 0.9
    brain = (fr["gm"] + fr["wm"] + fr["csf"]) > 0.5
    mixed = np.mean(np.max([fr["gm"], fr["wm"], fr["csf"]], axis=0)[brain] < 0.8)
    m, sd = d.mean(-1), d.std(-1, ddof=1)
    size = p["AcquisitionVoxelSize"][0]
    sigma = np.sqrt(p["AslscanSimulation"]["Acquisition"]["NoiseVariance"])
    show_image(ax, take_slice(m, k), f"{size:g} mm in-plane ({d.shape[0]}×{d.shape[1]}), σ {sigma:.0f}", vmin=-10, vmax=50)
    print(f"{name}: voxel {size:g} x {size:g} x 5 mm ({size * size * 5:.0f} mm³), sigma {sigma:.0f}: GM ΔM {m[gm_roi].mean():.1f}, "
          f"per-pair SD {sd[gm_roi].mean():.1f}, SNR per pair {m[gm_roi].mean() / sd[gm_roi].mean():.2f}; mixed voxels {mixed:.0%} of the brain")
fig.tight_layout()
```

The gray matter difference signal is 30 image units in all three runs, and the per-pair SNR
goes 0.27, 0.54, 1.11: a factor of two per step, as the volume ratio (0.51 and 2.04)
predicts. The 2.5 mm image resolves the cortical ribbon in principle but loses it in the
grain; the 5 mm image shows a clean cortex that is blurred into its neighbors, and the
printed fraction of mixed voxels (no tissue above 80 %) rises from 28 % to 38 % of the brain
between them. The 3.5 mm voxel of the reference protocol is the compromise.
[Chapter 12](../03-preprocessing/12-partial-volume.md) uses the same runs to show what
the mixing does to the CBF values and how partial-volume correction recovers the
pure-tissue perfusion.

## Echo time

The labeled water in the voxel is blood, whose $T_2$ at 3 T is about 165 ms, while gray
matter's is 80 ms and white matter's 110 ms ([Chapter 1](../01-mri-physics/01-spins-and-relaxation.md)).
Between excitation and the echo the difference signal decays as $e^{-\mathrm{TE}/T_{2b}}$ and
the static tissue as $e^{-\mathrm{TE}/T_2}$. Two consequences: a short TE keeps more of both,
and the ratio of the difference to the tissue signal, which the calibration of
[Chapter 16](../04-quantification/16-calibration.md) relies on, drifts with TE unless the
two decays are accounted for. Spin-echo EPI at a 64 × 68 matrix reaches a TE near 12 ms
with partial Fourier and a short echo train; gradient-echo EPI, which is common for ASL,
runs 10 to 20 ms. The `te-sweep` dataset holds the reference protocol at 12, 30, and 60 ms.

```{code-cell} python
:tags: [hide-input]
tes = data.load_dataset("te-sweep")
te_ms, dm_te, ctrl_te = [], [], []
for name in ("te12", "te30", "te60"):
    run = tes.run(name)
    p, ctx = run.sidecar(), run.context()
    mag = run.mag()
    gm_roi = run.fractions()["gm"] > 0.9
    ctrl = mag[..., [i for i, r in enumerate(ctx) if r == "control"]].mean(-1)
    te_ms.append(p["EchoTime"] * 1000); dm_te.append(quant.subtract(mag, ctx).mean(-1)[gm_roi].mean()); ctrl_te.append(ctrl[gm_roi].mean())
te_ms, dm_te, ctrl_te = map(np.array, (te_ms, dm_te, ctrl_te))
te_fine = np.linspace(0, 70, 200)
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.3))
axes[0].plot(te_fine, dm_te[0] * np.exp(-(te_fine - 12) / 1000 / presets.T2_BLOOD), color=INK["secondary"], lw=1, ls="--", label="exp(−TE/T2 blood), T2 165 ms")
axes[0].plot(te_ms, dm_te, "o", color=TISSUE_COLORS["GM"], ms=7, label="measured GM ΔM")
axes[0].set(xlabel="TE (ms)", ylabel="control − label (image units)", ylim=(0, 35), title="the difference decays with the blood's T2")
axes[1].plot(te_fine, ctrl_te[0] * np.exp(-(te_fine - 12) / 1000 / GM.t2), color=INK["secondary"], lw=1, ls="--", label="exp(−TE/T2 GM), T2 80 ms")
axes[1].plot(te_ms, ctrl_te, "o", color=TISSUE_COLORS["GM"], ms=7, label="measured GM control")
axes[1].set(xlabel="TE (ms)", ylabel="control signal (image units)", ylim=(0, 7000), title="the tissue decays with its own T2")
for ax in axes:
    ax.legend(fontsize=8, loc="lower left")
fig.tight_layout()
for te, dm, c in zip(te_ms, dm_te, ctrl_te):
    print(f"TE {te:2.0f} ms: GM ΔM {dm:5.2f} ({dm / dm_te[0]:.3f} of TE 12; blood decay predicts {np.exp(-(te - 12) / 1000 / presets.T2_BLOOD):.3f}), "
          f"control {c:6.0f} ({c / ctrl_te[0]:.3f}; tissue decay predicts {np.exp(-(te - 12) / 1000 / GM.t2):.3f}), ΔM/control {dm / c * 100:.2f} %")
```

The measured points fall on the two predicted decays to three decimals: the difference
signal loses 25 % between 12 and 60 ms, the control 45 %. The ratio of the two, which is
what a calibration against the control image or an M0 scan at the same TE measures, rises
from 0.48 % to 0.66 % over the same range, a 36 % error in CBF if the $T_2$ mismatch is
ignored. Keep TE as short as the readout allows, and record it, because the correction
needs it.

## 2D or 3D readout

The simulator, and every pipeline dataset in this book, is a 2D multi-slice EPI: one slice
is excited and read at a time, 40 ms apart, so the 20th slice is read 0.76 s after the
first. In the kinetic model that is a different post-labeling delay for every slice, and the
figure shows what it does to the difference signal down the slab at the reference timing.

```{code-cell} python
:tags: [hide-input]
offsets = np.asarray(protocols.slice_offsets(p_ref))
fig, ax = plt.subplots(figsize=(7, 3.2))
for name in ("GM", "WM"):
    tt = T[name]
    per_slice = kinetic.delta_m(REF.post_labeling_delay + TAU + offsets, tt.perfusion, tt.att, tt.t1, tt.m0)
    ax.plot(np.arange(len(offsets)), per_slice / per_slice[0], "o-", color=TISSUE_COLORS[name], ms=4, label=name)
    print(f"{name}: the last slice reads {per_slice[-1] / per_slice[0]:.2f} of the first slice's difference signal (PLD {REF.post_labeling_delay + offsets[-1]:.2f} s instead of {REF.post_labeling_delay:.1f} s)")
ax.set(xlabel="slice (acquired in order, 40 ms apart)", ylabel="ΔM relative to the first slice", ylim=(0, 1.1),
       xticks=range(0, 20, 2), title="2D readout: every slice has its own post-labeling delay")
ax.legend()
fig.tight_layout()
```

The top slice has 56 % of the bottom slice's gray matter signal for the same perfusion.
That is not an error as long as the quantification uses each slice's own delay
([Chapter 14](../04-quantification/14-cbf-quantification.md)), but it is an SNR gradient
across the brain, and it means background suppression, which nulls the tissue at one moment,
cannot be optimal for every slice ([Chapter 9](../03-preprocessing/09-background-suppression.md)).

A 3D readout excites the whole slab and reads it in one shot or a few segments, so every
voxel shares one delay, the SNR is higher (the whole volume contributes to each readout),
and background suppression can null the tissue for the entire volume at once. The white
paper recommends 3D readouts for these reasons, with 3D GRASE
{cite:p}`gunther2005,fernandezseara2005` and stack-of-spirals as the usual choices; segmented 3D readouts
with background suppression were shown to give the best SNR per unit time in a systematic
comparison {cite:p}`vidorreta2013`. The costs are blurring along the slice direction, because
the long echo train decays with $T_2$ across the k-space partitions, and a longer TR when
the readout is segmented. The simulator does not model 3D readouts, so this book's images
are 2D throughout; where a result depends on the choice (slice timing, background suppression
efficiency), the chapter says so.

## Background suppression, the M0 scan, and field strength

**Background suppression** adds inversion pulses between labeling and readout, timed so
that the static tissue is near zero when the image is read while the label, inverted along
with it, keeps its magnitude. The difference signal is unchanged (apart from a few percent
lost per pulse), but the fluctuations that scale with the static signal, from motion and
physiology, are suppressed with it. The white paper recommends it for every ASL protocol.
Its cost is that the suppressed series cannot serve as its own M0 image and that a 2D readout
sees each slice at a different point of the tissue's recovery.
[Chapter 9](../03-preprocessing/09-background-suppression.md) simulates the pulses and
measures what they buy.

**The M0 scan** calibrates the difference signal to blood magnetization. It needs a long TR
(the reference uses 8 s, so that even CSF recovers), the same readout and TE as the ASL
series, and no background suppression. It takes a few seconds and it is the most commonly
omitted part of an ASL protocol; without it CBF can only be estimated from the control
images with a saturation correction ([Chapter 16](../04-quantification/16-calibration.md)).

**Field strength** enters twice. The equilibrium magnetization, and so the raw SNR, grows
about linearly with $B_0$, and the blood $T_1$ is longer at 3 T (1.65 s, {cite:p}`lu2004`)
than at 1.5 T (about 1.35 s, the white paper's value), so more label survives the transit
and the delay. The figure shows the gray matter curve with each $T_{1b}$.

```{code-cell} python
:tags: [hide-input]
t = np.linspace(0, 6, 601)
fig, ax = plt.subplots(figsize=(7, 3.2))
for t1b, field, color in ((1.35, "1.5 T", PALETTE[0]), (1.65, "3 T", PALETTE[1])):
    dm = kinetic.delta_m(t, GM.perfusion, GM.att, GM.t1, GM.m0, t1b=t1b) / GM.m0 * 100
    ax.plot(t, dm, color=color, label=f"{field}: T1 blood {t1b:.2f} s")
ax.axvline(TAU + REF.post_labeling_delay, color=INK["secondary"], lw=0.8, ls=":")
ax.set(xlabel="time since the start of labeling (s)", ylabel="ΔM / M0 (%)", xlim=(0, 6), title="the same gray matter labeled at two field strengths")
ax.legend()
fig.tight_layout()
r_t1 = dm_gm(1.8, t1b=1.65) / dm_gm(1.8, t1b=1.35)
print(f"GM at the reference timing: T1b 1.65 s gives {r_t1:.2f} x the difference signal of T1b 1.35 s; "
      f"with the 2 x higher magnetization at 3 T, about {2 * r_t1:.1f} x the SNR of 1.5 T")
```

At the reference timing the longer blood $T_1$ alone gives 1.1 times the signal, and with the
doubled magnetization, about 2.2 times the SNR of the same protocol at 1.5 T: an ASL
protocol that takes 5 min at 3 T would need about 25 min at 1.5 T for the same result. This, more than any other factor, is why ASL became routine only with 3 T scanners.
The costs of 3 T, larger susceptibility offsets and higher SAR from the labeling train,
are the subjects of [Chapter 11](../03-preprocessing/11-susceptibility-distortion.md) and of
the labeling-duration limits above.

## Worked example: a five-minute protocol for an older population

A study of adults over 70 at 3 T, with a 2D EPI readout as the simulator models it, wants
CBF maps in five minutes of scanning. The choices, in the order they constrain each other:

```{code-cell} python
:tags: [hide-input]
ld, pld, att_expected = 1.8, 2.0, 1.5             # white paper: PLD 2.0 s for the elderly; expected slow ATT
p_old = protocols.pcasl(pld=pld, ld=ld, n_pairs=1, tr=ld + pld + OVERHEAD)
tr = p_old["RepetitionTimePreparation"]
t_m0 = 2 * REF.m0_repetition_time                  # one M0 volume at TR 8 s plus a dummy repetition
pairs = int((300 - t_m0) // (2 * tr))
total = t_m0 + 2 * pairs * tr
dm_first = dm_gm(pld, att=att_expected, tau=ld)
dm_last = kinetic.delta_m(pld + ld + offsets[-1], GM.perfusion, att_expected, GM.t1, GM.m0, tau=ld) / GM.m0
dm_units = dm_first * GM.m0 * SCALE                 # fraction of M0 -> M0 units -> image units
snr_pair = dm_units / (np.sqrt(2) * 40)
print(f"LD {ld:.1f} s, PLD {pld:.1f} s, 20 slices at 40 ms -> last slice read at {protocols.readout_end(p_old, ['control', 'label']):.2f} s; TR {tr:.1f} s")
print(f"M0 scan at TR {REF.m0_repetition_time:.0f} s: {t_m0:.0f} s; pairs that fit in 5 min: {pairs} (2 x {pairs} x {tr:.1f} s = {2 * pairs * tr:.0f} s); total {total:.0f} s = {total / 60:.2f} min")
print(f"expected GM signal at ATT {att_expected:.1f} s: {dm_first:.2%} of M0 in the first slice ({dm_units:.0f} image units), {dm_last:.2%} in the last; "
      f"PLD {pld:.1f} s keeps {dm_first / dm_gm(att_expected, att=att_expected, tau=ld):.0%} of the signal at the peak (PLD = ATT)")
print(f"at sigma 40: SNR {snr_pair:.2f} per pair, {snr_pair * np.sqrt(pairs):.1f} after {pairs} pairs in the first slice, "
      f"{snr_pair * np.sqrt(pairs) * dm_last / dm_first:.1f} in the last")
print(f"margin against transit time: exact up to ATT {pld:.1f} s; at ATT 2.3 s the formula would report "
      f"{quant.cbf_pcasl(kinetic.delta_m(pld + ld, GM.perfusion, 2.3, t1_as_blood, GM.m0), GM.m0, pld, tau=ld) / GM.perfusion:.0%} of the true CBF (T1' = T1b)")
fig, ax = plt.subplots(figsize=(8, 1.8))
left = 0
for lab, dur, color in (("M0 scan", t_m0, PALETTE[2]), (f"{pairs} control-label pairs", 2 * pairs * tr, PALETTE[1])):
    ax.barh(0, dur / 60, left=left / 60, color=color, height=0.5, label=lab)
    left += dur
ax.axvline(5, color=INK["primary"], ls="--", lw=1)
ax.set(xlim=(0, 5.5), yticks=[], xlabel="scan time (min)", title=f"where the five minutes go (TR {tr:.1f} s)")
ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0)); ax.grid(False)
fig.tight_layout()
```

| Parameter | Choice | Why | Cost | Tested in |
|---|---|---|---|---|
| Labeling | PCASL, LD 1.8 s | efficiency 0.85 without a separate coil; 1.8 s captures 75 % of the plateau | SAR; 1.8 s of every TR | [Chapter 4](./04-labeling-schemes.md) |
| PLD | 2.0 s | longer than the 1.5 s transit time expected in this population, with margin to 2.0 s | 68 % of the signal a 1.5 s delay would give at ATT 1.5 s | this chapter; [Chapter 14](../04-quantification/14-cbf-quantification.md) |
| TR | 4.7 s | the minimum that holds LD + PLD + the 0.76 s readout, rounded up | none: the tissue is at steady state | this chapter |
| Pairs | 30 | what fits in 5 min after the M0 scan | SNR about 3 in gray matter at σ 40 (2 to 3.4 across the slab) | [Chapter 8](../03-preprocessing/08-noise.md) |
| Voxel | 3.5 × 3.5 × 5 mm | the white paper's range; SNR over resolution | cortex and ventricle walls are mixtures | [Chapter 12](../03-preprocessing/12-partial-volume.md) |
| TE | 12 ms, spin-echo EPI | shortest the readout allows; 7 % of the label lost | none | [Chapter 16](../04-quantification/16-calibration.md) |
| Readout | 2D EPI, 20 slices, 40 ms apart | what the simulator models; 3D GRASE preferred on a scanner | per-slice delay; top slice at 56 % of the bottom's signal | [Chapter 14](../04-quantification/14-cbf-quantification.md) |
| Background suppression | on, two pulses | the white paper's recommendation; suppresses motion and physiological noise | the series cannot calibrate itself | [Chapter 9](../03-preprocessing/09-background-suppression.md) |
| M0 scan | separate, TR 8 s, same readout | required for absolute CBF | 16 s | [Chapter 16](../04-quantification/16-calibration.md) |

The protocol takes 298 s, just under the five minutes, and the expected gray matter SNR
after 30 pairs is about 3.4 in the first slice and 1.9 in the last; that is enough for a
regional CBF map at this voxel size, not for a voxelwise one, which is why a study wanting the latter would add
background suppression and a 3D readout rather than more pairs.

## What this implies for acquisition

- **Compute before scanning.** The difference signal from the kinetic model at the chosen
  delay and transit time, the TR from LD + PLD + readout, the pairs from the time budget,
  and the SNR from σ, the voxel, and $\sqrt{N}$: each is one line.
- **Set the delay by the population, not by the SNR curve.** The SNR optimum is at the
  transit time; the safe choice is beyond the longest transit time you expect, at a known
  cost in signal.
- **Spend TR on labeling rather than on margin.** Longer labeling raises the signal even
  after paying for the longer TR; TR beyond the readout's need buys nothing.
- **Keep TE short and record it**, and keep the same TE and readout for the M0 scan.
- **Prefer larger voxels, 3D readouts, and background suppression to more pairs**: the
  first three raise the SNR per pair, the last only its square root.
- **Always acquire the M0 scan.** It costs seconds and without it CBF is not absolute.

## Further reading

The white paper's protocol recommendations {cite:p}`alsop2015` and their update for
multi-timepoint acquisitions {cite:p}`woods2024`; the readout comparison of
{cite:t}`vidorreta2013`; 3D GRASE for ASL {cite:p}`gunther2005,fernandezseara2005`; the
blood $T_1$ at 3 T {cite:p}`lu2004`; the PCASL labeling scheme and its efficiency
{cite:p}`dai2008`.
