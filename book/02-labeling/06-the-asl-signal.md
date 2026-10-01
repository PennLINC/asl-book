---
title: "6. The ASL signal: control, label, difference, M0"
kernelspec:
  name: python3
  display_name: Python 3
---

:::{admonition} Simulated datasets in this chapter
:class: note
- **Built in this page:** the reference protocol simulated on the packaged slab by the toy tier, with and without noise ([Appendix B](../appendices/b-data-manifest.md#app-b-package-data)).
- **`ref-pcasl`**: the reference 2D PCASL series from the simulator, 30 pairs with a separate M0 scan and the standard noise ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-ref-pcasl)).
- **`ref-clean`**: the same series with the noise switched off ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-ref-clean)).

Pipeline-tier datasets are simulated offline by aslscan ([Chapter 0.2](../00-frontmatter/the-simulated-datasets.md)).
:::

## Learning goals

After this chapter you can:

- name the volumes of an ASL series, say what each contains, and read them from a BIDS
  dataset
- explain why the perfusion signal is a difference of two nearly identical images, and how
  large that difference is relative to the images and to the noise
- state how averaging pairs improves the difference, and measure its temporal SNR
- explain why each slice of a 2D readout has its own post-labeling delay, and see it in the data
- say what the M0 scan is for

```{code-cell} python
:tags: [hide-cell]
import numpy as np
import matplotlib.pyplot as plt

from aslbook import data, kinetic, phantom, presets, protocols, quant, synth
from aslbook.plotting import INK, PALETTE, TISSUE_COLORS, set_style, show_image, show_slice, take_slice

set_style()
ph = phantom.slab()
fr = phantom.fractions(ph)
K = phantom.DISPLAY_SLICE
GM, WM = presets.TISSUES["GM"], presets.TISSUES["WM"]
r = presets.REFERENCE
```

## The volumes of a series

An ASL acquisition alternates two kinds of image. Before a **label** image, the arterial
blood below the brain is inverted ([Chapter 4](./04-labeling-schemes.md)), and after a delay
the inverted water has reached the capillaries and exchanged into the tissue, where it
lowers the longitudinal magnetization slightly. Before a **control** image nothing is
inverted, but everything else is the same: the same radio-frequency power, the same timing,
the same readout. The two images are therefore identical except for the labeled water, and
their difference is the perfusion signal. Because a single difference is noisy, the pair is
repeated many times, and because the difference is a magnetization in unknown units, an
**M0 scan** with no labeling and a long repetition time is acquired to normalize it.

In BIDS {cite:p}`clement2022` the series is one 4-D file, and the table `aslcontext.tsv`
names each volume. The sidecar records the protocol. Here is the reference protocol as the
simulator read it:

```{code-cell} python
:tags: [hide-input]
p = protocols.pcasl(n_pairs=r.n_pairs)
ctx = protocols.aslcontext(p)
for key in ("ArterialSpinLabelingType", "LabelingDuration", "PostLabelingDelay", "RepetitionTimePreparation", "EchoTime",
            "AcquisitionVoxelSize", "M0Type", "BackgroundSuppression", "TotalAcquiredPairs"):
    print(f"{key:<26} {p[key]}")
print(f"{'SliceTiming':<26} [{p['SliceTiming'][0]}, {p['SliceTiming'][1]}, ..., {p['SliceTiming'][-1]}] ({len(p['SliceTiming'])} slices)")
print(f"{'aslcontext.tsv':<26} {ctx[0]}, {ctx[1]}, {ctx[2]}, {ctx[3]}, ... ({len(ctx)} volumes)")
```

The delays are in seconds and the voxel size in mm. `M0Type: Separate` says the calibration
image is a scan of its own with its own sidecar; other datasets include `m0scan` rows in
the series instead ([Chapter 16](../04-quantification/16-calibration.md)).

## What each image contains

The control image is the ordinary spin-echo EPI image of the brain: the tissue's
longitudinal magnetization at the moment of excitation, which after many repetitions has
settled at the saturation-recovery steady state of [Chapter 1](../01-mri-physics/01-spins-and-relaxation.md),

$$M_z^{\mathrm{tissue}} = M_0 \left(1 - e^{-\mathrm{TR}/T_1}\right),$$

times the transverse decay $e^{-\mathrm{TE}/T_2}$ to the echo time. The label image
contains the same tissue signal minus the difference the labeled blood delivered, which
[Chapter 5](./05-kinetic-model.md) gives as $\Delta M(t)$ from the kinetic model, times the
transverse decay of *blood*, $e^{-\mathrm{TE}/T_{2b}}$, because the labeled water carries
the relaxation of the compartment it is in during the readout. The simulator keeps these
as separate compartments; the toy tier does the same in one line per tissue class.

```{code-cell} python
:tags: [hide-input]
t_read = r.labeling_duration + r.post_labeling_delay
print(f"{'':>6} {'M0':>6} {'saturation':>11} {'T2 decay':>9} {'control':>8} {'dM (M0 units)':>14} {'dM x T2b':>9} {'dM/control':>11}")
for t in (GM, WM):
    sat = 1 - np.exp(-r.repetition_time / t.t1)
    e2 = np.exp(-r.echo_time / t.t2)
    control = r.signal_scale * t.m0 * sat * e2
    dm = kinetic.delta_m(t_read, t.perfusion, t.att, t.t1, t.m0, tau=r.labeling_duration)
    diff = r.signal_scale * dm * np.exp(-r.echo_time / presets.T2_BLOOD)
    print(f"{t.name:>6} {t.m0:6.1f} {sat:11.3f} {e2:9.3f} {control:8.0f} {dm:14.3f} {diff:9.1f} {100 * diff / control:10.2f} %")
```

Read the last column: the difference is about half a percent of the control image in gray
matter and a fifth of a percent in white matter. Everything about ASL follows from that
number. A subtraction of two images, each with its own noise, has to resolve a change of
one part in two hundred; anything that changes the control or label image by more than
that, such as head motion or a physiological fluctuation, is larger than the signal
([Chapters 9](../03-preprocessing/09-background-suppression.md) and
[10](../03-preprocessing/10-motion.md)).

## See it: one pair, and thirty

The toy tier simulates the reference protocol on the packaged slab: the tissue and blood
compartments per class from the tissue fractions, with noise of standard deviation 40 in
each of the real and imaginary channels, the level of the pipeline datasets.

```{code-cell} python
:tags: [hide-input]
s = synth.series(p, ctx, noise_sd=40.0, seed=1)
s0 = synth.series(p, ctx, noise_sd=0.0)
d = quant.subtract(s.mag, s.ctx)
d0 = quant.subtract(s0.mag, s0.ctx)
gm = fr["GM"] > 0.9
fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.3))
show_slice(axes[0], s.mag[..., 0], K, "control (volume 0)", vmin=0, vmax=7000)
show_slice(axes[1], s.mag[..., 1], K, "label (volume 1)", vmin=0, vmax=7000)
show_slice(axes[2], d[..., 0], K, "control − label, one pair", vmin=-150, vmax=150, cmap="RdBu_r")
show_slice(axes[3], d.mean(-1), K, "mean of 30 pairs", vmin=-40, vmax=40, cmap="RdBu_r")
fig.tight_layout()
print(f"gray matter: control {s.mag[gm][:, 0].mean():.0f}, label {s.mag[gm][:, 1].mean():.0f}, "
      f"one pair's difference {d[gm][:, 0].mean():.1f} ± {d[gm][:, 0].std():.1f} (mean ± SD over GM voxels), mean of 30 pairs {d.mean(-1)[gm].mean():.1f} ± {d.mean(-1)[gm].std():.1f}")
```

The control and label images are indistinguishable by eye, as they should be: the
difference is a percent of the window. A single pair's difference is dominated by noise;
the perfusion signal is in it (the mean over gray matter is right) but no single voxel shows
it. After thirty pairs the gray matter ribbon appears, with the noise reduced by the square
root of thirty.

## Averaging pairs

Each pair's difference has noise with standard deviation $\sqrt{2}\,\sigma$ in magnitude
images at this SNR, where $\sigma$ is the per-channel noise, because two independent noisy
images are subtracted. The mean of $N$ pairs has $\sqrt{2}\sigma / \sqrt{N}$. The
signal-to-noise ratio of the mean difference in gray matter therefore grows as $\sqrt{N}$,
and the scan time as $N$: doubling the SNR costs four times the scan.

```{code-cell} python
:tags: [hide-input]
ns = np.array([1, 2, 4, 8, 15, 30])
snr = []
for n in ns:
    m = d[..., :n].mean(-1)
    snr.append(m[gm].mean() / (m[gm] - d0.mean(-1)[gm]).std())
snr = np.array(snr)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.3))
ax1.plot(ns, snr, "o", color=PALETTE[1], label="measured in gray matter")
ax1.plot(ns, snr[0] * np.sqrt(ns), color=INK["secondary"], lw=1, ls="--", label="√N from one pair")
ax1.set(xlabel="pairs averaged", ylabel="SNR of the mean difference", xscale="log", yscale="log")
ax1.set_xticks(ns, [str(n) for n in ns]); ax1.set_yticks([0.5, 1, 2, 4], ["0.5", "1", "2", "4"])
ax1.legend()
show_image(ax2, take_slice(quant.tsnr(d), K), "temporal SNR (mean / SD over the 30 pairs)", kind="scalar", vmin=0, vmax=1, colorbar=True)
fig.tight_layout()
print("SNR of the mean difference in gray matter: " + ", ".join(f"{n} pairs {v:.2f}" for n, v in zip(ns, snr)))
```

The points follow the dashed square-root law. One pair gives an SNR near 0.5 in gray matter;
thirty give about 3, which is the regime of a typical clinical ASL scan and the reason ASL
maps are averaged over regions rather than read voxel by voxel. The right panel is the
temporal SNR, the mean over pairs divided by the standard deviation over pairs in each
voxel: below 1 nearly everywhere, highest in the deep gray matter where the voxels are
purest.

## Slice timing: every slice has its own delay

A 2D readout excites the slices one after another, here 40 ms apart, so the last of the
twenty slices is read 0.76 s after the first. The label goes on decaying with the tissue's
T1 during that time ([Chapter 5](./05-kinetic-model.md)), so the difference in the last
slice is smaller than in the first at the same nominal delay. The simulator evaluates the
kinetic model at each slice's actual readout time, `PostLabelingDelay + SliceTiming[z]`, and
so does the toy tier.

```{code-cell} python
:tags: [hide-input]
z = np.arange(r.n_slices)
offsets = np.array(protocols.slice_offsets(p))
gm_z = np.array([d0.mean(-1)[:, :, k][fr["GM"][:, :, k] > 0.9].mean() if (fr["GM"][:, :, k] > 0.9).sum() > 20 else np.nan for k in z])
model = r.signal_scale * np.exp(-r.echo_time / presets.T2_BLOOD) * kinetic.delta_m(t_read + offsets, GM.perfusion, GM.att, GM.t1, GM.m0, tau=r.labeling_duration)
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.plot(z, gm_z, "o", color=TISSUE_COLORS["GM"], label="noise-free toy series, pure gray matter")
ax.plot(z, model, color=INK["secondary"], lw=1.2, ls="--", label="kinetic model at PLD + slice offset")
ax.set(xlabel="slice (acquired in order, 40 ms apart)", ylabel="control − label (image units)", ylim=(0, 35))
ax.legend(loc="lower left")
fig.tight_layout()
print(f"gray matter difference: slice 0 {model[0]:.1f}, slice 19 {model[-1]:.1f} ({100 * (1 - model[-1] / model[0]):.0f} % lower)")
```

The measured points sit on the model line, and the last slice's difference is more than
40 % lower than the first's. A quantification that uses one delay for every slice would
underestimate perfusion in the upper slices by that much; [Chapter 14](../04-quantification/14-cbf-quantification.md)
corrects it slice by slice. A 3D readout acquires the whole volume at one delay and has no
such gradient ([Chapter 7](./07-acquisition-parameters.md)).

## The M0 scan

The difference is a magnetization in the scanner's arbitrary units. To turn it into a flow,
the kinetic model divides it by the equilibrium magnetization of arterial blood, which is
the tissue's equilibrium magnetization $M_0$ over the partition coefficient
([Chapter 14](../04-quantification/14-cbf-quantification.md)). The control image is not
$M_0$: it is saturated by the repetition time (about 3 % in gray matter at TR 4.5 s, 22 % in
CSF) and decayed by the echo time. The M0 scan is the same readout with a long repetition
time, 8 s here, so that the tissue has recovered almost fully; it is acquired once and
carries the same T2 decay as the controls, which the calibration of
[Chapter 16](../04-quantification/16-calibration.md) accounts for.

```{code-cell} python
:tags: [hide-input]
fig, axes = plt.subplots(1, 3, figsize=(10, 3.3))
show_slice(axes[0], s.mag[..., ::2].mean(-1), K, "mean control (TR 4.5 s)", vmin=0, vmax=7000)
show_slice(axes[1], s.m0scan, K, "M0 scan (TR 8 s)", vmin=0, vmax=7000)
with np.errstate(divide="ignore", invalid="ignore"):
    ratio = np.where(ph["mask"], s0.m0scan / s0.mag[..., ::2].mean(-1), np.nan)
show_image(axes[2], take_slice(ratio, K), "M0 scan / mean control (noise-free)", kind="scalar", vmin=1.0, vmax=1.3, colorbar=True)
fig.tight_layout()
for name, t in presets.TISSUES.items():
    print(f"{name}: M0 scan / control = {(1 - np.exp(-r.m0_repetition_time / t.t1)) / (1 - np.exp(-r.repetition_time / t.t1)):.3f}")
```

The two images look alike; the ratio map shows where they differ: hardly at all in white
matter, whose T1 is short, by 3 % in gray matter, and by 20 % in the ventricles, where CSF
with its 3 s T1 is far from recovered at TR 4.5 s. Using the control image as M0 would
overestimate perfusion by those factors.

## See it: the simulated series

The pipeline dataset is the same protocol acquired by the simulator's scanner model:
k-space, T2* decay along the readout, the truncation ringing of a 64 × 68 matrix, and noise
added in k-space.

```{code-cell} python
:tags: [hide-input]
run = data.load_dataset("ref-pcasl").run()
mag, rctx, sc = run.mag(), run.context(), run.sidecar()
m0 = run.m0scan()
frac = run.fractions()
gm_r = frac["gm"] > 0.9
dr = quant.subtract(mag, rctx)
print(f"series {mag.shape}; {rctx.count('control')} control and {rctx.count('label')} label volumes; M0 scan {m0.shape}")
print(f"sidecar: {sc['ArterialSpinLabelingType']}, LD {sc['LabelingDuration']} s, PLD {sc['PostLabelingDelay']} s, TR {sc['RepetitionTimePreparation']} s, "
      f"TE {1000 * sc['EchoTime']:.0f} ms, matrix {sc['AslscanSimulation']['Grid']['AcquisitionMatrix']}, simulated on {sc['AslscanSimulation']['Grid']['SimulationMatrix']}")
fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.3))
show_slice(axes[0], mag[..., 0], K, "control (volume 0)", vmin=0, vmax=7000)
show_slice(axes[1], m0, K, "M0 scan", vmin=0, vmax=7000)
show_slice(axes[2], dr[..., 0], K, "control − label, one pair", vmin=-150, vmax=150, cmap="RdBu_r")
show_slice(axes[3], dr.mean(-1), K, "mean of 30 pairs", vmin=-40, vmax=40, cmap="RdBu_r")
fig.tight_layout()
```

Compared with the toy images above, the control image has the faint ringing of a real EPI
image at every sharp edge, and the difference images look the same: the perfusion signal
and the noise are what they were, because the acquisition changes the images but not the
label. The M0 scan is brighter in the ventricles, as the ratio map predicted.

## Measure it: signal, noise, and the noise-free reference

The noise level can be read two ways: from the voxels outside the head, where the magnitude
of pure complex noise has the Rayleigh mean $\sigma\sqrt{\pi/2}$
({cite:p}`gudbjartsson1995`; [Chapter 8](../03-preprocessing/08-noise.md)), or from the standard deviation over time of
a voxel whose true signal does not change, such as a white matter voxel across the thirty
control volumes. The perfusion signal comes from the noise-free run of the same protocol.

```{code-cell} python
:tags: [hide-input]
clean = data.load_dataset("ref-clean").run()
dc = quant.subtract(clean.mag(), clean.context()).mean(-1)
from scipy import ndimage
empty = ~ndimage.binary_dilation(run.mask(), iterations=5)  # well away from the head
sigma_bg = mag[empty].mean() / np.sqrt(np.pi / 2)
wm_r = frac["wm"] > 0.9
controls = [i for i, v in enumerate(rctx) if v == "control"]
sigma = mag[wm_r][:, controls].std(axis=1, ddof=1).mean()
one = dr[gm_r][:, 0]
mean30 = dr.mean(-1)
print(f"noise sigma: {sigma:.1f} from the temporal SD of white matter across the controls; {sigma_bg:.1f} from the background's Rayleigh mean "
      f"(configured sigma {sc['AslscanSimulation']['Acquisition']['NoiseVariance'] ** 0.5:.0f}; the background estimate is lifted by the faint truncation sidelobes of the head, which are not noise)")
print(f"gray matter control {mag[gm_r][:, 0].mean():.0f} (SNR {mag[gm_r][:, 0].mean() / sigma:.0f}); one pair's difference {one.mean():.1f} ± {one.std():.1f} (SNR {one.mean() / one.std():.2f})")
print(f"mean of 30 pairs: {mean30[gm_r].mean():.1f}; noise-free difference {dc[gm_r].mean():.1f}; residual SD {(mean30 - dc)[gm_r].std():.1f} (SNR {mean30[gm_r].mean() / (mean30 - dc)[gm_r].std():.1f})")
dm_gt = clean.truth("deltam")
labels = [i for i, v in enumerate(clean.context()) if v == "label"]
gt = dm_gt[..., labels].mean(-1) * r.signal_scale * np.exp(-r.echo_time / presets.T2_BLOOD)
ratio = dc[gm_r] / gt[gm_r]
print(f"noise-free difference / (100 x exp(-TE/T2b) x deltam truth) in gray matter: {np.median(ratio):.3f} (the readout's T2* decay and ringing account for the rest)")
```

The numbers match the toy tier: a σ of 40 from the temporal spread of the controls, exactly
as configured (the background estimate comes out a little higher because the empty voxels
are not quite empty: the truncated k-space of the head leaves faint sidelobes there, which
lift the Rayleigh mean; [Chapter 2](../01-mri-physics/02-epi-and-reconstruction.md) shows
this), a gray matter control SNR above 150, a single-pair SNR near 0.5, and a 30-pair SNR
near 3. The last line
is the bridge between the tiers and the truth: the noise-free difference from the
simulator's scanner is within a percent of the kinetic model's difference scaled by the
blood's T2 decay, which is what the toy tier computes directly. The chapters that follow use
either tier as the point requires, and score both against the same truth.

## What this implies for acquisition

- **The signal is a half-percent difference.** Every choice is an SNR choice. Keep the echo
  time short, the voxels large, and the pairs many; the white paper's 3 to 4 mm in-plane
  voxels and 4 to 8 mm slices, averaged for about four minutes {cite:p}`alsop2015`, are a
  compromise, not a limit.
- **Scan time buys SNR slowly.** Four times the pairs for twice the SNR. A 5-minute scan
  gives voxelwise SNR near 3 in gray matter; regional averages are what ASL reports reliably.
- **Acquire an M0 scan.** The control image is saturated by the repetition time, most in
  CSF; a long-TR M0 scan removes the guesswork from the calibration.
- **2D readouts spread the delay across slices.** Record the slice timing in the sidecar
  and correct for it in quantification, or use a 3D readout.
- **Keep the pairs interleaved and the order recorded.** `aslcontext.tsv` is the record of
  which volume is which; a pipeline that guesses the order gets the sign of the signal
  wrong.

## Further reading

The white paper's description of the standard acquisition {cite:p}`alsop2015`, the ASL
extension of BIDS that names the volumes and the sidecar fields {cite:p}`clement2022`, and
the kinetic model behind the difference {cite:p}`buxton1998`.
