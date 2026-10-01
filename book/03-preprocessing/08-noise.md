---
title: "8. Thermal noise, averaging, and denoising"
kernelspec:
  name: python3
  display_name: Python 3
---

:::{admonition} Simulated datasets in this chapter
:class: note
- **Built in this page:** a cloud of noisy complex measurements and its magnitude histogram; a 60-pair synthetic series on the packaged phantom slab (`aslbook.synth`) for the averaging law; an MP-PCA denoiser in plain numpy and the random noise matrices that set the threshold of its NORDIC-style variant ([Appendix B](../appendices/b-data-manifest.md#app-b-package-data)).
- **`ref-pcasl`**: the reference protocol with the standard noise, σ = 40 image units; the series that is averaged, smoothed and denoised ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-ref-pcasl)).
- **`ref-clean`**: the same acquisition with the noise switched off: the noise-free reference every error below is measured against ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-ref-clean)).
- **`noise-sweep`**: the reference protocol at σ = 10, 40 and 80, plus a run with eight receive coils and GRAPPA 2; all four are denoised, the σ = 80 and GRAPPA runs also as complex series rebuilt from their magnitude and phase images ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-noise-sweep)).

Pipeline-tier datasets are simulated offline by aslscan ([Chapter 0.2](../00-frontmatter/the-simulated-datasets.md)).
:::

## Learning goals

After this chapter you can:

- explain why magnitude noise is Rician, why that matters in the background and not in an ASL control image, and what it does to the control-label difference
- measure the noise level of an ASL series two ways and say when each is trustworthy
- state the averaging law and turn it into a CBF noise floor for a given number of pairs
- recognize the noise of a multi-coil, parallel-imaging acquisition, which is neither uniform across the image nor independent between voxels
- apply spatial smoothing and local PCA denoising (MP-PCA, and a NORDIC-style variant on complex data) to an ASL series, and measure what each removes and what it costs in bias and resolution

```{code-cell} python
:tags: [hide-cell]
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, special

from aslbook import data, phantom, presets, protocols, quant, synth
from aslbook.plotting import INK, PALETTE, TISSUE_COLORS, fit_vs_truth, set_style, show_image, show_slice, take_slice

set_style()
K = phantom.DISPLAY_SLICE
TE, T2B, SCALE = presets.REFERENCE.echo_time, presets.T2_BLOOD, presets.REFERENCE.signal_scale


def rician_mean(a, sigma):
    """Mean of the magnitude of a complex value ``a`` plus Gaussian noise of SD ``sigma`` per component."""
    x = np.asarray(a, float) ** 2 / (4 * sigma**2)
    return sigma * np.sqrt(np.pi / 2) * ((1 + 2 * x) * special.ive(0, x) + 2 * x * special.ive(1, x))


def truth_deltam(run, pairs=None):
    """The noise-free control-label difference in image units: the ``deltam`` truth of the label
    rows, scaled like the images and decayed with the blood T2 to the echo time."""
    ctx = run.context()
    rows = [i for i, r in enumerate(ctx) if r == "label"]
    rows = rows if pairs is None else rows[:pairs]
    return run.truth("deltam")[..., rows].mean(axis=-1) * SCALE * np.exp(-TE / T2B)


def control_mean(mag, ctx):
    return mag[..., [i for i, r in enumerate(ctx) if r == "control"]].mean(axis=-1)


def side_colorbar(fig, im, label, right=0.9):
    """A colorbar in its own axis at the right edge, so that the image panels keep one size."""
    fig.tight_layout(rect=(0, 0, right, 1))
    cax = fig.add_axes([right + 0.01, 0.2, 0.012, 0.6])
    fig.colorbar(im, cax=cax, label=label)
```

## The physics: Gaussian in k-space, Rician in the image

Thermal noise from the body and the receive chain enters every k-space sample as
independent, zero-mean Gaussian noise. The Fourier transform is linear, so the complex image
carries the same kind of noise: every voxel's real and imaginary parts are its true value plus
a Gaussian draw of standard deviation σ. The scanner then discards the phase and keeps the
magnitude, the distance of the measured point from the origin of the complex plane, and that
step is not linear. The figure shows it for two voxels measured 3000 times each: one with no
true signal and one with a true signal of 3 σ. On the left are the clouds of measurements;
in the middle, the histograms of their magnitudes; on the right, the mean magnitude as a
function of the true signal.

```{code-cell} python
:tags: [hide-input]
rng = np.random.default_rng(0)
n, sigma = 3000, 1.0
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
for a, color, label in [(0.0, PALETTE[3], "no signal"), (3.0, PALETTE[0], "true signal 3 σ")]:
    z = a + sigma * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    axes[0].scatter(z.real, z.imag, s=3, alpha=0.25, color=color, rasterized=True)
    axes[0].plot([a], [0], "o", color=color, mec=INK["primary"], ms=8, label=f"{label}: true value")
    axes[1].hist(np.abs(z), bins=40, density=True, color=color, alpha=0.5, label=f"{label}: mean {np.abs(z).mean():.2f} σ")
axes[0].axhline(0, color=INK["secondary"], lw=0.8)
axes[0].axvline(0, color=INK["secondary"], lw=0.8)
axes[0].set(xlim=(-4, 7), ylim=(-4, 4), xlabel="real part (units of σ)", ylabel="imaginary part (units of σ)", title="two voxels, 3000 measurements each")
axes[0].set_aspect("equal")
axes[0].legend(loc="upper left")
m = np.linspace(0, 7, 300)
axes[1].plot(m, m * np.exp(-m**2 / 2), color="0.3", lw=1.5, label="Rayleigh distribution")
axes[1].axvline(np.sqrt(np.pi / 2), color=INK["secondary"], ls=":", lw=1)
axes[1].set(xlabel="magnitude (units of σ)", ylabel="density", title="the magnitudes")
axes[1].legend()
a_axis = np.linspace(0, 6, 200)
axes[2].plot(a_axis, a_axis, "--", color="0.5", lw=1, label="identity: no bias")
axes[2].plot(a_axis, rician_mean(a_axis, 1.0), color=PALETTE[1], label="mean magnitude")
axes[2].axhline(np.sqrt(np.pi / 2), color=INK["secondary"], ls=":", lw=1, label="noise floor, 1.25 σ")
axes[2].set(xlabel="true signal (units of σ)", ylabel="mean magnitude (units of σ)", title="mean magnitude vs. true signal", xlim=(0, 6), ylim=(0, 6.3))
axes[2].set_aspect("equal")
axes[2].legend(loc="upper left")
fig.tight_layout()
for a in [0, 1, 3, 10]:
    print(f"true signal {a:>2} σ: mean magnitude {rician_mean(a, 1.0):.3f} σ")
```

The cloud with no signal sits on the origin, and every point in it has a positive distance
from the origin, so its magnitudes average to 1.25 σ rather than zero. The cloud at 3 σ is
already far enough out that its magnitude is nearly the distance along the real axis, and
the mean is 3.16 σ, a bias of 5 %. At 10 σ the bias is half a percent. The magnitude of a
complex Gaussian measurement follows the **Rician** distribution {cite:p}`gudbjartsson1995`,
which is the **Rayleigh** distribution (mean 1.25 σ, standard deviation 0.66 σ;
{cite:t}`henkelman1985`) where there
is no signal and is indistinguishable from a Gaussian of standard deviation σ once the
signal exceeds a few σ. The magnitude operation therefore folds noise into a positive bias
where the signal is weak and leaves it alone where the signal is strong. The bias is a
known function of the true signal and σ (the curve in the right panel), and it can be
inverted: {cite:t}`koay2006` give an exact scheme that recovers the true signal and σ from
the mean and variance of magnitude measurements.

**ASL images are strong-signal images.** The reference protocol's gray matter control voxel
is about 6200 image units and σ is 40 ([Chapter 6](../02-labeling/06-the-asl-signal.md)):
an SNR of about 150, where the Rician bias is 0.13 units, one part in fifty thousand. The
magnitude noise in every tissue voxel is Gaussian with standard deviation σ, and the
Rayleigh floor lives only outside the head. What ASL has to fight is the size of its signal
of interest:

$$
\Delta M = M_\mathrm{control} - M_\mathrm{label}
$$

is the difference of two independent measurements, each with noise σ, so **one pair's
difference has noise $\sqrt{2}\,\sigma \approx 57$ units** around a true value of about 30
in gray matter. The SNR of one pair is about 0.5: the perfusion signal is smaller than its
noise. Averaging $N$ pairs divides the noise by $\sqrt{N}$, so the mean of 30 pairs has an
SNR of about 2.7, and an SNR of 10 in a single 3.5 mm voxel would take 400 pairs, or half
an hour. This is why every ASL protocol is a long average and why voxels are large.

## See it: the averaging law on the toy series

The synthetic series below evaluates the signal model on the packaged slab for 60 pairs of
the reference protocol with σ = 40 and no readout, so its noise-free difference is known
exactly. The left panel is the histogram of the noise of one pair's difference in the pure
gray matter voxels (the measured difference minus the true one); the right panel is the
SNR of the mean difference as a function of the number of pairs averaged, with the
$\sqrt{N}$ law drawn through the single-pair value.

```{code-cell} python
:tags: [hide-input]
p60 = protocols.pcasl(n_pairs=60)
s = synth.series(p60, noise_sd=40, seed=0)
ph = phantom.slab()
gm_toy = ph["gm"] >= 0.9
d_toy = quant.subtract(s.mag, s.ctx)
label_rows = [i for i, r in enumerate(s.ctx) if r == "label"]
true_toy = s.deltam[..., label_rows[0]] * SCALE * np.exp(-TE / T2B)  # every pair has the same truth
resid1 = (d_toy[..., 0] - true_toy)[gm_toy]
snr_pair = true_toy[gm_toy].mean() / resid1.std()
ns = np.array([1, 2, 3, 5, 10, 15, 20, 30, 40, 60])
snr_n = [true_toy[gm_toy].mean() / (d_toy[..., :n].mean(axis=-1) - true_toy)[gm_toy].std() for n in ns]

fig, (ax_h, ax_n) = plt.subplots(1, 2, figsize=(10, 3.4))
x = np.linspace(-250, 250, 300)
sd = np.sqrt(2) * 40
ax_h.hist(resid1, bins=60, density=True, color=TISSUE_COLORS["GM"], alpha=0.6, label=f"one pair, GM: SD {resid1.std():.1f}")
ax_h.plot(x, np.exp(-x**2 / (2 * sd**2)) / (sd * np.sqrt(2 * np.pi)), color="0.3", lw=1.5, label="Gaussian, SD √2 σ = 56.6")
ax_h.axvline(0, color=INK["secondary"], lw=0.8)
ax_h.set(xlabel="measured − true difference (image units)", ylabel="density", title="noise of one pair's difference")
ax_h.legend()
ax_n.plot(ns, snr_pair * np.sqrt(ns), "--", color="0.5", lw=1, label="√N law")
ax_n.plot(ns, snr_n, "o", color=PALETTE[0], label="measured, GM")
ax_n.set(xlabel="pairs averaged", ylabel="SNR of the mean difference", title="averaging", xscale="log", yscale="log")
ax_n.set_xticks([1, 2, 5, 10, 20, 60], ["1", "2", "5", "10", "20", "60"])
ax_n.set_yticks([0.5, 1, 2, 4], ["0.5", "1", "2", "4"])
ax_n.xaxis.set_minor_formatter(plt.NullFormatter())
ax_n.yaxis.set_minor_formatter(plt.NullFormatter())
ax_n.legend(loc="upper left")
fig.tight_layout()
print(f"GM true difference {true_toy[gm_toy].mean():.1f} units; one-pair noise SD {resid1.std():.1f} (√2 σ = {sd:.1f}); SNR per pair {snr_pair:.2f}")
for n in [10, 30, 60]:
    print(f"{n:>2} pairs: SNR {snr_n[list(ns).index(n)]:.2f}   (√N law: {snr_pair * np.sqrt(n):.2f})")
```

The histogram is Gaussian with the predicted standard deviation of 56.6 units: two
independent draws of σ = 40 added in quadrature. Nothing is left of the Rician shape,
because both images sit at SNR 150. The points on the right lie on the $\sqrt{N}$ line
from one pair (SNR 0.52) to 60 (SNR 4.1); the mean of the reference protocol's 30 pairs has
an SNR of 2.9. The law holds because each pair's noise is independent of every other
pair's, which is true of thermal noise and untrue of the slow physiological and
motion-related fluctuations of real data; the simulator has only thermal noise, so the line
here is a best case.

## The reference and its artifact-free twin

The pipeline series `ref-pcasl` is the same protocol with the readout: k-space, 2× oversampled
simulation grid, magnitude reconstruction, 30 pairs. `ref-clean` is the identical acquisition
with the noise variance set to zero, which is the reference for every error in this chapter:
its mean difference is the perfusion signal as the readout delivers it, with no noise at all.
The figure shows the display slice of one control image, the difference of one pair, the mean
difference of all 30 pairs, and the noise-free mean difference.

```{code-cell} python
:tags: [hide-input]
ref = data.load_dataset("ref-pcasl").run("pcasl")
clean = data.load_dataset("ref-clean").run("pcasl")
mag, ctx, p = ref.mag(), ref.context(), ref.sidecar()
fr, mask = ref.fractions(), ref.mask()
gm, wm = fr["gm"] >= 0.9, fr["wm"] >= 0.9
d_ref = quant.subtract(mag, ctx)
dm_ref = d_ref.mean(axis=-1)
dm_clean = quant.subtract(clean.mag(), clean.context()).mean(axis=-1)
dm_truth = truth_deltam(ref)

fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
show_slice(axes[0], mag[..., 0], K, "control image", vmin=0, vmax=7000)
show_slice(axes[1], d_ref[..., 0], K, "one pair: control − label", kind="diff", vmin=-150, vmax=150)
show_slice(axes[2], dm_ref, K, "mean of 30 pairs", kind="diff", vmin=-60, vmax=60)
im = show_slice(axes[3], dm_clean, K, "noise-free mean (ref-clean)", kind="diff", vmin=-60, vmax=60)
side_colorbar(fig, im, "difference (image units)")
print(f"GM control {control_mean(mag, ctx)[gm].mean():.0f} units; GM difference per pair {d_ref[gm].mean():.1f}; "
      f"noise-free GM difference {dm_clean[gm].mean():.1f}; deltam truth in image units {dm_truth[gm].mean():.1f}")
print(f"noise-free mean difference vs deltam truth in the brain: RMSE {quant.score(dm_clean, dm_truth, mask)['rmse']:.2f} units "
      f"(the readout's ringing and box averaging; Chapter 5)")
```

One pair is noise with a faint brain-shaped tint; the mean of 30 is a recognizable
perfusion image with gray matter above white matter; the noise-free version is what those
30 pairs are trying to estimate. The gray matter control signal is 6246 units and the
difference per pair 30 units, the numbers the physics section used. The noise-free mean
differs from the `deltam` ground truth by 1.3 units RMSE, the residual of the readout
itself (Gibbs ringing and the box averaging onto the acquisition grid,
[Chapter 5](../02-labeling/05-kinetic-model.md)), so the `ref-clean` mean is used as the
truth for the difference image from here on and the `deltam` map for the CBF.

## Measure it: the noise level

A pipeline needs σ, for instance to set a threshold for outlier rejection or to report SNR.
It can be read **outside the head**, where the signal is zero and the magnitude is Rayleigh
distributed with mean 1.25 σ, standard deviation 0.66 σ and mean square 2 σ²
{cite:p}`henkelman1985,gudbjartsson1995`, or **in a
tissue with high SNR**, where the magnitude noise is Gaussian and the standard deviation of a
voxel's control signal over the series is σ directly, as long as nothing but noise changes
from volume to volume. The catch with the background is that it must really be empty. The
left panel is the noise-free control image with a tight intensity window: the readout
spreads a ringing halo of the 6000-unit head signal into the surrounding voxels, about 100
units within three voxels of the head and 13 units beyond eight. The middle panel is the
histogram of the σ = 40 run's magnitudes in background voxels at least eight voxels from any
tissue, with the Rayleigh curve for the σ measured in gray matter; the right panel is the
same for the eight-coil GRAPPA run.

```{code-cell} python
:tags: [hide-input]
total = fr["gm"] + fr["wm"] + fr["csf"]
dist = ndimage.distance_transform_edt(total == 0)
far = (total == 0) & (dist > 8)
near = (total == 0) & (dist <= 3)
sweep = data.load_dataset("noise-sweep")
runs = {name: sweep.run(name) for name in ["sigma10", "sigma40", "sigma80", "coils8-r2"]}
series = {name: (r.mag(), r.context()) for name, r in runs.items()}
diffs = {name: quant.subtract(m, c) for name, (m, c) in series.items()}
ctrl_idx = [i for i, r in enumerate(ctx) if r == "control"]

sig_tissue = {name: float(m[..., ctrl_idx][gm].std(axis=-1, ddof=1).mean()) for name, (m, c) in series.items()}
sig_bg = {name: float(np.sqrt(np.mean(m[far] ** 2) / 2)) for name, (m, c) in series.items()}

fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
show_slice(axes[0], clean.mag()[..., 0], K, "noise-free control, window 0–150", vmin=0, vmax=150)
for ax, name in zip(axes[1:], ["sigma40", "coils8-r2"]):
    m = series[name][0]
    sg = sig_tissue[name]
    v = m[far] / sg
    r = np.linspace(0, 5, 300)
    ax.hist(v, bins=60, density=True, color=PALETTE[3], alpha=0.6, label=f"background ≥ 8 voxels out: mean {v.mean():.2f} σ")
    ax.plot(r, r * np.exp(-r**2 / 2), color="0.3", lw=1.5, label="Rayleigh for the GM σ")
    ax.set(xlabel="magnitude (units of the GM σ)", ylabel="density", title=f"{name}: far background", xlim=(0, 6))
    ax.legend(fontsize=7)
fig.tight_layout()
print(f"noise-free background: mean {clean.mag()[..., 0][near].mean():.1f} units within 3 voxels of tissue, "
      f"{clean.mag()[..., 0][far].mean():.1f} units beyond 8 voxels")
print(f"{'run':>10} {'σ set':>6} {'σ from GM control series':>26} {'σ from far background':>22} {'ΔM per pair SNR (GM)':>22}")
for name, (m, c) in series.items():
    d = diffs[name]
    snr = d[gm].mean() / d[gm].std(axis=-1, ddof=1).mean()
    print(f"{name:>10} {np.sqrt(runs[name].simulation()['Acquisition']['NoiseVariance']):>6.0f} {sig_tissue[name]:>26.1f} {sig_bg[name]:>22.1f} {snr:>22.2f}")
```

The tissue estimate is exact at every noise level: 9.9, 39.7 and 79.4 for σ set to 10, 40
and 80. The far background gives 42.3 for σ = 40 and 81 for σ = 80, within 6 %, but 17
for σ = 10, nearly twice the true value, because the 13 units of ringing left far from the
head are no longer small compared with the noise. Within three voxels of the head the
ringing averages about 100 units and the background estimate is useless at any noise
level. In real data the same warning applies to ghosts, the scalp and the edge of the field
of view; a background estimate needs a region that is genuinely empty, and the tissue
estimate needs a series in which the tissue signal is genuinely constant, which
physiological fluctuation and motion violate. The `sigma40` histogram has the Rayleigh
shape of a single coil, with its mean pushed from 1.25 σ to 1.33 σ by the residual ringing.

The **eight-coil GRAPPA run** is different in two ways. First, eight coil images have to be
combined into one. The simplest combination, the root sum of squares
{cite:p}`roemer1990`, adds the squares of 16 Gaussian components, and its magnitude follows
a **non-central chi** distribution, whose floor is higher than the Rician one for the same
per-coil noise {cite:p}`constantinides1997`. The simulator instead weights each coil image
by the coil's known sensitivity before adding them, the combination
{cite:t}`roemer1990` showed to be optimal; that keeps the combined image complex, with
Gaussian noise in both channels, so its magnitude is still Rician, but the noise level now
depends on position. Second, GRAPPA {cite:p}`griswold2002` reconstructs the
skipped k-space lines from the coil data, which amplifies the noise by a further factor that
depends on position (the **g-factor**, a term introduced with SENSE
{cite:p}`pruessmann1999` and computed for GRAPPA by {cite:t}`breuer2009`;
[Chapter 2](../01-mri-physics/02-epi-and-reconstruction.md))
and correlates it between neighboring voxels. The histogram above does not match the
Rayleigh curve drawn for the gray matter σ: its mean is 0.63 of that σ, because the
noise level differs between the background and the tissue, and its shape is a mixture of
Rayleigh distributions at different noise levels. The "σ from the far background" of
this run bears no fixed relation to the tissue noise. The figure below shows the spatial
pattern: the standard deviation over the 30 pairs of each voxel's difference, for the
single-coil run and the GRAPPA run.

```{code-cell} python
:tags: [hide-input]
sd_maps = {name: diffs[name].std(axis=-1, ddof=1) for name in ["sigma40", "coils8-r2"]}
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4))
for ax, name in zip(axes, sd_maps):
    im = show_slice(ax, np.where(mask, sd_maps[name], np.nan), K, f"noise SD of the difference: {name}", kind="scalar", vmin=0, vmax=250)
side_colorbar(fig, im, "image units", right=0.86)
cx, cy = np.array(mask.shape[:2]) // 2
yy, xx = np.meshgrid(np.arange(mask.shape[1]), np.arange(mask.shape[0]))
center = mask & (np.hypot(xx - cx, yy - cy) < 10)[..., None]
for name, sd in sd_maps.items():
    print(f"{name:>10}: noise SD of the difference, center of the FOV {sd[center].mean():.0f}, rest of the brain {sd[mask & ~center].mean():.0f} units")
```

The single-coil noise is flat across the brain (56 units everywhere, √2 × 40). The GRAPPA
noise is 166 units at the center of the field of view and 135 toward the edges, where the
coils are close and their sensitivities differ enough to separate the aliased points. The
run's per-coil noise variance is the same 1600 as the single-coil run; the higher level in
the tissue is the price of skipping half of k-space (√2) times the g-factor. In a dataset
like this, one σ does not describe the image, and any threshold or SNR map must be computed
voxel by voxel from the series itself.

## See it: temporal SNR

The **temporal SNR** (tSNR) of a voxel is the mean of its difference series divided by the
standard deviation of that series, computed with `quant.tsnr`. It is the SNR of one pair,
and multiplying it by $\sqrt{N}$ gives the SNR of the mean of $N$ pairs. Because it is
measured from the data alone, it captures every source of fluctuation, not just thermal
noise, which makes it a common quality measure of an ASL series. The maps below are the
four runs of the noise sweep at the display slice.

```{code-cell} python
:tags: [hide-input]
fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
for ax, name in zip(axes, diffs):
    t = quant.tsnr(diffs[name])
    show_slice(ax, np.where(mask, t, np.nan), K, f"tSNR, {name}", kind="scalar", vmin=0, vmax=2.5)
    print(f"{name:>10}: tSNR per pair, GM {t[gm].mean():.2f}, WM {t[wm].mean():.2f}; SNR of the mean of 30 pairs, GM {np.sqrt(30) * t[gm].mean():.1f}")
fig.colorbar(axes[3].images[0], ax=axes, shrink=0.8, label="tSNR per pair")
```

At σ = 10 the cortex stands out with a tSNR of 2.2 per pair; at σ = 40 it is 0.55, and at
σ = 80 and in the GRAPPA run, 0.27 and 0.26, the map is mostly speckle. White matter, with a
third of the perfusion and a longer transit time, has a tSNR of 0.08 at the reference
noise, an SNR of 0.4 after 30 pairs: white matter perfusion is not measurable voxel by
voxel with this protocol, only as a regional average. The low sensitivity of ASL in white
matter, with its lower perfusion and longer transit time, is the subject of
{cite:t}`vanosch2009`.

## Measure it: the cost in CBF

The noise in the mean difference propagates into CBF through the white-paper formula
{cite:p}`alsop2015` ([Chapter 14](../04-quantification/14-cbf-quantification.md)), which is
linear in ΔM: every unit of noise in the mean difference becomes a fixed number of
ml/100 g/min. Here the mean difference of the first $N$ pairs is quantified with the M0 scan
used directly as the calibration image and the slice-dependent post-labeling delay of the
2D readout (`quant.slice_plds`). The panels are the reference run's 30-pair estimate
against the true perfusion map.

```{code-cell} python
:tags: [hide-input]
plds = quant.slice_plds(p["PostLabelingDelay"], p["SliceTiming"])[None, None, :]
cbf_truth = ref.truth("perfusion")
m0_clean = clean.m0scan()
cbf_clean = quant.cbf_pcasl(dm_clean, m0_clean, plds)
cbf30 = quant.cbf_pcasl(dm_ref, ref.m0scan(), plds)
fig, axes = fit_vs_truth(cbf30, cbf_truth, mask, "CBF, 30 pairs", k=K, unit="(ml/100 g/min)")
s_truth, s_clean = quant.score(cbf30, cbf_truth, gm), quant.score(cbf30, cbf_clean, gm)
s0 = quant.score(cbf_clean, cbf_truth, gm)
print(f"noise-free estimate in GM: {cbf_clean[gm].mean():.1f} ml/100 g/min against a truth of {cbf_truth[gm].mean():.1f} (bias {s0['bias']:+.1f})")
print(f"30 pairs, GM: RMSE vs truth {s_truth['rmse']:.1f}; RMSE vs the noise-free estimate {s_clean['rmse']:.1f}; bias vs noise-free {s_clean['bias']:+.1f}")
```

Two things are wrong with the estimate, and only one of them is noise. The noise-free
estimate itself is 47 ml/100 g/min in gray matter against a truth of 59 (60 in pure gray
matter, slightly less in voxels that are 90 % gray matter): the white-paper
formula assumes the label decays with the T1 of blood and takes the tissue M0 scan as the
blood M0, while the phantom's label exchanges into tissue and decays with the shorter tissue
T1, and the M0 scan carries the tissue's T2 decay rather than the blood's. That bias is the
topic of [Chapters 14](../04-quantification/14-cbf-quantification.md) and
[16](../04-quantification/16-calibration.md), and it is the same in every voxel of gray
matter. The noise is on top of it: 16.7 ml/100 g/min RMSE around the noise-free estimate
at 30 pairs, 28 % of the true value, which is the speckle in the difference map and the
vertical spread of the scatter. The next figure separates the two by scoring against the
noise-free estimate as the number of pairs and the noise level change.

```{code-cell} python
:tags: [hide-input]
pairs_used = [5, 10, 20, 30]
fig, ax = plt.subplots(figsize=(7, 3.4))
rmse_table = {}
for j, name in enumerate(diffs):
    m0 = runs[name].m0scan()
    rmse = [quant.score(quant.cbf_pcasl(diffs[name][..., :n].mean(axis=-1), m0, plds), cbf_clean, gm)["rmse"] for n in pairs_used]
    rmse_table[name] = rmse
    ax.plot(pairs_used, rmse, "o-", color=PALETTE[j], label=name)
    ax.plot(pairs_used, rmse[0] * np.sqrt(pairs_used[0] / np.array(pairs_used, float)), ":", color=PALETTE[j], lw=1)
ax.set(xlabel="pairs averaged", ylabel="CBF RMSE vs noise-free (ml/100 g/min)", title="noise in CBF vs pairs averaged (GM; dotted: 1/√N)", xticks=pairs_used, ylim=(0, None))
ax.legend()
fig.tight_layout()
print(f"{'run':>10}" + "".join(f"{n:>10} pairs" for n in pairs_used))
for name, rmse in rmse_table.items():
    print(f"{name:>10}" + "".join(f"{v:>16.1f}" for v in rmse))
```

Every curve follows $1/\sqrt{N}$ (dotted), and the four are spaced by the noise level: at
30 pairs the gray matter RMSE is 4.2 ml/100 g/min at σ = 10, 16.7 at σ = 40, 33 at σ = 80
and 45 in the GRAPPA run. Read the table backward to plan a scan: the reference protocol
needs 30 pairs to reach 17 ml/100 g/min per voxel and would need 120 to halve that. Per
voxel is the key qualifier: a gray matter region of 100 voxels averages the noise down by
another factor of 10, so regional CBF is precise long before voxelwise CBF is, which is
what smoothing trades on.

## Correction step by step: smoothing, and what it costs

Averaging over pairs uses the time axis; **spatial smoothing** uses the neighbors. A
Gaussian filter replaces each voxel of the mean difference with a weighted mean of its
neighborhood (the CBF formula is linear in ΔM, so the same can be done on the CBF map). The
noise falls roughly with the square root of the number of voxels averaged; the perfusion
map is blurred by the same kernel, so wherever the true CBF changes between neighbors, at
the boundary of cortex with white matter and CSF, the smoothed value is a mixture, and in a
3.5 mm grid almost every gray matter voxel has such a neighbor. The figure shows the
reference run's mean difference smoothed in-plane with Gaussian kernels of 0, 0.5, 1 and
1.5 voxels (standard deviation; the full width at half maximum is 2.35 times that), and
below it the noise-free map smoothed the same way, which isolates what the kernel does to
the anatomy.

```{code-cell} python
:tags: [hide-input]
sigmas_vox = [0.0, 0.5, 1.0, 1.5]
smooth = lambda vol, sv: ndimage.gaussian_filter(vol, (sv, sv, 0)) if sv > 0 else vol
fig, axes = plt.subplots(2, 4, figsize=(11, 5.6))
for j, sv in enumerate(sigmas_vox):
    show_slice(axes[0, j], smooth(dm_ref, sv), K, f"30 pairs, kernel SD {sv:g} vox", kind="diff", vmin=-60, vmax=60)
    im = show_slice(axes[1, j], smooth(dm_clean, sv), K, f"noise-free, kernel SD {sv:g} vox", kind="diff", vmin=-60, vmax=60)
side_colorbar(fig, im, "difference (image units)")
```

Across the top row the speckle disappears and the cortex emerges; across the bottom row the
same kernel fills the sulci, spreads gray matter signal into white matter, and lowers the
peak values of the thin cortical ribbon. The cost is measured below in two places: the bias
in the voxels that are at least 90 % gray matter, and the error at **edge voxels**, defined
here as brain voxels that are between 20 % and 80 % gray matter.

```{code-cell} python
:tags: [hide-input]
edge = mask & (fr["gm"] > 0.2) & (fr["gm"] < 0.8)
sv_axis = np.array([0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
rows = []
for sv in sv_axis:
    est = quant.cbf_pcasl(smooth(dm_ref, sv), ref.m0scan(), plds)
    blur = quant.cbf_pcasl(smooth(dm_clean, sv), m0_clean, plds)
    rows.append((quant.score(est, cbf_clean, gm)["rmse"], quant.score(est, cbf_clean, gm)["bias"],
                 quant.score(blur, cbf_clean, gm)["bias"], quant.score(blur, cbf_clean, edge)["rmse"]))
rows = np.array(rows)
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.plot(sv_axis, rows[:, 0], "o-", color=PALETTE[0], label="noisy: GM RMSE vs noise-free map")
ax.plot(sv_axis, rows[:, 2], "s-", color=PALETTE[1], label="kernel alone: bias in pure GM")
ax.plot(sv_axis, rows[:, 3], "^-", color=PALETTE[3], label="kernel alone: RMSE at edge voxels")
ax.axhline(0, color=INK["secondary"], lw=0.8)
ax.set(xlabel="Gaussian kernel SD (voxels of 3.5 mm)", ylabel="ml/100 g/min", title="what smoothing removes and what it costs (30 pairs)")
ax.legend()
fig.tight_layout()
print(f"{'kernel SD':>10} {'GM RMSE (noisy)':>16} {'GM bias (kernel)':>17} {'edge RMSE (kernel)':>19}")
for sv, (rmse, _, bias_k, edge_k) in zip(sv_axis, rows):
    print(f"{sv:>10.2f} {rmse:>16.1f} {bias_k:>17.1f} {edge_k:>19.1f}")
```

The blue curve is the total error of the smoothed noisy map: it falls from 16.7 to 9.5
ml/100 g/min at a kernel of 0.75 voxel and then rises again, because the kernel's own error
takes over. That error has two parts. Pure gray matter is biased low, by 2.4 ml/100 g/min
at half a voxel and 7.6 at one voxel, since the kernel mixes in the lower perfusion of the
neighboring white matter and CSF; it is a bias, not a fluctuation, and it does not shrink
with more pairs. Edge voxels are moved either way, by 7.9 ml/100 g/min RMSE at one voxel.
A kernel of half a voxel (FWHM about 4 mm, comparable to the voxel) removes a third of the
noise for a bias of 2.4 ml/100 g/min; an 8 mm FWHM kernel is one
voxel here and costs 7.6 ml/100 g/min in cortex, and a 12 mm one costs 13. The right kernel
depends on the question: a regional mean needs none, a voxelwise map of a thin cortex can
afford little, and the bias should be reported alongside the map.

:::{admonition} Beyond a Gaussian kernel
:class: note
Smoothing is the crudest denoiser because it ignores everything except distance. Better
ones use structure the noise does not have: **outlier rejection** drops the pairs whose
difference is far from the rest {cite:p}`tan2009,dolui2017` (a motion-corrupted pair can
carry ten times the noise of a clean one; [Chapter 10](../03-preprocessing/10-motion.md)),
**temporal filtering** removes slow drifts, and **partial-volume correction** replaces
smoothing's mixture with a model of it {cite:p}`asllani2008`
([Chapter 12](../03-preprocessing/12-partial-volume.md)). Local PCA, which learns the
structure of the series from the series itself, is the next section; denoisers learned from
training data are in [Chapter 19](../05-advanced/19-frontiers.md). All are measured the way
this chapter measured smoothing: error against a truth, split into noise removed and
structure destroyed.
:::

## Correction step by step: MP-PCA

Smoothing assumes that neighboring voxels are alike. **Marchenko-Pastur PCA** (MP-PCA)
{cite:p}`veraart2016`, introduced for diffusion MRI, assumes less: only that the series is
redundant. Take a small patch of neighboring voxels, here 5 × 5 × 5 = 125 of them, and the
60 values of each (30 controls and 30 labels). In this simulation the true signal of the
patch has two parts: the static tissue signal, with one spatial pattern that is the same in
every volume, and the perfusion signal, with another pattern whose sign alternates between
control and label. A real series adds a few more (drift, motion, pulsation), but far fewer
than 60. Principal component analysis (PCA) finds such shared patterns and ranks them by the
variance each one explains. Thermal noise is independent in every voxel and every volume, so
no pattern explains it, and it spreads its variance over all 60 components.

Random-matrix theory says how it spreads. Arrange the patch as a matrix $X$ with $N = 125$
rows (voxels) and $M = 60$ columns (volumes), subtract from every volume its mean over the
patch, and call $\lambda_1 \ge \dots \ge \lambda_M$ the variances of the principal
components (the eigenvalues of $X^\mathsf{T}X/N$). If the patch held nothing but noise of
standard deviation σ, they would lie within the **Marchenko-Pastur** range
{cite:p}`marchenko1967`

$$
\lambda_\pm = \sigma^2 \left(1 \pm \sqrt{\gamma}\right)^2, \qquad \gamma = M/N,
$$

whose mean is σ² and whose width is $4\sigma^2\sqrt{\gamma}$. With $\gamma = 0.48$ the range
runs from 0.09 σ² to 2.87 σ². A component above the upper edge carries signal; a component
inside the range cannot be told from noise. MP-PCA keeps the first kind, sets the second to
zero, and rebuilds the patch from what is left. The figure is this analysis for one patch
of the reference series centered in the cortex: each bar is one component's variance in
units of σ², sorted, on a logarithmic scale; the gray band is the Marchenko-Pastur range for
σ = 40; the open circles are the same analysis of the noise-free series (`ref-clean`).

```{code-cell} python
:tags: [hide-input]
def patch_spectrum(vol4d, center, r=2):
    """Variances of the principal components of one (2r+1)^3 patch, largest first."""
    i, j, k = center
    X = vol4d[i - r : i + r + 1, j - r : j + r + 1, k - r : k + r + 1].reshape(-1, vol4d.shape[-1]).astype(float)
    X = X - X.mean(axis=0)
    return np.linalg.eigvalsh(X.T @ X / X.shape[0])[::-1], X.shape[0]

center = (14, 40, K)
mag_clean = clean.mag()
ev_noisy, n_vox = patch_spectrum(mag, center)
ev_clean, _ = patch_spectrum(mag_clean, center)
s2_true = 40.0**2
gamma = mag.shape[-1] / n_vox
mp_lo, mp_hi = (1 - np.sqrt(gamma)) ** 2, (1 + np.sqrt(gamma)) ** 2   # in units of σ²
n_signal = int((ev_noisy / s2_true > mp_hi).sum())

fig, ax = plt.subplots(figsize=(7, 3.4))
idx = np.arange(1, len(ev_noisy) + 1)
ax.axhspan(mp_lo, mp_hi, color="0.85", zorder=0, label="range for pure noise (Marchenko-Pastur)")
ax.bar(idx[:n_signal], ev_noisy[:n_signal] / s2_true, color=PALETTE[0], width=0.7, label="noisy patch: kept (signal)")
ax.bar(idx[n_signal:], ev_noisy[n_signal:] / s2_true, color=PALETTE[3], width=0.7, label="noisy patch: discarded (noise)")
ax.plot(idx[:2], ev_clean[:2] / s2_true, "o", mfc="none", mec=INK["primary"], ms=6, label="noise-free patch (two components)")
ax.set(yscale="log", ylim=(1e-2, 1e4), xlim=(0.3, len(idx) + 0.7), xlabel="component (sorted by variance)",
       ylabel="variance (units of σ²)", title=f"PCA of one {n_vox}-voxel patch in the cortex, {mag.shape[-1]} volumes")
ax.legend(loc="upper right", fontsize=7)
fig.tight_layout()
box = tuple(slice(c - 2, c + 3) for c in center)
print(f"patch at {center}: {fr['gm'][box].mean():.0%} gray matter, {fr['wm'][box].mean():.0%} white matter")
print(f"Marchenko-Pastur range for {n_vox} voxels x {mag.shape[-1]} volumes: {mp_lo:.2f} to {mp_hi:.2f} σ²; "
      f"{n_signal} of {len(ev_noisy)} components above it (the largest: {ev_noisy[0] / s2_true:.0f} σ²)")
print(f"noise-free patch: static component {ev_clean[0] / s2_true:.0f} σ², perfusion component {ev_clean[1] / s2_true:.2f} σ² "
      f"({ev_clean[1] / s2_true / mp_hi:.0%} of the upper edge), all others {ev_clean[2] / s2_true:.0e} σ²")
```

One component stands three orders of magnitude above the band: the static tissue pattern,
the contrast between gray and white matter that every volume shares. The other 59 lie
inside the band. The noise-free patch has exactly two components, and the second, the
perfusion signal, has a variance of 0.15 σ², one twentieth of the upper edge: it sits inside
the noise range, and no threshold on the variances can separate it from the noise. The
denoiser therefore removes it along with the noise. What is left of the perfusion signal in
the denoised series, and why there is any, is what the measurements below are for.

The scanner does not supply σ, so the method estimates it from the same eigenvalues
{cite:p}`veraart2016,veraart2016b`. Suppose the $p$ largest components are signal. The remaining $M - p$
should then fill a Marchenko-Pastur range for a matrix with $M - p$ columns, which gives two
estimates of the noise variance, one from their mean and one from their width:

$$
\hat\sigma^2_\mathrm{mean}(p) = \frac{1}{M - p} \sum_{i > p} \lambda_i, \qquad
\hat\sigma^2_\mathrm{width}(p) = \frac{\lambda_{p+1} - \lambda_M}{4\sqrt{(M - p)/N}}.
$$

As long as a signal component is still counted among the noise, its large variance
stretches the width and the second estimate exceeds the first. The procedure raises $p$
from zero until $\hat\sigma^2_\mathrm{mean}(p) \ge \hat\sigma^2_\mathrm{width}(p)$, takes
that $p$ as the number of signal components and that mean as σ². The cell below is the
whole method in plain numpy. `mp_cut` is the rule just described, applied to all patches of
a slab at once; `patch_pca` cuts the series into patches, decomposes each, zeroes the
components the rule rejects, rebuilds, and averages the estimates of the patches that
overlap in each voxel. To keep the run to a few seconds it takes every second patch
position in-plane, which still covers each voxel with about 30 patches.

```{code-cell} python
from numpy.lib.stride_tricks import sliding_window_view


def mp_cut(ev, n_vox):
    """The Marchenko-Pastur rule on variances sorted largest first, shape (..., M): the number
    of signal components p and the noise variance of every patch."""
    n_noise = ev.shape[-1] - np.arange(ev.shape[-1])                      # M - p for p = 0 .. M - 1
    s2_mean = np.cumsum(ev[..., ::-1], axis=-1)[..., ::-1] / n_noise      # mean of the M - p smallest
    s2_width = (ev - ev[..., -1:]) / (4 * np.sqrt(n_noise / n_vox))       # their range / (4 sqrt(gamma))
    p = np.argmax(s2_mean >= s2_width, axis=-1)                           # the first p that is consistent
    return p, np.take_along_axis(s2_mean, p[..., None], axis=-1)[..., 0]


def patch_pca(vol4d, cut, patch=(5, 5, 5), step=2):
    """Local PCA denoising of an (x, y, z, volume) series, real or complex. Returns the
    denoised series and maps of the noise SD and of the number of components kept."""
    n_vox, n_vol = int(np.prod(patch)), vol4d.shape[-1]
    out = np.zeros(vol4d.shape, vol4d.dtype)
    hits, s2, kept = (np.zeros(vol4d.shape[:3]) for _ in range(3))
    # in-plane patch corners: every step-th position, plus the last one so that the far edge is covered
    ix, iy = (np.unique(np.r_[0 : n - w + 1 : step, n - w]) for n, w in zip(vol4d.shape[:2], patch[:2]))
    px, py = len(ix), len(iy)
    win = sliding_window_view(vol4d, patch, axis=(0, 1, 2))               # (corner x, y, z, volume, 5, 5, 5), a view
    for k in range(win.shape[2]):                                         # one slab of patches at a time
        X = np.moveaxis(win[:, :, k][np.ix_(ix, iy)], 2, -1).reshape(px, py, n_vox, n_vol)   # (patch x, patch y, voxels, volumes)
        mean = X.mean(axis=2, keepdims=True)
        X = X - mean                                                      # center every volume on its patch mean
        ev, V = np.linalg.eigh(np.swapaxes(X.conj(), -1, -2) @ X / n_vox)     # volumes x volumes covariance
        ev, V = ev[..., ::-1], V[..., ::-1]                               # largest first
        n_keep, sig2 = cut(ev, n_vox)
        V = V[..., : max(int(n_keep.max()), 1)]                           # no patch of this slab keeps more
        V = V * (np.arange(V.shape[-1]) < n_keep[..., None])[..., None, :]    # zero the noise components
        Xd = ((X @ V) @ np.swapaxes(V.conj(), -1, -2) + mean).reshape(px, py, *patch, n_vol)
        for a, b, c in np.ndindex(*patch):                                # put every patch voxel back in place
            sl = (ix[:, None] + a, iy[None, :] + b, k + c)
            out[sl] += Xd[:, :, a, b, c]
            hits[sl] += 1; s2[sl] += sig2; kept[sl] += n_keep
    return out / hits[..., None], np.sqrt(s2 / hits), kept / hits


den_ref, sig_ref, kept_ref = patch_pca(mag.astype(float), mp_cut)
print(f"ref-pcasl: estimated σ in GM {sig_ref[gm].mean():.1f}, in WM {sig_ref[wm].mean():.1f} (true: 40); "
      f"components kept per patch, mean over the brain: {kept_ref[mask].mean():.2f} of {mag.shape[-1]}")
```

The rule finds the noise level without being told: 39.3 in gray matter and 39.6 in white
matter against a true 40. It keeps 1.2 components per patch on average: the static pattern
everywhere, and a second component in a minority of patches.

## See it: the denoised series

The harder case is the `sigma80` run of the noise sweep, with twice the noise. The figure
shows, for both runs, the first control image before and after MP-PCA and the difference of
the first control-label pair before and after.

```{code-cell} python
:tags: [hide-input]
mag80 = series["sigma80"][0].astype(float)
den80, sig80, kept80 = patch_pca(mag80, mp_cut)
fig, axes = plt.subplots(2, 4, figsize=(12, 6.2))
for row, (label, raw, den) in enumerate([("σ = 40", mag, den_ref), ("σ = 80", mag80, den80)]):
    show_slice(axes[row, 0], raw[..., 0], K, f"{label}: control, raw", vmin=0, vmax=7000)
    show_slice(axes[row, 1], den[..., 0], K, f"{label}: control, MP-PCA", vmin=0, vmax=7000)
    show_slice(axes[row, 2], raw[..., 0] - raw[..., 1], K, "one pair's difference, raw", kind="diff", vmin=-60, vmax=60)
    im = show_slice(axes[row, 3], den[..., 0] - den[..., 1], K, "one pair's difference, MP-PCA", kind="diff", vmin=-60, vmax=60)
    d_den = quant.subtract(den, ctx)
    print(f"{label}: error of the images against the noise-free series in GM, raw {(raw - mag_clean)[gm].std():.1f}, MP-PCA {(den - mag_clean)[gm].std():.1f} units; "
          f"σ estimate in GM {(sig_ref if row == 0 else sig80)[gm].mean():.1f}; components kept {(kept_ref if row == 0 else kept80)[mask].mean():.2f}; "
          f"tSNR per pair in GM, raw {quant.tsnr(quant.subtract(raw, ctx))[gm].mean():.2f}, MP-PCA {quant.tsnr(d_den)[gm].mean():.1f}")
side_colorbar(fig, im, "difference (image units)")
```

The control images look the same before and after, because at this intensity window the
noise was already invisible (0.6 % of the tissue signal at σ = 40). The numbers show the
change: the error of a gray matter voxel against the noise-free series falls from 40 to 6.5
units, and from 80 to 13 in the noisier run. The size of the gain follows from what is
kept: with one component, every volume of a patch is rebuilt from two numbers of its own,
its patch mean and the amplitude of the one shared pattern, where it had 125 noisy values.
The difference of one pair shows the result directly. Before, it is noise at twice
(σ = 40) or four times (σ = 80) the perfusion signal; after, it is a perfusion-weighted
image. Two warnings come with that picture. Its noise has not gone but changed character:
what is left is shared by the whole patch, so it appears as smooth blotches, plainest in
the background, instead of speckle. And it is smoother than the noise-free mean difference
shown earlier: the cortical ribbon has lost its detail. The temporal SNR of the denoised
series, 5.9 per pair in gray matter against 0.55 before, counts only the first of these.
It measures the fluctuation that is left and is blind to the structure that was removed,
so after denoising it is no longer a measure of how good the perfusion map is. The next
section measures that against the truth.

The second output of the method is its noise map, shown as a ratio to the true σ.

```{code-cell} python
:tags: [hide-input]
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4))
for ax, (label, sig_map, true) in zip(axes, [("ref-pcasl", sig_ref, 40.0), ("sigma80", sig80, 80.0)]):
    im = show_slice(ax, sig_map / true, K, f"MP-PCA σ / true σ: {label}", kind="scalar", vmin=0, vmax=1.2)
    print(f"{label:>9}: estimated σ, GM {sig_map[gm].mean():.1f}, WM {sig_map[wm].mean():.1f}, brain median {np.median(sig_map[mask]):.1f} "
          f"(true {true:.0f}); far background {sig_map[far].mean():.1f} = {sig_map[far].mean() / true:.2f} σ")
side_colorbar(fig, im, "estimated σ / true σ", right=0.86)
```

Inside the head the map is flat and 1 to 2 % below the truth (39.5 and 78.9 at the median):
the patch mean and the kept component each absorb a little of the noise variance, which
the estimate then does not count. Far outside the head it drops to 0.68 σ and 0.66 σ. That
is not a failure of the method but the Rician distribution again: where there is no signal
the magnitude is Rayleigh distributed, and its standard deviation is 0.66 σ, not σ. A noise
map estimated from magnitude data is trustworthy where the SNR is high and a third too low
where there is no signal.

## Measure it: what MP-PCA does to CBF

The denoised series goes through the same subtraction, averaging and quantification as the
raw one. The table scores the CBF map of all 30 pairs against the noise-free estimate, as
the smoothing section did, for three runs of increasing noise, and adds the error against
the true perfusion map; the figure shows the mean difference of the reference run.

```{code-cell} python
:tags: [hide-input]
def cbf_scores(vol4d, run):
    """Mean difference, CBF map, and (GM RMSE, GM bias, WM bias, edge RMSE vs noise-free; GM RMSE vs truth)."""
    dm = quant.subtract(vol4d, ctx).mean(axis=-1)
    cbf = quant.cbf_pcasl(dm, run.m0scan(), plds)
    g, w = quant.score(cbf, cbf_clean, gm), quant.score(cbf, cbf_clean, wm)
    return dm, cbf, (g["rmse"], g["bias"], w["bias"], quant.score(cbf, cbf_clean, edge)["rmse"], quant.score(cbf, cbf_truth, gm)["rmse"])

mag10 = series["sigma10"][0].astype(float)
den10, sig10, kept10 = patch_pca(mag10, mp_cut)
print(f"{'run':>9} {'series':>7} {'kept':>5} {'GM RMSE':>8} {'GM bias':>8} {'WM bias':>8} {'edge RMSE':>10} {'GM RMSE vs truth':>17}")
for name, run, raw, den, kept in [("sigma10", runs["sigma10"], mag10, den10, kept10), ("ref-pcasl", ref, mag, den_ref, kept_ref),
                                  ("sigma80", runs["sigma80"], mag80, den80, kept80)]:
    for label, vol4d in [("raw", raw), ("MP-PCA", den)]:
        sc = cbf_scores(vol4d, run)[2]
        print(f"{name:>9} {label:>7} {f'{kept[mask].mean():.2f}' if label == 'MP-PCA' else '—':>5} {sc[0]:>8.1f} {sc[1]:>+8.1f} {sc[2]:>+8.1f} {sc[3]:>10.1f} {sc[4]:>17.1f}")
dm_den, cbf_den, _ = cbf_scores(den_ref, ref)
contrast = lambda vol: vol[gm].mean() - vol[wm].mean()
print(f"gray-white contrast of the CBF map: noise-free {contrast(cbf_clean):.1f}, raw {contrast(cbf30):.1f}, MP-PCA {contrast(cbf_den):.1f} ml/100 g/min "
      f"({1 - contrast(cbf_den) / contrast(cbf_clean):.0%} lost)")
fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
show_slice(axes[0], dm_ref, K, "mean of 30 pairs, raw", kind="diff", vmin=-60, vmax=60)
show_slice(axes[1], dm_den, K, "mean of 30 pairs, MP-PCA", kind="diff", vmin=-60, vmax=60)
im = show_slice(axes[2], dm_clean, K, "noise-free mean (ref-clean)", kind="diff", vmin=-60, vmax=60)
show_slice(axes[3], np.where(mask, dm_den - dm_clean, 0), K, "MP-PCA − noise-free (window ±20)", kind="diff", vmin=-20, vmax=20)
side_colorbar(fig, im, "difference (image units)")
```

At the reference noise level the error against the noise-free estimate falls from 16.7 to
5.8 ml/100 g/min in gray matter, further than any Gaussian kernel took it (9.5), and at
σ = 80 from 33.4 to 6.6. The two denoised numbers are nearly the same although the noise
doubled, which says that what remains is mostly not noise. It is bias: gray matter is
4.3 ml/100 g/min too low and white matter 6 to 7 too high, at both noise levels. The error map
(right panel) shows where: the cortex and the deep gray matter are underestimated, the
white matter next to them overestimated. The denoised map has lost a quarter of the
contrast between the two tissues (29.1 ml/100 g/min against 39.4 in the noise-free map).

The spectrum figure explains it. The perfusion component was below the noise edge and was
discarded, so the perfusion signal of the denoised series has only two sources. One is the
patch mean of each volume, which the centering set aside and which is returned untouched.
The other is whatever part of the perfusion pattern is proportional to the static pattern:
gray matter is both brighter and better perfused than white matter, so within a patch the
perfusion signal partly follows the control image, and that part rides along in the kept
component. At this noise level MP-PCA therefore replaces each voxel's own perfusion signal
with a fit to its 17.5 × 17.5 × 25 mm patch: the patch's mean perfusion signal plus a term
proportional to the control image. That is a far better model than a box average of the
same size, which is why the error is so low, but it is a spatial model, and the gain was
bought with resolution just as smoothing's was.

The `sigma10` row shows the other regime. With a sixteenth of the noise variance the
perfusion component clears the edge in most patches (1.95 components kept), and a kept
component passes through nearly unchanged: the mean difference of the denoised series is
then close to the mean difference of the raw series. The error barely moves, from 4.2 to
3.5 ml/100 g/min, and so does the bias (−0.4). MP-PCA cannot improve on the average of a
component it keeps, and it replaces a component it discards by a patch fit. Against the
true perfusion map, which adds the quantification bias of 12 ml/100 g/min found above, the
gray matter error goes from 20.5 to 16.7 at σ = 40 and from 12.7 to 13.0 at σ = 10.

## Measure it: is the residual only noise?

The usual check on a denoiser is its residual, the raw series minus the denoised one: if
only noise was removed, the residual has no structure. The figure shows the residual of the
first control image and its histogram inside the brain.

```{code-cell} python
:tags: [hide-input]
resid = mag - den_ref
r0 = resid[..., 0]
inner = ndimage.binary_erosion(mask)
fig, (ax_i, ax_h) = plt.subplots(1, 2, figsize=(9.5, 3.4), gridspec_kw={"width_ratios": [1, 1.4]})
show_slice(ax_i, r0, K, "raw − MP-PCA, first control image", kind="diff", vmin=-120, vmax=120)
x = np.linspace(-160, 160, 300)
ax_h.hist(r0[mask], bins=60, density=True, color=PALETTE[3], alpha=0.6, label=f"residual in the brain: SD {r0[mask].std():.1f}")
ax_h.plot(x, np.exp(-x**2 / (2 * 40.0**2)) / (40.0 * np.sqrt(2 * np.pi)), color="0.3", lw=1.5, label="Gaussian, SD σ = 40")
ax_h.set(xlabel="raw − MP-PCA (image units)", ylabel="density", title="the residual of one volume")
ax_h.legend(fontsize=7)
fig.tight_layout()
print(f"residual of one volume in the brain: mean {r0[mask].mean():+.1f}, SD {r0[mask].std():.1f}; correlation between neighbors along x "
      f"{np.corrcoef(r0[inner], np.roll(r0, 1, axis=0)[inner])[0, 1]:+.3f}, along y {np.corrcoef(r0[inner], np.roll(r0, 1, axis=1)[inner])[0, 1]:+.3f}")
dm_resid = quant.subtract(resid, ctx).mean(axis=-1)
print(f"the residual treated as an ASL series, mean control − label of 30 pairs: GM {dm_resid[gm].mean():+.1f}, WM {dm_resid[wm].mean():+.1f} units "
      f"(SD across voxels {dm_resid[mask].std():.1f}; standard error of the GM mean {dm_resid[gm].std() / np.sqrt(gm.sum()):.2f}); "
      f"gray-white contrast removed: {contrast(dm_resid):.1f} of {contrast(dm_clean):.1f} units ({contrast(dm_resid) / contrast(dm_clean):.0%})")
```

The residual passes that check. It shows no anatomy, it is Gaussian with a standard
deviation of 38.7 (nearly all of the 40 that was there), and neighboring voxels are
uncorrelated (−0.015 and −0.007). The check is not sensitive enough for ASL, though: a
perfusion difference of 3 units cannot be seen in one image of noise with a standard
deviation of 39. The sensitive test is to treat the residual as data: subtract label from
control and average the 30 pairs. Pure noise would average toward zero in every tissue.
The residual's mean difference is +2.8 units in gray matter and −3.3 in white matter, with
a standard error of 0.11: the removed part contains perfusion contrast, 6.1 of the 25.8
units between the two tissues, the same quarter that the CBF table found missing.

A second test asks what happens to a perfusion change that the anatomy does not predict.
The cell halves the perfusion signal of the reference series in a small block of gray
matter, by moving the label images there halfway toward the control images, denoises the
altered series, and compares the mean difference in the block with and without the
deficit.

```{code-cell} python
:tags: [hide-input]
label_idx = [i for i, r in enumerate(ctx) if r == "label"]
for half in (1, 3):
    region = np.zeros(mask.shape, bool)
    hz = min(half, 2)
    region[28 - half : 29 + half, 19 - half : 20 + half, K - hz : K + hz + 1] = True
    lesion = mag.astype(float)
    lesion[..., label_idx] += 0.5 * (dm_clean * region)[..., None]      # half the perfusion signal: the label moves halfway to the control
    dm_lesion = quant.subtract(patch_pca(lesion, mp_cut)[0], ctx).mean(axis=-1)
    deficit_raw = (dm_ref - quant.subtract(lesion, ctx).mean(axis=-1))[region].mean()
    deficit_den = (dm_den - dm_lesion)[region].mean()
    print(f"deficit of 50 % in {2 * half + 1} x {2 * half + 1} x {2 * hz + 1} voxels ({fr['gm'][region].mean():.0%} GM): the mean difference there drops by "
          f"{deficit_raw:.1f} units in the raw series and by {deficit_den:.1f} after MP-PCA ({deficit_den / deficit_raw:.0%} of the deficit survives)")
```

In the raw series the deficit is all there (it is noisy, but averaging over the block or
over more pairs recovers it). After MP-PCA 13 % of a deficit of 3 × 3 × 3 voxels
(10.5 × 10.5 × 15 mm) is left, and 44 % of one of 7 × 7 × 5 voxels, which is wider than
the patch. A focal change is a perfusion pattern that the control image does not share, so
it has no kept component to ride on; it is spread over the patches that contain it. This is
the cost that the gray matter error of 5.8 ml/100 g/min does not show, because the
phantom's perfusion is constant within each tissue and has no focal features to lose.

## The same idea on complex data: NORDIC

NORDIC (noise reduction with distribution corrected PCA) {cite:p}`moeller2021,vizioli2021`
applies the same local low-rank idea with three changes.

- **It works on the complex images**, before the magnitude is taken. Complex noise is
  Gaussian with zero mean in both channels at every signal level, so there is no Rician
  floor for the PCA to keep as if it were signal, and the noise statistics that the
  threshold relies on hold in the background as well as in the tissue.
- **It normalizes the noise.** After parallel imaging the noise level varies across the
  image, as the GRAPPA run showed. NORDIC divides the images by a map of that variation
  (the g-factor) so that the noise is identically distributed everywhere, denoises, and
  multiplies the map back in.
- **It does not estimate the threshold from the patch.** Once the noise has a known,
  uniform level, the largest component variance that a patch of pure noise produces can be
  computed in advance from random matrices of the patch's size. Every component below that
  value is removed.

The book's datasets carry a `part-phase` image next to every magnitude image, so the
complex series can be rebuilt as magnitude × exp(i phase). The simulated object has no
phase of its own, so the phase holds no signal and the noise is Gaussian in both channels.
The cell below is a NORDIC-style variant made from the pieces already on the page: the same
`patch_pca`, applied to the complex series divided by a noise map, with a fixed threshold in
place of `mp_cut`. The threshold is the mean largest component variance of 50 simulated
patches of pure complex noise. The noise level comes from the series: for the single-coil
`sigma80` run, one number, the temporal standard deviation of the white matter control
voxels; for the eight-coil GRAPPA run, a map of the temporal standard deviation of the
complex control images, lightly smoothed.

:::{admonition} A book-sized illustration, not NORDIC
:class: note
This variant shows NORDIC's principle and is not the published implementation. It has no
g-factor map from the reconstruction: the noise map is estimated from the series, which
works here only because nothing but noise changes between the simulated volumes. And it
does nothing about phase, because the simulated object has none; real images carry a phase
that varies across the image and from volume to volume, and a complex denoiser has to deal
with it. For real data use the authors' software.
:::

```{code-cell} python
rng = np.random.default_rng(0)
E = rng.standard_normal((50, 125, 60)) + 1j * rng.standard_normal((50, 125, 60))    # 50 patches of pure complex noise, SD 1 per channel
E -= E.mean(axis=1, keepdims=True)
THR = np.linalg.eigvalsh(np.swapaxes(E.conj(), -1, -2) @ E / 125)[:, -1].mean()      # their largest component variance, on average
known_cut = lambda ev, n_vox: ((ev > THR).sum(axis=-1), np.ones(ev.shape[:-1]))      # keep what exceeds it; the noise SD is 1 by construction


def complex_series(run):
    return run.mag().astype(float) * np.exp(1j * run.phase().astype(float))


def nordic_style(run, noise):
    """Patch PCA of the complex series divided by the noise map, with the known-noise threshold."""
    den, _, kept = patch_pca(complex_series(run) / noise[..., None], known_cut)
    return np.abs(den) * noise[..., None], kept


def temporal_noise_map(run, smooth=(1, 1, 0)):
    """Noise SD per channel from the complex control images over time, smoothed in-plane."""
    c = complex_series(run)[..., ctrl_idx]
    var = (np.abs(c - c.mean(axis=-1, keepdims=True)) ** 2).sum(axis=-1) / (2 * (len(ctrl_idx) - 1))
    return ndimage.gaussian_filter(np.sqrt(var), smooth)


noise80 = np.full(mask.shape, mag80[..., ctrl_idx][wm].std(axis=-1, ddof=1).mean())  # one number: the temporal SD of WM controls
noise8 = temporal_noise_map(runs["coils8-r2"])                                       # a map: the noise varies across the image
nordic80, nkept80 = nordic_style(runs["sigma80"], noise80)
nordic8, nkept8 = nordic_style(runs["coils8-r2"], noise8)
mag8 = series["coils8-r2"][0].astype(float)
den8, sig8, kept8 = patch_pca(mag8, mp_cut)
res8 = mag8[..., ctrl_idx] - mag8[..., ctrl_idx].mean(axis=-1, keepdims=True)
print(f"threshold: {THR:.2f} (Marchenko-Pastur upper edge for complex noise of unit SD per channel: {2 * (1 + np.sqrt(60 / 125)) ** 2:.2f})")
print(f"sigma80: noise level from the WM controls {noise80[0, 0, 0]:.1f}")
print(f"coils8-r2: noise map in the brain from {noise8[mask].min():.0f} to {noise8[mask].max():.0f}, mean {noise8[mask].mean():.0f}, GM {noise8[gm].mean():.0f}; "
      f"MP-PCA's own estimate in GM {sig8[gm].mean():.0f}; correlation of the noise between neighbors along y "
      f"{np.corrcoef(res8[inner].ravel(), np.roll(res8, 1, axis=1)[inner].ravel())[0, 1]:+.2f}, along x {np.corrcoef(res8[inner].ravel(), np.roll(res8, 1, axis=0)[inner].ravel())[0, 1]:+.2f}")
```

The threshold, 5.35 in units of the per-channel noise variance, is a little below the
Marchenko-Pastur upper edge of 5.73 for a complex matrix of this size, because the edge is
the limit for infinitely large matrices and a finite patch's largest component usually
falls short of it. The noise map of the GRAPPA run spans a factor of more than ten across
the brain, from 22 to 280 units, with a mean of 102; MP-PCA's own estimate in gray matter
was 78 where the map says 100. That run's noise breaks the other assumption too: GRAPPA
leaves the noise of neighboring voxels along the phase-encode axis correlated, here with a
coefficient of −0.40. The table compares the raw series, magnitude MP-PCA and the
NORDIC-style variant on both runs, with the mean of the far background as a measure of the
noise floor; the figure shows the three mean difference images of the GRAPPA run.

```{code-cell} python
:tags: [hide-input]
print(f"{'run':>10} {'series':>13} {'kept':>5} {'GM RMSE':>8} {'GM bias':>8} {'WM bias':>8} {'far background mean':>20}")
dm8 = {}
for name, run, rows in [("sigma80", runs["sigma80"], [("raw", mag80, None), ("MP-PCA", den80, kept80), ("NORDIC-style", nordic80, nkept80)]),
                        ("coils8-r2", runs["coils8-r2"], [("raw", mag8, None), ("MP-PCA", den8, kept8), ("NORDIC-style", nordic8, nkept8)])]:
    for label, vol4d, kept in rows:
        dm8[name, label], _, sc = cbf_scores(vol4d, run)
        print(f"{name:>10} {label:>13} {'—' if kept is None else f'{kept[mask].mean():.2f}':>5} {sc[0]:>8.1f} {sc[1]:>+8.1f} {sc[2]:>+8.1f} {vol4d[far].mean():>20.1f}")
print(f"noise-free far background mean: {mag_clean[far].mean():.1f}")
fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
for ax, label in zip(axes, ["raw", "MP-PCA", "NORDIC-style"]):
    show_slice(ax, dm8["coils8-r2", label], K, f"coils8-r2 mean: {label}", kind="diff", vmin=-60, vmax=60)
im = show_slice(axes[3], dm_clean, K, "noise-free mean (ref-clean)", kind="diff", vmin=-60, vmax=60)
side_colorbar(fig, im, "difference (image units)")
```

On the single-coil run the two methods give the same perfusion map: a gray matter error of
6.6 and 7.0 ml/100 g/min, with the same bias. That is the expected result. Tissue in an
ASL control image sits at an SNR above 60 even at σ = 80, where magnitude noise is already
Gaussian, so there is no floor for the complex data to avoid, and with uniform noise there
is nothing to normalize. The difference is in the background: magnitude MP-PCA leaves its
mean at 102 units, the Rayleigh floor of 1.25 σ, because a floor is a bias and not a
fluctuation, while the complex variant brings it down to 20, toward the 13 units of ringing
that are really there. For this protocol that matters little. It would matter for a series
whose static signal is itself near the noise level, which is what background suppression
aims for ([Chapter 9](../03-preprocessing/09-background-suppression.md)).

On the GRAPPA run the methods part. Magnitude MP-PCA finds no clean Marchenko-Pastur range
in patches whose noise is neither uniform nor independent: it keeps 17 components per patch,
most of them noise, and takes the gray matter error only from 44.7 to 27.3 ml/100 g/min.
The NORDIC-style variant, after normalization, keeps 3.4 and reaches 13.8. The extra
components over the single-coil run's 1.7 are consistent with the correlation of the noise
and the imperfection of an estimated noise map, either of which pushes some noise over the
threshold. The normalization has a price of its own in this example: the bias doubles, to
−8.6 ml/100 g/min in gray matter and +12.2 in white matter. Dividing by a noise map that
changes steeply across a patch changes the patch's static pattern, which then follows the
noise map as much as the anatomy, and less of the gray-white perfusion contrast rides on
the kept component. The image (third panel) is the smoothest of the three, and the cortex
in it is hard to tell from the white matter.

## Where denoising sits in an ASL pipeline

Local PCA denoising belongs at the very start, on the unprocessed control and label images.
Two orderings that look harmless are not, and the cell below tries both on the reference
series: denoising the 30 difference images instead of the 60 images, and denoising after a
resampling step, here a shift of half a voxel by linear interpolation, as motion or
distortion correction would apply.

```{code-cell} python
:tags: [hide-input]
d_den, sig_d, kept_d = patch_pca(d_ref.astype(float), mp_cut)             # the difference images instead of the images
sc_d = [quant.score(quant.cbf_pcasl(d_den.mean(axis=-1), ref.m0scan(), plds), cbf_clean, t) for t in (gm, wm)]
half_shift = lambda a: 0.5 * (a + np.roll(a, 1, axis=0))                  # linear interpolation halfway between neighbors along x
shifted, shifted_clean = half_shift(mag.astype(float)), half_shift(mag_clean.astype(float))
s_den, sig_s, kept_s = patch_pca(shifted, mp_cut)
print(f"{'series denoised':>28} {'kept':>5} {'σ estimate (true)':>18} {'image error in GM, before → after':>34}")
print(f"{'control and label images':>28} {kept_ref[mask].mean():>5.2f} {f'{sig_ref[gm].mean():.1f} (40.0)':>18} "
      f"{f'{(mag - mag_clean)[gm].std():.1f} → {(den_ref - mag_clean)[gm].std():.1f}':>34}")
print(f"{'the same after interpolation':>28} {kept_s[mask].mean():>5.2f} {f'{sig_s[gm].mean():.1f} ({40 / np.sqrt(2):.1f})':>18} "
      f"{f'{(shifted - shifted_clean)[gm].std():.1f} → {(s_den - shifted_clean)[gm].std():.1f}':>34}")
print(f"{'difference images':>28} {kept_d[mask].mean():>5.2f} {f'{sig_d[gm].mean():.1f} ({40 * np.sqrt(2):.1f})':>18}")
print(f"CBF from the denoised difference images: GM RMSE {sc_d[0]['rmse']:.1f}, GM bias {sc_d[0]['bias']:+.1f}, WM bias {sc_d[1]['bias']:+.1f} ml/100 g/min "
      f"(denoised before subtraction: {quant.score(cbf_den, cbf_clean, gm)['rmse']:.1f}, {quant.score(cbf_den, cbf_clean, gm)['bias']:+.1f}, {quant.score(cbf_den, cbf_clean, wm)['bias']:+.1f})")
```

Interpolation mixes the noise of neighboring voxels, and correlated noise does not fill a
Marchenko-Pastur range. On the interpolated series the rule keeps 11.7 components per patch
instead of 1.2, most of them noise, puts σ at 23.0 where it is 28.3, and removes less than
half of the noise (28.3 to 15.3 units) where it removed five sixths before (40.0 to 6.5).
The same goes for every step that interpolates or smooths: motion correction, distortion
correction, resampling to another grid, and scanner-side interpolation or zero-filling.
The GRAPPA run above showed the same failure with noise that the reconstruction had
correlated.

Denoising the difference images fails for another reason. Their noise is still independent
and the rule measures it correctly (55.7 against √2 × 40 = 56.6). But the subtraction has
removed the static pattern, and with it the kept component that carried part of the
perfusion contrast. Only 0.68 components are kept per patch on average, and where none is
kept the patch mean is all that is returned: the gray matter error is 13.0 ml/100 g/min
with a bias of −7.0, against 5.8 and −4.3 when the same denoiser runs before the
subtraction.

The risk is specific to ASL. In diffusion MRI, for which MP-PCA was designed, the signal of
interest is the image; in ASL it is a difference of half a percent of the image, and in a
series of 30 pairs its component in a patch falls below the noise edge, as the spectrum
showed. A shorter series lowers it further, since a component's variance grows with the
number of volumes that carry it. What the denoiser then returns is a locally fitted
perfusion map: low in noise, biased toward the patch, and with focal changes of the patch's
size or smaller largely removed. That can be an acceptable trade for a regional mean,
which needs no denoising to begin with, and it is a poor one for a voxelwise map of a small
lesion. Whatever is chosen, check it the way this section did, by the bias against a
reference and by the mean difference of the residual, not by how clean the images look or
by the temporal SNR.

Temporal averaging therefore remains the main tool. Averaging more pairs lowers the noise
as $1/\sqrt{N}$ without a model of the neighborhood and leaves no bias behind, where
smoothing and patch fitting each leave theirs. Outlier rejection, which removes the pairs
that averaging handles badly, is the topic of [Chapter 10](../03-preprocessing/10-motion.md).

## What this implies for acquisition

- **The number of pairs is the noise budget.** CBF noise per voxel falls as $1/\sqrt{N}$
  and nothing in postprocessing is free; choose the pairs from the SNR the question needs,
  and remember that regional means need far fewer than voxelwise maps.
- **Voxel volume buys SNR linearly.** Doubling the in-plane voxel area halves σ relative to
  the signal, the same gain as four times the pairs, at the cost of partial volume
  ([Chapter 7](../02-labeling/07-acquisition-parameters.md), [Chapter 12](../03-preprocessing/12-partial-volume.md)).
- **Parallel imaging is not free.** Skipping half of k-space raises the noise by √2 times
  the g-factor and makes it vary across the image; use it for the shorter readout and
  smaller distortion ([Chapter 11](../03-preprocessing/11-susceptibility-distortion.md)),
  not for SNR, and compute tSNR maps rather than one σ.
- **Measure σ where the data allow it**: in a background free of ringing and ghosts, or in a
  series where nothing but noise changes. Report the tSNR of the difference series, which is
  what the CBF estimate actually sees.
- **Smooth last, and say how much.** Any kernel comparable to the cortical thickness biases
  gray matter CBF downward; keep it small for voxelwise maps and report the bias.
- **If the series may be denoised, acquire it for that.** Save the phase images, switch off
  scanner-side interpolation and filtering, and keep whatever the reconstruction can export
  about the noise, such as a g-factor map. Denoise first, before any resampling and before the
  subtraction, and do not let denoising stand in for pairs: at this protocol's noise level
  it returns a patch-level fit, not a sharper measurement.

## Further reading

The Rician distribution and its floor {cite:p}`henkelman1985,gudbjartsson1995` and its
exact correction {cite:p}`koay2006`; multi-coil combination {cite:p}`roemer1990`, the noise
of root-sum-of-squares images {cite:p}`constantinides1997`, SENSE {cite:p}`pruessmann1999`,
GRAPPA {cite:p}`griswold2002` and its g-factor {cite:p}`breuer2009`; the Marchenko-Pastur
law {cite:p}`marchenko1967`, MP-PCA and its noise estimate
{cite:p}`veraart2016,veraart2016b`, and NORDIC {cite:p}`moeller2021,vizioli2021`; white
matter perfusion {cite:p}`vanosch2009`; the white-paper recommendations on averaging and
voxel size {cite:p}`alsop2015`; outlier rejection of difference images
{cite:p}`tan2009,dolui2017`.
