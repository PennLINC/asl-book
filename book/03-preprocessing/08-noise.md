---
title: "8. Thermal noise, averaging, and denoising"
kernelspec:
  name: python3
  display_name: Python 3
---

:::{admonition} Simulated datasets in this chapter
:class: note
- **Built in this page:** a cloud of noisy complex measurements and its magnitude histogram; a 60-pair synthetic series on the packaged phantom slab (`aslbook.synth`) for the averaging law ([Appendix B](../appendices/b-data-manifest.md#app-b-package-data)).
- **`ref-pcasl`**: the reference protocol with the standard noise, σ = 40 image units ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-ref-pcasl)).
- **`ref-clean`**: the same acquisition with the noise switched off: the noise-free reference every error below is measured against ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-ref-clean)).
- **`noise-sweep`**: the reference protocol at σ = 10, 40 and 80, plus a run with eight receive coils and GRAPPA 2 ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-noise-sweep)).

Pipeline-tier datasets are simulated offline by aslscan ([Chapter 0.2](../00-frontmatter/the-simulated-datasets.md)).
:::

## Learning goals

After this chapter you can:

- explain why magnitude noise is Rician, why that matters in the background and not in an ASL control image, and what it does to the control-label difference
- measure the noise level of an ASL series two ways and say when each is trustworthy
- state the averaging law and turn it into a CBF noise floor for a given number of pairs
- recognize the noise of a multi-coil, parallel-imaging acquisition, which is neither Gaussian nor uniform
- apply spatial smoothing and measure what it removes and what it costs at tissue edges

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
which is the **Rayleigh** distribution (mean 1.25 σ, standard deviation 0.66 σ) where there
is no signal and is indistinguishable from a Gaussian of standard deviation σ once the
signal exceeds a few σ. The magnitude operation therefore folds noise into a positive bias
where the signal is weak and leaves it alone where the signal is strong.

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
distributed with mean 1.25 σ, standard deviation 0.66 σ and mean square 2 σ², or **in a
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

The **eight-coil GRAPPA run** is different in two ways. Its magnitude is the root sum of
squares of eight coil images {cite:p}`roemer1990`, and the sum of squares of 16 Gaussian
components follows a **non-central chi** distribution, whose floor is higher than the
Rician one for the same per-coil noise. GRAPPA {cite:p}`griswold2002` then reconstructs the
skipped k-space lines from the coil data, which amplifies the noise by a factor that
depends on position (the **g-factor**, [Chapter 2](../01-mri-physics/02-epi-and-reconstruction.md))
and correlates it between neighboring voxels. The histogram above does not match the
Rayleigh curve drawn for the gray matter σ: its mean is 0.63 of that σ, because the
amplification differs between the background and the tissue, and its shape is a mixture of
non-central chi distributions at different noise levels. The "σ from the far background" of
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
noise, which makes it the standard quality measure of an ASL series. The maps below are the
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
voxel with this protocol, only as a regional average, which is the usual finding in ASL.

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
noise for a bias of 2.4 ml/100 g/min; an 8 mm FWHM kernel, common in group studies, is one
voxel here and costs 7.6 ml/100 g/min in cortex, and a 12 mm one costs 13. The right kernel
depends on the question: a regional mean needs none, a voxelwise map of a thin cortex can
afford little, and the bias should be reported alongside the map.

:::{admonition} Beyond a Gaussian kernel
:class: note
Smoothing is the crudest denoiser because it ignores everything except distance. Better
ones use structure the noise does not have: **outlier rejection** drops the pairs whose
difference is far from the rest (a motion-corrupted pair can carry ten times the noise of a
clean one; [Chapter 10](../03-preprocessing/10-motion.md)), **temporal filtering** removes
slow drifts, and **partial-volume correction** replaces smoothing's mixture with a model of
it ([Chapter 12](../03-preprocessing/12-partial-volume.md)). Methods that learn the local
structure of the series are the frontier ([Chapter 19](../05-advanced/19-frontiers.md)). All
are measured the way this chapter measured smoothing: error against a truth, split into
noise removed and structure destroyed.
:::

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

## Further reading

The Rician distribution and its floor {cite:p}`gudbjartsson1995`; multi-coil combination
{cite:p}`roemer1990` and GRAPPA {cite:p}`griswold2002`; the white-paper recommendations on
averaging and voxel size {cite:p}`alsop2015`; outlier rejection of difference images
{cite:p}`dolui2017`.
