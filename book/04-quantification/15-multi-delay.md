---
title: "15. Multi-delay ASL: transit time and CBF together"
kernelspec:
  name: python3
  display_name: Python 3
---

:::{admonition} Simulated datasets in this chapter
:class: note
- **Built in this page:** the kinetic curve of a pure gray and a pure white matter voxel sampled at six delays, and the residual surface of the fit over (CBF, ATT) with and without noise ([Appendix B](../appendices/b-data-manifest.md#app-b-package-data)).
- **`multi-pld`**: one PCASL series with five pairs at each of six delays from 0.5 to 3.0 s, from which CBF and the arterial transit time are fitted ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-multi-pld)).
- **`ref-pcasl`**: the single-delay reference (30 pairs at PLD 1.8 s), for the comparison at equal numbers of pairs ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-ref-pcasl)).

Pipeline-tier datasets are simulated offline by aslscan ([Chapter 0.2](../00-frontmatter/the-simulated-datasets.md)).
:::

## Learning goals

After this chapter you can:

- explain why sampling the kinetic curve at several delays makes the transit time an
  estimable quantity, and what a single delay cannot tell you
- fit the kinetic model to a multi-delay series by a grid search over the transit time
  with CBF solved linearly at each candidate
- read a residual surface over (CBF, ATT) and see the coupling between the two estimates
  as the shape of its valley
- say which delays identify the transit time, and what it costs in CBF precision to spread
  a fixed scan time over several delays
- read an ATT map and a multi-delay CBF map against the truth, and against the single-delay
  formula of [Chapter 14](./14-cbf-quantification.md)

```{code-cell} python
:tags: [hide-cell]
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

from aslbook import data, kinetic, phantom, plotting, presets, protocols, quant
from aslbook.plotting import INK, PALETTE, TISSUE_COLORS, set_style, show_slice

set_style()
K = phantom.DISPLAY_SLICE
GM, WM = presets.TISSUES["GM"], presets.TISSUES["WM"]
TAU, ALPHA = presets.REFERENCE.labeling_duration, presets.ALPHA["PCASL"]
ATT_GRID = np.arange(0.2, 3.0001, 0.05)      # the default grid of quant.fit_multi_pld


def div0(n, d):
    n, d = np.asarray(n, float), np.asarray(d, float)
    return np.divide(n, d, out=np.zeros(np.broadcast(n, d).shape), where=d != 0)


def by_delay(diff, plds_of_pairs):
    """Average the pairs of each delay: (x, y, z, pairs) -> (x, y, z, delays), and the delays."""
    delays = np.unique(plds_of_pairs)
    return np.stack([diff[..., plds_of_pairs == v].mean(-1) for v in delays], -1), delays


def curve(pld, cbf, att, t1, m0, offset=0.0):
    """The PCASL kinetic curve in image units at the delays pld, read `offset` s after the first slice."""
    return kinetic.delta_m(np.asarray(pld) + TAU + offset, cbf, att, t1, m0, tau=TAU, alpha=ALPHA)


def smooth_inplane(vol, sigma=1.0):
    """Gaussian smoothing within each slice (a 2D readout's slices have different delays)."""
    return ndimage.gaussian_filter(vol, (sigma, sigma) + (0,) * (vol.ndim - 2))


def score_line(name, est, truth, roi, unit, fmt=".1f"):
    s = quant.score(est, truth, roi)
    return f"{name:>28}: bias {s['bias']:+{fmt}}, RMSE {s['rmse']:{fmt}} {unit} (n {s['n']})"
```

## Sampling the curve

[Chapter 5](../02-labeling/05-kinetic-model.md) described the label-control difference as a
curve in time: nothing until the bolus arrives at the transit time $\delta$, a rise while it
arrives, a decay after. A single-delay acquisition reads one point of that curve and
converts it to CBF with a formula that assumes where on the curve the point lies
([Chapter 14](./14-cbf-quantification.md)): after the bolus has arrived, and decaying with
the T1 of blood. A multi-delay acquisition reads several points and fits the curve itself
{cite:p}`buxton1998`, so the transit time becomes a measured quantity and CBF no longer
depends on assuming it. The `multi-pld` dataset does this with the reference protocol's
labeling (LD 1.8 s) at six delays, 0.5 to 3.0 s in steps of 0.5 s, five pairs each, in one
series of 60 volumes at TR 6 s. The figure evaluates the model for the two pure tissues and
marks the six samples; the shaded band is the arriving phase of each tissue, where the
curve's shape carries the transit time.

```{code-cell} python
:tags: [hide-input]
plds6 = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
fine = np.linspace(0.0, 3.2, 321)
fig, ax = plt.subplots(figsize=(7.5, 3.4))
for name, t in [("GM", GM), ("WM", WM)]:
    m0_t = 100 * t.m0 * np.exp(-presets.REFERENCE.echo_time / presets.T2_BLOOD)     # this tissue's M0 in image units
    ax.plot(fine, curve(fine, t.perfusion, t.att, t.t1, m0_t), color=TISSUE_COLORS[name], label=f"{name}: CBF {t.perfusion:g}, ATT {t.att:g} s")
    ax.plot(plds6, curve(plds6, t.perfusion, t.att, t.t1, m0_t), "o", color=TISSUE_COLORS[name], mec=INK["primary"], ms=7)
    ax.axvspan(t.att - TAU, t.att, color=TISSUE_COLORS[name], alpha=0.08, lw=0)
    ax.axvline(t.att, color=TISSUE_COLORS[name], lw=0.8, ls="--")
ax.set(xlabel="post-labeling delay (s)", ylabel="ΔM (image units)", xlim=(0, 3.2), ylim=(0, 85),
       title="the kinetic curve (LD 1.8 s), sampled at six delays; shaded: the bolus is still arriving")
ax.legend(loc="upper right")
fig.tight_layout()
for name, t in [("GM", GM), ("WM", WM)]:
    print(f"{name}: samples in the arriving phase (PLD < ATT): {int((plds6 < t.att).sum())} of 6; "
          f"ΔM at the six delays, pure voxel: {np.round(curve(plds6, t.perfusion, t.att, t.t1, 100 * t.m0 * np.exp(-0.012 / 0.165)), 1)}")
```

For gray matter (transit time 0.8 s) one sample, at 0.5 s, falls before the bolus has fully
arrived; for white matter (1.2 s) two do. The curve's value at any single delay depends on
CBF and ATT together, but its shape, the position of the peak and the slope on either side,
depends on ATT alone, so the fit separates them only if the samples cover the peak. The
readout matters here as much as the nominal delays: in a 2D acquisition each slice reads the
curve later by its slice offset ([Chapter 14](./14-cbf-quantification.md)), so in the
upper slices the 0.5 s sample has moved past the gray matter peak, with a consequence shown
below.

## Fitting the model

The fit finds the CBF and ATT whose model curve is closest, in the least-squares sense, to
the measured $\Delta M$ at the delays. `quant.fit_multi_pld` does it in a way that is both
fast and free of local minima:

1. **Grid over ATT.** For each candidate transit time on a grid (0.2 to 3.0 s in 0.05 s
   steps), evaluate the model curve at the delays with unit CBF.
2. **CBF by a linear solve.** The difference is proportional to CBF (except through
   $T_1'$, which depends on it weakly), so the best CBF at that ATT is the least-squares
   scale factor between the unit curve and the data, $f = \langle d, g\rangle / \langle g,
   g\rangle$, and the residual follows in closed form.
3. **Pick the ATT with the smallest residual**, then re-evaluate the curve with $T_1'$ at
   the fitted CBF and rescale once.

The grid makes every candidate ATT visible, so the fit cannot get stuck, and it makes the
per-voxel cost a few matrix products: the whole slab fits in under a second. Its two
weaknesses are also visible: the ATT is quantized to the grid, and an ATT that lands on
either end of the grid means the data did not constrain it.

:::{dropdown} The least-squares algebra
With data $d_i$ at delays $i$ and the unit-CBF model $g_i(\delta)$, the residual at a
candidate $\delta$ after solving for CBF is
$R(\delta) = \sum_i d_i^2 - \bigl(\sum_i d_i g_i\bigr)^2 / \sum_i g_i^2$. The fit minimizes
$R$ over the grid. With a single delay, $R(\delta) = 0$ for every $\delta$ at which
$g(\delta) \neq 0$: one data point fits one parameter exactly, whatever the other is, which
is the algebraic form of "a single delay cannot measure the transit time".
:::

## See it: the residual surface and the coupling

The figure maps the residual over the (CBF, ATT) plane for one pure gray matter voxel,
sampled at the six delays; yellow is a small residual. The left panel uses noise-free data
read in the first slice; the middle panel the same noise-free voxel read in the display
slice, 0.36 s later; the right panel adds noise to the first-slice data at the level of the
`multi-pld` dataset (five pairs per delay: $\sigma = 40\sqrt{2}/\sqrt{5} \approx 25$ image
units per delay). The white contours enclose the (CBF, ATT) pairs whose residual is within
one and two standard errors of the minimum at that noise level, the region the noise
cannot distinguish from the best fit; the cross is the truth and the dot the fit.

```{code-cell} python
:tags: [hide-input]
m0_gm = 100 * GM.m0 * np.exp(-presets.REFERENCE.echo_time / presets.T2_BLOOD)
sigma = 40 * np.sqrt(2) / np.sqrt(5)
cbf_ax, att_ax = np.linspace(0, 130, 131), np.arange(0.2, 3.0001, 0.02)
rng = np.random.default_rng(15)
noise = sigma * rng.standard_normal(6)
panels = [("noise-free, first slice", 0.0, 0.0), ("noise-free, display slice (+0.36 s)", 0.0, 0.36), ("with noise, first slice", 1.0, 0.0)]
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
for ax, (title, n_amp, off) in zip(axes, panels):
    d = curve(plds6, GM.perfusion, GM.att, GM.t1, m0_gm, off) + n_amp * noise
    model = curve(plds6[None, None, :], cbf_ax[:, None, None], att_ax[None, :, None], GM.t1, m0_gm, off)   # (cbf, att, delay)
    resid = ((model - d) ** 2).sum(-1) / sigma**2
    fit_cbf, fit_att = quant.fit_multi_pld(d.reshape(1, 1, 1, 6), plds6, np.array(m0_gm).reshape(1, 1, 1), tau=TAU, t1_tissue=GM.t1, slice_offsets=[off])
    im = ax.imshow(np.log10(resid - resid.min() + 1), origin="lower", aspect="auto", cmap="viridis_r",
                   extent=(att_ax[0], att_ax[-1], cbf_ax[0], cbf_ax[-1]))
    ax.contour(att_ax, cbf_ax, resid - resid.min(), levels=[2.3, 6.2], colors="white", linewidths=1.0)
    ax.plot(GM.att, GM.perfusion, "+", color="white", ms=14, mew=2.2, label="truth")
    ax.plot(fit_att.ravel(), fit_cbf.ravel(), "o", color=PALETTE[7], mec="white", ms=7, label="fit")
    ax.set(xlabel="candidate ATT (s)", ylabel="candidate CBF (ml/100 g/min)", title=title, xlim=(0.2, 3.0), ylim=(0, 130))
    ax.grid(False)
    print(f"{title:>36}: fit CBF {fit_cbf.ravel()[0]:.1f}, ATT {fit_att.ravel()[0]:.2f} s; "
          f"within one standard error: CBF {cbf_ax[np.any(resid - resid.min() <= 2.3, axis=1)].min():.0f}-{cbf_ax[np.any(resid - resid.min() <= 2.3, axis=1)].max():.0f}, "
          f"ATT {att_ax[np.any(resid - resid.min() <= 2.3, axis=0)].min():.2f}-{att_ax[np.any(resid - resid.min() <= 2.3, axis=0)].max():.2f} s")
axes[0].legend(loc="upper right", fontsize=8)
fig.colorbar(im, ax=axes[2], shrink=0.8, label="log10 residual above the minimum")
fig.tight_layout()

# how the noise moves the fit: many draws of the same voxel
draws = curve(plds6, GM.perfusion, GM.att, GM.t1, m0_gm)[None, :] + sigma * rng.standard_normal((2000, 6))
mc_cbf, mc_att = quant.fit_multi_pld(draws.reshape(-1, 1, 1, 6), plds6, np.full((2000, 1, 1), m0_gm), tau=TAU, t1_tissue=GM.t1, slice_offsets=[0.0])
draws9 = curve(plds6, GM.perfusion, GM.att, GM.t1, m0_gm, 0.36)[None, :] + sigma * rng.standard_normal((2000, 6))
mc_cbf9, mc_att9 = quant.fit_multi_pld(draws9.reshape(-1, 1, 1, 6), plds6, np.full((2000, 1, 1), m0_gm), tau=TAU, t1_tissue=GM.t1, slice_offsets=[0.36])
print(f"2000 noisy draws, first slice:   CBF {mc_cbf.mean():.1f} ± {mc_cbf.std():.1f}, ATT {mc_att.mean():.2f} ± {mc_att.std():.2f} s, "
      f"correlation of the two errors {np.corrcoef(mc_cbf.ravel(), mc_att.ravel())[0, 1]:+.2f}, ATT on the grid edge {100 * ((mc_att <= 0.2001) | (mc_att >= 2.9999)).mean():.0f} %")
print(f"2000 noisy draws, display slice: CBF {mc_cbf9.mean():.1f} ± {mc_cbf9.std():.1f}, ATT {mc_att9.mean():.2f} ± {mc_att9.std():.2f} s, "
      f"ATT on the grid edge {100 * ((mc_att9 <= 0.2001) | (mc_att9 >= 2.9999)).mean():.0f} %")
```

Look at the shape of the yellow region in the left panel. It is not a round basin but a
valley running diagonally, from low ATT with low CBF to high ATT with high CBF: a later
arrival means the bolus has spent less time decaying in tissue at each delay, so a smaller
CBF explains the same signal, and a larger CBF with an earlier arrival explains it equally
well. This is the **coupling** between the two estimates. Even without noise, the region
that 25 units of noise could not tell from the minimum spans CBF 38 to 88 and ATT 0.2 to
1.4 s, along the valley. The valley has a floor, because the 0.5 s sample lies on the
rising part of the curve and pins the arrival. In the middle panel the same voxel is read
0.36 s later, the earliest sample is at 0.86 s, past the gray matter arrival, and every
sample is on the decaying part of the curve. There the floor is gone: in the arrived phase
the curve's shape is $e^{-t/T_1'}$ whatever the ATT, and ATT changes only the amplitude,
which CBF absorbs. Every ATT below 0.86 s fits exactly, the yellow band that runs to the
grid's left edge, and the fit picks
one of them by rounding error (here 0.35 s, with the CBF of 64 that goes with it on the
line). With noise (right panel) the fit slides along
the valley: this draw lands at CBF 63 and ATT 1.5 s, and the printed 2000 draws give
CBF 65 ± 17 and ATT 0.82 ± 0.42 s with a correlation of +0.56 between the two errors, the
coupling as a number. Read in the display slice the spread is wider still and the mean of
the ATT means nothing: below 0.86 s the noise picks any value, and above it any candidate
that a noise draw happens to favor.

## The ATT map and why it matters

The transit time is a physiological quantity in its own right. It lengthens with age, with
distance from the feeding arteries (the watershed regions between territories arrive last),
in white matter, and wherever a stenosis or an occlusion forces blood through collateral
routes; a region with a long ATT and normal CBF is a different clinical finding from a
region with a low CBF {cite:p}`macintosh2010,wang2003`. It also matters for the CBF map
itself. In a single-delay acquisition a long transit time produces the **transit-delay
artifact**: bright arterial signal where the label is still in the vessels and dark tissue
downstream that the bolus has not reached, which a reader may take for hypoperfusion. A
multi-delay fit assigns the late-arriving signal to a late ATT instead of to a low CBF, and
reports both. What the book's model cannot show is the arterial signal itself: the phantom
has no macrovascular compartment ([Chapter 0.2](../00-frontmatter/the-simulated-datasets.md)),
so the label here is always in tissue when it is read, and the intravascular component that
multi-delay fits handle with an extra compartment {cite:p}`chappell2010` is absent.

## See it: the multi-pld dataset

The sidecar of a multi-delay series gives `PostLabelingDelay` as a list with one entry per
volume; `quant.pld_of_pairs` reads the delay of each control-label pair from it, and the
pairs of each delay are averaged into one difference image per delay. The figure shows the
six on the display slice, with the same window.

```{code-cell} python
:tags: [hide-input]
run = data.load_dataset("multi-pld").run("multipld")
mag, ctx, p = run.mag(), run.context(), run.sidecar()
fr, mask = run.fractions(), run.mask()
gm, wm = fr["gm"] > 0.9, fr["wm"] > 0.9
perfused = fr["gm"] + fr["wm"]
pmask = perfused > 0.5                                             # the voxels with something to fit
deltam, plds = by_delay(quant.subtract(mag, ctx), quant.pld_of_pairs(p, ctx))
offs = np.asarray(protocols.slice_offsets(p))
m0 = quant.m0_correction(run.m0scan(), tr=run.m0scan_sidecar()["RepetitionTimePreparation"], te=p["EchoTime"])
truth_cbf, truth_att = run.truth("perfusion"), run.truth("att")
t1_map = div0(fr["gm"] * GM.t1 + fr["wm"] * WM.t1, perfused)      # T1 of the perfused part of each voxel

print(f"{mag.shape[-1]} volumes, {len(ctx) // 2} pairs, delays {plds} s, {int((quant.pld_of_pairs(p, ctx) == plds[0]).sum())} pairs each; "
      f"TR {p['RepetitionTimePreparation']:g} s, scan time {mag.shape[-1] * p['RepetitionTimePreparation'] / 60:.0f} min")
fig, axes = plotting.mosaic({f"PLD {v:.1f} s": np.where(mask, deltam[..., i], np.nan) for i, v in enumerate(plds)}, k=K, kind="magnitude", ncols=3, figsize_per=2.6)
for ax, i in zip(axes.ravel(), range(6)):
    ax.images[0].set_clim(0, 80)
print("mean ΔM per delay, GM:", np.round(deltam[gm].mean(0), 1), " WM:", np.round(deltam[wm].mean(0), 1),
      f" (noise SD per voxel and delay ≈ {deltam[gm].std(0).mean():.0f} image units)")
```

The gray matter difference is 73 image units at the shortest delay and 12 at the longest,
while the noise of a five-pair average is about 26 per voxel: the late delays are read at a
signal-to-noise ratio below one in individual voxels, and the white matter never rises
above one. Multi-delay ASL spends its pairs across the curve, so each point is far noisier
than the 30-pair single-delay image, and the fit has to recover two parameters from six
such points.

The fit runs on every voxel with more than half of its volume perfused, with the tissue T1
of each voxel taken from the pipeline's tissue fractions (a fraction-weighted mean of the
gray and white matter values; a real pipeline would use a fixed 1.3 s or a T1 map), each
slice's readout offset, and the calibrated M0 scan.

```{code-cell} python
:tags: [hide-input]
cbf, att = quant.fit_multi_pld(deltam, plds, m0, tau=p["LabelingDuration"], t1_tissue=t1_map, slice_offsets=offs, mask=pmask)
fig, axes = plotting.fit_vs_truth(cbf, truth_cbf, pmask, "multi-delay CBF", k=K, unit="(ml/100 g/min)", kind="cbf")
print(score_line("CBF, GM (fraction > 0.9)", cbf, truth_cbf, gm, "ml/100 g/min"))
print(score_line("CBF, WM (fraction > 0.9)", cbf, truth_cbf, wm, "ml/100 g/min"))
print(f"GM mean {cbf[gm].mean():.1f} (truth {truth_cbf[gm].mean():.1f}), WM mean {cbf[wm].mean():.1f} (truth {truth_cbf[wm].mean():.1f}); "
      f"SD across GM voxels {cbf[gm].std():.1f}")
```

The CBF map has the right tissue contrast and, unlike the single-delay map of
[Chapter 14](./14-cbf-quantification.md), no systematic shortfall: the model that is fitted
is the model that generated the data, T1 of tissue included. What remains is noise, and a
positive bias in both tissues that the noise creates: the fit clips CBF at zero and, along
the valley, a noise draw toward a later ATT comes with a larger CBF, so the errors are
skewed upward, +8 in gray matter. In white matter, where the signal is below the noise, the
bias (+23) is larger than the truth.

```{code-cell} python
:tags: [hide-input]
fig, axes = plotting.fit_vs_truth(att, truth_att, pmask, "ATT", k=K, unit="(s)", kind="att")
print(score_line("ATT, GM (fraction > 0.9)", att, truth_att, gm, "s", ".2f"))
print(score_line("ATT, WM (fraction > 0.9)", att, truth_att, wm, "s", ".2f"))
edge = (att <= ATT_GRID[0] + 1e-9) | (att >= ATT_GRID[-1] - 1e-9)
print(f"ATT on the grid edge: {100 * edge[pmask].mean():.0f} % of fitted voxels ({100 * (att[pmask] <= ATT_GRID[0] + 1e-9).mean():.0f} % at 0.2 s, "
      f"{100 * (att[pmask] >= ATT_GRID[-1] - 1e-9).mean():.0f} % at 3.0 s); in GM {100 * edge[gm].mean():.0f} %, in WM {100 * edge[wm].mean():.0f} %")
first_sample = plds[0] + offs
ident = first_sample < GM.att                                     # slices whose earliest GM sample precedes arrival
lo, hi = gm & ident[None, None, :], gm & ~ident[None, None, :]
print(f"GM ATT where the first sample precedes arrival (slices {np.flatnonzero(ident)[0]}-{np.flatnonzero(ident)[-1]}): "
      f"bias {quant.score(att, truth_att, lo)['bias']:+.2f}, RMSE {quant.score(att, truth_att, lo)['rmse']:.2f} s; "
      f"where it does not (slices {np.flatnonzero(~ident)[0]}-{np.flatnonzero(~ident)[-1]}): "
      f"bias {quant.score(att, truth_att, hi)['bias']:+.2f}, RMSE {quant.score(att, truth_att, hi)['rmse']:.2f} s")
```

The ATT map is the harder one to read. Its estimate panel shows the display slice, one of
the slices in which the earliest gray matter sample comes after arrival, and there the gray
matter values scatter over the whole grid; the scatter panel shows the same as vertical
streaks at the two true values. The gray matter bias is small (+0.09 s) but the RMSE is
0.55 s, most of a grid's worth of scatter; 13 % of all fitted voxels sit on the grid's
edges, a quarter of the white matter ones. The printed split by slice makes the
identifiability point on the pipeline data: in the slices read early enough for the 0.5 s
sample to precede the gray matter arrival the RMSE is 0.46 s with no bias; in the later
slices it is 0.64 s with a bias of +0.18 s. The difference is smaller than the toy
suggested because the noise dominates both. White matter, with its signal below the noise,
is unusable voxel by voxel at this scan length. Two remedies follow, and the next cells show
both: average over a region, or borrow strength from neighboring voxels.

## See it: the fitted curves

Averaging the difference over a tissue's voxels within one slice gives a curve with a
noise of a fraction of an image unit, and the fit to it is what the model promises. The
left panels show the gray matter curve in the first slice and the white matter curve in
the slice with most white matter, with the fitted curve through the points. The right panel
repeats the region fit in every slice with enough voxels and plots the fitted ATT against
the slice's readout offset: the identifiability of the previous section, now without noise
to hide behind.

```{code-cell} python
:tags: [hide-input]
def roi_fit(roi, t1, min_voxels=100):
    """Fit the ROI-mean curve of every slice with enough voxels; returns per-slice (cbf, att, curve, m0, n)."""
    n = np.array([roi[:, :, z].sum() for z in range(roi.shape[2])])
    ok = n >= min_voxels
    d = np.array([deltam[:, :, z][roi[:, :, z]].mean(0) if ok[z] else np.zeros(len(plds)) for z in range(len(n))])
    m = np.array([m0[:, :, z][roi[:, :, z]].mean() if ok[z] else 1.0 for z in range(len(n))])
    c, a = quant.fit_multi_pld(d[None, None], plds, m[None, None], tau=TAU, t1_tissue=t1, slice_offsets=offs,
                               mask=ok[None, None], att_grid=np.arange(0.2, 3.0001, 0.01))
    return c[0, 0], a[0, 0], d, m, n

fits = {"GM": roi_fit(gm, GM.t1), "WM": roi_fit(wm, WM.t1)}
show = {"GM": 0, "WM": int(np.argmax(fits["WM"][4]))}
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
for ax, name in zip(axes[:2], ["GM", "WM"]):
    c, a, d, m, n = fits[name]
    z = show[name]
    ax.plot(plds, d[z], "o", color=TISSUE_COLORS[name], mec=INK["primary"], ms=7, label=f"measured, {name} mean of slice {z} ({n[z]} voxels)")
    ax.plot(fine, curve(fine, c[z], a[z], presets.TISSUES[name].t1, m[z], offs[z]), color=TISSUE_COLORS[name], label=f"fit: CBF {c[z]:.1f}, ATT {a[z]:.2f} s")
    ax.plot(fine, curve(fine, presets.TISSUES[name].perfusion, presets.TISSUES[name].att, presets.TISSUES[name].t1, m[z], offs[z]), ":", color=INK["secondary"], label="truth")
    ax.set(xlabel="post-labeling delay (s)", ylabel="ΔM (image units)", xlim=(0, 3.2), ylim=(0, None), title=f"{name} region, slice {z} (readout offset {offs[z]:.2f} s)")
    ax.legend(fontsize=7.5, loc="lower left")
    print(f"{name} region, slice {z}: CBF {c[z]:.1f} (truth {truth_cbf[:, :, z][gm[:, :, z] if name == 'GM' else wm[:, :, z]].mean():.1f}), ATT {a[z]:.2f} s (truth {presets.TISSUES[name].att:g})")
ax = axes[2]
for name in ["GM", "WM"]:
    c, a, d, m, n = fits[name]
    ok = n >= 100
    ax.plot(offs[ok], a[ok], "o-", color=TISSUE_COLORS[name], label=f"{name} region fit")
    ax.axhline(presets.TISSUES[name].att, color=TISSUE_COLORS[name], lw=0.8, ls=":")
ax.axvline(GM.att - plds[0], color=TISSUE_COLORS["GM"], lw=0.8, ls="--")
ax.text(GM.att - plds[0] + 0.02, 2.6, "0.5 s sample\nafter GM arrival →", fontsize=7.5, color=TISSUE_COLORS["GM"])
ax.set(xlabel="slice readout offset (s)", ylabel="fitted ATT (s)", ylim=(0, 3.1), title="region fits per slice; dotted: truth")
ax.legend(fontsize=7.5, loc="upper left")
fig.tight_layout()
c, a, d, m, n = fits["GM"]
ok = n >= 100
print(f"GM region fits: CBF {c[ok].mean():.1f} ± {c[ok].std():.1f} over {ok.sum()} slices; ATT {a[ok & ident].mean():.2f} ± {a[ok & ident].std():.2f} s "
      f"in the {int((ok & ident).sum())} slices read before arrival, {a[ok & ~ident].mean():.2f} ± {a[ok & ~ident].std():.2f} s in the {int((ok & ~ident).sum())} later ones")
c, a, d, m, n = fits["WM"]
ok = n >= 100
print(f"WM region fits: CBF {c[ok].mean():.1f} ± {c[ok].std():.1f}, ATT {a[ok].mean():.2f} ± {a[ok].std():.2f} s over {ok.sum()} slices")
```

The region curves sit on the model, and the fits return the phantom's values: CBF 59 and
ATT 0.79 s for the first slice's gray matter, 19 and 1.14 s for the white matter of slice
11, and over all slices 61 ± 3 and 19 ± 2 ml/100 g/min. The right panel is the lesson of
the chapter in one line: the gray matter ATT is 0.77 ± 0.06 s in the eight slices whose
first sample precedes its arrival and 0.62 ± 0.37 s, anywhere between the grid's start and
the truth, in the eleven read later, even though the noise in those region curves is
negligible. White matter, whose 1.2 s arrival is preceded by the 0.5 s sample in every
slice, gives 1.21 ± 0.14 s throughout. Identifiability is a property of
where the samples fall on the curve, not of the signal-to-noise ratio; a longer scan would
not fix it, an earlier first delay (or a 3D readout, which has no slice offsets) would.

## Borrowing strength from neighbors

Multi-delay pipelines rarely fit voxels independently. BASIL's spatial prior
{cite:p}`chappell2009` regularizes the parameters toward their neighbors, and simpler
pipelines smooth the difference images before fitting. The cell does the simplest version,
a Gaussian of one voxel within each slice, and refits.

```{code-cell} python
:tags: [hide-input]
cbf_s, att_s = quant.fit_multi_pld(smooth_inplane(deltam), plds, smooth_inplane(m0), tau=TAU, t1_tissue=t1_map, slice_offsets=offs, mask=pmask)
fig, axes = plotting.fit_vs_truth(att_s, truth_att, pmask, "ATT, smoothed 1 voxel", k=K, unit="(s)", kind="att")
edge_s = (att_s <= ATT_GRID[0] + 1e-9) | (att_s >= ATT_GRID[-1] - 1e-9)
for name, roi in [("GM", gm), ("WM", wm)]:
    a0, a1 = quant.score(att, truth_att, roi), quant.score(att_s, truth_att, roi)
    c0, c1 = quant.score(cbf, truth_cbf, roi), quant.score(cbf_s, truth_cbf, roi)
    print(f"{name}: ATT bias {a0['bias']:+.2f} -> {a1['bias']:+.2f}, RMSE {a0['rmse']:.2f} -> {a1['rmse']:.2f} s; "
          f"CBF bias {c0['bias']:+.1f} -> {c1['bias']:+.1f}, RMSE {c0['rmse']:.1f} -> {c1['rmse']:.1f} ml/100 g/min (unsmoothed -> smoothed)")
print(f"ATT on the grid edge: {100 * edge[pmask].mean():.0f} % -> {100 * edge_s[pmask].mean():.0f} % of fitted voxels")
```

Smoothing brings the gray matter ATT RMSE from 0.55 to 0.31 s and the white matter's from
1.14 to 0.55 s, takes most voxels off the grid edges (13 % to 5 %), and the ATT map now
shows the anatomy: cortex early, deep white matter late. It also
introduces the partial-volume bias of [Chapter 12](../03-preprocessing/12-partial-volume.md):
the gray matter CBF falls below the truth because its voxels now contain their white
matter and CSF neighbors, and the white matter CBF is pulled up by the gray matter next to
it. A spatial prior does the same trade more carefully, but it is the same trade; the
remedy at the source is signal, which means more pairs per delay, a 3D readout with
background suppression, or larger voxels.

## Fitting a single delay, and the comparison

What the fit does with a single-delay dataset is worth seeing once. The cell fits the
model to the 2.0 s delay alone.

```{code-cell} python
:tags: [hide-input]
i20 = int(np.flatnonzero(plds == 2.0)[0])
cbf_1, att_1 = quant.fit_multi_pld(deltam[..., i20:i20 + 1], plds[i20:i20 + 1], m0, tau=TAU, t1_tissue=t1_map, slice_offsets=offs, mask=pmask)
print(f"one delay (2.0 s): ATT at the first grid point in {100 * (att_1[pmask] <= 0.2001).mean():.0f} % of voxels, elsewhere wherever rounding error put it "
      f"(SD across voxels {att_1[pmask].std():.2f} s); "
      f"CBF GM {cbf_1[gm].mean():.1f}, WM {cbf_1[wm].mean():.1f} ml/100 g/min")

ref = data.load_dataset("ref-pcasl").run("pcasl")
pr = ref.sidecar()
dm_ref = quant.subtract(ref.mag(), ref.context()).mean(-1)
m0_ref = quant.m0_correction(ref.m0scan(), tr=ref.m0scan_sidecar()["RepetitionTimePreparation"], te=pr["EchoTime"])
cbf_ref = quant.cbf_pcasl(dm_ref, m0_ref, quant.slice_plds(pr["PostLabelingDelay"], protocols.slice_offsets(pr))[None, None, :],
                          tau=pr["LabelingDuration"], alpha=pr["LabelingEfficiency"])
gm_r, wm_r = ref.fractions()["gm"] > 0.9, ref.fractions()["wm"] > 0.9
truth_r = ref.truth("perfusion")

methods = {"truth": (truth_cbf[gm].mean(), truth_cbf[wm].mean(), 0, 0),
           "single delay, formula\n(ref-pcasl, 30 pairs at 1.8 s)": (cbf_ref[gm_r].mean(), cbf_ref[wm_r].mean(), cbf_ref[gm_r].std(), cbf_ref[wm_r].std()),
           "multi-delay fit, per voxel\n(30 pairs over 6 delays)": (cbf[gm].mean(), cbf[wm].mean(), cbf[gm].std(), cbf[wm].std()),
           "multi-delay fit, smoothed": (cbf_s[gm].mean(), cbf_s[wm].mean(), cbf_s[gm].std(), cbf_s[wm].std()),
           "multi-delay fit, region curves": (fits["GM"][0][fits["GM"][4] >= 100].mean(), fits["WM"][0][fits["WM"][4] >= 100].mean(), 0, 0)}
fig, ax = plt.subplots(figsize=(8.5, 3.6))
x = np.arange(len(methods))
for j, (name, col) in enumerate([("GM", TISSUE_COLORS["GM"]), ("WM", TISSUE_COLORS["WM"])]):
    vals = [v[j] for v in methods.values()]
    sds = [v[j + 2] for v in methods.values()]
    ax.bar(x + (j - 0.5) * 0.36, vals, 0.34, color=col, label=f"{name} mean (bar: SD across voxels)")
    ax.errorbar(x + (j - 0.5) * 0.36, vals, yerr=sds, fmt="none", ecolor=INK["primary"], elinewidth=1, capsize=3)
ax.set_xticks(x, list(methods), fontsize=7.5)
ax.axhline(GM.perfusion, color=TISSUE_COLORS["GM"], lw=0.8, ls=":"); ax.axhline(WM.perfusion, color=TISSUE_COLORS["WM"], lw=0.8, ls=":")
ax.set(ylabel="CBF (ml/100 g/min)", ylim=(0, 100))
ax.legend(loc="upper left")
fig.tight_layout()
print(f"{'':>34} {'GM mean':>8} {'GM SD':>7} {'WM mean':>8} {'WM SD':>7}")
for name, v in methods.items():
    print(f"{name.replace(chr(10), ' '):>34} {v[0]:8.1f} {v[2]:7.1f} {v[1]:8.1f} {v[3]:7.1f}")
```

With one delay the residual is zero for every candidate ATT up to rounding error, so the
grid search returns whichever candidate rounding favored: 12 % of voxels land on the first
grid point and the rest are spread over the whole grid, and the CBF (69 in gray matter,
108 in white matter) follows that arbitrary choice. Any pipeline offered a single-delay
series and asked for an ATT will produce a map; it will not be a measurement.

The comparison bars are the chapter's accounting. The two acquisitions have the same
number of pairs, 30. The single-delay formula returns a gray matter mean of 44 with a
spread of 16 across voxels; the multi-delay fit returns a mean of 67 with a spread of 19,
and in white matter it trades a large negative bias (7 for 21) for a large positive one
(43) with a spread of 39. The multi-delay fit is unbiased in the sense that matters,
its model is the right one, but with 30 pairs its per-voxel precision is worse, because
each pair contributes to one of six points and because two parameters are fitted from the
same data. Region curves and smoothing recover the precision; the choice between them
depends on whether the question is regional or voxelwise.

## How many delays, and where

The chapter's figures give the rules that {cite:t}`woods2024` set out in detail:

- **The first delay must precede the earliest arrival** you care about, in every slice: the
  ATT is identified by samples on the rising part of the curve. With a 2D readout, subtract
  the largest slice offset from the shortest nominal delay before deciding whether it does.
- **The last delay must follow the latest arrival**, so that the tissue with the longest
  transit time has samples on both sides of its peak; 2.5 to 3.0 s in adults, longer in
  cerebrovascular disease.
- **Five or more delays** between those limits; more averages at the longer delays, where
  the signal is weakest, than at the shorter ones. The `multi-pld` dataset's equal
  averaging is the simplest choice, not the best one.
- **The scan is longer, or the map is noisier.** Spreading a fixed number of pairs over
  delays costs CBF precision; a multi-delay protocol that matches the precision of a
  single-delay one needs more time, and it is the ATT map and the freedom from the
  arrival assumption that pay for it.

Two designs measure CBF without the transit time at all. A long labeling and a long delay,
the white paper's own recommendation (LD 1.8 s, PLD 1.8 s or more), put the readout after
every arrival, at the cost of a decayed signal and the T1 bias of
[Chapter 14](./14-cbf-quantification.md) {cite:p}`alsop1996`. Time-encoded (Hadamard)
labeling ([Chapter 18](../05-advanced/18-time-encoded-and-look-locker.md)) encodes several
sub-boluses into one series and decodes a multi-delay curve from it with less noise per
delay than acquiring the delays one by one {cite:p}`dai2013`; Look-Locker readouts sample
the curve after a single label {cite:p}`gunther2001`, and the model-free approach of
{cite:t}`petersen2006` deconvolves the measured arterial input instead of assuming the
bolus shape.

## Measure it: the numbers

The numbers this chapter printed, collected. For the `multi-pld` dataset fitted voxel by
voxel with the tissue T1 of each voxel: CBF bias +8.3 and RMSE 21.1 ml/100 g/min in gray
matter, +22.6 and 45.1 in white matter; ATT bias +0.09 and RMSE 0.55 s in gray matter
(truth 0.8 s), +0.54 and 1.14 s in white matter (truth 1.2 s); 13 % of fitted voxels with
the ATT on the grid's edge (7 % at 0.2 s, 6 % at 3.0 s). With one voxel of in-plane
smoothing: ATT RMSE 0.31 and 0.55 s, CBF bias −4.2 and +16.2, 5 % on the edge. The region
fits recover 61 ± 3 and 0.77 ± 0.06 s in gray matter (in the slices where the ATT is
identifiable) and 19 ± 2 and 1.21 ± 0.14 s in white matter. The single-delay formula on the
reference dataset, with the same 30 pairs, gives 44 and 7 in the two tissues.

## What this implies for acquisition

- **Acquire several delays when the transit time is unknown or is the question**: elderly
  and cerebrovascular populations, watershed regions, anything where a single-delay map
  might show a transit-delay artifact.
- **Put the shortest delay before the earliest arrival in every slice**, and the longest
  after the latest; with a 2D readout account for the slice offsets when you do.
- **Weight the averages toward the long delays**, where the signal is smallest.
- **Budget the time.** At equal scan time the multi-delay CBF map is noisier than the
  single-delay one; plan for regional analysis, spatial regularization, or a longer scan.
- **Do not fit a transit time to a single-delay series.** The software will return one.

## Further reading

The kinetic model and its multi-delay fit are {cite:t}`buxton1998`; the recommendations
for multi-timepoint ASL are {cite:t}`woods2024`. Transit-time mapping and its physiology:
{cite:t}`macintosh2010` and {cite:t}`wang2003`; the macrovascular component that this
book's phantom lacks, {cite:t}`chappell2010`; the spatially regularized Bayesian fit,
{cite:t}`chappell2009`. The arrival-insensitive designs: {cite:t}`alsop1996`,
{cite:t}`dai2013`, {cite:t}`gunther2001`, and {cite:t}`petersen2006`.
