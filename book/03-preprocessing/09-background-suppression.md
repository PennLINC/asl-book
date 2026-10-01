---
title: "9. Background suppression"
kernelspec:
  name: python3
  display_name: Python 3
---

:::{admonition} Simulated datasets in this chapter
:class: note
- **Built in this page:** the longitudinal timelines of the three tissues under two inversion pulses (`kinetic.tissue_mz`), the pulse-time landscape, and the per-slice residual of a 2D readout ([Appendix B](../appendices/b-data-manifest.md#app-b-package-data)).
- **`bgsup`**: the reference protocol without pulses (`off`), with two inversion pulses at 2.25 and 3.50 s from the start of labeling (`on`, efficiency 0.95), with perfect pulses (`perfect`, efficiency 1), and with a presaturation pulse added (`presat`) ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-bgsup)).
- **`motion`**: the same random head motion applied to the reference series without (`random`) and with (`random-bgsup`) the two pulses ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-motion)).

Pipeline-tier datasets are simulated offline by aslscan ([Chapter 0.2](../00-frontmatter/the-simulated-datasets.md)).
:::

## Learning goals

After this chapter you can:

- explain why the static tissue signal, not the thermal noise, limits ASL in practice, and how inversion pulses during the delay remove it
- follow the longitudinal magnetization of each tissue through labeling, pulses and readout, and choose pulse times that null two tissues
- state what the pulses do to the label itself, including the sign flip with an odd number of pulses, and correct a quantification for it
- explain why 2D readouts suppress unevenly across slices and why magnitude images can lose the sign of the difference
- measure what suppression costs in SNR and in calibration, and what it buys against motion

```{code-cell} python
:tags: [hide-cell]
import numpy as np
import matplotlib.pyplot as plt

from aslbook import data, kinetic, phantom, presets, quant
from aslbook.plotting import INK, PALETTE, TISSUE_COLORS, set_style, show_image, show_slice, take_slice

set_style()
K = phantom.DISPLAY_SLICE
R = presets.REFERENCE
TE, T2B, SCALE, TR = R.echo_time, presets.T2_BLOOD, R.signal_scale, R.repetition_time
T_FIRST = R.labeling_duration + R.post_labeling_delay  # first slice excitation, s from the start of labeling
PULSES = [2.25, 3.50]
CORONAL = 34  # the coronal plane through the middle of the slab
ASPECT = R.voxel_mm[2] / R.voxel_mm[0]


def mz_timeline(t1, tr, t_read, pulses, eps, t_grid, presat=False):
    """Mz/M0 of a tissue at every time in ``t_grid`` (s from the start of labeling), starting
    from what the previous excitation at ``t_read - tr`` left (zero under presaturation),
    recovering with ``t1`` and multiplied by ``1 - 2 eps`` at each pulse. The value at
    ``t_read`` equals ``kinetic.tissue_mz``."""
    rec = lambda mz, dt: 1.0 - (1.0 - mz) * np.exp(-dt / t1)
    out = np.empty(len(t_grid))
    for i, t in enumerate(t_grid):
        mz, t0 = (0.0 if presat else rec(0.0, tr - t_read)), 0.0
        for p in sorted(pulses):
            if p < t:
                mz, t0 = rec(mz, p - t0) * (1 - 2 * eps), p
        out[i] = rec(mz, t - t0)
    return out


def signed(run):
    """The signed image: the real part of the complex image aslscan wrote as magnitude and phase."""
    return run.mag() * np.cos(run.phase())


def truth_deltam(run):
    """The noise-free difference in image units: the ``deltam`` truth of the label rows, scaled
    like the images and decayed with the blood T2 to the echo time."""
    rows = [i for i, r in enumerate(run.context()) if r == "label"]
    return run.truth("deltam")[..., rows].mean(axis=-1) * SCALE * np.exp(-TE / T2B)


def control_mean(vol, ctx):
    return vol[..., [i for i, r in enumerate(ctx) if r == "control"]].mean(axis=-1)


def side_colorbar(fig, im, label, right=0.9):
    fig.tight_layout(rect=(0, 0, right, 1))
    cax = fig.add_axes([right + 0.01, 0.2, 0.012, 0.6])
    fig.colorbar(im, cax=cax, label=label)
```

## The physics: the static signal is the enemy

The perfusion signal of the reference protocol is 30 image units in gray matter, sitting on
a static tissue signal of about 6200 ([Chapter 8](../03-preprocessing/08-noise.md)). The
subtraction of control and label removes the static part only if it is identical in the two
images. Thermal noise does not care about the static signal: it adds the same √2 σ ≈ 57
units to every pair whatever the tissue signal is. Everything else does care. If the head
moves a fraction of a voxel between the control and the label, an edge voxel's tissue signal
changes by tens of percent; pulsation, breathing, and drifts of the field or the gain change
it by a fraction of a percent. **A change of 1 % of 6200 is 62 units, twice the perfusion
signal**; the same change of a static signal of 600 would be 6 units. Every fluctuation
that scales with the static signal is reduced in proportion by reducing that signal at the
moment of the readout, provided the label keeps its size {cite:p}`ye2000`.

That is **background suppression**: one or more inversion pulses applied to the imaging
region during the post-labeling delay, timed so that the longitudinal magnetization of the
tissues passes through zero at the readout {cite:p}`dixon1991,ye2000`. The label rides
along. A 180° pulse of efficiency ε turns a longitudinal magnetization $M_z$ into
$(1 - 2\varepsilon) M_z$, and between pulses $M_z$ recovers toward $M_0$ with the tissue's
T1:

$$
M_z(t) = M_0 - \bigl(M_0 - M_z(t_0)\bigr)\, e^{-(t - t_0)/T_1} .
$$

The tissue does not start at $M_0$: in the steady state the previous 90° excitation left
$M_z = 0$ one TR before this readout, and it has recovered for $\mathrm{TR} - t_\mathrm{read}$
when labeling starts. Two pulses can null two T1 values at one time; the figure follows gray
matter, white matter and CSF of the phantom (T1 1.33, 0.83 and 3.0 s) through the reference
protocol with the pipeline's pulses at 2.25 and 3.50 s from the start of labeling, and, on
the right, what the same pulses do to the label.

```{code-cell} python
:tags: [hide-input]
t_mid = T_FIRST + R.readout_duration / 2
t_grid = np.linspace(0, TR, 901)
fig, (ax_t, ax_l) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.6, 1]})
ax_t.axvspan(0, R.labeling_duration, color=INK["grid"], alpha=0.8, lw=0)
ax_t.axvspan(T_FIRST, T_FIRST + R.readout_duration, color=PALETTE[3], alpha=0.25, lw=0)
ax_t.text(R.labeling_duration / 2, 1.06, "labeling", ha="center", fontsize=8, color=INK["secondary"])
ax_t.text(T_FIRST + R.readout_duration / 2, 1.06, "readout", ha="center", fontsize=8, color=INK["secondary"])
for p in PULSES:
    ax_t.axvline(p, color=INK["primary"], lw=1, ls="--")
    ax_t.text(p, -1.08, f"pulse {p:g} s", ha="center", fontsize=8)
for name, t in presets.TISSUES.items():
    ax_t.plot(t_grid, mz_timeline(t.t1, TR, t_mid, PULSES, 0.95, t_grid), color=TISSUE_COLORS[name], label=f"{name}, T1 {t.t1:g} s")
    ax_t.plot(t_grid, mz_timeline(t.t1, TR, t_mid, [], 0.95, t_grid), color=TISSUE_COLORS[name], lw=1, ls=":")
ax_t.axhline(0, color=INK["secondary"], lw=0.8)
ax_t.set(xlabel="time from the start of labeling (s)", ylabel="Mz / M0", title="tissue magnetization with two pulses (dotted: without)", xlim=(0, TR), ylim=(-1.15, 1.15))
ax_t.legend(loc="lower right")
t_lab = np.linspace(0, TR, 901)
factor = np.array([np.prod([1 - 2 * 0.95 for p in PULSES if p < t]) for t in t_lab])
ax_l.plot(t_lab, factor, color=PALETTE[6], label="label × ∏(1 − 2ε), ε = 0.95")
ax_l.axvspan(T_FIRST, T_FIRST + R.readout_duration, color=PALETTE[3], alpha=0.25, lw=0)
ax_l.axhline(0, color=INK["secondary"], lw=0.8)
ax_l.set(xlabel="time from the start of labeling (s)", ylabel="factor on the control − label difference", title="the label's factor", xlim=(0, TR), ylim=(-1.15, 1.15))
ax_l.legend(loc="lower left")
fig.tight_layout()
for name, t in presets.TISSUES.items():
    mz = [float(kinetic.tissue_mz(1.0, t.t1, TR, T_FIRST + o, PULSES, 0.95)) for o in (0.0, R.readout_duration / 2, R.readout_duration)]
    print(f"{name:>3}: Mz/M0 at the first, middle and last slice {mz[0]:+.3f}, {mz[1]:+.3f}, {mz[2]:+.3f}; without pulses {kinetic.tissue_se(1.0, t.t1, TR):.3f}")
print(f"label factor for {len(PULSES)} pulses at ε 0.95: {kinetic.label_factor(PULSES, 0.95):.2f}; at ε 1: {kinetic.label_factor(PULSES, 1.0):.0f}; "
      f"for one pulse at ε 0.95: {kinetic.label_factor(PULSES[:1], 0.95):+.2f}")
```

Without pulses (dotted) every tissue is near $M_0$ by the readout: 0.97 for gray matter,
1.00 for white matter, 0.78 for CSF with its long T1. With them, the first pulse at 2.25 s
inverts all three from near-full recovery; white matter, with the shortest T1, recovers
fastest and is the first to cross zero; the second pulse at 3.50 s inverts them again just
before the readout, and now the recovering curves of gray and white matter both cross zero
early in the readout window. In the middle of the readout they stand at +0.11 and +0.14 of
$M_0$, and CSF, which no two-pulse scheme can null together with them, at +0.16. The static
signal has fallen by about a factor of eight.

The right panel is the price. The label is a *difference* in longitudinal magnetization
between the control and the label condition, and each pulse multiplies that difference by
$(1 - 2\varepsilon)$ exactly as it multiplies the tissue: after $N$ pulses the difference is
scaled by

$$
(1 - 2\varepsilon)^N ,
$$

which is $0.81$ for two pulses at ε = 0.95 (`kinetic.label_factor`), $1$ for two perfect
pulses, and $-0.90$ for a single pulse at ε = 0.95. **An odd number of pulses flips the
sign of the difference: control − label becomes negative.** The loss is inflicted whether or
not the tissue is nulled, so a quantification step must divide the difference by the
factor, and the sidecar records what to divide by (`BackgroundSuppressionNumberPulses`, and
in these simulations `AslscanSimulation.BackgroundSuppressionLabelFactor`). The white
paper's guidance follows the same trade: every pulse costs some of the label, more pulses
null a wider range of T1 values, and it names two pulses as a good compromise
{cite:p}`alsop2015`; measuring the efficiency of real pulses and its effect on the label
is the subject of {cite:t}`garcia2005`.

The definition of ε matters when numbers from different sources are compared. In this
book and in aslscan, ε is the fraction of the longitudinal magnetization that is inverted,
$M_z^+ = (1 - 2\varepsilon)\, M_z^-$, so ε = 0.95 multiplies the control − label
difference by $|1 - 2\varepsilon| = 0.90$ per pulse, and by 0.81 for two. The white
paper's statement that each pulse costs about 5 % of the ASL signal {cite:p}`alsop2015`
corresponds to ε ≈ 0.975 in this definition, so the simulated pulses lose twice as much
label per pulse as the ones it describes.

## See it: choosing the pulse times

For a target time the two pulse times are a two-dimensional search; the numerical
optimization of the pulse timing for ASL, over a range of T1 values, is the subject of
{cite:t}`maleki2012`. The left panel is the
root mean square of $M_z/M_0$ over gray and white matter at the middle of the readout,
3.98 s, for every pair of pulse times before the first excitation at 3.60 s, on a
logarithmic color scale (`kinetic.optimal_suppression_times` performs the same search, but
bounds the pulses only by the target time, so for a 2D readout the bound at the first
excitation has to be imposed as here). The right panel shows how the residual at the middle slice and the label factor depend
on the pulse efficiency.

```{code-cell} python
:tags: [hide-input]
grid = np.linspace(0.0, T_FIRST - 0.01, 90)
cost = np.full((grid.size, grid.size), np.nan)
for i, a in enumerate(grid):
    for j, b in enumerate(grid):
        if b > a:
            mz = [kinetic.tissue_mz(1.0, t1, TR, t_mid, [a, b], 0.95) for t1 in (presets.TISSUES["GM"].t1, presets.TISSUES["WM"].t1)]
            cost[i, j] = np.sqrt(np.mean(np.square(mz)))
i_best, j_best = np.unravel_index(np.nanargmin(cost), cost.shape)  # the optimum among pulses before the first excitation
best, resid = (grid[i_best], grid[j_best]), np.nanmin(cost)
cost_pipe = np.sqrt(np.mean([kinetic.tissue_mz(1.0, t1, TR, t_mid, PULSES, 0.95) ** 2 for t1 in (presets.TISSUES["GM"].t1, presets.TISSUES["WM"].t1)]))

fig, (ax_c, ax_e) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.15, 1]})
im = ax_c.pcolormesh(grid, grid, np.log10(cost.T), cmap="viridis_r", vmin=-2.5, vmax=0, shading="nearest")
ax_c.plot(*best, "o", color="white", mec=INK["primary"], ms=8, label=f"optimum {best[0]:.2f}, {best[1]:.2f} s")
ax_c.plot(*PULSES, "s", color=PALETTE[1], mec=INK["primary"], ms=8, label=f"pipeline {PULSES[0]:g}, {PULSES[1]:g} s")
ax_c.axvspan(0, R.labeling_duration, color="white", alpha=0.15, lw=0)
ax_c.set(xlabel="first pulse (s)", ylabel="second pulse (s)", title="RMS of Mz/M0 over GM and WM at 3.98 s (log10)")
ax_c.set_aspect("equal")
ax_c.legend(loc="lower right")
fig.colorbar(im, ax=ax_c, shrink=0.85)
eps_axis = np.linspace(0.8, 1.0, 41)
for name, t in presets.TISSUES.items():
    ax_e.plot(eps_axis, [kinetic.tissue_mz(1.0, t.t1, TR, t_mid, PULSES, e) for e in eps_axis], color=TISSUE_COLORS[name], label=f"{name}: Mz/M0 at 3.98 s")
ax_e.plot(eps_axis, [kinetic.label_factor(PULSES, e) for e in eps_axis], color=PALETTE[6], ls="--", label="label factor (1 − 2ε)²")
ax_e.set(xlabel="inversion efficiency ε", ylabel="fraction", title=f"pulses at {PULSES[0]:g} and {PULSES[1]:g} s: the effect of ε", ylim=(0, 1.05))
ax_e.legend(loc="center left")
fig.tight_layout()
print(f"optimum for the middle of the readout: pulses at {best[0]:.2f} and {best[1]:.2f} s, residual RMS {resid:.3f}; "
      f"the pipeline's {PULSES[0]:g} and {PULSES[1]:g} s: residual {cost_pipe:.3f}")
```

The landscape has a narrow valley along its upper edge: the second pulse must come late,
and the later it is the more freedom the first pulse has. The constrained optimum for the
middle slice puts the pulses at 2.18 and 3.59 s, the second 10 ms before the first slice is
excited, with a residual of 0.004. The pipeline uses 2.25 and 3.50 s, rounded to a quarter
second and leaving a margin before the excitation, because in the simulator no pulse can
fall after the first slice's excitation; the residual is 0.13 at the middle slice, an
eightfold reduction of the static signal rather than one of more than two hundred. In the
right panel the label factor falls from 1 at ε = 1 to 0.81 at 0.95
and 0.36 at 0.80, while the nulling barely changes: the tissue residual at the middle slice
is a matter of timing, the label loss is a matter of pulse quality.

## See it: the slice dependence of a 2D readout

The pulses null the tissue at one instant, and a 2D readout excites its 20 slices over
0.76 s. Each slice sees the tissue at a different point of the recovery curve. The figure
evaluates `kinetic.tissue_mz` at every slice's excitation time for the three tissues, with
the pipeline's pulses (solid) and with a presaturation pulse at the start of labeling that
sets the tissue to zero instead of letting it recover from the previous readout (dashed).

```{code-cell} python
:tags: [hide-input]
offsets = np.array(R.slice_timing)
fig, ax = plt.subplots(figsize=(7.5, 3.4))
for name, t in presets.TISSUES.items():
    ax.plot(range(R.n_slices), [kinetic.tissue_mz(1.0, t.t1, TR, T_FIRST + o, PULSES, 0.95) for o in offsets], "o-", color=TISSUE_COLORS[name], label=name)
    ax.plot(range(R.n_slices), [kinetic.tissue_mz(1.0, t.t1, TR, T_FIRST + o, PULSES, 0.95, presaturation=True) for o in offsets], "--", color=TISSUE_COLORS[name], lw=1)
ax.axhline(0, color=INK["secondary"], lw=0.8)
ax.axvline(K, color=INK["secondary"], lw=0.8, ls=":")
ax.text(K + 0.2, -0.36, "display slice", fontsize=8, color=INK["secondary"])
ax.set(xlabel="slice (excited 40 ms apart, from inferior)", ylabel="Mz / M0 at excitation", title="residual tissue magnetization per slice (dashed: with presaturation)", ylim=(-0.4, 0.5), xticks=range(0, 20, 2))
ax.legend(loc="upper left")
fig.tight_layout()
zero_gm = next(z for z, o in enumerate(offsets) if kinetic.tissue_mz(1.0, presets.TISSUES["GM"].t1, TR, T_FIRST + o, PULSES, 0.95) > 0)
print(f"GM crosses zero between slices {zero_gm - 1} and {zero_gm}; at the last slice GM is at "
      f"{kinetic.tissue_mz(1.0, presets.TISSUES['GM'].t1, TR, T_FIRST + offsets[-1], PULSES, 0.95):+.2f} and WM at "
      f"{kinetic.tissue_mz(1.0, presets.TISSUES['WM'].t1, TR, T_FIRST + offsets[-1], PULSES, 0.95):+.2f} of M0")
```

The first slices are read while gray and white matter are still inverted, at −0.17 and
−0.35 of $M_0$; both cross zero between slices 5 and 6; the last slice, 0.76 s after the
first, sees them recovered to +0.33 and +0.46. Only a few slices near the crossing are truly
suppressed. Presaturation changes the starting point of the recovery and moves the crossing
by one slice without removing the spread. This is a property of any 2D readout: the
suppression is a snapshot and the readout is not. A 3D readout excites the whole volume at
one instant, so one pair of pulse times serves every voxel, which is why the white paper
recommends background suppression together with a segmented 3D readout
{cite:p}`alsop2015,vidorreta2013`, and why the pipeline's 2D series shows the effect at its
worst.

## The setting that produces it

In BIDS the pulses are `BackgroundSuppression: true`, `BackgroundSuppressionNumberPulses` and
`BackgroundSuppressionPulseTime` (seconds from the start of labeling) {cite:p}`clement2022`; the pipeline's
`bgsup` dataset passes `background_suppression: [2.25, 3.50]` to the protocol and the
simulator records what it resolved in `AslscanSimulation.BackgroundSuppression`.

```{code-cell} python
:tags: [hide-input]
bgsup = data.load_dataset("bgsup")
runs = {name: bgsup.run(name) for name in ["off", "on", "perfect", "presat"]}
factors = {name: run.simulation()["BackgroundSuppressionLabelFactor"] or 1.0 for name, run in runs.items()}
for name, run in runs.items():
    p, sim = run.sidecar(), run.simulation()
    bs = sim["BackgroundSuppression"] or {}  # null when the run has no pulses
    print(f"{name:>8}: pulses {p.get('BackgroundSuppressionPulseTime', 'none')}, efficiency {bs.get('InversionEfficiency', {}).get('Value', '-')}, "
          f"presaturation {bs.get('Presaturation', {}).get('Value', '-')}, label factor {factors[name]:.2f}, model {sim['BackgroundSuppressionModel']}")
```

## See it: the control images

The four runs share the phantom, the noise seed and every other setting. The images below
are the mean of the 30 control volumes at the display slice, all with the same intensity
window; the numbers are the gray matter, white matter and CSF means. The `off` run is the
reference series of the rest of the book.

```{code-cell} python
:tags: [hide-input]
ctx = runs["off"].context()
fr, mask = runs["off"].fractions(), runs["off"].mask()
gm, wm, csf = fr["gm"] >= 0.9, fr["wm"] >= 0.9, fr["csf"] >= 0.9
sig = {name: signed(run) for name, run in runs.items()}
ctrl = {name: control_mean(s, ctx) for name, s in sig.items()}

fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
for ax, name in zip(axes, runs):
    im = show_slice(ax, np.abs(ctrl[name]), K, f"control, {name}", vmin=0, vmax=7000)
side_colorbar(fig, im, "magnitude (image units)")
for name in runs:
    print(f"{name:>8}: control signal at slice {K}: GM {ctrl[name][:, :, K][gm[:, :, K]].mean():6.0f}, WM {ctrl[name][:, :, K][wm[:, :, K]].mean():6.0f}, "
          f"CSF {ctrl[name][:, :, K][csf[:, :, K]].mean():6.0f}; whole slab GM {ctrl[name][gm].mean():6.0f}")
```

At the display slice the gray matter control signal drops from 6261 units to 654, a factor
of nearly ten; white matter and CSF are cut similarly. What is left is not zero, and it is
not the same in every slice. The next figure looks at the whole slab in a coronal section
through the middle of the brain, using the **signed** image, the real part of the complex
image that aslscan writes as magnitude and phase (`part-mag` and `part-phase`), so that an
inverted tissue shows as negative.

```{code-cell} python
:tags: [hide-input]
fig, axes = plt.subplots(1, 3, figsize=(11, 2.6))
for ax, name, lim in zip(axes, ["off", "on", "presat"], [7000, 3000, 3000]):
    im = show_image(ax, take_slice(ctrl[name], CORONAL, "coronal"), f"signed control, {name} (window ±{lim})", kind="diff", vmin=-lim, vmax=lim, aspect=ASPECT)
side_colorbar(fig, im, "signed signal (image units)")
```

Without suppression the coronal section is uniformly bright. With it, the inferior slices
(bottom) are negative, a band of near-zero signal runs through the slab, and the superior
slices recover toward positive values, the picture the per-slice curve predicted. The
measurement below compares the mean signed control signal of the pure gray and white matter
voxels in each slice with `kinetic.tissue_mz` evaluated at that slice's excitation time,
scaled by the tissue's $M_0$ and T2 decay like the images.

```{code-cell} python
:tags: [hide-input]
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
worst = 0.0
for ax, name in zip(axes, ["on", "perfect", "presat"]):
    sim = runs[name].simulation()["BackgroundSuppression"]
    eps, presat = sim["InversionEfficiency"]["Value"], sim["Presaturation"]["Value"]
    for tissue, sel in [("GM", gm), ("WM", wm)]:
        t = presets.TISSUES[tissue]
        pred = np.array([SCALE * t.m0 * np.exp(-TE / t.t2) * kinetic.tissue_mz(1.0, t.t1, TR, T_FIRST + o, PULSES, eps, presat) for o in offsets])
        meas = np.array([ctrl[name][:, :, z][sel[:, :, z]].mean() if sel[:, :, z].sum() > 20 else np.nan for z in range(R.n_slices)])
        ok = np.isfinite(meas)
        worst = max(worst, np.nanmax(np.abs(meas - pred)) / (SCALE * t.m0))
        ax.plot(range(R.n_slices), pred, "-", color=TISSUE_COLORS[tissue], lw=1.5, label=f"{tissue}: predicted")
        ax.plot(np.flatnonzero(ok), meas[ok], "o", color=TISSUE_COLORS[tissue], mec=INK["primary"], label=f"{tissue}: measured")
    ax.axhline(0, color=INK["secondary"], lw=0.8)
    ax.set(xlabel="slice", title=f"{name}: ε {eps:g}" + (", presaturation" if presat else ""), xticks=range(0, 20, 4))
axes[0].set(ylabel="signed control signal (image units)")
axes[0].legend(fontsize=7)
fig.tight_layout()
print(f"largest difference between the measured and predicted control signal, any tissue, run or slice: {100 * worst:.1f} % of the tissue's M0")
```

The points sit on the curves in every run to within 0.7 % of $M_0$: the simulator's
suppression model is the timeline of this chapter, applied per slice. Two features are worth
noting. Perfect pulses (middle) do not null better than ε = 0.95 pulses at the display
slice; they change the starting point of the last recovery and shift the crossing slightly.
Presaturation (right) lowers CSF most, because CSF has the most to lose from the previous
TR's recovery, and moves the gray and white matter crossing one slice later.

## See it: the difference images and the label factor

The difference image is where suppression must not do harm. The panels show the coronal
section of the mean difference of 30 pairs for the `off` run, for the `on` run subtracted
from the magnitude images as a scanner delivers them, from the signed images, and from the
signed images divided by the label factor 0.81.

```{code-cell} python
:tags: [hide-input]
dm_mag = {name: quant.subtract(run.mag(), ctx).mean(axis=-1) for name, run in runs.items()}
dm_sig = {name: quant.subtract(sig[name], ctx).mean(axis=-1) for name in runs}
panels = [("off", dm_mag["off"]), ("on, magnitude images", dm_mag["on"]), ("on, signed images", dm_sig["on"]), ("on, signed ÷ 0.81", dm_sig["on"] / factors["on"])]
fig, axes = plt.subplots(1, 4, figsize=(12, 2.6))
for ax, (title, vol) in zip(axes, panels):
    im = show_image(ax, take_slice(vol, CORONAL, "coronal"), title, kind="diff", vmin=-60, vmax=60, aspect=ASPECT)
side_colorbar(fig, im, "mean difference (image units)")
ratio = lambda a, b, sel: a[sel].mean() / b[sel].mean()
print(f"GM ratio of the mean difference to the unsuppressed run: magnitude images {ratio(dm_mag['on'], dm_mag['off'], gm):+.3f}; "
      f"signed images: on {ratio(dm_sig['on'], dm_sig['off'], gm):.3f}, perfect {ratio(dm_sig['perfect'], dm_sig['off'], gm):.3f}, presat {ratio(dm_sig['presat'], dm_sig['off'], gm):.3f}")
per_slice = [ratio(dm_mag["on"][:, :, z], dm_mag["off"][:, :, z], gm[:, :, z]) for z in range(R.n_slices)]
print("per-slice GM ratio from magnitude images: " + " ".join(f"{r:+.2f}" for r in per_slice))
```

The magnitude-image difference has the wrong sign in the inferior slices: where the tissue
was still inverted at readout, the label image (tissue minus label, both negative) has the
*larger* magnitude, and control − label comes out as −0.81 of the unsuppressed value instead
of +0.81. Averaged over gray matter, the magnitude difference of the `on` run is
−0.003 of the `off` run's, a perfusion image that has canceled itself. The signed images
restore it: 0.811 of the unsuppressed difference for `on`, 1.000 for `perfect` and 0.811 for
`presat`, the label factors of the timeline to three decimals, in every slice. Dividing by
the factor (last panel) returns the perfusion image of the unsuppressed run, with the same
speckle, because thermal noise is untouched by the pulses.

Real scanners deliver magnitude images by default, and the phase is not always saved. The
white paper's answer is to subtract the complex images, because the difference of two
magnitude images that are near zero is ambiguous in sign {cite:p}`alsop2015`; the signed
image formed above is that subtraction. Where only magnitudes are available, the pulses
can instead be timed so that every slice is read with a small *positive* residual rather
than at the exact null, at the price of less suppression; with a 3D readout one null time
serves every voxel and the margin can be small. Either way, a pipeline that receives a suppressed 2D
series must know how the pulses were timed relative to every slice, which is why the
sidecar carries `BackgroundSuppressionPulseTime` and `SliceTiming` together.

## Measure it: what suppression costs

Three costs follow from the physics, and the runs put numbers on each.

```{code-cell} python
:tags: [hide-input]
truth = truth_deltam(runs["off"])
noise_sd = {name: quant.subtract(sig[name], ctx)[gm].std(axis=-1, ddof=1).mean() for name in runs}
print(f"{'run':>8} {'M0 scan GM':>11} {'suppressed control GM':>22} {'ΔM GM':>7} {'noise SD per pair':>18} {'SNR per pair':>13} {'RMSE vs truth (÷ factor)':>25}")
scores = {}
for name in runs:
    est = dm_sig[name] / factors[name]
    scores[name] = quant.score(est, truth, mask)["rmse"]
    print(f"{name:>8} {runs[name].m0scan()[gm].mean():>11.0f} {ctrl[name][gm].mean():>22.0f} {dm_sig[name][gm].mean():>7.1f} {noise_sd[name]:>18.1f} "
          f"{dm_sig[name][gm].mean() / noise_sd[name]:>13.2f} {scores[name]:>25.1f}")
```

- **The M0 scan must be acquired without the pulses.** The separate M0 scan of every run
  reads 6450 units in gray matter, because aslscan acquires it as a plain spin-echo image at
  its own TR ([Chapter 16](../04-quantification/16-calibration.md)); the suppressed control
  images, whose signed gray matter mean over the slab is 250 units and which run from
  −1100 to +2100 units between the first and the last slice, cannot calibrate anything. A
  protocol that takes M0 from the control mean loses that option the moment it switches
  suppression on, and `m0scan` volumes included in the series must be acquired without pulses.
- **The label loses the factor, the noise does not.** The gray matter difference falls from
  30.2 to 24.4 units while the noise of a pair stays at 56 units, so the SNR per pair drops
  from 0.54 to 0.44, and after dividing by 0.81 the error against the truth rises from 10.5
  to 12.9 units RMSE, the same 1/0.81. Recovering the SNR would take 1/0.81² = 1.5 times the
  pairs. In a simulator with only thermal noise, suppression is a pure loss.
- **The simulator's global-bolus approximation.** aslscan inverts the whole labeled bolus at
  every pulse, wherever the bolus is, which is what $(1-2\varepsilon)^N$ assumes, and the
  sidecar says so in `AslscanSimulation.BackgroundSuppressionModel: "global-bolus"`. That is
  what a non-selective pulse does, and it is the case the white paper describes, in which
  the blood to be labeled experiences every inversion pulse {cite:p}`alsop2015`.
  Implementations differ. A slab-selective pulse leaves label that is still in transit
  below the slab uninverted, so that label arrives with fewer inversions, and with the
  opposite sign if it missed an odd number of them; $(1-2\varepsilon)^N$ then no longer
  describes the retained label, and the factor to divide by depends on where the bolus was
  at each pulse.

## See it: motion with and without suppression

The gain is against everything that scales with the static signal. The `motion` dataset
applies the same six random head displacements, of up to 2 mm and 2°, to the reference
series with and without the pulses; the poses are identical, only the static signal
differs. The truth for each run is its own noise-free difference (`deltam`, which follows
the moved anatomy), so what remains is the artifact: static signal that failed to cancel
because the control and the label were in different positions. The panels show the error
of the mean difference at the display slice; the bars, the RMSE of each pair's difference
against its own truth.

```{code-cell} python
:tags: [hide-input]
motion = data.load_dataset("motion")
mruns = {name: motion.run(name) for name in ["random", "random-bgsup"]}
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4))
per_pair = {}
for ax, (name, run) in zip(axes, mruns.items()):
    fac = run.simulation()["BackgroundSuppressionLabelFactor"] or 1.0
    d = quant.subtract(signed(run), run.context()) / fac
    rows = [i for i, r in enumerate(run.context()) if r == "label"]
    truth_rows = run.truth("deltam")[..., rows] * SCALE * np.exp(-TE / T2B)
    per_pair[name] = [quant.score(d[..., k], truth_rows[..., k], mask)["rmse"] for k in range(d.shape[-1])]
    err = d.mean(axis=-1) - truth_rows.mean(axis=-1)
    im = show_slice(ax, np.where(mask, err, np.nan), K, f"mean difference − truth: {name}", kind="diff", vmin=-150, vmax=150)
    scores[name] = quant.score(d.mean(axis=-1), truth_rows.mean(axis=-1), mask)["rmse"]
side_colorbar(fig, im, "error (image units)", right=0.86)
fig2, ax2 = plt.subplots(figsize=(8, 3.2))
x = np.arange(30)
ax2.bar(x - 0.2, per_pair["random"], 0.4, color=PALETTE[0], label="random")
ax2.bar(x + 0.2, per_pair["random-bgsup"], 0.4, color=PALETTE[1], label="random-bgsup")
ax2.set(xlabel="pair", ylabel="RMSE of the pair's difference (image units)", title="error of each pair against its own truth (log scale)", yscale="log", ylim=(30, 2000))
ax2.legend(loc="upper right")
fig2.tight_layout()
moved = pd = mruns["random"].motion()
moved = moved[(moved.iloc[:, 1:] != 0).any(axis=1)]["volume"].tolist()
print(f"moved volumes {moved} (pairs {sorted({v // 2 for v in moved})}); "
      f"largest displacement {np.abs(mruns['random'].motion()[['trans_x', 'trans_y', 'trans_z']].values).max():.2f} mm")
for name in mruns:
    pp = np.array(per_pair[name])
    still = np.delete(pp, sorted({v // 2 for v in moved}))
    print(f"{name:>13}: RMSE of the mean difference {scores[name]:5.1f}; per pair, still pairs {still.mean():5.1f}, moved pairs {pp[sorted({v // 2 for v in moved})].mean():6.1f}")
```

Six volumes move, in four pairs. Without suppression a moved pair's difference is wrong by
about 920 units RMSE on average (up to 1100), sixteen times the noise of a still pair and
thirty times the perfusion signal, and the four pairs alone push the error of the 30-pair
mean to 63 units, twice the signal it is supposed to show; the error map is the outline of
the brain, the edges where a 1 mm shift changes a 6000-unit signal the most. With
suppression the same displacements cost about 270 units per moved pair, and the mean's
error is 23 units. The still pairs show the other side of the ledger: 57 units of noise per
pair without suppression, 70 with it, the 1/0.81 of the previous section. Suppression
traded a 23 % increase of the noise for a reduction of the motion artifact by a factor of
3.4, and the ratio would
be larger still with a 3D readout, whose single null leaves less static signal than the
per-slice average here. The rest of the motion problem, detecting the moved pairs and
correcting or discarding them, is [Chapter 10](../03-preprocessing/10-motion.md).

## Measure it: the difference image against the truth

```{code-cell} python
:tags: [hide-input]
print(f"{'run':>14} {'RMSE of the mean difference vs truth, brain (image units)':>60}")
for name in ["off", "on", "perfect", "presat", "random", "random-bgsup"]:
    print(f"{name:>14} {scores[name]:>60.1f}")
```

For the still series the ranking is set by the label factor alone: 10.5 units for `off` and
`perfect`, 12.9 for `on` and `presat`. For the moving series it is set by the static signal:
63 without suppression, 23 with it, from the same poses.

## What this implies for acquisition

- **Use background suppression** in any protocol quantified from many pairs: physiological
  fluctuation and motion, not thermal noise, limit real ASL, and both scale with the static
  signal {cite:p}`alsop2015`.
- **Pair it with a 3D readout.** A 2D readout is nulled in a few slices and inverted or
  recovered in the rest; a segmented 3D readout sees one null everywhere
  {cite:p}`vidorreta2013`.
- **Leave a small positive residual** rather than aiming at the exact null, unless the phase
  is saved: a magnitude image cannot tell −0.2 $M_0$ from +0.2 $M_0$.
- **Acquire the M0 scan without pulses**, and record the pulse times, their number and their
  efficiency in the sidecar so that the quantification can divide by $(1-2\varepsilon)^N$.
- **Budget the loss.** Two pulses at ε = 0.95 cost 19 % of the label and 1.5 times the pairs
  for the same thermal SNR; more pulses suppress more tissues and cost more label, and an odd
  number flips the sign of the difference.

## Further reading

Multiple inversion recovery to null static tissue {cite:p}`dixon1991`; background
suppression for ASL {cite:p}`ye2000`; the efficiency of inversion pulses and its effect on
the label {cite:p}`garcia2005`; the optimization of the pulse timing {cite:p}`maleki2012`;
the sidecar fields that record the pulses {cite:p}`clement2022`; the white paper's recommendations on suppression and 3D
readouts {cite:p}`alsop2015`; 2D versus 3D readouts with suppression {cite:p}`vidorreta2013`.
