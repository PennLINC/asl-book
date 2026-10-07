---
title: "9. Background suppression"
kernelspec:
  name: python3
  display_name: Python 3
---

:::{admonition} Simulated datasets in this chapter
:class: note
- **Built in this page:** the longitudinal timelines of the three tissues under two inversion pulses (`kinetic.tissue_mz`), the pulse-time landscape, and the per-slice residual of a 2D readout ([Appendix B](../appendices/b-data-manifest.md#app-b-package-data)).
- **`bgsup`**: the reference protocol without pulses (`off`), with two inversion pulses at 2.25 and 3.50 s from the start of labeling (`on`, efficiency 0.95), with perfect pulses (`perfect`, efficiency 1), with a presaturation pulse added (`presat`), and the same pulses under the simulator's bolus-position model, in which a pulse inverts only the label already inside the imaging slab (`on-slab`; `early-slab` moves the first pulse to 2.00 s so that the tail of the bolus misses it) ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-bgsup)).
- **`motion`**: the same random head motion applied to the reference series without (`random`) and with (`random-bgsup`) the two pulses ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-motion)).
- **`physio`**: the reference protocol with global cardiac, respiratory and drift factors on the tissue signal and a cardiac factor on the label, without (`off`) and with (`on`) the two pulses ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-physio)).
- **`readout-3d`**: the four-shot 3D GRASE and the stack-of-spirals readouts on the reference labeling, without pulses (`grase-4shot`, `spiral`) and with two pulses at 1.90 and 3.25 s, retimed for the slab's single excitation at 3.6 s (`grase-4shot-bgsup`, `spiral-bgsup`) ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-readout-3d)).

Pipeline-tier datasets are simulated offline by aslscan ([Chapter 0.2](../00-frontmatter/the-simulated-datasets.md)).
:::

## Learning goals

After this chapter you can:

- explain why the static tissue signal, not the thermal noise, limits ASL in practice, and how inversion pulses during the delay remove it
- follow the longitudinal magnetization of each tissue through labeling, pulses and readout, and choose pulse times that null two tissues
- state what the pulses do to the label itself, including the sign flip with an odd number of pulses, and correct a quantification for it
- explain why 2D readouts suppress unevenly across slices and why magnitude images can lose the sign of the difference
- measure what suppression costs in SNR and in calibration, and what it buys against motion
- tell the global-bolus model of the label factor from a bolus-position model, derive the retained label when the tail of the bolus misses a pulse, and measure what suppression does to physiological noise
- show that a 3D readout sees one tissue residual and one label factor in every partition, and time the pulses for the readout actually used

```{code-cell} python
:tags: [hide-cell]
import numpy as np
import matplotlib.pyplot as plt

from aslbook import data, kinetic, phantom, presets, protocols, quant
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
- **The factor 0.81 is a model.** $(1-2\varepsilon)^N$ assumes that every pulse inverts
  the whole bolus; the next section compares that assumption with a model in which part of
  the bolus is still outside the imaging slab when a pulse is played.

## The two suppression models: global bolus and bolus position

The label factor of the timeline, $(1-2\varepsilon)^N$, holds if every pulse inverts the
whole labeled bolus wherever the bolus happens to be. That is the simulator's default, the
**global-bolus** model (`AslscanSimulation.BackgroundSuppressionModel: "global-bolus"`), and
it is what a non-selective inversion does; it is also the case the white paper describes,
in which the blood to be labeled experiences every pulse {cite:p}`alsop2015`. With two
pulses it is the most label any implementation can retain: a parcel of label that misses
one of them arrives inverted once, with factor $(1-2\varepsilon) = -0.90$ instead of
$+0.81$, and subtracts from the rest.

Implementations differ. A **slab-selective** pulse inverts only what is inside the imaging
region, and at 2.0 or 2.25 s part of the bolus is still in the neck. aslscan's second model,
**bolus-position** (overlay `background_suppression.model = "bolus-position"` with
`pulse_region = "slab"` and `slab_entry_time = 0.3`), lets a pulse act on a parcel of label
only once that parcel is inside the slab: a parcel labeled at time $l$ enters the slab at
$l + 0.3$ s. The bolus is then a sum of parcels with different factors, no single number
describes it, `BackgroundSuppressionLabelFactor` is null, and the simulator writes the
suppressed difference as a separate truth, `deltamSuppressed`, so that the effective factor
of every voxel is `deltamSuppressed / deltam`. The `bgsup` dataset has the pipeline's pulses
under this model (`on-slab`) and the same model with the first pulse moved to 2.00 s, just
after labeling ends (`early-slab`).

```{code-cell} python
:tags: [hide-input]
sruns = {name: bgsup.run(name) for name in ["on-slab", "early-slab"]}
label_rows = [i for i, r in enumerate(ctx) if r == "label"]
factor_map = {}
for name, run in sruns.items():
    dm, ds = run.truth("deltam")[..., label_rows].mean(axis=-1), run.truth("deltamSuppressed")[..., label_rows].mean(axis=-1)
    factor_map[name] = np.divide(ds, dm, out=np.full(dm.shape, np.nan), where=np.abs(dm) > 1e-6)
for name, run in {"on": runs["on"], **sruns}.items():
    sim, bs = run.simulation(), run.simulation()["BackgroundSuppression"]
    fm = factor_map.get(name)
    effective = f"sidecar label factor {sim['BackgroundSuppressionLabelFactor']:.2f}" if fm is None else f"GM {np.nanmean(fm[gm]):.3f}, WM {np.nanmean(fm[wm]):.3f} (deltamSuppressed / deltam)"
    print(f"{name:>10}: pulses {run.sidecar()['BackgroundSuppressionPulseTime']}, model {sim['BackgroundSuppressionModel']}, "
          f"region {bs.get('PulseRegion', '-')}, slab entry {bs.get('SlabEntryTime', '-')} s; effective factor: {effective}")
```

With the pulses at 2.25 and 3.50 s the two models agree: the bolus-position factor is 0.810
in gray and in white matter, the global value. That is a check, not a coincidence. Labeling
ends at 1.8 s, so the last parcel enters the slab at 2.1 s, before the first pulse, and
every parcel sees both pulses. Moving the first pulse to 2.00 s changes the picture: the
parcels labeled after 1.7 s are not yet in the slab, miss it, and arrive inverted once; the
measured factor is 0.642 in gray matter and 0.594 in white matter. Both numbers follow from
the timeline. The parcels labeled in $[0, 1.7]$ s carry $(1-2\varepsilon)^2 = 0.81$ and
those labeled in $[1.7, 1.8]$ s carry $(1-2\varepsilon) = -0.90$. At the readout every parcel
has arrived (the bolus is complete at ATT + 1.8 s, 2.6 s in gray matter and 3.0 s in white
matter), has decayed in blood for the same transit time, and has decayed in the tissue for
$t - l - \mathrm{ATT}$ with the apparent $T_1' = 1/(1/T_1 + f/\lambda)$ of the kinetic
model ([Chapter 5](../02-labeling/05-kinetic-model.md)), so a parcel's weight at the readout
is proportional to $e^{l/T_1'}$: later parcels weigh more. The fraction of the difference
carried by the parcels labeled before $l_c = 1.7$ s is

$$
w = \frac{e^{l_c/T_1'} - 1}{e^{\tau/T_1'} - 1}, \qquad
F = (1-2\varepsilon)^2\, w + (1-2\varepsilon)\,(1 - w),
$$

with $\tau$ the labeling duration.

```{code-cell} python
:tags: [hide-input]
bs = sruns["early-slab"].simulation()["BackgroundSuppression"]
eps, entry = bs["InversionEfficiency"]["Value"], bs["SlabEntryTime"]
l_cut = sruns["early-slab"].sidecar()["BackgroundSuppressionPulseTime"][0] - entry  # labeled later: outside the slab at the first pulse
for tissue, sel in [("GM", gm), ("WM", wm)]:
    t = presets.TISSUES[tissue]
    t1p = kinetic.t1_prime(t.t1, t.perfusion)
    w = (np.exp(l_cut / t1p) - 1) / (np.exp(R.labeling_duration / t1p) - 1)
    f_two, f_one = (1 - 2 * eps) ** 2, 1 - 2 * eps
    print(f"{tissue}: T1' {t1p:.3f} s; the parcels labeled before {l_cut:.1f} s carry {w:.3f} of the difference at {f_two:.2f}, the rest {1 - w:.3f} at {f_one:+.2f}: "
          f"predicted factor {f_two * w + f_one * (1 - w):.3f}, measured {np.nanmean(factor_map['early-slab'][sel]):.3f}")
```

The prediction matches the measured factor in both tissues: 0.642 in gray matter and 0.590
against 0.594 in white matter. White matter differs from gray matter only through its own
$T_1'$ (0.827 s against 1.311 s): with a shorter $T_1'$ the early parcels have decayed more
by the readout, the inverted tail weighs more, and the net factor is lower. The slice timing
plays no part, because the cut at 1.7 s depends only on when a parcel was labeled, not on
when the slice is read; the per-slice factors of the `early-slab` run are the same in every
slice. The figure shows the per-voxel factor in a coronal section and what happens to the
difference image when a pipeline divides it by the global factor regardless.

```{code-cell} python
:tags: [hide-input]
for name, run in sruns.items():
    sig[name] = signed(run)
    dm_sig[name] = quant.subtract(sig[name], ctx).mean(axis=-1)
fig, axes = plt.subplots(1, 4, figsize=(13, 2.6))
im0 = show_image(axes[0], take_slice(np.where(mask, factor_map["early-slab"], np.nan), CORONAL, "coronal"), "early-slab: factor per voxel", kind="scalar", vmin=0.55, vmax=0.70, aspect=ASPECT)
fig.colorbar(im0, ax=axes[0], shrink=0.6, pad=0.02)
for ax, name in zip(axes[1:], ["on", "on-slab", "early-slab"]):
    im = show_image(ax, take_slice(dm_sig[name] / factors["on"], CORONAL, "coronal"), f"{name}, signed ÷ {factors['on']:.2f}", kind="diff", vmin=-60, vmax=60, aspect=ASPECT)
side_colorbar(fig, im, "mean difference (image units)")
for name in ["on", "on-slab", "early-slab"]:
    r = ratio(dm_sig[name], dm_sig["off"], gm)
    print(f"{name:>10}: GM ratio of the signed difference to the off run {r:.3f}; after dividing by the global factor {factors['on']:.2f}: {r / factors['on']:.3f}")
```

The factor map of `early-slab` is a tissue map: 0.64 in gray matter, 0.59 in white matter,
and nothing else, because the only thing that varies is $T_1'$. The `on` and `on-slab`
difference images divided by 0.81 are the unsuppressed image again (ratios 1.001 and 1.001
to the `off` run). The `early-slab` image divided by the same 0.81 is 0.794 of it: a
pipeline that trusts the global factor delivers a perfusion map about 20 % too low, with a
white matter value that is low by a further few percent, and nothing in the image says so.

Neither model is the truth for a given scanner. Which pulses are slab-selective, where the
slab's edge sits relative to the labeling plane, and how long a parcel takes to cross that
edge decide how much label each pulse catches, and without that knowledge the retained label
can only be bounded, between the global value and the value with the whole tail inverted.
The white paper's recommendation is precisely that the pulses invert all of the blood that
is to be labeled {cite:p}`alsop2015`, and measuring what real pulses do to the label is the
experiment of {cite:t}`garcia2005`. Where the implementation is not known, the factor is a
calibration to measure, not a number to look up.

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

## Measure it: physiological noise, with and without suppression

Motion is the largest fluctuation of the static signal, but not the most frequent. The
heartbeat, breathing and slow drifts change the tissue signal of every volume by a fraction
of a percent, and that fraction, multiplied by 6200 units, is what the first section of this
chapter called the enemy. The `physio` dataset puts the simulator's version of it on the
reference series: at each slice's readout the tissue signal is multiplied by a global factor
$1 + 0.02 \sin\phi_c + 0.01 \sin\phi_r + 0.01\, x_d$, with a cardiac phase $\phi_c$ at about
1 Hz, a respiratory phase $\phi_r$ at about 0.25 Hz, and a drift $x_d$ that wanders over
tens of seconds, and the label is multiplied by $1 + 0.03 \sin\phi_c$ averaged over the
labeling window. The same process runs without (`off`) and with (`on`) the two pulses, and
`run.physio()` holds the phases and factors the simulator applied, one row per volume and
slice. The left panel plots the factors at the display slice; the right panel, the mean
signed control signal of the pure gray matter voxels of that slice, volume by volume, in the
four runs that differ only in whether physiology and suppression are on.

```{code-cell} python
:tags: [hide-input]
physio = data.load_dataset("physio")
pruns = {name: physio.run(name) for name in ["off", "on"]}
psig = {name: signed(run) for name, run in pruns.items()}
table = pruns["off"].physio()
at_k = table[table["slice"] == K].sort_values("volume")
ctl = [i for i, r in enumerate(ctx) if r == "control"]
fig, (ax_f, ax_s) = plt.subplots(1, 2, figsize=(11, 3.4))
ax_f.plot(at_k["time"], at_k["tissue_factor"], "o-", color=PALETTE[0], lw=1, label=f"tissue factor at the readout of slice {K}")
ax_f.plot(at_k["time"], at_k["label_factor"], "s-", color=PALETTE[6], lw=1, label="label factor (mean over the labeling)")
ax_f.axhline(1, color=INK["secondary"], lw=0.8)
ax_f.set(xlabel="time (s)", ylabel="factor", title="the simulator's physiological factors, every volume")
ax_f.legend(loc="lower right")
series = {"physio/off": psig["off"], "bgsup/off": sig["off"], "physio/on": psig["on"], "bgsup/on": sig["on"]}
trace_sd = {}
for (name, s), color in zip(series.items(), [PALETTE[1], PALETTE[0], PALETTE[7], PALETTE[6]]):
    g = s[:, :, K][gm[:, :, K]][:, ctl].mean(axis=0)
    trace_sd[name] = g.std(ddof=1)
    ax_s.plot(np.arange(len(ctl)), g - g.mean(), "o-", lw=1, color=color, label=f"{name} (mean {g.mean():.0f})")
ax_s.set(xlabel="control volume", ylabel="GM mean − its time mean (image units)", title=f"gray matter control signal at slice {K}, volume by volume")
ax_s.legend(loc="upper right", ncol=2)
fig.tight_layout()
amp = pruns["off"].simulation()["Physio"]
print(f"tissue amplitudes: cardiac {amp['TissueAmplitudes']['Cardiac']}, respiratory {amp['TissueAmplitudes']['Respiratory']}, drift {amp['TissueAmplitudes']['Drift']}; "
      f"label: cardiac {amp['LabelAmplitudes']['Cardiac']}; cardiac frequency {amp['CardiacFrequency']} Hz, respiratory {amp['RespiratoryFrequency']} Hz, TR {TR} s")
steps = np.diff(at_k["cardiac_phase"].values) / (2 * np.pi)
print(f"the cardiac phase advances by {steps.mean():.1f} ± {steps.std():.2f} cycles between volumes; "
      f"SD of the tissue factor over the 60 volumes {at_k['tissue_factor'].std():.4f}, of the label factor {at_k['label_factor'].std():.4f}")
print("SD of the GM mean over the 30 control volumes: " + ", ".join(f"{name} {sd:.1f}" for name, sd in trace_sd.items()) + " image units")
```

The tissue factor wanders by a few percent and shows no rhythm: a 1 Hz heartbeat sampled
every 4.5 s advances by 4.6 cycles between volumes, give or take a fifth of a cycle, so
consecutive samples share no phase, and the cardiac term is a random number in every volume.
That is aliasing, and it is what a scanner does too: the physiological rhythms are far above
the Nyquist frequency of any ASL series, and what remains of them in the time series is noise
with the spectrum of a drift, not a peak that a filter could remove. The SD of the tissue
factor over the 60 volumes is 0.0187; the label factor, a cardiac modulation averaged over
1.8 s of labeling, varies by 0.0021. The right panel shows the consequence. The thermal noise
averages out over the few hundred gray matter voxels of the slice, so the `bgsup/off` trace
has an SD of 2.2 units; the `physio/off` trace follows the tissue factor with an SD of 92.2
units and swings of up to about 170, several times the perfusion signal, because a global
factor is the same in every voxel and does not average out; and the `physio/on` trace shows
the same factor applied to a static signal of about 650 units, which brings the SD down to
9.4 units. The measurement
below compares the four runs in pure gray and white matter: the SD over the 30 control
volumes, the SD of the 30 pair differences, the temporal SNR of the difference series
(`quant.tsnr`: the mean over pairs divided by the SD over pairs, voxel by voxel), and the
error of the mean difference against the truth after dividing by the label factor.

```{code-cell} python
:tags: [hide-input]
lab = [i for i, r in enumerate(ctx) if r == "label"]
print(f"{'run':>10} {'control SD GM':>14} {'control SD WM':>14} {'pair SD GM':>11} {'ΔM GM':>7} {'tSNR GM':>8} {'RMSE vs truth (÷ factor)':>25}")
for name, s in series.items():
    d = quant.subtract(s, ctx)
    fac = factors["on"] if name.endswith("/on") else 1.0
    csd = s[..., ctl].std(axis=-1, ddof=1)
    scores[name] = quant.score(d.mean(axis=-1) / fac, truth, mask)["rmse"]
    print(f"{name:>10} {csd[gm].mean():>14.1f} {csd[wm].mean():>14.1f} {d[gm].std(axis=-1, ddof=1).mean():>11.1f} {d.mean(axis=-1)[gm].mean():>7.1f} "
          f"{quant.tsnr(d)[gm].mean():>8.3f} {scores[name]:>25.1f}")
f_k = at_k["tissue_factor"].values
shift = (f_k[ctl] - f_k[lab]).mean()
gm_k = gm[:, :, K]
print(f"at slice {K} the mean over the 30 pairs of the control − label tissue factor is {shift:+.4f}, i.e. {shift * ctrl['off'][:, :, K][gm_k].mean():+.1f} units of the GM control signal; "
      f"the physio/off mean difference at that slice is {quant.subtract(psig['off'], ctx).mean(axis=-1)[:, :, K][gm_k].mean():+.1f} units against {dm_sig['off'][:, :, K][gm_k].mean():+.1f} without physiology")
```

Without suppression the physiological factors raise the SD of the control signal in gray
matter from 39.7 units, the thermal value, to 118.0 (107.2 in white matter), and the SD of a
pair's difference from 56.1 to 174.0: three times the thermal noise and six times the
perfusion signal. The mean of 30 pairs does not escape it. The 30 values of the control −
label tissue factor do not average to zero; at the display slice their mean is −0.0094,
which multiplied by the gray matter control signal is a shift of −58.8 units of the whole
difference image, and the measured gray matter difference of that slice is −30.5 units
where the run without physiology gives +28.4. A global factor does not produce noise in the
usual sense: it produces an error that is the same fraction of the static signal in every
voxel, a copy of the anatomy added to the perfusion map, with a magnitude set by how the
30 random phases happened to fall. The temporal SNR of the `physio/off` run is −0.024 in
gray matter, the sign being that of this realization, against 0.548 for thermal noise alone,
and the RMSE of its mean difference against the truth is 54.4 units, five times the 10.5 of
the thermal run.

With the pulses on, the same factors act on a static signal of a few hundred units instead
of 6200, and almost nothing is left of them: the control SD is 43.9 in gray matter against
39.7 thermal, the pair SD 60.9 against 56.1, the temporal SNR 0.448 against 0.444, and the
RMSE 16.4 against 12.9. The label's own 3 % cardiac modulation, averaged over the labeling,
is a fraction of a unit and stays, since suppression acts on the tissue and not on the
label. This reverses the conclusion of the cost section, and it is worth saying explicitly:
with thermal noise only, suppression lowers the temporal SNR of the difference series from
0.548 to 0.444; with physiological noise present, it raises it from −0.024 to 0.448, and
lowers the error of the mean difference from 54.4 to 16.4 units. The thermal-only
comparison is the one a simulator makes by default and the one that never happens on a
scanner.

The simulator's physiology is a model, and a modest one. The factors are global, identical
in every voxel of a slice, so there is no spatial structure to them; there is no pulsatile
arterial volume, no inflow of unsaturated blood, no change of the head's position with
breathing, and the amplitudes are fixed at 2, 1 and 1 %. Real physiological noise is larger
and less uniform: its variance scales with the square of the signal, as the original
measurements of physiological noise in functional MRI established {cite:p}`kruger2001`, which
is exactly why removing the static signal removes it, and its spatial pattern (large vessels,
CSF spaces, the brainstem) is what makes it harder to separate from perfusion than a global
factor is. The relative gain measured here is therefore a lower bound on what suppression
buys against physiology, with the same mechanism.

## See it: suppression with a 3D readout

The slice dependence of a 2D readout was the weak point of every picture above: a null
reached in a few slices, an inverted band below them and a recovered band above. A 3D
readout excites the whole slab at one instant and reads every partition from that single
state ([Chapter 2](../01-mri-physics/02-epi-and-reconstruction.md)). The `readout-3d` dataset has the
four-shot 3D GRASE and the eight-interleaf stack of spirals on the reference labeling, with
and without two pulses; the sidecar's `AslscanSimulation.Readout` records that every
partition of every shot is excited at 3.6 s, and `protocols.slice_offsets` returns a single
zero offset for a 3D sidecar. The pulses are not the 2D ones. The pair of the `bgsup`
dataset, 2.25 and 3.50 s, was tuned for the middle of a 0.76 s slice readout at 3.98 s, and
at a slab excitation 0.38 s earlier it would leave gray and white matter still inverted
(the per-slice curve of the first section, read at its first slice). A readout change
retimes the pulses, so the 3D runs carry their own pair, read below from the sidecar. The
figure shows the coronal section of the signed control image, as for the 2D runs above.

```{code-cell} python
:tags: [hide-input]
r3d = data.load_dataset("readout-3d")
truns = {name: r3d.run(name) for name in ["grase-4shot", "grase-4shot-bgsup", "spiral", "spiral-bgsup"]}
tctx = truns["grase-4shot"].context()
tsig = {name: signed(run) for name, run in truns.items()}
tctrl = {name: control_mean(s, tctx) for name, s in tsig.items()}
tdm = {name: quant.subtract(s, tctx).mean(axis=-1) for name, s in tsig.items()}
t_exc = truns["grase-4shot-bgsup"].simulation()["Readout"]["ExcitationTimes"][0]
PULSES_3D = truns["grase-4shot-bgsup"].sidecar()["BackgroundSuppressionPulseTime"]
fig, axes = plt.subplots(1, 3, figsize=(11, 2.6))
for ax, name, lim in zip(axes, ["grase-4shot", "grase-4shot-bgsup", "spiral-bgsup"], [7000, 1000, 1000]):
    im = show_image(ax, take_slice(tctrl[name], CORONAL, "coronal"), f"{name} (window ±{lim})", kind="diff", vmin=-lim, vmax=lim, aspect=ASPECT)
side_colorbar(fig, im, "signed control signal (image units)")
for name in ["grase-4shot-bgsup", "spiral-bgsup"]:
    p, sim = truns[name].sidecar(), truns[name].simulation()
    print(f"{name:>17}: {p['MRAcquisitionType']} {p['PulseSequenceType']}, {p['NumberShots']} shots, {len(tctx) // 2} pairs, pulses {p['BackgroundSuppressionPulseTime']}, "
          f"every partition excited at {t_exc} s, slice offsets {protocols.slice_offsets(p)}, model {sim['BackgroundSuppressionModel']}, label factor {sim['BackgroundSuppressionLabelFactor']:.2f}")
for tissue, t in presets.TISSUES.items():
    print(f"{tissue:>3}: Mz/M0 at {t_exc} s with pulses at {PULSES_3D[0]:g} and {PULSES_3D[1]:g} s: {kinetic.tissue_mz(1.0, t.t1, TR, t_exc, PULSES_3D, 0.95):+.3f}", end="; ")
print()
```

There is no band. With the pulses at 1.9 and 3.25 s the timeline puts gray matter at
−0.012 and white matter at −0.029 of $M_0$ at the 3.6 s excitation, and the suppressed 3D
control image (note the window, ±1000 against ±3000 for the 2D runs) is a faint, uniform
negative from the bottom of the slab to the top, with CSF (+0.099, the tissue no two-pulse
scheme nulls with the other two) the only bright structure. The next figure measures it
partition by partition: the ratio of the suppressed to the unsuppressed control signal in
pure gray and white matter, against `kinetic.tissue_mz` at 3.6 s with the sidecar's pulses
divided by the unsuppressed steady state, and the ratio of the signed difference images,
which is the label factor. Ratios are used rather than absolute values because the 3D
control signal is also lowered by the echo train and the through-plane point-spread
function ([Chapter 2](../01-mri-physics/02-epi-and-reconstruction.md)), which the
suppressed and the unsuppressed run share.

```{code-cell} python
:tags: [hide-input]
fig, (ax_c, ax_l) = plt.subplots(1, 2, figsize=(11, 3.4))
pred, per = {}, {}
for tissue in ["GM", "WM"]:
    t = presets.TISSUES[tissue]
    pred[tissue] = kinetic.tissue_mz(1.0, t.t1, TR, t_exc, PULSES_3D, 0.95) / kinetic.tissue_se(1.0, t.t1, TR)
    ax_c.axhline(pred[tissue], color=TISSUE_COLORS[tissue], lw=1, ls="--")
for base, marker in [("grase-4shot", "o"), ("spiral", "s")]:
    fr3 = truns[base].fractions()
    for tissue, sel in [("GM", fr3["gm"] >= 0.9), ("WM", fr3["wm"] >= 0.9)]:
        zs = [z for z in range(R.n_slices) if sel[:, :, z].sum() > 20]
        per[base, tissue] = np.array([tctrl[base + "-bgsup"][:, :, z][sel[:, :, z]].mean() / tctrl[base][:, :, z][sel[:, :, z]].mean() for z in zs])
        ax_c.plot(zs, per[base, tissue], marker, color=TISSUE_COLORS[tissue], mec=INK["primary"], label=f"{tissue}, {base}")
        if tissue == "GM":
            per[base, "label"] = np.array([tdm[base + "-bgsup"][:, :, z][sel[:, :, z]].mean() / tdm[base][:, :, z][sel[:, :, z]].mean() for z in zs])
            ax_l.plot(zs, per[base, "label"], marker, color=PALETTE[6], mec=INK["primary"], label=f"GM, {base}")
ax_l.axhline(factors["on"], color=PALETTE[6], lw=1, ls="--")
ax_c.axhline(0, color=INK["secondary"], lw=0.8)
ax_c.set(xlabel="partition", ylabel="suppressed / unsuppressed control", title=f"control signal per partition (dashed: tissue_mz at {t_exc:g} s)", xticks=range(0, 20, 4), ylim=(-0.06, 0.02))
ax_l.set(xlabel="partition", ylabel="suppressed / unsuppressed difference", title=f"label factor per partition (dashed: {factors['on']:.2f})", xticks=range(0, 20, 4), ylim=(0.7, 0.9))
ax_c.legend(fontsize=7, ncol=2, loc="lower right")
ax_l.legend(fontsize=7)
fig.tight_layout()
for base in ["grase-4shot", "spiral"]:
    print(f"{base:>11}: control ratio GM {per[base, 'GM'].mean():+.3f} ({per[base, 'GM'].min():+.3f} to {per[base, 'GM'].max():+.3f}; predicted {pred['GM']:+.3f}), "
          f"WM {per[base, 'WM'].mean():+.3f} ({per[base, 'WM'].min():+.3f} to {per[base, 'WM'].max():+.3f}; predicted {pred['WM']:+.3f}); "
          f"label factor GM {per[base, 'label'].mean():.3f} ({per[base, 'label'].min():.3f} to {per[base, 'label'].max():.3f})")
print("grase-4shot, GM control ratio per partition: " + " ".join(f"{v:+.3f}" for v in per["grase-4shot", "GM"]))
print("grase-4shot, GM label factor per partition:  " + " ".join(f"{v:.3f}" for v in per["grase-4shot", "label"]))
```

Every partition reads the same state. In the GRASE run the gray matter control ratio is
−0.012 in every partition, between −0.013 and −0.011, where the 2D run went from −0.17 to
+0.33 across the slab; white matter sits at −0.026 (−0.027 to −0.025) against the
prediction of −0.029, the small gap being the through-plane point-spread function mixing
each partition with its neighbors; the spiral gives the same numbers. The static signal of
gray matter is down to 1.2 % of its unsuppressed value, where the 2D `on` run's slab
average was 250 of 6246 units, 4 %. The label factor is 0.810 in every partition, 0.807 to 0.811 in the GRASE run,
0.809 to 0.811 in the spiral run. What a 2D readout gave slice by slice, a 3D readout gives
once.

```{code-cell} python
:tags: [hide-input]
t1s = (presets.TISSUES["GM"].t1, presets.TISSUES["WM"].t1)
best3, resid3 = kinetic.optimal_suppression_times(t1s, t_exc, TR, n_pulses=2, n_grid=60)
rms = lambda pulses: np.sqrt(np.mean([kinetic.tissue_mz(1.0, t1, TR, t_exc, pulses, 0.95) ** 2 for t1 in t1s]))
fr3 = truns["grase-4shot"].fractions()
gm3, wm3 = fr3["gm"] >= 0.9, fr3["wm"] >= 0.9
mag_on, mag_off = (quant.subtract(truns[name].mag(), tctx).mean(axis=-1) for name in ["grase-4shot-bgsup", "grase-4shot"])
print(f"RMS of Mz/M0 over GM and WM at the {t_exc:g} s excitation: the 2D pulses at {PULSES[0]:g} and {PULSES[1]:g} s {rms(PULSES):.3f}; "
      f"the 3D runs' pulses at {PULSES_3D[0]:g} and {PULSES_3D[1]:g} s {rms(PULSES_3D):.3f}; the grid optimum {best3[0]:.2f} and {best3[1]:.2f} s {resid3:.3f}; "
      f"the first 3D pulse falls {PULSES_3D[0] - R.labeling_duration:+.2f} s from the end of labeling")
print(f"ratio of the magnitude-image difference, grase-4shot-bgsup to grase-4shot: GM {ratio(mag_on, mag_off, gm3):+.3f}, WM {ratio(mag_on, mag_off, wm3):+.3f}")
```

Two remarks. First, the pulse times belong to the readout. The 2D pair leaves an RMS residual
of 0.276 at the 3.6 s excitation, the 3D runs' pair 0.022, and the grid optimum for 3.6 s,
pulses at 1.89 and 3.23 s, 0.005: the one-instant readout can be nulled far better than any
slice of the 2D series, but only with its own pulse times. The price is that the first pulse
moves early: at 1.9 s it falls 0.10 s after labeling ends, and under the bolus-position model
of the two-models section the parcels labeled in the last 0.2 s would still be outside the
slab and miss it, exactly the `early-slab` situation; the better the 3D nulling, the more the
retained label depends on the implementation. Second, the magnitude-sign problem changes
character. With one null time for the whole slab the residual has the same sign and size in
every partition, so the sign ambiguity is no longer a band across the slab but a uniform
property of the volume; the magnitude-image difference is −0.811 of the unsuppressed one in
white matter, a single global sign flip, and −0.776 in gray matter, where the residual of
about 70 units is close enough to zero that in some voxels the noise carries the control
across it and the pair loses its sign. A small positive residual chosen once for the whole
volume removes that at a uniform, known cost in suppression; a 2D readout cannot choose one
residual for every slice.

## Measure it: the difference image against the truth

```{code-cell} python
:tags: [hide-input]
print(f"{'run':>14} {'RMSE of the mean difference vs truth, brain (image units)':>60}")
for name in ["off", "on", "perfect", "presat", "random", "random-bgsup", "physio/off", "physio/on"]:
    print(f"{name:>14} {scores[name]:>60.1f}")
```

For the still series the ranking is set by the label factor alone: 10.5 units for `off` and
`perfect`, 12.9 for `on` and `presat`. For the moving series it is set by the static signal:
63 without suppression, 23 with it, from the same poses. For the series with physiological
fluctuation it is set by the static signal again: 54.4 without suppression, 16.4 with it.

## What this implies for acquisition

- **Use background suppression** in any protocol quantified from many pairs: physiological
  fluctuation and motion, not thermal noise, limit real ASL, and both scale with the static
  signal {cite:p}`alsop2015`.
- **Pair it with a 3D readout, and time the pulses for it.** A 2D readout is nulled in a
  few slices and inverted or recovered in the rest (−0.17 to +0.33 of $M_0$ in gray matter
  across the slab); the 3D GRASE and spiral runs read every partition at the same −0.012
  with the same label factor 0.810 {cite:p}`vidorreta2013`. The pulse times that serve a 2D
  readout's middle slice would leave a 3D slab excited at 3.6 s inverted; the 3D runs' pair
  at 1.90 and 3.25 s is a different one.
- **Leave a small positive residual** rather than aiming at the exact null, unless the phase
  is saved: a magnitude image cannot tell −0.2 $M_0$ from +0.2 $M_0$. With a 3D readout the
  margin is one number for the whole volume.
- **Acquire the M0 scan without pulses**, and record the pulse times, their number and their
  efficiency in the sidecar so that the quantification can divide by $(1-2\varepsilon)^N$.
- **Know the suppression implementation.** $(1-2\varepsilon)^N$ is the global-bolus value and
  an upper bound; if a pulse is slab-selective and part of the bolus is still outside the
  slab when it is played, the tail of the bolus arrives with the opposite sign, and the
  `early-slab` run loses 21 % of its perfusion signal to a factor no sidecar field records.
  Where the pulses, the slab edge and the transit into the slab are not documented, measure
  the factor against an unsuppressed acquisition rather than looking it up.
- **Budget the loss.** Two pulses at ε = 0.95 cost 19 % of the label and 1.5 times the pairs
  for the same thermal SNR; more pulses suppress more tissues and cost more label, and an odd
  number flips the sign of the difference.

## Further reading

Multiple inversion recovery to null static tissue {cite:p}`dixon1991`; background
suppression for ASL {cite:p}`ye2000`; the efficiency of inversion pulses and its effect on
the label {cite:p}`garcia2005`; the optimization of the pulse timing {cite:p}`maleki2012`;
the sidecar fields that record the pulses {cite:p}`clement2022`; the white paper's recommendations on suppression and 3D
readouts {cite:p}`alsop2015`; 2D versus 3D readouts with suppression {cite:p}`vidorreta2013`;
physiological noise scaling with the signal it rides on {cite:p}`kruger2001`.
