---
title: "14. CBF from a single delay"
kernelspec:
  name: python3
  display_name: Python 3
---

:::{admonition} Simulated datasets in this chapter
:class: note
- **Built in this page:** the kinetic model evaluated for pure gray and white matter over a range of post-labeling delays, to show what the single-delay formula's assumptions cost ([Appendix B](../appendices/b-data-manifest.md#app-b-package-data)).
- **`ref-pcasl`**: the reference 2D PCASL series, 30 pairs at PLD 1.8 s with its M0 scan: the worked example ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-ref-pcasl)).
- **`label-types`**: the same slab and timing under PCASL, CASL, and PASL, each quantified with its own formula and efficiency ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-label-types)).
- **`pld-sweep`**: single-delay PCASL at six delays from 0.5 to 3.0 s: what the formula returns when the delay is shorter, and longer, than the transit time ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-pld-sweep)).

Pipeline-tier datasets are simulated offline by aslscan ([Chapter 0.2](../00-frontmatter/the-simulated-datasets.md)).
:::

## Learning goals

After this chapter you can:

- derive the single-delay CBF formula of the ASL white paper from the kinetic model and name
  the three assumptions it makes
- find every symbol of the formula in a BIDS sidecar, or know which constant to assume when
  the sidecar is silent
- correct the delay of each slice of a 2D readout and say what happens if you do not
- read a single-delay CBF map knowing which way it is biased when the transit time is
  longer than the delay, and which way the choice of T1 biases it
- quantify PASL, CASL, and PCASL data so that they agree

```{code-cell} python
:tags: [hide-cell]
import numpy as np
import matplotlib.pyplot as plt

from aslbook import data, kinetic, phantom, plotting, presets, protocols, quant
from aslbook.plotting import INK, PALETTE, TISSUE_COLORS, set_style, show_slice

set_style()
K = phantom.DISPLAY_SLICE
GM, WM = presets.TISSUES["GM"], presets.TISSUES["WM"]
T1B, LAM = presets.T1_BLOOD, presets.LAMBDA


def div0(n, d):
    n, d = np.asarray(n, float), np.asarray(d, float)
    return np.divide(n, d, out=np.zeros(np.broadcast(n, d).shape), where=d != 0)


def cbf_tissue_t1(deltam, m0, pld, att, t1_tissue, *, tau, alpha, lam=LAM, t1b=T1B):
    """The arrived-phase solution of the kinetic model inverted for CBF without the T1b
    assumption: the label decays with T1b until it arrives and with T1' afterwards. Needs the
    transit time and the tissue T1; T1' depends weakly on CBF, so one refinement suffices."""
    f = np.zeros(np.broadcast(deltam, m0, pld, att, t1_tissue).shape)
    for _ in range(2):
        t1p = kinetic.t1_prime(t1_tissue, f, lam)
        with np.errstate(divide="ignore", invalid="ignore"):   # T1 is 0 outside the perfused tissue
            den = 2 * alpha * t1p * m0 / lam * np.exp(-att / t1b) * np.exp(-(pld - att) / t1p) * (1 - np.exp(-tau / t1p))
        f = 6000 * div0(deltam, np.nan_to_num(den))
    return f


def load(dataset, run_name):
    """A run's mean difference image and its calibrated M0, both in image units."""
    run = data.load_dataset(dataset).run(run_name)
    p, ctx = run.sidecar(), run.context()
    dm = quant.subtract(run.mag(), ctx).mean(-1)
    m0 = quant.m0_correction(run.m0scan(), tr=run.m0scan_sidecar()["RepetitionTimePreparation"], te=p["EchoTime"])
    return run, p, dm, m0


def per_slice_mean(img, roi, min_voxels=50):
    """Mean of img in roi for every slice with enough voxels, NaN elsewhere."""
    return np.array([img[:, :, z][roi[:, :, z]].mean() if roi[:, :, z].sum() >= min_voxels else np.nan for z in range(img.shape[2])])
```

## From the kinetic model to one formula

[Chapter 5](../02-labeling/05-kinetic-model.md) gave the label-control difference of a
(P)CASL experiment once the whole bolus has arrived (signal time $t \ge \delta + \tau$, that
is, $\mathrm{PLD} \ge \delta$):

$$
\Delta M = 2\,\frac{M_0}{\lambda}\, f\, \alpha\, T_1'\;
e^{-\delta/T_{1b}}\; e^{-(\mathrm{PLD}-\delta)/T_1'}\;\bigl(1 - e^{-\tau/T_1'}\bigr),
$$

with $f$ the perfusion, $\delta$ the arterial transit time, $\tau$ the labeling duration,
$T_{1b}$ the T1 of arterial blood, and $T_1'$ the apparent T1 of the label once it is in
tissue ($1/T_1' = 1/T_1 + f/\lambda$). The label decays with $T_{1b}$ while it travels
(the first exponential) and with $T_1'$ after it arrives (the second). The transit time
appears in both.

The ASL white paper {cite:p}`alsop2015` makes one more assumption: that the label decays
with the T1 of blood the whole time, $T_1' \to T_{1b}$. The two exponentials then merge,
$e^{-\delta/T_{1b}} e^{-(\mathrm{PLD}-\delta)/T_{1b}} = e^{-\mathrm{PLD}/T_{1b}}$, and the
transit time drops out. Solving for $f$ and converting from ml/g/s to ml/100 g/min (a factor
6000) gives the single-delay formula:

$$
\mathrm{CBF} = \frac{6000\,\lambda\,\Delta M\, e^{\mathrm{PLD}/T_{1b}}}
{2\,\alpha\, T_{1b}\, M_0\, \bigl(1 - e^{-\tau/T_{1b}}\bigr)}
\quad [\mathrm{ml/100\,g/min}].
$$

It rests on three assumptions, and the rest of this chapter measures what each costs:

1. **The whole bolus has arrived**, $\mathrm{PLD} \ge \delta$ in every voxel. Where it has
   not, $\Delta M$ is smaller than the formula expects and CBF is underestimated.
2. **The label decays with $T_{1b}$ throughout.** Where the label has been in tissue for
   a while, it has decayed with the shorter tissue T1 instead, and CBF is again
   underestimated, more so for a shorter tissue T1 and a longer time in tissue.
3. **No outflow**: the label stays where it arrived. The kinetic model's $T_1'$ carries the
   outflow term $f/\lambda$, which is small (about 1.5 % of $1/T_1$ in gray matter) and is
   absorbed into assumption 2.

For pulsed labeling the arrived-phase solution with a bolus of fixed duration $\mathrm{TI}_1$
(a QUIPSS II or Q2TIPS cut-off, [Chapter 4](../02-labeling/04-labeling-schemes.md)) gives,
under the same $T_1' \to T_{1b}$ assumption,

$$
\mathrm{CBF} = \frac{6000\,\lambda\,\Delta M\, e^{\mathrm{TI}/T_{1b}}}
{2\,\alpha\, \mathrm{TI}_1\, M_0},
$$

where TI is the inversion time (the delay from the labeling pulse to the readout) and the
bolus has arrived when $\mathrm{TI} \ge \delta + \mathrm{TI}_1$. The two formulas are
`quant.cbf_pcasl` and `quant.cbf_pasl`.

## Every symbol traced to the sidecar

The formula needs five numbers besides the two images. Three are acquisition parameters that
the BIDS sidecar records; two are constants that the white paper fixes and that the sidecar
does not carry. The table lists where each comes from; the cell below prints them for the
reference dataset, including the simulator's own record of what it used
(`AslscanSimulation.Resolved`), which a real scanner does not write.

| Symbol | Meaning | Source | Reference value |
|---|---|---|---|
| $\tau$ | labeling duration | `LabelingDuration` | 1.8 s |
| PLD | post-labeling delay | `PostLabelingDelay`, plus `SliceTiming[z]` in 2D | 1.8 s |
| $\alpha$ | labeling efficiency | `LabelingEfficiency` if present; else the type's default (0.85 PCASL, 0.98 PASL) | 0.85 |
| $T_{1b}$ | T1 of arterial blood | assumed from `MagneticFieldStrength`: 1.65 s at 3 T, 1.35 s at 1.5 T | 1.65 s |
| $\lambda$ | blood-brain partition coefficient of water | assumed | 0.9 ml/g |
| $M_0$ | equilibrium magnetization of blood | the M0 scan (`M0Type` Separate), corrected as below | image units |
| $\mathrm{TI}_1$ | PASL bolus duration | `BolusCutOffDelayTime` | 0.7 s |

```{code-cell} python
:tags: [hide-input]
run, p, dm, m0 = load("ref-pcasl", "pcasl")
fr, mask = run.fractions(), run.mask()
gm, wm = fr["gm"] > 0.9, fr["wm"] > 0.9
truth = run.truth("perfusion")
res = run.simulation()["Resolved"]
for key in ["ArterialSpinLabelingType", "MRAcquisitionType", "LabelingDuration", "PostLabelingDelay",
            "LabelingEfficiency", "MagneticFieldStrength", "M0Type", "TotalAcquiredPairs", "EchoTime"]:
    print(f"{key:>26}: {p[key]}")
st = p["SliceTiming"]
print(f"{'SliceTiming':>26}: {st[0]:g} ... {st[-1]:g} s ({len(st)} slices, {st[1] - st[0]:g} s apart)")
print(f"{'M0 scan TR':>26}: {run.m0scan_sidecar()['RepetitionTimePreparation']:g} s")
print("the simulator used: " + ", ".join(f"{k} {v['Value']:g} ({v['Source'].lower()})" for k, v in res.items() if isinstance(v, dict)))
```

The printed `Resolved` block confirms that the simulator labeled with efficiency 0.85, took
$T_{1b}$ 1.65 s and $\lambda$ 0.9 from the phantom, and the white paper's constants are
therefore exactly right for this data. With real data they are assumptions, and
[Chapter 16](./16-calibration.md) measures how far a wrong one moves the result.

## M0 calibration in one paragraph

$M_0$ in the formula is the equilibrium magnetization of arterial blood, and the M0 scan
measures something related but not equal: the tissue's magnetization after a partial T1
recovery over the scan's own TR, decayed with the tissue T2 by the echo time. The book's M0
scan uses TR 8 s (from its sidecar), so the recovery is nearly complete
($1 - e^{-8/1.33} = 0.998$ in gray matter), but the T2 mismatch matters: at TE 12 ms the
tissue signal has decayed by $e^{-0.012/0.080}$ while the labeled blood in the difference
image has decayed by $e^{-0.012/0.165}$, a 7 % difference. `quant.m0_correction` divides the
M0 scan by the recovery factor and swaps the tissue T2 decay for the blood's, with gray
matter's T1 and T2 for every voxel; that is what the cells above applied, with TR 8 s and
the TE of the sidecar. The white paper folds the same corrections into a scan-specific
factor; [Chapter 16](./16-calibration.md) covers the other sources of $M_0$ (included
volumes, the control mean, a CSF reference) and their corrections.

## See it: ΔM, M0, and CBF on the reference protocol

The three images the formula combines, on the display slice. The mean difference image is
the average of 30 control-label pairs; the M0 image is the corrected M0 scan; the CBF map
is the formula applied voxel by voxel with each slice's own delay (next section).

```{code-cell} python
:tags: [hide-input]
offs = np.asarray(protocols.slice_offsets(p))
pld_z = quant.slice_plds(p["PostLabelingDelay"], offs)      # one delay per slice
tau, alpha = p["LabelingDuration"], p["LabelingEfficiency"]
cbf = quant.cbf_pcasl(dm, m0, pld_z[None, None, :], tau=tau, alpha=alpha)

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
show_slice(axes[0], np.where(mask, dm, np.nan), K, "mean ΔM (image units)", kind="magnitude", vmin=0, vmax=50)
show_slice(axes[1], np.where(mask, m0, np.nan), K, "M0, corrected M0 scan (image units)", kind="magnitude", vmin=0, vmax=8000)
im = show_slice(axes[2], np.where(mask, cbf, np.nan), K, "CBF (ml/100 g/min)", kind="cbf")
fig.colorbar(im, ax=axes[2], shrink=0.8)
fig.tight_layout()
for name, roi in [("GM", gm), ("WM", wm)]:
    print(f"{name}: ΔM {dm[roi].mean():5.1f}, M0 {m0[roi].mean():.0f} image units, ΔM/M0 {100 * dm[roi].mean() / m0[roi].mean():.2f} %, "
          f"CBF {cbf[roi].mean():.1f} vs truth {truth[roi].mean():.1f} ml/100 g/min")
```

Look at the scale of the three images. The difference image is about 30 image units in gray
matter against an M0 of about 7000, a ratio of 0.43 %; the formula multiplies that ratio by
about $10^4$ to reach tens of ml/100 g/min. Anything that biases $\Delta M$ by one image
unit moves CBF by more than one ml/100 g/min, which is why the artifacts of Part III matter
so much. The CBF map has the right anatomy, cortex bright and white matter dark, but the
printed gray matter mean is 44 against a truth of 59: the formula is biased low, and the
sections below trace the bias to its assumptions.

## Slice timing in a 2D readout

A 2D EPI readout excites one slice at a time, here 20 slices 40 ms apart, so the last slice
is read 0.76 s after the first ([Chapter 6](../02-labeling/06-the-asl-signal.md)). The
label in the last slice has decayed for 0.76 s longer, by $e^{-0.76/1.65} = 0.63$ if it
decays with $T_{1b}$: a 37 % smaller $\Delta M$ for the same perfusion. The sidecar's
`SliceTiming` gives each slice's offset, and the correction is to quantify slice $z$ with
$\mathrm{PLD} + \mathrm{SliceTiming}[z]$ (`quant.slice_plds`). The figure plots the mean
gray matter CBF of each slice, quantified with the nominal delay for every slice and with
each slice's own delay.

```{code-cell} python
:tags: [hide-input]
cbf_nominal = quant.cbf_pcasl(dm, m0, p["PostLabelingDelay"], tau=tau, alpha=alpha)
per_nom, per_corr = per_slice_mean(cbf_nominal, gm), per_slice_mean(cbf, gm)
ok = np.isfinite(per_corr)
fig, ax = plt.subplots(figsize=(7.5, 3.4))
ax.plot(offs[ok], per_nom[ok], "o-", color=PALETTE[3], label="nominal PLD for every slice")
ax.plot(offs[ok], per_corr[ok], "o-", color=TISSUE_COLORS["GM"], label="PLD + SliceTiming[z] (corrected)")
ax.plot(offs, per_corr[0] * np.exp(-offs / T1B), "--", color=INK["secondary"], lw=1.2, label="first slice × exp(−offset / T1b)")
ax.axhline(GM.perfusion, color=INK["secondary"], lw=0.8, ls=":")
ax.text(0.77, GM.perfusion + 1, "truth 60", ha="right", fontsize=8, color=INK["secondary"])
ax.set(xlabel="slice readout offset after the first slice (s)", ylabel="mean GM CBF (ml/100 g/min)", ylim=(0, 70))
ax.legend(loc="lower left")
fig.tight_layout()
first, last = np.flatnonzero(ok)[[0, -1]]
print(f"slices {first}-{last} (offsets {offs[first]:g}-{offs[last]:g} s); GM CBF first -> last slice: "
      f"uncorrected {per_nom[first]:.1f} -> {per_nom[last]:.1f} ({100 * (per_nom[last] / per_nom[first] - 1):+.0f} %), "
      f"corrected {per_corr[first]:.1f} -> {per_corr[last]:.1f} ({100 * (per_corr[last] / per_corr[first] - 1):+.0f} %)")
print(f"per-slice bias without the correction, relative to the corrected value: "
      f"{np.nanmean(100 * (per_nom - per_corr) / per_corr):.0f} % on average, {100 * (per_nom[last] / per_corr[last] - 1):.0f} % in the last slice")
```

Without the correction the gray matter CBF falls from 45 in the first slice to 24 in the
last, a 47 % drop: a gradient across the head that has nothing to do with perfusion, and
that any reader of the map would take for one. Relative to the corrected value the
uncorrected estimate is 37 % low in the last slice, exactly the $e^{-0.76/1.65}$ of the
dashed line, and 20 % low on average over the slab. With each slice's own delay the
estimate is much flatter, but not flat: the remaining slope, 45 to 38, is real and is
explained in the section on $T_1'$ below. The label in later slices has spent longer in
tissue, decaying faster than the $T_{1b}$ correction assumes, so the correction is
incomplete by the same mechanism that biases the whole map. A 3D readout has no slice
timing (every slice is read at once), which is one of the reasons the white paper prefers
it {cite:p}`vidorreta2013`.

## Measure it: CBF against the truth

The book's four-panel comparison scores the corrected CBF map of the reference dataset
against the phantom's perfusion map inside the brain mask, and the printed numbers are the
bias (mean estimate minus truth), the root-mean-square error, and the correlation.

```{code-cell} python
:tags: [hide-input]
fig, axes = plotting.fit_vs_truth(cbf, truth, mask, "CBF", k=K, unit="(ml/100 g/min)", kind="cbf")
for name, roi in [("GM (fraction > 0.9)", gm), ("WM (fraction > 0.9)", wm), ("whole brain mask", mask)]:
    s = quant.score(cbf, truth, roi)
    print(f"{name:>20}: bias {s['bias']:+6.1f}, RMSE {s['rmse']:5.1f} ml/100 g/min, r {s['r']:.2f}, n {s['n']}")
print(f"GM mean {cbf[gm].mean():.1f} (truth {truth[gm].mean():.1f}), WM mean {cbf[wm].mean():.1f} (truth {truth[wm].mean():.1f}); "
      f"noise alone: SD of CBF in GM {cbf[gm].std():.1f} ml/100 g/min")
```

The estimate has the truth's anatomy and a correlation of 0.62 over the brain, driven by
the contrast between tissues; within one tissue the truth is constant, so the correlation
there is meaningless and the bias and RMSE carry the information. The difference map is
blue almost everywhere: gray matter is underestimated by 15 ml/100 g/min (26 %) and white
matter by 14 (65 %). The RMSE in gray matter, 22, is the bias combined with the noise (a
standard deviation of 16 across gray matter voxels at 30 pairs), so at this scan length the
two contribute about equally; [Chapter 8](../03-preprocessing/08-noise.md) covers the
noise. The bias is the subject of the next two sections.

## Where the bias comes from: T1b versus T1'

Assumption 1 holds for the reference protocol (PLD 1.8 s exceeds both transit times, 0.8 and
1.2 s), so the underestimate is assumption 2. The simulator implements the kinetic model
exactly as [Chapter 5](../02-labeling/05-kinetic-model.md) stated it: labeled water
exchanges into tissue the moment it arrives and decays with the tissue $T_1'$ from then on.
At PLD 1.8 s the gray matter label has been in tissue for between 1.0 and 2.8 s (the
trailing and leading ends of the bolus), and over that time $T_1'$ of 1.31 s loses more
signal than $T_{1b}$ of 1.65 s: for the mean of 1.9 s, $e^{-1.9/1.31}/e^{-1.9/1.65} =
0.76$, the 24 % the previous cell measured. White matter's T1 of 0.83 s is half the
blood's, and its label has spent up to 2.4 s in tissue, so its bias is far larger. The
figure evaluates `kinetic.delta_m` for a pure gray and a pure white matter voxel at delays
from 0.3 to 3.0 s and applies the white-paper formula to the result, so it isolates the
formula's assumptions from noise, partial volume, and readout effects. The dotted lines
drop assumption 2 only: they invert the arrived-phase solution with the true transit time
and tissue T1 kept (`cbf_tissue_t1` in the setup cell), which is what a multi-delay fit
([Chapter 15](./15-multi-delay.md)) does once it has estimated the transit time.

```{code-cell} python
:tags: [hide-input]
plds = np.round(np.arange(0.3, 3.001, 0.02), 3)
toy = {}
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.5))
for name, t in [("GM", GM), ("WM", WM)]:
    dmk = kinetic.delta_m(plds + tau, t.perfusion, t.att, t.t1, 1.0, tau=tau)          # ΔM / M0
    wp = quant.cbf_pcasl(dmk, 1.0, plds, tau=tau, alpha=presets.ALPHA["PCASL"])
    kin = cbf_tissue_t1(dmk, 1.0, plds, t.att, t.t1, tau=tau, alpha=presets.ALPHA["PCASL"])
    toy[name] = (dmk, wp, kin)
    axes[0].plot(plds, 100 * dmk, color=TISSUE_COLORS[name], label=f"{name}: CBF {t.perfusion:g}, ATT {t.att:g} s, T1 {t.t1:g} s")
    axes[1].plot(plds, 100 * wp / t.perfusion, color=TISSUE_COLORS[name], label=f"{name}, white-paper formula")
    axes[1].plot(plds, 100 * kin / t.perfusion, ":", color=TISSUE_COLORS[name], label=f"{name}, same with tissue T1 and ATT known")
    for ax in axes:
        ax.axvline(t.att, color=TISSUE_COLORS[name], lw=0.8, ls="--")
axes[0].set(xlabel="post-labeling delay (s)", ylabel="ΔM / M0 (%)", title="the kinetic model (LD 1.8 s)")
axes[1].axhline(100, color=INK["secondary"], lw=0.8); axes[1].axhline(90, color=INK["secondary"], lw=0.8, ls=":")
axes[1].set(xlabel="post-labeling delay (s)", ylabel="estimate / truth (%)", ylim=(0, 115), title="single-delay estimate (dashed: the ATT)")
axes[0].legend(fontsize=7.5); axes[1].legend(fontsize=7, loc="lower right")
fig.tight_layout()
i18 = np.flatnonzero(plds == 1.8)[0]
for name, t in [("GM", GM), ("WM", WM)]:
    dmk, wp, kin = toy[name]
    j = np.argmax(wp)
    print(f"{name}: at PLD 1.8 s the formula gives {100 * wp[i18] / t.perfusion:.0f} % of the truth; its best delay is "
          f"{plds[j]:.2f} s ({100 * wp[j] / t.perfusion:.0f} %); with the tissue T1 the estimate reaches 90 % at "
          f"PLD {plds[np.flatnonzero(kin >= 0.9 * t.perfusion)[0]]:.2f} s and 100 % at {t.att:g} s (the transit time)")
```

Look first at the left panel: $\Delta M$ rises while the bolus arrives, from the moment the
delay plus the labeling duration exceeds the transit time until the delay itself does
(dashed), then decays. The right panel is the estimate divided by the truth. The
white-paper formula (solid) never reaches the truth in either tissue: gray matter peaks at
89 % at PLD 0.8 s, its own transit time, and is at 76 % by 1.8 s; white matter peaks at
67 % at 1.2 s and is at 47 % by 1.8 s, falling further as the delay lengthens, because
every extra second in tissue costs a factor $e^{-1/0.83}/e^{-1/1.65}$. With the tissue T1
and the transit time kept (dotted) the estimate is exact as soon as the delay reaches the
transit time and within 10 % from 0.70 s in gray matter and 1.14 s in white matter, just
before it, because the front of the bolus carries most of the signal by then; below that
the whole-bolus assumption fails and the estimate drops steeply.

Two things follow. First, the formula's T1 assumption is not a small correction on this
phantom: the simulator sits at one end of the physical range, a single well-mixed
compartment with instantaneous exchange, so the cost of assuming $T_{1b}$ is as large here
as it can be. In a living brain part of the label is still in the capillaries at the
readout and decays with the blood's T1, which is the white paper's argument for the
simpler formula; the truth lies between, and closer to the blood's T1 for short delays
after arrival. Second, the bias is systematic and tissue-dependent, so it inflates the
gray-to-white contrast (measured below) and it changes with the population's T1
([Chapter 16](./16-calibration.md) measures the sensitivity to every assumed constant).
The next cell shows that nothing else is wrong with the pipeline: quantifying the same
difference image with the phantom's transit-time map and a T1 map built from the tissue
fractions recovers the truth in both tissues.

```{code-cell} python
:tags: [hide-input]
perfused = fr["gm"] + fr["wm"]
t1_map = div0(fr["gm"] * GM.t1 + fr["wm"] * WM.t1, perfused)      # T1 of the perfused part of each voxel
cbf_kin = cbf_tissue_t1(dm, m0, pld_z[None, None, :], run.truth("att"), t1_map, tau=tau, alpha=alpha)

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
show_slice(axes[0], np.where(mask, cbf, np.nan), K, "white-paper formula", kind="cbf")
im = show_slice(axes[1], np.where(mask, cbf_kin, np.nan), K, "with the tissue T1 and the true ATT", kind="cbf")
fig.colorbar(im, ax=axes[1], shrink=0.8)
fig.tight_layout()
for name, roi in [("GM", gm), ("WM", wm)]:
    a, b = quant.score(cbf, truth, roi), quant.score(cbf_kin, truth, roi)
    print(f"{name}: bias {a['bias']:+5.1f} -> {b['bias']:+5.1f}, RMSE {a['rmse']:4.1f} -> {b['rmse']:4.1f} ml/100 g/min (white-paper formula -> tissue T1 and ATT)")
```

With the true transit time and tissue T1 the bias falls from −15 to +1 in gray matter and
from −14 to −1 in white matter (the residual is truncation ringing at the tissue edges,
[Chapter 2](../01-mri-physics/02-epi-and-reconstruction.md)). The white matter RMSE more
than doubles, because with a T1 of 0.83 s the model expects very little signal at PLD
1.8 s, so every image unit of noise is worth more CBF. Nobody has the true transit-time
map of a patient, which is why the single-delay formula is used at all: it trades this
bias for one fewer unknown.

## The bias when ATT exceeds PLD

The `pld-sweep` dataset acquires the reference protocol at six delays (15 pairs each, TR 6 s
so that the longest fits). Each run is quantified with the formula and its own delay; the
figure plots the mean gray and white matter CBF against the delay with the truth as
horizontal lines and, dashed, the toy prediction from the previous figure averaged over the
slices in the same way.

```{code-cell} python
:tags: [hide-input]
sweep = data.load_dataset("pld-sweep")
names = ["pld05", "pld10", "pld15", "pld20", "pld25", "pld30"]
sweep_cbf, sweep_pld = {}, []
for rn in names:
    r, ps, dms, m0s = load("pld-sweep", rn)
    pz = quant.slice_plds(ps["PostLabelingDelay"], protocols.slice_offsets(ps))
    sweep_cbf[rn] = quant.cbf_pcasl(dms, m0s, pz[None, None, :], tau=ps["LabelingDuration"], alpha=ps["LabelingEfficiency"])
    sweep_pld.append(ps["PostLabelingDelay"])
sweep_pld = np.array(sweep_pld)

fig, ax = plt.subplots(figsize=(7.5, 3.6))
sweep_means = {}
for name, t, roi in [("GM", GM, gm), ("WM", WM, wm)]:
    means = sweep_means[name] = np.array([sweep_cbf[rn][roi].mean() for rn in names])
    n_z = np.array([roi[:, :, z].sum() for z in range(20)])
    pred = [np.average([quant.cbf_pcasl(kinetic.delta_m(v + tau + o, t.perfusion, t.att, t.t1, 1.0, tau=tau), 1.0, v + o, tau=tau)
                        for o in offs], weights=n_z) for v in plds]
    ax.plot(sweep_pld, means, "o-", color=TISSUE_COLORS[name], label=f"{name}, measured")
    ax.plot(plds, pred, "--", color=TISSUE_COLORS[name], lw=1.2, label=f"{name}, kinetic model + formula")
    ax.axhline(t.perfusion, color=TISSUE_COLORS[name], lw=0.8, ls=":")
    ax.axvline(t.att, color=TISSUE_COLORS[name], lw=0.8, ls=":")
print(f"{'PLD (s)':>8} {'GM CBF':>8} {'% truth':>8} {'WM CBF':>8} {'% truth':>8}")
for i, v in enumerate(sweep_pld):
    print(f"{v:8.1f} {sweep_means['GM'][i]:8.1f} {100 * sweep_means['GM'][i] / truth[gm].mean():8.0f} "
          f"{sweep_means['WM'][i]:8.1f} {100 * sweep_means['WM'][i] / truth[wm].mean():8.0f}")
for name, roi in [("GM", gm), ("WM", wm)]:
    j = np.argmax(sweep_means[name])
    print(f"{name}: highest single-delay estimate at PLD {sweep_pld[j]:.1f} s, {100 * sweep_means[name][j] / truth[roi].mean():.0f} % of the truth")
ax.set(xlabel="post-labeling delay (s)", ylabel="mean CBF (ml/100 g/min)", ylim=(0, 70), xlim=(0.3, 3.2))
ax.text(0.35, GM.perfusion + 1.5, "truth GM 60", fontsize=8); ax.text(0.35, WM.perfusion + 1.5, "truth WM 20", fontsize=8)
ax.text(GM.att + 0.03, 66, "ATT GM", fontsize=7, color=TISSUE_COLORS["GM"]); ax.text(WM.att + 0.03, 66, "ATT WM", fontsize=7, color=TISSUE_COLORS["WM"])
ax.legend(fontsize=7.5, loc="center", bbox_to_anchor=(0.62, 0.42))
fig.tight_layout()
```

The measured points sit on the dashed prediction, so the pipeline behaves as the model
says it should, ringing and noise aside. The curves are smoother than the pure-voxel ones
of the previous figure because each point averages 20 slices whose delays span 0.76 s. Below
its transit time each tissue is underestimated for the first reason (the bolus has not
arrived): white matter at a nominal PLD of 0.5 s returns 49 % of its truth. Above it, the
estimate falls with the delay for the second reason (the T1 of tissue), and the white
matter estimate never exceeds 57 % of 20 ml/100 g/min, at PLD 1.0 s; gray matter peaks at
84 % at the same delay. There is no delay at which a single-delay acquisition returns the
white matter truth with this formula; the best it does is near the transit time, where the
two biases are smallest together. This is a fair summary of single-delay white matter CBF
in practice: it is reported as low, and it is trusted less than the gray matter value
{cite:p}`vanosch2018`.

The maps make the same point spatially. At short delays the deep white matter and the
territories that fill last are missing; at long delays everything is dim and noisy, because
the signal has decayed and the formula's $e^{\mathrm{PLD}/T_{1b}}$ amplifies noise as
much as signal.

```{code-cell} python
:tags: [hide-input]
fig, axes = plotting.mosaic({f"PLD {v:.1f} s": np.where(mask, sweep_cbf[rn], np.nan) for rn, v in zip(names, sweep_pld)},
                            k=K, kind="cbf", ncols=3, figsize_per=2.6)
fig.suptitle("single-delay CBF (ml/100 g/min) from the pld-sweep runs", fontsize=10, y=1.0)
for rn, v in zip(names, sweep_pld):
    print(f"PLD {v:.1f} s: GM CBF SD across voxels {sweep_cbf[rn][gm].std():.1f}, fraction of negative GM voxels {100 * (sweep_cbf[rn][gm] < 0).mean():.0f} %")
```

The printed spread across gray matter voxels grows from PLD 2.0 s onward while the mean
falls: the noise amplification of a long delay. The white paper's recommendation of PLD
1.8 s in healthy adults and 2.0 s or more in the elderly and in patients is the compromise
between arrival (assumption 1) and this noise; the formula's second assumption then sets
a bias it accepts. Where transit times are longer than the delay, a multi-delay acquisition
([Chapter 15](./15-multi-delay.md)) measures them instead of assuming them.

## PASL, CASL, PCASL: one formula each

The `label-types` dataset labels the same slab with the three schemes of
[Chapter 4](../02-labeling/04-labeling-schemes.md), at the same delay: PCASL and CASL with
LD 1.8 s and PLD 1.8 s, PASL with TI 1.8 s and a Q2TIPS cut-off at 0.7 s. Each is quantified
with its own formula and its own efficiency: 0.85 for PCASL, 0.68 for the amplitude-modulated
CASL control, 0.98 for the pulsed inversion. The efficiency comes from the sidecar's
`LabelingEfficiency`, which here equals the simulator's resolved value.

```{code-cell} python
:tags: [hide-input]
lt_cbf, lt_alpha, lt_expect = {}, {}, {}
n_gm_z = np.array([gm[:, :, z].sum() for z in range(20)])
for rn in ["pcasl", "casl", "pasl"]:
    r, ps, dms, m0s = load("label-types", rn)
    a = r.simulation()["Resolved"]["LabelingEfficiency"]["Value"]
    assert a == ps["LabelingEfficiency"]
    lt, bolus = ps["ArterialSpinLabelingType"], protocols.bolus_duration(ps)
    pz = quant.slice_plds(ps["PostLabelingDelay"], protocols.slice_offsets(ps))       # PLD, or TI for PASL, per slice
    t_sig = pz if lt == "PASL" else pz + bolus                                        # the kinetic clock per slice
    dm_model = kinetic.delta_m(t_sig, GM.perfusion, GM.att, GM.t1, 1.0, label_type=lt, tau=bolus, alpha=a)
    if lt == "PASL":
        lt_cbf[rn] = quant.cbf_pasl(dms, m0s, pz[None, None, :], ti1=bolus, alpha=a)
        expect = quant.cbf_pasl(dm_model, 1.0, pz, ti1=bolus, alpha=a)
    else:
        lt_cbf[rn] = quant.cbf_pcasl(dms, m0s, pz[None, None, :], tau=bolus, alpha=a)
        expect = quant.cbf_pcasl(dm_model, 1.0, pz, tau=bolus, alpha=a)
    lt_alpha[rn], lt_expect[rn] = a, np.average(expect, weights=n_gm_z)
fig, axes = plotting.mosaic({f"{rn.upper()}, α {lt_alpha[rn]:g}": np.where(mask, lt_cbf[rn], np.nan) for rn in lt_cbf}, k=K, kind="cbf", figsize_per=2.9)
for rn, c in lt_cbf.items():
    print(f"{rn.upper():>5}: GM {c[gm].mean():5.1f} ({100 * c[gm].mean() / truth[gm].mean():.0f} % of truth; the model predicts {100 * lt_expect[rn] / GM.perfusion:.0f} %), "
          f"WM {c[wm].mean():5.1f} ml/100 g/min; GM CBF SD {c[gm].std():.1f}; with α 0.85 for everything GM would be {c[gm].mean() * lt_alpha[rn] / 0.85:.1f}")
```

PCASL and CASL give the same map once each uses its own efficiency: the simulator labels
them identically except for $\alpha$, and the formula divides it back out. The PASL map is
different in two ways that the model predicts (the printed expectation, from
`kinetic.delta_m` with the pulsed branch and the PASL formula). Its gray matter mean is
18 % higher than PCASL's, not because pulsed labeling is better but because its label has
spent less time in tissue: a bolus of 0.7 s read at TI 1.8 s has been in gray matter for
0.3 to 1.0 s, against 1.0 to 2.8 s for the PCASL bolus, so the $T_{1b}$ assumption costs it
10 % rather than 24 %. For the same reason its white matter is higher too, even though its
white matter bolus has not fully arrived ($\delta + \mathrm{TI}_1 = 1.9$ s exceeds the
TI). And the PASL map is noisier (a larger spread across gray matter voxels), because a
0.7 s bolus produces less $\Delta M$ per unit of perfusion than a 1.8 s one and the PASL
formula scales the noise up with it. The last printed number shows what using the PCASL
efficiency for everything would do: CASL underestimated by 20 %, PASL overestimated by
15 %. The efficiency is a property of the labeling, not of the brain, and it has to travel
with the data.

## The gray-to-white contrast

The single-delay map's contrast between tissues is a familiar number: the ratio of gray to
white matter CBF is about 3 in the phantom and in PET, and single-delay ASL reports larger
ratios. The histogram shows why.

```{code-cell} python
:tags: [hide-input]
fig, ax = plt.subplots(figsize=(7.5, 3.3))
bins = np.linspace(-40, 120, 81)
for name, roi in [("GM", gm), ("WM", wm)]:
    ax.hist(cbf[roi], bins, histtype="stepfilled", alpha=0.45, color=TISSUE_COLORS[name], label=f"{name} voxels, estimate")
    ax.axvline(truth[roi].mean(), color=TISSUE_COLORS[name], lw=1.5, ls="--")
    ax.axvline(cbf[roi].mean(), color=TISSUE_COLORS[name], lw=1.5)
ax.set(xlabel="CBF (ml/100 g/min)", ylabel="voxels", title="dashed: truth; solid: mean estimate")
ax.legend()
fig.tight_layout()
print(f"GM/WM ratio: truth {truth[gm].mean() / truth[wm].mean():.1f}, white-paper formula {cbf[gm].mean() / cbf[wm].mean():.1f}, "
      f"with the tissue T1 and ATT {cbf_kin[gm].mean() / cbf_kin[wm].mean():.1f}; negative WM voxels {100 * (cbf[wm] < 0).mean():.0f} %")
```

Both distributions sit left of their truth, white matter's by a larger fraction, so the
ratio of the means rises from 2.9 to 6.2; with the tissue T1 and the transit time it is
3.1. The white matter distribution also straddles zero: 35 % of its voxels are negative at
30 pairs, because its $\Delta M$ of 4 image units is within one noise standard deviation
of zero. A white matter CBF from single-delay ASL is
therefore a noisy, biased number, and a gray-to-white ratio from it is not comparable with
one from PET or from a multi-delay fit unless both are computed the same way.

## What this implies for acquisition

- **Record what the formula needs.** Labeling type, LD, PLD, the efficiency, the field
  strength, the M0 scan and its TR, and the slice timing all go in the sidecar; the
  constants you assume ($\lambda$, $T_{1b}$) go in the methods.
- **Correct the slice timing of a 2D readout**, or the CBF map carries a 37 % gradient from
  the first slice to the last that looks like perfusion.
- **Set the PLD longer than the longest transit time you expect** (1.8 s in healthy adults,
  2.0 s or more in older or vascular patients): below it, the formula underestimates
  steeply; above it, the loss is gradual and the map gets noisier.
- **Expect single-delay white matter CBF to be low and noisy** and do not compare it, or a
  gray-to-white ratio, across protocols or with other modalities.
- **Know which T1 the label decays with.** The formula's $T_{1b}$ gives one number, a tissue
  T1 another; in this phantom the difference is 24 % in gray matter. If the transit time
  and T1 matter to the question, acquire several delays ([Chapter 15](./15-multi-delay.md)).

## Further reading

The white paper {cite:p}`alsop2015` states the formula, its constants, and the recommended
delays; the kinetic model it comes from is {cite:t}`buxton1998`. The reduced transit-time
sensitivity of a long delay is {cite:t}`alsop1996`; the pulsed bolus cut-off that makes the
PASL formula possible is {cite:t}`wong1998` and {cite:t}`luh1999`. The 2D versus 3D readout
question is {cite:t}`vidorreta2013`. The uncertain status of white matter ASL is discussed by
{cite:t}`vanosch2018`, and the multi-delay recommendations that address the transit time are
{cite:t}`woods2024`.
