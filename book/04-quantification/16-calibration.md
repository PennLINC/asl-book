---
title: "16. Calibration: M0, λ, T1, α, and the readout"
kernelspec:
  name: python3
  display_name: Python 3
---

:::{admonition} Simulated datasets in this chapter
:class: note
- **Built in this page:** the saturation and T2 factors of the calibration image evaluated from the phantom's constants, and the partial derivatives of the CBF formula ([Appendix B](../appendices/b-data-manifest.md#app-b-package-data)).
- **`ref-pcasl`**: the reference series, its separate M0 scan at TR 8 s, and its mean control image ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-ref-pcasl)).
- **`m0-types`**: the same protocol calibrated three ways: a separate M0 scan at TR 8 s, a separate M0 scan at TR 2 s, and two `m0scan` volumes included in the series at TR 4.5 s ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-m0-types)).
- **`te-sweep`**: the reference protocol at echo times of 12, 30, and 60 ms ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-te-sweep)).
- **`label-types`**: PCASL, CASL, and PASL with their labeling efficiencies recorded in the sidecar ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-label-types)).
- **`multi-pld`**: the six-delay series, refitted with different assumed tissue T1 values ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-multi-pld)).

Pipeline-tier datasets are simulated offline by aslscan ([Chapter 0.2](../00-frontmatter/the-simulated-datasets.md)).
:::

## Learning goals

After this chapter you can:

- say what the calibration image in the CBF formula has to be, and turn any of the four
  images a dataset may offer (a separate M0 scan, included `m0scan` volumes, the mean
  control image, a CSF reference) into it
- correct a calibration image for the T1 saturation of its repetition time and for the T2
  difference between blood and tissue at the echo time, and say how large each correction is
- quote the percent change in CBF per unit change in λ, T1b, α, and the assumed tissue T1
- read the sidecar as the record of what was assumed

```{code-cell} python
:tags: [hide-cell]
import numpy as np
import matplotlib.pyplot as plt

from aslbook import data, kinetic, phantom, presets, protocols, quant
from aslbook.plotting import INK, PALETTE, TISSUE_COLORS, set_style, show_slice, take_slice

set_style()
K = phantom.DISPLAY_SLICE
GM, WM, CSF = (presets.TISSUES[t] for t in ("GM", "WM", "CSF"))
R = presets.REFERENCE
TAU, PLD, TE = R.labeling_duration, R.post_labeling_delay, R.echo_time


def summarize(run):
    """Mean difference image, mean control image, sidecar, per-slice PLDs and tissue masks of a run."""
    mag, ctx, p = run.mag(), run.context(), run.sidecar()
    dm = quant.subtract(mag, ctx).mean(-1)
    ctrl = mag[..., [i for i, r in enumerate(ctx) if r == "control"]].mean(-1)
    fr = run.fractions()
    masks = {t: fr[t.lower()] >= 0.9 for t in ("GM", "WM", "CSF")}
    pld = p["PostLabelingDelay"]
    plds = None if isinstance(pld, list) else quant.slice_plds(pld, protocols.slice_offsets(p))[None, None, :]
    return dict(mag=mag, ctx=ctx, p=p, dm=dm, ctrl=ctrl, fr=fr, masks=masks, plds=plds, offsets=np.asarray(protocols.slice_offsets(p)))


def cbf_model(dm, m0_tissue, att, t1, t):
    """Invert the kinetic model at the true transit time and tissue T1: dM per unit CBF from
    kinetic.delta_m with M0 = the (corrected) tissue M0, so that only the calibration is tested."""
    unit = kinetic.delta_m(t, 1.0, att, t1, m0_tissue)
    return np.divide(dm, unit, out=np.zeros_like(dm), where=unit > 0)


def gm_mean(img, masks):
    return float(img[masks["GM"]].mean())
```

## What the formula asks for

The single-delay formula of [Chapter 14](./14-cbf-quantification.md)
{cite:p}`alsop2015` divides the difference signal by the equilibrium magnetization of
arterial blood:

$$
f = \frac{6000\,\lambda\,\Delta M\,e^{\mathrm{PLD}/T_{1b}}}{2\,\alpha\,T_{1b}\,M_0\,\bigl(1-e^{-\tau/T_{1b}}\bigr)},
\qquad \frac{M_0}{\lambda} = M_{0b},
$$

where $M_0$ is the equilibrium magnetization of the tissue in the voxel and $\lambda$, the
blood-brain partition coefficient of water {cite:p}`herscovitch1985` (ml of water per gram
of tissue over ml of water per ml of blood), converts it to that of blood, $M_{0b}$. The formula is a ratio: $\Delta M$
and $M_0$ must be in the same units, measured with the same coil, the same receiver gain,
the same voxel, and the same readout. That is the whole purpose of the calibration image.
The ratio is small. On the reference protocol a pure gray-matter voxel has $\Delta M \approx
30$ against $M_0 \approx 6450$ image units, and any factor left in one of the two images
and not the other passes straight into the CBF.

No scanner produces $M_0$ directly. Every calibration image is a spin-echo image acquired
at some repetition time and echo time, so what it holds is

$$
S_{M_0} = k\,M_0\,\bigl(1-e^{-\mathrm{TR}/T_1}\bigr)\,e^{-\mathrm{TE}/T_2},
$$

with $k$ the scanner's scale, $T_1$ and $T_2$ those of the tissue in the voxel. The
difference image holds $\Delta M_{\text{meas}} = k\,\Delta M\,e^{-\mathrm{TE}/T_{2b}}$: the
labeled water is in blood (aslscan keeps it there; in vivo it takes on the tissue's $T_2$
once it has exchanged {cite:p}`gregori2013`, so the blood value is an upper bound on the difference). The
calibration therefore needs two corrections on top of the division by $\lambda$:

$$
M_0 = \frac{S_{M_0}}{1-e^{-\mathrm{TR}/T_1}}\;\frac{e^{-\mathrm{TE}/T_{2b}}}{e^{-\mathrm{TE}/T_2}},
$$

which is what `quant.m0_correction` computes. The first factor undoes the saturation of the
M0 scan's repetition time; the second replaces the tissue's transverse decay at the echo
time by the blood's, so that the ratio $\Delta M / M_0$ is the one the formula expects.

## Where the calibration image comes from

The white paper {cite:p}`alsop2015` and ASL-BIDS {cite:p}`clement2022` recognize four
situations, recorded in the sidecar's `M0Type`:

- **`Separate`**: a scan of its own, `sub-01_m0scan.nii.gz`, with its own sidecar
  (`run.m0scan()`, `run.m0scan_sidecar()`). It should use the same readout as the ASL
  series with a long repetition time and no background suppression. The reference protocol
  acquires it at TR 8 s.
- **`Included`**: `m0scan` rows inside the ASL series, listed in `aslcontext.tsv`. They are
  acquired at the series' own repetition time, 4.5 s here, so they are more saturated than a
  separate scan.
- **`Estimate`**: no image, only a single number in the sidecar's `M0Estimate`, typically
  from a reference region.
- **`Absent`**: nothing. If the series was acquired without background suppression, the
  mean control image is a calibration image at TR 4.5 s. If it was suppressed, the control
  images hold no static signal and there is no calibration; only relative CBF remains.

A fifth route uses the M0 scan differently: instead of the tissue in every voxel it reads
$M_0$ in pure CSF, whose water content is known, and scales it to blood. It is the route
when the partition coefficient of the tissue is not trusted, and it is discussed below.

## See it: the saturation of the calibration image

```{code-cell} python
:tags: [hide-input]
tr = np.linspace(0.05, 12, 400)
fig, ax = plt.subplots(figsize=(7, 3.2))
for t in (GM, WM, CSF):
    ax.plot(tr, 1 - np.exp(-tr / t.t1), color=TISSUE_COLORS[t.name], label=f"{t.name}, T1 {t.t1:g} s")
for x, name in [(2.0, "separate, TR 2 s"), (4.5, "included / control, TR 4.5 s"), (8.0, "separate, TR 8 s")]:
    ax.axvline(x, color=INK["secondary"], lw=0.8, ls=":")
    ax.text(x + 0.1, 0.08, name, rotation=90, va="bottom", fontsize=7, color=INK["secondary"])
ax.set(xlabel="repetition time of the calibration image (s)", ylabel="fraction of M0 recovered", ylim=(0, 1.02))
ax.legend(loc="lower right")
fig.tight_layout()

for x in (2.0, 4.5, 8.0):
    print(f"TR {x:>3g} s: GM {1 - np.exp(-x / GM.t1):.3f}   WM {1 - np.exp(-x / WM.t1):.3f}   CSF {1 - np.exp(-x / CSF.t1):.3f}")
print(f"T2 factor at TE {TE * 1000:g} ms: GM {np.exp(-TE / GM.t2):.3f}   blood {np.exp(-TE / presets.T2_BLOOD):.3f}   "
      f"ratio blood/GM {np.exp(-TE / presets.T2_BLOOD) / np.exp(-TE / GM.t2):.3f}")
```

Each curve is $1-e^{-\mathrm{TR}/T_1}$ for one tissue. At TR 8 s gray matter has recovered
0.998 of its equilibrium and the correction is negligible; at TR 4.5 s it has recovered
0.966, a 3.5 percent effect; at TR 2 s only 0.778, so a CBF computed from an uncorrected
TR 2 s M0 scan is 1/0.778 = 1.29 times too high. CSF, with its 3 s $T_1$, is the slow one:
0.93 even at TR 8 s and 0.49 at TR 2 s, which matters for the CSF route. The $T_2$
correction is independent of TR: at TE 12 ms the blood's factor is 0.930 and gray matter's
0.861, so an uncorrected calibration overestimates CBF by their ratio, 1.080, on every
route.

## See it: the four calibration images

```{code-cell} python
:tags: [hide-input]
m0types = data.load_dataset("m0-types")
runs = {name: m0types.run(name) for name in ("separate-tr8", "separate-tr2", "included")}
S = {name: summarize(r) for name, r in runs.items()}

# The calibration image of each route, in image units, and its repetition time.
routes = {}
routes["separate M0, TR 8 s"] = (runs["separate-tr8"].m0scan(), runs["separate-tr8"].m0scan_sidecar()["RepetitionTimePreparation"])
routes["separate M0, TR 2 s"] = (runs["separate-tr2"].m0scan(), runs["separate-tr2"].m0scan_sidecar()["RepetitionTimePreparation"])
rows = [i for i, r in enumerate(S["included"]["ctx"]) if r == "m0scan"]
routes["included m0scan rows, TR 4.5 s"] = (S["included"]["mag"][..., rows].mean(-1), S["included"]["p"]["RepetitionTimePreparation"])
routes["mean control, TR 4.5 s"] = (S["separate-tr8"]["ctrl"], S["separate-tr8"]["p"]["RepetitionTimePreparation"])

fig, axes = plt.subplots(1, 4, figsize=(11, 3.2))
for ax, (name, (img, tr_)) in zip(axes, routes.items()):
    show_slice(ax, img, K, title=name, vmin=0, vmax=7000)
fig.tight_layout()

masks = S["separate-tr8"]["masks"]
for name, (img, tr_) in routes.items():
    print(f"{name:<34} GM {img[masks['GM']].mean():7.0f}   WM {img[masks['WM']].mean():7.0f}   CSF {img[masks['CSF']].mean():7.0f}")
```

All four are the same brain at the same window (0 to 7000 image units). The TR 8 s scan
is the brightest: gray matter 6450, CSF 6110. The TR 2 s scan is darker everywhere, 5029
in gray matter, and its CSF has fallen to about half, so its contrast is $T_1$-weighted
rather than proton density. The included rows and the mean control image are the same
measurement at TR 4.5 s, 6246 in gray matter, and differ only in how many volumes were
averaged: two `m0scan` rows against fifteen control images.

## Measure it: CBF from each route, uncorrected and corrected

Every route is applied to the difference image of its own run, first as the raw image,
then after `quant.m0_correction` with that route's repetition time. Two numbers are
reported for each. The first is the white-paper formula's gray-matter CBF. The second
inverts the kinetic model at the true transit time and tissue $T_1$ (`cbf_model` in the
setup cell) with the same calibrated $M_0$: it removes the formula's own approximation, so
it tests the calibration alone.

```{code-cell} python
:tags: [hide-input]
run_by_route = {"separate M0, TR 8 s": "separate-tr8", "separate M0, TR 2 s": "separate-tr2",
                "included m0scan rows, TR 4.5 s": "included", "mean control, TR 4.5 s": "separate-tr8"}
truth = {n: (r.truth("perfusion"), r.truth("att"), r.truth("T1map")) for n, r in runs.items()}
t_signal = PLD + TAU + S["separate-tr8"]["offsets"][None, None, :]

results = {}
for name, (img, tr_) in routes.items():
    s = S[run_by_route[name]]
    m0c = quant.m0_correction(img, tr=tr_)
    wp_raw = quant.cbf_pcasl(s["dm"], img, s["plds"])
    wp_cor = quant.cbf_pcasl(s["dm"], m0c, s["plds"])
    perf, att, t1 = truth[run_by_route[name]]
    model = cbf_model(s["dm"], m0c, att, t1, t_signal)
    results[name] = (gm_mean(wp_raw, s["masks"]), gm_mean(wp_cor, s["masks"]), gm_mean(model, s["masks"]))
true_gm = gm_mean(truth["separate-tr8"][0], masks)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
x = np.arange(len(results))
w = 0.38
axes[0].bar(x - w / 2, [v[0] for v in results.values()], w, color=PALETTE[7], label="raw image")
axes[0].bar(x + w / 2, [v[1] for v in results.values()], w, color=PALETTE[0], label="m0_correction")
axes[0].axhline(results["separate M0, TR 8 s"][1], color=INK["secondary"], lw=0.8, ls="--")
axes[0].set(ylabel="GM CBF, white-paper formula (ml/100 g/min)", ylim=(0, 70))
axes[0].legend(loc="upper left")
axes[1].bar(x, [v[2] for v in results.values()], 0.6, color=TISSUE_COLORS["GM"])
axes[1].axhline(true_gm, color="k", lw=1, ls="--", label=f"truth in pure-GM voxels, {true_gm:.1f}")
axes[1].set(ylabel="GM CBF, kinetic model at the true ATT", ylim=(0, 70))
axes[1].legend(loc="lower left")
for ax in axes:
    ax.set_xticks(x, [n.replace(", ", "\n") for n in results], fontsize=7.5)
fig.tight_layout()

print(f"{'route':<34} {'raw':>6} {'corrected':>10} {'model':>7}   (truth in pure GM {true_gm:.1f})")
for name, (a, b, c) in results.items():
    print(f"{name:<34} {a:6.1f} {b:10.1f} {c:7.1f}")
```

Left, the white-paper formula. The raw images disagree by a factor of 1.4: 47.4 from the
TR 8 s scan, 60.8 from the TR 2 s scan, 49.0 from the included rows. After the correction
the four routes agree to within 0.1 ml/100 g/min, at 43.7. That is the point of this
chapter: the calibration image is interchangeable once its repetition time and echo time
are accounted for, and not before. The residual gap to the true 59.1 is not a calibration
error. It is the formula's assumption that the label decays with $T_{1b}$ after it
arrives, whereas the simulator's label exchanges into tissue at once and decays with the
tissue's shorter $T_1'$ ([Chapter 14](./14-cbf-quantification.md)); the right panel shows
that with the kinetic model evaluated at the true transit time, the same corrected $M_0$
returns 58.7 from every route, within 0.4 of the truth. The remaining bias is the
Rician floor and the residual partial volume in voxels that are 90 percent gray matter.

The uncorrected TR 2 s scan is the case to remember: a 39 percent overestimate, 29 from
the saturation and 8 from the $T_2$ mismatch, from a scan parameter that is sitting in
the M0 sidecar's `RepetitionTimePreparation`. Pipelines
read it (ASLPrep and ExploreASL both apply the saturation correction with an assumed
tissue $T_1$ {cite:p}`adebimpe2022,mutsaerts2020`), but only if the field is there.

## See it: the echo time and the T2 of blood

The labeled water and the static tissue decay at different rates during the readout.
The simulator takes the $T_2$ of arterial blood at 3 T as 165 ms, and the phantom's gray
matter has 80 ms. The measured ratio
$\Delta M / S_{M_0}$ in gray matter therefore drifts with the echo time as
$e^{-\mathrm{TE}/T_{2b}} / e^{-\mathrm{TE}/T_{2,\mathrm{GM}}}$, and CBF from an
uncorrected calibration rises with it. The `te-sweep` dataset acquires the reference
protocol at 12, 30, and 60 ms.

```{code-cell} python
:tags: [hide-input]
tesweep = data.load_dataset("te-sweep")
te_names = ("te12", "te30", "te60")
tes, ratio, cbf_raw, cbf_cor = [], [], [], []
for name in te_names:
    r = tesweep.run(name)
    s = summarize(r)
    m0 = r.m0scan()
    te = s["p"]["EchoTime"]
    tes.append(te)
    ratio.append(gm_mean(s["dm"], s["masks"]) / gm_mean(m0, s["masks"]))
    cbf_raw.append(gm_mean(quant.cbf_pcasl(s["dm"], m0, s["plds"]), s["masks"]))
    m0c = quant.m0_correction(m0, tr=8.0, te=te)
    cbf_cor.append(gm_mean(quant.cbf_pcasl(s["dm"], m0c, s["plds"]), s["masks"]))
tes, ratio = np.array(tes), np.array(ratio)
te_fine = np.linspace(0, 0.07, 100)
mismatch = lambda te: np.exp(-te / presets.T2_BLOOD) / np.exp(-te / GM.t2)
pred = mismatch(te_fine) / mismatch(tes[0])  # relative to the 12 ms acquisition, like the measurement
pred_at = mismatch(tes)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
axes[0].plot(te_fine * 1000, pred, color=INK["secondary"], lw=1.2, label="predicted: exp(−TE/T2b) / exp(−TE/T2,GM)")
axes[0].plot(tes * 1000, ratio / ratio[0], "o", color=TISSUE_COLORS["GM"], label="measured ΔM / M0 in GM")
axes[0].set(xlabel="echo time (ms)", ylabel="ratio relative to TE 12 ms", ylim=(0.95, 1.45))
axes[0].legend(loc="upper left")
axes[1].plot(tes * 1000, cbf_raw, "o-", color=PALETTE[7], label="raw M0 scan")
axes[1].plot(tes * 1000, cbf_cor, "o-", color=PALETTE[0], label="m0_correction (TR and TE)")
axes[1].set(xlabel="echo time (ms)", ylabel="GM CBF (ml/100 g/min)", ylim=(30, 70))
axes[1].legend(loc="upper left")
fig.tight_layout()

for te, rt, pr, a, b in zip(tes, ratio / ratio[0], pred_at / pred_at[0], cbf_raw, cbf_cor):
    print(f"TE {te * 1000:2.0f} ms: measured ratio x{rt:.3f}, predicted x{pr:.3f};  GM CBF raw {a:.1f}, corrected {b:.1f}")
```

Left, the gray-matter ratio $\Delta M / S_{M_0}$ relative to its value at 12 ms: it rises
by a factor of 1.123 at 30 ms and 1.361 at 60 ms, and the prediction from the two $T_2$
values is 1.123 and 1.362. Right, what that does to CBF: uncorrected, 47.4 at 12 ms
becomes 53.2 at 30 ms and 64.6 at 60 ms, a 36 percent spread from an acquisition
parameter that changes nothing about the perfusion; corrected with both factors, all three
echo times give 43.7 to 43.8. The correction depends on a blood $T_2$ that is not measured
and varies with oxygenation and hematocrit {cite:p}`zhao2007`, which is the argument for the short echo time
the white paper recommends {cite:p}`alsop2015`: at 12 ms the whole factor is 1.08 and an error of 30 ms in
$T_{2b}$ moves CBF by about 1 percent; at 60 ms the same error moves it by close to 7
percent.

## See it: the CSF reference

When the tissue partition coefficient is not trusted, or when a pipeline prefers a single
calibration number, $M_{0b}$ can be read from cerebrospinal fluid {cite:p}`chalela2000`;
{cite:t}`pinto2020` compare this reference-region calibration with the voxelwise one.
CSF is nearly pure water, blood is about 87 percent water by volume
{cite:p}`herscovitch1985`, so

$$
M_{0b} = \frac{M_{0,\mathrm{CSF}}}{\lambda_{\mathrm{CSF}}}, \qquad \lambda_{\mathrm{CSF}} \approx 1.15,
$$

which is the same operation as dividing tissue $M_0$ by $\lambda = 0.9$, with a different
constant (equivalently, multiplying by 0.87). $M_{0,\mathrm{CSF}}$ is the mean of the M0
scan in voxels that are pure CSF, corrected for CSF's own saturation (0.93 at TR 8 s) and
its own $T_2$ (300 ms), and the single value is then used for every voxel.

```{code-cell} python
:tags: [hide-input]
ref = data.load_dataset("ref-pcasl").run()
s = summarize(ref)
m0 = ref.m0scan()
csf_vox = s["masks"]["CSF"]
m0_csf = float(m0[csf_vox].mean())
m0_csf_c = m0_csf / (1 - np.exp(-8.0 / CSF.t1)) * np.exp(-TE / presets.T2_BLOOD) / np.exp(-TE / CSF.t2)
m0b_csf = m0_csf_c / 1.15
m0b_gm = float((quant.m0_correction(m0, tr=8.0) / presets.LAMBDA)[s["masks"]["GM"]].mean())
m0b_true = R.signal_scale * GM.m0 / presets.LAMBDA * np.exp(-TE / presets.T2_BLOOD)
lam_csf_phantom = CSF.m0 / (GM.m0 / presets.LAMBDA)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), gridspec_kw={"width_ratios": [1, 1.6]})
show_slice(axes[0], m0, K, title="M0 scan with the pure-CSF voxels (csf ≥ 0.9)", vmin=0, vmax=7000)
axes[0].contour(take_slice(csf_vox.astype(float), K), levels=[0.5], colors=[TISSUE_COLORS["CSF"]], linewidths=1.2)
names = ["tissue route\nM0,GM / 0.9", "CSF route\nM0,CSF / 1.15", f"CSF route\nM0,CSF / {lam_csf_phantom:.2f}\n(phantom's own ratio)", "true M0b"]
vals = [m0b_gm, m0b_csf, m0_csf_c / lam_csf_phantom, m0b_true]
axes[1].bar(names, vals, 0.6, color=[TISSUE_COLORS["GM"], TISSUE_COLORS["CSF"], TISSUE_COLORS["CSF"], INK["secondary"]])
axes[1].set(ylabel="blood M0 in image units (at TE 12 ms)", ylim=(0, 9000))
axes[1].tick_params(axis="x", labelsize=7.5)
fig.tight_layout()

print(f"pure-CSF voxels: {csf_vox.sum()};  M0 scan in CSF {m0_csf:.0f}, saturation- and T2-corrected {m0_csf_c:.0f}")
print(f"M0b from the tissue route {m0b_gm:.0f}, from CSF / 1.15 {m0b_csf:.0f} ({m0b_csf / m0b_true - 1:+.1%}), true {m0b_true:.0f}")
print(f"phantom M0: GM {GM.m0:.1f}, CSF {CSF.m0:.1f}, blood (GM/0.9) {GM.m0 / presets.LAMBDA:.1f}; CSF/blood = {lam_csf_phantom:.2f}")
```

Left, the 142 voxels that are at least 90 percent CSF, in the lateral ventricles. Right,
the blood $M_0$ each route produces. The tissue route gives 7762 against the true 7710,
within 0.7 percent. The CSF route with the textbook constant gives 5526, 28 percent low,
which would put CBF 39 percent high. The reason is the phantom, not the method: the
ASLDRO phantom {cite:p}`olivertaylor2021` assigns CSF an $M_0$ of 68.0 and gray matter 74.6, whereas the
two constants of the routes, 1.15 for CSF and 0.9 for tissue, imply a CSF $M_0$ that is
1.15/0.9 = 1.28 times the tissue's. The
phantom's own ratio of CSF to blood is 0.82, and with that constant the CSF route lands on
the truth (third bar). The lesson transfers: the CSF route replaces one assumed constant
($\lambda$) with another ($\lambda_{\mathrm{CSF}}$, plus CSF's $T_1$ and $T_2$), and it
depends on finding voxels that are actually pure CSF. In a brain with small ventricles at
3.5 mm voxels there may be only a handful, and a single value from them, with partial
volume from the ventricle wall, sets the scale of every CBF map.

## Measure it: the sensitivity of CBF to the assumed constants

Three constants in the formula are never measured in a routine scan: $\lambda$, $T_{1b}$,
and $\alpha$. The dependence on each is simple enough to write down. CBF is proportional
to $\lambda$ and to $1/\alpha$, so a fractional error in either is a fractional error in
CBF. $T_{1b}$ enters three times, and the derivative is evaluated numerically.

```{code-cell} python
:tags: [hide-input]
dm_gm, m0_gm = 30.0, np.exp(-TE / presets.T2_BLOOD) * R.signal_scale * GM.m0  # representative GM values
base = quant.cbf_pcasl(dm_gm, m0_gm, PLD)
step = lambda kw, v: 100 * (quant.cbf_pcasl(dm_gm, m0_gm, PLD, **{kw: v}) / base - 1)  # percent change
panels = [
    ("lam", np.linspace(0.80, 1.00, 41), "λ (ml/g)", {0.9: "whole brain", 0.98: "GM"}),
    ("t1b", np.linspace(1.45, 1.85, 41), "T1 of blood (s)", {1.65: "white paper", 1.56: "Hct 0.50", 1.78: "Hct 0.35"}),
    ("alpha", np.linspace(0.65, 1.00, 41), "labeling efficiency α", {0.85: "PCASL default", 0.80: "measured 0.80", 0.68: "CASL"}),
]
fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
for ax, (kw, xs, xl, marks) in zip(axes, panels):
    ax.plot(xs, [step(kw, v) for v in xs], color=PALETTE[0])
    ax.axhline(0, color=INK["secondary"], lw=0.6)
    for xv, lab in marks.items():
        ax.plot(xv, step(kw, xv), "o", color=PALETTE[1])
        ax.annotate(lab, (xv, step(kw, xv)), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set(xlabel=xl, ylabel="CBF change (%)")
fig.tight_layout()

print(f"lambda 0.90 -> 0.95: {step('lam', 0.95):+.1f} %   0.90 -> 0.98 (GM-specific): {step('lam', 0.98):+.1f} %")
print(f"T1b 1.65 -> 1.75 s: {step('t1b', 1.75):+.1f} %   1.65 -> 1.55 s: {step('t1b', 1.55):+.1f} %")
print(f"alpha 0.85 -> 0.90: {step('alpha', 0.90):+.1f} %   0.85 -> 0.80 (measured): {step('alpha', 0.80):+.1f} %")
for hct in (0.35, 0.42, 0.50):
    t1b_h = 1 / (0.52 * hct + 0.38)
    print(f"hematocrit {hct:.2f}: T1b = {t1b_h:.2f} s (Lu et al. 2004), CBF {step('t1b', t1b_h):+.1f} % relative to 1.65 s")
```

Each panel moves one constant with the others at their white-paper values. A change of
0.05 in $\lambda$ moves CBF by 5.6 percent, and using the gray-matter-specific 0.98
instead of the whole-brain 0.9 {cite:p}`herscovitch1985` raises gray-matter CBF by 8.9
percent. $T_{1b}$ is the
most sensitive of the three: a change of 0.1 s moves CBF by 8 to 10 percent in the
opposite direction (+10.4 percent for 1.55 s, −8.4 percent for 1.75 s), because it enters
the exponential of the delay, the exponential of the bolus, and the prefactor. The
hematocrit dependence measured by {cite:t}`lu2004`, $1/T_{1b} = 0.52\,\mathrm{Hct} +
0.38\ \mathrm{s^{-1}}$, spans 1.56 s at a hematocrit of 0.50 to 1.78 s at 0.35, so a
subject with polycythemia and one with anemia differ by 20 percent in reported CBF (+9.0
and −10.6 percent) from this constant alone. A change of 0.05 in $\alpha$ moves CBF by
5.6 to 6.2 percent; a PCASL efficiency measured at 0.80 rather than the assumed 0.85
raises CBF by 6.2 percent, and the CASL value of 0.68 {cite:p}`wang2005,wu2007` is 25 percent away. None of these
is a random error. Each is a fixed scale on every map from one scanner and protocol,
invisible in a within-study comparison and decisive when two studies' absolute values are
compared.

## Measure it: the assumed tissue T1 in a multi-delay fit

The single-delay formula does not contain the tissue $T_1$. The kinetic model does,
through $T_1'$, which sets how fast the difference decays after the bolus has arrived
([Chapter 5](../02-labeling/05-kinetic-model.md)), and a multi-delay fit
([Chapter 15](./15-multi-delay.md)) has to assume it. The `multi-pld` series is refitted
below with `quant.fit_multi_pld` at five assumed tissue $T_1$ values, using the
calibrated M0 scan and the slice-timing offsets.

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
t1_assumed = [0.9, 1.1, 1.33, 1.5, 1.7]
fits = {t1: quant.fit_multi_pld(d_mean, delays, m0c_mp, mask=mask_mp, slice_offsets=sm["offsets"], t1_tissue=t1) for t1 in t1_assumed}

gm_mp = sm["masks"]["GM"]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
axes[0].plot(t1_assumed, [fits[t1][0][gm_mp].mean() for t1 in t1_assumed], "o-", color=TISSUE_COLORS["GM"], label="fitted GM CBF")
axes[0].axhline(GM.perfusion, color="k", lw=0.8, ls="--", label="truth 60")
axes[1].plot(t1_assumed, [fits[t1][1][gm_mp].mean() for t1 in t1_assumed], "o-", color=TISSUE_COLORS["GM"], label="fitted GM ATT")
axes[1].axhline(GM.att, color="k", lw=0.8, ls="--", label="truth 0.8 s")
for ax, yl in zip(axes, ("fitted CBF in pure GM (ml/100 g/min)", "fitted ATT in pure GM (s)")):
    ax.axvline(GM.t1, color=INK["secondary"], lw=0.8, ls=":")
    ax.text(GM.t1 + 0.01, ax.get_ylim()[1], "phantom GM T1", rotation=90, va="top", fontsize=7, color=INK["secondary"])
    ax.set(xlabel="assumed tissue T1 (s)", ylabel=yl)
    ax.legend(loc="upper right")
fig.tight_layout()

for t1 in t1_assumed:
    c, a = fits[t1]
    print(f"assumed T1 {t1:.2f} s: GM CBF {c[sm['masks']['GM']].mean():5.1f}, ATT {a[sm['masks']['GM']].mean():.2f} s;   "
          f"WM CBF {c[sm['masks']['WM']].mean():5.1f}, ATT {a[sm['masks']['WM']].mean():.2f} s")
```

Dashed lines are the truth (60 ml/100 g/min and 0.8 s); the dotted vertical is the
phantom's gray-matter $T_1$ of 1.33 s. The assumed $T_1$ is the strongest lever in this
chapter. Between 1.1 and 1.5 s, gray-matter CBF moves from 82.1 to 59.2 and the transit
time from 1.00 to 0.83 s, about 1 percent in CBF per 0.01 s of $T_1$ over the range of
values a pipeline might pick (published gray-matter values at 3 T run from 1.33 s
{cite:p}`wansapura1999` to 1.82 s {cite:p}`stanisz2005`). At the true 1.33 s the fit gives 67.1 and 0.89 s, above
the truth because five pairs per delay leave the transit-time grid search biased by noise
([Chapter 15](./15-multi-delay.md)), and the coupling of ATT and CBF carries that bias
into the CBF. White matter is not shown: with a difference of about 4 image units against
a per-delay noise near 25, its fit returns a transit time of 1.7 s at every assumed
$T_1$, the sign of a fit the data do not constrain. A single assumed tissue $T_1$ is a
compromise in every voxel that is not pure gray matter, which is one reason the white
paper's single-delay formula leaves it out, and why multi-delay fits either take a
$T_1$ map or accept this dependence.

## The sidecar as the record

Every constant above is a number somebody chose, and the only place the choice survives
is the sidecar. ASL-BIDS makes `LabelingEfficiency` an optional field {cite:p}`clement2022`; aslscan
records
every constant it resolved, with its source, under `AslscanSimulation.Resolved`.

```{code-cell} python
:tags: [hide-input]
lt = data.load_dataset("label-types")
print(f"{'run':<6} {'type':<6} {'sidecar LabelingEfficiency':>27}   Resolved (value, source)")
for name in ("pcasl", "casl", "pasl"):
    r = lt.run(name)
    p, res = r.sidecar(), r.simulation()["Resolved"]
    keys = ("LabelingEfficiency", "LambdaBloodBrain", "T1ArterialBlood", "T2Blood")
    resolved = "; ".join(f"{k} {res[k]['Value']} ({res[k]['Source']})" for k in keys)
    print(f"{name:<6} {p['ArterialSpinLabelingType']:<6} {str(p.get('LabelingEfficiency')):>27}   {resolved}")
```

The three runs of `label-types` record 0.85 for PCASL, 0.68 for CASL, and 0.98 for PASL,
and the `Resolved` block says which came from the simulator's defaults, which from the
phantom, and which from the run's overlay. A real scanner writes none of this: the
efficiency of a PCASL labeling train depends on the field offset and the blood velocity
at the labeling plane {cite:p}`dai2008,zhao2017`, and unless it was measured with a
phase-contrast reference {cite:p}`aslan2010` the 0.85 is a convention. The habit to take
from the simulator is the record: put `LabelingEfficiency`, `M0Type`, the M0 scan's
`RepetitionTimePreparation` and `EchoTime`, and the assumed $\lambda$ and $T_{1b}$ where
the next person can find them.

## What this implies for acquisition

- **Acquire a separate M0 scan** with the ASL readout, no background suppression, and a
  repetition time of at least 5 s, or 8 s if CSF calibration is planned; at TR 2 s the
  saturation correction is 29 percent in gray matter and depends on an assumed $T_1$.
- **Keep the echo time short.** Every millisecond of TE grows the blood-tissue $T_2$
  mismatch, whose correction depends on a blood $T_2$ nobody measures; at 12 ms the
  factor is 8 percent and its uncertainty about 1 percent.
- **Record the constants.** `LabelingEfficiency`, `M0Type`, the M0 scan's TR and TE, and
  the $\lambda$ and $T_{1b}$ used at quantification. Absolute CBF from two sites can
  differ by 10 percent from these choices alone.
- **Measure what can be measured.** A phase-contrast scan gives $\alpha$; a hematocrit
  gives $T_{1b}$, whose 0.2 s range across subjects is worth 20 percent of CBF; a tissue
  $T_1$ map removes the strongest dependence of a multi-delay fit.
- **Use the same M0 route across a study.** Any route is fine once corrected; switching
  routes between subjects or sessions adds a scale that looks like a perfusion change.

## Further reading

The white paper's calibration section and its recommended constants
{cite:p}`alsop2015`; the pitfalls of calibration in post-processing, reviewed by
{cite:t}`pinto2020`; the CSF-based route {cite:p}`chalela2000`; the partition coefficient
of water {cite:p}`herscovitch1985`; the $T_1$ of blood and its hematocrit dependence at
3 T {cite:p}`lu2004`, and a model that adds oxygen saturation and field strength
{cite:p}`hales2016`; tissue relaxation times at 3 T {cite:p}`wansapura1999,stanisz2005`;
the phase-contrast measurement of PCASL efficiency
{cite:p}`aslan2010`; the ASL-BIDS fields that carry the calibration metadata
{cite:p}`clement2022`; and how two pipelines implement the corrections
{cite:p}`adebimpe2022,mutsaerts2020`.
