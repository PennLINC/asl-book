---
title: "12. Partial volume effects"
kernelspec:
  name: python3
  display_name: Python 3
---

:::{admonition} Simulated datasets in this chapter
:class: note
- **Built in this page:** the tissue fractions of every acquisition voxel from the packaged phantom slab, a noise-free reference series on it, an "atrophied" copy of the slab, and a toy perfusion map with a focal lesion ([Appendix B](../appendices/b-data-manifest.md#app-b-package-data)).
- **`ref-pcasl`**: the reference series, 30 pairs at noise σ ≈ 40 ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-ref-pcasl)).
- **`ref-clean`**: the same acquisition with the noise switched off ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-ref-clean)).
- **`voxel-sweep`**: the reference protocol at 2.5, 3.5 and 5 mm in-plane voxels ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-voxel-sweep)).

Pipeline-tier datasets are simulated offline by aslscan ([Chapter 0.2](../00-frontmatter/the-simulated-datasets.md)).
:::

## Learning goals

After this chapter you can:

- explain why the perfusion measured in a voxel is the fraction-weighted mean of the tissues it contains, and why that matters more for ASL than for most other MRI contrasts
- read the classic scatter of measured CBF against gray-matter fraction and say what its slope, intercept and spread mean
- say how partial volume biases a group's gray-matter CBF, the GM/WM ratio, and a comparison between groups with different cortical thickness
- apply the linear-regression partial-volume correction of {cite:t}`asllani2008`, and state what it costs in spatial resolution
- choose a voxel size knowing what it does to the fraction of the cortex that is "pure" gray matter

```{code-cell} python
:tags: [hide-cell]
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

from aslbook import data, kinetic, phantom, plotting, presets, protocols, quant, synth
from aslbook.plotting import INK, PALETTE, TISSUE_COLORS, set_style, show_slice

set_style()
K = phantom.DISPLAY_SLICE
GM, WM = presets.TISSUES["GM"], presets.TISSUES["WM"]


def kinetic_factor(pld, tau=presets.REFERENCE.labeling_duration):
    """Ratio of the white-paper model's ΔM to the general kinetic model's for pure gray matter at
    delay ``pld``: the white-paper formula times this factor returns the true CBF of a pure GM
    voxel (Chapter 14 measures the bias this removes)."""
    t1b, a, lam = presets.T1_BLOOD, presets.ALPHA["PCASL"], presets.LAMBDA
    wp = 2 * (GM.m0 / lam) * (GM.perfusion / 6000) * a * t1b * np.exp(-np.asarray(pld) / t1b) * (1 - np.exp(-tau / t1b))
    return wp / kinetic.delta_m(np.asarray(pld) + tau, GM.perfusion, GM.att, GM.t1, GM.m0, tau=tau)


def cbf_map(deltam, m0scan, p, m0_tr=8.0):
    """CBF from a mean difference image and an M0 scan: the M0 scan corrected for its saturation
    and T2 weighting, the white-paper formula with each slice's own delay, and the kinetic factor."""
    plds = quant.slice_plds(p["PostLabelingDelay"], protocols.slice_offsets(p))
    m0 = quant.m0_correction(m0scan, tr=m0_tr, te=p["EchoTime"])
    return quant.cbf_pcasl(deltam, m0, plds[None, None, :]) * kinetic_factor(plds)[None, None, :]


def mean_diff(run):
    return quant.subtract(run.mag(), run.context()).mean(-1)
```

## The physics: a voxel is a mixture

An ASL voxel of 3.5 × 3.5 × 5 mm holds 61 µl of brain. The cortex is 2 to 3 mm thick and
folded, so most cortical voxels also contain white matter, cerebrospinal fluid (CSF), or
both. Each tissue contributes its own signal in proportion to the volume it occupies. For
the control image that is a nuisance; for the difference image it is the whole
measurement: the labeled water that arrives in a voxel is the sum of what arrives in each
of its tissues,

$$
\Delta M_\text{voxel} = f_\text{GM}\,\Delta M_\text{GM} + f_\text{WM}\,\Delta M_\text{WM} + f_\text{CSF}\,\Delta M_\text{CSF},
$$

with $f$ the volume fractions and $\Delta M_\text{CSF} = 0$: CSF is not perfused. Because the
quantification of [Chapter 14](../04-quantification/14-cbf-quantification.md) is linear in
$\Delta M$, the CBF a pipeline assigns to the voxel is the same weighted mean of the tissue
values, about $f_\text{GM} \cdot 60 + f_\text{WM} \cdot 20$ ml/100 g/min in this phantom. A
voxel that is half gray matter and half CSF reports 30, not 60, with no noise and no
artifact involved. The effect is larger in ASL than in structural imaging for two reasons:
the voxels are big (the signal is small, so the voxels must be), and the contrast between
the tissues is large (a factor of three between gray and white matter, and zero in CSF).

**The aslscan setting that produces it** is not a setting but the acquisition grid itself.
The phantom is defined at 1 mm, and the simulator averages every quantity over the 1 mm
cells each acquisition voxel overlaps (`aslbook.grid.BoxResampler`, the rule of
[Chapter 0.2](../00-frontmatter/the-simulated-datasets.md)). The fraction of each tissue in
every voxel is therefore known exactly, and the pipeline writes it out
(`derivatives/aslbook/*_probseg.nii.gz`, loaded by `run.fractions()`). The ground truth
`perfusion` map is the same fraction-weighted mean. There is no artifact-free reference for
this chapter, only the 1 mm phantom the voxels were cut from.

:::{admonition} A bias that is not partial volume
:class: note
The white-paper formula assumes the label decays with the T1 of blood until the readout.
In this phantom, as in the general kinetic model of [Chapter 5](../02-labeling/05-kinetic-model.md),
the label decays with the tissue's T1 (1.33 s in GM, against 1.65 s in blood) once it has
arrived, so the formula returns about 44 ml/100 g/min for a pure gray-matter voxel whose
true value is 60. [Chapter 14](../04-quantification/14-cbf-quantification.md) measures that
bias. To keep it out of a chapter about mixing, every CBF map here is multiplied by the
per-slice factor (1.39 at the display slice, 1.31 to 1.48 across the slab) that makes the
formula exact for pure gray matter; the setup cell defines it. White matter, whose label arrives later (1.2 s) and
decays faster (T1 0.83 s), is then still underestimated, at about 14 instead of 20, and
that bias is in the kinetics, not in the mixing. The figures say where it shows.
:::

## See it: the fractions and the truth

```{code-cell} python
:tags: [hide-input]
ph = phantom.slab()
fine = phantom.fine_slice()
truth_toy = phantom.maps()["perfusion"]

fig, axes = plt.subplots(1, 5, figsize=(14, 3.2))
fine_cbf = phantom.class_maps_from_labels(fine["dseg"], "perfusion")
plotting.show_image(axes[0], np.rot90(fine_cbf), "the 1 mm phantom (CBF)", kind="cbf")
for ax, name in zip(axes[1:4], ("gm", "wm", "csf")):
    show_slice(ax, ph[name], K, f"{name.upper()} fraction, 3.5 mm voxels", kind="fraction")
show_slice(axes[4], truth_toy, K, "truth CBF: the weighted mean", kind="cbf")
fig.tight_layout()

gm = ph["gm"]
in_brain = ph["mask"]
print(f"voxels in the slab: {in_brain.sum()}; with any gray matter: {(gm > 0).sum()}; "
      f"at least half GM: {(gm >= 0.5).sum()}; at least 90 % GM: {(gm >= 0.9).sum()}; at least 99 %: {(gm >= 0.99).sum()}")
print(f"mean GM fraction of the voxels that are at least half GM: {gm[gm >= 0.5].mean():.2f}")
print(f"kinetic factor applied to the white-paper CBF: {kinetic_factor(1.8 + 0.04 * K):.2f} at the display slice, "
      f"{kinetic_factor(1.8):.2f} to {kinetic_factor(1.8 + 0.76):.2f} across the slab")
```

The 1 mm phantom (left) has three flat values: 60 in the cortex, 20 in white matter, 0 in
the ventricles and sulci. On the acquisition grid, the fraction maps show how those regions
are shared out: the cortical ribbon is one to two voxels wide and few of its voxels are
entirely gray. The truth CBF map on the right is what a perfect measurement would return,
and it already looks like a blurred version of the 1 mm map. Of the 26,046 voxels in the
slab, 25,060 contain some gray matter and 16,235 are at least half gray, but only 9,038
are at least 90 % gray and 5,542 at least 99 %; the voxels that are at least half gray have
a mean gray fraction of 0.87.

## See it: measured CBF against gray-matter fraction

The classic figure of partial volume in ASL plots each voxel's measured CBF against its
gray-matter fraction. The toy series makes it with no noise and no readout: the signal
stage of the simulator on the packaged slab, subtracted and quantified.

```{code-cell} python
:tags: [hide-input]
p_toy = protocols.pcasl(n_pairs=4)
s = synth.series(p_toy, noise_sd=0)
cbf_toy = cbf_map(quant.subtract(s.mag, s.ctx).mean(-1), s.m0scan, p_toy)

ref = data.load_dataset("ref-pcasl").run()
clean = data.load_dataset("ref-clean").run()
p_ref = ref.sidecar()
fr = ref.fractions()
mask = ref.mask()
truth = ref.truth("perfusion")
cbf_clean = cbf_map(mean_diff(clean), clean.m0scan(), p_ref)
cbf_ref = cbf_map(mean_diff(ref), ref.m0scan(), p_ref)

pure = {"GM": cbf_toy[fr["gm"] >= 0.999], "WM": cbf_toy[fr["wm"] >= 0.999]}
WM_APPARENT = pure["WM"].mean()     # what this quantification returns for pure white matter
rng = np.random.default_rng(0)
sel = rng.choice(np.flatnonzero(mask), 8000, replace=False)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
for ax, (title, cbf) in zip(axes, [("toy: no noise, no readout", cbf_toy), ("ref-clean: the readout, no noise", cbf_clean),
                                    ("ref-pcasl: 30 pairs, σ ≈ 40", cbf_ref)]):
    sc = ax.scatter(fr["gm"].ravel()[sel], cbf.ravel()[sel], c=fr["csf"].ravel()[sel], cmap="viridis", vmin=0, vmax=1, s=3, alpha=0.6, rasterized=True)
    ax.plot([0, 1], [WM_APPARENT, 60], color=INK["primary"], ls="--", lw=1)
    ax.set(xlabel="gray-matter fraction of the voxel", title=title, xlim=(0, 1))
axes[0].set(ylabel="measured CBF (ml/100 g/min)", ylim=(-40, 120))
fig.colorbar(sc, ax=axes, shrink=0.8, label="CSF fraction")

print(f"toy, voxels of one tissue only: GM {pure['GM'].mean():.1f} (n={pure['GM'].size}), WM {WM_APPARENT:.1f} (n={pure['WM'].size}) ml/100 g/min")
no_csf = mask & (fr["csf"] < 0.01) & (fr["gm"] + fr["wm"] > 0.99)
slope, icpt = np.polyfit(fr["gm"][no_csf], cbf_toy[no_csf], 1)
print(f"toy, voxels of GM and WM only: CBF = {icpt:.1f} + {slope:.1f} x GM fraction (r = {np.corrcoef(fr['gm'][no_csf], cbf_toy[no_csf])[0, 1]:.3f})")
edge = mask & (fr["gm"] + fr["wm"] + fr["csf"] < 0.9)
print(f"voxels partly outside the head ({edge.sum()}): measured CBF {cbf_toy[edge].mean():.1f} against a weighted truth of {truth[edge].mean():.1f}")
for name, cbf in (("ref-clean", cbf_clean), ("ref-pcasl", cbf_ref)):
    sc_ = quant.score(cbf, truth, mask)
    print(f"{name}: RMSE against the truth map {sc_['rmse']:.1f}, bias {sc_['bias']:+.1f} ml/100 g/min over the slab")
```

In the toy panel the voxels made of gray and white matter only lie on a straight line,
from white matter's apparent 9.9 at a gray fraction of 0 to 60 at a gray fraction of 1:
the fit gives CBF = 10.4 + 49.9 × GM fraction with a correlation of 0.999 (the small
residual is the slice dependence of the white-matter term, whose kinetic bias grows with
the delay). Voxels with CSF (yellow) fall below the line by their CSF fraction, because
CSF contributes volume and M0 but no labeled signal. The pure-tissue voxels return 60.0 in
gray matter and 9.9 in white matter: the gray value is exact by construction of the
kinetic factor, the white value is the kinetic bias of the note above. The horizontal band
at 60 between gray fractions of 0.5 and 1 is the edge of the head: voxels that are partly
outside the brain contain gray matter and nothing else, and because the M0 image shrinks
by the same fraction as the difference, the ratio is unaffected (the 1,461 such voxels
read 50.8 on average where the fraction-weighted truth is 36.4). Partial volume with air
cancels in the calibration; partial volume with CSF does not. The middle panel is the same
acquisition through the simulator's readout: the EPI reconstruction blurs and rings
slightly ([Chapter 2](../01-mri-physics/02-epi-and-reconstruction.md)), so neighboring
voxels leak into each other and the line thickens, but the pattern is the same. The right
panel is the reference series with its noise: the line becomes a cloud of width
± 20 ml/100 g/min, and the mixing is no longer visible voxel by voxel, although it is still
there. Over the whole slab the noise-free run has an RMSE of 7.0 against the truth (the
readout's blur plus the white-matter bias) and the noisy one 25.0, so at this noise level
the voxel-wise error is set by noise ([Chapter 8](./08-noise.md)); partial volume is a
bias, and biases show in averages.

## See it: how partial volume biases what a study reports

A study does not report voxels; it reports the mean CBF in "gray matter", a set of voxels
chosen by thresholding a gray-matter probability map. The threshold decides how much white
matter and CSF the average contains.

```{code-cell} python
:tags: [hide-input]
thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
rows = []
for t in thresholds:
    roi = fr["gm"] >= t
    rows.append((t, roi.sum(), truth[roi].mean(), cbf_ref[roi].mean(), cbf_clean[roi].mean()))
    print(f"GM fraction >= {t:.2f}: {roi.sum():5d} voxels, truth {truth[roi].mean():.1f}, ref-clean {cbf_clean[roi].mean():.1f}, ref-pcasl {cbf_ref[roi].mean():.1f} ml/100 g/min")
rows = np.array(rows)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.4))
ax1.plot(rows[:, 0], rows[:, 2], "-o", color=INK["primary"], label="truth (weighted mean)")
ax1.plot(rows[:, 0], rows[:, 4], "-s", color=PALETTE[2], label="ref-clean")
ax1.plot(rows[:, 0], rows[:, 3], "-^", color=TISSUE_COLORS["GM"], label="ref-pcasl")
ax1.axhline(60, color=INK["secondary"], ls=":", lw=1)
ax1.set(xlabel="GM fraction threshold of the mask", ylabel="mean CBF in the mask (ml/100 g/min)", title="what \"gray-matter CBF\" means", ylim=(40, 62))
ax1.legend(loc="lower right")
ax2.bar(rows[:, 0], rows[:, 1], width=0.035, color=TISSUE_COLORS["GM"])
ax2.set(xlabel="GM fraction threshold of the mask", ylabel="voxels in the mask", title="how many voxels qualify")
fig.tight_layout()
for t in (0.5, 0.9):
    g_, w_ = fr["gm"] >= t, fr["wm"] >= t
    print(f"GM/WM ratio at a threshold of {t}: truth {truth[g_].mean() / truth[w_].mean():.1f} (pure tissues 3.0), "
          f"ref-pcasl {cbf_ref[g_].mean() / cbf_ref[w_].mean():.1f}, ref-clean {cbf_clean[g_].mean() / cbf_clean[w_].mean():.1f}")
```

The measured gray-matter mean follows the truth's weighted mean, and both rise with the
threshold: 53.6 at a threshold of 0.5, 56.5 at 0.7, 59.1 at 0.9, 60.0 at 0.99 for the truth,
with the noisy series about a unit above each value (54.6, 57.3, 59.9, 61.0). None of these
is wrong; each is the mean perfusion of the tissue actually inside the mask. But a
"gray-matter CBF" of 54 and one of 60 differ by 11 %, which is the size of many reported
group effects, and the difference comes entirely from the threshold. The right panel shows
the price of a strict mask: at 0.99 only 5,542 voxels remain, and in a real segmentation,
which is itself uncertain at tissue boundaries, fewer still. The GM/WM ratio inherits the
same dependence: the pure tissues have a ratio of 3.0, and the truth in masks at a
threshold of 0.5 has 2.1, because the "white matter" mask contains cortex and the
"gray matter" mask white matter and CSF. The measured ratio is larger still, 6.1 at a
threshold of 0.9, but that is the kinetic bias in white matter, not partial volume.

### Two groups with the same perfusion

The consequence for group comparisons is that anything that changes the fractions changes
the measured CBF, with no change in perfusion. Cortical atrophy, with age or disease,
thins the cortex and widens the sulci: in the same voxels the gray fraction falls and the
CSF fraction rises. The toy below builds an "atrophied" slab by moving a fifth of every
voxel's gray matter to CSF, leaving the true perfusion of gray matter at 60, and measures
both slabs with the same noise-free protocol and the same mask.

```{code-cell} python
:tags: [hide-input]
shrink = 0.2
ph_atrophy = dict(ph)
ph_atrophy["gm"] = ph["gm"] * (1 - shrink)
ph_atrophy["csf"] = ph["csf"] + ph["gm"] * shrink
s_a = synth.series(p_toy, ph=ph_atrophy, noise_sd=0)
cbf_atrophy = cbf_map(quant.subtract(s_a.mag, s_a.ctx).mean(-1), s_a.m0scan, p_toy)

roi = ph["gm"] >= 0.7       # the same mask for both "groups", as a study would use a template mask
fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))
show_slice(axes[0], ph["gm"], K, "GM fraction: healthy slab", kind="fraction")
show_slice(axes[1], ph_atrophy["gm"], K, f"GM fraction: {100 * shrink:.0f} % of GM turned to CSF", kind="fraction")
show_slice(axes[2], cbf_atrophy, K, "measured CBF, atrophied slab", kind="cbf")
fig.tight_layout()
m_h, m_a = cbf_toy[roi].mean(), cbf_atrophy[roi].mean()
print(f"mean measured CBF in the GM mask (fraction >= 0.7, {roi.sum()} voxels): healthy {m_h:.1f}, atrophied {m_a:.1f} ml/100 g/min "
      f"({100 * (m_a / m_h - 1):+.0f} %), with the true GM perfusion 60 in both")
print(f"in voxels that were pure GM: healthy {cbf_toy[ph['gm'] >= 0.999].mean():.1f}, atrophied {cbf_atrophy[ph['gm'] >= 0.999].mean():.1f}")
```

In the same mask the atrophied slab reads 11 ml/100 g/min lower (46.0 against 56.8, a
19 % drop) although its gray matter is perfused at exactly the same rate; even the voxels
that used to be pure gray matter now read 48.5. A comparison between an older and a younger
group, or between patients and controls, measures this mixture of perfusion and anatomy
unless the fractions are accounted for, which is the job of the correction below
{cite:p}`asllani2008`. In practice the fractions come from a segmented T1-weighted image
resampled to the ASL grid, so the correction is only as good as the segmentation and the
registration between the two images.

## See it: voxel size

Smaller voxels contain fewer tissue boundaries. The `voxel-sweep` dataset acquires the
same slab at 2.5, 3.5 and 5 mm in-plane (slices stay 5 mm) with 15 pairs each; each run
carries its own fraction maps, computed on its own grid.

```{code-cell} python
:tags: [hide-input]
vs = data.load_dataset("voxel-sweep")
fig, axes = plt.subplots(2, 3, figsize=(11, 7))
sizes = {}
for col, name in enumerate(["vox25", "vox35", "vox50"]):
    run = vs.run(name)
    f = run.fractions()
    mm = run.sidecar()["AcquisitionVoxelSize"][0]
    cbf = cbf_map(mean_diff(run), run.m0scan(), run.sidecar())
    show_slice(axes[0, col], f["gm"], K, f"GM fraction, {mm:g} mm in-plane ({f['gm'].shape[0]}×{f['gm'].shape[1]})", kind="fraction")
    m = run.mask()
    sel = np.random.default_rng(1).choice(np.flatnonzero(m), 6000, replace=False)
    axes[1, col].scatter(f["gm"].ravel()[sel], cbf.ravel()[sel], s=3, alpha=0.4, color=TISSUE_COLORS["GM"], rasterized=True)
    axes[1, col].plot([0, 1], [WM_APPARENT, 60], color=INK["primary"], ls="--", lw=1)
    axes[1, col].set(xlim=(0, 1), ylim=(-60, 140), xlabel="GM fraction", title=f"{mm:g} mm: 15 pairs")
    n9, n99 = (f["gm"] >= 0.9).sum(), (f["gm"] >= 0.99).sum()
    vol_frac = f["gm"][f["gm"] >= 0.9].sum() / f["gm"].sum()
    sizes[mm] = (n9, n99, vol_frac, cbf[f["gm"] >= 0.7].mean(), cbf[f["gm"] >= 0.9].mean(), run.truth("perfusion")[f["gm"] >= 0.7].mean())
    print(f"{mm:g} mm: {n9:6d} voxels >= 90 % GM, {n99:6d} >= 99 %; {100 * vol_frac:.0f} % of the gray-matter volume lies in voxels >= 90 % GM; "
          f"mean CBF at threshold 0.7: {sizes[mm][3]:.1f} (truth {sizes[mm][5]:.1f}), at 0.9: {sizes[mm][4]:.1f}; "
          f"SD of CBF in voxels >= 90 % GM {cbf[f['gm'] >= 0.9].std():.0f}")
axes[1, 0].set(ylabel="measured CBF (ml/100 g/min)")
fig.tight_layout()
```

At 2.5 mm the cortical ribbon is two to three voxels wide and 63 % of the gray-matter
volume sits in voxels that are at least 90 % gray; at 3.5 mm it is 56 % and at 5 mm 48 %,
where the ribbon in the fraction map is one voxel wide with no interior. The scatter
panels show the two costs pulling in opposite directions: the mixing is weaker in small
voxels (more points at high fractions), but the noise scales with the voxel's volume as
it does on a scanner, so with 15 pairs the CBF of a pure gray-matter voxel has an SD of
59 ml/100 g/min at 2.5 mm, 30 at 3.5 mm and 15 at 5 mm. Matching the 3.5 mm noise at
2.5 mm takes four times the pairs, which is to say four times the scan
([Chapter 7](../02-labeling/07-acquisition-parameters.md)). The mean CBF at a threshold of
0.7 barely changes with voxel size (56 to 58 ml/100 g/min), because the threshold selects
the purest voxels at each size, but the number of voxels that qualify, and so the cortex a
study can measure, changes by a factor of five (19,897 against 3,840 at 90 %).

## The correction: linear regression over a neighborhood

{cite:t}`asllani2008` noticed that the mixing equation is a linear model with two unknowns
per voxel, the pure gray-matter and pure white-matter perfusions $f_\text{GM}$ and
$f_\text{WM}$, and known coefficients, the fractions:

$$
\text{CBF}_i = P_{\text{GM},i}\, f_\text{GM} + P_{\text{WM},i}\, f_\text{WM}, \qquad i \in \text{neighborhood}.
$$

One voxel gives one equation for two unknowns. But if the pure-tissue perfusions are
assumed constant over a small neighborhood (a $k \times k$ in-plane kernel, 5 × 5 in the
original paper), the $k^2$ voxels give $k^2$ equations, solved by least squares for the two
values at the center. The assumption is the cost: whatever structure gray-matter perfusion
has within the kernel is averaged out, so the corrected map has the resolution of the
kernel, not of the voxel. The kernel must also contain enough of both tissues for the two
columns to be distinguishable; where a neighborhood is all one tissue the two-unknown
system is singular, and `quant.pv_correct` falls back to the one-tissue solution there (the
present tissue's perfusion is still determined) and returns zero for the absent tissue.

```{code-cell} python
:tags: [hide-input]
f_gm5, f_wm5 = quant.pv_correct(cbf_ref, fr["gm"], fr["wm"], kernel=5, mask=mask)
f_gm3, f_wm3 = quant.pv_correct(cbf_ref, fr["gm"], fr["wm"], kernel=3, mask=mask)

def determined(f, frac, lo=0.3):
    """The corrected map where the regression had both tissues to work with (zero marks a
    neighborhood of one tissue, where the two unknowns cannot be separated)."""
    return np.where((frac >= lo) & (f != 0), f, np.nan)

fig, axes = plt.subplots(1, 4, figsize=(13, 3.3))
show_slice(axes[0], np.where(mask, cbf_ref, np.nan), K, "uncorrected CBF (ref-pcasl)", kind="cbf")
show_slice(axes[1], determined(f_gm5, fr["gm"]), K, "GM CBF, 5 × 5 kernel", kind="cbf")
show_slice(axes[2], determined(f_wm5, fr["wm"]), K, "WM CBF, 5 × 5 kernel", kind="cbf")
show_slice(axes[3], determined(f_gm3, fr["gm"]), K, "GM CBF, 3 × 3 kernel", kind="cbf")
fig.tight_layout()

gm_roi, wm_roi = fr["gm"] >= 0.5, fr["wm"] >= 0.5
for k, fg, fw in ((5, f_gm5, f_wm5), (3, f_gm3, f_wm3)):
    g_ok, w_ok = gm_roi & (fg != 0), wm_roi & (fw != 0)
    print(f"kernel {k}: GM CBF in voxels >= 50 % GM: mean {fg[g_ok].mean():.1f}, SD {fg[g_ok].std():.1f} (truth 60; undetermined in {gm_roi.sum() - g_ok.sum()} of {gm_roi.sum()}); "
          f"WM CBF in voxels >= 50 % WM: mean {fw[w_ok].mean():.1f}, SD {fw[w_ok].std():.1f} (truth 20, apparent {WM_APPARENT:.1f}; undetermined in {wm_roi.sum() - w_ok.sum()})")
print(f"uncorrected CBF in the same GM voxels: mean {cbf_ref[gm_roi].mean():.1f}, SD {cbf_ref[gm_roi].std():.1f}; truth there {truth[gm_roi].mean():.1f}")
c_gm5, c_wm5 = quant.pv_correct(cbf_clean, fr["gm"], fr["wm"], kernel=5, mask=mask)
print(f"the same on ref-clean (no noise), kernel 5: GM {c_gm5[gm_roi & (c_gm5 != 0)].mean():.1f}, WM {c_wm5[wm_roi & (c_wm5 != 0)].mean():.1f}")
```

The uncorrected map has bright cortex and dark white matter with the noise of 30 pairs on
top. The corrected gray-matter map is nearly flat, which is what the phantom's cortex is:
in the voxels that are at least half gray it has a mean of 61.9 with a 5 × 5 kernel,
against an uncorrected mean of 54.6 (the truth in the same voxels being 53.6, the
weighted mean). The two units above 60 are not noise: the noise-free `ref-clean` run gives
61.7, because the regression is handed sharp fractions and a map the EPI readout has
blurred, so a little of each voxel's cortex is credited to its neighbors' fractions. The
white-matter map is flat and dark at 9.5, the apparent pure-white value of this
quantification, not 20: the correction returns the pure-tissue values the quantification
assigns, and it cannot fix a bias that sits in the kinetic constants rather than in the
mixing. The 3 × 3 kernel gives the same mean with more noise (SD 11.8 against 7.4 in gray
matter): nine equations constrain two unknowns less well than twenty-five, and where the
nine voxels are nearly all one tissue the two columns are almost collinear and the
solution swings. Where they are entirely one tissue the function solves for that tissue
alone, so no voxel of either mask is left undetermined.

### The cost: a lesion smoothed away

The kernel's assumption of constant perfusion is false wherever perfusion actually varies.
The toy below gives the phantom's cortex a focal deficit, a 10 mm disc at 30 ml/100 g/min,
builds the noise-free mixed map from the fractions, and corrects it with both kernels.

```{code-cell} python
:tags: [hide-input]
RADIUS_MM = 10.0
vx, vy, vz = ph["voxel_mm"]
# put the deficit where the display slice has the most cortex within the radius
disc = np.hypot(*np.meshgrid(np.arange(-3, 4) * vx, np.arange(-3, 4) * vy, indexing="ij")) <= RADIUS_MM
cortex_score = ndimage.convolve(ph["gm"][:, :, K], disc.astype(float), mode="constant")
cx, cy = np.unravel_index(np.argmax(cortex_score), cortex_score.shape)
xx, yy, zz = np.indices(ph["gm"].shape)
lesion = np.hypot((xx - cx) * vx, (yy - cy) * vy, (zz - K) * vz) <= RADIUS_MM
gm_true = np.where(lesion, 30.0, 60.0)              # the true gray-matter perfusion map
cbf_mixed = ph["gm"] * gm_true + ph["wm"] * 20.0    # the noise-free measurement
l_gm5, _ = quant.pv_correct(cbf_mixed, ph["gm"], ph["wm"], kernel=5, mask=ph["mask"])
l_gm3, _ = quant.pv_correct(cbf_mixed, ph["gm"], ph["wm"], kernel=3, mask=ph["mask"])

fig, axes = plt.subplots(1, 4, figsize=(13, 3.3))
show_slice(axes[0], np.where(ph["gm"] >= 0.3, gm_true, np.nan), K, "true GM CBF with a 10 mm deficit", kind="cbf")
show_slice(axes[1], np.where(ph["mask"], cbf_mixed, np.nan), K, "measured (mixed), no noise", kind="cbf")
show_slice(axes[2], determined(l_gm3, ph["gm"]), K, "GM CBF, 3 × 3 kernel", kind="cbf")
show_slice(axes[3], determined(l_gm5, ph["gm"]), K, "GM CBF, 5 × 5 kernel", kind="cbf")
for ax in axes:   # the circle in display coordinates (rot90: column = x, row = last y - y)
    ax.add_patch(plt.Circle((cx, ph["gm"].shape[1] - 1 - cy), RADIUS_MM / vx, fill=False, color="white", lw=0.8, ls="--"))
fig.tight_layout()

core = lesion & (ph["gm"] >= 0.5)
outside = (~lesion) & (ph["gm"] >= 0.5)
print(f"GM CBF inside the deficit (voxels >= 50 % GM, n={core.sum()}): truth 30.0, "
      f"3 × 3 kernel {l_gm3[core & (l_gm3 != 0)].mean():.1f}, 5 × 5 kernel {l_gm5[core & (l_gm5 != 0)].mean():.1f} ml/100 g/min")
print(f"GM CBF in the unaffected cortex: 3 × 3 {l_gm3[outside & (l_gm3 != 0)].mean():.1f}, 5 × 5 {l_gm5[outside & (l_gm5 != 0)].mean():.1f} "
      f"(undetermined in {(outside & (l_gm3 == 0)).sum()} and {(outside & (l_gm5 == 0)).sum()} voxels)")
```

The 3 × 3 kernel recovers most of the deficit (37 inside the sphere against a truth of 30,
from a mixed map that reads about 25 there) and the 5 × 5 kernel less (41), because a
25-voxel kernel reaches past the 10 mm sphere from every voxel inside it. Far from the
lesion both return 60 (59.9 and 59.8), with no noise to disturb them. This is the trade: a large kernel gives stable estimates and
loses the spatial detail a lesion study wants; a small one keeps the detail and amplifies
the noise. The kernel should be chosen for the question, and a corrected map should never
be read at a finer scale than its kernel.

:::{dropdown} The Bayesian alternative
{cite:t}`chappell2011` put the same two-tissue model inside the kinetic-model fit of BASIL,
with a spatial prior on each tissue's perfusion instead of a fixed kernel: neighboring
voxels are encouraged, not forced, to share a value, and the strength of the encouragement
is estimated from the data. The result adapts the effective kernel to the local
signal-to-noise ratio and can give the two tissues different kinetic parameters (a longer
transit time for white matter, for instance), which the linear regression cannot. It is
the correction `oxford_asl --pvcorr` applies. Its assumptions are the same as the regression's
in one respect: the fractions are taken as known, so segmentation and registration errors
pass straight through.
:::

## Measure it: the corrected maps against the truth

The final check is the error of the whole map. The reconstruction
$P_\text{GM} f_\text{GM} + P_\text{WM} f_\text{WM}$ puts the corrected values back into the
voxels, so it can be compared with the fraction-weighted truth on equal terms.

```{code-cell} python
:tags: [hide-input]
recon5 = fr["gm"] * f_gm5 + fr["wm"] * f_wm5
recon3 = fr["gm"] * f_gm3 + fr["wm"] * f_wm3
for name, est in (("uncorrected", cbf_ref), ("reconstructed, 5 × 5", recon5), ("reconstructed, 3 × 3", recon3)):
    sc_ = quant.score(est, truth, mask)
    sg = quant.score(est, truth, gm_roi)
    print(f"{name:22s}: slab RMSE {sc_['rmse']:5.1f}, bias {sc_['bias']:+5.1f}; GM (>= 50 %) RMSE {sg['rmse']:5.1f}, bias {sg['bias']:+5.1f} ml/100 g/min")
fig, axes = plotting.fit_vs_truth(recon5, truth, mask, "CBF (5 × 5 reconstruction)", k=K, unit="(ml/100 g/min)")
```

Over the slab the uncorrected map has an RMSE of 25.0 against the truth and the
5 × 5 reconstruction 8.3, the 3 × 3 one 11.1. The improvement is mostly denoising, not
partial-volume correction: the regression pools 25 voxels, so its reconstruction is a
smoothed version of the data, and the truth it is scored against is itself the
fraction-weighted mean. The bias stays slightly negative (−2.0 uncorrected, −2.4 with the
5 × 5 kernel over the slab) because the reconstruction puts the underestimated
white-matter value back into every mixed voxel, where the uncorrected map's noise had
hidden it. In the 4-panel
the difference map is smooth noise with a faint dark white matter, and the scatter hugs
the identity line in gray matter and falls below it at low CBF. What the correction
changes is not the map's error but its interpretation: the corrected gray-matter value no
longer depends on the threshold, the cortical thickness, or the voxel size.

## What this implies for acquisition

- **Voxel size decides how much of the cortex is pure.** At 2.5 mm in-plane about 63 % of
  the gray-matter volume lies in voxels that are at least 90 % gray, at 5 mm about half. The
  smaller voxel needs longer averaging for the same signal-to-noise ratio.
- **Slice thickness matters as much as in-plane resolution.** The 5 mm slices here mix
  tissue along the head-foot axis that no in-plane resolution can separate; 3D readouts
  with isotropic voxels reduce that ([Chapter 7](../02-labeling/07-acquisition-parameters.md)).
- **Acquire a structural image and plan the registration.** The correction needs tissue
  fractions on the ASL grid; a T1-weighted image, its segmentation, and a careful
  registration to the distortion-corrected ASL images are part of the protocol, not an
  afterthought ([Chapter 11](./11-susceptibility-distortion.md)).
- **Report the mask.** A gray-matter CBF is a number attached to a threshold and a
  segmentation; report both, and prefer corrected pure-tissue values when groups differ in
  anatomy.
- **Choose the kernel for the question.** Group means tolerate a 5 × 5 kernel; focal
  deficits need 3 × 3 or the adaptive Bayesian prior, and more averaging to pay for it.

## Further reading

The regression method is {cite:t}`asllani2008`; the Bayesian spatial version in BASIL is
{cite:t}`chappell2011`. The consensus recommendations of {cite:t}`alsop2015` discuss partial
volume among the reasons ASL gray-matter values vary between studies, and the ASLPrep
{cite:p}`adebimpe2022` and ExploreASL {cite:p}`mutsaerts2020` pipelines both implement the
regression correction as an optional last step, which is where
[Chapter 13](./13-assembled-pipeline.md) places it.
