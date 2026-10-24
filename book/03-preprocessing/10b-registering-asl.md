---
title: "10.2 Registering an ASL series: contrasts, references, and what pipelines do"
kernelspec:
  name: python3
  display_name: Python 3
---

:::{admonition} Simulated datasets in this chapter
:class: note
- **Built in this page:** the rigid registration of [Chapter 10](./10-motion.md) with four similarity measures, the registration of each volume type to each candidate reference, the zig-zag regression, and the label-to-control offset step ([Appendix B](../appendices/b-data-manifest.md#app-b-package-data)).
- **`ref-pcasl`**: the still reference series, its M0 scan, and the contrasts of its volume types ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-ref-pcasl)).
- **`bgsup`**: the still series with two suppression pulses (`on`), whose control and label images look nothing like the M0 scan ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-bgsup)).
- **`readout-3d`**: the still four-shot GRASE (`grase-4shot`) and stack of spirals (`spiral`), whose contrast differs from the 2D series' ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-readout-3d)).
- **`motion`**: the slow drift (`drift`) and the random jumps (`random`), on which the references and the zig-zag regression are scored against the true poses ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-motion)).
- **`vascular`**: the multi-delay GRASE series with an arterial compartment (`uncrushed`) and without (`none`), for the short delays at which the label still holds arterial signal ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-vascular)).

Pipeline-tier datasets are simulated offline by aslscan ([Chapter 0.2](../00-frontmatter/the-simulated-datasets.md)).
:::

## Learning goals

After this chapter you can:

- list the contrasts an ASL series contains (control, label, M0, suppressed, each readout's own) and say why registering one to another is biased even when the head is still
- choose a similarity measure for a same-contrast and for a cross-contrast registration, and say what least squares does across contrasts
- explain the spurious label-control offset, the blur bias of an averaged reference, and the zig-zag regression, with the case each one fails on
- compare the reference images a pipeline can register to, and describe what ASLPrep, ExploreASL, ASLtbx and BASIL actually do

```{code-cell} python
:tags: [hide-cell]
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

from aslbook import data, motion, phantom, plotting, presets, quant
from aslbook.motion import COSTS, apply_poses, move_volume, pair_offsets, register_series, register_volume, regress_alternation, sample_points
from aslbook.plotting import INK, PALETTE, set_style, show_slice

set_style()
K = phantom.DISPLAY_SLICE
```

## The problem: one series, several contrasts

A functional MRI series is one contrast repeated; registering its volumes to one another
is the problem rigid-body registration was made for {cite:p}`friston1995,jenkinson2002`.
An ASL series is not one contrast. The control and the label differ by the perfusion
signal, half a percent, which is small but is exactly the signal, and a registration that
aligns the two on their *common* content is right only if nothing pulls it toward the
part that differs. The M0 scan, acquired with a long repetition time, has a different
contrast from the control: its cerebrospinal fluid has recovered and is bright, where in
the control it is dark. Under background suppression the control and label images are
the inverted and nulled tissue that [Chapter 9](./09-background-suppression.md) described,
with gray matter darker than white and the fluid brightest, nothing like the M0 scan they
are usually registered to. And each readout has its own contrast: the long echo trains of
a 3D GRASE or a stack of spirals weight the image by T2, so that white matter comes out
brighter than gray, the opposite of the 2D EPI. The table and the figure measure these on
the still series of the book, mean over the interior slices of the voxels that are at
least 90 % one tissue.

```{code-cell} python
:tags: [hide-input]
ref = data.load_dataset("ref-pcasl").run()
bs = data.load_dataset("bgsup").run("on")
gr = data.load_dataset("readout-3d").run("grase-4shot")
sp = data.load_dataset("readout-3d").run("spiral")
runs = {"2D EPI": ref, "2D EPI, suppressed": bs, "3D GRASE": gr, "stack of spirals": sp}
imgs, masks, fr = {}, {}, {}
for name, r in runs.items():
    m = r.signed().astype(float) if name.endswith("suppressed") else r.mag().astype(float)
    ctx = r.context()
    imgs[name] = {"control": m[..., [i for i, c in enumerate(ctx) if c == "control"]].mean(-1),
                  "label": m[..., [i for i, c in enumerate(ctx) if c == "label"]].mean(-1), "M0 scan": r.m0scan().astype(float)}
    masks[name] = r.mask(); masks[name][:, :, [0, -1]] = False
    fr[name] = r.fractions()
tissue_mean = lambda img, name, t: img[(fr[name][t] > 0.9) & masks[name]].mean()
print(f"{'image':>32} {'GM':>6} {'WM':>6} {'CSF':>6}   GM/WM")
for name in runs:
    for kind in ("control", "label", "M0 scan"):
        g, w, c = (tissue_mean(imgs[name][kind], name, t) for t in ("gm", "wm", "csf"))
        print(f"{name + ', ' + kind:>32} {g:>6.0f} {w:>6.0f} {c:>6.0f}   {g / w:.2f}")
fig, axes = plt.subplots(1, 5, figsize=(14, 3.1))
for ax, (name, kind) in zip(axes, [("2D EPI", "control"), ("2D EPI", "M0 scan"), ("2D EPI, suppressed", "control"), ("3D GRASE", "control"), ("stack of spirals", "control")]):
    img = imgs[name][kind]
    show_slice(ax, img, K, f"{name}: {kind}", vmin=0, vmax=1.3 * tissue_mean(img, name, "wm"))
fig.tight_layout()
```

The 2D control image reads 6249 in gray matter, 5747 in white matter and 5103 in the
fluid, a gray-to-white ratio of 1.09; its label is 30 units lower in gray matter and the
same elsewhere, which is the perfusion signal and nothing else. The M0 scan has the same
tissues at 6453 and 5774 but the fluid at 6110, brighter than white matter: a different
image. Under suppression the signed control is 349, 734 and 879, gray matter the darkest
tissue at a ratio of 0.47, and the suppressed M0 scan is of course the unsuppressed one.
The 3D GRASE control is 5807, 6063 and 6481, white brighter than gray (0.96) and the fluid
brightest; the spiral's is the same order at 0.94; and their M0 scans share that order,
with the fluid at 7400. Every registration below is between two of these images, and the
question in each case is what the similarity measure makes of the differences.

## Similarity measures, and what least squares does across contrasts

A registration minimizes a cost over the six pose parameters, and the cost is a statement
about how the two images are related. The page's registration offers four, the ones the
common engines use:

- **Least squares** (`ssd`, the mean squared difference): the images are the same up to
  noise. SPM's realign {cite:p}`friston1995`, AFNI's 3dvolreg {cite:p}`cox1999` and
  MCFLIRT's `leastsq` use it. It is the most precise cost within one contrast and has no
  idea what to do with a change of contrast, where a brighter fluid is simply an error
  to be reduced by moving the image.
- **Normalized correlation** (`ncc`): the images are the same up to a scale and an offset,
  MCFLIRT's default {cite:p}`jenkinson2002`.
- **Correlation ratio** (`corratio`): for each intensity of one image, the other image
  takes one value, any value; the cost is the variance that this mapping leaves
  unexplained {cite:p}`roche1998`. FLIRT's default for registration between modalities
  {cite:p}`jenkinson2001`.
- **Normalized mutual information** (`nmi`): the joint histogram of the two images is
  sharp, whatever the mapping {cite:p}`viola1997,studholme1999`; MCFLIRT's `normmi`.

The cell slides the control mean along $x$ against itself, against the M0 scan, and, for
the suppressed series, against the M0 scan, and plots each cost; the registration table
after it lets the full six-parameter search loose on the same pairs.

```{code-cell} python
:tags: [hide-input]
smooth = lambda x: ndimage.gaussian_filter(x, (1.5, 1.5, 1.0))
inner = masks["2D EPI"]
pts = sample_points(inner & ~np.isin(np.arange(20)[None, None, :] * np.ones(inner.shape, int), [1, 18]), 6000, seed=0)
ctrl, m0 = imgs["2D EPI"]["control"], imgs["2D EPI"]["M0 scan"]
ctrl_bs = imgs["2D EPI, suppressed"]["control"]
shifts = np.linspace(-3, 3, 61)
curves = {}
for moving_name, mov, target_name, target in (("control", ctrl, "control (same contrast)", ctrl), ("control", ctrl, "M0 scan (another contrast)", m0),
                                              ("suppressed control", ctrl_bs, "M0 scan", m0)):
    tv = smooth(target)[tuple(pts.astype(int))]
    for cost, f in COSTS.items():
        vals = np.array([f(move_volume(smooth(mov), [s, 0, 0], [0, 0, 0])[tuple(pts.astype(int))], tv) for s in shifts])
        curves[(target_name, cost)] = (vals - vals.min()) / (vals.max() - vals.min())
        print(f"{moving_name:>19} -> {target_name:<27} {cost:>8}: lowest cost at a shift of {shifts[np.argmin(vals)]:+.1f} mm")
fig, axes = plt.subplots(1, 3, figsize=(13, 3.3))
for ax, target_name, title in zip(axes, ("control (same contrast)", "M0 scan (another contrast)", "M0 scan"),
                                  ("control mean against itself", "control mean against the M0 scan", "suppressed control mean against the M0 scan")):
    for j, cost in enumerate(COSTS):
        ax.plot(shifts, curves[(target_name, cost)], color=PALETTE[j], lw=1.4, label=cost)
    ax.set(xlabel="shift of the moving image along x (mm)", ylabel="cost (scaled 0–1)", title=title)
    ax.axvline(0, color=INK["secondary"], lw=0.8, ls=":")
axes[0].legend(fontsize=7)
fig.tight_layout()
```

```{code-cell} python
:tags: [hide-input]
print(f"{'readout':>20} {'cost':>9} {'label -> control':>18} {'control -> M0':>16}   (|translation| in mm, largest rotation in degrees)")
offsets = {}
for name, r in runs.items():
    mk = masks[name]
    p_here = sample_points(mk & ~np.isin(np.arange(20)[None, None, :] * np.ones(mk.shape, int), [1, 18]), 6000, seed=0)
    for cost in COSTS:
        e_lc = register_volume(smooth(imgs[name]["label"]), smooth(imgs[name]["control"]), p_here, cost=cost)
        e_cm = register_volume(smooth(imgs[name]["control"]), smooth(imgs[name]["M0 scan"]), p_here, cost=cost)
        offsets[(name, cost)] = (e_lc, e_cm)
        fmt = lambda e: f"{np.linalg.norm(e[:3]):6.3f} / {np.abs(e[3:]).max():5.3f}"
        print(f"{name:>20} {cost:>9} {fmt(e_lc):>18} {fmt(e_cm):>16}")
```

Against itself and against the M0 scan every cost is a bowl with its bottom at zero on a
0.1 mm grid, mutual information a sharper one: the M0's bright fluid is a small
perturbation that a mapping-based cost absorbs and that least squares can live with. The
suppressed control against the M0 scan is another matter. Along the one axis the plot
can show, normalized correlation falls all the way to the edge at −3 mm, the correlation
ratio is noisy with its minimum at +1.4 mm, and least squares has a minimum at zero in
this one dimension only: given six, its search in the table below walks off to 228 mm
and 92°, normalized correlation to 649 mm, the correlation ratio to 0.9 mm, and mutual
information to 28 mm and 1.4°. Between a suppressed image and an unsuppressed one the
tissues have changed order, and a cost that assumes the images are alike will move the
image until they are. The four unsuppressed cases behave: the label lands on the control
within 0.02 mm for every readout and every cost, and the control on the M0 scan within
0.05 to 0.19 mm, the largest being least squares on the 2D series, which trades 0.19 mm of
shift for the fluid it cannot otherwise match. Those small numbers are the subject of
the next section; the large ones say that the M0 scan is not a registration target for a
suppressed series, with any cost, and that least squares is only for images of one
contrast.

## The spurious label-control offset, and the offset step

A pipeline that registers each volume type to a single reference, the M0 scan say,
registers the controls across one contrast difference and the labels across a slightly
different one, and the two registrations do not land in the same place. The difference is
an offset between the corrected labels and the corrected controls that was not in the
data, and because the subtraction sees any offset as a rim, it is paid for at the edges of
the brain on every pair. The cell registers the still series' control mean and label mean
to the M0 scan separately with each cost, measures the offset, and measures what it does
at a one-voxel rim of the brain, against a subtraction in which both images were moved by
the same pose. Then it does what the offset step does: registers the corrected label mean
onto the corrected control mean, which is a same-contrast least-squares problem, and
composes that correction into the labels' transform.

```{code-cell} python
:tags: [hide-input]
lab = imgs["2D EPI"]["label"]
rim = inner & ~ndimage.binary_erosion(masks["2D EPI"] | ~ref.mask(), iterations=1)
gm_dm = (ctrl - lab)[(fr["2D EPI"]["gm"] > 0.9) & inner].mean()
print(f"gray matter ΔM of the still series: {gm_dm:.1f} image units; its noise-free ΔM at the rim is about {np.abs(ctrl - lab)[rim].mean():.0f}")
print(f"{'cost':>9} {'control -> M0':>14} {'label -> M0':>12} {'label - control':>16} {'rim it makes':>13} {'offset step: left':>18} {'rim, resampled twice':>21} {'rim, composed':>14}")
for cost in COSTS:
    e_c = register_volume(smooth(ctrl), smooth(m0), pts, cost=cost)
    e_l = register_volume(smooth(lab), smooth(m0), pts, cost=cost)
    c_cor, l_cor = move_volume(ctrl, e_c[:3], e_c[3:], undo=True), move_volume(lab, e_l[:3], e_l[3:], undo=True)
    d_same = c_cor - move_volume(lab, e_c[:3], e_c[3:], undo=True)                   # both images moved by the control's pose: no offset, the same interpolation
    d_typewise = c_cor - l_cor
    off = register_volume(smooth(l_cor), smooth(c_cor), pts, cost="ssd")             # the offset step: the corrected label mean onto the corrected control mean
    l_twice = move_volume(l_cor, off[:3], off[3:], undo=True)                        # applied as a second resampling
    l_once = move_volume(lab, e_l[:3] + off[:3], e_l[3:] + off[3:], undo=True)       # composed into the label's transform, one resampling
    left = register_volume(smooth(l_twice), smooth(c_cor), pts, cost="ssd")
    err = lambda d: np.sqrt(np.mean((d - d_same)[rim] ** 2))
    print(f"{cost:>9} {np.linalg.norm(e_c[:3]):>14.3f} {np.linalg.norm(e_l[:3]):>12.3f} {np.linalg.norm(e_l[:3] - e_c[:3]):>16.3f} {err(d_typewise):>13.1f} {np.linalg.norm(left[:3]):>18.3f} {err(c_cor - l_twice):>21.1f} {err(c_cor - l_once):>14.1f}   (mm, mm, mm, image units RMS at the rim, mm, image units, image units)")
```

The offsets are small, and they matter. With least squares and normalized correlation the
controls and labels land within 0.005 and 0.008 mm of each other, and the rim they leave
is under 2 units. With the correlation ratio the label lands 0.085 mm from the control
and with mutual information 0.054 mm, the costs built to tolerate a contrast difference
being the ones that let the perfusion signal, which is a contrast difference between
label and control, pull the two apart; the rims they leave are 12.6 and 7.7 units RMS on
a signal that is about 26 at the rim and 30 in gray matter. A twentieth of a voxel is
invisible on a control image and half of the perfusion signal at the edge of the brain.
The offset step finds and removes it: after it the label mean sits on the control mean to
0.000 mm. Applied as a second resampling of the already corrected labels it buys nothing
at the rim (12.8 and 9.9), because a second interpolation blurs the edges by as much as
the offset moved them; composed into the labels' transform and applied once, it brings
the correlation-ratio rim from 12.6 to 3.4 units. That is the design in ASLPrep's
`typewise_offset` method and the reason the composition matters.

## Blurred references bias the registration

The reference is often an average: the mean control, the mean of every volume, a
template rebuilt from aligned volumes. An average of imperfectly aligned volumes is
blurred, and a sharp volume registered to a blurred copy of itself does not land on it.
The cell registers the still control mean to smoothed copies of itself with mutual
information, so that the right answer is the identity, and then registers the smoothed
copy to itself.

```{code-cell} python
:tags: [hide-input]
fwhms = [0.0, 2.5, 5.0, 8.0]
bias = {}
for name, r in (("2D EPI", ref), ("3D GRASE", gr)):
    img = imgs[name]["control"]; mk = masks[name]
    p_here = sample_points(mk & ~np.isin(np.arange(20)[None, None, :] * np.ones(mk.shape, int), [1, 18]), 6000, seed=0)
    for fwhm in fwhms:
        sig = fwhm / 2.355
        blurred = ndimage.gaussian_filter(img, (sig / 3.5, sig / 3.5, sig / 5.0)) if fwhm else img
        e_mis = register_volume(img, blurred, p_here, cost="nmi")
        e_match = register_volume(blurred, blurred, p_here, cost="nmi")
        bias[(name, fwhm)] = (np.linalg.norm(e_mis[:3]), np.linalg.norm(e_match[:3]))
        print(f"{name:>9}, reference blurred to {fwhm:3.1f} mm FWHM: a sharp volume lands {bias[(name, fwhm)][0]:.2f} mm off; blurred alike, {bias[(name, fwhm)][1]:.2f} mm")
fig, ax = plt.subplots(figsize=(6, 3.2))
for j, name in enumerate(("2D EPI", "3D GRASE")):
    ax.plot(fwhms, [bias[(name, f)][0] for f in fwhms], "o-", color=PALETTE[j], label=f"{name}: sharp volume, blurred reference")
    ax.plot(fwhms, [bias[(name, f)][1] for f in fwhms], "s--", color=PALETTE[j], mfc="none", label=f"{name}: both blurred")
ax.set(xlabel="smoothing of the reference (mm FWHM)", ylabel="registration error (mm)", title="a still head registered to a blurred copy of itself")
ax.legend(fontsize=7)
fig.tight_layout()
```

At 2.5 mm of smoothing the sharp volume lands 0.06 mm off on the 2D series and 0.02 mm on
the GRASE, at 5 mm 1.12 and 0.72 mm, at 8 mm 2.27 and 1.80 mm; with both images smoothed
alike the error is zero at every width. The bias is not the blur, it is the mismatch, and
it grows quickly once the smoothing exceeds the voxel. Two consequences for ASL. A
reference averaged from a series that drifted is blurred by the drift, so the volumes
registered to it inherit a bias that depends on how much they moved, and a reference
smoothed on purpose, which some pipelines do for short or noisy series, must be matched
by smoothing the moving volumes by the same kernel. The drift run below shows the first
case.

## Which reference? Scored on the drift run

The `drift` run of [Chapter 10](./10-motion.md) moves every volume a little farther along
a line, so every volume needs a pose and the reference is never a volume that exists.
The cells register all sixty volumes to each candidate reference and score the result
three ways: the error of the within-pair offset against the true poses, which is the
motion the subtraction sees; the RMS error of the mean difference image against the
unmoved truth over the brain; and the same error at the one-voxel rim, where an offset
shows. The true poses, undone exactly, give the floor, which is not zero because sixty
resampled volumes are sixty blurred ones.

```{code-cell} python
:tags: [hide-input]
mo = data.load_dataset("motion")
drift = mo.run("drift")
m_d, ctx_d = drift.mag().astype(float), drift.context()
tbl = drift.motion()
true_d = np.column_stack([tbl[["trans_x", "trans_y", "trans_z"]].values, np.degrees(tbl[["rot_x", "rot_y", "rot_z"]].values)])
ctrl_idx = [i for i, c in enumerate(ctx_d) if c == "control"]; lab_idx = [i for i, c in enumerate(ctx_d) if c == "label"]
lab_rows = [l for _, l in quant.pair_indices(ctx_d)]
to_iu = presets.REFERENCE.signal_scale * np.exp(-ref.sidecar()["EchoTime"] / presets.T2_BLOOD)
truth_dm = drift.truth("deltamStatic")[..., lab_rows].mean(-1) * to_iu
m0_d = drift.m0scan().astype(float)
rmsb = lambda a, b, m: np.sqrt(np.mean((a - b)[m] ** 2))
scores, est = {}, {}

def score(label, poses):
    d = quant.subtract(apply_poses(m_d, poses), ctx_d).mean(-1)
    po = pair_offsets(poses, ctx_d) - pair_offsets(true_d, ctx_d)
    scores[label] = (np.sqrt((po[:, :3] ** 2).mean()), rmsb(d, truth_dm, inner), rmsb(d, truth_dm, rim))
    print(f"{label:>44}: pair offset error {scores[label][0]:.3f} mm; mean ΔM RMSE {scores[label][1]:.1f}, at the rim {scores[label][2]:.1f} image units")

score("uncorrected", np.zeros_like(true_d))
score("the true poses", true_d - true_d.mean(0))
est["control mean"] = register_series(m_d, m_d[..., ctrl_idx].mean(-1), pts)
score("control mean, least squares", est["control mean"])
est["joint mean"] = register_series(m_d, m_d.mean(-1), pts)
score("mean of every volume, least squares", est["joint mean"])
```

```{code-cell} python
:tags: [hide-input]
est["M0, NMI"] = register_series(m_d, m0_d, pts, cost="nmi")
score("M0 scan, normalized mutual information", est["M0, NMI"])
est["M0, LS"] = register_series(m_d, m0_d, pts, cost="ssd")
score("M0 scan, least squares", est["M0, LS"])
own = est["control mean"].copy()
own[lab_idx] = register_series(m_d, m_d[..., lab_idx].mean(-1), pts, volumes=lab_idx)
score("own-type templates, not aligned to each other", own)
reg_own = apply_poses(m_d, own)
off = register_volume(smooth(reg_own[..., lab_idx].mean(-1)), smooth(reg_own[..., ctrl_idx].mean(-1)), pts)
own_off = own.copy(); own_off[lab_idx] += off
score(f"own-type templates + offset step ({np.linalg.norm(off[:3]):.3f} mm)", own_off)
```

```{code-cell} python
:tags: [hide-input]
two_pass = est["control mean"].copy()
template = apply_poses(m_d, two_pass)[..., ctrl_idx].mean(-1)                           # re-average the aligned controls, register again
two_pass = register_series(m_d, template, pts)
score("control mean, second pass on the aligned mean", two_pass)
dm_ref = quant.subtract(m_d, ctx_d).mean(-1)
est["ΔM, NMI"] = register_series(m_d, dm_ref, pts, cost="nmi")
score("mean ΔM of the raw pairs, NMI", est["ΔM, NMI"])
print(f"the mean ΔM used as a reference has a gray matter SNR of {dm_ref[(fr['2D EPI']['gm'] > 0.9) & inner].mean() / quant.subtract(m_d, ctx_d).std(-1)[(fr['2D EPI']['gm'] > 0.9) & inner].mean() * np.sqrt(30):.1f}")
fig, ax = plt.subplots(figsize=(9, 3.4))
labels = list(scores)
ax.barh(range(len(labels)), [scores[l][2] for l in labels], color=PALETTE[0])
ax.set(yticks=range(len(labels)), yticklabels=labels, xlabel="mean ΔM error at the rim of the brain (image units)", title="drift run: what each reference leaves at the edges")
ax.invert_yaxis()
fig.tight_layout()
```

The first four rows are the same-contrast references. Uncorrected, the drift leaves a
within-pair offset error of 0.031 mm and a mean ΔM error of 15.4 units (12.4 at the rim).
The true poses bring that to 9.1 and 10.6. The control mean, least squares, reaches 11.2
over the brain but 20.2 at the rim, worse than no correction there, and the mean of every
volume 23.5: both references are averages of a drifting series, blurred by 3 mm of drift,
and the volumes registered to them carry the bias of the previous section, each by an
amount that depends on where in the drift it sat. Registering a second time to the mean
of the aligned volumes does not help (22.7), because the volumes were aligned to a
blurred target in the first place. The sharp references do better. The M0 scan with
mutual information leaves the rim at 9.0, the floor, but its per-volume estimates jitter:
a within-pair offset error of 0.068 mm, four times the same-contrast references', which
averages out of the thirty-pair mean here and would not out of a short series. The M0
scan with least squares is the best single reference in this table (8.4 over the brain,
8.1 at the rim, 0.018 mm), and that is the lesson of the cost curves above in a milder
form: the 2D control and the M0 scan differ by a bright fluid and nothing else, and
least squares tolerates that while rewarding sharpness. Own-type templates, the labels
to the label mean and the controls to the control mean, align each type perfectly within
itself and leave the two types where they were relative to each other (0.031 mm, the
uncorrected offset); the offset step finds 0.043 mm between the templates and brings the
pair error to 0.009 mm, the smallest in the table, and the rim to 8.7. The last row is a
warning: the mean ΔM of the raw pairs, which a reader may be tempted to use because it
shows the perfusion and not the static tissue, has a gray matter SNR of 3.2, and a
registration to it is a registration to noise, with within-pair errors of 0.67 mm and a
mean ΔM that misses the truth by 80 units. A perfusion image is a usable reference only
when it is far sharper and cleaner than this one, which spiral acquisitions with many
pairs and background suppression can provide and a 2D series of thirty pairs cannot.

## The zig-zag regression

ASLtbx {cite:p}`wang2008` and ExploreASL {cite:p}`mutsaerts2020` register every volume,
control and label alike, to one reference with SPM's realign and then remove the
label-control alternation from each of the six parameter series by regression
{cite:p}`wang2012`: whatever part of the estimated motion flips sign with the labeling is
assumed to be the spurious offset and is subtracted. On a still head with a
cross-contrast reference that is exactly right. The cell runs the regression, ordinary
least squares and a Huber-weighted version that counts outliers less, on the page's
estimates for the random run, where six volumes jumped, and the drift run, where the
true motion is a ramp; the true poses are regressed the same way to show how much
alternation the real motion contains.

```{code-cell} python
:tags: [hide-input]
rnd = mo.run("random")
m_r, ctx_r = rnd.mag().astype(float), rnd.context()
tbl_r = rnd.motion()
true_r = np.column_stack([tbl_r[["trans_x", "trans_y", "trans_z"]].values, np.degrees(tbl_r[["rot_x", "rot_y", "rot_z"]].values)])
est_r = register_series(m_r, m_r.mean(-1), pts)
print(f"{'run':>7} {'estimate':>24} {'pair offset error (mm)':>23} {'alternation removed (mm, x y z)':>32}")
zz = {}
for name, e, true in (("random", est_r, true_r), ("drift", est["joint mean"], true_d)):
    ctx_here = ctx_r if name == "random" else ctx_d
    for how, robust in (("joint estimate", None), ("after OLS zig-zag removal", False), ("after Huber zig-zag removal", True)):
        p, removed = (e, np.zeros(6)) if robust is None else regress_alternation(e, ctx_here, robust=robust)
        zz[(name, how)] = p
        po = pair_offsets(p, ctx_here) - pair_offsets(true, ctx_here)
        print(f"{name:>7} {how:>24} {np.sqrt((po[:, :3] ** 2).mean()):>23.3f} {np.array2string(removed[:3], precision=3, suppress_small=True):>32}")
    _, in_truth = regress_alternation(true, ctx_here)
    print(f"{'':>7} {'alternation in the TRUE poses':>24} {'':>23} {np.array2string(in_truth[:3], precision=3, suppress_small=True):>32}")
fig, axes = plt.subplots(1, 2, figsize=(11, 3.3))
axes[0].plot(true_r[:, 0] - true_r[:, 0].mean(), "k-", lw=1, label="true x")
for j, how in enumerate(("joint estimate", "after OLS zig-zag removal", "after Huber zig-zag removal")):
    axes[0].plot(zz[("random", how)][:, 0], ["o", "s", "^"][j], color=PALETTE[j], ms=3, mfc="none", label=how)
axes[0].set(xlabel="volume", ylabel="x (mm)", title="random run: the estimates")
axes[0].legend(fontsize=6)
axes[1].plot(pair_offsets(true_d, ctx_d)[:, 0], "k-", lw=1, label="true")
for j, how in enumerate(("joint estimate", "after OLS zig-zag removal", "after Huber zig-zag removal")):
    axes[1].plot(pair_offsets(zz[("drift", how)], ctx_d)[:, 0], ["o", "s", "^"][j], color=PALETTE[j], ms=3, mfc="none", label=how)
axes[1].set(xlabel="pair", ylabel="label − control, x (mm)", title="drift run: the motion within each pair")
axes[1].legend(fontsize=6)
fig.tight_layout()
```

Neither run has a spurious alternation to remove, since the page's reference is the mean
of every volume and its same-contrast cost leaves label and control where they are; what
the regression finds is real motion that happens to project onto the alternation. In the
random run the six jumps, which fell on labels and controls unequally, contain 0.063 mm
of alternation along $x$ in the true poses themselves; ordinary least squares removes
0.052 mm of it from every volume, puts a zig-zag into the 54 still volumes (left panel,
orange), and doubles the within-pair error from 0.047 to 0.093 mm. The Huber fit, which
down-weights the six outliers, removes 0.004 mm and leaves the error at 0.038. In the
drift run the within-pair motion is the ramp's step from one volume to the next, 0.05 mm
in $x$ on every pair, and that step *is* an alternation relative to the pair's mean: both
fits remove 0.027 mm of it, halve the within-pair motion they were meant to preserve, and
double the within-pair error from 0.018 to 0.035 mm (right panel). Robustness cannot help,
because a drift is not an outlier. The regression is exact when the only alternation in
the estimates is spurious, and harmful whenever the head's own motion has one, which a
drift always does and a jump often does; the offset step of the previous sections removes
the same spurious offset without touching the estimates of individual volumes.

## Short delays: the label holds arterial signal

At a delay shorter than the arterial transit time the labeled blood is still in the
arteries, and the label image carries it as bright vessels that the control image does
not have ([Chapter 19](../05-advanced/19-frontiers.md)). A registration of the label to
the control, or of both to a reference, then has a contrast difference to fit that is
not the head's position. The `vascular` dataset is a multi-delay four-shot GRASE series
with an arterial compartment (`uncrushed`) and without (`none`), four pairs at each of
seven delays; the cell registers each delay's label mean to its control mean.

```{code-cell} python
:tags: [hide-input]
vas = data.load_dataset("vascular")
print(f"{'run':>10} {'PLD (s)':>8} {'GM ΔM':>6} {'label -> control, NMI (mm / deg)':>34}")
for name in ("none", "uncrushed"):
    r = vas.run(name)
    m, ctx_v, p = r.mag().astype(float), r.context(), r.sidecar()
    plds = np.array(p["PostLabelingDelay"], float)
    mk = r.mask(); mk[:, :, [0, -1]] = False
    p_here = sample_points(mk & ~np.isin(np.arange(20)[None, None, :] * np.ones(mk.shape, int), [1, 18]), 6000, seed=0)
    gm_v = r.fractions()["gm"] > 0.9
    for pld in (0.1, 0.3, 0.5, 1.8):
        ci = [i for i, c in enumerate(ctx_v) if c == "control" and plds[i] == pld]
        li = [i for i, c in enumerate(ctx_v) if c == "label" and plds[i] == pld]
        cm, lm = m[..., ci].mean(-1), m[..., li].mean(-1)
        e = register_volume(smooth(lm), smooth(cm), p_here, cost="nmi")
        print(f"{name:>10} {pld:>8.1f} {(cm - lm)[gm_v & mk].mean():>6.0f} {np.linalg.norm(e[:3]):>22.3f} / {np.abs(e[3:]).max():.3f}")
```

Without an arterial compartment the label lands on the control within 0.005 mm at every
delay. With it, the label at 0.1 and 0.3 s holds a difference signal of 235 and 241 units,
four times the tissue's, and lands 0.051 and 0.043 mm off; at 0.5 s, when most of the
label has reached the tissue, 0.016 mm; at 1.8 s nothing. That is the arterial signal
acting as a contrast difference, and it is a lower bound: the simulator spreads its
arterial blood over every gray matter voxel as a fraction, where a brain concentrates it
in a few bright vessels with sharp edges that a cost function can lock onto, and a
multi-delay pipeline that corrects a label-control offset per delay should not trust the
estimate it gets from the earliest delays. ASLPrep's offset step borrows the nearest
later delay's estimate below 0.5 s for that reason.

## What the pipelines do

The published pipelines made these choices long before the experiments above, mostly by
inheriting a tool from fMRI and adapting it to the pairs. The table is what each does to
the series before subtraction, as documented at the time of writing; versions change, and
the documentation of each tool is the authority.

| Pipeline | Engine and cost | Reference | The label-control offset | Afterwards |
|---|---|---|---|---|
| ASLPrep {cite:p}`adebimpe2022` | MCFLIRT {cite:p}`jenkinson2002`, normalized correlation | one reference for the run, chosen in the order CBF volumes, ΔM volumes, an M0 volume in the series, the separate M0 scan, the control mean; each volume *type* is registered to it separately (`typewise`) | left in place by default; the opt-in `typewise_offset` method registers the corrected label mean of each delay onto the corrected control mean with AFNI's 3dvolreg {cite:p}`cox1999` and composes that into the label transforms (delays under 0.5 s, 0.8 s for PASL, borrow the nearest other delay's estimate) | SCORE, then SCRUB, on the CBF series {cite:p}`dolui2017,dolui2016`; framewise displacement in the confounds |
| ExploreASL {cite:p}`mutsaerts2020` | SPM realign {cite:p}`friston1995`, least squares, two passes to the mean | the mean of the whole series, control and label together | the zig-zag regression on the realignment parameters {cite:p}`wang2012` | ENABLE's motion-sorted exclusion of pairs {cite:p}`shirzadi2018`; the M0 realigned separately |
| ASLtbx {cite:p}`wang2008` | SPM realign, least squares, to the first volume | the first volume | the zig-zag regression {cite:p}`wang2012` | optional outlier cleaning |
| BASIL / oxford_asl {cite:p}`chappell2009,chappell2023` | MCFLIRT (`--mc`) | the series itself, with the calibration image aligned to it afterwards so that the ASL data are interpolated once | not addressed | the kinetic model is fitted to the differences |
| A scanner with prospective correction, PROMO on 3D PCASL {cite:p}`zun2014` | navigators between the shots, the head tracked and the next shot steered | the head itself | none to speak of, since every shot is acquired in the same frame | the volumes arrive aligned; residual motion is still estimated offline |

Three patterns run through the table. First, every retrospective pipeline registers to a
single image, and the cross-contrast cases, label volumes to an M0 or to a mixed mean,
are where the spurious offsets of this chapter come from; the two answers in use are to
correct the offset after the fact (`typewise_offset`, the zig-zag regression) or to live
with it. Second, the least-squares engines, SPM realign and 3dvolreg, are the most precise
within one contrast and the ones that fail across contrasts, which is why ExploreASL's
suppressed series are realigned to their own mean and never to the M0. Third, outlier
handling comes after registration in all of them, because, as [Chapter 10](./10-motion.md)
showed, registration does not repair a pair whose two halves moved relative to each other;
it only makes the rest of the series agree.

The 3D readouts add a case the table does not show: the GRASE and spiral references have
white matter brighter than gray (GM/WM 0.95 above), the opposite of the 2D EPI and of a
T1-weighted anatomical image. A boundary-based coregistration to the anatomy
{cite:p}`greve2009` assumes a sign for that contrast, and given the wrong one it can fail
silently, landing several millimeters off while staying inside its fallback rule. A
pipeline that measures the gray-white contrast of its reference and sets the sign from it
avoids the failure; a perfusion image, whose gray matter is always the bright tissue, is
the other way out, when its SNR allows it.

## What this implies for acquisition

- **Give the pipeline a same-contrast target.** An unsuppressed series can be registered
  to its own control mean or, with least squares, to a sharp M0 scan; a suppressed series
  can only be registered to itself, and a pipeline that sends it to the M0 scan with a
  least-squares engine will fail outright, with any other engine quietly. If the protocol
  is suppressed, acquire the M0 scan anyway and register *it* to the series, not the
  series to it.
- **Expect an offset of a few hundredths of a millimeter between label and control** from
  any cross-contrast registration, and look for a pipeline that removes it (the offset
  step) or does not create it (own-type references with a same-contrast engine). On a
  perfusion signal of half a percent, a twentieth of a voxel at the rim is half the
  signal there.
- **Do not regress the alternation out of a series that drifted or jumped.** The zig-zag
  regression is right only when the head's own motion has no alternation; a slow drift
  always does.
- **Keep references sharp.** An average of a moving series is blurred and biases what is
  registered to it; a sharp single image of a mildly different contrast can be the
  better target, and smoothing applied to a reference must be applied to the moving
  volumes too.
- **Know the contrast of your 3D reference.** The GRASE and spiral readouts have white
  matter brighter than gray, and a boundary-based coregistration to the anatomy fails
  silently with the wrong sign; check the overlay, or set the sign from the measured
  gray-white ratio.
- **With many pairs and suppression, a perfusion image can be the reference;** with
  thirty 2D pairs it is noise. Spiral protocols that register to a mean ΔM are using
  the SNR they have, not a rule that transfers.

## Further reading

Rigid registration and its costs {cite:p}`friston1995,jenkinson2001,jenkinson2002,roche1998,viola1997,studholme1999`;
AFNI's 3dvolreg {cite:p}`cox1996,cox1999`; the zig-zag regression {cite:p}`wang2012` in
ASLtbx {cite:p}`wang2008` and ExploreASL {cite:p}`mutsaerts2020`; ASLPrep's type-wise
correction and offset step {cite:p}`adebimpe2022`; BASIL and oxford_asl
{cite:p}`chappell2009,chappell2023`; boundary-based coregistration {cite:p}`greve2009`;
weighting pairs by their noise rather than their motion {cite:p}`tanenbaum2015`;
prospective correction for 3D PCASL {cite:p}`zun2014`; the arterial signal at short delays
{cite:p}`alsop2015`.
