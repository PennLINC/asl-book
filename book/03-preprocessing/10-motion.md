---
title: "10. Head motion"
kernelspec:
  name: python3
  display_name: Python 3
---

:::{admonition} Simulated datasets in this chapter
:class: note
- **Built in this page:** the true poses replayed on the reference series, and a rigid registration and an outlier rule written in the page ([Appendix B](../appendices/b-data-manifest.md#app-b-package-data)).
- **`motion`**: the reference series with random head jumps on six volumes (`random`), with a slow linear drift over the whole series (`drift`), and the jumps again under background suppression (`random-bgsup`), each with the true pose of every volume ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-motion)).
- **`ref-pcasl`**: the same series without motion, simulated with the same seed, so that it is the motion-free version of `random` volume for volume ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-ref-pcasl)).

Pipeline-tier datasets are simulated offline by aslscan ([Chapter 0.2](../00-frontmatter/the-simulated-datasets.md)).
:::

## Learning goals

After this chapter you can:

- explain why a millimeter of head motion, harmless in most MRI, corrupts an ASL difference image
- distinguish motion between the two volumes of a pair from motion between pairs, and say what each does to the mean perfusion image
- register the volumes of an ASL series rigidly, check the poses against the truth, see what registration leaves behind, and reject corrupted pairs with an outlier rule that knows when it cannot help
- choose acquisition settings that reduce the damage at the source

```{code-cell} python
:tags: [hide-cell]
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, optimize

from aslbook import data, phantom, presets, quant
from aslbook.plotting import INK, PALETTE, set_style, show_slice

set_style()
K = phantom.DISPLAY_SLICE
VOX = np.array(presets.REFERENCE.voxel_mm)          # mm along the image axes x, y, z

def rotation(rx, ry, rz):
    """R = Rz Ry Rx for angles in degrees, the simulator's order."""
    rx, ry, rz = np.radians([rx, ry, rz])
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def pose_to_voxels(t_mm, r_deg, shape, undo=False):
    """ndimage's (matrix, offset) for a pose p' = R (p - c) + c + t in mm about the
    field-of-view center c. undo=False resamples an image the way the pose moves the head
    (output voxel q takes the input at pose^-1 q); undo=True moves it back."""
    c = (np.array(shape[:3]) - 1) / 2 * VOX
    R, t = rotation(*r_deg), np.asarray(t_mm, float)
    M, off = (R, c + t - R @ c) if undo else (R.T, c - R.T @ (c + t))
    return M * VOX[None, :] / VOX[:, None], off / VOX      # mm -> voxel coordinates

def move_volume(vol, t_mm, r_deg, undo=False, order=1):
    M, off = pose_to_voxels(t_mm, r_deg, vol.shape, undo)
    return ndimage.affine_transform(vol, M, offset=off, order=order, mode="nearest")

def register_volume(moving, target, pts):
    """Rigid registration: the pose (tx, ty, tz in mm, rx, ry, rz in degrees) whose undoing
    best maps `moving` onto `target`, by the mean squared difference at the sample voxels
    `pts` (3, N), minimized with Powell's method from the identity."""
    tv = target[tuple(pts.astype(int))]

    def cost(p):
        M, off = pose_to_voxels(p[:3], p[3:], moving.shape, undo=True)
        return np.mean((ndimage.map_coordinates(moving, M @ pts + off[:, None], order=1, mode="nearest") - tv) ** 2)

    return optimize.minimize(cost, np.zeros(6), method="Powell", options={"xtol": 1e-3, "ftol": 1e-6}).x
```

## The physics

Nothing about head motion is specific to ASL: the head is a rigid body, and between two
volumes it can translate and rotate, six numbers in all. What is specific to ASL is the
size of the signal. The perfusion signal is the *difference* between a control and a label
volume ([Chapter 6](../02-labeling/06-the-asl-signal.md)): in the reference protocol a gray
matter voxel reads about 6200 image units in the control image and 30 less in the label
image, half a percent. The subtraction assumes that the two volumes contain the same tissue
in every voxel; if the head moved between them, they do not.

Consider a voxel at the edge of the brain, or at a boundary between gray matter and
cerebrospinal fluid. A shift of 1 mm, less than a third of the 3.5 mm voxel, changes what
fraction of the voxel is tissue and what fraction is fluid or air, and its control signal
changes by a corresponding fraction of its full value: hundreds of units, tens of percent.
The difference of a moved control and a still label therefore contains the perfusion
signal, 30 units, plus a bright and dark rim of hundreds of units along every edge, positive
on the side the head moved toward and negative on the other, which dwarfs the signal.

Two things follow. First, what matters is the motion *within* a pair, between the control
and the label that are subtracted: motion between pairs only blurs the mean when misaligned
clean images are averaged, while motion within one pair gives that pair a rim one or two
orders of magnitude above the signal, and one such pair can dominate the mean of thirty.
Second, the rim scales with the *static* signal, not the perfusion signal, so anything that
reduces the static signal reduces the rim in proportion; that is the case for background
suppression ([Chapter 9](./09-background-suppression.md)), measured at the end of this chapter.

## The aslscan setting that produces it

The `motion` overlay gives every volume of the series a rigid pose. The simulator applies
it to the finished simulation-grid images of that volume, before the acquisition stage adds
k-space, noise, and the rest ([Chapter 0.2](../00-frontmatter/the-simulated-datasets.md)),
so the anatomy of a moved volume is the same anatomy in a new place (one consequence: the
slice timing travels with the anatomy, which a real 2D readout would not do). A pose moves
a point at $p$ to

$$p' = R\,(p - c) + c + t,$$

with $c$ the center of the field of view, $t$ the translation in mm along the image's
voxel axes $x$, $y$, $z$ (3.5, 3.5, and 5 mm), and $R = R_z R_y R_x$ the rotation, in
degrees in the overlay and the sidecar's `AslscanSimulation.Motion` block, in radians in the
ground-truth table `desc-motion_gt.tsv`. `random` gives six volumes (5, 12, 13, 27, 40, 41)
independent jumps of up to 2 mm and 2°; `drift` moves every volume a little farther along a
line, 3 mm and 1° from first to last. The random run shares its noise seed with `ref-pcasl`
(the motion seed is salted separately), so the two series have the same noise in every
voxel and differ by the motion alone, which is verified first.

```{code-cell} python
:tags: [hide-input]
ds = data.load_dataset("motion")
runs = {name: ds.run(name) for name in ("random", "drift", "random-bgsup")}
ref = data.load_dataset("ref-pcasl").run()
mag = {name: r.mag().astype(float) for name, r in runs.items()}
mag_ref = ref.mag().astype(float)
ctx = ref.context()
poses = {name: r.motion() for name, r in runs.items()}
mask = ref.mask()
inner = mask.copy(); inner[:, :, [0, -1]] = False          # the top and bottom slices are scored separately, see below
gm = ref.fractions()["gm"] > 0.7
moved_vols = [5, 12, 13, 27, 40, 41]; still_vols = [v for v in range(60) if v not in moved_vols]
print(f"largest difference between motion/random and ref-pcasl on the {len(still_vols)} unmoved volumes: "
      f"{np.abs(mag['random'][..., still_vols] - mag_ref[..., still_vols]).max():.1f} image units")
print("moved volumes of the random run: " + ", ".join(f"{v} ({ctx[v]})" for v in moved_vols))
fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
for ax, name in zip(axes, ("random", "drift")):
    m = poses[name]
    for j, c in enumerate(["trans_x", "trans_y", "trans_z"]):
        ax.plot(m["volume"], m[c], color=PALETTE[j], lw=1.5, label=c.replace("trans_", "t"))
    for j, c in enumerate(["rot_x", "rot_y", "rot_z"]):
        ax.plot(m["volume"], np.degrees(m[c]), color=PALETTE[j], lw=1.5, ls="--", label=c.replace("rot_", "r"))
    ax.set(xlabel="volume", ylabel="mm (solid), degrees (dashed)", title=f"{name}: the true pose of every volume")
    ax.legend(ncol=2, loc="upper left")
fig.tight_layout()
```

The unmoved volumes are identical to the reference to the last bit, so every difference
between the two series below is caused by motion. The jumps land on both kinds of row:
volume 5 is a label whose control (volume 4) is still, and volumes 12 and 13, like 40 and
41, are the control and the label of one pair, each with its own pose. The drift is small
per volume (0.05 mm) and large over the run.

## See it: motion alone, with the noise subtracted away

Because the noise is shared, subtracting the reference from a moved volume shows the pure
motion artifact. Volume 5 moved by 0.7 mm right, 0.7 mm posterior, 0.3 mm up, and turned
1.5° about the vertical axis.

```{code-cell} python
:tags: [hide-input]
v = 5
change = mag["random"][..., v] - mag_ref[..., v]
t_true = poses["random"].loc[v, ["trans_x", "trans_y", "trans_z"]].values.astype(float)
r_true = np.degrees(poses["random"].loc[v, ["rot_x", "rot_y", "rot_z"]].values.astype(float))
replay = move_volume(mag_ref[..., v], t_true, r_true)
wrong = move_volume(mag_ref[..., v], -t_true, -r_true)
rms = lambda a, b: np.sqrt(np.mean((a - b)[inner] ** 2))
print(f"volume {v}: motion changes the image by up to {np.abs(change[inner]).max():.0f} units, RMS {rms(mag['random'][..., v], mag_ref[..., v]):.0f} over the brain")
print(f"the reference volume moved by the true pose matches the moved volume to RMS {rms(replay, mag['random'][..., v]):.0f} "
      f"(with the pose's sign flipped: {rms(wrong, mag['random'][..., v]):.0f})")
fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.3))
show_slice(axes[0], mag_ref[..., v], K, f"volume {v}, reference (still)", vmin=0, vmax=7000)
show_slice(axes[1], mag["random"][..., v], K, f"volume {v}, moved", vmin=0, vmax=7000)
show_slice(axes[2], change, K, "moved − still: motion alone", kind="diff", vmin=-2000, vmax=2000)
fig.colorbar(axes[2].images[0], ax=axes[2], shrink=0.75, label="image units")
fig.tight_layout()
```

The two images look the same; a shift of a fifth of a voxel is invisible to the eye. Their
difference is not: every edge has a rim of ±2000 units and more, up to 4500 at the lateral
edges the 1.5° turn swept farthest, a hundred times the perfusion signal of 30; the RMS
change over the brain is 430 units. The printed check pins down the convention: moving the
reference volume by the table's pose, exactly as the formula says, reproduces the moved
volume to an RMS of 140 units (the replay's interpolation residual), while the sign-flipped
pose leaves 700. The table describes where the head went, not the way back.

## See it: pairs versus volumes

The subtraction turns the sixty volumes into thirty difference images. The bar chart
measures how far each departs from the mean difference image (RMS over the brain), the
panels show three of them, and the second figure is the mean of all thirty against the
motion-free reference and the truth.

```{code-cell} python
:tags: [hide-input]
d_rand = quant.subtract(mag["random"], ctx)
d_ref = quant.subtract(mag_ref, ctx)
pairs = quant.pair_indices(ctx)

def deviation(d, keep=None):
    keep = np.ones(d.shape[-1], bool) if keep is None else keep
    m = d[..., keep].mean(-1)
    return np.array([np.sqrt(np.mean((d[..., k] - m)[inner] ** 2)) for k in range(d.shape[-1])])

dev = deviation(d_rand)
moved_pairs = sorted({k for k, (c, l) in enumerate(pairs) if c in moved_vols or l in moved_vols})
print("deviation of each pair's difference image from the mean (RMS over the brain, image units):")
for k in (0, 2, 6, 13, 20):
    print(f"  pair {k:>2} (volumes {pairs[k][0]:>2} control, {pairs[k][1]:>2} label): {dev[k]:6.0f}   moved: {[v for v in pairs[k] if v in moved_vols]}")
fig = plt.figure(figsize=(12, 3.4))
gs = fig.add_gridspec(1, 4, width_ratios=[1.6, 1, 1, 1])
ax = fig.add_subplot(gs[0])
ax.bar(np.arange(30), dev, color=[PALETTE[1] if k in moved_pairs else PALETTE[0] for k in range(30)])
ax.set(xlabel="pair", ylabel="RMS deviation from the mean ΔM", title="random run: which pairs are corrupted")
for k in moved_pairs:
    ax.text(k, dev[k] + 15, "volume " + "+".join(str(v) for v in pairs[k] if v in moved_vols), ha="center", fontsize=7)
for j, (k, title) in enumerate([(0, "pair 0: both still"), (2, "pair 2: label moved"), (6, "pair 6: both moved, differently")]):
    show_slice(fig.add_subplot(gs[j + 1]), d_rand[..., k], K, title, kind="diff", vmin=-1500, vmax=1500)
fig.tight_layout()

lab_rows = [l for _, l in pairs]
to_image_units = presets.REFERENCE.signal_scale * np.exp(-ref.sidecar()["EchoTime"] / presets.T2_BLOOD)
truth_dm = {name: r.truth("deltamStatic")[..., lab_rows].mean(-1) * to_image_units for name, r in runs.items()}
truth_dm["random-bgsup"] *= runs["random-bgsup"].simulation()["BackgroundSuppressionLabelFactor"]
print(f"mean ΔM, RMS error against the unmoved truth over the brain: reference {rms(d_ref.mean(-1), truth_dm['random']):.1f}, "
      f"random run {rms(d_rand.mean(-1), truth_dm['random']):.1f} image units (gray matter ΔM is about 30)")
fig2, axes = plt.subplots(1, 3, figsize=(9.5, 3.3))
show_slice(axes[0], d_ref.mean(-1), K, "mean ΔM, no motion", kind="diff", vmin=-80, vmax=80)
show_slice(axes[1], d_rand.mean(-1), K, "mean ΔM, random motion", kind="diff", vmin=-80, vmax=80)
show_slice(axes[2], d_rand.mean(-1) - truth_dm["random"], K, "error against the truth", kind="diff", vmin=-80, vmax=80)
fig2.colorbar(axes[2].images[0], ax=axes[2], shrink=0.75, label="image units")
fig2.tight_layout()
```

A still pair (blue bars) deviates from the mean by the noise alone, 66 units, and its
difference image is noise with a faint brain in it. Pair 2, whose label moved while its
control stayed, deviates by 440 units: the whole rim of the previous figure is in it. Pair
6 is instructive: both of its volumes moved, by different amounts, so its rim is the
difference of the two poses rather than either one; it happens to be about as large (400),
and pair 20, the other pair with two moved volumes, is the worst in the run (810) because
volumes 40 and 41 moved in opposite directions. Pair 13 (volume 27, 0.8 mm with almost no
rotation) shows that the smallest jump in the run still produces a rim three times the
noise. The mean of the thirty difference images inherits all of them: four corrupted pairs
raise its RMS error from 10.5 units, the noise floor of a 30-pair average, to 38, and the
error map is the rim pattern again, now with the ventricles outlined as well as the brain.

:::{dropdown} The truth maps of a motion run, and why the edge slices are scored apart
A motion run carries two ground-truth difference series: `deltam`, the truth *moved by each
volume's pose*, and `deltamStatic`, the truth in the unmoved anatomy, the target of any
motion correction. Every score here is against the mean of `deltamStatic` over the label
rows, in image units (signal scale × the blood's $T_2$ decay at TE = 0.93 × 100), over the
brain voxels of slices 1 to 18: the simulator fills every sample outside the slab with zero,
so a through-plane motion of even a fraction of a slice empties most of an edge slice
(volume 12, moved 0.4 mm along $z$, keeps 1 % of its bottom slice). On a scanner the excited
slab moves with the head and the edge slices are only partly affected; either way they
should not set the score, and the registration below samples only the interior slices.
:::

## Correction step by step: rigid registration

A pipeline does not know which volumes moved, so it registers every volume to a reference:
it searches for the six pose parameters whose undoing makes the volume most similar to the
reference, then resamples the volume with them. FSL's MCFLIRT {cite:p}`jenkinson2002` and
SPM's realign are the tools most ASL pipelines use {cite:p}`adebimpe2022,mutsaerts2020`.
The version in the setup cell is small enough to read in full: the reference is the mean
control image, both images are smoothed by about a voxel, the similarity is the mean
squared difference at 6000 brain voxels, and Powell's method searches from the identity.
The smoothing matters more than it looks: resampling a noisy volume also averages its
noise, so without it the search prefers a shift of half a voxel, where the noise is
averaged most, to the true shift.

```{code-cell} python
:tags: [hide-input]
rng = np.random.default_rng(0)
sample = inner.copy(); sample[:, :, [1, -2]] = False
pts = np.array(np.unravel_index(rng.choice(np.flatnonzero(sample.ravel()), 6000, replace=False), mask.shape), float)
smooth = lambda x: ndimage.gaussian_filter(x, (1.5, 1.5, 1.0))
controls = [c for c, _ in pairs]
est = {}
for name in ("random", "drift"):
    target = smooth(mag[name][..., controls].mean(-1))
    est[name] = np.array([register_volume(smooth(mag[name][..., v]), target, pts) for v in range(60)])

fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
for j, name in enumerate(("random", "drift")):
    m = poses[name]
    true_t, true_r = m[["trans_x", "trans_y", "trans_z"]].values, np.degrees(m[["rot_x", "rot_y", "rot_z"]].values)
    true_t = true_t - true_t.mean(0); true_r = true_r - true_r.mean(0)     # the reference is the mean pose
    err_t = est[name][:, :3] - true_t; err_r = est[name][:, 3:] - true_r
    print(f"{name}: registration error, RMS over 60 volumes: translations {np.sqrt((err_t ** 2).mean(0)).round(2)} mm (x, y, z), "
          f"rotations {np.sqrt((err_r ** 2).mean(0)).round(2)} degrees; largest {np.abs(err_t).max():.2f} mm, {np.abs(err_r).max():.2f} degrees")
    for ax, true, e, unit in [(axes[2 * j], true_t, est[name][:, :3], "mm"), (axes[2 * j + 1], true_r, est[name][:, 3:], "degrees")]:
        for a, lab in enumerate("xyz"):
            ax.plot(np.arange(60), true[:, a], color=PALETTE[a], lw=1.2, label=f"{lab} true")
            ax.plot(np.arange(60), e[:, a], "o", color=PALETTE[a], ms=3, mfc="none", label=f"{lab} estimated")
        ax.set(xlabel="volume", ylabel=unit, title=f"{name}: {'translations' if unit == 'mm' else 'rotations'}")
    axes[2 * j].legend(ncol=3, fontsize=6, loc="lower left")
fig.tight_layout()
```

The lines are the true poses relative to the mean pose, which is what a registration to the
mean control image can know; the circles are the estimates. For the random run the six
jumps are recovered to 0.05 mm and 0.05° RMS, with the largest miss 0.18 mm, and the still
volumes come out still. For the drift the estimates follow the ramp in the plane; the two
through-plane rotations, a quarter of a degree end to end, are only partly recovered, since
5 mm slices give the search little to work with along $z$. The poses are then undone by
resampling each volume, and the difference images are recomputed:

```{code-cell} python
:tags: [hide-input]
reg = {name: np.stack([move_volume(mag[name][..., v], est[name][v, :3], est[name][v, 3:], undo=True) for v in range(60)], -1)
       for name in ("random", "drift")}
d_reg = {name: quant.subtract(reg[name], ctx) for name in reg}
d_raw = {"random": d_rand, "drift": quant.subtract(mag["drift"], ctx)}
dev_reg = deviation(d_reg["random"])
print("random run, deviation of the corrupted pairs before -> after registration: " + "; ".join(f"pair {k} {dev[k]:.0f} -> {dev_reg[k]:.0f}" for k in moved_pairs))
print(f"volume 5 against the still reference: RMS {rms(mag['random'][..., 5], mag_ref[..., 5]):.0f} before, {rms(reg['random'][..., 5], mag_ref[..., 5]):.0f} after registration, "
      f"{rms(move_volume(mag['random'][..., 5], t_true, r_true, undo=True), mag_ref[..., 5]):.0f} with the true pose")
for name in ("random", "drift"):
    print(f"{name}: mean ΔM RMS error {rms(d_raw[name].mean(-1), truth_dm[name]):.1f} before, {rms(d_reg[name].mean(-1), truth_dm[name]):.1f} after registration")
fig, axes = plt.subplots(1, 4, figsize=(12, 3.3))
show_slice(axes[0], d_rand[..., 2], K, "pair 2 before registration", kind="diff", vmin=-1500, vmax=1500)
show_slice(axes[1], d_reg["random"][..., 2], K, "pair 2 after registration", kind="diff", vmin=-1500, vmax=1500)
show_slice(axes[2], d_reg["random"][..., 2], K, "the same, ×5", kind="diff", vmin=-300, vmax=300)
show_slice(axes[3], d_reg["random"].mean(-1) - truth_dm["random"], K, "mean ΔM error after registration", kind="diff", vmin=-80, vmax=80)
fig.tight_layout()
```

Registration shrinks the rim of pair 2 but does not remove it: the pair's deviation falls
from 440 to 230, and at five times the contrast (third panel) the rim is plainly still
there. This is not a failure of the pose estimate, which is within 0.1 mm. Moving a volume
back means interpolating it, which blurs every edge by an amount negligible for the anatomy
and large compared with ΔM; the simulator's move was itself an interpolation, so the
corrected volume has been resampled twice and no longer matches one that never was. Undoing
the *true* pose leaves the same residual (260 against the still reference, from 430). For
the drift, where every volume is off by a little and no pair by much, registration does
nearly all the work: the mean ΔM error falls from 15.4 to 11.2 units, close to the noise
floor of 10.5. For the random run it falls from 38 to 18; the rest sits in the four
corrupted pairs, and the way to deal with them is to leave them out.

## Correction step by step: rejecting pairs

Outlier rejection treats the thirty difference images as thirty measurements of the same
map and discards the ones that disagree with the rest. The SCORE algorithm
{cite:p}`dolui2017` first discards volumes whose mean CBF is far from the median, then
iteratively removes the volume whose removal most improves the correlation of the mean map
with the tissue maps. The rule here is a simplified first step: compute each pair's RMS
deviation from the current mean, express it as a robust $z$-score (distance from the median
in units of the median absolute deviation, which the outliers cannot inflate), discard the
worst pair if its $z$ exceeds 2.5, recompute the mean, and repeat until nothing does.

```{code-cell} python
:tags: [hide-input]
def reject_pairs(d, threshold=2.5):
    keep = np.ones(d.shape[-1], bool)
    while True:
        dv = deviation(d, keep)
        med = np.median(dv[keep]); mad = 1.4826 * np.median(np.abs(dv[keep] - med))
        z = np.where(keep, (dv - med) / mad, -np.inf)
        if z.max() <= threshold:
            return keep, z
        keep[np.argmax(z)] = False

flags = {}
for name, series in [("random, uncorrected", d_rand), ("random, registered", d_reg["random"]),
                     ("drift, uncorrected", d_raw["drift"]), ("drift, registered", d_reg["drift"])]:
    keep, z = flags[name], _ = reject_pairs(series)
    print(f"{name:>22}: rejected pairs {np.flatnonzero(~keep).tolist()}; largest z among the kept pairs {z[keep].max():.1f}")
fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
for ax, name in zip(axes, ("random", "drift")):
    dv = deviation(d_reg[name])
    keep = flags[f"{name}, registered"]
    ax.bar(np.arange(30), dv, color=[PALETTE[1] if not keep[k] else PALETTE[0] for k in range(30)])
    med = np.median(dv[keep]); mad = 1.4826 * np.median(np.abs(dv[keep] - med))
    ax.axhline(med + 2.5 * mad, color=INK["secondary"], ls="--", lw=1)
    ax.text(29.5, med + 2.5 * mad, "median + 2.5 MAD", ha="right", va="bottom", fontsize=7, color=INK["secondary"])
    ax.set(xlabel="pair", ylabel="RMS deviation from the mean ΔM", title=f"{name}, after registration: rejected pairs in orange")
fig.tight_layout()
```

On the random run the rule finds pairs 2, 6, 13, and 20, exactly the pairs that contain a
moved volume, before and after registration; the largest $z$ among the kept pairs is 2.3.
On the drift run it finds nothing, before or after: the drift has no outliers, only a slow
slide, and a rule that looks for outliers has nothing to find. That is the case registration
was for. The arch in the drift panel is not motion: the volumes at the two ends of the run
were resampled by nearly half a voxel, which averages their noise, while the middle
volumes, already near the mean position, were barely resampled and keep their full noise.

## Residual error versus truth

The two corrections in every combination, scored on the mean difference image and on the CBF
computed from it with the run's own M0 scan ([Chapter 14](../04-quantification/14-cbf-quantification.md)) against the unmoved truth; the motion-free reference sets the noise floor.

```{code-cell} python
:tags: [hide-input]
p = ref.sidecar()
plds = quant.slice_plds(p["PostLabelingDelay"], p["SliceTiming"])[None, None, :]
truth_cbf = ref.truth("perfusion")
cbf_of = lambda run, dm: quant.cbf_pcasl(dm, run.m0scan().astype(float), plds, tau=p["LabelingDuration"])
rows, cbf_maps = [], {}
for name in ("random", "drift"):
    big = (np.abs(est[name][:, :3]).max(1) > 0.1) | (np.abs(est[name][:, 3:]).max(1) > 0.1)   # estimated motion above 0.1 mm or 0.1 degree
    reg_sel = np.where(big[None, None, None, :], reg[name], mag[name])                          # resample only those volumes
    cases = {"uncorrected": (d_raw[name], None), "registered": (d_reg[name], None),
             "rejected": (d_raw[name], flags[f"{name}, uncorrected"]), "registered + rejected": (d_reg[name], flags[f"{name}, registered"]),
             f"registered ({big.sum()} vols) + rejected": (quant.subtract(reg_sel, ctx), flags[f"{name}, registered"])}
    for label, (d, keep) in cases.items():
        dm = d.mean(-1) if keep is None else d[..., keep].mean(-1)
        cbf = cbf_of(runs[name], dm)
        cbf_maps[(name, label)] = cbf
        rows.append((name, label, 30 if keep is None else int(keep.sum()), rms(dm, truth_dm[name]), quant.score(cbf, truth_cbf, inner & gm)["rmse"]))
cbf_ref = cbf_of(ref, d_ref.mean(-1))
rows.append(("reference", "no motion", 30, rms(d_ref.mean(-1), truth_dm["random"]), quant.score(cbf_ref, truth_cbf, inner & gm)["rmse"]))
print(f"{'run':>10} {'correction':>34} {'pairs':>5} {'ΔM RMSE':>9} {'CBF RMSE in GM':>15}")
for r in rows:
    print(f"{r[0]:>10} {r[1]:>34} {r[2]:>5} {r[3]:>9.1f} {r[4]:>15.1f}")

fig, axes = plt.subplots(1, 4, figsize=(12, 3.3), layout="constrained")
for ax, label in zip(axes, ["uncorrected", "registered", "rejected", "registered + rejected"]):
    show_slice(ax, np.where(inner, cbf_maps[("random", label)] - truth_cbf, np.nan), K, f"CBF error, {label}", kind="diff", vmin=-60, vmax=60)
fig.colorbar(axes[3].images[0], ax=axes.tolist(), shrink=0.75, label="ml/100 g/min")
```

Read the table by run. For the random jumps, rejection is what works: dropping the four
corrupted pairs brings the ΔM error from 38 to 11 units and the gray matter CBF error from
70 to 22 ml/100 g/min, which is the reference's own noise-limited error of 21; the lost
pairs cost a few percent of noise. Registration alone gets halfway (18 units, CBF 33), and
registration followed by rejection removes the same four pairs but ends *worse* than
rejection alone, at 14 units and a CBF error of 31: the registration estimated poses of a
few hundredths of a millimeter for the 54 still volumes, and resampling a volume by a
hundredth of a voxel still changes its edge voxels by tens of units, differently for the
control and the label of a pair, which is visible on a signal of 30. Leaving alone every
volume whose estimated motion is below 0.1 mm and 0.1° (fifth row: only the six moved
volumes are resampled) gives back what rejection alone achieves. For the drift, rejection
does nothing and registration does nearly everything: 15.4 to 11.2 units, and a CBF error
of 27 against the floor of 21, the difference being the blur of sixty resampled volumes.
The error maps say the same in pictures. A real pipeline does both, in this order:
registration for the slow motion every subject has, then rejection for the pairs a jump has
corrupted; ASLPrep, for example, realigns with MCFLIRT and then applies SCORE and its
successor SCRUB {cite:p}`adebimpe2022`. The drift case also shows why the reference
matters: the mean control image of 3 mm of drift is itself blurred, so pipelines refine the
reference by re-averaging the aligned volumes and registering again.

## See it: the same jumps under background suppression

The rim scales with the static signal. The `random-bgsup` run applies the same six poses
to the reference protocol with two suppression pulses ([Chapter 9](./09-background-suppression.md)),
which leave gray and white matter near zero at the first slice's readout and cut the
perfusion signal only by the label factor of 0.81. As in Chapter 9, the inferior slices are
read out while the tissue is still slightly inverted, so the subtraction uses the signed
image (magnitude times the cosine of the phase), divided by the label factor.

```{code-cell} python
:tags: [hide-input]
signed = mag["random-bgsup"] * np.cos(runs["random-bgsup"].phase().astype(float))    # the inferior slices are read out while the tissue is still inverted
d_bs = quant.subtract(signed, ctx) / runs["random-bgsup"].simulation()["BackgroundSuppressionLabelFactor"]
dev_bs = deviation(d_bs)
print(f"gray matter control signal: {mag['random'][..., 4][inner & gm].mean():.0f} without, {mag['random-bgsup'][..., 4][inner & gm].mean():.0f} with suppression (slice-averaged); "
      f"deviation of pair 2: {dev[2]:.0f} without, {dev_bs[2]:.0f} with; of a still pair: {dev[0]:.0f} and {dev_bs[0]:.0f}")
keep4 = flags["random, uncorrected"]
print(f"what the four corrupted pairs add to the mean ΔM (RMS over the brain): {rms(d_rand.mean(-1), d_rand[..., keep4].mean(-1)):.1f} units without, {rms(d_bs.mean(-1), d_bs[..., keep4].mean(-1)):.1f} with suppression")
fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.3))
show_slice(axes[0], d_rand[..., 2], K, "pair 2, no suppression", kind="diff", vmin=-1500, vmax=1500)
show_slice(axes[1], d_bs[..., 2], K, "pair 2, background suppression", kind="diff", vmin=-1500, vmax=1500)
show_slice(axes[2], d_bs[..., 2], K, "the same, ×5", kind="diff", vmin=-300, vmax=300)
fig.tight_layout()
```

The same jump on the same volume gives pair 2 a deviation of 440 units without suppression
and 145 with it, against a noise-only deviation of 66 to 69. The gray matter control signal
fell eightfold, from 6160 to 790, and the rim by three: the pulses are timed to null gray
and white matter and leave the cerebrospinal fluid, with its long $T_1$, far from zero, so
the ventricle and sulcus edges keep a rim (visible in the ×5 panel) while the brain's outer
edge loses most of its own. What the four corrupted pairs add to the mean difference image
is 36 units RMS without suppression and 12 with it, about the mean's noise floor.
Suppression does not correct motion; it makes the artifact small enough that registration
and rejection have little left to do, at the price of the factor 0.81 on the perfusion
signal and of an M0 scan from elsewhere.

## What this implies for acquisition

- **Background suppression is the single most effective setting.** It reduces the motion
  artifact in proportion to the static signal it removes, one reason the ASL white paper
  recommends it {cite:p}`alsop2015`; physiological fluctuations shrink with it too.
- **Keep the pair together.** Control and label as close in time as the sequence allows, so
  that a slow drift affects both nearly equally, as the drift run shows.
- **Acquire enough pairs to afford losing some.** Rejection cost four of thirty pairs here
  and a few percent of noise; a protocol with ten pairs cannot spare four.
- **Record and report the motion.** The pose estimates are the quality metric; framewise
  displacement {cite:p}`power2012` summarizes them, and the rejected pairs belong beside it.
- **Padding, instruction, and short scans** reduce motion at the source; no correction
  matches a subject who did not move.

## Further reading

Rigid-body registration {cite:p}`jenkinson2002`; outlier rejection for ASL {cite:p}`dolui2017`;
the pipeline order in ASLPrep {cite:p}`adebimpe2022` and ExploreASL {cite:p}`mutsaerts2020`;
framewise displacement {cite:p}`power2012`; the white paper {cite:p}`alsop2015`.
