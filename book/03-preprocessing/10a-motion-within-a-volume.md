---
title: "10.1 Motion inside a volume: what each readout makes of it"
kernelspec:
  name: python3
  display_name: Python 3
---

:::{admonition} Simulated datasets in this chapter
:class: note
- **Built in this page:** the motion trace replayed from its recipe, a rigid registration of the affected volumes, a slice-by-slice registration, and the outlier rule of [Chapter 10](./10-motion.md) ([Appendix B](../appendices/b-data-manifest.md#app-b-package-data)).
- **`motion-readouts`**: one timed head-motion trace (a nod, a step, and a slow drift) replayed on the reference 2D EPI (`epi`), the same with the trace sampled once per volume (`epi-volume`) and with spin history (`epi-spinhistory`), on a four-fold multiband EPI (`mb4`), and on the single-shot 3D GRASE (`grase-1shot`), the four-shot 3D GRASE (`grase`) and the stack of spirals (`spiral`), each 2D or 3D shot acquired at the pose of its own excitation time; with motion-free partners `epi-still` and `mb4-still` ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-motion-readouts)).
- **`readout-3d`**: the motion-free partners of the three 3D runs (`grase-1shot`, `grase-4shot`, `spiral`, same seed) ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-readout-3d)).

Pipeline-tier datasets are simulated offline by aslscan ([Chapter 0.2](../00-frontmatter/the-simulated-datasets.md)).
:::

## Learning goals

After this chapter you can:

- say how long a volume takes to acquire for a 2D EPI, a multiband EPI, a single-shot 3D readout, and a segmented 3D GRASE or stack of spirals, and therefore what "motion within a volume" means for each
- recognize the four signatures: staggered slices and dark or bright slice bands in 2D, groups of slices in multiband, replicas of the brain along the phase-encode axis in a segmented GRASE, and rings and smears outside the brain in a stack of spirals
- explain why no rigid transform undoes any of them, what a slice-wise registration recovers in 2D, and what is left to rejection
- read a timed motion trace against the shot times of a sequence and predict which volumes it will damage

```{code-cell} python
:tags: [hide-cell]
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, optimize

from aslbook import data, motion, phantom, plotting, quant, traces
from aslbook.motion import deviation, move_volume, reject_pairs, register_volume, sample_points
from aslbook.plotting import INK, PALETTE, set_style, show_slice

set_style()
K = phantom.DISPLAY_SLICE
```

## The physics

[Chapter 10](./10-motion.md) moved whole volumes: every slice of a moved volume had the same
pose, and the only question was whether the control and the label of a pair agreed. That
is what motion looks like when the head moves *between* volumes. But a volume is not an
instant. The reference 2D EPI excites its twenty slices one after another, 40 ms apart, so
a volume is spread over 0.76 s; a multiband EPI excites several slices at once and reads
the stack in a few shots, 0.2 s here; a single-shot 3D readout excites the whole slab once
and reads it in one echo train of a few hundred milliseconds; and a segmented 3D readout
excites the slab once per shot, each shot with its own labeling and one repetition time
apart, so a four-shot GRASE volume takes 18 s and an eight-interleaf stack of spirals 36 s
([Chapter 7](../02-labeling/07-acquisition-parameters.md)). A head that moves during that
time does not move the volume. It moves parts of the volume relative to one another, and
what the parts are depends on the readout.

- **2D, one slice per shot.** Each slice is a rigid picture of the head at its own time,
  so a movement during the readout gives a stack of slices that do not fit together, and
  the displacement between neighbouring slices is the head's velocity times 40 ms. A
  through-plane movement adds a second effect: tissue carried into a slice that has not
  yet been excited is excited twice in one repetition and comes out dark, and tissue
  carried out of a slice before its turn is skipped and comes out bright in the next
  volume, which is spin history {cite:p}`friston1996`.
- **Multiband, several slices per shot.** The shots are fewer and the readout shorter, so
  less motion fits into a volume, but the slices that share a shot share a pose, and a
  stagger shows as groups of slices rather than single ones {cite:p}`feinberg2013`.
- **3D, single shot.** One excitation, one pose for the whole volume. Motion between
  volumes is the only motion there is, and [Chapter 10](./10-motion.md) applies as it
  stands. (Motion *during* the echo train blurs and ghosts the volume, which the
  simulator does not model.)
- **3D, segmented.** Each shot reads part of k-space, every fourth phase-encode line or one
  spiral interleaf, at its own pose. The volume is assembled from parts of k-space that
  describe the head in different places, and the reconstruction turns the disagreement
  into replicas of the brain along the segmented axis for a Cartesian GRASE
  {cite:p}`wood1985` and into rings and smears for a spiral, spread over the whole volume.
  A movement that would have cost one pair in 2D costs the whole volume, but a movement
  between two volumes costs nothing, as before.

Every one of these is invisible to a rigid registration of whole volumes, which has six
parameters for a volume that no longer has six degrees of freedom. The rest of the chapter
makes the same head movement happen during each readout and looks at what it leaves.

## The aslscan setting that produces it

The `motion` overlay's `trajectory` can be a *timed* trace: a TSV with a `time` column in
seconds from the series' first acquisition and the six pose parameters, sampled as fast as
one likes, the format a motion camera or a navigator would export. Each 2D shot (the slices
excited together) and each 3D segment then takes the trace's pose at its own excitation
time (`within_volume_sampling = "shot"`, the default for a timed trace), and the simulator
writes every shot's time and pose to `desc-motionShots_gt.tsv`. With
`within_volume_sampling = "volume"` every shot of a volume takes the pose at the volume's
first shot instead, which is what a simulator with one pose per volume renders, and
`spin_history = true` steps every phantom voxel's longitudinal magnetization through the
slice excitations at their shots' poses. The pipeline writes the trace from a recipe in
`datasets.yaml` ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-motion-readouts)), the
same nods, steps and drifts aslscan's own `tools/motion_trace.py` composes.

Three approximations to keep in mind. A shot's echo train is read at one pose, so motion
during a train is not modeled; the slices or segments are resampled from finished images,
as in Chapter 10, so a moved slice is the same anatomy in a new place; and the labeling
plane does not move with the head.

```{code-cell} python
:tags: [hide-input]
ds = data.load_dataset("motion-readouts")
r3 = data.load_dataset("readout-3d")
moved = {name: ds.run(name) for name in ("epi", "epi-volume", "epi-spinhistory", "mb4", "grase-1shot", "grase", "spiral")}
still = {"epi": ds.run("epi-still"), "epi-volume": ds.run("epi-still"), "epi-spinhistory": ds.run("epi-still"), "mb4": ds.run("mb4-still"),
         "grase-1shot": r3.run("grase-1shot"), "grase": r3.run("grase-4shot"), "spiral": r3.run("spiral")}
recipe = next(r for r in ds.provenance()["runs"] if r["name"] == "epi")["overlay"]["motion"]["trace"]
trace = traces.synthesize(recipe)
shots = {name: moved[name].motion_shots() for name in ("epi", "mb4", "grase-1shot", "grase", "spiral")}
print(traces.describe(recipe))
for name, sh in shots.items():
    per_vol = sh.groupby("volume").size().iloc[0]
    print(f"{name:>12}: {sh['volume'].nunique()} volumes of {per_vol} shot(s), a volume lasts {sh[sh.volume == 0]['time'].max() - sh[sh.volume == 0]['time'].min():.2f} s "
          f"from its first to its last excitation and the next begins {sh[sh.volume == 1]['time'].min() - sh[sh.volume == 0]['time'].min():.1f} s after it")

fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), gridspec_kw={"width_ratios": [1.6, 1, 1]})
t, tz, ty, tx, rx, rz = trace[:, 0], trace[:, 3], trace[:, 2], trace[:, 1], np.degrees(trace[:, 4]), np.degrees(trace[:, 6])
for ax, (lo, hi) in zip(axes, [(0, 140), (34.8, 36.2), (88.5, 95)]):
    sel = (t >= lo) & (t <= hi)
    ax.plot(t[sel], tx[sel], color=PALETTE[0], lw=1.3, label="x (mm)")
    ax.plot(t[sel], ty[sel], color=PALETTE[1], lw=1.3, label="y (mm)")
    ax.plot(t[sel], tz[sel], color=PALETTE[2], lw=1.3, label="z (mm)")
    ax.plot(t[sel], rx[sel], color=PALETTE[3], lw=1.3, ls="--", label="pitch (deg)")
    ax.plot(t[sel], rz[sel], color=PALETTE[4], lw=1.3, ls="--", label="yaw (deg)")
    for j, (name, sh) in enumerate(shots.items()):
        s = sh[(sh.time >= lo) & (sh.time <= hi)]
        ax.plot(s["time"], np.full(len(s), -0.6 - 0.25 * j), "|", color=INK["secondary"], ms=5)
        if ax is not axes[0]:
            ax.text(lo + 0.01 * (hi - lo), -0.6 - 0.25 * j, name, fontsize=6, va="center", ha="left")
    ax.set(xlabel="time from the first labeling (s)", xlim=(lo, hi), ylim=(-2.0, 2.2))
axes[0].set(ylabel="mm, degrees", title="the trace: nod at 35 s, step at 90 s, drift")
axes[1].set(title="the nod against each readout's shots")
axes[2].set(title="the step")
axes[0].legend(ncol=3, fontsize=6, loc="upper left")
fig.tight_layout()
```

The recipe is printed first and plotted in the left panel: a nod, 1 mm anterior and 1.5 mm
up with 2° of pitch, that begins at 35.05 s and is over 0.6 s later; a step at 90.3 s, 1.5 mm
to the left with 1° of yaw, that stays; and a slow ramp of 1 mm and 0.7° over the first
270 s. The nod is timed so that it begins 0.05 s before the seventh repetition's readout,
and the step so that it falls between the twentieth repetition's labeling and its readout.
The ticks under the trace are the shot times of each sequence, and they are the whole
point of the dataset: the same movement is read by the 2D EPI as twenty slices spread over
0.76 s, by the multiband EPI as five shots over 0.2 s, by the single-shot GRASE as one
instant, and by the segmented readouts as four or eight excitations 4.5 s apart, so that a
four-shot GRASE volume lasts 13.5 s from its first excitation to its last and a spiral
volume 31.5 s. The middle panel shows what each sees of the nod: the 2D slices sample its
whole shape, the multiband shots its rise, and each 3D readout catches it with exactly one
excitation, at a quarter of its height, the last shot of GRASE volume 1 and the last
interleaf of spiral volume 0. The right panel shows the step falling between two 2D
volumes, between two GRASE volumes, and between the fourth and fifth interleaves of spiral
volume 2.

## See it: the same nod in a 2D volume

The run `epi` is the reference 2D protocol with the trace, `epi-still` the same series
without it (same seed, so the same noise in every voxel), and `epi-volume` the trace
sampled once per volume at the volume's first shot, which is what a simulator with one
pose per volume, like the one [Chapter 10](./10-motion.md) used, renders of the same
movement.

```{code-cell} python
:tags: [hide-input]
mag = {name: r.mag().astype(float) for name, r in moved.items()}
ref = {name: r.mag().astype(float) for name, r in still.items()}
ctx = {name: r.context() for name, r in moved.items()}
masks = {name: still[name].mask() for name in moved}
rms = lambda a, b, m: np.sqrt(np.mean((a - b)[m] ** 2))
V2 = 7                                                            # the 2D volume whose readout the nod falls in
m2, s2, msk = masks["epi"], None, masks["epi"]
sh7 = shots["epi"][shots["epi"].volume == V2]
print(f"epi volume {V2} ({ctx['epi'][V2]}): the readout runs from {sh7.time.min():.2f} to {sh7.time.max():.2f} s; the pose of its slices:")
for _, row in sh7.iloc[[0, 1, 2, 3, 6, 10, 11, 12, 13, 14, 19]].iterrows():
    print(f"  slice {int(row.shot):>2} at {row.time:.2f} s: y {row.trans_y:+.2f} mm, z {row.trans_z:+.2f} mm, pitch {np.degrees(row.rot_x):+.2f} deg")
per_slice = {name: [rms(mag[name][:, :, k, V2], ref[name][:, :, k, V2], msk[:, :, k]) for k in range(20)] for name in ("epi", "epi-volume")}
print(f"RMS change of volume {V2} over the brain: epi {rms(mag['epi'][..., V2], ref['epi'][..., V2], msk):.0f} image units, "
      f"epi-volume (the trace sampled once per volume) {rms(mag['epi-volume'][..., V2], ref['epi-volume'][..., V2], msk):.0f}; a still volume differs from its partner by {rms(mag['epi'][..., 0], ref['epi'][..., 0], msk):.0f}")
X = msk.shape[0] // 2
sag = lambda vol: vol[X, :, :].T[::-1]
fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), gridspec_kw={"width_ratios": [1, 1, 1.3, 1.5]})
show_slice(axes[0], mag["epi"][..., V2] - ref["epi"][..., V2], K, f"epi, volume {V2}: moved − still", kind="diff", vmin=-2000, vmax=2000)
show_slice(axes[1], mag["epi-volume"][..., V2] - ref["epi"][..., V2], K, "epi-volume: the same, one pose", kind="diff", vmin=-2000, vmax=2000)
axes[2].imshow(sag(mag["epi"][..., V2] - ref["epi"][..., V2]), cmap="RdBu_r", vmin=-2000, vmax=2000, aspect=5 / 3.5)
axes[2].set(title="epi: moved − still, sagittal", xticks=[], yticks=[])
axes[3].bar(np.arange(20) - 0.2, per_slice["epi"], 0.4, color=PALETTE[0], label="epi (a pose per slice)")
axes[3].bar(np.arange(20) + 0.2, per_slice["epi-volume"], 0.4, color=PALETTE[1], label="epi-volume (one pose)")
ax2 = axes[3].twinx()
ax2.plot(sh7.shot, sh7.trans_z, "k.-", lw=1, ms=3, label="z of the slice's shot (mm)")
ax2.set(ylabel="mm", ylim=(0, 2.5))
axes[3].set(xlabel="slice", ylabel="RMS change (image units)", title="change per slice")
axes[3].legend(fontsize=6, loc="upper right"); ax2.legend(fontsize=6, loc="center right")
fig.tight_layout()
```

Volume 7, a label, is read from 35.10 to 35.86 s. Its first slice catches the nod a quarter
of the way up (0.5 mm up, 0.6° of pitch), the next two its rise, slices 3 to 10 its full
height (1 mm anterior, 1.63 mm up, 2.1°), slices 11 to 13 its fall, and the last six the
head back where the drift had left it. The volume differs from its still twin by 410 image
units RMS over the brain, a still volume by 4. Rendered with one pose per volume
(`epi-volume`, the pose of the first slice), the same movement changes the volume by 152:
a per-volume model sees a quarter of the nod and spreads it over every slice, and it
misses more than half of the damage. The axial difference images look alike at first, a
rim where the head moved, but the sagittal view and the per-slice bars show the stagger:
the change is largest in slices 1 and 2, around 760 units, where the head had risen more
than a millimeter in two slice intervals and the lowest slices of the slab, at the temporal
lobes, change content fastest with height; it stays near 450 through the plateau, and it
falls to the still level, about 30, once the nod is over. The rendering with one pose has
nothing of this shape, which is why the per-volume model of Chapter 10 could not have
shown it.

## Correction step by step: a rigid pose does not fit a staggered volume

A pipeline sees the staggered volume as one volume and registers it as one. The cell fits
the best rigid pose to volume 7 against its still twin, which is the most any volume
registration can do, and then does what a slice-to-volume method does: registers each
slice on its own, in the plane, with three parameters.

```{code-cell} python
:tags: [hide-input]
inner = msk.copy(); inner[:, :, [0, -1]] = False
pts = sample_points(inner & ~np.isin(np.arange(20)[None, None, :] * np.ones(msk.shape, int), [1, 18]), 6000, seed=0)
smooth = lambda x: ndimage.gaussian_filter(x, (1.5, 1.5, 1.0))
fits = {}
for name in ("epi", "epi-volume"):
    est = register_volume(smooth(mag[name][..., V2]), smooth(ref[name][..., V2]), pts)
    fits[name] = (est, move_volume(mag[name][..., V2], est[:3], est[3:], undo=True))
    floor = move_volume(move_volume(ref[name][..., V2], est[:3], est[3:]), est[:3], est[3:], undo=True)      # the interpolation floor: a still volume moved and moved back
    print(f"{name:>10}: best rigid pose y {est[1]:+.2f} z {est[2]:+.2f} mm, pitch {est[3]:+.2f} deg; residual against the still volume {rms(fits[name][1], ref[name][..., V2], inner):.0f} "
          f"(before {rms(mag[name][..., V2], ref[name][..., V2], inner):.0f}; interpolation alone leaves {rms(floor, ref[name][..., V2], inner):.0f})")

def register_slice(moving, target, k):
    """In-plane rigid registration of one slice: the shift (tx, ty in mm) and rotation
    (degrees) that moved it, found by undoing candidates until the slice matches the target."""
    tv = target[:, :, k]
    c = (np.array(tv.shape) - 1) / 2
    smooth2 = lambda x: ndimage.gaussian_filter(x, 1.5)
    def undo(p):
        th = np.radians(p[2]); R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        return ndimage.affine_transform(moving[:, :, k], R, offset=c - R @ c + np.array(p[:2]) / 3.5, order=1, mode="nearest")
    cost = lambda p: np.mean((smooth2(undo(p)) - smooth2(tv))[msk[:, :, k]] ** 2)
    p = optimize.minimize(cost, np.zeros(3), method="Powell", options={"xtol": 1e-3, "ftol": 1e-6}).x
    return p, undo(p)

slicewise = np.zeros_like(mag["epi"][..., V2]); est_slices = []
for k in range(20):
    p, slicewise[:, :, k] = register_slice(mag["epi"][..., V2], ref["epi"][..., V2], k)
    est_slices.append(p)
est_slices = np.array(est_slices)
# the in-plane displacement each slice's shot pose gives the slice's centre (the pitch moves a slice in y by its height times the angle)
centre = (np.array(msk.shape) - 1) / 2 * np.array([3.5, 3.5, 5.0])
true_inplane = []
for _, row in sh7.iterrows():
    p0 = np.array([centre[0], centre[1], row.shot * 5.0])
    p1 = motion.rotation(np.degrees(row.rot_x), 0, 0) @ (p0 - centre) + centre + np.array([row.trans_x, row.trans_y, row.trans_z])
    true_inplane.append(p1[:2] - p0[:2])
true_inplane = np.array(true_inplane)
res_rigid = [rms(fits["epi"][1][:, :, k], ref["epi"][:, :, k, V2], msk[:, :, k]) for k in range(20)]
res_slice = [rms(slicewise[:, :, k], ref["epi"][:, :, k, V2], msk[:, :, k]) for k in range(20)]
flat = sh7.trans_z.values < 0.2                                   # the slices acquired after the nod, moved in the plane only by the residual pitch
print(f"slice-by-slice (in-plane) registration: residual against the still volume {rms(slicewise, ref['epi'][..., V2], inner):.0f} image units "
      f"(one rigid pose left {rms(fits['epi'][1], ref['epi'][..., V2], inner):.0f}); in-plane y shift recovered to {np.abs(est_slices[flat, 1] - true_inplane[flat, 1]).max():.2f} mm on the {flat.sum()} slices the nod had left, "
      f"off by up to {np.abs(est_slices[~flat, 1] - true_inplane[~flat, 1]).max():.2f} mm on the {(~flat).sum()} slices it displaced through the plane")
print("residual per slice, one rigid pose -> slice-by-slice: " + ", ".join(f"{k}: {res_rigid[k]:.0f} -> {res_slice[k]:.0f}" for k in (0, 1, 2, 3, 6, 10, 12, 14, 19)))
fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
show_slice(axes[0], fits["epi"][1] - ref["epi"][..., V2], K, "epi: after the best rigid pose", kind="diff", vmin=-2000, vmax=2000)
show_slice(axes[1], slicewise - ref["epi"][..., V2], K, "epi: after slice-by-slice registration", kind="diff", vmin=-2000, vmax=2000)
show_slice(axes[2], fits["epi-volume"][1] - ref["epi"][..., V2], K, "epi-volume: after the best rigid pose", kind="diff", vmin=-2000, vmax=2000)
axes[3].bar(np.arange(20) - 0.2, res_rigid, 0.4, color=PALETTE[1], label="one rigid pose")
axes[3].bar(np.arange(20) + 0.2, res_slice, 0.4, color=PALETTE[0], label="slice-by-slice, in plane")
ax2 = axes[3].twinx()
ax2.plot(range(20), sh7.trans_z.values, "k.-", lw=1, ms=3, label="through-plane shift of the slice (mm)")
ax2.set(ylabel="mm", ylim=(0, 2.5))
axes[3].set(xlabel="slice", ylabel="residual (image units)", title="what each correction leaves, per slice")
axes[3].legend(fontsize=6, loc="upper right"); ax2.legend(fontsize=6, loc="center right")
fig.tight_layout()
```

The best single pose is a compromise, 0.65 mm anterior, 0.4 mm up and 1.3° of pitch, between
the slices that had the full nod and those that had none, and undoing it brings the volume
from 420 to 325 units from its twin. That number has a floor: moving a still volume by the
same pose and back leaves 269, because interpolation blurs edges on an image whose
perfusion signal lives in the edges, so most of the 325 is not the stagger. But the first
panel shows that the rim is still there, now of both signs, and the rendering with one
pose (third panel), where there was a pose to find, comes out clean. The slice-by-slice
registration does better where it can and worse where it cannot. On the six slices the
nod had already left, it recovers their in-plane shift to 0.10 mm and takes their residual
from about 300 to the floor (slice 14: 326 to 22). On the slices the nod displaced through
the plane, its estimates are off by up to 2.61 mm and the residual of slices 1 and 2 rises
from about 550 to about 780: a slice that moved 1.5 mm along the slice direction contains
different tissue, and an in-plane search that is told to match it finds the wrong answer
with confidence. The in-plane part of a stagger is recoverable slice by slice; the
through-plane part is not recoverable from the slice at all, because the information is
in the slices that were not acquired. A full slice-to-volume registration estimates all
six parameters of each slice against a volume and does better than this page's three, but
the missing tissue stays missing.

## See it: spin history

The nod is through-plane at 10 mm/s on its rise, and the slices are acquired bottom to top.
Tissue excited by slice 1 and carried up into slice 2 before slice 2 fires is excited
twice, 40 ms apart, and has nothing left to give. The `epi-spinhistory` run is the `epi`
run with this bookkeeping switched on, so the difference between the two series is the
spin-history term alone.

```{code-cell} python
:tags: [hide-input]
gm = still["epi"].fractions()["gm"] > 0.7
sh_corr = mag["epi-spinhistory"] - mag["epi"]                      # the two runs differ by the spin-history term alone
gm_mean = lambda vol, k: vol[:, :, k][gm[:, :, k]].mean()
vz = np.gradient(sh7.trans_z.values, sh7.time.values)             # through-plane velocity at each slice's excitation
rows = [(k, gm_mean(sh_corr[..., V2], k), gm_mean(sh_corr[..., V2 + 1], k), vz[k]) for k in range(20)]
print(f"spin history, GM mean of (epi-spinhistory − epi) per slice, in image units (a GM slice reads about {gm_mean(ref['epi'][..., V2], 5):.0f}):")
print(f"  {'slice':>5} {'volume 7':>9} {'volume 8':>9} {'z velocity (mm/s)':>18}")
for k, a, b, v in rows:
    if k in (0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 19):
        print(f"  {k:>5} {a:>9.0f} {b:>9.0f} {v:>18.1f}")
print(f"largest change in any slice: {min(r[1] for r in rows):.0f} image units ({100 * min(r[1] for r in rows) / gm_mean(ref['epi'][..., V2], 1):.1f} % of the GM signal)")
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), gridspec_kw={"width_ratios": [1.3, 1.3, 1.4]})
axes[0].imshow(sag(sh_corr[..., V2]), cmap="RdBu_r", vmin=-600, vmax=600, aspect=5 / 3.5); axes[0].set(title=f"spin history, volume {V2} (sagittal)", xticks=[], yticks=[])
axes[1].imshow(sag(sh_corr[..., V2 + 1]), cmap="RdBu_r", vmin=-600, vmax=600, aspect=5 / 3.5); axes[1].set(title=f"volume {V2 + 1}, the next", xticks=[], yticks=[])
axes[2].bar(np.arange(20) - 0.2, [r[1] for r in rows], 0.4, color=PALETTE[0], label=f"volume {V2}")
axes[2].bar(np.arange(20) + 0.2, [r[2] for r in rows], 0.4, color=PALETTE[1], label=f"volume {V2 + 1}")
axes[2].set(xlabel="slice", ylabel="GM signal change (image units)", title="per slice")
axes[2].legend(fontsize=7)
fig.tight_layout()
```

The effect is a band. Slices 1, 2 and 3, excited while the head rose at 12, 7 and 2 mm/s,
lose 537, 357 and 158 image units of their gray matter signal, 8.8 % at most, which is
thirty times the perfusion signal of 28 and the dark stripes of the sagittal panel. Slices
11 to 13, during the fall, lose about 1 %, because the pitch's through-plane component
still runs ahead of the slice order in part of the brain. In the next volume (orange bars)
slices 10 to 13 read 12 to 23 units *brighter*: the tissue that the fall had carried out of
their way was skipped for one repetition and has recovered further than the steady state.
These are the sizes aslscan's own expectation table gives, a dip of several percent in a
slice or two at an abrupt through-plane movement and nothing at all from a slow drift,
and they explain the stripes a reader has seen in moved 2D volumes: they are not a
reconstruction artifact but the longitudinal magnetization keeping score. No registration
touches them, since nothing is out of place; they are an intensity change in a slice, and
the only remedies are to drop the volume, to model the dip, or to acquire with
background suppression, under which the tissue signal that carries the history is small.

## See it: multiband, five poses for twenty slices

The `mb4` run excites four slices at once, spread evenly through the stack (slices 0, 5,
10 and 15 together, then 2, 7, 12 and 17, and so on), in five shots 40 ms apart
{cite:p}`feinberg2013`; `mb4-still` is its twin without the trace.

```{code-cell} python
:tags: [hide-input]
mb = moved["mb4"].sidecar()
sh_mb = shots["mb4"][shots["mb4"].volume == V2]
print(f"multiband factor {mb['MultibandAccelerationFactor']}: {len(sh_mb)} shots per volume, the readout lasts {sh_mb.time.max() - sh_mb.time.min() + 0.04:.2f} s")
for _, row in sh_mb.iterrows():
    print(f"  shot {int(row.shot)} at {row.time:.2f} s excites slices {row.slices}: y {row.trans_y:+.2f} mm, z {row.trans_z:+.2f} mm, pitch {np.degrees(row.rot_x):+.2f} deg")
per_slice_mb = [rms(mag["mb4"][:, :, k, V2], ref["mb4"][:, :, k, V2], masks["mb4"][:, :, k]) for k in range(20)]
shot_of = {}
for _, row in sh_mb.iterrows():
    for s in str(row.slices).split(","):
        shot_of[int(s)] = int(row.shot)
print(f"RMS change of volume {V2} over the brain: mb4 {rms(mag['mb4'][..., V2], ref['mb4'][..., V2], masks['mb4']):.0f} image units (epi {rms(mag['epi'][..., V2], ref['epi'][..., V2], msk):.0f})")
cor = lambda vol: vol[:, msk.shape[1] // 2, :].T[::-1]
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), gridspec_kw={"width_ratios": [1.3, 1.3, 1.6]})
axes[0].imshow(cor(mag["epi"][..., V2] - ref["epi"][..., V2]), cmap="RdBu_r", vmin=-2000, vmax=2000, aspect=5 / 3.5); axes[0].set(title="epi: moved − still, coronal", xticks=[], yticks=[])
axes[1].imshow(cor(mag["mb4"][..., V2] - ref["mb4"][..., V2]), cmap="RdBu_r", vmin=-2000, vmax=2000, aspect=5 / 3.5); axes[1].set(title="mb4: moved − still, coronal", xticks=[], yticks=[])
axes[2].bar(range(20), per_slice_mb, color=[PALETTE[shot_of[k]] for k in range(20)])
axes[2].set(xlabel="slice (colour: the shot that excited it)", ylabel="RMS change (image units)", title="mb4: change per slice")
fig.tight_layout()
```

The readout now lasts 0.20 s and the volume has five poses instead of twenty, the rise of
the nod and its first 0.04 s of plateau. The volume as a whole is changed about as much as
the single-band one, 416 units against 410, because the nod was timed to begin at the
readout; what differs is the shape. In the coronal difference image every fifth slice
shares a shot and therefore a displacement, so the stagger is a comb rather than a ramp,
and the per-slice bars repeat with the period of the slice groups: slice 1, excited by
shot 3 at the full nod, changes by 1100 units, slice 0, excited by shot 0 at a quarter of
it, by 200, and the pattern recurs at slices 5 and 6, 10 and 11, 15 and 16. Two things
follow for a pipeline. A slice-wise correction has only five unknown poses rather than
twenty, which is a better-posed problem, and a multiband volume that is staggered is
staggered everywhere in the stack at once, so no range of slices is spared. The shorter
readout is the real gain: a nod of 0.6 s that fills most of a 0.76 s readout fills all of a
0.2 s one, but the chance that a given movement falls inside the readout at all shrinks in
proportion.

## See it: a segmented 3D GRASE

The `grase` run is the four-shot GRASE of [Chapter 7](../02-labeling/07-acquisition-parameters.md)
with the trace; its twin is `readout-3d`'s `grase-4shot`, with the same seed. Each shot
reads every fourth phase-encode line of every partition, so the four shots are four
interleaved combs of k-space, acquired 4.5 s apart.

```{code-cell} python
:tags: [hide-input]
V3 = 1                                                            # the GRASE volume whose last shot catches the nod
sh_g = shots["grase"][shots["grase"].volume == V3]
print(f"grase volume {V3} ({ctx['grase'][V3]}): shots at " + ", ".join(f"{t:.1f}" for t in sh_g.time) + " s; pose of each shot: " +
      "; ".join(f"shot {int(r.shot)} y {r.trans_y:+.2f} z {r.trans_z:+.2f} mm pitch {np.degrees(r.rot_x):+.2f} deg" for _, r in sh_g.iterrows()))
mask3 = masks["grase"]
far = ~ndimage.binary_dilation(mask3, iterations=3)
q = mask3.shape[1] // 4
zone4 = (np.roll(mask3, q, axis=1) | np.roll(mask3, -q, axis=1)) & far
pct = lambda img, z: 100 * np.abs(img[z]).mean() / np.abs(img[mask3]).mean()
per_vol = {name: [rms(mag[name][..., v], ref[name][..., v], masks[name]) for v in range(mag[name].shape[-1])] for name in ("epi", "grase-1shot", "grase", "spiral")}
print(f"RMS change per volume (image units), grase: " + ", ".join(f"{v:.0f}" for v in per_vol["grase"][:8]) + " ...; grase-1shot: " + ", ".join(f"{v:.0f}" for v in per_vol["grase-1shot"][:9]) + " ...")
print(f"signal in the FOV/4 zones outside the brain, as a percentage of the brain mean: still volume {V3} {pct(ref['grase'][..., V3], zone4):.1f} %, "
      f"moved volume {V3} {pct(mag['grase'][..., V3], zone4):.1f} %, (moved − still) {pct(mag['grase'][..., V3] - ref['grase'][..., V3], zone4):.0f} %")
fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
show_slice(axes[0], mag["grase"][..., V3] - ref["grase"][..., V3], K, f"grase, volume {V3}: moved − still", kind="diff", vmin=-600, vmax=600)
axes[0].contour(plotting.take_slice(zone4.astype(float), K), levels=[0.5], colors=[PALETTE[1]], linewidths=0.6)
show_slice(axes[1], mag["grase"][..., 5] - ref["grase"][..., 5], K, "grase, volume 5: after the step", kind="diff", vmin=-2000, vmax=2000)
show_slice(axes[2], mag["grase-1shot"][..., V2] - ref["grase-1shot"][..., V2], K, f"grase-1shot, volume {V2}: the nod", kind="diff", vmin=-600, vmax=600)
axes[3].plot(per_vol["epi"], ".-", color=PALETTE[0], lw=1, ms=4, label="epi")
axes[3].plot(per_vol["grase-1shot"], ".-", color=PALETTE[2], lw=1, ms=4, label="grase-1shot")
axes[3].plot(per_vol["grase"], ".-", color=PALETTE[1], lw=1, ms=4, label="grase (4 shots)")
axes[3].plot(per_vol["spiral"], ".-", color=PALETTE[3], lw=1, ms=4, label="spiral (8 shots)")
axes[3].set(xlabel="volume", ylabel="RMS change over the brain", title="the same trace, volume by volume", yscale="log")
axes[3].legend(fontsize=7)
fig.tight_layout()
```

Volume 1's first three shots see the drift and nothing else, a tenth of a millimeter; its
fourth, at 35.1 s, catches the nod a quarter of the way up: 0.25 mm anterior, 0.5 mm up,
0.6° of pitch. A quarter of a small nod in a quarter of k-space is enough. The first panel
is the moved volume minus its twin, and it is three copies of the brain's edges, above and
below the brain at a quarter of the field of view and, fainter, at half, the replicas that
a disagreement between every fourth line produces {cite:p}`wood1985`; the orange contour
marks where the brain shifted by a quarter field of view would lie, and the difference
image has 74 % as much signal there as inside the brain. The volume changed by 45 units
RMS, a tenth of what the 2D slices suffered from the same nod, but it changed everywhere.
The step, which the 2D run took between two volumes, falls here before volume 5's first
shot, so volume 5 is simply displaced by 1.5 mm and 1°: a rim of 2000 units (second
panel), the whole-volume motion of Chapter 10, which a registration undoes. The
single-shot run (third panel) has the easiest time of all: its volume 7 is one excitation
at a quarter of the nod, displaced rigidly by 77 units' worth and otherwise intact, and
its curve in the fourth panel is the 2D run's divided by five. The fourth panel is the
dataset in one picture: the same trace, volume by volume, is a spike of 410 at volume 7
and a step to 380 at volume 20 for the 2D EPI, a spike of 77 and a step to 226 for the
single-shot GRASE, a bump of 45 at volume 1 and a step to 292 at volume 5 for the four-shot
GRASE, and for the spiral a rise through volumes 2 and 3 that the next section explains.
What each readout calls a damaged volume is a different set of volumes, and the drift
itself, invisible to the 2D run, has become a slow within-volume inconsistency in the
segmented ones: between volume 1 and volume 4 the GRASE run's change grows from 29 to 52
units with no event at all, because its four shots are 13.5 s apart and the head is 0.15 mm
higher at the last than at the first.

## See it: a stack of spirals

The `spiral` run reads each volume as eight interleaves, one per shot, 36 s per volume;
its twin is `readout-3d`'s `spiral`. A spiral has no phase-encode axis, so a disagreement
between interleaves is not a set of replicas along one direction but a swirl spread over
the plane, and the spiral's reconstruction already leaves a faint haze outside the brain
([Chapter 2](../01-mri-physics/02-epi-and-reconstruction.md)).

```{code-cell} python
:tags: [hide-input]
sh_s = shots["spiral"][shots["spiral"].volume == 2]
print("spiral volume 2 (control): interleaves at " + ", ".join(f"{t:.1f}" for t in sh_s.time) + " s; x of each: " + ", ".join(f"{x:+.2f}" for x in sh_s.trans_x) + " mm")
masks_s = masks["spiral"]
outside = ~ndimage.binary_dilation(masks_s, iterations=2)
out_pct = lambda img: 100 * np.abs(img[outside]).mean() / np.abs(img[masks_s]).mean()
print(f"mean |signal| outside the brain as a percentage of the brain mean: still volume 2 {out_pct(ref['spiral'][..., 2]):.1f} %, "
      f"moved volume 0 (nod at the last interleaf) {out_pct(mag['spiral'][..., 0]):.1f} %, volume 2 (step between interleaves 3 and 4) {out_pct(mag['spiral'][..., 2]):.1f} %, volume 3 (all interleaves after the step) {out_pct(mag['spiral'][..., 3]):.1f} %")
print(f"RMS change per volume, spiral: " + ", ".join(f"{v:.0f}" for v in per_vol["spiral"][:6]) + " ...")
fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
show_slice(axes[0], mag["spiral"][..., 0] - ref["spiral"][..., 0], K, "spiral, volume 0: nod at the last interleaf", kind="diff", vmin=-600, vmax=600)
show_slice(axes[1], mag["spiral"][..., 2] - ref["spiral"][..., 2], K, "volume 2: the step between interleaves", kind="diff", vmin=-2000, vmax=2000)
show_slice(axes[2], ref["spiral"][..., 2], K, "still volume 2, window 0–1000", vmin=0, vmax=1000)
show_slice(axes[3], mag["spiral"][..., 2], K, "moved volume 2, window 0–1000", vmin=0, vmax=1000)
fig.tight_layout()
```

Volume 2 is the one the step falls into: its first four interleaves were read with the
head at +0.15 mm, its last four at −1.3 mm, so half of its k-space describes a head 1.5 mm
to the left of the other half. The difference image (second panel) is a rim inside the
brain, as for any displacement, with rings around it, and the two windowed panels show
what the rings are: in the still volume the space outside the brain holds the spiral's
haze at 1.2 % of the brain's mean signal, and in the moved volume it holds 1.8 %, arcs
concentric with the brain that no 2D or Cartesian 3D readout produces. Volume 0, which
caught the nod on its last interleaf, shows the same arcs faintly (first panel; 27 units
RMS, against 32 for the next, drift-only volume), and volume 3, read entirely after the
step, is displaced rigidly and its outside is back at 1.1 %. Two things about the spiral
matter to a pipeline. The artifact is outside the brain as much as inside it, where a
registration cost that samples the whole field of view will see it and a brain mask will
not, and it is spread smoothly rather than as sharp replicas, so it looks like a change
of the image's background rather than a copy of the anatomy. And a volume lasts 36 s, so
the slow drift of the first 270 s, 0.35 mm from an interleaf to the last of the same
volume, puts a small inconsistency into every volume of the first four minutes.

## Correction step by step: what registration leaves

Each readout's affected volumes are registered to their still twins with the page's
rigid registration, and the residual is compared with what undoing the true per-volume
pose leaves. For a volume that moved as a whole, the two should agree and the residual
should be the interpolation floor; for a volume whose parts moved differently, there is
no pose to find.

```{code-cell} python
:tags: [hide-input]
cases = [("epi", V2, "nod inside the readout"), ("epi", 21, "after the step (between volumes)"), ("mb4", V2, "nod inside the readout"),
         ("grase-1shot", V2, "nod, single shot"), ("grase", V3, "nod at one of four shots"), ("grase", 5, "after the step (between volumes)"),
         ("spiral", 0, "nod at one of eight interleaves"), ("spiral", 2, "step between interleaves"), ("spiral", 3, "after the step (between volumes)")]
def to_deg(tbl):
    return np.column_stack([tbl[["trans_x", "trans_y", "trans_z"]].values, np.degrees(tbl[["rot_x", "rot_y", "rot_z"]].values)])

poses = {name: to_deg(moved[name].motion()) for name in moved}            # the volume's pose: the trace at its first shot
print(f"{'run':>12} {'vol':>4} {'what':>34} {'before':>7} {'registered':>11} {'true pose':>10}   (RMS against the still partner, image units)")
left = {}
for name, v, what in cases:
    m, s, mk = mag[name][..., v], ref[name][..., v], masks[name]
    inn = mk.copy(); inn[:, :, [0, -1]] = False
    p_here = sample_points(inn & ~np.isin(np.arange(mk.shape[2])[None, None, :] * np.ones(mk.shape, int), [1, mk.shape[2] - 2]), 6000, seed=0)
    est = register_volume(smooth(m), smooth(s), p_here)
    back = move_volume(m, est[:3], est[3:], undo=True)
    true_back = move_volume(m, poses[name][v, :3], poses[name][v, 3:], undo=True)
    left[(name, v)] = (rms(m, s, inn), rms(back, s, inn), rms(true_back, s, inn))
    print(f"{name:>12} {v:>4} {what:>34} {left[(name, v)][0]:>7.0f} {left[(name, v)][1]:>11.0f} {left[(name, v)][2]:>10.0f}")
```

The volumes that moved between excitations are the registration's bread and butter: the
2D volume after the step goes from 381 to 224, GRASE volume 5 from 295 to 152 and spiral
volume 3 from 268 to 137, each within a few units of what the true pose achieves (241, 156
and 140), the rest being the interpolation floor. The volumes that moved *during* their
acquisition are a different matter. The staggered 2D volume goes from 420 to 325, and the
multiband one from 426 to 275, better than the true first-shot pose (354 and 336) because
the search found a compromise pose, and no better than that. The GRASE volume with the
nod on one shot stays at 45, and the spiral volume with the nod on one interleaf at 27:
the registration returns the identity, correctly, because the replicas and arcs are not a
displacement of anything. The spiral volume split by the step, whose two halves differ by
1.5 mm, goes from 133 to 111, the compromise again, and undoing either half's pose would
be worse (158 for the first-shot pose). This is the line between Chapter 10 and this one:
a pose per volume corrects what happened between volumes and leaves what happened inside
them exactly as it was.

## Measure it: the mean difference image, per readout

The mean difference image is scored against the unmoved truth (`deltamStatic`, scaled to
image units by the still series' gray matter), in percent of the gray matter ΔM, for
four versions of each run: the still twin, which sets the noise floor of a 15-pair
average; the moved series as acquired; the moved series with the true pose of every
volume undone, the best a volume registration can do; and that series after Chapter 10's
outlier rule has dropped the pairs it finds. The rule's scale is floored at 10 % of the
median deviation: the simulator's thermal noise is the same in every pair to a fraction
of a percent, so without a floor a pair can be flagged for a change of noise texture alone
after resampling, which no real series with physiological fluctuations would show.

```{code-cell} python
:tags: [hide-input]
print(f"{'run':>15} {'GM ΔM':>6} {'still partner':>13} {'uncorrected':>12} {'true poses':>11} {'+ rejection':>12} {'rejected pairs':>16}   (RMS error of the mean ΔM against the truth, % of the GM ΔM)")
summary, dev_tables = {}, {}
for name in ("epi", "epi-volume", "epi-spinhistory", "mb4", "grase-1shot", "grase", "spiral"):
    mk = masks[name]; inn = mk.copy(); inn[:, :, [0, -1]] = False
    gm_here = (still[name].fractions()["gm"] > 0.7) & inn
    lab_rows = [l for _, l in quant.pair_indices(ctx[name])]
    truth = moved[name].truth("deltamStatic")[..., lab_rows].mean(-1)
    d_still = quant.subtract(ref[name], ctx[name])
    truth = truth * d_still.mean(-1)[gm_here].mean() / truth[gm_here].mean()      # M0 units to image units, by the still series' gray matter
    gm_dm = truth[gm_here].mean()
    d_raw = quant.subtract(mag[name], ctx[name])
    d_cor = quant.subtract(motion.apply_poses(mag[name], poses[name]), ctx[name])
    keep, z = reject_pairs(d_cor, inn, mad_floor=0.1)
    dev_tables[name] = (deviation(d_still, inn), deviation(d_raw, inn), deviation(d_cor, inn), z)
    e = lambda d: 100 * rms(d, truth, inn) / gm_dm
    summary[name] = (e(d_still.mean(-1)), e(d_raw.mean(-1)), e(d_cor.mean(-1)), e(d_cor[..., keep].mean(-1)), np.flatnonzero(~keep).tolist())
    print(f"{name:>15} {gm_dm:>6.1f} {summary[name][0]:>12.0f}% {summary[name][1]:>11.0f}% {summary[name][2]:>10.0f}% {summary[name][3]:>11.0f}% {str(summary[name][4]):>16}")
print("deviation of each pair's ΔM from the mean after the true poses (image units), grase: " + ", ".join(f"{v:.0f}" for v in dev_tables["grase"][2]) +
      f"; the still partner's pairs {dev_tables['grase'][0].min():.1f} to {dev_tables['grase'][0].max():.1f}")
print("the same for the spiral: " + ", ".join(f"{v:.0f}" for v in dev_tables["spiral"][2]) + f"; still partner {dev_tables['spiral'][0].min():.1f} to {dev_tables['spiral'][0].max():.1f}")
for name in ("grase", "spiral"):
    z = dev_tables[name][3]
    print(f"{name}: final z-scores of the kept pairs {np.round(z[np.isfinite(z)], 1).tolist()}")
fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
for ax, (name, v) in zip(axes, [("epi", V2), ("mb4", V2), ("grase", V3), ("spiral", 2)]):
    d_cor = quant.subtract(motion.apply_poses(mag[name], poses[name]), ctx[name])
    d_still = quant.subtract(ref[name], ctx[name])
    k = v // 2
    show_slice(ax, d_cor[..., k] - d_still[..., k], K, f"{name}, pair {k} after the true poses: ΔM error", kind="diff", vmin=-600, vmax=600)
fig.tight_layout()
```

Read the table across. A still 15-pair 2D series misses the truth by 53 % of its gray
matter ΔM, the noise floor; the nod and the step raise that to 117 %, undoing every
volume's pose brings it to 97 %, and dropping pair 3, the one with the staggered label,
to 48 %, a little below the floor because the resampled volumes have had their noise
averaged. The rendering with one pose per volume tells a milder story, 66 % uncorrected
and 62 % after the poses: a simulator that moves whole volumes would have halved the
damage, which is the reason this dataset exists. Spin history makes the uncorrected
series worse (133 %) and changes nothing after rejection, because the banded volume is the
rejected one. The multiband run goes from 100 % to 79 % to 41 % by the same route. The
single-shot GRASE needs little: 32 % uncorrected, 30 % after the poses, 23 % after dropping
pair 3, near its floor of 21 %, because its only within-volume motion is none. The
four-shot GRASE goes from 59 % to 35 % with the poses and to 18 % after the rule drops three
pairs: pair 2, whose two volumes sit on either side of the step (deviation 105 against a
still pair's 12), pair 0, whose label carries the nod's replicas (36), and then pair 1
(z = 9.6 once the other two are out of the mean), whose volumes were read while the head
drifted and whose shots therefore disagree by a tenth of a millimeter. The spiral goes
from 30 % to 38 % with the poses, *worse*, because undoing a 1.5 mm, 1° pose resamples
eleven of its fifteen pairs and blurs their edges while the broken pair stays broken, and
then to 16 % after the rule drops four pairs: the split volume (136), the nod's arcs (26),
and pairs 2 and 3, read during the drift, which stand out at z = 11 once the first two are
gone against a still-pair deviation of 6.6. The error images of the rejected pairs are the
sections above in ΔM: a rim of both signs for the staggered 2D and multiband labels,
three ghosts of the brain for the GRASE pair, a rim with arcs around it for the spiral.
Each readout lost a different number of pairs to the same head: one in 2D, three in a
four-shot GRASE, four in a stack of spirals, and the segmented readouts lost them to the
drift as well as to the nod and the step.

## What this implies for acquisition

- **Know how long your volume is.** The window in which motion corrupts a volume rather
  than displaces it is 0.76 s for the reference 2D EPI, 0.2 s with multiband 4, one
  excitation for a single-shot 3D readout, and 13.5 s or 31.5 s for the segmented GRASE and
  spiral. A nod of half a second is a staggered volume in 2D and, if it hits an excitation,
  a ghosted volume in 3D; a slow drift that 2D does not notice is a small inconsistency in
  every segmented volume.
- **Expect to reject, not repair, within-volume motion.** No rigid pose fits a staggered
  or ghosted volume, and slice-wise registration recovers only the in-plane part of a 2D
  stagger. Acquire enough pairs to lose the ones that straddle a movement, and more of
  them with a segmented readout, which loses whole volumes to a single bad shot.
- **Fewer and shorter shots.** Multiband shortens the 2D window; a single-shot 3D readout
  removes it; a segmented readout with fewer shots has fewer chances to be hit and a
  shorter span for the drift to act over.
- **Background suppression scales every one of these artifacts** with the static signal,
  the stagger, the replicas and the arcs alike, as it scaled the rim in Chapter 10; the
  spin-history bands, which live in the tissue signal, shrink with it too.
- **Prospective correction is the answer for segmented 3D.** Tracking the head between
  shots and steering the next one, as PROMO does for 3D PCASL {cite:p}`zun2014` and as
  navigator and camera systems do in general {cite:p}`maclaren2013`, acquires every shot in
  the same frame; nothing done afterwards to a product image can.
- **Record the motion as a trace, not only per volume.** A camera or navigator trace at
  the sequence's own clock, like the one this dataset replays, says which shots a
  movement fell into, and that is what decides whether a volume can be kept.

## Further reading

Spin history and the movement-related signal changes it causes {cite:p}`friston1996`; ghosts
from inconsistent k-space segments {cite:p}`wood1985`; simultaneous multi-slice ASL
{cite:p}`feinberg2013`; the segmented 3D readouts {cite:p}`fernandezseara2005,vidorreta2017`
and the comparison of 2D and 3D sequences {cite:p}`vidorreta2013`; prospective motion
correction in general {cite:p}`maclaren2013` and for 3D PCASL {cite:p}`zun2014`; the
outlier rules of [Chapter 10](./10-motion.md) {cite:p}`tan2009,dolui2017`.
