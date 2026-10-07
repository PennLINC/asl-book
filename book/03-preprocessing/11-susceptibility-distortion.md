---
title: "11. Susceptibility distortion"
kernelspec:
  name: python3
  display_name: Python 3
---

:::{admonition} Simulated datasets in this chapter
:class: note
- **Built in this page:** a one-dimensional pile-up demonstration, and the phantom's synthetic field map box-averaged onto the acquisition grid ([Appendix B](../appendices/b-data-manifest.md#app-b-package-data)).
- **`sdc`**: the reference protocol on the slab with its synthetic field map, acquired with the phase encoding posterior-anterior (`ap`, `PhaseEncodingDirection` `j-`) and reversed (`pa`, `j`), plus the same protocol on the slab without a field map (`undistorted`), all at TE 20 ms, a total readout time of 35 ms, and 15 pairs ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-sdc)).
- **`readout-3d`** (four of its runs): the four-shot 3D GRASE (`grase-4shot`: four interleaved k_y segments of 17 lines, line spacing 0.3 ms, `TotalReadoutTime` 5 ms) and the stack of spirals (`spiral`: eight 5 ms interleaves) on the slab, and the same two on the slab with the field map (`grase-4shot-fieldmap`, `spiral-fieldmap`), 15 pairs each ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-readout-3d)).

Pipeline-tier datasets are simulated offline by aslscan ([Chapter 0.2](../00-frontmatter/the-simulated-datasets.md)).
:::

## Learning goals

After this chapter you can:

- predict, from a field map and the sidecar's `TotalReadoutTime`, where an EPI image is displaced and by how many voxels
- read the sign of the displacement from `PhaseEncodingDirection` and check it on a blip-up/blip-down pair
- undo the distortion when the field is known, including the intensity correction, and explain how the field is estimated from a reversed-polarity pair when it is not
- say what distortion does to a CBF map, why the M0 division hides part of it, and why the hidden part still matters
- explain why a segmented 3D GRASE is displaced a quarter as far as a single-shot readout of the same line spacing, why a stack of spirals blurs instead of displacing, and what each needs from a pipeline

```{code-cell} python
:tags: [hide-cell]
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from scipy import ndimage, optimize

from aslbook import data, grid, phantom, presets, quant
from aslbook.plotting import INK, PALETTE, TISSUE_COLORS, fit_vs_truth, set_style, show_image, show_slice, take_slice

set_style()
K = 2                       # the display slice of this chapter: the field lobes sit in the inferior slices
VOX = np.array(presets.REFERENCE.voxel_mm)


def unwarp(img, disp):
    """Undo a displacement along the phase-encode axis (axis 1). `disp[x, y, z]` is where the
    signal that belongs at row y landed, in voxels relative to y. Each column is resampled at
    y + disp(y) and multiplied by the Jacobian 1 + d disp/dy, which restores the intensity
    that pile-up concentrated and stretching diluted."""
    out = np.zeros_like(img)
    y = np.arange(img.shape[1], dtype=float)
    for i in range(img.shape[0]):
        for k in range(img.shape[2]):
            d = disp[i, :, k]
            out[i, :, k] = np.interp(y + d, y, img[i, :, k]) * (1.0 + np.gradient(d))
    return out


def distort(img, disp):
    """The forward model: the signal at row y lands at y + disp(y), with intensity conserved
    (the inverse map is interpolated and the Jacobian d y_true / d y_landed applied)."""
    out = np.zeros_like(img)
    y = np.arange(img.shape[1], dtype=float)
    for i in range(img.shape[0]):
        for k in range(img.shape[2]):
            landed = y + disp[i, :, k]
            order = np.argsort(landed)
            y_true = np.interp(y, landed[order], y[order], left=np.nan, right=np.nan)
            out[i, :, k] = np.nan_to_num(np.interp(y_true, y, img[i, :, k]) * np.gradient(y_true))
    return out
```

## The physics

The static field is not uniform inside the head. Air in the frontal sinuses and the
mastoids, the dense petrous bone, and brain tissue have different magnetic
susceptibilities, and near the interfaces between them the field deviates from its nominal
value by up to a hundred hertz or more at 3 T. Spins there precess at a shifted frequency
$\Delta f$.

ASL images are read out with echo-planar imaging {cite:p}`mansfield1977`, one whole slice
per excitation ([Chapter 2](../01-mri-physics/02-epi-and-reconstruction.md)). Along the frequency-encode
axis, position is read from the frequency within one line, which takes well under a
millisecond, and a frequency offset moves signal by a small fraction of a voxel. Along the
**phase-encode** axis, position is inferred from how much phase a spin accumulates from one
k-space line to the next, and the lines follow one another for tens of milliseconds. A spin
that precesses $\Delta f$ too fast gains an extra phase of $2\pi\,\Delta f\,t_\mathrm{esp}$
per line, with $t_\mathrm{esp}$ the effective echo spacing, exactly as if it sat farther
along the axis. Across the $N$ lines of the readout that extra phase adds up to a
displacement of

$$\Delta y\ (\text{voxels}) = \Delta f \cdot t_\mathrm{esp} \cdot N = \Delta f \cdot \text{TotalReadoutTime},$$

because a spin whose phase advances by one full turn over the readout is reconstructed one
voxel away {cite:p}`jezzard1995`. `TotalReadoutTime` in the BIDS sidecar
{cite:p}`gorgolewski2016` is defined as the
effective echo spacing times the number of reconstructed phase-encode lines less one, so
the product gives the shift directly in voxels of the reconstructed image; in-plane
acceleration shortens it, partial Fourier does not. Multiply by the voxel size for
millimeters. The sign depends on the direction the blips step through k-space, which the
sidecar records as `PhaseEncodingDirection` (`j` or `j-` for the second array axis):
reversing the blips reverses the displacement.

Where the displacement varies from one row to the next, the signal of several true rows
lands in one voxel (**pile-up**, bright) or one row's signal is spread over several
(**stretching**, dark). The intensity change is the Jacobian of the mapping: if the signal at
$y$ lands at $y + d(y)$, the observed intensity is the true intensity divided by
$1 + \mathrm{d}d/\mathrm{d}y$, since the signal is conserved but the length it occupies is not.

For ASL this matters in two ways. The field offsets are largest in the inferior brain, above
the sinuses and beside the petrous bones, which is where the orbitofrontal and inferior
temporal gray matter sits; the ASL readout is long (the reference protocol's 20 ms would
already move signal one voxel at 50 Hz, and the `sdc` protocol's 35 ms moves it 3.5) and
in-plane acceleration is rarely used because the signal cannot afford it. And CBF is a
*ratio*, ΔM over M0, of two images distorted the same way, so the intensity part of the
artifact largely cancels; what remains is that the value is in the wrong place, which is
what every comparison with a segmentation, an atlas, or another modality feels.

## The aslscan setting that produces it

The `slab-fieldmap` phantom carries a synthetic field map in hertz at 1 mm, written by the
pipeline's phantom preparation: a +120 Hz lobe above the frontal sinuses, two −90 Hz lobes at
the temporal bones, and a shallow superior-inferior gradient from imperfect shimming. The
simulator applies it in the acquisition stage, with `PhaseEncodingDirection` from the
protocol (`ReversePhase` in the sidecar's simulation block for `j`), and the `sdc` protocol
sets `TotalReadoutTime` to 0.035 s. The acquisition grid is 64 × 68 × 20 voxels of
3.5 × 3.5 × 5 mm, corner-aligned with the phantom's 197 × 233 × 100 mm, so the field on the
acquisition grid is the box average of the 1 mm field over each voxel, by the same rule the
simulator uses for its ground-truth maps ([Chapter 0.2](../00-frontmatter/the-simulated-datasets.md)).

```{code-cell} python
:tags: [hide-input]
ds = data.load_dataset("sdc")
runs = {name: ds.run(name) for name in ("ap", "pa", "undistorted")}
p = runs["ap"].sidecar()
TRT = p["TotalReadoutTime"]
field = phantom.slab()["fieldmap"].astype(float)  # the 1 mm field box-averaged to the acquisition grid, packaged with the book
disp = {"ap": -field * TRT, "pa": +field * TRT}        # voxels along y; the sign is verified below
mask = runs["undistorted"].mask()
gm = runs["undistorted"].fractions()["gm"] > 0.7
print(f"phase encoding: ap {runs['ap'].sidecar()['PhaseEncodingDirection']}, pa {runs['pa'].sidecar()['PhaseEncodingDirection']}; "
      f"TotalReadoutTime {TRT} s; the simulator's line time {p['AslscanSimulation']['Acquisition']['TLineMs']:.3f} ms × 68 lines = {p['AslscanSimulation']['Acquisition']['TLineMs'] * 68:.1f} ms")
print(f"field on the acquisition grid: {field[mask].min():.0f} to {field[mask].max():.0f} Hz inside the brain -> displacement up to "
      f"{np.abs(field[mask]).max() * TRT:.1f} voxels ({np.abs(field[mask]).max() * TRT * VOX[1]:.0f} mm)")
print(f"gray matter voxels displaced by more than one voxel: {100 * np.mean(np.abs(field[gm]) * TRT > 1):.0f} %; "
      f"median displacement in gray matter of slices 0-5: {np.median(np.abs(field[gm & (np.arange(20)[None, None, :] <= 5)])) * TRT * VOX[1]:.1f} mm; "
      f"in the book's usual display slice {phantom.DISPLAY_SLICE}: at most {np.abs(field[:, :, phantom.DISPLAY_SLICE][mask[:, :, phantom.DISPLAY_SLICE]]).max() * TRT:.2f} voxels")

fig = plt.figure(figsize=(12, 3.4), layout="constrained")
gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1, 1.2])
ax = fig.add_subplot(gs[0])
im = ax.imshow(field[field.shape[0] // 2].T, origin="lower", cmap="RdBu_r", vmin=-120, vmax=120, aspect="equal",
               extent=(0, field.shape[1] * VOX[1], 0, field.shape[2] * VOX[2]))
ax.axhline((K + 0.5) * VOX[2], color=INK["primary"], lw=0.8, ls="--"); ax.text(3, (K + 0.5) * VOX[2] + 2, f"slice {K}", fontsize=7)
ax.set(title="field map, midline sagittal (anterior right)", xlabel="y (mm)", ylabel="z (mm)"); ax.grid(False)
ax = fig.add_subplot(gs[1])
show_image(ax, take_slice(np.where(mask, field, np.nan), K), f"field on the acquisition grid, slice {K}", cmap="RdBu_r", vmin=-120, vmax=120)
cb = fig.colorbar(im, ax=ax, shrink=0.8); cb.set_label("Hz")
sec = cb.ax.secondary_yaxis("left", functions=(lambda hz: hz * TRT, lambda vox: vox / TRT)); sec.set_ylabel("displacement (voxels)")
ax = fig.add_subplot(gs[2])
ax.bar(np.arange(20), [np.abs(field[:, :, k][mask[:, :, k]]).max() * TRT for k in range(20)], color=PALETTE[0])
ax.set(xlabel="slice", ylabel="largest displacement (voxels)", title="where the distortion is")
```

The lobes sit at the bottom of the slab: on the sagittal section the positive frontal lobe
and the negative temporal lobe (the latter off the midline, visible here only as its
inferior tail) are strongest in the lowest 20 mm, and the bar chart shows the largest
displacement per slice falling from four voxels in slices 0 to 3 to a fraction of a voxel
above slice 7. That is why this chapter displays slice 2 rather than the book's usual slice 9,
where the shift is 0.21 voxel at most. The color bar carries a second scale converting hertz
to voxels at this readout time: 114 Hz is 4.0 voxels, 14 mm. About 8 % of gray matter
voxels move by more than a voxel, and in the inferior slices the median displacement in
gray matter is half a millimeter.

## See it: pile-up and stretching in one dimension

The forward model in the setup cell is applied to a column of uniform tissue with a
Gaussian field bump, for both polarities. Because the tissue is uniform, every change in
intensity is the Jacobian.

```{code-cell} python
:tags: [hide-input]
y = np.arange(68, dtype=float)
bump = 100.0 * np.exp(-((y - 50) / 5.0) ** 2)                  # Hz, a lobe near the front of the head
profile = ((y > 10) & (y < 60)).astype(float)[None, :, None]   # uniform tissue
fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
axes[0].plot(y, -bump * TRT, color=PALETTE[0], label="j−: signal lands at y − Δf·TRT")
axes[0].plot(y, +bump * TRT, color=PALETTE[1], label="j: signal lands at y + Δf·TRT")
axes[0].set(xlabel="row y (posterior → anterior)", ylabel="displacement (voxels)", title="a 100 Hz lobe at 35 ms readout")
axes[0].legend(loc="upper left")
axes[1].plot(y, profile[0, :, 0], color=INK["secondary"], ls=":", lw=1.5, label="true (uniform)")
for sign, name, c in ((-1, "j−", PALETTE[0]), (+1, "j", PALETTE[1])):
    axes[1].plot(y, distort(profile, sign * bump[None, :, None] * TRT)[0, :, 0], color=c, label=name)
axes[1].set(xlabel="row y", ylabel="intensity (true = 1)", ylim=(0, 3), title="the distorted profile: pile-up above 1, stretching below")
axes[1].legend(loc="upper left")
fig.tight_layout()
```

The bump displaces rows near 50 by up to 3.5 voxels, forward for one polarity and backward
for the other. On the side of the lobe where the displacement grows with $y$, consecutive
rows are pulled apart and the tissue is stretched and dimmed; on the side where it shrinks,
rows are pushed together and the signal piles up into a bright band. The two polarities
swap which side is which, and that is the whole basis of blip-up/blip-down correction.

## The artifact-free reference and the two polarities

The `undistorted` run is the same protocol, seed, and noise on the slab without a field map.
The mean control image of each run is shown at slice 2, with the true brain outline from the
undistorted run's tissue fractions, and the difference of each distorted run from the
reference.

```{code-cell} python
:tags: [hide-input]
mag = {name: r.mag().astype(float) for name, r in runs.items()}
ctx = runs["ap"].context()
ctrl = {name: m[..., [i for i, c in enumerate(ctx) if c == "control"]].mean(-1) for name, m in mag.items()}
rms = lambda a, b, m=mask: np.sqrt(np.mean((a - b)[m] ** 2))
print("mean control image, RMS difference from the undistorted reference over the brain: "
      f"ap {rms(ctrl['ap'], ctrl['undistorted']):.0f}, pa {rms(ctrl['pa'], ctrl['undistorted']):.0f} image units "
      f"(noise alone: {rms(ctrl['undistorted'], ctrl['undistorted'] + 0) + 40 * np.sqrt(2 / 15):.0f}); in slice {K}: {rms(ctrl['ap'], ctrl['undistorted'], mask & (np.arange(20)[None, None, :] == K)):.0f}")
outline = lambda ax: ax.contour(take_slice(mask.astype(float), K), levels=[0.5], colors=[PALETTE[3]], linewidths=0.8)
fig, axes = plt.subplots(1, 5, figsize=(14.5, 3.2), layout="constrained")
for ax, name, title in zip(axes[:3], ("undistorted", "ap", "pa"), ("undistorted (reference)", "ap (j−): frontal lobe pushed back", "pa (j): pushed forward")):
    show_slice(ax, ctrl[name], K, title, vmin=0, vmax=7500); outline(ax)
show_slice(axes[3], ctrl["ap"] - ctrl["undistorted"], K, "ap − reference", kind="diff", vmin=-3000, vmax=3000)
show_slice(axes[4], ctrl["pa"] - ctrl["undistorted"], K, "pa − reference", kind="diff", vmin=-3000, vmax=3000)
fig.colorbar(axes[4].images[0], ax=axes[4], shrink=0.75, label="image units")
```

Compare the front of the brain with the outline. In the `ap` image the frontal gray matter
has fallen behind the outline and piled up into a bright band, and the two temporal poles,
under the negative lobes, have moved the other way; in the `pa` image every displacement is
reversed. The difference images are the same pattern with opposite signs. The RMS
difference from the reference over the whole brain is 570 units for `ap` and 490 for `pa`,
against a noise-only difference of about 15 between two 15-pair means; in slice 2 it is
1100, a fifth of the tissue signal.

## Reading the sign from the metadata, and checking it

The sidecar says `j-` for `ap` and `j` for `pa`. The question a correction must answer is
which way a positive field offset moves the signal in each. The forward model settles it:
the undistorted control image is distorted with both signs of the displacement and compared
with the acquired images.

```{code-cell} python
:tags: [hide-input]
for sign, label in ((-1, "signal lands at y − Δf·TRT"), (+1, "signal lands at y + Δf·TRT")):
    model = distort(ctrl["undistorted"], sign * field * TRT)
    print(f"{label}: RMS difference from ap {rms(model, ctrl['ap']):.0f}, from pa {rms(model, ctrl['pa']):.0f}")
fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2))
show_slice(axes[0], ctrl["ap"], K, "ap, acquired", vmin=0, vmax=7500)
show_slice(axes[1], distort(ctrl["undistorted"], -field * TRT), K, "reference distorted by −Δf·TRT", vmin=0, vmax=7500)
show_slice(axes[2], distort(ctrl["undistorted"], -field * TRT) - ctrl["ap"], K, "difference", kind="diff", vmin=-3000, vmax=3000)
fig.tight_layout()
```

With `PhaseEncodingDirection` `j-`, the signal of a voxel with field offset $\Delta f$ is
displaced by $-\Delta f \cdot \text{TotalReadoutTime}$ voxels along the array's second axis,
toward smaller $y$ (posterior), and with `j` by $+\Delta f \cdot \text{TotalReadoutTime}$,
toward anterior: the model with the negative sign reproduces the `ap` image to 70 units, the
noise of the comparison, and misses the `pa` image by 910, and the positive sign does the
reverse. The 1 mm field, the sidecar's readout time, and the reconstructed voxel grid are
all that is needed; no matrix-size factor enters, because `TotalReadoutTime` is already
defined per reconstructed line. This is the convention the rest of the chapter uses; a
reader with real data should run the same check, because a reoriented NIfTI file with an
unchanged sidecar can point the label the wrong way, and the next section shows what that
costs.

## Correction step by step: unwarp from a known field

With the field known, the correction inverts the forward model column by column: the value
that belongs at row $y$ is read from the distorted image at $y + d(y)$, and multiplied by
$1 + \mathrm{d}d/\mathrm{d}y$ to undo the pile-up and stretching. That is the `unwarp` in
the setup cell, six lines with `np.interp`. It is what a field-map-based correction does
{cite:p}`jezzard1995` once the field has been measured with a dual-echo gradient-echo scan,
unwrapped, and registered to the EPI; here the field is the simulator's own.

```{code-cell} python
:tags: [hide-input]
corr = {name: unwarp(ctrl[name], disp[name]) for name in ("ap", "pa")}
wrong = unwarp(ctrl["ap"], -disp["ap"])
print("mean control image, RMS difference from the undistorted reference: "
      f"ap {rms(ctrl['ap'], ctrl['undistorted']):.0f} -> {rms(corr['ap'], ctrl['undistorted']):.0f} after the unwarp; "
      f"pa {rms(ctrl['pa'], ctrl['undistorted']):.0f} -> {rms(corr['pa'], ctrl['undistorted']):.0f}; "
      f"mean of the two corrected images {rms((corr['ap'] + corr['pa']) / 2, ctrl['undistorted']):.0f}; ap unwarped with the wrong sign {rms(wrong, ctrl['undistorted']):.0f}")
fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
show_slice(axes[0], ctrl["ap"], K, "ap, distorted", vmin=0, vmax=7500); outline(axes[0])
show_slice(axes[1], corr["ap"], K, "ap, unwarped", vmin=0, vmax=7500); outline(axes[1])
show_slice(axes[2], corr["ap"] - ctrl["undistorted"], K, "unwarped − reference", kind="diff", vmin=-3000, vmax=3000)
show_slice(axes[3], wrong, K, "unwarped with the wrong sign", vmin=0, vmax=7500); outline(axes[3])
fig.tight_layout()
```

The unwarped `ap` image sits on the outline, the pile-up band is gone, and its RMS
difference from the reference falls from 570 to 58 units, a tenfold reduction to within a
few times the noise; what remains is a thin fringe at tissue edges, where any resampling is
least exact, and it is largest where the pile-up was strongest, because signal that was
summed into one voxel cannot be separated again by resampling it. The `pa` correction
lands at 87, worse because its pile-up sits on the frontal gray matter rather than in front
of it. The mean of the two corrected images is not better than the better one, since it
averages the good estimate of each region with the poor one. The wrong sign moves the signal
the same way the field did and doubles the displacement: 970 units, worse than no correction.

## Correction step by step: estimating the field from the reversed pair

When no field map was acquired, the pair itself supplies the field. The two images are the
same anatomy displaced by $d$ and by $-d$, so the displacement that unwarps one onto the
other is $2d$, and the field follows {cite:p}`andersson2003`. `topup`, the implementation
of that method in FSL {cite:p}`smith2004`, solves for a smooth three-dimensional field that
makes the two corrected images agree; the version
below is one column at a time. The displacement along a column is written as a sum of nine
smooth bumps, and the nine weights are chosen so that the `ap` column unwarped by $d$ and
the `pa` column unwarped by $-d$ agree as closely as possible, with a small penalty on
rough solutions. Each column is independent, so the map is noisier than `topup`'s; it is
run on the mean control images of slice 2.

```{code-cell} python
:tags: [hide-input]
centers = np.linspace(0, 67, 9)
basis = np.exp(-((y[:, None] - centers[None, :]) / 6.0) ** 2)       # (68 rows, 9 bumps)


def estimate_column(a, b, smoothness=50.0):
    def cost(w):
        d = basis @ w
        ua = np.interp(y + d, y, a) * (1 + np.gradient(d))          # a unwarped by d
        ub = np.interp(y - d, y, b) * (1 - np.gradient(d))          # b unwarped by -d
        return np.mean((ua - ub) ** 2) + smoothness * np.mean(np.diff(w) ** 2)
    return basis @ optimize.minimize(cost, np.zeros(len(centers)), method="Powell", options={"xtol": 1e-2, "ftol": 1e-4}).x


d_est = np.zeros((64, 68))
for i in range(64):
    if mask[i, :, K].any():
        d_est[i] = estimate_column(ctrl["ap"][i, :, K], ctrl["pa"][i, :, K])
mk = mask[:, :, K]
print(f"estimated displacement of the ap image in slice {K}: RMS error against the truth {np.sqrt(np.mean((d_est - disp['ap'][:, :, K])[mk] ** 2)):.2f} voxels "
      f"(the true displacement has RMS {np.sqrt(np.mean(disp['ap'][:, :, K][mk] ** 2)):.2f}, correlation {np.corrcoef(d_est[mk], disp['ap'][:, :, K][mk])[0, 1]:.3f})")
col = int(np.argmax(np.abs(disp["ap"][:, :, K]).max(1)))
ap_est = unwarp(ctrl["ap"][:, :, K][..., None], d_est[..., None])[..., 0]
print(f"ap control in slice {K} unwarped with the estimated field: RMS difference from the reference {np.sqrt(np.mean((ap_est - ctrl['undistorted'][:, :, K])[mk] ** 2)):.0f} "
      f"(with the true field {np.sqrt(np.mean((corr['ap'][:, :, K] - ctrl['undistorted'][:, :, K])[mk] ** 2)):.0f}, uncorrected {np.sqrt(np.mean((ctrl['ap'][:, :, K] - ctrl['undistorted'][:, :, K])[mk] ** 2)):.0f})")
fig = plt.figure(figsize=(12, 3.3), layout="constrained")
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.4])
ax = fig.add_subplot(gs[0]); show_image(ax, np.rot90(np.where(mk, disp["ap"][:, :, K], np.nan)), "true displacement of ap (voxels)", cmap="RdBu_r", vmin=-4, vmax=4)
ax = fig.add_subplot(gs[1]); im = show_image(ax, np.rot90(np.where(mk, d_est, np.nan)), "estimated from the ap/pa pair", cmap="RdBu_r", vmin=-4, vmax=4)
fig.colorbar(im, ax=ax, shrink=0.8, label="voxels")
ax = fig.add_subplot(gs[2])
ax.plot(y, disp["ap"][col, :, K], color=INK["secondary"], ls=":", lw=1.5, label="true")
ax.plot(y, d_est[col], color=PALETTE[1], label="estimated")
ax.set(xlabel="row y (posterior → anterior)", ylabel="displacement (voxels)", title=f"column x = {col}, through the frontal lobe")
ax.legend(loc="lower left")
```

The estimate reproduces the map: the frontal lobe's −4 voxels and the temporal lobes'
+3, with an RMS error of 0.17 voxel against a true displacement of RMS 1.1, and the profile
along the worst column follows the truth into the lobe and out of it. Unwarping the `ap`
image with the estimated field reaches an RMS of 200 units in this slice, from 1100
uncorrected, against 85 with the true field; the difference is the column-to-column jitter
of the estimate. Beyond the front of the brain, where the column holds no signal to match,
the estimate is unconstrained and wanders (the profile past row 60); a real implementation
regularizes across columns and slices, as `topup` does with its three-dimensional spline
field. A simpler use of the pair, averaging
the two corrected images, was measured above; the pair's real value is the field estimate,
which then corrects every volume of the series from its one polarity.

## Residual error versus truth: CBF

The same displacement moves the control, the label, and the M0 scan, so the difference
image and the calibration image are distorted alike and CBF, their ratio, keeps roughly the
right *values* in the wrong *places*. The scoring truth, `perfusion`, is on the undistorted
grid, so it sees both kinds of error. CBF is computed from the mean difference image with
each run's own M0 scan and the slice-dependent delay ([Chapter 14](../04-quantification/14-cbf-quantification.md)),
for the three runs and for the `ap` run after unwarping both its difference image and its M0
scan with the known field.

```{code-cell} python
:tags: [hide-input]
plds = quant.slice_plds(p["PostLabelingDelay"], p["SliceTiming"])[None, None, :]
truth = runs["undistorted"].truth("perfusion")
dm = {name: quant.subtract(mag[name], ctx).mean(-1) for name in runs}
m0 = {name: r.m0scan().astype(float) for name, r in runs.items()}
cbf = {name: quant.cbf_pcasl(dm[name], m0[name], plds, tau=p["LabelingDuration"]) for name in runs}
cbf["ap unwarped"] = quant.cbf_pcasl(unwarp(dm["ap"], disp["ap"]), unwarp(m0["ap"], disp["ap"]), plds, tau=p["LabelingDuration"])
low = gm & (np.arange(20)[None, None, :] <= 5)
print(f"{'CBF from':>14} {'RMSE in GM':>11} {'GM, slices 0-5':>15} {'GM mean, slices 0-5':>20}   (truth there: {truth[low].mean():.1f})")
for name in ("undistorted", "ap", "pa", "ap unwarped"):
    print(f"{name:>14} {quant.score(cbf[name], truth, gm)['rmse']:>11.1f} {quant.score(cbf[name], truth, low)['rmse']:>15.1f} {cbf[name][low].mean():>20.1f}")
per_slice = {name: [quant.score(cbf[name], truth, gm & (np.arange(20)[None, None, :] == k))["rmse"] for k in range(20)] for name in cbf}
fig, ax = plt.subplots(figsize=(7, 3.0))
for name, c in (("undistorted", INK["secondary"]), ("ap", PALETTE[0]), ("pa", PALETTE[1]), ("ap unwarped", PALETTE[2])):
    ax.plot(np.arange(20), per_slice[name], "o-", color=c, ms=3, lw=1.5, label=name)
ax.set(xlabel="slice", ylabel="CBF RMSE in gray matter (ml/100 g/min)", title="where the distortion hits the CBF map")
ax.legend()
fig.tight_layout()
smooth = lambda c: ndimage.gaussian_filter(np.where(mask, c, 0.0), (1.5, 1.5, 0)) / np.maximum(ndimage.gaussian_filter(mask.astype(float), (1.5, 1.5, 0)), 1e-3)
gm_outline = lambda ax: ax.contour(take_slice(runs["undistorted"].fractions()["gm"], K), levels=[0.5], colors=[PALETTE[3]], linewidths=0.7)
fig2, axes = plt.subplots(1, 5, figsize=(14.5, 3.2), layout="constrained")
for ax, name in zip(axes[:3], ("undistorted", "ap", "ap unwarped")):
    show_slice(ax, np.where(mask, smooth(cbf[name]), np.nan), K, f"CBF, {name} (smoothed)", kind="cbf"); gm_outline(ax)
for ax, name in zip(axes[3:], ("ap", "ap unwarped")):
    show_slice(ax, np.where(mask, smooth(cbf[name]) - smooth(truth), np.nan), K, f"error, {name}", kind="diff", vmin=-25, vmax=25)
fig2.colorbar(axes[4].images[0], ax=axes[4], shrink=0.75, label="ml/100 g/min")
```

The per-slice curve is the map of the artifact. Above slice 7 the three runs are
indistinguishable and their error is the noise of a 15-pair average, about 28 ml/100 g/min
per gray matter voxel. In slices 0 to 5 the distorted runs double it, to about 50, and the
unwarp brings it back to 22, slightly below the undistorted run's 25 because the resampling
smooths the noise a little. The maps, smoothed by 1.5 voxels so that the anatomy shows
through the noise, make the same point at slice 2 with the true gray matter outline drawn
on them: the distorted CBF map is a plausible perfusion image whose frontal gray matter
sits behind the outline and whose temporal gray matter sits inside it, and its error map is
a pair of bands, too high where gray matter values landed on white matter and too low
where white matter values landed on gray; after the unwarp the ribbon is on the outline and
the bands are gone. Note also the mean gray matter CBF in the inferior slices: 47.7
distorted against 49.0 undistorted, a difference of a few percent, while the per-voxel
error doubled. A regional mean hides the distortion, which is why a check against the
segmentation, not a summary number, is what reveals it.

:::{admonition} What the M0 division hides, and what it does not
:class: note
Because the M0 scan is distorted the same way as the ASL series, the Jacobian cancels in the
ratio: a piled-up voxel has too much ΔM and too much M0 in the same proportion. That is
why the CBF values in the distorted maps are roughly right. It does not cancel the
displacement, and it does not survive any step that uses an undistorted image: partial-volume
correction with tissue fractions from the T1-weighted image ([Chapter 12](./12-partial-volume.md)),
region-of-interest analysis with an atlas, or an M0 estimated from a different acquisition.
The `kitchen-sink` dataset of [Chapter 13](./13-assembled-pipeline.md) puts the reversed-polarity
correction into the full pipeline for this reason.
:::

## 3D readouts under the same field: a segmented GRASE and a spiral

The displacement formula has no term for the number of excitations, only for the phase a
spin accrues between consecutive reconstructed lines. That is what a segmented 3D readout
changes. In the four-shot 3D GRASE of the `readout-3d` dataset
{cite:p}`gunther2005,fernandezseara2005`, each echo of the train reads every fourth $k_y$
line, and the four shots interleave their lines into one 68-line plane. Within one echo the
lines are 0.3 ms apart, but they are every fourth line of the plane; the three lines in
between come from the other shots, read at the same moment of their own echoes. The
off-resonance phase therefore climbs by $2\pi\,\Delta f \times 0.3$ ms every four lines, a
ramp of 0.075 ms per reconstructed line: the effective echo spacing is the line spacing over
the segment count, and `TotalReadoutTime`, the effective echo spacing times the 67 line
intervals, is a quarter of the single-shot value. The displacement is still
$\Delta f \cdot \text{TotalReadoutTime}$ along the phase-encode axis, with the same sign
rule, and the unwarp is the same as in two dimensions. The through-plane axis is phase
encoded too, one partition per echo of the train, but each partition is read at a spin
echo, where the refocusing pulse has undone the static off-resonance phase, so the
through-plane axis is displaced by nothing; what the echo train does to it is the T2
window ([Chapter 2](../01-mri-physics/02-epi-and-reconstruction.md)), not the field.

A stack of spirals has no phase-encode axis in the plane. Each interleaf starts at the
center of k-space at its echo and spirals outward for 5 ms, so a spin at $\Delta f$ has
accrued a phase $2\pi\,\Delta f\,\tau$ by the time the trajectory reaches the radius it
samples at time $\tau$: the phase error is a function of the k-space radius, not of one
axis, and the point-spread function of an off-resonant voxel is the transform of that
radial phase ramp, a blur that spreads in every direction with a faint ring around it
{cite:p}`noll1991b`. Nothing moves; edges soften, most where the field is largest. The
simulator reconstructs the spiral by density-weighted least squares on the nominal
trajectory, as its sidecar's `Reconstruction` block says, with no field map, so the blur is
baked into the magnitude image it writes.

```{code-cell} python
:tags: [hide-input]
ds3 = data.load_dataset("readout-3d")
r3 = {name: ds3.run(name) for name in ("grase-4shot", "grase-4shot-fieldmap", "spiral", "spiral-fieldmap")}
pg, ps = r3["grase-4shot-fieldmap"].sidecar(), r3["spiral-fieldmap"].sidecar()
rog, ros = pg["AslscanSimulation"]["Readout"], ps["AslscanSimulation"]["Readout"]
TRT3 = pg["TotalReadoutTime"]
print(f"4-shot GRASE: line spacing {rog['LineSpacingMs']} ms over {rog['KySegments']} ky segments -> EffectiveEchoSpacing {pg['EffectiveEchoSpacing'] * 1e3:.3f} ms; "
      f"TotalReadoutTime {TRT3 * 1e3:.3f} ms (= {pg['EffectiveEchoSpacing'] * 1e3:.3f} ms x 67 line intervals); PhaseEncodingDirection {pg['PhaseEncodingDirection']}; "
      f"{rog['NumberShots']} shots, {rog['EchoTrainLength']} echoes of {rog['LinesPerEcho']} lines, echo spacing {rog['EchoSpacingMs']} ms")
print(f"displacement at 100 Hz: 4-shot GRASE {100 * TRT3:.2f} voxel; the single-shot reference protocol ({presets.REFERENCE.total_readout_time * 1e3:.0f} ms) "
      f"{100 * presets.REFERENCE.total_readout_time:.1f}; sdc ({TRT * 1e3:.0f} ms) {100 * TRT:.1f}")
print(f"spiral: {ros['Interleaves']} interleaves of {ros['Trajectory']['ReadoutTimeMs']} ms ({ros['Trajectory']['SamplesPerInterleaf']} samples at {ros['Trajectory']['DwellTimeMs'] * 1e3:.0f} us), "
      f"sidecar TotalReadoutTime {ps.get('TotalReadoutTime')}, PhaseEncodingDirection {ps.get('PhaseEncodingDirection')}; "
      f"phase accrued by the end of an interleaf at the field peak ({field[mask].max():.0f} Hz): {field[mask].max() * ros['Trajectory']['ReadoutTimeMs'] * 1e-3:.2f} cycle")
print("spiral reconstruction:", ros["Reconstruction"]["Method"])
ctx3 = r3["grase-4shot"].context()
mag3 = {name: r.mag().astype(float)[:64] for name, r in r3.items()}   # the spiral's 68 x 68 matrix pads the phantom's 64 columns with empty ones
print(f"spiral columns beyond the GRASE grid hold brain: {r3['spiral'].mask()[64:].any()}; tissue fractions identical on the common grid: "
      f"{np.array_equal(r3['spiral'].fractions()['gm'][:64], runs['undistorted'].fractions()['gm'])}")
ctrl3 = {name: m[..., [i for i, c in enumerate(ctx3) if c == "control"]].mean(-1) for name, m in mag3.items()}
dm3 = {name: quant.subtract(m, ctx3).mean(-1) for name, m in mag3.items()}
wm99 = runs["undistorted"].fractions()["wm"] > 0.99
resid = lambda name: (mag3[name][wm99] - mag3[name][wm99].mean(-1, keepdims=True)).ravel()
print(f"each field-map run shares its twin's noise realization (correlation of the pure-white-matter residuals: GRASE {np.corrcoef(resid('grase-4shot'), resid('grase-4shot-fieldmap'))[0, 1]:.5f}, "
      f"spiral {np.corrcoef(resid('spiral'), resid('spiral-fieldmap'))[0, 1]:.5f}), so every difference below is the field's doing")
```

The sidecar carries the quarter directly: an `EffectiveEchoSpacing` of 0.075 ms, the
0.3 ms line spacing over four segments, and a `TotalReadoutTime` of 5.025 ms, so 100 Hz
moves the GRASE image 0.50 voxel where the single-shot reference protocol moves it 2.0 and
the `sdc` runs 3.5. The spiral sidecar has neither key, because there is no axis to
apply them to; what it has is a 5 ms interleaf, over which a spin at the lobe's peak of
114 Hz turns 0.57 of a cycle, and a reconstruction description that names no field map.
Each field-map run was simulated with its twin's seed, and the pure-white-matter residuals
of the two correlate to 0.99995 for the GRASE and 0.99974 for the spiral, so the differences
that follow are the field's effect alone, not noise.

```{code-cell} python
:tags: [hide-input]
twin, fm = ctrl3["grase-4shot"], ctrl3["grase-4shot-fieldmap"]
in_slice = lambda k: mask & (np.arange(20)[None, None, :] == k)
print(f"4-shot GRASE control image, RMS difference of the field-map run from its twin: {rms(fm, twin):.0f} units over the brain, {rms(fm, twin, in_slice(K)):.0f} in slice {K} "
      f"(the 2D sdc ap run against undistorted: {rms(ctrl['ap'], ctrl['undistorted']):.0f} and {rms(ctrl['ap'], ctrl['undistorted'], in_slice(K)):.0f})")
times = {"2.5": 0.5, "5.0 (the sidecar)": 1, "10": 2, "20 (single shot)": 4, "35 (sdc)": 7}
misfit = {t: rms(distort(twin, -field * TRT3 * f), fm) for t, f in times.items()}
print("twin distorted by -Δf x readout time, RMS difference from the field-map run: " + "; ".join(f"{t} ms {v:.0f}" for t, v in misfit.items())
      + f"; +Δf x 5.0 ms (the wrong sign) {rms(distort(twin, +field * TRT3), fm):.0f}")


def crossing(c, frac, lo=35, hi=60):
    """Where a profile falls through `frac` of its plateau (the maximum over c[lo:hi]) beyond `lo`, by linear interpolation."""
    level = frac * c[lo:hi].max()
    j = np.where(c[lo:] > level)[0].max() + lo
    return j + (c[j] - level) / (c[j] - c[j + 1])


print(f"anterior brain edge along column x = {col}, slice {K} (half-maximum crossing): twin at row {crossing(twin[col, :, K], 0.5):.2f}, field-map run at {crossing(fm[col, :, K], 0.5):.2f} "
      f"-> moved {crossing(fm[col, :, K], 0.5) - crossing(twin[col, :, K], 0.5):+.2f} voxel; -Δf x TotalReadoutTime is {-field[col, int(round(crossing(twin[col, :, K], 0.5))), K] * TRT3:+.2f} at that row, "
      f"{-field[col, :, K].max() * TRT3:+.2f} at the lobe's peak; 20-80 % edge width {crossing(twin[col, :, K], 0.2) - crossing(twin[col, :, K], 0.8):.2f} -> {crossing(fm[col, :, K], 0.2) - crossing(fm[col, :, K], 0.8):.2f} voxels")
fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.2))
show_slice(axes[0], fm, K, "4-shot GRASE with the field", vmin=0, vmax=7500); outline(axes[0])
show_slice(axes[1], fm - twin, K, "field-map run − twin", kind="diff", vmin=-1500, vmax=1500)
show_slice(axes[2], distort(twin, -field * TRT3) - twin, K, "twin distorted by −Δf·5 ms, − twin", kind="diff", vmin=-1500, vmax=1500)
axes[3].plot([float(t.split()[0]) for t in times], list(misfit.values()), "o-", color=PALETTE[0])
axes[3].set(xlabel="readout time in the model (ms)", ylabel="RMS misfit (image units)", title="which readout time explains the image")
fig.tight_layout()
```

The GRASE field-map run differs from its twin by 77 units over the brain and 150 in
slice 2, against 570 and 1100 for the 2D `sdc` run under the same field. The difference
image has the shape of the 2D one, a posterior-anterior dipole at the frontal edge and the
reverse at the temporal poles, at a fraction of the amplitude, and the twin distorted by
$-\Delta f \times 5$ ms reproduces it: the misfit is 27 units at the sidecar's readout time,
91 at twice it, 244 at the single-shot 20 ms and 483 at the `sdc` 35 ms, and the wrong sign
at 5 ms gives 150, the same as no model at all. The displacement is the sidecar's, a
quarter of a single shot's, with the 2D sign convention. The edge measurement along the
column through the lobe is a coarser check: the half-maximum crossing moves 0.42 voxel
posterior, where the field at that row predicts 0.27 and the lobe's peak 0.57, and the
20-80 % edge width grows from 1.5 to 2.8 voxels, because the displacement shrinks toward
the front of the lobe and stretches the edge, exactly as in the one-dimensional model
earlier. Nothing has moved along $z$, as the spin-echo encoding of the partitions
predicts.

```{code-cell} python
:tags: [hide-input]
sp, spf = ctrl3["spiral"], ctrl3["spiral-fieldmap"]
bands = ((0, 10), (10, 30), (30, 60), (60, 130))
in_band = lambda lo, hi: mask & (np.abs(field) >= lo) & (np.abs(field) < hi)
print("spiral control image, RMS difference of the field-map run from its twin by field band: "
      + "; ".join(f"|Δf| {lo}-{hi} Hz {rms(spf, sp, in_band(lo, hi)):.0f} ({in_band(lo, hi).sum()} voxels)" for lo, hi in bands)
      + "; the 4-shot GRASE in the same bands: " + ", ".join(f"{rms(fm, twin, in_band(lo, hi)):.0f}" for lo, hi in bands))
print(f"whole brain {rms(spf, sp):.0f}; slice {K} {rms(spf, sp, in_slice(K)):.0f}; slice {phantom.DISPLAY_SLICE} {rms(spf, sp, in_slice(phantom.DISPLAY_SLICE)):.0f}; "
      f"GM control signal {sp[gm].mean():.0f} -> {spf[gm].mean():.0f} in all GM, {sp[gm & (field > 60)].mean():.0f} -> {spf[gm & (field > 60)].mean():.0f} under the frontal lobe")
steep = np.abs(np.gradient(sp, axis=0)) + np.abs(np.gradient(sp, axis=1))   # how much structure the twin has to blur, in units per voxel
for name, lobe in (("frontal lobe (Δf > 60 Hz)", mask & (field > 60)), ("temporal lobes (Δf < −60 Hz)", mask & (field < -60))):
    print(f"{name}: {lobe.sum()} voxels, RMS difference spiral {rms(spf, sp, lobe):.0f}, GRASE {rms(fm, twin, lobe):.0f}; the twin's in-plane intensity gradient there {steep[lobe].mean():.0f} units/voxel")
fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.2))
show_slice(axes[0], sp, K, "spiral, no field", vmin=0, vmax=7500); outline(axes[0])
show_slice(axes[1], spf, K, "spiral with the field", vmin=0, vmax=7500); outline(axes[1])
show_slice(axes[2], spf - sp, K, f"difference, slice {K}", kind="diff", vmin=-600, vmax=600)
show_slice(axes[3], spf - sp, phantom.DISPLAY_SLICE, f"difference, slice {phantom.DISPLAY_SLICE}", kind="diff", vmin=-600, vmax=600)
fig.colorbar(axes[3].images[0], ax=axes[3], shrink=0.75, label="image units")
fig.tight_layout()
```

In the spiral images nothing has moved: the outline sits on both. What the field did is in the
difference image, and it is not where the 2D or the GRASE artifact was largest. The frontal
lobe, where the field peaks, is nearly blank except for a spot at the midline edge above the
sinus, and the strong pattern sits at the two temporal poles: a dark core along the inside
of the lateral brain edge with a bright rim outside it, the blur of a sharp edge. Slice 9,
where the field is a few hertz, is blank. The RMS difference climbs with the field band,
from 7 units where $|\Delta f| < 10$ Hz to 146 above 60 Hz, against 27 and 267 for the
GRASE in the same voxels, and the two lobes split differently for the two readouts: under
the frontal lobe the spiral differs from its twin by 79 units and the GRASE by 279, under the
temporal lobes by 218 and 245. The displacement moves everything the field covers, the
blur only changes an image where there is structure to smear, and the frontal peak sits
inside tissue whose intensity changes by 304 units per voxel, the temporal lobes on the brain
edge where it changes by 861. The gray matter signal under the frontal lobe is unchanged,
5168 against 5179.

```{code-cell} python
:tags: [hide-input]
jt = int(np.unravel_index(np.argmin(field[:, :, K]), field.shape[:2])[1])   # the row through the temporal pole, where the field is most negative
lateral = lambda c, frac: 63 - crossing(c[::-1], frac, 32, 54)             # the left brain edge along a row, in x
print(f"anterior edge along column x = {col}, slice {K}, spiral: half-maximum crossing {crossing(sp[col, :, K], 0.5):.2f} -> {crossing(spf[col, :, K], 0.5):.2f} "
      f"({crossing(spf[col, :, K], 0.5) - crossing(sp[col, :, K], 0.5):+.2f} voxel); 20-80 % edge width {crossing(sp[col, :, K], 0.2) - crossing(sp[col, :, K], 0.8):.2f} -> {crossing(spf[col, :, K], 0.2) - crossing(spf[col, :, K], 0.8):.2f} voxels")
print(f"left temporal edge along row y = {jt} ({field[int(round(lateral(sp[:, jt, K], 0.5))), jt, K]:.0f} Hz at the edge): spiral crossing {lateral(sp[:, jt, K], 0.5):.2f} -> {lateral(spf[:, jt, K], 0.5):.2f}, "
      f"width {lateral(sp[:, jt, K], 0.8) - lateral(sp[:, jt, K], 0.2):.2f} -> {lateral(spf[:, jt, K], 0.8) - lateral(spf[:, jt, K], 0.2):.2f} voxels; "
      f"GRASE crossing {lateral(twin[:, jt, K], 0.5):.2f} -> {lateral(fm[:, jt, K], 0.5):.2f}, width {lateral(twin[:, jt, K], 0.8) - lateral(twin[:, jt, K], 0.2):.2f} -> {lateral(fm[:, jt, K], 0.8) - lateral(fm[:, jt, K], 0.2):.2f}")
print(f"spiral difference along that row at x = 5 to 14: {np.round(spf[5:15, jt, K] - sp[5:15, jt, K]).astype(int)}")
fig, axes = plt.subplots(1, 3, figsize=(13, 3.2), sharey=True)
panels = ((twin, fm, y, lambda a: a[col, :, K], field[col, :, K], (30, 67), "row y (posterior → anterior)", f"4-shot GRASE, column x = {col}"),
          (sp, spf, y, lambda a: a[col, :, K], field[col, :, K], (30, 67), "row y (posterior → anterior)", f"spiral, column x = {col}"),
          (sp, spf, np.arange(64), lambda a: a[:, jt, K], field[:, jt, K], (0, 25), "column x (left → right)", f"spiral, row y = {jt}, temporal pole"))
for ax, (a, b, xs, pick, f, xlim, xlabel, title) in zip(axes, panels):
    ax.plot(xs, pick(a), color=INK["secondary"], ls=":", lw=1.5, label="no field")
    ax.plot(xs, pick(b), color=PALETTE[0], label="with the field")
    ax.set(xlim=xlim, xlabel=xlabel, title=f"{title}, slice {K}")
    sec = ax.twinx(); sec.plot(xs, f, color=PALETTE[3], lw=0.8); sec.set_ylim(-150, 150)
    sec.set_ylabel("field (Hz)", color=PALETTE[3]) if ax is axes[-1] else sec.set_yticks([])
axes[0].set_ylabel("mean control (image units)"); axes[0].legend(loc="lower left")
fig.tight_layout()
```

The profiles put the two mechanisms side by side. Along the column through the frontal
lobe the GRASE curve is the dotted one shifted and stretched where the field rises; the
spiral curve is the dotted one with its corners rounded, the frontal edge's half-maximum
crossing at the same row to 0.04 voxel and its 20-80 % width grown from 1.64 to 1.82
voxels. Along the row through the temporal pole, where the field is −68 Hz on the brain
edge itself, the spiral's crossing is unchanged to 0.01 voxel while its width grows from
1.32 to 1.59, and the difference along the row alternates in sign from one voxel to the
next, +619, −326, +294, −109: the ring of the off-resonance point-spread function, an
overshoot that a sharp edge turns into a visible fringe. The GRASE's lateral edge does not
widen at all (1.28 to 1.25), because its displacement is along $y$, not $x$. Blur has no
sign to read from a sidecar and no reversed polarity to pair it with; its magnitude is set
by the field and the interleaf duration, and the places it shows are the places with edges.

```{code-cell} python
:tags: [hide-input]
m0_3 = {name: r.m0scan().astype(float)[:64] for name, r in r3.items()}
cbf3 = {name: quant.cbf_pcasl(dm3[name], m0_3[name], r3[name].sidecar()["PostLabelingDelay"], tau=r3[name].sidecar()["LabelingDuration"]) for name in r3}
frontal, temporal = gm & (field > 60), gm & (field < -60)
print(f"frontal gray matter under the +120 Hz lobe (Δf > 60 Hz, GM fraction > 0.7): {frontal.sum()} voxels, truth {truth[frontal].mean():.1f}; temporal under the −90 Hz lobes (Δf < −60 Hz): {temporal.sum()} voxels")
print(f"{'field-map run vs its twin':>26} {'ΔM RMS diff, frontal GM':>24} {'all GM':>7} {'CBF RMS diff, frontal':>22} {'temporal':>9} {'CBF RMSE vs truth, frontal: twin -> field':>42} {'frontal mean':>14}")
pairs = (("grase-4shot-fieldmap", "grase-4shot", dm3, cbf3), ("spiral-fieldmap", "spiral", dm3, cbf3), ("2D sdc ap", "undistorted", dm, cbf))
for a, b, d_, c_ in pairs:
    a_ = "ap" if a == "2D sdc ap" else a
    print(f"{a:>26} {rms(d_[a_], d_[b], frontal):>24.1f} {rms(d_[a_], d_[b], gm):>7.1f} {rms(c_[a_], c_[b], frontal):>22.1f} {rms(c_[a_], c_[b], temporal):>9.1f} "
          f"{quant.score(c_[b], truth, frontal)['rmse']:>20.1f} -> {quant.score(c_[a_], truth, frontal)['rmse']:<18.1f} {c_[b][frontal].mean():>5.1f} -> {c_[a_][frontal].mean():.1f}")
emptied = frontal & (m0["ap"] < 0.5 * np.median(m0["undistorted"][frontal]))
print(f"the 2D CBF difference is the M0 division's: {emptied.sum()} of the {frontal.sum()} frontal voxels hold less than half the typical M0 in the ap scan "
      f"(the tissue moved away; minimum {m0['ap'][frontal].min():.0f} against {m0['undistorted'][frontal].min():.0f} undistorted); without them the RMS difference is {rms(cbf['ap'], cbf['undistorted'], frontal & ~emptied):.1f}")
fig, axes = plt.subplots(1, 6, figsize=(16, 2.9), layout="constrained")
for ax, name in zip(axes[:4], r3):
    show_slice(ax, dm3[name], K, f"ΔM, {name}", vmin=0, vmax=60); gm_outline(ax)
show_slice(axes[4], dm3["grase-4shot-fieldmap"] - dm3["grase-4shot"], K, "GRASE: field − twin", kind="diff", vmin=-15, vmax=15)
show_slice(axes[5], dm3["spiral-fieldmap"] - dm3["spiral"], K, "spiral: field − twin", kind="diff", vmin=-15, vmax=15)
fig.colorbar(axes[5].images[0], ax=axes[5], shrink=0.75, label="image units")
```

The four ΔM maps are the same perfusion image, and the difference images carry the two
patterns at the scale of ΔM, about 30 units in gray matter: a dipole at the frontal edge and
along the posterior rim for the GRASE, the temporal-pole blur and the midline spot for the
spiral, with a speckle at every tissue edge in both from the shallow shim gradient's
fraction-of-a-voxel shifts. In the 242 frontal gray matter voxels under the positive lobe
the field-map runs differ from their twins by 2.1 ΔM units for the GRASE and 2.7 for the
spiral, against 0.6 and 0.5 across all gray matter, and the 2D `sdc` run differs from its
reference by 13.6 in the same voxels. In CBF the GRASE pair differs by 2.0 ml/100 g/min
there and the spiral pair by 4.6, and 2.5 each under the temporal lobes; the 2D pair
differs by 129, a number the M0 division manufactures: 26 of the 242 frontal voxels lie
beyond the displaced brain edge of the `ap` run, where its M0 scan holds a fraction of the
tissue value (as little as 64 image units against 4413) and the ratio explodes, and without
them the difference is 14.7. Against the truth, both 3D twins already sit low
in this thin cortex (47.9 against 56.4, the through-plane point-spread function and partial
volume, [Chapter 12](./12-partial-volume.md)), and the field adds 0.3 ml/100 g/min to the
GRASE's RMSE and 1.7 to the spiral's, with the regional mean moving by 0.3 and 1.4. A
regional mean hides the spiral's blur as thoroughly as it hid the 2D displacement.

What each readout needs from a pipeline follows from the mechanism. The GRASE needs the
same unwarp as the 2D images, with the field map and the sidecar's `PhaseEncodingDirection`
and `TotalReadoutTime`; that `TotalReadoutTime` is the effective one, already divided by the
segment count, and it is the number the unwarp must use, not the line spacing times the
line count. A reversed-polarity pair serves it as well as it serves EPI. The spiral cannot
be unwarped, because nothing is displaced; its correction is a field map used inside the
reconstruction, a conjugate-phase or time-segmented gridding that gives each sample's phase
back before it is placed {cite:p}`noll1991b`, or the iterative model-based reconstruction
that includes the field in the forward model {cite:p}`sutton2003`. Once the magnitude image
is written, the blur is in it, and a pipeline that receives spiral magnitude images can
only note that the frontal and temporal cortex is softened, not restore it. The simulator's
spiral is an ideal trajectory with no gradient delays or eddy currents, an approximation its
sidecar lists; on a scanner those add a second, field-independent blur that calibration of
the trajectory, not a field map, removes.

## What this implies for acquisition

- **Shorten the readout.** The displacement is proportional to `TotalReadoutTime`, so
  in-plane acceleration reduces it directly, at a cost in SNR that ASL feels more than most
  sequences ([Chapter 7](../02-labeling/07-acquisition-parameters.md)); partial Fourier
  shortens the echo time but not the displacement.
- **Or segment it.** 3D readouts such as GRASE {cite:p}`gunther2005,fernandezseara2005` and
  stack-of-spirals {cite:p}`vidorreta2013` are among the reasons the white paper prefers 3D
  {cite:p}`alsop2015`, and under the same field the measurements above put numbers on it:
  the four-shot GRASE's displacement is the sidecar's 5 ms, a quarter of a single shot's,
  and its field-map run differs from its twin by 77 image units where the 2D `sdc` run
  differs by 570; the spiral displaces nothing and blurs instead, 146 units where the field
  exceeds 60 Hz against the GRASE's 267, and only where there are edges to smear. In the
  frontal gray matter ΔM the field costs the GRASE 2.1 units and the spiral 2.7 against
  their twins, where the 2D run pays 13.6. The GRASE is corrected like EPI; the spiral only
  in the reconstruction.
- **Acquire the reversed polarity.** A few volumes with the blips reversed, or the M0 scan
  in both polarities, are enough to estimate the field; the ASL series is then corrected
  from its one polarity. A gradient-echo field map serves the same purpose if it is
  registered to the EPI.
- **Record the metadata.** `PhaseEncodingDirection` and `TotalReadoutTime` are the two
  fields a distortion correction reads. BIDS requires them when a field-map-based
  correction is intended {cite:p}`gorgolewski2016,clement2022`, and aslscan requires them
  to simulate the distortion; a reoriented image with a stale sidecar doubles the
  distortion, as the wrong-sign panel shows.
- **Look at the inferior slices.** The artifact lives above the sinuses and beside the
  petrous bones; a check of the frontal and temporal gray matter against the outline of the
  anatomical image, at the lowest slices, catches it.
- **Shim** before the ASL series; every hertz removed at the source is a fraction of a
  voxel not moved.

## Further reading

Field-map-based correction {cite:p}`jezzard1995`; the reversed-polarity estimate of the
field {cite:p}`andersson2003`; the metadata BIDS records for it {cite:p}`gorgolewski2016,clement2022`;
the ASLPrep pipeline {cite:p}`adebimpe2022`, which applies these corrections through
SDCFlows, the distortion-correction workflows that grew out of fMRIPrep
{cite:p}`esteban2019`; the white paper's discussion of readouts {cite:p}`alsop2015`, the
3D GRASE readout {cite:p}`gunther2005,fernandezseara2005` and the comparison of 2D and 3D
readouts including the stack of spirals {cite:p}`vidorreta2013`; off-resonance in spiral
imaging and its correction in the reconstruction, the conjugate-phase method
{cite:p}`noll1991b` and the iterative field-corrected reconstruction {cite:p}`sutton2003`.
