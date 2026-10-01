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

Pipeline-tier datasets are simulated offline by aslscan ([Chapter 0.2](../00-frontmatter/the-simulated-datasets.md)).
:::

## Learning goals

After this chapter you can:

- predict, from a field map and the sidecar's `TotalReadoutTime`, where an EPI image is displaced and by how many voxels
- read the sign of the displacement from `PhaseEncodingDirection` and check it on a blip-up/blip-down pair
- undo the distortion when the field is known, including the intensity correction, and explain how the field is estimated from a reversed-polarity pair when it is not
- say what distortion does to a CBF map, why the M0 division hides part of it, and why the hidden part still matters

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

ASL images are read out with echo-planar imaging, one whole slice per excitation
([Chapter 2](../01-mri-physics/02-epi-and-reconstruction.md)). Along the frequency-encode
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
voxel away {cite:p}`jezzard1995`. `TotalReadoutTime` in the BIDS sidecar is defined as the
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
other is $2d$, and the field follows {cite:p}`andersson2003`. FSL's `topup` solves for a
smooth three-dimensional field that makes the two corrected images agree; the version
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

## What this implies for acquisition

- **Shorten the readout.** The displacement is proportional to `TotalReadoutTime`, so
  in-plane acceleration reduces it directly, at a cost in SNR that ASL feels more than most
  sequences ([Chapter 7](../02-labeling/07-acquisition-parameters.md)); partial Fourier
  shortens the echo time but not the displacement. 3D readouts such as GRASE and
  stack-of-spirals distort differently and usually less along the slow axis, which is one of
  the reasons the white paper prefers them {cite:p}`alsop2015`.
- **Acquire the reversed polarity.** A few volumes with the blips reversed, or the M0 scan
  in both polarities, are enough to estimate the field; the ASL series is then corrected
  from its one polarity. A gradient-echo field map serves the same purpose if it is
  registered to the EPI.
- **Record the metadata.** `PhaseEncodingDirection` and `TotalReadoutTime` are what every
  correction reads, and ASL-BIDS requires them {cite:p}`clement2022`; a reoriented image
  with a stale sidecar doubles the distortion, as the wrong-sign panel shows.
- **Look at the inferior slices.** The artifact lives above the sinuses and beside the
  petrous bones; a check of the frontal and temporal gray matter against the outline of the
  anatomical image, at the lowest slices, catches it.
- **Shim** before the ASL series; every hertz removed at the source is a fraction of a
  voxel not moved.

## Further reading

Field-map-based correction {cite:p}`jezzard1995`; the reversed-polarity estimate of the
field {cite:p}`andersson2003`; the metadata ASL-BIDS records for it {cite:p}`clement2022`;
the ASLPrep pipeline, which applies these corrections through SDCFlows
{cite:p}`adebimpe2022`; the white paper's discussion of readouts {cite:p}`alsop2015`.
