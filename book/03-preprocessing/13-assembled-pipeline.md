---
title: "13. Remaining artifacts and the assembled pipeline"
kernelspec:
  name: python3
  display_name: Python 3
---

:::{admonition} Simulated datasets in this chapter
:class: note
- **Built in this page:** the suppression timeline of the kitchen-sink protocol and the box-averaged field map of the phantom ([Appendix B](../appendices/b-data-manifest.md#app-b-package-data)).
- **`readout`**: partial Fourier 6/8, a 5 % Nyquist ghost, and two k-space spikes per slice, one run each ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-readout)).
- **`ref-pcasl`**: the artifact-free reference the `readout` runs are compared with ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-ref-pcasl)).
- **`sdc`**: used once, to fix the sign of the field-map displacement against its undistorted run ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-sdc)).
- **`kitchen-sink`**: everything at once, as a blip-up/blip-down pair with the same head motion ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-kitchen-sink)).

Pipeline-tier datasets are simulated offline by aslscan ([Chapter 0.2](../00-frontmatter/the-simulated-datasets.md)).
:::

## Learning goals

After this chapter you can:

- recognize partial-Fourier blurring, a Nyquist ghost and k-space spikes in a control image and in a difference image, and say what to do about each
- run the corrections of Chapters [8](./08-noise.md) to [12](./12-partial-volume.md) in order on one series and score each step against the truth
- explain why registration comes before subtraction, why distortion correction must treat the M0 image the same way as the series, and why partial-volume correction comes last
- compute the quality-control numbers a pipeline reports (temporal SNR, negative-voxel fraction, GM/WM ratio, motion summary) and read them

The chapter has two parts: three readout imperfections that have no chapter of their
own, and then the preprocessing of Part III assembled into one pipeline and applied step
by step to the `kitchen-sink` dataset, which has every artifact of the book on at once.

```{code-cell} python
:tags: [hide-cell]
import json
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from scipy import ndimage

from aslbook import data, grid, kinetic, phantom, plotting, presets, protocols, quant
from aslbook.plotting import INK, PALETTE, TISSUE_COLORS, set_style, show_slice

set_style()
K = phantom.DISPLAY_SLICE
GM_T = presets.TISSUES["GM"]


def kinetic_factor(pld, tau=presets.REFERENCE.labeling_duration):
    """White-paper ΔM over kinetic-model ΔM for pure GM at delay ``pld`` (Chapter 12)."""
    t1b, a, lam = presets.T1_BLOOD, presets.ALPHA["PCASL"], presets.LAMBDA
    wp = 2 * (GM_T.m0 / lam) * (GM_T.perfusion / 6000) * a * t1b * np.exp(-np.asarray(pld) / t1b) * (1 - np.exp(-tau / t1b))
    return wp / kinetic.delta_m(np.asarray(pld) + tau, GM_T.perfusion, GM_T.att, GM_T.t1, GM_T.m0, tau=tau)


def cbf_from(deltam, m0, p):
    """The white-paper formula with each slice's delay, times the kinetic factor; ``m0`` is
    whatever calibration image the step under test provides."""
    plds = quant.slice_plds(p["PostLabelingDelay"], protocols.slice_offsets(p))
    return quant.cbf_pcasl(deltam, m0, plds[None, None, :]) * kinetic_factor(plds)[None, None, :]


def rigid(vol, q, order=1):
    """Undo a head movement of rotations (rx, ry, rz; radians, Rz Ry Rx about the centre) and
    translations (tx, ty, tz; mm): read the head point at x from R x + t."""
    tx, ty, tz, rx, ry, rz = q
    cx, sx = np.cos(rx), np.sin(rx); cy, sy = np.cos(ry), np.sin(ry); cz, sz = np.cos(rz), np.sin(rz)
    R = (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
         @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]))
    M = np.diag(1 / VOX) @ R @ np.diag(VOX)
    return ndimage.affine_transform(vol, M, offset=CENTRE - M @ CENTRE + np.array([tx, ty, tz]) / VOX, order=order, mode="nearest")


def register(vol, ref, where, n_iter=10):
    """Rigid registration by Gauss-Newton least squares on the voxels in ``where``: the
    residual's derivative is the image gradient (translations) times the lever arm (rotations)."""
    q = np.zeros(6)
    X = (np.stack(np.indices(vol.shape), -1)[where] - CENTRE) * VOX
    for _ in range(n_iter):
        w = rigid(vol, q)
        r = (w - ref)[where]
        g = np.stack([np.gradient(w, axis=i) / VOX[i] for i in range(3)], -1)[where]
        J = np.column_stack([g] + [(g * np.cross(np.eye(3)[k], X)).sum(1) for k in range(3)])
        dq = -np.linalg.solve(J.T @ J, J.T @ r)
        q = q + dq
        if np.abs(dq[:3]).max() < 1e-3 and np.abs(dq[3:]).max() < 1e-5:
            break
    return q


def unwarp(img, disp):
    """Undo a displacement ``disp`` (voxels, along y) column by column: ``out[y] = img[y + disp[y]]``
    times the Jacobian ``1 + d disp/dy`` that restores intensity."""
    out = np.empty_like(img)
    y = np.arange(img.shape[1], dtype=float)
    for x in range(img.shape[0]):
        for z in range(img.shape[2]):
            d = disp[x, :, z]
            out[x, :, z] = np.interp(y + d, y, img[x, :, z]) * (1 + np.gradient(d))
    return out


def fd(poses):
    """Framewise displacement (Power et al. 2012): translations plus rotations on a 50 mm sphere."""
    d = np.abs(np.diff(poses, axis=0))
    return np.r_[0.0, d[:, :3].sum(1) + 50.0 * d[:, 3:].sum(1)]


TABLE = []
def score_step(name, cbf):
    """Bias, RMSE and median absolute error against the truth in gray matter, kept for the table."""
    g = quant.score(cbf, truth, gm_score)
    medae = float(np.median(np.abs(cbf - truth)[gm_score]))
    TABLE.append((name, g["bias"], g["rmse"], medae))
    print(f"{name:56s} GM bias {g['bias']:+6.1f}  RMSE {g['rmse']:5.1f}  median |error| {medae:5.1f}")
    return cbf
```

## Part one: three readout imperfections

The EPI readout of [Chapter 2](../01-mri-physics/02-epi-and-reconstruction.md) can go wrong in
ways that no ASL-specific step corrects. The `readout` dataset switches on three of them,
one per run, in the reference protocol with 15 pairs: `partial_fourier: 0.75` (only 6/8 of
the phase-encode lines are acquired, the rest filled by conjugate symmetry),
`ghost_offset: 0.05` (odd and even readout lines disagree, so 5 % of the object reappears
half a field of view away along the phase-encode axis), and `n_spikes: 2` with
`spike_amplitude: 4.0` (two k-space samples per slice replaced by a large value).

```{code-cell} python
:tags: [hide-input]
ref = data.load_dataset("ref-pcasl").run()
rd = data.load_dataset("readout")
runs = {"reference": ref, "partial Fourier 6/8": rd.run("pf68"), "5 % Nyquist ghost": rd.run("ghost"), "2 k-space spikes per slice": rd.run("spikes")}
mask_ref, fr_ref = ref.mask(), ref.fractions()
ghost_zone = np.roll(mask_ref, mask_ref.shape[1] // 2, axis=1) & ~ndimage.binary_dilation(mask_ref, iterations=3)
background = ~ndimage.binary_dilation(mask_ref, iterations=4)

fig, axes = plt.subplots(2, 4, figsize=(12.5, 6.4))
for col, (name, run) in enumerate(runs.items()):
    c0 = run.mag()[..., 0]
    show_slice(axes[0, col], c0, K, name, vmin=0, vmax=7000)
    show_slice(axes[1, col], c0, K, "background window (0-400)", vmin=0, vmax=400)
    axes[1, col].contour(plotting.take_slice(ghost_zone.astype(float), K), levels=[0.5], colors=[PALETTE[1]], linewidths=0.6)
    print(f"{name:28s}: signal in the ghost zone {100 * c0[ghost_zone].mean() / c0[mask_ref].mean():4.1f} % of the brain mean; background SD {c0[background].std():6.1f}")
fig.tight_layout()
```

**Partial Fourier** (second column) is invisible at the normal window and nearly so in the
background window: the missing high-frequency lines soften edges along the phase-encode
axis (vertical here) by a fraction of a voxel. It is a deliberate trade for a shorter
echo time, and the only remedy is to know it is there: a difference image made from
blurred controls and labels is itself blurred, which mixes tissues a little more than the
voxel size already does ([Chapter 12](./12-partial-volume.md)). **The ghost** (third
column) appears in the background window as two faint copies of the brain above and
below it, where the orange outline marks half a field of view away: the signal there is
5.2 % of the brain's mean against 1.0 % in the reference (the reference's 1 % is the
Rician noise floor). Scanners calibrate it out with a reference scan, or estimate the
phase correction from the image itself {cite:p}`buonocore1997`, and a residual of a
percent is normal; a larger ghost is a hardware or calibration fault that can only be
detected, by measuring the background as here, and the run excluded or repeated. **The
spikes** (fourth column) are the loudest: each spike is one bright k-space sample, a plane
wave across the whole slice, so two spikes give two sets of stripes over everything and
the background SD is 5,200 image units where the reference's is 31. A spike hits one
volume and one slice at a time; it is found by looking at the images or the raw k-space,
and the volume is discarded.

The question for ASL is whether these subtract out. A ghost of the control looks like a
ghost of the label, and the two differ by the same 1 % as the images they copy. The
figure tests that argument.

```{code-cell} python
:tags: [hide-input]
fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.7))
for ax, (name, run) in zip(axes, runs.items()):
    d = quant.subtract(run.mag(), run.context())
    dm = d.mean(-1)
    vmax = 45 if "spike" not in name else 3000
    show_slice(ax, dm, K, f"mean ΔM: {name}", kind="diff", vmin=-vmax, vmax=vmax)
    print(f"{name:28s}: mean ΔM in GM {dm[fr_ref['gm'] >= 0.9].mean():6.1f}, SD of the mean ΔM outside the head {dm[~mask_ref].std():7.1f}, "
          f"in the ghost zone {dm[ghost_zone].std():7.1f}; per-pair SD in GM {d[fr_ref['gm'] >= 0.9].std():6.1f}")
fig.tight_layout()
```

The reference and partial-Fourier difference images are the same map to the eye, and the
numbers nearly agree: a gray-matter ΔM of 30.2 against 29.4, the blur having averaged a
little white matter into the cortex, and a per-pair SD in gray matter of 50 against 57,
because the zero-filled lines carry no noise. The ghost run's difference image has a
faint ghost of the *perfusion* map in the background (the SD of the mean difference is
14.1 in the ghost zone against 12.1 outside the head; 7.6 against 7.9 in the reference),
because the ghost copies the control and the label separately and their difference is
5 % of the perfusion signal; the ghost that lands on the brain is a 5 % error in ΔM there.
The spikes do not subtract out at all: every volume has spikes of its own at its own
k-space positions, so the difference of two spiked images is two sets of stripes, and the
mean over 15 pairs is stripes with a per-pair SD in gray matter of 7,600 against 57 in
the reference. Nothing downstream can repair that; the run is lost unless the spiked
volumes are few enough to drop.

## Part two: the assembled pipeline

The `kitchen-sink` dataset is the reference protocol with everything on: the phantom with
its field map, a TE of 20 ms and a total readout time of 35 ms (distortion of up to four
voxels, [Chapter 11](./11-susceptibility-distortion.md)), eight coils with GRAPPA 2
{cite:p}`griswold2002`, partial Fourier 6/8, a 2 % ghost, two background-suppression pulses at 2.25 and 3.50 s
([Chapter 9](./09-background-suppression.md)), random head motion on six volumes
([Chapter 10](./10-motion.md)), and noise. It comes as two runs, `ap` and `pa`, with
opposite phase-encode polarity and the same motion, each with its own unsuppressed M0
scan. The truth is `perfusion` on the undistorted, unmoved grid.

Each step is scored the same way: the CBF map the pipeline would produce at that point,
against the truth, as bias, RMSE and median absolute error in gray matter (voxels at
least half gray). The RMSE is sensitive to a few voxels with enormous errors, the median
to the typical voxel, and the pair says which kind of error a step removed. The two edge
slices of the slab are left out of every score and of the registration, because a head
that moves through-plane carries signal out of a 2D slab that nothing brings back.

### Step 1: look at the data

```{code-cell} python
:tags: [hide-input]
ks = data.load_dataset("kitchen-sink")
ap, pa = ks.run("ap"), ks.run("pa")
p, ctx, sim = ap.sidecar(), ap.context(), ap.simulation()
mag_ap, phase_ap = ap.mag(), ap.phase()
fr, mask, truth = ap.fractions(), ap.mask(), ap.truth("perfusion")
VOX = np.array(p["AcquisitionVoxelSize"], float)
CENTRE = (np.array(mag_ap.shape[:3]) - 1) / 2
interior = np.zeros(mask.shape, bool); interior[:, :, 1:-1] = True
gm_score = (fr["gm"] >= 0.5) & interior

keys = ["ArterialSpinLabelingType", "LabelingDuration", "PostLabelingDelay", "EchoTime", "TotalReadoutTime", "PhaseEncodingDirection",
        "ParallelReductionFactorInPlane", "PartialFourier", "BackgroundSuppression", "BackgroundSuppressionPulseTime", "M0Type", "TotalAcquiredPairs"]
print(json.dumps({k: p[k] for k in keys}))
print(f"AslscanSimulation: {sim['Acquisition']['NCoils']} coils, ghost {sim['Acquisition']['GhostOffset']}, motion mode {sim['Motion']['Mode']}, "
      f"suppression label factor {sim['BackgroundSuppressionLabelFactor']:.2f}")
print(f"series {mag_ap.shape}, rows {ctx[:2]}...; M0 scan TR {ap.m0scan_sidecar()['RepetitionTimePreparation']} s, phase-encode {ap.m0scan_sidecar()['PhaseEncodingDirection']}")

signed_ap = mag_ap * np.cos(phase_ap)
signed_pa = pa.mag() * np.cos(pa.phase())
fig, axes = plt.subplots(2, 4, figsize=(12.5, 6.4))
for row, z in enumerate((K, 3)):
    show_slice(axes[row, 0], mag_ap[..., 0], z, f"control magnitude, slice {z}", vmin=0, vmax=3000)
    show_slice(axes[row, 1], phase_ap[..., 0], z, f"control phase, slice {z}", kind="phase", vmin=-np.pi, vmax=np.pi)
    show_slice(axes[row, 2], signed_ap[..., 0], z, f"signed control, slice {z}", kind="diff", vmin=-3000, vmax=3000)
    show_slice(axes[row, 3], mag_ap[..., 0] - mag_ap[..., 1], z, f"control − label (magnitude), slice {z}", kind="diff", vmin=-150, vmax=150)
fig.tight_layout()
```

The sidecar says what to expect: PCASL at the reference timing, but with
`BackgroundSuppression` on and its pulse times, a 20 ms echo time, a 35 ms readout,
GRAPPA 2, partial Fourier, and phase encoding `j-`. The simulation block adds the eight
coils, the 2 % ghost, the random motion, and a suppression label factor of 0.81: two
pulses of efficiency 0.95 leave $(1 - 2 \cdot 0.95)^2$ of the label
([Chapter 9](./09-background-suppression.md)).

The images say something the sidecar does not. At the display slice the control looks
like a suppressed brain: gray matter at about 600 image units instead of 6,200, CSF
brighter than tissue. At slice 3 the magnitude looks similar, but the phase is π across
the tissue, which means the tissue's longitudinal magnetization was *negative* when it
was read out: the pulses inverted it and it had not yet recovered through zero. The
signed image (magnitude times the cosine of the phase, the real part) shows it directly,
and the last column shows the consequence: at slice 3 the magnitude difference
control − label is negative in gray matter, because subtracting the label from a negative
tissue signal makes the magnitude larger, not smaller. A pipeline that subtracts
magnitudes gets the sign of perfusion wrong in the lower third of this slab.

```{code-cell} python
:tags: [hide-input]
t_read = p["PostLabelingDelay"] + p["LabelingDuration"] + np.array(p["SliceTiming"])
timeline = {name: np.array([kinetic.tissue_mz(t.m0, t.t1, p["RepetitionTimePreparation"], tr_, p["BackgroundSuppressionPulseTime"], 0.95) / t.m0
                            for tr_ in t_read]) for name, t in presets.TISSUES.items()}
d_mag = quant.subtract(mag_ap, ctx).mean(-1)
d_sig = quant.subtract(signed_ap, ctx).mean(-1)
gm8 = fr["gm"] >= 0.8
zs = np.arange(1, 19)          # the interior slices; the edge slices lose signal to through-plane motion
per_slice = lambda d: [d[:, :, z][gm8[:, :, z]].mean() for z in zs]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.4))
for name, mz in timeline.items():
    ax1.plot(range(20), mz, "-o", ms=3, color=TISSUE_COLORS[name], label=name)
ax1.axhline(0, color=INK["secondary"], lw=1)
ax1.set(xlabel="slice", ylabel="tissue Mz / M0 at readout", title="the suppression timeline, slice by slice")
ax1.legend()
ax2.plot(zs, per_slice(d_mag), "-o", ms=3, color=PALETTE[7], label="control − label, magnitude")
ax2.plot(zs, per_slice(d_sig), "-o", ms=3, color=PALETTE[0], label="control − label, signed (real part)")
ax2.axhline(0, color=INK["secondary"], lw=1)
ax2.set(xlabel="slice", ylabel="mean ΔM in GM (image units)", title="the measured difference, slice by slice")
ax2.legend()
fig.tight_layout()
zero_gm = int(np.argmax(timeline["GM"] > 0))
print(f"GM crosses zero at slice {zero_gm} ({t_read[zero_gm]:.2f} s after labeling starts); WM at slice {int(np.argmax(timeline['WM'] > 0))}")
print(f"mean ΔM in GM over the interior slices: magnitude subtraction {d_mag[gm8 & interior].mean():5.1f}, signed subtraction {d_sig[gm8 & interior].mean():5.1f} image units")
m0_raw = ap.m0scan()
score_step("magnitude subtraction, raw M0 scan, nothing else", cbf_from(d_mag, m0_raw, p))
cbf_signed = score_step("signed subtraction (the real part)", cbf_from(d_sig, m0_raw, p))
```

The timeline from the sidecar (left) predicts it: with the readout starting 3.6 s after
labeling and slices 40 ms apart, gray matter is still inverted for the first six slices
and crosses zero at slice 6 (3.84 s after the start of labeling), white matter at slice 7,
while CSF, with its 3 s T1, is positive throughout. The measured differences (right, for
the interior slices) follow the prediction: the magnitude subtraction gives −25 in gray
matter for the first slices and +22 from slice 6 on, the signed subtraction +25 falling
to +16 across the slab, the decay of the label with the slice delay. The magnitude mean
over the interior is 3.4 image units, the signed one 21.7. On a scanner the phase is
usually discarded and the difference has no sign to recover; the timing must then be
chosen so that every slice is read out with the tissue on the same side of zero, which is
the first item of the closing list. Here the phase was kept, and the pipeline uses the
signed images from now on. The first two rows of the score table show what the choice is
worth: a gray-matter bias of −39 ml/100 g/min with a median error of 40 becomes a bias
of −3 with a median error of 20, the rest being noise.

### Step 2: motion correction

Every volume is registered by a rigid transform with six parameters, estimated by
Gauss-Newton least squares on the interior voxels ([Chapter 10](./10-motion.md)); the
transform undoes a head movement of three rotations about the volume's centre and three
translations, in the convention the simulator records in `desc-motion_gt.tsv`. Controls
are registered to the mean control and labels to the mean label. That detail matters
more under background suppression than anywhere else: the CSF is unsuppressed and several
times brighter than the tissue, so the intensity steps at its borders are enormous
compared with the perfusion signal, and a label registered against a control reference
is pulled by the very contrast the subtraction is after. The bias is a few hundredths of
a voxel, and it costs a tenth of ΔM. The `pa` run shares the poses by construction (same
seed) and is registered the same way.

```{code-cell} python
:tags: [hide-input]
reg_where = mask & interior

def motion_correct(signed):
    refs = {"control": signed[..., ::2].mean(-1), "label": signed[..., 1::2].mean(-1)}
    poses = np.array([register(signed[..., v], refs[ctx[v]], reg_where) for v in range(signed.shape[-1])])
    return poses, np.stack([rigid(signed[..., v], poses[v]) for v in range(signed.shape[-1])], axis=-1)

poses_ap, moco_ap = motion_correct(signed_ap)
poses_pa, moco_pa = motion_correct(signed_pa)
true_poses = ap.motion().values[:, 1:].astype(float)
moved = [int(v) for v in np.flatnonzero(np.abs(true_poses).sum(1) > 0)]
vols = np.arange(signed_ap.shape[-1])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.4), sharex=True)
for i, lab in enumerate("xyz"):
    ax1.plot(vols, true_poses[:, i], color=PALETTE[i], lw=1, alpha=0.5, label=f"true {lab}")
    ax1.plot(vols, poses_ap[:, i], "o", ms=3, color=PALETTE[i], label=f"estimated {lab}")
    ax2.plot(vols, np.degrees(true_poses[:, 3 + i]), color=PALETTE[i], lw=1, alpha=0.5)
    ax2.plot(vols, np.degrees(poses_ap[:, 3 + i]), "o", ms=3, color=PALETTE[i])
ax1.set(xlabel="volume", ylabel="translation (mm)", title="translations: recorded pose and estimate")
ax2.set(xlabel="volume", ylabel="rotation (degrees)", title="rotations")
ax1.legend(ncol=2, fontsize=7)
fig.tight_layout()

err = poses_ap - true_poses
still = np.setdiff1d(vols, moved)
print(f"moved volumes {moved}: RMS error of the estimate {np.sqrt((err[moved, :3] ** 2).mean()):.2f} mm in translation "
      f"({', '.join(f'{lab} {np.sqrt((err[moved, i] ** 2).mean()):.2f}' for i, lab in enumerate('xyz'))}), "
      f"{np.degrees(np.sqrt((err[moved, 3:] ** 2).mean())):.2f} degrees in rotation")
print(f"still volumes: RMS estimate {np.sqrt((poses_ap[still, :3] ** 2).mean()):.2f} mm, {np.degrees(np.sqrt((poses_ap[still, 3:] ** 2).mean())):.2f} degrees (should be zero)")
fdisp = fd(poses_ap)
print(f"framewise displacement: mean {fdisp.mean():.2f} mm, max {fdisp.max():.2f} mm at volume {fdisp.argmax()}; "
      f"volumes above 0.5 mm: {[int(v) for v in np.flatnonzero(fdisp > 0.5)]}")
ref_ap = signed_ap[..., ::2].mean(-1)
print(f"residual of volume 41 against the mean control in the interior: before {np.abs(signed_ap[..., 41] - ref_ap)[reg_where].mean():.0f}, "
      f"after {np.abs(moco_ap[..., 41] - ref_ap)[reg_where].mean():.0f}, a still volume {np.abs(signed_ap[..., 3] - ref_ap)[reg_where].mean():.0f} image units")
d_moco = quant.subtract(moco_ap, ctx)
cbf_moco = score_step("+ motion correction (ap)", cbf_from(d_moco.mean(-1), m0_raw, p))
```

The estimates land on the recorded poses for the six moved volumes: the translations
agree within 0.1 mm in-plane and 0.3 mm through-plane, the rotations within a tenth of a
degree. The through-plane translation is the least certain, as it must be with 5 mm
slices and a 100 mm slab, and the still volumes are estimated as still to within 0.06 mm
and 0.04°. The framewise displacement trace {cite:p}`power2012`, the motion summary a
pipeline reports, is above 0.5 mm only at the moved volumes and the ones after them (a jump back is a
displacement too). Volume 41, which moved by 1.8 mm and 1°, differs from the mean
control by 92 image units in the interior before registration, the edge pattern of
[Chapter 10](./10-motion.md), and by 50 after, against 34 for a still volume; the
remainder is the interpolation of a noisy image. In the table, motion correction moves
the gray-matter RMSE from 59.2 to 56.9 and the median error from 20.3 to 19.2: four
corrupted pairs out of thirty are a small part of a noisy average, and at this noise
level the map is limited by noise, not by motion.

### Step 3: outlier rejection

Registration undoes the displacement but not everything: the moved volumes were
interpolated, which smooths them, and any within-volume or through-plane effect remains.
The rule used here is the first pass of the SCORE algorithm {cite:p}`dolui2017` with a
wider threshold: it takes each pair's mean difference in gray matter and discards the
pairs that deviate from the median by more than three robust standard deviations (1.4826
times the median absolute deviation), where SCORE takes the mean gray matter CBF and a
threshold of 2.5. SCORE's second pass, which goes on removing the volume most correlated
with the mean CBF map for as long as the within-tissue variance of that map falls, and
SCRUB {cite:p}`dolui2016`, which then down-weights outlying values voxel by voxel instead
of discarding volumes, are run on the `motion` dataset in [Chapter 10](./10-motion.md).

```{code-cell} python
:tags: [hide-input]
def outlier_pairs(diff, roi, n_sd=3.0):
    m = np.array([diff[..., i][roi].mean() for i in range(diff.shape[-1])])
    z = (m - np.median(m)) / (1.4826 * np.median(np.abs(m - np.median(m))))
    return m, np.abs(z) > n_sd

gm_pair, reject_ap = outlier_pairs(d_moco, gm_score)
d_pa = quant.subtract(moco_pa, pa.context())
_, reject_pa = outlier_pairs(d_pa, gm_score)
gm_pair_raw, _ = outlier_pairs(quant.subtract(signed_ap, ctx), gm_score)

fig, ax = plt.subplots(figsize=(8, 3.2))
pairs = np.arange(len(gm_pair))
ax.plot(pairs, gm_pair_raw, "o", ms=4, color=INK["secondary"], alpha=0.5, label="before motion correction")
ax.plot(pairs, gm_pair, "o", ms=4, color=PALETTE[0], label="after motion correction")
ax.plot(pairs[reject_ap], gm_pair[reject_ap], "x", ms=9, color=PALETTE[7], label="rejected")
thr = 3.0 * 1.4826 * np.median(np.abs(gm_pair - np.median(gm_pair)))
ax.axhspan(np.median(gm_pair) - thr, np.median(gm_pair) + thr, color=PALETTE[0], alpha=0.1, lw=0)
ax.set(xlabel="pair", ylabel="mean ΔM in GM (image units)", title="per-pair gray-matter difference and the rejection band")
ax.legend(loc="upper right", fontsize=7)
fig.tight_layout()
print(f"ap: rejected pairs {[int(i) for i in np.flatnonzero(reject_ap)]} (pairs with a moved volume: {sorted({v // 2 for v in moved})}); "
      f"pa: rejected {[int(i) for i in np.flatnonzero(reject_pa)]}")
dm_ap = d_moco[..., ~reject_ap].mean(-1)
dm_pa = d_pa[..., ~reject_pa].mean(-1)
cbf_clean = score_step(f"+ outlier rejection ({(~reject_ap).sum()} pairs kept)", cbf_from(dm_ap, m0_raw, p))
```

Before motion correction the four pairs with a moved volume are off the band, at 15 or
26 image units against 19.5 for the rest, because edge signal of hundreds of units leaks
into the difference and does not average to zero over the cortex. After registration
three are still outside three robust standard deviations, which are narrow because the
still pairs agree to within 0.3 units; the fourth, pair 20, whose two volumes both moved,
lands inside the band in `ap` and outside it in `pa`. The rule rejects pairs 2, 6 and 13
in `ap` and those plus 20 in `pa`. The table shows what that buys here: nothing. The RMSE
rises from 56.9 to 65.3 and the median error from 19.2 to 19.7, because registration had
already repaired the pairs and the remaining error is noise, which grows when 10 % of
the averaging is thrown away; the RMSE grows more than the median because the frontal
voxels, where the distortion has stretched the M0 scan to almost nothing, amplify noise
enormously (the next step deals with them). Rejection earns its place with the pairs
registration cannot fix, such as motion during a readout or a head that left the slab,
and it must be judged by what it removes, not applied by reflex.

### Step 4: distortion correction

The field map moves signal along the phase-encode axis by field × total readout time, in
opposite directions in the `ap` and `pa` runs ([Chapter 11](./11-susceptibility-distortion.md)).
The pipeline here has the luxury of the true field: the phantom's field map, box-averaged
to the acquisition grid exactly as the simulator averages everything else. In a study it
would come from a field-map scan {cite:p}`jezzard1995` or be estimated from the
blip-up/blip-down pair, which is what FSL's `topup` {cite:p}`andersson2003,smith2004` and
the pipelines that wrap it do. The sign of the displacement is not a convention
to remember but a thing to check, so the cell fixes it against the `sdc` dataset's
undistorted run, then unwarps each run's mean difference image and M0 scan column by
column with a Jacobian and averages the two runs.

```{code-cell} python
:tags: [hide-input]
field = phantom.slab()["fieldmap"].astype(float)   # the 1 mm field box-averaged to the acquisition grid, packaged with the book
disp = field * p["TotalReadoutTime"]        # voxels along y, sign to be fixed

sdc = data.load_dataset("sdc")
u, a = sdc.run("undistorted").mag()[..., 0], sdc.run("ap").mag()[..., 0]
errs = {s: np.abs(unwarp(a, s * disp) - u)[mask].mean() for s in (+1, -1)}
SIGN_AP = min(errs, key=errs.get)
print(f"unwarp error of the sdc ap control against the undistorted run: sign +1 {errs[1]:.0f}, sign -1 {errs[-1]:.0f}, uncorrected {np.abs(a - u)[mask].mean():.0f} -> "
      f"displacement for j- is {SIGN_AP:+d} x field x readout time; largest {np.abs(disp)[mask].max():.1f} voxels")

dm_sdc = 0.5 * (unwarp(dm_ap, SIGN_AP * disp) + unwarp(dm_pa, -SIGN_AP * disp))
m0_sdc = 0.5 * (unwarp(m0_raw, SIGN_AP * disp) + unwarp(pa.m0scan(), -SIGN_AP * disp))
dm_gt = ap.truth("deltamStatic")[..., 1] * 100 * sim["BackgroundSuppressionLabelFactor"] * np.exp(-p["EchoTime"] / presets.T2_BLOOD)

z = 2
fig, axes = plt.subplots(1, 5, figsize=(15, 3.7))
show_slice(axes[0], field, z, f"field (Hz), slice {z}", kind="diff", vmin=-120, vmax=120)
show_slice(axes[1], dm_ap, z, "mean ΔM, ap (j-)", kind="diff", vmin=-45, vmax=45)
show_slice(axes[2], dm_pa, z, "mean ΔM, pa (j)", kind="diff", vmin=-45, vmax=45)
show_slice(axes[3], dm_sdc, z, "both unwarped and averaged", kind="diff", vmin=-45, vmax=45)
show_slice(axes[4], dm_gt, z, "truth ΔM (static, scaled)", kind="diff", vmin=-45, vmax=45)
fig.tight_layout()
low = mask & (np.abs(disp) > 1.0) & interior
for name, dmap in (("ap alone", dm_ap), ("ap and pa averaged, no unwarp", 0.5 * (dm_ap + dm_pa)), ("both unwarped and averaged", dm_sdc)):
    print(f"{name:32s}: RMSE against the truth ΔM in the {low.sum()} voxels displaced by more than one voxel {np.sqrt(((dmap - dm_gt) ** 2)[low].mean()):5.1f}")
print(f"M0 scan of the ap run in gray matter: smallest {m0_raw[gm_score].min():.0f}, median {np.median(m0_raw[gm_score]):.0f} image units")
cbf_avg = score_step("+ ap and pa averaged without unwarping", cbf_from(0.5 * (dm_ap + dm_pa), 0.5 * (m0_raw + pa.m0scan()), p))
cbf_sdc = score_step("+ distortion correction (field map, ap and pa averaged)", cbf_from(dm_sdc, m0_sdc, p))
```

The check against the `sdc` dataset says the displacement for `j-` is −1 × field ×
readout time (an unwarp error of 29 image units with that sign, 364 with the other, 203
uncorrected), up to 4.0 voxels in this slab. At slice 2 the field reaches 100 Hz over the
frontal sinuses, and the two runs show the frontal cortex piled up and stretched in
opposite directions. Averaging them without unwarping smears the cortex in both
directions but halves the displacement error; unwarping each with the field and then
averaging brings the frontal gray matter back to where the truth has it. In the 1,464
voxels displaced by more than one voxel the RMSE against the truth difference image is
17.7 for the `ap` run alone, 13.2 for the plain average and 12.9 for the unwarped
average. The larger effect in the CBF table comes from the calibration image: in the `ap`
run the M0 scan is stretched to almost nothing over the sinuses (its smallest gray-matter
value is 43 image units against a median of 5,800), and dividing by it there turns noise
into CBF values of hundreds. The `pa` scan is compressed in the same voxels, the average
of the two has no empty voxels, and the gray-matter RMSE falls from 65 to 33; the unwarp
then takes it to 31.5 and the median error from 16.4 to 15.3. Averaging the two runs also
doubles the pairs, which lowers the noise everywhere. The M0 scan is unwarped with the
same field and the same sign as its series, and that matters: a distorted M0 dividing an
undistorted difference would put the frontal calibration in the wrong voxels.

### Steps 5 to 7: the suppression factor, the calibration, and partial volume

Two multiplicative corrections remain. Background suppression left 0.81 of the label, so
the difference is divided by the label factor from the sidecar's simulation block (in a
study, computed from the pulses' efficiency, or measured). The M0 scan is then corrected
for its own saturation at TR 8 s and for the difference between the tissue's T2 and the
blood's at the 20 ms echo time ([Chapter 16](../04-quantification/16-calibration.md)),
and the white-paper formula {cite:p}`alsop2015` with each slice's delay does the rest. Last, because it needs
a map from which every other error has been removed and fractions on the same,
undistorted grid, comes the linear regression of {cite:t}`asllani2008` with a 5 × 5 kernel
([Chapter 12](./12-partial-volume.md)).

```{code-cell} python
:tags: [hide-input]
lf = sim["BackgroundSuppressionLabelFactor"]
cbf_lf = score_step(f"+ divided by the suppression label factor {lf:.2f}", cbf_from(dm_sdc / lf, m0_sdc, p))
m0_cal = quant.m0_correction(m0_sdc, tr=ap.m0scan_sidecar()["RepetitionTimePreparation"], te=p["EchoTime"])
cbf_cal = score_step("+ M0 corrected for TR saturation and T2 at TE", cbf_from(dm_sdc / lf, m0_cal, p))
f_gm, f_wm = quant.pv_correct(cbf_cal, fr["gm"], fr["wm"], kernel=5, mask=mask)
recon = fr["gm"] * f_gm + fr["wm"] * f_wm
score_step("+ partial-volume correction (5 × 5), reconstructed", recon)
ok = gm_score & (f_gm != 0)
print(f"pure gray-matter CBF in voxels >= 50 % GM: mean {f_gm[ok].mean():.1f}, SD {f_gm[ok].std():.1f} (truth 60); "
      f"calibrated map in the same voxels: mean {cbf_cal[gm_score].mean():.1f}, truth {truth[gm_score].mean():.1f}")
fig, axes = plotting.fit_vs_truth(np.where(interior, cbf_cal, np.nan), truth, mask & interior, "CBF, calibrated", k=K, unit="(ml/100 g/min)")
```

The label factor is the largest single correction in the table: dividing by 0.81 raises
every value by 23 % and takes the gray-matter bias from −4.6 to +7.0. The M0 correction
goes the other way: at a 20 ms echo time the tissue in the M0 scan has decayed more than
the blood in the difference image (T2 80 ms against 165 ms), so the M0 is 14 % too small
and the CBF 14 % too large until the two are put on the same footing, and the bias
settles at −0.5 with a median error of 15.8. What remains is noise (a per-voxel RMSE of
34 at this noise level, with eight coils and GRAPPA) and the white-matter kinetics of
[Chapter 12](./12-partial-volume.md): the 4-panel shows a speckled cortex around 60, the
white matter dark, and a scatter centered on the identity line at 60 and below it at 20.
The corrected gray-matter map has a mean of 60.4 in the voxels that are at least half
gray (truth 60), against 53.1 for the calibrated map in the same voxels (truth 53.6, the
weighted mean), and the reconstruction's RMSE of 13.4 and median error of 5.5 are far
below the calibrated map's because the regression averages 25 voxels.

```{code-cell} python
:tags: [hide-input]
steps = ["magnitude subtraction", "signed", "+ motion", "+ outliers", "+ ap/pa average", "+ unwarp", "+ label factor", "+ M0 correction", "+ PV"]
maps = [cbf_from(d_mag, m0_raw, p), cbf_signed, cbf_moco, cbf_clean, cbf_avg, cbf_sdc, cbf_lf, cbf_cal, recon]
fig, axes = plt.subplots(2, 5, figsize=(14.5, 6.2))
for ax, name, m in zip(axes.ravel(), steps, maps):
    show_slice(ax, np.where(mask, m, np.nan), K, name, kind="cbf")
show_slice(axes[1, 4], np.where(mask, truth, np.nan), K, "truth", kind="cbf")
fig.tight_layout()
```

The mosaic tells the whole story at one slice: the first map is dark and speckled, the
sign fix restores contrast, motion correction and rejection change little at this slice,
the `ap`/`pa` average and the unwarp halve the speckle, the two multiplicative steps
raise and then trim the whole map, the partial-volume reconstruction is a smooth version
of the truth, and the last panel is the truth.

### Step 8: the quality-control numbers

```{code-cell} python
:tags: [hide-input]
d_kept = d_moco[..., ~reject_ap]
tsnr = quant.tsnr(d_kept)
print(f"temporal SNR of the difference series (ap, {d_kept.shape[-1]} pairs): median in GM {np.median(tsnr[gm_score]):.2f}, in WM {np.median(tsnr[(fr['wm'] >= 0.9) & interior]):.2f}")
print(f"negative CBF voxels in the slab: {100 * (cbf_cal[mask & interior] < 0).mean():.1f} %")
print(f"GM/WM ratio (voxels >= 90 % of each tissue): {cbf_cal[(fr['gm'] >= 0.9) & interior].mean() / cbf_cal[(fr['wm'] >= 0.9) & interior].mean():.1f}")
print(f"motion: mean FD {fdisp.mean():.2f} mm, max FD {fdisp.max():.2f} mm, {len(moved)} volumes moved, {reject_ap.sum()} of {len(reject_ap)} pairs rejected")
print(f"\n{'step':58s} {'GM bias':>8s} {'GM RMSE':>8s} {'median |error|':>15s}")
for name, b, r, s in TABLE:
    print(f"{name:58s} {b:8.1f} {r:8.1f} {s:15.1f}")
```

These four numbers, with the per-volume framewise displacement, are what a report should
show first. The temporal SNR of the difference series is 0.41 in gray matter and 0.08 in
white matter: a single pair's difference is well below its own noise, which is normal for
ASL and the reason for 30 pairs. The negative-voxel fraction, 11.5 % of the slab, says how
much of the map is below the noise floor and is dominated by white matter and CSF. The
GM/WM ratio of 5.1 is to be compared with the phantom's pure-tissue ratio of 3 (60
against 20 ml/100 g/min); it is high here for the reason
[Chapter 12](./12-partial-volume.md) gave, the white-matter kinetics. The motion summary,
a mean framewise displacement of 0.91 mm with a maximum of 6.6 mm, six moved volumes and
three rejected pairs, decides whether the subject is kept and enters the group analysis
as a covariate. ASLPrep condenses evidence of the same kind into one number per CBF map,
the quality evaluation index (QEI), built from the structural similarity between the CBF
map and the tissue maps, the spatial variability of the map, and the fraction of gray
matter voxels with negative CBF {cite:p}`dolui2017qei,adebimpe2022`.

## Measure it: the error after each step

The table printed above is the chapter's result, in ml/100 g/min against the truth in
gray matter (voxels at least half gray, interior slices):

| Step | GM bias | GM RMSE | GM median error |
|---|---|---|---|
| Magnitude subtraction, raw M0 scan | −38.6 | 78.2 | 40.4 |
| Signed subtraction (the real part) | −3.2 | 59.2 | 20.3 |
| + motion correction | −2.5 | 56.9 | 19.2 |
| + outlier rejection (27 pairs kept) | −4.7 | 65.3 | 19.7 |
| + `ap` and `pa` averaged, no unwarp | −4.7 | 33.0 | 16.4 |
| + distortion correction (field map, both runs) | −4.6 | 31.5 | 15.3 |
| + suppression label factor 0.81 | +7.0 | 39.0 | 18.9 |
| + M0 corrected for TR and T2 | −0.5 | 33.7 | 15.8 |
| + partial-volume correction (5 × 5 reconstruction) | −0.6 | 13.4 | 5.5 |

Three steps carry most of the table. The sign fix is worth 36 units of bias; without it
the map is wrong in kind, not in degree. The distortion step, mostly through the
calibration image and the doubled averaging, halves the RMSE. The two multiplicative
corrections are together worth 12 % of the final value and cost nothing to apply, which
makes them the easiest to get wrong in a study whose sidecar does not record the
suppression or the M0 scan's timing. Motion correction and rejection are small here
because the motion was small (six volumes, at most 2 mm) and the noise large; on a
quieter series or a more restless subject they would dominate. Partial-volume correction
lowers the error more than any other step, but as [Chapter 12](./12-partial-volume.md)
showed, most of that is the smoothing of a 25-voxel regression, and its real product is
the pure-tissue value: 60.4 against a truth of 60.

## How the standard pipelines order these steps

| Step | ASLPrep {cite:p}`adebimpe2022` | ExploreASL {cite:p}`mutsaerts2020` | BASIL / oxford_asl {cite:p}`chappell2009` | White paper {cite:p}`alsop2015` |
|---|---|---|---|---|
| Motion correction | whole series, before subtraction (MCFLIRT {cite:p}`jenkinson2002`) | SPM realignment {cite:p}`friston1995` with a regressor for the control-label intensity difference | MCFLIRT before the fit | image registration may be applied; prospective correction where available |
| Outlier rejection | optional: SCORE {cite:p}`dolui2017`, then SCRUB {cite:p}`dolui2016`, on the CBF time series | ENABLE {cite:p}`shirzadi2018`: pairs sorted by their motion and excluded while that improves the temporal SNR | none by default | inspect the individual difference images and exclude the artifactual ones |
| Distortion correction | field map or blip-up/blip-down, through SDCFlows {cite:p}`esteban2019` | `topup` on a reversed-polarity pair, when one was acquired | `--fmap` options, applied to both | not addressed |
| Subtraction, suppression factor | pairwise; efficiency times a suppression term | pairwise; efficiency corrected for suppression | inside the kinetic-model fit | efficiency reduced per pulse |
| Calibration | M0 scan, an M0 value in the metadata, or the control mean; optional scaling and smoothing | M0 scan, masked and smoothed, T1 correction for a short TR | voxel-wise or CSF reference, TR and T2 terms | separate M0, TR ≥ 5 s or T1 correction |
| Partial-volume correction | optional, through BASIL | optional, last, regression {cite:p}`asllani2008` | optional, inside the fit, with a spatial prior {cite:p}`chappell2011,groves2009` | not addressed |
| QC | FD, coregistration and normalization overlap, QEI {cite:p}`dolui2017qei` | temporal SNR, spatial CoV, motion | none | visual: gray-white contrast, plausible gray matter CBF, the individual difference images |

The order is the same in all of them, for reasons the steps above made visible.
**Registration comes before subtraction** because a difference image made from
misaligned volumes is an edge map that no later step can undo; once the pairs are
subtracted, the motion has become perfusion. **Distortion correction comes before
calibration** only if the M0 image is corrected the same way, because calibration is a
voxel-wise ratio and the two images must agree on where each voxel is; correcting one
and not the other is worse than correcting neither. **The multiplicative corrections can
go anywhere** after subtraction, since scalars commute, which is why they are the easiest
to forget and the largest in the table. **Partial-volume correction comes last** because
it fits a model to the CBF map with tissue fractions on the undistorted grid, and every
uncorrected artifact in the map is fitted as perfusion.

## What this implies for acquisition

- **Keep the tissue on one side of zero.** With 2D readouts and background suppression,
  choose pulse times so that no slice is read out with an inverted tissue, or accept
  weaker suppression at the first slices; a magnitude-only pipeline cannot recover the
  sign. If the scanner can save the phase, save it.
- **Acquire what the corrections need.** A blip-up/blip-down pair or a field map for the
  distortion; an unsuppressed M0 scan with the same readout, polarity and echo time as
  the series; a structural image for the fractions. A pipeline corrects only what the
  acquisition recorded.
- **Record the label efficiency and the suppression pulses in the sidecar.** The largest
  correction in the table is a number that has to come from the protocol, and ASL-BIDS has
  fields for both {cite:p}`clement2022`.
- **Look at the background.** Ghosts and spikes are found by windowing the images to the
  noise floor, not by any subtraction; a control image's background is the cheapest QC
  there is.
- **Plan for rejected pairs.** Three of thirty pairs went here; the scan should be long
  enough that the survivors still meet the SNR the study needs.

## Further reading

The pipelines: ASLPrep {cite:p}`adebimpe2022`, ExploreASL {cite:p}`mutsaerts2020`, BASIL
{cite:p}`chappell2009`; the consensus paper {cite:p}`alsop2015`. Outlier rejection by
SCORE {cite:p}`dolui2017` and the voxel-wise robust estimate that follows it, SCRUB
{cite:p}`dolui2016`, or by motion-sorted exclusion, ENABLE {cite:p}`shirzadi2018`; the
quality evaluation index {cite:p}`dolui2017qei`; framewise
displacement {cite:p}`power2012`; ghost correction from the image phase
{cite:p}`buonocore1997`; the reverse-polarity distortion estimate {cite:p}`andersson2003`
and the SDCFlows workflows that grew out of fMRIPrep {cite:p}`esteban2019`; the
partial-volume regression {cite:p}`asllani2008` and the spatial-prior alternative
{cite:p}`chappell2011,groves2009`; the GRAPPA reconstruction whose noise structure the
kitchen sink inherits {cite:p}`griswold2002`.
