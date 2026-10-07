---
title: "2. Spatial encoding, EPI, and reconstruction"
kernelspec:
  name: python3
  display_name: Python 3
---

:::{admonition} Simulated datasets in this chapter
:class: note
- **Built in this page:** the k-space of a 1 mm control-image slice built from the packaged label map, its reconstruction at the acquisition matrix, an EPI timing diagram, and Rician noise draws ([Appendix B](../appendices/b-data-manifest.md#app-b-package-data)).
- **`ref-pcasl`**: the reference PCASL run: one control image, its phase, the M0 scan, and the sidecar's readout metadata ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-ref-pcasl)).
- **`ref-clean`**: the same run with the noise switched off, used once to show what the "empty" background holds ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-ref-clean)).
- **`readout-3d`**: the same labeling and timing read with 3D GRASE in one shot (`grase-1shot`), in four shots (`grase-4shot`), in four shots with 130° refocusing (`grase-4shot-130`), and with a stack of spirals (`spiral`): their control images, the sidecars' echo amplitudes and trajectory, and the noise per volume ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-readout-3d)).

Pipeline-tier datasets are simulated offline by aslscan ([Chapter 0.2](../00-frontmatter/the-simulated-datasets.md)).
:::

## Learning goals

After this chapter you can:

- explain what k-space is and why the scanner records the Fourier transform of the image
  rather than the image itself, and relate the field of view and voxel size to how k-space is
  sampled
- describe the single-shot EPI readout, compute its timing from the sidecar, and state why
  an off-resonance displaces the image along the phase-encode direction
- say in one paragraph each what partial Fourier, multiple receive coils, and GRAPPA change
- explain how Gaussian noise in k-space becomes Rician noise in a magnitude image, and
  measure the noise level of a series from its background
- recognize Gibbs ringing and say why the simulator builds its object on a finer grid than
  the image
- describe a 3D GRASE readout as one excitation of the slab and one partition per spin echo,
  read the echo amplitudes of the sidecar as a $k_z$ window, and turn it into the
  through-plane point-spread function and the signal it costs
- say how a stack of spirals differs from an EPI block in its trajectory, its artifacts and
  its reconstruction, and do the per-volume and per-unit-time noise accounting of a
  segmented 3D readout

```{code-cell} python
:tags: [hide-cell]
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, stats

from aslbook import data, grid, phantom, presets
from aslbook.plotting import INK, PALETTE, TISSUE_COLORS, set_style, show_image, show_slice, take_slice

set_style()
r = presets.REFERENCE
NX, NY = r.matrix                      # 64 x 68 acquisition matrix (x: readout, y: phase encode)
DX = r.voxel_mm[0]                     # 3.5 mm
FOV = (NX * DX, NY * DX)               # 224 x 238 mm

fft2c = lambda a: np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(a)))
ifft2c = lambda k: np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k)))

def show_kspace(ax, k, title=None, **kw):
    """log magnitude of a centered k-space array, oriented like the images (rot90)."""
    return show_image(ax, np.rot90(np.log10(np.abs(k) + 1e-3 * np.abs(k).max())), title, **kw)
```

## Gradients as encoding

Without a gradient every spin in a slice precesses at the same frequency and the receiver
cannot tell where the signal came from. A gradient coil adds a small field that grows
linearly with position along one direction, so the precession frequency depends on position
(the idea from which MR imaging started, {cite:t}`lauterbur1973`)
and, after the gradient has been on for a while, each spin's phase is proportional to its
position: the phase winds across the object like a corkscrew, tighter the longer the gradient
stays on. The receiver still records only the sum over the whole object, as one complex
number. Adding up arrows that point in different directions is the same as multiplying the
object by a striped pattern with the corkscrew's pitch and summing, so one recorded number
says how much of the object varies on that spatial scale. That number is one sample of
**k-space** {cite:p}`ljunggren1983,twieg1983`, and $\mathbf{k}$ is the number of turns per
millimeter, in cycles/mm:

$$s(\mathbf{k}) = \int \rho(\mathbf{r})\, e^{-2\pi i\, \mathbf{k}\cdot\mathbf{r}}\, d\mathbf{r},$$

the Fourier transform of the image $\rho(\mathbf{r})$ ({cite:t}`haacke1999` and
{cite:t}`nishimura2010` give the derivation). A pulse sequence is a plan for moving
through k-space (the gradient history sets $\mathbf{k}$) and recording samples along the
way; reconstruction is the inverse Fourier transform. Along the **readout** direction a
gradient stays on while the receiver samples, and a whole line of k-space is collected in
well under a millisecond. The **phase-encode** direction is stepped between lines by a short
gradient pulse, so it is the slow direction, and the artifacts of Part III act along it.
Slice selection uses the same principle during excitation: a gradient along $z$ makes the RF
pulse resonant only within one slab, which is how the 20 slices of the reference protocol
are excited one after another.

## See it: a control image from k-space

The object below is one axial slice of the simulated brain at the phantom's own 1 mm
resolution, at the level of the display slice, with each tissue class given the
control-image signal of [Chapter 1](../01-mri-physics/01-spins-and-relaxation.md),
$M_0 (1 - e^{-\mathrm{TR}/T_1})\, e^{-\mathrm{TE}/T_2}$ at TR 4.5 s and TE 12 ms, times the
simulator's intensity scale of 100. It is placed in the acquisition field of view (64 × 68
voxels of 3.5 mm: 224 × 238 mm), transformed, and reconstructed from only the central
64 × 68 k-space samples, which is what a 3.5 mm acquisition records: the 1 mm grid's k-space
extends to 0.5 cycles/mm, the acquisition keeps only the samples within 0.143 cycles/mm.

```{code-cell} python
:tags: [hide-input]
fine = phantom.fine_slice()
dseg1 = fine["dseg"]                                              # (x, y) at 1 mm, 197 x 233
signal_of = {n: t.m0 * (1 - np.exp(-r.repetition_time / t.t1)) * np.exp(-r.echo_time / t.t2) * r.signal_scale for n, t in presets.TISSUES.items()}
obj = np.zeros((int(FOV[0]), int(FOV[1])))                        # the 1 mm object inside the acquisition FOV (corner-aligned, like the simulator)
for n, t in presets.TISSUES.items():
    obj[: dseg1.shape[0], : dseg1.shape[1]][dseg1 == t.label] = signal_of[n]
k_fine = fft2c(obj)
c0, c1 = obj.shape[0] // 2, obj.shape[1] // 2
k_acq = k_fine[c0 - NX // 2 : c0 + NX // 2, c1 - NY // 2 : c1 + NY // 2]  # the central 64 x 68 samples
img_acq = ifft2c(k_acq) * (NX * NY) / obj.size                    # scaled so that a uniform region keeps its intensity

fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
show_image(axes[0], np.rot90(obj), "object at 1 mm (control-image signal)", vmin=0, vmax=7000)
show_kspace(axes[1], k_fine, "log |k-space| of the 1 mm object")
axes[1].add_patch(plt.Rectangle((c1 - NY // 2 - 0.5, c0 - NX // 2 - 0.5), NY, NX, fill=False, color=PALETTE[1], lw=1))
show_image(axes[2], np.rot90(np.abs(img_acq)), "central 64 × 68 samples → 3.5 mm image", vmin=0, vmax=7000)
fig.tight_layout()
print("control-image signal per class (image units): " + ", ".join(f"{n} {v:.0f}" for n, v in signal_of.items()))
print(f"k-space sample spacing 1/FOV = {1 / FOV[0]:.4f} cycles/mm; outermost acquired sample 1/(2 x {DX:g} mm) = {1 / (2 * DX):.3f} cycles/mm; "
      f"the 1 mm object's k-space extends to {0.5:.1f} cycles/mm")
```

Left: the object, with gray matter brightest (6205 image units), white matter next, and CSF
darkest because its long T1 leaves it only three quarters recovered at TR 4.5 s. Middle: the
magnitude of its k-space on a log scale; the orange box is the central 64 × 68 samples.
Most of the energy is inside the box: the center of k-space holds the contrast and the
coarse shape, the periphery holds the edges. Right: the image reconstructed from the box
alone, the 3.5 mm control image this slice would give before noise. It already carries the
ringing the next section isolates.

Two numbers describe how k-space is sampled, and each has a characteristic failure. The
**spacing** between samples, $\Delta k = 1/\mathrm{FOV}$, sets the field of view: sampling
too coarsely for the object makes the image wrap around on itself (aliasing). The
**outermost** sample, $k_\mathrm{max} = 1/(2\,\Delta x)$, sets the voxel size: stopping
closer to the center gives larger voxels and softer edges. The matrix is
$N = \mathrm{FOV}/\Delta x$ samples per axis.

```{code-cell} python
:tags: [hide-input]
low = np.zeros_like(k_acq); low[NX // 2 - 16 : NX // 2 + 16, NY // 2 - 17 : NY // 2 + 17] = k_acq[NX // 2 - 16 : NX // 2 + 16, NY // 2 - 17 : NY // 2 + 17]
every_other = np.where((np.arange(NY) % 2 == 0)[None, :], k_acq, 0)
fig, axes = plt.subplots(2, 3, figsize=(9, 6.2))
show_kspace(axes[0, 0], k_acq, "k-space: all 64 × 68 samples")
show_kspace(axes[0, 1], low, "central 32 × 34 kept")
show_kspace(axes[0, 2], every_other, "every other $k_y$ line kept")
show_image(axes[1, 0], np.rot90(np.abs(img_acq)), "image: 3.5 mm voxels", vmin=0, vmax=7000)
show_image(axes[1, 1], np.rot90(np.abs(ifft2c(low) * (NX * NY) / obj.size)), "image: 7 mm voxels", vmin=0, vmax=7000)
show_image(axes[1, 2], np.rot90(np.abs(ifft2c(every_other) * 2 * (NX * NY) / obj.size)), "image: FOV halved, wrapped", vmin=0, vmax=7000)
fig.tight_layout()
```

Keeping only the central half of the samples in each direction (middle column) gives 7 mm
voxels: the same brain, blurred. Keeping every other phase-encode line (right column) halves
the field of view along that axis, and the parts of the head that no longer fit wrap around
to the other side. Parallel imaging {cite:p}`pruessmann1999,griswold2002` deliberately does
the second and undoes the wrap with the receive coils (below).

## Truncation and Gibbs ringing

Every acquisition stops at some outermost sample. A Fourier representation cut off at a
finite frequency overshoots at every sharp edge by about 9 % of the step, and the overshoot
does not shrink with more samples; it only moves closer to the edge. In an image it appears
as ripples parallel to every sharp boundary, and the sharpest boundaries in the brain are
between CSF and tissue and at the brain surface. The simulator's ground truth for a slice is
not this ringing image but the volume-weighted mean of the object over each 3.5 mm voxel,
the box average every truth map and tissue fraction in the book is built on (`aslbook.grid`).
The figure computes that box average of the same 1 mm slice and subtracts it: what remains
is the ringing.

```{code-cell} python
:tags: [hide-input]
rs = grid.BoxResampler(obj.shape + (1,), (1.0, 1.0, 1.0), (NX, NY, 1), (DX, DX, 1.0))
box = rs.mean(obj[:, :, None])[:, :, 0]                          # the same slice box-averaged onto the 3.5 mm grid
trunc = np.abs(img_acq)
diff = trunc - box
brain = box > 0
row = 34                                                          # a left-right profile through the lateral ventricles
fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.4))
show_image(axes[0], np.rot90(trunc), "from the central k-space samples", vmin=0, vmax=7000)
show_image(axes[1], np.rot90(box), "box average (the simulator's truth)", vmin=0, vmax=7000)
show_image(axes[2], np.rot90(diff), "difference: the ringing", kind="diff", vmin=-800, vmax=800)
axes[2].axhline(NY - 1 - row, color=PALETTE[1], lw=0.8)
axes[3].plot(box[:, row], color="0.3", lw=1, ls="--", label="box average")
axes[3].plot(trunc[:, row], color=PALETTE[1], lw=1.5, label="from k-space")
axes[3].set(xlabel="x (voxels)", ylabel="image units", title="profile along the marked row", xlim=(8, 56))
axes[3].legend(loc="lower center")
fig.tight_layout()
deep = ndimage.distance_transform_edt(brain) > 2
print(f"more than 2 voxels inside the brain surface: largest overshoot {diff[deep].max():.0f} image units ({diff[deep].max() / signal_of['GM'] * 100:.0f} % of the GM signal), "
      f"RMS ringing {np.sqrt((diff[deep] ** 2).mean()):.0f} ({np.sqrt((diff[deep] ** 2).mean()) / signal_of['GM'] * 100:.1f} % of GM)")
outside = ndimage.distance_transform_edt(~brain) > 3
print(f"mean |signal| more than 3 voxels outside the brain, where the box average is exactly 0: {trunc[outside].mean():.0f} image units")
```

The two images are nearly identical at a glance, but the difference panel shows alternating
bands of over- and undershoot, one voxel apart, lining every CSF boundary and the brain
surface, and the profile shows the k-space image overshooting the box average on both sides
of each ventricle. Deep inside the brain the largest excursion is about a tenth of the
gray-matter signal and the root-mean-square ringing is 2.6 % of it. Outside the brain, where
the true object is exactly zero, the reconstruction is not: the truncation spreads faint
sidelobes over the whole field of view, 38 image units on average beyond three voxels from
the brain, an effect that returns when the noise level is measured from the background.

This is why aslscan builds its object on a grid twice as fine as the image (`Oversample: 2`
in the sidecar: a 128 × 136 simulation matrix for a 64 × 68 acquisition) and truncates its
k-space to the acquisition matrix. A phantom drawn directly on the 64 × 68 grid would have no
edges finer than a voxel and would ring nowhere, which no real image does. Ringing is a
property of every acquisition, not a malfunction; its effect on perfusion values at tissue
borders is part of the partial-volume story of [Chapter 12](../03-preprocessing/12-partial-volume.md).

## The EPI readout

A conventional sequence records one line of k-space per excitation. Echo-planar imaging
(EPI, {cite:p}`mansfield1977`) records the whole plane after a single excitation: the readout
gradient alternates direction, sweeping back and forth along $k_x$, while short phase-encode
blips step $k_y$ one line at a time. The path through k-space is a zigzag, and the time
between successive lines is the **echo spacing**. The reference sidecar records a
`TotalReadoutTime` of 20 ms for 68 phase-encode lines; the simulator divides it evenly, so
one line is recorded every 0.294 ms (`TLineMs`), and the train is centered on the echo time
of 12 ms.

```{code-cell} python
:tags: [hide-input]
sim = data.load_dataset("ref-pcasl").run().simulation()["Acquisition"]
t_line, n_lines = sim["TLineMs"], NY
te_ms = r.echo_time * 1e3
t0 = te_ms - n_lines / 2 * t_line
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [0.8, 1.6]})
n = 12  # lines drawn in the k-space sketch
for i in range(n):
    xs = np.linspace(-1, 1, 20) * (1 if i % 2 == 0 else -1)
    ax1.plot(xs, np.full(20, i - n / 2 + 0.5), color=plt.cm.viridis(i / (n - 1)), lw=1.5)
    if i < n - 1:
        ax1.annotate("", xy=(xs[-1], i - n / 2 + 1.4), xytext=(xs[-1], i - n / 2 + 0.6), arrowprops=dict(arrowstyle="->", color="0.5", lw=1))
ax1.set(xlabel="$k_x$ (readout)", ylabel="$k_y$ (phase encode)", title="the EPI zigzag (color = time)", xticks=[], yticks=[])
ax1.grid(False)
t = np.linspace(-1, 24, 3000)
rf = np.exp(-((t - 0) / 0.25) ** 2) * np.sinc((t - 0) * 4)
gx = np.where((t >= t0) & (t <= t0 + n_lines * t_line), np.sign(np.sin(np.pi * (t - t0) / t_line)), 0)
gy = np.where((t >= t0) & (t <= t0 + n_lines * t_line) & (((t - t0) % t_line) > 0.85 * t_line), 1.0, 0)
for y, sig, lab in [(3, rf, "RF (90°)"), (2, gx * 0.8, "readout gradient $G_x$"), (1, gy * 0.8, "phase-encode blips $G_y$"), (0, np.abs(gx) * 0.6, "sampling (ADC)")]:
    ax2.plot(t, y + sig, color=PALETTE[0] if y == 3 else INK["primary"], lw=0.8)
    ax2.text(-0.5, y + 0.5, lab, fontsize=8, ha="right", va="center")
ax2.axvline(te_ms, color=PALETTE[1], lw=1, ls="--")
ax2.text(te_ms + 0.3, 3.8, f"TE {te_ms:.0f} ms:\ncenter of k-space", fontsize=8, color=PALETTE[1], va="top")
ax2.annotate("", xy=(t0 + n_lines * t_line, -0.6), xytext=(t0, -0.6), arrowprops=dict(arrowstyle="<->", color="0.4", lw=1))
ax2.text(te_ms, -0.75, f"{n_lines} lines × {t_line:.3f} ms = {n_lines * t_line:.0f} ms (TotalReadoutTime)", fontsize=8, ha="center", va="top", color="0.4")
ax2.set(xlim=(-1, 24), ylim=(-1.4, 4.1), yticks=[], xlabel="time after the excitation (ms)", title="the reference readout, as the simulator times it")
ax2.grid(False)
ax2.spines["left"].set_visible(False)
fig.tight_layout()
print(f"TLineMs {t_line:.4f} ms x {n_lines} lines = {t_line * n_lines:.1f} ms = TotalReadoutTime {r.total_readout_time * 1e3:.0f} ms; "
      f"train from {t0:.0f} to {t0 + n_lines * t_line:.0f} ms after the excitation")
```

Left: the zigzag. Every line runs the opposite way to the one before, which is why a timing
mismatch between odd and even lines produces the Nyquist ghost {cite:p}`buonocore1997` of
[Chapter 13](../03-preprocessing/13-assembled-pipeline.md). Right: the reference readout
as the simulator times it, sampling from 2 to 22 ms with the center of k-space, which
decides the contrast, at TE. The simulator applies the spin-echo T2 envelope at TE and the
readout timing separately; drawn literally, a spin echo at 12 ms would need its 180° pulse
at 6 ms, inside the train, so a real spin-echo EPI at this matrix would use a longer TE or
partial Fourier (below). The train is short by the standards of diffusion or functional EPI
(30–90 ms) because the ASL matrix is small, and that shortness limits its distortion.

The **total readout time** sets how far an off-resonance moves the image
{cite:p}`jezzard1995`. A spin precessing
$\Delta f$ hertz away from the scanner's assumed frequency accumulates extra phase that
grows through the train, line by line, exactly as a phase-encode gradient would, so the
reconstruction places its signal at the wrong position along the phase-encode axis:

$$\Delta y = \Delta f \times \mathrm{TotalReadoutTime} \quad \text{(in voxels)}.$$

At 100 Hz, close to the largest offset of the phantom's synthetic field map, which sits
above the frontal sinuses
([Chapter 11](../03-preprocessing/11-susceptibility-distortion.md)), that is 100 × 0.020 = 2 voxels, or
7 mm, along `PhaseEncodingDirection`, which the sidecar records as `j-`: the second image
axis, posterior–anterior, in the negative sense. A real offset varies across the head, so
rows move by different amounts and tissue piles up on one side and stretches on the other;
[Chapter 11](../03-preprocessing/11-susceptibility-distortion.md) simulates and corrects
that. Here only the uniform case is checked.

```{code-cell} python
:tags: [hide-input]
df = 100.0  # Hz, uniform over the slice
t_lines = (np.arange(NY) - NY // 2) * t_line * 1e-3                       # s from the center line, one per ky line
k_off = k_acq * np.exp(2j * np.pi * df * t_lines)[None, :]                # each line picks up the phase of its own moment
shifted = np.abs(ifft2c(k_off) * (NX * NY) / obj.size)
prof_a, prof_b = trunc.sum(0), shifted.sum(0)                             # profiles along phase encode
xc = np.fft.ifft(np.fft.fft(prof_b) * np.conj(np.fft.fft(prof_a))).real  # cross-correlation: the lag with the largest value is the shift
lag = np.argmax(xc); lag = lag - NY if lag > NY / 2 else lag
fig, axes = plt.subplots(1, 2, figsize=(7, 3.6))
show_image(axes[0], np.rot90(trunc), "on resonance", vmin=0, vmax=7000)
show_image(axes[1], np.rot90(shifted), f"uniform {df:.0f} Hz offset", vmin=0, vmax=7000)
for ax in axes:
    ax.contour(np.rot90(trunc), levels=[2500], colors=[PALETTE[1]], linewidths=0.7)
fig.tight_layout()
print(f"{df:.0f} Hz x {r.total_readout_time * 1e3:.0f} ms readout: predicted shift {df * r.total_readout_time:.1f} voxels along phase encode; measured {abs(lag):d} voxels ({abs(lag) * DX:.1f} mm)")
```

The orange outline is the brain's edge on resonance; with a uniform 100 Hz offset the whole
slice has moved 2 voxels (7 mm) along the phase-encode axis, as predicted. A longer readout
would move it proportionally farther.

## Three refinements of the readout

**Partial Fourier.** For an object with no phase, k-space is symmetric about its center, so
half of the lines are redundant. A partial-Fourier acquisition
{cite:p}`feinberg1986,noll1991` skips a fraction of the lines
on one side (typically 5/8 to 7/8 are acquired) and lets the reconstruction supply the rest
from the symmetry: the train is shorter and the center of k-space is reached sooner, which
allows a shorter TE. But real images do have phase, from field inhomogeneity, coil phase,
and motion, so the symmetry is only approximate, and the missing lines cost resolution along
phase encode; the `readout` dataset of [Chapter 13](../03-preprocessing/13-assembled-pipeline.md)
acquires 6/8 of the lines and shows the blurring. The sidecar records the fraction as
`PartialFourier` (1.0 here).

**Multiple receive coils.** A modern head coil is an array of 8 to 64 small coils, each
most sensitive to the tissue nearest it {cite:p}`roemer1990`. Each records its own k-space
and its own image, weighted by its sensitivity, and the images are combined, most simply as
the root of the sum of the squares. Small coils give more signal per unit noise near the
surface than one large coil, and their sensitivities differ, which is the information
parallel imaging uses. The reference protocol uses a single coil (`NCoils: 1`) so that its
noise is the simplest case; the `coils8-r2` run of [Chapter 8](../03-preprocessing/08-noise.md)
uses eight.

**GRAPPA.** In-plane acceleration by a factor $R$ keeps every $R$-th phase-encode line,
which halves (for $R = 2$) the train, the total readout time, and therefore the distortion.
Because each coil sees the object through a smooth sensitivity, a skipped k-space sample can
be predicted from its neighbors across all coils; GRAPPA {cite:p}`griswold2002` learns those
weights from a fully sampled band of central lines (`AcsLines: 24` in the sidecar). The
reconstruction removes the wrap of the aliasing figure but amplifies the noise by a factor
that varies across the image (the g-factor, defined for SENSE by {cite:t}`pruessmann1999`
and computed for GRAPPA by {cite:t}`breuer2009`)
and correlates it between neighboring voxels, so one σ no longer describes the image
([Chapter 8](../03-preprocessing/08-noise.md)).

## Magnitude, phase, and noise

Every k-space sample is complex, so every reconstructed voxel is complex: a magnitude and a
phase. The **magnitude** is the amount of transverse magnetization the voxel held at the
echo, the image everyone looks at. The **phase** is the angle it had turned through relative
to the scanner's reference, which includes an arbitrary constant from the receiver, so only
phase differences carry information: between voxels (a field map), between echoes, or
between acquisitions. BIDS {cite:p}`gorgolewski2016` stores both, as `part-mag` and
`part-phase` files.

Thermal noise enters in k-space, as independent Gaussian noise of the same variance in
every sample's real and imaginary parts. The Fourier transform is linear and orthogonal, so
the image is the noise-free image plus complex Gaussian noise of standard deviation $\sigma$
per component in every voxel. Taking the magnitude changes the distribution: a voxel with
true signal $A$ has a **Rician** magnitude {cite:p}`gudbjartsson1995`. Where $A \gg \sigma$
it is close to $A$ plus Gaussian noise of standard deviation $\sigma$; where $A = 0$ it is
Rayleigh distributed, with mean $\sigma\sqrt{\pi/2} \approx 1.25\,\sigma$ and standard
deviation $0.66\,\sigma$ {cite:p}`henkelman1985`, so a magnitude image has no zero-mean
background and never shows a negative value. The figure draws the reference noise level, $\sigma = 40$ image units
(`NoiseVariance: 1600` per component in the sidecar), on four true signals.

```{code-cell} python
:tags: [hide-input]
sigma = np.sqrt(r.noise_variance)
rng = np.random.default_rng(0)
fig, axes = plt.subplots(1, 4, figsize=(11.5, 2.9), sharey=True)
for ax, a_over_s in zip(axes, [0, 1, 3, 10]):
    A = a_over_s * sigma
    mag = np.abs(A + sigma * (rng.standard_normal(40000) + 1j * rng.standard_normal(40000)))
    ax.hist(mag, bins=60, density=True, color=PALETTE[0], alpha=0.5)
    x = np.linspace(0, mag.max(), 400)
    ax.plot(x, stats.rice.pdf(x, A / sigma, scale=sigma), color=INK["primary"], lw=1.2, label="Rician pdf")
    ax.axvline(A, color=PALETTE[1], lw=1, ls="--", label="true signal A")
    ax.set(title=f"A = {a_over_s} σ" + (" (background)" if a_over_s == 0 else ""), xlabel="magnitude (image units)")
    print(f"A = {a_over_s:2d} σ: mean magnitude {mag.mean():6.1f} (A = {A:5.1f}; excess {mag.mean() - A:5.1f} = {(mag.mean() - A) / sigma:.2f} σ), SD {mag.std():5.1f} = {mag.std() / sigma:.2f} σ")
axes[0].set(ylabel="density")
axes[0].legend(loc="upper right", fontsize=7)
fig.tight_layout()
```

In the background ($A = 0$) the histogram sits entirely above zero with a mean of 50.1, which
is 1.25 σ; at $A = \sigma$ the bias is still 0.55 σ; at $A = 3\sigma$ it has shrunk to
0.17 σ, and at $A = 10\sigma$ the distribution is Gaussian with the right mean and a
standard deviation equal to σ. Tissue in the reference series sits at about 150 σ, so its
noise is Gaussian for every practical purpose; the Rician floor matters for the
perfusion-weighted difference, whose signal is a few σ, and for the background.

## The reference series, as the scanner would give it

Everything above used a synthetic slice and the page's own Fourier transform. The pipeline
tier acquires the whole slab with aslscan: the same object, oversampled, sampled by a
spin-echo EPI train, with noise added in k-space. The figure shows the first control volume
of the reference series at the display slice, its phase, and the separate M0 scan, as loaded
from the BIDS files {cite:p}`clement2022`; the printout lists the sidecar fields that
describe the readout.

```{code-cell} python
:tags: [hide-input]
run = data.load_dataset("ref-pcasl").run()
p, ctx, k = run.sidecar(), run.context(), phantom.DISPLAY_SLICE
ctrl0 = np.asarray(run.image("part-mag_asl.nii.gz").dataobj[..., 0], np.float32)
phase0 = np.asarray(run.image("part-phase_asl.nii.gz").dataobj[..., 0], np.float32)
m0 = run.m0scan()
fig, axes = plt.subplots(1, 3, figsize=(9.5, 4))
show_slice(axes[0], ctrl0, k, f"volume 0 ({ctx[0]}), magnitude", vmin=0, vmax=7500)
show_slice(axes[1], phase0, k, "the same volume, phase", kind="phase", vmin=-np.pi, vmax=np.pi)
show_slice(axes[2], m0, k, f"M0 scan (TR {run.m0scan_sidecar()['RepetitionTimePreparation']:g} s)", vmin=0, vmax=7500)
fig.tight_layout()
st = p["SliceTiming"]
print(f"EchoTime {p['EchoTime'] * 1e3:.0f} ms | TotalReadoutTime {p['TotalReadoutTime'] * 1e3:.0f} ms | PhaseEncodingDirection {p['PhaseEncodingDirection']} | "
      f"AcquisitionVoxelSize {p['AcquisitionVoxelSize']} mm | SliceTiming {len(st)} slices, {st[0]:g} to {st[-1]:g} s in steps of {st[1] - st[0]:g} s")
acq, grd = p["AslscanSimulation"]["Acquisition"], p["AslscanSimulation"]["Grid"]
print(f"AslscanSimulation.Acquisition: TLineMs {acq['TLineMs']:.4f}, NCoils {acq['NCoils']}, PartialFourier {acq['PartialFourier']}, NoiseVariance {acq['NoiseVariance']:.0f} (sigma {np.sqrt(acq['NoiseVariance']):.0f}); "
      f"Grid: Oversample {grd['Oversample']}, SimulationMatrix {grd['SimulationMatrix']} -> AcquisitionMatrix {grd['AcquisitionMatrix']}")
gm = run.fraction("gm") > 0.9
print(f"mean control signal in voxels that are > 90 % GM: {ctrl0[gm].mean():.0f} image units, SNR {ctrl0[gm].mean() / np.sqrt(acq['NoiseVariance']):.0f}")
```

The control image is the box-averaged object of the toy figures with ringing and noise on
top: gray matter brightest, CSF darkest. Its phase is close to zero inside the brain (the
simulated object has no phase, and at this SNR the noise turns it by a fraction of a degree)
and random outside, where noise alone sets it. The M0 scan is the same readout at TR 8 s, so
CSF, three quarters recovered in the control image, is nearly fully recovered here and the
tissue–CSF contrast is smaller ([Chapter 16](../04-quantification/16-calibration.md)). The
printout is the metadata every correction in Part III needs: the total readout time and
phase-encode direction for distortion, the slice timing for the per-slice delay of
[Chapter 6](../02-labeling/06-the-asl-signal.md), the echo time and the voxel size. Gray
matter sits at 6246 image units, 156 σ.

## Measure it: the noise level from the background

A magnitude image's background is a Rayleigh sample of the noise, so σ can be estimated from
it without any knowledge of the object: divide the mean background magnitude by
$\sqrt{\pi/2}$ {cite:p}`henkelman1985`. The cell does that in the voxels whose tissue fractions are all zero, in two
versions: every such voxel, and only those at least three voxels from any tissue.

```{code-cell} python
:tags: [hide-input]
fr = run.fractions()
empty = (fr["gm"] + fr["wm"] + fr["csf"]) == 0
far = ndimage.distance_transform_edt(empty) > 3
rayleigh = lambda m: m.mean() / np.sqrt(np.pi / 2)
print(f"sigma from the Rayleigh mean: all {empty.sum()} empty voxels {rayleigh(ctrl0[empty]):.1f}; the {far.sum()} at least 3 voxels from tissue {rayleigh(ctrl0[far]):.1f} (configured: {sigma:.0f})")
clean = np.asarray(data.load_dataset("ref-clean").run().image("part-mag_asl.nii.gz").dataobj[..., 0], np.float32)
print(f"the noise-free run in the same far voxels: mean {clean[far].mean():.1f} image units (truncation sidelobes), not 0")
corrected = np.sqrt(((ctrl0[far] ** 2).mean() - (clean[far] ** 2).mean()) / 2)  # Rician: E|S|^2 = A^2 + 2 sigma^2
print(f"sigma with the sidelobes accounted for, sqrt((E|S|^2 - A^2) / 2): {corrected:.1f}")
wm = fr["wm"] > 0.99
controls = np.asarray(run.image("part-mag_asl.nii.gz").dataobj[..., ::2], np.float32)
print(f"sigma from the SD over the 30 control volumes in {wm.sum()} pure-WM voxels (Gaussian regime): {controls[wm].std(axis=-1, ddof=1).mean():.1f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.4), gridspec_kw={"width_ratios": [1, 1.4]})
show_slice(ax1, clean, k, "noise-free control, window 0–100: the sidelobes", vmin=0, vmax=100)
ax1.contour(take_slice(empty.astype(float), k), levels=[0.5], colors=[PALETTE[1]], linewidths=0.7)
ax2.hist(ctrl0[far], bins=80, range=(0, 250), density=True, color=PALETTE[0], alpha=0.5, label="background voxels (≥ 3 from tissue)")
x = np.linspace(0, 250, 400)
ax2.plot(x, stats.rayleigh.pdf(x, scale=sigma), color=INK["primary"], lw=1.2, label=f"Rayleigh, σ = {sigma:.0f} (configured)")
ax2.plot(x, stats.rayleigh.pdf(x, scale=rayleigh(ctrl0[far])), color=PALETTE[1], lw=1.2, ls="--", label=f"Rayleigh, σ = {rayleigh(ctrl0[far]):.0f} (from the mean)")
ax2.set(xlabel="magnitude (image units)", ylabel="density", title="the background is not quite Rayleigh")
ax2.legend(fontsize=7)
fig.tight_layout()
```

The naive estimate is too high: 53 from every empty voxel, 45 from those at least three
voxels from any tissue, against the configured 40. The reason is the ringing of the section
above. The noise-free run shows that the "empty" voxels carry a mean signal of 21 image
units, the truncation sidelobes of the oversampled brain (left panel, windowed to 0–100,
with the orange line marking where the tissue fractions reach zero; about half the toy
figure's 38, because the simulator's object is on a 1.75 mm grid whose edges one box average
has already softened). A Rician voxel with a small true signal has a larger mean than a
Rayleigh one, so the sidelobes bias the estimate upward, most near the brain and least in
the corners. Accounting for them with the Rician identity $E|S|^2 = A^2 + 2\sigma^2$
{cite:p}`gudbjartsson1995` gives 40.0, and the standard deviation across the 30 control volumes in pure white-matter voxels,
where the noise is Gaussian, gives 39.8. The histogram shows the mismatch directly: the
background sits to the right of the Rayleigh curve for σ = 40 and has a longer tail. A
background-based estimate is therefore a slight overestimate whenever the object rings into
the background, which is always; a temporal estimate from the series itself
([Chapter 8](../03-preprocessing/08-noise.md)) does not have this problem.

## 3D readouts: a slab, an echo train, and a stack of partitions

The reference protocol excites and reads its 20 slices one after another, 40 ms apart, so
each slice sees the label at its own delay ([Chapter 6](../02-labeling/06-the-asl-signal.md)).
A **3D readout** excites the whole 100 mm slab with one pulse, so every voxel is read at the
same delay and there is no slice timing to correct, and it encodes the through-plane
direction the way the in-plane phase-encode direction is encoded: a gradient pulse along $z$
steps $k_z$ between acquisitions, and k-space becomes a stack of $n_z = 20$ planes, the
**partitions**. The partition spacing is $1/\mathrm{FOV}_z$ and the outermost partition sets
the slice thickness, exactly as in-plane, and the reconstruction is a three-dimensional
inverse Fourier transform. In-plane nothing changes: the truncation, the ringing and the
distortion of the sections above are the same in 3D.

The question is how to read 20 partitions after one excitation. 3D GRASE (gradient and spin
echo, {cite:p}`oshio1991`) follows the excitation with a train of refocusing pulses and reads
one partition per spin echo with an EPI block, so a train of 20 echoes covers the stack; its
use for ASL is due to {cite:t}`gunther2005` and {cite:t}`fernandezseara2005`, and the
consensus paper recommends a segmented 3D readout {cite:p}`alsop2015`. The price is that the
echoes decay. With 180° refocusing pulses the $e$-th echo has amplitude $e^{-e\,\mathrm{ESP}/T_2}$,
the T2 decay at its own time (echo spacing ESP times echo number). Below 180° each pulse
leaves part of the magnetization longitudinal, where it does not decay with T2, and later
pulses bring it back as **stimulated echoes** that partly restore the later echoes
{cite:p}`hennig1988`; the extended phase graph {cite:p}`weigel2015` computes the amplitudes of
the whole train, and the simulator does so per tissue class with that class's T1 and T2
(`T2Mode: class`). Because each echo reads one partition, the train multiplies the $k_z$
axis by a **window** $w(k_z)$, and the image is convolved along $z$ with the inverse DFT of
that window, the through-plane **point-spread function** (PSF):

$$\rho_\mathrm{3D}(x, y, z) = \sum_{z'} \rho(x, y, z')\, h(z - z'), \qquad h(z) = \frac{1}{n_z}\sum_{k_z} w(k_z)\, e^{2\pi i k_z z / n_z},$$

with the sum over $z'$ circular, since the DFT is periodic. The order in which the echoes
visit the partitions decides what the window looks like. **Centric** ordering
(`KzOrder: centric`) reads $k_z = 0$ at the first echo, then $\pm 1, \pm 2, \ldots$, so the
window is largest at the center and falls off toward the edges: the contrast comes from the
strongest echo and the decay costs resolution, a blur, not signal. A linear order would put
the center at the tenth echo and lose most of the gray-matter signal. Even with centric
ordering the train must be short against T2, 80 ms in gray matter, and a 20-echo train in
which every echo reads all 68 lines of a partition is not: that is the reason for
**segmentation**. The four-shot run reads 17 of the 68 lines per echo, so each shot's train
is a quarter as long, and a volume is assembled from `NumberShots` shots, each with its own
labeling, one TR apart; the volume takes `NumberShots` TRs, and anything that changes between
shots, motion, physiology, or the label itself, becomes an inconsistency between k-space
segments rather than between volumes
([Chapter 10](../03-preprocessing/10-motion.md), [Chapter 13](../03-preprocessing/13-assembled-pipeline.md)).

The sidecar of a 3D run has `MRAcquisitionType: 3D`, no `SliceTiming`, `NumberShots`, and a
`FlipAngle` that is the train's refocusing angle (the excitation is 90°); its
`EffectiveEchoSpacing` and `TotalReadoutTime` are those of the segmented EPI block, the line
spacing divided by the number of segments. The simulator's resolved train is under
`AslscanSimulation.Readout`, and its `Approximations` list states where the model is
idealized: the slab is exactly the field of view in $z$ with a flat profile (so the
circular blur wraps the first partition into the last, which a real protocol avoids by
oversampling in $z$), the partitions are the acquisition grid's slices, the train leaves no
longitudinal magnetization behind, and the coils are uniform along $z$. The cell prints the
three GRASE runs' trains and plots the echo amplitudes the extended phase graph gave each
tissue (`EchoAmplitudes.PerEcho`) and the same amplitudes in partition order
(`EchoAmplitudes.KzModulation`), the window.

```{code-cell} python
:tags: [hide-input]
ds3 = data.load_dataset("readout-3d")
grase = {n: ds3.run(n) for n in ["grase-1shot", "grase-4shot", "grase-4shot-130"]}
ro = {n: g.simulation()["Readout"] for n, g in grase.items()}
ea = {n: g.simulation()["EchoAmplitudes"] for n, g in grase.items()}
for n, g in grase.items():
    R, p3 = ro[n], g.sidecar()
    print(f"{n}: NumberShots {R['NumberShots']}, KySegments {R['KySegments']} x LinesPerEcho {R['LinesPerEcho']}, EchoSpacingMs {R['EchoSpacingMs']:.2f}, EchoTrainLength {R['EchoTrainLength']} "
          f"(train {R['EchoSpacingMs'] * R['EchoTrainLength']:.0f} ms), refocusing {R['RefocusingFlipAngle']['Value']:.0f} deg, KzOrder {R['KzOrder']} (CentreEcho {R['CentreEcho']}), VolumeDuration {R['VolumeDuration']:g} s | "
          f"EchoTime {p3['EchoTime'] * 1e3:.1f} ms, TotalReadoutTime {p3['TotalReadoutTime'] * 1e3:.2f} ms, EffectiveEchoSpacing {p3['EffectiveEchoSpacing'] * 1e3:.3f} ms, SliceTiming {'present' if 'SliceTiming' in p3 else 'absent'}")
print("Approximations: " + "; ".join(ro["grase-4shot"]["Approximations"]))
TIS = {"GM": "grey_matter", "WM": "white_matter", "CSF": "csf", "blood": "blood"}
COL = dict(TISSUE_COLORS, blood=PALETTE[3])
echo, kz = np.arange(1, 21), np.arange(-10, 10)
fig, axes = plt.subplots(2, 3, figsize=(11, 5.4), sharey=True)
for j, n in enumerate(grase):
    for t, lab in TIS.items():
        axes[0, j].plot(echo, ea[n]["PerEcho"][lab], "o-", ms=3, lw=1, color=COL[t], label=t)
        axes[1, j].plot(kz, ea[n]["KzModulation"][lab], "o-", ms=3, lw=1, color=COL[t])
    axes[0, j].set(title=f"{n}: ESP {ro[n]['EchoSpacingMs']:.0f} ms, {ro[n]['RefocusingFlipAngle']['Value']:.0f}°", xlabel="echo number")
    axes[1, j].set(xlabel="$k_z$ (partition index)")
axes[0, 0].set(ylabel="echo amplitude (PerEcho)"); axes[1, 0].set(ylabel="$k_z$ window (KzModulation)"); axes[0, 0].legend(fontsize=7)
fig.tight_layout()
for n in grase:
    g = ea[n]["PerEcho"]["grey_matter"]
    print(f"{n}: GM echo amplitudes {g[0]:.3f}, {g[1]:.3f}, {g[2]:.3f} ... {g[-1]:.3f}")
```

Top row: the echo amplitudes. The one-shot train (left) reads all 68 lines per echo at
0.3 ms per line, so its echo spacing is 22.55 ms and the train lasts 451 ms; gray matter
starts at 0.754 (the T2 decay to the first echo, which is also the echo time) and is at
0.004 by the last echo. Blood, with its longer T2, and CSF decay more slowly. Four shots of
17 lines (middle) bring the spacing to 10 ms and the train to 200 ms: gray matter runs from
0.882 to 0.082. At 130° (right) the first echo is lower, 0.725, because part of the
magnetization was stored longitudinally instead of being refocused, and the second echo,
0.782, is larger than the first: the stimulated echo returning. From the third echo on the
130° train decays more slowly than the 180° one, its tail at 0.096 against 0.082. Bottom row:
the same amplitudes in partition order. Centric ordering makes the window peak at $k_z = 0$
and fall off to both sides; at 130° the peak sits at $k_z = +1$, the second echo, a one-partition
offset that shows up as a phase ramp in the image and that the magnitude removes. The next
cell takes the inverse DFT of each window, on a grid twenty times finer than the slices so
that the width can be read off, and reports the full width at half maximum.

```{code-cell} python
:tags: [hide-input]
def psf_of(window, pad=20):
    """|inverse DFT| of a k_z window, evaluated on a grid ``pad`` times finer than the slices."""
    w = np.asarray(window, complex); n = len(w)
    kz = np.arange(-n // 2, n - n // 2); z = np.arange(-n // 2, n // 2, 1 / pad)
    return z, (w[None, :] * np.exp(2j * np.pi * kz[None, :] * z[:, None] / n)).sum(1) / n

def fwhm(z, p):
    above = z[np.abs(p) >= np.abs(p).max() / 2]
    return above.max() - above.min()

z, flat = psf_of(np.ones(20))
fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
for ax, n in zip(axes, grase):
    ax.plot(z, np.abs(flat), color="0.6", lw=1, ls="--", label="flat window")
    line = f"{n}:"
    for t in ["GM", "WM", "CSF"]:
        zz, p = psf_of(ea[n]["KzModulation"][TIS[t]])
        ax.plot(zz, np.abs(p), color=COL[t], lw=1.3, label=t)
        line += f" {t} FWHM {fwhm(zz, p):.1f} slices, peak {np.abs(p).max():.2f};"
    print(line)
    ax.set(xlim=(-4, 4), xlabel="z (slices)", title=n)
axes[0].set(ylabel="|PSF|"); axes[0].legend(fontsize=7)
fig.tight_layout()
print(f"flat window (no decay): FWHM {fwhm(z, flat):.1f} slices, peak 1.00; the peak of a PSF is the mean of its window and its sum over the slices is the window at k_z = 0, the first echo")
```

The dashed curve is the PSF of a flat window, the sinc-like kernel that a 20-partition
acquisition gives with no decay at all: 1.2 slices wide at half maximum, with the sidelobes
of the truncation section. The one-shot train widens gray matter's PSF to 3.5 slices of 5 mm and
lowers its peak to 0.15: the long train has cost gray matter most of its through-plane
resolution, while CSF, whose T2 is 300 ms, keeps a 1.5-slice PSF. Four shots bring gray matter to 1.8 slices and white
matter to 1.6; the 130° train is marginally narrower (1.7 slices) because its window is
flatter, at the cost of a lower peak. Two numbers summarize each PSF: its peak, which is the
mean of the window, and its sum over the slices, which is the window at $k_z = 0$, the
first echo's amplitude. The sum is what a region much thicker than the PSF keeps of its
signal; the peak is what a single-slice feature keeps. The cortex, a few millimeters thick,
sits between the two, which is why the 3D gray-matter signal is lower than the 2D one in the
images that follow.

## See it: one control image, five readouts

The figure shows the mean control image of the reference 2D series beside the three GRASE
runs and the stack of spirals of the next section, all on the same object, labeling and
timing, at the display slice and in a coronal section through the ventricles (the slices
are 5 mm over 3.5 mm voxels, so the coronal panels are stretched accordingly).

```{code-cell} python
:tags: [hide-input]
def control_mean(rn):
    """The mean of a run's control volumes, and the control volumes themselves (x, y, z, n)."""
    m = rn.mag(); rows = [i for i, c in enumerate(rn.context()) if c == "control"]
    return m[..., rows].mean(-1), m[..., rows]

spiral = ds3.run("spiral")
RUNS = {"ref-pcasl (2D EPI)": run, **grase, "spiral": spiral}
cm, series, gm3 = {}, {}, {}
for n, rn in RUNS.items():
    cm[n], series[n] = control_mean(rn)
    gm3[n] = rn.fraction("gm") > 0.9
CORONAL, ASPECT = 34, r.voxel_mm[2] / r.voxel_mm[0]  # the coronal plane through the ventricles; 5 mm slices over 3.5 mm voxels
fig, axes = plt.subplots(2, 5, figsize=(13, 5.0), gridspec_kw={"height_ratios": [1, 0.62]})
for j, (n, v) in enumerate(cm.items()):
    show_slice(axes[0, j], v, k, n, vmin=0, vmax=7500)
    show_image(axes[1, j], take_slice(v, CORONAL, plane="coronal"), None, vmin=0, vmax=7500, aspect=ASPECT)
fig.tight_layout()
print("mean control signal in voxels that are > 90 % GM: " + ", ".join(f"{n} {v[gm3[n]].mean():.0f}" for n, v in cm.items()))
print(f"matrix: 2D and GRASE {cm['grase-4shot'].shape}, spiral {cm['spiral'].shape}; the spiral's 68 x 68 field of view is corner-aligned with the 64 x 68 one")
```

Axially (top row) the 3D images are smoother than the 2D one, in two ways. Their noise is
lower, which the last section measures, and the cortex is less distinct, because every slice
now holds a blurred share of its neighbors: the gray-matter mean falls from 6246 in 2D to
4718 for the one-shot GRASE, 5752 for the four-shot, 4838 at 130°, and 5547 for the spiral.
The coronal sections (bottom row) show the through-plane blur directly. The 2D slab has a
sharp top and bottom; in the 3D runs the brain's outline is smeared along $z$, most in the
one-shot run, and the first and last slices carry a faint copy of each other, the circular
wrap of the ideal-slab approximation. The ventricles, whose CSF has the narrowest PSF, stay
the sharpest feature.

## Measure it: the 3D control image from the 2D physics

The claim is that the 3D images differ from the 2D one by the $k_z$ window alone. The cell
tests it. For each tissue it takes the fraction map of the acquisition grid times that
tissue's control signal, $M_0 (1 - e^{-\mathrm{TR}/T_1}) \times 100$ without the $e^{-\mathrm{TE}/T_2}$ factor,
because the window already carries the decay: its value at $k_z = 0$ is the first echo's
amplitude, $e^{-\mathrm{TE}/T_2}$ at the 3D run's echo time. It convolves that object along $z$,
circularly, with the tissue's PSF from the sidecar window, sums the tissues, and takes the
magnitude. For comparison it also predicts the image with the first echo's amplitude alone,
without the blur, and it predicts the 2D image the same way (the fractions times
`signal_of`, no window at all) so that the residual ringing and noise of a prediction from
box-averaged fractions are known.

```{code-cell} python
:tags: [hide-input]
sig_te0 = {n: t.m0 * (1 - np.exp(-r.repetition_time / t.t1)) * r.signal_scale for n, t in presets.TISSUES.items()}  # the control signal before the T2 decay

def predict_3d(rn):
    """The box-averaged object convolved along z (circularly) with each tissue's PSF, and the same
    object times the k_z = 0 amplitude alone (no blur)."""
    frs, w = rn.fractions(), rn.simulation()["EchoAmplitudes"]["KzModulation"]
    blurred, flat = np.zeros(frs["gm"].shape, complex), np.zeros(frs["gm"].shape)
    for t in ["GM", "WM", "CSF"]:
        obj, win = frs[t.lower()] * sig_te0[t], np.asarray(w[TIS[t]])
        blurred += np.fft.ifft(np.fft.fft(obj, axis=2) * np.fft.ifftshift(win)[None, None, :], axis=2)
        flat += obj * win[10]                                                   # k_z = 0 is index 10 of -10 ... 9
    return np.abs(blurred), flat

rms = lambda a, b, m: np.sqrt(((a - b)[m] ** 2).mean())
n2, m2 = "ref-pcasl (2D EPI)", cm["ref-pcasl (2D EPI)"]
pred2d = sum(fr[t.lower()] * signal_of[t] for t in ["GM", "WM", "CSF"])
print(f"{n2}: measured GM {m2[gm3[n2]].mean():.0f}, predicted without any window {pred2d[gm3[n2]].mean():.0f}, RMS difference in the brain {rms(pred2d, m2, run.mask()):.0f} image units (ringing and noise)")
for n in ["grase-1shot", "grase-4shot", "grase-4shot-130", "spiral"]:
    blurred, flat = predict_3d(RUNS[n]); m, g, brain = cm[n], gm3[n], RUNS[n].mask()
    print(f"{n}: measured GM {m[g].mean():.0f}; predicted with the window {blurred[g].mean():.0f} (RMS difference {rms(blurred, m, brain):.0f}); "
          f"with the first-echo amplitude alone, no blur, {flat[g].mean():.0f} (RMS {rms(flat, m, brain):.0f})")
    if n == "grase-4shot":
        shown = (m, flat, blurred, blurred - m)
fig, axes = plt.subplots(1, 4, figsize=(12, 2.7))
titles = ["measured: grase-4shot control mean", "object × first-echo amplitude", "convolved with the PSF along z", "predicted − measured"]
for ax, img, title, kw in zip(axes, shown, titles, [dict(vmin=0, vmax=7500)] * 3 + [dict(kind="diff", vmin=-600, vmax=600)]):
    show_image(ax, take_slice(img, CORONAL, plane="coronal"), title, aspect=ASPECT, **kw)
fig.tight_layout()
```

The 2D prediction misses the measured image by 119 image units RMS, the ringing and noise
a box-averaged object cannot reproduce. With the window, the 3D predictions are closer than
that: 62 for the one-shot run, 77 for the four-shot, 70 at 130°, 80 for the spiral, and the
gray-matter means agree to within about 20 image units (5752 measured against 5732
predicted for the four-shot GRASE). Without the blur, with only the first echo's amplitude
applied, the prediction is off by 578 RMS and puts gray matter at 6336 against the measured
5752: the amplitude alone barely changes the 2D value of 6246, since the four-shot echo time
is 10 ms rather than 12, and the PSF explains the rest of the loss. The coronal panels make the same point: the
second panel is the 2D object with a lower signal, sharp; the third, after the convolution,
is the measured image, including the smeared slab ends; the difference is noise, ringing at
the brain surface, and nothing structured along $z$. The lower 3D gray-matter signal is
therefore physics, the thin cortex sharing its signal with its neighbors through the PSF,
and the same convolution acts on the perfusion-weighted difference, where almost all of the
signal is in that cortex.

## The stack of spirals

An EPI block is not the only way to read a partition. A **spiral** trajectory
{cite:p}`meyer1992` starts at the center of k-space and winds outward under two sinusoidal
gradients, and the simulator's **stack of spirals** reads one spiral per spin echo of the
train, one interleaf of each partition per shot; the consensus paper names it beside 3D
GRASE as a recommended readout {cite:p}`alsop2015`. One interleaf is an Archimedean spiral
with `Turns` revolutions out to `KMax`, and the $s$-th of $N$ interleaves is rotated by
$2\pi s/N$:

$$k_s(u) = k_\mathrm{max}\, u\, e^{\,i\,(2\pi n_\mathrm{turns}\, u + 2\pi s/N)}, \qquad 0 \le u \le 1,$$

so that across all interleaves the turns are `1/RadialOversampling` cycles per field of
view apart, finer than the $1/\mathrm{FOV}$ that avoids aliasing. The simulator traverses
the spiral at constant speed in k-space (constant angular velocity in a short central region,
`CentreRegionMs`), and its `SamplingBound` is that the speed times the dwell time stays below
one cycle per field of view, so that consecutive samples are never farther apart than the
grid they are meant to represent; it does not model gradient amplitude or slew limits,
which on a scanner set how fast the outer turns can be traversed. There is no phase-encode
axis, so three things disappear with it: `PhaseEncodingDirection` and the Nyquist ghost,
since no two lines are read in opposite directions; partial Fourier, since no lines exist to
skip; and the shift of an off-resonant voxel. The phase a field offset accrues still grows
with the time since the center of k-space, but that time grows along the spiral in every
direction at once, so the voxel is spread radially and blurred rather than displaced
([Chapter 11](../03-preprocessing/11-susceptibility-distortion.md)). Two things appear.
The samples do not lie on a grid, so the inverse DFT cannot be applied as it stands; the
reconstruction either grids them, interpolating each sample onto the Cartesian grid with a
weight for the local sampling density before the FFT {cite:p}`jackson1991,pipe1999`, or
solves the forward model by least squares. The simulator does the latter, as its
`Reconstruction` block describes, with Pipe-Menon density weights, restricted to the disc
of radius `KMax`; the in-plane resolution is therefore set by a disc rather than a square
and the in-plane point-spread function is circularly symmetric. And because the trajectory
is radially symmetric, the matrix is square (68 × 68) rather than 64 × 68.

```{code-cell} python
:tags: [hide-input]
sp, p_sp = spiral.simulation()["Readout"], spiral.sidecar(); traj = sp["Trajectory"]
kmax, turns, n_int, t_ro, dwell, n_s = traj["KMax"], traj["Turns"], sp["Interleaves"], traj["ReadoutTimeMs"], traj["DwellTimeMs"], traj["SamplesPerInterleaf"]
a = kmax / (2 * np.pi * turns)                                                # k = a θ e^{iθ}: an Archimedean spiral
theta = np.linspace(0, 2 * np.pi * turns, 20001)
arc = a / 2 * (theta * np.sqrt(1 + theta ** 2) + np.arcsinh(theta))             # its arc length
th_t = np.interp(arc[-1] * np.arange(n_s) * dwell / t_ro, arc, theta)           # constant speed along the spiral: θ at every sample
fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
for s in range(n_int):
    kk = a * th_t * np.exp(1j * (th_t + 2 * np.pi * s / n_int))
    axes[0].plot(kk.real, kk.imag, lw=0.6, color=plt.cm.viridis(s / (n_int - 1)))
axes[0].add_patch(plt.Circle((0, 0), kmax, fill=False, color="0.5", lw=0.8, ls="--"))
axes[0].add_patch(plt.Rectangle((-kmax, -kmax), 2 * kmax, 2 * kmax, fill=False, color="0.5", lw=0.8, ls=":"))
axes[0].set(aspect="equal", xlabel="$k_x$ (cycles/FOV)", ylabel="$k_y$ (cycles/FOV)", title=f"{n_int} interleaves (color = shot)")
kk = a * th_t * np.exp(1j * th_t)
axes[1].plot(kk.real[:300], kk.imag[:300], ".-", ms=2, lw=0.5, color=PALETTE[0])
axes[1].set(aspect="equal", xlabel="$k_x$", ylabel="$k_y$", title="the first 300 samples of one interleaf")
show_slice(axes[2], cm["spiral"], k, "spiral: control mean", vmin=0, vmax=7500)
show_slice(axes[3], cm["grase-4shot"], k, "grase-4shot: control mean", vmin=0, vmax=7500)
fig.tight_layout()
print(f"Trajectory: KMax {kmax:g} cycles/FOV, Turns {turns:g} per interleaf, Interleaves {n_int}, ReadoutTimeMs {t_ro:g}, DwellTimeMs {dwell:g}, SamplesPerInterleaf {n_s}, "
      f"RadialOversampling {traj['RadialOversampling']['Value']:g}, CentreRegionMs {traj['CentreRegionMs']:.3f} ({traj['CentreRegionMs'] / dwell:.1f} samples)")
print(f"turns {kmax / (turns * n_int):.3f} cycles/FOV apart across the interleaves (= 1/RadialOversampling); arc length {arc[-1]:.0f} cycles/FOV per interleaf in {t_ro:g} ms, "
      f"so speed x dwell = {arc[-1] / t_ro * dwell:.2f} cycles/FOV per sample; SamplingBound: {traj['SamplingBound']}")
print(f"samples per partition: {n_int * n_s} on the spiral against {NY * NY} grid points; EchoTime {p_sp['EchoTime'] * 1e3:.0f} ms = EchoSpacingMs {sp['EchoSpacingMs']:g}; NumberShots {sp['NumberShots']}, VolumeDuration {sp['VolumeDuration']:g} s")
print("sidecar: " + ", ".join(f"{f} {p_sp[f] if f in p_sp else 'absent'}" for f in ["PhaseEncodingDirection", "TotalReadoutTime", "EffectiveEchoSpacing", "PartialFourier", "DwellTime"]))
print(f"Reconstruction: {sp['Reconstruction']['Method']}; DensityCompensation: {sp['Reconstruction']['DensityCompensation']}")
print(f"RMS difference between the spiral and the four-shot GRASE control means over the shared 64 columns: {rms(cm['spiral'][:NX], cm['grase-4shot'], RUNS['grase-4shot'].mask()):.0f} image units")
```

Left: the eight interleaves, each 5.1 turns out to 34 cycles per field of view (the
dashed circle; the dotted square is the 68 × 68 Cartesian grid, whose corners the spiral
never visits), each rotated by an eighth of a turn from the last so that their turns
interleave 0.833 cycles/FOV apart. Second panel: the first 300 of an interleaf's 1250
samples, 4 µs apart; at constant speed the samples are 0.44 cycles/FOV apart along the path,
within the bound of 1. The sketch neglects the constant-angular-velocity center, which lasts
0.015 ms, 3.8 samples. The 5 ms interleaf is a quarter of the one-shot GRASE's 20 ms EPI block and
the same as the four-shot's segment, but it has to be repeated eight times to cover the
partition at this density, so a volume takes eight shots, 36 s, twice the four-shot GRASE's.
The two control images on the right are the same object through the two readouts, with the
same echo train (ESP 12 ms against 10), and they differ by 187 image units RMS in the brain
over the columns they share: their noise, their slightly different windows and echo times,
and their different in-plane truncation, a disc against a rectangle, which rings
differently at every edge. The spiral sidecar has no `PhaseEncodingDirection`,
`TotalReadoutTime`, or `EffectiveEchoSpacing`, which is correct and which a pipeline that
expects them must tolerate; `DwellTime` is there instead. Off-resonance is not applied in
this run (the `FieldmapPresent` flag is false); the `fieldmap` variants of this dataset and
[Chapter 11](../03-preprocessing/11-susceptibility-distortion.md) show the blur, which the
simulator applies by time segmentation of the trajectory (`TimeSegmentation`).

## Measure it: the noise of a 3D volume

The 3D reconstruction sums $n_z$ partitions, each carrying independent noise, so a voxel's
noise standard deviation is $1/\sqrt{n_z}$ of the 2D value at the same noise per k-space
sample; the simulator states this for the Cartesian readout and says that the spiral's
noise after the least squares is measured, not asserted. The cell measures both the way the
2D noise was measured above, as the temporal standard deviation over the control volumes in
pure white matter, and then does the accounting the $\sqrt{n_z}$ alone leaves out.

```{code-cell} python
:tags: [hide-input]
dur = {n: RUNS[n].simulation().get("Readout", {}).get("VolumeDuration", r.repetition_time) for n in RUNS}
sd, gm_mean = {}, {}
for n in ["ref-pcasl (2D EPI)", "grase-4shot", "spiral"]:
    pure = RUNS[n].fraction("wm") > 0.99
    sd[n], gm_mean[n] = series[n].std(axis=-1, ddof=1)[pure].mean(), cm[n][gm3[n]].mean()
    print(f"{n}: temporal SD over the {series[n].shape[-1]} control volumes in {pure.sum()} pure-WM voxels {sd[n]:.2f} image units; GM control {gm_mean[n]:.0f}, SNR per volume {gm_mean[n] / sd[n]:.0f}; VolumeDuration {dur[n]:g} s")
s2, s3, ss = sd.values()
print(f"ratio 2D / GRASE {s2 / s3:.2f} (the simulator's statement is 1/sqrt(nz): sqrt(20) = {np.sqrt(20):.2f}); 2D / spiral {s2 / ss:.2f}; spiral / GRASE {ss / s3:.2f}")
print(f"GRASE sidecar Noise: {RUNS['grase-4shot'].simulation()['Readout']['Noise']}\nspiral sidecar Noise: {RUNS['spiral'].simulation()['Readout']['Noise']}")
for n in ["grase-4shot", "spiral"]:
    nv = dur[n] / dur["ref-pcasl (2D EPI)"]
    print(f"in one {n} volume's {dur[n]:g} s the 2D protocol acquires {nv:.0f} volumes, whose mean has SD {s2 / np.sqrt(nv):.1f}: the 3D gain per unit scan time is {s2 / np.sqrt(nv) / sd[n]:.2f} "
          f"(sqrt(20 / {nv:.0f}) = {np.sqrt(20 / nv):.2f}); in GM SNR, with the lower 3D signal, {(gm_mean[n] / sd[n]) / (gm_mean['ref-pcasl (2D EPI)'] / (s2 / np.sqrt(nv))):.2f}")
R4 = RUNS["grase-4shot"].simulation()["Readout"]
n_cart, t_cart = NX * R4["LinesPerEcho"] * R4["KySegments"], R4["LinesPerEcho"] * R4["KySegments"] * R4["LineSpacingMs"]
print(f"samples per partition: GRASE {n_cart} in {t_cart:.1f} ms of sampling, spiral {n_int * n_s} in {n_int * t_ro:g} ms; sqrt({n_cart}/{n_int * n_s}) = {np.sqrt(n_cart / (n_int * n_s)):.2f}, "
      f"times the disc's sqrt(pi/4) = {np.sqrt(n_cart / (n_int * n_s) * np.pi / 4):.2f}, against the measured spiral / GRASE {ss / s3:.2f}")
fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2))
for ax, n in zip(axes, sd):
    show_slice(ax, series[n].std(axis=-1, ddof=1), k, f"{n}: temporal SD", vmin=0, vmax=60)
fig.tight_layout()
```

The 2D series gives 39.75, the configured σ. The four-shot GRASE gives 8.81, a ratio of
4.51 against the stated $\sqrt{20} = 4.47$, so the Cartesian statement holds. The spiral
gives 4.78, a ratio of 8.32 to the 2D value and 0.54 of the GRASE value, lower than the
sidecar's "about the Cartesian 3D value". The reason is the sample count: the spiral reads
10000 samples per partition in 40 ms of sampling, the GRASE 4352 in 20.4 ms, and at the
same noise per sample (the simulator's convention, which on a scanner holds when the
receiver bandwidth is the same) the least squares averages more of them;
$\sqrt{4352/10000} = 0.66$, and 0.58 with the disc band that discards the corners of
k-space, close to the measured 0.54. The spiral's gain is bought with sampling time, not for
free. The maps on the right show the three noise levels at the same window: the 2D map at
40 inside the brain and lower outside (the Rayleigh background's SD of 0.66 σ), the 3D maps
nearly flat and dark.

The honest accounting is per unit scan time. A four-shot volume takes 18 s against 4.5 s,
and in 18 s the 2D protocol averages four volumes to an SD of 19.9, so the GRASE's advantage
per unit time is 2.26, which is $\sqrt{20/4} = 2.24$: the $\sqrt{n_z}$ gain divided by the
$\sqrt{\text{shots}}$ it cost. The spiral's eight shots take 36 s, eight 2D volumes, and its
advantage per unit time is 2.94. Both figures are for the noise alone; the 3D gray-matter
signal is lower, by the first-echo amplitude and the blur of the sections above, so the
control-image SNR gain per unit time in gray matter is 2.08 for the GRASE and 2.61 for the
spiral, and the perfusion-weighted difference, confined to the cortex, loses more to the
blur than the control image does. The 2D versus 3D comparison in practice, with background
suppression and the artifacts each readout brings, is the subject of
{cite:t}`vidorreta2013` and of [Chapter 13](../03-preprocessing/13-assembled-pipeline.md).

## What this implies for acquisition

- **Voxel size and FOV are k-space decisions.** Smaller voxels need more lines, a longer
  train, and therefore more distortion and blur; the ASL matrix is small (64 × 68) because
  the perfusion signal is too weak for small voxels, and its readout is correspondingly short.
- **The readout length drives the EPI artifacts.** Distortion is the off-resonance times
  `TotalReadoutTime`, along `PhaseEncodingDirection`; both fields must be correct before any
  distortion correction ([Chapter 11](../03-preprocessing/11-susceptibility-distortion.md)).
  In-plane acceleration shortens the total readout time; partial Fourier shortens only the
  train and the minimum TE.
- **The magnitude image has a noise floor.** Background and low-signal voxels are biased
  upward by up to 1.25 σ; the perfusion-weighted difference is only a few σ per pair
  ([Chapter 8](../03-preprocessing/08-noise.md)).
- **Ringing is universal**, at the CSF boundaries, at the brain surface, and into the
  "empty" background; estimate the noise from the series in time, not from the background,
  when precision matters.
- **Keep the phase.** It costs nothing to store and carries the field information a
  distortion correction may need.
- **A 3D readout reads the whole slab at one delay** and gains $\sqrt{n_z}$ in noise per
  volume, but a volume costs `NumberShots` TRs, so the gain per unit scan time is
  $\sqrt{n_z / \text{shots}}$, and the first echo's amplitude and the through-plane blur
  take back part of the gray-matter signal.
- **The echo train sets the through-plane blur.** Its length against T2 decides the $k_z$
  window and the PSF; segmenting the train over shots keeps it short, at the price that
  anything changing between shots (motion, physiology, the label) becomes an inconsistency
  inside a volume ([Chapter 10](../03-preprocessing/10-motion.md),
  [Chapter 13](../03-preprocessing/13-assembled-pipeline.md)). A spiral trades the EPI
  block's ghost and displacement for an off-resonance blur and a non-Cartesian
  reconstruction.

## Further reading

Imaging with gradients {cite:p}`lauterbur1973`, its k-space description
{cite:p}`ljunggren1983,twieg1983` and echo-planar imaging {cite:p}`mansfield1977`; the
noise of magnitude images {cite:p}`henkelman1985,gudbjartsson1995`; coil arrays
{cite:p}`roemer1990`, SENSE {cite:p}`pruessmann1999`, GRAPPA {cite:p}`griswold2002` and
its g-factor {cite:p}`breuer2009`; partial Fourier acquisition and reconstruction
{cite:p}`feinberg1986,noll1991`; EPI distortion from field offsets {cite:p}`jezzard1995`;
GRASE {cite:p}`oshio1991` and its single-shot 3D use for ASL
{cite:p}`gunther2005,fernandezseara2005`, the echo amplitudes of refocusing trains below
180° {cite:p}`hennig1988` and the extended phase graph {cite:p}`weigel2015`; spiral imaging
{cite:p}`meyer1992`, gridding {cite:p}`jackson1991` and density compensation
{cite:p}`pipe1999`; the recommended 3D readouts for ASL {cite:p}`alsop2015` and a 2D versus
3D comparison {cite:p}`vidorreta2013`; the BIDS standard {cite:p}`gorgolewski2016`; the
textbooks {cite:t}`haacke1999` and {cite:t}`nishimura2010`.
