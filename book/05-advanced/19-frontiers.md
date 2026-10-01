---
title: "19. Frontiers"
kernelspec:
  name: python3
  display_name: Python 3
---

:::{admonition} Simulated datasets in this chapter
:class: note
- **Built in this page:** single-voxel signals from the kinetic model of [Chapter 5](../02-labeling/05-kinetic-model.md), extended in the page with a second compartment, a dispersion kernel, an arterial term, and a 3D echo train ([Appendix B](../appendices/b-data-manifest.md#app-b-package-data)).

None of the methods of this chapter is simulated by aslscan; the last section says what adding each would take.
:::

## Learning goals

After this chapter you can:

- describe what velocity-selective labeling, vessel encoding, multi-echo readouts, 3D
  readouts, dispersion and macrovascular models, and learned processing each add to the
  chain this book has followed
- explain, with a simulation, why a velocity-selective label is insensitive to transit
  time, why the echo-time decay of ΔM reads the exchange of label into tissue, and why a
  segmented 3D GRASE readout blurs through-plane
- state the bias that dispersion and arterial signal produce in a standard multi-delay
  fit, and what a model that includes them has to add
- say what aslscan does not model and how each omission would be added

```{code-cell} python
:tags: [hide-cell]
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.stats import gamma as gamma_dist

from aslbook import kinetic, presets
from aslbook.plotting import INK, PALETTE, TISSUE_COLORS, set_style

set_style()
GM = presets.TISSUES["GM"]
SCALE = presets.REFERENCE.signal_scale
T1B, LAM = presets.T1_BLOOD, presets.LAMBDA
M0B = GM.m0 / LAM


def deltam(t, perfusion=GM.perfusion, att=GM.att, **kw):
    """The gray-matter kinetic curve in image units."""
    return SCALE * kinetic.delta_m(t, perfusion, att, GM.t1, GM.m0, **kw)
```

## Velocity-selective labeling

Every labeling scheme so far defines the label by *position*: a plane or a slab below the
brain, from which the labeled blood has to travel to the tissue. The arterial transit time
is the price of that geometry, and Chapters [15](../04-quantification/15-multi-delay.md) and
[18](./18-time-encoded-and-look-locker.md) spent their scan time measuring it. Velocity-selective
ASL {cite:p}`wong2006` defines the label by *velocity* instead. A labeling module of
radiofrequency pulses interleaved with bipolar gradients, applied without any spatial
selection, leaves the magnetization of stationary and slowly moving spins alone and
saturates spins that move faster than a cutoff velocity $V_c$ along the gradient
direction. Arterial blood decelerates continuously from the large arteries (tens of
cm/s) to the capillaries (about 1 mm/s), so with $V_c$ of 1 to 2 cm/s the label is created
in the arterioles of every voxel, wherever they are, and reaches the tissue within a
fraction of a second. The control module has the same pulses without the velocity
weighting.

The ideal saturation profile is a cosine in velocity, $M_z(v) = M_0 \cos(\pi v / 2V_c)$,
which is zero at $V_c$ and oscillates beyond it. In a real artery the oscillation does not
matter: the laminar flow profile spans velocities from zero to twice the mean, the cosine
averages over them, and the average falls to zero at a mean velocity of $V_c$ and stays
near zero beyond it. With the second module applied just before the readout to end the
bolus, the signal is that of a bolus of duration TI that arrived at time zero:

$$
\Delta M(\mathrm{TI}) = 2\,\alpha\, M_{0\mathrm{b}}\, f\, \mathrm{TI}\, e^{-\mathrm{TI}/T_{1\mathrm{b}}} ,
$$

the pulsed-label curve of [Chapter 5](../02-labeling/05-kinetic-model.md) with the transit
time set to zero. A saturation module produces a difference of at most $M_{0\mathrm{b}}$
where an inversion produces $2 M_{0\mathrm{b}}$, so its efficiency in this formula is at
most 0.5, which velocity-selective *inversion* modules developed later recover.

```{code-cell} python
:tags: [hide-input]
VC = 2.0                                                    # cm/s
v = np.linspace(0, 6, 601)
plug = np.cos(np.pi * v / (2 * VC))
laminar = np.where(v > 0, VC / (np.pi * np.maximum(v, 1e-9)) * np.sin(np.pi * v / VC), 1.0)
ti = np.linspace(0, 3.0, 301)
curves = {
    "velocity-selective (α = 0.5, no transit)": deltam(ti, att=0.0, label_type="PASL", tau=10.0, alpha=0.5),
    "pulsed (FAIR), ATT 0.8 s": deltam(ti, att=0.8, label_type="PASL", tau=0.7),
    "pulsed (FAIR), ATT 2.0 s": deltam(ti, att=2.0, label_type="PASL", tau=0.7),
}
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 3.3))
ax0.plot(v, plug, color=PALETTE[0], label="one velocity: cos(πv / 2$V_c$)")
ax0.plot(v, laminar, color=PALETTE[1], label="averaged over a laminar profile")
ax0.axhline(0, color="k", lw=0.6)
ax0.axvline(VC, color=INK["secondary"], ls=":", lw=1)
ax0.text(VC + 0.08, 0.85, "$V_c$", fontsize=9)
ax0.axvspan(VC, 6, color=PALETTE[1], alpha=0.08)
ax0.text(4.3, 0.35, "labeled: arteries\nand arterioles", ha="center", fontsize=8, color=PALETTE[1])
ax0.text(0.9, -0.55, "spared:\ntissue, capillaries", ha="center", fontsize=8, color=INK["secondary"])
ax0.set(xlabel="velocity along the gradient (cm/s)", ylabel="$M_z$ after the module / M₀", title=f"a velocity-selective saturation profile, $V_c$ = {VC:.0f} cm/s")
ax0.legend(fontsize=7, loc="upper right")
for (name, c), color in zip(curves.items(), (PALETTE[2], TISSUE_COLORS["GM"], PALETTE[3])):
    ax1.plot(ti, c, color=color, label=name)
ax1.set(xlabel="TI (s)", ylabel="ΔM (image units)", title="gray matter: the label does not wait for transit")
ax1.legend(fontsize=7)
fig.tight_layout()
at = 1.8
print("ΔM at TI = 1.8 s (image units): " + ", ".join(f"{n} {np.interp(at, ti, c):.1f}" for n, c in curves.items()))
```

Left: the saturation profile. For spins at a single velocity (blue) the profile is a cosine
that first reaches zero at the cutoff; averaged over the laminar profile of an artery
(orange), everything above the cutoff is effectively saturated, and everything well below
it, tissue water and capillary blood, is spared. Right: the resulting curve for a
gray-matter voxel against a pulsed label with a bolus cut-off. The velocity-selective
signal rises from time zero and is the same whatever the transit time, at the reduced
efficiency of a saturation label; the spatially labeled signal is larger when the transit
time is short, and at TI = 1.8 s has not arrived at all when it is 2.0 s (printed). This is
the case that motivates the method: long or unknown transit times, as in cerebrovascular
disease, the elderly, and any territory fed by collaterals. Its costs are the lower
efficiency, a static-tissue signal loss and diffusion weighting from the labeling
gradients, sensitivity to eddy currents in the control-label match, and a bolus whose
duration is set by the timing rather than measured; the readout itself, and every artifact
of Part III, are unchanged.

## Vessel-encoded ASL and territory mapping

The pseudo-continuous labeling of [Chapter 4](../02-labeling/04-labeling-schemes.md) inverts
every artery crossing its plane. Adding a gradient across the plane during the pulse train
makes the inversion efficiency vary sinusoidally with position along that gradient, so that
arteries at the maxima are labeled and arteries half a period away are left in the control
state. Cycling through a few such patterns, left-right and anterior-posterior with
different phases, gives each artery a distinct label-control signature across the cycles,
and a linear decoding of the same kind as
[Chapter 18](./18-time-encoded-and-look-locker.md), or a clustering of the signatures,
assigns every voxel's perfusion to the artery that delivered it: a *territory map* of the
carotid and vertebral supplies {cite:p}`vanosch2018`. Because every cycle labels about half
of the arteries, the total perfusion image costs no more scan time than a conventional
PCASL scan of the same length. Super-selective variants rotate the gradient during the
train so that only one vessel of choice is labeled. The maps are used to see collateral
flow after an occlusion, to plan and follow bypass surgery, and to attribute a perfusion
deficit to its vessel.

```{code-cell} python
:tags: [hide-input]
vessels = {"R ICA": (-1.5, 1.0), "L ICA": (1.5, 1.0), "R VA": (-0.6, -1.4), "L VA": (0.6, -1.4)}
x = np.linspace(-3.5, 3.5, 201)
y = np.linspace(-2.6, 2.6, 151)
X, Y = np.meshgrid(x, y)
cycles = {
    "cycle 1: label all (plain PCASL)": np.ones_like(X),
    "cycle 2: left-right modulation": 0.5 * (1 + np.cos(np.pi * (X + 1.5) / 3.0)),
    "cycle 3: anterior-posterior modulation": 0.5 * (1 + np.cos(np.pi * (Y - 1.0) / 2.4)),
}
fig, axes = plt.subplots(1, 3, figsize=(11, 3.1))
for ax, (title, eff) in zip(axes, cycles.items()):
    ax.imshow(eff, extent=(x[0], x[-1], y[0], y[-1]), origin="lower", cmap="Oranges", vmin=0, vmax=1.3, alpha=0.8)
    for name, (vx, vy) in vessels.items():
        e = float(eff[np.argmin(abs(y - vy)), np.argmin(abs(x - vx))])
        ax.add_patch(Circle((vx, vy), 0.35, facecolor=PALETTE[1] if e > 0.5 else "white", edgecolor=INK["primary"], lw=1.2))
        ax.text(vx, vy - 0.75, f"{name}\n{'label' if e > 0.5 else 'control'}", ha="center", va="top", fontsize=7)
    ax.set(title=title, xticks=[], yticks=[], xlabel="labeling plane, seen from below")
    ax.grid(False)
fig.tight_layout()
print("signatures (1 = label) across the three cycles: " + "; ".join(
    f"{n} {[int(float(eff[np.argmin(abs(y - vy)), np.argmin(abs(x - vx))]) > 0.5) for eff in cycles.values()]}" for n, (vx, vy) in vessels.items()))
```

Three encoding cycles seen in the labeling plane: the orange shading is the inversion
efficiency the modulated pulse train produces across the plane, and each circle is one of
the four arteries, filled when it lies at a labeled position. With a control cycle added,
the four cycles give every artery a distinct signature (printed), and the decoding
separates the four territories in the same way the Hadamard decoding separated the
sub-boli of [Chapter 18](./18-time-encoded-and-look-locker.md). The geometry of the real
arteries, which are not symmetric, is measured from an angiogram before the scan and the
modulation periods are set to it.

## Multi-echo ASL and water exchange

The kinetic model assumes that labeled water leaves the capillaries the moment it arrives
and thereafter shares the tissue's relaxation. Water is not a freely diffusible tracer at
the capillary wall on the time scale of ASL: its exchange from blood to tissue takes a
few hundred milliseconds, so at the readout part of the label is still intravascular
{cite:p}`stlawrence2000`. The echo time can tell the two apart. Label in blood decays
with the blood's $T_2$ of 165 ms at 3 T; label that has entered gray matter decays with the
tissue's 80 ms. The difference signal at several echo times is then a two-component decay
whose composition is the fraction of label still in blood, and a multi-echo (multi-TE)
ASL readout measures the exchange time from it {cite:p}`gregori2013`. The cell extends the
kinetic model with a blood compartment that empties into the tissue with an exchange time
$T_\mathrm{ex}$, and reads the result at a range of echo times.

```{code-cell} python
:tags: [hide-input]
T2_T, T2_B = GM.t2, presets.T2_BLOOD
DT = 0.001


def two_compartment(t_ex, t_end=3.6, tau=1.8, tissue=GM):
    """Label in blood and in tissue against time: delivery into blood, exchange at 1/t_ex into tissue,
    T1b decay in blood and T1' decay in tissue (M0 units of the voxel)."""
    n = int(round(t_end / DT))
    tg = (np.arange(n) + 0.5) * DT
    f, alpha = tissue.perfusion / 6000.0, presets.ALPHA["PCASL"]
    t1p = kinetic.t1_prime(tissue.t1, tissue.perfusion)
    delivery = 2 * M0B * f * alpha * np.exp(-tissue.att / T1B) * ((tg > tissue.att) & (tg < tissue.att + tau))
    mb, mt = np.zeros(n), np.zeros(n)
    b = t = 0.0
    for i in range(n):
        b, t = b + DT * (delivery[i] - b * (1 / t_ex + 1 / T1B)), t + DT * (b / t_ex - t / t1p)
        mb[i], mt[i] = b, t
    return tg, SCALE * mb, SCALE * mt


TE = np.linspace(0, 0.15, 151)
T_EX = (0.1, 0.3, 0.6, 1.0)
tg, mb_ref, mt_ref = two_compartment(0.001)
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 3.3))
tg, mb, mt = two_compartment(0.3)
ax0.plot(tg, mb + mt, color=INK["secondary"], lw=1.2, label="total label")
ax0.plot(tg, mb, color=PALETTE[7], label="in blood (T2 165 ms)")
ax0.plot(tg, mt, color=TISSUE_COLORS["GM"], label="in tissue (T2 80 ms)")
ax0.plot(tg, deltam(tg), "k:", lw=1, label="kinetic model (instant exchange)")
ax0.axvline(3.6, color="k", lw=0.6)
ax0.set(xlabel="time from the start of labeling (s)", ylabel="label (image units)", title="T_ex = 0.3 s: where the label is, up to the readout at 3.6 s")
ax0.legend(fontsize=7)
rows = []
for t_ex, color in zip(T_EX, plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(T_EX)))):
    tg, mb, mt = two_compartment(t_ex)
    dm_te = mb[-1] * np.exp(-TE / T2_B) + mt[-1] * np.exp(-TE / T2_T)
    ax1.plot(1e3 * TE, dm_te / dm_te[0], color=color, label=f"T_ex = {t_ex:.1f} s")
    sel = TE <= 0.1
    t2_app = -1 / np.polyfit(TE[sel], np.log(dm_te[sel]), 1)[0]
    rows.append((t_ex, mb[-1] / (mb[-1] + mt[-1]), t2_app, (mb[-1] + mt[-1]) / deltam(3.6)))
ax1.plot(1e3 * TE, np.exp(-TE / T2_B), color=PALETTE[7], ls="--", lw=1, label="all in blood")
ax1.plot(1e3 * TE, np.exp(-TE / T2_T), color=TISSUE_COLORS["GM"], ls="--", lw=1, label="all in tissue")
ax1.set(xlabel="echo time (ms)", ylabel="ΔM(TE) / ΔM(0)", title="the decay of ΔM with TE reads the exchange", yscale="log")
ax1.legend(fontsize=7)
fig.tight_layout()
print(f"instant exchange check: two-compartment total at 3.6 s {mb_ref[-1] + mt_ref[-1]:.1f} vs kinetic.delta_m {deltam(3.6):.1f} image units")
for t_ex, fb, t2a, ratio in rows:
    print(f"T_ex {t_ex:.1f} s: {fb:.0%} of the label still in blood at the readout, apparent T2 {1e3 * t2a:.0f} ms, "
          f"total label {ratio:.2f} of the instant-exchange model; CBF from the blood-T2 calibration at TE 30 ms off by {np.exp(-0.03 / t2a) / np.exp(-0.03 / T2_B) - 1:+.0%}")
```

Left: with an exchange time of 0.3 s, the label arrives into the blood compartment (red),
crosses into the tissue (orange), and once the bolus ends the blood compartment empties
within a second; the total (gray) is close to the instant-exchange curve (dotted) because
the two compartments' $T_1$ values are similar. Right: the echo-time decay of the
difference signal at the reference readout, 1.8 s after a 1.8 s bolus, for four exchange
times. At that late readout the exchange is nearly complete for exchange times up to a few
hundred milliseconds: 1 % of the label is still in blood at 0.3 s, 9 % at 0.6 s, and 23 %
at 1.0 s, and the apparent $T_2$ printed above moves from the tissue's 80 ms only to 94 ms.
A multi-TE measurement of the exchange time therefore reads out at short delays, while the
bolus is still arriving and the blood compartment is full, where the two curves of the
left panel are far apart. The right panel carries a second message for single-echo
quantification. [Chapter 16](../04-quantification/16-calibration.md) assumed the label has
the blood's $T_2$ at the echo time, which is also aslscan's assumption; at a late readout
most of the label has exchanged, the difference signal at TE = 30 ms is 13 to 18 % smaller
than that assumption predicts, and CBF is underestimated by the same amount. At the
reference echo time of 12 ms the error is less than half of that. Water exchange is also a
quantity of interest in itself, as a measure of blood-brain barrier permeability.

## 3D readouts

The white paper recommends a segmented 3D readout with background suppression
{cite:p}`alsop2015`, and the book's 2D EPI reference protocol is the simulator's limit, not
the field's practice. Two 3D readouts are used. 3D GRASE {cite:p}`gunther2005,fernandezseara2005`
excites the whole slab, then follows a train of refocusing pulses, each spin echo carrying
an EPI readout of one through-plane partition of k-space ($k_z$); a stack of spirals does
the same with a spiral trajectory in each partition. Both read every voxel at the same
time after the label, which removes the slice timing of
[Chapter 6](../02-labeling/06-the-asl-signal.md) and lets a single pair of background
suppression pulses null the whole volume, and both have the SNR of a volume excitation. The
cost is the echo train: the partitions are acquired one spin echo apart, and the signal
decays with $T_2$ across the train, so the $k_z$ axis is weighted by a decaying window,
whose Fourier transform is a through-plane point-spread function wider than one partition.
Centric ordering puts $k_z = 0$ at the first echo and the decay on the edges of k-space, so
the effect is blur rather than signal loss; a single-shot train over 20 partitions lasts
longer than $T_2$, which is why the readout is segmented across several excitations
{cite:p}`vidorreta2013`.

```{code-cell} python
:tags: [hide-input]
N_KZ, ESP = 20, 0.030                                        # partitions; s per spin echo


def kz_window(t2, n_segments):
    """Weight of each k_z partition from T2 decay along a centric-ordered echo train shared by the segments."""
    kz = np.arange(-N_KZ // 2, N_KZ // 2)
    order = np.argsort(np.abs(kz - 0.25))                    # 0, -1, 1, -2, 2, ... : centric
    echo = np.empty(N_KZ)
    echo[order] = np.arange(N_KZ) // n_segments               # echo index of each partition
    return kz, np.exp(-(echo + 1) * ESP / t2), (echo + 1) * ESP


def psf(window, pad=16):
    """Through-plane point-spread function of a k_z window (zero-padded in the center of k-space)
    and the full width at half maximum of its main lobe, in slices."""
    big = np.zeros(pad * N_KZ)
    start = pad * N_KZ // 2 - N_KZ // 2
    big[start:start + N_KZ] = window
    p = np.abs(np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(big))))
    z = np.fft.fftshift(np.fft.fftfreq(pad * N_KZ, d=1.0 / N_KZ))   # slices
    p /= p.max()
    c = np.argmax(p)
    left, right = c, c
    while p[left - 1] >= 0.5:
        left -= 1
    while p[right + 1] >= 0.5:
        right += 1
    return z, p, z[right] - z[left]


configs = [("single shot", 1), ("2 segments", 2), ("4 segments", 4)]
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 3.3))
for (name, seg), color in zip(configs, PALETTE[:3]):
    kz, w, te = kz_window(GM.t2, seg)
    ax0.plot(kz, w, "o-", ms=4, color=color, label=f"{name}: train {N_KZ // seg * ESP * 1e3:.0f} ms")
    z, p, fwhm = psf(w)
    ax1.plot(z, p, color=color, label=f"{name}: FWHM {fwhm:.2f} slices")
    print(f"{name:12s}: echo train {N_KZ // seg * ESP * 1e3:4.0f} ms, k_z edge weight {w.min():.3f}, through-plane FWHM {fwhm:.2f} slices (tissue T2 {1e3 * GM.t2:.0f} ms)")
z, p, fwhm_b = psf(kz_window(presets.T2_BLOOD, 1)[1])
ax1.plot(z, p, color=PALETTE[7], ls="--", lw=1, label=f"single shot, blood T2: FWHM {fwhm_b:.2f}")
ax0.set(xlabel="k_z partition", ylabel="weight from T2 decay", title=f"the k_z window of a centric 3D GRASE train, {ESP * 1e3:.0f} ms per echo, T2 {1e3 * GM.t2:.0f} ms")
ax0.legend(fontsize=7)
ax1.set(xlabel="through-plane position (slices)", ylabel="point-spread function", xlim=(-4, 4), title="what a point becomes along z")
ax1.legend(fontsize=7)
fig.tight_layout()
print(f"single shot with the label's T2 of {1e3 * presets.T2_BLOOD:.0f} ms: FWHM {fwhm_b:.2f} slices")
```

Left: the weight each $k_z$ partition receives in a centric-ordered train, for the book's
20 partitions at 30 ms per echo. In a single shot the outer partitions are read 600 ms
after excitation and keep a thousandth of the gray-matter signal; with four segments the
train is 150 ms and the edge keeps 15 %. Right: the resulting through-plane point-spread
function, whose main-lobe widths are printed. A single-shot train at this echo spacing
blurs a point over several slices; segmentation narrows it toward the 1.1 slices of an
unweighted window, at the cost of one excitation, and therefore one label, per segment,
and of sensitivity to motion between segments. The
dashed curve is the same single-shot train for the label's longer $T_2$: the label and the
static tissue are blurred by different amounts, so the subtraction of a 3D GRASE pair is
not exactly the tissue-free difference the 2D chapters assumed. Stack-of-spirals readouts
have the same $k_z$ window with a spiral's own in-plane blurring from off-resonance in
place of EPI's displacement.

## Bolus dispersion and the macrovascular signal

The kinetic model's bolus arrives as a sharp-edged box: nothing before the transit time,
the full concentration after it. Real labeled blood spreads along its path, because the
velocities across a vessel differ and the paths through the arterial tree differ, so the
edges of the bolus are smeared by the time they reach the tissue. And before the label
reaches the capillaries it passes through arteries within the voxel, where it contributes
a difference signal that has nothing to do with perfusion into that voxel. Both effects
are largest at short delays, which is where a multi-delay or time-encoded acquisition
samples. A common description {cite:p}`chappell2010` adds two things to the model: a
dispersion kernel, here a gamma-variate, convolved with the delivery, and an arterial
compartment with a blood volume fraction $a\mathrm{BV}$, whose label arrives at an earlier
arterial arrival time $\delta_\mathrm{a}$ and is never exchanged:

$$
\Delta M_\mathrm{a}(t) = 2\,\alpha\, M_{0\mathrm{b}}\; a\mathrm{BV}\; c_\mathrm{a}(t) ,
$$

with $c_\mathrm{a}$ the (dispersed) arterial concentration, decayed by $T_{1\mathrm{b}}$.
The cell builds the delivery as the integral it is, so the kernel can be inserted, and
samples the total at the six delays of the multi-delay dataset of
[Chapter 15](../04-quantification/15-multi-delay.md).

```{code-cell} python
:tags: [hide-input]
TAU, ALPHA = 1.8, presets.ALPHA["PCASL"]
DTG = 0.002
tg = (np.arange(int(5.2 / DTG)) + 0.5) * DTG
T1P = kinetic.t1_prime(GM.t1, GM.perfusion)
PLDS = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
T_SAMPLE = PLDS + TAU


def concentration(arrival, disp_mean, shape=2.0):
    """Arterial concentration at the voxel (fraction of M0b, times 2 alpha): a box of the labeling duration
    leaving the plane, delayed by `arrival`, spread by a gamma kernel of the given mean, decayed with T1b."""
    box = ((tg >= 0) & (tg < TAU)).astype(float)
    if disp_mean <= 0:
        kernel = np.zeros_like(tg); kernel[0] = 1.0 / DTG
    else:
        kernel = gamma_dist.pdf(tg, shape, scale=disp_mean / shape)
    spread = np.convolve(box, kernel * np.exp(-tg / T1B))[: len(tg)] * DTG
    shifted = np.interp(tg - arrival, tg, spread, left=0.0)
    return 2 * ALPHA * np.exp(-arrival / T1B) * shifted


def tissue_signal(c, perfusion=GM.perfusion):
    """ΔM of the tissue from a delivery c: f M0b ∫ c(t') exp(-(t - t')/T1') dt' (image units)."""
    resid = np.exp(-tg / T1P)
    return SCALE * M0B * perfusion / 6000.0 * np.convolve(c, resid)[: len(tg)] * DTG


DELTA_A = GM.att - 0.3                                       # arterial arrival, 0.3 s before the tissue
cases = {"kinetic model": (0.0, 0.0), "dispersion 0.3 s": (0.3, 0.0), "arterial 1 %": (0.0, 0.01), "both": (0.3, 0.01)}
signals = {}
for name, (disp, abv) in cases.items():
    tissue = tissue_signal(concentration(GM.att, disp))
    arterial = SCALE * M0B * abv * concentration(DELTA_A, disp)
    signals[name] = (tissue, arterial)
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 3.3))
ax0.plot(tg, concentration(GM.att, 0.0) / (2 * ALPHA), color=INK["secondary"], label="box (kinetic model)")
ax0.plot(tg, concentration(GM.att, 0.3) / (2 * ALPHA), color=PALETTE[0], label="dispersed, kernel mean 0.3 s")
ax0.set(xlabel="time from the start of labeling (s)", ylabel="label concentration at the voxel / M₀ᵦ", title="the bolus as it arrives (ATT 0.8 s, LD 1.8 s)")
ax0.legend(fontsize=7)
ax1.plot(tg, signals["kinetic model"][0], color=INK["secondary"], label="tissue, box")
ax1.plot(tg, deltam(tg), "k:", lw=1, label="kinetic.delta_m (check)")
ax1.plot(tg, signals["both"][0], color=PALETTE[0], label="tissue, dispersed")
ax1.plot(tg, signals["both"][1], color=PALETTE[7], label="arterial, aBV 1 %, dispersed")
ax1.plot(tg, sum(signals["both"]), color=PALETTE[1], lw=2.5, alpha=0.7, label="total")
ax1.plot(T_SAMPLE, np.interp(T_SAMPLE, tg, sum(signals["both"])), "ko", ms=5, label="the six delays")
ax1.set(xlabel="time from the start of labeling (s)", ylabel="ΔM (image units)", title="gray matter with dispersion and an arterial compartment")
ax1.legend(fontsize=7)
for ax in (ax0, ax1):
    ax.set_xlim(0, 5.0)
fig.tight_layout()
print(f"box delivery vs kinetic.delta_m at the six delays: max |Δ| {np.abs(np.interp(T_SAMPLE, tg, signals['kinetic model'][0]) - deltam(T_SAMPLE)).max():.2f} image units")
print(f"peak tissue ΔM {signals['kinetic model'][0].max():.0f}; arterial peak at aBV 1 %: {signals['arterial 1 %'][1].max():.0f} image units")
```

Left: the box that the kinetic model assumes and its dispersed version, whose edges are
spread over a few tenths of a second and whose onset is the same transit time. Right: what
they do to the signal. Dispersion rounds the corners of the tissue curve (blue against
gray) and shifts its peak later. The arterial compartment (red) is a large signal at short
delays even at 1 % blood volume: a fraction of a percent of the voxel's water at the blood's
full label amplitude is comparable with the perfusion signal, which is a percent of the
tissue's water accumulated over the labeling duration, as the printed peaks show. The
arterial signal has passed by the delays the white paper recommends, which is one reason
for its long PLD, and it is exactly what the vascular crushing gradients of
[Chapter 4](../02-labeling/04-labeling-schemes.md) remove. The next cell fits the standard
model to the six samples of each case.

```{code-cell} python
:tags: [hide-input]
ATT_GRID = np.arange(0.2, 2.501, 0.01)


def fit_standard(samples):
    """Grid over ATT, CBF by least squares, one T1' refinement: the fit of quant.fit_multi_pld on one voxel."""
    best = None
    for a in ATT_GRID:
        g = deltam(T_SAMPLE, perfusion=1.0, att=a)
        f = samples @ g / (g @ g)
        r = np.sum((samples - f * g) ** 2)
        if best is None or r < best[0]:
            best = (r, f, a)
    _, f, a = best
    g = deltam(T_SAMPLE, perfusion=f, att=a)
    return f * (samples @ g) / (g @ g), a


for name, (tissue, arterial) in signals.items():
    cbf, att = fit_standard(np.interp(T_SAMPLE, tg, tissue + arterial))
    print(f"{name:17s}: fitted CBF {cbf:5.1f} ml/100 g/min ({cbf / GM.perfusion - 1:+.0%}), ATT {att:.2f} s ({att - GM.att:+.2f} s)")

abvs = np.linspace(0, 0.02, 21)
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 3.2))
for disp, color, label in ((0.0, INK["secondary"], "no dispersion"), (0.3, PALETTE[0], "dispersion 0.3 s")):
    c_t, c_a = concentration(GM.att, disp), concentration(DELTA_A, disp)
    fits = np.array([fit_standard(np.interp(T_SAMPLE, tg, tissue_signal(c_t) + SCALE * M0B * abv * c_a)) for abv in abvs])
    ax0.plot(100 * abvs, fits[:, 0], color=color, label=label)
    ax1.plot(100 * abvs, fits[:, 1], color=color, label=label)
ax0.axhline(GM.perfusion, color="k", lw=0.8, ls=":")
ax1.axhline(GM.att, color="k", lw=0.8, ls=":")
ax0.set(xlabel="arterial blood volume in the voxel (%)", ylabel="fitted CBF (ml/100 g/min)", title="standard multi-delay fit: CBF (truth 60)")
ax1.set(xlabel="arterial blood volume in the voxel (%)", ylabel="fitted ATT (s)", title="standard multi-delay fit: ATT (truth 0.8 s)")
ax0.legend(fontsize=7)
fig.tight_layout()
```

The printed fits are the bias of the standard model. Dispersion alone leaves CBF within
1 % of the truth and moves the fitted transit time later by 0.29 s, the kernel's mean: the
model absorbs a rounded bolus into a later arrival, and a transit-time map from dispersed
data is a map of mean arrival rather than onset. The arterial signal is the larger
problem: at 1 % blood volume the fit explains the bright early sample with an arrival at
0.43 s, close to the arterial arrival, and a CBF of 71 ml/100 g/min, 18 % too high; with
dispersion as well it is 78, 30 % too high. The figure sweeps the blood volume: the CBF
bias grows with it, faster when the bolus is also dispersed, and the fitted transit time
collapses to the arterial arrival once the blood volume passes half a percent. The
remedies are the ones the literature uses: crush the arterial signal, start sampling after
it has passed, or fit a model that contains it, so that $a\mathrm{BV}$ and
$\delta_\mathrm{a}$ become two more maps, well determined only where the short-delay
samples are many and clean. Time-encoded acquisitions, whose earliest delays are short,
need one of the three.

## Learned denoising and quantification

Neural networks trained on paired data now denoise ASL difference images, predict the CBF
map of a full-length scan from a few pairs, and, in some work, quantify CBF and transit
time directly from multi-delay or time-encoded series without an explicit kinetic model.
The gains in apparent image quality are large, and the caveats are the same as for every
learned method. The output is only as general as the training data, which for ASL is
scarce, single-site, and mostly healthy; the errors are plausible-looking rather than
noisy, which makes a hallucinated cortical ribbon harder to detect than the noise floor
of [Chapter 8](../03-preprocessing/08-noise.md); and a network trained on a kinetic model
inherits that model's omissions. A simulator with a known answer is the natural way to
test these methods, which is the closing topic.

## Simulation as validation

Every quantification chapter of this book ended with an estimate scored against the truth,
and that was possible only because the data were simulated. Real ASL has no ground truth:
positron emission tomography agrees with ASL within its own errors, and test-retest
reproducibility tests only agreement with oneself. A simulator that takes a BIDS protocol
and a phantom through the kinetic model, the signal equations, and a k-space acquisition,
and writes the answer it started from, lets a method be scored on the quantity it claims
to measure, under the artifacts it will meet, at the parameters of a specific protocol.
That is the loop the book has closed with aslscan and the ASLDRO phantom
{cite:p}`olivertaylor2021`: the toy tier for every curve, the pipeline tier for every image.

The limits of the simulator are the limits of the validation, and the chapters have said
where they apply. The table lists what aslscan does not model, which chapter of this book
would change if it did, and what adding it would take.

| Not modeled | Where it matters | What adding it takes |
|---|---|---|
| Macrovascular compartment | Chapters 15, 18, 19: short-delay samples, time-encoded fits | An arterial blood volume map and an arterial arrival map in the phantom; one more term in the signal stage, the arterial concentration times $2\alpha M_{0\mathrm{b}}\,a\mathrm{BV}$, with its own $T_2$; a `VascularCrushing` flag that removes it |
| Bolus dispersion | Chapters 15, 18: multi-delay fits | A kernel (gamma-variate with two parameters, possibly maps) convolved with the delivery in `kinetic.delta_m`, which then becomes a numerical integral as in this chapter |
| Water exchange | Chapter 16, this chapter: TE dependence, calibration | A two-compartment kinetic model with an exchange-time map, and the transverse decay applied per compartment at the echo time instead of blood $T_2$ for all of the label |
| Physiological noise | Chapters 8, 9, 10: why background suppression matters, tSNR | Multiplicative fluctuations of the static signal per volume (cardiac and respiratory components, a slow drift) and a pulsatile term on the arterial signal, scaled with the static signal so that suppression reduces them |
| 3D readouts | Chapters 6, 7, 9, 17: slice timing, suppression timing, SNR | A slab excitation, a $k_z$ echo train with its $T_2$ window per compartment, segmentation across excitations with motion between them; no slice offsets |
| Spatially selective suppression pulses | Chapter 9: the label's own suppression factor | Tracking where the label is at each pulse (still in the arteries below the imaging region, or in it) instead of the global-bolus factor $\prod(1 - 2\epsilon)$ |
| Velocity-selective and vessel-encoded labeling | This chapter | A labeling stage that acts on a velocity or on a position in the labeling plane rather than on a global efficiency; vessel encoding needs an arterial territory map in the phantom |

Each row is a direction in which the simulator can grow, and each extension makes one more
section of this book testable rather than descriptive. The phantom itself is the other
limit: one perfusion, one transit time, and one $T_1$ per tissue class make the answer key
exact and the tissue unrealistic, and the first extension a reader is likely to want is a
phantom with spatially varying maps, which every function in `aslbook` already accepts.

## What this implies for acquisition

- **Long or unknown transit times** are the case for velocity-selective labeling; its
  price is efficiency, not timing.
- **A territory question** needs vessel encoding, at no cost in the perfusion image.
- **A short echo time** keeps the exchange question out of a single-echo CBF measurement;
  at 30 ms the blood-$T_2$ calibration can be off by a tenth if most of the label has
  exchanged, and only a multi-echo acquisition can tell.
- **A segmented 3D readout with background suppression** is the recommended readout; know
  its echo train length, which sets the through-plane blur.
- **Short delays contain arterial signal.** Crush it, avoid it, or model it; do not fit the
  standard model to samples before the bolus has cleared the arteries.
- **Ask how a learned method was validated**, and prefer methods scored against a
  simulated truth under the artifacts of your protocol.

## Further reading

The reviews of advanced ASL methods {cite:p}`vanosch2018,hernandezgarcia2019`;
velocity-selective ASL {cite:p}`wong2006`; restricted water exchange and its $T_2$
signature {cite:p}`stlawrence2000,gregori2013`; 3D GRASE {cite:p}`gunther2005,fernandezseara2005`
and the comparison of readouts {cite:p}`vidorreta2013`; the macrovascular model
{cite:p}`chappell2010`; the multi-timepoint recommendations, which discuss dispersion and
arterial signal in time-encoded data {cite:p}`woods2024`; and the phantom this book's
simulations start from {cite:p}`olivertaylor2021`.
