---
title: "1. Spins, relaxation, and the longitudinal magnetization"
kernelspec:
  name: python3
  display_name: Python 3
---

:::{admonition} Simulated datasets in this chapter
:class: note
- **Built in this page:** Bloch-equation curves and animations, an isochromat spin-echo simulation, and the longitudinal signal equations evaluated with the tissue and blood constants of the simulated brain ([Appendix B](../appendices/b-data-manifest.md#app-b-package-data)).

No pipeline-tier dataset is used here; the first images appear in [Chapter 2](../01-mri-physics/02-epi-and-reconstruction.md).
:::

## Learning goals

After this chapter you can:

- state where the MR signal comes from and what T1, T2, and T2* describe
- write the longitudinal magnetization after a 90° pulse (saturation recovery) and after a
  180° pulse (inversion recovery), and explain why an inverted population's difference from
  equilibrium decays with T1, which is how an ASL label decays
- compute the steady-state signal of each tissue of the simulated brain at the reference
  repetition time of 4.5 s and at the M0 scan's 8 s
- explain, from a simulation, why a spin echo recovers the signal that a gradient echo loses
- state the practical consequences of the T1 of blood (1.65 s) for the timing of an ASL
  acquisition

```{code-cell} python
:tags: [hide-cell]
import numpy as np
import matplotlib.pyplot as plt

from aslbook import kinetic, presets
from aslbook.plotting import INK, PALETTE, TISSUE_COLORS, animate, set_style

set_style()
TISSUES = presets.TISSUES
BLOOD_COLOR = PALETTE[7]  # arterial blood, red, used only in this part of the book
COLORS = {**TISSUE_COLORS, "blood": BLOOD_COLOR}
```

## Magnetization and precession

Hydrogen nuclei (protons) carry a magnetic moment. In the scanner's static field $B_0$ they
do not simply line up with the field. Like a spinning top tilted in gravity, each moment
swings around the field direction, a motion called precession. The rate of that swing, the
Larmor frequency, is proportional to the field:

$$f_0 = \frac{\gamma}{2\pi} B_0, \qquad \frac{\gamma}{2\pi} = 42.58\ \mathrm{MHz/T}.$$

Thermal motion randomizes the moments almost completely; the net alignment along the field
is a few parts per million. That small net magnetization per unit volume, $M_0$, is what MRI
measures, and every ASL formula ends with a division by it
([Chapter 14](../04-quantification/14-cbf-quantification.md)).

```{code-cell} python
:tags: [hide-input]
print(f"Larmor frequency at {presets.B0_T:.0f} T: {presets.GAMMA_BAR_MHZ_PER_T * presets.B0_T:.1f} MHz")
```

The precession frequency follows the local field. Any deviation from $B_0$, whether from the
susceptibility of tissue and air or from a gradient applied on purpose, changes the frequency
in proportion. A gradient is used deliberately to encode position
([Chapter 2](../01-mri-physics/02-epi-and-reconstruction.md)); an unwanted offset near the
sinuses displaces the image ([Chapter 11](../03-preprocessing/11-susceptibility-distortion.md))
and, for pseudo-continuous labeling, reduces how well the blood is labeled
([Chapter 4](../02-labeling/04-labeling-schemes.md)).

## Excitation and relaxation

A radio-frequency (RF) pulse at the Larmor frequency tips the magnetization away from the
field direction by a chosen flip angle. After a 90° pulse the magnetization lies in the
transverse plane, precesses, and induces a voltage in the receive coil: that voltage is the MR
signal. After a 180° pulse it points against the field and gives no signal until a later
pulse tips it into the plane. ASL uses both: 90° pulses to read out images, and 180°
(inversion) pulses to label blood and to suppress the static tissue. The animation follows
the net magnetization of a voxel, drawn as one arrow, through a 90° pulse and the recovery
that follows, with the precession slowed enormously (the real arrow turns 128 million times
per second at 3 T) and T1 only three times T2 rather than the ten to twenty times of brain
tissue. The orange line is the arrow's shadow on the transverse plane, which is what the
receive coil detects.

```{code-cell} python
:tags: [hide-input]
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3-D projection)

N_REST, N_TIP, N_FREE = 6, 14, 70
PREC = 2 * np.pi / 10          # radians of precession per frame
T1_DISP, T2_DISP = 54.0, 18.0  # relaxation times in frames
path = [(0.0, 0.0, 1.0)] * N_REST
for i in range(1, N_TIP + 1):  # the RF pulse tips the vector while it precesses
    tip, phi = np.pi / 2 * i / N_TIP, PREC * i
    path.append((np.sin(tip) * np.cos(phi), np.sin(tip) * np.sin(phi), np.cos(tip)))
for i in range(1, N_FREE + 1):  # free precession: transverse part decays (T2), longitudinal part regrows (T1)
    phi, mxy, mz = PREC * (N_TIP + i), np.exp(-i / T2_DISP), 1 - np.exp(-i / T1_DISP)
    path.append((mxy * np.cos(phi), mxy * np.sin(phi), mz))
path = np.array(path)

fig = plt.figure(figsize=(8.4, 3.8))
ax3 = fig.add_subplot(1, 2, 1, projection="3d")
ax2 = fig.add_subplot(1, 2, 2)
ax3.set_axis_off()
ax3.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-0.05, 1.1))
ax3.view_init(elev=22, azim=-60)
ring = np.linspace(0, 2 * np.pi, 100)
ax3.plot(np.cos(ring), np.sin(ring), 0, color="0.85", lw=0.8)
ax3.plot([0, 0], [0, 0], [0, 1.1], color="0.6", lw=0.8, ls="--")
ax3.text(0, 0, 1.15, "B₀", color="0.4", fontsize=9)
vec, = ax3.plot([], [], [], color=PALETTE[0], lw=3)
shadow, = ax3.plot([], [], [], color=PALETTE[1], lw=2)
stage = fig.text(0.26, 0.93, "", ha="center", va="top", fontsize=9)
frames_t = np.arange(len(path))
l_xy, = ax2.plot([], [], color=PALETTE[1], label="transverse $M_{xy}$ (the signal)")
l_z, = ax2.plot([], [], color=PALETTE[0], label="longitudinal $M_z$ (along B₀)")
ax2.set(xlim=(0, len(path)), ylim=(0, 1.05), xticks=[], xlabel="time", ylabel="magnetization / M₀")
ax2.legend(loc="center right", fontsize=7)
fig.tight_layout(rect=(0, 0, 1, 0.88))

def frame(i):
    x, y, z = path[i]
    vec.set_data_3d([0, x], [0, y], [0, z])
    shadow.set_data_3d([0, x], [0, y], [0, 0])
    l_xy.set_data(frames_t[: i + 1], np.hypot(path[: i + 1, 0], path[: i + 1, 1]))
    l_z.set_data(frames_t[: i + 1], path[: i + 1, 2])
    stage.set_text("at rest: aligned with B₀" if i < N_REST else
                   "90° pulse: tipped into the transverse plane" if i < N_REST + N_TIP else
                   "precessing; transverse part decays (T2),\nlongitudinal part regrows (T1)")

animate(fig, frame, range(len(path)), fps=12, width=680, dpi=70,
        alt="an arrow representing the net magnetization points along the main field, is tipped into the transverse plane by an RF pulse while it precesses, then spirals back up: its transverse component shrinks with T2 while its longitudinal component regrows with T1; a plot alongside traces both components over time")
```

Two processes return the magnetization to equilibrium, and both are visible above:

- **T1 (longitudinal relaxation)** rebuilds the component along the field, $M_z$. It sets
  how much magnetization is available for the next excitation, and it is the clock on which
  an ASL label fades.
- **T2 (transverse relaxation)** is the loss of the transverse component, $M_{xy}$, as
  neighboring spins dephase one another. It sets how much signal remains at the echo time TE.

In practice the transverse signal decays faster than T2 alone predicts, because the field is
never perfectly uniform across a voxel and spins in slightly different fields drift out of
phase. This faster decay is **T2\***. The extra loss from the static field differences has its
own time constant, T2′, and the rates add: $1/T_2^* = 1/T_2 + 1/T_2'$. The part caused by the
static field is reversible, and the spin echo at the end of this chapter reverses it.

In symbols, the Bloch equations {cite:p}`bloch1946` describe the two relaxations in a frame
that rotates at the Larmor frequency, once the RF pulse is over:

$$\frac{dM_z}{dt} = \frac{M_0 - M_z}{T_1}, \qquad \frac{dM_{xy}}{dt} = -\frac{M_{xy}}{T_2}.$$

The transverse equation has the plain exponential solution $M_{xy}(t) = M_{xy}(0)\,e^{-t/T_2}$.
The longitudinal one is the subject of this chapter, because ASL is a story about $M_z$.

### The tissues of the simulated brain

The simulated brain is piecewise constant: one T1, T2, T2*, and $M_0$ per tissue class, with
the values of the ASLDRO 3 T phantom ([Chapter 0.2](../00-frontmatter/the-simulated-datasets.md)).
Arterial blood, which ASL labels, has a T1 of 1.65 s and a T2 of 165 ms, the values the ASL
white paper recommends at 3 T {cite:p}`alsop2015,lu2004`.

```{code-cell} python
:tags: [hide-input]
print(f"{'tissue':>6} {'T1 (s)':>7} {'T2 (ms)':>8} {'T2* (ms)':>9} {'T2′ (ms)':>9} {'M0 (a.u.)':>10}")
for t in TISSUES.values():
    print(f"{t.name:>6} {t.t1:7.2f} {t.t2 * 1e3:8.0f} {t.t2star * 1e3:9.0f} {t.t2prime * 1e3:9.0f} {t.m0:10.1f}")
print(f"{'blood':>6} {presets.T1_BLOOD:7.2f} {presets.T2_BLOOD * 1e3:8.0f} {'—':>9} {'—':>9} {'—':>10}")
```

The ordering matters more than the exact values. White matter has the shortest T1, gray
matter a longer one, blood longer still, and CSF the longest by far. T2 runs the other way
for tissue (white matter longer than gray) and is longest in CSF. The phantom assigns no T2*
to blood; the simulated readout is a spin echo, so only its T2 enters.

```{code-cell} python
:tags: [hide-input]
t = np.linspace(0, 6, 601)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.1))
for name, tis in TISSUES.items():
    ax1.plot(t, 1 - np.exp(-t / tis.t1), color=COLORS[name], label=f"{name} (T1 {tis.t1:.2f} s)")
    ax2.plot(t[t <= 0.25] * 1e3, np.exp(-t[t <= 0.25] / tis.t2), color=COLORS[name], label=f"{name} (T2 {tis.t2 * 1e3:.0f} ms)")
ax1.plot(t, 1 - np.exp(-t / presets.T1_BLOOD), color=BLOOD_COLOR, ls="--", label=f"blood (T1 {presets.T1_BLOOD:.2f} s)")
ax2.plot(t[t <= 0.25] * 1e3, np.exp(-t[t <= 0.25] / presets.T2_BLOOD), color=BLOOD_COLOR, ls="--", label=f"blood (T2 {presets.T2_BLOOD * 1e3:.0f} ms)")
te = presets.REFERENCE.echo_time * 1e3
ax2.axvline(te, color="0.6", lw=1, ls=":")
ax2.text(te + 3, 0.5, f"TE = {te:.0f} ms", fontsize=8, color="0.4")
ax1.set(xlabel="time after a 90° pulse (s)", ylabel="$M_z$ / $M_0$", title="T1 recovery", ylim=(0, 1.02))
ax2.set(xlabel="time after a 90° pulse (ms)", ylabel="$M_{xy}$ / $M_{xy}(0)$", title="T2 decay", ylim=(0, 1.02))
ax1.legend(loc="lower right")
ax2.legend(loc="upper right")
fig.tight_layout()
```

Left: after a 90° pulse, white matter is back near equilibrium in about 3 s, gray matter
takes longer, blood longer still, and CSF is only about two thirds recovered at 4 s. Right:
at the reference echo time of 12 ms every tissue keeps most of its transverse magnetization,
CSF the most and gray matter the least. The curves for blood and tissue differ, and
[Chapter 16](../04-quantification/16-calibration.md) accounts for that when the perfusion
signal, which comes from blood, is divided by a calibration image, which comes from tissue.

## The longitudinal axis: saturation and inversion

Solving the longitudinal Bloch equation from any starting value $M_z(0)$ gives

$$M_z(t) = M_0 - \bigl(M_0 - M_z(0)\bigr)\, e^{-t/T_1}.$$

Read it as: the *difference from equilibrium* decays exponentially with T1, whatever its
starting size or sign. Two starting values matter in ASL.

**Saturation recovery.** A 90° pulse leaves $M_z(0) = 0$, so
$M_z(t) = M_0\,(1 - e^{-t/T_1})$. If the next 90° pulse comes a repetition time TR later, the
magnetization it finds, and therefore the signal it produces, is $M_0(1 - e^{-\mathrm{TR}/T_1})$.
This is the static tissue signal of every ASL image.

**Inversion recovery.** A 180° pulse leaves $M_z(0) = -M_0$, so
$M_z(t) = M_0\,(1 - 2e^{-t/T_1})$. The magnetization passes through zero at $t = T_1 \ln 2$
and then recovers toward $+M_0$. The difference from equilibrium starts at $2M_0$ and decays
as $2M_0\,e^{-t/T_1}$.

An ASL label is an inverted population of arterial blood water; the control condition leaves
the same blood at equilibrium. The difference image therefore measures exactly the labeled
blood's difference from equilibrium, and the equation says how it fades: as $e^{-t/T_1}$,
first with the T1 of blood while the water is in the arteries, then with the T1 of the
tissue once it has exchanged into it ([Chapter 5](../02-labeling/05-kinetic-model.md)). The
label is worth $2M_0$ at the moment of inversion, not $M_0$; that factor of 2 appears in
every ASL quantification formula.

```{code-cell} python
:tags: [hide-input]
t = np.linspace(0, 5, 501)
pld_end = presets.REFERENCE.labeling_duration  # 1.8 s: the reference post-labeling delay
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.2))
for name, t1 in [("blood", presets.T1_BLOOD)] + [(n, tis.t1) for n, tis in TISSUES.items()]:
    ls = "--" if name == "blood" else "-"
    ax1.plot(t, 1 - 2 * np.exp(-t / t1), color=COLORS[name], ls=ls, label=name)
    ax2.plot(t, np.exp(-t / t1), color=COLORS[name], ls=ls, label=name)
    ax1.plot(t1 * np.log(2), 0, "o", color=COLORS[name], ms=4)
ax1.axhline(0, color="0.6", lw=0.8)
ax1.set(xlabel="time after a 180° pulse (s)", ylabel="$M_z$ / $M_0$", title="inversion recovery (dots: the null point $T_1 \\ln 2$)", ylim=(-1.02, 1.02))
ax2.axvline(pld_end, color="0.6", lw=1, ls=":")
ax2.text(pld_end + 0.08, 0.9, f"{pld_end:.1f} s", fontsize=8, color="0.4")
ax2.set(xlabel="time after a 180° pulse (s)", ylabel="$(M_0 - M_z)\\,/\\,2M_0 = e^{-t/T_1}$", title="the label: difference from equilibrium", ylim=(0, 1.02))
ax1.legend(loc="lower right")
ax2.legend(loc="upper right")
fig.tight_layout()
remaining = {"blood": np.exp(-pld_end / presets.T1_BLOOD), **{n: np.exp(-pld_end / tis.t1) for n, tis in TISSUES.items()}}
print(f"fraction of an inverted label remaining after {pld_end:.1f} s: " + ", ".join(f"{k} {v:.3f}" for k, v in remaining.items()))
print(f"null point T1 ln 2: blood {presets.T1_BLOOD * np.log(2):.2f} s, " + ", ".join(f"{n} {tis.t1 * np.log(2):.2f} s" for n, tis in TISSUES.items()))
```

Left: every curve starts at $-M_0$, crosses zero at its own null point, and approaches
$+M_0$. White matter nulls first (0.58 s), then gray matter (0.92 s), blood (1.14 s), and CSF
last (2.08 s); background suppression ([Chapter 9](../03-preprocessing/09-background-suppression.md))
times its inversion pulses so that the tissues are near their null points at readout.
Right: the same curves as the difference from equilibrium, the quantity the label carries.
At 1.8 s, the reference post-labeling delay, a label in blood retains 0.336 of its initial
value; it would retain only 0.258 with gray matter's T1 and 0.114 with white matter's. The
decay of the label is the single largest loss in ASL.

The animation makes the same point with two arrows: the same blood water under the control
condition (left, at equilibrium) and after inversion (right). The shaded gap between the two
curves is the label, and it closes with the T1 of blood.

```{code-cell} python
:tags: [hide-input]
T1B = presets.T1_BLOOD
N_FR, T_END = 48, 4.0
ts = np.linspace(0, T_END, N_FR)
mz_label = 1 - 2 * np.exp(-ts / T1B)

fig = plt.figure(figsize=(8.4, 3.4))
gs = fig.add_gridspec(1, 3, width_ratios=[0.5, 0.5, 1.6])
axc, axl, axp = (fig.add_subplot(gs[i]) for i in range(3))
for ax, title in [(axc, "control"), (axl, "label")]:
    ax.set(xlim=(-1, 1), ylim=(-1.15, 1.15), xticks=[], yticks=[], title=title)
    ax.axhline(0, color="0.8", lw=0.8)
    ax.grid(False)
    ax.set_axis_off()
axc.annotate("", xy=(0, 1), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=PALETTE[0], lw=3))
arrow_l = axl.annotate("", xy=(0, -1), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=BLOOD_COLOR, lw=3))
axp.plot(ts, np.ones_like(ts), color=PALETTE[0], label="control: $M_z = M_0$")
line_l, = axp.plot([], [], color=BLOOD_COLOR, label="label: $M_z = M_0(1 - 2e^{-t/T_1})$")
fill = [axp.fill_between([], [], [], color=BLOOD_COLOR, alpha=0.15)]
axp.axhline(0, color="0.8", lw=0.8)
axp.set(xlim=(0, T_END), ylim=(-1.05, 1.05), xlabel="time after inversion (s)", ylabel="$M_z$ / $M_0$",
        title=f"arterial blood, T1 = {T1B:.2f} s")
axp.legend(loc="lower right", fontsize=7)
txt = axp.text(0.03, 0.95, "", transform=axp.transAxes, fontsize=8, va="top", color=INK["secondary"])
fig.tight_layout()

def frame(i):
    arrow_l.xy = (0, mz_label[i])
    line_l.set_data(ts[: i + 1], mz_label[: i + 1])
    fill[0].remove()
    fill[0] = axp.fill_between(ts[: i + 1], mz_label[: i + 1], 1, color=BLOOD_COLOR, alpha=0.15)
    txt.set_text(f"t = {ts[i]:.1f} s: label = control − label = {1 - mz_label[i]:.2f} M₀")

animate(fig, frame, range(N_FR), fps=8, width=720, dpi=70,
        alt="two arrows: the control arrow stays pointing up at equilibrium while the label arrow starts pointing down after inversion, shrinks through zero, and grows back up; a plot alongside shades the gap between the two curves, which closes exponentially with the T1 of blood")
```

The gap starts at $2M_0$ and has closed to $0.67 M_0$ after 1.8 s and to $0.18 M_0$ after
4 s. The label is a wasting asset: every second spent waiting for it to reach the tissue
costs about 45 % of what is left.

## See it: the static signal at the repetition time

The reference protocol reads an image every 4.5 s, and the M0 calibration scan every 8 s
([Chapter 0.2](../00-frontmatter/the-simulated-datasets.md)). With a 90° pulse every TR,
$M_z$ is zero after each pulse and $M_0(1 - e^{-\mathrm{TR}/T_1})$ before the next, whatever
it was before the first pulse, so the steady state is reached at once (`kinetic.tissue_se`
evaluates it). The curves are that steady state for each tissue as a function of TR, scaled
by its $M_0$, in the units of the ground-truth M0 map.

```{code-cell} python
:tags: [hide-input]
r = presets.REFERENCE
tr = np.linspace(0.05, 12, 400)
fig, ax = plt.subplots(figsize=(7.5, 3.3))
for name, tis in TISSUES.items():
    ax.plot(tr, kinetic.tissue_se(tis.m0, tis.t1, tr), color=COLORS[name], label=name)
    ax.axhline(tis.m0, color=COLORS[name], lw=0.8, ls=":")
for x, label in [(r.repetition_time, f"ASL series: TR {r.repetition_time:g} s"), (r.m0_repetition_time, f"M0 scan: TR {r.m0_repetition_time:g} s")]:
    ax.axvline(x, color="0.5", lw=1, ls="--")
    ax.text(x + 0.12, 88, label, fontsize=8, color="0.4", va="top")
ax.set(xlabel="repetition time TR (s)", ylabel="$M_z$ before the pulse (M0 units)", ylim=(0, 90),
       title="steady-state signal vs TR (dotted: each tissue's $M_0$)")
ax.legend(loc="lower right")
fig.tight_layout()
print(f"{'tissue':>6} {'TR 4.5 s':>9} {'TR 8 s':>8}   (fraction of M0 recovered)")
for name, tis in TISSUES.items():
    print(f"{name:>6} {1 - np.exp(-r.repetition_time / tis.t1):9.3f} {1 - np.exp(-r.m0_repetition_time / tis.t1):8.3f}")
```

The curves rise toward each tissue's $M_0$ (dotted) at a rate set by its T1. At the
reference TR of 4.5 s white matter is fully recovered (0.996 of $M_0$) and gray matter nearly
so (0.966), but CSF has recovered only 0.777; at the M0 scan's TR of 8 s gray matter is at
0.998 and CSF at 0.931. So the control image is not an M0 image: its gray matter is 3.4 %
below equilibrium, which is why quantification divides by a separately calibrated $M_0$ and
why that calibration undoes the saturation of the scan it came from
([Chapter 16](../04-quantification/16-calibration.md)). A shorter TR would save time but
cost gray matter signal in proportion to $1 - e^{-\mathrm{TR}/T_1}$: 0.895 at TR 3 s
([Chapter 7](../02-labeling/07-acquisition-parameters.md)). The label itself does not see
the TR; it is fresh blood every time.

## Spin echo versus gradient echo

After the 90° pulse, spins in a voxel precess at slightly different frequencies because the
field is not perfectly uniform; their contributions drift out of phase and the summed signal,
the free induction decay (FID), falls off with T2*. A **gradient echo** switches a gradient
on in one direction and then the other, undoing only the dephasing it caused itself, so it
remains T2*-weighted. A **spin echo** {cite:p}`hahn1950` applies a 180° pulse at time TE/2,
which reverses the accumulated phase of every spin; each continues at its own rate, so at
time TE the phases realign and only the T2 loss remains.

The simulation follows several thousand spins whose frequency offsets are drawn from a
Lorentzian distribution, which gives the static dephasing an exponential envelope with time
constant T2′. Gray matter's own T2′ is long (377 ms), so to make the effect visible the
simulation uses the T2′ of 20 ms found near air-filled sinuses, with gray matter's T2 of 80 ms.

```{code-cell} python
:tags: [hide-input]
T2, T2P, TE_SIM = 80.0, 20.0, 40.0  # ms
T2STAR = 1 / (1 / T2 + 1 / T2P)
rng = np.random.default_rng(0)
omega = (1 / T2P) * np.tan(np.pi * (rng.uniform(size=6000) - 0.5))  # Lorentzian offsets, rad/ms: their mean phasor decays as exp(-t/T2')
t = np.linspace(0, 100, 2001)

def signal(t, te=None):
    """|sum of spin phasors| with T2 decay; a 180° pulse at te/2 negates every phase."""
    tau = t if te is None else np.where(t < te / 2, t, t - te)  # phase = omega * tau after the reversal
    return np.exp(-t / T2) * np.abs(np.exp(1j * np.outer(tau, omega)).mean(axis=1))

fid, echo = signal(t), signal(t, TE_SIM)
fig, ax = plt.subplots(figsize=(7.5, 3.2))
ax.plot(t, fid, label="FID (90° only): gradient echoes sample this")
ax.plot(t, echo, label=f"spin echo (90°, then 180° at {TE_SIM / 2:.0f} ms)")
ax.plot(t, np.exp(-t / T2), color="0.5", lw=1, ls="--", label=f"T2 decay ({T2:.0f} ms)")
ax.plot(t, np.exp(-t / T2STAR), color="0.5", lw=1, ls=":", label=f"T2* decay ({T2STAR:.0f} ms)")
for x, lab in [(TE_SIM / 2, "180°"), (TE_SIM, "TE")]:
    ax.axvline(x, color="0.8", lw=1)
    ax.text(x + 1, 0.95, lab, fontsize=8, color="0.4")
ax.set(xlabel="time after the 90° pulse (ms)", ylabel="signal (fraction of maximum)", ylim=(0, 1.02))
ax.legend()
fig.tight_layout()
i_te = np.argmin(np.abs(t - TE_SIM))
print(f"at TE = {TE_SIM:.0f} ms: spin echo {echo[i_te]:.3f} (T2 predicts {np.exp(-TE_SIM / T2):.3f}); FID {fid[i_te]:.3f} (T2* predicts {np.exp(-TE_SIM / T2STAR):.3f})")
te = presets.REFERENCE.echo_time
print(f"signal remaining at the reference TE of {te * 1e3:.0f} ms, spin echo exp(-TE/T2) vs gradient echo exp(-TE/T2*):")
for name, tis in TISSUES.items():
    print(f"  {name:>3}: spin echo {np.exp(-te / tis.t2):.3f}   gradient echo {np.exp(-te / tis.t2star):.3f}")
print(f"  blood: spin echo {np.exp(-te / presets.T2_BLOOD):.3f}")
```

The FID (blue) is gone within about 50 ms, following the T2* envelope. The 180° pulse at
20 ms makes the dephased spins refocus: the signal climbs back and peaks at 40 ms on the T2
curve, at 0.607, exactly the T2 prediction. After the echo the spins dephase again. The
printed table applies the two envelopes to the simulated tissues at the reference echo time
of 12 ms: a spin echo keeps 0.861 of gray matter's signal and 0.897 of white matter's; a
gradient echo, with the tissues' own long T2′, keeps 0.834 and 0.797. The difference is
modest at 12 ms in a well-shimmed field, but it grows near the sinuses and with longer TE.

The simulated readout is a spin-echo EPI (the sidecar's `AcqContrast` is `se`), so the images
in this book are T2-weighted at 12 ms. Many clinical 2D ASL readouts use gradient-echo EPI
and are T2*-weighted; 3D GRASE readouts are spin-echo based {cite:p}`alsop2015`. The choice
sets the transverse factor on the perfusion signal and the calibration image
([Chapter 16](../04-quantification/16-calibration.md)) and the signal loss near air–tissue
interfaces.

## Measure it: the blood–tissue T1 comparison

Where the label is when it decays matters, because the T1 of blood is longer than the T1 of
either tissue. The figure follows a label that leaves the labeling plane at $t = 0$ and
arrives in the tissue after the arterial transit time (0.8 s in gray matter, 1.2 s in white
matter): it decays with the T1 of blood on the way and with the tissue's T1 after exchange.
The reference curves keep the label in blood, or in tissue, throughout.

```{code-cell} python
:tags: [hide-input]
t = np.linspace(0, 4, 401)
T1B, PLD = presets.T1_BLOOD, presets.REFERENCE.post_labeling_delay
fig, ax = plt.subplots(figsize=(7.5, 3.3))
ax.plot(t, np.exp(-t / T1B), color=BLOOD_COLOR, ls="--", label=f"stays in blood (T1 {T1B:.2f} s)")
paths = {}
for name in ("GM", "WM"):
    tis = TISSUES[name]
    two_stage = np.where(t < tis.att, np.exp(-t / T1B), np.exp(-tis.att / T1B) * np.exp(-(t - tis.att) / tis.t1))
    paths[name] = two_stage
    ax.plot(t, two_stage, color=COLORS[name], label=f"blood until ATT {tis.att:.1f} s, then {name} (T1 {tis.t1:.2f} s)")
    ax.plot(t, np.exp(-t / tis.t1), color=COLORS[name], lw=1, ls=":", label=f"{name} throughout")
    ax.plot(tis.att, np.exp(-tis.att / T1B), "o", color=COLORS[name], ms=4)
ax.axvline(PLD, color="0.6", lw=1, ls=":")
ax.text(PLD + 0.05, 0.92, f"PLD {PLD:.1f} s", fontsize=8, color="0.4")
ax.set(xlabel="time since the label was created (s)", ylabel="label remaining, $e^{-\\int dt/T_1}$", ylim=(0, 1.02),
       title="the label decays with the T1 of wherever it is (dots: arrival in the tissue)")
ax.legend(loc="center right", fontsize=7)
fig.tight_layout()
i = np.argmin(np.abs(t - PLD))
print(f"label remaining at {PLD:.1f} s: in blood throughout {np.exp(-PLD / T1B):.3f}; "
      + "; ".join(f"blood then {n} {paths[n][i]:.3f} (in {n} throughout {np.exp(-PLD / TISSUES[n].t1):.3f})" for n in ("GM", "WM")))
```

At the 1.8 s delay, a label that stayed in blood keeps 0.336; one that entered gray matter at
0.8 s keeps 0.290 and one that entered white matter at 1.2 s keeps 0.235. The single-delay
formula of [Chapter 14](../04-quantification/14-cbf-quantification.md) assumes the first
case; the general kinetic model of [Chapter 5](../02-labeling/05-kinetic-model.md) accounts
for the second through an apparent tissue relaxation time T1′. The gap between the curves,
about 14 % for gray matter here, is the size of that approximation. It grows when the transit
time is short relative to the delay and vanishes when the tissue T1 equals the blood T1,
nearly the case for gray matter at 3 T, which is why the white paper's formula works as well
as it does.

## What this implies for acquisition

- **The label fades with T1, and blood's T1 is 1.65 s at 3 T.** Every 1.8 s costs two thirds
  of the label. Delays cannot be shortened at will, because the label must first arrive; the
  balance between arrival and decay is the central trade of
  [Chapter 7](../02-labeling/07-acquisition-parameters.md).
- **Field strength lengthens T1.** At 1.5 T the blood T1 is about 1.35 s and the label fades
  faster; at 3 T, and more so at 7 T, more label survives the delay.
- **TR sets the static signal, not the label.** At TR 4.5 s gray matter is 3.4 % short of
  equilibrium; the M0 scan uses 8 s to be nearly saturation-free. Any scan that serves as
  $M_0$ must have its TR recorded and corrected for ([Chapter 16](../04-quantification/16-calibration.md)).
- **Inversion is the tool of the trade.** The 180° pulse that creates a label, applied to the
  tissue at the right times, nulls it: background suppression is inversion recovery timed so
  that $1 - 2e^{-t/T_1}$ is near zero at readout ([Chapter 9](../03-preprocessing/09-background-suppression.md)).
- **A short TE keeps the signal; the readout type sets the envelope.** At 12 ms the
  transverse loss is 10–15 % and differs between blood and tissue, which the calibration
  must reconcile.

## Further reading

The original descriptions of nuclear induction {cite:p}`bloch1946` and the spin echo
{cite:p}`hahn1950`; textbook treatments in {cite:t}`haacke1999` and {cite:t}`nishimura2010`;
the T1 of blood at 3 T {cite:p}`lu2004` and the constants recommended for ASL
{cite:p}`alsop2015`.
