---
title: "3. Perfusion, and blood water as a tracer"
kernelspec:
  name: python3
  display_name: Python 3
---

:::{admonition} Simulated datasets in this chapter
:class: note
- **Built in this page:** a one-compartment tracer model integrated step by step, the label's T1 decay, and the packaged phantom slab's tissue fractions, perfusion and transit-time maps ([Appendix B](../appendices/b-data-manifest.md#app-b-package-data)).

This chapter uses no pipeline-tier dataset. The phantom it draws is the one every aslscan run in the book is made from ([Chapter 0.2](../00-frontmatter/the-simulated-datasets.md)).
:::

## Learning goals

After this chapter you can:

- define cerebral blood flow, give its units, and place gray and white matter on its scale
- write the one-compartment equation for a freely diffusible tracer and say what $\lambda$ converts
- explain what arterial spin labeling delivers to a voxel and why it disappears with the T1 of blood
- say what the arterial transit time is and why it differs between tissues and people
- estimate the size of the ASL signal from first principles and explain why it is noise-limited

```{code-cell} python
:tags: [hide-cell]
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle

from aslbook import kinetic, phantom, presets
from aslbook.plotting import INK, PALETTE, TISSUE_COLORS, animate, set_style, show_slice

set_style()
GM, WM = presets.TISSUES["GM"], presets.TISSUES["WM"]
T1B, LAM, R = presets.T1_BLOOD, presets.LAMBDA, presets.REFERENCE
K = phantom.DISPLAY_SLICE
```

## What perfusion is

**Perfusion** is the delivery of blood to the capillary bed of a tissue: the volume of
arterial blood passing through the capillaries of a given mass of tissue per unit time,
where oxygen, glucose and water are exchanged with the cells. In the brain it is called
**cerebral blood flow (CBF)**, written $f$, in milliliters of blood per 100 grams of tissue per minute (ml/100 g/min).

Perfusion is not blood flowing *through* a region. A large artery crossing a voxel carries
a great deal of blood that is on its way elsewhere and perfuses nothing where it is seen. A
perfusion measurement should count the blood that arrives in the voxel's capillaries; blood
still in arteries at the time of the measurement is a contaminant, the *macrovascular*
signal of [Chapter 5](./05-kinetic-model.md). The simulated brain has no arteries at all,
only capillary delivery.

Kety and Schmidt measured a whole-brain average of about 54 ml/100 g/min in young adults by
following an inhaled inert gas {cite:p}`kety1948`, and the number has held up. It hides a
large difference between tissues: gray matter, with its dense synapses and high metabolic
rate, is perfused at roughly 40 to 100 ml/100 g/min depending on the person, the region
and the method; white matter at about 20. The simulated brain makes these numbers exact:
$f = 60$ in every gray matter voxel, $f = 20$ in every white matter voxel, and 0 in CSF.

In the formulas below $f$ is needed per second and per gram, so ml/100 g/min is divided by
$100 \times 60 = 6000$: a gray matter flow of 60 ml/100 g/min is $0.01$ ml of blood per gram
of tissue per second. Every second, one percent of the tissue's own volume in blood
arrives, and that one percent is the origin of every number in this chapter.

## The tracer view

A tracer is a substance carried by the blood whose amount in the tissue can be measured
over time. Kety's insight was that for a tracer that diffuses freely across the capillary
wall, so that the tissue and the blood leaving it are always in equilibrium, the amount in
the tissue obeys a single balance equation {cite:p}`kety1951`. With $C_a(t)$ the tracer
concentration in arterial blood (per ml) and $C_t(t)$ in the tissue (per gram), tracer
arrives at rate $f\,C_a(t)$ and leaves with the venous blood at $C_t / \lambda$:

$$
\frac{dC_t}{dt} = f\,C_a(t) - \frac{f}{\lambda}\,C_t(t).
$$

The **blood–brain partition coefficient** $\lambda$ is the ratio of the tracer's
concentration in tissue to its concentration in blood at equilibrium, in ml of blood per
gram of tissue: it converts between content per gram of tissue and content per ml of
blood. For water, the tracer in ASL, $\lambda = 0.9$ ml/g {cite:p}`herscovitch1985`: a
gram of brain holds as much water as 0.9 ml of blood. The same $\lambda$ reappears in
[Chapter 14](../04-quantification/14-cbf-quantification.md) to convert the equilibrium
magnetization of tissue into that of blood.

Two time scales fall out of the equation. Delivery fills the tissue at a rate $f$ times
the arterial concentration; clearance empties it with time constant $\lambda / f$, which for
gray matter is $0.9 / 0.01 = 90$ s. ASL boluses are a second or two long, so clearance hardly
acts before the measurement: the tissue simply accumulates $f$ times the bolus duration.

## Arterial water as an endogenous tracer

Every tracer method needs something to inject or inhale, except one. Detre, Williams and
colleagues realized that the water already in arterial blood can be made into a tracer by
changing its magnetization {cite:p}`detre1992,williams1992`. Invert the longitudinal
magnetization of the water in the blood flowing toward the brain
([Chapter 1](../01-mri-physics/01-spins-and-relaxation.md)), and the blood carries a
*deficit* of magnetization into the tissue; water crosses the capillary wall almost freely,
so the inverted water mixes with the tissue water and lowers the voxel's magnetization. A
second image without the inversion (the *control*) has the full magnetization, and the
difference $\Delta M$ is proportional to the labeled water that arrived, hence to $f$.

In the tracer equation the "concentration" is now the deficit of magnetization relative to
equilibrium. If the labeling inverts the arterial water with **efficiency** $\alpha$ (1 for
a perfect inversion; [Chapter 4](./04-labeling-schemes.md)), the arterial deficit at the
moment of labeling is $2\alpha M_{0b}$, where $M_{0b}$ is the equilibrium magnetization of
blood water per ml of blood, the tissue's $M_0$ divided by $\lambda$; the factor 2 is
inversion, from $+M_0$ to $-M_0$.

This tracer has a property that nitrous oxide does not: it disappears on its own. An
inverted magnetization relaxes toward equilibrium with the time constant T1, and the
difference between labeled and control blood shrinks as $e^{-t/T_1}$. Arterial blood at 3 T
has $T_{1b} = 1.65$ s {cite:p}`lu2004`, so the label has a half-life of
$T_{1b} \ln 2 = 1.14$ s, as if the tracer were radioactive: the measurement must be made
within a few seconds of labeling, and the amount that survives depends on how long the
blood took to arrive. Once in the tissue, the water relaxes with the tissue's shorter T1
instead. The equation for the labeled water is therefore the Kety equation with a decay
term, the general kinetic model of [Chapter 5](./05-kinetic-model.md) {cite:p}`buxton1998`:

$$
\frac{d\,\Delta M}{dt} = f\,\Delta M_a(t) - \frac{f}{\lambda}\,\Delta M - \frac{\Delta M}{T_1}.
$$

```{code-cell} python
:tags: [hide-input]
print(f"label half-life in blood: T1b ln 2 = {T1B * np.log(2):.2f} s")
for x, what in ((GM.att, "GM transit time"), (WM.att, "WM transit time"), (R.labeling_duration + R.post_labeling_delay, "the reference readout")):
    print(f"t = {x:.1f} s ({what}): {100 * np.exp(-x / T1B):5.1f} % of the label survives in blood")
```

When the first labeled blood reaches gray matter, 0.8 s after inversion, 62 % of the label
is left; at the white matter arrival time, 1.2 s, 48 %; label made at the very start of the
reference protocol's 1.8 s labeling period has 11 % left when the image is read 3.6 s
later. Blood water is a tracer with an expiry, and protocol design ([Chapter 7](./07-acquisition-parameters.md)) is a negotiation with it.

## Arterial transit time

Labeling happens in the neck or at the base of the brain, several centimeters below the
tissue being imaged. Blood covers that distance at 20–40 cm/s in the carotid arteries and
slows as the arteries branch and narrow, so the first labeled water reaches a voxel's
capillaries only after a delay: the **arterial transit time (ATT)**, $\delta$ in the
kinetic model, typically 0.5–1.5 s in healthy adult gray matter and 0.5–2 s across brains
and regions. It is longer

- in **white matter**, fed by long penetrating arterioles at a third of the gray matter flow;
- in **watershed regions** at the borders between the territories of the major arteries;
- in the **elderly**, whose flow is slower and whose vessels are longer and more tortuous;
- in **vascular disease**: a stenosed carotid or a collateral route can delay arrival by a
  second or more, beyond what a standard protocol was designed for.

It is shorter in children and under hypercapnia. The phantom uses 0.8 s for gray matter
and 1.2 s for white matter. Because the label decays while it travels, a longer ATT means
less label for the same $f$, and imaging before the label has arrived measures none
([Chapter 5](./05-kinetic-model.md)); [Chapter 15](../04-quantification/15-multi-delay.md) shows how to measure the ATT itself.

## See it: the phantom's perfusion and transit-time maps

The simulated brain is piecewise constant: each voxel of the acquisition grid holds a
fraction of gray matter, white matter and CSF, and every property of the voxel is the
fraction-weighted mean of the tissue constants. The fractions on the display slice:

```{code-cell} python
:tags: [hide-input]
fr = phantom.fractions()
fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.3))
for ax, name in zip(axes, ("GM", "WM", "CSF")):
    im = show_slice(ax, fr[name], K, f"{name} fraction", kind="fraction")
fig.colorbar(im, ax=axes.tolist(), shrink=0.75)
mask = phantom.slab()["mask"]
mixed = (mask & (fr["GM"] > 0.1) & (fr["WM"] > 0.1)).sum()
print(f"voxels in the brain mask: {mask.sum()}; at least 90 % one tissue: " + ", ".join(f"{n} {(fr[n] > 0.9).sum()}" for n in fr))
print(f"voxels holding more than 10 % of both GM and WM: {mixed} ({100 * mixed / mask.sum():.0f} % of the mask)")
```

Anterior is at the top. The gray matter fraction traces the cortical ribbon and the deep
nuclei; the white matter fraction fills the centrum semiovale; CSF is brightest in the
ventricles and sulci. At 3.5 × 3.5 × 5 mm many voxels are mixtures: of the 26 046 voxels in
the brain mask, 9038 are at least 90 % gray matter and 4782 at least 90 % white matter, and
a third hold more than a tenth of each. Mixed voxels are the subject of
[Chapter 12](../03-preprocessing/12-partial-volume.md); this chapter's numbers refer to the pure ones.

```{code-cell} python
:tags: [hide-input]
maps = phantom.maps()
in_brain = lambda a: np.where(mask, a, np.nan)
fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), layout="constrained")
im0 = show_slice(axes[0], in_brain(maps["perfusion"]), K, "CBF (ml/100 g/min)", kind="cbf")
fig.colorbar(im0, ax=axes[0], shrink=0.8)
im1 = show_slice(axes[1], np.where(maps["att"] > 0, maps["att"], np.nan), K, "arterial transit time (s)", kind="att")
fig.colorbar(im1, ax=axes[1], shrink=0.8)
```

The CBF map, on the book's fixed 0–90 ml/100 g/min window, is gray matter at 60, white
matter at 20 and CSF at 0 (the ventricles are black), blurred only by the tissue mixing of
each voxel. The transit-time map, on the fixed 0–2 s window, is its complement: 0.8 s where
gray matter dominates and 1.2 s in the white matter (CSF has no transit time and is left
blank, as is everything outside the brain). Every ground-truth map the pipeline writes
([Appendix E](../appendices/e-truth-map-catalogue.md)) is such a fraction-weighted image.

## See it: labeled water arriving in a voxel

The animation integrates the labeled-water equation for a gray matter voxel under the
reference protocol: a bolus of inverted arterial water 1.8 s long that starts arriving
0.8 s after labeling begins, with $f = 60$ ml/100 g/min, $\lambda = 0.9$, $\alpha = 0.85$,
$T_{1b} = 1.65$ s and a tissue T1 of 1.33 s. On the left, the artery's color shows the
label it carries and the voxel's color the label it has accumulated; on the right are the
two curves, in units of the tissue's $M_0$. The blood was inverted to $2 \alpha / \lambda = 189$ % of $M_0$
at the labeling plane, and every ml that arrives has spent the same 0.8 s in transit, so
during the bolus it arrives at a constant $189\,\% \times e^{-0.8/1.65} = 116$ %.

```{code-cell} python
:tags: [hide-input]
DT = 0.001
t = np.arange(0, 6.0 + DT / 2, DT)

def arterial_label(t, att, tau=R.labeling_duration, alpha=presets.ALPHA["PCASL"]):
    """Deficit of arriving arterial water in tissue-M0 units; every arriving ml was labeled ATT earlier."""
    return np.where((t > att) & (t <= att + tau), 2 * alpha / LAM * np.exp(-att / T1B), 0.0)

def integrate(inflow, f_per_s, t1_tissue=None):
    """dM/dt = f inflow(t) - (f / lambda) M - M / T1 by forward Euler; t1_tissue None means no decay."""
    m, rate = np.zeros_like(t), f_per_s / LAM + (0.0 if t1_tissue is None else 1.0 / t1_tissue)
    for i in range(1, len(t)):
        m[i] = m[i - 1] + DT * (f_per_s * inflow[i - 1] - rate * m[i - 1])
    return m

curves = {}
for name, tis in (("GM", GM), ("WM", WM)):
    ma = arterial_label(t, tis.att)
    curves[name] = (ma, integrate(ma, tis.perfusion / 6000, tis.t1))
    closed = kinetic.delta_m(t, tis.perfusion, tis.att, tis.t1, 1.0)
    print(f"{name}: peak dM/M0 {100 * closed.max():.2f} % at t = {t[closed.argmax()]:.2f} s; "
          f"step-by-step integration vs the closed-form model: max difference {100 * np.abs(curves[name][1] - closed).max():.4f} % of M0")
```

```{code-cell} python
:tags: [hide-input]
ma, mt = curves["GM"]
blend = lambda color, a: tuple(1 - a * (1 - c) for c in to_rgb(color))
fig = plt.figure(figsize=(9, 3.6))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.3], hspace=0.55)
axs = fig.add_subplot(gs[:, 0]); axa = fig.add_subplot(gs[0, 1]); axt = fig.add_subplot(gs[1, 1])
axs.set(xlim=(0, 11.5), ylim=(-0.8, 6), aspect="equal"); axs.set_axis_off()
artery = Rectangle((0.2, 2.2), 2.0, 1.6, color="white", ec=INK["secondary"]); axs.add_patch(artery)
tissue = Rectangle((4.0, 0.8), 4.4, 4.4, color="white", ec=INK["secondary"]); axs.add_patch(tissue)
for xy, xytext, color, ls in [((4.0, 3.0), (2.2, 3.0), INK["primary"], "-"), ((10.2, 3.0), (8.4, 3.0), INK["primary"], "-"), ((6.2, -0.5), (6.2, 0.8), "0.5", "--")]:
    axs.annotate("", xy=xy, xytext=xytext, arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5, ls=ls))
for x, y, s, kw in [(1.2, 4.1, "artery", {}), (6.2, 5.5, "tissue voxel", {}), (3.1, 3.4, "f·ΔMₐ", {}), (9.3, 3.4, "f·ΔM/λ", {}),
                    (10.4, 3.0, "vein", dict(ha="left", va="center")), (6.5, 0.0, "T1 decay", dict(ha="left", color="0.4"))]:
    axs.text(x, y, s, fontsize=8, **{"ha": "center", **kw})
level = axs.text(6.2, 3.0, "", ha="center", va="center", fontsize=9)
la, = axa.plot([], [], color=PALETTE[7]); lt, = axt.plot([], [], color=TISSUE_COLORS["GM"])
ca = axa.axvline(0, color="0.6", lw=1); ct = axt.axvline(0, color="0.6", lw=1)
axa.set(xlim=(0, 6), ylim=(0, 200), ylabel="arterial ΔMₐ (% of M0)", title="arriving label", xticklabels=[])
axt.set(xlim=(0, 6), ylim=(0, 1.35), ylabel="tissue ΔM (% of M0)", xlabel="time from the start of labeling (s)")
axt.axvline(R.labeling_duration + R.post_labeling_delay, color="0.8", lw=1, ls="--")
axt.text(3.65, 1.15, "readout", fontsize=7, color=INK["secondary"])
frames = np.arange(0, len(t), 150)

def frame(i):
    artery.set_facecolor(blend(PALETTE[7], ma[i] / ma.max()))
    tissue.set_facecolor(blend(TISSUE_COLORS["GM"], mt[i] / mt.max()))
    level.set_text(f"ΔM\n{100 * mt[i]:.2f} % of M0")
    la.set_data(t[: i + 1], 100 * ma[: i + 1]); lt.set_data(t[: i + 1], 100 * mt[: i + 1])
    ca.set_xdata([t[i], t[i]]); ct.set_xdata([t[i], t[i]])
    axs.set_title(f"t = {t[i]:.2f} s", fontsize=9)

animate(fig, frame, frames, fps=6, width=720, dpi=72,
        alt="a schematic artery feeding a tissue voxel: the artery colors red while the labeled bolus passes between 0.8 and 2.6 s, the voxel colors orange as it accumulates label, and two curves grow with time, the arterial deficit as a flat step and the tissue deficit as a rise to just over one percent followed by a slow decay")
```

Nothing happens for the first 0.8 s: the label is in transit. Then the arterial curve steps
up to a constant level and the tissue curve rises, at first almost linearly, since the
arriving flow and its label are constant and the clearance term is negligible. The tissue's
own T1 decay bends the curve over, so it peaks at 1.14 % of $M_0$ just as the last of the
bolus arrives at 2.6 s, then decays with the tissue's T1. The reference protocol reads out
at 3.6 s, one second past the peak, where 0.53 % is left. The printed check shows that the
step-by-step integration reproduces the closed-form model aslscan uses (`kinetic.delta_m`)
to about a thousandth of a percent of $M_0$: the animation and the simulator are the same
physics. The next figure separates the two ingredients by comparing the label with an
imaginary tracer that does not decay, for gray and white matter.

```{code-cell} python
:tags: [hide-input]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharex=True)
for name, tis in (("GM", GM), ("WM", WM)):
    f = tis.perfusion / 6000
    unit_bolus = ((t > tis.att) & (t <= tis.att + R.labeling_duration)).astype(float)
    ideal = 100 * integrate(unit_bolus, f)
    axes[0].plot(t, ideal, color=TISSUE_COLORS[name], label=f"{name}: f {tis.perfusion:.0f}, ATT {tis.att} s")
    axes[0].axhline(100 * f * R.labeling_duration, color=TISSUE_COLORS[name], lw=0.8, ls=":")
    axes[1].plot(t, 100 * curves[name][1], color=TISSUE_COLORS[name], label=f"{name}: T1 {tis.t1} s")
    print(f"{name}: ideal tracer plateau {ideal.max():.2f} % (f x bolus = {100 * f * R.labeling_duration:.2f} %); labeled water peak {100 * curves[name][1].max():.2f} % of M0")
axes[0].set(title="a tracer that does not decay", xlabel="time (s)", ylabel="tissue content (% of arterial)")
axes[1].set(title="labeled water: arrives decayed, then keeps decaying", xlabel="time (s)", ylabel="ΔM (% of M0)")
axes[0].legend(loc="center right"); axes[1].legend(loc="center right")
fig.tight_layout()
```

On the left, a tracer at unit arterial concentration, delivered for 1.8 s, accumulates to
just under $f \times 1.8$ s in the tissue (dotted lines: 1.80 % in gray matter, 0.60 % in
white; the curves reach 1.78 % and 0.60 %) and stays there, because clearance over these
few seconds is negligible. On the right is the label: it arrives already reduced by its
journey and then relaxes with the tissue's T1. White matter is hit twice, by the longer
transit and the shorter T1: its curve peaks at 0.22 % against gray matter's 1.14 %, a
fivefold ratio for a threefold difference in flow.

## The size of the effect

The pieces are in hand for an estimate of $\Delta M / M_0$ that fits on one line. The
arterial deficit is $2\alpha M_0 / \lambda$. Over a bolus of duration $\tau$ (the labeling
duration, LD) a gram of tissue receives $f$ ml of blood per second, but blood labeled earlier
has decayed more, so the delivered label integrates to $f\,T_{1b}\,(1 - e^{-\tau/T_{1b}})$
rather than $f\tau$. If the whole bolus then decays in blood for a post-labeling delay
(PLD) of $w$ before the readout:

$$
\frac{\Delta M}{M_0} \approx \frac{2\alpha f T_{1b}}{\lambda}\,e^{-w/T_{1b}}\,\left(1 - e^{-\tau/T_{1b}}\right).
$$

This is the ASL white paper's formula {cite:p}`alsop2015` read backwards, the one
[Chapter 14](../04-quantification/14-cbf-quantification.md) inverts to get CBF from an
image. For the reference protocol:

```{code-cell} python
:tags: [hide-input]
tau, w, alpha = R.labeling_duration, R.post_labeling_delay, presets.ALPHA["PCASL"]
steps = [("arterial deficit\n2α/λ", 2 * alpha / LAM),
         ("× blood delivered\nf·T1b·(1 − e^(−LD/T1b))", (GM.perfusion / 6000) * T1B * (1 - np.exp(-tau / T1B))),
         ("× decay over PLD\ne^(−PLD/T1b)", np.exp(-w / T1B))]
running = np.cumprod([s[1] for s in steps])
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.bar(range(3), 100 * running, color=[PALETTE[7], TISSUE_COLORS["GM"], "0.6"], width=0.55)
for i, v in enumerate(running):
    ax.text(i, 100 * v * 1.25, f"{100 * v:.0f} %" if v > 0.1 else f"{100 * v:.2f} %", ha="center", fontsize=9)
ax.set(yscale="log", ylim=(0.1, 600), xticks=range(3), xticklabels=[s[0] for s in steps], ylabel="ΔM / M0 (%, log scale)",
       title="gray matter, reference protocol: LD 1.8 s, PLD 1.8 s, α 0.85")
ax.grid(axis="x", visible=False)
fig.tight_layout()
crude = {n: 2 * alpha * (tis.perfusion / 6000) * T1B / LAM * np.exp(-w / T1B) * (1 - np.exp(-tau / T1B)) for n, tis in (("GM", GM), ("WM", WM))}
for name, tis in (("GM", GM), ("WM", WM)):
    print(f"{name}: f = {tis.perfusion:.0f} ml/100 g/min -> dM/M0 = {100 * crude[name]:.3f} %")
```

Read the bars left to right, on a logarithmic scale. Inverting the arterial water makes a
deficit of 189 % of the tissue's $M_0$ per ml of blood. In 1.8 s a gram of gray matter
receives 1.1 % of its volume in blood after allowing for decay during the bolus, so the
delivered label is 2.1 % of $M_0$. Waiting 1.8 s more for the label to reach and fill the
tissue costs a factor of three: 0.70 %. Across the range of gray matter flows the estimate
runs from about 0.5 % to 1 %; in white matter it is 0.23 %. The consequence is that ASL is
limited by noise. The reference protocol's gray matter control image has a signal-to-noise
ratio of about 155 ([Chapter 6](./06-the-asl-signal.md)); the difference of two such
images has $\sqrt{2}$ times the noise of one and a signal of 0.7 % of the control:

```{code-cell} python
:tags: [hide-input]
snr_pair = presets.REFERENCE_GM_SNR * crude["GM"] / np.sqrt(2)
print(f"one control-label pair: SNR of dM in a GM voxel = 155 x {100 * crude['GM']:.2f} % / sqrt(2) = {snr_pair:.2f}")
print(f"{R.n_pairs} pairs averaged: {snr_pair * np.sqrt(R.n_pairs):.1f};  pairs needed for SNR 10 per voxel: {int(np.ceil((10 / snr_pair) ** 2))}")
```

A single pair gives a difference image whose gray matter voxels have an SNR below one
(0.76): the perfusion signal is smaller than the noise. Thirty pairs, the reference
protocol's four and a half minutes, bring a gray matter voxel to about 4; an SNR of 10 per
voxel would need about 170 pairs, half an hour of scanning. This is why ASL protocols
average dozens of pairs, why voxels are large, and why anything that adds variance to the
control image ([Chapter 9](../03-preprocessing/09-background-suppression.md),
[Chapter 10](../03-preprocessing/10-motion.md)) matters so much.

## See it: the difference next to the image it hides in

The kinetic model evaluated on the phantom's maps gives the noise-free difference image of
the reference protocol. The figure puts it next to the static tissue signal the control image
is made of (the spin-echo steady state at TR 4.5 s of [Chapter 1](../01-mri-physics/01-spins-and-relaxation.md)), both as a percentage of $M_0$.

```{code-cell} python
:tags: [hide-input]
t_ref = R.labeling_duration + R.post_labeling_delay
static = sum(fr[n] * kinetic.tissue_se(tis.m0, tis.t1, R.repetition_time) for n, tis in presets.TISSUES.items())
dm = sum(fr[n] * kinetic.delta_m(t_ref, tis.perfusion, tis.att, tis.t1, tis.m0) for n, tis in presets.TISSUES.items())
pct = lambda a: 100 * np.divide(a, maps["m0"], out=np.zeros_like(a), where=maps["m0"] > 0)  # zero outside the brain
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), layout="constrained")
show_slice(axes[0], pct(static), K, "control signal (% of M0)", vmin=0, vmax=100)
show_slice(axes[1], pct(dm), K, "ΔM on the same window", vmin=0, vmax=100)
im = show_slice(axes[2], in_brain(pct(dm)), K, "ΔM on its own window (% of M0)", vmin=0, vmax=0.6, cmap="inferno")
fig.colorbar(im, ax=axes[2], shrink=0.8)
for name in ("GM", "WM", "CSF"):
    print(f"{name:3} (voxels at least 90 % {name}): control {pct(static)[fr[name] > 0.9].mean():5.1f} % of M0, dM {pct(dm)[fr[name] > 0.9].mean():.3f} % of M0")
```

On the left, the control image: gray matter at 97 % of its $M_0$, white matter at 100 %,
and CSF, whose T1 of 3 s has not recovered by 4.5 s, at 78 %. In the middle, the difference
on the same brightness scale is black, because 0.5 % of $M_0$ is half a gray level in an
eight-bit display. On its own window, on the right, it is a perfusion map, gray matter
bright and white matter faint: what every ASL acquisition is trying to recover from under
the image on the left and the noise added to it.

## Measure it: the estimate against the kinetic model

The one-line estimate treats the label as if it decayed with the blood T1 until the readout
and ignores the transit time. The kinetic model does neither: the label decays with
$T_{1b}$ until it arrives at the tissue's ATT, and with the tissue's T1 after that. Both are
evaluated at the reference timing, 3.6 s after the start of labeling, for pure gray matter
and pure white matter.

```{code-cell} python
:tags: [hide-input]
print(f"dM/M0 at t = {t_ref:.1f} s (LD {R.labeling_duration} s + PLD {R.post_labeling_delay} s)")
print(f"{'':4} {'estimate':>10} {'kinetic model':>14} {'ratio':>7}")
for name, tis in (("GM", GM), ("WM", WM)):
    model = float(kinetic.delta_m(t_ref, tis.perfusion, tis.att, tis.t1, 1.0))
    print(f"{name:4} {100 * crude[name]:9.3f} % {100 * model:13.3f} % {model / crude[name]:7.2f}")
    print(f"     T1' of {name} (tissue T1 with outflow) = {kinetic.t1_prime(tis.t1, tis.perfusion):.2f} s")
```

The estimate is 0.695 % for gray matter and the model gives 0.531 %, a ratio of 0.76; for
white matter, 0.232 % against 0.108 %, a ratio of 0.47. The estimate is right in its order
of magnitude, which is what it was for, and high in both tissues for one reason: in the
model the labeled water spends most of the delay inside the tissue, relaxing with the
tissue's T1 (1.31 s for gray matter once the outflow term is included, 0.83 s for white
matter) rather than the blood's 1.65 s. The gap is larger in white matter because its T1
is shorter and its label arrives later. [Chapter 5](./05-kinetic-model.md) works through
the model, and [Chapter 14](../04-quantification/14-cbf-quantification.md) shows what the
last column does to a CBF computed with the one-line formula, which is the standard one.

## What this implies for acquisition

- **The signal is about one percent of the image**, and proportional to CBF. Every
  protocol choice is a trade between more signal per pair and more pairs.
- **The label expires with $T_{1b}$**: a half-life of 1.1 s. Every second between labeling
  and readout costs a factor of $e^{-1/1.65} = 0.55$, so the delay is kept as short as the
  transit time allows.
- **The transit time sets the minimum delay.** Imaging before the label has arrived
  measures nothing; the safe delay depends on the population, which is why the white paper
  recommends different delays for children, adults and the elderly ([Chapter 7](./07-acquisition-parameters.md)).
- **White matter is a harder measurement than gray matter**: a third of the flow, a later
  arrival and a shorter T1 leave it a fifth of the gray matter signal at the reference timing.
- **$\lambda$, $T_{1b}$ and $\alpha$ are assumed, not measured**, and each multiplies the
  CBF estimate directly. Record them and where they came from ([Chapter 16](../04-quantification/16-calibration.md)).

## Further reading

The nitrous oxide method {cite:p}`kety1948`, the theory of inert-gas exchange behind the
one-compartment model {cite:p}`kety1951`, and the partition coefficient of water
{cite:p}`herscovitch1985`. Arterial water as a tracer: the proposal {cite:p}`detre1992` and
its first demonstration {cite:p}`williams1992`; the blood T1 at 3 T {cite:p}`lu2004`; the
consensus recommendations that fix the constants used here {cite:p}`alsop2015`; and the
general kinetic model the animation integrated, {cite:t}`buxton1998`.
