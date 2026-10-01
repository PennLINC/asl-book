---
title: "5. The general kinetic model"
kernelspec:
  name: python3
  display_name: Python 3
---

:::{admonition} Simulated datasets in this chapter
:class: note
- **Built in this page:** the Buxton kinetic model evaluated for the phantom's tissue classes (`aslbook.kinetic`, the same code the simulator runs), on the packaged phantom slab ([Appendix B](../appendices/b-data-manifest.md#app-b-package-data)).
- **`pld-sweep`**: single-delay PCASL at six post-labeling delays from 0.5 to 3.0 s, 15 pairs each at TR 6 s, so that whole acquisitions sample the kinetic curve ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-pld-sweep)).
- **`ref-clean`**: the reference protocol with the noise switched off, whose control-label difference is the simulator's `deltam` ground truth seen through the readout ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-ref-clean)).

Pipeline-tier datasets are simulated offline by aslscan ([Chapter 0.2](../00-frontmatter/the-simulated-datasets.md)).
:::

## Learning goals

After this chapter you can:

- write the label-control difference as delivery times relaxation times clearance, and name
  the three phases of the curve (the bolus has not arrived, is arriving, has arrived)
- state the (P)CASL and PASL solutions of the model and the role of each symbol: transit
  time, labeling duration, post-labeling delay, blood T1, tissue T1, labeling efficiency
- explain the "PLD longer than ATT" rule: why the single-delay signal stops depending on
  the transit time once the whole bolus has arrived, and what limits that independence
- read a multi-delay curve and say what the model leaves out

```{code-cell} python
:tags: [hide-cell]
import numpy as np
import matplotlib.pyplot as plt

from aslbook import data, kinetic, phantom, presets, protocols, quant
from aslbook.plotting import INK, PALETTE, TISSUE_COLORS, set_style, show_image, take_slice

set_style()
T = presets.TISSUES
REF = presets.REFERENCE
TAU = REF.labeling_duration            # the reference labeling duration, 1.8 s
T_REF = TAU + REF.post_labeling_delay  # the reference readout time of the first slice, 3.6 s
T2B = presets.T2_BLOOD
SCALE = REF.signal_scale * np.exp(-REF.echo_time / T2B)   # M0 units -> image units for labeled blood at TE
print(f"one M0 unit of labeled blood is {SCALE:.1f} image units at TE {REF.echo_time * 1000:.0f} ms")
```

## Delivery, relaxation, clearance

[Chapter 3](./03-perfusion-and-tracers.md) treated labeled arterial water as a tracer, and
[Chapter 4](./04-labeling-schemes.md) described how the labeling schemes create it. This
chapter follows the tracer into the tissue. The quantity to predict is the difference in
longitudinal magnetization between the control and label conditions of one voxel, $\Delta M(t)$,
as a function of the time $t$ since labeling began. {cite:t}`buxton1998` wrote it as a
convolution of three factors, and the form is worth reading before any formula:

$$
\Delta M(t) = 2\, M_{0b}\, f \int_0^t c(t')\, r(t - t')\, m(t - t')\, \mathrm{d}t' .
$$

- **Delivery**, $c(t')$: the fraction of the arterial blood arriving at time $t'$ that is
  labeled. Blood that left the labeling plane at time zero reaches the voxel after the
  arterial transit time $\delta$ (ATT), so $c$ is zero before $\delta$. While it is nonzero,
  the label has been decaying in blood with $T_{1b}$ since it was created, so $c$ carries a
  factor $e^{-t'/T_{1b}}$ (for pulsed labeling) or $e^{-\delta/T_{1b}}$ (for continuous
  labeling, where every part of the bolus spends the same time $\delta$ in transit).
- **Clearance**, $r(t - t')$: the fraction of the water that arrived at $t'$ and is still in
  the voxel at $t$. For a freely diffusible tracer the voxel empties at the rate $f/\lambda$,
  so $r(t) = e^{-f t / \lambda}$.
- **Relaxation**, $m(t - t')$: the fraction of the label that arrived at $t'$ and has not
  relaxed by $t$. Once in the tissue the label decays with the tissue's $T_1$, so
  $m(t) = e^{-t/T_1}$.

The factor $2 M_{0b} f$ in front says that a fully inverted bolus differs from control by twice
the blood's equilibrium magnetization per unit volume, $M_{0b} = M_0/\lambda$, delivered at the
rate $f$; the labeling efficiency $\alpha$ scales the delivery function. The symbols follow the
[notation page](../00-frontmatter/notation.md): $f$ is CBF, $\lambda$ the blood-brain
partition coefficient (0.9 ml/g, {cite:p}`herscovitch1985`), $\tau$ the labeling (bolus) duration, $w$ the post-labeling
delay, and $\delta$ the transit time.

Clearance and relaxation act on the same water at the same time, so their product is a single
exponential with a shorter time constant:

$$
r(t)\, m(t) = e^{-t/T_1'}, \qquad \frac{1}{T_1'} = \frac{1}{T_1} + \frac{f}{\lambda}.
$$

$T_1'$ is the *apparent* tissue $T_1$: the time constant with which label disappears from the
voxel by relaxation and outflow together. For brain tissue outflow is slow compared with
relaxation, so $T_1'$ is almost $T_1$:

```{code-cell} python
:tags: [hide-input]
for name in ("GM", "WM"):
    tt = T[name]
    t1p = kinetic.t1_prime(tt.t1, tt.perfusion)
    print(f"{name}: f = {tt.perfusion:.0f} ml/100 g/min -> f/lambda = {tt.perfusion / 6000 / presets.LAMBDA:.4f} /s, "
          f"1/T1 = {1 / tt.t1:.3f} /s; T1' = {t1p:.3f} s vs T1 = {tt.t1:.2f} s ({(tt.t1 - t1p) / tt.t1:.1%} shorter)")
```

Gray matter's $T_1'$ is 1.311 s against a $T_1$ of 1.33 s, 1.5 % shorter; white matter's is
0.3 % shorter. At the perfusion rates of the brain the clearance term is a small correction,
and the reason it appears in every formula is completeness, not size. The difference between
$T_1'$ and the *blood* $T_1$ (1.65 s at 3 T, {cite:p}`lu2004`) is another matter, and the
last part of this chapter measures what it costs.

## The three phases and the (P)CASL solution

For continuous and pseudo-continuous labeling the delivery function is a rectangle: nothing
before $\delta$, then a constant $\alpha\, e^{-\delta/T_{1b}}$ for the duration $\tau$ of the
labeling, then nothing. The integral has three regimes, and the curve has three phases:

$$
\Delta M(t) = 2 M_{0b} f\, \alpha\, T_1'\, e^{-\delta/T_{1b}} \times
\begin{cases}
0 & t \le \delta \quad \text{(not arrived)} \\[4pt]
1 - e^{-(t-\delta)/T_1'} & \delta < t < \delta + \tau \quad \text{(arriving)} \\[4pt]
e^{-(t-\delta-\tau)/T_1'} \left(1 - e^{-\tau/T_1'}\right) & t \ge \delta + \tau \quad \text{(arrived)}
\end{cases}
$$

While the bolus is arriving, label accumulates in the voxel faster than it decays, and the
curve rises toward a plateau of $2 M_{0b} f \alpha T_1' e^{-\delta/T_{1b}}$ that it would reach
only with an infinitely long label. Once the trailing edge of the bolus has arrived, at
$t = \delta + \tau$, no more label is delivered, and what is in the voxel decays with $T_1'$.
The curve peaks at exactly that moment. The factor $e^{-\delta/T_{1b}}$ is paid once, for the
transit: it is the label lost in the arteries before any of it reached the voxel.

The PASL solution differs because the bolus is created all at once and its whole length
decays in blood with $T_{1b}$ while it travels, so the delivery function carries
$e^{-t'/T_{1b}}$ rather than a constant. The arriving and arrived phases become

$$
\Delta M(t) = 2 M_{0b} f\, \alpha\, e^{-t/T_{1b}} \times
\begin{cases}
(t - \delta)\, q_p(t) & \delta < t < \delta + \tau \\[4pt]
\tau\, q_p(t) & t \ge \delta + \tau
\end{cases}
$$

with $q_p(t)$ a correction factor close to one that accounts for the difference between
$T_{1b}$ and $T_1'$; its full form is in the dropdown. The PASL $\tau$ is the bolus duration
fixed by the QUIPSS II or Q2TIPS cut-off {cite:p}`wong1998,luh1999`
([Chapter 4](./04-labeling-schemes.md)), and the
clock $t$ starts at the labeling pulse, so $t$ is the inversion time TI.

:::{dropdown} The PASL correction factor
With $k = 1/T_{1b} - 1/T_1'$,

$$
q_p(t) = \frac{e^{k t}\left(e^{-k\delta} - e^{-k t}\right)}{k\,(t - \delta)} \quad (\delta < t < \delta + \tau), \qquad
q_p(t) = \frac{e^{k t}\left(e^{-k\delta} - e^{-k(\delta+\tau)}\right)}{k\,\tau} \quad (t \ge \delta + \tau).
$$

When $T_1' = T_{1b}$, $k \to 0$ and $q_p \to 1$: the arriving phase is then a straight line
of slope $2 M_{0b} f \alpha e^{-t/T_{1b}}$ and the arrived phase is $\tau$ times that. The code
in `aslbook.kinetic.delta_m` is the simulator's implementation of both solutions, including
its guarded divisions, so every curve on this page is the curve the pipeline used.
:::

## See it: the curve for gray and white matter

The figure evaluates the PCASL solution with the phantom's constants for pure gray matter
(f 60 ml/100 g/min, ATT 0.8 s, T1 1.33 s) and pure white matter (f 20, ATT 1.2 s, T1 0.83 s),
for the reference labeling duration of 1.8 s. The phantom's tissue $T_1$ values are those
measured at 3 T by {cite:t}`wansapura1999`, the short end of the published range;
{cite:t}`stanisz2005` report 1.82 s and 1.08 s in excised tissue. The vertical axis is $\Delta M$ as a percentage
of the tissue's $M_0$, the natural unit of the answer key. The shading marks the three phases
for gray matter, and the dotted line is the reference readout time, $\tau + w$ = 3.6 s, at
which the first slice of every reference-protocol image is read.

```{code-cell} python
:tags: [hide-input]
t = np.linspace(0, 6, 601)
fig, ax = plt.subplots(figsize=(7.5, 3.4))
gm = T["GM"]
for a, b, label in [(0, gm.att, "not arrived"), (gm.att, gm.att + TAU, "arriving"), (gm.att + TAU, 6, "arrived")]:
    ax.axvspan(a, b, color=INK["grid"], alpha=0.55 if label == "arriving" else 0.25, lw=0)
    ax.text((a + b) / 2, 1.28, label, ha="center", fontsize=8, color=INK["secondary"])
ax.fill_between([0, TAU], 1.15, 1.2, color=PALETTE[3], lw=0)
ax.text(TAU / 2, 1.08, "label on (τ = 1.8 s)", ha="center", fontsize=8, color=INK["secondary"])
for name in ("GM", "WM"):
    tt = T[name]
    dm = kinetic.delta_m(t, tt.perfusion, tt.att, tt.t1, tt.m0) / tt.m0 * 100
    ax.plot(t, dm, color=TISSUE_COLORS[name], label=f"{name}: f {tt.perfusion:.0f}, ATT {tt.att:.1f} s, T1 {tt.t1:.2f} s")
    ax.plot(tt.att + TAU, dm.max(), "o", color=TISSUE_COLORS[name], ms=6)
ax.axvline(T_REF, color=INK["secondary"], lw=1, ls=":")
ax.text(T_REF + 0.05, 0.98, "reference readout\n(PLD 1.8 s)", fontsize=8, color=INK["secondary"])
ax.set(xlabel="time since the start of labeling (s)", ylabel="ΔM / M0 (%)", xlim=(0, 6), ylim=(0, 1.35),
       title="PCASL, labeling duration 1.8 s: the difference signal of pure tissue")
ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.45), fontsize=8)
fig.tight_layout()
for name in ("GM", "WM"):
    tt = T[name]
    dm = kinetic.delta_m(t, tt.perfusion, tt.att, tt.t1, tt.m0) / tt.m0
    print(f"{name}: peak {dm.max():.2%} of M0 at t = {t[dm.argmax()]:.1f} s (ATT + tau); "
          f"{dm[np.searchsorted(t, T_REF)]:.2%} of M0 at the reference readout")
```

Look first at the size of the effect: the gray matter curve never exceeds 1.1 % of $M_0$,
and at the reference readout it is 0.53 %, the number [Chapter 3](./03-perfusion-and-tracers.md)
estimated on the back of an envelope and the reason the difference image of
[Chapter 6](./06-the-asl-signal.md) is so much noisier than the control image it is cut from.
Second, the shape: gray matter's signal is zero for 0.8 s, rises for 1.8 s, peaks at 2.6 s
(the dots mark $\delta + \tau$), and decays. White matter arrives 0.4 s later, rises more
slowly because its $T_1'$ is short, and peaks at 3.0 s at a fifth of the gray matter value:
a third of the perfusion and a shorter $T_1'$. At the reference readout both tissues are in
the arrived phase, as the reference protocol intends.

The next figure varies one parameter at a time around the gray matter curve. Each panel keeps
the other constants at the phantom's values.

```{code-cell} python
:tags: [hide-input]
sweeps = {
    "ATT δ (s)": ("att", [0.5, 0.8, 1.2, 1.6, 2.0]),
    "labeling duration τ (s)": ("tau", [0.7, 1.0, 1.8, 3.0]),
    "blood T1 (s)": ("t1b", [1.2, 1.4, 1.65, 1.9]),
    "CBF f (ml/100 g/min)": ("perfusion", [20, 40, 60, 80, 100]),
}
fig, axes = plt.subplots(2, 2, figsize=(9, 5.6), sharex=True, sharey=True)
for ax, (title, (key, values)) in zip(axes.ravel(), sweeps.items()):
    for v, color in zip(values, PALETTE):
        kw = dict(perfusion=gm.perfusion, att=gm.att, tau=TAU, t1b=presets.T1_BLOOD)
        kw[key] = v
        dm = kinetic.delta_m(t, kw["perfusion"], kw["att"], gm.t1, gm.m0, tau=kw["tau"], t1b=kw["t1b"]) / gm.m0 * 100
        ax.plot(t, dm, color=color, lw=1.6, label=f"{v:g}")
    ax.axvline(T_REF, color=INK["secondary"], lw=1, ls=":")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
for ax in axes[1]:
    ax.set_xlabel("time since the start of labeling (s)")
for ax in axes[:, 0]:
    ax.set_ylabel("ΔM / M0 (%)")
axes[0, 0].set(xlim=(0, 6), ylim=(0, 1.9))
fig.tight_layout()
```

- **Transit time** shifts the curve right and lowers it: every extra 0.4 s in the arteries
  costs $e^{-0.4/1.65}$, about 21 %, of the label. At the reference readout (dotted) a voxel
  with an ATT of 2.0 s has barely begun to fill: the failure mode of single-delay ASL in
  slow-flow territories ([Chapter 14](../04-quantification/14-cbf-quantification.md)).
- **Labeling duration** sets how long the curve keeps rising: 3.0 s raises the peak by a
  third over 1.8 s at the price of a longer TR; 0.7 s gives less than half the signal.
- **Blood T1** scales the transit loss and, for PASL, the whole decay; its lengthening with
  field strength is one of the two reasons ASL works better at 3 T {cite:p}`alsop2015`
  ([Chapter 7](./07-acquisition-parameters.md)).
- **CBF** scales the curve almost linearly, which is what makes quantification possible; the
  "almost" is the weak dependence of $T_1'$ on $f$.

## See it: pulsed against continuous labeling

The two solutions side by side, for the reference PCASL (τ 1.8 s, α 0.85) and for a PASL bolus
cut off at 0.7 s (α 0.98) as in the `label-types` dataset of
[Chapter 4](./04-labeling-schemes.md). The PASL clock starts at the labeling pulse, so its
time axis is the inversion time TI; the reference PASL readout at TI 1.8 s is marked.

```{code-cell} python
:tags: [hide-input]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), sharey=True)
for ax, name in zip(axes, ("GM", "WM")):
    tt = T[name]
    pc = kinetic.delta_m(t, tt.perfusion, tt.att, tt.t1, tt.m0, label_type="PCASL", tau=TAU) / tt.m0 * 100
    pa = kinetic.delta_m(t, tt.perfusion, tt.att, tt.t1, tt.m0, label_type="PASL", tau=0.7) / tt.m0 * 100
    ax.plot(t, pc, color=TISSUE_COLORS[name], label="PCASL, τ 1.8 s, α 0.85")
    ax.plot(t, pa, color=TISSUE_COLORS[name], ls="--", label="PASL, bolus 0.7 s, α 0.98")
    ax.axvline(T_REF, color=INK["secondary"], lw=1, ls=":")
    ax.axvline(1.8, color=INK["secondary"], lw=1, ls="-.")
    ax.set(title=f"{name}", xlabel="time since labeling began (s); PASL: TI", xlim=(0, 6), ylim=(0, 1.3))
    ax.legend(fontsize=8, loc="center right")
    print(f"{name}: PCASL peak {pc.max():.2f} % of M0, PASL peak {pa.max():.2f} % ({pa.max() / pc.max():.2f} of PCASL); "
          f"at the reference readouts PCASL {pc[360]:.2f} %, PASL {pa[180]:.2f} % ({pa[180] / pc[360]:.2f} of PCASL)")
axes[0].set_ylabel("ΔM / M0 (%)")
axes[0].text(T_REF + 0.05, 1.2, "PCASL readout (PLD 1.8 s)", fontsize=7, color=INK["secondary"])
axes[0].text(1.85, 1.2, "PASL readout (TI 1.8 s)", fontsize=7, color=INK["secondary"])
fig.tight_layout()
```

The pulsed curve rises linearly rather than exponentially during the arriving phase (its
delivery is not constant but decays with $T_{1b}$, and the $q_p$ factor bends it only
slightly), peaks at $\delta + \tau$ like the continuous one, and then falls faster, because
after the bolus has arrived the whole of it keeps decaying with $e^{-t/T_{1b}}$ in front. The
height ratio is set mostly by the bolus durations: 0.7 s of label against 1.8 s, partly
offset by PASL's higher efficiency and by the shorter time its label has spent decaying at
the peak. The printed peaks give a ratio of about 0.5 in gray matter. At the readout times
the `label-types` runs actually use (PCASL at PLD 1.8 s, PASL at TI 1.8 s) the gap is
smaller, 0.46 % against 0.53 % of $M_0$, because the PASL readout catches its curve much
nearer its peak; that is the amplitude ratio [Chapter 4](./04-labeling-schemes.md) measures
between the two runs.

## The "PLD longer than ATT" rule

The single-delay quantification of [Chapter 14](../04-quantification/14-cbf-quantification.md)
rests on one property of the arrived phase. Write the time of the readout as $t = \tau + w$
and substitute into the arrived branch of the PCASL solution:

$$
\Delta M(\tau + w) = 2 M_{0b} f \alpha\, T_1' \left(1 - e^{-\tau/T_1'}\right)
e^{-\delta/T_{1b}}\; e^{-(w - \delta)/T_1'} .
$$

If the label decayed at the same rate in blood and in tissue, $T_1' = T_{1b}$, the two
exponentials would combine into $e^{-w/T_{1b}}$ and the transit time would drop out
entirely: the signal at a delay $w \ge \delta$ would be the same whatever $\delta$ is, as
long as the whole bolus has arrived. This is the observation of {cite:t}`alsop1996`, and it
is the basis of the recommendation to choose a post-labeling delay longer than the longest
transit time expected in the population {cite:p}`alsop2015`. Under that condition one image
at one delay gives CBF without knowing the transit time, through the white-paper formula
that [Chapter 14](../04-quantification/14-cbf-quantification.md) applies:

$$
f = \frac{6000\, \lambda\, \Delta M\, e^{w/T_{1b}}}{2\, \alpha\, T_{1b}\, M_0 \left(1 - e^{-\tau/T_{1b}}\right)} .
$$

The independence is exact only when $T_1' = T_{1b}$. In the model, and in the simulator, the
label decays with the tissue's $T_1'$ from the moment it arrives, 1.31 s in gray matter
against 1.65 s in blood. The figure shows both cases, each curve one transit time plotted
against the post-labeling delay, with the formula's implied signal dashed.

```{code-cell} python
:tags: [hide-input]
w = np.linspace(0, 3.0, 301)
atts = [0.5, 0.8, 1.2, 1.6, 2.0]
f, m0 = gm.perfusion, gm.m0
wp = 2 * (m0 / presets.LAMBDA) * (f / 6000) * presets.ALPHA["PCASL"] * presets.T1_BLOOD * np.exp(-w / presets.T1_BLOOD) * (1 - np.exp(-TAU / presets.T1_BLOOD))
T1_AS_BLOOD = 1 / (1 / presets.T1_BLOOD - f / 6000 / presets.LAMBDA)   # the tissue T1 that makes T1' exactly T1b
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
for ax, t1_tissue, title in zip(axes, (T1_AS_BLOOD, gm.t1), ("label decays with T1' = T1b = 1.65 s after arrival", "label decays with the tissue T1 = 1.33 s (the phantom)")):
    for att, color in zip(atts, PALETTE):
        dm = kinetic.delta_m(TAU + w, f, att, t1_tissue, m0)
        ax.plot(w, dm / m0 * 100, color=color, lw=1.6, label=f"ATT {att:.1f} s")
        ax.plot(att, kinetic.delta_m(TAU + att, f, att, t1_tissue, m0) / m0 * 100, "o", color=color, ms=5)
    ax.plot(w, wp / m0 * 100, "k--", lw=1.2, label="white-paper formula")
    ax.set(title=title, xlabel="post-labeling delay w (s)", xlim=(0, 3))
axes[0].set_ylabel("ΔM / M0 (%)")
axes[0].legend(fontsize=7, ncol=2)
fig.tight_layout()
short = kinetic.delta_m(TAU + 1.0, f, 1.6, T1_AS_BLOOD, m0) / (wp[np.searchsorted(w, 1.0)])
late = kinetic.delta_m(TAU + 2.5, f, gm.att, gm.t1, m0) / kinetic.delta_m(TAU + 1.8, f, gm.att, gm.t1, m0)
print(f"left panel, ATT 1.6 s read at PLD 1.0 s: {short:.0%} of the formula's value")
print(f"right panel, GM at PLD 2.5 s: {late:.0%} of the signal at PLD 1.8 s (exp(-0.7/T1'))")
```

On the left (the tissue given the $T_1$ that makes $T_1' = T_{1b}$ exactly) every curve,
once past its own transit time (the dots mark $w = \delta$), lies on the dashed line: the
signal is the same at any delay longer than the transit time, and the formula recovers $f$
exactly. On the right, the phantom's case, the curves decay faster than the formula assumes
and no longer coincide: a voxel with a long transit time has spent more of the delay in
blood, where the label lasts longer, and keeps slightly more signal. The formula
underestimates CBF in proportion to the gap, which grows with the delay; the measurement
below puts a number on it.

Two more things follow. A delay shorter than the transit time is a large error: at
$w$ = 1.0 s the 1.6 s curve on the left is at 54 % of the formula's value, so a slow-flow
region would be reported at half its true CBF. And a long delay is safe but expensive: in
the phantom's gray matter the signal at 2.5 s is 59 % of that at 1.8 s at the same noise, so
SNR falls as $e^{-w/T_1'}$ with every second of caution
([Chapter 7](./07-acquisition-parameters.md)).

## The multi-delay view

A multi-delay acquisition samples the curve at several post-labeling delays and fits both
$f$ and $\delta$ to it {cite:p}`mezue2014,woods2024`
([Chapter 15](../04-quantification/15-multi-delay.md)). The
`pld-sweep` dataset does this with whole acquisitions: six single-delay PCASL series on the
same slab at delays of 0.5 to 3.0 s in steps of 0.5 s, 15 pairs each, TR 6 s so that the
longest delay fits. In a 2D readout each slice has its own readout time,
PLD + τ + `SliceTiming[z]`, so the model is evaluated at every voxel's own slice time and
averaged over the region, exactly as the simulator did.

The figure compares, in pure gray matter (voxels more than 90 % GM) and pure white matter,
the measured mean control-label difference at each delay (±2 standard errors over pairs),
the model curve for the pure tissue at the region's slice times scaled from $M_0$ units to
image units by $100 \times e^{-\mathrm{TE}/T_{2b}}$ (the simulator's intensity scale and
the blood's $T_2$ decay at TE 12 ms, with the simulator's blood $T_2$ of 165 ms; the
measured value depends on oxygenation and hematocrit, {cite:p}`zhao2007`), and the `deltam`
ground truth of the same voxels.

```{code-cell} python
:tags: [hide-input]
sweep = data.load_dataset("pld-sweep")
runs = ["pld05", "pld10", "pld15", "pld20", "pld25", "pld30"]
rows = []                                   # (pld, tissue, measured, se, truth, n_pairs)
for name in runs:
    run = sweep.run(name)
    p, ctx = run.sidecar(), run.context()
    d = quant.subtract(run.mag(), ctx)      # (x, y, z, pairs)
    fr = run.fractions()
    lab = [i for i, r in enumerate(ctx) if r == "label"]
    gt = run.truth("deltam")[..., lab[0]] * SCALE
    offsets = np.asarray(protocols.slice_offsets(p))
    for tissue in ("GM", "WM"):
        roi = fr[tissue.lower()] > 0.9
        per_pair = d[roi].mean(axis=0)      # ROI mean of each pair
        rows.append((p["PostLabelingDelay"], tissue, per_pair.mean(), per_pair.std(ddof=1) / np.sqrt(per_pair.size), gt[roi].mean()))
roi_slices = {tissue: np.bincount(np.nonzero(fr[tissue.lower()] > 0.9)[2], minlength=len(offsets)) for tissue in ("GM", "WM")}

def model_curve(tissue, plds, tau=TAU):
    tt = T[tissue]
    per_slice = np.stack([kinetic.delta_m(plds + tau + off, tt.perfusion, tt.att, tt.t1, tt.m0) for off in offsets])
    wts = roi_slices[tissue] / roi_slices[tissue].sum()
    return (per_slice * wts[:, None]).sum(0) * SCALE

pld_fine = np.linspace(0, 3.2, 321)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
for ax, tissue in zip(axes, ("GM", "WM")):
    sel = [r for r in rows if r[1] == tissue]
    x, y, se, gt_ = (np.array([r[i] for r in sel]) for i in (0, 2, 3, 4))
    ax.plot(pld_fine, model_curve(tissue, pld_fine), color=TISSUE_COLORS[tissue], lw=1.6, label="model, pure tissue, ROI slice times")
    ax.plot(x, gt_, "s", color=INK["secondary"], ms=5, mfc="none", label="deltam ground truth in the ROI")
    ax.errorbar(x, y, yerr=2 * se, fmt="o", color=TISSUE_COLORS[tissue], ms=6, capsize=3, label="measured (±2 SE)")
    ax.set(title=f"{tissue} > 90 %: {roi_slices[tissue].sum()} voxels", xlabel="post-labeling delay (s)", ylabel="control − label (image units)", xlim=(0, 3.2))
    ax.legend(fontsize=7)
    for pld, _, m, s, g in sel:
        print(f"{tissue} PLD {pld:.1f} s: measured {m:6.2f} ± {s:.2f}, model {model_curve(tissue, np.array([pld]))[0]:6.2f}, truth {g:6.2f}")
fig.tight_layout()
```

The measured points sit on the model curve at every delay: in gray matter within 0.2 image
units (0.3 %) of the pure-tissue model and within 2 % of the ground truth of the same
voxels. Truth and pure-tissue model differ by 1 to 2 % because a voxel "more than 90 % GM"
still holds a little white matter, whose curve is lower and later. The white matter points
are noisier (a fifth of the signal at the same noise) and lie a few percent below the truth
at the shortest delays, where the readout's blurring moves a little of the bright gray
matter signal across the boundary. The curve is the first figure's, seen through the
readout: the decline from 0.5 to 3.0 s is $e^{-2.5/T_1'}$, a factor of six in gray matter,
which is what a delay costs.

## See it: the toy tier is the pipeline tier minus the readout

The `ref-clean` run is the reference protocol with the noise switched off. Its control-label
difference should equal the `deltam` ground truth, scaled by the same
$100 \times e^{-\mathrm{TE}/T_{2b}}$, in every voxel, up to what the 2D EPI readout does to
an image. The figure shows the noise-free difference, the scaled truth, and their
difference on the display slice.

```{code-cell} python
:tags: [hide-input]
clean = data.load_dataset("ref-clean").run("pcasl")
ctx_c = clean.context()
d_clean = quant.subtract(clean.mag(), ctx_c).mean(axis=-1)
lab_c = [i for i, r in enumerate(ctx_c) if r == "label"]
gt_clean = clean.truth("deltam")[..., lab_c[0]] * SCALE
mask_c = clean.mask()
err = d_clean - gt_clean
k = phantom.DISPLAY_SLICE
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
show_image(axes[0], take_slice(d_clean, k), "noise-free control − label", vmin=0, vmax=40)
show_image(axes[1], take_slice(gt_clean, k), "deltam truth × 100 e^(−TE/T2b)", vmin=0, vmax=40)
im = show_image(axes[2], take_slice(np.where(mask_c, err, np.nan), k), "difference (image units)", kind="diff", vmin=-4, vmax=4)
fig.colorbar(im, ax=axes[2], shrink=0.8)
fig.tight_layout()
fr_c = clean.fractions()
gm_pure = fr_c["gm"] > 0.9
inside = mask_c & (gt_clean > 5)                 # voxels with at least ~1/6 of the gray matter signal
ratio = d_clean[inside] / gt_clean[inside]
print(f"ratio over {inside.sum()} voxels with truth > 5: median {np.median(ratio):.3f}, "
      f"interquartile range {np.percentile(ratio, 25):.3f}-{np.percentile(ratio, 75):.3f}")
print(f"pure GM (> 90 %): ROI-mean difference {d_clean[gm_pure].mean():.2f} vs truth {gt_clean[gm_pure].mean():.2f} image units "
      f"(ratio {d_clean[gm_pure].mean() / gt_clean[gm_pure].mean():.3f}); voxelwise error SD {err[gm_pure].std():.2f}")
print(f"whole brain: error SD {err[mask_c].std():.2f} image units, {err[mask_c].std() / gt_clean[gm_pure].mean():.1%} of the GM signal; "
      f"summed over the brain, {d_clean[mask_c].sum() / gt_clean[mask_c].sum():.4f} of the truth")
```

The two images are the same image. The third panel is their difference on a ±4 image-unit
window, a tenth of the gray matter signal. It is not zero: it carries a fine pattern of
alternating sign along tissue boundaries, strongest around the ventricles and at the cortex
(the same ringing is visible as faint horizontal striping outside the head in the left
image). That pattern is the readout. The truth box-averages
the 1 mm phantom onto acquisition voxels; the readout instead truncates its k-space at the
acquisition matrix, and a truncated sharp edge rings
([Chapter 2](../01-mri-physics/02-epi-and-reconstruction.md)). Its size is 1.3 image units
(the printed error SD), about 4 % of the gray matter signal, and it nearly sums to nothing:
over the whole brain the noise-free difference holds the same total as the truth to within
0.3 %, and in pure gray matter the region mean is 1.8 % above the truth. That is the
systematic error a noise-free measurement of this phantom carries before any quantification,
and it justifies the statement the book relies on: the toy tier (`aslbook.synth`, the
kinetic model per voxel) is the pipeline tier without the readout, and the readout is a
few-percent perturbation of the difference image concentrated at edges.

## Measure it: peaks, transit times, and the cost of T1' ≠ T1b

Two measurements. First, where the curves peak. The model says $\Delta M$ of the first slice peaks at
PLD = ATT, 0.8 s in gray matter and 1.2 s in white matter, and each later slice earlier by
its offset. The data sample the curve every 0.5 s, so the cell restricts the regions to the
first four slices (offsets up to 0.12 s) and fits the model's transit time to the six
measured points by a grid search with the amplitude free, the way
[Chapter 15](../04-quantification/15-multi-delay.md) fits every voxel.

```{code-cell} python
:tags: [hide-input]
early = np.zeros(len(offsets), bool); early[:4] = True
meas_early = {"GM": [], "WM": []}
for name in runs:
    run = sweep.run(name)
    d = quant.subtract(run.mag(), run.context())
    fr = run.fractions()
    for tissue in ("GM", "WM"):
        roi = (fr[tissue.lower()] > 0.9) & early[None, None, :]
        meas_early[tissue].append(d[roi].mean())
plds = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
att_grid = np.arange(0.2, 2.01, 0.02)
for tissue in ("GM", "WM"):
    tt = T[tissue]
    y = np.array(meas_early[tissue])
    best = None
    for a in att_grid:
        g = np.mean([kinetic.delta_m(plds + TAU + off, tt.perfusion, a, tt.t1, tt.m0) for off in offsets[:4]], axis=0)
        amp = (g @ y) / (g @ g)
        rss = ((y - amp * g) ** 2).sum()
        if best is None or rss < best[0]:
            best = (rss, a, amp)
    model_first = kinetic.delta_m(plds + TAU, tt.perfusion, tt.att, tt.t1, tt.m0)
    print(f"{tissue}: model peak at PLD = ATT = {tt.att:.1f} s; among the six sampled delays the model is highest at "
          f"{plds[model_first.argmax()]:.1f} s and the data at {plds[y.argmax()]:.1f} s; "
          f"ATT fitted to the six points: {best[1]:.2f} s (truth {tt.att:.1f} s)")
```

The gray matter data are highest at the 0.5 s sample and the white matter data at 1.0 s,
where the model puts its highest samples given that the true peaks (0.8 and 1.2 s) fall
between grid points. The fit recovers 0.80 s and 1.20 s, the phantom's values, from region
means of 15 pairs; a per-voxel fit has far less signal to work with
([Chapter 15](../04-quantification/15-multi-delay.md)).

Second, the cost of assuming $T_1' = T_{1b}$. The white-paper formula applied to the model's
own noise-free signal at the reference timing returns:

```{code-cell} python
:tags: [hide-input]
for tissue in ("GM", "WM"):
    tt = T[tissue]
    dm_ref = kinetic.delta_m(T_REF, tt.perfusion, tt.att, tt.t1, tt.m0)
    cbf_formula = quant.cbf_pcasl(dm_ref, tt.m0, REF.post_labeling_delay, tau=TAU)
    t1_as_blood = 1 / (1 / presets.T1_BLOOD - tt.perfusion / 6000 / presets.LAMBDA)
    dm_blood = kinetic.delta_m(T_REF, tt.perfusion, tt.att, t1_as_blood, tt.m0)
    print(f"{tissue}: formula gives {cbf_formula:5.1f} ml/100 g/min for a true {tt.perfusion:.0f} "
          f"({cbf_formula / tt.perfusion - 1:+.0%}); with T1' = T1b the model's signal would be "
          f"{dm_blood / dm_ref:.2f} x larger and the formula exact")
```

With the label decaying at the tissue's rate after arrival, the formula underestimates gray
matter CBF by about 24 % and white matter by about 53 % at PLD 1.8 s: the largest
systematic error in [Chapter 14](../04-quantification/14-cbf-quantification.md)'s
single-delay maps of this phantom. Real tissue is kinder than the model: water spends some
time in the capillaries before exchanging, still decaying with $T_{1b}$
{cite:p}`stlawrence2000`, so the true error lies between zero and the model's figure. The
white paper chose $T_{1b}$ knowingly, as the assumption wrong by the least when the
exchange time is unknown {cite:p}`alsop2015`.

## What the model leaves out

The model is a single well-mixed compartment fed by a rectangular bolus. Four things it
omits matter in real data:

- **Dispersion.** The bolus does not arrive with a sharp edge: laminar flow and branching
  spread it over a few hundred milliseconds, so the arriving phase begins earlier and rises
  more gently than the rectangle, and a fitted transit time is a compromise between first
  and mean arrival. Models convolve the bolus with a gamma or Gaussian kernel
  {cite:p}`hrabe2004,chappell2013`; the simulator does not.
- **Macrovascular signal.** Label still in an artery at readout is counted as if it had
  perfused the voxel, a large bright spurious signal at delays below the transit time.
  Vascular crushing or a longer delay removes it, and a multi-delay fit can model it as a
  separate arterial component {cite:p}`chappell2010`; the simulator has no arterial
  compartment, so its `pld05` run is cleaner than a real one.
- **Restricted exchange.** Water crosses the capillary wall in a finite time, and meanwhile
  the label decays with $T_{1b}$, not $T_1'$ {cite:p}`stlawrence2000,zhou2001,parkes2002`; multi-echo ASL
  ([Chapter 19](../05-advanced/19-frontiers.md)) can measure the exchange time because the
  two compartments also differ in $T_2$ {cite:p}`gregori2013`.
- **One T1 per tissue.** $T_{1b}$ varies by about 10 % between subjects with hematocrit and
  oxygenation {cite:p}`lu2004,hales2016`, which enters CBF as an equal error ([Chapter 16](../04-quantification/16-calibration.md)).

## What this implies for acquisition

- **Choose the post-labeling delay longer than the longest transit time you expect**, so
  that every voxel is in the arrived phase and a single delay gives CBF without knowing the
  transit time. Longer than needed costs signal as $e^{-w/T_1'}$; shorter costs accuracy in
  the slowest regions first.
- **Longer labeling raises the signal** in proportion to $1 - e^{-\tau/T_1'}$: 1.8 s captures
  three quarters of the plateau, 3 s about 90 %, and each second of label is a second of TR.
- **In a 2D readout each slice has its own delay**, later by its slice offset. The model
  predicts the per-slice signal exactly, and the quantification must use the per-slice
  delay ([Chapter 14](../04-quantification/14-cbf-quantification.md)).
- **If transit times are the question, sample the curve.** Several delays spanning the
  expected transit times let $f$ and $\delta$ be fitted together
  ([Chapter 15](../04-quantification/15-multi-delay.md)), at the cost of signal per delay.
- **Know which T1 your formula assumes.** The blood-T1 assumption of the white-paper formula
  is a choice, and on a phantom with instant exchange it underestimates gray matter CBF by a
  quarter; on a real head the error is smaller and unknown.

## Further reading

The general kinetic model {cite:p}`buxton1998`; the transit-time-insensitive single-delay
scheme {cite:p}`alsop1996` and its recommendation in the white paper {cite:p}`alsop2015`;
exchange between capillary and tissue {cite:p}`stlawrence2000,zhou2001,parkes2002` and
its measurement through $T_2$ {cite:p}`gregori2013`; the
pulsed bolus and its cut-off {cite:p}`wong1998,luh1999`; multi-delay modeling of transit
time and dispersion {cite:p}`chappell2010,hrabe2004,chappell2013,woods2024`; the constants
the model needs, the partition coefficient {cite:p}`herscovitch1985`, the blood $T_1$
{cite:p}`lu2004,hales2016` and the tissue $T_1$ {cite:p}`wansapura1999,stanisz2005`.
