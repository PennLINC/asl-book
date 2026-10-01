---
title: "4. Labeling schemes: PASL, CASL, PCASL"
kernelspec:
  name: python3
  display_name: Python 3
---

:::{admonition} Simulated datasets in this chapter
:class: note
- **Built in this page:** labeling geometry and timing diagrams, and the kinetic model's bolus for each scheme on a pure gray matter voxel ([Appendix B](../appendices/b-data-manifest.md#app-b-package-data)).
- **`label-types`**: the same slab and the same 2D readout under PCASL (efficiency 0.85), CASL (0.68) and PASL with a Q2TIPS cut-off at 0.7 s and an inversion time of 1.8 s (0.98) ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-label-types)).

Pipeline-tier datasets are simulated offline by aslscan ([Chapter 0.2](../00-frontmatter/the-simulated-datasets.md)).
:::

## Learning goals

After this chapter you can:

- describe how pulsed, continuous and pseudo-continuous labeling each invert arterial
  water, and where in the head they do it
- explain why the duration of a pulsed bolus is unknown and how a QUIPSS II or Q2TIPS
  cut-off fixes it
- say what a control condition must match and why the amplitude-modulated CASL control
  costs efficiency
- define the labeling efficiency $\alpha$, quote the standard value for each scheme, and
  find it in a BIDS sidecar
- predict the ratio of the difference signals of two schemes from their efficiencies and
  their timing, and check it on simulated images

```{code-cell} python
:tags: [hide-cell]
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle

from aslbook import data, kinetic, phantom, presets, protocols, quant
from aslbook.plotting import INK, PALETTE, TISSUE_COLORS, sequence_diagram, set_style, show_slice

set_style()
GM = presets.TISSUES["GM"]
R = presets.REFERENCE
K = phantom.DISPLAY_SLICE
```

## What labeling has to do

[Chapter 3](./03-perfusion-and-tracers.md) established the tracer: arterial water whose
longitudinal magnetization has been inverted. Making it is the labeling problem, and it
has three parts. The inversion must happen *upstream*, in blood that has not yet reached
the imaged tissue, and must not touch the tissue itself. It must produce a bolus of
labeled blood whose amount, or at least whose duration, is known, because the amount
delivered is what the signal is proportional to. And it must come with a **control**
condition that leaves the tissue in exactly the same state except for the label, so that
control minus label isolates the labeled water. Three families of methods solve these in
different ways, and the differences matter for the signal's size, for what has to be
assumed to quantify it, and for what can go wrong.

## Pulsed labeling

**Pulsed ASL (PASL)** inverts a thick slab of tissue and blood, 15 to 20 cm in the white
paper's recommendation {cite:p}`alsop2015`, below the imaging region with a single
adiabatic inversion pulse a few milliseconds long. The
first implementations differed in where the slab was and how the control was made. EPISTAR
inverted a slab below the imaging slices and, for control, the mirror-image slab above
them {cite:p}`edelman1994`. FAIR inverted everything with a non-selective pulse and, for
control, only the imaging slab with a selective one, so that the difference is the blood
that flowed in from outside the slab {cite:p}`kim1995,kwong1995`; inversion-recovery images
sensitive to inflowing blood had already been used to map brain activation
{cite:p}`kwong1992`. PICORE inverts the slab below the
slices and applies the same pulse off-resonance, away from any tissue, for the control
{cite:p}`wong1997`.
All three deliver the same thing: at $t = 0$, all the arterial blood in the labeling
region is inverted at once, and from then on it drains into the brain.

The difficulty is the bolus. The labeled blood is whatever the slab held, and how long it
takes to leave the slab depends on the slab's length, the blood velocity and the geometry
of the vessels, which vary from person to person and are not known. A pulsed bolus has a
duration $\tau$ that the experiment does not control, and without $\tau$ the signal
cannot be turned into a flow. QUIPSS II solves this by cutting the bolus short
deliberately: at a time $\mathrm{TI}_1$ after the inversion, a saturation pulse applied
to the labeling slab destroys whatever labeled blood is still in it {cite:p}`wong1998`. If
$\mathrm{TI}_1$ is shorter than the time the slab takes to empty, the bolus that reaches
the brain has a known duration, $\tau = \mathrm{TI}_1$. Q2TIPS replaces the single
saturation with a train of thin-slice saturation pulses at the leading edge of the slab,
repeated from $\mathrm{TI}_1$ until shortly before the readout, which makes the cut-off
sharper and less sensitive to the slab profile {cite:p}`luh1999`. The book's PASL protocol
is FAIR with a Q2TIPS cut-off at 0.7 s and an inversion time of 1.8 s; its labeling
efficiency is 0.98, the white paper's value for pulsed labeling {cite:p}`alsop2015`,
because a single adiabatic pulse inverts nearly perfectly.

The pulsed clock runs from the inversion pulse. The delay between it and the readout is
the **inversion time (TI)**; BIDS {cite:p}`clement2022` records it as `PostLabelingDelay` all the same, and
the kinetic model's time $t$ is TI. Because the whole bolus was labeled at $t = 0$, all of
it has decayed by $e^{-t/T_{1b}}$ when it is imaged, which is the pulsed branch of the
model in [Chapter 5](./05-kinetic-model.md).

## Continuous labeling

**Continuous ASL (CASL)** does not invert a region; it inverts a *plane*
{cite:p}`williams1992`. A long, low-power RF pulse, lasting seconds, is applied
together with a magnetic field gradient along the direction of flow, so that the RF is on
resonance only in one thin plane across the feeding arteries in the neck. Blood flowing
through the plane experiences the RF frequency sweeping past its resonance as its
position changes, and if the sweep is slow enough compared with the RF amplitude, its
magnetization follows the effective field and ends inverted: a **flow-driven adiabatic
inversion**, the adiabatic fast passage that {cite:t}`dixon1986` used to label flowing
blood for angiography. Blood that reaches the plane is labeled continuously for as long as the RF is
on, so the bolus duration $\tau$ is the **labeling duration (LD)**, set by the operator.
That is the great advantage over pulsed labeling: the bolus is long and its duration is
known. The interval from the end of labeling to the readout is the **post-labeling delay
(PLD)** {cite:p}`alsop1996`, and the kinetic clock $t = \mathrm{LD} + \mathrm{PLD}$ runs
from the start of labeling.

Continuous labeling has two costs. The first is **magnetization transfer (MT)**. The
labeling RF is applied for seconds at a frequency offset from the imaging slices; it
does not excite their free water, but it does saturate the broad resonance of the
protons bound to macromolecules, which exchange magnetization with the free water and
lower the tissue signal {cite:p}`zhang1995`. The loss is far larger than the perfusion
signal, so the control must reproduce the same MT. For a single slice it can: apply the control
RF at the mirror-image offset, on the other side of the slice, with the gradient
reversed, and the tissue sees the same off-resonance power. For many slices at different
offsets no single mirror works. The **amplitude-modulated control** solves this by
modulating the control RF with a sinusoid, which makes two closely spaced inversion
planes instead of one; blood passing both is inverted twice, net unlabeled, while the RF
power and its MT are the same as in the label condition {cite:p}`alsop1998`. The double
inversion is imperfect, and the label itself is imperfect, so the efficiency of CASL with
this control, about 0.68 at 3 T {cite:p}`wang2005,wu2007`, is lower than that of the inversion
alone. The second
cost is hardware: a seconds-long continuous RF pulse is more than the body transmit
hardware of most clinical scanners is built to deliver, which is the white paper's stated
reason for preferring the pseudo-continuous form {cite:p}`alsop2015`.

## Pseudo-continuous labeling

**Pseudo-continuous ASL (PCASL)** produces the same flow-driven inversion with a train of
short RF pulses instead of one long one {cite:p}`dai2008`. Pulses of about half a
millisecond are played every millisecond or so, each with a slice-selective gradient
across the labeling plane, and a smaller net gradient between pulses. Blood moving through
the plane sees the train as an approximately continuous adiabatic sweep and is inverted;
the pulses are short enough for any scanner's transmit coil. The control plays the same
pulses with their phase alternating by 180° from pulse to pulse, so that the net effect on
the flowing spins cancels while the average RF power, and hence the MT, is identical.
Simulations put the best achievable inversion efficiency at 0.85 over the range of
arterial velocities, with 0.80 measured {cite:p}`wu2007`, and a later in vivo measurement
against phase-contrast flow gave 0.86 {cite:p}`aslan2010`; 0.85 is the value the ASL white
paper adopts and the simulator uses by default {cite:p}`alsop2015`.

The pulse train has a weakness that continuous labeling does not: between pulses, the
spins at the labeling plane accumulate phase from any **off-resonance** there, and a
field offset at the labeling plane detunes the train and reduces the efficiency
{cite:p}`zhao2017`. The
efficiency also depends on the velocity of the blood {cite:p}`aslan2010`, falling when
blood moves too fast or when its velocity pulses through the cardiac cycle
{cite:p}`zhao2017`; both are why the labeling plane is placed where the carotid and
vertebral arteries run straight and where the field is uniform, and why some
implementations calibrate the RF phase per subject. In the *balanced* implementation the
label and control use the same gradient waveform and differ only in RF phase; in the
*unbalanced* one the control's mean gradient is set to zero. Both were implemented early
{cite:p}`wu2007`, and in simulation and experiment the unbalanced scheme was the more
robust of the two to off-resonance {cite:p}`zhao2017`; the white paper prefers it for the
same reason {cite:p}`alsop2015`. The simulator models none of this: its
PCASL is a bolus of duration LD with efficiency $\alpha$.

## The control, and what $\alpha$ means

Whatever the scheme, the control has one job: to match the label in everything except
the label. MT, the RF's heating, eddy currents from the labeling gradients, the timing of
every pulse, and the excitation history of the imaged slices must be identical, because
the difference image is the perfusion signal plus whatever the control failed to match,
and the perfusion signal is one percent of the image ([Chapter 3](./03-perfusion-and-tracers.md)).
A control that mismatches the tissue signal by one part in a thousand has already added a
tenth of the signal.

The **labeling efficiency** $\alpha$ is the fraction of the ideal inversion that the
scheme achieves: the arterial magnetization difference between label and control is
$2\alpha M_{0b}$, with $\alpha = 1$ for a perfect inversion. It folds in everything that
reduces the label, including the imperfect control of CASL, and it multiplies the signal
and therefore the CBF estimate directly: a CBF computed with $\alpha = 0.85$ when the true
efficiency was 0.75 is 13 % too low. BIDS records the value the acquisition assumes in the
sidecar field `LabelingEfficiency` {cite:p}`clement2022`. The simulator takes a default per labeling type
(`presets.ALPHA`) unless the protocol overrides it, and records the value it used and where
it came from under `AslscanSimulation.Resolved.LabelingEfficiency`:

```{code-cell} python
:tags: [hide-input]
print("simulator defaults:", presets.ALPHA)
ds = data.load_dataset("label-types")
runs = {name: ds.run(name) for name in ("pcasl", "casl", "pasl")}
for name, run in runs.items():
    p = run.sidecar()
    bolus = f"LabelingDuration {p['LabelingDuration']} s" if "LabelingDuration" in p else f"BolusCutOffDelayTime {p['BolusCutOffDelayTime']} s ({p['BolusCutOffTechnique']})"
    eff = p["AslscanSimulation"]["Resolved"]["LabelingEfficiency"]
    print(f"{name:5}: ArterialSpinLabelingType {p['ArterialSpinLabelingType']:5}  {bolus:40} PostLabelingDelay {p['PostLabelingDelay']} s  "
          f"LabelingEfficiency {p['LabelingEfficiency']} (resolved: {eff['Value']}, source {eff['Source']})")
```

The PCASL and PASL runs use the defaults; the CASL run's efficiency was set to 0.68 by the
dataset's overlay, and the sidecar says so. The simulator does not model magnetization
transfer or the amplitude-modulated control themselves; the lower efficiency is how the
book represents their cost.

## See it: where the label is made

The three sketches are sagittal views, not to scale, with the two feeding arteries drawn
in red and the book's 10 cm imaging slab dashed in blue.

```{code-cell} python
:tags: [hide-input]
ORANGE, YELLOW, BLUE, RED = TISSUE_COLORS["GM"], PALETTE[3], PALETTE[0], PALETTE[7]

def head(ax, title):
    ax.add_patch(Rectangle((-3.2, -7), 6.4, 9, color="0.92", zorder=0))      # neck
    ax.add_patch(Ellipse((0, 8), 13, 15, color="0.92", zorder=0))            # head
    ax.add_patch(Ellipse((0, 8.5), 10, 11, color="0.84", zorder=1))          # brain
    for x0 in (-1.3, 1.3):                                                   # carotid + vertebral arteries
        ax.plot([x0, x0, 0.6 * x0, 2.5 * x0], [-7, 1.5, 5, 9], color=RED, lw=2, zorder=2, solid_capstyle="round")
    ax.add_patch(Rectangle((-7, 5), 14, 8, fill=False, ec=BLUE, lw=1.5, ls="--", zorder=3))
    ax.text(7.3, 9, "imaging\nslab", color=BLUE, fontsize=8, va="center")
    ax.set(xlim=(-10, 12), ylim=(-8, 17), aspect="equal", title=title)
    ax.set_axis_off()

fig, axes = plt.subplots(1, 3, figsize=(11, 4.2))
head(axes[0], "PASL: a slab inverted at once")
axes[0].add_patch(Rectangle((-7, -7), 14, 11, color=ORANGE, alpha=0.3, zorder=1))
axes[0].add_patch(Rectangle((-7, 3), 14, 1, fill=False, hatch="////", ec=YELLOW, lw=0, zorder=4))
axes[0].text(-9.8, -1.5, "inversion slab\n(one 10 ms pulse\nat t = 0)", color=ORANGE, fontsize=8, va="center")
axes[0].text(7.3, 3.5, "Q2TIPS saturation\nfrom TI1 = 0.7 s", color=YELLOW, fontsize=8, va="center")
head(axes[1], "CASL: a plane, RF on for 1.8 s")
axes[1].add_patch(Rectangle((-4.5, -3.2), 9, 0.4, color=ORANGE, zorder=4))
axes[1].text(-9.8, -3, "labeling plane:\nflow-driven\nadiabatic inversion", color=ORANGE, fontsize=8, va="center")
axes[1].annotate("", xy=(-1.3, -0.5), xytext=(-1.3, -5.5), arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.2))
axes[1].text(-2.0, -5.8, "flow", color=RED, fontsize=7, ha="right")
head(axes[2], "PCASL: a plane, pulses every 1 ms")
axes[2].add_patch(Rectangle((-4.5, -3.2), 9, 0.4, color=ORANGE, zorder=4))
for k in range(9):
    axes[2].plot([-9.5 + 0.55 * k] * 2, [-4.6, -3.7 + 0.5 * (k % 2 == 0)], color=ORANGE, lw=1.2)
axes[2].text(-9.8, -2.2, "labeling plane:\n~1800 short pulses,\nalternating phase\nfor control", color=ORANGE, fontsize=8, va="bottom")
fig.tight_layout()
```

In pulsed labeling (left) the inverted region is a slab 10 cm thick immediately below the
imaging slab, with a small gap so that the inversion's imperfect edges do not touch the
imaged tissue; the Q2TIPS saturation at its leading edge (hatched) is what turns the slab's
unknown drainage into a bolus of known duration. In continuous and pseudo-continuous
labeling (middle, right) the label is made at a single plane across the neck, and the bolus
is however much blood crossed the plane while the RF was on. The two differ in how the RF
is delivered, not in where.

## See it: the timing of each scheme

The timing diagrams below use the book's parameters: a 2D readout of 20 slices 40 ms
apart (0.76 s from the first to the last), a TR of 4.5 s, and for the continuous schemes
a labeling duration of 1.8 s and a post-labeling delay of 1.8 s. The PASL row has a
Q2TIPS cut-off at 0.7 s and an inversion time of 1.8 s, so its readout begins 1.8 s after
the labeling pulse, while the continuous schemes read out at 3.6 s.

```{code-cell} python
:tags: [hide-input]
ro = R.readout_duration
LD, PLD = R.labeling_duration, R.post_labeling_delay
schemes = {
    f"PASL (FAIR + Q2TIPS): TI {PLD} s, cut-off 0.7 s": (
        [("labeling", 0.0, 0.03, ORANGE), ("cut-off", 0.0, 0.0, "none")]
        + [("cut-off", s, s + 0.008, YELLOW) for s in np.arange(0.7, 1.3, 0.025)]
        + [("readout", PLD, PLD + ro, BLUE)], [("TI", 0.0, PLD)]),
    f"CASL: LD {LD} s, PLD {PLD} s": (
        [("labeling", 0.0, LD, ORANGE), ("cut-off", 0.0, 0.0, "none"), ("readout", LD + PLD, LD + PLD + ro, BLUE)],
        [("LD", 0.0, LD), ("PLD", LD, LD + PLD)]),
    f"PCASL: LD {LD} s, PLD {PLD} s": (
        [("labeling", s, s + 0.006, ORANGE) for s in np.arange(0.0, LD, 0.02)]
        + [("cut-off", 0.0, 0.0, "none"), ("readout", LD + PLD, LD + PLD + ro, BLUE)],
        [("LD", 0.0, LD), ("PLD", LD, LD + PLD)]),
}
fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
for ax, (title, (blocks, spans)) in zip(axes, schemes.items()):
    sequence_diagram(ax, blocks, R.repetition_time, labels={"cut-off": "bolus cut-off", "readout": "readout (20 slices)"})
    ax.set_ylim(-0.6, 3.3)
    for name, t0, t1 in spans:
        ax.annotate("", xy=(t1, 2.85), xytext=(t0, 2.85), arrowprops=dict(arrowstyle="<->", color=INK["secondary"], lw=1))
        ax.text((t0 + t1) / 2, 2.95, f"{name} {t1 - t0:.1f} s", ha="center", fontsize=8, color=INK["secondary"])
    ax.set_title(title, fontsize=9, loc="left")
    ax.text(R.repetition_time, -0.5, f"TR {R.repetition_time} s", ha="right", fontsize=8, color=INK["secondary"])
for ax in axes[:-1]:
    ax.set_xlabel("")
fig.tight_layout()
```

Read each row from left to right. The pulsed scheme spends 10 ms labeling, then a train
of saturation pulses at the leading edge of the slab from 0.7 s stops the bolus, and the
slices are read from 1.8 s. The continuous schemes label for the whole first 1.8 s (CASL as
one long pulse, PCASL as a train too dense to resolve at this scale: 1800 pulses in the
orange band), wait 1.8 s more, and read the same 20 slices from 3.6 s. In all three the
last slice is read 0.76 s after the first, so each slice has its own delay, a fact that
[Chapter 6](./06-the-asl-signal.md) and [Chapter 14](../04-quantification/14-cbf-quantification.md)
return to. The rest of the 4.5 s TR is spent waiting for the tissue and the blood to
recover before the next labeling.

## See it: the bolus each scheme delivers

The kinetic model of [Chapter 5](./05-kinetic-model.md) gives the label in a pure gray
matter voxel as a function of time for each scheme. Here it is evaluated with each
scheme's own efficiency and bolus duration, with the phantom's gray matter ($f = 60$
ml/100 g/min, ATT 0.8 s, T1 1.33 s), and the readout time of each scheme marked.

```{code-cell} python
:tags: [hide-input]
t = np.linspace(0, 5, 501)
curves = {
    "PCASL": dict(label_type="PCASL", tau=LD, alpha=presets.ALPHA["PCASL"], t_read=LD + PLD, color=ORANGE, ls="-"),
    "CASL": dict(label_type="CASL", tau=LD, alpha=0.68, t_read=LD + PLD, color=PALETTE[4], ls="--"),
    "PASL": dict(label_type="PASL", tau=0.7, alpha=presets.ALPHA["PASL"], t_read=PLD, color=PALETTE[6], ls="-"),
}
fig, ax = plt.subplots(figsize=(7.5, 3.4))
at_readout = {}
for name, c in curves.items():
    dm = 100 * kinetic.delta_m(t, GM.perfusion, GM.att, GM.t1, 1.0, label_type=c["label_type"], tau=c["tau"], alpha=c["alpha"])
    at_readout[name] = 100 * float(kinetic.delta_m(c["t_read"], GM.perfusion, GM.att, GM.t1, 1.0, label_type=c["label_type"], tau=c["tau"], alpha=c["alpha"]))
    ax.plot(t, dm, color=c["color"], ls=c["ls"], label=f"{name}: τ {c['tau']} s, α {c['alpha']}")
    ax.plot(c["t_read"], at_readout[name], "o", color=c["color"], ms=7)
    print(f"{name:5}: peak {dm.max():.2f} % of M0 at t = {t[dm.argmax()]:.1f} s; at its readout (t = {c['t_read']} s) {at_readout[name]:.3f} %")
ax.axvline(GM.att, color="0.8", lw=1, zorder=0); ax.text(GM.att + 0.03, 1.2, "ATT", fontsize=8, color=INK["secondary"])
ax.set(xlabel="time from the start of labeling (s)", ylabel="ΔM in pure GM (% of M0)", xlim=(0, 5), ylim=(0, 1.3),
       title="dots: the value at each scheme's readout")
ax.legend(loc="upper right")
fig.tight_layout()
```

Nothing arrives before the transit time. The two continuous curves have the same shape,
rising while the 1.8 s bolus arrives and decaying after it has all arrived at 2.6 s, and
differ only by the ratio of their efficiencies, 0.68 / 0.85 = 0.80. The pulsed curve is
different in kind: its bolus is 0.7 s long, so it stops rising at 1.5 s, and it is read at
1.8 s, when the label has had less time to decay. Despite a bolus two and a half times
shorter, PASL delivers 0.463 % of $M_0$ at its readout against PCASL's 0.531 %, a ratio of
0.87, because it is read 1.8 s earlier. That advantage is not free: the pulsed bolus is
small and its readout early, so it is more vulnerable to a transit time longer than
expected ([Chapter 5](./05-kinetic-model.md)), and per unit of scan time the longer bolus
of PCASL wins ([Chapter 7](./07-acquisition-parameters.md)).

## See it: the same brain under the three schemes

The `label-types` dataset is the slab simulated end to end, with noise, once with each
scheme and otherwise identical. The mean of the 30 control-label differences of each run,
on the display slice and one shared window:

```{code-cell} python
:tags: [hide-input]
diff = {name: quant.subtract(run.mag(), run.context()).mean(-1) for name, run in runs.items()}
fr = runs["pcasl"].fractions()
gm, wm = fr["gm"] > 0.9, fr["wm"] > 0.9
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), layout="constrained")
for ax, (name, d) in zip(axes, diff.items()):
    p = runs[name].sidecar()
    im = show_slice(ax, d, K, f"{p['ArterialSpinLabelingType']}, α = {p['LabelingEfficiency']}", vmin=0, vmax=40)
    print(f"{name:5}: mean difference in GM {d[gm].mean():5.2f}, in WM {d[wm].mean():5.2f} image units")
fig.colorbar(im, ax=axes[2], shrink=0.8, label="control − label (image units)")
```

The three images show the same perfusion pattern, gray matter bright and white matter
faint, at three amplitudes: about 30 image units in gray matter under PCASL, 24 under
CASL and 26 under PASL, against a control image of about 6200. The noise is the same in
all three (the difference of two images with noise σ = 40 has σ ≈ 57 per pair, 10 after
30 pairs), so the scheme with the largest signal has the best difference image.

## Measure it: the amplitude ratios against the efficiencies

The ratios of the gray matter difference signals can be predicted two ways. Between PCASL
and CASL, which share the timing, the ratio should be the ratio of efficiencies. Between
PCASL and PASL the timing differs too, so the prediction needs the kinetic model, evaluated
with each run's bolus duration and efficiency from its sidecar and at each slice's own
readout time (the slice offsets of the 2D readout). The ground truth `deltam` written by
the simulator provides the exact answer to compare both against.

```{code-cell} python
:tags: [hide-input]
p0 = runs["pcasl"].sidecar()
offsets = np.asarray(protocols.slice_offsets(p0))
z_gm = np.nonzero(gm)[2]                      # the slice of every pure-GM voxel
rows, ref = {}, "pcasl"
for name, run in runs.items():
    p, ctx = run.sidecar(), run.context()
    t_slice = protocols.signal_times(p, ctx)[1] + offsets[z_gm]   # a label row's signal time, per voxel
    model = kinetic.delta_m(t_slice, GM.perfusion, GM.att, GM.t1, 1.0, label_type=protocols.label_type(p),
                            tau=protocols.bolus_duration(p), alpha=p["LabelingEfficiency"]).mean()
    truth = run.truth("deltam")[..., ctx.index("label")][gm].mean()
    rows[name] = dict(alpha=p["LabelingEfficiency"], measured=diff[name][gm].mean(), model=model, truth=truth)
print(f"{'scheme':6} {'alpha':>5} | {'measured':>8} {'alpha only':>10} {'kinetic model':>13} {'truth':>6}   (GM ratio to {ref.upper()})")
for name, r in rows.items():
    r0 = rows[ref]
    print(f"{name:6} {r['alpha']:5.2f} | {r['measured'] / r0['measured']:8.3f} {r['alpha'] / r0['alpha']:10.3f} "
          f"{r['model'] / r0['model']:13.3f} {r['truth'] / r0['truth']:6.3f}")
```

Against PCASL, the CASL images are 0.801 times as strong, the efficiency ratio predicts
0.800, the model 0.800 and the truth 0.800: with the timing held fixed, the efficiency is
the whole story, and the measurement recovers it to three decimals over 9038 gray matter
voxels. The PASL images are 0.872 times as strong. The efficiencies alone would predict
1.153, since 0.98 exceeds 0.85; that prediction is wrong because it ignores that PASL's
bolus is 0.7 s rather than 1.8 s and that it is read 1.8 s earlier. The kinetic model,
given both from the sidecar, predicts 0.872, and the truth is 0.872. The lesson is the
one [Chapter 14](../04-quantification/14-cbf-quantification.md) will build on: the
efficiency, the bolus duration and the timing all enter the signal, they are all in the
sidecar, and the quantification formula must use each scheme's own.

## What this implies for acquisition

- **Prefer PCASL** on standard hardware: the white paper's recommendation
  {cite:p}`alsop2015`, for its long, known bolus and its compatibility with body coils. Use its efficiency of 0.85 unless it
  was measured, and record it in `LabelingEfficiency`.
- **Place the labeling plane** perpendicular to straight segments of the carotid and
  vertebral arteries, away from the air spaces of the neck: off-resonance at the plane
  lowers PCASL's efficiency and does so differently on each side.
- **If PASL is used, use a cut-off** (QUIPSS II or Q2TIPS) and record
  `BolusCutOffDelayTime`; without it the bolus duration is unknown and the signal cannot be
  quantified. Keep the inversion time short enough to catch the small, early bolus and long
  enough to exceed the transit time.
- **Match the control to the label.** Any mismatch in MT, timing or gradients appears in
  the difference image at full tissue signal. For CASL this is what the amplitude-modulated
  control buys, and its efficiency of 0.68 is the price.
- **Different schemes need different formulas**: the same brain gave three difference
  images whose amplitudes differ by tens of percent for reasons the sidecar fully records.

## Further reading

The first pulsed methods, EPISTAR {cite:p}`edelman1994`, FAIR {cite:p}`kim1995,kwong1995`
and PICORE {cite:p}`wong1997`, and the
cut-offs that make them quantitative, QUIPSS II {cite:p}`wong1998` and Q2TIPS
{cite:p}`luh1999`. Continuous labeling {cite:p}`williams1992` and the adiabatic fast
passage behind it {cite:p}`dixon1986`, its magnetization transfer {cite:p}`zhang1995`, the
post-labeling delay {cite:p}`alsop1996`, and the amplitude-modulated control
{cite:p}`alsop1998` and its efficiency at 3 T {cite:p}`wang2005,wu2007`.
Pseudo-continuous labeling {cite:p}`dai2008`, its labeling efficiency in theory and
experiment {cite:p}`wu2007` and in vivo {cite:p}`aslan2010`, and its off-resonance and
velocity sensitivity {cite:p}`zhao2017`. The consensus recommendations on which scheme to
use and with what efficiency {cite:p}`alsop2015`, and the BIDS fields that record the
scheme and its parameters {cite:p}`clement2022`.
