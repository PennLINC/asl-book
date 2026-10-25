---
title: What ASL measures, and why it matters
subtitle: Blood flow, transit time, arterial blood, and what each is used for
kernelspec:
  name: python3
  display_name: Python 3
---

## Learning goals

After this page you can:

- say why the brain's blood supply is worth measuring
- define cerebral blood flow (CBF), the arterial transit time (ATT), the arterial bolus
  arrival time (aBAT) and the arterial blood volume (aBV) in one sentence each
- say what each of the four does to the signal of a voxel, and which of them one standard
  acquisition can measure
- name a use of each in research or the clinic
- explain why ASL suits tasks and comparisons that are slow, and what it gives up for that

```{code-cell} python
:tags: [hide-cell]
import numpy as np
import matplotlib.pyplot as plt

from aslbook import kinetic, presets
from aslbook.plotting import INK, PALETTE, TISSUE_COLORS, set_style

set_style()
GM, R = presets.TISSUES["GM"], presets.REFERENCE
T1B, LAM, ALPHA = presets.T1_BLOOD, presets.LAMBDA, presets.ALPHA["PCASL"]
LD, T_READ = R.labeling_duration, R.labeling_duration + R.post_labeling_delay
```

## Why blood flow

The brain is about 2 % of the body's weight and uses about 20 % of its oxygen at rest
{cite:p}`raichle2002`. It stores almost no fuel, so it depends from second to second on the
blood that brings oxygen and glucose to it: stop the supply and function fails within
seconds, and tissue dies within minutes. Blood flow is also matched to demand, locally and
quickly. When a region of cortex becomes active its blood flow rises, by tens of percent in
the first measurements of the effect, far more than its oxygen consumption does
{cite:p}`fox1986`. The rate at which blood reaches tissue is therefore a measure of two
things: whether the tissue is being supplied, and how hard it is working.

**Arterial spin labeling (ASL)** measures that rate with MRI and without an injected tracer.
It magnetically labels the water in the blood on its way to the brain, takes an image after
the labeled blood has arrived and another without the label, and subtracts them
{cite:p}`detre1992,williams1992`. What is left is the labeled water delivered to each voxel,
in an image that can be turned into blood flow in physiological units. Because nothing is
injected, the measurement can be repeated as often as needed, in children and in patients
who cannot receive a contrast agent {cite:p}`alsop2015`.

## What an ASL acquisition can measure

The difference image depends on how much labeled blood arrived, when it arrived, and where
in the voxel it was when the image was taken. Four quantities describe this, and they are
the main outputs, or *derivatives*, of an ASL analysis.

- **Cerebral blood flow (CBF)**, written $f$: the volume of arterial blood delivered to the
  capillaries of 100 g of tissue per minute, in ml/100 g/min. It is the quantity ASL exists
  to measure. Gray matter values from 40 to 100 ml/100 g/min are normal; white matter has
  about a third of the gray matter value or less {cite:p}`alsop2015`.
- **Arterial transit time (ATT)**, written $\delta$: the time the labeled blood takes from
  where it was labeled to the capillaries of a voxel, usually 0.5 to 2 s. It is also called
  the bolus arrival time (BAT). Before the ATT no label has reached the tissue, and an image
  taken then measures no perfusion however high the flow.
- **Arterial bolus arrival time (aBAT)**, written $\delta_\mathrm{a}$: the time the label
  takes to reach the *arteries* inside a voxel, which it passes through before it reaches
  the capillaries, so aBAT is shorter than ATT. The simulator writes it as the `aATT` truth
  map.
- **Arterial blood volume (aBV)**: the fraction of a voxel's volume occupied by arterial
  blood, labeled blood that is still in transit and will perfuse tissue elsewhere, on the
  order of one percent of a gray matter voxel {cite:p}`chappell2010`.

CBF and ATT describe the tissue; aBAT and aBV describe the arteries that run through it.
Other quantities are built from these. **Cerebrovascular reactivity** is the change in CBF
when the arteries are made to dilate, for example by breathing air with extra carbon
dioxide, which tests whether the vessels still have reserve. The exchange of labeled water
across the capillary wall, which measures the blood-brain barrier, and the arterial
signal itself are topics of [Chapter 19](../05-advanced/19-frontiers.md).

## See it: what each quantity does to the signal

The curves are the label in one pure gray matter voxel through one repetition of the
book's reference protocol (pseudo-continuous labeling for 1.8 s from $t = 0$, read out at
3.6 s), in percent of the tissue's equilibrium magnetization $M_0$. The tissue part is the
kinetic model of [Chapter 5](../02-labeling/05-kinetic-model.md) with the phantom's gray
matter (CBF 60 ml/100 g/min, ATT 0.8 s); the arterial part is labeled blood that enters the
voxel's arteries at the aBAT, flows on, and leaves after one labeling duration, the
plug-flow arterial term of [Chapter 19](../05-advanced/19-frontiers.md), with an aBV of
1 % arriving 0.3 s before the tissue's label. The top-left panel takes the voxel apart; the
other three change one quantity at a time.

```{code-cell} python
:tags: [hide-input]
t = np.linspace(0, 5, 1001)
tissue = lambda f=GM.perfusion, att=GM.att: 100 * kinetic.delta_m(t, f, att, GM.t1, 1.0, tau=LD)


def arterial(abv=0.01, abat=GM.att - 0.3):
    """Label inside the voxel's arteries (% of M0): present from the aBAT for one labeling duration."""
    return 100 * 2 * ALPHA * abv / LAM * np.exp(-abat / T1B) * ((t > abat) & (t <= abat + LD))


at = lambda curve, x: float(np.interp(x, t, curve))
fig, axes = plt.subplots(2, 2, figsize=(11, 6.2), sharex=True, sharey=True)
ax = axes[0, 0]
ax.plot(t, tissue(), color=TISSUE_COLORS["GM"], label="tissue: CBF 60, ATT 0.8 s")
ax.plot(t, arterial(), color=PALETTE[7], label="arteries: aBV 1 %, aBAT 0.5 s")
ax.plot(t, tissue() + arterial(), color=INK["primary"], lw=1.2, ls="--", label="the voxel")
for x, name, dx in ((GM.att - 0.3, "aBAT", -0.15), (GM.att, "ATT", 0.15)):
    ax.annotate(name, xy=(x, 0), xytext=(x + dx, 0.5), ha="center", fontsize=8, color=INK["secondary"],
                arrowprops=dict(arrowstyle="-|>", color=INK["secondary"], lw=1))
ax.set_title("one gray matter voxel, taken apart")
variants = [
    (axes[0, 1], "CBF (ml/100 g/min)", [30, 60, 90], 60, lambda v: tissue(f=v) + arterial()),
    (axes[1, 0], "ATT (s)", [0.8, 1.4, 2.0], 0.8, lambda v: tissue(att=v) + arterial(abat=v - 0.3)),
    (axes[1, 1], "aBV (%)", [0, 1, 2], 1, lambda v: tissue() + arterial(abv=v / 100)),
]
for ax, name, values, ref, curve in variants:
    others = iter((PALETTE[0], PALETTE[4]))
    for v in values:
        c, color = curve(v), INK["primary"] if v == ref else next(others)   # the reference voxel in black
        ax.plot(t, c, color=color, label=f"{name.split()[0]} {v}")
        print(f"{name:20} {v:4}: at the reference readout (t = {T_READ} s) {at(c, T_READ):.3f} %, "
              f"after a delay of 0.3 s (t = {LD + 0.3} s) {at(c, LD + 0.3):.3f} % of M0")
    ax.set_title(f"changing {name}")
for ax in axes.flat:
    ax.axvline(T_READ, color="0.75", lw=1, zorder=0)
    ax.text(T_READ + 0.04, 2.2, "reference\nreadout", fontsize=7, color=INK["secondary"], va="top")
    ax.legend(loc="upper right", fontsize=7)
    ax.set(xlim=(0, 5), ylim=(0, 4))
for ax in axes[1]:
    ax.set_xlabel("time from the start of labeling (s)")
for ax in axes[:, 0]:
    ax.set_ylabel("ΔM (% of M0)")
fig.tight_layout()
```

Top left: nothing happens until the label reaches the voxel's arteries at the aBAT. The
arterial label is large while it lasts, larger than the tissue's even at 1 % of the voxel,
because the label in the arteries is all still there: it has not been diluted into the
tissue water or begun to relax with the tissue's shorter T1. It leaves as fast as it came.
The tissue label starts at the ATT, rises while the bolus arrives, peaks when the last of
it has arrived, and decays.

Top right: CBF scales the tissue curve. At the reference readout the voxel holds 0.268,
0.531 and 0.789 % of $M_0$ for 30, 60 and 90 ml/100 g/min, nearly in proportion to the
flow (not exactly, because faster flow also washes the label out of the tissue faster),
which is what lets one image at one delay be turned into CBF
([Chapter 14](../04-quantification/14-cbf-quantification.md)).

Bottom left: ATT moves the curve later. At the reference readout it matters little as long
as the whole bolus has arrived: 0.531 % at 0.8 s and 0.583 % at 1.4 s, slightly more,
because the later label spent more of its life in blood, whose T1 is longer than the
tissue's. At 2.0 s the end of the bolus has not yet arrived (it does at 3.8 s) and the
signal is 0.519 %. This insensitivity is the purpose of a long post-labeling delay
{cite:p}`alsop1996`. Early in the curve the opposite is true: after a delay of 0.3 s the
same voxel holds 2.354, 1.408 and 0.728 % of $M_0$ for the three transit times. A
single-delay image says nothing about the ATT, and an image taken before the label has
arrived reads as low flow; to measure both the curve has to be sampled at several delays
([Chapter 15](../04-quantification/15-multi-delay.md)).

Bottom right: aBV changes nothing at the reference readout (0.531 % for all three values),
because by then the arterial label has passed through. After a delay of 0.3 s it dominates:
the voxel holds 0.959 % of $M_0$ with no arterial blood and 3.749 % with 2 %. The arteries
are invisible to the standard protocol, which is by design, and they are the largest signal
at short delays, which a multi-delay acquisition must either model or remove
([Chapter 19](../05-advanced/19-frontiers.md)).

## Why each one matters

**CBF** is the clinical and scientific quantity. Lower flow marks tissue that is not being
supplied, as in acute stroke {cite:p}`chalela2000`, or that is doing less, as in the
reduced flow of the posterior cingulate, precuneus and parietal cortex in Alzheimer's
disease, a pattern close to that of reduced glucose metabolism {cite:p}`wolk2012`. Higher
flow marks tissue that is working harder or has grown new vessels: high-grade tumors and
seizure foci during a seizure are hyperperfused {cite:p}`haller2016`. Gray matter flow
declines with age and differs between people of the same age by up to a factor of two
{cite:p}`parkes2004`. Because ASL measures CBF in ml/100 g/min, a value can be compared
across sessions, scanners and people, and it agrees with the reference standard, positron
emission tomography with labeled water, within the precision of either
{cite:p}`heijtel2014`.

**ATT** is first a source of error: the delay before the image has to be longer than the
longest transit time in the brain, or the CBF there is underestimated
([Chapter 14](../04-quantification/14-cbf-quantification.md)), which is why the consensus
delays depend on the age of the population {cite:p}`alsop2015`. It is also a measurement in
its own right. Transit times are long where blood reaches tissue by an indirect route: in
the watershed regions between arterial territories, and in narrowed or blocked arteries
whose territory is fed by collateral vessels. In moyamoya disease, the bright arterial
signal that a long transit time leaves in the image is a sign of collateral flow
{cite:p}`zaharchuk2011`; a map of ATT measures the delay directly {cite:p}`woods2024`.

**aBAT and aBV** describe the arteries. In most analyses they are what must be removed so
that the tissue's CBF and ATT are not biased by label that is only passing through
{cite:p}`chappell2010`. The arterial signal is also information: how much arterial blood a
region holds and when its label arrives describe the arterial tree that feeds it, and
labeling methods that separate the territories of individual arteries build on the same
signal {cite:p}`vanosch2018`.

## Task-based ASL

ASL can follow brain activity as well as map the brain at rest. Every control-label pair
is an estimate of CBF, so a series of pairs is a CBF time course, and a task that raises
the flow in a region raises its difference signal. This is **perfusion fMRI**, the ASL
counterpart of the blood-oxygenation-level-dependent (BOLD) signal that most functional MRI
uses {cite:p}`detre2002`.

The two have opposite strengths. BOLD has more signal, an image every second or two, and
much better sensitivity for tasks that alternate quickly. But a BOLD image has no
meaningful baseline, and the scanner's slow drifts are as large as the effects it measures,
so slow comparisons are hard. ASL subtracts images taken seconds apart, and anything the
two share, the drift included, cancels. Its noise is therefore nearly the same at every
time scale, and it is the more sensitive of the two for tasks whose blocks last minutes
{cite:p}`aguirre2002`; it detects activation with blocks of several minutes, and
differences between sessions on different days {cite:p}`wang2003lowfreq`. The price is a
smaller signal and one CBF estimate per pair, every 9 s with the reference protocol. ASL
is the natural choice when the state of interest changes slowly or is sustained: a drug,
pain, sleep deprivation, or the difference between two scanning sessions
{cite:p}`detre2002`.

## See it: a slow task, measured as flow

The toy below is a gray matter region of 200 voxels scanned for 24 minutes with the
reference protocol, in blocks of 4 minutes of rest and 4 of a task that raises its CBF from
60 to 75 ml/100 g/min. The scanner's gain drifts up by 1.5 % over the run, with a slower
oscillation of 0.5 % on top. The noise is that of the reference series averaged over the
region; there is no physiological noise, and the drift is the same in the control and the
label images.

```{code-cell} python
:tags: [hide-input]
rng = np.random.default_rng(0)
PAIR = 2 * R.repetition_time                                # one control-label pair every 9 s
tp = np.arange(0, 24 * 60, PAIR)
task = ((tp // 240) % 2 == 1)                                # 4 min rest, 4 min task, three cycles
S0, DM0, SIGMA = 6200.0, 30.0, 40.0 / np.sqrt(200)          # GM control and ΔM (image units); noise of a 200-voxel mean
drift = 1 + 0.015 * tp / tp[-1] + 0.005 * np.sin(2 * np.pi * tp / 600)
dm = DM0 * np.where(task, 75 / 60, 1.0)
control = S0 * drift + rng.normal(0, SIGMA, tp.size)
label = (S0 - dm) * drift + rng.normal(0, SIGMA, tp.size)
cbf = GM.perfusion * (control - label) / DM0                # 30 image units of ΔM per 60 ml/100 g/min

fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 5.4), sharex=True)
for ax in (ax0, ax1):
    for start in np.arange(240, tp[-1], 480):
        ax.axvspan(start / 60, (start + 240) / 60, color=PALETTE[3], alpha=0.15, lw=0)
ax0.plot(tp / 60, control, color=INK["secondary"], lw=1, label="control")
ax0.plot(tp / 60, label, color=PALETTE[6], lw=1, label="label")
ax0.set(ylabel="region mean (image units)", title="the images: shaded blocks are the task")
ax0.legend(loc="upper left")
ax1.plot(tp / 60, cbf, ".", color=INK["secondary"], ms=3, label="one pair")
blocks = tp // 240
means = np.array([cbf[blocks == b].mean() for b in np.unique(blocks)])
edges = 4 * np.arange(means.size + 1)                       # block edges in minutes
ax1.step(edges, np.r_[means, means[-1]], where="post", color=TISSUE_COLORS["GM"], label="block mean")
ax1.set(xlabel="time (min)", ylabel="CBF (ml/100 g/min)", title="control − label, as flow")
ax1.legend(loc="upper left")
fig.tight_layout()
print(f"drift of the control image over the run: {np.ptp(S0 * drift):.0f} image units; change of the label image with the task: {DM0 * 15 / 60:.1f}")
print(f"label image, task blocks minus rest blocks: {label[task].mean() - label[~task].mean():+.1f} image units (the task alone: {-DM0 * 15 / 60:+.1f})")
se = cbf[~task].std(ddof=1) / np.sqrt((~task).sum())
print(f"CBF from the pairs: rest {cbf[~task].mean():.1f}, task {cbf[task].mean():.1f} ml/100 g/min (truth 60 and 75); "
      f"standard error of a condition's mean {se:.1f}")
```

In the images (top) the drift is everything: the control image climbs and dips over a range
of 122 image units, while the task changes the label image by 7.5. Comparing the label
image in the task blocks with the rest blocks gives +15.8 image units, twice the size of
the effect and of the wrong sign, because the task blocks come later in the run, when the
gain is higher. The differences (bottom) do not drift: the pairs scatter around 60 in the
rest blocks and 75 in the task blocks, and their means are 61.6 and 75.9 ml/100 g/min, with
a standard error of 1.0 each: the task's 15 ml/100 g/min is measured, and the drift, which
scales the difference by less than 1.5 %, barely touches it. A real series adds physiological noise and imperfect subtraction
([Chapter 8](../03-preprocessing/08-noise.md), [Chapter 10](../03-preprocessing/10-motion.md)),
but the principle holds: what ASL compares is flow, and flow has a baseline.

## What this book covers

The book's simulated brain is at rest and simple: one CBF and one ATT per tissue
([Chapter 0.2](./the-simulated-datasets.md)), so that every estimate can be scored against
the truth. Most of the book is about getting CBF right: how the label is made
([Chapter 4](../02-labeling/04-labeling-schemes.md)), how the signal depends on CBF and
ATT ([Chapter 5](../02-labeling/05-kinetic-model.md)), what corrupts it and how to correct it
(Part III), and how to turn it into CBF with one delay
([Chapter 14](../04-quantification/14-cbf-quantification.md)) and into CBF and ATT with
several ([Chapter 15](../04-quantification/15-multi-delay.md)). The arterial compartment,
with its aBV and aBAT, is simulated and measured in
[Chapter 19](../05-advanced/19-frontiers.md). Task-based ASL is not simulated: everything
the book shows for one CBF map applies to each pair of a perfusion fMRI series, and the
analysis of the time series itself is the subject of the fMRI literature
{cite:p}`detre2002`.

## Further reading

The brain's energy budget {cite:p}`raichle2002` and the coupling of flow to activity
{cite:p}`fox1986`. The first ASL measurements {cite:p}`detre1992,williams1992`, the
consensus recommendations for clinical use {cite:p}`alsop2015`, and the recommendations
for multi-delay ASL, which measures ATT as well as CBF, with its clinical applications
{cite:p}`woods2024`. Clinical applications of ASL {cite:p}`haller2016`, its use in
neurodegenerative disease {cite:p}`wolk2012`, and collateral flow in moyamoya disease
{cite:p}`zaharchuk2011`. The arterial signal and how to separate it {cite:p}`chappell2010`,
and vessel-selective labeling {cite:p}`vanosch2018`. Perfusion fMRI and BOLD
{cite:p}`detre2002`, their sensitivity at low task frequencies {cite:p}`aguirre2002`, and
ASL with very low task frequency {cite:p}`wang2003lowfreq`.
