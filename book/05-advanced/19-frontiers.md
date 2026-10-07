---
title: "19. Frontiers"
kernelspec:
  name: python3
  display_name: Python 3
---

:::{admonition} Simulated datasets in this chapter
:class: note
- **Built in this page:** single-voxel signals from the kinetic model of [Chapter 5](../02-labeling/05-kinetic-model.md), extended in the page with a second compartment, a dispersion kernel and an arterial term ([Appendix B](../appendices/b-data-manifest.md#app-b-package-data)).
- **`multi-te`**: PCASL with a water exchange time of 0.5 s, three echoes of every excitation: a spin-echo readout at 24, 48 and 72 ms with PLD 1.0 s (`se-pld10`), and a gradient-echo readout at 60 degrees at 12, 33 and 54 ms with PLD 1.0 s (`ge-pld10`) and 1.8 s (`ge-pld18`) ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-multi-te)).
- **`vascular`**: the reference labeling at seven delays from 0.1 to 1.8 s, read with the four-shot 3D GRASE, without an arterial compartment (`none`), with one (`uncrushed`: 2 % of a gray matter voxel, 0.8 % of white matter, arriving 0.3 s before the tissue's label) and with it under vascular crushing at VENC 4 cm/s (`crushed`) ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-vascular)).

Three of this chapter's methods are now simulated by aslscan: multi-echo readouts with exchange, the arterial compartment with crushing (both here), and the 3D readouts ([Chapter 2](../01-mri-physics/02-epi-and-reconstruction.md)). Pipeline-tier datasets are simulated offline by aslscan ([Chapter 0.2](../00-frontmatter/the-simulated-datasets.md)); the last section says what adding the rest would take.
:::

## Learning goals

After this chapter you can:

- describe what velocity-selective labeling, vessel encoding, multi-echo readouts, 3D
  readouts, dispersion and macrovascular models, and learned processing each add to the
  chain this book has followed
- explain, with a simulation, why a velocity-selective label is insensitive to transit
  time and why the echo-time decay of ΔM reads the exchange of label into tissue
- measure the fraction of the label still in blood from three echoes of the `multi-te`
  runs, and say why it falls between a delay of 1.0 and 1.8 s
- state the bias that dispersion and arterial signal produce in a standard multi-delay
  fit, measure it on the `vascular` runs, and say what vascular crushing removes and what
  it leaves
- say what aslscan does not model and how each omission would be added

```{code-cell} python
:tags: [hide-cell]
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.stats import gamma as gamma_dist

from aslbook import data, kinetic, phantom, presets, quant
from aslbook.plotting import INK, PALETTE, TISSUE_COLORS, set_style, show_slice

set_style()
GM, WM = presets.TISSUES["GM"], presets.TISSUES["WM"]
K = phantom.DISPLAY_SLICE
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
most 0.5, which velocity-selective *inversion* modules developed later recover
{cite:p}`qin2016`.

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
state {cite:p}`wong2007`. Cycling through a few such patterns, left-right and anterior-posterior with
different phases, gives each artery a distinct label-control signature across the cycles,
and a linear decoding of the same kind as
[Chapter 18](./18-time-encoded-and-look-locker.md)
assigns every voxel's perfusion to the artery that delivered it: a *territory map* of the
carotid and vertebral supplies {cite:p}`vanosch2018`. Because every cycle labels about half
of the arteries, the total perfusion image costs no more scan time than a conventional
PCASL scan of the same length. Super-selective variants rotate the gradient during the
train so that only one vessel of choice is labeled {cite:p}`helle2010`. The maps are used to see collateral
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
{cite:p}`stlawrence2000,zhou2001,parkes2002`. The echo time can tell the two apart. Label in blood decays
with the blood's $T_2$, 165 ms in this book's simulations (the measured value depends on
oxygenation and hematocrit {cite:p}`zhao2007`); label that has entered gray matter decays
with the tissue's, 80 ms in the phantom. The difference signal at several echo times is then a two-component decay
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
reference echo time of 12 ms the error is less than half of that.

The simulator's exchange model, switched on in the `multi-te` dataset, is simpler than the
toy's two compartments. It is a single pass: label that arrived in the voxel at time $t'$
has crossed into the tissue by time $t$ with probability $1 - e^{-(t - t')/T_\mathrm{ex}}$,
irreversibly, and the kinetic model's $T_1'$ applies to both parts, so the total `deltam`
is exactly the curve of [Chapter 5](../02-labeling/05-kinetic-model.md) and only the
transverse decay is split: the part still in the capillary (`deltamIntravascular`) decays
with the blood's $T_2$ and the exchanged remainder with the tissue's. The toy's compartments
also differ in $T_1$, the blood's 1.65 s against $T_1'$, which is why its total was close
to, but not exactly, the instant-exchange curve. Three runs with an exchange time of 0.5 s
read three echoes of every excitation: a spin-echo readout at 24, 48 and 72 ms with a
delay of 1.0 s, and a gradient-echo readout at 60 degrees at 12, 33 and 54 ms with a delay
of 1.0 s and of 1.8 s. The cell takes the mean difference in pure gray and pure white
matter (fraction at least 0.95) at each echo and compares it with the truth's two parts
decayed to the echo time. In the gradient-echo runs the decays are the $T_2^*$ ones; the
blood compartment has no $T_2^*$ of its own in the phantom, so the simulator gives it the
blood's $T_2$ and the tissue's $T_2'$, and the label carries the excitation's
$\sin 60^\circ$, which cancels in every ratio. The cell then fits the three measured points
with the two known exponentials and reads the intravascular fraction from the two
amplitudes.

```{code-cell} python
:tags: [hide-input]
RUN_TITLES = {"se-pld10": "spin echo, PLD 1.0 s", "ge-pld10": "gradient echo, PLD 1.0 s", "ge-pld18": "gradient echo, PLD 1.8 s"}
mte = data.load_dataset("multi-te")
echoes = {}
for name, title in RUN_TITLES.items():
    run = mte.run(name)
    p = run.sidecar(echo=1)
    sim = p["AslscanSimulation"]
    tes = np.array(sim["MultiEcho"]["EchoTimes"])
    ge = sim["MultiEcho"]["EchoFormation"] == "gradient"
    label_gain = np.sin(np.deg2rad(p["FlipAngle"])) if ge else 1.0      # a gradient echo reads sin(a) ΔM
    ctx, fr = run.context(), run.fractions()
    lab = [i for i, r in enumerate(ctx) if r == "label"]
    total = run.truth("deltam")[..., lab].mean(-1)
    iv = run.truth("deltamIntravascular")[..., lab].mean(-1)
    meas = np.stack([quant.subtract(run.mag(echo=e), ctx).mean(-1) for e in run.echoes()], -1)
    print(f"{name} ({title}): TE {np.round(1e3 * tes).astype(int)} ms, exchange time {sim['Exchange']['ExchangeTime']} s, "
          f"model '{sim['Exchange']['Model']}'")
    for tname, tis in (("GM", GM), ("WM", WM)):
        roi = fr[tname.lower()] >= 0.95
        t2t = tis.t2star if ge else tis.t2
        t2b = 1 / (1 / T2_B + 1 / tis.t2prime) if ge else T2_B
        e_b, e_t = np.exp(-tes / t2b), np.exp(-tes / t2t)
        m, sem = meas[roi].mean(0), meas[roi].std(0) / np.sqrt(roi.sum())
        f_true = iv[roi].mean() / total[roi].mean()
        dm0 = SCALE * label_gain * total[roi].mean()                      # ΔM(TE = 0) in image units
        a_b, a_t = np.linalg.lstsq(np.stack([e_b, e_t], 1), m, rcond=None)[0]
        echoes[name, tname] = (tes, m, sem, f_true, t2b, t2t, dm0)
        print(f"  {tname} (decay constants: blood {1e3 * t2b:.0f} ms, tissue {1e3 * t2t:.0f} ms): measured {np.round(m, 1)} ± {sem.mean():.1f}, "
              f"predicted {np.round(dm0 * (f_true * e_b + (1 - f_true) * e_t), 1)}; fraction in blood fitted {a_b / (a_b + a_t):.3f}, truth {f_true:.3f}")
```

```{code-cell} python
:tags: [hide-input]
te_axis = np.linspace(0, 0.08, 161)
fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), sharey=True)
for ax, (name, title) in zip(axes, RUN_TITLES.items()):
    tes, m, sem, f_true, t2b, t2t, dm0 = echoes[name, "GM"]
    ax.plot(1e3 * te_axis, np.exp(-te_axis / t2b), color=PALETTE[7], ls="--", lw=1, label=f"all in blood ({1e3 * t2b:.0f} ms)")
    ax.plot(1e3 * te_axis, np.exp(-te_axis / t2t), color=TISSUE_COLORS["GM"], ls="--", lw=1, label=f"all in tissue ({1e3 * t2t:.0f} ms)")
    ax.plot(1e3 * te_axis, f_true * np.exp(-te_axis / t2b) + (1 - f_true) * np.exp(-te_axis / t2t), color=INK["secondary"], lw=1.2,
            label=f"truth: {100 * f_true:.1f} % in blood")
    ax.errorbar(1e3 * tes, m / dm0, yerr=sem / dm0, fmt="o", color=PALETTE[0], ms=5, capsize=3, label="measured, pure GM")
    ax.set(xlabel="echo time (ms)", yscale="log", title=f"{name}: {title}")
    ax.legend(fontsize=7)
axes[0].set_ylabel("ΔM(TE) / ΔM(0)")
fig.tight_layout()
```

The measured gray-matter points fall on the curve the truth predicts, between the two
limiting decays, and the two-exponential fit returns the fraction of the label still in
blood: 0.124 against a truth of 0.125 in the spin-echo run at a delay of 1.0 s, 0.125
against 0.125 in the gradient-echo run at the same delay, and 0.026 against 0.025 at 1.8 s.
An exchange time of 0.5 s lies in the range the toy swept, and the two delays show why
the readout time decides whether it can be measured: at 1.0 s the first slices read
2.8 s after the start of labeling, when the bolus has only just finished arriving and
an eighth of the label is fresh enough to be still in the capillary; at 1.8 s almost all of
it has exchanged, and a fraction of 2.5 % is at the edge of what three echoes resolve. In
the gradient-echo runs the blood compartment's effective constant is 115 ms against the
tissue's 66 ms (printed), a smaller contrast than the spin echo's 165 against 80 ms, and
the fit still succeeds because the pure gray matter signal is large. White matter is the
opposite case. Its bolus, with a transit time of 1.2 s, is still arriving when the first
slices read, so a larger share of its label is intravascular, 0.179 at 1.0 s and 0.037 at
1.8 s by the truth; but the pure white matter signal is a sixth of gray matter's, its two
decay constants are closer (110 against 165 ms in the spin echo, 53 against 63 ms in the
gradient echo), and the fitted fractions, printed negative, are meaningless. A multi-echo
measurement of exchange is a gray matter measurement at short delays. The measured points
sit about 2 % above the prediction in gray matter and a few percent below it in white
matter at every echo, an offset the reference series shares and that comes from the
acquisition grid at the cortical ribbon ([Chapter 2](../01-mri-physics/02-epi-and-reconstruction.md)),
not from the exchange model; being the same at every echo, it cancels in the fit.

```{code-cell} python
:tags: [hide-input]
for name, title in RUN_TITLES.items():
    tes, m, sem, f_true, t2b, t2t, dm0 = echoes[name, "GM"]
    true_decay = f_true * np.exp(-tes / t2b) + (1 - f_true) * np.exp(-tes / t2t)
    blood_only = np.exp(-tes / t2b)
    print(f"{title}, GM: ΔM(TE)/ΔM(TE₁) measured {np.round(m / m[0], 3)}, blood decay alone {np.round(blood_only / blood_only[0], 3)}; "
          "a calibration that gives all of the label the blood's constant underestimates ΔM(0), and CBF, by "
          + ", ".join(f"{100 * (1 - true_decay[i] / blood_only[i]):.0f} % at {1e3 * te:.0f} ms" for i, te in enumerate(tes)))
```

This is the calibration message for [Chapter 16](../04-quantification/16-calibration.md),
now measured. The single-echo correction divides the difference by $e^{-\mathrm{TE}/T_{2\mathrm{b}}}$,
as if the whole label still had the blood's $T_2$; here the label has mostly exchanged and
decays faster, so the measured ratio between the first and the last echo is 0.577 in the
spin-echo run where the blood's decay alone would give 0.748. Divided by the blood factor,
the difference comes out too small, and CBF with it: by 13 % at 24 ms and 32 % at 72 ms in
the spin-echo run, and by 7 % at 12 ms and 29 % at 54 ms in the gradient-echo run at the
reference delay. The error is the price of a long echo time, and the shortest echo the
readout allows keeps it below a tenth, which is the white paper's recommendation for a
different reason. The transfer of water from blood to tissue is also a quantity of
interest in itself: it is what the multi-TE measurement of {cite:t}`gregori2013` set out to
map, and what {cite:t}`ohene2021` used to detect a higher permeability of the blood-brain
barrier to water in the aging brain.

## 3D readouts

The white paper recommends a segmented 3D readout with background suppression
{cite:p}`alsop2015`: 3D GRASE {cite:p}`gunther2005,fernandezseara2005`, which excites the
whole slab and follows it with a train of refocusing pulses, each spin echo carrying an EPI
readout of one through-plane partition of k-space ($k_z$), or a stack of spirals, which
does the same with a spiral trajectory in each partition. Both are simulated by aslscan,
and [Chapter 2](../01-mri-physics/02-epi-and-reconstruction.md) measures them on the
`readout-3d` dataset: every partition is read at the same time after the label, which
removes the slice timing of [Chapter 6](../02-labeling/06-the-asl-signal.md) and lets one
pair of suppression pulses null the whole volume; the echo train's amplitudes, which the
simulator computes from an extended phase graph and writes to the sidecar, form the $k_z$
window whose inverse transform is the through-plane point-spread function, and Chapter 2
measures the blur that window produces on the phantom; segmenting the train across several
shots shortens it and narrows the point-spread function, at the cost of one labeling per
shot and of motion between the shots; and the volume's noise is $1/\sqrt{n_z}$ of a 2D
acquisition's at the same noise variance, because every echo reads the whole slab, which is
the SNR advantage the comparison of readouts measured in vivo {cite:p}`vidorreta2013`. This
chapter therefore keeps no 3D toy of its own; the `vascular` runs below use the four-shot
GRASE so that every voxel shares each delay.

What the 3D model still leaves out is listed in each run's sidecar under
`Readout.Approximations`. The slab is ideal, exactly the field of view in z with a uniform
profile and no $k_z$ aliasing, where a real slab has an edge profile and is oversampled in z
to keep its wrap out of the image. The train is a spin-echo train, so there is no 3D
gradient-echo readout, which a multi-echo or Look-Locker acquisition with a 3D readout
would need. The train leaves $M_z = 0$ at the excitation and its longitudinal state is not
carried to the next repetition. And the coils are uniform in z. These return as rows of the
table at the end of the chapter.

## Bolus dispersion and the macrovascular signal

The kinetic model's bolus arrives as a sharp-edged box: nothing before the transit time,
the full concentration after it. Real labeled blood spreads along its path, because the
velocities across a vessel differ and the paths through the arterial tree differ, so the
edges of the bolus are smeared by the time they reach the tissue. And before the label
reaches the capillaries it passes through arteries within the voxel, where it contributes
a difference signal that has nothing to do with perfusion into that voxel. Both effects
are largest at short delays, which is where a multi-delay or time-encoded acquisition
samples. A common description adds two things to the model: a
dispersion kernel {cite:p}`hrabe2004,chappell2013`, here a gamma-variate, convolved with
the delivery, and an arterial compartment {cite:p}`chappell2010` with a blood volume fraction $a\mathrm{BV}$, whose label arrives at an earlier
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

The `vascular` dataset switches the arterial compartment on in the simulator. It is the
toy's arterial term with no dispersion: the label in a voxel's arteries is a plug-flow
compartment, $2\alpha\,(M_0/\lambda)\,a\mathrm{BV}\,e^{-\delta_\mathrm{a}/T_{1\mathrm{b}}}$
while the bolus passes the voxel's arteries, $\delta_\mathrm{a} \le t < \delta_\mathrm{a} + \mathrm{LD}$,
and zero otherwise, so that it is present while the delay is shorter than the arterial
arrival and vanishes abruptly after it, with $a\mathrm{BV}$ 2 % of a gray matter voxel and
0.8 % of a white matter one, and $\delta_\mathrm{a}$ 0.5 and 0.9 s, 0.3 s before the
tissue's transit time {cite:p}`chappell2010`. The reference labeling is read at seven
delays from 0.1 to 1.8 s, four pairs each, with the four-shot 3D GRASE of
[Chapter 2](../01-mri-physics/02-epi-and-reconstruction.md) so that every voxel shares
each delay, three times: without the compartment (`none`), with it (`uncrushed`), and with
it under vascular crushing at VENC 4 cm/s (`crushed`), whose surviving fraction of the
arterial signal the simulator computes from a laminar velocity distribution and writes to
the sidecar. The cell takes the mean difference in pure gray and white matter at each delay
and compares the gray matter values with the truths, `deltam` plus `deltamArterial` times
the survival, scaled by 100 times the label's first-echo amplitude as in Chapter 5 and by
the ratio measured in the `none` run, which carries the through-plane point-spread function
of the 3D readout.

```{code-cell} python
:tags: [hide-input]
vasc = data.load_dataset("vascular")
RUN_COLORS = {"none": INK["secondary"], "uncrushed": PALETTE[7], "crushed": PALETTE[0]}
curves, at_03, fit_inputs = {}, {}, {}
for name in RUN_COLORS:
    run = vasc.run(name)
    p, ctx, sim, fr = run.sidecar(), run.context(), run.simulation(), run.fractions()
    diff = quant.subtract(run.mag(), ctx)
    pld_pair = quant.pld_of_pairs(p, ctx)
    plds = np.unique(pld_pair)
    by_delay = np.stack([diff[..., pld_pair == v].mean(-1) for v in plds], -1)
    lab = [i for i, r in enumerate(ctx) if r == "label"]
    dm = run.truth("deltam")[..., lab]
    art = run.truth("deltamArterial")[..., lab] if name != "none" else np.zeros_like(dm)
    survival = sim["VascularCrushing"]["Survival"][0] if p["VascularCrushing"] else [1.0, 1.0]   # per tissue, GM then WM
    amp = sim["EchoAmplitudes"]["PerEcho"]["blood"][0]                                            # the label at the first echo
    for j, tname in enumerate(("GM", "WM")):
        roi = fr[tname.lower()] >= 0.95
        truth = np.stack([(dm + survival[j] * art)[..., pld_pair == v].mean(-1) for v in plds], -1)
        curves[name, tname] = (plds, by_delay[roi].mean(0), SCALE * amp * truth[roi].mean(0))
    at_03[name] = np.where(run.mask(), by_delay[..., plds == 0.3][..., 0], np.nan)
    m0 = quant.m0_correction(run.m0scan(), tr=run.m0scan_sidecar()["RepetitionTimePreparation"], te=p["EchoTime"])
    fit_inputs[name] = (by_delay, plds, m0, fr, run.truth("perfusion"), run.truth("att"), p["LabelingDuration"])
readout_factor = (curves["none", "GM"][1] / curves["none", "GM"][2]).mean()
print(f"delays {plds} s; label amplitude at the first echo {amp:.3f}; measured / predicted ΔM in pure GM of the none run {readout_factor:.3f} "
      f"(the 3D readout's through-plane point-spread function); survival of the arterial signal under crushing: GM {survival[0]:.3f}, WM {survival[1]:.3f}")
for name in RUN_COLORS:
    plds, m, t = curves[name, "GM"]
    print(f"{name:9s} GM ΔM per delay: measured {np.round(m, 1)}, predicted {np.round(readout_factor * t, 1)}; arterial truth at PLD 0.3 s "
          f"{(t[2] / (SCALE * amp) - curves['none', 'GM'][2][2] / (SCALE * amp)):.2f} M0 units against tissue {curves['none', 'GM'][2][2] / (SCALE * amp):.2f}")
```

```{code-cell} python
:tags: [hide-input]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.3))
for ax, tname in zip(axes, ("GM", "WM")):
    for name, color in RUN_COLORS.items():
        plds, m, t = curves[name, tname]
        ax.plot(plds, m, "o", color=color, label=f"{name}: measured")
        if tname == "GM":
            ax.plot(plds, readout_factor * t, "-", color=color, lw=1.2, alpha=0.8, label=f"{name}: truth × {readout_factor:.2f}")
        else:
            ax.plot(plds, m, "-", color=color, lw=1.2, alpha=0.8)
    ax.set(xlabel="post-labeling delay (s)", ylabel="mean ΔM (image units)", title=f"pure {tname}: the arterial signal and its crushing")
    ax.legend(fontsize=7)
fig.tight_layout()
```

Left: pure gray matter. Without the compartment (gray) the difference follows the kinetic
curve, scaled by the 0.92 the `none` run measures against its truth, which is the
through-plane point-spread function of the 3D readout mixing the cortex with its neighbors
([Chapter 2](../01-mri-physics/02-epi-and-reconstruction.md)). With it (red), the four
delays below the arterial arrival of 0.5 s carry 236 to 245 image units where the tissue
alone gives 56 to 64: the arterial compartment is 2.07 M0 units against the tissue's 0.71,
three times the perfusion signal, from 2 % of the voxel's water, as the toy's peaks showed
for 1 %. At 0.5 s it is gone, abruptly, because the simulator's bolus has no dispersion;
the toy's dispersed arterial curve had a tail here. Crushing at VENC 4 cm/s (blue) leaves
0.198 of it, the laminar average the sidecar states, so the crushed run still carries
92 to 100 image units at the short delays, and from 0.5 s the three runs coincide. The
lines are the truths scaled by the `none` run's factor, and they pass through the measured
points of all three runs: the compartment and its crushing are what the sidecar says they
are. Right: pure white matter, measured only. Its arterial arrival is 0.9 s, so its
arterial signal persists through the 0.5 s delay and is gone at 1.0 s, and its crushing
leaves 0.341, more than gray matter's, because the simulator's white matter arteries are
slower (6 cm/s against 10). The white matter truths are not drawn: the point-spread
function mixes the neighboring gray matter's arterial signal into these voxels, and a
single factor does not describe it.

```{code-cell} python
:tags: [hide-input]
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
for ax, name in zip(axes, RUN_COLORS):
    im = show_slice(ax, at_03[name], K, f"{name}: mean ΔM at PLD 0.3 s", kind="magnitude", vmin=0, vmax=250)
fig.subplots_adjust(left=0.01, right=0.9, top=0.92, bottom=0.02, wspace=0.05)
fig.colorbar(im, ax=axes.tolist(), shrink=0.8, label="image units")
```

The same at PLD 0.3 s on the display slice, one window. The arterial signal is in every
gray matter voxel, because the phantom gives every gray matter voxel the same arterial
blood volume; that is unlike a real brain, where the arterial blood volume is concentrated
in the voxels that contain the larger arteries, the Sylvian fissures and the
interhemispheric and basal cisterns, and the uncrushed short-delay image shows bright
vessels on a dimmer cortex rather than a uniformly brighter cortex. The pattern the
phantom cannot show is the one that makes the arterial signal recognizable by eye; its
amount per voxel is what the fit sees, and that is what the next cell tests: the standard
multi-delay fit of [Chapter 15](../04-quantification/15-multi-delay.md), with the labeling
duration and M0 from the separate scan, to all seven delays of each run, scored in pure
gray matter against the truth, and once more to the delays from 0.5 s on, which is the
"avoid it" remedy.

```{code-cell} python
:tags: [hide-input]
fits = {}
for name in RUN_COLORS:
    by_delay, plds, m0, fr, truth_cbf, truth_att, tau = fit_inputs[name]
    perfused = fr["gm"] + fr["wm"]
    pmask = perfused > 0.5
    t1_map = np.divide(fr["gm"] * GM.t1 + fr["wm"] * WM.t1, perfused, out=np.zeros_like(perfused), where=perfused > 0)
    cbf, att = quant.fit_multi_pld(by_delay, plds, m0, tau=tau, t1_tissue=t1_map, mask=pmask)
    fits[name] = (cbf, att, pmask)
    gm = fr["gm"] >= 0.95
    late = plds >= 0.5
    cbf_late, att_late = quant.fit_multi_pld(by_delay[..., late], plds[late], m0, tau=tau, t1_tissue=t1_map, mask=gm)
    sc, sa = quant.score(cbf, truth_cbf, gm), quant.score(att, truth_att, gm)
    print(f"{name:9s}: pure GM, all seven delays: CBF {cbf[gm].mean():.1f} ml/100 g/min (truth {truth_cbf[gm].mean():.1f}, bias {sc['bias']:+.1f}), "
          f"ATT {att[gm].mean():.2f} s (truth {truth_att[gm].mean():.2f}, bias {sa['bias']:+.2f}; at the grid's lower edge of 0.2 s in {100 * (att[gm] <= 0.2001).mean():.0f} % of voxels); "
          f"delays from 0.5 s only: CBF {cbf_late[gm].mean():.1f}, ATT {att_late[gm].mean():.2f} s")
fig, axes = plt.subplots(2, 3, figsize=(11, 6.2))
for j, name in enumerate(RUN_COLORS):
    cbf, att, pmask = fits[name]
    show_slice(axes[0, j], np.where(pmask, cbf, np.nan), K, f"{name}: fitted CBF", kind="cbf")
    show_slice(axes[1, j], np.where(pmask, att, np.nan), K, f"{name}: fitted ATT", kind="att")
fig.tight_layout()
```

The fits are the toy's bias, measured. Without the compartment the standard model
recovers 60.5 ml/100 g/min and 0.79 s in pure gray matter, the baseline of this readout
and this fit. With the uncrushed compartment it returns 128.5 ml/100 g/min and an arrival
of 0.20 s, the lower edge of the fit's grid in every gray matter voxel: four samples three
times too high at the short delays are explained as a large perfusion arriving as early
as the model allows, and the maps (red run, top and bottom middle) are uniformly wrong,
not noisy, because every gray matter voxel carries the same arterial signal. The crushed
run sits between the two in the way that matters for practice: its CBF, 60.0, is back at
the baseline, since a fifth of the arterial signal at four of the seven delays is not
enough to pull the amplitude, but its arrival, 0.30 s, is still half a second early, and
the ATT map (bottom right) is as uniformly wrong as the uncrushed one. Crushing protects
the perfusion estimate before it protects the transit-time estimate. Dropping the delays
below 0.5 s restores both for all three runs (printed): 61.3 and 0.75 s for the uncrushed
data, 61.1 and 0.78 s for the crushed, which is the white paper's long delay in multi-delay
form. The third remedy, a model with the arterial term, has in this phantom the easiest
possible case, a compartment that is the same in every voxel and switches off at one
instant; a real one is dispersed, and the toy above shows what that does to the fit.

## Learned denoising and quantification

Neural networks trained on paired data now denoise ASL difference images, predict the CBF
map of a full-length scan from a few pairs {cite:p}`kim2018`, and, in some work, quantify CBF and transit
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
positron emission tomography with labeled water is the usual reference, and comparisons
with it carry its own errors {cite:p}`ye2000pet,heijtel2014`, and test-retest
reproducibility tests only agreement with oneself. A simulator that takes a BIDS protocol
and a phantom through the kinetic model, the signal equations, and a k-space acquisition,
and writes the answer it started from, lets a method be scored on the quantity it claims
to measure, under the artifacts it will meet, at the parameters of a specific protocol.
That is the loop the book has closed with aslscan and the ASLDRO phantom
{cite:p}`olivertaylor2021`: the toy tier for every curve, the pipeline tier for every image.

The limits of the simulator are the limits of the validation, and the chapters have said
where they apply. Several of the methods of this chapter are in the simulator, each with
an approximation of its own: the macrovascular compartment, modeled as
plug flow with one arterial blood volume and one arterial arrival per tissue, without
dispersion and without vessel geometry (the `vascular` runs above); water exchange, modeled
as a single-pass split of the label's $T_2$, not as a two-compartment $T_1$ model (the
`multi-te` runs); physiological noise, modeled as global factors per slice or shot, without
spatial structure or a pulsatile arterial volume (the `physio` runs of
[Chapter 9](../03-preprocessing/09-background-suppression.md)); the 3D readouts, with an
ideal slab ([Chapter 2](../01-mri-physics/02-epi-and-reconstruction.md)); and spatially
selective suppression pulses, through the bolus-position model
([Chapter 9](../03-preprocessing/09-background-suppression.md)). The table lists what
aslscan still does not model, which chapter of this book would change if it did, and what
adding it would take.

| Not modeled | Where it matters | What adding it takes |
|---|---|---|
| Bolus dispersion | Chapters 15, 18 and this one: multi-delay and time-encoded fits, the abrupt arterial cut-off of the `vascular` runs | A kernel (gamma-variate with two parameters, possibly maps) convolved with the delivery in `kinetic.delta_m` and with the arterial compartment, which then become numerical integrals as in this chapter |
| A two-compartment $T_1$ exchange model | Chapter 5 and this chapter: the total label, not only its $T_2$ split | Separate longitudinal decay for the intravascular label ($T_{1\mathrm{b}}$) and the exchanged label ($T_1'$), which the single-pass model keeps at $T_1'$ for both; the `deltam` truth would then leave the Buxton curve |
| Slab profile and $k_z$ aliasing | Chapters 2 and 17: the end partitions of a 3D volume, the choice of slab oversampling | An excitation profile along z and a field of view that can be smaller than the excited object, so that the edges wrap, with the through-plane oversampling that real 3D ASL uses against it |
| 3D gradient-echo trains | This chapter and Chapter 18: multi-echo and Look-Locker acquisitions with a 3D readout | An echo train whose amplitudes follow $T_2^*$ and the field map's phase per echo instead of the extended phase graph's spin echoes |
| The echo train's longitudinal state | Chapters 2, 9, 10: the steady state of a segmented 3D readout, suppression timing across shots | Carrying what the train leaves of $M_z$ into the next repetition instead of counting recovery from zero at the excitation |
| Labeling-plane motion | Chapters 4, 10: labeling efficiency under head motion | Moving the labeling plane with the head, so the efficiency and the arterial path change per volume, instead of moving only the finished images |
| Look-Locker with 3D or multi-echo readouts | Chapter 18 | The depletion model of Chapter 18 applied through a 3D train's excitations, and several echoes of each Look-Locker readout |
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
- **A short echo time** keeps the exchange question out of a single-echo CBF measurement.
  At the reference delay 2.5 % of the gray matter label was still in blood, and a
  calibration that gives the whole label the blood's constant underestimated CBF by 7 % at
  12 ms and 29 % at 54 ms; only a multi-echo acquisition can tell, and it must read at a
  short delay, where an eighth of the label was still intravascular at 1.0 s.
- **A segmented 3D readout with background suppression** is the recommended readout;
  [Chapter 2](../01-mri-physics/02-epi-and-reconstruction.md) measures its through-plane
  blur and its noise advantage, and the `vascular` runs show it giving every voxel the
  same delay.
- **Short delays contain arterial signal.** With 2 % of a voxel arterial the signal at
  delays below the arterial arrival was three times the perfusion signal, the standard fit
  to the seven delays doubled CBF (128.5 against 60.5) and put the arrival at the grid's
  edge; crushing at VENC 4 cm/s left a fifth of it and restored CBF (60.0) but not the
  arrival (0.30 s against 0.79); dropping the delays below the arterial arrival restored
  both. Crush it, avoid it, or model it; do not fit the standard model to samples before the
  bolus has cleared the arteries.
- **Ask how a learned method was validated**, and prefer methods scored against a
  simulated truth under the artifacts of your protocol.

## Further reading

The reviews of advanced ASL methods {cite:p}`vanosch2018,hernandezgarcia2019`;
velocity-selective ASL {cite:p}`wong2006` and its inversion variant {cite:p}`qin2016`;
vessel-encoded {cite:p}`wong2007` and super-selective {cite:p}`helle2010` labeling;
restricted water exchange {cite:p}`stlawrence2000,zhou2001,parkes2002`, its $T_2$
signature {cite:p}`gregori2013` and a multi-TE study of blood-brain barrier permeability
{cite:p}`ohene2021`; 3D GRASE {cite:p}`gunther2005,fernandezseara2005`
and the comparison of readouts {cite:p}`vidorreta2013`; models of dispersion
{cite:p}`hrabe2004,chappell2013` and the macrovascular model {cite:p}`chappell2010`;
deep learning applied to ASL {cite:p}`kim2018`; the multi-timepoint recommendations, which discuss dispersion and
arterial signal in time-encoded data {cite:p}`woods2024`; and the phantom this book's
simulations start from {cite:p}`olivertaylor2021`.
