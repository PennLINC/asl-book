---
title: "18. Time-encoded (Hadamard) and Look-Locker ASL"
kernelspec:
  name: python3
  display_name: Python 3
---

:::{admonition} Simulated datasets in this chapter
:class: note
- **Built in this page:** the kinetic model of [Chapter 5](../02-labeling/05-kinetic-model.md) evaluated sub-bolus by sub-bolus on the pure tissues and on the packaged phantom slab, with noise added in the page; a Look-Locker readout train simulated from the same model ([Appendix B](../appendices/b-data-manifest.md#app-b-package-data)).

Neither acquisition of this chapter is simulated by aslscan, so there is no pipeline-tier dataset here: everything is toy tier ([Chapter 0.2](../00-frontmatter/the-simulated-datasets.md)).
:::

## Learning goals

After this chapter you can:

- explain how a Hadamard matrix turns one labeling period into seven sub-boli whose
  signals are recovered from one series by a decoding sum
- derive the noise of a decoded sub-bolus, state the gain over a conventional multi-delay
  series of the same duration, and measure it
- fit CBF and the arterial transit time to a decoded series with the same code as
  [Chapter 15](../04-quantification/15-multi-delay.md)
- simulate a Look-Locker readout train, say how each readout consumes the label, and
  correct the sampled curve for it
- say what the QUASAR sequence adds to a Look-Locker acquisition

```{code-cell} python
:tags: [hide-cell]
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from scipy.linalg import hadamard

from aslbook import kinetic, phantom, plotting, presets, quant, synth
from aslbook.plotting import INK, PALETTE, TISSUE_COLORS, set_style

set_style()
GM, WM = presets.TISSUES["GM"], presets.TISSUES["WM"]
SCALE = presets.REFERENCE.signal_scale  # image units per M0 unit
SIGMA = 40.0  # the reference protocol's noise, image units
```

## Why encode the time axis

[Chapter 15](../04-quantification/15-multi-delay.md) sampled the kinetic curve by acquiring
separate control-label pairs at each post-labeling delay. That divides the scan time: with
seven delays, each delay receives one seventh of the pairs, and the difference image at
each delay is √7 noisier than the single-delay image the same scan time would have bought.
Time-encoded ASL removes the division {cite:p}`dai2013,woods2024`. The labeling period is
cut into sub-boli, each sub-bolus is switched between label and control according to a row
of a Hadamard matrix, and every acquired volume carries information about every delay. A
linear decoding of the series recovers the signal of each sub-bolus separately, and, as the
accounting below shows, each recovered signal has the noise of a scan that had spent the
whole time on that one delay.

## Sub-boli and the Hadamard matrix

A Hadamard matrix $H_N$ of order $N$ is a square matrix of entries $\pm 1$ whose columns
are mutually orthogonal: $H_N^\mathsf{T} H_N = N I$. Sylvester's construction gives them
for every power of two, and `scipy.linalg.hadamard(8)` builds the $8 \times 8$ one used
here. Its first column is all $+1$; the remaining $N - 1 = 7$ columns each contain four
$+1$ and four $-1$ entries.

The labeling period is divided into $N - 1 = 7$ blocks. In encoded volume $i$, block $j$
is *labeled* if $H_{ij} = -1$ and *control* if $H_{ij} = +1$, using columns $1$ to $7$ of
the matrix (column 0, all $+1$, is dropped). Volume 0 is therefore an ordinary control
image, and each of the other seven volumes has four labeled blocks and three control
blocks. Writing $\Delta M_j$ for the label-control difference that block $j$ alone would
produce at the readout, and $L_{ij} \in \{0, 1\}$ for its label state, the signal of
volume $i$ is

$$
S_i = M_\mathrm{s} - \sum_{j=1}^{N-1} L_{ij}\, \Delta M_j + n_i ,
\qquad L_{ij} = \tfrac{1}{2}\,(1 - H_{ij}),
$$

with $M_\mathrm{s}$ the static tissue signal and $n_i$ the noise. The sub-boli add because
the kinetic model is linear in the delivered label: the difference from a bolus labeled
during $[a, b]$ is the integral of the delivery over that interval, and integrals over
adjacent intervals add. Decoding is a weighted sum over the volumes with the matrix's own
column:

$$
\Delta M_k = \frac{2}{N} \sum_{i=0}^{N-1} H_{ik}\, S_i .
$$

:::{dropdown} Why the decoding sum isolates one sub-bolus
Substitute $S_i$ into the sum. The static term vanishes because column $k \geq 1$ is
orthogonal to the all-ones column 0: $\sum_i H_{ik} = 0$. The label terms give
$\sum_i H_{ik} L_{ij} = \tfrac{1}{2}\sum_i H_{ik} - \tfrac{1}{2}\sum_i H_{ik} H_{ij} =
-\tfrac{N}{2}\,\delta_{jk}$ by the same orthogonality. So $\sum_i H_{ik} S_i =
\tfrac{N}{2}\,\Delta M_k + \sum_i H_{ik} n_i$, and the factor $2/N$ gives $\Delta M_k$ plus
a noise term whose variance is $(2/N)^2 \cdot N \sigma^2 = 4\sigma^2/N$.
:::

The timing used in this chapter is the simplest one: seven sub-boli of equal duration
$\tau = 0.4$ s, so the whole labeling lasts 2.8 s, and the readout follows the last
sub-bolus after 0.3 s. Counting from the start of labeling, block $j$ is labeled during
$[a_j, a_j + \tau]$ with $a_j = 0.4\,(j - 1)$ s, and its own post-labeling delay, the time
from its end to the readout, is $\mathrm{PLD}_j = 0.3 + 0.4\,(7 - j)$ s: 2.7 s for the
first block down to 0.3 s for the last. The kinetic model's clock starts at the beginning of
a bolus, so block $j$ contributes `kinetic.delta_m` evaluated at the time $t - a_j$ with
`tau=0.4`: a (P)CASL bolus of its own duration measured from its own start. The same
number is the difference between a bolus labeled during $[0, a_j + \tau]$ and one labeled
during $[0, a_j]$; the cell checks that both constructions agree. Time-encoded ASL is
nearly always paired with a 3D readout, and this chapter follows that: every voxel is read
at the same time, with no 2D slice offsets.

```{code-cell} python
:tags: [hide-input]
N = 8
H = hadamard(N)                       # +1 / -1, first column all +1
L = (1 - H[:, 1:]) // 2               # (N, 7): 1 = labeled, 0 = control
D = 2.0 / N * H[:, 1:].T              # (7, N): the decoding matrix
TAU = 0.4                             # s, every sub-bolus
PLD_LAST = 0.3                        # s, from the end of labeling to the readout
n_sub = N - 1
starts = TAU * np.arange(n_sub)       # a_j
t_read = n_sub * TAU + PLD_LAST       # 3.1 s from the start of labeling
plds = t_read - (starts + TAU)        # PLD_j, 2.7 ... 0.3 s
assert np.allclose(D @ (-L), np.eye(n_sub))

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1, 1.6]})
cmap = ListedColormap([PALETTE[1], INK["grid"]])
ax0.imshow(H, cmap=cmap, vmin=-1, vmax=1)
for i in range(N):
    for j in range(N):
        ax0.text(j, i, "−" if H[i, j] < 0 else "+", ha="center", va="center", fontsize=9,
                 color="white" if H[i, j] < 0 else INK["secondary"])
ax0.set(xticks=range(N), xticklabels=["(0)"] + [f"{j}" for j in range(1, N)], yticks=range(N),
        xlabel="column: sub-bolus j (column 0 dropped)", ylabel="row: encoded volume i", title="Hadamard matrix H₈ (− = label)")
ax0.grid(False)
for i in range(N):
    for j in range(n_sub):
        ax1.add_patch(Rectangle((starts[j], i - 0.38), TAU, 0.76, facecolor=PALETTE[1] if L[i, j] else INK["grid"], edgecolor="white", lw=1))
    ax1.plot([t_read], [i], "k>", ms=6)
ax1.set(xlim=(-0.05, 3.4), ylim=(N - 0.5, -0.5), yticks=range(N), ylabel="encoded volume i",
        xlabel="time from the start of labeling (s)", title="the seven sub-boli in each volume (orange = label, gray = control; ▶ readout)")
ax1.grid(False)
fig.tight_layout()
print(f"labeling {n_sub * TAU:.1f} s in {n_sub} sub-boli of {TAU:.1f} s, readout at {t_read:.1f} s; PLD of each sub-bolus (s): {plds}")
print(f"labeled sub-boli per volume: {L.sum(axis=1)}")
```

The left panel is the matrix: orange entries are $-1$ and mark a labeled block. The right
panel reads the same matrix as a pulse sequence: each row is one acquired volume, time runs
left to right through the seven 0.4 s blocks, and the arrow is the readout at 3.1 s. Volume
0 is all control; every other volume labels four of the seven blocks, and every pair of
columns disagrees in exactly half of the rows, which is the orthogonality the decoding
relies on.

```{code-cell} python
:tags: [hide-input]
def sub_bolus(t, tissue, j):
    """Label-control difference at time t (from the start of labeling) of sub-bolus j alone, M0 units."""
    return kinetic.delta_m(np.asarray(t) - starts[j], tissue.perfusion, tissue.att, tissue.t1, tissue.m0, tau=TAU)


def sub_bolus_by_difference(t, tissue, j):
    """The same, as the difference of two boluses that start together and end apart."""
    long = kinetic.delta_m(t, tissue.perfusion, tissue.att, tissue.t1, tissue.m0, tau=starts[j] + TAU)
    short = kinetic.delta_m(t, tissue.perfusion, tissue.att, tissue.t1, tissue.m0, tau=starts[j]) if j > 0 else 0.0
    return long - short


t_axis = np.linspace(0, 3.5, 701)
dm_true = {t.name: SCALE * np.array([sub_bolus(t_read, t, j) for j in range(n_sub)]) for t in (GM, WM)}
dm_diff = SCALE * np.array([sub_bolus_by_difference(t_read, GM, j) for j in range(n_sub)])
whole = SCALE * kinetic.delta_m(t_read, GM.perfusion, GM.att, GM.t1, GM.m0, tau=n_sub * TAU)

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 3.4))
shades = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, n_sub))
for j in range(n_sub):
    ax0.plot(t_axis, SCALE * sub_bolus(t_axis, GM, j), color=shades[j], lw=1.6, label=f"j = {j + 1}, PLD {plds[j]:.1f} s")
ax0.plot(t_axis, SCALE * kinetic.delta_m(t_axis, GM.perfusion, GM.att, GM.t1, GM.m0, tau=n_sub * TAU), color=INK["secondary"], ls="--", lw=1.2, label="one 2.8 s bolus")
ax0.axvline(t_read, color="k", lw=0.8)
ax0.text(t_read + 0.03, 60, "readout", fontsize=8)
ax0.set(xlabel="time from the start of labeling (s)", ylabel="ΔM (image units)", title="gray matter: each sub-bolus on its own")
ax0.legend(fontsize=7, ncol=2)
pld_axis = np.linspace(0.0, 3.0, 301)
for tissue in (GM, WM):
    ax1.plot(pld_axis, SCALE * kinetic.delta_m(pld_axis + TAU, tissue.perfusion, tissue.att, tissue.t1, tissue.m0, tau=TAU), color=TISSUE_COLORS[tissue.name], lw=1.2, alpha=0.6)
    ax1.plot(plds, dm_true[tissue.name], "o", color=TISSUE_COLORS[tissue.name], label=f"{tissue.name}: the seven sub-boli")
ax1.set(xlabel="post-labeling delay of the sub-bolus (s)", ylabel="ΔM (image units)", title="what decoding should return (noise-free)")
ax1.legend()
fig.tight_layout()
print(f"GM sub-bolus ΔM at the readout (image units): {np.round(dm_true['GM'], 1)}")
print(f"WM sub-bolus ΔM at the readout (image units): {np.round(dm_true['WM'], 1)}")
print(f"direct vs difference construction, max |Δ|: {np.abs(dm_true['GM'] - dm_diff).max():.1e} image units")
print(f"sum of the seven GM sub-boli {dm_true['GM'].sum():.1f} = one 2.8 s bolus at PLD 0.3 s {whole:.1f}")
```

Left: the seven sub-bolus curves of a gray-matter voxel, each a short (P)CASL bolus that
rises once its label arrives 0.8 s after its own start, plateaus for its 0.4 s duration,
and decays. At the readout (vertical line) the early sub-boli have decayed and the last one
(j = 7, labeled from 2.4 to 2.8 s) has not yet arrived, so it contributes nothing; their
sum is the dashed curve of a single 2.8 s bolus, as printed. Right: the noise-free decoded
values sit on the curve of a 0.4 s bolus against its delay, which is exactly what a
conventional multi-delay scan with LD 0.4 s would measure at each of the seven delays. The
white matter values are small and vanish for the two shortest delays, because its transit
time of 1.2 s exceeds them.

## The noise accounting

Each encoded volume has noise of standard deviation $\sigma$, and the decoding sum adds $N$
of them with weights $\pm 2/N$, so a decoded sub-bolus has variance $4\sigma^2/N$: for
$N = 8$, a standard deviation of $\sigma/\sqrt{2}$. A conventional pair gives a difference
image with variance $2\sigma^2$, and the average of $N/2$ such pairs, which take the same
$N$ volumes, has variance $2\sigma^2/(N/2) = 4\sigma^2/N$. That is the statement to
remember: one Hadamard block of $N$ volumes gives *every one* of its $N - 1$ sub-boli the
noise of $N/2$ averaged pairs, as if the whole block had been spent on each delay. The
$\sqrt{N/2}$ often quoted (here 2) is the gain of a decoded sub-bolus over one conventional
pair.

At equal scan time the comparison is with a conventional series that must divide its
volumes over the $N - 1$ delays. Spending the same $N$ volumes on $N - 1$ delays leaves
$N / (2(N - 1))$ pairs per delay, with variance $4(N - 1)\sigma^2/N$, so the time-encoded
series is better in variance by $N - 1$ and in standard deviation by $\sqrt{N - 1}$, which
for this matrix is $\sqrt{7} = 2.65$. The cell simulates both at exactly the same scan
time: seven Hadamard blocks (56 volumes) against four pairs at each of the seven delays
(also 56 volumes), each with the reference protocol's noise of 40 image units, for a pure
gray- and a pure white-matter voxel, repeated over 2000 noise realizations. The noise is
added to the signal directly as Gaussian noise; at a static signal of thousands of image
units the Rician magnitude of [Chapter 8](../03-preprocessing/08-noise.md) is
indistinguishable from it.

```{code-cell} python
:tags: [hide-input]
R_BLOCKS = 7                                  # Hadamard blocks
PAIRS_PER_DELAY = 4                           # conventional pairs at each delay: 7 x 4 x 2 = 56 volumes too
N_REAL = 2000
TR = 4.0                                      # s: 2.8 s labeling + 0.3 s + a readout
rng = np.random.default_rng(18)


def encode(dm_sub, static):
    """Noise-free encoded volumes S_i (last axis N) from the sub-bolus differences dm_sub (..., 7)."""
    return static - np.tensordot(dm_sub, L.T, axes=([-1], [0]))


def simulate_voxel(tissue, n_real, rng):
    """Decoded (n_real, 7) and conventional (n_real, 7) ΔM estimates at equal scan time, image units."""
    dm = dm_true[tissue.name]
    static = SCALE * kinetic.tissue_se(tissue.m0, tissue.t1, TR)
    enc = encode(dm, static)[None, None, :] + SIGMA * rng.standard_normal((n_real, R_BLOCKS, N))
    decoded = (enc @ D.T).mean(axis=1)                                   # (n_real, 7)
    ctrl = static + SIGMA * rng.standard_normal((n_real, PAIRS_PER_DELAY, n_sub))
    lab = static - dm + SIGMA * rng.standard_normal((n_real, PAIRS_PER_DELAY, n_sub))
    conventional = (ctrl - lab).mean(axis=1)
    return decoded, conventional


sims = {t.name: simulate_voxel(t, N_REAL, rng) for t in (GM, WM)}
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
for ax, tissue in zip(axes, (GM, WM)):
    dec, conv = sims[tissue.name]
    ax.plot(pld_axis, SCALE * kinetic.delta_m(pld_axis + TAU, tissue.perfusion, tissue.att, tissue.t1, tissue.m0, tau=TAU), color=INK["secondary"], lw=1.2, label="true curve")
    ax.errorbar(plds - 0.03, dec[0], yerr=dec.std(axis=0), fmt="o", color=PALETTE[0], capsize=3, label="time-encoded, decoded (56 volumes)")
    ax.errorbar(plds + 0.03, conv[0], yerr=conv.std(axis=0), fmt="s", color=PALETTE[3], capsize=3, label="conventional multi-delay (56 volumes)")
    ax.set(xlabel="post-labeling delay of the sub-bolus (s)", ylabel="ΔM (image units)", title=f"{tissue.name}: one realization, bars = SD over {N_REAL} realizations")
    ax.axhline(0, color="k", lw=0.6)
axes[0].legend(fontsize=7)
fig.tight_layout()
sd_dec = np.mean([sims[n][0].std(axis=0).mean() for n in sims])
sd_conv = np.mean([sims[n][1].std(axis=0).mean() for n in sims])
print(f"noise SD of one decoded sub-bolus: predicted {SIGMA * 2 / np.sqrt(N) / np.sqrt(R_BLOCKS):.1f}, measured {sd_dec:.1f} image units")
print(f"noise SD of one conventional delay: predicted {SIGMA * np.sqrt(2) / np.sqrt(PAIRS_PER_DELAY):.1f}, measured {sd_conv:.1f} image units")
print(f"SNR gain at equal scan time: measured {sd_conv / sd_dec:.2f}, predicted sqrt(N - 1) = {np.sqrt(N - 1):.2f}")
```

The markers are one noise realization of each scheme and the bars its spread over all
realizations. The decoded time-encoded values (circles) scatter about the true curve with a
standard deviation of 10.7 image units; the conventional values at the same scan time
(squares) scatter with 28.4, and the measured ratio 2.67 is the predicted $\sqrt{7} = 2.65$
within the sampling error of 2000 realizations. For
gray matter the time-encoded series resolves the shape of the curve from a single 3.7 min
scan of one 3.5 × 3.5 × 5 mm voxel; the conventional series does not. For white matter,
whose sub-bolus signals are 1 to 5 image units, neither scheme resolves a single voxel at
this noise level, and the gain is the same factor.

:::{admonition} What the gain does not buy
:class: note
The decoded sub-boli can be summed to recover the difference image of one 2.8 s bolus at
PLD 0.3 s, but the seven noise terms add: the sum has variance $7 \cdot 4\sigma^2/N$, which
is $\sqrt{7}$ worse than a dedicated single-delay scan of the same 56 volumes. Time encoding
buys the timing information at no cost relative to a conventional *multi-delay* scan; it
does not turn a multi-delay scan into a single-delay one for free.
:::

## Fitting CBF and ATT to the decoded series

Because every sub-bolus has the same duration, the decoded series is a multi-delay series
with LD 0.4 s and seven delays, and `quant.fit_multi_pld` of
[Chapter 15](../04-quantification/15-multi-delay.md) fits it unchanged with `tau=0.4`: a grid
over the transit time, CBF by least squares at each candidate, the residual choosing between
them. The cell fits every one of the 2000 realizations of both schemes for both tissues.

```{code-cell} python
:tags: [hide-input]
ATT_GRID = np.arange(0.2, 2.501, 0.02)
fits = {}
for scheme in ("time-encoded", "conventional"):
    k = 0 if scheme == "time-encoded" else 1
    dm = np.stack([sims["GM"][k], sims["WM"][k]], axis=1)[:, :, None, :]       # (n_real, 2, 1, 7)
    m0 = SCALE * np.broadcast_to(np.array([GM.m0, WM.m0])[None, :, None], (N_REAL, 2, 1))
    t1 = np.broadcast_to(np.array([GM.t1, WM.t1])[None, :, None], (N_REAL, 2, 1))
    cbf, att = quant.fit_multi_pld(dm, plds, np.ascontiguousarray(m0), tau=TAU, t1_tissue=t1, att_grid=ATT_GRID)
    fits[scheme] = (cbf[:, :, 0], att[:, :, 0])

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 3.2))
colors = {"time-encoded": PALETTE[0], "conventional": PALETTE[3]}
for scheme, (cbf, att) in fits.items():
    ax0.hist(cbf[:, 0], bins=np.linspace(0, 150, 61), color=colors[scheme], alpha=0.6, label=scheme)
    ax1.hist(att[:, 0], bins=np.arange(0.19, 2.52, 0.04), color=colors[scheme], alpha=0.6, label=scheme)
ax0.axvline(GM.perfusion, color="k", lw=1)
ax1.axvline(GM.att, color="k", lw=1)
ax0.set(xlabel="fitted CBF (ml/100 g/min)", ylabel="realizations", title=f"pure GM voxel: fitted CBF (truth {GM.perfusion:.0f}), {N_REAL} noise realizations")
ax1.set(xlabel="fitted ATT (s)", title=f"pure GM voxel: fitted ATT (truth {GM.att:.1f} s)")
ax0.legend(fontsize=7)
fig.tight_layout()
for scheme, (cbf, att) in fits.items():
    for col, tissue in enumerate((GM, WM)):
        print(f"{scheme:13s} {tissue.name}: CBF {cbf[:, col].mean():5.1f} ± {cbf[:, col].std():4.1f} (truth {tissue.perfusion:.0f}), "
              f"ATT {att[:, col].mean():.2f} ± {att[:, col].std():.2f} s (truth {tissue.att:.1f})")
```

For a pure gray-matter voxel the time-encoded fit returns CBF 63.5 ± 18.5 ml/100 g/min
and ATT 0.81 ± 0.25 s (mean ± SD over the realizations), centered near the truth of 60 and
0.8 s. The conventional fit at the same scan time returns 99.6 ± 77.7 and 1.15 ± 0.67 s:
four times the spread, a pile of realizations clipped at zero, and a mean biased upward,
because the fit is nonlinear: a noise pattern that looks like a late arrival is explained
by a long transit time and a large CBF. The spikes in the ATT histogram sit at the sample
delays (0.7, 1.1, 1.5 s, ...) and at the grid's ends: with seven samples, a transit time
that puts one sample exactly at the arrival edge fits many noise patterns equally well.
The white-matter fits, printed above, are dominated by noise for
both schemes at this voxel size: a single 3.7 min scan does not measure white-matter
transit time voxel by voxel with either encoding. The point of the comparison is the ratio
of the spreads, not their absolute size, and the encoding changes only that ratio.

## See it: the slab

The same two acquisitions on the packaged slab, with the tissue fractions of every voxel
mixing the three classes as in `aslbook.synth`, and Rician noise of 40 image units on each
volume. The calibration image is exact (the true M0 map in image units, no separate scan),
so the comparison isolates the encoding.

```{code-cell} python
:tags: [hide-input]
ph = phantom.slab()
fr = phantom.fractions(ph)
maps = phantom.maps(ph)
rng = np.random.default_rng(1818)


def class_terms(tissue):
    static = SCALE * kinetic.tissue_se(tissue.m0, tissue.t1, TR)
    dm = SCALE * np.array([sub_bolus(t_read, tissue, j) for j in range(n_sub)])
    return static, dm


# time-encoded: R_BLOCKS blocks of N volumes
clean_enc = np.zeros(ph["gm"].shape + (N,), np.float32)
clean_ctrl = np.zeros(ph["gm"].shape, np.float32)
clean_lab = np.zeros(ph["gm"].shape + (n_sub,), np.float32)
for name, tissue in presets.TISSUES.items():
    static, dm = class_terms(tissue)
    clean_enc += fr[name][..., None] * encode(dm, static)[None, None, None, :]
    clean_ctrl += fr[name] * static
    clean_lab += fr[name][..., None] * (static - dm)[None, None, None, :]
decoded = np.zeros(clean_lab.shape, np.float32)
for r in range(R_BLOCKS):
    enc = synth.add_noise(clean_enc, SIGMA, rng)
    decoded += enc @ D.T.astype(np.float32) / R_BLOCKS
# conventional: PAIRS_PER_DELAY pairs at each delay
conventional = np.zeros(clean_lab.shape, np.float32)
for r in range(PAIRS_PER_DELAY):
    for j in range(n_sub):
        conventional[..., j] += (synth.add_noise(clean_ctrl, SIGMA, rng) - synth.add_noise(clean_lab[..., j], SIGMA, rng)) / PAIRS_PER_DELAY

show = [4, 2, 0]  # sub-boli with PLD 1.1, 1.9, 2.7 s
fig, axes = plt.subplots(2, 3, figsize=(9, 6))
for col, j in enumerate(show):
    plotting.show_slice(axes[0, col], decoded[..., j], phantom.DISPLAY_SLICE, f"time-encoded, PLD {plds[j]:.1f} s", vmin=0, vmax=30)
    plotting.show_slice(axes[1, col], conventional[..., j], phantom.DISPLAY_SLICE, f"conventional, PLD {plds[j]:.1f} s", vmin=0, vmax=30)
fig.tight_layout()
gm_pure, wm_pure = fr["GM"] >= 0.9, fr["WM"] >= 0.9
print(f"56 volumes each, TR {TR:.0f} s: {56 * TR / 60:.1f} min per scheme")
print("noise SD in pure GM about the truth, per delay (image units):")
print(f"  time-encoded {np.round([np.std(decoded[gm_pure, j] - dm_true['GM'][j]) for j in range(n_sub)], 1)}")
print(f"  conventional {np.round([np.std(conventional[gm_pure, j] - dm_true['GM'][j]) for j in range(n_sub)], 1)}")
```

The top row is the decoded time-encoded series, the bottom row the conventional one, both
at the same three delays, the same window (0 to 30 image units), and the same scan time.
The cortical ribbon and its decay from PLD 1.1 to 2.7 s are visible in the top row and
barely in the bottom one; the noise measured over the pure gray-matter voxels is the
$\sqrt{7}$ ratio of the previous section, now on images.

```{code-cell} python
:tags: [hide-input]
m0_img = SCALE * maps["m0"]
fit_mask = (fr["GM"] + fr["WM"]) >= 0.5
results = {}
for scheme, series in (("time-encoded", decoded), ("conventional", conventional)):
    cbf, att = quant.fit_multi_pld(series, plds, m0_img, tau=TAU, t1_tissue=maps["t1"], mask=fit_mask, att_grid=np.arange(0.2, 3.0, 0.05))
    results[scheme] = (cbf, att)
cbf_te, att_te = results["time-encoded"]
fig, axes = plotting.fit_vs_truth(cbf_te, maps["perfusion"], fit_mask, "CBF, time-encoded", k=phantom.DISPLAY_SLICE, unit="(ml/100 g/min)")
for scheme, (cbf, att) in results.items():
    for name, m in (("GM", gm_pure), ("WM", wm_pure)):
        s = quant.score(cbf, maps["perfusion"], m)
        print(f"{scheme:13s} CBF in pure {name}: bias {s['bias']:+5.1f}, RMSE {s['rmse']:5.1f} ml/100 g/min (n = {s['n']})")
```

The 4-panel of [Chapter 14](../04-quantification/14-cbf-quantification.md) for the
time-encoded CBF: the estimate, the truth, their difference, and the voxelwise scatter,
within the voxels that are at least half perfused tissue. The map is noisy at this voxel
size and scan time: in pure gray matter the time-encoded CBF has a bias of +4.3 and an RMSE
of 21.5 ml/100 g/min, and in pure white matter, where the sub-bolus signals are a few image
units, the nonlinear fit's upward bias reaches +31. The conventional fit at the same scan
time has a gray-matter bias of +53 and an RMSE of 124 ml/100 g/min, six times the
time-encoded RMSE: the $\sqrt{7}$ in the noise becomes more than $\sqrt{7}$ in the fitted
CBF, because the conventional series sits below the SNR at which the fit is linear.

```{code-cell} python
:tags: [hide-input]
fig, axes = plotting.fit_vs_truth(att_te, maps["att"], fit_mask, "ATT, time-encoded", k=phantom.DISPLAY_SLICE, kind="att", unit="(s)")
for scheme, (cbf, att) in results.items():
    for name, m in (("GM", gm_pure), ("WM", wm_pure)):
        s = quant.score(att, maps["att"], m)
        print(f"{scheme:13s} ATT in pure {name}: bias {s['bias']:+.2f}, RMSE {s['rmse']:.2f} s")
```

The transit-time map from the same fit. In gray matter the estimate clusters around the
true 0.8 s (bias +0.02 s, RMSE 0.30 s); in white matter the small signal lets the fit
wander over its grid (RMSE 0.84 s). The conventional scheme's gray-matter transit time has
an RMSE of 0.88 s, no better than its white matter: at this scan time it measures neither.
Two remarks on design. Equal sub-boli are the simplest choice, not the best: because the
earliest sub-boli decay longest, designs that lengthen them (T1-adjusted durations) equalize
the decoded SNR across delays, and "free-lunch" designs make the first block long enough to
serve as a single-delay measurement on its own {cite:p}`woods2024`. And at the shortest
delays the decoded signal contains labeled blood still in arteries, which the kinetic model
does not describe; [Chapter 19](./19-frontiers.md) shows what that does to a fit.

## Look-Locker readouts

A different way to sample the curve after one label is to read it many times. A
Look-Locker readout {cite:p}`gunther2001` follows one labeling with a train of excitations
at a low flip angle $\theta$, separated by $\Delta t$, each followed by its own image
readout. Each excitation reads $\sin\theta$ of the longitudinal magnetization present at
that moment, and leaves $\cos\theta$ of it along the longitudinal axis. The static tissue
settles into a driven steady state, but the label is not replenished: a portion of label
that has arrived in the voxel is multiplied by $\cos\theta$ at every subsequent readout,
while label still on its way in the arteries, outside the imaging slab, is untouched.

The simulation below writes the kinetic model as the integral it is, delivery times decay,
so the readouts can be inserted: the label delivered at time $t'$ and read at $t_k$ has
decayed by $e^{-(t_k - t')/T_1'}$ and has been hit by every readout between $t'$ and $t_k$,

$$
\Delta M_\mathrm{LL}(t_k) = \int_0^{t_k} 2 M_{0\mathrm{b}}\, f\, \alpha\, c(t')\,
e^{-(t_k - t')/T_1'}\, (\cos\theta)^{\,n(t', t_k)}\, \mathrm{d}t' ,
$$

with $n(t', t_k)$ the number of readouts in $(t', t_k)$ and $c(t')$ the arterial
concentration of [Chapter 5](../02-labeling/05-kinetic-model.md). With $\theta = 0$ the
integral is `kinetic.delta_m`, which the cell checks. The signal actually read is
$\sin\theta \cdot \Delta M_\mathrm{LL}$; the figure divides by $\sin\theta$ to show the
longitudinal label the train has left. The train here is a pulsed (FAIR) label with a
0.7 s bolus cut-off, [Chapter 4](../02-labeling/04-labeling-schemes.md), followed by ten
readouts every 0.3 s from 0.3 to 3.0 s.

```{code-cell} python
:tags: [hide-input]
T_RO = np.round(np.arange(0.3, 3.01, 0.3), 3)   # readout times, s
DT_RO = 0.3
TAU_PASL = 0.7                                   # bolus cut-off, s
THETAS = (20.0, 35.0, 50.0)


def look_locker(tissue, theta_deg, perfusion=None, att=None, t_ro=T_RO, label_type="PASL", tau=TAU_PASL, dt=0.002):
    """ΔM left along the longitudinal axis at each readout of a Look-Locker train (M0 units)."""
    perfusion = tissue.perfusion if perfusion is None else perfusion
    att = tissue.att if att is None else att
    tg = (np.arange(int(round(t_ro[-1] / dt))) + 0.5) * dt
    f, m0b, alpha = perfusion / 6000.0, tissue.m0 / presets.LAMBDA, presets.ALPHA[label_type]
    t1p = kinetic.t1_prime(tissue.t1, perfusion)
    inflow = (tg > att) & (tg < att + tau)
    c = np.where(inflow, np.exp(-tg / presets.T1_BLOOD) if label_type == "PASL" else np.exp(-att / presets.T1_BLOOD), 0.0)
    delivery = 2 * m0b * f * alpha * c
    cos = np.cos(np.radians(theta_deg))
    out = np.zeros(len(t_ro))
    for k, tk in enumerate(t_ro):
        before = tg < tk
        n_hits = k - np.searchsorted(t_ro, tg[before], side="right")   # readouts strictly between t' and t_k
        out[k] = np.sum(delivery[before] * np.exp(-(tk - tg[before]) / t1p) * cos ** n_hits) * dt
    return out


def t1_look_locker(t1, theta_deg, dt=DT_RO):
    """The apparent T1 of a Look-Locker train: the readouts add a decay of -ln(cos θ) per interval."""
    return 1.0 / (1.0 / t1 - np.log(np.cos(np.radians(theta_deg))) / dt)


t_fine = np.linspace(0, 3.0, 601)
true_fine = SCALE * kinetic.delta_m(t_fine, GM.perfusion, GM.att, GM.t1, GM.m0, label_type="PASL", tau=TAU_PASL)
true_ro = SCALE * kinetic.delta_m(T_RO, GM.perfusion, GM.att, GM.t1, GM.m0, label_type="PASL", tau=TAU_PASL)
check = SCALE * look_locker(GM, 0.0)
ll = {th: SCALE * look_locker(GM, th) for th in THETAS}
fig, axes = plt.subplots(1, 3, figsize=(12, 3.3), sharey=True)
for ax, th in zip(axes, THETAS):
    corrected_model = SCALE * kinetic.delta_m(t_fine, GM.perfusion, GM.att, t1_look_locker(GM.t1, th), GM.m0, label_type="PASL", tau=TAU_PASL)
    ax.plot(t_fine, true_fine, color=INK["secondary"], lw=1.2, label="true ΔM(t), no readouts")
    ax.plot(t_fine, corrected_model, color=PALETTE[0], ls="--", lw=1.2, label="model with T1 → T1_LL")
    ax.plot(T_RO, ll[th], "o", color=TISSUE_COLORS["GM"], label="Look-Locker samples / sin θ")
    ax.set(xlabel="time from the labeling pulse (s)", title=f"θ = {th:.0f}°: T1_LL = {t1_look_locker(GM.t1, th):.2f} s, sin θ = {np.sin(np.radians(th)):.2f}")
axes[0].set(ylabel="longitudinal ΔM (image units)")
axes[0].legend(fontsize=7)
fig.tight_layout()
print(f"θ = 0 check: max |simulated − kinetic.delta_m| = {np.abs(check - true_ro).max():.2f} image units (peak {true_ro.max():.1f})")
for th in THETAS:
    k_peak = np.argmax(true_ro)
    print(f"θ = {th:.0f}°: label left at the peak readout {ll[th][k_peak] / true_ro[k_peak]:.2f}, at the last readout {ll[th][-1] / true_ro[-1]:.2f} of the undisturbed curve")
```

Each panel is one flip angle. The gray curve is the undisturbed kinetic curve of a
gray-matter voxel; the orange points are what the train leaves along the longitudinal axis
at each readout. The first two readouts fall before the label arrives and see nothing; the
samples then track the curve at first and fall below it faster and faster, because each
readout removes $1 - \cos\theta$ of the label that has already arrived. At 20° the loss at
the last readout is a third; at 50° the train has consumed nearly all of the label by
3 s, as printed. The dashed curve is the standard correction {cite:p}`gunther2001`: the
readouts act like an extra decay of the tissue label of $-\ln(\cos\theta)$ per interval,
so the kinetic model is evaluated with the apparent

$$
\frac{1}{T_{1,\mathrm{LL}}} = \frac{1}{T_1} - \frac{\ln \cos\theta}{\Delta t} ,
$$

in place of the tissue $T_1$ (it is 1.04 s at 20° and 0.45 s at 50° for a $T_1$ of 1.33 s).
The corrected model follows the samples closely but not exactly: it spreads each readout's
loss continuously over the interval, while the real loss happens at the readout, so label
delivered just after a readout is charged for it too early. The next cell fits CBF and the
transit time to the samples three ways: ignoring the readouts, with $T_{1,\mathrm{LL}}$, and
with the exact discrete model above.

```{code-cell} python
:tags: [hide-input]
ATT_LL = np.arange(0.2, 2.001, 0.02)


def grid_fit(samples, model):
    """CBF (linear) and ATT (grid) by least squares, like quant.fit_multi_pld: unit-CBF curves,
    then one refinement of T1' at the fitted CBF."""
    best = None
    for a in ATT_LL:
        g = model(1.0, a)
        f = samples @ g / (g @ g)
        resid = np.sum((samples - f * g) ** 2)
        if best is None or resid < best[0]:
            best = (resid, f, a)
    _, f, a = best
    g = model(f, a)
    return f * (samples @ g) / (g @ g), a


rows = []
for th in THETAS:
    samples = ll[th]
    models = {
        "ignore the readouts": lambda f, a: SCALE * kinetic.delta_m(T_RO, f, a, GM.t1, GM.m0, label_type="PASL", tau=TAU_PASL),
        "T1 → T1_LL": lambda f, a, th=th: SCALE * kinetic.delta_m(T_RO, f, a, t1_look_locker(GM.t1, th), GM.m0, label_type="PASL", tau=TAU_PASL),
        "exact discrete model": lambda f, a, th=th: SCALE * look_locker(GM, th, perfusion=f, att=a),
    }
    for name, model in models.items():
        cbf, att = grid_fit(samples, model)
        rows.append((th, name, cbf, att))
        print(f"θ = {th:.0f}°, {name:21s}: CBF {cbf:5.1f} ml/100 g/min ({cbf / GM.perfusion - 1:+.0%}), ATT {att:.2f} s (truth 60, 0.80)")

n_ro = np.arange(1, len(T_RO) + 1)
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 3.2))
for th in THETAS:
    c, s = np.cos(np.radians(th)), np.sin(np.radians(th))
    ax0.plot(n_ro, c ** (n_ro - 1), "o-", ms=4, label=f"θ = {th:.0f}°")
    ax1.plot(n_ro, s * c ** (n_ro - 1), "o-", ms=4, label=f"θ = {th:.0f}°")
ax0.set(xlabel="readouts since the label arrived", ylabel="fraction of the label left", title="what each readout leaves: cosⁿ⁻¹ θ")
ax1.set(xlabel="readout number since arrival", ylabel="signal read, per unit label", title="what each readout gives: sin θ · cosⁿ⁻¹ θ")
ax1.legend()
fig.tight_layout()
biases = {th: next(r[2] for r in rows if r[0] == th and r[1] == "ignore the readouts") / GM.perfusion - 1 for th in THETAS}
print("CBF bias when the readouts are ignored: " + ", ".join(f"{th:.0f}° {b:+.0%}" for th, b in biases.items()))
```

The printed fits are the Look-Locker bias. Ignoring the readouts and fitting the ordinary
model to the samples underestimates CBF by 10 % at 20°, 28 % at 35°, and 46 % at 50°, and
shortens the fitted transit time (0.64 s instead of 0.80 s at 50°), because a curve that
falls too fast looks like one that arrived early. The $T_{1,\mathrm{LL}}$ correction
recovers the transit time exactly and reduces the CBF error to +3, +9, and +21 %, with the
sign reversed: it charges the label delivered between two readouts for a readout it has not
yet experienced. The exact discrete model returns 60.0 and 0.80 s at every angle, at the
price of a model that depends on the train's timing. The figure shows the trade behind the choice of $\theta$: a higher flip angle reads
more signal at the first readout after arrival ($\sin\theta$, right panel) but leaves less
for the later ones (left), and the sum over the train is what sets the SNR of the fitted
curve. Flip angles of 25 to 35° are typical. The gain of the method is that ten samples of
the curve come from one label, at $\sin\theta$ of the signal each, instead of ten labels;
its cost is the flip-angle dependence of the curve, and, in 2D, the slice-by-slice timing
of every readout in the train.

QUASAR {cite:p}`petersen2006` builds on a Look-Locker pulsed acquisition to make the
quantification model-free. The train is acquired twice, once with vascular crushing
gradients that dephase the labeled blood still moving in arteries and once without. The
difference between the two trains is the arterial signal in the voxel, which is the local
arterial input function, and the crushed train is the tissue response. Deconvolving the
one from the other, as dynamic susceptibility contrast perfusion imaging does, gives CBF
without assuming a transit time, a bolus shape, or a tissue $T_1$; the sequence also
acquires the train at a second flip angle so that the tissue $T_1$ and the flip-angle
effect can be measured rather than assumed. Its price is scan time and a low SNR per
sample, and its lasting contribution was to show how much of the ASL signal at short
delays is arterial.

## What this implies for acquisition

- **A multi-delay question is a time-encoded question.** At the same scan time, a
  Hadamard-encoded series measures every delay with $\sqrt{N - 1}$ lower noise than
  separate pairs; with the 8 × 8 matrix that is a factor 2.65, and it does not depend on
  the noise level.
- **Pair time encoding with a 3D readout and background suppression**, as the consensus
  recommendations do {cite:p}`woods2024`; a 2D readout gives each slice its own set of
  seven delays.
- **Choose the sub-bolus durations for the question**: equal blocks are simplest,
  T1-adjusted blocks equalize the SNR across delays, and a long first block gives a
  single-delay image for free.
- **Decoded short-delay signals contain arterial blood**; fit them with a model that has a
  macrovascular term, or start the delays late enough for the population's transit times.
- **A Look-Locker curve is flip-angle dependent**: quantify it with the $T_{1,\mathrm{LL}}$
  correction or an explicit model of the train, never with the single-readout formula.

## Further reading

Hadamard-encoded CASL {cite:p}`dai2013` and the consensus recommendations for
multi-timepoint ASL, which cover time-encoded designs and their quantification
{cite:p}`woods2024`; the Look-Locker sampling strategy for ASL {cite:p}`gunther2001` and
QUASAR {cite:p}`petersen2006`; the kinetic model every sub-bolus obeys {cite:p}`buxton1998`;
and the reviews of advanced ASL methods {cite:p}`vanosch2018,hernandezgarcia2019`.
