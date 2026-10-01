---
title: "0.3 Notation and units"
kernelspec:
  name: python3
  display_name: Python 3
---

Terms and symbols as used throughout the book, with units and the chapter that introduces
each. [Appendix D](../appendices/d-glossary.md) is the full glossary.

## Terms

Acquisition terms that recur in the book, each in one plain sentence. The chapter column
says where the term is explained properly.

| Term | In plain words | Chapter |
|---|---|---|
| label image, control image | the two images of every pair: one taken after arterial blood was inverted, one without; their difference is the perfusion signal | 4, 6 |
| ΔM, the difference image | control minus label: the signal carried into each voxel by labeled blood, about 1 % of the control image in gray matter | 6 |
| M0 scan | an image with no labeling and a long TR, which gives the equilibrium magnetization the difference is divided by | 6, 16 |
| labeling duration (LD) | how long arterial blood is inverted in (pseudo-)continuous ASL; the length of the bolus | 4 |
| post-labeling delay (PLD) | the wait between the end of labeling and the readout, chosen to let the bolus arrive | 5, 7 |
| inversion time (TI) | in pulsed ASL, the time from the labeling pulse to the readout | 4 |
| bolus cut-off | in pulsed ASL, a saturation pulse that ends the bolus at a known time, so its duration is known | 4 |
| arterial transit time (ATT) | the time labeled blood takes from the labeling plane to a voxel's capillaries | 3, 5, 15 |
| labeling efficiency (α) | the fraction of arterial magnetization the labeling actually inverts | 4 |
| background suppression | inversion pulses during the delay timed so the static tissue has almost no signal at readout, while the label keeps its difference | 9 |
| EPI (echo-planar imaging) | the fast readout that records a whole slice's k-space after one excitation | 2 |
| slice timing | in a 2D readout, each slice is excited at a different time, so each has its own effective delay | 6, 14 |
| `aslcontext.tsv` | the BIDS table that names each volume of the series (control, label, m0scan, deltam) | 6 |
| sidecar | the BIDS JSON file next to the series that records the protocol; the quantification reads its fields | 6, 14 |

## Physics and signal

| Symbol | Meaning | Unit | Chapter |
|---|---|---|---|
| $B_0$ | static magnetic field (3 T in the phantom) | T | 1 |
| $\gamma/2\pi$ | gyromagnetic ratio of the proton; 42.58 MHz/T | MHz/T | 1 |
| $M_0$ | equilibrium longitudinal magnetization of tissue (the phantom's `M0map`) | arbitrary | 1, 6 |
| $M_z$, $M_{xy}$ | longitudinal and transverse magnetization | arbitrary | 1 |
| $T_1$, $T_2$, $T_2^*$ | relaxation times of tissue (GM 1.33 s; 80 ms; 66 ms) | s | 1 |
| $T_{1b}$, $T_{2b}$ | relaxation times of arterial blood (1.65 s; 165 ms at 3 T) | s | 3, 5 |
| $T_1'$ | apparent tissue T1 with the label's outflow, $1/T_1' = 1/T_1 + f/\lambda$ | s | 5 |
| TR, TE | repetition time, echo time | s | 1, 2 |
| $\Delta f$ | off-resonance (field offset) | Hz | 2, 11 |
| $\sigma$ | noise standard deviation of one real or imaginary channel | image units | 8 |
| TotalReadoutTime | effective duration of the EPI readout, which sets the distortion per Hz | s | 2, 11 |

## Perfusion and the kinetic model

| Symbol | Meaning | Unit | Chapter |
|---|---|---|---|
| $f$, CBF | cerebral blood flow (perfusion); GM 60, WM 20 in the phantom | ml/100 g/min | 3 |
| $\lambda$ | blood-brain partition coefficient of water; 0.9 | ml/g | 3 |
| $M_{0b}$ | equilibrium magnetization of arterial blood, $M_0 / \lambda$ | arbitrary | 3, 16 |
| $\alpha$ | labeling efficiency (PCASL 0.85, CASL 0.68 here, PASL 0.98) | – | 4 |
| $\tau$ | bolus duration: the labeling duration for (P)CASL, the cut-off delay for PASL | s | 4, 5 |
| $\delta$, ATT | arterial transit time (GM 0.8 s, WM 1.2 s in the phantom) | s | 3, 5 |
| $w$, PLD | post-labeling delay | s | 5 |
| $t$ | signal time from the start of labeling: PLD + $\tau$ for (P)CASL, TI for PASL, plus the slice's offset in a 2D readout | s | 5, 6 |
| $\Delta M(t)$ | label-control difference in longitudinal magnetization at time $t$ | M0 units | 5 |
| $\epsilon$ | inversion efficiency of one background-suppression pulse (0.95) | – | 9 |
| $N$ | number of control-label pairs | – | 6, 8 |
| SNR, tSNR | signal-to-noise ratio; temporal SNR of the difference series | – | 8 |

## Data and files

| Term | Meaning |
|---|---|
| `part-mag_asl`, `part-phase_asl` | BIDS magnitude and phase images of the complex ASL series |
| `aslcontext.tsv` | one row per volume: `control`, `label`, `m0scan`, or `deltam` |
| `m0scan` | the separate calibration image (`M0Type: Separate`) with its own sidecar |
| `desc-<name>_gt` | the simulator's ground-truth maps, next to the series ([Appendix E](../appendices/e-truth-map-catalogue.md)) |
| `label-GM_probseg` | the pipeline's tissue fraction of every acquisition voxel |
| overlay | the simulator's TOML file of settings BIDS does not record ([Appendix A](../appendices/a-aslscan-cookbook.md)) |

## Conventions

- Volumes are arrays `(x, y, z)`, with x left-right, y posterior-anterior (the phase-encode
  axis), and z inferior-superior, at 3.5 × 3.5 × 5 mm; a series appends the volume axis
  last. Axial slices are shown with anterior at the top.
- Image intensities are the simulator's units: the equilibrium magnetization times 100,
  times the saturation and T2 factors of the readout, so a gray matter control voxel is
  about 6200, a gray matter M0 scan voxel about 6450, and a gray matter difference about 30.
  The ground-truth `deltam` map is in the units of `M0map` (about 0.3 in pure gray matter).
- Unless stated otherwise, the constants are the phantom's and the white paper's: $\lambda$
  0.9, $T_{1b}$ 1.65 s, $\alpha$ 0.85 for PCASL, and the tissue values of the table in
  [Chapter 1](../01-mri-physics/01-spins-and-relaxation.md).

```{code-cell} python
:tags: [remove-cell]
import aslbook
assert aslbook.__version__
```
