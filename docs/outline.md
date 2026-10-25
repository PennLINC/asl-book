# Arterial spin labeling: an executable book — outline

Status: written 2026-09-30; extended 2026-10-06 for the 3D, time-encoded, Look-Locker, multi-echo, vascular and physiological-noise simulations. Companion: [chapter-conventions.md](chapter-conventions.md)
(the notebook standard every chapter follows) and the diffusion book this one is modeled on
(`../diffusion-book`).

## Premise

Every figure in the book is produced by code the reader can run. Realistic data come from
**aslscan** (a headless ASL simulator: a BIDS ASL protocol + a digital brain phantom → a BIDS
ASL series with the artifacts of a 2D EPI, a 3D GRASE or a stack-of-spirals acquisition, plus the ground-truth maps
the series was made from). Because the phantom is piecewise constant (one perfusion, transit
time, T1, T2, T2*, M0 per tissue class) the answer key is exact: every quantification chapter
ends with a quantitative "estimate vs. truth" comparison.

Two kinds of simulation feed the book, and each chapter says which it uses:

| Tier | What | Where it runs | Used for |
|---|---|---|---|
| **Toy** | the kinetic model and the signal equations evaluated in the page on the packaged phantom slab (`aslbook.synth`: aslscan's signal stage in numpy, no k-space), plus small Bloch / k-space / noise simulations | at book build time, seconds | physics intuition, every curve, anything that needs a parameter sweep too fine for the pipeline |
| **Pipeline** | aslscan runs on the slab (`pipelines/`): 2D SE-EPI (the reference), 3D GRASE and stack-of-spirals readouts, with k-space, oversampling (ringing), noise, coils/GRAPPA, partial Fourier, ghosts, spikes, a field map, background suppression (global-bolus or bolus-position), motion (per volume or per shot), physiological noise, time encoding, Look-Locker readouts, multi-echo readouts, exchange, an arterial compartment and crushing | offline, ~8 s per 2D or GRASE run, minutes per spiral run; outputs versioned and fetched at build time | realistic images, artifacts, preprocessing walkthroughs, quantification vs. truth |

The toy tier reproduces aslscan's `deltam` ground truth to float32 precision on every voxel
(verified in `aslbook/tests`), so the two tiers are one simulator with and without its
acquisition stage.

## The phantom and the reference protocol

- **Phantom:** ASLDRO's `hrgt_icbm_2009a_nls_3t` (1 mm; GM f 60 ml/100 g/min, ATT 0.8 s, T1
  1.33 s, T2 80 ms; WM f 20, ATT 1.2 s, T1 0.83 s, T2 110 ms; CSF f 0, T1 3.0 s; blood T1
  1.65 s, T2 165 ms, λ 0.9), cropped to a 100 mm axial slab (20 slices of 5 mm, from the
  temporal lobes to just below the vertex). A synthetic field map variant has ±100 Hz lobes
  above the frontal sinuses and at the temporal bones.
- **Reference protocol (`aslbook.presets.REFERENCE`):** 2D PCASL, LD 1.8 s, PLD 1.8 s, TR
  4.5 s, TE 12 ms, 3.5 × 3.5 × 5 mm, matrix 64 × 68, 20 slices 40 ms apart (readout 0.76 s),
  30 control-label pairs, separate M0 scan at TR 8 s, no background suppression, noise σ ≈ 40
  image units (GM control SNR ≈ 150). It follows the ASL white paper (Alsop et al. 2015)
  within what aslscan simulates (2D EPI; the white paper prefers 3D readouts and background
  suppression, which the book treats explicitly).
- **Display:** axial slice 9 of 20 (through the lateral ventricles), anterior up.

## Conventions used throughout

- Notation table in the front matter (ΔM, M0, f (CBF), λ, α, τ (LD), PLD/w, TI, ATT/δ, T1, T1b,
  T1', TR, TE, …); every chapter links to it.
- Every chapter: *Learning goals* → the physics → *See it* (figures) → *Measure it* (a number
  against truth) → *What this implies for acquisition* → *Further reading*.
- Tissue colors fixed: GM orange, WM blue, CSF aqua. CBF maps `inferno` 0–90; ΔM and
  difference images with a zero-centered diverging map where signed.
- Raw data are always shown as the scanner would give them: BIDS `part-mag`/`part-phase`
  NIfTI + `aslcontext.tsv` + sidecar.

---

## Part 0 — Front matter

### 0.1 How to read and run this book
Audience, the executable cells, the two tiers, running locally, reproduction, conventions.

### 0.2 The simulated datasets
The phantom (maps, why piecewise constant), what aslscan does (kinetics → signal → acquisition,
with the simulator diagram), the reference protocol, the dataset table, the ground truth and
what it is not (one value per tissue, no dispersion, an ideal 3D slab, global
physiological noise; the arterial compartment, exchange, physiological noise and the
bolus-position suppression model exist but are off unless a dataset turns them on).

### 0.3 Notation and units

---

## Part I — MRI physics for perfusion imaging

### Chapter 1 — Spins, relaxation, and the longitudinal magnetization
Larmor precession, excitation, T1/T2/T2*; the Bloch equations with emphasis on the
*longitudinal* axis: saturation recovery `M0(1−e^{−TR/T1})`, inversion recovery
`M0(1−2e^{−t/T1})`, why an inverted spin's difference from equilibrium decays with T1 (the
label decays this way); spin echo vs gradient echo; tissue T1/T2 table. *Toy:* Bloch curves,
a T1 recovery animation, the inversion animation, the spin-echo Bloch simulator; the
steady-state signal vs TR per tissue; the blood-tissue T1 comparison. *Pipeline:* none.

### Chapter 2 — Spatial encoding, EPI, and reconstruction
k-space, FOV/matrix/Nyquist, the EPI trajectory and readout time, distortion ∝ field offset ×
readout time along phase encode, partial Fourier, multi-coil and GRAPPA, magnitude/phase,
noise from k-space to image (Gaussian → Rician), Gibbs ringing from truncation; then the
3D readouts: a slab excitation and a train of spin echoes reading one k_z partition each
(GRASE: an EPI block per echo; stack of spirals: an interleaf per echo), the echo amplitudes
(T2 decay, stimulated echoes below 180°) as a k_z window and the through-plane point-spread
function, segmentation over shots, the sqrt(n_z) noise advantage, the spiral trajectory and
its off-resonance blur. Condensed: this chapter exists so Part III can refer to it. *Toy:*
the packaged 1 mm slice → k-space → image at the ASL matrix; truncation ringing; an EPI
timing diagram; the k_z window → PSF. *Pipeline:* `ref-pcasl` (one control image, its
phase, the M0 scan, the sidecar's readout metadata); `readout-3d` (the same control image
with one-shot and four-shot GRASE at 180° and 130°, and the spiral; the sidecar's echo
amplitudes against the measured through-plane blur; the noise per volume).

---

## Part II — Perfusion and its labeling

### Chapter 3 — Perfusion, and blood water as a tracer
CBF definition and units (ml/100 g/min; typical GM 40–100, WM ~20); the tracer view: Kety
(1951) / Kety–Schmidt; a freely diffusible tracer, the partition coefficient λ (0.9 ml/g);
what arterial water carries when its magnetization is inverted; arterial transit time and
why it varies (0.5–2 s; longer in WM, in the elderly, in disease); the tiny size of the
effect: a rough estimate of ΔM/M0 ≈ 1 % from f, λ, T1b. *Toy:* a compartment-model
animation (labeled water arriving, exchanging, decaying), the order-of-magnitude
calculation, an illustration of the phantom's CBF and ATT maps.

### Chapter 4 — Labeling schemes: PASL, CASL, PCASL
Pulsed labeling (EPISTAR, FAIR, PICORE): a slab inverted at once; the bolus-duration problem
and QUIPSS II / Q2TIPS cut-offs. Continuous labeling: flow-driven adiabatic inversion at a
plane, the amplitude-modulated control, efficiency ~0.68 and the magnetization-transfer
problem. Pseudo-continuous labeling: the pulse train, efficiency 0.85, its sensitivity to
off-resonance and velocity (mention). Control conditions; what "label efficiency" means and
how the sidecar records it. *Toy:* labeling geometry diagrams; sagittal snapshots and an
animation of the label traveling from the PCASL plane and the PASL inversion region to the
brain, with the Q2TIPS cut-off (`aslbook.sketch`); a sequence diagram per scheme;
the bolus as a function of time for each scheme (`kinetic.delta_m` with the three branches).
*Pipeline:* `label-types` (the same slab under PCASL, CASL, PASL: ΔM images and their
amplitude ratios vs the efficiency and bolus duration).

### Chapter 5 — The general kinetic model
Buxton et al. 1998: delivery, clearance, relaxation; the three phases (not arrived,
arriving, arrived); the PCASL and PASL solutions; T1' and why it hardly matters; the roles
of ATT, τ, PLD, T1b; the "PLD > ATT" rule and the plateau; multi-delay curves; what the
model leaves out (dispersion, macrovascular signal, exchange). *Toy:* ΔM(t) curves for GM
and WM, the effect of each parameter (small multiples), the animation of the bolus passing.
*Pipeline:* `pld-sweep` (the measured GM and WM ΔM at six delays against the model curve
evaluated from the truth), `ref-clean` (the noise-free difference equals the deltam truth up to
the readout).

### Chapter 6 — The ASL signal: control, label, difference, M0
What the volumes are; the tissue steady state at TR; the difference image and its size
relative to the static signal and to the noise; averaging N pairs; the temporal SNR; the M0
scan and why it is needed; 2D slice timing: each slice has its own PLD; BIDS ASL files
(`aslcontext.tsv`, sidecar fields). *Toy:* `synth.series` on the reference protocol: the
volumes, the pairwise differences, the mean difference vs N, the slice-dependence of ΔM.
*Pipeline:* `ref-pcasl`, `ref-clean` (the same on real images; the SNR of the mean
difference measured; the noise σ estimated from the background).

### Chapter 7 — Acquisition parameter choices
For each: what it controls, what it costs, a simulated sweep. PLD and LD (SNR vs ATT
robustness; the white paper's recommendations by population); TR (must hold LD + PLD +
readout); number of pairs (SNR ∝ √N, scan time); voxel size (SNR ∝ volume, partial
volume); TE (T2 of blood vs tissue); 2D vs 3D readouts (slice timing against one delay for
the slab, the SNR per volume and per unit time, the through-plane blur, the volume that
takes several TRs); background suppression (pointer to Ch. 9); M0 scan choices (pointer to
Ch. 16); field strength (T1b, T1 longer at 3 T). Closing worked example: a 5-minute
protocol. *Pipeline:* `pld-sweep`, `voxel-sweep`, `te-sweep`, `noise-sweep`, `readout-3d`.

---

## Part III — Artifacts and preprocessing

Each chapter follows one template: **(a)** the physics of the artifact, **(b)** the aslscan
setting that produces it, **(c)** the artifact-free reference from the same phantom, **(d)**
the correction step by step, **(e)** residual error vs. truth (ΔM and CBF error maps),
**(f)** what acquisition choices reduce the problem at the source.

### Chapter 8 — Thermal noise, averaging, and denoising
Gaussian in k-space → Rician magnitude → the difference of two Rician images is nearly
Gaussian at high SNR; noise in ΔM is √2 σ per pair; averaging; tSNR maps; outlier pairs;
the multi-coil GRAPPA case (spatially varying, still Rician after the sensitivity-weighted combination); MP-PCA and a NORDIC-style complex variant; simple denoising (Gaussian smoothing, and the
bias it introduces at tissue edges); the CBF map noise floor at 30 pairs. *Pipeline:*
`noise-sweep`, `ref-clean`; CBF error vs number of pairs used.

### Chapter 9 — Background suppression
Why the static signal is the enemy (motion and physiological fluctuations scale with it,
ΔM does not); inversion pulses timed to null tissues at readout: the longitudinal timeline
(`kinetic.tissue_mz`); optimizing two pulse times for GM and WM; the slice dependence in 2D
(later slices recover); the label's own factor `(1−2ε)^N` and the sign flip with one pulse;
the cost: M0 cannot come from the suppressed series; the global-bolus model against the
bolus-position model (a pulse inverts only the label inside the slab); what suppression buys
against physiological noise, measured; suppression with a 3D readout, where one null serves
the slab. *Toy:* the timeline figure, the optimization, the slice dependence.
*Pipeline:* `bgsup` (off / on / perfect / presat / on-slab / early-slab): control images, ΔM
images, their SNR, the effective suppression factor; `physio` (off / on: tSNR with
physiological noise, without and with the pulses); `readout-3d` (the four-shot GRASE and the
spiral with the pulses); `motion` (random vs random-bgsup).

### Chapter 10 — Head motion
Rigid motion between volumes; why ASL is unusually sensitive (a 1 % signal difference vs a
few-percent edge signal change from a 1 mm shift); pairs vs volumes; motion correction by
registration (scipy-based rigid registration of magnitude volumes to the mean control, or
the simulator's true poses replayed); outlier handling: a simple deviation rule, then SCORE
(Dolui 2017: mean-GM-CBF outliers at 2.5 MAD, then removal of the volume most correlated
with the mean map while the pooled within-tissue variance falls) and SCRUB (Dolui 2016:
robust voxelwise reweighting with a tissue prior, run on the volumes SCORE kept), each
implemented in the page and scored against truth; the drift case; motion between the shots
of a segmented 3D readout, which ghosts along the segmented axis and which no volume
registration can undo. *Pipeline:* `motion` (random, drift, random-bgsup), ground-truth
poses from `desc-motion_gt.tsv`: score the estimated poses and the ΔM/CBF error before and
after correction and rejection; `readout-3d` (grase-4shot-shotmotion and its
`desc-motionEvents_gt.tsv`).

### Chapter 10.1 — Motion inside a volume: what each readout makes of it
A volume is not an instant: 0.76 s of slices for the 2D EPI, 0.2 s for a four-fold
multiband EPI, one excitation for a single-shot 3D readout, 18 s for the four-shot GRASE
and 36 s for the eight-interleaf stack of spirals. One timed motion trace (a 0.6 s nod
inside a readout, a step that stays, a slow drift) replayed on each readout with every 2D
shot or 3D segment at the pose of its own excitation time (aslscan P9/P10): staggered
slices and spin-history bands in 2D, groups of slices in multiband, replicas of the brain
along the phase-encode axis in the segmented GRASE, rings and smears outside the brain in
the spiral, nothing but a rigid displacement in the single-shot 3D; the per-volume view of
the same trace for comparison; why no rigid pose fits a staggered volume, what a
slice-by-slice registration recovers, and what is left to rejection, scored per readout
against the motion-free partner. *Pipeline:* `motion-readouts` (epi, epi-volume,
epi-spinhistory, mb4, grase-1shot, grase, spiral, with still partners), `readout-3d`.

### Chapter 10.2 — Registering an ASL series: contrasts, references, and what pipelines do
The contrasts in one series (control, label, M0, suppressed, each readout's own, white
matter brighter than gray in the 3D readouts); the four similarity measures and their cost
curves, least squares across contrasts (converges on the unsuppressed series, diverges
between a suppressed series and the M0); the spurious label-control offset of a common
cross-contrast reference and the offset step that removes it; the blur bias of an averaged
reference; the reference images scored on the drift run (control mean, mean of every
volume, M0 by NMI and by least squares, own-type templates with and without the offset
step, a second pass on the aligned mean, the mean ΔM); the zig-zag regression (perfect on a
still head, aliases spikes, halves a drift's within-pair step); the arterial signal at short
delays; what ASLPrep, ExploreASL, ASLtbx, BASIL/oxford_asl and the engines (MCFLIRT, SPM
realign, 3dvolreg, ANTs) do, and prospective correction. *Pipeline:* `ref-pcasl`, `bgsup`
(on), `readout-3d` (grase-4shot, spiral), `motion` (drift, random), `vascular`.

### Chapter 11 — Susceptibility distortion
Field map → displacement along phase encode ∝ Δf × TotalReadoutTime; pile-up and
stretching; where it hits ASL (frontal and temporal GM); blip-up/blip-down pairs and the
fieldmap-based unwarp (a didactic unwarp from the known field: shift each column by the
field × readout time, intensity-corrected by the Jacobian); a reverse-PE estimate of the
field; the 3D readouts under the same field map: a segmented GRASE whose effective echo
spacing is the line spacing over the segment count, so the displacement is a quarter of the
single-shot one, and the spiral, which blurs instead of displacing. *Pipeline:* `sdc` (ap,
pa, undistorted): distortion in the control image and in CBF; the unwarped result vs the
undistorted reference; `readout-3d` (grase-4shot-fieldmap, spiral-fieldmap).

### Chapter 12 — Partial volume effects
Voxels of 3.5 × 3.5 × 5 mm mix GM, WM, CSF; CBF in a mixed voxel is the fraction-weighted
mean; the GM-fraction dependence of measured CBF (the classic scatter); PV correction by
linear regression (Asllani 2008) and its smoothing cost; the truth: the exact tissue
fractions of every voxel from the pipeline. *Pipeline:* `ref-pcasl`, `voxel-sweep`: the
scatter of CBF vs GM fraction, PV-corrected GM CBF vs the true 60 and WM vs 20.

### Chapter 13 — Remaining artifacts and the assembled pipeline
Nyquist ghosts, k-space spikes, partial Fourier blurring (`readout` dataset), the
segmentation ghost of a 3D readout whose shots differ (physiological noise, `readout-3d`),
and how they look in ΔM; then the assembled pipeline: motion correction → (distortion correction) →
subtraction with outlier rejection → M0 calibration → CBF → PV correction → QC metrics
(tSNR, motion, negative CBF fraction, GM/WM ratio); how ASLPrep / BASIL / the ASL white
paper order these. *Pipeline:* `kitchen-sink` corrected end to end with the book's own
functions, scored against truth at each step.

---

## Part IV — Quantification

### Chapter 14 — CBF from a single delay
The white-paper formula and its assumptions (PLD > ATT, no outflow, T1' ≈ T1b); every
symbol traced to the sidecar; M0 calibration and λ; slice-timing correction in 2D; PASL's
formula; the bias when ATT exceeds PLD; the GM/WM contrast. *Pipeline:* `ref-pcasl`,
`label-types`, `pld-sweep`: CBF maps vs truth (the 4-panel), bias and RMSE in GM and WM,
per-slice bias without slice-timing correction.

### Chapter 15 — Multi-delay ASL: transit time and CBF together
Sampling the curve; fitting the kinetic model (grid search over ATT, linear in CBF); the
ATT map; the coupling between ATT and CBF errors; how many delays; the trade against
single-delay SNR; the arrival-time-insensitive alternatives (long LD/PLD; time-encoded,
pointer to Ch. 18). *Pipeline:* `multi-pld` (fit vs truth for CBF and ATT), `pld-sweep`.

### Chapter 16 — Calibration: M0, λ, T1, α, and the readout
Where M0 comes from (separate scan, included volumes, the control mean, estimated); the
M0 scan's TR and its T1 saturation; the T2 mismatch between blood and tissue at TE, and
what a finite exchange time does to it (pointer to the multi-echo data of Ch. 19); the
CSF reference method; sensitivity of CBF to λ, T1b, α, and to the assumed tissue T1; the
sidecar as the record of what was assumed. *Pipeline:* `m0-types`, `te-sweep`,
`label-types`.

### Chapter 17 — What your data allow
The decision table: single vs multi delay, background suppression, 2D vs 3D, M0 type, the
number of pairs, the PLD relative to the population's ATT → which quantities are estimable
and with what caveat; worked cases (an inherited single-PLD dataset without M0; a
multi-PLD dataset with 2D readout; a pediatric protocol).

---

## Part V — Advanced acquisitions

### Chapter 18 — Time-encoded (Hadamard) and Look-Locker ASL
Encoding sub-boli with a Hadamard matrix and decoding the multi-delay signal from one
series; Look-Locker readouts sampling the curve after one label; both at the toy tier
(`kinetic.delta_m` for each sub-bolus; the decoding matrix), with the SNR accounting; then
at the pipeline tier: the encoded series the simulator acquired (`sourcedata`), its decoding
in the page against the simulator's, the decoded noise against the prediction, the 2D
slice-dependence of the sub-bolus delays and the GRASE variant without it, the fit of CBF
and ATT scored against truth and against the conventional series of the same scan time; the
Look-Locker series: what each readout read (`deltamRead`) against the undepleted curve, the
tissue's driven steady state, and the three fits (ignore, T1_LL, exact) on real images.
*Pipeline:* `hadamard` (epi, grase, conventional), `look-locker`.

### Chapter 19 — Frontiers
Velocity-selective and vessel-encoded ASL (toy); multi-TE ASL and water exchange, now
measured on the `multi-te` dataset (the two-compartment decay of ΔM with TE, the exchanged
fraction fitted against the `deltamIntravascular` truth, the calibration bias of a
single-echo assumption); the macrovascular signal, measured on the `vascular` dataset
(uncrushed vs crushed vs none, scored against `deltamArterial`, the standard fit's bias);
dispersion (toy); a pointer to the 3D readouts of Ch. 2 and what remains unmodeled;
deep-learning denoising; and simulation as validation, with the table of what aslscan still
does not model. *Pipeline:* `multi-te`, `vascular`.

---

## Appendices
- **A. aslscan cookbook**: the protocol and overlay behind every dataset, rendered from
  `pipelines/config/datasets.yaml` through `aslbook.cookbook`.
- **B. Data manifest**: every file the notebooks load, rendered from the registry.
- **C. Software environment.**
- **D. Glossary.**
- **E. Ground-truth map catalogue** (`aslbook.truth.TRUTH_MAPS`).

## Dataset dependency summary

| Dataset id | Runs | What varies | Chapters |
|---|---|---|---|
| `ref-pcasl` | pcasl | nothing: the reference | 0, 2, 6, 8, 12, 13, 14, 16, 17 |
| `ref-clean` | pcasl | noise off | 5, 6, 8, 12, 14 |
| `label-types` | pcasl, casl, pasl | the labeling scheme | 4, 5, 14 |
| `pld-sweep` | pld05 … pld30 | the post-labeling delay (TR 6 s, 15 pairs) | 5, 7, 14, 15 |
| `multi-pld` | multipld | one series with six delays | 15, 17 |
| `noise-sweep` | sigma10, sigma40, sigma80, coils8-r2 | the noise level; coils + GRAPPA | 8 |
| `bgsup` | off, on, perfect, presat, on-slab, early-slab | background suppression, and the two suppression models | 9, 10 |
| `motion` | random, drift, random-bgsup | head motion | 10 |
| `sdc` | ap, pa, undistorted | field map + phase-encode polarity | 11, 13 |
| `voxel-sweep` | vox25, vox35, vox50 | in-plane voxel size | 7, 12 |
| `m0-types` | separate-tr8, separate-tr2, included | the calibration image | 16 |
| `te-sweep` | te12, te30, te60 | the echo time | 7, 16 |
| `readout` | pf68, ghost, spikes | readout imperfections | 13 |
| `kitchen-sink` | ap, pa | everything at once | 13 |
| `readout-3d` | grase-1shot, grase-4shot, grase-4shot-130, spiral, grase-4shot-bgsup, spiral-bgsup, grase-4shot-fieldmap, spiral-fieldmap, grase-4shot-physio, grase-4shot-shotmotion | the 3D readouts, alone and with suppression, a field map, physiology, shot motion | 2, 7, 9, 10, 11, 13 |
| `physio` | off, on | physiological noise without and with suppression | 8, 9 |
| `hadamard` | epi, grase, conventional | time-encoded labeling and its conventional twin | 18 |
| `look-locker` | pasl-ll | eight readouts after each pulsed label | 18 |
| `multi-te` | ge-pld10, ge-pld18, se-pld10 | three echoes per excitation, a finite exchange time | 16, 19 |
| `vascular` | none, uncrushed, crushed | the arterial compartment and vascular crushing | 5, 19 |
| `motion-readouts` | epi-still, epi, epi-volume, epi-spinhistory, mb4-still, mb4, grase-1shot, grase, spiral | one timed motion trace on four readouts, each shot at its own pose | 10 |
