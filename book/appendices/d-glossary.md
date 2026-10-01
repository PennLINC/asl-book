---
title: "Appendix D: Glossary"
kernelspec:
  name: python3
  display_name: Python 3
---

Terms as they are used in this book, with the chapter that introduces each. Every entry
starts with a plain-language sentence; the sentences after it give the technical meaning.
Symbols are those of the [notation page](../00-frontmatter/notation.md); chapter numbers
link to the front matter (0.x), the chapters (1 to 19), and the appendices (A to E).

| Term | Meaning | Chapter |
|---|---|---|
| Adiabatic inversion | Flipping the magnetization with a pulse whose frequency sweeps through resonance, so that the flip does not depend on the exact pulse amplitude. Continuous labeling uses a flow-driven version: blood moving through a gradient sees the sweep as it crosses the labeling plane {cite:p}`williams1992`. | 4 |
| Amplitude-modulated control | The control condition of continuous labeling. The labeling pulse is modulated so that it creates two closely spaced inversion planes that cancel, leaving the blood uninverted while depositing the same radiofrequency power, and so the same magnetization transfer, as the label {cite:p}`alsop1998`. | 4 |
| Arterial blood volume (aBV) | The fraction of a voxel occupied by arterial blood. It sets the size of the macrovascular signal in a model that includes one; Chapter 19 simulates values from 0 to 2 %. | 19 |
| Arterial input function | The concentration of label in the arteries feeding a voxel as a function of time. The kinetic model assumes a box; QUASAR measures it from the crushed and uncrushed difference {cite:p}`petersen2006`. | 5, 18 |
| Arterial transit time (ATT, δ) | How long labeled blood takes to travel from the labeling region to the tissue of a voxel. Longer in white matter, in the elderly, and in cerebrovascular disease {cite:p}`alsop2015,woods2024`; the phantom's values are 0.8 s (gray matter) and 1.2 s (white matter). | 3, 5 |
| ASLDRO | The digital reference object whose brain this book simulates {cite:p}`olivertaylor2021`. Its `hrgt_icbm_2009a_nls_3t` phantom gives one perfusion, transit time, T1, T2, T2*, and M0 per tissue class. | 0.2 |
| aslscan | The headless ASL simulator behind the pipeline tier. It takes a BIDS protocol and the phantom through the kinetic model, the signal equations, and a 2D spin-echo EPI acquisition, and writes the ground-truth maps beside the series. | 0.2, A |
| `aslcontext.tsv` | The table that says what each volume of an ASL series is. One row per volume with `volume_type` of `control`, `label`, `m0scan`, or others; the subtraction is impossible without it. | 6 |
| Background suppression | Inversion pulses timed so that the static tissue signal is near zero at the readout while the label survives {cite:p}`dixon1991,ye2000`. It reduces the motion and physiological fluctuations that scale with the static signal; it costs the M0 information of the suppressed volumes and flips the sign of the label with each pulse. | 9 |
| BASIL | The FSL {cite:p}`smith2004` toolbox for Bayesian inference of the kinetic model {cite:p}`chappell2009`, including multi-delay fits with dispersion and macrovascular components, and partial-volume correction with a spatial prior {cite:p}`chappell2011,groves2009`. | 12, 13, 15 |
| BIDS | A standard way of naming and organizing neuroimaging files. The Brain Imaging Data Structure {cite:p}`gorgolewski2016`: for ASL {cite:p}`clement2022`, a `_asl.nii.gz` series, its `_aslcontext.tsv`, an optional `_m0scan`, and a JSON sidecar with the labeling and readout parameters. | 2, 6 |
| Blip-up/blip-down | Two acquisitions distorted in opposite directions along the phase-encode axis, from which the susceptibility field can be estimated {cite:p}`andersson2003`. | 11 |
| Bolus | The parcel of labeled blood that one labeling period creates. For (P)CASL it is as long in time as the labeling duration; for PASL its duration is set by the slab and the flow unless a cut-off fixes it. | 4, 5 |
| Bolus cut-off | A saturation pulse applied to the labeling slab after a fixed time, which gives a pulsed label a known bolus duration. QUIPSS II uses one pulse {cite:p}`wong1998`, Q2TIPS a train of thin ones {cite:p}`luh1999`. | 4 |
| Buxton model | See general kinetic model. | 5 |
| Calibration | Turning ΔM into CBF in absolute units by dividing by the equilibrium magnetization of blood, obtained from an M0 image and the partition coefficient. | 14, 16 |
| CASL | Continuous arterial spin labeling: a long radiofrequency pulse and a gradient invert blood as it flows through a plane for one to a few seconds {cite:p}`detre1992,williams1992`. Efficiency about 0.68 with the amplitude-modulated control at 3 T {cite:p}`wang2005,wu2007`; it deposits much power and needs a control that matches its magnetization transfer. | 4 |
| CBF (f) | Cerebral blood flow: how much blood the tissue receives per unit time, in ml of blood per 100 g of tissue per minute. Gray matter values from 40 to 100 can be normal {cite:p}`alsop2015`, and white matter is about 20 in PET {cite:p}`leenders1990`; the phantom's values are 60 and 20. | 3 |
| Control image | The image acquired without labeling, against which the label image is subtracted. It must match the label image in everything but the blood's magnetization, including magnetization transfer and suppression. | 4, 6 |
| CSF reference | Using the cerebrospinal fluid signal, whose proton density is known relative to blood, to calibrate M0 when no separate M0 scan exists {cite:p}`chalela2000`. | 16 |
| ΔM | The label-control difference of the longitudinal magnetization, the perfusion-weighted signal. About 1 % of the static signal; unsigned in this book's truth maps; in image units in the series and in M0 units in the ground truth. | 6 |
| `deltam` (ground truth) | The noise-free label-control difference aslscan writes for every labeled volume, `desc-deltam_gt`, in the M0 map's units, before the readout. The toy tier reproduces it to float32 precision. | 0.2, E |
| Dispersion | The spreading of the bolus's edges along its path, from the spread of velocities within and between vessels. Modeled by convolving the delivery with a kernel {cite:p}`hrabe2004,chappell2013`; it biases the fitted transit time later. | 5, 19 |
| Echo time (TE) | The time from the excitation to the center of the readout. The signal decays with T2 (spin echo) or T2* (gradient echo) over it; 12 ms in the reference protocol. | 1, 7 |
| EPI | Echo-planar imaging: a readout that collects a whole 2D image after one excitation by zigzagging through k-space in tens of milliseconds {cite:p}`mansfield1977`. It is fast and sensitive to field offsets along the phase-encode axis. | 2 |
| EPISTAR | The first pulsed labeling scheme: a slab below the imaging region is inverted; the control inverts a slab above it {cite:p}`edelman1994`. | 4 |
| Exchange (of water) | Labeled water moving from the capillaries into the tissue. The kinetic model assumes it is instantaneous; it takes a few hundred milliseconds {cite:p}`stlawrence2000,parkes2002`, and the label still in blood decays with the blood's T2 at the echo time. | 19 |
| FAIR | Flow-sensitive alternating inversion recovery: a pulsed scheme whose label is a non-selective inversion and whose control is a slice-selective one, so that only inflowing blood differs {cite:p}`kim1995`. | 4 |
| Field map | A map of how far the magnetic field departs from its nominal value at each point, in Hz. The EPI displacement is computed from it {cite:p}`jezzard1995`; the phantom's synthetic one has lobes above the frontal sinuses and at the temporal bones. | 11 |
| General kinetic model | The model of {cite:t}`buxton1998` that gives ΔM as the label delivered by the flow, minus what has relaxed and what has flowed out. Three phases: not arrived (t ≤ ATT), arriving, and arrived; separate solutions for pulsed and continuous labels. | 5 |
| GRAPPA | Parallel imaging that fills in skipped lines of k-space from neighboring lines across coils, with weights fitted on fully sampled calibration lines {cite:p}`griswold2002`. It shortens the readout and changes the noise distribution. | 2, 8 |
| GRASE | A 3D readout that follows a slab excitation with a train of spin echoes, each carrying an EPI readout of one through-plane partition of k-space {cite:p}`gunther2005,fernandezseara2005`. T2 decay across the train blurs the image through-plane, so the train is segmented. | 19 |
| Hadamard encoding | Switching each sub-bolus of one labeling period between label and control according to a row of a Hadamard matrix, so that one series carries every delay and a linear decoding recovers each sub-bolus's signal {cite:p}`gunther2007,dai2013`. | 18 |
| Image units | The intensity scale of the simulated series: aslscan multiplies the signal by `signal_scale` = 100, so a gray-matter control voxel is about 6200 and its ΔM at the reference timing about 37 in the first slice and about 30 averaged over the slab. | 0.2, 6 |
| Inversion efficiency | See labeling efficiency. | 4 |
| Inversion recovery | The return of the longitudinal magnetization from −M0 to M0 after a 180° pulse, `M0 (1 − 2 e^(−t/T1))`. The label's difference from equilibrium decays this way. | 1 |
| k-space | The form in which the scanner records an image: one Fourier coefficient per sample {cite:p}`ljunggren1983,twieg1983`. The image is its inverse Fourier transform; its extent sets the resolution and its sampling the field of view. | 2 |
| Label image | The image acquired with the blood inverted, whose subtraction from the control image gives ΔM. | 4, 6 |
| Labeling duration (LD, τ) | How long the (P)CASL labeling lasts, and therefore the bolus duration. 1.8 s in the reference protocol and the white paper. | 4, 7 |
| Labeling efficiency (α) | The fraction of the ideal inversion that the labeling achieves, including the imperfect inversion and the label lost to any suppression pulses. About 0.85 for PCASL and 0.98 for PASL {cite:p}`alsop2015`, 0.68 for CASL {cite:p}`wang2005,wu2007`; a sidecar records it as `LabelingEfficiency`. | 4 |
| Labeling plane | The plane, perpendicular to the feeding arteries in the neck, through which blood flows to be inverted in continuous and pseudo-continuous labeling. | 4 |
| Look-Locker | Reading the curve many times after one label with a train of low-flip-angle excitations {cite:p}`gunther2001`. Each readout consumes part of the label, which the quantification must model. | 18 |
| M0, M0 scan | The equilibrium magnetization of a voxel, and the image that measures it: a long-TR acquisition without labeling or suppression. CBF calibration divides ΔM by the blood's M0, which the tissue's M0 gives through the partition coefficient. | 6, 16 |
| Macrovascular signal | ΔM from labeled blood still in arteries within the voxel, not from perfusion of that voxel. Large at short delays; removed by vascular crushing, avoided by a long delay, or fitted with an arterial compartment {cite:p}`chappell2010`. | 5, 19 |
| Magnetization transfer | Signal loss in tissue caused by radiofrequency power applied off resonance, through the exchange between free and bound water. Continuous labeling produces it; the control must produce the same amount or the subtraction is wrong. | 4 |
| MP-PCA | Marchenko-Pastur principal component analysis: a denoising method that, in each small patch of a series, keeps the principal components whose variance exceeds the range random-matrix theory predicts for pure noise, and estimates the noise level in the same step {cite:p}`veraart2016`. | 8 |
| Multi-delay (multi-PLD) | Acquiring at several post-labeling delays so that the kinetic curve can be fitted for both CBF and the transit time {cite:p}`woods2024`. Also multi-TI for pulsed labels. | 15 |
| Multi-TE ASL | Acquiring ΔM at several echo times so that the label in blood and in tissue, whose T2 differ (165 and 80 ms in this book's simulations), can be told apart, which measures the exchange {cite:p}`gregori2013`. | 19 |
| NORDIC | Noise reduction with distribution corrected PCA: patch-wise low-rank denoising of the complex images after normalizing them by the noise (g-factor) map, with the threshold set from the known noise level rather than estimated {cite:p}`moeller2021,vizioli2021`. | 8 |
| Nyquist ghost | A copy of the image shifted by half the field of view along the phase-encode axis, from a mismatch between the odd and even lines of an EPI readout. | 13 |
| Outlier rejection | Dropping the control-label pairs whose difference image is inconsistent with the others, usually because of motion {cite:p}`tan2009,dolui2017`. | 10, 13 |
| Overlay (aslscan) | The per-dataset settings the pipeline gives the simulator on top of the phantom and the protocol: the noise level, motion, field map, readout imperfections, and suppression options that make each dataset of Appendix A. | A |
| Partial Fourier | Skipping part of k-space on one side and filling it in from the other by symmetry. It shortens the echo time at the cost of some SNR and sharpness. | 2, 13 |
| Partial volume | A voxel containing more than one tissue, so that its CBF is the fraction-weighted mean of the tissues' CBF. Large at the reference voxel size; the pipeline's tissue fractions make the true mixture known. | 12 |
| Partition coefficient (λ) | How much water a gram of tissue holds relative to a milliliter of blood, 0.9 ml/g for the whole brain {cite:p}`herscovitch1985`. It converts the tissue's M0 into the blood's in the calibration. | 3, 14 |
| PASL | Pulsed arterial spin labeling: a slab of blood is inverted at once with a short pulse. High efficiency and low power; a bolus duration that a cut-off must fix. | 4 |
| PCASL | Pseudo-continuous arterial spin labeling: a train of short pulses with a gradient that inverts blood flowing through a plane, as continuous labeling does, without a continuous transmitter {cite:p}`dai2008`. Efficiency about 0.85; the white paper's recommended scheme {cite:p}`alsop2015`. | 4 |
| Phase-encode direction | The image axis that EPI samples slowly, along which field offsets displace and ghosts appear. `j-` (anterior to posterior) in the reference protocol. | 2, 11 |
| PICORE | Proximal inversion with control for off-resonance effects: a pulsed scheme whose control applies the inversion pulse off resonance without a gradient, matching the label's magnetization transfer {cite:p}`wong1997`. | 4 |
| Pipeline tier | The datasets simulated offline by aslscan with its k-space acquisition, fetched at build time. Used for realistic images, artifacts, and quantification against truth. | 0.1, 0.2 |
| Post-labeling delay (PLD, w) | The time from the end of labeling to the readout, during which the label travels to the tissue {cite:p}`alsop1996`. It must exceed the transit time for the single-delay formula to hold; 1.8 s in the reference protocol. | 4, 5, 7 |
| Presaturation | Saturating the imaging region before labeling so that the tissue magnetization starts from zero, which makes the suppression timeline independent of the previous readout. | 9 |
| Q2TIPS | A bolus cut-off made of a train of thin saturation pulses at the distal end of the labeling slab, which cuts the bolus more sharply than QUIPSS II {cite:p}`luh1999`. | 4 |
| QEI | Quality evaluation index: a single number between 0 and 1 for a CBF map, combining its similarity to the map expected from the tissue segmentation, its spatial variability, and its share of negative gray-matter voxels {cite:p}`dolui2017qei`. | 13 |
| QUASAR | A Look-Locker pulsed acquisition acquired with and without vascular crushing, whose difference measures the arterial input function so that CBF follows by deconvolution without a kinetic model {cite:p}`petersen2006`. | 18 |
| QUIPSS II | A single saturation pulse applied to the labeling slab after a fixed time, which fixes the pulsed bolus duration at that time {cite:p}`wong1998`. | 4 |
| Readout | The part of the sequence that acquires the image after the label has arrived: 2D EPI slice by slice in this book's simulations, 3D GRASE or a stack of spirals in the field's practice. | 2, 6, 19 |
| Rician noise | The distribution of the noise in a magnitude image: the magnitude of complex Gaussian noise, which does not average to zero where the signal is weak {cite:p}`gudbjartsson1995`. The difference of two high-SNR magnitude images is nearly Gaussian. | 2, 8 |
| Saturation recovery | The regrowth of the longitudinal magnetization from zero after a 90° pulse, `M0 (1 − e^(−TR/T1))`. It sets the static signal of every volume and the T1 saturation of the M0 scan. | 1, 16 |
| SCORE | Structural correlation-based outlier rejection: first drops the CBF volumes whose mean gray-matter value is more than 2.5 median absolute deviations from the median, then repeatedly removes the volume most correlated with the mean map for as long as the pooled within-tissue variance of the mean map keeps falling {cite:p}`dolui2017`. | 10 |
| SCRUB | Structural correlation with robust Bayesian estimation: a voxelwise robust mean of the volumes SCORE kept, in which each volume is down-weighted where it departs from the others and the estimate is drawn toward a prior from the tissue's mean perfusion {cite:p}`dolui2016`. | 10 |
| Sidecar | The JSON file beside a BIDS series that records the acquisition: labeling type, duration, delays, suppression, readout timing, M0 type, and what was assumed. Every symbol in the CBF formula traces to it. | 6, 14 |
| Signal time (t) | The kinetic model's clock: seconds from the start of labeling to a slice's readout, PLD + τ for (P)CASL and the inversion time for PASL, plus the slice's offset in a 2D readout. | 5, 6 |
| Slice timing | The delay of each slice of a 2D readout relative to the first: 40 ms per slice here, so the last of 20 slices is read 0.76 s later and at a longer effective delay. It must enter the quantification. | 6, 14 |
| SNR, temporal SNR | Signal-to-noise ratio: the signal divided by the noise standard deviation. Temporal SNR of the difference series: its mean over pairs divided by its standard deviation over pairs, per voxel. | 6, 8 |
| Spin echo | Signal that returns when a 180° pulse reverses the dephasing from static field differences, so that the decay to the echo is T2 rather than T2* {cite:p}`hahn1950`. The book's simulated readout is a spin-echo EPI. | 1, 2 |
| Stack of spirals | A 3D readout whose through-plane partitions are each read with a spiral trajectory in k-space; the same partition train as GRASE with the spiral's own off-resonance blurring. One of the two segmented 3D readouts the white paper recommends {cite:p}`alsop2015`. | 19 |
| Sub-bolus | One block of a time-encoded labeling period, labeled or control according to the Hadamard matrix; its own short (P)CASL bolus with its own delay. | 18 |
| Susceptibility distortion | Warping of EPI images near air-tissue boundaries: a displacement along the phase-encode axis proportional to the field offset times the total readout time. | 11 |
| T1, T1 of blood (T1b), T1' | How quickly the longitudinal magnetization regrows; the label's difference from equilibrium decays with it. T1b is 1.65 s at 3 T {cite:p}`lu2004`, the decay of the label in transit; T1' is the tissue T1 shortened by the outflow, `1/T1' = 1/T1 + f/λ`, which hardly differs from T1. | 1, 3, 5 |
| T2, T2* | How quickly the transverse signal decays through spin interactions (T2, recovered by nothing) and through static field differences as well (T2*, shorter, recovered by a spin echo). In this book's simulations at 3 T, blood 165 ms and gray matter 80 ms; the blood value depends on oxygenation and hematocrit {cite:p}`zhao2007`. | 1 |
| Territory mapping | Assigning each voxel's perfusion to the artery that delivered it, by vessel-encoded {cite:p}`wong2007` or super-selective {cite:p}`helle2010` labeling. | 19 |
| Time-encoded ASL | A (P)CASL acquisition whose labeling period is split into sub-boli encoded with a Hadamard matrix, so that one series yields a multi-delay measurement at the SNR of averaging {cite:p}`dai2013,teeuwisse2014`. | 18 |
| Total readout time | The duration of the EPI train that sets the susceptibility displacement: the effective echo spacing times the number of phase-encode lines minus one, `TotalReadoutTime` in the sidecar. | 2, 11 |
| Toy tier | The kinetic model and the signal equations evaluated in the page on the packaged phantom slab (`aslbook.synth`), without k-space. Used for every curve and for parameter sweeps. | 0.1, 0.2 |
| TR | The repetition time: the interval between successive labelings, which must hold the labeling duration, the delay, and the whole readout. 4.5 s in the reference protocol. | 1, 7 |
| Truth maps | The known answers the simulator writes beside each dataset: perfusion, transit time, T1, T2, T2*, M0, the tissue fractions, and the noise-free `deltam`. | 0.2, E |
| Vascular crushing | Bipolar gradients before the readout that dephase moving spins, removing the label still in arteries at the cost of some perfusion signal. `VascularCrushing` in the sidecar. | 4, 19 |
| Velocity-selective ASL | Labeling defined by velocity rather than position: a module saturates or inverts spins moving faster than a cutoff wherever they are, so the label is created in the arterioles of every voxel and the transit time drops out {cite:p}`wong2006,qin2016`. | 19 |
| Vessel-encoded ASL | PCASL with a gradient across the labeling plane that labels some arteries and not others in each cycle, so that a decoding gives each artery's territory {cite:p}`wong2007`. | 19 |
| Voxel | The three-dimensional pixel of an MRI image, 3.5 × 3.5 × 5 mm in the reference protocol. Its signal is the sum of everything inside it. | 2, 12 |
| White paper | The consensus recommendations for clinical ASL {cite:p}`alsop2015`: PCASL, LD 1.8 s, PLD 1.8 to 2.0 s, a segmented 3D readout with background suppression, a separate M0 scan, and the single-delay CBF formula. Its multi-timepoint successor is {cite:t}`woods2024`. | 7, 14 |

```{code-cell} python
:tags: [remove-cell]
import aslbook
assert aslbook.__version__
```
