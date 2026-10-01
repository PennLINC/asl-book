---
title: Arterial spin labeling, executed
subtitle: An executable book on perfusion MRI with arterial spin labeling, from spins to cerebral blood flow
kernelspec:
  name: python3
  display_name: Python 3
---

Every figure in this book is produced by code you can run, and every perfusion estimate is
scored against a known answer. The answer exists because the brain in these pages is
simulated: a digital phantom with one perfusion rate, transit time, T1, T2, and equilibrium
magnetization per tissue passes through **aslscan**, a headless arterial spin labeling (ASL)
simulator that evaluates the kinetic model in every voxel, forms the label and control
images, and acquires them with a model of a 2D spin-echo echo-planar scanner, k-space and
all. The same phantom yields the ground-truth maps that no scanner can provide.

:::{admonition} Status
:class: note
All chapters and appendices are written and execute at both tiers: the toy tier, in which
the kinetic model and the signal equations are evaluated in the page on the packaged phantom
slab, and the pipeline tier, in which full aslscan simulations of the same slab are loaded
from the data release. The datasets are simulated by the repository's offline pipeline
([Appendix A](appendices/a-aslscan-cookbook.md)).
:::

## What is in the book

| Part | Chapters | What you will be able to do |
|---|---|---|
| Front matter | 0.1–0.3 | run the book, know the simulated datasets, read the notation |
| I. MRI physics for perfusion imaging | 1–2 | follow the longitudinal magnetization through saturation and inversion, and know what an EPI readout does to an image |
| II. Perfusion and its labeling | 3–7 | define perfusion, label arterial water three ways, predict the difference signal with the kinetic model, read an ASL series, and choose the delay, duration, TR, voxel, and number of pairs |
| III. Artifacts and preprocessing | 8–13 | recognize noise, background suppression, head motion, susceptibility distortion, partial volume effects, and readout imperfections in simulated data, correct them, and measure what remains |
| IV. Quantification | 14–17 | compute CBF from a single delay, fit transit time and CBF from several delays, calibrate with M0 and the assumed constants, and decide what a given acquisition allows |
| V. Advanced acquisitions | 18–19 | reason about time-encoded and Look-Locker sampling, and where the field is heading |

## Two kinds of simulation

Small **toy** simulations (the kinetic model, the longitudinal signal equations, a Bloch
equation, the k-space of one slice) are written in the notebook and run when the book
builds; those that need a brain use the packaged slab of the simulated brain, and they
reproduce the simulator's own difference signal voxel for voxel. Brain-level **pipeline**
data come from full aslscan runs made offline and versioned; notebooks download them.
[Chapter 0.2](00-frontmatter/the-simulated-datasets.md) explains both.

```{code-cell} python
:tags: [hide-input]
import aslbook
from aslbook import data

print(f"aslbook {aslbook.__version__}; registered datasets: {', '.join(data.registered_datasets()) or 'none yet'}")
```
