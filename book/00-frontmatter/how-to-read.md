---
title: "0.1 How to read and run this book"
subtitle: Executable cells, data, environment, reproduction
kernelspec:
  name: python3
  display_name: Python 3
---

## Who this book is for

This book is for people who are new to MRI and need to work with arterial spin labeling
(ASL) data: to plan an acquisition, to judge whether a dataset supports the quantification
someone wants from it, to run a processing pipeline and read its quality report, or to
interpret a perfusion map someone else produced. It covers the physics only as far as the
physics explains what appears in the data and what can be done about it. Every artifact is
shown on simulated data whose correct answer is known, every correction is scored against
that answer, and every quantification is applied where its assumptions hold and where they
do not.

```{code-cell} python
:tags: [hide-cell]
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from aslbook.plotting import INK, PALETTE, set_style

set_style()
```

The chapters are meant to be read in order the first time, because each step of the chain
below depends on the one before it. Part I covers how the magnetization behaves and how an
image is made, Part II how blood is labeled and what signal the label produces, Part III
what goes wrong and how it is fixed, Part IV how the corrected images become a perfusion
map, and Part V what lies beyond a standard acquisition.

```{code-cell} python
:tags: [hide-input]
steps = [  # (part, label, chapters)
    ("I", "magnetization\nrelaxes with T1", "Ch 1"),
    ("I", "an EPI readout\nmakes an image", "Ch 2"),
    ("II", "arterial water\nis labeled", "Ch 3–4"),
    ("II", "the label arrives\nand decays", "Ch 5–7"),
    ("III", "artifacts, and\nhow to correct them", "Ch 8–13"),
    ("IV", "differences become\nperfusion maps", "Ch 14–17"),
    ("V", "beyond a standard\nacquisition", "Ch 18–19"),
]
parts = {"I": "how an image is made", "II": "how perfusion is measured", "III": "what goes wrong",
         "IV": "what is quantified", "V": "further"}
colors = dict(zip(parts, PALETTE[:5]))
W, GAP = 1.45, 0.28
fig, ax = plt.subplots(figsize=(11.8, 2.6))
ax.set(xlim=(-0.05, len(steps) * (W + GAP) - GAP + 0.05), ylim=(0, 2.25))
ax.set_axis_off()
xs = [i * (W + GAP) for i in range(len(steps))]
for i, (x, (part, label, ch)) in enumerate(zip(xs, steps)):
    c = colors[part]
    ax.add_patch(FancyBboxPatch((x, 0.35), W, 1.05, boxstyle="round,pad=0.02,rounding_size=0.1", fc=c + "22", ec=c, lw=1.4))
    ax.text(x + W / 2, 1.02, label, ha="center", va="center", fontsize=8.6, color=INK["primary"], linespacing=1.25)
    ax.text(x + W / 2, 0.52, ch, ha="center", va="center", fontsize=8, color=INK["secondary"])
    if i:
        ax.annotate("", xy=(x, 0.875), xytext=(x - GAP, 0.875),
                    arrowprops=dict(arrowstyle="-|>", color=INK["secondary"], lw=1.2, shrinkA=0, shrinkB=0))
for part, desc in parts.items():
    idx = [i for i, s in enumerate(steps) if s[0] == part]
    x0, x1 = xs[idx[0]], xs[idx[-1]] + W
    ax.plot([x0, x1], [1.62, 1.62], color=colors[part], lw=2.5, solid_capstyle="butt")
    ax.text((x0 + x1) / 2, 1.72, f"Part {part}\n{desc}", ha="center", va="bottom", fontsize=8.6, color=INK["primary"], linespacing=1.2)
fig.tight_layout()
```

Each box is one step from the scanner to a finished perfusion map, and the arrows give the
order in which the book takes them. [Chapter 17](../04-quantification/17-what-your-data-allow.md)
collects the requirements of everything before it into one decision table and is the page
to return to.

## Executable cells

Every figure and every number in the text is produced by code in the page. The code is
collapsed by default; each code cell has a *Source* toggle that shows it. Reading the book
does not require reading the code, but the code is short and deliberately literal, and it
is the exact procedure behind each figure. Cells that only set up imports are hidden
entirely.

Each chapter opens with a box listing the simulated datasets it uses, with links to their
descriptions in [Appendix A](../appendices/a-aslscan-cookbook.md#app-a-datasets). It then
follows one structure: *learning goals*, the physics, a *See it* section that produces the
figures, a *Measure it* section that reports a number against the known answer, *What this
implies for acquisition*, and further reading. In Part III the middle sections follow the
artifact template: the physics of the artifact, the simulator setting that produces it, the
artifact-free reference, the correction step by step, the residual against truth, and the
acquisition choices that reduce it.

## Two kinds of simulation

The figures come from two sources, and every chapter says which.
[Chapter 0.2](./the-simulated-datasets.md) describes each in full.

- **Toy tier.** The kinetic model and the longitudinal signal equations, evaluated in the
  page on the packaged slab of the simulated brain, together with small simulations of
  spins, k-space, and noise. These run in seconds when the book is built. The toy tier is
  the simulator's signal stage without its scanner: it reproduces the simulator's own
  difference signal in every voxel, which [Chapter 5](../02-labeling/05-kinetic-model.md)
  verifies.
- **Pipeline tier.** Full aslscan simulations of the same slab, with the k-space, noise,
  coils, distortion, suppression pulses, and head motion of a real acquisition, made
  offline by the repository's pipeline and downloaded by the pages that use them. Fourteen
  datasets, listed in [Chapter 0.2](./the-simulated-datasets.md#the-datasets).

Both are reproducible: the toy tier from the repository alone, the pipeline tier from the
repository plus the simulator and its phantom, which [Appendix A](../appendices/a-aslscan-cookbook.md)
documents run by run.

## Running the book yourself

The repository holds the pages, the helper package `aslbook`, and the pipeline. To execute
any chapter locally:

```bash
git clone https://github.com/PennLINC/asl-book
cd asl-book
micromamba create -n aslbook -f environment.yml
micromamba run -n aslbook pip install -e .
OMP_NUM_THREADS=1 micromamba run -n aslbook myst start --execute
```

The last command serves the book at a local address and re-executes a page whenever it is
edited. The chapters can also be opened as notebooks in JupyterLab, since each page is a
MyST Markdown notebook. [Appendix C](../appendices/c-software-environment.md) lists the
versions of every package the published build used.

The pipeline-tier datasets are fetched on first use by the pages that need them and cached;
to build against a local pipeline output instead, set `ASLBOOK_DATA` to its directory
([Appendix B](../appendices/b-data-manifest.md)).

## Reproducing the simulations

[Appendix A](../appendices/a-aslscan-cookbook.md) shows, for every dataset, the BIDS sidecar
and `aslcontext.tsv` the simulator read, the overlay of settings that BIDS does not record,
and the command line, all rendered from the pipeline's own configuration. Each dataset
directory carries a `provenance.json` with the same information plus the simulator version
and commit. The pipeline itself is a Snakemake workflow in the repository's `pipelines`
directory; it requires the aslscan binary and the phantom, which are distributed separately
from the book.

## Conventions

This section is reference material. It uses terms that Chapters 1 through 6 define, and it
will make sense by the end of [Chapter 6](../02-labeling/06-the-asl-signal.md); come back to
it then.

- Spelling is American; the tissue colors are fixed throughout (gray matter orange, white
  matter blue, CSF aqua); CBF maps use one color scale from 0 to 90 ml/100 g/min; signed
  difference maps use a diverging scale centered on zero.
- Images of the simulated brain are axial slices with anterior at the top, through the
  lateral ventricles unless the point needs another level. The phase-encode axis of every
  simulated acquisition is anterior-posterior, so distortion and ghosts move things up and
  down in these pictures.
- Units: perfusion in ml/100 g/min, times in seconds (TE in ms where stated), field offsets
  in Hz, displacements in voxels or mm as labeled, image intensities in the simulator's
  arbitrary units, in which a gray matter control voxel is about 6200 and its label-control
  difference about 30. The notation page (0.3) lists every symbol.
- The reference protocol is the book's 2D pseudo-continuous ASL acquisition: labeling
  duration 1.8 s, post-labeling delay 1.8 s, TR 4.5 s, TE 12 ms, 3.5 × 3.5 × 5 mm voxels, 20
  slices 40 ms apart, 30 control-label pairs, a separate M0 scan at TR 8 s. It follows the
  ASL white paper's recommendations for 2D acquisitions within what the simulator models,
  and every other dataset changes one thing at a time from it.
