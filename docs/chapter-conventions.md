# Chapter conventions

The notebook standard every page follows. Modeled on `../diffusion-book` (read one of its
chapters, e.g. `book/03-preprocessing/08-noise.md`, for the voice).

## File

One MyST Markdown notebook per chapter under `book/<part>/<nn>-<slug>.md`, with the front
matter:

```
---
title: "5. The general kinetic model"
kernelspec:
  name: python3
  display_name: Python 3
---
```

## Structure

1. **Datasets box** right after the front matter:

   ```
   :::{admonition} Simulated datasets in this chapter
   :class: note
   - **Built in this page:** <what the toy tier computes here> ([Appendix B](../appendices/b-data-manifest.md#app-b-package-data)).
   - **`pld-sweep`**: <one line> ([Appendix A](../appendices/a-aslscan-cookbook.md#ds-pld-sweep)).

   Pipeline-tier datasets are simulated offline by aslscan ([Chapter 0.2](../00-frontmatter/the-simulated-datasets.md)).
   :::
   ```

2. `## Learning goals` — "After this chapter you can:" + 3–5 bullets.
3. A hidden setup cell (`:tags: [hide-cell]`) with the imports and `set_style()`.
4. The physics, with MyST math (`$$ ... $$`, inline `$...$`), plain language first, every
   symbol from the notation page. Use `:::{dropdown} ...` for detail a first reader can skip,
   `:::{admonition} ... :class: note` for asides.
5. `## See it: ...` sections with figures; `## Measure it: ...` with at least one number
   against the truth where a truth exists (print it, and quote it in the prose that follows).
6. `## What this implies for acquisition` — 3–6 bullets.
7. `## Further reading` — `{cite:p}` / `{cite:t}` keys from `book/references.bib`.

Part III chapters use the artifact template: physics → the aslscan setting that produces it
(name the overlay key or sidecar field) → the artifact-free reference → the correction step
by step → residual vs truth → acquisition choices.

## Code cells

- ```` ```{code-cell} python ```` with `:tags: [hide-input]` on every cell (code collapsed,
  output shown); `:tags: [hide-cell]` for setup; no tag when the code itself is the lesson.
- Minimal, readable, top-level; helpers from `aslbook`. No cell over ~40 lines; none over
  ~60 s. Seed every random draw.
- Print the numbers the prose quotes with f-strings; the prose then repeats them (a reader
  who does not open the code must still get the number).
- Figures: `fig.tight_layout()` last; sizes about `(7, 3.2)` for one panel, `(11, 3.2)` for
  four; titles short; axis labels with units. Colors from `PALETTE` / `TISSUE_COLORS`.
- Every figure is followed by prose that says what to look at and what it shows.
- Never load a whole 4-D pipeline series when a slice or a mean suffices for the point, but
  do not hesitate to load it when needed (≈20 MB each, fast).

## Orientation and display

- Volumes are `(x, y, z[, volume])` in the phantom's frame (x left-right, y
  posterior-anterior, z inferior-superior). Show axial slices with
  `plotting.show_slice(ax, vol, k)` or `plotting.take_slice(vol, k)` (rot90: anterior up).
- The display slice is `phantom.DISPLAY_SLICE` (9). Use it unless the point needs another.
- CBF maps: `kind="cbf"` (inferno, 0–90 ml/100 g/min). ΔM maps: `kind="magnitude"` with an
  explicit `vmin=0, vmax=...`, or `kind="diff"` when signed. Fractions: `kind="fraction"`.
- The 4-panel comparison is `plotting.fit_vs_truth(est, truth, mask, "CBF", unit="ml/100 g/min")`.

## The two tiers in code

Toy tier (in the page):

```python
from aslbook import kinetic, phantom, presets, protocols, quant, synth
p = protocols.pcasl(n_pairs=30)            # the reference protocol's sidecar
s = synth.series(p, noise_sd=40, seed=0)   # SimSeries: .mag (x,y,z,v), .ctx, .deltam, .m0scan
d = quant.subtract(s.mag, s.ctx)           # (x,y,z,pairs) control - label
```

Pipeline tier:

```python
from aslbook import data
run = data.load_dataset("ref-pcasl").run()            # or .run("sigma40") for multi-run datasets
mag, ctx, p = run.mag(), run.context(), run.sidecar()  # images, rows, BIDS sidecar (+ AslscanSimulation)
m0 = run.m0scan(); truth_cbf = run.truth("perfusion"); dm_gt = run.truth("deltam")
fr = run.fractions(); mask = run.mask()               # GM/WM/CSF fraction of each voxel
```

Image units: aslscan scales by `signal_scale = 100`, so a GM control voxel is ≈ 6200 and a
GM ΔM ≈ 37 in the first slice at the reference timing (0.396 M0 units × 100 × the blood T2
factor 0.93), falling to ≈ 21 in the last slice and averaging ≈ 30 over the slab; the
ground-truth `deltam` is in M0 units; `m0scan` is in image units and carries the M0 scan's
saturation and T2 factors.

The quantities the phantom makes exact: GM CBF 60, WM 20 ml/100 g/min; GM ATT 0.8, WM 1.2 s.
The per-voxel truth from `run.truth("perfusion")` is the fraction-weighted mean.

## Voice

Plain declarative sentences. American spelling. Define every term where it first appears.
No "we"; address the reader as "you" sparingly. No marketing adjectives. Say what a figure
shows, then what it means, then what to do about it. Quote numbers the code printed.
Where the simulator's model is an approximation (background suppression's global bolus; no
macrovascular compartment; motion applied to finished images), say so where it matters.
