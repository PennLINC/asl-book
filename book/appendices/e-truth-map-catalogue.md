---
title: "Appendix E: Ground-truth map catalogue"
subtitle: What the simulator writes next to every series, and which chapter uses each
kernelspec:
  name: python3
  display_name: Python 3
---

aslscan writes the answer key next to every simulated series, under
`sub-01/perf/ground-truth/`, resampled from the 1 mm phantom to the acquisition grid by the
rules of `aslbook.grid`: the volume-weighted mean of each phantom quantity over the phantom
voxels an acquisition voxel overlaps, a majority vote for the labels, and for the transit
time the mean over the perfused tissues only. The pipeline adds the tissue fractions of every
acquisition voxel. The catalogue below is rendered from `aslbook.truth.TRUTH_MAPS`.

```{code-cell} python
:tags: [hide-cell]
from aslbook import truth
```

```{code-cell} python
:tags: [hide-input]
print(f"{'map':<20} {'units':<12} {'chapters':<20} definition")
for name, (units, definition, chapters) in truth.TRUTH_MAPS.items():
    print(f"{name:<20} {units:<12} {chapters:<20} {definition}")
```

## File names

| Map | File |
|---|---|
| perfusion, att, T1map, T2map, M0map, dseg | `sub-01_desc-<map>_gt.nii.gz` with a JSON sidecar giving the units and the resampling rule |
| deltam, deltamStatic | `sub-01_desc-deltam_gt.nii.gz` (4-D, one volume per series row), `sub-01_desc-deltamStatic_gt.nii.gz` in motion runs |
| motion | `sub-01_desc-motion_gt.tsv`: `volume`, `trans_x/y/z` (mm), `rot_x/y/z` (radians), and any within-volume events |
| tissue fractions | `derivatives/aslbook/sub-01/perf/sub-01_label-{GM,WM,CSF}_probseg.nii.gz` |

## What the deltam truth is, exactly

For every `label` (and `deltam`) row the simulator evaluates the general kinetic model
{cite:p}`buxton1998` in every 1 mm phantom voxel at that row's signal time plus the slice's readout offset, and
box-averages the result onto the acquisition grid; `control` and `m0scan` rows hold zero. It
is in the units of `M0map`, before the intensity scale, the T2 decay of blood at the echo
time, and, under background suppression, the pulses' label factor. To compare it with a
measured control-label difference in image units, multiply by the intensity scale (100),
by $e^{-\mathrm{TE}/T_{2b}}$, and by the label factor from the sidecar
([Chapter 5](../02-labeling/05-kinetic-model.md) does this on the noise-free reference). In
motion runs the map is the truth moved by the same poses as the data, so it disagrees with
the unmoved `deltamStatic` at every boundary of a moved volume.

## The transit-time map at boundaries

A voxel that is half gray matter (ATT 0.8 s) and half white matter (1.2 s) has a truth of
1.0 s, the perfusion-blind mean; CSF, which is never perfused, is excluded from the mean. A
fit that weights the two tissues by their perfusion, as the signal does, lands nearer the
gray matter value. The transit-time comparison of
[Chapter 15](../04-quantification/15-multi-delay.md) is therefore exact only in pure-tissue
voxels, which is why it reports the pure-tissue means as well as the voxelwise score.
