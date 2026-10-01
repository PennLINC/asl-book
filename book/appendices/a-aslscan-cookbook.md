---
title: "Appendix A: aslscan cookbook"
subtitle: The protocol, overlay, and command behind every dataset
kernelspec:
  name: python3
  display_name: Python 3
---

This appendix is generated from the pipeline configuration
(`pipelines/config/datasets.yaml`), the one file that describes what the offline pipeline
runs. Each dataset entry is expanded below into its runs by the same code the pipeline
driver executes (`aslbook.cookbook`), so the cookbook cannot drift from the pipeline. For
each run the simulator reads two BIDS files, the `*_asl.json` sidecar and the
`*_aslcontext.tsv`, exactly as it would from a real dataset, and an optional TOML overlay
with the settings BIDS does not record.

```{code-cell} python
:tags: [hide-cell]
from aslbook import cookbook

cfg = cookbook.load_config()
```

## The simulator in one paragraph

aslscan takes a BIDS ASL protocol and a phantom directory of NIfTI maps (perfusion, transit
time, T1, T2, T2*, M0, tissue labels, optionally a field map) and writes a BIDS ASL dataset:
`part-mag` and `part-phase` images with one volume per `aslcontext.tsv` row, the sidecars,
a separate M0 scan when `M0Type` is `Separate`, and the ground truth on the acquisition
grid. The kinetics stage evaluates the general kinetic model {cite:p}`buxton1998` per phantom
voxel at each slice's readout time; the signal stage forms the tissue and labeled-blood
compartments (saturation recovery or the background-suppression timeline for tissue; the
signed difference for blood); the acquisition stage, shared with the diffusion simulator
TRXScan through the `mrsim-acq` library, acquires each slice as a 2D single-shot spin-echo
EPI with T2 and T2' decay, field-map distortion, partial Fourier, ghosts, spikes, receive
coils, GRAPPA, k-space noise, and head motion. It reproduces ASLDRO's kinetic and signal
models {cite:p}`olivertaylor2021`.

## Tools and phantoms

```{code-cell} python
:tags: [hide-input]
t = cfg["tools"]
print(f"aslscan commit {t['aslscan_commit']} (mrsim-acq {t['mrsim_acq_commit']}), built with --features cli,kspace,par")
for key, ph in cfg["phantoms"].items():
    (x0, x1), (y0, y1), (z0, z1) = ph["crop"]
    print(f"phantom {key:<14} {ph['source'].split('/')[-1]} cropped to x {x0}:{x1}, y {y0}:{y1}, z {z0}:{z1} (1 mm)"
          + ("; synthetic field map" if ph["fieldmap"] else ""))
```

The source phantom is ASLDRO's `hrgt_icbm_2009a_nls_3t`, converted to the layout above by
aslscan's `tools/hrgt_to_bids.py` and cropped by the pipeline to the 100 mm slab that 20
slices of 5 mm tile exactly. The field map of `slab-fieldmap` is synthetic: a +120 Hz lobe
above the frontal sinuses, −90 Hz lobes at the temporal bones, and a small linear term
([Chapter 11](../03-preprocessing/11-susceptibility-distortion.md)).

## Shared settings

Every run starts from the reference protocol and overlay and changes what its dataset entry
says. The sidecar fields the datasets vary, and what each controls:

```{code-cell} python
:tags: [hide-input]
for k, v in cookbook.SIDECAR_GLOSSARY.items():
    print(f"{k:<32} {v}")
```

The overlay keys used by the book's datasets:

```{code-cell} python
:tags: [hide-input]
for k, v in cookbook.KEY_GLOSSARY.items():
    print(f"{k:<44} {v}")
```

The reference protocol's sidecar and the default overlay, as the `ref-pcasl` run reads them:

```{code-cell} python
:tags: [hide-input]
cookbook.print_protocol(cfg, "ref-pcasl")
print()
cookbook.print_overlay(cfg, "ref-pcasl")
```

(app-a-datasets)=
## The datasets

Each chapter lists the datasets it uses in the box at its top and links here. For each
dataset this section gives the pipeline's description, the chapters that use it, the runs
with the sidecar fields and overlay keys that differ from the reference, and the command
lines. Every run directory is a BIDS dataset exactly as aslscan wrote it (`sub-01/perf/`),
with the ground truth under `sub-01/perf/ground-truth/` (listed in `.bidsignore`) and the
pipeline's tissue fractions under `derivatives/aslbook/`.

```{code-cell} python
:tags: [hide-input]
import json

REF = cookbook.expand(cfg, "ref-pcasl")[0]

def diff_of(run):
    """The sidecar fields and overlay keys that differ from the reference run."""
    out = []
    for k, v in run.protocol.items():
        if REF.protocol.get(k) != v:
            s = json.dumps(v) if not (isinstance(v, list) and len(v) > 8) else f"[{v[0]}, ..., {v[-1]}] ({len(v)} values)"
            out.append(f"{k} = {s}")
    def walk(d, ref, prefix=""):
        for k, v in d.items():
            if isinstance(v, dict):
                walk(v, (ref or {}).get(k, {}), f"{prefix}{k}.")
            elif (ref or {}).get(k) != v:
                out.append(f"overlay {prefix}{k} = {json.dumps(v)}")
    walk(run.overlay, REF.overlay)
    if run.phantom != REF.phantom:
        out.append(f"phantom = {run.phantom}")
    kinds = {k: run.ctx.count(k) for k in ("m0scan", "control", "label") if k in run.ctx}
    ref_kinds = {k: REF.ctx.count(k) for k in ("m0scan", "control", "label") if k in REF.ctx}
    if kinds != ref_kinds:
        out.append("aslcontext: " + ", ".join(f"{n} {k}" for k, n in kinds.items()))
    return out

def describe(ds):
    entry = cfg["datasets"][ds]
    print(entry["description"].strip())
    print("chapters: " + ", ".join(str(c) for c in entry["chapters"]))
    for run in cookbook.expand(cfg, ds):
        d = diff_of(run)
        print(f"\nrun {run.name}: " + ("; ".join(d) if d else "the reference"))
        print("  " + run.command(cfg))
```

(ds-ref-pcasl)=
### ref-pcasl

```{code-cell} python
:tags: [hide-input]
describe("ref-pcasl")
```

Expected files of a run directory:

```{code-cell} python
:tags: [hide-input]
cookbook.print_tree(cfg, "ref-pcasl")
```

(ds-ref-clean)=
### ref-clean

```{code-cell} python
:tags: [hide-input]
describe("ref-clean")
```

(ds-label-types)=
### label-types

```{code-cell} python
:tags: [hide-input]
describe("label-types")
```

(ds-pld-sweep)=
### pld-sweep

```{code-cell} python
:tags: [hide-input]
describe("pld-sweep")
```

(ds-multi-pld)=
### multi-pld

```{code-cell} python
:tags: [hide-input]
describe("multi-pld")
```

(ds-noise-sweep)=
### noise-sweep

```{code-cell} python
:tags: [hide-input]
describe("noise-sweep")
```

(ds-bgsup)=
### bgsup

```{code-cell} python
:tags: [hide-input]
describe("bgsup")
```

(ds-motion)=
### motion

```{code-cell} python
:tags: [hide-input]
describe("motion")
```

Motion runs also write `sub-01_desc-motion_gt.tsv` (the applied pose per volume) and
`sub-01_desc-deltamStatic_gt.nii.gz` (the unmoved difference truth) next to the other
ground-truth maps.

(ds-sdc)=
### sdc

```{code-cell} python
:tags: [hide-input]
describe("sdc")
```

(ds-voxel-sweep)=
### voxel-sweep

```{code-cell} python
:tags: [hide-input]
describe("voxel-sweep")
```

(ds-m0-types)=
### m0-types

```{code-cell} python
:tags: [hide-input]
describe("m0-types")
```

(ds-te-sweep)=
### te-sweep

```{code-cell} python
:tags: [hide-input]
describe("te-sweep")
```

(ds-readout)=
### readout

```{code-cell} python
:tags: [hide-input]
describe("readout")
```

(ds-kitchen-sink)=
### kitchen-sink

```{code-cell} python
:tags: [hide-input]
describe("kitchen-sink")
```

## Regenerating a dataset

Every dataset directory carries a `provenance.json` with the exact command lines, the
protocol and overlay of every run, and the aslscan version and commit. To regenerate one:

```bash
cd pipelines
micromamba run -n aslbook snakemake -c 4 <dataset-id>
```

The Snakefile prepares the phantom slabs, runs the commands above, adds the tissue
fractions, and writes the provenance file. A run takes about eight seconds on the slab.
