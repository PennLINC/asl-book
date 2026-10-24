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
time, T1, T2, T2*, M0, tissue labels, optionally a field map) and writes a BIDS ASL dataset {cite:p}`clement2022`:
`part-mag` and `part-phase` images with one volume per `aslcontext.tsv` row, the sidecars,
a separate M0 scan when `M0Type` is `Separate`, and the ground truth on the acquisition
grid. The kinetics stage evaluates the general kinetic model {cite:p}`buxton1998` per phantom
voxel at each slice's readout time; the signal stage forms the tissue and labeled-blood
compartments (saturation recovery or the background-suppression timeline for tissue; the
signed difference for blood; optionally the label split by exchange, an arterial
compartment, and physiological factors); the acquisition stage, shared with the diffusion
simulator TRXScan through the `mrsim-acq` library, acquires each slice as a 2D single-shot
EPI, or the whole slab as a segmented 3D GRASE or stack-of-spirals echo train, with T2 and
T2' decay, field-map distortion, partial Fourier, ghosts, spikes, receive coils, GRAPPA
{cite:p}`griswold2002`, k-space noise, and head motion, and decodes a time-encoded series.
It reproduces ASLDRO's kinetic and signal models {cite:p}`olivertaylor2021`.

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

The source phantom is ASLDRO's `hrgt_icbm_2009a_nls_3t` {cite:p}`olivertaylor2021`, converted to the layout above by
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
    kinds = {k: run.ctx.count(k) for k in ("m0scan", "control", "label", "deltam") if k in run.ctx}
    ref_kinds = {k: REF.ctx.count(k) for k in ("m0scan", "control", "label", "deltam") if k in REF.ctx}
    if kinds != ref_kinds:
        out.append("aslcontext: " + ", ".join(f"{n} {k}" for k, n in kinds.items()))
    if run.echo_times:
        out.append(f"one sidecar per echo: {', '.join(f'{1e3 * te:g} ms' for te in run.echo_times)}")
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

(ds-readout-3d)=
### readout-3d

```{code-cell} python
:tags: [hide-input]
describe("readout-3d")
```

A 3D sidecar has no `SliceTiming` and no `TotalReadoutTime` (the simulator writes the
effective spacing it resolved); `FlipAngle` is the echo train's refocusing angle, and the
in-plane segmentation, the actual EPI line spacing, and the spiral's interleaves, readout
time and dwell time, none of which BIDS records, come from the overlay's `[readout]` table.
Every resolved value of the train (echo spacing, echo train length, lines per echo, the
echo amplitudes of every tissue and the resulting k-space modulation) is in
`AslscanSimulation.Readout` and `AslscanSimulation.EchoAmplitudes`
([Chapter 2](../01-mri-physics/02-epi-and-reconstruction.md)). A volume of a four-shot
readout takes four repetition times, which the NIfTI time step records.

(ds-physio)=
### physio

```{code-cell} python
:tags: [hide-input]
describe("physio")
```

The factors the simulator applied to every volume and slice, with the true cardiac and
respiratory phases and the drift, are in `sub-01_desc-physio_gt.tsv`
([Chapter 9](../03-preprocessing/09-background-suppression.md)).

(ds-hadamard)=
### hadamard

```{code-cell} python
:tags: [hide-input]
describe("hadamard")
```

The sidecar's arrays describe the *decoded* volumes, as the BIDS convention for
time-encoded data has it: each `deltam` row's `LabelingDuration` is its sub-bolus and its
`PostLabelingDelay` the time from the end of that sub-bolus to the readout. The simulator
acquires the encoded volumes, keeps them under `sourcedata/` with a table of which sub-boli
each one labeled, and writes the decoded series as the main dataset
([Chapter 18](../05-advanced/18-time-encoded-and-look-locker.md)); the files of a run:

```{code-cell} python
:tags: [hide-input]
cookbook.print_tree(cfg, "hadamard")
```

(ds-look-locker)=
### look-locker

```{code-cell} python
:tags: [hide-input]
describe("look-locker")
```

Each labeling is followed by eight readouts, one volume each, so the `PostLabelingDelay`
array runs through the eight inversion times once per cycle and `FlipAngle` is the
readouts' excitation. Beside the undepleted `deltam` truth the simulator writes
`desc-deltamRead_gt`, what each readout actually read after the earlier readouts of the
cycle consumed part of the label, and `desc-lookLocker_gt.tsv`, the tissue magnetization
before every readout ([Chapter 18](../05-advanced/18-time-encoded-and-look-locker.md)).

(ds-multi-te)=
### multi-te

```{code-cell} python
:tags: [hide-input]
describe("multi-te")
```

A multi-echo run reads one sidecar per echo (repeated `--asl-json`, identical except for
`EchoTime`) and writes one series per echo in the BIDS `echo-N` layout, with one
`aslcontext.tsv` and one ground truth shared by all echoes:

```{code-cell} python
:tags: [hide-input]
cookbook.print_tree(cfg, "multi-te")
```

(ds-vascular)=
### vascular

```{code-cell} python
:tags: [hide-input]
describe("vascular")
```

The arterial compartment's inputs are per tissue (the phantom carries no arterial maps), and
the simulator writes them back as maps (`desc-aBV_gt`, `desc-aATT_gt`) beside the arterial
signal of every volume before crushing (`desc-deltamArterial_gt`), so that the crushed and
uncrushed series can be scored against it
([Chapter 19](../05-advanced/19-frontiers.md)).

(ds-motion-readouts)=
### motion-readouts

```{code-cell} python
:tags: [hide-input]
describe("motion-readouts")
```

The `motion.trace` table is not an aslscan key but the pipeline's recipe for one: `aslbook.traces`
writes it as `inputs/<run>/motion.tsv`, a timed trace with a `time` column at the given rate
(the same nods, steps and drifts aslscan's `tools/motion_trace.py` composes), and the overlay
the simulator reads says `trajectory = "<that file>"`. The simulator then samples the trace at
every 2D shot's or 3D segment's own excitation time (`within_volume_sampling = "shot"`, the
default for a timed trace) and writes the poses to `desc-motionShots_gt.tsv`; `epi-volume`
asks for `"volume"` instead, and `epi-spinhistory` adds `spin_history = true`, whose
correction is written as `desc-spinHistory_gt`. The 3D runs' motion-free partners are
`readout-3d`'s `grase-1shot`, `grase-4shot` and `spiral`, which share their protocol and seed
([Chapter 10.1](../03-preprocessing/10a-motion-within-a-volume.md)).

```{code-cell} python
:tags: [hide-input]
from aslbook import traces
print(traces.describe(cookbook.expand(cfg, "motion-readouts")[1].overlay["motion"]["trace"]))
cookbook.print_tree(cfg, "motion-readouts")
```

## Regenerating a dataset

Every dataset directory carries a `provenance.json` with the exact command lines, the
protocol and overlay of every run, and the aslscan version and commit. To regenerate one:

```bash
cd pipelines
micromamba run -n aslbook snakemake -c 4 <dataset-id>
```

The Snakefile (the pipeline is a Snakemake workflow {cite:p}`molder2021`) prepares the phantom slabs, runs the commands above, adds the tissue
fractions, and writes the provenance file. A 2D run takes about eight seconds on the slab
and a 3D GRASE run about as long; a stack-of-spirals run takes minutes, because the
non-Cartesian reconstruction is iterative.
