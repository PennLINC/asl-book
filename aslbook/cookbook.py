"""Render the aslscan inputs and commands behind every dataset (Appendix A).

The pipeline configuration ``pipelines/config/datasets.yaml`` is the single description of
what the offline pipeline runs. This module expands each dataset entry into its runs: for each
run, the BIDS sidecar and ``aslcontext.tsv`` the simulator reads (built by
:mod:`aslbook.protocols`), the TOML overlay with the settings BIDS does not record, and the
``aslscan`` command line. The pipeline driver (``pipelines/scripts/run_dataset.py``) executes
exactly this expansion, so the cookbook cannot drift from the pipeline.

A dataset entry has: ``description``; ``protocol`` (``type`` names a builder in
:mod:`aslbook.protocols`, the rest are its keyword arguments); optional ``overlay`` (nested
tables merged over the defaults); optional ``phantom`` (a key of the ``phantoms`` table);
``variants`` (a list of ``{name, protocol?, overlay?, phantom?, seed?}`` overrides, one run
each) or ``sweep`` (``{key, values, labels}`` where ``key`` is a dotted path into
``protocol`` or ``overlay``); and ``chapters``.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import protocols

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "pipelines" / "config" / "datasets.yaml"

#: Overlay keys used by the book's datasets, with what each controls.
KEY_GLOSSARY: dict[str, str] = {
    "seed": "random seed of the noise and motion draws; every run records it",
    "acquisition.oversample": "simulate the object on a grid this many times finer than the acquisition in-plane, so truncation ringing is intrinsic (Chapter 2)",
    "acquisition.matrix": "in-plane acquisition matrix; the field of view is matrix times voxel size, corner-aligned with the phantom",
    "acquisition.noise_variance": "variance of each of the real and imaginary noise components in image units; 0 is noise-free (Chapter 8)",
    "acquisition.n_coils": "number of ring-arranged receive coils, combined with Roemer weights (Chapter 2)",
    "acquisition.partial_fourier": "fraction of phase-encode lines acquired (Chapter 13)",
    "acquisition.ghost_offset": "Nyquist ghost strength (Chapter 13)",
    "acquisition.n_spikes": "k-space spikes per slice (Chapter 13)",
    "acquisition.signal_scale": "overall intensity scale of the images (100: an M0 of 74.6 gives about 7460 before saturation and T2 decay)",
    "m0.repetition_time": "repetition time of the separate M0 scan, which nothing in the ASL sidecar carries (Chapter 16)",
    "background_suppression.inversion_efficiency": "fraction of the longitudinal magnetization each suppression pulse inverts (Chapter 9)",
    "background_suppression.presaturation": "a saturation pulse on the imaging region at the start of labeling (Chapter 9)",
    "motion.mode": "off, random (impulses that return to baseline), linear (a drift), or trajectory (a TSV replayed per volume) (Chapter 10)",
    "motion.trans_mm": "per-axis translation amplitudes in mm for random and linear motion",
    "motion.rot_deg": "per-axis rotation amplitudes in degrees",
    "motion.volumes": "the volumes random and linear motion affect",
    "kinetic.label_efficiency": "labeling efficiency alpha (Chapter 4)",
    "kinetic.t1_arterial_blood": "T1 of arterial blood in s (Chapter 5)",
    "kinetic.lambda_blood_brain": "blood-brain partition coefficient in ml/g (Chapter 3)",
    "signal.t2_blood": "T2 of blood in s, applied to the labeled compartment during the readout (Chapter 16)",
}

#: Sidecar fields the datasets vary, with what each controls.
SIDECAR_GLOSSARY: dict[str, str] = {
    "ArterialSpinLabelingType": "PCASL, CASL, or PASL: the kinetic branch and the default efficiency (Chapter 4)",
    "LabelingDuration": "the bolus duration tau of (P)CASL (Chapter 5)",
    "PostLabelingDelay": "the delay from the end of labeling (PASL: from the labeling pulse) to the readout; a list gives one value per volume (Chapter 5)",
    "BolusCutOffDelayTime": "PASL: the time at which the bolus is cut off, which becomes tau (Chapter 4)",
    "RepetitionTimePreparation": "the repetition time; the tissue's saturation recovery is evaluated at it (Chapter 6)",
    "AcquisitionVoxelSize": "the acquisition voxel, mm; slices are not oversampled (Chapter 7)",
    "SliceTiming": "the excitation time of each 2D slice; each slice's delay grows by its offset (Chapter 6)",
    "BackgroundSuppressionPulseTime": "the inversion pulses, in seconds from the start of labeling (Chapter 9)",
    "M0Type": "Separate (an M0 scan of its own), Included (m0scan rows), Estimate, or Absent (Chapter 16)",
    "PhaseEncodingDirection": "j or j-: the polarity along which the field map displaces the image (Chapter 11)",
    "ParallelReductionFactorInPlane": "GRAPPA acceleration; needs several coils (Chapter 2)",
}


@dataclass
class RunSpec:
    """One simulator run: everything needed to write its inputs and run it."""

    dataset: str
    name: str
    protocol: dict
    ctx: list[str]
    overlay: dict
    phantom: str
    seed: int
    description: str = ""

    @property
    def label(self) -> str:
        return self.name

    def input_dir(self, data_root: Path) -> Path:
        return data_root / self.dataset / "inputs" / self.name

    def out_dir(self, data_root: Path) -> Path:
        return data_root / self.dataset / self.name

    def command(self, cfg: dict, data_root: Path | str = "data") -> str:
        data_root = Path(data_root)
        ph = cfg["phantoms"][self.phantom]
        inp = self.input_dir(data_root).as_posix()
        return (
            f"aslscan --asl-json {inp}/asl.json --aslcontext {inp}/aslcontext.tsv "
            f"--overlay {inp}/overlay.toml --phantom {ph['dir']} --out {self.out_dir(data_root).as_posix()} "
            f"--seed {self.seed}"
        )


def load_config(path: Path | str = CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _set_path(d: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def build_protocol(spec: dict) -> tuple[dict, list[str]]:
    """A sidecar and context from a ``protocol`` mapping (``type`` plus keyword arguments)."""
    spec = dict(spec)
    kind = spec.pop("type", "pcasl")
    m0_rows = spec.pop("m0_rows", 1)
    order = spec.pop("order", "control-first")
    builder = getattr(protocols, kind)
    p = builder(**spec)
    return p, protocols.aslcontext(p, m0_rows=m0_rows, order=order)


def expand(cfg: dict, dataset_id: str) -> list[RunSpec]:
    """All runs of one dataset, in the order the pipeline executes them."""
    ds = cfg["datasets"][dataset_id]
    d = cfg["defaults"]
    base = {
        "protocol": _deep_merge(d.get("protocol", {}), ds.get("protocol", {})),
        "overlay": _deep_merge(d.get("overlay", {}), ds.get("overlay", {})),
        "phantom": ds.get("phantom", d["phantom"]),
        "seed": ds.get("seed", d["seed"]),
    }
    variants: list[dict] = []
    if "sweep" in ds:
        sw = ds["sweep"]
        labels = sw.get("labels") or [f"{sw['key'].split('.')[-1]}{v}" for v in sw["values"]]
        for label, value in zip(labels, sw["values"]):
            v = {"name": label, "description": f"{sw['key']} = {value}"}
            head = sw["key"].split(".")[0]
            rest = sw["key"].split(".", 1)[1]
            v[head] = {}
            _set_path(v[head], rest, value)
            variants.append(v)
    if "variants" in ds:
        variants.extend(ds["variants"])
    if not variants:
        variants = [{"name": ds.get("run", "run")}]
    runs = []
    for v in variants:
        proto = _deep_merge(base["protocol"], v.get("protocol", {}))
        overlay = _deep_merge(base["overlay"], v.get("overlay", {}))
        p, ctx = build_protocol(proto)
        runs.append(RunSpec(
            dataset=dataset_id, name=v["name"], protocol=p, ctx=ctx, overlay=overlay,
            phantom=v.get("phantom", base["phantom"]), seed=v.get("seed", base["seed"]),
            description=v.get("description", ""),
        ))
    return runs


def to_toml(d: dict) -> str:
    """A small TOML writer for the overlay: scalars, lists, and one level of tables."""

    def fmt(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return repr(float(v)) if isinstance(v, float) else str(v)
        if isinstance(v, str):
            return json.dumps(v)
        if isinstance(v, (list, tuple)):
            return "[" + ", ".join(fmt(x) for x in v) + "]"
        raise TypeError(f"cannot write {v!r} to TOML")

    lines = []
    for k, v in d.items():
        if not isinstance(v, dict):
            lines.append(f"{k} = {fmt(v)}")
    for k, v in d.items():
        if isinstance(v, dict):
            lines.append(f"\n[{k}]")
            for kk, vv in v.items():
                if isinstance(vv, dict):
                    continue
                lines.append(f"{kk} = {fmt(vv)}")
            for kk, vv in v.items():
                if isinstance(vv, dict):
                    lines.append(f"\n[{k}.{kk}]")
                    for k3, v3 in vv.items():
                        lines.append(f"{k3} = {fmt(v3)}")
    return "\n".join(lines) + "\n"


def write_inputs(run: RunSpec, data_root: Path) -> Path:
    """Write ``asl.json``, ``aslcontext.tsv`` and ``overlay.toml`` for a run; returns the directory."""
    d = run.input_dir(data_root)
    d.mkdir(parents=True, exist_ok=True)
    (d / "asl.json").write_text(json.dumps(run.protocol, indent=1) + "\n")
    (d / "aslcontext.tsv").write_text("volume_type\n" + "\n".join(run.ctx) + "\n")
    (d / "overlay.toml").write_text(to_toml(run.overlay))
    return d


# ------------------------------------------------------------------- rendering (Appendix A)
def print_commands(cfg: dict, dataset_id: str) -> None:
    for run in expand(cfg, dataset_id):
        print(f"# {run.name}" + (f": {run.description}" if run.description else ""))
        print(run.command(cfg))


def print_protocol(cfg: dict, dataset_id: str, run_name: str | None = None, keys: tuple[str, ...] | None = None) -> None:
    """The sidecar fields of a run (the first by default), and its context summary."""
    runs = expand(cfg, dataset_id)
    run = runs[0] if run_name is None else next(r for r in runs if r.name == run_name)
    p = run.protocol
    keys = keys or tuple(p)
    for k in keys:
        v = p[k]
        if isinstance(v, list) and len(v) > 8:
            v = f"[{v[0]}, {v[1]}, ..., {v[-1]}] ({len(v)} values)"
        print(f"{k:<34} {v}")
    ctx = run.ctx
    kinds = {k: ctx.count(k) for k in ("m0scan", "control", "label", "deltam") if k in ctx}
    print(f"{'aslcontext':<34} {len(ctx)} volumes: " + ", ".join(f"{n} {k}" for k, n in kinds.items()))


def print_overlay(cfg: dict, dataset_id: str, run_name: str | None = None) -> None:
    runs = expand(cfg, dataset_id)
    run = runs[0] if run_name is None else next(r for r in runs if r.name == run_name)
    print(to_toml(run.overlay).strip())


def expected_files(run: RunSpec) -> list[str]:
    """The files a run directory holds once aslscan and the pipeline have run."""
    files = [
        "dataset_description.json", "README", ".bidsignore",
        "sub-01/perf/sub-01_part-mag_asl.nii.gz", "sub-01/perf/sub-01_part-mag_asl.json",
        "sub-01/perf/sub-01_part-phase_asl.nii.gz", "sub-01/perf/sub-01_part-phase_asl.json",
        "sub-01/perf/sub-01_aslcontext.tsv",
    ]
    if run.protocol.get("M0Type") == "Separate":
        files += ["sub-01/perf/sub-01_m0scan.nii.gz", "sub-01/perf/sub-01_m0scan.json"]
    gt = "sub-01/perf/ground-truth/sub-01_desc-{}_gt.nii.gz"
    files += [gt.format(n) for n in ("perfusion", "att", "T1map", "T2map", "M0map", "dseg", "deltam")]
    if run.overlay.get("motion", {}).get("mode", "off") != "off":
        files += [gt.format("deltamStatic"), "sub-01/perf/ground-truth/sub-01_desc-motion_gt.tsv"]
    files += [f"derivatives/aslbook/sub-01/perf/sub-01_label-{t}_probseg.nii.gz" for t in ("GM", "WM", "CSF")]
    return files


def print_tree(cfg: dict, dataset_id: str) -> None:
    runs = expand(cfg, dataset_id)
    print(f"{dataset_id}/")
    print("  provenance.json")
    for i, run in enumerate(runs):
        print(f"  {run.name}/")
        if i == 0:
            for f in expected_files(run):
                print(f"    {f}")
        else:
            print("    (the same files)")


def datasets_using(cfg: dict, chapter) -> list[str]:
    return [d for d, ds in cfg["datasets"].items() if chapter in ds.get("chapters", [])]
