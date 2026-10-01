"""Simulate one dataset of the book with aslscan.

    micromamba run -n aslbook python scripts/run_dataset.py --dataset ref-pcasl [--data-root ../data]

Expands the dataset's entry in ``config/datasets.yaml`` into runs (``aslbook.cookbook``),
writes each run's inputs (``asl.json``, ``aslcontext.tsv``, ``overlay.toml``) under
``<data-root>/<dataset>/inputs/<run>/``, runs aslscan into ``<data-root>/<dataset>/<run>/``,
adds the tissue fractions of every acquisition voxel as ``derivatives/aslbook`` (the
box-averaged 1 mm label map, by the rule the simulator resamples with), and writes the
dataset's ``provenance.json`` with the commands, timings, and the pinned simulator commits.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np

from aslbook import cookbook, grid


def tissue_fractions(run_dir: Path, phantom_dir: Path) -> None:
    """Write the GM/WM/CSF fraction of each acquisition voxel next to the run."""
    gt = run_dir / "sub-01" / "perf" / "ground-truth" / "sub-01_desc-dseg_gt.nii.gz"
    acq = nib.load(gt)
    src = nib.load(phantom_dir / "dseg.nii.gz")
    labels = np.asarray(src.dataobj).astype(np.int16)
    rs = grid.BoxResampler(labels.shape, [float(v) for v in src.header.get_zooms()[:3]], acq.shape[:3], [float(v) for v in acq.header.get_zooms()[:3]])
    fr = rs.fractions(labels, values=(1, 2, 3))
    out = run_dir / "derivatives" / "aslbook"
    (out / "sub-01" / "perf").mkdir(parents=True, exist_ok=True)
    (out / "dataset_description.json").write_text(json.dumps({
        "Name": "aslbook tissue fractions", "BIDSVersion": "1.9.0", "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "asl-book pipelines/scripts/run_dataset.py", "Description": "box average of the 1 mm phantom label map onto the acquisition grid (aslbook.grid)"}],
    }, indent=2) + "\n")
    for label, name in ((1, "GM"), (2, "WM"), (3, "CSF")):
        img = nib.Nifti1Image(fr[label].astype(np.float32), acq.affine)
        img.set_data_dtype(np.float32)
        nib.save(img, out / "sub-01" / "perf" / f"sub-01_label-{name}_probseg.nii.gz")
        (out / "sub-01" / "perf" / f"sub-01_label-{name}_probseg.json").write_text(json.dumps({"Units": "fraction", "Description": f"volume fraction of {name} in each acquisition voxel"}, indent=2) + "\n")


def git_head(repo: str) -> str:
    try:
        return subprocess.run(["git", "-C", repo, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/datasets.yaml")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data-root", default="../data")
    ap.add_argument("--root", default="..", help="repository root, which phantom `dir` entries are relative to")
    ap.add_argument("--force", action="store_true", help="rerun runs whose output exists")
    a = ap.parse_args()
    cfg = cookbook.load_config(a.config)
    root = Path(a.root).resolve()
    data_root = Path(a.data_root).resolve()
    tools = cfg["tools"]
    binary = tools["aslscan_bin"]
    for repo, key in ((tools["aslscan_repo"], "aslscan_commit"), (tools["mrsim_acq_repo"], "mrsim_acq_commit")):
        head = git_head(repo)
        if head != "unknown" and head != tools[key]:
            raise SystemExit(f"{repo} is at {head}, the configuration pins {tools[key]}; update the pin or check out the commit")
    version = subprocess.run([binary, "--version"], capture_output=True, text=True).stdout.strip()

    runs = cookbook.expand(cfg, a.dataset)
    prov = {
        "dataset": a.dataset, "description": cfg["datasets"][a.dataset].get("description", ""),
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "aslscan": {"version": version, "commit": tools["aslscan_commit"], "mrsim_acq_commit": tools["mrsim_acq_commit"]},
        "runs": [],
    }
    for run in runs:
        out = run.out_dir(data_root)
        phantom_dir = root / cfg["phantoms"][run.phantom]["dir"]
        if out.exists() and not a.force:
            print(f"{a.dataset}/{run.name}: exists, skipping (use --force to rerun)")
        else:
            if out.exists():
                shutil.rmtree(out)
            inp = cookbook.write_inputs(run, data_root)
            cmd = [binary, "--asl-json", str(inp / "asl.json"), "--aslcontext", str(inp / "aslcontext.tsv"),
                   "--overlay", str(inp / "overlay.toml"), "--phantom", str(phantom_dir), "--out", str(out), "--seed", str(run.seed)]
            t0 = time.time()
            res = subprocess.run(cmd, capture_output=True, text=True)
            dt = time.time() - t0
            (inp / "aslscan.log").write_text(res.stdout + res.stderr)
            if res.returncode != 0:
                raise SystemExit(f"{a.dataset}/{run.name} failed:\n{res.stdout}\n{res.stderr}")
            tissue_fractions(out, phantom_dir)
            print(f"{a.dataset}/{run.name}: {dt:.1f} s")
        prov["runs"].append({
            "name": run.name, "description": run.description, "phantom": run.phantom, "seed": run.seed,
            "command": run.command(cfg), "protocol": run.protocol, "aslcontext": run.ctx, "overlay": run.overlay,
        })
    (data_root / a.dataset / "provenance.json").write_text(json.dumps(prov, indent=1) + "\n")


if __name__ == "__main__":
    main()
