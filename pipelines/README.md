# Offline data pipeline

Everything here runs in WSL, in the `aslbook` environment, and produces the datasets the
notebooks load. The book itself never runs any of this: notebooks fetch finished outputs from a
data release (or from `$ASLBOOK_DATA` while developing).

```bash
wsl -e bash -lc "cd /mnt/c/Users/tsalo/Documents/linc/asl-book/pipelines && micromamba run -n aslbook snakemake -c 4 --until all"
```

| Piece | Purpose |
|---|---|
| `config/datasets.yaml` | single source of truth: the pinned simulator commits, the phantoms, the shared protocol and overlay, every dataset's runs |
| `scripts/prepare_phantom.py` | crops aslscan's 1 mm ASLDRO phantom to the book's 100 mm slab, adds the synthetic field map, and packages the tissue fractions on the acquisition grid into `aslbook/data/` for the toy tier |
| `scripts/run_dataset.py` | expands one dataset entry into aslscan runs through `aslbook.cookbook`, writes each run's `asl.json` / `aslcontext.tsv` / `overlay.toml`, runs the simulator, adds the tissue-fraction derivatives, writes `provenance.json` |
| `scripts/package_release.py` | copies every file to flat release assets, writes the sha256 registry `aslbook/registry.txt` |
| `Snakefile` | the `phantoms` rule and one `simulate` rule per dataset |

Outputs land in `../data/<dataset-id>/<run>/`, one BIDS dataset per run exactly as aslscan
wrote it (with its `sub-01/perf/ground-truth`), plus `derivatives/aslbook` with the GM/WM/CSF
fraction of every acquisition voxel. Each run takes about 8 s on the 20-slice slab; the whole
table simulates in under ten minutes.

Tools the scripts expect: the `aslscan` binary built with `--features cli,kspace,par` at the
commit pinned in the configuration (its `../mrsim-acq` checkout at the pinned commit too).
