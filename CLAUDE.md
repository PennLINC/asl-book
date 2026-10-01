# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

An executable Jupyter Book 2 (mystmd) on arterial spin labeling MRI, modeled on
`../diffusion-book`. Chapters are MyST Markdown notebooks under `book/`; the helper package
`aslbook/` holds the kinetic model, the toy simulator, quantification, the data loader and
plotting; `pipelines/` runs the aslscan simulator offline to produce the datasets.

## Environment

- micromamba environment: **`aslbook`** (WSL side), from `environment.yml`. Run everything as
  `wsl -e bash -lc "cd /mnt/c/Users/tsalo/Documents/linc/asl-book && micromamba run -n aslbook <command>"`.
- The simulated datasets live in `data/` (gitignored); set
  `ASLBOOK_DATA=/mnt/c/Users/tsalo/Documents/linc/asl-book/data` to build against them.
- Simulator: `../../rust-trx/aslscan/target/release/aslscan` at the commit pinned in
  `pipelines/config/datasets.yaml`; the phantom source is `rust-trx/aslscan/work/phantom-3t`.

## Commands

```bash
micromamba run -n aslbook pytest                                   # package tests
micromamba run -n aslbook python tools/run_chapter.py book/<part>/<chapter>.md --figures _build/figs/x
                                                                   # execute one chapter's cells outside the build
ASLBOOK_DATA=$PWD/data OMP_NUM_THREADS=1 micromamba run -n aslbook myst build --html --execute
micromamba run -n aslbook python tools/inject_static.py            # lightbox after the build
cd pipelines && micromamba run -n aslbook python scripts/run_dataset.py --dataset <id>   # simulate one dataset
```

## Conventions

- `docs/chapter-conventions.md` is the notebook standard; `docs/outline.md` the chapter plan
  and dataset table. Every chapter executes end to end and quotes numbers its code printed.
- The execution cache (`_build/execute`) is keyed on notebook text only: delete it after
  changing `aslbook/`.
- Do not hand-edit `aslbook/registry.txt` (written by `pipelines/scripts/package_release.py`).
