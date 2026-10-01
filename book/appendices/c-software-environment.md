---
title: "Appendix C: Software environment"
subtitle: Versions used to build this edition
kernelspec:
  name: python3
  display_name: Python 3
---

The book is built by executing every notebook in one Python environment, described by
`environment.yml` at the root of the repository. The versions below are read from the
environment that built the page you are reading, so they are the versions behind every
figure and number in it.

```{code-cell} python
:tags: [hide-input]
import importlib.metadata as md
import platform

print(f"Python {platform.python_version()} on {platform.system()} {platform.machine()}")
for pkg in ["numpy", "scipy", "nibabel", "matplotlib", "pandas", "scikit-image", "pooch", "pyyaml", "mystmd", "aslbook"]:
    try:
        print(f"{pkg:<14} {md.version(pkg)}")
    except md.PackageNotFoundError:
        print(f"{pkg:<14} not installed")
```

The numerical work in every page rests on NumPy {cite:p}`harris2020` and SciPy
{cite:p}`virtanen2020`, and the figures are drawn with Matplotlib {cite:p}`hunter2007`.

## Creating the environment

```bash
micromamba create -n aslbook -f environment.yml
micromamba run -n aslbook pip install -e .
```

The environment file:

```{code-cell} python
:tags: [hide-input]
from pathlib import Path

import aslbook

env = Path(aslbook.__file__).resolve().parents[1] / "environment.yml"
print(env.read_text() if env.exists() else "environment.yml not found next to the package")
```

## Building the book

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 myst build --html --execute
```

The notebooks execute in parallel, one kernel each; one BLAS thread per kernel keeps the
build short. `myst start --execute` serves a live preview that rebuilds on every edit. The
execution cache in `_build/execute` is keyed on the notebook text, so after a change to the
`aslbook` package the cache must be cleared for the outputs to reflect it.

## Offline tools

The simulated datasets are produced by a separate pipeline that is not part of the book
build ([Appendix A](./a-aslscan-cookbook.md)):

| Tool | Role | Where it runs |
|---|---|---|
| aslscan (with the mrsim-acq library) | the simulator and its ground truth | a native binary built from the commit pinned in the pipeline configuration, with `cargo build --release --features cli,kspace,par` |
| ASLDRO's `hrgt_icbm_2009a_nls_3t` {cite:p}`olivertaylor2021` | the phantom, converted by aslscan's `tools/hrgt_to_bids.py` | once, before the pipeline |
| Snakemake {cite:p}`molder2021` | pipeline driver | the `aslbook` environment |

## Reproducibility

Every dataset's `provenance.json` records the aslscan version and commit, and every sidecar
carries the seed of its noise and motion draws, so a run is reproducible bit for bit from
the same binary. Every number in the book comes from a seeded random number generator, so
the toy-tier results are identical between builds on the same versions.
