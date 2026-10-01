# asl-book

Arterial spin labeling executable book, built with Jupyter Book 2 (mystmd). Figures come from
small in-notebook simulations (the kinetic model and signal equations evaluated on a packaged
phantom slab) and from simulated datasets produced offline by
[aslscan](https://github.com/PennLINC/aslscan); every perfusion estimate is scored against the
analytic ground truth of the same simulated brain.

## Quickstart

All commands run inside the `aslbook` environment (WSL on this machine):

```bash
micromamba create -n aslbook -f environment.yml
micromamba run -n aslbook pip install -e ".[test]"
micromamba run -n aslbook pytest                        # helper-package unit tests
micromamba run -n aslbook myst start --execute          # live preview at http://localhost:3000
micromamba run -n aslbook myst build --html --execute   # static site in _build/html
micromamba run -n aslbook python tools/inject_static.py # then add the image lightbox
```

The theme cannot run page scripts, so click-to-enlarge images are added after the build:
`tools/inject_static.py` copies `book/_static/lightbox.{css,js}` into `_build/html` and
links them from every page. The CI workflow runs it before deploying.

Set `OMP_NUM_THREADS=1` (and the OpenBLAS/MKL equivalents) before building: the notebooks
execute in parallel, and one BLAS thread per kernel avoids oversubscription. Set
`ASLBOOK_DATA=/path/to/data` to build against local pipeline output instead of the data
release. The execution cache is keyed on notebook text only, so after editing anything in
`aslbook/` delete `_build/execute` (or the cached outputs will silently reflect the old
code). The offline data pipeline lives in [pipelines/](pipelines/README.md).

## Layout

| Path | What |
|---|---|
| `myst.yml` | book configuration and table of contents |
| `book/` | chapters as MyST Markdown notebooks, `references.bib` |
| `aslbook/` | helper package: data loader, kinetic model, toy simulator, quantification, plotting, packaged phantom slab |
| `pipelines/` | the offline pipeline that runs aslscan (config, phantom preparation, driver, release packaging) |
| `docs/` | planning documents |

## Planning documents

- [docs/outline.md](docs/outline.md) - chapter-by-chapter outline and the dataset dependency table
