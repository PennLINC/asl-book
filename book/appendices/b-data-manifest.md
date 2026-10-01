---
title: "Appendix B: Data manifest"
subtitle: Every file the notebooks load, with checksums
kernelspec:
  name: python3
  display_name: Python 3
---

The notebooks obtain every simulated file through one loader, `aslbook.data.load_dataset`,
which resolves a dataset id to a directory: a local pipeline output if `ASLBOOK_DATA` points
at one, otherwise a download from the data release verified against the checksums in the
package's registry. Each run of a dataset is one archive in the release, fetched and
extracted the first time a page asks for a file of that run. The registry is written by the
pipeline's release script and never edited by hand, and this appendix is rendered from it,
so the manifest is the registry.

```{code-cell} python
:tags: [hide-cell]
import os
from pathlib import Path

from aslbook import cookbook, data

registry_path = Path(data.__file__).with_name("registry.txt")
cfg = cookbook.load_config()
```

## The data release

```{code-cell} python
:tags: [hide-input]
entries = data._registry()
print(f"release tag {data.DATA_TAG}; registry {registry_path.name} with {len(entries)} files")
if os.environ.get("ASLBOOK_DATA"):
    print(f"this build reads local pipeline output from ASLBOOK_DATA = {os.environ['ASLBOOK_DATA']}")
```

## Archives by dataset

For each dataset: its provenance file and one archive per run, each with the first twelve
hexadecimal digits of its SHA-256 checksum from the registry and, where the run is present
locally, its size.

```{code-cell} python
:tags: [hide-input]
def local_size(ds, run):
    try:
        d = data.load_dataset(ds)
    except KeyError:
        return None
    if not d.local or not (d.path / run).is_dir():
        return None
    return sum(f.stat().st_size for f in (d.path / run).rglob("*") if f.is_file())

total = 0
for ds in cfg["datasets"]:
    print(f"{ds}/")
    for key in ("provenance.json", *(f"{r.name}.tar" for r in cookbook.expand(cfg, ds))):
        digest = entries.get(f"{ds}/{key}", "")
        size = local_size(ds, key[:-4]) if key.endswith(".tar") else None
        total += size or 0
        print(f"  {key:<24} {digest[:12] or '(not registered)':<16} {f'{size / 1e6:.0f} MB' if size else ''}")
print(f"\n{total / 1e6:.0f} MB in all")
```

Every archive holds one BIDS dataset as aslscan wrote it; the layout of a run directory is
listed in [Appendix A](./a-aslscan-cookbook.md#ds-ref-pcasl). Until the data release is
published the digests describe the archives the pipeline built locally.

(app-b-package-data)=
## Package data

Two small files ship inside the `aslbook` package for the toy tier, written by the pipeline's
`prepare_phantom.py` from the same cropped phantom the simulator reads:

```{code-cell} python
:tags: [hide-input]
from aslbook import phantom

ph = phantom.slab()
fine = phantom.fine_slice()
pkg = Path(phantom.__file__).with_name("data")
for name in ("phantom_slab.npz", "phantom_slice_1mm.npz"):
    print(f"{name:<24} {(pkg / name).stat().st_size / 1e3:6.0f} kB")
print(f"\nphantom_slab.npz: gm, wm, csf fractions and the majority label on the {ph['gm'].shape} acquisition grid at {ph['voxel_mm']} mm")
print(f"  provenance: {ph['provenance']}")
print(f"phantom_slice_1mm.npz: the 1 mm label map {fine['dseg'].shape} at z = {fine['z_mm']:.1f} mm of the slab (display slice {fine['acq_slice']})")
```

Everything else the toy tier uses is computed in the page from these fractions and the
per-tissue constants of [Chapter 0.2](../00-frontmatter/the-simulated-datasets.md).
