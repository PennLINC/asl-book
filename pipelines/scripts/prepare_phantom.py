"""Prepare the phantoms the pipeline simulates and the slab the book packages.

    micromamba run -n aslbook python scripts/prepare_phantom.py --config config/datasets.yaml [--package]

For every entry of the configuration's ``phantoms`` table: crop the source phantom (aslscan's
``hrgt_to_bids.py`` output at 1 mm) to the configured slab, copying every map and sidecar with
the affine shifted to the crop, and, when ``fieldmap`` is true, add a synthetic B0 field map
in Hz. With ``--package``, also write ``aslbook/data/phantom_slab.npz`` (the tissue fractions
on the reference acquisition grid, by aslscan's box-averaging rule) and
``phantom_slice_1mm.npz`` (one 1 mm slice of the label map) for the toy tier.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml

from aslbook import grid, phantom, presets

MAPS = ("perfusion", "att", "T1map", "T2map", "T2starmap", "M0map", "dseg")


def crop_phantom(source: Path, crop, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (x0, x1), (y0, y1), (z0, z1) = crop
    for name in MAPS:
        img = nib.load(source / f"{name}.nii.gz")
        cropped = img.slicer[x0:x1, y0:y1, z0:z1]
        nib.save(cropped, out / f"{name}.nii.gz")
        shutil.copy2(source / f"{name}.json", out / f"{name}.json")
    meta = json.loads((source / "phantom.json").read_text())
    meta["Crop"] = [list(c) for c in crop]
    meta["CroppedBy"] = "asl-book pipelines/scripts/prepare_phantom.py"
    (out / "phantom.json").write_text(json.dumps(meta, indent=2) + "\n")


def synthetic_fieldmap(dseg_img: nib.Nifti1Image) -> np.ndarray:
    """A smooth B0 offset in Hz on the phantom grid, shaped like the susceptibility field of a
    head: a positive lobe above the frontal sinuses (anterior, inferior), negative lobes at the
    temporal bones (lateral, inferior), decaying with distance, and zero far from them."""
    dseg = np.asarray(dseg_img.dataobj)
    nx, ny, nz = dseg.shape
    x, y, z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    # brain extent to place the lobes relative to the anatomy
    inside = dseg > 0
    xs, ys, zs = np.nonzero(inside)
    xc = xs.mean()
    y_ant, y_post = np.percentile(ys, 97), np.percentile(ys, 3)
    z_inf = np.percentile(zs, 5)
    field = np.zeros(dseg.shape)
    # frontal sinus lobe: +120 Hz, anterior-inferior, wide in x
    field += 120.0 * np.exp(-(((x - xc) / 28.0) ** 2 + ((y - (y_ant - 8)) / 22.0) ** 2 + ((z - (z_inf + 4)) / 22.0) ** 2))
    # temporal (ear canal / petrous) lobes: -90 Hz, lateral-inferior, mid-posterior
    for sx in (-1, 1):
        field += -90.0 * np.exp(-(((x - (xc + sx * 62)) / 20.0) ** 2 + ((y - (y_post + 80)) / 26.0) ** 2 + ((z - (z_inf - 2)) / 20.0) ** 2))
    # a gentle superior-inferior gradient from imperfect shimming
    field += 8.0 * (z - nz / 2) / nz
    return field.astype(np.float32)


def package_slab(slab_dir: Path, out_dir: Path, fieldmap_dir: Path | None = None) -> None:
    dseg_img = nib.load(slab_dir / "dseg.nii.gz")
    dseg = np.asarray(dseg_img.dataobj).astype(np.int16)
    vox = tuple(float(v) for v in dseg_img.header.get_zooms()[:3])
    r = presets.REFERENCE
    dims = grid.acquisition_dims(dseg.shape, vox, r.voxel_mm, matrix=r.matrix)
    rs = grid.BoxResampler(dseg.shape, vox, dims, r.voxel_mm)
    fr = rs.fractions(dseg, values=(1, 2, 3))
    majority = rs.majority(dseg)
    affine = dseg_img.affine.copy()
    affine[:3, :3] = np.diag(r.voxel_mm)  # corner-aligned: same corner, coarser cells
    field = np.zeros(dims, np.float32)
    if fieldmap_dir is not None:
        field = rs.mean(np.asarray(nib.load(fieldmap_dir / "fieldmap.nii.gz").dataobj, float)).astype(np.float32)
    np.savez_compressed(
        out_dir / "phantom_slab.npz",
        gm=fr[1].astype(np.float32), wm=fr[2].astype(np.float32), csf=fr[3].astype(np.float32),
        dseg=majority.astype(np.int16), fieldmap=field, voxel_mm=np.array(r.voxel_mm), affine=affine,
        provenance=f"ASLDRO hrgt_icbm_2009a_nls_3t via aslscan hrgt_to_bids.py, cropped to {slab_dir.name}, box-averaged to {dims} at {r.voxel_mm} mm by aslbook.grid",
    )
    k = phantom.DISPLAY_SLICE
    z_mm = (k + 0.5) * r.voxel_mm[2]
    z1 = int(z_mm)  # the 1 mm slice through the center of the display slice
    np.savez_compressed(out_dir / "phantom_slice_1mm.npz", dseg=dseg[:, :, z1], z_mm=z_mm, acq_slice=k)
    print(f"packaged {dims} slab ({int((fr[1] + fr[2] + fr[3] > 0.5).sum())} brain voxels) and the 1 mm slice at z = {z1} mm")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/datasets.yaml")
    ap.add_argument("--root", default="..", help="repository root, which `dir` entries are relative to")
    ap.add_argument("--package", action="store_true", help="also write the packaged slab into aslbook/data")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    root = Path(a.root).resolve()
    dirs = {}
    for key, ph in cfg["phantoms"].items():
        out = root / ph["dir"]
        crop_phantom(Path(ph["source"]), ph["crop"], out)
        if ph.get("fieldmap"):
            dseg_img = nib.load(out / "dseg.nii.gz")
            fmap = synthetic_fieldmap(dseg_img)
            nib.save(nib.Nifti1Image(fmap, dseg_img.affine), out / "fieldmap.nii.gz")
            (out / "fieldmap.json").write_text(json.dumps({"Units": "Hz", "Source": "synthetic (asl-book prepare_phantom.py): frontal-sinus and temporal lobes plus a linear shim term"}, indent=2) + "\n")
            print(f"{key}: field map {fmap.min():.0f} to {fmap.max():.0f} Hz")
        print(f"{key}: cropped to {nib.load(out / 'dseg.nii.gz').shape} at {out}")
        dirs[key] = out
    if a.package:
        fmap_dirs = [root / ph["dir"] for ph in cfg["phantoms"].values() if ph.get("fieldmap")]
        package_slab(dirs["slab"], Path(phantom.__file__).with_name("data"), fmap_dirs[0] if fmap_dirs else None)


if __name__ == "__main__":
    main()
