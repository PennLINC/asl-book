import json

import nibabel as nib
import numpy as np
import pytest

from aslbook import data


def test_unknown_dataset_raises_helpful_error(monkeypatch):
    monkeypatch.delenv("ASLBOOK_DATA", raising=False)
    with pytest.raises(KeyError, match="not in the registry"):
        data.load_dataset("does-not-exist")


def test_local_override_and_run_access(tmp_path, monkeypatch):
    perf = tmp_path / "toy" / "run1" / "sub-01" / "perf"
    (perf / "ground-truth").mkdir(parents=True)
    img = nib.Nifti1Image(np.zeros((4, 4, 2, 2), dtype=np.float32), np.eye(4))
    nib.save(img, perf / "sub-01_part-mag_asl.nii.gz")
    (perf / "sub-01_part-mag_asl.json").write_text(json.dumps({"AslscanSimulation": {"Seed": 1}}))
    (perf / "sub-01_aslcontext.tsv").write_text("volume_type\ncontrol\nlabel\n")
    nib.save(nib.Nifti1Image(np.ones((4, 4, 2), np.float32), np.eye(4)), perf / "ground-truth" / "sub-01_desc-perfusion_gt.nii.gz")
    monkeypatch.setenv("ASLBOOK_DATA", str(tmp_path))
    ds = data.load_dataset("toy")
    assert ds.local and ds.runs() == ["run1"]
    run = ds.run()
    assert run.mag().shape == (4, 4, 2, 2) and run.context() == ["control", "label"]
    assert run.simulation()["Seed"] == 1 and run.truth("perfusion").mean() == 1.0
    with pytest.raises(FileNotFoundError):
        run.file("missing.nii.gz")


def test_registry_parses_and_asset_names():
    assert isinstance(data.registered_datasets(), list)
    assert data.asset_name("a/b/c.nii.gz") == "a__b__c.nii.gz"
