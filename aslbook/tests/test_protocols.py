import json

import pytest

from aslbook import cookbook, presets, protocols


def test_pcasl_sidecar_and_context():
    p = protocols.pcasl(n_pairs=3)
    ctx = protocols.aslcontext(p)
    assert ctx == ["control", "label"] * 3
    assert p["ArterialSpinLabelingType"] == "PCASL" and p["BackgroundSuppression"] is False
    assert protocols.bolus_duration(p) == presets.REFERENCE.labeling_duration
    assert protocols.signal_times(p, ctx) == [3.6] * 6
    assert max(protocols.slice_offsets(p)) == pytest.approx(0.76)
    assert protocols.readout_end(p, ctx) <= p["RepetitionTimePreparation"]
    json.dumps(p)  # serializable


def test_included_m0_rows_come_first():
    p = protocols.pcasl(n_pairs=2, m0_type="Included")
    ctx = protocols.aslcontext(p, m0_rows=2)
    assert ctx[:2] == ["m0scan", "m0scan"] and len(ctx) == 6
    assert protocols.signal_times(p, ctx)[:2] == [0.0, 0.0]


def test_pasl_uses_the_cutoff_as_tau_and_ti_as_time():
    p = protocols.pasl(ti=1.8, cutoff=0.7, n_pairs=1)
    assert protocols.bolus_duration(p) == 0.7
    assert protocols.signal_times(p, protocols.aslcontext(p)) == [1.8, 1.8]


def test_multi_pld_lists_one_delay_per_volume():
    p = protocols.multi_pld([0.5, 1.0], pairs_per_pld=2, tr=6.0)
    ctx = protocols.aslcontext(p)
    assert p["PostLabelingDelay"] == [0.5] * 4 + [1.0] * 4 and len(ctx) == 8
    assert protocols.signal_times(p, ctx)[-1] == pytest.approx(2.8)


def test_background_suppression_fields():
    p = protocols.pcasl(background_suppression=[2.1, 3.2])
    assert p["BackgroundSuppression"] is True and p["BackgroundSuppressionNumberPulses"] == 2


def test_cookbook_expands_every_dataset_into_valid_runs():
    cfg = cookbook.load_config()
    for ds in cfg["datasets"]:
        runs = cookbook.expand(cfg, ds)
        assert runs, ds
        names = [r.name for r in runs]
        assert len(names) == len(set(names)), ds
        for r in runs:
            assert protocols.readout_end(r.protocol, r.ctx) <= r.protocol["RepetitionTimePreparation"], (ds, r.name)
            assert r.phantom in cfg["phantoms"]
            toml = cookbook.to_toml(r.overlay)
            assert "[acquisition]" in toml and "aslscan" in r.command(cfg)
            for k in cookbook.expected_files(r):
                assert not k.startswith("/")


def test_sweep_and_variants_combine():
    cfg = cookbook.load_config()
    runs = cookbook.expand(cfg, "noise-sweep")
    assert [r.name for r in runs] == ["sigma10", "sigma40", "sigma80", "coils8-r2"]
    assert runs[0].overlay["acquisition"]["noise_variance"] == 100
    assert runs[3].overlay["acquisition"]["n_coils"] == 8


def test_toml_writer_round_trips_through_tomllib():
    import tomllib

    d = {"seed": 3, "acquisition": {"oversample": 2, "matrix": [64, 68], "noise_variance": 1600.0, "window": "hann"},
         "motion": {"mode": "random", "trans_mm": [2.0, 2.0, 1.0], "within_volume": {"dropout_rate": 0.1, "severity": 0.5}},
         "background_suppression": {"presaturation": True}}
    back = tomllib.loads(cookbook.to_toml(d))
    assert back == d
