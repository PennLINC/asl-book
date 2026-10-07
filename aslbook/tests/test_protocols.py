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


def test_3d_sidecars_drop_the_2d_fields_and_carry_the_train():
    p = protocols.grase(n_shots=4, te=0.010, n_pairs=2)
    assert p["MRAcquisitionType"] == "3D" and p["PulseSequenceType"] == "3Dgrase" and p["NumberShots"] == 4
    assert "SliceTiming" not in p and "TotalReadoutTime" not in p and p["PhaseEncodingDirection"] == "j-"
    assert protocols.slice_offsets(p) == [0.0] and protocols.is_3d(p)
    s = protocols.spiral(interleaves=8, te=0.012, n_pairs=2)
    assert s["PulseSequenceType"] == "spiral" and s["NumberShots"] == 8 and "PhaseEncodingDirection" not in s
    m = protocols.grase(label="multi_pld", plds=[0.1, 1.8], pairs_per_pld=2, n_shots=4)
    assert m["PostLabelingDelay"] == [0.1] * 4 + [1.8] * 4 and m["MRAcquisitionType"] == "3D"
    with pytest.raises(ValueError):
        protocols.readout_3d(protocols.pcasl(), "radial", 1)


def test_hadamard_rows_describe_the_decoded_volumes():
    p = protocols.hadamard(order=8, sub_bolus=0.4, pld=0.3, n_cycles=2)
    ctx = protocols.aslcontext(p)
    assert ctx == ["deltam"] * 14 and p["TotalAcquiredPairs"] == 2
    assert p["LabelingDuration"] == [0.4] * 14
    assert p["PostLabelingDelay"][:7] == [2.7, 2.3, 1.9, 1.5, 1.1, 0.7, 0.3]
    assert protocols.sub_boli(p) == [0.4] * 7 and protocols.bolus_duration(p) == pytest.approx(2.8)
    assert protocols.signal_times(p, ctx) == [pytest.approx(3.1)] * 14  # every decoded row of a cycle shares its excitation
    assert protocols.is_time_encoded(p) and "time-encoded" in protocols.describe(p)
    inc = protocols.hadamard(order=4, n_cycles=1, m0_type="Included")
    assert protocols.aslcontext(inc) == ["m0scan"] + ["deltam"] * 3 and inc["LabelingDuration"][0] == 0.0
    g = protocols.hadamard(order=8, n_cycles=1, readout="grase", n_shots=4)
    assert g["MRAcquisitionType"] == "3D" and g["EchoTime"] == 0.010


def test_look_locker_cycles_alternate_control_and_label():
    p = protocols.look_locker(tis=[0.8, 1.3, 1.8], flip_deg=35, n_pairs=2, tr=5.0)
    ctx = protocols.aslcontext(p)
    assert p["LookLocker"] is True and p["FlipAngle"] == 35
    assert protocols.readouts_per_cycle(p) == 3
    assert ctx == ["control"] * 3 + ["label"] * 3 + ["control"] * 3 + ["label"] * 3
    assert protocols.signal_times(p, ctx)[:3] == [0.8, 1.3, 1.8]
    assert protocols.readout_end(p, ctx) <= p["RepetitionTimePreparation"]
    assert "Look-Locker x 3" in protocols.describe(p)


def test_multi_te_gives_one_sidecar_per_echo():
    sides = protocols.multi_te([0.012, 0.033, 0.054], contrast="ge", flip_deg=60, n_pairs=2, pld=1.0)
    assert [s["EchoTime"] for s in sides] == [0.012, 0.033, 0.054]
    assert all(s["FlipAngle"] == 60 for s in sides)
    assert {k: v for k, v in sides[0].items() if k != "EchoTime"} == {k: v for k, v in sides[2].items() if k != "EchoTime"}
    se = protocols.multi_te([0.024, 0.048], contrast="se", n_pairs=1)
    assert se[0]["FlipAngle"] == 90


def test_cookbook_handles_multi_echo_and_3d_runs():
    cfg = cookbook.load_config()
    runs = {r.name: r for r in cookbook.expand(cfg, "multi-te")}
    r = runs["ge-pld10"]
    assert r.echo_times == [0.012, 0.033, 0.054] and r.sidecar_files() == ["asl.json", "asl-echo-2.json", "asl-echo-3.json"]
    assert r.command(cfg).count("--asl-json") == 3
    assert any("echo-3_part-mag" in f for f in cookbook.expected_files(r))
    had = {r.name: r for r in cookbook.expand(cfg, "hadamard")}["epi"]
    assert "sourcedata/sub-01/perf/sub-01_rawvolumes.tsv" in cookbook.expected_files(had)
    assert set(had.ctx) == {"deltam"} and len(had.ctx) == 49
    r3 = {r.name: r for r in cookbook.expand(cfg, "readout-3d")}["spiral"]
    assert r3.overlay["readout"]["interleaves"] == 8 and r3.overlay["acquisition"]["matrix"] == [68, 68]
    assert r3.protocol["NumberShots"] == 8
