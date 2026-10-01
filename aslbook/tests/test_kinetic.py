import numpy as np
import pytest

from aslbook import kinetic, presets

# The aslscan unit-test constants: GM in the ASLDRO 3 T phantom.
F, DT, T1T, M0 = 60.0, 0.8, 1.33, 74.622


def t1p(f=F, t1t=T1T, lam=0.9):
    return 1.0 / (1.0 / t1t + (f / 6000.0) / lam)


def test_not_arrived_is_exactly_zero():
    for t in [0.5, DT, 0.0, -1.0]:
        assert kinetic.delta_m(t, F, DT, T1T, M0, label_type="PCASL", tau=1.8) == 0.0
        assert kinetic.delta_m(t, F, DT, T1T, M0, label_type="PASL", tau=0.7) == 0.0


def test_pcasl_arrived_matches_closed_form():
    t = 3.6
    want = 2 * (M0 / 0.9) * (F / 6000) * t1p() * 0.85 * np.exp(-DT / 1.65) * np.exp(-(t - 1.8 - DT) / t1p()) * (1 - np.exp(-1.8 / t1p()))
    got = kinetic.delta_m(t, F, DT, T1T, M0, label_type="PCASL", tau=1.8, alpha=0.85)
    assert got == pytest.approx(want, rel=1e-12)


def test_pcasl_arriving_matches_closed_form():
    t = 1.5
    want = 2 * (M0 / 0.9) * (F / 6000) * t1p() * 0.85 * np.exp(-DT / 1.65) * (1 - np.exp(-(t - DT) / t1p()))
    assert kinetic.delta_m(t, F, DT, T1T, M0, tau=1.8) == pytest.approx(want, rel=1e-12)


def test_pasl_arrived_matches_closed_form():
    t, tau, alpha = 1.8, 0.7, 0.98
    kk = 1 / 1.65 - 1 / t1p()
    q = np.exp(kk * t) * (np.exp(-kk * DT) - np.exp(-kk * (DT + tau))) / (kk * tau)
    want = 2 * (M0 / 0.9) * (F / 6000) * alpha * tau * np.exp(-t / 1.65) * q
    assert kinetic.delta_m(t, F, DT, T1T, M0, label_type="PASL", tau=tau) == pytest.approx(want, rel=1e-12)


def test_broadcasts_over_time_and_voxels():
    t = np.linspace(0, 5, 51)
    curve = kinetic.delta_m(t, F, DT, T1T, M0, tau=1.8)
    assert curve.shape == t.shape and curve[t <= DT].max() == 0 and curve.max() > 0
    maps = kinetic.delta_m(3.6, np.full((4, 5), F), np.full((4, 5), DT), T1T, M0, tau=1.8)
    assert maps.shape == (4, 5) and np.allclose(maps, maps[0, 0])


def test_delta_m_is_linear_in_perfusion_up_to_t1prime():
    a = kinetic.delta_m(3.6, 30.0, DT, T1T, M0, tau=1.8)
    b = kinetic.delta_m(3.6, 60.0, DT, T1T, M0, tau=1.8)
    assert 1.9 < b / a < 2.0  # T1' shortens slightly with perfusion


def test_tissue_se_and_zero_t1_guard():
    assert kinetic.tissue_se(M0, T1T, 4.0) == pytest.approx(M0 * (1 - np.exp(-4.0 / T1T)), rel=1e-15)
    assert kinetic.tissue_se(M0, 0.0, 4.0) == 0.0


def test_suppression_timeline_closed_forms():
    tr, t_read = 4.57, 3.8
    assert kinetic.tissue_mz(1.0, 1.33, tr, t_read) == pytest.approx(kinetic.tissue_se(1.0, 1.33, tr), rel=1e-12)
    one = kinetic.tissue_mz(1.0, 1.33, tr, t_read, [t_read - 1e-12], epsilon=1.0)
    assert one == pytest.approx(-kinetic.tissue_se(1.0, 1.33, tr), rel=1e-9)
    two = kinetic.tissue_mz(1.0, 1.33, tr, t_read, [2.5, 2.5], epsilon=1.0)
    assert two == pytest.approx(kinetic.tissue_se(1.0, 1.33, tr), rel=1e-12)
    assert kinetic.tissue_mz(1.0, 1.33, tr, t_read, [2.05, 3.276], epsilon=0.0) == pytest.approx(kinetic.tissue_se(1.0, 1.33, tr), rel=1e-12)
    # the aslscan design spec's worked asl002 example (first slice): GM 0.156, WM 0.175
    assert kinetic.tissue_mz(1.0, 1.33, tr, t_read, [2.05, 3.276], epsilon=1.0) == pytest.approx(0.156, abs=1e-3)
    assert kinetic.tissue_mz(1.0, 0.83, tr, t_read, [2.05, 3.276], epsilon=1.0) == pytest.approx(0.175, abs=1e-3)
    assert kinetic.label_factor([1, 2], 1.0) == 1.0
    assert kinetic.label_factor([1, 2], 0.95) == pytest.approx(0.81)
    assert kinetic.label_factor([], 0.95) == 1.0


def test_ir_reduces_to_saturation_recovery_at_90_degrees_without_inversion():
    e = np.exp(-4.0 / T1T)
    assert kinetic.tissue_ir(M0, T1T, 4.0, 1.0, fa_deg=90, fa_inv_deg=0) == pytest.approx(M0 * (1 - e), rel=1e-12)


def test_presets_are_consistent():
    assert set(presets.TISSUES) == {"GM", "WM", "CSF"}
    assert presets.REFERENCE.n_slices == len(presets.REFERENCE.slice_timing)
    r = presets.REFERENCE
    assert r.labeling_duration + r.post_labeling_delay + r.readout_duration <= r.repetition_time
