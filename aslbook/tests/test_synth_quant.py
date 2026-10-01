import numpy as np
import pytest

from aslbook import grid, kinetic, phantom, presets, protocols, quant, synth


def test_axis_weights_tile_and_cover():
    w = grid.axis_weights(7, 1.0, 2, 3.5)
    assert [round(sum(x for _, x in ps), 9) for ps in w] == [1.0, 1.0]
    assert w[0] == [(0, pytest.approx(1 / 3.5)), (1, pytest.approx(1 / 3.5)), (2, pytest.approx(1 / 3.5)), (3, pytest.approx(0.5 / 3.5))]
    part = grid.axis_weights(5, 1.0, 2, 3.5)  # the second cell is only partly covered
    assert sum(x for _, x in part[1]) == pytest.approx(1.5 / 3.5)


def test_box_resampler_conserves_mass_and_fractions():
    labels = np.zeros((7, 7, 5), np.int16)
    labels[:4] = 1
    labels[4:] = 2
    rs = grid.BoxResampler(labels.shape, (1, 1, 1), (2, 2, 1), (3.5, 3.5, 5))
    fr = rs.fractions(labels, values=(1, 2))
    assert np.allclose(fr[1] + fr[2], 1.0)
    assert fr[1][0, 0, 0] == pytest.approx(1.0) and fr[1][1, 0, 0] == pytest.approx(0.5 / 3.5)
    assert rs.majority(labels)[1, 0, 0] == 2


def test_packaged_slab_is_consistent():
    ph = phantom.slab()
    assert ph["gm"].shape == (64, 68, 20)
    total = ph["gm"] + ph["wm"] + ph["csf"]
    assert total.max() <= 1 + 1e-5 and ph["mask"].sum() > 20000
    m = phantom.maps(ph)
    pure = ph["gm"] > 0.999
    assert np.allclose(m["perfusion"][pure], 60.0, atol=0.1) and np.allclose(m["att"][pure], 0.8, atol=1e-3)
    assert phantom.fine_slice()["dseg"].shape == (197, 233)


def test_toy_series_matches_the_kinetic_model_in_pure_gray_matter():
    p = protocols.pcasl(n_pairs=2)
    s = synth.series(p, noise_sd=0.0)
    ph = phantom.slab()
    pure = ph["gm"] > 0.999
    gm = presets.TISSUES["GM"]
    assert s.mag.shape == (64, 68, 20, 4) and s.m0scan.shape == (64, 68, 20)
    z = 0
    dm = kinetic.delta_m(3.6, gm.perfusion, gm.att, gm.t1, gm.m0, tau=1.8)
    diff = quant.subtract(s.mag, s.ctx)[..., 0]
    vox = pure[:, :, z]
    assert np.allclose(diff[:, :, z][vox], 100 * dm * np.exp(-0.012 / presets.T2_BLOOD), rtol=1e-4)
    assert np.allclose(s.deltam[:, :, z, 1][vox], dm, rtol=1e-5)
    # the last slice reads 0.76 s later, so its difference is smaller (post-bolus decay)
    assert diff[:, :, -1][pure[:, :, -1]].mean() < diff[:, :, 0][vox].mean()


def test_noise_and_subtraction_statistics():
    p = protocols.pcasl(n_pairs=20)
    s = synth.series(p, noise_sd=40.0, seed=1)
    ph = phantom.slab()
    bg = (ph["gm"] + ph["wm"] + ph["csf"]) == 0  # no tissue at all: pure noise
    assert s.mag[bg].mean() == pytest.approx(40 * np.sqrt(np.pi / 2), rel=0.05)  # Rayleigh floor
    d = quant.subtract(s.mag, s.ctx)
    assert d.shape[-1] == 20
    ds = quant.subtract(s.mag, s.ctx, method="surround")
    assert ds.shape == d.shape
    t = quant.tsnr(d)
    assert np.isfinite(t).all()


def test_white_paper_formula_recovers_truth_from_noise_free_deltam():
    gm = presets.TISSUES["GM"]
    dm = kinetic.delta_m(3.6, gm.perfusion, gm.att, gm.t1, gm.m0, tau=1.8)
    m0b = gm.m0  # the formula wants the tissue M0 (lambda converts to blood)
    cbf = quant.cbf_pcasl(dm, m0b, 1.8)
    # exact only when T1' -> T1b, ATT -> 0 assumptions hold; here the bias is the model's
    assert 40 < cbf < 70


def test_multi_pld_fit_recovers_cbf_and_att():
    gm = presets.TISSUES["GM"]
    plds = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    d = np.zeros((2, 2, 1, 6))
    truth_f, truth_att = 55.0, 1.1
    d[..., :] = kinetic.delta_m(plds + 1.8, truth_f, truth_att, gm.t1, gm.m0, tau=1.8)
    m0 = np.full((2, 2, 1), gm.m0)
    cbf, att = quant.fit_multi_pld(d, plds, m0, tau=1.8, t1_tissue=gm.t1)
    assert cbf[0, 0, 0] == pytest.approx(truth_f, rel=0.02)
    assert att[0, 0, 0] == pytest.approx(truth_att, abs=0.05)


def test_pv_correction_recovers_pure_tissue_values():
    rng = np.random.default_rng(0)
    gm = rng.uniform(0.1, 0.9, (12, 12, 1))
    wm = 1 - gm
    cbf = 60 * gm + 20 * wm
    f_gm, f_wm = quant.pv_correct(cbf, gm, wm, kernel=5)
    inner = (slice(2, -2), slice(2, -2), slice(None))
    assert np.allclose(f_gm[inner], 60, atol=1e-6) and np.allclose(f_wm[inner], 20, atol=1e-6)


def test_score():
    s = quant.score(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 4.0]), np.array([True, True, True]))
    assert s["bias"] == pytest.approx(-1 / 3) and s["n"] == 3 and 0.9 < s["r"] <= 1.0
