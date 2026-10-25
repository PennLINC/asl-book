import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from aslbook import presets, sketch

T1B, LD = presets.T1_BLOOD, presets.REFERENCE.labeling_duration


@pytest.fixture(params=list(sketch.ARTERY_PATHS))
def artery(request):
    return sketch.Artery(sketch.ARTERY_PATHS[request.param])


def test_pcasl_label_reaches_the_tissue_after_the_transit_time(artery):
    end = np.array([artery.length])
    assert artery.att("PCASL") == pytest.approx(presets.TISSUES["GM"].att)
    assert artery.state(end, 0.79, "PCASL")[0] == 0.0
    assert artery.state(end, 0.81, "PCASL")[0] == pytest.approx(np.exp(-artery.att("PCASL") / T1B))


def test_pcasl_labels_only_blood_that_crossed_the_plane_during_labeling(artery):
    below, above = artery.s_plane - 0.1, artery.s_plane + 0.1
    during = artery.state(np.array([below, above]), 1.0, "PCASL")
    assert during[0] == 0.0 and during[1] == pytest.approx(np.exp(-0.1 / artery.speed / T1B))
    after = artery.state(np.array([above]), LD + 0.2, "PCASL")   # crossed after the RF stopped
    assert after[0] == 0.0


def test_pasl_inverts_everything_below_the_top_at_once(artery):
    s = np.array([0.0, artery.s_top - 0.1, artery.s_top + 0.1])
    assert list(artery.state(s, 0.0, "PASL")) == [1.0, 1.0, 0.0]
    assert artery.att("PASL") < artery.att("PCASL")


def test_pasl_cutoff_saturates_blood_crossing_the_band_between_ti1_and_ti2(artery):
    ti1, ti2 = sketch.CUTOFF_TIMES
    t = 1.0
    s_early = artery.s_cut + (t - 0.5 * ti1) * artery.speed     # crossed before TI1: still labeled
    s_late = artery.s_cut + (t - 0.5 * (ti1 + t)) * artery.speed  # crossed between TI1 and now: saturated
    lab = artery.state(np.array([s_early, s_late]), t, "PASL")
    assert lab[0] == pytest.approx(np.exp(-t / T1B))
    assert lab[1] == -1.0


def test_the_whole_pulsed_bolus_is_cut_before_ti2(artery):
    # the tail of the inverted blood, out at the coil's edge, crosses the band before the train ends
    tail_crosses = (artery.s_cut + sketch.COIL_BELOW) / artery.speed
    assert sketch.CUTOFF_TIMES[0] < tail_crosses < sketch.CUTOFF_TIMES[1]


def test_sketch_draws_every_scheme():
    fig, axes = plt.subplots(1, 2)
    for ax, scheme in zip(axes, ("PCASL", "PASL")):
        sk = sketch.LabelingSketch(ax, scheme)
        for t in (0.0, 1.0, sketch.readout_start(scheme) + 0.1, 4.4):
            sk.update(t)
        assert "readout" in sketch.phase(sketch.readout_start(scheme) + 0.1, scheme)
    plt.close(fig)
