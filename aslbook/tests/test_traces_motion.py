import numpy as np
import pytest

from aslbook import motion, traces


def test_trace_components_add_and_hold():
    spec = {"rate": 10, "duration": 20,
            "nod": "start=2,rise=1,hold=1,fall=1,trans=0:0:2,rot=0.1:0:0",
            "step": "start=10,rise=0.5,trans=1:0:0",
            "drift": "trans=0:0:0,rot=0:0:0.02"}
    rows = traces.synthesize(spec)
    assert rows.shape == (201, 7) and rows[0, 0] == 0.0 and rows[-1, 0] == pytest.approx(20.0)
    at = lambda t: rows[int(round(t * 10)), 1:]
    assert at(1.0)[2] == 0.0 and at(3.5)[2] == pytest.approx(2.0) and at(3.5)[3] == pytest.approx(0.1)
    assert at(2.5)[2] == pytest.approx(1.0)           # the half-cosine edge is halfway at mid-rise
    assert at(6.0)[2] == pytest.approx(0.0, abs=1e-12)  # back after the fall
    assert at(9.0)[0] == 0.0 and at(15.0)[0] == pytest.approx(1.0)  # the step holds
    assert at(20.0)[5] == pytest.approx(0.02) and at(10.0)[5] == pytest.approx(0.01)  # the drift is linear
    pose = traces.pose_at(spec, [3.5, 15.0])
    assert pose[0, 2] == pytest.approx(2.0) and pose[1, 0] == pytest.approx(1.0)


def test_trace_rejects_bad_specs(tmp_path):
    with pytest.raises(ValueError, match="needs rise"):
        traces.parse_component("nod", "start=1,fall=1")
    with pytest.raises(ValueError, match="unknown keys"):
        traces.parse_component("drift", "start=1")
    path = traces.write({"rate": 5, "duration": 2, "drift": "trans=1:0:0"}, tmp_path / "m.tsv")
    lines = path.read_text().splitlines()
    assert lines[0].split("\t") == traces.COLUMNS and len(lines) == 12
    assert "drift" in traces.describe({"rate": 5, "duration": 2, "drift": "trans=1:0:0"})


def test_pose_roundtrip_and_registration_recovers_a_shift():
    rng = np.random.default_rng(0)
    x, y, z = np.mgrid[0:32, 0:32, 0:12]
    vol = np.exp(-(((x - 15.5) / 8) ** 2 + ((y - 15.5) / 9) ** 2 + ((z - 5.5) / 3) ** 2)) * 1000
    vol = vol * (1 + 0.4 * np.sin(x / 1.5) * np.cos(y / 2.0) + 0.3 * np.cos(x / 3.0 + y / 2.5))   # texture, so rotations are visible
    t, r = [1.2, -0.7, 0.4], [0.0, 0.0, 3.0]
    moved = motion.move_volume(vol, t, r)
    back = motion.move_volume(moved, t, r, undo=True)
    assert np.abs(back - vol)[4:-4, 4:-4, 2:-2].max() < 0.1 * vol.max()
    mask = vol > 0.2 * vol.max()
    pts = motion.sample_points(mask, min(800, int(mask.sum()) // 2), seed=1)
    est = motion.register_volume(moved + rng.normal(0, 2, vol.shape), vol, pts)
    assert np.allclose(est[:3], t, atol=0.2) and abs(est[5] - 3.0) < 0.5
    for cost in ("ncc", "corratio", "nmi"):
        est = motion.register_volume(moved, vol, pts, cost=cost)
        assert np.allclose(est[:3], t, atol=0.3), cost


def test_fd_alternation_regression_and_pair_offsets():
    ctx = ["control", "label"] * 4
    poses = np.zeros((8, 6))
    poses[:, 0] = [0, 0.3, 0, 0.3, 0, 0.3, 0, 0.3]      # a pure label offset
    poses[:, 5] = np.linspace(0, 0.7, 8)                 # and a drift
    fd = motion.framewise_displacement(poses)
    assert fd[0] == 0 and fd[1] == pytest.approx(0.3 + np.radians(0.1) * 50)
    cleaned, removed = motion.regress_alternation(poses, ctx)
    assert removed[0] == pytest.approx(-0.15) and np.allclose(cleaned[:, 0], 0.15)
    assert removed[5] == pytest.approx(-0.05)           # half the within-pair drift step goes with it
    off = motion.pair_offsets(poses, ctx)
    assert off.shape == (4, 6) and np.allclose(off[:, 0], 0.3)
    robust, _ = motion.regress_alternation(poses, ctx, robust=True)
    assert np.allclose(robust[:, 0], 0.15, atol=1e-6)
