"""Rigid head motion in the page: poses, resampling, a small registration, and the summaries
pipelines report (Chapters 10, 10.1, 10.2).

The conventions are the simulator's. A pose moves a point at ``p`` (mm, image axes) to
``R (p - c) + c + t`` with ``c`` the centre of the field of view, ``t`` the translation and
``R = Rz Ry Rx`` the rotation; angles are in degrees here and in the overlay, in radians in the
ground-truth tables. The registration is deliberately small enough to read: a similarity
measure at a few thousand sample voxels, minimized over the six parameters by Powell's method.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage, optimize

from . import presets

VOX = np.array(presets.REFERENCE.voxel_mm, float)


def rotation(rx: float, ry: float, rz: float) -> np.ndarray:
    """``R = Rz Ry Rx`` for angles in degrees, the simulator's order."""
    rx, ry, rz = np.radians([rx, ry, rz])
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def pose_to_voxels(t_mm, r_deg, shape, undo: bool = False, vox=VOX):
    """ndimage's ``(matrix, offset)`` for a pose about the field-of-view centre. ``undo=False``
    resamples an image the way the pose moves the head (output voxel ``q`` takes the input at
    ``pose^-1 q``); ``undo=True`` moves it back."""
    vox = np.asarray(vox, float)
    c = (np.array(shape[:3]) - 1) / 2 * vox
    R, t = rotation(*r_deg), np.asarray(t_mm, float)
    M, off = (R, c + t - R @ c) if undo else (R.T, c - R.T @ (c + t))
    return M * vox[None, :] / vox[:, None], off / vox


def move_volume(vol, t_mm, r_deg, undo: bool = False, order: int = 1, vox=VOX):
    """Resample a volume by a pose (trilinear by default), see :func:`pose_to_voxels`."""
    M, off = pose_to_voxels(t_mm, r_deg, vol.shape, undo, vox)
    return ndimage.affine_transform(vol, M, offset=off, order=order, mode="nearest")


def sample_points(mask: np.ndarray, n: int = 6000, seed: int = 0) -> np.ndarray:
    """``n`` voxel coordinates ``(3, n)`` drawn without replacement from a mask."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(np.flatnonzero(mask.ravel()), n, replace=False)
    return np.array(np.unravel_index(idx, mask.shape), float)


# ------------------------------------------------------------------ similarity measures
def _entropy(h):
    p = h / h.sum()
    p = p[p > 0]
    return -(p * np.log(p)).sum()


def cost_ssd(a, b):
    """Mean squared difference: the least-squares cost (MCFLIRT ``leastsq``, SPM realign,
    AFNI 3dvolreg), right only when both images have the same contrast."""
    return float(np.mean((a - b) ** 2))


def cost_ncc(a, b):
    """One minus the normalized correlation (MCFLIRT ``normcorr``): tolerates a scale and an
    offset between the images, not a change of contrast."""
    return float(1 - np.corrcoef(a, b)[0, 1])


def cost_corratio(a, b, bins: int = 32):
    """One minus the correlation ratio of ``b`` given ``a`` (FLIRT's default): the fraction of
    the variance of ``b`` not explained by the mean of ``b`` within each intensity bin of ``a``.
    Any one-to-one mapping of intensities, not only a linear one, scores perfectly."""
    edges = np.quantile(a, np.linspace(0, 1, bins + 1))
    k = np.clip(np.searchsorted(edges, a, side="right") - 1, 0, bins - 1)
    n = np.bincount(k, minlength=bins)
    m = np.bincount(k, weights=b, minlength=bins) / np.maximum(n, 1)
    within = np.sum((b - m[k]) ** 2)
    return float(within / np.sum((b - b.mean()) ** 2))


def cost_nmi(a, b, bins: int = 32):
    """Minus the normalized mutual information (MCFLIRT ``normmi``): the joint histogram's
    entropy against the marginals'. Any mapping of intensities that is consistent across the
    image scores well, which is why it is the choice across contrasts."""
    h, _, _ = np.histogram2d(a, b, bins=bins)
    return float(-(_entropy(h.sum(1)) + _entropy(h.sum(0))) / _entropy(h))


COSTS = {"ssd": cost_ssd, "ncc": cost_ncc, "corratio": cost_corratio, "nmi": cost_nmi}


def register_volume(moving, target, pts, cost: str = "ssd", x0=None, vox=VOX, xtol: float = 1e-3):
    """Rigid registration: the pose (tx, ty, tz in mm, rx, ry, rz in degrees) whose undoing
    best maps ``moving`` onto ``target`` at the sample voxels ``pts`` (3, N), by the named
    cost, minimized with Powell's method from ``x0`` (the identity by default)."""
    f = COSTS[cost]
    tv = target[tuple(pts.astype(int))]

    def objective(p):
        M, off = pose_to_voxels(p[:3], p[3:], moving.shape, undo=True, vox=vox)
        return f(ndimage.map_coordinates(moving, M @ pts + off[:, None], order=1, mode="nearest"), tv)

    x0 = np.zeros(6) if x0 is None else np.asarray(x0, float)
    return optimize.minimize(objective, x0, method="Powell", options={"xtol": xtol, "ftol": 1e-6}).x


def register_series(series, target, pts, cost: str = "ssd", smooth=(1.5, 1.5, 1.0), volumes=None, vox=VOX):
    """Register every volume (or the listed ``volumes``) of a series ``(x, y, z, v)`` to a
    target image, both smoothed by ``smooth`` voxels first; returns ``(len(volumes), 6)``."""
    sm = (lambda x: ndimage.gaussian_filter(x, smooth)) if smooth else (lambda x: x)
    tgt = sm(target)
    volumes = range(series.shape[-1]) if volumes is None else volumes
    return np.array([register_volume(sm(series[..., v]), tgt, pts, cost=cost, vox=vox) for v in volumes])


def apply_poses(series, poses, vox=VOX, order: int = 1):
    """Undo one estimated pose per volume by resampling; poses ``(v, 6)`` in mm and degrees."""
    return np.stack([move_volume(series[..., v], poses[v, :3], poses[v, 3:], undo=True, order=order, vox=vox)
                     for v in range(series.shape[-1])], -1)


# ------------------------------------------------------------------ summaries and corrections
def framewise_displacement(poses, radius_mm: float = 50.0) -> np.ndarray:
    """Power's framewise displacement {cite:p}`power2012` from poses ``(v, 6)`` in mm and
    degrees: the sum of the absolute changes of the six parameters between consecutive volumes,
    rotations converted to arc length on a sphere of ``radius_mm``. The first value is 0."""
    p = np.asarray(poses, float).copy()
    p[:, 3:] = np.radians(p[:, 3:]) * radius_mm
    return np.concatenate([[0.0], np.abs(np.diff(p, axis=0)).sum(1)])


def alternation(ctx) -> np.ndarray:
    """The label-control regressor of a context: +1 on control rows, -1 on label rows, 0 on
    the rest."""
    return np.array([1.0 if c == "control" else -1.0 if c == "label" else 0.0 for c in ctx])


def regress_alternation(poses, ctx, robust: bool = False, n_iter: int = 20):
    """Remove the control-label alternation from each pose parameter by least squares (ASLtbx
    and ExploreASL's step {cite:p}`wang2012`): fit an intercept and the ±1 regressor over the
    control and label volumes and subtract the fitted alternation. ``robust`` uses Huber
    weights (iteratively reweighted least squares) so that spikes count less. Returns the
    cleaned poses and the removed amplitude per parameter."""
    z = alternation(ctx)
    rows = z != 0
    X = np.column_stack([np.ones(rows.sum()), z[rows]])
    p = np.asarray(poses, float).copy()
    removed = np.zeros(p.shape[1])
    for j in range(p.shape[1]):
        y = p[rows, j]
        w = np.ones_like(y)
        for _ in range(n_iter if robust else 1):
            W = np.sqrt(w)[:, None]
            beta, *_ = np.linalg.lstsq(X * W, y * W[:, 0], rcond=None)
            if not robust:
                break
            r = y - X @ beta
            s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-12
            w = np.minimum(1.0, 1.345 * s / np.maximum(np.abs(r), 1e-12))
        removed[j] = beta[1]
        p[rows, j] -= beta[1] * z[rows]
    return p, removed


def pair_offsets(poses, ctx) -> np.ndarray:
    """The pose difference label minus control within each pair ``(pairs, 6)``: the motion
    the subtraction sees."""
    from . import quant

    pairs = quant.pair_indices(ctx)
    p = np.asarray(poses, float)
    return np.array([p[l] - p[c] for c, l in pairs])


# ------------------------------------------------------------------ pair-level outliers (Chapter 10's rule)
def deviation(d: np.ndarray, mask: np.ndarray, keep=None) -> np.ndarray:
    """Each pair's RMS deviation from the mean difference image over ``mask``: the series
    ``d`` is ``(x, y, z, pairs)``; ``keep`` restricts the mean to some pairs."""
    keep = np.ones(d.shape[-1], bool) if keep is None else keep
    m = d[..., keep].mean(-1)
    return np.array([np.sqrt(np.mean((d[..., k] - m)[mask] ** 2)) for k in range(d.shape[-1])])


def reject_pairs(d: np.ndarray, mask: np.ndarray, threshold: float = 2.5, mad_floor: float = 0.0):
    """Chapter 10's outlier rule: discard the pair farthest from the mean while its robust
    z-score (median and 1.4826 MAD of the deviations of the kept pairs) exceeds ``threshold``.
    ``mad_floor`` is a lower bound on the scale as a fraction of the median deviation: the
    simulator's thermal noise is the same in every pair to a fraction of a percent, so
    without a floor a few resampled pairs can be flagged for a change of noise texture
    alone, which no real series with physiological fluctuations would show. Returns the
    kept pairs (bool) and the final z-scores."""
    keep = np.ones(d.shape[-1], bool)
    while True:
        dv = deviation(d, mask, keep)
        med = np.median(dv[keep])
        mad = max(1.4826 * np.median(np.abs(dv[keep] - med)), mad_floor * med, 1e-12)
        z = np.where(keep, (dv - med) / mad, -np.inf)
        if z.max() <= threshold:
            return keep, z
        keep[np.argmax(z)] = False
