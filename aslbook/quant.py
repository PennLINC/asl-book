"""Subtraction, CBF quantification, multi-delay fitting, partial-volume correction, scoring.

These are the operations an ASL pipeline performs, written small enough to read. They take
plain arrays: series as ``(x, y, z, volume)``, maps as ``(x, y, z)``. Times in seconds,
perfusion in ml/100 g/min.
"""

from __future__ import annotations

import numpy as np

from . import kinetic, presets


# ----------------------------------------------------------------------------- subtraction
def pair_indices(ctx: list[str]) -> list[tuple[int, int]]:
    """``(control index, label index)`` for each consecutive control-label pair, in either order."""
    idx = [i for i, r in enumerate(ctx) if r in ("control", "label")]
    pairs = []
    for a, b in zip(idx[::2], idx[1::2]):
        if {ctx[a], ctx[b]} != {"control", "label"}:
            raise ValueError(f"rows {a} and {b} are not a control-label pair")
        pairs.append((a, b) if ctx[a] == "control" else (b, a))
    return pairs


def subtract(series: np.ndarray, ctx: list[str], method: str = "simple") -> np.ndarray:
    """Control minus label per pair: ``simple`` pairwise, or ``surround`` (each label against
    the mean of its two neighboring controls, which cancels linear drifts). Returns
    ``(x, y, z, n_pairs)``."""
    pairs = pair_indices(ctx)
    if method == "simple":
        return np.stack([series[..., c] - series[..., l] for c, l in pairs], axis=-1)
    if method == "surround":
        controls = [c for c, _ in pairs]
        out = []
        for k, (c, l) in enumerate(pairs):
            neighbors = [controls[j] for j in (k - 1, k + 1) if 0 <= j < len(controls)] + [c]
            out.append(series[..., neighbors].mean(axis=-1) - series[..., l])
        return np.stack(out, axis=-1)
    raise ValueError(method)


def pld_of_pairs(p: dict, ctx: list[str]) -> np.ndarray:
    """The post-labeling delay of each control-label pair (the label row's)."""
    pld = p["PostLabelingDelay"]
    if not isinstance(pld, list):
        return np.full(len(pair_indices(ctx)), float(pld))
    asl_rows = [i for i, r in enumerate(ctx) if r != "m0scan"]
    by_row = dict(zip(asl_rows, pld))
    return np.array([by_row[l] for _, l in pair_indices(ctx)])


def tsnr(diff: np.ndarray) -> np.ndarray:
    """Temporal SNR of the difference series: mean over pairs / SD over pairs."""
    sd = diff.std(axis=-1, ddof=1)
    return np.divide(diff.mean(axis=-1), sd, out=np.zeros(sd.shape), where=sd > 0)


# --------------------------------------------------------------------------- quantification
def slice_plds(pld: float, slice_offsets) -> np.ndarray:
    """The effective post-labeling delay of every slice of a 2D readout: the nominal delay plus
    the slice's offset in the readout."""
    return pld + np.asarray(slice_offsets, float)


def cbf_pcasl(
    deltam: np.ndarray,
    m0: np.ndarray,
    pld,
    *,
    tau: float = presets.REFERENCE.labeling_duration,
    alpha: float = presets.ALPHA["PCASL"],
    lam: float = presets.LAMBDA,
    t1b: float = presets.T1_BLOOD,
) -> np.ndarray:
    """The white-paper single-delay (P)CASL formula (Alsop et al. 2015):

    ``CBF = 6000 lambda dM exp(PLD/T1b) / (2 alpha T1b M0 (1 - exp(-tau/T1b)))``

    in ml/100 g/min. ``m0`` is the TISSUE equilibrium magnetization in the units of ``deltam``
    (the M0 scan, after :func:`m0_correction`); the formula's ``lambda`` converts it to the
    blood's, so do not divide by ``lambda`` first. ``pld`` may be an array broadcasting
    against ``deltam`` (one value per slice for a 2D readout). Voxels with ``m0 <= 0`` give 0.
    """
    pld = np.asarray(pld, float)
    num = 6000.0 * lam * deltam * np.exp(pld / t1b)
    den = 2.0 * alpha * t1b * m0 * (1.0 - np.exp(-tau / t1b))
    return np.divide(num, den, out=np.zeros(np.broadcast(num, den).shape), where=den > 0)


def cbf_pasl(
    deltam: np.ndarray,
    m0: np.ndarray,
    ti,
    *,
    ti1: float = 0.7,
    alpha: float = presets.ALPHA["PASL"],
    lam: float = presets.LAMBDA,
    t1b: float = presets.T1_BLOOD,
) -> np.ndarray:
    """The white-paper PASL formula: ``CBF = 6000 lambda dM exp(TI/T1b) / (2 alpha TI1 M0)``."""
    ti = np.asarray(ti, float)
    num = 6000.0 * lam * deltam * np.exp(ti / t1b)
    den = 2.0 * alpha * ti1 * m0
    return np.divide(num, den, out=np.zeros(np.broadcast(num, den).shape), where=den > 0)


def m0_correction(m0scan: np.ndarray, *, tr: float, t1: float = presets.TISSUES["GM"].t1,
                  te: float = presets.REFERENCE.echo_time, t2_tissue: float = presets.TISSUES["GM"].t2,
                  t2_blood: float = presets.T2_BLOOD) -> np.ndarray:
    """Turn a measured M0 image into the tissue M0 the formula wants: undo the T1 saturation of
    the M0 scan's repetition time and replace the tissue's T2 decay at TE by the blood's.
    ``t1`` may be a map (e.g. a run's ``T1map`` truth) so that CSF, whose T1 is 3 s, is not
    corrected with the gray matter value."""
    return m0scan / (1.0 - np.exp(-tr / t1)) * np.exp(-te / t2_blood) / np.exp(-te / t2_tissue)


def smooth_m0(m0: np.ndarray, sigma_vox: float = 2.0) -> np.ndarray:
    """Gaussian-smooth an M0 image (common practice: the calibration image is low in
    structure and the smoothing suppresses its noise). On a coarse grid the blur mixes the
    background and white matter into gray matter voxels and biases CBF there upward by tens
    of percent at ``sigma_vox = 2``; mask the image or keep the kernel small."""
    from scipy import ndimage

    return ndimage.gaussian_filter(m0, sigma_vox)


# -------------------------------------------------------------------------- multi-delay fit
def fit_multi_pld(
    deltam: np.ndarray,
    plds: np.ndarray,
    m0: np.ndarray,
    *,
    tau: float = presets.REFERENCE.labeling_duration,
    label_type: str = "PCASL",
    t1_tissue: float | np.ndarray = presets.TISSUES["GM"].t1,
    alpha: float | None = None,
    lam: float = presets.LAMBDA,
    t1b: float = presets.T1_BLOOD,
    att_grid: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    slice_offsets=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit CBF and the arterial transit time to a multi-delay difference series voxel by voxel.

    ``deltam`` is ``(x, y, z, n)`` with one volume per delay in ``plds`` (the mean over the
    pairs at that delay), in image units; ``m0`` the calibration image in the same units. For
    each candidate transit time on ``att_grid`` the model curve is evaluated with unit
    perfusion (the difference is linear in CBF apart from the weak dependence of T1' on it),
    the best-fitting CBF follows by least squares, and the transit time with the smallest
    residual wins; the CBF is then re-evaluated with T1' at the fitted CBF. With
    ``slice_offsets`` (one per z) the readout time of each slice is used. Returns
    ``(cbf, att)`` maps in ml/100 g/min and s.
    """
    if att_grid is None:
        att_grid = np.arange(0.2, 3.0001, 0.05)
    if alpha is None:
        alpha = presets.ALPHA[label_type]
    plds = np.asarray(plds, float)
    nx, ny, nz, n = deltam.shape
    tau_k = 0.0 if label_type == "PASL" else tau
    offsets = np.zeros(nz) if slice_offsets is None else np.asarray(slice_offsets, float)
    if mask is None:
        mask = m0 > 0
    cbf = np.zeros((nx, ny, nz))
    att = np.zeros((nx, ny, nz))
    t1t = np.broadcast_to(np.asarray(t1_tissue, float), (nx, ny, nz))
    for z in range(nz):
        t = plds + tau_k + offsets[z]  # signal times of this slice
        vox = np.flatnonzero(mask[:, :, z])
        if vox.size == 0:
            continue
        d = deltam[:, :, z, :].reshape(-1, n)[vox]  # (V, n)
        m = m0[:, :, z].reshape(-1)[vox]
        t1 = t1t[:, :, z].reshape(-1)[vox]
        # unit-CBF curves for every (att, voxel, time): (A, V, n)
        curves = np.stack([
            kinetic.delta_m(t[None, :], 1.0, a, t1[:, None], m[:, None], label_type=label_type, tau=tau, alpha=alpha, lam=lam, t1b=t1b)
            for a in att_grid
        ])
        gg = np.einsum("avn,avn->av", curves, curves)
        dg = np.einsum("vn,avn->av", d, curves)
        f = np.divide(dg, gg, out=np.zeros_like(dg), where=gg > 0)  # (A, V)
        resid = np.einsum("vn,vn->v", d, d)[None, :] - f * dg
        best = np.argmin(resid, axis=0)
        f_best = f[best, np.arange(vox.size)]
        a_best = att_grid[best]
        # one refinement of T1' at the fitted CBF
        g = kinetic.delta_m(t[None, :], f_best[:, None], a_best[:, None], t1[:, None], m[:, None],
                            label_type=label_type, tau=tau, alpha=alpha, lam=lam, t1b=t1b)
        gg2 = np.einsum("vn,vn->v", g, g)
        scale = np.divide(np.einsum("vn,vn->v", d, g), gg2, out=np.ones_like(gg2), where=gg2 > 0)
        cbf[:, :, z].reshape(-1)[vox] = np.clip(f_best * scale, 0, None)
        att[:, :, z].reshape(-1)[vox] = a_best
    return cbf, att


# ----------------------------------------------------------------------- partial volume
def pv_correct(cbf: np.ndarray, gm: np.ndarray, wm: np.ndarray, kernel: int = 5, min_frac: float = 0.05,
               mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Linear-regression partial-volume correction (Asllani et al. 2008): within a
    ``kernel x kernel`` in-plane neighborhood, solve ``cbf_i = gm_i f_GM + wm_i f_WM`` for the
    pure-tissue perfusions by least squares. Returns ``(cbf_gm, cbf_wm)`` maps, zero where the
    neighborhood holds too little of a tissue (its summed fraction below ``min_frac``)."""
    from numpy.lib.stride_tricks import sliding_window_view

    r = kernel // 2
    pad = ((r, r), (r, r), (0, 0))
    c = np.pad(cbf, pad)
    g = np.pad(gm, pad)
    w = np.pad(wm, pad)
    keep = np.pad(np.ones(cbf.shape, bool) if mask is None else mask, pad)
    # windows: (x, y, z, kernel, kernel), rearranged so the last axis enumerates neighbors
    win = lambda a: np.moveaxis(sliding_window_view(a, (kernel, kernel), axis=(0, 1)), (3, 4), (-2, -1)).reshape(*cbf.shape, -1)
    C, G, W, K = win(c), win(g), win(w), win(keep).astype(float)
    G, W, C = G * K, W * K, C * K
    gg, ww, gw = (G * G).sum(-1), (W * W).sum(-1), (G * W).sum(-1)
    gc, wc = (G * C).sum(-1), (W * C).sum(-1)
    det = gg * ww - gw * gw
    with np.errstate(divide="ignore", invalid="ignore"):
        f_gm = (ww * gc - gw * wc) / det
        f_wm = (gg * wc - gw * gc) / det
        # a neighborhood holding one tissue only has a singular 2 x 2 system, but that
        # tissue's perfusion is still determined: fall back to the one-tissue solution
        f_gm_one = gc / gg
        f_wm_one = wc / ww
    ok = det > 1e-9 * np.maximum(gg * ww, 1e-30)
    has_gm, has_wm = G.sum(-1) > min_frac, W.sum(-1) > min_frac
    f_gm = np.where(ok, f_gm, np.where(has_gm & ~has_wm, f_gm_one, 0.0))
    f_wm = np.where(ok, f_wm, np.where(has_wm & ~has_gm, f_wm_one, 0.0))
    f_gm = np.where(has_gm, f_gm, 0.0)
    f_wm = np.where(has_wm, f_wm, 0.0)
    return np.nan_to_num(f_gm), np.nan_to_num(f_wm)


# ------------------------------------------------------------------------------- scoring
def score(fit: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Bias (mean fit - truth), RMSE, and Pearson correlation inside ``mask``."""
    f, t = np.asarray(fit, float)[mask], np.asarray(truth, float)[mask]
    ok = np.isfinite(f) & np.isfinite(t)
    f, t = f[ok], t[ok]
    if f.size < 2:
        return {"bias": np.nan, "rmse": np.nan, "r": np.nan, "n": int(f.size)}
    r = np.corrcoef(f, t)[0, 1] if f.std() > 0 and t.std() > 0 else np.nan
    return {"bias": float((f - t).mean()), "rmse": float(np.sqrt(((f - t) ** 2).mean())), "r": float(r), "n": int(f.size)}


def roi_means(img: np.ndarray, fractions: dict[str, np.ndarray], threshold: float = 0.9) -> dict[str, float]:
    """Mean of ``img`` in the voxels that are at least ``threshold`` of each tissue."""
    return {name: float(img[fr >= threshold].mean()) if (fr >= threshold).any() else np.nan for name, fr in fractions.items()}
