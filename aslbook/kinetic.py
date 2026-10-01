"""The Buxton general kinetic model and the longitudinal signal equations, as aslscan evaluates them.

Everything here is a faithful numpy port of aslscan's ``kinetic``, ``mrsignal`` and
``longitudinal`` modules (themselves diffed against ASLDRO), so a number computed in the page
is the number the simulator used. Seconds throughout; perfusion in ml/100 g/min as the phantom
stores it, converted to per-second inside. The functions broadcast over arrays, so a whole
volume's worth of voxels or a whole time axis can be evaluated in one call.
"""

from __future__ import annotations

import numpy as np

from . import presets


def _div0(n, d):
    """ASLDRO's guarded division: zero where the denominator is zero."""
    n, d = np.asarray(n, float), np.asarray(d, float)
    return np.divide(n, d, out=np.zeros(np.broadcast(n, d).shape), where=d != 0)


def t1_prime(t1_tissue, perfusion, lam: float = presets.LAMBDA):
    """The apparent tissue T1 with the label's outflow: ``1/T1' = 1/T1 + f/lambda`` (s)."""
    f = np.asarray(perfusion, float) / 6000.0
    return _div0(1.0, _div0(1.0, t1_tissue) + (f / lam if lam != 0 else 0.0))


def delta_m(
    t,
    perfusion,
    att,
    t1_tissue,
    m0,
    *,
    label_type: str = "PCASL",
    tau: float = presets.REFERENCE.labeling_duration,
    alpha: float | None = None,
    lam: float = presets.LAMBDA,
    t1b: float = presets.T1_BLOOD,
):
    """The label-control difference in longitudinal magnetization at signal time ``t``.

    ``t`` is measured from the start of labeling (for PASL, from the labeling pulse). The
    inputs broadcast: a time axis against a voxel's constants, or a volume of maps against one
    time. ``perfusion`` is in ml/100 g/min, ``att`` the arterial transit time (s),
    ``t1_tissue`` the tissue T1 (s), ``m0`` the tissue equilibrium magnetization (same units as
    the result). ``tau`` is the bolus duration: the labeling duration for (P)CASL, the cut-off
    delay for PASL. The result is zero before the bolus arrives (``t <= att``), follows the
    inflow curve while it arrives (``att < t < att + tau``), and decays after it has fully
    arrived.
    """
    if alpha is None:
        alpha = presets.ALPHA[label_type]
    t = np.asarray(t, float)
    f = np.asarray(perfusion, float) / 6000.0
    dt = np.asarray(att, float)
    t1t = np.asarray(t1_tissue, float)
    m0 = np.asarray(m0, float)
    m0b = m0 / lam if lam != 0 else np.zeros_like(m0)
    t1p = t1_prime(t1t, perfusion, lam)

    t, f, dt, t1p, m0b = np.broadcast_arrays(t, f, dt, t1p, m0b)
    arriving = (dt < t) & (t < dt + tau)
    arrived = t >= dt + tau
    out = np.zeros(t.shape)

    if label_type == "PASL":
        kk = (1.0 / t1b if t1b != 0 else 0.0) - _div0(1.0, t1p)
        decay = np.exp(-t / t1b) if t1b > 0 else np.zeros_like(t)
        with np.errstate(over="ignore", invalid="ignore"):
            num_a = np.exp(kk * t) * (np.exp(-kk * dt) - np.exp(-kk * t))
            q_a = _div0(num_a, kk * (t - dt))
            num_b = np.exp(kk * t) * (np.exp(-kk * dt) - np.exp(-kk * (dt + tau)))
            q_b = _div0(num_b, kk * tau)
        out = np.where(arriving, 2 * m0b * f * (t - dt) * alpha * decay * q_a, out)
        out = np.where(arrived, 2 * m0b * f * alpha * tau * decay * q_b, out)
    elif label_type in ("CASL", "PCASL"):
        decay = np.exp(-dt / t1b) if t1b != 0 else np.zeros_like(dt)
        with np.errstate(over="ignore", invalid="ignore"):
            q_a = 1.0 - np.exp(-_div0(t - dt, t1p))
            q_b = 1.0 - np.exp(-_div0(tau, t1p))
            tail = np.exp(-_div0(t - tau - dt, t1p))
        out = np.where(arriving, 2 * m0b * f * t1p * alpha * decay * q_a, out)
        out = np.where(arrived, 2 * m0b * f * t1p * alpha * decay * tail * q_b, out)
    else:
        raise ValueError(f"unknown labeling type {label_type!r}")
    return np.where(np.isfinite(out), out, 0.0)


def tissue_se(m0, t1, tr):
    """Longitudinal magnetization before the 90-degree excitation of a spin-echo readout
    repeated every ``tr``: ``m0 (1 - exp(-tr/t1))``; zero for a zero T1."""
    m0, t1 = np.asarray(m0, float), np.asarray(t1, float)
    return m0 * (1.0 - np.exp(-_div0(tr, t1)))


def tissue_mz(m0, t1, tr, t_read, pulse_times=(), epsilon: float = 0.95, presaturation: bool = False):
    """The tissue's signed longitudinal magnetization at readout under background suppression.

    Time runs from the start of labeling. The tissue starts from what the previous 90-degree
    excitation left, recovered for ``tr - t_read`` (or from zero under ``presaturation``),
    recovers toward ``m0`` with ``t1`` between events, and each suppression pulse at ``p``
    (seconds from labeling start, before ``t_read``) multiplies it by ``1 - 2 epsilon``. With
    no pulses this is :func:`tissue_se`. Broadcasts over ``m0``, ``t1`` and ``t_read`` (one
    readout time per slice, for instance); a pulse at or after a given ``t_read`` is ignored
    for that readout.
    """
    m0, t1, t_read = np.broadcast_arrays(np.asarray(m0, float), np.asarray(t1, float), np.asarray(t_read, float))
    inv = _div0(1.0, t1)  # zero rate where t1 == 0 -> the guard below returns 0

    def recover(mz, dt):
        return m0 - (m0 - mz) * np.exp(-dt * inv)

    mz = np.zeros_like(m0) if presaturation else recover(np.zeros_like(m0), tr - t_read)
    t = np.zeros_like(t_read)
    for p in sorted(pulse_times):
        applies = p < t_read
        kicked = recover(mz, p - t) * (1.0 - 2.0 * epsilon)
        mz = np.where(applies, kicked, mz)
        t = np.where(applies, p, t)
    out = recover(mz, t_read - t)
    return np.where(t1 == 0, 0.0, out)


def label_factor(pulse_times=(), epsilon: float = 0.95) -> float:
    """The factor every suppression pulse leaves on the label-control difference:
    ``prod (1 - 2 epsilon)``, i.e. ``(-1)^N`` for perfect pulses."""
    return float(np.prod([1.0 - 2.0 * epsilon for _ in pulse_times])) if len(pulse_times) else 1.0


def tissue_ir(m0, t1, tr, ti, fa_deg: float = 90.0, fa_inv_deg: float = 180.0):
    """The inversion-recovery steady state (Tofts 2009, eq. 7) without the transverse factor."""
    fa, fai = np.radians(fa_deg), np.radians(fa_inv_deg)
    m0, t1 = np.asarray(m0, float), np.asarray(t1, float)
    e_tr, e_ti = np.exp(-_div0(tr, t1)), np.exp(-_div0(ti, t1))
    num = m0 * (1 - (1 - np.cos(fai)) * e_ti - np.cos(fai) * e_tr)
    den = 1 - np.cos(fa) * np.cos(fai) * e_tr
    return np.sin(fa) * _div0(num, den)


def optimal_suppression_times(
    t1s,
    t_read: float,
    tr: float,
    n_pulses: int = 2,
    t_min: float = 0.0,
    epsilon: float = 0.95,
    n_grid: int = 60,
    t_max: float | None = None,
) -> tuple[list[float], float]:
    """Pulse times that minimize the summed squared tissue magnetization at ``t_read`` over the
    T1 values in ``t1s``, by a grid search between ``t_min`` and ``t_max`` (default: just
    before ``t_read``). For a 2D readout pass the first slice's excitation time as ``t_max``
    and the slice to null as ``t_read``, since no pulse can follow the first excitation.
    Returns the times and the residual (the root mean square of ``Mz / M0`` over the tissues)."""
    import itertools

    grid = np.linspace(t_min, (t_read if t_max is None else t_max) - 1e-3, n_grid)
    best, best_t = np.inf, None
    for combo in itertools.combinations(grid, n_pulses):
        mz = [tissue_mz(1.0, t1, tr, t_read, combo, epsilon) for t1 in t1s]
        cost = float(np.sqrt(np.mean(np.square(mz))))
        if cost < best:
            best, best_t = cost, [float(c) for c in combo]
    return best_t, best
