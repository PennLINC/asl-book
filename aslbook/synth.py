"""Toy ASL series built in the page from the packaged phantom.

This is aslscan's signal stage in numpy, without its acquisition stage: for every volume and
every slice, each tissue class contributes its fraction times its longitudinal magnetization
(the spin-echo steady state, or the background-suppression timeline) plus, on label rows,
minus the class's label-control difference from the kinetic model, evaluated at that slice's
readout time. The transverse decay to the echo time is applied per compartment (tissue T2,
blood T2), the sum is scaled like the simulator's images, and complex Gaussian noise is added
before the magnitude is taken. There is no k-space, so no distortion, ringing, ghosting or
coil effects: those are what the pipeline tier is for. The noise-free difference is returned
as the answer key, in the M0 map's units exactly as aslscan writes ``desc-deltam_gt``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import kinetic, phantom, presets, protocols


@dataclass
class Suppression:
    pulse_times: list[float] = field(default_factory=list)
    epsilon: float = 0.95
    presaturation: bool = False


@dataclass
class SimSeries:
    """A simulated series: magnitude images ``mag[x, y, z, volume]`` in image units, the
    context rows, the noise-free ``deltam`` truth per volume (M0 units, unsigned), the M0
    scan when the protocol has a separate one, the protocol, and the noise level."""

    mag: np.ndarray
    ctx: list[str]
    deltam: np.ndarray
    m0scan: np.ndarray | None
    protocol: dict
    noise_sd: float
    signal_scale: float

    @property
    def n_pairs(self) -> int:
        return sum(1 for r in self.ctx if r == "label")


def class_signals(p: dict, ctx: list[str], *, bs: Suppression | None = None, alpha: float | None = None,
                  t1b: float = presets.T1_BLOOD, lam: float = presets.LAMBDA):
    """Per class, per volume, per slice: the tissue magnetization and the unsigned label-control
    difference at readout, both in M0 units and before any transverse decay.

    Returns ``(tissue, blood)``, each ``{class: array (n_volumes, n_slices)}``.
    """
    lt = protocols.label_type(p)
    tau = protocols.bolus_duration(p)
    times = np.asarray(protocols.signal_times(p, ctx))
    offsets = np.asarray(protocols.slice_offsets(p))
    tr = float(p["RepetitionTimePreparation"])
    t_read = times[:, None] + offsets[None, :]
    tissue, blood = {}, {}
    pulses = list(p.get("BackgroundSuppressionPulseTime", [])) if p.get("BackgroundSuppression") else []
    if bs is not None:
        pulses = list(bs.pulse_times)
    eps = bs.epsilon if bs else 0.95
    presat = bs.presaturation if bs else False
    factor = kinetic.label_factor(pulses, eps)
    for name, t in presets.TISSUES.items():
        mz = np.empty(t_read.shape)
        for v, row in enumerate(ctx):
            if row == "m0scan" or not pulses and not presat:
                mz[v] = kinetic.tissue_se(t.m0, t.t1, tr)
            else:
                mz[v] = kinetic.tissue_mz(t.m0, t.t1, tr, t_read[v], pulses, eps, presat)
        dm = kinetic.delta_m(t_read, t.perfusion, t.att, t.t1, t.m0, label_type=lt, tau=tau, alpha=alpha, lam=lam, t1b=t1b)
        for v, row in enumerate(ctx):
            if row in ("m0scan", "control"):
                dm[v] = 0.0
            elif row == "label":
                dm[v] *= factor
        tissue[name], blood[name] = mz, dm
    return tissue, blood


def series(
    p: dict,
    ctx: list[str] | None = None,
    ph: dict | None = None,
    *,
    noise_sd: float = 0.0,
    seed: int = 0,
    signal_scale: float = presets.REFERENCE.signal_scale,
    bs: Suppression | None = None,
    alpha: float | None = None,
    t1b: float = presets.T1_BLOOD,
    t2b: float = presets.T2_BLOOD,
    lam: float = presets.LAMBDA,
    m0_tr: float = presets.REFERENCE.m0_repetition_time,
    t2_decay: bool = True,
    complex_noise: bool = True,
) -> SimSeries:
    """Simulate a series for protocol ``p`` (a sidecar from :mod:`aslbook.protocols`) on the
    packaged slab. ``noise_sd`` is the standard deviation of each of the real and imaginary
    noise components in image units (``signal_scale`` times M0 units), so the tissue SNR is
    about ``signal_scale * M0 * (1 - exp(-TR/T1)) / noise_sd``."""
    ph = ph or phantom.slab()
    ctx = ctx or protocols.aslcontext(p)
    fr = phantom.fractions(ph)
    te = float(p["EchoTime"])
    tissue, blood = class_signals(p, ctx, bs=bs, alpha=alpha, t1b=t1b, lam=lam)
    nx, ny, nz = ph["gm"].shape
    nv = len(ctx)
    clean = np.zeros((nx, ny, nz, nv), np.float32)
    deltam = np.zeros((nx, ny, nz, nv), np.float32)
    for name, t in presets.TISSUES.items():
        f = fr[name]
        e_t = np.exp(-te / t.t2) if t2_decay else 1.0
        e_b = np.exp(-te / t2b) if t2_decay else 1.0
        for v, row in enumerate(ctx):
            sign = -1.0 if row == "label" else 1.0
            for z in range(nz):
                clean[:, :, z, v] += f[:, :, z] * (tissue[name][v, z] * e_t + sign * blood[name][v, z] * e_b)
                if row == "label":
                    deltam[:, :, z, v] += f[:, :, z] * abs(blood[name][v, z])
    clean *= signal_scale
    rng = np.random.default_rng(seed)
    mag = add_noise(clean, noise_sd, rng, complex_noise)

    m0scan = None
    if p.get("M0Type") == "Separate":
        m0c = np.zeros((nx, ny, nz), np.float32)
        for name, t in presets.TISSUES.items():
            e_t = np.exp(-te / t.t2) if t2_decay else 1.0
            m0c += fr[name] * kinetic.tissue_se(t.m0, t.t1, m0_tr) * e_t
        m0scan = add_noise(m0c * signal_scale, noise_sd, rng, complex_noise)
    return SimSeries(mag, ctx, deltam, m0scan, p, noise_sd, signal_scale)


def add_noise(clean: np.ndarray, noise_sd: float, rng, complex_noise: bool = True) -> np.ndarray:
    """Add Gaussian noise of standard deviation ``noise_sd`` per component: complex (then the
    magnitude, Rician) by default, or real-valued."""
    if noise_sd <= 0:
        return clean.astype(np.float32)
    if complex_noise:
        z = clean + noise_sd * (rng.standard_normal(clean.shape) + 1j * rng.standard_normal(clean.shape))
        return np.abs(z).astype(np.float32)
    return (clean + noise_sd * rng.standard_normal(clean.shape)).astype(np.float32)


def add_motion(series: SimSeries, shifts: dict[int, tuple[float, float]], order: int = 1) -> SimSeries:
    """A toy rigid motion: shift the named volumes in-plane by ``(dx, dy)`` voxels (linear
    interpolation), the way a moving head shifts a finished image. Returns a copy."""
    from scipy import ndimage

    mag = series.mag.copy()
    for v, (dx, dy) in shifts.items():
        for z in range(mag.shape[2]):
            mag[:, :, z, v] = ndimage.shift(mag[:, :, z, v], (dx, dy), order=order, mode="constant")
    return SimSeries(mag, series.ctx, series.deltam, series.m0scan, series.protocol, series.noise_sd, series.signal_scale)
