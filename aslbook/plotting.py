"""House style for figures: slice mosaics, error maps, and fit-vs-truth panels.

Every phantom figure in the book uses the same axial slice and the same intensity windows so
the reader learns one brain. Color maps are colorblind-safe: ``gray`` for magnitude,
``twilight`` for phase, ``viridis`` for scalar maps, ``RdBu_r`` (zero-centered) for
differences; CBF maps use ``inferno`` with a fixed 0-90 ml/100 g/min window.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

#: Categorical palette (colorblind-validated, fixed order, never cycled past 8).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

#: Tissue colors are fixed for the whole book: GM orange, WM blue, CSF aqua.
TISSUE_COLORS = {"GM": PALETTE[1], "WM": PALETTE[0], "CSF": PALETTE[2]}

#: Text and grid tones (never used for data).
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "grid": "#e4e3df"}

CMAPS = {"magnitude": "gray", "phase": "twilight", "scalar": "viridis", "diff": "RdBu_r", "cbf": "inferno", "att": "viridis", "fraction": "viridis"}

#: Fixed windows for quantities with natural ranges.
FIXED_WINDOWS = {"cbf": (0.0, 90.0), "att": (0.0, 2.0), "fraction": (0.0, 1.0)}

#: Intensity windows (percentiles) per contrast otherwise.
WINDOWS = {"magnitude": (1, 99), "scalar": (1, 99), "diff": (2, 98)}


def set_style() -> None:
    """Apply the book's matplotlib style: thin marks, recessive axes, the fixed palette."""
    mpl.rcParams.update({
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK["secondary"],
        "axes.labelcolor": INK["primary"],
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.grid": True,
        "grid.color": INK["grid"],
        "grid.linewidth": 0.6,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.color": INK["secondary"],
        "ytick.color": INK["secondary"],
        "legend.fontsize": 8,
        "legend.frameon": False,
        "figure.dpi": 110,
        "figure.figsize": (7.0, 3.2),
        "image.cmap": "gray",
        "image.interpolation": "nearest",
    })


def take_slice(vol: np.ndarray, k: int | None = None, plane: str = "axial") -> np.ndarray:
    """One slice of an ``(x, y, z)`` array oriented for display: axial with anterior at the
    top, coronal and sagittal with superior at the top. ``k`` defaults to the middle."""
    axis = {"sagittal": 0, "coronal": 1, "axial": 2}[plane]
    if k is None:
        k = vol.shape[axis] // 2
    sl = np.take(vol, k, axis=axis)
    return np.rot90(sl)


def _limits(img: np.ndarray, kind: str) -> tuple[float, float]:
    if kind in FIXED_WINDOWS:
        return FIXED_WINDOWS[kind]
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, WINDOWS.get(kind, (1, 99)))
    if kind == "diff":
        m = max(abs(lo), abs(hi))
        return -m, m
    return float(lo), float(hi)


def show_image(ax, img: np.ndarray, title: str | None = None, kind: str = "magnitude", colorbar: bool = False, **kw):
    """Display a 2-D array (already oriented) with the book's color maps and no axes decoration."""
    if np.iscomplexobj(img):
        img = np.abs(img)
    kw.setdefault("cmap", CMAPS[kind])
    if "vmin" not in kw and "vmax" not in kw:
        kw["vmin"], kw["vmax"] = _limits(img, kind)
    im = ax.imshow(img, **kw)
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    if colorbar:
        ax.figure.colorbar(im, ax=ax, shrink=0.75)
    return im


def show_slice(ax, vol: np.ndarray, k: int | None = None, title: str | None = None, kind: str = "magnitude", **kw):
    """``show_image`` of the axial slice ``k`` of an ``(x, y, z)`` volume."""
    return show_image(ax, take_slice(vol, k), title, kind, **kw)


def mosaic(vols: dict[str, np.ndarray], k: int | None = None, kind: str = "magnitude", share_window: bool = True,
           ncols: int | None = None, figsize_per: float = 2.8, colorbar: bool = True):
    """Side-by-side axial slices of several volumes with a shared window; returns ``(fig, axes)``."""
    names = list(vols)
    ncols = ncols or len(names)
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per * ncols, figsize_per * nrows * 1.05), squeeze=False)
    slices = {n: take_slice(vols[n], k) for n in names}
    if share_window:
        lo, hi = _limits(np.concatenate([s.ravel() for s in slices.values()]), kind)
    for ax, n in zip(axes.ravel(), names):
        vmin, vmax = (lo, hi) if share_window else _limits(slices[n], kind)
        im = ax.imshow(slices[n], cmap=CMAPS[kind], vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(n, fontsize=9)
        ax.axis("off")
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    if colorbar:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7)
    return fig, axes


def fit_vs_truth(fit: np.ndarray, truth: np.ndarray, mask: np.ndarray, name: str, k: int | None = None,
                 kind: str = "cbf", unit: str = "", max_points: int = 20000, seed: int = 0, lims=None):
    """The book's standard 4-panel: fitted map | truth | difference | scatter with identity line."""
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.3))
    f2, t2, m2 = (take_slice(a, k) for a in (fit, truth, mask.astype(float)))
    lo, hi = lims if lims is not None else _limits(np.concatenate([t2[m2 > 0], f2[m2 > 0]]), kind)
    axes[0].imshow(np.where(m2 > 0, f2, np.nan), cmap=CMAPS[kind], vmin=lo, vmax=hi)
    axes[0].set_title(f"{name}: estimate", fontsize=9)
    im0 = axes[1].imshow(np.where(m2 > 0, t2, np.nan), cmap=CMAPS[kind], vmin=lo, vmax=hi)
    axes[1].set_title(f"{name}: truth", fontsize=9)
    fig.colorbar(im0, ax=axes[1], shrink=0.8)
    diff = np.where(m2 > 0, f2 - t2, np.nan)
    dlo, dhi = _limits(diff, "diff")
    im = axes[2].imshow(diff, cmap=CMAPS["diff"], vmin=dlo, vmax=dhi)
    axes[2].set_title("estimate − truth", fontsize=9)
    fig.colorbar(im, ax=axes[2], shrink=0.8)
    for ax in axes[:3]:
        ax.axis("off")
    f, t = fit[mask.astype(bool)], truth[mask.astype(bool)]
    ok = np.isfinite(f) & np.isfinite(t)
    f, t = f[ok], t[ok]
    if f.size > max_points:
        sel = np.random.default_rng(seed).choice(f.size, max_points, replace=False)
        f, t = f[sel], t[sel]
    axes[3].scatter(t, f, s=2, alpha=0.3, rasterized=True)
    axes[3].plot([lo, hi], [lo, hi], "k--", lw=1)
    axes[3].set(xlabel=f"truth {unit}".strip(), ylabel=f"estimate {unit}".strip(), xlim=(lo, hi), ylim=(lo, hi))
    axes[3].set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return fig, axes


def animate(fig, update, frames, alt: str, fps: float = 2, width: int = 480, dpi: float | None = None):
    """Render a matplotlib animation as an inline looping GIF; returns an ``IPython.display.HTML``.

    ``update(frame)`` redraws the figure for one frame. The GIF is embedded as a data URI, so
    the built book needs no extra files. ``alt`` is the text description for screen readers.
    """
    import base64
    import html
    import os
    import tempfile

    from IPython.display import HTML
    from matplotlib.animation import FuncAnimation, PillowWriter

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps)
    fd, path = tempfile.mkstemp(suffix=".gif", prefix="aslbook_")
    os.close(fd)
    try:
        anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
        with open(path, "rb") as f:
            gif_b64 = base64.b64encode(f.read()).decode("ascii")
    finally:
        os.remove(path)
    plt.close(fig)
    return HTML(f'<img src="data:image/gif;base64,{gif_b64}" alt="{html.escape(alt)}" style="width: {width}px; max-width: 100%;">')


def sequence_diagram(ax, blocks: list[tuple[str, float, float, str]], t_max: float, labels: dict[str, str] | None = None):
    """Draw a timing diagram: ``blocks`` are ``(lane, start, end, color)`` in seconds."""
    lanes = []
    for lane, *_ in blocks:
        if lane not in lanes:
            lanes.append(lane)
    for i, lane in enumerate(lanes):
        y = len(lanes) - 1 - i
        ax.plot([0, t_max], [y, y], color=INK["grid"], lw=1)
        ax.text(-0.05, y, (labels or {}).get(lane, lane), ha="right", va="center", fontsize=8, color=INK["primary"])
    for lane, t0, t1, color in blocks:
        y = len(lanes) - 1 - lanes.index(lane)
        ax.fill_between([t0, t1], y - 0.35, y + 0.35, color=color, alpha=0.75, lw=0)
    ax.set(xlim=(-0.02 * t_max, t_max), ylim=(-0.6, len(lanes) - 0.4), yticks=[], xlabel="time from the start of labeling (s)")
    ax.grid(False)
    for s in ("left",):
        ax.spines[s].set_visible(False)
