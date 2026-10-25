"""Sagittal sketches of labeling: where the label is made and how it travels to the brain.

A schematic head, not to scale, facing left, with two feeding arteries (a carotid in front,
a vertebral behind) drawn as paths from the bottom of the neck to the tissue each one feeds.
Blood moves along each path in plug flow, at a speed set so that blood leaving the PCASL
labeling plane reaches the tissue after the gray matter transit time. ``Artery.state`` says
whether the blood at a position along a path at a time is fresh, labeled (and how much of
its label T1 decay has left), or saturated by the pulsed scheme's bolus cut-off;
``LabelingSketch`` draws one scheme at one time, for snapshots and for ``plotting.animate``.

The pulsed scheme is the book's FAIR with a Q2TIPS cut-off: at t = 0 all the blood below the
imaging region, out to the edge of the transmit coil, is inverted, and from TI1 to TI2 a
train of saturation pulses at the top of the inverted region destroys the label of whatever
blood crosses it.
"""

from __future__ import annotations

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse, Polygon, Rectangle

from . import kinetic, presets
from .plotting import INK, PALETTE, TISSUE_COLORS

R = presets.REFERENCE
GM = presets.TISSUES["GM"]

PLANE_Y = -3.0              #: height of the (P)CASL labeling plane, across the neck
INVERTED_TOP = 4.0          #: the pulsed inversion covers everything below this height ...
COIL_BELOW = 6.0            #: ... and reaches this far below the bottom of the drawing
CUTOFF_BAND = (3.0, 4.0)    #: the Q2TIPS saturation band at the top of the inverted region
CUTOFF_TIMES = (0.7, 1.3)   #: TI1 and TI2: the saturation train runs between them (s)
IMAGING = (5.0, 13.0)       #: the imaging slab, bottom and top
NECK_BOTTOM = -8.0

#: The two feeding arteries, from the bottom of the neck to the tissue they feed.
ARTERY_PATHS = {
    "carotid": np.array([(-1.4, NECK_BOTTOM), (-1.4, 1.0), (-0.7, 3.8), (-2.6, 9.5)]),
    "vertebral": np.array([(1.8, NECK_BOTTOM), (1.8, 1.5), (0.8, 4.2), (2.6, 10.0)]),
}

FRESH = tuple(1 - 0.45 * (1 - c) for c in to_rgb(PALETTE[7]))   # unlabeled arterial blood
LABELED = to_rgb(PALETTE[6])                                     # inverted blood water
SATURATED = to_rgb(INK["secondary"])                             # label destroyed by the cut-off
BRAIN = to_rgb("0.84")
TISSUE = to_rgb(TISSUE_COLORS["GM"])
SCHEME_COLOR = TISSUE_COLORS["GM"]                               # the labeling RF, as in Chapter 4
CUTOFF_COLOR = PALETTE[3]
IMAGING_COLOR = PALETTE[0]


class Artery:
    """A feeding artery as a polyline, with plug flow along it at a constant speed."""

    def __init__(self, points: np.ndarray, att: float = GM.att):
        self.points = np.asarray(points, float)
        self.arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(self.points, axis=0).T))])
        self.length = float(self.arc[-1])
        self.s_plane = self.at_height(PLANE_Y)
        self.s_top = self.at_height(INVERTED_TOP)
        self.s_cut = self.at_height(np.mean(CUTOFF_BAND))
        self.speed = (self.length - self.s_plane) / att

    def at_height(self, y: float) -> float:
        """Arc length at which the path reaches height ``y`` (every path rises monotonically)."""
        return float(np.interp(y, self.points[:, 1], self.arc))

    def xy(self, s) -> np.ndarray:
        s = np.asarray(s, float)
        return np.stack([np.interp(s, self.arc, self.points[:, 0]), np.interp(s, self.arc, self.points[:, 1])], -1)

    def att(self, scheme: str) -> float:
        """Transit time from where the scheme's label starts to the end of the path (s)."""
        start = self.s_top if scheme == "PASL" else self.s_plane
        return (self.length - start) / self.speed

    def state(self, s, t: float, scheme: str = "PCASL", tau: float = R.labeling_duration) -> np.ndarray:
        """The label carried by the blood at arc length ``s`` at time ``t`` after labeling begins.

        Positive values are the fraction of the inversion left after T1 decay in blood, 0 is
        fresh blood, and -1 is blood whose label the Q2TIPS train destroyed (PASL only).
        """
        s = np.asarray(s, float)
        out = np.zeros(s.shape)
        if scheme == "PASL":
            s0 = s - self.speed * t                                   # where this blood was at t = 0
            inverted = (s0 <= self.s_top) & (s0 >= -COIL_BELOW)
            crossed = t - (s - self.s_cut) / self.speed               # when it crosses the saturation band
            saturated = (s >= self.s_cut) & (crossed >= CUTOFF_TIMES[0]) & (crossed <= CUTOFF_TIMES[1])
            out[inverted] = np.exp(-t / presets.T1_BLOOD)
            out[saturated] = -1.0
            return out
        age = (s - self.s_plane) / self.speed                         # time since it crossed the plane
        labeled = (age >= 0) & (t - age >= 0) & (t - age <= tau)
        out[labeled] = np.exp(-age[labeled] / presets.T1_BLOOD)
        return out


def readout_start(scheme: str) -> float:
    """Start of the 2D readout in the book's protocols: TI 1.8 s for PASL, LD + PLD otherwise."""
    return R.post_labeling_delay if scheme == "PASL" else R.labeling_duration + R.post_labeling_delay


def tissue_label(t, att: float, scheme: str = "PCASL") -> np.ndarray:
    """ΔM / M0 of pure gray matter fed through a path with transit time ``att`` (the kinetic model)."""
    tau = CUTOFF_TIMES[0] if scheme == "PASL" else R.labeling_duration
    return kinetic.delta_m(t, GM.perfusion, att, GM.t1, 1.0, label_type=scheme, tau=tau)


def phase(t: float, scheme: str) -> str:
    """What the sequence is doing at time ``t``."""
    ro = readout_start(scheme)
    if ro <= t < ro + R.readout_duration:
        return "readout"
    if t >= ro + R.readout_duration:
        return "recovery before the next TR"
    if scheme == "PASL":
        if t < 0.05:
            return "inversion pulse"
        if CUTOFF_TIMES[0] <= t <= CUTOFF_TIMES[1]:
            return "Q2TIPS cut-off"
        return "bolus drains" if t < CUTOFF_TIMES[0] else "inversion time"
    return "labeling" if t < R.labeling_duration else "post-labeling delay"


def draw_head(ax, title: str | None = None, blood: str | None = None) -> None:
    """The static sagittal head (facing left), its two arteries, and the imaging slab.

    The arteries are empty tubes for ``LabelingSketch`` to fill, or drawn in ``blood``.
    """
    ax.add_patch(Rectangle((-3.0, NECK_BOTTOM), 6.6, 10, color="0.92", zorder=0))
    ax.add_patch(Ellipse((0.3, 8.3), 13, 15, color="0.92", zorder=0))
    ax.add_patch(Polygon([(-5.6, 7.6), (-7.8, 5.0), (-5.3, 4.2)], color="0.92", zorder=0))     # nose
    ax.add_patch(Ellipse((0.3, 8.8), 10.4, 10.8, color=BRAIN, zorder=1))
    for pts in ARTERY_PATHS.values():
        ax.plot(*pts.T, color="white", lw=6, zorder=2, solid_capstyle="round", solid_joinstyle="round")
        if blood:
            ax.plot(*pts.T, color=blood, lw=2, zorder=2, solid_capstyle="round", solid_joinstyle="round")
    ax.add_patch(Rectangle((-7, IMAGING[0]), 14, IMAGING[1] - IMAGING[0], fill=False, ec=IMAGING_COLOR, lw=1.5, ls="--", zorder=6))
    ax.text(7.3, 9, "imaging\nslab", color=IMAGING_COLOR, fontsize=8, va="center")
    ax.text(-9.8, 15.8, "← anterior", fontsize=7, color=INK["secondary"])
    ax.set(xlim=(-10, 12), ylim=(NECK_BOTTOM, 17), aspect="equal")
    if title:
        ax.set_title(title)
    ax.set_axis_off()


class LabelingSketch:
    """One scheme's sagittal sketch on ``ax``; ``update(t)`` redraws the blood, tissue and pulses."""

    def __init__(self, ax, scheme: str = "PCASL", n_seg: int = 240):
        self.scheme = scheme
        self.arteries = [Artery(p) for p in ARTERY_PATHS.values()]
        draw_head(ax, f"{scheme}: {'a slab inverted at once' if scheme == 'PASL' else 'a plane, labeling for 1.8 s'}")
        self.s = [np.linspace(0, a.length, n_seg + 1) for a in self.arteries]
        self.lines = []
        for a, s in zip(self.arteries, self.s):
            xy = a.xy(s)
            lc = LineCollection(np.stack([xy[:-1], xy[1:]], 1), linewidths=3.2, capstyle="round", zorder=3)
            ax.add_collection(lc)
            self.lines.append(lc)
        self.tissue = [ax.add_patch(Circle(tuple(p[-1]), 1.3, color=BRAIN, zorder=2.5)) for p in ARTERY_PATHS.values()]
        tt = np.linspace(0, 5, 501)                                   # one tissue scale for every scheme: the PCASL peak
        self.norm = max(tissue_label(tt, a.att("PCASL")).max() for a in self.arteries)
        if scheme == "PASL":
            self.pulse = ax.add_patch(Rectangle((-7, NECK_BOTTOM), 14, INVERTED_TOP - NECK_BOTTOM, color=SCHEME_COLOR, zorder=1.5))
            self.cut = ax.add_patch(Rectangle((-7, CUTOFF_BAND[0]), 14, CUTOFF_BAND[1] - CUTOFF_BAND[0], fill=False,
                                              hatch="////", ec=CUTOFF_COLOR, lw=0, zorder=4))
            ax.text(-9.8, -1.5, "inversion\nregion", color=SCHEME_COLOR, fontsize=8, va="center")
            ax.text(7.3, 3.5, "Q2TIPS\nsaturation", color=CUTOFF_COLOR, fontsize=8, va="center")
        else:
            self.pulse = ax.add_patch(Rectangle((-4.5, PLANE_Y - 0.2), 9, 0.4, color=SCHEME_COLOR, zorder=4))
            ax.text(-9.8, PLANE_Y, "labeling\nplane", color=SCHEME_COLOR, fontsize=8, va="center")
        self.readout = ax.add_patch(Rectangle((-7, IMAGING[0]), 14, IMAGING[1] - IMAGING[0], color=IMAGING_COLOR, alpha=0.0, zorder=5))
        self.clock = ax.text(1.0, 16.4, "", ha="center", fontsize=8.5, color=INK["primary"])

    def update(self, t: float) -> None:
        for a, s, lc in zip(self.arteries, self.s, self.lines):
            lab = a.state(0.5 * (s[:-1] + s[1:]), t, self.scheme)
            frac = np.clip(lab, 0, 1)[:, None]
            rgb = np.where(lab[:, None] < 0, SATURATED, np.array(FRESH) + frac * (np.array(LABELED) - FRESH))
            lc.set_color(rgb)
        for a, c in zip(self.arteries, self.tissue):
            level = float(np.clip(tissue_label(t, a.att(self.scheme), self.scheme) / self.norm, 0, 1))
            c.set_facecolor(tuple(b + level * (x - b) for b, x in zip(BRAIN, TISSUE)))
        if self.scheme == "PASL":
            self.pulse.set_alpha(0.55 if t < 0.05 else 0.08)
            self.cut.set_visible(CUTOFF_TIMES[0] <= t <= CUTOFF_TIMES[1])
        else:
            self.pulse.set_alpha(1.0 if t < R.labeling_duration else 0.2)
        what = phase(t, self.scheme)
        self.readout.set_alpha(0.18 if what == "readout" else 0.0)
        self.clock.set_text(f"t = {t:.2f} s: {what}")


def legend_handles() -> list:
    """Legend entries for the colors of the sketch."""
    line = lambda c, lab: Line2D([], [], color=c, lw=3.2, label=lab)
    return [line(FRESH, "fresh arterial blood"), line(LABELED, "labeled blood (fades with T1)"),
            line(SATURATED, "label destroyed by the cut-off"),
            Line2D([], [], marker="o", ls="", ms=9, color=TISSUE, label="label in gray matter")]
