"""Timed head-motion traces for the simulator, built from nods, steps and drifts.

aslscan's ``[motion] trajectory`` accepts a TSV with a ``time`` column (seconds from the start
of the series' first acquisition) and the six pose parameters, sampled as fast as one likes;
each 2D shot or 3D segment then takes the pose at its own excitation time (Chapter 10.1). The
pipeline describes such a trace in ``datasets.yaml`` as a small recipe (``motion.trace``) and
this module writes the file, so Appendix A can print the recipe instead of twenty thousand rows.

The recipe follows aslscan's ``tools/motion_trace.py``, component for component, so a trace
written here equals one written there:

- ``nod``: ``start=,rise=,hold=,fall=,trans=x:y:z,rot=x:y:z[,shape=cosine|linear]``, a rise
  from zero to the amplitude, a hold, and a fall back; edges are half cosines by default
- ``step``: ``start=,rise=,trans=,rot=[,shape=]``, a rise with no fall
- ``drift``: ``trans=,rot=``, a linear ramp from zero at the first sample to the amplitude at
  the last

Translations are in mm, rotations in radians (``Rz Ry Rx`` about the field-of-view centre),
and the components add.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

COLUMNS = ["time", "trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
_ALLOWED = {
    "nod": {"start", "rise", "hold", "fall", "trans", "rot", "shape"},
    "step": {"start", "rise", "trans", "rot", "shape"},
    "drift": {"trans", "rot"},
}
_REQUIRED = {"nod": ("start", "rise", "fall"), "step": ("start", "rise"), "drift": ()}


def _vec(s: str, what: str) -> list[float]:
    parts = str(s).split(":")
    if len(parts) != 3:
        raise ValueError(f"{what} = {s!r}: expected x:y:z")
    v = [float(p) for p in parts]
    if not all(math.isfinite(x) for x in v):
        raise ValueError(f"{what} = {s!r}: not finite")
    return v


def parse_component(kind: str, spec: str) -> dict:
    """``key=value,...`` into a component dict (the generator's own representation)."""
    if kind not in _ALLOWED:
        raise ValueError(f"unknown trace component {kind!r}: nod, step or drift")
    fields = {}
    for item in spec.split(","):
        if "=" not in item:
            raise ValueError(f"{kind} {spec!r}: {item!r} is not key=value")
        k, v = item.split("=", 1)
        fields[k.strip()] = v.strip()
    unknown = set(fields) - _ALLOWED[kind]
    if unknown:
        raise ValueError(f"{kind}: unknown keys {sorted(unknown)}; expected {sorted(_ALLOWED[kind])}")
    c = {
        "kind": kind,
        "trans": _vec(fields.get("trans", "0:0:0"), f"{kind} trans"),
        "rot": _vec(fields.get("rot", "0:0:0"), f"{kind} rot"),
        "shape": fields.get("shape", "cosine"),
    }
    if c["shape"] not in ("cosine", "linear"):
        raise ValueError(f"{kind} shape = {c['shape']!r}: expected cosine or linear")
    for k in _REQUIRED[kind]:
        if k not in fields:
            raise ValueError(f"{kind} needs {k}=")
    for k in ("start", "rise", "hold", "fall"):
        if k not in _ALLOWED[kind]:
            continue
        x = float(fields.get(k, "0"))
        if not math.isfinite(x) or (k != "start" and x < 0):
            raise ValueError(f"{kind} {k} = {x}: not a finite{'' if k == 'start' else ' non-negative'} number")
        c[k] = x
    return c


def _edge(u: np.ndarray, shape: str) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    return u if shape == "linear" else 0.5 * (1.0 - np.cos(np.pi * u))


def _profile(c: dict, t: np.ndarray) -> np.ndarray:
    """The component's scale (0..1) at the sample times ``t``."""
    if c["kind"] == "drift":
        return (t - t[0]) / (t[-1] - t[0]) if t[-1] != t[0] else np.zeros_like(t)
    s = c["start"]
    up = np.where(t < s, 0.0, 1.0 if c["rise"] == 0.0 else _edge((t - s) / c["rise"], c["shape"]))
    if c["kind"] == "step":
        return up
    down_at = s + c["rise"] + c["hold"]
    down = 1.0 - (np.where(t < down_at, 0.0, 1.0) if c["fall"] == 0.0 else _edge((t - down_at) / c["fall"], c["shape"]))
    return np.where(t < down_at, up, down)


def components(spec: dict) -> list[dict]:
    """The components of a recipe: the keys ``nod``, ``step`` and ``drift``, each one spec
    string or a list of them."""
    out = []
    for kind in ("nod", "step", "drift"):
        v = spec.get(kind)
        if v is None:
            continue
        for s in ([v] if isinstance(v, str) else list(v)):
            out.append(parse_component(kind, s))
    return out


def synthesize(spec: dict) -> np.ndarray:
    """The trace of a recipe ``{rate, duration, start?, nod?, step?, drift?}`` as an array
    ``(samples, 7)``: time, three translations (mm), three rotations (rad)."""
    rate, duration, start = float(spec["rate"]), float(spec["duration"]), float(spec.get("start", 0.0))
    if not (math.isfinite(rate) and rate > 0) or not (math.isfinite(duration) and duration > 0):
        raise ValueError("a trace needs a positive rate (Hz) and duration (s)")
    n = int(round(duration * rate)) + 1
    t = start + np.arange(n) / rate
    rows = np.zeros((n, 7))
    rows[:, 0] = t
    for c in components(spec):
        rows[:, 1:] += _profile(c, t)[:, None] * np.array(c["trans"] + c["rot"])[None, :]
    return rows


def pose_at(spec: dict, times) -> np.ndarray:
    """The recipe's pose at arbitrary times, linearly interpolated as the simulator does
    between its samples: ``(len(times), 6)`` in mm and radians."""
    rows = synthesize(spec)
    times = np.atleast_1d(np.asarray(times, float))
    return np.stack([np.interp(times, rows[:, 0], rows[:, k]) for k in range(1, 7)], -1)


def write(spec: dict, path: Path | str) -> Path:
    """Write the recipe's trace as the TSV aslscan reads; returns the path."""
    path = Path(path)
    rows = synthesize(spec)
    with open(path, "w", newline="\n") as f:
        f.write("\t".join(COLUMNS) + "\n")
        for r in rows:
            f.write("\t".join(f"{v:.6g}" for v in r) + "\n")
    return path


def describe(spec: dict) -> str:
    """One line per component, in words, for the cookbook."""
    lines = [f"sampled at {spec['rate']} Hz for {spec['duration']} s"]
    for c in components(spec):
        tr = ", ".join(f"{v:g}" for v in c["trans"])
        ro = ", ".join(f"{math.degrees(v):.2g}" for v in c["rot"])
        if c["kind"] == "drift":
            lines.append(f"drift: a linear ramp to ({tr}) mm and ({ro}) degrees at the last sample")
        elif c["kind"] == "step":
            lines.append(f"step at {c['start']:g} s: a {c['shape']} rise of {c['rise']:g} s to ({tr}) mm and ({ro}) degrees, held")
        else:
            lines.append(f"nod at {c['start']:g} s: rise {c['rise']:g} s, hold {c['hold']:g} s, fall {c['fall']:g} s, to ({tr}) mm and ({ro}) degrees")
    return "\n".join(lines)
