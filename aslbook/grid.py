"""The box resampler aslscan uses to take the 1 mm phantom to an acquisition grid.

The acquisition grid has ``ceil(extent / voxel)`` cells per axis, corner-aligned with the
phantom (the first cells share the phantom's corner), so a 3.5 mm voxel overlaps three or four
1 mm cells per axis with fractional weights. Every ground-truth map on the acquisition grid is
the volume-weighted mean of the phantom values it overlaps (a majority vote for labels, a
masked mean for the transit time), and so are the tissue fractions the book scores
partial-volume corrections against. This module reproduces those rules in numpy.
"""

from __future__ import annotations

import math

import numpy as np


def axis_weights(n_src: int, d_src: float, n_dst: int, d_dst: float) -> list[list[tuple[int, float]]]:
    """For each target cell, the ``(source index, overlap fraction)`` pairs along one axis.

    Source cells are ``[i d_src, (i+1) d_src)`` and target cells ``[j d_dst, (j+1) d_dst)``,
    both from the shared corner at 0; the weight is the overlap length divided by ``d_dst``,
    so the weights of a fully covered target cell sum to 1 and those of a partial last cell
    to the covered fraction.
    """
    out = []
    for j in range(n_dst):
        lo, hi = j * d_dst, (j + 1) * d_dst
        i0, i1 = max(0, int(math.floor(lo / d_src))), min(n_src, int(math.ceil(hi / d_src)))
        pairs = []
        for i in range(i0, i1):
            w = (min(hi, (i + 1) * d_src) - max(lo, i * d_src)) / d_dst
            if w > 1e-9:
                pairs.append((i, w))
        out.append(pairs)
    return out


def acquisition_dims(src_dims, src_vox, dst_vox, matrix=None) -> tuple[int, int, int]:
    """``ceil(extent / voxel)`` per axis, the first two replaced by ``matrix`` when given."""
    dims = [int(math.ceil(n * ds / dd - 1e-9)) for n, ds, dd in zip(src_dims, src_vox, dst_vox)]
    if matrix is not None:
        dims[0], dims[1] = int(matrix[0]), int(matrix[1])
    return tuple(dims)


class BoxResampler:
    """Source grid to target grid box averaging, both corner-aligned and axis-aligned."""

    def __init__(self, src_dims, src_vox, dst_dims, dst_vox):
        self.src_dims, self.dst_dims = tuple(src_dims), tuple(dst_dims)
        self.axes = [axis_weights(n, ds, m, dd) for n, ds, m, dd in zip(src_dims, src_vox, dst_dims, dst_vox)]
        # dense per-axis weight matrices (dst x src); the 3-D box average is their tensor product
        self.mats = []
        for ax, pairs in enumerate(self.axes):
            m = np.zeros((self.dst_dims[ax], self.src_dims[ax]))
            for j, ps in enumerate(pairs):
                for i, w in ps:
                    m[j, i] = w
            self.mats.append(m)

    def mean(self, src: np.ndarray) -> np.ndarray:
        """Volume-weighted mean of ``src`` (3-D, source grid) on the target grid."""
        a = np.asarray(src, float)
        a = np.einsum("ji,ikl->jkl", self.mats[0], a)
        a = np.einsum("kj,ijl->ikl", self.mats[1], a)
        a = np.einsum("lk,ijk->ijl", self.mats[2], a)
        return a

    def masked_mean(self, src: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Mean of ``src`` over the overlapped source cells where ``mask`` holds; 0 where none do."""
        num = self.mean(np.where(mask, src, 0.0))
        den = self.mean(mask.astype(float))
        return np.divide(num, den, out=np.zeros_like(num), where=den > 0)

    def fractions(self, labels: np.ndarray, values=(1, 2, 3)) -> dict[int, np.ndarray]:
        """Volume fraction of each label value in every target cell."""
        return {v: self.mean(labels == v) for v in values}

    def majority(self, labels: np.ndarray, values=(0, 1, 2, 3)) -> np.ndarray:
        """Majority label by overlap weight, ties to the lower label."""
        fr = np.stack([self.mean(labels == v) for v in values])
        return np.asarray(values)[np.argmax(fr, axis=0)]
