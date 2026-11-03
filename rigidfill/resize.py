"""Upsampling from the network grid to the native image, as U2Flow does it.

U2Flow's ``resize_flow`` interpolates bilinearly with aligned corners
(``torch.nn.functional.interpolate(..., mode="bilinear", align_corners=True)``) and multiplies
``u`` by ``W / w`` and ``v`` by ``H / h``. These functions reproduce it in numpy (float32, the
same index and weight rule, and the same arithmetic), so the package needs no deep-learning
framework.

A NaN anywhere in a native pixel's 2 x 2 interpolation support makes that pixel NaN, also when
its weight is zero (as in PyTorch): a native pixel next to an undefined grid value is undefined.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def _axis(n_in: int, n_out: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Indices and weights of one axis (PyTorch's align_corners=True rule, float32)."""
    if n_in == n_out:
        i0 = np.arange(n_out)
        return i0, i0, np.ones(n_out, np.float32), np.zeros(n_out, np.float32)
    scale = np.float32(n_in - 1) / np.float32(n_out - 1) if n_out > 1 else np.float32(0.0)
    src = scale * np.arange(n_out, dtype=np.float32)
    i0 = np.minimum(np.floor(src).astype(np.int64), n_in - 1)
    l1 = np.clip(src - i0.astype(np.float32), np.float32(0.0), np.float32(1.0))
    l1 = l1.astype(np.float32)
    i1 = i0 + (i0 < n_in - 1).astype(np.int64)
    return i0, i1, (np.float32(1.0) - l1).astype(np.float32), l1


def _fma(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """``a * b + c`` with one rounding to float32 (a fused multiply-add): the product of two
    float32 numbers is exact in float64."""
    return (a.astype(np.float64) * b.astype(np.float64) + c.astype(np.float64)).astype(np.float32)


def upsample_bilinear(x, size: Sequence[int]) -> np.ndarray:
    """Bilinear resize with aligned corners of ``(H, W)`` or ``(C, H, W)`` to ``size = (H', W')``.

    No value scaling (use :func:`upsample_flow` for flows). Returns float32. Each output is
    ``fma(t0, wy0, t1 * wy1)`` with ``t_i = fma(x_i0, wx0, x_i1 * wx1)``, the arithmetic of
    PyTorch's CPU kernel when compiled with fused multiply-adds (as on Apple silicon), which it
    then matches bit for bit; elsewhere the two agree to float32 rounding.
    """
    a = np.asarray(x, dtype=np.float32)
    squeeze = a.ndim == 2
    if squeeze:
        a = a[None]
    _, h, w = a.shape
    H, W = int(size[0]), int(size[1])
    y0, y1, ly0, ly1 = _axis(h, H)
    x0, x1, lx0, lx1 = _axis(w, W)
    rows0, rows1 = a[:, y0], a[:, y1]
    top = _fma(rows0[:, :, x0], lx0, rows0[:, :, x1] * lx1)
    bot = _fma(rows1[:, :, x0], lx0, rows1[:, :, x1] * lx1)
    out = _fma(top, ly0[:, None], bot * ly1[:, None])
    return out[0] if squeeze else out


def upsample_flow(flow, size: Sequence[int]) -> np.ndarray:
    """A grid flow ``(2, h, w)`` at the native size ``(H, W)``: U2Flow's ``resize_flow``.

    Bilinear with aligned corners, then ``u / (w / W)`` and ``v / (h / H)``. Returns float32
    ``(2, H, W)``; NaN where the interpolation support touches an undefined grid value.
    """
    f = np.asarray(flow, dtype=np.float32)
    _, h, w = f.shape
    H, W = int(size[0]), int(size[1])
    out = upsample_bilinear(f, (H, W))
    out[0] = out[0] / np.float32(w / float(W))
    out[1] = out[1] / np.float32(h / float(H))
    return out


def upsample_score(alpha, size: Sequence[int]) -> np.ndarray:
    """An uncertainty map ``(h, w)`` at the native size, without value scaling.

    A native pixel whose interpolation support touches a non-finite grid value gets ``-inf``,
    so an uncertainty gate (``alpha > tau``) never selects it.
    """
    a = np.asarray(alpha, dtype=np.float32)
    fin = np.isfinite(a)
    out = upsample_bilinear(np.where(fin, a, np.float32(0.0)), size)
    if not fin.all():
        out[upsample_bilinear((~fin).astype(np.float32), size) > 0] = -np.inf
    return out
