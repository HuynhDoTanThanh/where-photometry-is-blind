"""Forward-backward consistency: visibility, and an uncertainty score for networks without one.

For a flow ``F`` (frame a -> b) and the opposite flow ``F'`` (b -> a) of the same pass, let
``w(p) = F'(p + F(p))``, sampled bilinearly with zero padding outside the grid. Then

* the squared residual is ``r(p) = |F(p) + w(p)|^2``;
* a pixel is visible when ``r <= 0.01 (|F|^2 + |w|^2) + 0.5`` (the occlusion test of UnFlow,
  used with these constants by U2Flow, ARFlow and Sun-RAFT);
* for networks without an uncertainty head, RigidFill uses ``alpha = ln(r + 1e-3)``
  (:func:`fb_score`). Where ``p + F(p)`` leaves the grid, ``w = 0`` and ``r = |F|^2``, so the gate
  then acts as a motion threshold rather than a confidence.

All quantities are in grid pixels. Computations are float32, like the networks' own code.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

#: Constants of the forward-backward occlusion test.
OCC_SCALE = 0.01
OCC_BIAS = 0.5
#: Floor added to the residual before the logarithm in :func:`fb_score`.
FB_EPS = 1e-3


def warp_zeros(x, flow) -> np.ndarray:
    """``x`` sampled at ``p + flow(p)``: bilinear, zero outside the grid (``grid_sample`` with
    ``padding_mode="zeros", align_corners=True``).

    Args:
        x: ``(C, H, W)`` or ``(H, W)`` values on the grid.
        flow: ``(2, H, W)`` displacement in grid pixels.
    """
    a = np.asarray(x, dtype=np.float32)
    squeeze = a.ndim == 2
    if squeeze:
        a = a[None]
    f = np.asarray(flow, dtype=np.float32)
    _, h, w = a.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    px, py = xs + f[0], ys + f[1]
    fin = np.isfinite(px) & np.isfinite(py)
    px, py = np.where(fin, px, np.float32(-2.0)), np.where(fin, py, np.float32(-2.0))
    x0, y0 = np.floor(px), np.floor(py)
    x1, y1 = x0 + 1, y0 + 1
    out = np.zeros_like(a)
    for cx, cy, wgt in ((x0, y0, (x1 - px) * (y1 - py)), (x1, y0, (px - x0) * (y1 - py)),
                        (x0, y1, (x1 - px) * (py - y0)), (x1, y1, (px - x0) * (py - y0))):
        ok = fin & (cx >= 0) & (cx <= w - 1) & (cy >= 0) & (cy <= h - 1)
        ix = np.where(ok, cx, 0).astype(np.int64)
        iy = np.where(ok, cy, 0).astype(np.int64)
        out += np.where(ok[None], a[:, iy, ix] * wgt[None], np.float32(0.0))
    if not fin.all():
        out[:, ~fin] = np.nan
    return out[0] if squeeze else out


def fb_consistency(flow, flow_reverse, scale: float = OCC_SCALE,
                   bias: float = OCC_BIAS) -> Tuple[np.ndarray, np.ndarray]:
    """Squared forward-backward residual and visibility of ``flow``.

    Args:
        flow: ``(2, H, W)`` flow a -> b.
        flow_reverse: ``(2, H, W)`` flow b -> a from the same network pass.

    Returns:
        ``(r, visible)``: float32 ``(H, W)`` squared residual in grid px^2, and bool ``(H, W)``.
    """
    f = np.asarray(flow, dtype=np.float32)
    wb = warp_zeros(flow_reverse, f)
    d = f + wb
    r = (d * d).sum(0)
    mag = (f * f).sum(0) + (wb * wb).sum(0)
    visible = ~(r > np.float32(scale) * mag + np.float32(bias))
    return r.astype(np.float32), visible


def fb_score(flow, flow_reverse, eps: float = FB_EPS) -> np.ndarray:
    """Uncertainty substitute ``alpha = ln(r + 1e-3)`` for networks without an uncertainty head.

    Use it in place of a predicted log-variance, with the ``"arflow"`` or ``"sunraft"`` gate
    presets (or gate thresholds selected for your network).
    """
    r, _ = fb_consistency(flow, flow_reverse)
    return np.log(r + np.float32(eps)).astype(np.float32)
