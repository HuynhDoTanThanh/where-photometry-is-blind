"""Small visualisation helpers for the examples (numpy only)."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def _color_wheel() -> np.ndarray:
    """The 55-colour wheel of the Middlebury flow benchmark (Baker et al., IJCV 2011)."""
    segments = ((15, (255, 0, 0), (255, 255, 0)), (6, (255, 255, 0), (0, 255, 0)),
                (4, (0, 255, 0), (0, 255, 255)), (11, (0, 255, 255), (0, 0, 255)),
                (13, (0, 0, 255), (255, 0, 255)), (6, (255, 0, 255), (255, 0, 0)))
    rows = []
    for n, a, b in segments:
        t = np.arange(n, dtype=np.float64)[:, None] / n
        rows.append(np.asarray(a, np.float64) * (1 - t) + np.asarray(b, np.float64) * t)
    return np.floor(np.concatenate(rows, 0))


def flow_to_color(flow, max_flow: Optional[float] = None) -> np.ndarray:
    """RGB uint8 ``(H, W, 3)`` colour coding of a ``(2, H, W)`` flow (hue = direction,
    saturation = magnitude relative to ``max_flow``, default the largest finite magnitude).
    Undefined pixels are black."""
    f = np.asarray(flow, dtype=np.float64)
    u, v = f[0], f[1]
    ok = np.isfinite(u) & np.isfinite(v)
    u, v = np.where(ok, u, 0.0), np.where(ok, v, 0.0)
    mag = np.hypot(u, v)
    if max_flow is None:
        max_flow = float(mag[ok].max()) if ok.any() else 1.0
    r = np.clip(mag / max(max_flow, 1e-9), 0.0, 1.0)
    wheel = _color_wheel()
    n = wheel.shape[0]
    ang = np.arctan2(-v, -u) / np.pi
    fk = (ang + 1) / 2 * (n - 1)
    k0 = np.floor(fk).astype(int)
    k1 = (k0 + 1) % n
    frac = (fk - k0)[..., None]
    col = ((1 - frac) * wheel[k0] + frac * wheel[k1]) / 255.0
    col = 1 - r[..., None] * (1 - col)
    out = np.floor(255 * col).astype(np.uint8)
    out[~ok] = 0
    return out


def overlay_mask(rgb, mask, color: Sequence[int] = (0, 200, 0), alpha: float = 0.5
                 ) -> np.ndarray:
    """``rgb`` with ``mask`` tinted in ``color``."""
    img = np.asarray(rgb, dtype=np.float64).copy()
    m = np.asarray(mask, dtype=bool)
    img[m] = (1 - alpha) * img[m] + alpha * np.asarray(color, np.float64)
    return np.clip(img, 0, 255).astype(np.uint8)


def error_map(flow, flow_gt, valid=None, max_err: float = 10.0) -> np.ndarray:
    """RGB uint8 end-point-error map (dark blue = 0 px, through yellow, to red = ``max_err``
    px or more); pixels outside ``valid`` are grey."""
    e = np.hypot(*(np.asarray(flow, np.float64) - np.asarray(flow_gt, np.float64)))
    t = np.clip(np.nan_to_num(e, nan=max_err) / max_err, 0, 1)
    rise, fall = np.clip(2 * t, 0, 1), np.clip(2 - 2 * t, 0, 1)
    out = np.stack([255 * rise, 255 * fall * rise, 96 * (1 - t)], -1)
    if valid is not None:
        out[~np.asarray(valid, bool)] = 96
    return out.astype(np.uint8)


def outlier_overlay(rgb, outlier_mask, valid=None) -> np.ndarray:
    """``rgb`` in grey with outlier pixels in red (and, if ``valid`` is given, unlabelled
    pixels darker)."""
    img = np.asarray(rgb, np.float64)
    grey = img.mean(-1, keepdims=True).repeat(3, -1) * 0.8 + 40
    if valid is not None:
        grey[~np.asarray(valid, bool)] *= 0.5
    grey[np.asarray(outlier_mask, bool)] = (230, 30, 30)
    return np.clip(grey, 0, 255).astype(np.uint8)


def tile(images, cols: int = 2, pad: int = 4) -> np.ndarray:
    """RGB images of one size in a grid, white padding."""
    ims = [np.asarray(i, np.uint8) for i in images]
    h, w = ims[0].shape[:2]
    rows = -(-len(ims) // cols)
    canvas = np.full((rows * h + (rows + 1) * pad, cols * w + (cols + 1) * pad, 3), 255,
                     np.uint8)
    for k, im in enumerate(ims):
        r, c = divmod(k, cols)
        y, x = pad + r * (h + pad), pad + c * (w + pad)
        canvas[y:y + h, x:x + w] = im
    return canvas
