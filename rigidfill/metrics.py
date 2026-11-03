"""KITTI flow metrics: end-point error and the Fl outlier rate."""
from __future__ import annotations

from typing import Optional

import numpy as np


def endpoint_error(flow, gt) -> np.ndarray:
    """``(H, W)`` float32 end-point error of ``flow`` against ``gt`` (both ``(2, H, W)``)."""
    d = np.asarray(flow, np.float32) - np.asarray(gt, np.float32)
    return np.sqrt(d[0] * d[0] + d[1] * d[1])


def outliers(flow, gt, valid: Optional[np.ndarray] = None) -> np.ndarray:
    """KITTI 2015 outliers: end-point error above 3 px and above 5% of the true magnitude.

    Returns bool ``(H, W)``, False outside ``valid``.
    """
    epe = endpoint_error(flow, gt)
    g = np.asarray(gt, np.float32)
    mag = np.sqrt(g[0] * g[0] + g[1] * g[1])
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (epe > np.float32(3.0)) & ((epe / mag) > np.float32(0.05))
    return out if valid is None else out & np.asarray(valid, bool)


def fl(flow, gt, valid) -> float:
    """Fl (%): the share of valid pixels that are outliers. Pool counts over images (sum of
    :func:`outliers` over sum of valid pixels) for a dataset-level Fl."""
    v = np.asarray(valid, bool)
    return 100.0 * float(outliers(flow, gt, v).sum()) / max(int(v.sum()), 1)
