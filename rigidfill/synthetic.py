"""A synthetic driving scene with exact flows, for the demo and the tests.

A camera drives forward down a corridor (a ground plane, two side walls and a far wall). The
true forward and backward flows of frame t follow from the scene's depth and the camera
motion. A simulated network gets the in-frame flow right but under-estimates the flow of the
pixels that leave frame t+1, and reports a high uncertainty there, which is the failure
RigidFill corrects.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .geometry import rigid_flow, scale_intrinsics


def kitti_like_K(size: Sequence[int] = (375, 1242)) -> np.ndarray:
    """Intrinsics similar to KITTI's left colour camera, rescaled to ``size = (H, W)``."""
    K = np.array([[721.5377, 0.0, 609.5593], [0.0, 721.5377, 172.854], [0.0, 0.0, 1.0]])
    return scale_intrinsics(K, (375, 1242), size)


def yaw(deg: float) -> np.ndarray:
    """Rotation about the camera's vertical (y) axis."""
    a = np.radians(deg)
    return np.array([[np.cos(a), 0.0, np.sin(a)], [0.0, 1.0, 0.0], [-np.sin(a), 0.0, np.cos(a)]])


def corridor_depth(K, size: Sequence[int], half_width: float = 8.0, camera_height: float = 1.6,
                   far: float = 60.0) -> np.ndarray:
    """z-depth of a corridor seen from its centre line: ground plane, side walls, far wall."""
    h, w = int(size[0]), int(size[1])
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        d = np.linalg.inv(np.asarray(K, np.float64)) @ np.stack(
            [xs, ys, np.ones_like(xs)]).reshape(3, -1)
    dx, dy = d[0].reshape(h, w), d[1].reshape(h, w)
    z = np.full((h, w), far)
    with np.errstate(divide="ignore", invalid="ignore"):
        for cand in (np.where(dy > 0, camera_height / dy, np.inf),
                     np.where(dx > 0, half_width / dx, np.inf),
                     np.where(dx < 0, -half_width / dx, np.inf)):
            z = np.minimum(z, cand)
    return z


def motion(R_c2w: np.ndarray, c: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """``(R, t)`` mapping camera-t points to a camera at pose ``(R_c2w, c)`` in camera t's frame."""
    R = np.asarray(R_c2w, np.float64).T
    return R, -R @ np.asarray(c, np.float64)


@dataclass
class Scene:
    """Everything RigidFill needs for one synthetic frame triple, plus the truth.

    Flows are on the network grid; ``flow_fw_true_native`` is the true forward flow at native
    resolution and ``oof_native`` marks the native pixels whose true target leaves frame t+1.
    """

    K: np.ndarray
    image_size: Tuple[int, int]
    grid: Tuple[int, int]
    flow_fw: np.ndarray
    flow_bw: np.ndarray
    alpha_fw: np.ndarray
    alpha_bw: np.ndarray
    vis_fw: np.ndarray
    vis_bw: np.ndarray
    flow_fw_true: np.ndarray
    oof: np.ndarray
    flow_fw_true_native: np.ndarray
    oof_native: np.ndarray
    true_scale: float


def _flow_and_flags(depth, K, R, t):
    f = rigid_flow(depth, K, R, t)
    h, w = depth.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    tx, ty = xs + f[0], ys + f[1]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        X = (np.linalg.inv(K) @ np.stack([xs, ys, np.ones_like(xs)]).reshape(3, -1)) \
            * depth.ravel()
        z2 = (R @ X + t.reshape(3, 1))[2].reshape(h, w)
    inside = (tx >= 0) & (tx <= w - 1) & (ty >= 0) & (ty <= h - 1) & (z2 > 0)
    return f, inside


def make_scene(image_size: Sequence[int] = (375, 1242), grid: Optional[Sequence[int]] = (256, 832),
               forward: float = 1.0, backward: float = 0.9, yaw_deg: float = 1.0,
               oof_factor: float = 0.6, noise_px: float = 0.0, seed: int = 0) -> Scene:
    """A frame triple of the corridor scene.

    Args:
        image_size: native ``(H, W)``.
        grid: the network's grid (``None``: the native size).
        forward, backward: distance (m) driven from t to t+1 and from t-1 to t.
        yaw_deg: yaw of each step.
        oof_factor: the simulated network predicts ``oof_factor`` times the true flow on pixels
            that leave frame t+1 (1.0 = perfect).
        noise_px: Gaussian noise (grid px) added to both simulated flows.
        seed: seed of the noise.
    """
    native = (int(image_size[0]), int(image_size[1]))
    grid = native if grid is None else (int(grid[0]), int(grid[1]))
    K = kitti_like_K(native)
    Rf, tf = motion(yaw(yaw_deg), [0.05, 0.0, forward])
    Rb, tb = motion(yaw(-yaw_deg), [-0.04, 0.0, -backward])

    Kg = scale_intrinsics(K, native, grid)
    Dg = corridor_depth(Kg, grid)
    true_fw, in_fw = _flow_and_flags(Dg, Kg, Rf, tf)
    true_bw, in_bw = _flow_and_flags(Dg, Kg, Rb, tb)
    rng = np.random.default_rng(seed)
    flow_fw = np.where(in_fw[None], true_fw, oof_factor * true_fw)
    flow_bw = true_bw.copy()
    if noise_px > 0:
        flow_fw = flow_fw + rng.normal(0.0, noise_px, flow_fw.shape)
        flow_bw = flow_bw + rng.normal(0.0, noise_px, flow_bw.shape)
    alpha_fw = np.where(in_fw, -2.0, 5.0)
    alpha_bw = np.where(in_bw, -2.0, 5.0)

    Dn = corridor_depth(K, native)
    true_n, in_n = _flow_and_flags(Dn, K, Rf, tf)
    return Scene(K=K, image_size=native, grid=grid, flow_fw=flow_fw.astype(np.float32),
                 flow_bw=flow_bw.astype(np.float32), alpha_fw=alpha_fw.astype(np.float32),
                 alpha_bw=alpha_bw.astype(np.float32), vis_fw=in_fw, vis_bw=in_bw,
                 flow_fw_true=true_fw, oof=~in_fw, flow_fw_true_native=true_n,
                 oof_native=~in_n,
                 true_scale=float(np.linalg.norm(tf) / np.linalg.norm(tb)))
