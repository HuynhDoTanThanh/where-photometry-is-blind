"""Camera motion, depth and rigid flow from a network's own flows.

These are the geometric steps of RigidFill, all computed on the network's input grid:

1. :func:`correspondences` builds, for one flow direction, the pixels whose correspondence is
   visible and stays inside the frame, and the subset that is reliable enough for pose
   estimation (uncertainty ``alpha <= ln 2``);
2. :func:`estimate_pose` fits an essential matrix with OpenCV's five-point RANSAC and decomposes
   it with ``cv2.recoverPose`` into a rotation and a unit translation;
3. :func:`triangulate` gives the depth of every usable correspondence (linear triangulation);
4. :func:`fit_geometry` runs steps 1-3 for the backward pair (t -> t-1) and the forward pair
   (t -> t+1) and aligns their units with the median depth ratio ``lambda``;
5. :func:`rigid_flow` reprojects every pixel with a backward depth into camera t+1.

Conventions: pixel ``(x, y)`` is column ``x`` and row ``y``, and integer coordinates are pixel
centres. Flows are arrays of shape ``(2, H, W)`` holding ``(u, v)``. Depth is the z coordinate
in camera t. A pose ``(R, t)`` maps a point of camera t to the other camera:
``X_other = R @ X_t + t``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

#: Pose cutoff on the uncertainty: only pixels with ``alpha <= ln 2`` (sigma^2 <= 2) fit a pose.
POSE_ALPHA_MAX = math.log(2.0)
#: RANSAC inlier threshold of ``cv2.findEssentialMat``, in grid pixels.
RANSAC_THRESHOLD_PX = 1.0
#: RANSAC confidence of ``cv2.findEssentialMat``.
RANSAC_CONFIDENCE = 0.999
#: Minimum number of correspondences, of inliers in front of both cameras, and of pixels with
#: both depths (for ``lambda``).
MIN_POINTS = 8


class RigidFillError(RuntimeError):
    """No rigid flow can be computed for this image; the caller keeps the network's flow.

    Attributes:
        stage: ``"pose"`` (too few correspondences, no unique essential matrix, or too few
            inliers in front of both cameras) or ``"scale"`` (too few pixels with both depths,
            or ``lambda`` not finite and positive).
        reason: a human-readable explanation.
    """

    def __init__(self, stage: str, reason: str) -> None:
        super().__init__(f"{stage}: {reason}")
        self.stage = stage
        self.reason = reason


def scale_intrinsics(K, src_size: Sequence[int], dst_size: Sequence[int]) -> np.ndarray:
    """Intrinsics after resizing an image from ``src_size`` to ``dst_size`` (both ``(H, W)``).

    Each axis is scaled separately (the aspect ratio need not be kept), with the pixel-centre
    convention of ``cv2.resize``: ``f' = s f`` and ``c' = s (c + 0.5) - 0.5``.
    """
    K = np.asarray(K, dtype=np.float64)
    sx = dst_size[1] / src_size[1]
    sy = dst_size[0] / src_size[0]
    out = K.copy()
    out[0, 0], out[0, 1], out[1, 1] = sx * K[0, 0], sx * K[0, 1], sy * K[1, 1]
    out[0, 2] = sx * (K[0, 2] + 0.5) - 0.5
    out[1, 2] = sy * (K[1, 2] + 0.5) - 0.5
    return out


@dataclass
class Correspondences:
    """The correspondences ``p -> p + F(p)`` of one flow field on its grid.

    Attributes:
        xs, ys: pixel coordinates of every grid pixel, float64 ``(H, W)``.
        tx, ty: target coordinates ``p + F(p)``, float64 ``(H, W)``.
        usable: visible pixels whose target lies inside the frame (the set M_d of the paper);
            these are triangulated.
        pose_set: the subset of ``usable`` with ``alpha <= alpha_max`` (the set P_d); these fit
            the pose. Equal to ``usable`` when no uncertainty is given.
    """

    xs: np.ndarray
    ys: np.ndarray
    tx: np.ndarray
    ty: np.ndarray
    usable: np.ndarray
    pose_set: np.ndarray

    def points(self, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """``(pts_t, pts_target)``, each float64 ``(N, 2)``, in row-major pixel order."""
        return (np.stack([self.xs[mask], self.ys[mask]], 1),
                np.stack([self.tx[mask], self.ty[mask]], 1))


def correspondences(flow, visible=None, alpha=None,
                    alpha_max: float = POSE_ALPHA_MAX) -> Correspondences:
    """Usable and pose-quality correspondences of one flow field.

    Args:
        flow: ``(2, H, W)`` flow from frame t to the other frame, in grid pixels.
        visible: ``(H, W)`` bool, the network's forward-backward visibility; ``None`` treats every
            pixel as visible.
        alpha: ``(H, W)`` uncertainty of ``flow`` (log-variance, or :func:`rigidfill.fb_score`);
            ``None`` uses every usable pixel for the pose.
        alpha_max: pose cutoff; pixels with ``alpha <= alpha_max`` fit the pose.
    """
    flow = np.asarray(flow, dtype=np.float64)
    h, w = flow.shape[1:]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    tx, ty = xs + flow[0], ys + flow[1]
    vis = np.ones((h, w), bool) if visible is None else np.asarray(visible, dtype=bool)
    inside = (tx >= 0) & (tx <= w - 1) & (ty >= 0) & (ty <= h - 1) & np.isfinite(tx)
    usable = vis & inside
    if alpha is None:
        pose_set = usable
    else:
        pose_set = usable & (np.asarray(alpha, dtype=np.float64) <= alpha_max)
    return Correspondences(xs, ys, tx, ty, usable, pose_set)


def estimate_pose(pts1, pts2, K, threshold: float = RANSAC_THRESHOLD_PX,
                  confidence: float = RANSAC_CONFIDENCE,
                  seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, int]:
    """Relative camera motion from correspondences ``pts1 -> pts2`` (five-point RANSAC).

    ``cv2.findEssentialMat`` (RANSAC, ``threshold`` px, ``confidence``) estimates the essential
    matrix and ``cv2.recoverPose`` (its default depth limit of 50 baselines) picks the
    decomposition that puts the most inliers in front of both cameras.

    Args:
        pts1, pts2: float64 ``(N, 2)`` pixel coordinates. RANSAC depends on their order.
        K: ``(3, 3)`` intrinsics of the grid the points live on.
        threshold: RANSAC inlier threshold in pixels.
        confidence: RANSAC confidence.
        seed: if given, ``cv2.setRNGSeed(seed)`` is called first. The paper's implementation set
            it to the image index; OpenCV's RANSAC does not use it in the versions tested, so the
            pose is a deterministic function of the ordered points.

    Returns:
        ``(R, t, n)`` with ``X_2 = R X_1 + t``, ``|t| = 1`` and ``n`` the inliers in front.

    Raises:
        RigidFillError: fewer than 8 points, no unique essential matrix, or fewer than 8
            inliers in front of both cameras (for example a camera that barely translates).
    """
    import cv2

    pts1 = np.ascontiguousarray(pts1, dtype=np.float64)
    pts2 = np.ascontiguousarray(pts2, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    if pts1.shape[0] < MIN_POINTS:
        raise RigidFillError("pose", f"{pts1.shape[0]} correspondences")
    if seed is not None:
        cv2.setRNGSeed(int(seed) % 2**31)
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=confidence,
                                   threshold=threshold)
    if E is None or E.shape != (3, 3):
        raise RigidFillError("pose", "findEssentialMat gave no single essential matrix")
    n, R, t, _ = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
    if n < MIN_POINTS:
        raise RigidFillError("pose", f"{n} inliers in front of both cameras")
    return R, t.reshape(3), int(n)


def triangulate(pts1, pts2, K, R, t) -> np.ndarray:
    """Linear (DLT) triangulation of ``pts1 -> pts2`` under ``K[I|0]`` and ``K[R|t]``.

    Each point is solved by a singular value decomposition of its 4 x 4 system.

    Returns:
        float64 ``(N,)`` depth (z) in camera 1; NaN where the solution is not finite or lies
        behind either camera.
    """
    pts1 = np.asarray(pts1, dtype=np.float64)
    pts2 = np.asarray(pts2, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    if pts1.shape[0] == 0:
        return np.zeros(0)
    # errstate: some BLAS builds (numpy 2.0 with Accelerate) raise spurious matmul warnings
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = K @ np.hstack([R, t.reshape(3, 1)])
        u1, v1, u2, v2 = pts1[:, 0], pts1[:, 1], pts2[:, 0], pts2[:, 1]
        A = np.stack([u1[:, None] * P1[2] - P1[0], v1[:, None] * P1[2] - P1[1],
                      u2[:, None] * P2[2] - P2[0], v2[:, None] * P2[2] - P2[1]], 1)
        _, _, Vt = np.linalg.svd(A)
        Xh = Vt[:, -1, :]
        X = Xh[:, :3] / Xh[:, 3:4]
        z1 = X[:, 2]
        z2 = (X @ R.T + t)[:, 2]
        ok = np.isfinite(z1) & (z1 > 0) & (z2 > 0)
    return np.where(ok, z1, np.nan)


def triangulate_dense(corr: Correspondences, K, R, t) -> np.ndarray:
    """Depth map ``(H, W)`` of every usable correspondence of ``corr`` (NaN elsewhere)."""
    D = np.full(corr.xs.shape, np.nan)
    if corr.usable.any():
        p1, p2 = corr.points(corr.usable)
        D[corr.usable] = triangulate(p1, p2, K, R, t)
    return D


@dataclass
class RigidGeometry:
    """Camera motion and depth of frame t, estimated from the network's flows.

    Attributes:
        R_fw, t_fw: forward motion t -> t+1, ``X_{t+1} = R_fw X_t + t_fw``; ``t_fw`` is in the
            units of ``depth_bw`` (``lambda`` times the unit translation).
        R_bw, t_bw: backward motion t -> t-1 with ``|t_bw| = 1``.
        depth_bw: ``(H, W)`` depth of frame t triangulated from the backward flow, in units of
            ``|t_bw|``; NaN where undefined.
        scale: ``lambda``, the median of ``D_b / D_f`` over pixels with both depths.
        n_inliers_bw, n_inliers_fw: RANSAC inliers in front of both cameras.
        n_both: pixels with both depths.
    """

    R_fw: np.ndarray
    t_fw: np.ndarray
    R_bw: np.ndarray
    t_bw: np.ndarray
    depth_bw: np.ndarray
    scale: float
    n_inliers_bw: int
    n_inliers_fw: int
    n_both: int


def fit_geometry(flow_fw, flow_bw, K_grid, *, vis_fw=None, vis_bw=None, alpha_fw=None,
                 alpha_bw=None, alpha_max: float = POSE_ALPHA_MAX,
                 ransac_threshold: float = RANSAC_THRESHOLD_PX,
                 ransac_confidence: float = RANSAC_CONFIDENCE,
                 seed: Optional[int] = None) -> RigidGeometry:
    """Both relative poses, the backward depth and the scale ``lambda`` of one frame triple.

    Args:
        flow_fw: ``(2, H, W)`` forward flow t -> t+1 on the grid.
        flow_bw: ``(2, H, W)`` backward flow t -> t-1 on the same grid.
        K_grid: ``(3, 3)`` intrinsics of the grid (see :func:`scale_intrinsics`).
        vis_fw, vis_bw: visibility of each flow (``None``: all visible).
        alpha_fw, alpha_bw: uncertainty of each flow, used for the pose sets (``None``: every
            usable pixel fits the pose).
        alpha_max: pose cutoff (``ln 2``).
        ransac_threshold, ransac_confidence: settings of ``cv2.findEssentialMat``.
        seed: optional OpenCV seed; the forward fit uses ``seed + 1``.

    Raises:
        RigidFillError: when a pose or the scale cannot be estimated.
    """
    cb = correspondences(flow_bw, vis_bw, alpha_bw, alpha_max)
    Rb, tb, nb = estimate_pose(*cb.points(cb.pose_set), K_grid, ransac_threshold,
                               ransac_confidence, seed)
    Db = triangulate_dense(cb, K_grid, Rb, tb)
    cf = correspondences(flow_fw, vis_fw, alpha_fw, alpha_max)
    Rf, tf, nf = estimate_pose(*cf.points(cf.pose_set), K_grid, ransac_threshold,
                               ransac_confidence, None if seed is None else seed + 1)
    Df = triangulate_dense(cf, K_grid, Rf, tf)
    with np.errstate(invalid="ignore"):
        both = np.isfinite(Db) & np.isfinite(Df) & (Df > 0)
    if both.sum() < MIN_POINTS:
        raise RigidFillError("scale", f"{int(both.sum())} pixels triangulated both ways")
    s = float(np.median(Db[both] / Df[both]))
    if not np.isfinite(s) or s <= 0:
        raise RigidFillError("scale", f"scale {s}")
    return RigidGeometry(R_fw=np.asarray(Rf, np.float64), t_fw=s * tf,
                         R_bw=np.asarray(Rb, np.float64), t_bw=tb, depth_bw=Db, scale=s,
                         n_inliers_bw=nb, n_inliers_fw=nf, n_both=int(both.sum()))


def rigid_flow(depth, K, R, t) -> np.ndarray:
    """Flow of a rigid scene: ``pi(K (R D(p) K^-1 p~ + t)) - p`` for every pixel with a depth.

    Args:
        depth: ``(H, W)`` z-depth of frame 1; NaN or non-positive means undefined.
        K: ``(3, 3)`` intrinsics of this grid (one K for both cameras).
        R, t: motion with ``X_2 = R X_1 + t``, ``t`` in the units of ``depth``.

    Returns:
        float64 ``(2, H, W)``; NaN where the depth is undefined or the point lands on the
        plane z = 0 of camera 2. No other test is made in camera 2: a projection may leave the
        image (the case RigidFill targets), and a point behind camera 2 also yields a value.
    """
    d = np.asarray(depth, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    h, w = d.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        rays = np.linalg.inv(K) @ np.stack([xs, ys, np.ones_like(xs)], 0).reshape(3, -1)
        X = rays.reshape(3, h, w) * d[None]
        X2 = (np.asarray(R, np.float64) @ X.reshape(3, -1)
              + np.asarray(t, np.float64).reshape(3, 1)).reshape(3, h, w)
        P = K @ X2.reshape(3, -1)
        z = P[2]
        xq = (P[0] / z).reshape(h, w)
        yq = (P[1] / z).reshape(h, w)
        zc = X2.reshape(3, -1)[2].reshape(h, w)
        valid = np.isfinite(d) & (d > 0) & np.isfinite(zc) & (zc != 0)
    return np.stack([np.where(valid, xq - xs, np.nan), np.where(valid, yq - ys, np.nan)], 0)


def round_depth_half(depth) -> np.ndarray:
    """Depth after storing it as half-precision inverse depth, as the paper's implementation did.

    Depths whose inverse under- or overflows float16 become NaN.
    """
    D = np.asarray(depth, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(D) & (D > 0)
    inv = np.where(valid, 1.0 / np.where(valid, D, 1.0), 0.0).astype(np.float16)
    inv = inv.astype(np.float64)
    ok = valid & np.isfinite(inv) & (inv > 0)
    with np.errstate(divide="ignore"):
        return np.where(ok, 1.0 / np.where(ok, inv, 1.0), np.nan)
