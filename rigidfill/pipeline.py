"""RigidFill end to end: :func:`correct_flow`."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np

from . import geometry as G
from .gate import GateParams, apply_gate, resolve_gate, uncertainty_gate
from .resize import upsample_flow, upsample_score


@dataclass
class RigidFillInfo:
    """What :func:`correct_flow` did.

    Attributes:
        ok: a rigid flow was computed (``False``: the output is the network's flow everywhere).
        failure: why not, e.g. ``"pose: 6 inliers in front of both cameras"``; ``None`` if ok.
        gate: bool ``(H, W)`` native pixels where the rigid flow replaced the network's flow.
        rigid_flow: float32 ``(2, H, W)`` rigid flow T at native resolution (NaN where
            undefined), or ``None``.
        rigid_flow_grid: float64 ``(2, h, w)`` rigid flow on the network grid, or ``None``.
        base_flow: float32 ``(2, H, W)`` the network's flow S at native resolution.
        alpha: float32 ``(H, W)`` the uncertainty at native resolution.
        geometry: poses, backward depth and scale (:class:`rigidfill.RigidGeometry`), or ``None``.
        K_grid: the intrinsics rescaled to the grid.
        params: the gate thresholds used.
    """

    ok: bool
    failure: Optional[str]
    gate: np.ndarray
    rigid_flow: Optional[np.ndarray]
    rigid_flow_grid: Optional[np.ndarray]
    base_flow: np.ndarray
    alpha: np.ndarray
    geometry: Optional[G.RigidGeometry]
    K_grid: np.ndarray
    params: GateParams

    @property
    def n_replaced(self) -> int:
        """Number of native pixels replaced by the rigid flow."""
        return int(self.gate.sum())

    @property
    def replaced_fraction(self) -> float:
        """Share of the native pixels replaced by the rigid flow."""
        return float(self.gate.mean())


def rigid_flow_from_flows(flow_fw, flow_bw, K_grid, *, vis_fw=None, vis_bw=None, alpha_fw=None,
                          alpha_bw=None, half_precision_depth: bool = True,
                          alpha_max: float = G.POSE_ALPHA_MAX,
                          ransac_threshold: float = G.RANSAC_THRESHOLD_PX,
                          ransac_confidence: float = G.RANSAC_CONFIDENCE,
                          seed: Optional[int] = None, threads: int = 1):
    """The rigid flow T on the grid and the geometry behind it (steps 1-3 of RigidFill).

    Returns ``(T_grid, geometry)``; raises :class:`rigidfill.RigidFillError` on failure. See
    :func:`correct_flow` for the arguments.
    """
    geo = G.fit_geometry(flow_fw, flow_bw, K_grid, vis_fw=vis_fw, vis_bw=vis_bw,
                         alpha_fw=alpha_fw, alpha_bw=alpha_bw, alpha_max=alpha_max,
                         ransac_threshold=ransac_threshold, ransac_confidence=ransac_confidence,
                         seed=seed, threads=threads)
    depth = G.round_depth_half(geo.depth_bw) if half_precision_depth else geo.depth_bw
    return G.rigid_flow(depth, K_grid, geo.R_fw, geo.t_fw), geo


def correct_flow(flow_fw, flow_bw, alpha_fw, K, image_size: Optional[Sequence[int]] = None, *,
                 vis_fw=None, vis_bw=None, alpha_bw=None, base_flow=None,
                 gate: Union[str, GateParams] = "u2flow", tau: Optional[float] = None,
                 d_max: Optional[float] = None, d_min: Optional[float] = None,
                 half_precision_depth: bool = True, alpha_max: float = G.POSE_ALPHA_MAX,
                 ransac_threshold: float = G.RANSAC_THRESHOLD_PX,
                 ransac_confidence: float = G.RANSAC_CONFIDENCE,
                 seed: Optional[int] = None, threads: int = 1):
    """Correct a frozen network's forward flow with the rigid flow of the scene (RigidFill).

    All flow inputs live on the network's input grid ``(h, w)`` (for U2Flow on KITTI,
    256 x 832); the output is at the native image size ``image_size = (H, W)``.

    Steps: (1) camera motion t -> t-1 and t -> t+1 by five-point RANSAC on visible, in-frame
    pixels with ``alpha <= ln 2``; (2) depth of frame t by triangulating every visible, in-frame
    backward correspondence, and the scale ``lambda`` = median ratio of backward to forward
    depth; (3) rigid flow ``T = pi(K [R_f D_b K^-1 p~ + lambda t_f]) - p`` wherever the backward
    depth exists; (4) T, alpha and S brought to native resolution with U2Flow's bilinear
    upsampling; (5) ``O = T`` where ``alpha > tau`` and ``d_min < ||T - S|| <= d_max``, else S.
    If no pose or scale can be estimated, the output is S and ``info.ok`` is ``False``.

    Args:
        flow_fw: ``(2, h, w)`` forward flow t -> t+1 (the flow to correct).
        flow_bw: ``(2, h, w)`` backward flow t -> t-1, from a second pass on frames (t-1, t).
        alpha_fw: ``(h, w)`` uncertainty of ``flow_fw``: U2Flow's log-variance, or
            :func:`rigidfill.fb_score` for networks without an uncertainty head.
        K: ``(3, 3)`` intrinsics of the native image (rescaled to the grid internally).
        image_size: native ``(H, W)``; default: the grid size (flows already at native size).
        vis_fw, vis_bw: ``(h, w)`` bool visibility of each flow by the forward-backward test
            (:func:`rigidfill.fb_consistency`); ``None`` treats every pixel as visible.
        alpha_bw: ``(h, w)`` uncertainty of ``flow_bw`` (filters the backward pose set; the
            paper uses it for U2Flow).
        base_flow: ``(2, H, W)`` the network's flow at native resolution, if you already have
            it; default ``upsample_flow(flow_fw, image_size)``.
        gate: ``"u2flow"`` (tau 2.75, d_max 100 px), ``"arflow"``, ``"sunraft"`` or
            :class:`rigidfill.GateParams`.
        tau, d_max, d_min: override the preset's thresholds (``d_max``, ``d_min`` in native px).
        half_precision_depth: store the backward depth as float16 inverse depth before
            rendering, as the paper's implementation did (set ``False`` for full precision).
        alpha_max: pose cutoff (default ``ln 2``).
        ransac_threshold, ransac_confidence: ``cv2.findEssentialMat`` settings (1 px, 0.999).
        seed: optional OpenCV RNG seed (see :func:`rigidfill.estimate_pose`).
        threads: CPU threads for the geometry (see :func:`rigidfill.fit_geometry`); the output
            is the same as with the default of one thread. The paper's runtime (212.7 ms per
            KITTI image with all network passes) used 24.

    Returns:
        ``(flow_out, info)``: float32 ``(2, H, W)`` corrected flow and a
        :class:`RigidFillInfo`.
    """
    flow_fw = np.asarray(flow_fw)
    if flow_fw.ndim != 3 or flow_fw.shape[0] != 2:
        raise ValueError(f"flow_fw must be (2, h, w), got {flow_fw.shape}")
    grid = tuple(int(x) for x in flow_fw.shape[1:])
    if np.asarray(flow_bw).shape != flow_fw.shape:
        raise ValueError(f"flow_bw {np.asarray(flow_bw).shape} and flow_fw {flow_fw.shape} differ")
    if np.asarray(alpha_fw).shape != grid:
        raise ValueError(f"alpha_fw must be {grid}, got {np.asarray(alpha_fw).shape}")
    native = grid if image_size is None else (int(image_size[0]), int(image_size[1]))
    params = resolve_gate(gate, tau=tau, d_max=d_max, d_min=d_min)
    K_grid = G.scale_intrinsics(K, native, grid)
    S = upsample_flow(flow_fw, native) if base_flow is None else np.asarray(base_flow,
                                                                            np.float32)
    if S.shape != (2, *native):
        raise ValueError(f"base_flow must be (2, {native[0]}, {native[1]}), got {S.shape}")
    alpha = upsample_score(alpha_fw, native)
    try:
        T_grid, geo = rigid_flow_from_flows(
            flow_fw, flow_bw, K_grid, vis_fw=vis_fw, vis_bw=vis_bw, alpha_fw=alpha_fw,
            alpha_bw=alpha_bw, half_precision_depth=half_precision_depth, alpha_max=alpha_max,
            ransac_threshold=ransac_threshold, ransac_confidence=ransac_confidence, seed=seed,
            threads=threads)
    except G.RigidFillError as e:
        info = RigidFillInfo(ok=False, failure=str(e), gate=np.zeros(native, bool),
                             rigid_flow=None, rigid_flow_grid=None, base_flow=S, alpha=alpha,
                             geometry=None, K_grid=K_grid, params=params)
        return S.copy(), info
    T = upsample_flow(T_grid, native)
    g = uncertainty_gate(alpha, T, S, params)
    out = apply_gate(g, T, S)
    info = RigidFillInfo(ok=True, failure=None, gate=g, rigid_flow=T, rigid_flow_grid=T_grid,
                         base_flow=S, alpha=alpha, geometry=geo, K_grid=K_grid, params=params)
    return out, info
