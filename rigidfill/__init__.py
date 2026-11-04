"""RigidFill: inference-time geometric correction of optical flow in driving scenes.

A frozen flow network learns nothing about pixels that leave the next frame. RigidFill estimates
the camera motion from the network's own reliable correspondences, triangulates the backward
correspondences of frame t to get its depth, reprojects every pixel into camera t+1, and
replaces the network's flow by this rigid flow where the network is uncertain.

Quick start::

    import rigidfill
    out, info = rigidfill.correct_flow(flow_fw, flow_bw, alpha_fw, K, image_size=(375, 1242),
                                       vis_fw=vis_fw, vis_bw=vis_bw, alpha_bw=alpha_bw)

See README.md for the inputs each network provides.
"""
from .consistency import fb_consistency, fb_score, warp_zeros
from .gate import PRESETS, GateParams, apply_gate, resolve_gate, uncertainty_gate
from .geometry import (
    POSE_ALPHA_MAX,
    RANSAC_CONFIDENCE,
    RANSAC_THRESHOLD_PX,
    Correspondences,
    RigidFillError,
    RigidGeometry,
    correspondences,
    estimate_pose,
    fit_geometry,
    rigid_flow,
    scale_intrinsics,
    triangulate,
)
from .io import (
    read_image,
    read_kitti_flow,
    read_kitti_intrinsics,
    write_image,
    write_kitti_flow,
)
from .metrics import endpoint_error, fl, outliers
from .pipeline import RigidFillInfo, correct_flow, rigid_flow_from_flows
from .resize import upsample_bilinear, upsample_flow, upsample_score
from .viz import flow_to_color

__version__ = "1.1.0"

__all__ = [
    "POSE_ALPHA_MAX", "PRESETS", "RANSAC_CONFIDENCE", "RANSAC_THRESHOLD_PX", "Correspondences",
    "GateParams", "RigidFillError", "RigidFillInfo", "RigidGeometry", "apply_gate", "correct_flow",
    "correspondences", "endpoint_error", "estimate_pose", "fb_consistency", "fb_score",
    "fit_geometry", "fl", "flow_to_color", "outliers", "read_image", "read_kitti_flow",
    "read_kitti_intrinsics", "resolve_gate", "rigid_flow", "rigid_flow_from_flows",
    "scale_intrinsics", "triangulate", "uncertainty_gate", "upsample_bilinear", "upsample_flow",
    "upsample_score", "warp_zeros", "write_image", "write_kitti_flow",
]
