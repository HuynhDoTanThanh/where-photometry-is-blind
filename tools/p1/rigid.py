"""Rigid flow from depth, relative pose and intrinsics: F_rig, OOF, O_T, valid_T, b_T and c_T,
the one implementation the P1 renderer, refine_pose, the gate and (later) the training loader
share (plan W2-14 names it core/teacher/rigid.py; core/ belongs to the codebase skill, so it
lives here and the loader imports it, todo_for_integrator).

How to use (library only; numpy, float64 throughout):
    K1 = zoom_K(K, (375, 1242), (256, 832))                 # proposal eq. 3
    M = relative_pose(T1_c2w, T2_c2w)                       # X_2 = M X_1 (4x4)
    r = rigid_flow(depth1, K1, M)                           # r.flow (2,h,w), r.xq, r.yq, r.z
    out = render_pair(inv1, inv2, K, M, grid_hw)            # both directions + every flag
    up, ok = resize_bilinear(inv_depth, (256, 832), valid)  # half-pixel centres, strict
    K = fit_intrinsics(local_points)                        # pi3: fx, fy, cx, cy from X/Z, Y/Z

Conventions (fixed; tests/test_p1_rigid.py checks each against an analytic case).
- Pixels: x = column, y = row; integer coordinates are pixel centres, so a target q is in frame
  iff 0 <= x_q <= W-1 and 0 <= y_q <= H-1, exactly tools/eval/regions.py's IN [V regions.py
  in_frame]: R2 (occluded, in frame) and R3 (out of frame) use the same boundary as OOF here.
- Depth is z-depth (camera z), never distance along the ray; caches store inverse depth, and 0 or
  a non-finite value means invalid. Poses are OpenCV camera-to-world 4x4 (+x right, +y down, +z
  forward), MapAnything's and pi3's convention [sourced-V their READMEs at 3d10cf7 / 9fa3ddb,
  2026-10-03]; the relative transform from frame 1 to frame 2 is M = inv(T2) @ T1.
- F_rig(p) = pi(K (R D1(p) K^-1 [p;1] + t)) - p with ONE K for both projections [P proposal
  final.architecture.equations[0]: '[x', y', z'] = K(R X_t(p) + t), X_t(p) = D_t(p) K^-1 [p;1]'].
  A window has one camera, so a per-frame K (MapAnything predicts one per view, pi3's is fitted
  per frame) would turn K jitter between frames 10 and 11 into spurious flow that grows toward
  the border [M tests/test_p1_rigid.py: an identity pose with K11 = 1.01 K10 gave 6.48 px at
  native resolution]; the caller picks the window's K (p1lib/geomio.py window_K: the source
  frame's teacher K, or the calibration) and records it. Scale: a common factor on depth and
  translation cancels, so depth and pose must come from one forward pass or from PnP on that
  depth [P same equation].
- Resizing (proposal eq. 2-3): x_T = (x + 0.5) W_T / w - 0.5 and c' = s (c + 0.5) - 0.5, the
  half-pixel convention of cv2.resize, which U2Flow's Zoom uses [V core/utils/transforms/
  sep_transforms.py:56]. resize_bilinear implements it in numpy (edge clamped) so the Mac and the
  VM agree without cv2's version (Mac 5.0.0, VM 4.14.0 [V env/lock_G0.txt:42]).
- Strict validity: an interpolated depth is valid only when all four neighbours are valid, so no
  depth is invented across a hole or an invalid border [E: a mixed value is a surface that is
  not there].
- OOF(p) = q outside the frame, or z' <= 0 [P eq. 'OOF và che theo teacher'], with a 1e-6 px
  slack (TOL) for float64 round-off [M: an identity pose maps x = 0 to -3e-14]; KITTI GT flows
  are multiples of 1/64 px, so the slack never moves a pixel across regions.py's boundary [E].
- Target-depth test with eps_z [P same equation]: O_T(p) = z' > (1 + eps) s D2(q) (p is hidden
  behind the surface frame 2 sees); front(p) = z' < (1 - eps) s D2(q) (p would hide that surface,
  which frame 2 contradicts: valid_T = 0). D2(q) is the strict bilinear sample of inverse depth,
  inverted; where it is unavailable (q out of frame, an invalid neighbour) both are False and
  zunknown(p) is True.
- b_T(p) = b0 + ||F(p; (1+rho) D) - F(p; D)||_1 + lambda_c e_cyc(p) [P eq. 'Thang teacher'];
  e_cyc(p) = ||F12(p) + F21(p + F12(p))||_1 where F21 can be sampled, else 0 [E: the proposal
  names e_cyc without defining it; L1 so that b_T keeps one norm]; valid_T = 0 when b_T > b_max.
- c_T(p) = clip((rank_W(C(p)) - 0.2) / 0.6, 0, 1) [P eq. 'Confidence teacher']; rank_W is the
  window-wide rank every geometry cache stores as uint8 (conf_rank; rank_u8 here,
  teacher_cache.rank_uint8 in teacher-factory's caches).

Hand-set constants. eps_z 0.05, b0 0.5 px, rho 0.05, lambda_c 0.5, b_max 8 px and the c_T ramp
(0.2, 0.6) are the proposal's [P], not checked against the literature: the SOTA check belongs to
the method skill before GATE-1.json or prereg-v2 freezes any of them (todo). Every render records
the constants it used, and the geometry caches keep only depth, confidence, pose and K, so a
change of constant is a CPU re-render, never a GPU rerun.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS_Z = 0.05          # [P] proposal equations[3]
B0 = 0.5              # [P] equations[4], px at the training grid
RHO = 0.05            # [P] equations[4]
LAMBDA_C = 0.5        # [P] equations[4]
B_MAX = 8.0           # [P] equations[4]
C_LO, C_SPAN = 0.2, 0.6  # [P] equations[5]
TOL = 1e-6  # px: frame-boundary slack for float64 round-off (identity pose maps x = 0 to -3e-14)
CONSTANTS = {"eps_z": EPS_Z, "b0": B0, "rho": RHO, "lambda_c": LAMBDA_C, "b_max": B_MAX,
             "c_lo": C_LO, "c_span": C_SPAN, "standard": "P (proposal final.architecture."
             "equations 3-5); SOTA check pending (method)"}


# ------------------------------------------------------------------------------ intrinsics
def zoom_K(K, src_hw: tuple[int, int], dst_hw: tuple[int, int]) -> np.ndarray:
    """K after resizing an image from src_hw = (H0, W0) to dst_hw = (H1, W1), aspect not kept."""
    K = np.asarray(K, dtype=np.float64)
    sx, sy = dst_hw[1] / src_hw[1], dst_hw[0] / src_hw[0]
    out = K.copy()
    out[0, 0], out[0, 1], out[1, 1] = sx * K[0, 0], sx * K[0, 1], sy * K[1, 1]
    out[0, 2] = sx * (K[0, 2] + 0.5) - 0.5
    out[1, 2] = sy * (K[1, 2] + 0.5) - 0.5
    return out


def to_grid(c, src_n: int, dst_n: int):
    """A coordinate on a src_n-pixel axis expressed on a dst_n-pixel axis (half-pixel centres)."""
    return (np.asarray(c, dtype=np.float64) + 0.5) * (dst_n / src_n) - 0.5


def fit_intrinsics(local_points, valid=None) -> np.ndarray:
    """Pinhole K whose projection best reproduces each pixel from its camera-frame point:
    x = fx X/Z + cx and y = fy Y/Z + cy, least squares over valid pixels with Z > 0 (pi3 gives
    local points, not K [sourced-V Pi3 README at 9fa3ddb, 2026-10-03])."""
    P = np.asarray(local_points, dtype=np.float64)
    h, w = P.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    ok = np.isfinite(P).all(axis=-1) & (P[..., 2] > 1e-9)
    if valid is not None:
        ok &= np.asarray(valid, dtype=bool)
    if ok.sum() < 10:
        raise ValueError("fit_intrinsics: fewer than 10 valid points")
    a = P[..., 0][ok] / P[..., 2][ok]
    b = P[..., 1][ok] / P[..., 2][ok]
    fx, cx = np.linalg.lstsq(np.stack([a, np.ones_like(a)], 1), xs[ok], rcond=None)[0]
    fy, cy = np.linalg.lstsq(np.stack([b, np.ones_like(b)], 1), ys[ok], rcond=None)[0]
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


# ------------------------------------------------------------------------------ poses
def relative_pose(T1_c2w, T2_c2w) -> np.ndarray:
    """M with X_2 = M X_1 for camera-to-world poses T1, T2."""
    return np.linalg.inv(np.asarray(T2_c2w, np.float64)) @ np.asarray(T1_c2w, np.float64)


def rt_to_M(R, t) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = np.asarray(R, np.float64), np.asarray(t, np.float64).reshape(3)
    return M


def rotation_angle_deg(R) -> float:
    """The rotation angle, as atan2(|vee(R - R^T)| / 2, (tr R - 1) / 2): arccos of the trace
    alone is ill-conditioned near 0 (float32 poses read as 0.015 deg instead of 6e-6 deg [M
    tests/test_p1_geometry.py, 2026-10-03])."""
    R = np.asarray(R, np.float64)
    c = (np.trace(R) - 1.0) / 2.0
    sv = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2.0
    return float(np.degrees(np.arctan2(np.linalg.norm(sv), c)))


def pose_deviation(Ma, Mb) -> dict:
    """How far two relative poses are apart: rotation angle of Ra^T Rb, the angle between the
    translation directions, and the ratio of translation norms (scale-free inputs give 1)."""
    Ma, Mb = np.asarray(Ma, np.float64), np.asarray(Mb, np.float64)
    ta, tb = Ma[:3, 3], Mb[:3, 3]
    na, nb = float(np.linalg.norm(ta)), float(np.linalg.norm(tb))
    ang_t = float(np.degrees(np.arccos(np.clip(ta @ tb / (na * nb), -1, 1)))) \
        if na > 0 and nb > 0 else None
    return {"rot_deg": rotation_angle_deg(Ma[:3, :3].T @ Mb[:3, :3]), "trans_dir_deg": ang_t,
            "trans_norm_ratio": (nb / na) if na > 0 else None}


# ------------------------------------------------------------------------------ resampling
def _valid_of(a, valid=None) -> np.ndarray:
    ok = np.isfinite(a)
    if valid is not None:
        ok &= np.asarray(valid, dtype=bool)
    return ok


def sample_bilinear(img, xq, yq, valid=None) -> tuple[np.ndarray, np.ndarray]:
    """Values of img (h, w) at float points; ok only inside [0, w-1] x [0, h-1] with all four
    neighbours valid (an integer coordinate uses its own pixel and the next one at weight 0)."""
    img = np.asarray(img, np.float64)
    h, w = img.shape
    ok_src = _valid_of(img, valid)
    xq, yq = np.asarray(xq, np.float64), np.asarray(yq, np.float64)
    inside = np.isfinite(xq) & np.isfinite(yq) & (xq >= -TOL) & (xq <= w - 1 + TOL) \
        & (yq >= -TOL) & (yq <= h - 1 + TOL)
    xs = np.clip(np.where(inside, xq, 0.0), 0, w - 1)
    ys = np.clip(np.where(inside, yq, 0.0), 0, h - 1)
    x0 = np.minimum(np.floor(xs).astype(np.int64), max(w - 2, 0))
    y0 = np.minimum(np.floor(ys).astype(np.int64), max(h - 2, 0))
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    wx, wy = xs - x0, ys - y0
    src = np.where(ok_src, img, 0.0)
    w00, w01 = (1 - wx) * (1 - wy), wx * (1 - wy)
    w10, w11 = (1 - wx) * wy, wx * wy
    v = src[y0, x0] * w00 + src[y0, x1] * w01 + src[y1, x0] * w10 + src[y1, x1] * w11
    # a neighbour of weight 0 may be invalid; every other one must be valid
    ok = (inside & (ok_src[y0, x0] | (w00 == 0)) & (ok_src[y0, x1] | (w01 == 0))
          & (ok_src[y1, x0] | (w10 == 0)) & (ok_src[y1, x1] | (w11 == 0)))
    return np.where(ok, v, np.nan), ok


def resize_bilinear(img, dst_hw: tuple[int, int], valid=None) -> tuple[np.ndarray, np.ndarray]:
    """Resize (h, w) -> dst_hw with half-pixel centres, the edge clamped (cv2.resize
    INTER_LINEAR's geometry), and strict validity."""
    img = np.asarray(img, np.float64)
    h, w = img.shape
    H, W = dst_hw
    xs = np.clip(to_grid(np.arange(W), W, w), 0, w - 1)
    ys = np.clip(to_grid(np.arange(H), H, h), 0, h - 1)
    gx, gy = np.meshgrid(xs, ys)
    return sample_bilinear(img, gx, gy, valid)


def depth_from_inv(inv) -> np.ndarray:
    inv = np.asarray(inv, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(np.isfinite(inv) & (inv > 0), 1.0 / inv, np.nan)
    return d


# ------------------------------------------------------------------------------ the flow
@dataclass
class Flow:
    flow: np.ndarray    # (2, h, w), nan where depth is invalid
    xq: np.ndarray      # target x
    yq: np.ndarray      # target y
    z: np.ndarray       # target z' in camera 2
    valid: np.ndarray   # depth valid and z' finite


def backproject(depth, K) -> np.ndarray:
    d = np.asarray(depth, np.float64)
    h, w = d.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    Kinv = np.linalg.inv(np.asarray(K, np.float64))
    rays = np.stack([xs, ys, np.ones_like(xs)], 0).reshape(3, -1)
    return (Kinv @ rays).reshape(3, h, w) * d[None]


def project(X, K) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    K = np.asarray(K, np.float64)
    shp = X.shape[1:]
    P = K @ X.reshape(3, -1)
    z = P[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        x, y = P[0] / z, P[1] / z
    return x.reshape(shp), y.reshape(shp), X.reshape(3, -1)[2].reshape(shp)


def rigid_flow(depth, K, M) -> Flow:
    """F_rig of every pixel of frame 1 under X_2 = M X_1, one K for both projections (eq. 0)."""
    d = np.asarray(depth, np.float64)
    h, w = d.shape
    X = backproject(d, K)
    M = np.asarray(M, np.float64)
    X2 = (M[:3, :3] @ X.reshape(3, -1) + M[:3, 3:4]).reshape(3, h, w)
    xq, yq, z = project(X2, K)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    valid = np.isfinite(d) & (d > 0) & np.isfinite(z) & (z != 0)
    flow = np.stack([np.where(valid, xq - xs, np.nan), np.where(valid, yq - ys, np.nan)], 0)
    return Flow(flow, np.where(valid, xq, np.nan), np.where(valid, yq, np.nan),
                np.where(valid, z, np.nan), valid)


def oof_mask(r: Flow, hw: tuple[int, int]) -> np.ndarray:
    H, W = hw
    with np.errstate(invalid="ignore"):
        inside = (r.xq >= -TOL) & (r.xq <= W - 1 + TOL) & (r.yq >= -TOL) \
            & (r.yq <= H - 1 + TOL) & (r.z > 0)
    return r.valid & ~inside


def depth_test(r: Flow, inv_depth2, oof, s: float = 1.0, eps: float = EPS_Z,
               valid2=None) -> dict:
    """O_T, front (a valid_T veto) and zunknown, from frame 2's inverse depth sampled at q."""
    inv_q, ok = sample_bilinear(inv_depth2, r.xq, r.yq, valid=_pos(inv_depth2, valid2))
    ok &= r.valid & ~oof
    with np.errstate(divide="ignore", invalid="ignore"):
        D2 = np.where(ok, 1.0 / inv_q, np.nan) * s
        occ = ok & (r.z > (1 + eps) * D2)
        front = ok & (r.z < (1 - eps) * D2)
    return {"occ": occ, "front": front, "zunknown": r.valid & ~oof & ~ok}


def _pos(inv, valid=None) -> np.ndarray:
    inv = np.asarray(inv, np.float64)
    ok = np.isfinite(inv) & (inv > 0)
    return ok if valid is None else ok & np.asarray(valid, bool)


def cycle_error(flow12, flow21) -> np.ndarray:
    """||F12(p) + F21(p + F12(p))||_1 where F21 is defined around the target, else 0."""
    _, h, w = flow12.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    xq, yq = xs + flow12[0], ys + flow12[1]
    u, oku = sample_bilinear(flow21[0], xq, yq)
    v, okv = sample_bilinear(flow21[1], xq, yq)
    ok = oku & okv & np.isfinite(flow12).all(0)
    e = np.abs(flow12[0] + u) + np.abs(flow12[1] + v)
    return np.where(ok, e, 0.0)


def rank_u8(conf_stack, valid_stack) -> np.ndarray:
    """Window-wide percentile rank of a confidence stack as uint8 (0 = lowest, 255 = highest);
    ties share their mean rank; invalid pixels get 0 (they are invalid anyway)."""
    c = np.asarray(conf_stack, np.float64)
    ok = np.asarray(valid_stack, bool) & np.isfinite(c)
    out = np.zeros(c.shape, np.uint8)
    vals = c[ok]
    if vals.size == 0:
        return out
    if vals.size == 1:
        out[ok] = 255
        return out
    order = np.argsort(vals, kind="stable")
    ranks = np.empty(vals.size, np.float64)
    ranks[order] = np.arange(vals.size, dtype=np.float64)
    uniq, inv = np.unique(vals, return_inverse=True)
    mean_rank = np.bincount(inv, weights=ranks) / np.bincount(inv)
    out[ok] = np.rint(255.0 * mean_rank[inv] / (vals.size - 1)).astype(np.uint8)
    return out


def c_T(rank) -> np.ndarray:
    r = np.asarray(rank, np.float64) / 255.0
    return np.clip((r - C_LO) / C_SPAN, 0.0, 1.0)


# ------------------------------------------------------------------------------ the renderer
def render_pair(inv1, inv2, K, M12, grid_hw: tuple[int, int], s21: float = 1.0,
                valid1=None, valid2=None, consts: dict | None = None,
                need_target_depth: bool = False) -> dict:
    """Both directions of one pair on grid_hw. inv1 / inv2: inverse depth at the teacher's
    resolution (one model grid); K: the window's ONE intrinsics there (eq. 0; the caller picks
    it, p1lib/geomio.py window_K); M12: X_2 = M12 X_1. Depth is upsampled to the grid with
    strict bilinear resizing and rendered with K zoomed to the grid (the canonical mode;
    render_model_mode() is the alternative the W2-14 resolution check compares with). inv2 may be
    None (self-triangulation has no D_{t+1}): the 1->2 depth test is then unknown everywhere and
    the 2->1 direction is not rendered. s21 scales frame 2's depth onto frame 1's (1 when both
    come from one forward pass). valid_T follows the proposal (depth valid, no 'front' veto,
    b_T <= b_max); zunknown is a separate flag, and need_target_depth=True also vetoes it."""
    k = dict(CONSTANTS)
    k.update(consts or {})
    H, W = grid_hw
    inv1 = np.asarray(inv1, np.float64)
    g1, ok1 = resize_bilinear(inv1, grid_hw, _pos(inv1, valid1))
    Kg = zoom_K(K, inv1.shape, grid_hw)
    have2 = inv2 is not None
    if have2:
        inv2 = np.asarray(inv2, np.float64)
        if inv2.shape != inv1.shape:
            raise ValueError(f"render_pair: frames on two grids {inv1.shape} / {inv2.shape}")
        g2, ok2 = resize_bilinear(inv2, grid_hw, _pos(inv2, valid2))
    else:
        g2, ok2 = None, None
    out = {"grid": [H, W], "constants": k}
    dirs = [("12", g1, ok1, np.asarray(M12, np.float64), g2, ok2, s21)]
    if have2:
        # frame 2's depth is put in frame 1's units before it moves (below), so its test
        # against frame 1's depth needs no further scale
        dirs.append(("21", g2, ok2, np.linalg.inv(M12), g1, ok1, 1.0))
    for tag, ga, oka, M, gb, okb, s in dirs:
        Da = depth_from_inv(np.where(oka, ga, np.nan))
        if tag == "21":
            Da = Da * s21  # frame 2's depth in frame 1's units before it moves
        r = rigid_flow(Da, Kg, M)
        oof = oof_mask(r, grid_hw)
        if gb is not None:
            dt = depth_test(r, np.where(okb, gb, np.nan), oof, s=s, eps=k["eps_z"])
        else:
            z = np.zeros((H, W), bool)
            dt = {"occ": z, "front": z, "zunknown": r.valid & ~oof}
        r2 = rigid_flow(Da * (1.0 + k["rho"]), Kg, M)
        sens = np.abs(r2.flow - r.flow).sum(0)
        out[f"flow_{tag}"] = r.flow
        out[f"z_{tag}"] = r.z
        out[f"valid_depth_{tag}"] = r.valid
        out[f"oof_{tag}"] = oof
        out[f"occ_{tag}"] = dt["occ"]
        out[f"front_{tag}"] = dt["front"]
        out[f"zunknown_{tag}"] = dt["zunknown"]
        out[f"sens_{tag}"] = np.where(r.valid, sens, np.nan)
    for tag, other in (("12", "21"), ("21", "12")):
        if f"flow_{tag}" not in out:
            continue
        if f"flow_{other}" in out:
            ecyc = cycle_error(out[f"flow_{tag}"], out[f"flow_{other}"])
        else:
            ecyc = np.zeros((H, W))
        b = k["b0"] + out[f"sens_{tag}"] + k["lambda_c"] * ecyc
        out[f"ecyc_{tag}"] = ecyc
        out[f"b_{tag}"] = b
        out[f"valid_{tag}"] = (out[f"valid_depth_{tag}"] & ~out[f"front_{tag}"]
                               & np.isfinite(b) & (b <= k["b_max"]))
        if need_target_depth:
            out[f"valid_{tag}"] &= ~out[f"zunknown_{tag}"] | out[f"oof_{tag}"]
    return out


def render_model_mode(inv1, K, M12, grid_hw: tuple[int, int], valid1=None):
    """F_rig 1->2 rendered at the teacher's resolution, then carried to the grid by eq. 2: the
    target field is sampled at each grid pixel's model coordinate and mapped back. Only the flow
    (for the W2-14 resolution decision)."""
    inv1 = np.asarray(inv1, np.float64)
    r = rigid_flow(depth_from_inv(np.where(_pos(inv1, valid1), inv1, np.nan)), K, M12)
    return carry_flow(r.flow, grid_hw)


def carry_flow(flow, dst_hw: tuple[int, int]) -> np.ndarray:
    """A flow field (2, h, w) expressed on another pixel grid by eq. 2: each destination pixel's
    source coordinate is found, the source target field is sampled there (strict bilinear) and
    mapped back. Used for RAFT's native flow on the training grid and for render_model_mode.
    It is NOT U2Flow's resize_flow (align_corners=True), which stays the evaluator's own
    convention for student predictions (metric section 1)."""
    flow = np.asarray(flow, np.float64)
    _, h, w = flow.shape
    H, W = dst_hw
    ys0, xs0 = np.mgrid[0:h, 0:w].astype(np.float64)
    gx = np.clip(to_grid(np.arange(W), W, w), 0, w - 1)
    gy = np.clip(to_grid(np.arange(H), H, h), 0, h - 1)
    mx, my = np.meshgrid(gx, gy)
    qx, okx = sample_bilinear(xs0 + flow[0], mx, my)
    qy, oky = sample_bilinear(ys0 + flow[1], mx, my)
    Qx, Qy = to_grid(qx, w, W), to_grid(qy, h, H)
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    ok = okx & oky
    return np.stack([np.where(ok, Qx - xs, np.nan), np.where(ok, Qy - ys, np.nan)], 0)


def flow_diff_stats(a, b) -> dict:
    ok = np.isfinite(a).all(0) & np.isfinite(b).all(0)
    if not ok.any():
        return {"n": 0, "max_px": None, "mean_px": None}
    d = np.sqrt(((a - b) ** 2).sum(0))[ok]
    return {"n": int(ok.sum()), "max_px": float(d.max()), "mean_px": float(d.mean()),
            "p99_px": float(np.percentile(d, 99))}
