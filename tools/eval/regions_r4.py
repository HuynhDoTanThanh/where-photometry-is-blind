#!/usr/bin/env python3
"""Metric regions v2: split R4 = V_occ & FG into R4m (moving), R4s (static) and R4u
(undetermined) by a GT ego-motion fit, and mark R4m-epi (moving instances that lie on their
ego-motion epipolar lines). Evaluation only: every input is GT (flow_occ, flow_noc, obj_map,
disp_occ_0) or public calibration; nothing here ever reaches a teacher or a training run.

STAGED (2026-10-03, Tier 2 package T2C). Final path: tools/eval/regions_r4.py. It sits here
because any untracked file under tools/eval/ makes tools/colab/launch.sh refuse every launch
(exit 64, its dirty check covers tools/eval) and moves eval_sha, which score_cached.py checks
against each run's [V tools/colab/launch.sh:277-285; tools/eval/score_cached.py:266-268;
tools/eval/regions.py eval_files]. The integrator moves it with the regions.py diff in one
commit, at a moment when no scoring of an earlier run is pending (a new eval_sha group).

What it does
    parse_calib(text)              P_rect_02 / P_rect_03 of calib_cam_to_cam -> K, f, baseline
    decode_disp(buf)               KITTI disparity PNG -> (float32 disparity, valid) [devkit]
    CalibSource(root= | zip_path=) calib text per id, sha256-checked against the meta bundle's
                                   FILES.sha256 (inside the zip, or <root>/.bundles/u2f-meta-v1)
    find_calib(gt_root)            $U2_K15_CALIB, <gt_root>/kitti15/calib (VM DATA_ROOT layout),
                                   then <gt_root>/../bundles/u2f-meta-v1.zip (the Mac layout)
    fit_ego(...)                   PnP-RANSAC of R1 points (back-projected with disp_occ_0) to
                                   their GT targets p + F_gt in frame 2: (R, t), diagnostics
    split_r4(...)                  the masks R4m, R4s, R4u, R4m-epi and the per-instance table;
                                   asserts S7 (R4m, R4s, R4u partition R4; R4m-epi within R4m)
    Library, plus a dev-only sanity CLI:
        python tools/eval/regions_r4.py sanity [--gt-root D] [--calib PATH] [--ids 000001,...]
            [--json OUT]
        It reads dev GT only (splits/k15_split_v1.json dev_ids; a held-out id or a path under
        sealed/ is refused), asserts S1-S3 through regions.masks and S7 here, and prints the
        R4 family counts and the ego-fit diagnostics. It never prints an R2 count (regions.md
        S6: the R2 near-empty threshold is fixed before any R2 count is seen) and never reads
        a prediction.
Exit codes (CLI): 0 every check passed · 1 a sanity check failed (S1-S3, S7) · 2 refused (a
    held-out id, a sealed path, missing GT or calib) · 64 usage.

Pinned definitions (regions.md 'The R4 split'; [P] proposal final.region_protocol; [E] decided
here on 2026-10-03 BEFORE any R4 number existed; changing one is regions_v3, metric skill s9)
    Camera     rectified camera 2 (image_2, the scored view): K = P_rect_02[:, :3], f = K[0, 0];
               baseline b = (P_rect_02[0, 3] - P_rect_03[0, 3]) / f metres; depth Z = f * b / d.
               The 2.7 mm z offsets of P_rect_02/03 [M 000001] are ignored: under 1e-3 relative
               at Z >= 3 m [E].
    Disparity  disp_occ_0 PNG uint16 / 256, valid where the raw value > 0 [devkit readme;
               regions.md 'Inputs'].
    Ego fit    points: R1 = V_noc & B pixels with valid disparity; targets p + F_gt (flow_occ
               values on V_noc, equal to flow_noc there by S3). Deterministic stride subsample
               to at most EGO_MAX_PTS; cv2.solvePnPRansac (SOLVEPNP_EPNP, reprojection
               EGO_RANSAC_PX, EGO_RANSAC_ITERS iterations, confidence EGO_RANSAC_CONF, after
               cv2.setRNGSeed(EGO_SEED)), then cv2.solvePnPRefineLM on the inliers. The fit fails
               (whole image -> R4u) with fewer than EGO_MIN_PTS points or an inlier share under
               EGO_MIN_INLIER_FRAC. cv2 RANSAC differs between OpenCV versions [A4], so
               lock_hash is in every region row's protocol key.
    F_ego      for an FG pixel with valid disparity: back-project, X' = R X + t, project; minus p.
    Moving     an instance is one obj_map value > 0 (every value its own instance [M 2026-10-03:
               the first 8 dev obj_maps hold uint8 values 1-5 and 14]; one object each [E]).
               It is decided on its pixels in V_occ with valid disparity; fewer than
               MIN_INST_PX of them -> R4u. Moving iff
               median |F_gt - F_ego| > MOVING_ABS_PX [P] AND median (|F_gt - F_ego| /
               (|F_gt| + 1e-8)) > MOVING_REL [P; the '5% of what' pinned as the per-pixel relative
               residual, mirroring the outlier rule, E]. Else static. The decision covers ALL the
               instance's R4 pixels (with or without disparity).
    R4m-epi    a moving instance whose median Sampson distance over its R4 pixels is
               < EPI_SAMPSON_PX [P], with x1 = p, x2 = p + F_gt and F = K^-T [t]x R K^-1;
               the distance is sqrt of the first-order (Sampson) error, in px [E: the usual
               geometric reading of 'Sampson distance in px', Hartley & Zisserman eq. 11.9].
               When |t| < EPI_MIN_T_M the epipolar geometry is degenerate (a stopped car):
               no instance of that image is R4m-epi, and info.ego.epi_degenerate says so [E].
    R4u        every R4 pixel of an undetermined instance, or all of R4 when the ego fit fails;
               never guessed between moving and static [regions.md: a guess would bias the R4m
               non-inferiority row].

Why it is built this way
    - The claim is regional (regions.md): R4m is where a rigid teacher must NOT be trusted, so the
      moving/static split is the instrument behind H4's R4m non-inferiority margin and GATE-1 (c).
      It must exist, pinned, before any model's R4m number does.
    - Undetermined is its own region (R4u) and its share is printed: a silent default (static)
      would move independently-moving pixels into R4s and flatter a leaky teacher.
    - Deterministic: fixed RANSAC seed, stride subsample (no random draw), float64 geometry.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _code_root() -> Path:
    """The repo root, from tools/eval/ (final) or tools/staging/eval/ (staged)."""
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "AGENTS.md").is_file() and (p / "tools" / "eval").is_dir():
            return p
    return here.parents[2]


CODE_ROOT = _code_root()
REGIONS_VERSION = "regions_v2"
R4_REGIONS = ("R4m", "R4s", "R4u", "R4m-epi")
META_BUNDLE = "u2f-meta-v1"
CALIB_REL = "kitti15/calib/training/calib_cam_to_cam/{id}.txt"
SPLIT_REL = "splits/k15_split_v1.json"

# ---- pinned constants (docstring 'Pinned definitions'; regions_v2) ----
MOVING_ABS_PX = 3.0         # [P] proposal region_protocol
MOVING_REL = 0.05           # [P] '> 5%'; read as the per-pixel relative residual [E]
EPI_SAMPSON_PX = 1.0        # [P] proposal region_protocol
MIN_INST_PX = 50            # [E] pixels with disparity needed to decide an instance
EGO_MIN_PTS = 500           # [E] R1 points with disparity needed for the fit
EGO_MIN_INLIER_FRAC = 0.5   # [E] RANSAC inlier share below which the fit is not trusted
EGO_MAX_PTS = 20000         # [E] stride subsample cap (speed only)
EGO_RANSAC_PX = 1.0         # [E] KITTI-15 background GT flow is itself rigid (scan + ego-motion)
EGO_RANSAC_ITERS = 2000     # [E]
EGO_RANSAC_CONF = 0.9999    # [E]
EGO_SEED = 20261002         # [E] the split seed, reused so no new number is chosen
EPI_MIN_T_M = 0.05          # [E] |t| under 5 cm per frame (0.5 m/s at 10 Hz): degenerate
EPS = 1e-8

PINNED = {"MOVING_ABS_PX": MOVING_ABS_PX, "MOVING_REL": MOVING_REL,
          "EPI_SAMPSON_PX": EPI_SAMPSON_PX, "MIN_INST_PX": MIN_INST_PX,
          "EGO_MIN_PTS": EGO_MIN_PTS, "EGO_MIN_INLIER_FRAC": EGO_MIN_INLIER_FRAC,
          "EGO_MAX_PTS": EGO_MAX_PTS, "EGO_RANSAC_PX": EGO_RANSAC_PX,
          "EGO_RANSAC_ITERS": EGO_RANSAC_ITERS, "EGO_RANSAC_CONF": EGO_RANSAC_CONF,
          "EGO_SEED": EGO_SEED, "EPI_MIN_T_M": EPI_MIN_T_M}


class R4Error(AssertionError):
    """S7 failed, or an input is malformed: the split cannot be trusted."""


class CalibError(RuntimeError):
    """Calibration missing, unparsable or not matching its recorded sha256."""


# ------------------------------------------------------------------------------ inputs
def parse_calib(text: str) -> dict:
    """calib_cam_to_cam text -> {"K", "f", "baseline_m", "P2", "P3"} (float64)."""
    rows: dict[str, np.ndarray] = {}
    for line in text.splitlines():
        key, sep, rest = line.partition(":")
        if not sep:
            continue
        try:
            rows[key.strip()] = np.array([float(x) for x in rest.split()], np.float64)
        except ValueError:
            continue  # calib_time is a date, not numbers
    try:
        p2 = rows["P_rect_02"].reshape(3, 4)
        p3 = rows["P_rect_03"].reshape(3, 4)
    except (KeyError, ValueError):
        raise CalibError("calib_cam_to_cam needs P_rect_02 and P_rect_03 (3 x 4)") from None
    k = p2[:, :3].copy()
    f = float(k[0, 0])
    if not (f > 0 and abs(k[1, 1] - f) < 1e-6 * f and k[2, 2] == 1.0):
        raise CalibError(f"P_rect_02 is not a rectified pinhole matrix (f={f})")
    b = float((p2[0, 3] - p3[0, 3]) / f)
    if not (0.1 < b < 2.0):
        raise CalibError(f"stereo baseline {b:.4f} m is implausible for KITTI (about 0.54 m)")
    return {"K": k, "f": f, "baseline_m": b, "P2": p2, "P3": p3}


def decode_disp(buf: bytes) -> tuple[np.ndarray, np.ndarray]:
    """KITTI disparity PNG (uint16, value / 256, 0 = invalid) -> (float32 disparity, bool valid)."""
    import cv2

    raw = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_UNCHANGED)
    if raw is None or raw.ndim != 2 or raw.dtype != np.uint16:
        raise R4Error("disp_occ_0 is not a single-channel uint16 PNG")
    valid = raw > 0
    return (raw.astype(np.float32) / np.float32(256.0)), valid


def _listing(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        if line.strip():
            sha, rel = line.split(None, 1)
            out[rel.strip()] = sha
    return out


class CalibSource:
    """Calibration per KITTI-15 training id, from a directory laid out like DATA_ROOT
    (<root>/kitti15/calib/training/calib_cam_to_cam/<id>.txt) or from the u2f-meta-v1 zip.
    Every file is checked against the bundle's FILES.sha256 (a file missing from the list is an
    error, not a pass). `pack_sha` is the zip's sha256 (provenance for metrics.json), or None."""

    def __init__(self, root: Path | str | None = None, zip_path: Path | str | None = None,
                 verify: bool = True) -> None:
        if (root is None) == (zip_path is None):
            raise ValueError("CalibSource needs exactly one of root or zip_path")
        self.root = Path(root) if root is not None else None
        self.zip_path = Path(zip_path) if zip_path is not None else None
        self._zip = zipfile.ZipFile(self.zip_path) if self.zip_path is not None else None
        self.listing: dict[str, str] | None = None
        self.pack_sha: str | None = None
        if self._zip is not None:
            h = hashlib.sha256()
            with open(self.zip_path, "rb") as f:
                for block in iter(lambda: f.read(8 << 20), b""):
                    h.update(block)
            self.pack_sha = h.hexdigest()
            if verify:
                try:
                    self.listing = _listing(self._zip.read("FILES.sha256").decode("utf-8"))
                except KeyError:
                    raise CalibError(f"{self.zip_path} has no FILES.sha256") from None
        elif verify:
            lst = self.root / ".bundles" / META_BUNDLE / "FILES.sha256"
            if lst.is_file():
                self.listing = _listing(lst.read_text("utf-8"))
        self.verified = self.listing is not None
        self._cache: dict[str, dict] = {}

    def read_text(self, img_id: str) -> str:
        rel = CALIB_REL.format(id=img_id)
        try:
            data = self._zip.read(rel) if self._zip is not None else (self.root / rel).read_bytes()
        except (KeyError, OSError):
            raise CalibError(f"calibration {rel} is missing") from None
        if self.listing is not None:
            want = self.listing.get(rel)
            if want is None or hashlib.sha256(data).hexdigest() != want:
                raise CalibError(f"{rel} does not match the {META_BUNDLE} FILES.sha256")
        return data.decode("utf-8")

    def read(self, img_id: str) -> dict:
        if img_id not in self._cache:
            self._cache[img_id] = parse_calib(self.read_text(img_id))
        return self._cache[img_id]

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()


def find_calib(gt_root: Path | str | None) -> CalibSource | None:
    """The calibration for a GT root, or None when none is found (regions.masks then puts all
    of R4 in R4u and says why). Order: $U2_K15_CALIB (a dir or a zip), <gt_root>/kitti15/calib
    (the VM's DATA_ROOT after fetch.py extracts u2f-meta-v1), <gt_root>/../bundles/u2f-meta-v1.zip
    (the Mac: U2FLOW_data/gt beside U2FLOW_data/bundles)."""
    env = os.environ.get("U2_K15_CALIB")
    cands: list[Path] = [Path(env)] if env else []
    if gt_root is not None:
        g = Path(gt_root)
        cands += [g, g.parent / "bundles" / f"{META_BUNDLE}.zip"]
    for c in cands:
        if c.is_file() and c.suffix == ".zip":
            return CalibSource(zip_path=c)
        if c.is_dir() and (c / "kitti15" / "calib" / "training" / "calib_cam_to_cam").is_dir():
            return CalibSource(root=c)
    return None


# ------------------------------------------------------------------------------ geometry
def backproject(xs: np.ndarray, ys: np.ndarray, d: np.ndarray, k: np.ndarray,
                b: float) -> np.ndarray:
    """(N, 3) camera-2 points of pixels (xs, ys) with disparity d > 0 (float64)."""
    f, cx, cy = k[0, 0], k[0, 2], k[1, 2]
    z = f * b / d.astype(np.float64)
    return np.stack([(xs - cx) * z / f, (ys - cy) * z / f, z], axis=1)


def project(pts: np.ndarray, k: np.ndarray) -> np.ndarray:
    """(N, 2) pixels of (N, 3) camera points; points at z <= 0 give NaN."""
    z = pts[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = k[0, 0] * pts[:, 0] / z + k[0, 2]
        v = k[1, 1] * pts[:, 1] / z + k[1, 2]
    bad = ~(z > 1e-9)
    u[bad] = np.nan
    v[bad] = np.nan
    return np.stack([u, v], axis=1)


def skew(t: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -t[2], t[1]], [t[2], 0.0, -t[0]], [-t[1], t[0], 0.0]])


def fundamental(k: np.ndarray, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    """F with x2^T F x1 = 0 for X2 = R X1 + t, one camera K: K^-T [t]x R K^-1."""
    ki = np.linalg.inv(k)
    return ki.T @ skew(np.asarray(t, np.float64).ravel()) @ r @ ki


def sampson_px(x1: np.ndarray, x2: np.ndarray, fm: np.ndarray) -> np.ndarray:
    """sqrt of the first-order (Sampson) error of each correspondence, in px (N,)."""
    h1 = np.concatenate([x1, np.ones((len(x1), 1))], axis=1)
    h2 = np.concatenate([x2, np.ones((len(x2), 1))], axis=1)
    fx1 = h1 @ fm.T                    # rows: F x1
    ftx2 = h2 @ fm                     # rows: F^T x2
    num = np.einsum("ij,ij->i", h2, fx1) ** 2
    den = fx1[:, 0] ** 2 + fx1[:, 1] ** 2 + ftx2[:, 0] ** 2 + ftx2[:, 1] ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(num / den)


@dataclass
class EgoFit:
    ok: bool
    r: np.ndarray | None = None
    t: np.ndarray | None = None
    n_pts: int = 0
    n_inliers: int = 0
    median_resid_px: float | None = None
    t_norm_m: float | None = None
    reason: str = ""

    def to_json(self) -> dict:
        return {"ok": self.ok, "n_pts": self.n_pts, "n_inliers": self.n_inliers,
                "inlier_frac": round(self.n_inliers / self.n_pts, 6) if self.n_pts else None,
                "median_resid_px": None if self.median_resid_px is None
                else round(self.median_resid_px, 6),
                "t_norm_m": None if self.t_norm_m is None else round(self.t_norm_m, 6),
                "reason": self.reason}


def fit_ego(flow: np.ndarray, r1: np.ndarray, disp: np.ndarray, dvalid: np.ndarray,
            k: np.ndarray, b: float) -> EgoFit:
    """PnP-RANSAC of the R1 points with valid disparity to their GT targets (docstring)."""
    import cv2

    ys, xs = np.nonzero(r1 & dvalid)
    n = len(xs)
    if n < EGO_MIN_PTS:
        return EgoFit(False, n_pts=n, reason=f"{n} R1 points with disparity < {EGO_MIN_PTS}")
    step = max(1, -(-n // EGO_MAX_PTS))  # ceil: a deterministic stride, no random draw
    ys, xs = ys[::step], xs[::step]
    obj = backproject(xs.astype(np.float64), ys.astype(np.float64), disp[ys, xs], k, b)
    img = np.stack([xs + flow[0, ys, xs].astype(np.float64),
                    ys + flow[1, ys, xs].astype(np.float64)], axis=1)
    cv2.setRNGSeed(EGO_SEED)
    ok, rvec, tvec, inl = cv2.solvePnPRansac(
        obj, img, k, None, iterationsCount=EGO_RANSAC_ITERS, reprojectionError=EGO_RANSAC_PX,
        confidence=EGO_RANSAC_CONF, flags=cv2.SOLVEPNP_EPNP)
    m = len(xs)
    if not ok or inl is None:
        return EgoFit(False, n_pts=m, reason="solvePnPRansac found no model")
    inl = inl.ravel()
    if len(inl) < EGO_MIN_INLIER_FRAC * m or len(inl) < 6:
        return EgoFit(False, n_pts=m, n_inliers=len(inl),
                      reason=f"inlier share {len(inl) / m:.3f} < {EGO_MIN_INLIER_FRAC}")
    rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], k, None, rvec, tvec)
    r, _ = cv2.Rodrigues(rvec)
    t = tvec.ravel().astype(np.float64)
    pred = project(obj @ r.T + t, k)
    res = np.linalg.norm(pred - img, axis=1)
    return EgoFit(True, r=r, t=t, n_pts=m, n_inliers=int(len(inl)),
                  median_resid_px=float(np.nanmedian(res[inl])), t_norm_m=float(np.linalg.norm(t)))


# ------------------------------------------------------------------------------ the split
def split_r4(flow: np.ndarray, v_occ: np.ndarray, v_noc: np.ndarray, obj_map: np.ndarray,
             disp: np.ndarray, dvalid: np.ndarray, calib: dict) -> tuple[dict, dict]:
    """Masks {R4m, R4s, R4u, R4m-epi} (bool, H x W) and info {ego, instances, checks}.

    flow is the GT flow_occ (2, H, W) float32; v_occ/v_noc the GT validity (bool); obj_map the
    instance map (0 = background); disp/dvalid from decode_disp; calib from parse_calib."""
    h, w = v_occ.shape
    for name, a in (("flow", flow[0]), ("v_noc", v_noc), ("obj_map", obj_map), ("disp", disp),
                    ("dvalid", dvalid)):
        if a.shape != (h, w):
            raise R4Error(f"{name} has shape {a.shape}, expected {(h, w)}")
    k, b = calib["K"], calib["baseline_m"]
    bg, fg = obj_map == 0, obj_map > 0
    r4 = v_occ & fg
    r1 = v_noc & bg
    ego = fit_ego(flow, r1, disp, dvalid, k, b)
    masks = {n: np.zeros((h, w), bool) for n in R4_REGIONS}
    insts: list[dict] = []
    epi_degenerate = bool(ego.ok and ego.t_norm_m is not None and ego.t_norm_m < EPI_MIN_T_M)
    fm = fundamental(k, ego.r, ego.t) if ego.ok and not epi_degenerate else None
    for iid in (int(x) for x in np.unique(obj_map[r4])):
        sel = r4 & (obj_map == iid)
        rec: dict = {"id": iid, "n_px": int(sel.sum())}
        if not ego.ok:
            rec.update(cls="undetermined", why="ego fit failed")
            masks["R4u"] |= sel
            insts.append(rec)
            continue
        use = sel & dvalid
        ys, xs = np.nonzero(use)
        rec["n_disp"] = len(xs)
        if len(xs) < MIN_INST_PX:
            rec.update(cls="undetermined", why=f"{len(xs)} px with disparity < {MIN_INST_PX}")
            masks["R4u"] |= sel
            insts.append(rec)
            continue
        pts = backproject(xs.astype(np.float64), ys.astype(np.float64), disp[ys, xs], k, b)
        tgt = project(pts @ ego.r.T + ego.t, k)
        fego = tgt - np.stack([xs, ys], axis=1).astype(np.float64)
        fgt = np.stack([flow[0, ys, xs], flow[1, ys, xs]], axis=1).astype(np.float64)
        res = np.linalg.norm(fgt - fego, axis=1)
        rel = res / (np.linalg.norm(fgt, axis=1) + EPS)
        med_res, med_rel = float(np.nanmedian(res)), float(np.nanmedian(rel))
        moving = med_res > MOVING_ABS_PX and med_rel > MOVING_REL
        rec.update(med_res_px=round(med_res, 6), med_rel=round(med_rel, 6),
                   cls="moving" if moving else "static")
        masks["R4m" if moving else "R4s"] |= sel
        if moving:
            if fm is None:
                rec["epi"] = None
            else:
                yy, xx = np.nonzero(sel)
                x1 = np.stack([xx, yy], axis=1).astype(np.float64)
                x2 = x1 + np.stack([flow[0, yy, xx], flow[1, yy, xx]], axis=1)
                med_s = float(np.nanmedian(sampson_px(x1, x2, fm)))
                rec["med_sampson_px"] = round(med_s, 6)
                rec["epi"] = med_s < EPI_SAMPSON_PX
                if rec["epi"]:
                    masks["R4m-epi"] |= sel
        insts.append(rec)
    # S7: R4m, R4s, R4u are disjoint and their union is R4; R4m-epi lies within R4m.
    parts = ("R4m", "R4s", "R4u")
    union = np.zeros((h, w), bool)
    for p in parts:
        if np.any(union & masks[p]):
            raise R4Error(f"S7 failed: {p} overlaps an earlier R4 part")
        union |= masks[p]
    if not np.array_equal(union, r4):
        raise R4Error("S7 failed: R4m, R4s and R4u do not partition R4")
    if np.any(masks["R4m-epi"] & ~masks["R4m"]):
        raise R4Error("S7 failed: R4m-epi is not within R4m")
    n4 = int(r4.sum())
    info = {"ego": dict(ego.to_json(), epi_degenerate=epi_degenerate), "instances": insts,
            "checks": {"S7": True, "R4_px": n4, "R4u_px": int(masks["R4u"].sum()),
                       "R4u_share": round(int(masks["R4u"].sum()) / n4, 6) if n4 else None}}
    return masks, info


# ------------------------------------------------------------------------------ sanity CLI
def _load_regions():
    p = CODE_ROOT / "tools" / "eval" / "regions.py"
    spec = importlib.util.spec_from_file_location("u2f_regions_for_r4", p)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {p}")
    mod = sys.modules.get(spec.name) or importlib.util.module_from_spec(spec)
    if spec.name not in sys.modules:
        sys.modules[spec.name] = mod  # dataclasses resolve their module through sys.modules
        spec.loader.exec_module(mod)
    return mod


def dev_ids(root: Path = CODE_ROOT) -> list[str]:
    doc = json.loads((root / SPLIT_REL).read_text("utf-8"))
    return [str(x) for x in doc["dev_ids"]]


def sanity(gt_root: Path, calib: CalibSource, ids: list[str]) -> tuple[int, dict]:
    """S1-S3 (regions.masks) and S7 per dev image; returns (exit code, report). No R2 count and
    no prediction is ever read or printed."""
    rg = _load_regions()
    src = rg.GTSource("kitti15", root=gt_root)
    if not src.verified:
        raise CalibError(f"{gt_root}: no FILES.sha256 for {rg.GT_BUNDLE['kitti15']}; the GT "
                         "cannot be verified")
    per, failures = [], []
    tot = {"R4_px": 0, "R4m_px": 0, "R4s_px": 0, "R4u_px": 0, "R4m-epi_px": 0}
    cls_n = {"moving": 0, "static": 0, "undetermined": 0}
    for img_id in ids:
        try:
            gt = src.read(img_id)
            rg.masks(gt, "kitti15")                       # asserts S1, S2, S3
            disp, dvalid = decode_disp(src.read_bytes("disp_occ_0", img_id))
            m, info = split_r4(gt.flow_occ, gt.valid_occ >= 0.5, gt.valid_noc >= 0.5,
                               gt.obj_map, disp, dvalid, calib.read(img_id))
        except AssertionError as e:                     # RegionError, R4Error
            failures.append(f"{img_id}: {e}")
            continue
        row = {"id": img_id, "ego": info["ego"],
               "n_inst": {c: sum(1 for i in info["instances"] if i["cls"] == c) for c in cls_n},
               "px": {r: int(m[r].sum()) for r in R4_REGIONS}, "R4_px": info["checks"]["R4_px"]}
        for c in cls_n:
            cls_n[c] += row["n_inst"][c]
        tot["R4_px"] += row["R4_px"]
        for r in R4_REGIONS:
            tot[f"{r}_px"] += row["px"][r]
        per.append(row)
    n4 = tot["R4_px"]
    share = {r: round(tot[f"{r}_px"] / n4, 6) if n4 else None for r in ("R4m", "R4s", "R4u")}
    share["R4m-epi_of_R4m"] = round(tot["R4m-epi_px"] / tot["R4m_px"], 6) if tot["R4m_px"] \
        else None
    ego_fail = [r["id"] for r in per if not r["ego"]["ok"]]
    resid = sorted(r["ego"]["median_resid_px"] for r in per if r["ego"]["ok"])
    rep = {"regions_version": REGIONS_VERSION, "pinned": PINNED, "n_images": len(per),
           "ids": ids, "failures": failures, "totals_px": tot, "shares": share,
           "instances": cls_n, "ego_failed": ego_fail,
           "epi_degenerate": [r["id"] for r in per if r["ego"].get("epi_degenerate")],
           "ego_median_resid_px": {"min": resid[0] if resid else None,
                                   "median": resid[len(resid) // 2] if resid else None,
                                   "max": resid[-1] if resid else None},
           "calib_pack_sha": calib.pack_sha, "per_image": per,
           "standard": "measured", "split": "dev", "gt_only": True}
    src.close()
    return (1 if failures else 0), rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="regions_r4.py", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="verb")
    s = sub.add_parser("sanity", help="S1-S3 and S7 on dev GT; R4 counts; no predictions")
    s.add_argument("--gt-root", default="/Volumes/Shin/U2FLOW_data/gt")
    s.add_argument("--calib", help="a DATA_ROOT-like dir or the u2f-meta-v1 zip "
                                   "(default: find_calib)")
    s.add_argument("--ids", help="comma-separated dev ids (default: every dev id)")
    s.add_argument("--json", help="write the full report here")
    sub.add_parser("pinned", help="print the pinned constants")
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 64 if e.code else 0
    if a.verb == "pinned":
        print(json.dumps({"regions_version": REGIONS_VERSION, **PINNED}, sort_keys=True))
        return 0
    if a.verb != "sanity":
        ap.print_help()
        return 64
    gt_root = Path(a.gt_root)
    if "sealed" in gt_root.resolve().parts:
        print("regions_r4: refused: a path under sealed/ (AGENTS.md hard rule 4)", file=sys.stderr)
        return 2
    dev = dev_ids()
    ids = [x.strip() for x in a.ids.split(",") if x.strip()] if a.ids else dev
    bad = [x for x in ids if x not in set(dev)]
    if bad:
        print(f"regions_r4: refused: {bad[:5]} not dev ids of {SPLIT_REL} (the held-out half is "
              "sealed; this tool reads dev GT only)", file=sys.stderr)
        return 2
    try:
        calib = (CalibSource(zip_path=a.calib) if a.calib and a.calib.endswith(".zip") else
                 CalibSource(root=a.calib) if a.calib else find_calib(gt_root))
        if calib is None:
            raise CalibError("no calibration found (set --calib or $U2_K15_CALIB)")
        code, rep = sanity(gt_root, calib, ids)
    except (CalibError, OSError) as e:
        print(f"regions_r4: {e}", file=sys.stderr)
        return 2
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n", "utf-8")
    for r in rep["per_image"]:
        e = r["ego"]
        print(f"{r['id']} ego {'ok' if e['ok'] else 'FAIL'} inl {e['inlier_frac']} resid "
              f"{e['median_resid_px']} |t| {e['t_norm_m']} inst m/s/u "
              f"{r['n_inst']['moving']}/{r['n_inst']['static']}/{r['n_inst']['undetermined']} "
              f"px R4 {r['R4_px']} m {r['px']['R4m']} s {r['px']['R4s']} u {r['px']['R4u']} "
              f"epi {r['px']['R4m-epi']}")
    for f in rep["failures"]:
        print(f"FAIL {f}")
    summ = {k: rep[k] for k in ("n_images", "totals_px", "shares", "instances", "ego_failed",
                                "epi_degenerate", "ego_median_resid_px", "calib_pack_sha")}
    print("R4_SANITY " + json.dumps(summ, sort_keys=True))
    return code


if __name__ == "__main__":
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    sys.exit(main())
