#!/usr/bin/env python3
"""Metric v1 kernel: KITTI GT decode, region masks R1-R4 (+R3-rel, and the R4m/R4s/R4u/R4m-epi
split of regions_r4.py), per-image counts, the split aggregation and eval_sha. One code path for
every number tools/eval writes.

What it does
    decode_kitti_flow / decode_obj_map   the GT PNG decode of readFlowKITTI [V frame_utils.py:
                                         102-107], from bytes (a directory or a zip member)
    GTSource                             reads one image's GT from U2FLOW_data/gt (or the VM's
                                         DATA_ROOT) and checks each file against FILES.sha256
    masks(gt, dataset)                   the regions of references/regions.md, with S1-S4
    score_image(pred, gt, dataset)       per region: n (valid px), out (outliers), epe_sum
                                         (float64), and out3 for KITTI-12; plus the U2Flow
                                         per-image noc/occ EPE (zeros appended, trap T5)
    aggregate(per_image, dataset)        U2Flow's validate_kitti_2015 numbers (epe, fl, noc, occ,
                                         fl_noc, fl_occ) and the region table, from counts
    eval_sha(code_root)                  sha256 over every file that can move a number (metric
                                         skill section 9)
    Library only; `python tools/eval/regions.py --eval-sha` prints the evaluator hash.

The frozen semantics (metric skill section 1; evaluate.py at 152bb38, blob 3d3a14de7f6b)
    EPE map    sqrt(du^2 + dv^2) in float32 against the flow_occ values [V evaluate.py:543-545].
               float32 squares, a float32 add of two terms and a float32 sqrt are each correctly
               rounded, so this numpy form equals torch.sum(..., dim=0).sqrt() bit for bit [E].
    Outlier    epe > 3 and epe / (mag + 1e-8) > 0.05, all in float32 with float32 constants
               [V evaluate.py:552]; KITTI-12 drops the 1e-8 [V :860] (label Fl(K15-rule)) and
               also gets Out3 = epe > 3 (official KITTI-12 Out-All/Out-Noc rule, A4).
    Valid      third PNG channel >= 0.5 [V :550]. Every GT quantity is masked before use (T12).
    Fl         100 * sum(out) / sum(n) from integer counts (T8: never a float32 mean).
    EPE        image-mean: mean over images of epe_sum / n; pooled: sum(epe_sum) / sum(n).
    noc/occ    U2Flow's per-image noc EPE is against the flow_noc values, 0.0 when an image has
               no noc pixel [V :557-562]; occ is 0.0 when valid_occ - valid_noc is all zero
               [V :568-570]. KITTI-12 has no noc guard (NaN, kept as None) and an int 0 for occ
               [V :867-872]. These exist for parity only: label 'noc/occ EPE (U2Flow, zeros
               appended)'. Region EPE uses only images with n_R > 0 and reports n_images.

Regions (references/regions.md, regions_v1; coordinates x = column, y = row, F = (u, v))
    IN      0 <= x + u <= W-1 and 0 <= y + v <= H-1, evaluated on V_occ only
    K15     all = V_occ; noc = V_noc; occ = V_occ & ~V_noc; bg = V_occ & B; fg = V_occ & FG
            R1 = V_noc & B; R2 = V_occ & ~V_noc & B & IN; R3 = V_occ & ~V_noc & B & ~IN
            R3-rel = R3 & |F_gt| <= 100 px; R2+R3 = R2 | R3; R4 = V_occ & FG, split by
            regions_r4.split_r4 (regions_v2: a GT ego-motion fit from disp_occ_0 and the
            calib_cam_to_cam of u2f-meta-v1) into R4m / R4s / R4u, with R4m-epi within R4m; every
            constant is pinned in regions_r4.py. Without disparity or calibration the whole of R4
            is R4u and checks["R4"] says why. B = obj_map == 0, FG = obj_map > 0 [V devkit
            evaluate_scene_flow.cpp:139].
    K12     all, noc, occ; K12-R1 = V_noc; K12-R2 = V_occ & ~V_noc & IN; K12-R3 = V_occ & ~V_noc
            & ~IN (no obj_map ships with KITTI-12).
    Sanity  S2 (V_noc subset of V_occ) and S3 (flow_occ == flow_noc on V_noc, exact: both decode
            the same 1/64 px integers) are asserted per image before anything else; S1 (R1-R4,
            or K12-R1..3, partition V_occ: disjoint, union = V_occ, so the counts add up) is
            asserted per image; S7 (R4m, R4s, R4u partition R4) is asserted in regions_r4.
            S4 (V_noc & ~IN) is recorded per image, never printed: regions.md
            S6 fixes the R2 'near-empty' threshold in prereg-v1 BEFORE anyone looks at any count,
            so R2_NEAR_EMPTY stays None [U] and nothing here merges R2 into R3.

Why it is built this way
    - topaneu's parity test ran upstream against upstream, so the replica drifted while the
      suite stayed green (metric skill). Here the 152bb38 blob runs beside this kernel in
      tests/test_eval.py (link 1) and the Mac re-scores the VM's dumps with it (link 2).
    - Rates are never stored, only counts (statistics.md): every pooled number and every
      bootstrap replicate is recomputed from n, out and epe_sum.
    - eval_sha hashes every file of section 9, not evaluate.py alone: a resize_flow edit would
      otherwise move numbers with nothing in the protocol key showing it.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[2]
METRIC_VERSION = "v1"
REGIONS_VERSION = "regions_v2"  # v2: the R4 split (regions_r4.py); v1 rows stay in their group
PARITY_REF = "152bb38"
# git blob ids at 152bb38 (evaluator-traps.md "Map of evaluate.py"); link 1 stages these.
BLOB_IDS = {"evaluate.py": "3d3a14de7f6b", "core/utils/flow_utils.py": "c33e72c7bd09"}
# Metric skill section 9: every file that can move a number. tools/eval/*.py is added at runtime.
EVAL_FILES = (
    "evaluate.py",
    "core/utils/flow_utils.py",
    "core/utils/frame_utils.py",
    "core/utils/transforms/sep_transforms.py",
    "core/flow_datasets.py",
    "core/utils/utils.py",
    "core/Networks/__init__.py",
    "core/Networks/raft_2f_u.py",
    "core/Networks/corr.py",
    "core/Networks/extractor.py",
    "core/Networks/update.py",
)
K15_REGIONS = ("all", "noc", "occ", "bg", "fg", "R1", "R2", "R3", "R3-rel", "R2+R3", "R4", "R4m",
               "R4s", "R4u", "R4m-epi")
K12_REGIONS = ("all", "noc", "occ", "K12-R1", "K12-R2", "K12-R3")
R3_REL_MAX_PX = 100.0          # [P] pre-registered; SOTA check owned by method before prereg-v1
R2_NEAR_EMPTY: float | None = None   # [U] S6: fixed in prereg-v1 before any count is seen
GT_BUNDLE = {"kitti15": "u2f-k15-gt-dev-v1", "kitti12": "u2f-k12-gt-v1"}
GT_DIRS = {"kitti15": ("flow_occ", "flow_noc", "obj_map"), "kitti12": ("flow_occ", "flow_noc")}
R4_DIRS = ("disp_occ_0",)   # KITTI-15 only, for the R4 split (in u2f-k15-gt-dev-v1 [M])

_ABS = np.float32(3.0)
_REL = np.float32(0.05)
_EPS = np.float32(1e-8)


class RegionError(AssertionError):
    """A sanity check of regions.md failed (S1, S2 or S3): the GT or the masks are wrong."""


class IntegrityError(RuntimeError):
    """A file does not match the sha256 that was recorded for it."""


_R4: list = []


def r4mod():
    """tools/eval/regions_r4.py (the R4 split), loaded beside this file whatever sys.path is."""
    if not _R4:
        p = Path(__file__).with_name("regions_r4.py")
        spec = importlib.util.spec_from_file_location("u2f_regions_r4", p)
        if spec is None or spec.loader is None:
            raise IntegrityError(f"cannot load {p}")
        mod = sys.modules.get(spec.name) or importlib.util.module_from_spec(spec)
        if spec.name not in sys.modules:
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
        _R4.append(mod)
    return _R4[0]


# ------------------------------------------------------------------------------ hashing
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_blob_id(data: bytes) -> str:
    """git's object id of a blob (sha1 of 'blob <len>\\0' + data), computed without git."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def eval_files(code_root: Path = CODE_ROOT) -> list[str]:
    own = sorted(f"tools/eval/{p.name}" for p in (code_root / "tools" / "eval").glob("*.py"))
    return list(EVAL_FILES) + own


def eval_sha(code_root: Path = CODE_ROOT) -> tuple[str, dict[str, str]]:
    """sha256 of the sorted '<sha256>  <path>' lines of every evaluator file (section 9). A
    missing file is an error: an evaluator that cannot be hashed cannot be keyed."""
    files = {}
    for rel in eval_files(code_root):
        p = code_root / rel
        if not p.is_file():
            raise IntegrityError(f"eval_sha: {rel} is missing under {code_root}")
        files[rel] = sha256_file(p)
    body = "".join(f"{files[k]}  {k}\n" for k in sorted(files))
    return sha256_bytes(body.encode()), files


# ------------------------------------------------------------------------------ GT decode
def _imdecode(buf: bytes, flags: int) -> np.ndarray:
    import cv2

    arr = cv2.imdecode(np.frombuffer(buf, np.uint8), flags)
    if arr is None:
        raise IntegrityError("not a decodable PNG")
    return arr


def decode_kitti_flow(buf: bytes) -> tuple[np.ndarray, np.ndarray]:
    """readFlowKITTI on bytes: (H, W, 2) float32 flow and (H, W) float32 valid channel."""
    import cv2

    flow = _imdecode(buf, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    flow = flow[:, :, ::-1].astype(np.float32)
    flow, valid = flow[:, :, :2], flow[:, :, 2]
    flow = (flow - 2**15) / 64.0
    return flow, valid


def decode_obj_map(buf: bytes) -> np.ndarray:
    import cv2

    m = _imdecode(buf, cv2.IMREAD_UNCHANGED)
    if m.ndim != 2:
        raise IntegrityError(f"obj_map has shape {m.shape}, expected a single channel")
    return m


@dataclass
class GTImage:
    """One image's GT at full resolution: flows (2, H, W) float32, valid maps (H, W) float32."""

    flow_occ: np.ndarray
    valid_occ: np.ndarray
    flow_noc: np.ndarray
    valid_noc: np.ndarray
    obj_map: np.ndarray | None = None
    disp_occ: np.ndarray | None = None     # disp_occ_0 (float32 px) for the R4 split
    disp_valid: np.ndarray | None = None
    calib: dict | None = None              # regions_r4.parse_calib of calib_cam_to_cam
    r4_why: str = ""                       # why the R4 split is unavailable, when it is

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.valid_occ.shape[0]), int(self.valid_occ.shape[1])


def gt_from_arrays(flow_occ_hw2, valid_occ, flow_noc_hw2, valid_noc, obj_map=None) -> GTImage:
    return GTImage(np.ascontiguousarray(np.transpose(flow_occ_hw2, (2, 0, 1)), np.float32),
                   np.asarray(valid_occ, np.float32),
                   np.ascontiguousarray(np.transpose(flow_noc_hw2, (2, 0, 1)), np.float32),
                   np.asarray(valid_noc, np.float32), obj_map)


class GTSource:
    """GT of one dataset, from a directory laid out like the bundles (<root>/<dataset>/training/
    <dir>/<id>_10.png) or from a zip (members '<prefix>training/<dir>/<id>_10.png'). With a
    FILES.sha256 list (the bundle's own, at <root>/.bundles/<name>/) every file read is
    checked against it; a file missing from the list is an integrity error, not a pass.
    KITTI-15 also reads disp_occ_0 and the calibration (calib: a regions_r4.CalibSource, or
    "auto" = regions_r4.find_calib(root)) for the R4 split; with require_r4 a missing one is an
    IntegrityError, else that image's R4 is all R4u with the reason recorded."""

    def __init__(self, dataset: str, root: Path | str | None = None, zip_path: Path | str |
                 None = None, zip_prefix: str = "", verify: bool = True, calib="auto",
                 require_r4: bool = False) -> None:
        if dataset not in GT_DIRS:
            raise ValueError(f"unknown dataset {dataset!r}")
        if (root is None) == (zip_path is None):
            raise ValueError("GTSource needs exactly one of root or zip_path")
        self.dataset, self.zip_prefix = dataset, zip_prefix
        self.root = Path(root) if root is not None else None
        self._zip = zipfile.ZipFile(zip_path) if zip_path is not None else None
        self.listing: dict[str, str] | None = None
        self.bundle_dir: Path | None = None
        if self.root is not None and verify:
            self.bundle_dir = self.root / ".bundles" / GT_BUNDLE[dataset]
            lst = self.bundle_dir / "FILES.sha256"
            if lst.is_file():
                self.listing = {}
                for line in lst.read_text("utf-8").splitlines():
                    if line.strip():
                        sha, rel = line.split(None, 1)
                        self.listing[rel.strip()] = sha
        self.verified = self.listing is not None
        self.require_r4 = require_r4
        if dataset != "kitti15":
            calib = None
        elif calib == "auto":
            calib = r4mod().find_calib(self.root)
        self.calib = calib
        if require_r4 and dataset == "kitti15" and calib is None:
            raise IntegrityError(
                "require_r4: no KITTI-15 calibration found (set U2_K15_CALIB to the extracted "
                "u2f-meta-v1 dir or zip, or put it at <root>/kitti15/calib or "
                "<root>/../bundles/u2f-meta-v1.zip); refusing to let R4 fall into R4u")

    def rel(self, d: str, img_id: str) -> str:
        return f"{self.dataset}/training/{d}/{img_id}_10.png"

    def read_bytes(self, d: str, img_id: str) -> bytes:
        if self._zip is not None:
            return self._zip.read(f"{self.zip_prefix}training/{d}/{img_id}_10.png")
        rel = self.rel(d, img_id)
        data = (self.root / rel).read_bytes()
        if self.listing is not None:
            want = self.listing.get(rel)
            if want is None or sha256_bytes(data) != want:
                raise IntegrityError(f"{rel} does not match {self.bundle_dir}/FILES.sha256")
        return data

    def read(self, img_id: str, need_obj: bool = True) -> GTImage:
        fo, vo = decode_kitti_flow(self.read_bytes("flow_occ", img_id))
        fn, vn = decode_kitti_flow(self.read_bytes("flow_noc", img_id))
        gt = gt_from_arrays(fo, vo, fn, vn, None)
        if self.dataset == "kitti15" and need_obj:
            gt.obj_map = decode_obj_map(self.read_bytes("obj_map", img_id))
            r4 = r4mod()
            try:
                gt.disp_occ, gt.disp_valid = r4.decode_disp(self.read_bytes("disp_occ_0", img_id))
                if self.calib is None:
                    raise r4.CalibError("no calibration source (regions_r4.find_calib found none)")
                gt.calib = self.calib.read(img_id)
            except (KeyError, OSError, r4.CalibError) as e:
                if self.require_r4:
                    raise IntegrityError(f"{img_id}: the R4 split needs disp_occ_0 and calib: "
                                         f"{e}") from None
                gt.r4_why = f"{type(e).__name__}: {e}"
        return gt

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()


# ------------------------------------------------------------------------------ kernel
def epe_map(pred: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """(H, W) float32 end-point error of pred against flow, both (2, H, W) float32."""
    d = np.asarray(pred, np.float32) - np.asarray(flow, np.float32)
    return np.sqrt(d[0] * d[0] + d[1] * d[1])


def magnitude(flow: np.ndarray) -> np.ndarray:
    f = np.asarray(flow, np.float32)
    return np.sqrt(f[0] * f[0] + f[1] * f[1])


def outliers(epe: np.ndarray, mag: np.ndarray, eps: bool = True) -> np.ndarray:
    """KITTI-15 rule [V evaluate.py:552]; eps=False is KITTI-12's form [V :860]."""
    with np.errstate(divide="ignore", invalid="ignore"):
        den = mag + _EPS if eps else mag
        return (epe > _ABS) & ((epe / den) > _REL)


def in_frame(gt: GTImage, v_occ: np.ndarray) -> np.ndarray:
    """IN: the GT target lands inside frame 2; False wherever V_occ is False (masked first)."""
    h, w = gt.shape
    ys, xs = np.mgrid[0:h, 0:w]
    tx = xs.astype(np.float64) + gt.flow_occ[0].astype(np.float64)
    ty = ys.astype(np.float64) + gt.flow_occ[1].astype(np.float64)
    ok = (tx >= 0) & (tx <= w - 1) & (ty >= 0) & (ty <= h - 1)
    return ok & v_occ


def masks(gt: GTImage, dataset: str, only_all: bool = False) -> tuple[dict, dict]:
    """Region masks and per-image checks. Raises RegionError on S1, S2 or S3."""
    v_occ = gt.valid_occ >= 0.5
    if only_all:
        return {"all": v_occ}, {}
    v_noc = gt.valid_noc >= 0.5
    if np.any(v_noc & ~v_occ):
        raise RegionError("S2 failed: a flow_noc-valid pixel is not flow_occ-valid "
                          "(evaluate.py's occ selection would drop it; trap T5)")
    s3 = float(np.max(np.abs(gt.flow_occ[:, v_noc] - gt.flow_noc[:, v_noc]))) if v_noc.any() \
        else 0.0
    if s3 != 0.0:
        raise RegionError("S3 failed: flow_occ differs from flow_noc on V_noc (trap T7)")
    occ_only = v_occ & ~v_noc
    inside = in_frame(gt, v_occ)
    m = {"all": v_occ, "noc": v_noc, "occ": occ_only}
    if dataset == "kitti15":
        if gt.obj_map is None:
            raise RegionError("KITTI-15 regions need obj_map")
        if gt.obj_map.shape != gt.shape:
            raise RegionError("obj_map and flow GT differ in size")
        bg, fg = gt.obj_map == 0, gt.obj_map > 0
        mag = magnitude(gt.flow_occ)
        m.update(bg=v_occ & bg, fg=v_occ & fg, R1=v_noc & bg, R2=occ_only & bg & inside,
                 R3=occ_only & bg & ~inside, R4=v_occ & fg)
        m["R3-rel"] = m["R3"] & (mag <= R3_REL_MAX_PX)
        m["R2+R3"] = m["R2"] | m["R3"]
        parts = ("R1", "R2", "R3", "R4")
        r4 = r4mod()
        if gt.disp_occ is not None and gt.calib is not None:
            try:
                r4m, r4info = r4.split_r4(gt.flow_occ, v_occ, v_noc, gt.obj_map, gt.disp_occ,
                                          gt.disp_valid, gt.calib)   # asserts S7
            except r4.R4Error as e:
                raise RegionError(str(e)) from None
            m.update(r4m)
        else:
            m.update({k: np.zeros_like(v_occ) for k in r4.R4_REGIONS})
            m["R4u"] = m["R4"].copy()
            r4info = {"split": "unavailable", "why": gt.r4_why or "no disp_occ_0 or calib"}
    elif dataset == "kitti12":
        m.update({"K12-R1": v_noc, "K12-R2": occ_only & inside, "K12-R3": occ_only & ~inside})
        parts = ("K12-R1", "K12-R2", "K12-R3")
    else:
        raise ValueError(f"unknown dataset {dataset!r}")
    union = np.zeros_like(v_occ)
    total = 0
    for p in parts:
        if np.any(union & m[p]):
            raise RegionError(f"S1 failed: {p} overlaps an earlier region")
        union |= m[p]
        total += int(m[p].sum())
    if not np.array_equal(union, v_occ) or total != int(v_occ.sum()):
        raise RegionError("S1 failed: the regions do not partition V_occ")
    checks = {"S3_max_abs": s3, "S4_noc_not_in": int((v_noc & ~inside).sum())}
    if dataset == "kitti15":
        checks["R4"] = r4info
    return m, checks


def score_image(pred: np.ndarray, gt: GTImage, dataset: str, only_all: bool = False) -> dict:
    """Per-image counts. pred is the (2, H, W) float32 flow after resize_flow (never
    re-interpolated here). Returns {"r": {region: {"n", "out", "epe_sum"[, "out3"]}},
    "u2f_noc", "u2f_occ", "checks"}."""
    pred = np.asarray(pred, np.float32)
    if pred.shape != (2, *gt.shape):
        raise ValueError(f"prediction shape {pred.shape} != GT (2, {gt.shape[0]}, {gt.shape[1]})")
    region_masks, checks = masks(gt, dataset, only_all)
    epe = epe_map(pred, gt.flow_occ)
    mag = magnitude(gt.flow_occ)
    k15 = dataset == "kitti15"
    out = outliers(epe, mag, eps=k15)
    out3 = epe > _ABS
    r = {}
    for name, mk in region_masks.items():
        rec = {"n": int(mk.sum()), "out": int(out[mk].sum()),
               "epe_sum": float(epe[mk].sum(dtype=np.float64))}
        if not k15:
            rec["out3"] = int(out3[mk].sum())
        r[name] = rec
    res = {"r": r, "checks": checks}
    if only_all:
        return res
    v_noc = gt.valid_noc >= 0.5
    noc_err = epe_map(pred, gt.flow_noc)
    if v_noc.any():
        res["u2f_noc"] = float(noc_err[v_noc].mean(dtype=np.float64))
    else:
        res["u2f_noc"] = 0.0 if k15 else None   # K12 has no guard: NaN [V evaluate.py:867]
    diff = gt.valid_occ - gt.valid_noc
    if (diff == 0).all():
        res["u2f_occ"] = 0.0
    else:
        sel = diff >= 0.5
        res["u2f_occ"] = float(epe[sel].mean(dtype=np.float64)) if sel.any() else None
    return res


# ------------------------------------------------------------------------------ aggregation
def _nanmean(xs: list) -> float | None:
    vals = [math.nan if x is None else float(x) for x in xs]
    if not vals:
        return None
    m = math.fsum(vals) / len(vals)
    return None if math.isnan(m) else m


def _rate(out: int, n: int) -> float | None:
    return 100.0 * out / n if n else None


def region_summary(per_image: list[dict], region: str) -> dict:
    n = sum(p["r"][region]["n"] for p in per_image)
    out = sum(p["r"][region]["out"] for p in per_image)
    es = math.fsum(p["r"][region]["epe_sum"] for p in per_image)
    nz = [p["r"][region] for p in per_image if p["r"][region]["n"] > 0]
    rec = {"n_valid": n, "n_out": out, "epe_sum": es, "n_images": len(nz),
           "fl": _rate(out, n), "epe_pooled": es / n if n else None,
           "epe_img_mean": (math.fsum(x["epe_sum"] / x["n"] for x in nz) / len(nz)) if nz
           else None}
    if per_image and "out3" in per_image[0]["r"][region]:
        o3 = sum(p["r"][region]["out3"] for p in per_image)
        rec.update(n_out3=o3, out3=_rate(o3, n))
    return rec


def aggregate(per_image: list[dict], dataset: str, only_all: bool = False) -> dict:
    """U2Flow's split numbers plus the region table, from per-image counts only."""
    if not per_image:
        raise ValueError("no images to aggregate")
    names = ("all",) if only_all else (K15_REGIONS if dataset == "kitti15" else K12_REGIONS)
    regions = {r: region_summary(per_image, r) for r in names}
    a = regions["all"]
    u2f = {"epe": _nanmean([p["r"]["all"]["epe_sum"] / p["r"]["all"]["n"]
                            if p["r"]["all"]["n"] else None for p in per_image]),
           "fl": a["fl"]}
    if not only_all:
        u2f.update(noc=_nanmean([p.get("u2f_noc") for p in per_image]),
                   occ=_nanmean([p.get("u2f_occ") for p in per_image]),
                   fl_noc=regions["noc"]["fl"] or 0.0, fl_occ=regions["occ"]["fl"] or 0.0)
    return {"metric_version": METRIC_VERSION, "regions_version": REGIONS_VERSION,
            "fl_label": "Fl" if dataset == "kitti15" else "Fl(K15-rule)",
            "n_images": len(per_image), "n_pixels": a["n_valid"], "u2f": u2f,
            "regions": regions}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eval-sha", action="store_true", help="print the evaluator hash and files")
    a = ap.parse_args(argv)
    if a.eval_sha:
        sha, files = eval_sha()
        print(json.dumps({"eval_sha": sha, "files": files}, indent=1, sort_keys=True))
        return 0
    ap.print_help()
    return 64


if __name__ == "__main__":
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    sys.exit(main())
