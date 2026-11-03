#!/usr/bin/env python3
"""arbitrate.py v1: the hierarchical, out-of-fold SAM-tree gate that decides which pixels a rigid
teacher may supervise (plan row W3-01; proposal final.architecture.arbitration_rule A1-A8), and
the comparison variants that bound it. CPU, numpy, library plus a small CLI.

What it does (one direction of one frame; every array is on the training grid)
    load_records / build_tree   SAM's records (the --rle JSON of tools/vm/sam_masks.py: uncompressed
                                COCO RLE + area per mask) -> a containment tree [A1]: a mask's
                                parent is the smallest larger mask holding >= 90 % of it; pixels no
                                mask covers become 8-connected components cut on a 32 x 32 grid;
                                a full-frame mask is dropped (UnSAMFlow's full_seg does the same)
    build_pair_evidence         the OUT-OF-FOLD pose [A3, A6]: leaf segments are split into two
                                seeded halves A and B (refine_pose.split_halves); the rigid flow of
                                every pixel of A is rendered with a pose fitted by PnP on evidence
                                pixels of B only, and vice versa (refine_pose.pnp reused). Out of
                                fold PER PIXEL: a pixel's residual never uses a pose fitted on its
                                own leaf's half. A smallest mask decides on its own pixels only,
                                so it is judged with a pose it did not shape; an ANCESTOR's
                                n_s, a_s, m_s (veto, deferral) pool the pixels of both folds, as
                                proposal A6 specifies, so they include pixels judged by a pose
                                fitted on the leaf being decided (a property of the rule, a
                                limitation for the G1 memo). A fold pose must pass the A3 numbers
                                for T_ref (ratio >= 0.5, >= 300 inliers; [E] here); a failed
                                fold gets no rigid flow and its pixels are WITHHELD: gate() zeroes
                                the weight of every pixel that has no rigid flow in some pair (a
                                node's verdict can reach such a pixel by deferral to a small
                                parent, whose statistics never counted it), and a withheld pixel
                                counts as uncovered in every coverage/leakage count. It is
                                reported in PairEvidence.info, GateResult.fold_failures and
                                the CLI. Known gap: a fold whose RANSAC consensus is a dominant
                                mover can pass these checks and renders the mover as rigid; the
                                two folds' poses then disagree, which raises the WARNING
                                fold_poses_disagree (GateResult.fold_warnings, CLI) but gates
                                nothing (tests/test_arbitrate.py pins the gap); the CLI prints
                                the warning count beside the figures, and a consumer (W3-03) must
                                gate on it as A3 does for T_ref, or report the count likewise
    gate                        per node: n_s, a_s, m_s pooled over the pairs [A5, A6] -> RIGID /
                                MOVER / UNDECIDED [A7]; the hierarchy rule [A7]: a pixel is taught
                                only when its smallest mask is RIGID and no ancestor is MOVER or
                                has > 50 % of its area in MOVER children; a smallest mask short of
                                evidence defers to its smallest ancestor with evidence and area
                                <= 25 % of the image; R1 (recount on PairEvidence.ev_relaxed, a_s
                                >= 0.85: implemented and tested, but NOTHING builds ev_relaxed yet,
                                it needs the evidence network's sigma^2 <= 4 mask from the real
                                caches), R2 (one pair only: a_s >= 0.85), R3 (outlier
                                carve-out, 5 px) and R4 (UNDECIDED weighs 0) of A7; then A8 (weight
                                min(1, n_s/200) x pair_w, zero within 3 px of a non-RIGID segment);
                                and A3's third pair_w rule (the road segment must be RIGID),
                                which refine_pose.py records as 'pending: arbitrate.py'
    variants                    none (w = valid_T) · per-pixel with static default · per-pixel with
                                reject default · gate · GT oracle (audit only; the caller passes the
                                moving mask, nothing here reads GT). Variants (a)-(e) of the
                                proposal (object-class gates, 4D confidence, Sampson, superpixels)
                                are NOT here
    regions / coverage_leakage  R1, R2, R3, R4m masks -> counts (never rates; statistics.md):
                                n and k on R2+R3 (coverage) and on R4m (leakage). The masks of a
                                KITTI-15 GT image come from tools/eval/regions.py (regions_from_gt),
                                so the definitions are the evaluator's own. R4m exists only in the
                                staged regions_v2 (tools/staging/eval/regions_r4.py): until it lands
                                the caller passes r4m, or the report says 'R4' and means every FG
                                pixel
    synthetic_frames            the CPU world of the tests: tools/p1/p1lib/synth.py's analytic scene
                                (ground, wall, box) seen by the moving camera, plus one
                                independently moving patch; exact GT and a hand-written SAM tree
                                per frame

How to invoke (Mac, .venv/bin/python)
    tools/p1/arbitrate.py synth [--frames 10] [--seed 0] [--json OUT]
        10 synthetic frames, all variants, coverage of R2+R3 and leakage onto R4m (pooled counts);
        prints 'SYNTHETIC WORLD: test design, not evidence' first and last, the R4m source and
        every failed fold; exit 1 when the oracle leaks (a broken audit), else 0
    tools/p1/arbitrate.py tree --json RECORDS.json[.gz] --hw 256 832
        the tree of one frame's SAM records (a path under any sealed directory is refused)
    Library: build_tree, build_pair_evidence, gate, variants_for_frame, coverage_leakage.
Exit codes: 0 ok · 1 an audit check failed · 2 a precondition (missing, truncated or malformed
input, sealed path) · 64. Library calls with no evidence pair raise ValueError.

Why it is built this way
- Per-pixel gates cannot be right on both axes at once: R2 and R3 pixels (occluded, out of frame)
  carry no visible evidence, so a per-pixel rule must guess; 'static' leaks onto movers and
  'reject' covers nothing there, by construction [P proposal]. A segment decision is made on
  pixels that DO have evidence and then applies to the whole segment, blind pixels included.
- The hierarchy exists because SAM merges: a parent that is mostly a mover vetoes the child that
  looks rigid by chance (a car's window); an evidence-poor smallest mask borrows its parent's
  decision only while the parent is small enough not to be 'the whole road'.
- Out of fold, because a segment that shaped the pose confirms itself (proposal C2).
- Counts, not rates: every pooled number is recomputed from n and k.

Constants. Every threshold of the rule is the proposal's own [P final.architecture.arbitration_rule,
prereg/inputs/proposal-full.json], pre-specified and not tuned here; the SOTA check at their exact
values belongs to the method skill before GATE-1.json freezes them (as for rigid.py). The values
marked [E] are implementation decisions of this file where the proposal is silent.
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import sys
import zlib
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import refine_pose as RP  # noqa: E402
import rigid  # noqa: E402

UNDECIDED, RIGID, MOVER = 0, 1, 2
LABEL_NAMES = {UNDECIDED: "UNDECIDED", RIGID: "RIGID", MOVER: "MOVER"}

# ---- the proposal's rule (A1-A8), KITTI values; Sintel values are not used here ----
MIN_SEG_PX = 64            # [P A1] a smaller segment is UNDECIDED
PARENT_CONTAIN = 0.90      # [P A1] P is the parent of s when >= 90 % of s lies in P
GRID_CELL_PX = 32          # [P A1] cut of the uncovered (id 0) components
MIN_EVIDENCE = 40          # [P A7] n_s floor for RIGID and for MOVER
RIGID_MIN_AREA_FRAC = 0.10  # [P A7] n_s >= max(40, 0.1 |s|)
A_RIGID = 0.80             # [P A7] a_s for RIGID
A_RIGID_HI = 0.85          # [P A7 R1, R2] a_s when evidence is relaxed or one pair is missing
M_RIGID_PX = 1.0           # [P A7] m_s (median residual) for RIGID
A_MOVER = 0.50             # [P A7] a_s <= 0.5 with n_s >= 40 is MOVER
INLIER_ABS_PX = 2.0        # [P A5] r <= max(2.0 px, 0.05 ||F_rig||)
INLIER_REL = 0.05          # [P A5]
PARENT_MAX_AREA_FRAC = 0.25  # [P A7] a deferred decision is taken at an ancestor <= 25 % of image
MOVER_CHILD_FRAC = 0.50    # [P A7] a parent with more than half of its area in MOVER children
CARVE_MIN_PX = 20          # [P A7 R3] outlier components of at least 20 px ...
CARVE_DILATE_PX = 5        # [P A7 R3] ... dilated by 5 px
N_SAT = 200                # [P A8] w_seg = min(1, n_s / 200) ...
BORDER_PX = 3              # [P A8] ... zero within 3 px of a non-RIGID segment
# A fold's pose must pass the same two checks the proposal states for T_ref (A3: "PnP has an inlier
# ratio < 0.5 or fewer than 300 inliers" gives pair_w = 0). The proposal does not state them for
# the fold poses: applying them there is an [E] choice, taken from refine_pose.PAIR_W so that the
# two tools cannot drift apart.
FOLD_MIN_INLIER_RATIO = RP.PAIR_W["min_inlier_ratio"]  # [E] from [P A3], 0.5
FOLD_MIN_INLIERS = RP.PAIR_W["min_inliers"]            # [E] from [P A3], 300
# [E] the proposal says "dilate by 5 px" and "within 3 px" without a metric: Euclidean, in pixels
# [E] the proposal does not say how a parent's 'child area' is counted: the union of its direct
#     children's masks that are MOVER, inside the parent
# [E] a segment under 64 px counts as 'short of evidence' for the defer-to-parent rule


@dataclass
class Params:
    """The rule's thresholds; the defaults are the proposal's. Changing one is a deviation and the
    caller records it (as refine_pose.py does for pair_w)."""

    min_seg_px: int = MIN_SEG_PX
    min_evidence: int = MIN_EVIDENCE
    rigid_min_area_frac: float = RIGID_MIN_AREA_FRAC
    a_rigid: float = A_RIGID
    a_rigid_hi: float = A_RIGID_HI
    m_rigid_px: float = M_RIGID_PX
    a_mover: float = A_MOVER
    inlier_abs_px: float = INLIER_ABS_PX
    inlier_rel: float = INLIER_REL
    parent_max_area_frac: float = PARENT_MAX_AREA_FRAC
    mover_child_frac: float = MOVER_CHILD_FRAC
    carve_min_px: int = CARVE_MIN_PX
    carve_dilate_px: int = CARVE_DILATE_PX
    n_sat: int = N_SAT
    border_px: int = BORDER_PX
    fold_min_inlier_ratio: float = FOLD_MIN_INLIER_RATIO
    fold_min_inliers: int = FOLD_MIN_INLIERS


# ------------------------------------------------------------------------------ SAM records
def _sam_masks():
    """tools/vm/sam_masks.py (the producer; rle_decode lives there, no second copy here)."""
    spec = importlib.util.spec_from_file_location("sam_masks", REPO / "tools" / "vm" /
                                                  "sam_masks.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("sam_masks", mod)
    spec.loader.exec_module(mod)
    return mod


def refuse_sealed(path) -> None:
    if "sealed" in Path(path).resolve().parts:
        raise PermissionError(f"{path}: under a sealed directory (AGENTS.md hard rule 4)")


def load_records(path: Path | str) -> list[dict]:
    """SAM's records of one frame: a JSON list (or .json.gz) of {segmentation: uncompressed RLE,
    area, ...} as sam_masks.py --rle writes them."""
    refuse_sealed(path)
    p = Path(path)
    try:
        raw = gzip.open(p, "rb").read() if p.suffix == ".gz" else p.read_bytes()
    except (EOFError, gzip.BadGzipFile, zlib.error) as e:  # a truncated or corrupt .gz
        raise ValueError(f"{path}: unreadable archive ({type(e).__name__}: {e})") from e
    recs = json.loads(raw)
    if not isinstance(recs, list):
        raise ValueError(f"{path}: expected a JSON list of SAM records")
    check_records(recs, str(path))
    return recs


def check_records(records: list, where: str = "records") -> None:
    """Every record needs an uncompressed RLE 'segmentation' ({size, counts}); a clear ValueError
    instead of a KeyError deep in the decoder."""
    for i, r in enumerate(records):
        seg = r.get("segmentation") if isinstance(r, dict) else None
        if not isinstance(seg, dict) or "size" not in seg or "counts" not in seg:
            raise ValueError(f"{where}: record {i} has no uncompressed-RLE 'segmentation' "
                             "(size, counts)")


def decode_records(records: list[dict], grid_hw: tuple[int, int]) -> list[np.ndarray]:
    """Bool masks on the grid, in SAM's order. A mask as large as the native frame is dropped, as
    UnSAMFlow's full_seg does (sam_masks.full_seg paints a label map and drops it internally, so
    it cannot be called as a filter; tests/test_arbitrate.py checks that our smallest-mask map and
    full_seg's partition agree); masks are resized with INTER_NEAREST, as sep_transforms.Zoom
    does."""
    check_records(records)
    sm = _sam_masks()
    H, W = grid_hw
    out = []
    for r in records:
        seg = r["segmentation"]
        h, w = seg["size"]
        m = sm.rle_decode(seg)
        if int(m.sum()) >= h * w:
            continue
        if (h, w) != (H, W):
            m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        if m.any():
            out.append(m)
    return out


# ------------------------------------------------------------------------------ the tree [A1]
@dataclass
class Tree:
    hw: tuple[int, int]
    masks: np.ndarray    # (N, H, W) bool, ascending area; pseudo-segments last
    area: np.ndarray     # (N,) int, pixels on the grid
    parent: np.ndarray   # (N,) int, -1 for a root
    pseudo: np.ndarray   # (N,) bool: a cell of an uncovered component
    leaf: np.ndarray     # (H, W) int32: the smallest node containing the pixel

    @property
    def n(self) -> int:
        return int(self.area.size)

    def ancestors(self, i: int) -> list[int]:
        out, a = [], int(self.parent[i])
        while a >= 0:
            out.append(a)
            a = int(self.parent[a])
        return out

    def children(self, i: int) -> list[int]:
        return [int(c) for c in np.flatnonzero(self.parent == i)]


def build_tree_from_masks(masks: list[np.ndarray], hw: tuple[int, int],
                          contain: float = PARENT_CONTAIN, cell: int = GRID_CELL_PX) -> Tree:
    H, W = hw
    ms = [np.asarray(m, bool) for m in masks]
    areas = np.array([int(m.sum()) for m in ms], np.int64)
    order = np.argsort(areas, kind="stable")  # ascending; equal areas keep SAM order
    ms = [ms[i] for i in order]
    areas = areas[order]
    n = len(ms)
    parent = np.full(n, -1, np.int64)
    if n:
        flat = np.stack([m.reshape(-1) for m in ms]).astype(np.float32)
        inter = flat @ flat.T
        for i in range(n):
            cand = np.flatnonzero(inter[i, i + 1:] >= contain * areas[i])
            if cand.size:
                parent[i] = i + 1 + int(cand[0])
    leaf = np.full((H, W), -1, np.int32)
    for i in reversed(range(n)):  # largest first, so the smallest mask is painted last
        leaf[ms[i]] = i
    unc = leaf < 0
    pseudo_masks = []
    if unc.any():
        _, cc = cv2.connectedComponents(unc.astype(np.uint8), connectivity=8)
        ys, xs = np.mgrid[0:H, 0:W]
        key = (cc.astype(np.int64) * 1000 + (ys // cell)) * 1000 + (xs // cell)
        _, inv = np.unique(key[unc], return_inverse=True)
        ids = np.full((H, W), -1, np.int64)
        ids[unc] = inv
        for k in range(int(inv.max()) + 1):
            pseudo_masks.append(ids == k)
            leaf[ids == k] = n + k
    allm = ms + pseudo_masks
    area = np.array([int(m.sum()) for m in allm], np.int64)
    pseudo = np.array([False] * n + [True] * len(pseudo_masks))
    par = np.concatenate([parent, np.full(len(pseudo_masks), -1, np.int64)])
    stack = np.stack(allm) if allm else np.zeros((0, H, W), bool)
    return Tree((H, W), stack, area, par, pseudo, leaf)


def build_tree(records: list[dict], grid_hw: tuple[int, int]) -> Tree:
    return build_tree_from_masks(decode_records(records, grid_hw), grid_hw)


# ------------------------------------------------------------------------------ pair evidence
@dataclass
class PairEvidence:
    """One pair's evidence for the frame being gated: flow_e (2, H, W) the evidence network's
    flow, ev (H, W) bool E^P [A4: low-alpha, visible, valid_T, not OOF, not O_T; built by the
    caller], rig (2, H, W) the rigid flow rendered with the OUT-OF-FOLD pose (NaN where the fold's
    pose failed). ev_relaxed is R1's looser E^P (sigma^2 <= 4), optional."""

    flow_e: np.ndarray
    ev: np.ndarray
    rig: np.ndarray
    ev_relaxed: np.ndarray | None = None
    name: str = ""
    info: dict = field(default_factory=dict)


def oof_rigid_flows(depth, K, flow_e, L, labels, seed: int,
                    min_ratio: float = FOLD_MIN_INLIER_RATIO,
                    min_inliers: int = FOLD_MIN_INLIERS) -> tuple[np.ndarray, dict]:
    """Rigid flow of every pixel under an out-of-fold pose. depth (H, W) z-depth (NaN invalid), K
    the window's one K, flow_e (2, H, W), L (H, W) bool the pixels PnP may use, labels (H, W)
    int > 0 the leaf segments. Segments are split in two halves A, B (refine_pose.split_halves);
    pixels of A get the pose fitted on L outside A (that is, in B) and pixels of B the pose fitted
    on L in A.

    A fold pose is used only if PnP returns one with an inlier ratio >= min_ratio and >=
    min_inliers inliers ([E] the A3 numbers for T_ref). Otherwise the fold is left NaN (no
    evidence there: its nodes become UNDECIDED, nothing is taught from it: fail closed) and the
    failure is recorded, never silent: info['folds'][name] has ok False and 'failed' (the reason),
    and info['failed_folds'] lists them for GateResult and the CLI. info['pose_deviation'] is the
    rotation and translation-direction deviation of the two folds' poses (when both exist); it is
    a diagnostic, not a gate (see the header: the dominant-mover case)."""
    H, W = depth.shape
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    tx, ty = xs + flow_e[0], ys + flow_e[1]
    Lm = np.asarray(L, bool) & np.isfinite(depth) & (depth > 0) & np.isfinite(tx) & np.isfinite(ty)
    A, _B = RP.split_halves(labels, seed)
    in_a = np.isin(labels, sorted(A))
    X = np.moveaxis(rigid.backproject(np.where(Lm, depth, np.nan), K), 0, -1).reshape(-1, 3)
    x2 = np.stack([tx.reshape(-1), ty.reshape(-1)], 1)
    rig = np.full((2, H, W), np.nan)
    info = {"folds": {}, "failed_folds": []}
    poses = {}
    for name, fold in (("A", in_a), ("B", ~in_a)):
        fit = np.flatnonzero((Lm & ~fold).reshape(-1))  # outside the evaluated fold
        rng = np.random.default_rng(seed + (1 if name == "A" else 2))
        pick = RP.stratified_sample(fit, labels.reshape(-1)[fit], RP.MAX_SAMPLES, rng)
        res = RP.pnp(X[pick], x2[pick], K, seed)
        rec = {"n_pixels": int(fold.sum()), "n_fit": int(fit.size)}
        info["folds"][name] = rec
        if res is None:
            reason = "pnp failed"
        else:
            rec.update(n_in=res["n_in"], n=res["n"], rmse_px=res["rmse_px"], M=res["M"])
            poses[name] = res["M"]
            ratio = res["n_in"] / res["n"] if res["n"] else 0.0
            reason = None
            if ratio < min_ratio:
                reason = f"inlier ratio {ratio:.3f} < {min_ratio:g}"
            elif res["n_in"] < min_inliers:
                reason = f"{res['n_in']} inliers < {min_inliers:g}"
        rec["ok"] = reason is None
        rec["failed"] = reason
        if reason is not None:
            info["failed_folds"].append({"fold": name, "reason": reason})
            continue
        flow = rigid.rigid_flow(depth, K, res["M"]).flow
        rig[:, fold] = flow[:, fold]
    info["fold_poses_disagree"] = False
    if len(poses) == 2:
        dev = rigid.pose_deviation(poses["A"], poses["B"])
        info["pose_deviation"] = dev
        # a warning, not a gate: two background fits agree (synthetic frames: <= 0.16 deg and
        # <= 1.7 deg [M]); a dominant mover in one fold's consensus pulls them far apart. The A3
        # numbers for T_ref against T_4D are the yardstick [E]
        info["fold_poses_disagree"] = bool(
            dev["rot_deg"] > RP.PAIR_W["max_rot_deg"]
            or (dev["trans_dir_deg"] is not None
                and dev["trans_dir_deg"] > RP.PAIR_W["max_tdir_deg"]))
    return rig, info


def build_pair_evidence(depth, K, flow_e, ev, tree: Tree, seed: int, ev_relaxed=None,
                        name: str = "", p: Params | None = None) -> PairEvidence:
    """PairEvidence of one pair with the out-of-fold pose; folds follow the tree's leaf nodes."""
    p = p or Params()
    labels = tree.leaf.astype(np.int64) + 1  # split_halves reads labels > 0
    rig, info = oof_rigid_flows(np.asarray(depth, np.float64), np.asarray(K, np.float64),
                                np.asarray(flow_e, np.float64), ev, labels, seed,
                                p.fold_min_inlier_ratio, p.fold_min_inliers)
    return PairEvidence(np.asarray(flow_e, np.float64), np.asarray(ev, bool), rig,
                        None if ev_relaxed is None else np.asarray(ev_relaxed, bool), name, info)


def residuals(pe: PairEvidence, p: Params, relaxed: bool = False):
    """r = ||F_E - F_rig||, the evidence mask and the inlier mask [A5]."""
    with np.errstate(invalid="ignore"):
        d = pe.flow_e - pe.rig
        r = np.sqrt(d[0] ** 2 + d[1] ** 2)
        mag = np.sqrt(pe.rig[0] ** 2 + pe.rig[1] ** 2)
        thr = np.maximum(p.inlier_abs_px, p.inlier_rel * mag)
        base = pe.ev_relaxed if (relaxed and pe.ev_relaxed is not None) else pe.ev
        ok = base & np.isfinite(r)
        inl = ok & (r <= thr)
    return r, ok, inl


# ------------------------------------------------------------------------------ statistics [A6]
def exclusive_area(tree: Tree) -> np.ndarray:
    """Pixels whose smallest mask is the node: the node's own full_seg segment."""
    return np.bincount(tree.leaf.reshape(-1), minlength=tree.n).astype(np.int64)


def node_stats(tree: Tree, pairs: list[PairEvidence], p: Params, relaxed: bool = False,
               exclusive: bool = False) -> dict:
    """n_s, inlier count, a_s, m_s of every node, pooled over the pairs. exclusive=False counts the
    pixels of the node's whole mask (what an ancestor's decision rests on); exclusive=True counts
    only the pixels whose smallest mask is the node (what the node's own decision as a smallest
    mask rests on: a mask that holds a mover must not lose its own rigid pixels to it [E, the
    proposal reads 'the smallest mask'; the full_seg segment is that mask's own region]). a_s and
    m_s are NaN where n_s = 0."""
    N = tree.n
    mf = None if exclusive else tree.masks.reshape(N, -1)
    n = np.zeros(N, np.int64)
    k = np.zeros(N, np.int64)
    vals: list[list[np.ndarray]] = [[] for _ in range(N)]
    leaf = tree.leaf.reshape(-1)
    for pe in pairs:
        r, ok, inl = residuals(pe, p, relaxed)
        idx = np.flatnonzero(ok.reshape(-1))
        if idx.size == 0:
            continue
        rf, inf = r.reshape(-1)[idx], inl.reshape(-1)[idx]
        if exclusive:
            lf = leaf[idx]
            n += np.bincount(lf, minlength=N)
            k += np.bincount(lf[inf], minlength=N)
            for i in np.unique(lf):
                vals[i].append(rf[lf == i])
        else:
            sub = mf[:, idx]
            n += sub.sum(1)
            k += (sub & inf).sum(1)
            for i in range(N):
                if sub[i].any():
                    vals[i].append(rf[sub[i]])
    m = np.array([float(np.median(np.concatenate(v))) if v else np.nan for v in vals])
    a = np.where(n > 0, k / np.maximum(n, 1), np.nan)
    return {"n": n, "k": k, "a": a, "m": m}


def decide(tree: Tree, st: dict, p: Params, n_pairs: int, st_relaxed: dict | None = None,
           area: np.ndarray | None = None):
    """RIGID / MOVER / UNDECIDED of every node [A7], whether it is short of evidence, and the
    n_s used. area: |s| (default the mask area; the exclusive area for a smallest mask). One pair
    only (R2): RIGID needs a_s >= 0.85. R1: a node short of evidence at the strict mask is
    recounted on the relaxed one and needs a_s >= 0.85. A segment under 64 px is UNDECIDED [A1]."""
    N = tree.n
    area = tree.area if area is None else area
    label = np.zeros(N, np.int8)
    n, a, m = st["n"].copy(), st["a"].copy(), st["m"].copy()
    a_need = np.full(N, p.a_rigid if n_pairs >= 2 else p.a_rigid_hi)
    need_n = np.maximum(p.min_evidence, p.rigid_min_area_frac * area)
    if st_relaxed is not None:  # R1
        redo = (n < need_n) & (st_relaxed["n"] > n)
        n = np.where(redo, st_relaxed["n"], n)
        a = np.where(redo, st_relaxed["a"], a)
        m = np.where(redo, st_relaxed["m"], m)
        a_need = np.where(redo, np.maximum(a_need, p.a_rigid_hi), a_need)
    with np.errstate(invalid="ignore"):
        is_mover = (n >= p.min_evidence) & (a <= p.a_mover)
        is_rigid = (n >= need_n) & (a >= a_need) & (m <= p.m_rigid_px)
    big = area >= p.min_seg_px
    label[is_rigid & big & ~is_mover] = RIGID
    label[is_mover & big] = MOVER
    short = ~big | ((label == UNDECIDED) & (n < need_n))
    return label, short, n


def mover_child_fraction(tree: Tree, label: np.ndarray) -> np.ndarray:
    """Share of each node's area covered by its direct children that are MOVER [A7, E]."""
    out = np.zeros(tree.n)
    for i in range(tree.n):
        kids = [c for c in tree.children(i) if label[c] == MOVER]
        if kids:
            u = np.any(tree.masks[kids], axis=0) & tree.masks[i]
            out[i] = u.sum() / max(1, tree.area[i])
    return out


@dataclass
class Decisions:
    """The two readings of every node. full: the whole mask (an ancestor's verdict); own: the
    node's exclusive pixels (its verdict as a smallest mask)."""

    label_full: np.ndarray
    short_full: np.ndarray
    n_full: np.ndarray
    label_own: np.ndarray
    short_own: np.ndarray
    n_own: np.ndarray
    stats_full: dict
    stats_own: dict


def decisions(tree: Tree, pairs: list[PairEvidence], p: Params) -> Decisions:
    relaxed = any(pe.ev_relaxed is not None for pe in pairs)
    out = []
    for excl in (False, True):
        st = node_stats(tree, pairs, p, exclusive=excl)
        rl = node_stats(tree, pairs, p, relaxed=True, exclusive=excl) if relaxed else None
        area = exclusive_area(tree) if excl else None
        out.append((st, *decide(tree, st, p, len(pairs), rl, area)))
    (sf, lf, shf, nf), (so, lo, sho, no) = out
    return Decisions(lf, shf, nf, lo, sho, no, sf, so)


def resolve(tree: Tree, d: Decisions, p: Params):
    """The decision node and the verdict of every node taken as a smallest mask [A7 hierarchy].
    A smallest mask with evidence decides on its own pixels; one short of evidence takes the
    verdict of its smallest ancestor with evidence and area <= 25 % of the image. The pixel is
    vetoed when any ancestor is MOVER or has more than half of its area in MOVER children.
    Returns (decision_node, eff_label, vetoed, accept, w_node), w_node = min(1, n_s/200) (A8,
    before pair_w and the border zeroing)."""
    N = tree.n
    H, W = tree.hw
    mov = mover_child_fraction(tree, d.label_full)
    dec = np.full(N, -1, np.int64)
    eff = np.zeros(N, np.int8)
    veto = np.zeros(N, bool)
    n_dec = np.zeros(N, np.int64)
    for s in range(N):
        chain = tree.ancestors(s)
        if not d.short_own[s]:
            eff[s], n_dec[s], dec[s] = d.label_own[s], d.n_own[s], s
        else:
            for a in chain:
                if tree.area[a] > p.parent_max_area_frac * H * W:
                    break
                if not d.short_full[a]:
                    eff[s], n_dec[s], dec[s] = d.label_full[a], d.n_full[a], a
                    break
        veto[s] = any(d.label_full[a] == MOVER or mov[a] > p.mover_child_frac for a in chain)
    accept = (eff == RIGID) & ~veto
    w_node = np.where(accept, np.minimum(1.0, n_dec / p.n_sat), 0.0)
    return dec, eff, veto, accept, w_node


def need_pairs(pairs) -> None:
    if not pairs:
        raise ValueError("arbitrate: no evidence pairs (a frame needs at least one pair)")


# ------------------------------------------------------------------------------ the gate [A7, A8]
@dataclass
class GateResult:
    w: np.ndarray            # (H, W) float32 weight, A8 (no valid_T applied)
    accept_px: np.ndarray    # (H, W) bool, w > 0
    label: np.ndarray        # per node
    short: np.ndarray
    decision_node: np.ndarray
    vetoed: np.ndarray
    accept: np.ndarray
    n: np.ndarray
    a: np.ndarray
    m: np.ndarray
    pair_w: float
    pair_w_eff: float
    road: dict
    carved_px: int
    eff_label: np.ndarray    # per node as a smallest mask: the verdict it takes (before the veto)
    decisions: Decisions
    fold_failures: list      # [{pair, fold, reason}]: a fold pose that failed, its pixels got no
    #                          rigid flow (fail closed); must be reported wherever a result is
    fold_warnings: list      # [{pair, warning, rot_deg, ...}]: the two fold poses disagree
    no_pose_px: int          # pixels withheld because some pair gave them no rigid flow


def carve_outliers(accepted: np.ndarray, pairs: list[PairEvidence], p: Params) -> np.ndarray:
    """R3: inside accepted (RIGID) pixels, 8-connected components of outlier evidence pixels of at
    least 20 px, dilated by 5 px; True where a pixel is carved out."""
    bad = np.zeros(accepted.shape, bool)
    for pe in pairs:
        _, ok, inl = residuals(pe, p)
        bad |= ok & ~inl
    bad &= accepted
    if not bad.any():
        return np.zeros_like(bad)
    cnt, cc, stats, _ = cv2.connectedComponentsWithStats(bad.astype(np.uint8), connectivity=8)
    keep = np.zeros(bad.shape, np.uint8)
    for c in range(1, cnt):
        if stats[c, cv2.CC_STAT_AREA] >= p.carve_min_px:
            keep[cc == c] = 1
    if not keep.any():
        return np.zeros_like(bad)
    k = 2 * p.carve_dilate_px + 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(keep, ker) > 0


def road_node(tree: Tree) -> int:
    """The largest segment touching the bottom border [A3 third rule]: the road. 'Segment' is the
    full_seg region (a node's exclusive pixels)."""
    bottom = np.unique(tree.leaf[-1])
    return int(bottom[np.argmax(exclusive_area(tree)[bottom])]) if bottom.size else -1


def gate(tree: Tree, pairs: list[PairEvidence], pair_w: float = 1.0, p: Params | None = None,
         road_rule: bool = True) -> GateResult:
    """The hierarchical gate of one direction of one frame; see the module header. pair_w is the
    direction's A3 weight (refine_pose.pair_weight); with road_rule, a road segment that is not
    accepted sets it to 0 (A3's third rule)."""
    p = p or Params()
    need_pairs(pairs)
    d = decisions(tree, pairs, p)
    dec, eff, veto, accept, w_node = resolve(tree, d, p)
    label, short, n = d.label_full, d.short_full, d.n_full
    road = road_node(tree)
    road_ok = bool(accept[road]) if road >= 0 else False
    pw = float(pair_w) if (road_ok or not road_rule) else 0.0
    wpx = (w_node[tree.leaf] * pw).astype(np.float32)
    carved = carve_outliers(wpx > 0, pairs, p)
    # a pixel with no rigid flow in some pair (its fold's pose failed, or its depth is invalid)
    # never gets weight: deferral could otherwise hand it a rigid parent's verdict on evidence
    # that never included it (a mover in a failed fold under a small rigid-looking parent)
    no_pose = np.zeros(wpx.shape, bool)
    for pe in pairs:
        no_pose |= ~np.isfinite(pe.rig).all(axis=0)
    # A8's 3 px rule is about NON-RIGID segments: it is measured from the pixels no segment
    # decision accepts AND the withheld (no-pose) pixels (review minor 15: a pixel next to a
    # withheld one has no verified neighbour), before the carve-out, so a carved blob widens by
    # R3's 5 px and no more
    rejected = (wpx <= 0) | no_pose
    near = np.zeros_like(rejected)
    if rejected.any():
        dist = cv2.distanceTransform((~rejected).astype(np.uint8), cv2.DIST_L2,
                                     cv2.DIST_MASK_PRECISE)
        near = dist <= p.border_px
    wpx[carved | near | no_pose] = 0.0
    folds = [{"pair": pe.name, **f} for pe in pairs for f in pe.info.get("failed_folds", [])]
    warns = [{"pair": pe.name, "warning": "fold poses disagree (dominant-mover suspect)",
              **pe.info["pose_deviation"]} for pe in pairs
             if pe.info.get("fold_poses_disagree")]
    return GateResult(wpx, wpx > 0, label, short, dec, veto, accept, n, d.stats_full["a"],
                      d.stats_full["m"], float(pair_w), pw, {"node": road, "accepted": road_ok},
                      int(carved.sum()), eff, d, folds, warns, int(no_pose.sum()))


# ------------------------------------------------------------------------------ the variants
def pixel_inliers(pairs: list[PairEvidence], p: Params | None = None):
    """Per pixel over the pairs: has evidence in any pair, and an outlier in any pair where it
    has evidence [E: conservative; the proposal gives the single-pair rule only]."""
    p = p or Params()
    need_pairs(pairs)
    has = np.zeros(pairs[0].ev.shape, bool)
    bad = np.zeros_like(has)
    for pe in pairs:
        _, ok, inl = residuals(pe, p)
        has |= ok
        bad |= ok & ~inl
    return has, bad


def variants_for_frame(tree: Tree, pairs: list[PairEvidence], valid_T: np.ndarray,
                       gt_moving: np.ndarray | None = None, pair_w: float = 1.0,
                       p: Params | None = None) -> dict:
    """Every variant on one frame, each ANDed with valid_T so coverage is of usable teacher
    pixels: {name: {"sel": bool mask, "w": float weight}}. gt_moving (the oracle's input, audit
    only) is optional."""
    p = p or Params()
    need_pairs(pairs)
    vt = np.asarray(valid_T, bool)
    # a per-pixel rule judges the evaluated pair alone (pairs[0], the forward pair): pooling a
    # second pair would hand a pixel that is blind in this pair evidence from the other one
    has, bad = pixel_inliers(pairs[:1], p)
    g = gate(tree, pairs, pair_w, p)
    out = {
        "none": vt,
        "per_pixel_static": vt & ~bad,           # blind pixels default to static
        "per_pixel_reject": vt & has & ~bad,     # blind pixels default to rejected
        "gate": vt & g.accept_px,
    }
    res = {k: {"sel": v, "w": v.astype(np.float32)} for k, v in out.items()}
    res["gate"]["w"] = np.where(out["gate"], g.w, 0.0).astype(np.float32)
    if gt_moving is not None:
        o = vt & ~np.asarray(gt_moving, bool)
        res["oracle"] = {"sel": o, "w": o.astype(np.float32)}
    res["_gate"] = g
    return res


# ------------------------------------------------------------------------------ regions, report
def _regions_module():
    spec = importlib.util.spec_from_file_location("u2f_regions", REPO / "tools" / "eval" /
                                                  "regions.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("u2f_regions", mod)
    spec.loader.exec_module(mod)
    return mod


def regions_from_gt(gt, grid_hw: tuple[int, int] | None = None, r4m=None,
                    r4m_source: str | None = None) -> dict:
    """R1, R2, R3 and R4m masks of a KITTI-15 GTImage by tools/eval/regions.py's own definitions
    (regions.masks; read only, never edited). r4m: the R4m mask when regions_v2 supplies it; when
    None the key 'R4m' holds R4 (every FG pixel) and 'r4m_source' says so. Masks are resized to
    grid_hw with nearest-neighbour when it differs from the GT size."""
    rg = _regions_module()
    m, _ = rg.masks(gt, "kitti15")
    src = r4m_source or ("regions_v2 R4m" if r4m is not None else "R4 (regions_v2 not landed)")
    out = {"R1": m["R1"], "R2": m["R2"], "R3": m["R3"],
           "R4m": np.asarray(r4m, bool) if r4m is not None else m["R4"], "r4m_source": src,
           "r4_label": "R4m" if r4m is not None else "R4"}  # the label printed with a leakage
    if grid_hw is not None and tuple(grid_hw) != tuple(m["R1"].shape):
        H, W = grid_hw
        for k in ("R1", "R2", "R3", "R4m"):
            out[k] = cv2.resize(out[k].astype(np.uint8), (W, H),
                                interpolation=cv2.INTER_NEAREST) > 0
    return out


COUNT_KEYS = ("R1", "R2", "R3", "R2+R3", "R4m")


def coverage_leakage(sel: np.ndarray, regions: dict, w: np.ndarray | None = None) -> dict:
    """Counts: for each region n (pixels) and k (selected pixels); coverage of R2+R3 is
    k/n of 'R2+R3' and leakage onto R4m is k/n of 'R4m'. w_R4m is the summed weight on R4m."""
    sel = np.asarray(sel, bool)
    r = {k: np.asarray(regions[k], bool) for k in ("R1", "R2", "R3", "R4m")}
    r["R2+R3"] = r["R2"] | r["R3"]
    c = {f"n_{k}": int(r[k].sum()) for k in COUNT_KEYS}
    c.update({f"k_{k}": int((sel & r[k]).sum()) for k in COUNT_KEYS})
    c["w_R4m"] = float(np.asarray(w, np.float64)[r["R4m"]].sum()) if w is not None else None
    c["r4m_source"] = regions.get("r4m_source", "unlabelled")
    c["r4_label"] = regions.get("r4_label", "R4")   # an unlabelled mask is never called R4m
    return c


def aggregate(counts: list[dict]) -> dict:
    """Pooled counts over frames, then the rates (statistics.md: counts are summed, rates are
    derived). Frames whose 'R4m' came from different sources are never pooled (ValueError): a
    leakage over a mix of R4 and R4m pixels would carry a label that is true for neither."""
    if not counts:
        raise ValueError("aggregate: no frames")
    sources = {(c["r4m_source"], c["r4_label"]) for c in counts}
    if len(sources) > 1:
        raise ValueError(f"aggregate: frames with different R4m sources: {sorted(sources)}")
    tot = {k: sum(c[k] for c in counts) for k in counts[0]
           if k.startswith(("n_", "k_"))}
    tot["w_R4m"] = (sum(c["w_R4m"] for c in counts) if counts[0]["w_R4m"] is not None
                    else None)
    tot["r4m_source"], tot["r4_label"] = counts[0]["r4m_source"], counts[0]["r4_label"]
    return tot


def rates(c: dict) -> dict:
    """Rates from pooled counts. 'leakage_R4m' is the share of the region called r4_label (R4m,
    or R4 until regions_v2 lands); both labels travel with the number."""
    def frac(k, n):
        return (c[k] / c[n]) if c[n] else None
    return {"coverage_R2+R3": frac("k_R2+R3", "n_R2+R3"), "coverage_R2": frac("k_R2", "n_R2"),
            "coverage_R3": frac("k_R3", "n_R3"), "coverage_R1": frac("k_R1", "n_R1"),
            "leakage_R4m": frac("k_R4m", "n_R4m"),
            "weight_R4m_per_px": (c["w_R4m"] / c["n_R4m"]) if c["w_R4m"] is not None
            and c["n_R4m"] else None,
            "r4_label": c["r4_label"], "r4m_source": c["r4m_source"]}


# ------------------------------------------------------------------------------ synthetic world
MOVER_ROWS, MOVER_COLS = (0.625, 0.875), (0.74, 0.95)  # fractions of H, W: the moving patch
SPLIT_COL = 0.69                                       # ground_left | ground_right (contains it)
WALL_MAX_ROW = 0.58                                    # the wall's last row is above ground


def _layers(depth, K, T_c2w):
    """Which analytic surface each pixel sees (wall z = 40, ground y = 1.5, box z = 12)."""
    H, W = depth.shape
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    d = T_c2w[:3, :3] @ (np.linalg.inv(K) @ np.stack([xs, ys, np.ones_like(xs)]).reshape(3, -1))
    pt = T_c2w[:3, 3:4] + depth.reshape(1, -1) * d
    pz, py = pt[2].reshape(H, W), pt[1].reshape(H, W)
    box = np.isclose(pz, 12.0, atol=1e-6)
    ground = np.isclose(py, 1.5, atol=1e-6) & ~box
    wall = np.isclose(pz, 40.0, atol=1e-6) & ~box & ~ground
    return box, ground, wall


def _synthetic_records(depth, K, T_c2w, hw) -> tuple[list[dict], np.ndarray]:
    """A hand-written SAM tree of the scene: full frame (dropped), wall (and its two halves),
    ground, box, ground_left (and its three pieces) and ground_right (the mover patch inside it,
    as its child). Returns the records and the mover patch mask."""
    sm = _sam_masks()
    H, W = hw
    box, ground, wall = _layers(depth, K, T_c2w)
    r0, r1 = int(MOVER_ROWS[0] * H), int(MOVER_ROWS[1] * H)
    c0, c1 = int(MOVER_COLS[0] * W), int(MOVER_COLS[1] * W)
    patch = np.zeros((H, W), bool)
    patch[r0:r1, c0:c1] = True
    patch &= ground
    ci = np.arange(W)[None, :]
    right = ground & (ci >= int(SPLIT_COL * W))
    left = ground & ~(ci >= int(SPLIT_COL * W))
    # finer pieces, so that both out-of-fold halves hold ground and wall pixels: with the coarse
    # tree (wall, ground, box) a fold fitted on the GROUND alone was degenerate (PnP: 113 inliers
    # of 2524, rmse 36 px on frame 5 [M reviewer of CYC-001]; the wall alone fitted fine). That
    # is the case oof_rigid_flows's fold check now reports instead of using
    thirds = [left & (ci >= int(f0 * W)) & (ci < int(f1 * W))
              for f0, f1 in ((0.0, 0.23), (0.23, 0.46), (0.46, 1.0))]
    halves = [wall & (ci < W // 2), wall & (ci >= W // 2)]
    masks = [np.ones((H, W), bool), wall, ground, box, left, right, patch] + thirds + halves
    recs = [{"segmentation": sm.rle_encode(m), "area": int(m.sum()),
             "bbox": [0, 0, W, H]} for m in masks if m.any()]
    return recs, patch


def synthetic_frames(n_frames: int = 10, hw: tuple[int, int] = (48, 160), seed: int = 0,
                     noise_px: float = 0.3, mover_px: float = 8.0, grid: bool = True):
    """The CPU world: frames t = 1..n_frames of the stub scene, each with its forward (t -> t+1)
    and backward (t -> t-1) pair. Per frame, a dict: depth, K, valid_T, pair_inputs [(flow_e, ev)],
    records (SAM JSON records), regions (R1, R2, R3, R4m by tools/eval/regions.py on the exact GT),
    gt (the GTImage behind them), gt_moving, and the window id. The evidence network is
    simulated: flow_e = GT + noise (sigma
    noise_px) and E^P = visible (V_noc) & valid_T & not OOF & not O_T; a mover patch of the ground
    moves mover_px px per frame in x (sign follows the direction), so some of its pixels leave the
    frame or end up unseen. Simulated alpha and visibility are [E]; the geometry is exact."""
    from p1lib import synth  # noqa: PLC0415

    rg = _regions_module()
    H, W = hw
    K = np.array(synth.k_for(hw), np.float64).reshape(3, 4)[:, :3]
    rng = np.random.default_rng(seed)
    frames = []
    for t in range(1, n_frames + 1):
        T = synth.Stub.scene_pose(t - 10)
        depth = synth.Stub.scene_depth(K, T, hw)
        records, patch = _synthetic_records(depth, K, T, hw)
        pair_inputs, fwd = [], None
        for sign, u in ((+1, t + 1), (-1, t - 1)):
            Tu = synth.Stub.scene_pose(u - 10)
            M = rigid.relative_pose(T, Tu)
            rend = rigid.render_pair(1.0 / depth, 1.0 / synth.Stub.scene_depth(K, Tu, hw), K, M,
                                     hw)
            base = np.nan_to_num(rend["flow_12"], nan=0.0)
            gtf = base.copy()
            gtf[0][patch] += sign * mover_px
            v_occ = rend["valid_depth_12"]
            tx = np.arange(W)[None, :] + gtf[0]
            ty = np.arange(H)[:, None] + gtf[1]
            inside = (tx >= 0) & (tx <= W - 1) & (ty >= 0) & (ty <= H - 1)
            v_noc = v_occ & np.where(patch, inside, ~rend["oof_12"] & ~rend["occ_12"])
            ev = v_noc & rend["valid_12"] & ~rend["oof_12"] & ~rend["occ_12"]
            flow_e = gtf + rng.normal(0.0, noise_px, gtf.shape)
            pair_inputs.append((flow_e, ev))
            if sign > 0:
                fwd = (gtf, v_occ, v_noc, rend["valid_12"])
        gtf, v_occ, v_noc, valid_T = fwd
        gt = rg.gt_from_arrays(np.moveaxis(gtf, 0, -1).astype(np.float32), v_occ,
                               np.moveaxis(gtf, 0, -1).astype(np.float32), v_noc,
                               patch.astype(np.uint8))
        regions = regions_from_gt(gt, r4m=patch & v_occ,
                                  r4m_source="synthetic: every FG pixel is a mover")
        frames.append({"id": f"synth_{t:02d}", "depth": depth, "K": K, "valid_T": valid_T,
                       "pair_inputs": pair_inputs, "records": records, "regions": regions,
                       "gt_moving": patch, "gt": gt, "seed": seed * 1000 + t})
    return frames


def run_frame(fr: dict, with_oracle: bool = True, pair_w: float = 1.0, p: Params | None = None,
              records: list[dict] | None = None) -> dict:
    """One synthetic (or prepared) frame through tree, out-of-fold evidence and every variant."""
    tree = build_tree(records if records is not None else fr["records"], fr["depth"].shape)
    pairs = [build_pair_evidence(fr["depth"], fr["K"], fe, ev, tree, fr["seed"] + i,
                                 name=("fwd", "bwd")[i], p=p)
             for i, (fe, ev) in enumerate(fr["pair_inputs"])]
    vs = variants_for_frame(tree, pairs, fr["valid_T"], fr["gt_moving"] if with_oracle else None,
                            pair_w, p)
    g = vs.pop("_gate")
    return {"counts": {k: coverage_leakage(v["sel"], fr["regions"], v["w"])
                       for k, v in vs.items()}, "gate": g, "tree": tree,
            "id": fr.get("id", ""), "fold_failures": g.fold_failures,
            "fold_warnings": g.fold_warnings, "n_folds": 2 * len(pairs)}


VARIANT_ORDER = ("none", "per_pixel_static", "per_pixel_reject", "gate", "oracle")
BANNER = "SYNTHETIC WORLD: test design, not evidence"


def run_synthetic(n_frames: int = 10, seed: int = 0) -> tuple[dict, dict]:
    """(pooled counts per variant, fold report). The fold report lists every fold pose that
    failed its check, by frame, pair and fold."""
    per = [run_frame(fr) for fr in synthetic_frames(n_frames, seed=seed)]
    agg = {v: aggregate([x["counts"][v] for x in per]) for v in VARIANT_ORDER}
    folds = {"n_folds": sum(x["n_folds"] for x in per),
             "failed": [{"frame": x["id"], **f} for x in per for f in x["fold_failures"]],
             "warnings": [{"frame": x["id"], **f} for x in per for f in x["fold_warnings"]]}
    return agg, folds


# ------------------------------------------------------------------------------ CLI
def _fmt(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def format_table(agg: dict) -> list[str]:
    """The result table of pooled counts per variant. The leakage column carries the region's own
    name (R4m, or R4 while regions_v2 has not landed) and the source line says where it came
    from."""
    g = agg["gate"]
    lab = g["r4_label"]
    lines = [f"{lab} source: {g['r4m_source']}",
             f"{'variant':<18}{'cov R2+R3':>10}{'cov R2':>8}{'cov R3':>8}{'cov R1':>8}"
             f"{'leak ' + lab:>10}   (n_R2+R3={g['n_R2+R3']}, n_{lab}={g['n_R4m']})"]
    for v in VARIANT_ORDER:
        if v not in agg:
            continue
        r = rates(agg[v])
        lines.append(f"{v:<18}{_fmt(r['coverage_R2+R3']):>10}{_fmt(r['coverage_R2']):>8}"
                     f"{_fmt(r['coverage_R3']):>8}{_fmt(r['coverage_R1']):>8}"
                     f"{_fmt(r['leakage_R4m']):>10}")
    return lines


def cmd_synth(a) -> int:
    agg, folds = run_synthetic(a.frames, a.seed)
    print(BANNER)
    print(f"arbitrate synth: {a.frames} frames, seed {a.seed}")
    print("\n".join(format_table(agg)))
    nf = len(folds["failed"])
    nw = len(folds["warnings"])
    print(f"fold poses: {folds['n_folds']} fitted, {nf} failed their check "
          f"(ratio >= {FOLD_MIN_INLIER_RATIO:g}, inliers >= {FOLD_MIN_INLIERS}); {nw} "
          "pose-disagreement warnings (dominant-mover suspects, not gated); "
          "a failed fold's pixels get no teacher (fail closed)")
    for f in folds["failed"]:
        print(f"ARBITRATE_FOLD_FAILED frame={f['frame']} pair={f['pair']} fold={f['fold']} "
              f"reason=\"{f['reason']}\"")
    for f in folds["warnings"]:
        print(f"ARBITRATE_FOLD_WARN frame={f['frame']} pair={f['pair']} {f['warning']}: "
              f"rot_deg={f['rot_deg']:.3f} trans_dir_deg={f['trans_dir_deg']:.2f}")
    print(BANNER)
    if a.json:
        out = {"evidence": False, "banner": BANNER, "r4m_source": agg["gate"]["r4m_source"],
               "r4_label": agg["gate"]["r4_label"], "folds": folds,
               "variants": {v: {"counts": agg[v], "rates": rates(agg[v])}
                            for v in VARIANT_ORDER}}
        Path(a.json).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    if agg["oracle"]["k_R4m"] != 0:
        print("ARBITRATE_ERROR the oracle leaks onto the moving region: the audit is broken")
        return 1
    return 0


def cmd_tree(a) -> int:
    try:
        recs = load_records(a.json)
        t = build_tree(recs, (a.hw[0], a.hw[1]))
    except (OSError, ValueError, KeyError, EOFError) as e:  # PermissionError is an OSError
        print(f"ARBITRATE_ERROR {a.json}: {type(e).__name__}: {e}")
        return 2
    print(f"nodes={t.n} pseudo={int(t.pseudo.sum())} roots={int((t.parent < 0).sum())} "
          f"min_area={int(t.area.min()) if t.n else 0} max_area={int(t.area.max()) if t.n else 0}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("synth", help="all variants on the synthetic frames")
    s.add_argument("--frames", type=int, default=10)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--json")
    t = sub.add_parser("tree", help="the tree of one frame's SAM records")
    t.add_argument("--json", required=True)
    t.add_argument("--hw", type=int, nargs=2, required=True)
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 64 if e.code else 0
    if a.cmd == "synth":
        return cmd_synth(a)
    if a.cmd == "tree":
        return cmd_tree(a)
    ap.print_help()
    return 64


if __name__ == "__main__":
    sys.exit(main())
