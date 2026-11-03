#!/usr/bin/env python3
"""arbitration_cv.py: IDEA-003. Can a GT-FREE per-pixel selector between the student S (U2Flow
kitti.pth) and a rigid teacher render T recover part of the per-pixel oracle headroom that
tools/p1/composite_bound.py measured (IDEA-002, EV-0401)? Dev half of KITTI-15, cached flows
only, CPU on the Mac; it computes numbers, writes no ledger row (hard rule 7) and starts no VM.

Selectors (each switches a pixel to T only where T is finite; S elsewhere)
    alpha   U2Flow's log-variance alpha_10to11 > a
    fb      S's forward-backward inconsistency |f12(p) + f21(p + f12(p))| > f (native px)
    band    lo < |T - S| <= hi (endpoint disagreement, native px) AND alpha > a
    gate    the W3-01 gate (tools/p1/arbitrate.py variant 'gate') run on cached inputs through
            tools/p1/arbitrate_run.py's frame_inputs, with the GRID tree (32 x 32 cells): no SAM
            file exists for frame 10 (EV-0300), so this is the gate's diagnostic floor, never the
            method. It has no threshold (the proposal's constants), so its CV gain equals its
            in-sample gain. Its geometry is the teacher's own eval cache and pose (calib K), and
            its mask (256 x 832, frame 10, pair 10->11) is resized to native by nearest neighbour
            on the align_corners=True grid (the geometry of resize_flow, alpha and fb) and
            applied to the named render T of the same source and window
Teacher provenance (header.teachers[].derived_from; none of the teachers is independent of S):
    selftri/native  an S-DERIVED THREE-FRAME RIGID REFIT: depth and pose of frame 10 are
                    triangulated from U2Flow's own flow_10to09 and flow_10to11 on low-alpha,
                    visible pixels (tools/p1/self_triangulate.py:1-17); it uses frame 09, which
                    S's two-frame 10->11 prediction never sees. A gain with selftri is S plus a
                    rigid refit of S's own three-frame flow, not an external geometry teacher
    pi3/ref, mapanything/ref  the foundation model's depth, with the relative pose T_ref fitted
                    by PnP to U2Flow's evidence (exp(alpha) <= 2, vis; tools/p1/refine_pose.py:
                    1-10): an external depth with an S-fitted pose
Thresholds come from FIXED grids declared below (A_GRID, FB_GRID, D_EDGES; +inf = never switch,
-inf = switch wherever T is finite and the score is not itself -inf, i.e. a pixel touched by a
non-finite alpha is never switched by an alpha rule), chosen before any dev number was seen
(the alpha range from two KITTI-12 items). Choice of thresholds: 2-fold cross-validation over
dev images. The folds are a seeded permutation of the 100 sorted dev ids (--fold-seed,
recorded with both id lists); every KITTI-15 image id is its own scene, so an image id is the
grouping unit. Each fold's thresholds are the grid point maximising the Fl-all gain (pooled
outliers) on the OTHER fold (ties: fewer switched pixels, then grid order); the pooled CV gain
scores every image with thresholds tuned on the fold it is not in. In-sample rows (tuned and
scored on all 100) are labelled OPTIMISTIC.
Rows: one per (teacher, selector); 'nested[selector]' picks teacher + thresholds inside training;
'nested[all]' picks teacher + selector + thresholds inside training (the honest best).
Statistics (metric s5): Fl px-pooled over V_occ (all), R1 and R3 of regions_v1 (R2 withheld-S6,
as in composite_bound), gain = Fl(S) - Fl(row) in pp, paired image-bootstrap CI
(composite_bound.arm_stats -> tools/eval/stats.py). The CI treats the fold thresholds as fixed
(it does not resample the tuning). Teacher use = switched pixels / GT-valid pixels, and / all
pixels of the frame.

Resolution (how S, alpha and fb meet the GT). S is the cache's native_10to11 (float32, resize_flow
of the 256 x 832 prediction to the GT's own (H, W), the evaluator's path), read exactly as
composite_bound / region_table do (region_table.open_entry, then pixel_maps against flow_occ).
alpha_10to11 (256 x 832, fp16) is resized to (H, W) with torch bilinear align_corners=True, the
geometry of resize_flow, with no value scaling (a score in log grid-px^2, thresholded as such).
fb is computed on the 256 x 832 grid with U2Flow's own core.utils.warp_utils.flow_warp (bilinear,
align_corners=True) from flow_10to11 and flow_11to10; its u and v are then scaled by W/832 and
H/256 (resize_flow's scale) and its norm resized like alpha. A pixel whose p + f12(p) leaves the
grid has fb = +inf (an out-of-frame guess by S's own flow, GT-free); that flag is resized
bilinearly and >= 0.5 marks the native pixel.

Frozen mode (IDEA-004, --frozen IDEA-004). ONE pre-declared rule, no grid, no CV, no tuning:
EV-0402's in-sample choice (teacher selftri/native/w03-11; switch S -> T at a pixel iff T is
finite AND alpha > 2.75 AND |T - S| <= 100 px, no lower bound), hard-coded as FROZEN_* and
registered by IDEA-004 before any KITTI-12 number existed. Scored on KITTI-12 training (ids
0-193, Mac GT; all, noc, occ, K12-R1, K12-R3; K12-R2 withheld-S6) with Fl(K15-rule), Out-All 3 px,
Out3 on V_noc (scored against flow_occ, so NOT the official Out-Noc), EPE px-pooled and img-mean
for S and the composite, the paired image-bootstrap gain and the share of pixels switched; then
on the KITTI-15 dev half, which must reproduce EV-0402's in-sample row exactly (exit 1
otherwise). Any --frozen-* value other than the constant, or any teacher, gate or fold option, is
exit 64 in this mode. The output is FROZEN_JSON.

How to invoke (repo root, the Mac venv):
    .venv/bin/python tools/p1/arbitration_cv.py --seed 20261008 [--fold-seed 20261008] \
        [--teachers pi3/ref,mapanything/ref,selftri/native] [--no-gate] \
        [--gate-geom pi3=DIR --gate-geom mapanything=DIR --gate-geom selftri=DIR] \
        [--B 10000] [--alpha 0.05] [--json-out reports/p1_region/arbitration_cv.json | --no-json]
    .venv/bin/python tools/p1/arbitration_cv.py --frozen IDEA-004 --seed 20261008 [--B 10000] \
        [--frozen-teacher selftri/native/w03-11 --frozen-alpha-gt 2.75 --frozen-d-le-px 100] \
        [--json-out reports/p1_region/arbitration_frozen_k12.json | --no-json]
Exit codes: region_table.py's (0 ok, 1 error, 2 precondition, 7 integrity, 50 seal, 64 usage).

Why it is built this way.
- One kernel: GT, region masks, cache readers, the seal refusals, the outlier map and the
  statistics come from composite_bound.py / region_table.py (and through them tools/eval); the
  gate comes from arbitrate.py / arbitrate_run.py. This file adds the selectors and the CV.
- Exact and cheap: for a fixed pixel the switch changes the outlier count by out_S - out_T, so
  every grid point's counts are read off one 2-D cumulative histogram per image.
- A selector that cannot see GT can still be tuned on it; the CV is what keeps the tuning off the
  scored images, and tests/test_arbitration_cv.py proves no fold is scored with its own tuning.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

import composite_bound as CB  # noqa: E402

RT, ek, rg, st, C = CB.RT, CB.ek, CB.rg, CB.st, CB.C
Fail = RT.Fail
TOOL = "tools/p1/arbitration_cv.py"
DEFAULT_JSON = "reports/p1_region/arbitration_cv.json"
DS = "kitti15"
REGIONS = CB.REGIONS                      # ("all", "R1", "R3")
GRID = (256, 832)
DEFAULT_FOLD_SEED = 20261008
KILL_PP = 0.2                             # IDEA-003 kill (GATE-1.d.delta_hat_min_pp, SOTA-pending)
DEFAULT_TEACHERS = ("pi3/ref", "mapanything/ref", "selftri/native")
DATA = Path("/Volumes/Shin/U2FLOW_data")
GATE_GEOM = {"mapanything": DATA / "work/p1geom/derived/evalmapa/v2",
             "pi3": DATA / "work/p1geom/derived/evalpi3/v2",
             "selftri": DATA / "derived/evalselftri/v2"}

INF = float("inf")
A_GRID = np.array([-INF, *np.arange(-4.5, 6.01, 0.25), INF])
FB_GRID = np.array([-INF, 0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50, 100,
                    INF])
D_EDGES = np.array([-1.0, 0.5, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 50, 100, INF])
D_TRIVIAL = np.array([-1.0, INF])
SELECTORS = ("alpha", "fb", "band", "gate")
DERIVED_FROM = {
    "selftri/native": "S-derived three-frame rigid refit: depth and pose triangulated from "
                      "U2Flow's own flow_10to09 + flow_10to11 on low-alpha visible pixels "
                      "(tools/p1/self_triangulate.py); uses frame 09, unseen by S's 10->11 "
                      "prediction; not independent of S",
    "pi3/ref": "pi3 depth; relative pose T_ref fitted by PnP to U2Flow evidence (exp(alpha) <= 2, "
               "vis; tools/p1/refine_pose.py): external depth, S-fitted pose",
    "mapanything/ref": "MapAnything depth; relative pose T_ref fitted by PnP to U2Flow evidence "
                       "(tools/p1/refine_pose.py): external depth, S-fitted pose",
    "pi3/native": "pi3 depth and pose as the model outputs them",
    "mapanything/native": "MapAnything depth and pose as the model outputs them",
}
S_DERIVED = ("selftri/",)

# The frozen rule of IDEA-004 (registered 2026-10-08, before any KITTI-12 number existed): EV-0402's
# in-sample choice, row band[selftri/native/w03-11] of reports/p1_region/arbitration_cv.json, tuned
# on all 100 KITTI-15 dev images. Applied with NO tuning (--frozen IDEA-004); the CLI refuses any
# other value, teacher or tuning option in that mode.
FROZEN_MODE = "frozen-IDEA-004"
FROZEN_TEACHER = "selftri/native/w03-11"   # EV-0402 / IDEA-004
FROZEN_ALPHA_GT = 2.75                     # switch iff alpha_10to11 > 2.75 (EV-0402 / IDEA-004)
FROZEN_D_LE_PX = 100.0                     # and |T - S| <= 100 px, no lower bound (EV-0402)
FROZEN_KILL_PP = 0.2                       # IDEA-004 kill: Fl and Out-All gains, and their CIs
FROZEN_JSON = "reports/p1_region/arbitration_frozen_k12.json"
FROZEN_REPRO_ROW = ("band[selftri/native/w03-11]", "in-sample-OPTIMISTIC")   # EV-0402, 0.673 pp
FROZEN_REGIONS = {"kitti12": ("all", "noc", "occ", "K12-R1", "K12-R3"),
                  "kitti15": ("all", "R1", "R3")}
FROZEN_STATS = {"kitti12": ("fl", "out3", "epe_pooled", "epe_img_mean"),
                "kitti15": ("fl", "epe_pooled", "epe_img_mean")}

# quantities of a per-image table (switched pixels and outlier gain out_S - out_T among them)
SW_IMG, SW_ALL, G_ALL, SW_R1, G_R1, SW_R3, G_R3 = range(7)
NQ = 7


# ------------------------------------------------------------------------------ GT-free signals
def resize_map(x: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """A (h, w) score map to (H, W): bilinear, align_corners=True (resize_flow's geometry), no
    value scaling."""
    import torch

    t = torch.from_numpy(np.ascontiguousarray(x, np.float32))[None, None]
    return torch.nn.functional.interpolate(t, tuple(hw), mode="bilinear",
                                           align_corners=True)[0, 0].numpy()


def fb_native(f12: np.ndarray, f21: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """|f12(p) + f21(p + f12(p))| on the grid, in native px, resized to hw; +inf where p + f12(p)
    leaves the grid (decided on the grid, flag resized bilinearly, >= 0.5)."""
    import torch

    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from core.utils.warp_utils import flow_warp  # noqa: PLC0415  (U2Flow's own warp)

    f12 = np.asarray(f12, np.float32)
    f21 = np.asarray(f21, np.float32)
    h, w = f12.shape[1:]
    H, W = hw
    with torch.no_grad():
        a = torch.from_numpy(f12)[None]
        b = torch.from_numpy(f21)[None]
        fb = (a + flow_warp(b, a))[0].numpy()
    xs = np.arange(w, dtype=np.float32)[None] + f12[0]
    ys = np.arange(h, dtype=np.float32)[:, None] + f12[1]
    oof = (xs < 0) | (xs > w - 1) | (ys < 0) | (ys > h - 1) | ~np.isfinite(fb).all(axis=0)
    mag = np.hypot(fb[0] * (W / w), fb[1] * (H / h))
    mag = np.where(oof, 0.0, mag).astype(np.float32)
    out = resize_map(mag, hw)
    out[resize_map(oof.astype(np.float32), hw) >= 0.5] = INF
    return out


def nearest_ac(mask: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """A (h, w) mask to (H, W) by nearest neighbour on the align_corners=True grid (native pixel
    y samples grid row round(y (h-1)/(H-1))), the geometry resize_flow, alpha and fb use; cv2
    INTER_NEAREST would use half-pixel centres instead."""
    h, w = mask.shape
    H, W = hw
    ys = np.floor(np.arange(H) * ((h - 1) / max(H - 1, 1)) + 0.5).astype(int)   # half up
    xs = np.floor(np.arange(W) * ((w - 1) / max(W - 1, 1)) + 0.5).astype(int)
    return np.asarray(mask, bool)[ys[:, None], xs[None, :]]


def alpha_native(alpha: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """alpha resized like resize_flow's geometry; a native pixel touched by a non-finite grid
    alpha gets -inf (never switched by the alpha rule)."""
    a = np.asarray(alpha, np.float32)
    fin = np.isfinite(a)
    out = resize_map(np.where(fin, a, 0.0), hw)
    if not fin.all():
        out[resize_map((~fin).astype(np.float32), hw) > 0] = -INF
    return out


# ------------------------------------------------------------------------------ per-image tables
def image_table(score: np.ndarray, d: np.ndarray | None, fin_t: np.ndarray, out_s: np.ndarray,
                out_t: np.ndarray, masks: dict, a_grid: np.ndarray, d_edges: np.ndarray
                ) -> np.ndarray:
    """(NQ, na, ne) int64: Q[q, j, e] = sum of quantity q over pixels with finite T,
    score > a_grid[j] and d <= d_edges[e]. A band lo < d <= hi is Q[.., hi] - Q[.., lo]."""
    na, ne = len(a_grid), len(d_edges)
    sel = np.asarray(fin_t, bool)
    s = np.asarray(score, np.float64)[sel]
    s = np.where(np.isnan(s), -INF, s)
    dd = np.zeros(s.shape) if d is None else np.asarray(d, np.float64)[sel]
    ba = np.searchsorted(a_grid, s, "left")           # s > a_grid[j] for every j < ba
    bd = np.searchsorted(d_edges, dd, "left")         # d_edges[bd-1] < d <= d_edges[bd]
    flat = ba * ne + bd
    v = masks["all"][sel]
    g = out_s[sel].astype(np.int64) - out_t[sel].astype(np.int64)
    r1, r3 = masks["R1"][sel], masks["R3"][sel]
    weights = (np.ones(s.shape, np.int64), v, g * v, r1, g * r1, r3, g * r3)
    Q = np.empty((NQ, na, ne), np.int64)
    for q, wq in enumerate(weights):
        h = np.bincount(flat, weights=wq.astype(np.float64), minlength=(na + 1) * ne)
        h = np.rint(h).astype(np.int64).reshape(na + 1, ne)
        tail = np.cumsum(h[::-1], axis=0)[::-1]        # tail[k] = sum over ba >= k
        Q[q] = np.cumsum(tail[1:], axis=1)            # ba >= j + 1  <=>  s > a_grid[j]
    return Q


def base_counts(out_s: np.ndarray, masks: dict) -> dict:
    return {**CB.counts(out_s, masks, REGIONS), "px": int(out_s.size)}


def choices(na: int, ne: int) -> np.ndarray:
    """(k, 3) of (j, lo, hi), lo < hi: every threshold of the score grid x every d band."""
    los, his = np.triu_indices(ne, 1)
    j = np.repeat(np.arange(na), len(los))
    return np.stack([j, np.tile(los, na), np.tile(his, na)], 1)


def eval_choices(Q: np.ndarray, ch: np.ndarray) -> np.ndarray:
    """(N, NQ, na, ne) per-image tables -> (N, NQ, k) counts of every choice."""
    return Q[:, :, ch[:, 0], ch[:, 2]] - Q[:, :, ch[:, 0], ch[:, 1]]


def pick(Qc: np.ndarray, idx) -> int:
    """The choice with the largest Fl-all gain pooled over images idx (the denominators do not
    depend on the choice); ties: fewer switched GT-valid pixels, then the first."""
    idx = np.asarray(idx, int)
    gain = Qc[idx, G_ALL].sum(axis=0)
    sw = Qc[idx, SW_ALL].sum(axis=0)
    best = np.flatnonzero(gain == gain.max())
    return int(best[np.argmin(sw[best])])


def composite(base: list[dict], Qc: np.ndarray, choice_of: np.ndarray) -> tuple[list[dict], dict]:
    """Per-image {region: {n, out}} of the composite where image i uses choice choice_of[i], and
    the pooled teacher-use counts."""
    per, use = [], {"sw_valid": 0, "n_valid": 0, "sw_img": 0, "px": 0}
    gq = {"all": G_ALL, "R1": G_R1, "R3": G_R3}
    for i, b in enumerate(base):
        c = int(choice_of[i])
        per.append({r: {"n": b[r]["n"], "out": b[r]["out"] - int(Qc[i, gq[r], c])}
                    for r in REGIONS})
        use["sw_valid"] += int(Qc[i, SW_ALL, c])
        use["n_valid"] += b["all"]["n"]
        use["sw_img"] += int(Qc[i, SW_IMG, c])
        use["px"] += b["px"]
    return per, use


def fl_gain(base: list[dict], per: list[dict], idx) -> float:
    n = sum(base[i]["all"]["n"] for i in idx)
    return 100.0 * sum(base[i]["all"]["out"] - per[i]["all"]["out"] for i in idx) / n if n else 0.0


def cross_validate(base: list[dict], Qc: np.ndarray, folds: list[list[int]], picker=pick
                   ) -> tuple[np.ndarray, list[dict]]:
    """2-fold CV: fold k's images are scored with the choice picked on the OTHER fold only.
    Returns (choice per image, fold records)."""
    if len(folds) != 2 or set(folds[0]) & set(folds[1]):
        raise ValueError("two disjoint folds are needed")
    choice_of = np.full(len(base), -1, int)
    recs = []
    for k in (0, 1):
        train, test = folds[1 - k], folds[k]
        c = picker(Qc, train)
        choice_of[np.asarray(test, int)] = c
        per, _ = composite(base, Qc, np.full(len(base), c))
        recs.append({"tuned_on": "AB"[1 - k], "scored_on": "AB"[k], "choice": c,
                     "train_gain_pp": fl_gain(base, per, train),
                     "test_gain_pp": fl_gain(base, per, test)})
    if (choice_of < 0).any():
        raise ValueError("an image is in no fold")
    return choice_of, recs


def make_folds(ids: list[str], seed: int) -> list[list[int]]:
    perm = np.random.default_rng(seed).permutation(len(ids))
    h = len(ids) // 2
    return [sorted(perm[:h].tolist()), sorted(perm[h:].tolist())]


# ------------------------------------------------------------------------------ the gate
def gate_masks(src: str, geom: Path, window: str, ids: list[str], shapes: dict, evi, root: Path
               ) -> tuple[dict, list[dict]]:
    """{seq: native bool mask} of arbitrate's 'gate' variant (grid tree, no SAM) for one teacher
    source and window, and the frames that could not run (with the reason)."""
    import arbitrate as AR  # noqa: PLC0415
    import arbitrate_run as ARR  # noqa: PLC0415
    import p1_windows as MW  # noqa: PLC0415
    from p1lib import geomio as GI  # noqa: PLC0415

    C.refuse_sealed(geom)
    if not geom.is_dir():
        raise Fail(RT.EXIT_PRE, f"{geom}: gate geometry cache not on disk")
    rows = ARR.select_rows(MW.load_plan(MW.PLAN), DS, window, ids, None)
    ARR.check_dev(DS, rows, root)
    geo = GI.GeomIndex(geom)
    first = next((r for r in rows if geo.get(r["id"])), None)
    kch = GI.resolve_k(geo.get(first["id"])[1], "calib") if first else "calib"
    tree = ARR.grid_tree()
    out, skipped = {}, []
    for row in rows:
        try:
            fi = ARR.frame_inputs(geo, evi, row, kch)
        except ARR.Skip as e:
            skipped.append({"id": row["seq"], "reason": str(e)})
            continue
        pairs = [AR.build_pair_evidence(fi["depth"], fi["K"], fe, ev, tree, fi["seed"] + i,
                                        name=fi["pair_names"][i])
                 for i, (fe, ev) in enumerate(fi["pair_inputs"])]
        sel = AR.variants_for_frame(tree, pairs, fi["valid_T"])["gate"]["sel"]
        out[row["seq"]] = nearest_ac(sel, shapes[row["seq"]])
    return out, skipped


# ------------------------------------------------------------------------------ loading
def dev_evidence(path: Path, dev: set, idx=None, ds: str = DS):
    """(seq, arrays) of the dev KITTI-15 items of the U2Flow evidence cache, in seq order. The
    cache also holds alpha and grid flows for held-out ids (the evidence dump covers whole
    windows), so region_table's flow-key seal (any non-dev item holding the key = exit 50) cannot
    be used for alpha. Like arbitrate_run.py (check_dev), a non-dev item is never requested,
    used or scored: its id is parsed from the index and skipped before idx.get. Its bytes may
    still be decoded in memory, because CacheIndex.get unpacks a whole shard and dev and held-out
    items share shards (and region_table.open_entry's own seal loop gets every non-dev item on
    purpose); no held-out GT is read and no held-out array reaches a table."""
    from p1lib import cache as CA  # noqa: PLC0415
    from p1lib import kitti as KT  # noqa: PLC0415

    C.refuse_sealed(path)
    idx = CA.CacheIndex(path) if idx is None else idx      # idx: a stand-in in tests
    items = []
    for item in idx.ids():
        d = KT.parse_id(item)
        if (d["ds"] == ds and d["sub"] == "training" and d["side"] == "image_2"
                and d["seq"] in dev):
            items.append((d["seq"], item))
    for seq, item in sorted(items):
        got = idx.get(item)
        if got is not None:
            yield seq, got[0]


def load_student(entry: dict, root: Path, dev: set, held: set, gtc, ds: str = DS,
                 with_fb: bool = True) -> dict:
    """S's native flow and outlier map, and the native alpha (and fb) maps, per id: the dev ids
    for kitti15; for kitti12 pass dev = the 194 KITTI-12 ids."""
    _, streams = RT.open_entry({**entry, "datasets": [ds]}, root, dev, held)
    factory, pred_sha, _ = streams[ds]
    S = {}
    for seq, flow in factory():
        shape, flow_occ, _ = gtc.get(seq)
        fin, _, out = RT.pixel_maps(flow, shape, flow_occ, ds)
        if not fin.all():
            raise Fail(RT.EXIT_ERR, f"student {seq}: non-finite flow")
        S[seq] = {"shape": shape, "flow": flow, "out": out}
    for seq, arr in dev_evidence(Path(entry["path"]), dev, ds=ds):
        if seq not in S:
            continue
        sh = S[seq]["shape"]
        S[seq]["alpha"] = alpha_native(arr["alpha_10to11"], sh).astype(np.float32)
        if with_fb:
            S[seq]["fb"] = fb_native(arr["flow_10to11"], arr["flow_11to10"], sh)
    miss = [s for s, v in S.items() if "alpha" not in v]
    if miss:
        raise Fail(RT.EXIT_ERR, f"no alpha/flow evidence for {len(miss)} {ds} ids, e.g. "
                   f"{miss[0]}")
    return {"pred_sha": pred_sha, "maps": S}


# ------------------------------------------------------------------------------ rows
def describe(sel: str, member: str, ch: np.ndarray, c: int) -> dict:
    j, lo, hi = (int(x) for x in ch[c])
    d = {"member": member, "selector": sel}
    if sel == "alpha":
        d["alpha_gt"] = float(A_GRID[j])
    elif sel == "fb":
        d["fb_gt_px"] = float(FB_GRID[j])
    elif sel == "band":
        d.update(alpha_gt=float(A_GRID[j]), d_gt_px=float(D_EDGES[lo]),
                 d_le_px=float(D_EDGES[hi]))
    else:
        d["gate"] = "arbitrate.py gate, grid tree (no threshold)"
    return d


def make_row(name: str, members: list[tuple[str, str, np.ndarray, np.ndarray]], base, folds,
             boot, draws, kind: str) -> dict:
    """members: (key, selector, Qc (N, NQ, k), choice table). Concatenates the members' choices
    so a nested row picks member and thresholds together."""
    Qc = np.concatenate([m[2] for m in members], axis=2)
    owner = [(mi, c) for mi, m in enumerate(members) for c in range(m[2].shape[2])]

    def desc(c):
        mi, cc = owner[c]
        return describe(members[mi][1], members[mi][0], members[mi][3], cc)

    if kind == "cv":
        choice_of, recs = cross_validate(base, Qc, folds)
        for r in recs:
            r["choice"] = desc(r["choice"])
    else:
        c = pick(Qc, range(len(base)))
        choice_of, recs = np.full(len(base), c), [{"tuned_on": "all", "scored_on": "all",
                                                   "choice": desc(c)}]
    per, use = composite(base, Qc, choice_of)
    return {"row": name, "kind": "cv" if kind == "cv" else "in-sample-OPTIMISTIC",
            "members": [m[0] + ":" + m[1] for m in members], "folds": recs,
            "teacher_use_valid_pct": 100.0 * use["sw_valid"] / max(use["n_valid"], 1),
            "teacher_use_image_pct": 100.0 * use["sw_img"] / max(use["px"], 1),
            "regions": CB.arm_stats(per, base, boot, draws)}


def selector_tables(S: dict, ids: list[str], T: dict, gate: dict | None, mask_of) -> dict:
    """{selector: (Qc (N, NQ, k), choices)} for one teacher; T = {seq: (flow, fin, out)}."""
    specs = {"alpha": (A_GRID, D_TRIVIAL), "fb": (FB_GRID, D_TRIVIAL), "band": (A_GRID, D_EDGES)}
    if gate is not None:
        specs["gate"] = (np.array([0.5]), D_TRIVIAL)
    tabs = {k: [] for k in specs}
    for seq in ids:
        shape, m = mask_of(seq)
        s = S["maps"][seq]
        got = T.get(seq)
        if got is None:
            flow_t, fin_t, out_t = None, np.zeros(shape, bool), np.zeros(shape, bool)
        else:
            flow_t, fin_t, out_t = got
        d = None
        if flow_t is not None:
            d = np.where(fin_t, np.hypot(*(np.nan_to_num(flow_t) - s["flow"])), INF)
        for k, (ag, de) in specs.items():
            if k == "alpha":
                sc, dd = s["alpha"], None
            elif k == "fb":
                sc, dd = s["fb"], None
            elif k == "band":
                sc, dd = s["alpha"], d
            else:
                gm = gate.get(seq)
                sc = np.where(gm, 1.0, -INF) if gm is not None else np.full(shape, -INF)
                dd = None
            tabs[k].append(image_table(sc, dd, fin_t, s["out"], out_t & fin_t, m, ag, de))
    out = {}
    for k, (ag, de) in specs.items():
        ch = choices(len(ag), len(de))
        out[k] = (eval_choices(np.stack(tabs[k]), ch), ch)
    return out


# ------------------------------------------------------------------------------ output
def _f(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


def render(doc: dict) -> str:
    h = doc["header"]
    out = [f"# IDEA-003 GT-free arbitration, 2-fold CV (dev half, KITTI-15, n={h['n_images']})",
           "", f"Fl px-pooled %; gain = Fl(S) - Fl(row) pp, paired image-bootstrap "
           f"{round(100 * (1 - h['statistics']['alpha']))}% CI (B={h['statistics']['B']}, "
           f"seed {h['statistics']['seed']}, diagnostic); folds seed {h['folds']['seed']}. "
           "S Fl-all " + _f(h["S"]["fl_all"], 3) + ".", "",
           "| row | kind | Fl-all | gain all [CI] | gain R1 | gain R3 | use valid % | "
           "use img % | fold A<-B gain | fold B<-A gain |", "|---" * 10 + "|"]
    for r in doc["rows"]:
        a, r1, r3 = (r["regions"][k] for k in REGIONS)
        ci = a.get("gain_ci")
        fg = [x.get("test_gain_pp") for x in r["folds"]]
        out.append(f"| {r['row']} | {'CV' if r['kind'] == 'cv' else 'IN-SAMPLE (OPTIMISTIC)'} "
                   f"| {_f(a['fl'])} | {_f(a['gain_pp'])}"
                   + (f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "")
                   + f" | {_f(r1['gain_pp'])} | {_f(r3['gain_pp'])} "
                   f"| {_f(r['teacher_use_valid_pct'], 1)} | {_f(r['teacher_use_image_pct'], 1)} "
                   f"| {_f(fg[0]) if r['kind'] == 'cv' else '-'} "
                   f"| {_f(fg[1]) if r['kind'] == 'cv' else '-'} |")
    out += ["", f"Kill read (IDEA-003, user decides): {h['kill_read']['text']}", ""]
    return "\n".join(out)


def _header_of(p: Path) -> dict:
    try:
        h = json.loads(p.read_text("utf-8")).get("header", {})
        return h if isinstance(h, dict) else {}
    except (OSError, ValueError, AttributeError):
        return {}


def _is_ours(p: Path, suffix: str) -> bool:
    """The CV JSON only: the frozen mode's JSON (and the reverse) is never overwritten."""
    h = _header_of(p)
    return h.get("tool") == TOOL and h.get("mode") is None


def _is_ours_frozen(p: Path, suffix: str) -> bool:
    h = _header_of(p)
    return h.get("tool") == TOOL and h.get("mode") == FROZEN_MODE


def _choice_text(c: dict) -> str:
    if c["selector"] == "alpha":
        return f"{c['member']} alpha > {c['alpha_gt']:g} (alpha only)"
    if c["selector"] == "band":
        return (f"{c['member']} alpha > {c['alpha_gt']:g} and {c['d_gt_px']:g} < |T-S| <= "
                f"{c['d_le_px']:g} px (alpha + band)")
    if c["selector"] == "fb":
        return f"{c['member']} fb > {c['fb_gt_px']:g} px"
    return f"{c['member']} gate"


def kill_read(rows: list[dict]) -> dict:
    """IDEA-003's kill on the CV Fl-all gains. Ties between rows with the same gain prefer
    nested[all] (chosen inside training), then the first row. The text names each fold's choice
    and says when the winning member is S-derived (selftri)."""
    cv = [r for r in rows if r["kind"] == "cv"]
    g = max(r["regions"]["all"]["gain_pp"] for r in cv)
    top = [r for r in cv if r["regions"]["all"]["gain_pp"] == g]
    best = next((r for r in top if r["row"] == "nested[all]"), top[0])
    nest = next(r for r in cv if r["row"] == "nested[all]")
    nested = nest["regions"]["all"]["gain_pp"]
    fold_choices = [f"fold {f['scored_on']} (tuned on {f['tuned_on']}): {_choice_text(f['choice'])}"
                    for f in nest["folds"]]
    members = {f["choice"]["member"] for f in nest["folds"]}
    s_derived = any(m.startswith(S_DERIVED) for m in members)
    fires = g < KILL_PP
    text = (f"best CV Fl-all gain = {g:.2f} pp ({best['row']}; the max over CV rows, itself a "
            f"selection on CV results) {'<' if fires else '>='} {KILL_PP} pp: kill "
            f"{'FIRES' if fires else 'does not fire'}; nested[all] (teacher + selector + "
            f"thresholds chosen inside training) = {nested:.2f} pp; its choices: "
            + "; ".join(fold_choices)
            + (". The chosen teacher is an S-DERIVED THREE-FRAME RIGID REFIT (selftri: "
               "triangulated from U2Flow's own 10->09 and 10->11 flows), not an external "
               "geometry teacher" if s_derived else ""))
    return {"best_cv_gain_pp": g, "best_cv_row": best["row"], "nested_all_cv_gain_pp": nested,
            "nested_all_fold_choices": [f["choice"] for f in nest["folds"]],
            "s_derived_teacher": s_derived, "threshold_pp": KILL_PP, "fires": fires,
            "fires_on_nested_all": nested < KILL_PP, "text": text}


def oracle_context(root: Path, pred_sha: str, s_fl: float) -> dict:
    """composite_bound.json's ORACLE rows for the same S (headroom), and the S cross-check."""
    p = root / CB.DEFAULT_JSON
    doc = json.loads(p.read_text("utf-8"))
    src = [s for s in doc["header"]["sources"] if s["key"] == "u2flow"]
    if not src or src[0]["pred_sha"] != pred_sha:
        raise Fail(RT.EXIT_ERR, f"{p}: no u2flow source with pred_sha {pred_sha}")
    rows = {r["arm"]: r["regions"]["all"] for r in doc["rows"] if r["student"] == "u2flow"}
    ref = rows["S"]["fl"]
    if abs(ref - s_fl) > 1e-6:
        raise Fail(RT.EXIT_ERR, f"S Fl-all {s_fl} != composite_bound.json {ref}")
    return {"against": CB.DEFAULT_JSON, "S_fl_all_abs_diff": abs(ref - s_fl),
            "oracle_gain_pp": {k: v["gain_pp"] for k, v in rows.items()
                               if k.startswith(("ORACLE-pixel[", "ORACLE-best-of-named"))}}


# ------------------------------------------------------------------------------ frozen (IDEA-004)
def frozen_switch(alpha: np.ndarray, flow_s: np.ndarray, flow_t: np.ndarray) -> np.ndarray:
    """IDEA-004's rule: switch S -> T iff T is finite AND alpha > 2.75 AND |T - S| <= 100 px.
    The comparisons of the band selector's table (image_table): strict on alpha (a -inf or NaN
    alpha never switches), <= on the endpoint distance, no lower bound."""
    flow_t = np.asarray(flow_t, np.float32)
    fin_t = np.isfinite(flow_t).all(axis=0)
    d = np.hypot(*(np.nan_to_num(flow_t) - np.asarray(flow_s, np.float32)))
    return fin_t & (np.asarray(alpha) > FROZEN_ALPHA_GT) & (d <= FROZEN_D_LE_PX)


def frozen_counts(flow: np.ndarray, shape, flow_occ: np.ndarray, masks: dict, ds: str,
                  regions) -> dict:
    """{region: {n, out, epe_sum[, out3]}} of one full-coverage flow, through region_table's
    image_counts (regions.py's kernel; out3 = EPE > 3 px, KITTI-12 only)."""
    got = RT.image_counts(flow, shape, flow_occ, masks, ds, list(regions))
    res = {}
    for r, c in got.items():
        if c["n_cov"] != c["n"]:
            raise Fail(RT.EXIT_ERR, "a non-finite flow pixel in a frozen-rule arm")
        res[r] = {"n": c["n"], "out": c["out_fc"], "epe_sum": c["epe_sum"]}
        if "out3_fc" in c:
            res[r]["out3"] = c["out3_fc"]
    return res


def frozen_image(s: dict, flow_t, shape, flow_occ, masks: dict, ds: str, regions) -> tuple:
    """(S counts, composite counts, switched-pixel counts) of one image; flow_t None = no
    teacher render for this image (the composite is S)."""
    cs = frozen_counts(s["flow"], shape, flow_occ, masks, ds, regions)
    if flow_t is None:
        sw = np.zeros(shape, bool)
        cc = cs
    else:
        sw = frozen_switch(s["alpha"], s["flow"], flow_t)
        comp = np.where(sw[None], np.asarray(flow_t, np.float32), s["flow"])
        cc = frozen_counts(comp, shape, flow_occ, masks, ds, regions)
    use = {"px": int(sw.size), "sw_img": int(sw.sum()),
           **{f"sw_{r}": int((sw & masks[r]).sum()) for r in regions}}
    return cs, cc, use


def frozen_label(ds: str, region: str, stat: str) -> str:
    """The quoted name of one statistic. KITTI-12 noc is scored against flow_occ (V1Provider,
    EV-0300), so its EPE > 3 px rate is NOT the official Out-Noc (which uses flow_noc)."""
    agg = "img-mean" if stat == "epe_img_mean" else "px-pooled"
    name = {"fl": "Fl(K15-rule)" if ds == "kitti12" else "Fl", "out3": "Out3",
            "epe_pooled": "EPE", "epe_img_mean": "EPE"}[stat]
    if stat == "out3" and region == "all":
        name = "Out-All 3px"
    elif stat == "out3" and region in ("noc", "K12-R1"):
        name = "Out3 on V_noc vs flow_occ (not official Out-Noc)"
    return f"{name} {agg}"


def frozen_stats(per_s: list[dict], per_c: list[dict], regions, stats, boot: dict,
                 ds: str) -> dict:
    """{region: {n_images, n_pixels, stat: {S, frozen, gain, gain_ci, agg, label}}};
    gain = S - frozen (positive = the rule helps), paired image bootstrap (tools/eval/stats.py),
    the image as the unit, the pooled statistic recomputed from counts in every replicate."""
    agg = {"fl": "px-pooled", "out3": "px-pooled", "epe_pooled": "px-pooled",
           "epe_img_mean": "img-mean"}
    res = {}
    for r in regions:
        a, b = RT.region_arrays(per_s, r), RT.region_arrays(per_c, r)
        rec = {"n_images": int((a["n"] > 0).sum()), "n_pixels": int(a["n"].sum())}
        if rec["n_pixels"] == 0:
            res[r] = {**rec, "status": "n=0"}
            continue
        rec["status"] = "ok"
        for stat in stats:
            sv, cv = st.statistic(a, stat), st.statistic(b, stat)
            ent = {"S": sv, "frozen": cv, "gain": sv - cv, "gain_ci": None, "agg": agg[stat],
                   "label": frozen_label(ds, r, stat)}
            if boot["B"] > 0:
                g = st.paired_image_bootstrap(a, b, stat=stat, B=boot["B"], seed=boot["seed"],
                                              alpha=boot["alpha"])
                ent["gain"], ent["gain_ci"] = g["estimate"], g["ci"]
            rec[stat] = ent
        res[r] = rec
    return res


def frozen_kill_read(k12: dict) -> dict:
    """IDEA-004's kill on KITTI-12 'all': fires if the Fl(K15-rule) gain < 0.2 pp, or the
    Out-All gain < 0.2 pp, or either CI includes 0. No CI (B = 0) and no gain below the bar =
    not evaluated: fires is None, never False."""
    reasons, readable = [], True
    for name, key in (("Fl(K15-rule)", "fl"), ("Out-All", "out3")):
        rec = k12["all"][key]
        if rec["gain"] < FROZEN_KILL_PP:
            reasons.append(f"{name} gain {rec['gain']:.3f} pp < {FROZEN_KILL_PP} pp")
        ci = rec["gain_ci"]
        if ci is None:
            readable = False
        elif ci[0] <= 0.0 <= ci[1]:
            reasons.append(f"{name} CI [{ci[0]:.3f}, {ci[1]:.3f}] includes 0")
    fires = True if reasons else (False if readable else None)
    fl, o3 = k12["all"]["fl"], k12["all"]["out3"]
    verdict = {True: "FIRES (" + "; ".join(reasons) + ")", False: "does not fire",
               None: "NOT EVALUATED (no CI: B = 0)"}[fires]
    text = (f"KITTI-12 (194 images) Fl(K15-rule) gain {fl['gain']:.3f} pp, Out-All gain "
            f"{o3['gain']:.3f} pp: kill {verdict}")
    return {"threshold_pp": FROZEN_KILL_PP, "fires": fires, "readable": readable,
            "reasons": reasons, "text": text}


def frozen_repro(root: Path, k15: dict, use: dict, s_sha: str, t_sha: str, boot: dict) -> dict:
    """The KITTI-15 dev run of the frozen rule must equal EV-0402's in-sample row (same choice,
    same S and teacher shards, same gains; the CI too when B and the seed match)."""
    p = root / DEFAULT_JSON
    doc = json.loads(p.read_text("utf-8"))
    rows = [r for r in doc["rows"] if (r["row"], r["kind"]) == FROZEN_REPRO_ROW]
    if len(rows) != 1:
        raise Fail(RT.EXIT_ERR, f"{p}: no single row {FROZEN_REPRO_ROW}")
    ref = rows[0]
    ch = ref["folds"][0]["choice"]
    if (ch["member"], ch["selector"], ch["alpha_gt"], ch["d_le_px"]) != (
            FROZEN_TEACHER, "band", FROZEN_ALPHA_GT, FROZEN_D_LE_PX) or ch["d_gt_px"] >= 0:
        raise Fail(RT.EXIT_ERR, f"{p}: the EV-0402 row's choice {ch} is not the frozen rule")
    h = doc["header"]
    t_ref = [t["pred_sha"] for t in h["teachers"] if t["key"] == FROZEN_TEACHER]
    if h["S"]["pred_sha"] != s_sha or t_ref != [t_sha]:
        raise Fail(RT.EXIT_ERR, f"{p}: S or teacher pred_sha differs from this run's")
    res = {"against": f"{DEFAULT_JSON} row {FROZEN_REPRO_ROW[0]} ({FROZEN_REPRO_ROW[1]})",
           "evidence": "EV-0402", "choice": ch}
    for r in REGIONS:
        want, got = ref["regions"][r]["gain_pp"], k15[r]["fl"]["gain"]
        if abs(want - got) > 1e-9:
            raise Fail(RT.EXIT_ERR, f"reproduction: K15 dev {r} gain {got} != EV-0402 {want}")
        res[f"gain_pp_{r}"] = {"ev0402": want, "here": got, "abs_diff": abs(want - got)}
    sb = h["statistics"]
    if (sb["B"], sb["seed"], sb["alpha"]) == (boot["B"], boot["seed"], boot["alpha"]):
        want, got = ref["regions"]["all"]["gain_ci"], k15["all"]["fl"]["gain_ci"]
        if max(abs(x - y) for x, y in zip(want, got, strict=True)) > 1e-9:
            raise Fail(RT.EXIT_ERR, f"reproduction: K15 dev CI {got} != EV-0402 {want}")
        res["gain_ci_all"] = {"ev0402": want, "here": got}
    for k, ours in (("teacher_use_valid_pct", 100.0 * use["sw_all"] / use["n_valid"]),
                    ("teacher_use_image_pct", 100.0 * use["sw_img"] / use["px"])):
        if abs(ref[k] - ours) > 1e-9:
            raise Fail(RT.EXIT_ERR, f"reproduction: {k} {ours} != EV-0402 {ref[k]}")
        res[k] = {"ev0402": ref[k], "here": ours}
    res["equal"] = True
    return res


def frozen_dataset(ds: str, root: Path, gt_root: Path, s_entry: dict, t_entry: dict,
                   boot: dict) -> dict:
    """The frozen rule on one dataset: KITTI-12 ids 0-193, or the KITTI-15 dev half (never a
    held-out or all-200 id: region_table's seal checks run inside open_entry)."""
    dev_l, held_l = ek.split_ids(root)
    dev, held = set(dev_l), set(held_l)
    ids = sorted(dev_l) if ds == "kitti15" else list(ek.K12_IDS)
    regions = FROZEN_REGIONS[ds]
    gt = rg.GTSource(ds, root=gt_root)
    gtc = RT.GTCache(gt, RT.V1Provider(), ds, 0)
    try:
        S = load_student(s_entry, root, dev if ds == "kitti15" else set(ids), held, gtc, ds=ds,
                         with_fb=False)
        if sorted(S["maps"]) != ids:
            raise Fail(RT.EXIT_ERR, f"u2flow {ds}: {len(S['maps'])} images, need {len(ids)}")
        _, streams = RT.open_entry({**t_entry, "datasets": [ds]}, root, dev, held)
        if ds not in streams:
            raise Fail(RT.EXIT_PRE, f"{t_entry['path']}: holds no {ds} predictions")
        factory, t_sha, _ = streams[ds]
        per = {}

        def one(seq, flow_t):
            shape, flow_occ, packed = gtc.get(seq)
            m = {r: RT.unpack(packed[r], shape) for r in regions}
            per[seq] = frozen_image(S["maps"][seq], flow_t, shape, flow_occ, m, ds, regions)

        for seq, flow_t in factory():
            if seq not in S["maps"]:
                raise Fail(RT.EXIT_ERR, f"teacher {ds} {seq}: not an id of the split")
            one(seq, flow_t)
            S["maps"][seq] = {k: S["maps"][seq][k] for k in ("flow", "alpha")}
        no_t = [i for i in ids if i not in per]
        for seq in no_t:
            one(seq, None)
    finally:
        gt.close()
    per_s = [per[i][0] for i in ids]
    per_c = [per[i][1] for i in ids]
    use = {k: sum(per[i][2][k] for i in ids) for k in per[ids[0]][2]}
    use["n_valid"] = sum(p["all"]["n"] for p in per_s)
    stats = frozen_stats(per_s, per_c, regions, FROZEN_STATS[ds], boot, ds)
    switched = {"valid_pct": 100.0 * use["sw_all"] / use["n_valid"],
                "image_pct": 100.0 * use["sw_img"] / use["px"],
                "per_region_pct": {r: (100.0 * use[f"sw_{r}"] / stats[r]["n_pixels"]
                                       if stats[r]["n_pixels"] else None) for r in regions},
                "counts": use}
    try:
        gtb = ek.gt_bundle(ds, gt_root, root)
    except (OSError, ValueError, rg.IntegrityError) as ex:
        gtb = {"name": rg.GT_BUNDLE[ds], "gt_pack_sha": None, "integrity": f"unknown: {ex}"}
    return {"dataset": ds, "split": RT.SPLIT_OF[ds], "n_images": len(ids),
            "teacher_images": len(ids) - len(no_t), "teacher_missing_ids": no_t, "gt": gtb,
            "S_pred_sha": S["pred_sha"], "T_pred_sha": t_sha, "regions": stats,
            "switched": switched}


def frozen_render(doc: dict) -> str:
    h = doc["header"]
    b = h["statistics"]
    out = [f"# IDEA-004 frozen rule ({h['rule']['text']})", "",
           f"Gain = S - frozen (positive = better); paired image-bootstrap "
           f"{round(100 * (1 - b['alpha']))}% CI (B={b['B']}, seed {b['seed']}).", ""]
    for res in doc["results"]:
        ds = res["dataset"]
        out += [f"## {ds} ({res['split']}, n={res['n_images']}, teacher on "
                f"{res['teacher_images']}); switched {res['switched']['valid_pct']:.2f}% of "
                "GT-valid px, " f"{res['switched']['image_pct']:.2f}% of all px", "",
                "| region | metric | S | frozen | gain [CI] |", "|---|---|---|---|---|"]
        for r, rec in res["regions"].items():
            if rec.get("status") != "ok":
                out.append(f"| {r} | - | n=0 | | |")
                continue
            for k in FROZEN_STATS[ds]:
                e = rec[k]
                ci = e["gain_ci"]
                out.append(f"| {r} | {e['label']} | {e['S']:.3f} | {e['frozen']:.3f} | "
                           f"{e['gain']:.3f}" + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "")
                           + " |")
        out.append("")
    if h.get("reproduction"):
        rp = h["reproduction"]
        out.append(f"Reproduction (KITTI-15 dev vs EV-0402 in-sample): Fl-all gain "
                   f"{rp['gain_pp_all']['here']:.4f} pp = {rp['gain_pp_all']['ev0402']:.4f} pp "
                   f"(|diff| {rp['gain_pp_all']['abs_diff']:.1e}).")
    out += [f"Kill read (IDEA-004, user decides): {h['kill_read']['text']}", ""]
    return "\n".join(out)


def run_frozen(a) -> int:
    root = ek.repo_root()
    gt_root = Path(a.gt_root)
    C.refuse_sealed(gt_root, CB.DEFAULT_SPEC)
    spec_p = root / CB.DEFAULT_SPEC
    spec = json.loads(spec_p.read_text("utf-8"))
    boot = RT.bootstrap_settings(a, root)
    json_out = (None if a.no_json else
                RT.out_path(a.json_out or FROZEN_JSON, root, ".json", is_ours=_is_ours_frozen))
    if not gt_root.is_dir():
        raise Fail(RT.EXIT_PRE, f"{gt_root}: GT root not found")
    students_e, named_e, _ = CB._resolve(spec, root)
    t_hit = [e for e in named_e if CB.tkey(e) == FROZEN_TEACHER]
    if len(t_hit) != 1:
        raise Fail(RT.EXIT_PRE, f"{FROZEN_TEACHER}: not exactly once in {spec['teachers_spec']}")
    t_entry = t_hit[0]
    s_entry = next(e for e in students_e if e["source"] == "u2flow")
    results = [frozen_dataset(ds, root, gt_root, s_entry, t_entry, boot)
               for ds in ("kitti12", "kitti15")]
    k12, k15 = results
    for res, names, o3 in ((k12, {"R1": "K12-R1", "R3": "K12-R3"}, True),
                           (k15, {"R1": "R1", "R3": "R3"}, False)):
        s_row = {"regions": {mine: {"fl": res["regions"][mine]["fl"]["S"],
                                    "n_pixels": res["regions"][mine]["n_pixels"],
                                    **({"out3": res["regions"][mine]["out3"]["S"]} if o3 else {})}
                             for mine in names.values()}}
        res["S_crosscheck"] = CB.crosscheck_region_table(root, res["S_pred_sha"], s_row,
                                                         ds=res["dataset"], names=names, out3=o3)
    k15["S_crosscheck_composite_bound"] = oracle_context(root, k15["S_pred_sha"],
                                                         k15["regions"]["all"]["fl"]["S"])
    repro = frozen_repro(root, {r: k15["regions"][r] for r in REGIONS},
                         k15["switched"]["counts"],
                         k15["S_pred_sha"], k15["T_pred_sha"], boot)
    esha, _ = rg.eval_sha(rg.CODE_ROOT)
    header = {
        "tool": TOOL, "mode": FROZEN_MODE, "idea": "IDEA-004", "standard": "measured",
        "rule": {"teacher": FROZEN_TEACHER, "alpha_gt": FROZEN_ALPHA_GT,
                 "d_le_px": FROZEN_D_LE_PX, "d_gt_px": None, "source": "EV-0402 in-sample choice "
                 "(reports/p1_region/arbitration_cv.json), frozen by IDEA-004 before any KITTI-12 "
                 "number; no tuning here",
                 "text": f"T={FROZEN_TEACHER} where T finite and alpha > {FROZEN_ALPHA_GT:g} and "
                         f"|T - S| <= {FROZEN_D_LE_PX:g} px, else S"},
        "teacher": {"key": FROZEN_TEACHER, "path": t_entry["path"],
                    "derived_from": DERIVED_FROM["selftri/native"]},
        "S": {"source": "u2flow kitti.pth", "path": s_entry["path"],
              "flow_key": s_entry.get("flow_key"),
              "resizing": "native_10to11 (resize_flow to the GT's (H, W), the evaluator path); "
                          "alpha_10to11 bilinear align_corners=True, no value scaling (alpha_"
                          "native, the dev path)"},
        "metric_version": rg.METRIC_VERSION, "eval_sha": esha, "git": ek.git_info(),
        "regions_provider": RT.V1Provider.name,
        "regions_note": "KITTI-12: all = V_occ, noc = V_noc, occ = V_occ & ~V_noc, K12-R1 = V_noc "
                        "(= noc), K12-R3 = occ & out-of-frame; K12-R2 withheld-S6 (R2_NEAR_EMPTY "
                        "None). The real K12 GT fails regions.py's S3 (flow_noc != flow_occ on "
                        "V_noc, EV-0300 / [M 2026-10-06]): region_table.V1Provider scores noc "
                        "against flow_occ (evaluate.py convention), so S3 is vacuous, not "
                        "weakened. KITTI-15 dev: all, R1, R3 (as EV-0402).",
        "metrics": {"fl": "Fl px-pooled; KITTI-12 = Fl(K15-rule) (EPE > 3 and > 5% of |F_gt|, "
                          "no epsilon, evaluate.py:860); never presented as Out-All",
                    "out3": "EPE > 3 px, px-pooled (KITTI-12 official rule): Out-All on 'all'. "
                            "On 'noc' (= K12-R1) it is scored against flow_occ, NOT flow_noc, so "
                            "it is not the official Out-Noc",
                    "epe_pooled": "EPE px-pooled", "epe_img_mean": "EPE mean over images with "
                    "n_R > 0"},
        "statistics": {"unit": "image", "gain": "S - frozen, stats.paired_image_bootstrap",
                       **{k: boot[k] for k in ("B", "seed", "alpha", "seed_source",
                                               "settings_source")}},
        "code": CB.code_identity(root, [HERE / "arbitration_cv.py", HERE / "composite_bound.py",
                                        HERE / "region_table.py", spec_p]),
        "reproduction": repro,
        "inputs_never_read": "KITTI-15 held-out and all-200 (region_table seal checks)",
        "ledger": "no results row (hard rule 7); record as note.py evidence --kind measurement",
    }
    header["kill_read"] = frozen_kill_read(k12["regions"])
    doc = RT._jsonable({"header": header, "results": results})
    if json_out:
        ek.atomic_write(json_out, json.dumps(doc, indent=1) + "\n")
    sys.stdout.write(frozen_render(doc) + "\n")
    return RT.EXIT_OK


def check_frozen_args(a) -> None:
    """--frozen IDEA-004 applies ONE pre-declared rule: a --frozen-* value other than the frozen
    constant, or any teacher, gate or fold option, is a usage error (exit 64); the --frozen-*
    flags are refused without --frozen."""
    vals = (("--frozen-teacher", a.frozen_teacher, FROZEN_TEACHER),
            ("--frozen-alpha-gt", a.frozen_alpha_gt, FROZEN_ALPHA_GT),
            ("--frozen-d-le-px", a.frozen_d_le_px, FROZEN_D_LE_PX))
    if not a.frozen:
        given = [f for f, v, _ in vals if v is not None]
        if given:
            raise Fail(RT.EXIT_USAGE, f"{', '.join(given)}: only with --frozen IDEA-004")
        return
    for flag, val, want in vals:
        if val is not None and val != want:
            raise Fail(RT.EXIT_USAGE, f"{flag} {val}: IDEA-004 froze {want} (EV-0402); the "
                       "frozen mode applies no other value (no tuning)")
    tuning = [f for f, v in (("--teachers", a.teachers), ("--gate-geom", a.gate_geom or None),
                             ("--no-gate", a.no_gate or None), ("--fold-seed", a.fold_seed))
              if v is not None]
    if tuning:
        raise Fail(RT.EXIT_USAGE, f"{', '.join(tuning)}: a selector/tuning option is refused in "
                   "the frozen mode (IDEA-004 fixes teacher and thresholds)")


def run(a) -> int:
    root = ek.repo_root()
    gt_root = Path(a.gt_root)
    C.refuse_sealed(gt_root, CB.DEFAULT_SPEC)
    spec_p = root / CB.DEFAULT_SPEC
    spec = json.loads(spec_p.read_text("utf-8"))
    boot = RT.bootstrap_settings(a, root)
    json_out = (None if a.no_json else
                RT.out_path(a.json_out or DEFAULT_JSON, root, ".json", is_ours=_is_ours))
    a.teachers = a.teachers or ",".join(DEFAULT_TEACHERS)
    a.fold_seed = DEFAULT_FOLD_SEED if a.fold_seed is None else a.fold_seed
    if not gt_root.is_dir():
        raise Fail(RT.EXIT_PRE, f"{gt_root}: GT root not found")
    students_e, named_e, _ = CB._resolve(spec, root)
    want = [t.strip() for t in a.teachers.split(",") if t.strip()]
    teach_e = [e for e in named_e if f"{e['source']}/{e['pose']}" in want]
    if {f"{e['source']}/{e['pose']}" for e in teach_e} != set(want):
        raise Fail(RT.EXIT_USAGE, f"--teachers {want}: not all among the named rigid teachers")
    s_entry = next(e for e in students_e if e["source"] == "u2flow")
    geom = {**GATE_GEOM, **{k: Path(v) for k, v in (g.split("=", 1) for g in a.gate_geom)}}
    dev_l, held_l = ek.split_ids(root)
    dev, held = set(dev_l), set(held_l)
    ids = sorted(dev_l)
    folds = make_folds(ids, a.fold_seed)
    gt = rg.GTSource(DS, root=gt_root)
    gtc = RT.GTCache(gt, RT.V1Provider(), DS, len(ids) + 1)
    rows, gate_info, prov = [], {}, []
    try:
        def mask_of(i):
            shape, _, packed = gtc.get(i)
            return shape, {r: RT.unpack(packed[r], shape) for r in REGIONS}

        S = load_student(s_entry, root, dev, held, gtc)
        if sorted(S["maps"]) != ids:
            raise Fail(RT.EXIT_ERR, f"u2flow: {len(S['maps'])} dev images, need {len(ids)}")
        base = [base_counts(S["maps"][i]["out"], mask_of(i)[1]) for i in ids]
        shapes = {i: S["maps"][i]["shape"] for i in ids}
        n = len(ids)
        draws = (np.random.default_rng(boot["seed"]).integers(0, n, (boot["B"], n))
                 if boot["B"] > 0 else None)
        s_row = CB.arm_stats(base, base, {**boot, "B": 0}, None)
        evi = None
        tables = {}
        for e in teach_e:
            key = CB.tkey(e)
            T = {}
            p, streams = RT.open_entry({**e, "datasets": [DS]}, root, dev, held)
            factory, pred_sha, _ = streams[DS]
            for seq, flow in factory():
                fin, _, out = RT.pixel_maps(flow, shapes[seq], gtc.get(seq)[1], DS)
                T[seq] = (np.asarray(flow, np.float32), fin, out)
            prov.append({"key": key, "path": e["path"], "pred_sha": pred_sha, "n_images": len(T),
                         "derived_from": DERIVED_FROM.get(f"{e['source']}/{e['pose']}",
                                                          "unstated")})
            gate = None
            if not a.no_gate:
                if evi is None:
                    from p1lib import cache as CA  # noqa: PLC0415
                    evi = CA.CacheIndex(Path(s_entry["path"]))
                gate, skipped = gate_masks(e["source"], geom[e["source"]], e["window"], ids,
                                           shapes, evi, root)
                gate_info[key] = {"geometry": str(geom[e["source"]]), "frames": len(gate),
                                  "skipped": skipped, "tree": "grid (no SAM at frame 10, EV-0300)",
                                  "k": "calib"}
                print(f"arbitration_cv: gate {key}: {len(gate)} frames, {len(skipped)} skipped",
                      file=sys.stderr)
            tables[key] = selector_tables(S, ids, T, gate, mask_of)
            del T
    finally:
        gt.close()
    for key, tabs in tables.items():
        for sel, (Qc, ch) in tabs.items():
            for kind in ("cv", "in"):
                rows.append(make_row(f"{sel}[{key}]", [(key, sel, Qc, ch)], base, folds, boot,
                                     draws, kind))
    for sel in SELECTORS:
        mem = [(k, sel, *t[sel]) for k, t in tables.items() if sel in t]
        if mem:
            for kind in ("cv", "in"):
                rows.append(make_row(f"nested[{sel}]", mem, base, folds, boot, draws, kind))
    mem = [(k, sel, *t[sel]) for k, t in tables.items() for sel in t]
    for kind in ("cv", "in"):
        rows.append(make_row("nested[all]", mem, base, folds, boot, draws, kind))
    s_fl = s_row["all"]["fl"]
    esha, _ = rg.eval_sha(rg.CODE_ROOT)
    try:
        gtb = ek.gt_bundle(DS, gt_root, root)
    except (OSError, ValueError, rg.IntegrityError) as ex:
        gtb = {"name": rg.GT_BUNDLE[DS], "gt_pack_sha": None, "integrity": f"unknown: {ex}"}
    header = {
        "tool": TOOL, "idea": "IDEA-003", "precondition": "EV-0401 (composite_bound.json)",
        "standard": "measured", "split": "dev", "dataset": DS, "n_images": len(ids),
        "agg": "px-pooled", "metric_version": rg.METRIC_VERSION,
        "regions_provider": RT.V1Provider.name, "eval_sha": esha, "gt": gtb, "git": ek.git_info(),
        "code": CB.code_identity(root, [HERE / "arbitration_cv.py", HERE / "composite_bound.py",
                                        HERE / "region_table.py", HERE / "arbitrate.py",
                                        HERE / "arbitrate_run.py", spec_p]),
        "S": {"source": "u2flow", "path": s_entry["path"], "pred_sha": S["pred_sha"],
              "fl_all": s_fl, "fl_R1": s_row["R1"]["fl"], "fl_R3": s_row["R3"]["fl"],
              "crosscheck": oracle_context(root, S["pred_sha"], s_fl)},
        "teachers": prov, "gate": gate_info if not a.no_gate else "skipped (--no-gate)",
        "folds": {"seed": a.fold_seed, "unit": "image id (each KITTI-15 id is its own scene)",
                  "A": [ids[i] for i in folds[0]], "B": [ids[i] for i in folds[1]]},
        "grids": {"alpha_gt": A_GRID.tolist(), "fb_gt_px": FB_GRID.tolist(),
                  "d_edges_px": D_EDGES.tolist(),
                  "note": "fixed before any dev number; +inf = never switch, -inf = switch "
                          "wherever T is finite; the alpha range read from 2 KITTI-12 items"},
        "objective": "max Fl-all gain (pooled) on the training fold; ties: fewer switched "
                     "GT-valid pixels, then grid order",
        "resizing": "S native_10to11 (resize_flow, evaluator path); alpha bilinear "
                    "align_corners=True, no value scaling; fb on the grid with "
                    "core.utils.warp_utils.flow_warp, u*W/832, v*H/256, norm resized likewise; "
                    "out-of-grid target = +inf (flag resized, >= 0.5)",
        "statistics": {"unit": "image", "gain_ci": "stats.paired_image_bootstrap via "
                       "composite_bound.arm_stats; fold thresholds held fixed (tuning not "
                       "resampled)", **{k: boot[k] for k in ("B", "seed", "alpha", "seed_source",
                                                             "settings_source")}},
        "s6": "R2 withheld-S6 (regions.R2_NEAR_EMPTY is None); Fl-all counts every V_occ pixel",
        "ledger": "no results row (hard rule 7); record as note.py evidence --kind measurement",
    }
    header["kill_read"] = kill_read(rows)
    doc = RT._jsonable({"header": header, "rows": rows})
    if json_out:
        ek.atomic_write(json_out, json.dumps(doc, indent=1) + "\n")
    sys.stdout.write(render(doc) + "\n")
    return RT.EXIT_OK


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gt-root", default="/Volumes/Shin/U2FLOW_data/gt")
    ap.add_argument("--teachers", help=f"default {','.join(DEFAULT_TEACHERS)}")
    ap.add_argument("--gate-geom", action="append", default=[], help="SOURCE=GEOMETRY_CACHE_DIR")
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--fold-seed", type=int, help=f"default {DEFAULT_FOLD_SEED}")
    ap.add_argument("--json-out", help=f"default {DEFAULT_JSON} ({FROZEN_JSON} with --frozen)")
    ap.add_argument("--frozen", choices=["IDEA-004"],
                    help="IDEA-004's frozen rule (no tuning) on KITTI-12 and the K15 dev half")
    ap.add_argument("--frozen-teacher", help=f"optional; must be {FROZEN_TEACHER}")
    ap.add_argument("--frozen-alpha-gt", type=float, help=f"optional; must be {FROZEN_ALPHA_GT}")
    ap.add_argument("--frozen-d-le-px", type=float, help=f"optional; must be {FROZEN_D_LE_PX}")
    ap.add_argument("--no-json", action="store_true")
    ap.add_argument("--B", type=int)
    ap.add_argument("--alpha", type=float)
    ap.add_argument("--seed", type=int)
    try:
        a = ap.parse_args(argv)
    except SystemExit as ex:
        return RT.EXIT_USAGE if ex.code else RT.EXIT_OK
    try:
        check_frozen_args(a)
        return run_frozen(a) if a.frozen else run(a)
    except Fail as e:
        print(f"arbitration_cv: {'REFUSED (seal)' if e.code == RT.EXIT_SEAL else 'error'}: {e}",
              file=sys.stderr)
        return e.code
    except C.Stop as e:
        print(f"arbitration_cv: {e}", file=sys.stderr)
        return e.code
    except rg.IntegrityError as e:
        print(f"arbitration_cv: integrity: {e}", file=sys.stderr)
        return RT.EXIT_INTEGRITY
    except (OSError, ValueError, KeyError) as e:
        print(f"arbitration_cv: {type(e).__name__}: {e}", file=sys.stderr)
        return RT.EXIT_ERR


if __name__ == "__main__":
    sys.exit(main())
