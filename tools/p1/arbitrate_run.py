#!/usr/bin/env python3
"""arbitrate_run.py: runs tools/p1/arbitrate.py (the SAM-tree gate and its comparison variants) on
REAL caches on the Mac (CPU) and reports pooled R2+R3 coverage and R4 leakage per source and
variant, with the fold failures and fold_poses_disagree warnings printed beside every figure
(plan row W3-01, cycle CYC-001 step 2). It computes numbers and writes JSON under
reports/p1_region/ only: no ledger row (hard rule 7), no VM.

What it does (one window of the plan = one frame, frame 10, with its 10->11 and 10->09 pairs)
    geometry   a teacher cache (p1lib/geomio.py GeomIndex: evalmapa/evalpi3/evalcolmap/evalselftri
               v2): depth of frame 10 and the teacher poses of frames 9, 10, 11, brought to the
               256 x 832 grid with the window's ONE K (--k; eq. 0) as refine_pose.py does
    evidence   U2Flow's dump_eval cache (flow, alpha, vis of 10to11 and 10to09), sequence level
    E^P        e^alpha <= 2 and vis and valid_T and not OOF and not O_T [P A4]; valid_T, OOF and
               O_T from rigid.render_pair of the pair (the same call refine_pose.py makes)
    tree       'sam': the SAM masks of frame 10, 'grid': no SAM, the 32 x 32 cells of the whole
               frame (A1's rule for pixels no mask covers, applied to every pixel: a diagnostic
               floor for the gate, never the method). Without SAM files the 'sam' tree is skipped
               and says so (see "What blocks SAM" below)
    variants   none, per_pixel_static, per_pixel_reject, gate, oracle
               (arbitrate.variants_for_frame); the oracle is the GT moving mask (KITTI-15
               obj_map > 0, audit only) and is absent for KITTI-12 (no obj_map).
               Per-pixel variants judge the forward pair with the out-of-fold rigid flow of
               the tree they sit beside, so every variant of a tree sees the same flow
    regions    R1, R2, R3 from tools/eval/regions.py (arbitrate.regions_from_gt); 'R4' is every FG
               pixel until regions_v2 lands, and every figure says R4, never R4m. KITTI-12 has no
               R4: its leakage is n/a, only coverage counts
    report     per (source, tree, variant): pooled counts n and k (never averaged rates), coverage
               of R2+R3, leakage onto R4, and beside them n_folds, failed folds, pose-disagreement
               warnings and withheld pixels of that tree. 'none' and 'oracle' do not depend on a
               fold; they carry the figures of the tree they were computed under

What blocks SAM [M 2026-10-06]. The sam:v1 pack holds the flat full_seg label maps and the key
objects of every loader frame and never frames 09-12 (sam_masks.py: the loaders skip them), so
there is NO SAM file for frame 10, the evaluated frame of every window; the --rle JSONs (the
unflattened hierarchy A1 asks for) were not packed either, and no SAM checkpoint is on the Mac.
sam_tree() reads whatever the pack has (full_seg labels as leaf masks, key objects as larger masks:
a two-level tree, weaker than the AMG hierarchy) and returns None when the frame's PNG is absent.
Closing the gap needs sam_masks.py to produce frame 10 of the eval windows (a VM job, colab-fleet).

How to invoke (repo root; the venv of the Mac tools):
    D=/Volumes/Shin/U2FLOW_data
    .venv/bin/python tools/p1/arbitrate_run.py run \
        --source mapanything=$D/work/p1geom/derived/evalmapa/v2 --k calib \
        --evidence-cache $D/runs/p1-dump-eval-u2flow-r1/out/cache/diag/u2flow \
        --gt-root $D/gt [--sam-dir DIR] \
        [--dataset kitti15|kitti12] [--n 10 | --all] [--window w03-11] \
        --json-out reports/p1_region/arbitrate_k15_n10.json
    A SOURCE is NAME=PATH, repeatable. --n N takes N dev windows spread evenly over the sorted dev
    ids (no cherry-picking); --all takes every dev window (KITTI-12: ids 000000-000193).
Exit codes: 0 ok · 1 a check failed · 2 precondition (missing cache, GT, plan) · 50 a held-out or
sealed path or id (STOP, AGENTS.md section 4) · 64 usage.

Why it is built this way
- The gate is a library (arbitrate.py); this file only feeds it. No threshold lives here: every
  constant is arbitrate.py's (the proposal's), and the per-source K choice (--k) is W2-14's.
- Dev half and KITTI-12 only: a KITTI-15 id outside splits/k15_split_v1.json's dev_ids is exit 50
  before anything is read; the GT root is the verified dev GT and every path goes through
  refuse_sealed.
- One teacher pose per pair (the cache's own, not T_ref): the gate's folds replace T_ref (A3), and
  pair_w is 1.0 because refine_pose.py's pair weight needs a refined pose (W3-03's join).
- Counts are pooled before any rate (statistics.md); frames that cannot run (no pose, no cache
  item, no evidence) are listed with a reason, never dropped silently.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "tools" / "eval"))

import arbitrate as AR  # noqa: E402
import eval_kitti as ek  # noqa: E402
import p1_windows as MW  # noqa: E402
import regions as rg  # noqa: E402
import rigid  # noqa: E402
from p1lib import cache as CA  # noqa: E402
from p1lib import common as C  # noqa: E402
from p1lib import geomio as GI  # noqa: E402

GRID = (256, 832)
ALPHA_MAX = float(np.log(2.0))   # [P A4] e^alpha <= 2 (refine_pose.ALPHA_MAX is the same number)
PAIRS = (("10to11", 11, "fwd"), ("10to09", 9, "bwd"))   # the forward pair first
TREES = ("sam", "grid")
VARIANTS = AR.VARIANT_ORDER
OUT_DIR = REPO / "reports" / "p1_region"


# ------------------------------------------------------------------------------ SAM
def sam_tree(sam_dir: Path | None, row: dict, frame: int, grid_hw=GRID):
    """Tree of one frame from the sam:v1 pack layout, or None when the frame's full_seg PNG is
    absent (frames 09-12 are never produced). Leaf masks: every label > 0 of the full_seg map;
    larger masks: every key-object channel (uint8 0/255, npz key 'key_objects'); both
    nearest-resized to the grid. This is the pack's two levels, not the AMG hierarchy."""
    if sam_dir is None:
        return None
    lab = GI.sam_labels(Path(sam_dir), row, frame, grid_hw)
    if lab is None:
        return None
    masks = [lab == v for v in np.unique(lab) if v > 0]
    ko = (Path(sam_dir) / f"{GI.SAM_DS[row['ds']]}_mv_sam1_key_objects" / row["sub"] / row["side"]
          / f"{row['seq']}_{frame:02d}.npz")
    if ko.is_file():
        with np.load(ko) as z:
            arr = z["key_objects"]
        H, W = grid_hw
        for k in range(arr.shape[2] if arr.ndim == 3 else 0):
            m = cv2.resize((arr[:, :, k] > 127).astype(np.uint8), (W, H),
                           interpolation=cv2.INTER_NEAREST) > 0
            if m.any() and not m.all():
                masks.append(m)
    return AR.build_tree_from_masks(masks, grid_hw)


def grid_tree(grid_hw=GRID):
    """No SAM: the 32 x 32 cells of the whole frame (A1's cut of pixels no mask covers)."""
    return AR.build_tree_from_masks([], grid_hw)


# ------------------------------------------------------------------------------ one frame
class Skip(Exception):
    """A frame that cannot run: the reason is listed in the report."""


def frame_inputs(geo, evi, row: dict, kch: str, grid=GRID) -> dict:
    """depth, K, valid_T and the evidence pairs of one window's frame 10 on the grid."""
    got = geo.get(row["id"])
    if got is None:
        raise Skip("no geometry item")
    ga, gm = got
    ev_got = evi.get(GI.evidence_id(row))
    if ev_got is None:
        raise Skip("no evidence item")
    ea, _ = ev_got
    if 10 not in gm["frames"]:
        raise Skip("window without frame 10")
    i10 = gm["frames"].index(10)
    K = GI.window_K(ga, gm, i10, kch)                     # ONE K for the window (eq. 0)
    g10 = GI.geometry_frame(ga, gm, i10, grid, K_model=K)
    if not g10["pose_ok"]:
        raise Skip("no teacher pose at frame 10")
    pairs, names = [], []
    valid_T = None
    for d, other, name in PAIRS:
        if other not in gm["frames"] or f"flow_{d}" not in ea:
            continue
        g2 = GI.geometry_frame(ga, gm, gm["frames"].index(other), grid, K_model=K)
        if not g2["pose_ok"]:
            continue
        M = rigid.relative_pose(g10["pose_c2w"], g2["pose_c2w"])
        rend = rigid.render_pair(g10["inv"], g2["inv"], g10["K"], M, grid, valid1=g10["valid"],
                                 valid2=g2["valid"])
        e = GI.evidence(ea, d, grid)
        ev = rend["valid_12"] & ~rend["oof_12"] & ~rend["occ_12"]      # valid_T, not OOF, not O_T
        if "alpha" in e:
            ev &= e["alpha"] <= ALPHA_MAX
        if "vis" in e:
            ev &= e["vis"]
        ev &= np.isfinite(e["flow"]).all(axis=0)
        pairs.append((e["flow"], ev))
        names.append(name)
        if valid_T is None:
            valid_T = rend["valid_12"]
    if not pairs:
        raise Skip("no pair with poses and evidence")
    depth = rigid.depth_from_inv(np.where(g10["valid"], g10["inv"], np.nan))
    return {"id": row["id"], "depth": depth, "K": g10["K"], "valid_T": valid_T,
            "pair_inputs": pairs, "pair_names": names,
            "seed": MW.__dict__.get("seed_of", lambda *a: 0)(row["id"]) if False else
            int(abs(hash_id(row["id"])) % 2**31)}


def hash_id(item_id: str) -> int:
    """A stable per-window seed (sha256 of the id, not Python's salted hash)."""
    import hashlib
    return int(hashlib.sha256(item_id.encode()).hexdigest()[:8], 16)


def regions_for(gt, dataset: str, grid=GRID) -> dict:
    """R1, R2, R3 and 'R4' (every FG pixel; labelled R4, not R4m) on the grid."""
    if dataset == "kitti15":
        return AR.regions_from_gt(gt, grid)
    m, _ = rg.masks(gt, "kitti12")
    out = {"R1": m["K12-R1"], "R2": m["K12-R2"], "R3": m["K12-R3"],
           "R4m": np.zeros(gt.shape, bool), "r4m_source": "n/a (KITTI-12 has no obj_map)",
           "r4_label": "R4"}
    H, W = grid
    for k in ("R1", "R2", "R3", "R4m"):
        out[k] = cv2.resize(out[k].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    return out


def run_tree(fi: dict, tree, regions: dict, oracle: bool, p: AR.Params | None = None) -> dict:
    """Every variant of one frame under one tree: counts, plus that tree's fold report."""
    pairs = [AR.build_pair_evidence(fi["depth"], fi["K"], fe, ev, tree, fi["seed"] + i,
                                    name=fi["pair_names"][i], p=p)
             for i, (fe, ev) in enumerate(fi["pair_inputs"])]
    vs = AR.variants_for_frame(tree, pairs, fi["valid_T"],
                               regions["R4m"] if oracle else None, 1.0, p)
    g = vs.pop("_gate")
    return {"counts": {k: AR.coverage_leakage(v["sel"], regions, v["w"]) for k, v in vs.items()},
            "n_folds": 2 * len(pairs), "failed_folds": len(g.fold_failures),
            "disagree": len(g.fold_warnings), "no_pose_px": g.no_pose_px,
            "n_pairs": len(pairs), "fold_failures": g.fold_failures,
            "fold_warnings": g.fold_warnings}


# ------------------------------------------------------------------------------ selection
def dev_ids_for(dataset: str, root: Path) -> list[str]:
    if dataset == "kitti12":
        return list(ek.K12_IDS)
    dev, _held = ek.split_ids(root)
    return sorted(dev)


def select_rows(plan: dict, dataset: str, win: str, ids: list[str], n: int | None) -> list[dict]:
    """The plan's rows of (dataset, window) whose seq is in ids (dev only), n of them spread
    evenly over the sorted ids, or all of them when n is None."""
    by_seq = {r["seq"]: r for r in plan["eval_windows"]
              if r["ds"] == dataset and r["win"] == win and r["sub"] == "training"
              and r["side"] == "image_2"}
    rows = [by_seq[i] for i in sorted(ids) if i in by_seq]
    if n is None or n >= len(rows):
        return rows
    pick = np.unique(np.round(np.linspace(0, len(rows) - 1, n)).astype(int))
    return [rows[i] for i in pick]


def check_dev(dataset: str, rows: list[dict], root: Path) -> None:
    """Hard rule 4: a KITTI-15 window outside the dev half is exit 50 before anything is read."""
    if dataset != "kitti15":
        return
    dev, _ = ek.split_ids(root)
    bad = [r["seq"] for r in rows if r["seq"] not in set(dev)]
    if bad:
        raise C.Stop(C.EXIT_SEAL, f"{len(bad)} KITTI-15 window(s) outside the dev half "
                     f"(e.g. {bad[0]}): held-out is never an input (hard rule 4)")


# ------------------------------------------------------------------------------ one source
def run_source(name: str, geo, evi, rows: list[dict], gt_src, dataset: str, kch: str,
               sam_dir: Path | None, p: AR.Params | None = None, grid=GRID) -> dict:
    per = {t: [] for t in TREES}
    folds = {t: {"n_folds": 0, "failed_folds": 0, "disagree": 0, "no_pose_px": 0, "frames": 0,
                 "n_pairs": 0, "failures": [], "warnings": []} for t in TREES}
    skipped, no_sam = [], []
    for row in rows:
        try:
            fi = frame_inputs(geo, evi, row, kch, grid)
        except Skip as e:
            skipped.append({"id": row["id"], "reason": str(e)})
            continue
        try:
            gt = gt_src.read(row["seq"], need_obj=dataset == "kitti15")
            regions = regions_for(gt, dataset, grid)
        except Exception as e:  # noqa: BLE001
            # the evaluator's own sanity check (S1-S3): listed, not hidden. Matched by name:
            # arbitrate re-executes regions.py per call, so the class is never the same object
            if type(e).__name__ != "RegionError":
                raise
            skipped.append({"id": row["id"], "reason": f"regions.py: {e}"})
            continue
        trees = {"grid": grid_tree(grid)}
        st = sam_tree(sam_dir, row, 10, grid)
        if st is None:
            no_sam.append(row["id"])
        else:
            trees["sam"] = st
        for t, tree in trees.items():
            res = run_tree(fi, tree, regions, oracle=dataset == "kitti15", p=p)
            per[t].append(res["counts"])
            f = folds[t]
            for k in ("n_folds", "failed_folds", "disagree", "no_pose_px", "n_pairs"):
                f[k] += res[k]
            f["frames"] += 1
            f["failures"] += [{"frame": row["id"], **x} for x in res["fold_failures"]]
            f["warnings"] += [{"frame": row["id"], **x} for x in res["fold_warnings"]]
    out = {"source": name, "k": kch, "n_windows": len(rows), "skipped": skipped,
           "frames_without_sam": no_sam, "trees": {}}
    for t in TREES:
        if not per[t]:
            out["trees"][t] = None
            continue
        variants = {}
        for v in VARIANTS:
            if not all(v in c for c in per[t]):
                continue
            agg = AR.aggregate([c[v] for c in per[t]])
            variants[v] = {"counts": agg, "rates": AR.rates(agg)}
        out["trees"][t] = {"variants": variants, "folds": folds[t]}
    return out


# ------------------------------------------------------------------------------ report
def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def format_report(doc: dict) -> list[str]:
    lines = [f"ARBITRATE_RUN {doc['dataset']} window {doc['window']} windows={doc['n_windows']} "
             f"(R4 = every FG pixel until regions_v2 lands; counts pooled, not averaged)"]
    for s in doc["sources"]:
        lines.append(f"  source {s['source']} (K {s['k']}): windows {s['n_windows']}, "
                     f"skipped {len(s['skipped'])}, without SAM at frame 10 "
                     f"{len(s['frames_without_sam'])}")
        for t in TREES:
            tr = s["trees"][t]
            if tr is None:
                lines.append(f"    tree {t}: not run (no frame)")
                continue
            f = tr["folds"]
            beside = (f"folds {f['n_folds']} failed {f['failed_folds']} "
                      f"pose-disagree {f['disagree']} withheld_px {f['no_pose_px']}")
            for v, d in tr["variants"].items():
                c, r = d["counts"], d["rates"]
                lines.append(
                    f"    [{t}] {v:17s} cov R2+R3 {_pct(r['coverage_R2+R3'])} "
                    f"({c['k_R2+R3']}/{c['n_R2+R3']})  leak {c['r4_label']} "
                    f"{_pct(r['leakage_R4m'])} ({c['k_R4m']}/{c['n_R4m']})  | {beside}")
    return lines


def out_path(arg: str) -> Path:
    p = Path(arg)
    p = p if p.is_absolute() else REPO / p
    r = p.resolve()
    if OUT_DIR.resolve() not in r.parents:
        raise C.Stop(C.EXIT_USAGE, f"{arg}: output only under reports/p1_region/")
    return r


# ------------------------------------------------------------------------------ CLI
def run(a) -> int:
    paths = [a.evidence_cache, a.gt_root, a.sam_dir] + [s.split("=", 1)[-1] for s in a.source]
    C.refuse_sealed(*paths)
    gt_root = Path(a.gt_root)
    if not gt_root.is_dir():
        raise C.Stop(C.EXIT_PRE, f"{gt_root}: GT root not found")
    plan = MW.load_plan(Path(a.plan))
    ids = dev_ids_for(a.dataset, REPO)
    rows = select_rows(plan, a.dataset, a.window, ids, None if a.all else a.n)
    if not rows:
        raise C.Stop(C.EXIT_PRE, "no window selected")
    check_dev(a.dataset, rows, REPO)
    gt_src = rg.GTSource(a.dataset, root=gt_root)
    evi = CA.CacheIndex(Path(a.evidence_cache))
    sam_dir = Path(a.sam_dir) if a.sam_dir else None
    doc = {"schema": 1, "tool": "tools/p1/arbitrate_run.py", "dataset": a.dataset,
           "window": a.window, "n_windows": len(rows), "selection": "all dev" if a.all else
           f"{a.n} evenly spaced over the sorted dev ids", "ids": [r["seq"] for r in rows],
           "r4_note": "R4 = every FG pixel; R4m exists only after regions_v2 lands",
           "params": "arbitrate.Params defaults (the proposal's constants)",
           "sam_dir": str(sam_dir) if sam_dir else None, "sources": []}
    for spec in a.source:
        name, _, path = spec.partition("=")
        geo = GI.GeomIndex(Path(path))
        kch = GI.resolve_k(geo.get(next(r["id"] for r in rows if geo.get(r["id"])))[1], a.k) \
            if any(geo.get(r["id"]) for r in rows[:3]) else (a.k or "teacher")
        doc["sources"].append(run_source(name, geo, evi, rows, gt_src, a.dataset, kch, sam_dir))
    for line in format_report(doc):
        print(line)
    if a.json_out:
        out = out_path(a.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=1) + "\n", "utf-8")
        print(f"ARBITRATE_RUN_JSON {out}")
    return C.EXIT_OK


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--source", action="append", required=True, help="NAME=GEOMETRY_CACHE_DIR")
    r.add_argument("--evidence-cache", required=True)
    r.add_argument("--gt-root", required=True)
    r.add_argument("--sam-dir")
    r.add_argument("--dataset", choices=("kitti15", "kitti12"), default="kitti15")
    r.add_argument("--window", default="w03-11")
    r.add_argument("--k", choices=GI.K_CHOICES)
    r.add_argument("--n", type=int, default=10)
    r.add_argument("--all", action="store_true")
    r.add_argument("--plan", default=str(MW.PLAN))
    r.add_argument("--json-out")
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return C.EXIT_USAGE if e.code else C.EXIT_OK
    try:
        return run(a)
    except C.Stop as e:
        print(f"ARBITRATE_RUN_ERROR {e.msg}", file=sys.stderr)
        return e.code


if __name__ == "__main__":
    sys.exit(main())
