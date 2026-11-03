#!/usr/bin/env python3
"""render_frig.py: F_rig for every teacher x pose x window combination, with every flag the gate
and the loader use (plan rows W2-14, W2-16). CPU, on the Mac.

What it does.
    render      one geometry cache (a teacher_cache root from tools/teacher/run_geometry.py, or a
                P1 cache from run_colmap.py / self_triangulate.py; p1lib/geomio.py) x one
                pose source x the windows asked for: for the pair 10 -> 11 of each window,
                rigid.render_pair on the 256 x 832 grid with the window's ONE K (--k), both
                directions:
                  flow_12 / flow_21 fp16 (2,256,832), b_12 / b_21 fp16, c_12 uint8 (c_T x 255),
                  packbits valid_*, oof_*, occ_* (O_T), front_*, zunknown_*
                and, with --native, for KITTI-15 dev and KITTI-12 windows only, float32 (NaN
                where the teacher has no depth): native_12 = the grid flow carried to the frame's
                native resolution by U2Flow's own resize_flow, exactly as a student prediction
                is (the gate row), and native_direct_12 = F_rig rendered directly at native
                resolution (a diagnostic row only).
                Pose 'native' = the teacher's own (T_4D, SfM for COLMAP, the essential-matrix
                pose for self-triangulation, PoseNet for Monodepth2); 'ref' = refine_pose.py's
                T_ref, which must have been made with the same --k (else exit 3).
                Output: OUT/cache/diag/frig/<teacher>_<pose>_<win>/ (shards, manifest).
    check-res   the W2-14 storage decision on N windows: F_rig from the stored cache (fp16 at
                the teacher's resolution, upsampled as the loader would) against a reference:
                --ref-cache (the same windows stored at the training grid or in fp32) or the
                cache's own inv_depth_f32; plus the model-resolution vs grid render; inside the
                frame and on its outer ring; writes OUT/frig_resolution_check.json; prints the
                plan's 0.05 px verdict.
    export-preds  the native flows of a frig cache in eval_kitti.py's prediction format
                (OUT/<kitti15|kitti12>/preds/PREDS.json + shards), NaN kept where the teacher
                has no depth; --which grid (default: native_12, the gate row) or direct
                (native_direct_12, diagnostic). COVERAGE.json carries the provenance a teacher
                row needs: gpu_name of the geometry's (or, for self-triangulation, the
                evidence's) GPU with gpu_source, the render host and the pose's cv2. Scoring
                them with coverage is the metric skill's (W3-03).
    merge-cache shards of several runs of one recipe (e.g. --shard k/n on different VMs) into
                one cache dir; refuses different identities or one shard with two hashes.

How to invoke (Mac, .venv/bin/python):
    tools/p1/render_frig.py render --geom-cache C/diag/mapanything --pose native --win all \
        --k teacher|calib --out-dir O [--native] [--max-prio 0]
    tools/p1/render_frig.py render --geom-cache C/diag/mapanything --pose ref --k K \
        --pose-cache O/cache/diag/pose_ref/mapanything --win w03-11 --out-dir O
    tools/p1/render_frig.py check-res --geom-cache D/derived/w214mapa --k K --n 20 --out-dir O
    tools/p1/render_frig.py export-preds --frig-cache D --out-dir O [--which grid|direct]
    tools/p1/render_frig.py merge-cache --src A --src B --dst D
Output lines: as p1lib/jobrun.py, plus FRIG_RESOLUTION {json}, FRIG_EXPORT, FRIG_MERGE.
Exit codes: 0 ok · 1 windows left, or check-res above the plan's bound · 2 precondition ·
3 refused (merge of different recipes) · 50 seal · 64 usage.

Why it is built this way.
- One renderer for every source [P architecture overview], so a teacher table compares
  geometry, not code paths; the constants (rigid.CONSTANTS) are recorded in every manifest and
  live only here, so changing one is a CPU re-render.
- Self-triangulation has no depth for frame 11: its 1 -> 2 direction is rendered without the
  target-depth test (zunknown everywhere) and 2 -> 1 is absent; COLMAP frames that are not
  registered fail their window.
- One K per window [P eq. 0]: the K of the source frame 10 ('teacher') or the calibration
  ('calib'); a cache whose K is not the calibration (MapAnything predicts it, pi3's is fitted)
  needs --k, which is W2-14's decision and is recorded in the identity (p1lib/geomio.py
  window_K). Two Ks would turn K jitter between frames into flow [M tests/test_p1_rigid.py].
- The gate row is scored the way the student is [P region_protocol: 'du doan duoc upsample
  bang resize_flow nhu evaluate.py. Tat ca chay qua mot duong code duy nhat']: the grid flow
  (the training teacher, arbitration 'tren luoi huan luyen') through U2Flow's resize_flow; the
  direct native render is kept beside it as a diagnostic. resize_flow spreads a NaN to its
  bilinear neighbours, so a pixel next to a hole is uncovered (fail-closed).
- --native only for windows whose GT may be read on the Mac (KITTI-15 dev, KITTI-12): a
  held-out id never gets a scorable prediction here.
- Provenance: the render runs on the Mac (CPU, numpy float64); the GPU work is the teacher's
  network (or U2Flow's evidence for self-triangulation) on the G4. A teacher row's gpu_name is
  that GPU (gpu_source says which), the render host and cv2 sit beside it (prereg/drafts/
  GATE-1.json _row_conventions); a cache that recorded no GPU exports gpu_name null, which no
  gate selector matches.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import p1_windows as MW  # noqa: E402
import rigid  # noqa: E402
from p1lib import cache as CA  # noqa: E402
from p1lib import common as C  # noqa: E402
from p1lib import geomio as GI  # noqa: E402
from p1lib import jobrun as JR  # noqa: E402
from p1lib import kitti as KT  # noqa: E402

GRID = (256, 832)
TOOL_FILES = [HERE / "render_frig.py", HERE / "rigid.py", HERE / "p1lib" / "geomio.py",
              C.REPO / "core" / "utils" / "flow_utils.py"]  # resize_flow: the gate row's path
FLAGS = ("valid", "oof", "occ", "front", "zunknown")
RES_BOUND_PX = 0.05  # [P W2-14] (method thresholds.json Frig.lowres_tolerance_px)


def scorable(row: dict) -> bool:
    return row["ds"] == "kitti12" or row.get("k15_half") == "dev"


def resize_flow_like_the_student(flow: np.ndarray, native_hw) -> np.ndarray:
    """(2, h, w) grid flow -> (2, H0, W0) float32 by U2Flow's core.utils.flow_utils.resize_flow
    (align_corners=True bilinear, u scaled by W0/w, v by H0/h), the function eval_kitti.py
    applies to every student prediction [V core/utils/flow_utils.py:67-75]."""
    import torch

    sys.path.insert(0, str(C.REPO))
    from core.utils.flow_utils import resize_flow

    t = torch.from_numpy(np.asarray(flow, np.float32)[None].copy())
    with torch.no_grad():
        return resize_flow(t, tuple(int(x) for x in native_hw))[0].numpy().astype(np.float32)


def render_window(ga: dict, gm: dict, M: np.ndarray, grid, native: bool, row: dict,
                  k: str = "teacher") -> tuple:
    i10, i11 = gm["frames"].index(10), gm["frames"].index(11)
    K = GI.window_K(ga, gm, i10, k)  # ONE K for the window (eq. 0)
    g10 = GI.geometry_frame(ga, gm, i10, grid, K_model=K)
    g11 = GI.geometry_frame(ga, gm, i11, grid, K_model=K)
    has2 = bool(g11["valid_model"].any())
    inv2 = g11["inv_model"] if has2 else None
    out = rigid.render_pair(g10["inv_model"], inv2, K, M, grid,
                            valid1=g10["valid_model"], valid2=g11["valid_model"] if has2
                            else None)
    arrays = {}
    for tag in ("12", "21"):
        if f"flow_{tag}" not in out:
            continue
        arrays[f"flow_{tag}"] = out[f"flow_{tag}"].astype(np.float16)
        arrays[f"b_{tag}"] = out[f"b_{tag}"].astype(np.float16)
        for f in FLAGS:
            arrays[f"{f}_{tag}"] = CA.pack_bits(out[f"{f}_{tag}"])
    if g10["conf_rank"] is not None:
        r, _ = rigid.resize_bilinear(g10["conf_rank"].astype(np.float64), grid)
        arrays["c_12"] = np.rint(255 * rigid.c_T(np.nan_to_num(r))).astype(np.uint8)
    if native and scorable(row):
        nat = tuple(gm["native_hw"])
        arrays["native_12"] = resize_flow_like_the_student(out["flow_12"], nat)
        on = rigid.render_pair(g10["inv_model"], None, K, M, nat, valid1=g10["valid_model"])
        arrays["native_direct_12"] = on["flow_12"].astype(np.float32)
    meta = {"grid": list(grid), "native_hw": gm["native_hw"], "frames": gm["frames"],
            "has_target_depth": has2, "constants": rigid.CONSTANTS, "win": gm["win"],
            "k15_half": gm.get("k15_half"), "M": M.tolist(), "k": k, "K_model": K.tolist(),
            "valid_frac_12": float(out["valid_12"].mean()),
            "oof_frac_12": float(out["oof_12"].mean()),
            "native_rule": "native_12 = resize_flow(grid flow_12) (the gate row); "
                           "native_direct_12 = direct render (diagnostic)"}
    return arrays, meta


def cmd_render(a) -> int:
    if a.pose == "ref" and not a.pose_cache:
        raise C.Stop(C.EXIT_USAGE, "--pose ref needs --pose-cache")
    geo = GI.GeomIndex(Path(a.geom_cache))
    teacher = a.teacher or geo.manifest.get("source", "teacher")
    pose_idx = CA.CacheIndex(Path(a.pose_cache)) if a.pose == "ref" else None
    k, k_src = resolve_k_of(geo, a.k)
    if pose_idx is not None and (pose_idx.manifest.get("settings") or {}).get("k") != k:
        raise C.Stop(C.EXIT_REFUSED, f"{a.pose_cache} was refined with K "
                     f"'{(pose_idx.manifest.get('settings') or {}).get('k')}', this render "
                     f"uses '{k}': one K per window for the pose and its render")
    plan = MW.load_plan(Path(a.plan))
    by_id = {r["id"]: r for r in plan["eval_windows"]}
    wins = list(KT.EVAL_WINDOWS) if a.win == "all" else [a.win]
    grid = tuple(a.grid)
    rc = C.EXIT_OK
    for win in wins:
        rows = [by_id[i] for i in geo.ids() if i in by_id]  # ref (5-15) windows: not rendered
        rows = [r for r in rows if r["win"] == win
                and (a.max_prio is None or r["prio"] <= a.max_prio)]
        if not rows:
            continue
        rows.sort(key=lambda r: (r["prio"], r["id"]))
        name = f"{teacher}_{a.pose}_{win}"
        settings = {"teacher": teacher, "pose": a.pose, "win": win, "grid": list(grid),
                    "native": bool(a.native), "constants": rigid.CONSTANTS,
                    "k": k, "k_source": k_src,
                    "native_rule": "resize_flow(grid) gate row + direct diagnostic",
                    "geom_identity": geo.manifest.get("identity"),
                    "pose_identity": pose_idx.manifest.get("identity") if pose_idx else None,
                    "pose_cv2": (pose_idx.manifest.get("settings") or {}).get("cv2")
                    if pose_idx else None}
        identity = {"settings": settings, "code": C.code_identity(TOOL_FILES)}
        out = Path(a.out_dir)
        job = JR.ItemJob(kind="frig", source=name, items=rows,
                         cache_dir=out / "cache" / "diag" / "frig" / name,
                         work_dir=out / "work" / f"frig-{name}", shard_size=a.shard_size,
                         shard=None, identity=identity, settings=settings,
                         allow_failures=True, limit=a.limit)

        def produce(row: dict):
            ga, gm = geo.get(row["id"])
            i10, i11 = gm["frames"].index(10), gm["frames"].index(11)
            if "pose_ok" in ga and not (ga["pose_ok"][i10] and ga["pose_ok"][i11]):
                raise JR.ItemFailed("pose", "frame 10 or 11 has no teacher pose")
            if a.pose == "native":
                M = rigid.relative_pose(ga["pose_c2w"][i10], ga["pose_c2w"][i11])
            else:
                got = pose_idx.get(row["id"])
                if got is None or "M_ref_10-11" not in got[0]:
                    raise JR.ItemFailed("pose", "no refined pose for 10-11")
                M = got[0]["M_ref_10-11"]
            arrays, meta = render_window(ga, gm, M, grid, a.native, row, k)
            return arrays, dict(meta, **GI.producer_gpu(geo, gm))

        job.progress.resolved({"tool": "tools/p1/render_frig.py", "combination": name,
                               "n_items": job.total})
        rc = max(rc, job.run(produce))
    return rc


def resolve_k_of(geo: GI.GeomIndex, k: str | None) -> tuple[str, str]:
    """(the K choice, the cache's k_source) from its first item (one teacher, one k_source)."""
    ids = geo.ids()
    if not ids:
        raise C.Stop(C.EXIT_PRE, "the geometry cache holds no item")
    _a, m = geo.get(ids[0])
    return GI.resolve_k(m, k), GI.k_source(m)


def ring_mask(hw) -> np.ndarray:
    m = np.zeros(hw, bool)
    m[[0, -1], :] = m[:, [0, -1]] = True
    return m


def _stats(a, b, ring) -> dict:
    d = np.sqrt(((a - b) ** 2).sum(0))
    out = {}
    for name, sel in (("interior", ~ring), ("ring", ring)):
        v = d[sel & np.isfinite(d)]
        out[name] = ({"n": int(v.size), "max_px": float(v.max()), "mean_px": float(v.mean()),
                      "p99_px": float(np.percentile(v, 99))} if v.size else {"n": 0})
    return out


def _frame10(ga: dict, gm: dict, k: str, key: str = "inv_depth"):
    i10, i11 = gm["frames"].index(10), gm["frames"].index(11)
    M = rigid.relative_pose(ga["pose_c2w"][i10], ga["pose_c2w"][i11])
    valid = CA.unpack_bits(ga["valid"], tuple(gm["shape"]))[i10]
    return ga[key][i10].astype(np.float64), valid, GI.window_K(ga, gm, i10, k), M


def cmd_check_res(a) -> int:
    """Reference = the same window's float32 depth (a run_geometry.py --store-f32 cache, or a P1
    cache made with it), or the window of --ref-cache (its float32 depth when it has one, else
    its stored depth; e.g. a 20-window teacher run stored at the training grid): the candidate
    is always the stored fp16 depth, rendered as the loader would (grid mode), with the same
    one K per window (--k) on both sides."""
    geo = GI.GeomIndex(Path(a.geom_cache))
    refc = GI.GeomIndex(Path(a.ref_cache)) if a.ref_cache else None
    k, k_src = resolve_k_of(geo, a.k)
    grid = tuple(a.grid)
    ring = ring_mask(grid)
    res = []
    for item in geo.ids():
        ga, gm = geo.get(item)
        if 10 not in gm["frames"] or 11 not in gm["frames"]:
            continue
        inv, valid, K, M = _frame10(ga, gm, k)
        cand = rigid.render_pair(inv, None, K, M, grid, valid1=valid)["flow_12"]
        if refc is not None:
            got = refc.get(item)
            if got is None:
                continue
            ra, rm = got
            rkey = "inv_depth_f32" if "inv_depth_f32" in ra else "inv_depth"
            rinv, rvalid, rK, rM = _frame10(ra, rm, k, rkey)
            ref = rigid.render_pair(rinv, None, rK, rM, grid, valid1=rvalid)["flow_12"]
            model = rigid.render_model_mode(inv, K, M, grid, valid1=valid)
        elif "inv_depth_f32" in ga:
            f32, _, _, _ = _frame10(ga, gm, k, "inv_depth_f32")
            ref = rigid.render_pair(f32, None, K, M, grid, valid1=valid)["flow_12"]
            model = rigid.render_model_mode(f32, K, M, grid, valid1=valid)
        else:
            continue
        res.append({"item": item, "stored_vs_reference": _stats(cand, ref, ring),
                    "model_vs_grid": _stats(model, cand, ring)})
        if len(res) >= a.n:
            break
    if not res:
        raise C.Stop(C.EXIT_PRE, "no window with a reference: give --ref-cache, or a cache that "
                                 "carries float32 depth (run_geometry.py --store-f32, or a P1 "
                                 "cache)")
    worst = max(r["stored_vs_reference"]["interior"].get("max_px", 0.0) for r in res)
    doc = {"n": len(res), "bound_px": RES_BOUND_PX, "stored_interior_max_px": worst,
           "stored_ok": worst <= RES_BOUND_PX, "windows": res,
           "reference": a.ref_cache or "inv_depth_f32 of the same cache",
           "k": k, "k_source": k_src,
           "geom_identity": geo.manifest.get("identity"), "utc": C.utc_now(),
           "note": "decision input for W2-14/W2-15; the ring is the grid's outer pixel row and "
                   "column, whose centres lie outside the teacher's pixel-centre hull"}
    C.atomic_json(Path(a.out_dir) / "frig_resolution_check.json", doc)
    C.say("FRIG_RESOLUTION", json.dumps({k: doc[k] for k in ("n", "stored_interior_max_px",
                                                             "stored_ok")}))
    return C.EXIT_OK if doc["stored_ok"] else C.EXIT_FAIL


def cmd_export(a) -> int:
    sys.path.insert(0, str(C.REPO / "tools" / "eval"))
    import eval_kitti as ek  # noqa: PLC0415

    idx = CA.CacheIndex(Path(a.frig_cache))
    key = {"grid": "native_12", "direct": "native_direct_12"}[a.which]
    writers: dict = {}
    n = 0
    gpus: set = set()
    srcs: set = set()
    for item in idx.ids():
        arr, meta = idx.get(item)
        if key not in arr:
            continue
        d = KT.parse_id(item)
        if d["ds"] == "kitti15" and meta.get("k15_half") != "dev":
            raise C.Stop(C.EXIT_SEAL, f"{item}: a native prediction of a non-dev KITTI-15 id")
        w = writers.setdefault(d["ds"], ek.ShardWriter(Path(a.out_dir) / d["ds"] / "preds"))
        w.add(d["seq"], arr[key].astype(np.float32))
        gpus.add(meta.get("gpu_name"))
        srcs.add(meta.get("gpu_source"))
        n += 1
    shas = {ds: w.close() for ds, w in writers.items()}
    st = idx.manifest.get("settings") or {}
    one = len(gpus) == 1 and None not in gpus
    C.atomic_json(Path(a.out_dir) / "COVERAGE.json", {
        "source": idx.manifest.get("source"), "pred_sha": shas, "n": n,
        "which": a.which, "flow_key": key,
        "row_kind": "gate" if a.which == "grid" else "diagnostic (never a gate row)",
        "teacher_label_suffix": "" if a.which == "grid" else ":direct",
        "env": {"gpu_name": next(iter(gpus)) if one else None,
                "gpu_source": next(iter(srcs)) if one else None,
                "gpu_names_seen": sorted(str(g) for g in gpus),
                "render_host": idx.manifest.get("host"), "pose_cv2": st.get("pose_cv2"),
                "k": st.get("k"), "k_source": st.get("k_source"),
                "rule": "gpu_name = the GPU that ran the teacher's network (or U2Flow's evidence "
                        "for self-triangulation); null when unrecorded or mixed, which no gate "
                        "selector matches (prereg/drafts/GATE-1.json _row_conventions)"},
        "nan": "NaN marks pixels with no teacher depth: score with coverage (metric skill)"})
    C.say("FRIG_EXPORT", f"n={n} {json.dumps(shas, sort_keys=True)}")
    return C.EXIT_OK


def cmd_merge(a) -> int:
    dst = Path(a.dst)
    C.refuse_sealed(dst, *a.src)
    docs = [CA.CacheIndex(Path(s)).manifest for s in a.src]
    ids = {json.dumps(d.get("identity"), sort_keys=True) for d in docs}
    if len(ids) != 1:
        raise C.Stop(C.EXIT_REFUSED, "the sources were made by different recipes")
    dst.mkdir(parents=True, exist_ok=True)
    shards: dict = {}
    for s, d in zip(a.src, docs, strict=True):
        for sh in d["shards"]:
            have = shards.get(sh["file"])
            if have and have["sha256"] != sh["sha256"]:
                raise C.Stop(C.EXIT_REFUSED, f"{sh['file']}: two different contents")
            if not have:
                if C.sha256_file(Path(s) / sh["file"]) != sh["sha256"]:
                    raise C.Stop(C.EXIT_PRE, f"{s}/{sh['file']}: sha256 differs from manifest")
                shutil.copyfile(Path(s) / sh["file"], dst / sh["file"])
                shards[sh["file"]] = sh
    fails = [f for d in docs for f in d.get("failures", [])]
    n_items = sum(len(v["items"]) for v in shards.values())
    planned = docs[0].get("n_items_planned")
    doc = dict(docs[0], shards=[shards[k] for k in sorted(shards)],
               n_items=n_items, failures=fails, shard="merged",
               complete=planned is not None and n_items == planned and not fails,
               merged_from=[str(s) for s in a.src], utc=C.utc_now())
    CA.write_manifest(dst, doc)
    C.say("FRIG_MERGE", f"{dst} shards={len(shards)} items={doc['n_items']}")
    return C.EXIT_OK


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("--geom-cache", required=True)
    r.add_argument("--pose", default="native", choices=("native", "ref"))
    r.add_argument("--pose-cache")
    r.add_argument("--teacher", help="name for the output dirs (default: the cache's kind)")
    r.add_argument("--win", default="all", choices=("all", *KT.EVAL_WINDOWS))
    r.add_argument("--out-dir", required=True)
    r.add_argument("--plan", default=str(MW.PLAN))
    r.add_argument("--max-prio", type=int)
    r.add_argument("--native", action="store_true")
    r.add_argument("--k", choices=GI.K_CHOICES, help="the window's one K: teacher (frame 10's "
                   "own) or calib; required unless the cache's K is the calibration")
    r.add_argument("--grid", type=int, nargs=2, default=list(GRID))
    r.add_argument("--shard-size", type=int, default=25)
    r.add_argument("--limit", type=int)
    c = sub.add_parser("check-res")
    c.add_argument("--geom-cache", required=True)
    c.add_argument("--out-dir", required=True)
    c.add_argument("--n", type=int, default=20)
    c.add_argument("--ref-cache")
    c.add_argument("--k", choices=GI.K_CHOICES)
    c.add_argument("--grid", type=int, nargs=2, default=list(GRID))
    e = sub.add_parser("export-preds")
    e.add_argument("--frig-cache", required=True)
    e.add_argument("--out-dir", required=True)
    e.add_argument("--which", choices=("grid", "direct"), default="grid")
    m = sub.add_parser("merge-cache")
    m.add_argument("--src", action="append", required=True)
    m.add_argument("--dst", required=True)
    try:
        a = ap.parse_args(argv)
    except SystemExit as ex:
        return C.EXIT_USAGE if ex.code else C.EXIT_OK
    try:
        for k in ("geom_cache", "pose_cache", "out_dir", "frig_cache", "ref_cache"):
            C.refuse_sealed(getattr(a, k, None))
        return {"render": cmd_render, "check-res": cmd_check_res, "export-preds": cmd_export,
                "merge-cache": cmd_merge}[a.cmd](a)
    except C.Stop as ex:
        C.say("P1_ERROR", ex.msg)
        return ex.code


if __name__ == "__main__":
    sys.exit(main())
