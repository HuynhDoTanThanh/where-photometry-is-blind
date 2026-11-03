#!/usr/bin/env python3
"""refine_pose.py: PnP-refined and out-of-fold relative poses of a teacher's depth against the
student's evidence (plan row W2-14; proposal eq. 'Pose out-of-fold'). CPU, on the Mac.

What it does. For each evaluation window of a geometry cache (a teacher_cache root written by
tools/teacher/run_geometry.py, or run_colmap.py's / self_triangulate.py's P1 cache) and each
pair available in it, 10 -> 11 and (w03-11 only) 10 -> 09:
    L      = {exp(alpha_E) <= 2} & vis_E & valid depth & valid_T under the teacher's own pose,
             on the 256 x 832 grid (evidence from dump_eval.py --source u2flow)
    T_ref  = solvePnPRansac(EPnP, 1 px, 1000 iterations) on at most 5000 samples of L,
             stratified by SAM segment, then solvePnPRefineLM on the inliers [P eq. 'Pose
             out-of-fold': 'ngưỡng 1 px, 1000 vòng lặp, tối đa 5000 mẫu phân tầng']
    T_A, T_B  the same from L restricted to the B half / the A half of the SAM segments (a seeded
             split stratified by size), so a segment is tested with a pose it did not shape
    dev    rotation and translation-direction deviation of T_ref from the teacher's pose (T_4D
             or SfM)
    pair_w per pair [P final.architecture.arbitration_rule A3]: 0 when T_ref's PnP has an
             inlier ratio < 0.5 or fewer than 300 inliers (or PnP failed), or when T_ref departs
             from the teacher's pose by > 0.5 deg of rotation or > 10 deg of translation
             direction; else 1; the reasons are recorded. A3's third rule (the largest segment
             touching the bottom border, the road, is not RIGID) needs the arbitration's
             segment decisions and belongs to arbitrate.py (W3-01): recorded as 'pending'.
Every computation uses the window's ONE K (--k, p1lib/geomio.py window_K; eq. 0), recorded in
the identity; render_frig.py refuses a T_ref made with another K.
Items go to OUT/cache/diag/pose_ref/<teacher>/ (p1lib/cache.py), failures to failures.csv.

How to invoke (Mac, .venv/bin/python):
    tools/p1/refine_pose.py --geom-cache C/diag/mapanything --evidence-cache C/diag/u2flow \
        --sam-dir /Volumes/Shin/U2FLOW_data/derived/sam/v1 --out-dir O --k teacher|calib \
        [--max-prio 0]   (A3's thresholds are the defaults; overriding one is a deviation)
Output lines: as p1lib/jobrun.py.
Exit codes: 0 every window done (a window whose PnP fails is recorded, not an error) · 1 left
or interrupted · 2 precondition (a cache missing) · 3 stamp mismatch · 50 seal · 64 usage.

Why it is built this way.
- PnP on the teacher's own depth makes depth and translation share a scale, which F_rig needs
  [P eq. 0 'hoặc từ PnP trên chính depth đó'].
- Out-of-fold halves stop a segment from confirming itself [P C2]; the split is seeded by the
  window id (sha256), so reruns and other machines draw the same halves.
- cv2 RANSAC results move between OpenCV versions [A4] (Mac 5.0.0, VM 4.14.0 [V
  env/lock_G0.txt:42]): the plan asks for the same cv2 on both [P W2-14]; this tool runs on the
  Mac for the evaluation windows, records cv2.__version__ in every item and the host in the
  manifest, and render_frig carries the pose's cv2 into every exported row. Training-window
  T_ref made later on the VM uses cv2 4.14.0: a pose-parity check (the same windows on both
  hosts, like G1.P4 for R4) is open (prereg/drafts/GATE-1.json _open).
- pair_w's thresholds are the proposal's own (A3), applied as pre-specified: setting them after
  dev data exists would be a forking path. Their SOTA check is the method skill's
  (references/thresholds.json pose.pair_w.*, pose.pnp_inliers.*).
"""
from __future__ import annotations

import argparse
import hashlib
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

GRID = (256, 832)
ALPHA_MAX = float(np.log(2.0))   # exp(alpha) <= 2 [P eq. 'Pose out-of-fold']
REPROJ_PX, ITERS, MAX_SAMPLES = 1.0, 1000, 5000  # [P same equation]
MIN_CORR = 6  # [E] EPnP needs >= 4; 6 leaves RANSAC a choice
# pair_w = 0 rules [P final.architecture.arbitration_rule A3]
PAIR_W = {"min_inlier_ratio": 0.5, "min_inliers": 300, "max_rot_deg": 0.5, "max_tdir_deg": 10.0}
TOOL_FILES = [HERE / "refine_pose.py", HERE / "rigid.py", HERE / "p1lib" / "geomio.py"]


def seed_of(*parts: str) -> int:
    return int(hashlib.sha256(":".join(parts).encode()).hexdigest()[:8], 16)


def stratified_sample(idx: np.ndarray, strata: np.ndarray, k: int, rng) -> np.ndarray:
    """At most k of idx, allocated to strata in proportion to their size (at least 1 each)."""
    if idx.size <= k:
        return idx
    labs, counts = np.unique(strata, return_counts=True)
    alloc = np.maximum(1, np.floor(counts / counts.sum() * k)).astype(int)
    out = []
    for lab, a in zip(labs, alloc, strict=True):
        members = idx[strata == lab]
        out.append(rng.choice(members, size=min(a, members.size), replace=False))
    return np.sort(np.concatenate(out))[:k] if out else idx[:0]


def split_halves(labels: np.ndarray, seed: int) -> tuple[set, set]:
    """Segments (label > 0) split in two halves stratified by size: sorted by area, taken in
    consecutive pairs, and each pair shuffled by the seeded generator."""
    labs, counts = np.unique(labels[labels > 0], return_counts=True)
    order = labs[np.argsort(-counts, kind="stable")]
    rng = np.random.default_rng(seed)
    A, B = set(), set()
    for i in range(0, len(order), 2):
        pair = list(order[i:i + 2])
        rng.shuffle(pair)
        A.add(int(pair[0]))
        if len(pair) > 1:
            B.add(int(pair[1]))
    return A, B


def pnp(X: np.ndarray, x2: np.ndarray, K: np.ndarray, seed: int) -> dict | None:
    import cv2

    if X.shape[0] < MIN_CORR:
        return None
    cv2.setRNGSeed(int(seed) % 2**31)
    ok, rvec, tvec, inl = cv2.solvePnPRansac(
        X.astype(np.float64), x2.astype(np.float64), K, None, flags=cv2.SOLVEPNP_EPNP,
        reprojectionError=REPROJ_PX, iterationsCount=ITERS, confidence=0.999)
    if not ok or inl is None or len(inl) < MIN_CORR:
        return None
    inl = inl.reshape(-1)
    rvec, tvec = cv2.solvePnPRefineLM(X[inl], x2[inl], K, None, rvec, tvec)
    R, _ = cv2.Rodrigues(rvec)
    M = rigid.rt_to_M(R, tvec.reshape(3))
    proj, _ = cv2.projectPoints(X[inl], rvec, tvec, K, None)
    err = np.linalg.norm(proj.reshape(-1, 2) - x2[inl], axis=1)
    return {"M": M, "n_in": int(inl.size), "n": int(X.shape[0]),
            "rmse_px": float(np.sqrt(np.mean(err ** 2)))}


def refine_pair(g1: dict, g2: dict | None, ev: dict, M_native: np.ndarray, labels,
                seed: int, consts: dict | None = None) -> dict:
    """Everything refine_pose computes for one pair on the grid; library entry for tests."""
    H, W = ev["flow"].shape[1:]
    if g2 is not None:
        # g1["K"] is the window's one K on the grid (geometry_frame K_model; eq. 0)
        rend = rigid.render_pair(g1["inv"], g2["inv"], g1["K"], M_native, (H, W),
                                 valid1=g1["valid"], valid2=g2["valid"], consts=consts)
        valid_T = rend["valid_12"]
    else:
        valid_T = g1["valid"]
    L = g1["valid"] & valid_T
    if "alpha" in ev:
        L &= ev["alpha"] <= ALPHA_MAX
    if "vis" in ev:
        L &= ev["vis"]
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    tx, ty = xs + ev["flow"][0], ys + ev["flow"][1]
    L &= np.isfinite(tx) & np.isfinite(ty)
    X = np.moveaxis(rigid.backproject(rigid.depth_from_inv(np.where(L, g1["inv"], np.nan)),
                                      g1["K"]), 0, -1)
    flat = np.flatnonzero(L.reshape(-1))
    strata = (labels.reshape(-1)[flat] if labels is not None
              else (ys.reshape(-1)[flat] // 32) * 100 + xs.reshape(-1)[flat] // 32)
    rng = np.random.default_rng(seed)
    Xf = X.reshape(-1, 3)
    x2f = np.stack([tx.reshape(-1), ty.reshape(-1)], 1)
    out = {"n_L": int(flat.size)}
    pick = stratified_sample(flat, strata, MAX_SAMPLES, rng)
    ref = pnp(Xf[pick], x2f[pick], g1["K"], seed)
    out["ref"] = ref
    if labels is not None:
        A, B = split_halves(labels, seed)
        segs = labels.reshape(-1)[flat]
        for name, half in (("A", B), ("B", A)):  # T_A from L in S_B, T_B from L in S_A
            sel = flat[np.isin(segs, list(half))]
            s = stratified_sample(sel, labels.reshape(-1)[sel], MAX_SAMPLES, rng)
            out[name] = pnp(Xf[s], x2f[s], g1["K"], seed)
        out["halves"] = {"A": sorted(A), "B": sorted(B)}
    if ref is not None:
        out["dev"] = rigid.pose_deviation(M_native, ref["M"])
    return out


def pair_weight(res: dict, thr: dict | None = None) -> tuple[int, list[str]]:
    """A3's pair_w (0 or 1) of one refined pair and the reasons it is 0."""
    t = dict(PAIR_W, **(thr or {}))
    ref = res.get("ref")
    if ref is None:
        return 0, ["pnp failed"]
    why = []
    ratio = ref["n_in"] / ref["n"] if ref["n"] else 0.0
    if ratio < t["min_inlier_ratio"]:
        why.append(f"inlier ratio {ratio:.3f} < {t['min_inlier_ratio']:g}")
    if ref["n_in"] < t["min_inliers"]:
        why.append(f"{ref['n_in']} inliers < {t['min_inliers']:g}")
    dev = res.get("dev") or {}
    if dev.get("rot_deg") is not None and dev["rot_deg"] > t["max_rot_deg"]:
        why.append(f"rotation {dev['rot_deg']:.3f} deg > {t['max_rot_deg']:g}")
    if dev.get("trans_dir_deg") is not None and dev["trans_dir_deg"] > t["max_tdir_deg"]:
        why.append(f"translation direction {dev['trans_dir_deg']:.2f} deg > "
                   f"{t['max_tdir_deg']:g}")
    return (0 if why else 1), why


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--geom-cache", required=True)
    ap.add_argument("--evidence-cache", required=True)
    ap.add_argument("--sam-dir")
    ap.add_argument("--teacher", help="name for the output dir (default: the cache's kind)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--plan", default=str(MW.PLAN))
    ap.add_argument("--max-prio", type=int)
    ap.add_argument("--k", choices=GI.K_CHOICES, help="the window's one K: teacher (frame 10's "
                    "own) or calib; required unless the cache's K is the calibration")
    ap.add_argument("--max-rot-deg", type=float, default=PAIR_W["max_rot_deg"])
    ap.add_argument("--max-tdir-deg", type=float, default=PAIR_W["max_tdir_deg"])
    ap.add_argument("--min-inlier-ratio", type=float, default=PAIR_W["min_inlier_ratio"])
    ap.add_argument("--min-inliers", type=int, default=PAIR_W["min_inliers"])
    ap.add_argument("--work-dir")
    ap.add_argument("--shard-size", type=int, default=25)
    ap.add_argument("--limit", type=int)
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return C.EXIT_USAGE if e.code else C.EXIT_OK
    try:
        import cv2

        C.refuse_sealed(a.geom_cache, a.evidence_cache, a.sam_dir, a.out_dir, a.work_dir)
        geo = GI.GeomIndex(Path(a.geom_cache))
        evi = CA.CacheIndex(Path(a.evidence_cache))
        plan = MW.load_plan(Path(a.plan))
        teacher = a.teacher or geo.manifest.get("source", "teacher")
        by_id = {r["id"]: r for r in plan["eval_windows"]}
        rows = [by_id[i] for i in geo.ids() if i in by_id]
        if a.max_prio is not None:
            rows = [r for r in rows if r["prio"] <= a.max_prio]
        rows.sort(key=lambda r: (r["prio"], r["id"]))
        if not rows:
            raise C.Stop(C.EXIT_PRE, "no evaluation window of the plan in this geometry cache")
        kch = GI.resolve_k(geo.get(rows[0]["id"])[1], a.k)
        thr = {"max_rot_deg": a.max_rot_deg, "max_tdir_deg": a.max_tdir_deg,
               "min_inlier_ratio": a.min_inlier_ratio, "min_inliers": a.min_inliers}
        deviates = sorted(x for x in thr if thr[x] != PAIR_W[x])
        settings = {"teacher": teacher, "geom_identity": geo.manifest.get("identity"),
                    "evidence_identity": evi.manifest.get("identity"), "pair_w": thr,
                    "pair_w_source": "proposal arbitration_rule A3" + (
                        f"; DEVIATION in {', '.join(deviates)}" if deviates else ""),
                    "pair_w_road_rule": "pending: arbitrate.py (W3-01)",
                    "k": kch, "pnp": {"reproj_px": REPROJ_PX, "iters": ITERS,
                                    "max_samples": MAX_SAMPLES, "alpha_max": "ln 2"},
                    "cv2": cv2.__version__, "sam": bool(a.sam_dir)}
        identity = {"settings": settings, "code": C.code_identity(TOOL_FILES)}
        out = Path(a.out_dir)
        work = Path(a.work_dir) if a.work_dir else out / "work" / f"pose_ref-{teacher}"
        job = JR.ItemJob(kind="pose-ref", source=teacher, items=rows,
                         cache_dir=out / "cache" / "diag" / "pose_ref" / teacher, work_dir=work,
                         shard_size=a.shard_size, shard=None, identity=identity,
                         settings=settings, allow_failures=True, limit=a.limit)
        sam = Path(a.sam_dir) if a.sam_dir else None

        def produce(row: dict):
            ga, gm = geo.get(row["id"])
            got = evi.get(GI.evidence_id(row))
            if got is None:
                raise JR.ItemFailed("input", "no evidence item for this sequence")
            ea, _ = got
            i10 = gm["frames"].index(10)
            K = GI.window_K(ga, gm, i10, kch)  # ONE K for every frame of the window (eq. 0)
            g10 = GI.geometry_frame(ga, gm, i10, GRID, K_model=K)
            labels = GI.sam_labels(sam, row, 10, GRID)
            arrays, meta = {}, {"pairs": {}, "cv2": cv2.__version__, "thresholds": thr,
                                "k": kch, "K_model": K.tolist()}
            for tag, other, d in (("10-11", 11, "10to11"), ("10-09", 9, "10to09")):
                if other not in gm["frames"]:
                    continue
                j = gm["frames"].index(other)
                g2 = GI.geometry_frame(ga, gm, j, GRID, K_model=K)
                if not (g10["pose_ok"] and g2["pose_ok"]):
                    meta["pairs"][tag] = {"status": "no teacher pose"}
                    continue
                M_nat = rigid.relative_pose(g10["pose_c2w"], g2["pose_c2w"])
                has_inv2 = bool(g2["valid"].any())
                res = refine_pair(g10, g2 if has_inv2 else None, GI.evidence(ea, d, GRID),
                                  M_nat, labels, seed_of(row["id"], tag))
                pm = {"n_L": res["n_L"], "status": "ok" if res["ref"] else "pnp failed"}
                for k in ("ref", "A", "B"):
                    if res.get(k):
                        arrays[f"M_{k}_{tag}"] = res[k]["M"]
                        pm[k] = {x: res[k][x] for x in ("n_in", "n", "rmse_px")}
                arrays[f"M_native_{tag}"] = M_nat
                if "dev" in res:
                    pm["dev"] = res["dev"]
                pm["pair_w"], pm["pair_w_reasons"] = pair_weight(res, thr)
                pm["pair_w_road_rule"] = "pending (arbitrate.py, W3-01)"
                if "halves" in res:
                    pm["halves"] = res["halves"]
                meta["pairs"][tag] = pm
            if not any(k.startswith("M_ref_") for k in arrays):
                raise JR.ItemFailed("pnp", "no pair gave a refined pose")
            return arrays, meta

        job.progress.resolved({"tool": "tools/p1/refine_pose.py", "teacher": teacher,
                               "n_items": job.total})
        return job.run(produce)
    except C.Stop as e:
        C.say("P1_ERROR", e.msg)
        return e.code


if __name__ == "__main__":
    sys.exit(main())
