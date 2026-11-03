#!/usr/bin/env python3
"""exact_refit_par.py: the frozen selftri rule's refit made faster, two ways.

    Rp   all on the CPU, with outputs byte-identical to R, the timed single-threaded CPU path
         (selftri_train_maps.selftri_maps on freshly computed evidence; EV-0628, EV-0672):
         same inputs, same library calls per point, same outputs (EV-0849, Mac). R itself is not
         the cached implementation behind the paper's accuracy numbers (Supp. S6: KITTI-12 Fl
         3.636 % timed vs 3.613 % cached); nothing here touches that difference.
    Rpg  Rp's pose (the same bits) with the dense DLT in torch float64 on a GPU
         (gpu_refit.triangulate_torch, PX-F6's DLT): NOT bit-exact (EV-0672 measured
         max 7.6e-6 px native for that DLT), so it is checked as PX-F6 checked Rg (masks,
         failure reasons and per-image Fl counts equal to R's).
Neither changes the rule; both are timed by tools/p1/time_arbitration.py (Rp@N, Rpg@N, N = the
chunk threads) and checked against R by tools/p1/fast_refit.py on the machine that times them.

How to invoke
    library:  selftri_maps_par(ev_fwd, ev_bwd, K_native, native_hw, seed, pool_dir, pool_chunk,
              nch, log, dev=None)  (pool_dir: ThreadPoolExecutor(2); pool_chunk:
              ThreadPoolExecutor(nch); dev: a torch device for Rpg, None for Rp)
    check:    python tools/p1/exact_refit_par.py check --evidence-dir DIR --threads N
              DIR holds <seq>.npz with fwd_/bwd_{flow,alpha,vis} (the float32 model outputs;
              rule_on casts them to float64, so float32 storage is lossless), K, hw.
              Per item: R and Rp on the same evidence; byte equality of the switched output,
              every map array and the meta; timing. Exit 0 all equal · 1 a mismatch · 64 usage.

Where R spends its time (Mac profile 2026-11-01, 5 KITTI-12 items, one thread): the two
cv2.recoverPose calls about 43 % (OpenCV 4.14 triangulates ALL N correspondences, outliers
included, under each of the 4 pose candidates: 4 N 4x4 SVDs [V opencv 4.14.0 calib3d/src/
five-point.cpp:623-669, triangulate.cpp:113-133]) and the two dense DLTs about 55 % (one 4x4 SVD
per pixel); findEssentialMat, correspondences, render and the rule about 2 %.

Why each change is expected to leave every bit unchanged (Rp; Rpg keeps 1, 3 and 4). This is an
argument per point, not a proof about the libraries: OpenCV's and LAPACK's inner loops may take
other SIMD or tail paths when an array is split, so equality is claimed only where it was checked
byte for byte (all 194 KITTI-12 ids, per platform; the G4 check in fast_refit.py decides).
1. recoverPose: the input mask is applied AFTER the per-point cheirality tests and only
   countNonZero of each candidate's mask decides [V five-point.cpp:684-737], so points the
   RANSAC rejected never count: calling it on the inliers alone gives the same 4 counts.
   triangulatePoints solves each point alone, so the counts are sums over any split of the
   points. decomposeEssentialMat(E) returns the same R1, R2, t bits in every call. If every chunk
   returns the same candidate h, h is also the FIRST global maximum of OpenCV's >= chain (an
   earlier j tying h globally would tie it in every chunk, and each chunk would then have picked
   j), so R, t are the same bits and n is the sum. If chunks disagree: the original serial call.
2. DLT: np.linalg.svd per pixel on the same 16 numbers; the chunking only changes which call
   handles which pixel. The z2 product and the masks stay on the full arrays, as in R.
3. The parallax angle is not used: selftri_maps never reads res['angle'] (Rp skips it).
4. The backward and forward sides (RANSAC + DLT each) overlap; the backward failure is still
   raised first, as in R.
Bit equality is a property of the platform's LAPACK and OpenCV builds, so it is CHECKED, not
assumed: Mac arm64 with cv2 4.14.0 and numpy 2.1.3 (the G4 lock's versions) on all 194 KITTI-12
items, 2026-11-01 (EV-0849). The G4 (x86_64) must pass the same check in the job that times it
(tools/p1/fast_refit.py); nothing is claimed beyond the checked images.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402


def recover_pose_par(E, p1, p2, K, mask, pool, nch: int):
    """cv2.recoverPose(E, p1, p2, K, mask=mask) -> (n, R, t, path), path chunked|fallback|serial."""
    import cv2

    sel = mask.ravel() != 0
    q1, q2 = p1[sel], p2[sel]
    if nch <= 1 or q1.shape[0] < 4 * nch:
        n, R, t, _ = cv2.recoverPose(E, p1, p2, K, mask=mask)
        return n, R, t, "serial"
    idx = np.array_split(np.arange(q1.shape[0]), nch)
    res = list(pool.map(lambda c: cv2.recoverPose(E, np.ascontiguousarray(q1[c]),
                                                  np.ascontiguousarray(q2[c]), K), idx))
    R0, t0 = res[0][1], res[0][2]
    if all(np.array_equal(r[1], R0) and np.array_equal(r[2], t0) for r in res):
        return int(sum(r[0] for r in res)), R0, t0, "chunked"
    n, R, t, _ = cv2.recoverPose(E, p1, p2, K, mask=mask)
    return n, R, t, "fallback"


def relative_motion_par(p1, p2, K, seed, pool, nch, log):
    """self_triangulate.relative_motion with recover_pose_par."""
    import cv2
    import self_triangulate as ST
    from p1lib import jobrun as JR

    if p1.shape[0] < 8:
        raise JR.ItemFailed("essential", f"{p1.shape[0]} correspondences")
    cv2.setRNGSeed(int(seed) % 2**31)
    E, mask = cv2.findEssentialMat(p1, p2, K, method=cv2.RANSAC, prob=ST.PROB,
                                   threshold=ST.RANSAC_PX)
    if E is None or E.shape != (3, 3):
        raise JR.ItemFailed("essential", "findEssentialMat gave no single E")
    n, R, t, path = recover_pose_par(E, p1, p2, K, mask, pool, nch)
    log.append(path)
    if n < 8:
        raise JR.ItemFailed("recoverPose", f"{n} points in front")
    return R, t.reshape(3), int(n)


def _last_vt(K, R, t, u1, v1, u2, v2):
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t.reshape(3, 1)])
    A = np.stack([u1[:, None] * P1[2] - P1[0], v1[:, None] * P1[2] - P1[1],
                  u2[:, None] * P2[2] - P2[0], v2[:, None] * P2[2] - P2[1]], 1)
    return np.linalg.svd(A)[2][:, -1, :]


def triangulate_par(xs, ys, tx, ty, m, K, R, t, pool, nch: int) -> np.ndarray:
    """self_triangulate.triangulate's depth D, line for line; A and the SVD in 4 nch chunks."""
    H, W = xs.shape
    D = np.full((H, W), np.nan)
    if not m.any():
        return D
    u1, v1, u2, v2 = xs[m], ys[m], tx[m], ty[m]
    parts = np.array_split(np.arange(u1.shape[0]), max(1, 4 * nch))
    Xh = np.concatenate(list(pool.map(
        lambda c: _last_vt(K, R, t, u1[c], v1[c], u2[c], v2[c]), parts)))
    with np.errstate(divide="ignore", invalid="ignore"):
        X = Xh[:, :3] / Xh[:, 3:4]
    z1 = X[:, 2]
    z2 = (X @ R.T + t)[:, 2]
    ok = np.isfinite(z1) & (z1 > 0) & (z2 > 0)
    D[m] = np.where(ok, z1, np.nan)
    return D


def make_self_triangulate(pool_dir, pool_chunk, nch: int, log: list, dev=None):
    """self_triangulate.self_triangulate, sides overlapped, without the angle; dev: the DLT in
    torch on that device (Rpg), else triangulate_par (Rp)."""
    import rigid
    import self_triangulate as ST
    from p1lib import jobrun as JR

    def dlt(x, y, tx_, ty_, m_all, K, R, t):
        if dev is None:
            return triangulate_par(x, y, tx_, ty_, m_all, K, R, t, pool_chunk, nch)
        import gpu_refit as GR

        return GR.triangulate_torch(x, y, tx_, ty_, m_all, K, R, t, dev)[0]

    def self_triangulate_par(ev_b: dict, ev_f: dict, K: np.ndarray, seed: int) -> dict:
        xs, ys, txb, tyb, mb_all, mb_e = ST.correspondences(ev_b)
        xf, yf, txf, tyf, mf_all, mf_e = ST.correspondences(ev_f)

        def side(x, y, tx_, ty_, m_e, m_all, sd):
            R, t, n = relative_motion_par(np.stack([x[m_e], y[m_e]], 1),
                                          np.stack([tx_[m_e], ty_[m_e]], 1), K, sd,
                                          pool_chunk, nch, log)
            return R, t, n, dlt(x, y, tx_, ty_, m_all, K, R, t)

        fb = pool_dir.submit(side, xs, ys, txb, tyb, mb_e, mb_all, seed)
        ff = pool_dir.submit(side, xf, yf, txf, tyf, mf_e, mf_all, seed + 1)
        try:
            Rb, tb, nb, Db = fb.result()
            Rf, tf, nf, Df = ff.result()
        finally:
            ff.cancel()
            try:
                ff.result()
            except Exception:  # noqa: BLE001  (raised above, or not needed after a failure)
                pass
        both = np.isfinite(Db) & np.isfinite(Df) & (Df > 0)
        if both.sum() < 8:
            raise JR.ItemFailed("scale", f"{int(both.sum())} pixels triangulated both ways")
        s = float(np.median(Db[both] / Df[both]))
        if not np.isfinite(s) or s <= 0:
            raise JR.ItemFailed("scale", f"scale {s}")
        return {"D": Db, "angle": None, "M_b": rigid.rt_to_M(Rb, tb),
                "M_f": rigid.rt_to_M(Rf, s * tf), "s": s, "n_b": nb, "n_f": nf,
                "n_both": int(both.sum())}

    return self_triangulate_par


def selftri_maps_par(ev_fwd, ev_bwd, K_native, native_hw, seed, pool_dir, pool_chunk, nch, log,
                     dev=None):
    """selftri_train_maps.selftri_maps UNCHANGED, with its self_triangulate swapped (Rp, or Rpg
    when dev is a torch device)."""
    import selftri_train_maps as STM

    orig = STM.ST.self_triangulate
    STM.ST.self_triangulate = make_self_triangulate(pool_dir, pool_chunk, nch, log, dev)
    try:
        return STM.selftri_maps(ev_fwd, ev_bwd, K_native, native_hw, seed)
    finally:
        STM.ST.self_triangulate = orig


def switched_output(ev_fwd: dict, extra: dict) -> np.ndarray:
    """The grid flow time_arbitration.rule_on passes to resize_flow."""
    f12 = extra["flow_12"]
    return ev_fwd["flow"] if f12 is None else np.where(extra["rule"][None], np.nan_to_num(f12),
                                                       ev_fwd["flow"])


def check(evidence_dir: Path, nch: int, out: Path | None) -> int:
    import cv2
    import selftri_train_maps as STM

    files = sorted(glob.glob(str(evidence_dir / "*.npz")))
    if not files:
        print(f"EXACT_REFIT_FAIL no evidence in {evidence_dir}", flush=True)
        return 64
    pool_dir, pool_chunk = ThreadPoolExecutor(2), ThreadPoolExecutor(nch)
    rows, t_ref, t_par = [], [], []
    for f in files:
        z = np.load(f)
        ev = {s: {"flow": z[f"{s}_flow"].astype(np.float64),
                  "alpha": z[f"{s}_alpha"].astype(np.float64), "vis": z[f"{s}_vis"]}
              for s in ("fwd", "bwd")}
        K, hw = z["K"], tuple(int(v) for v in z["hw"])
        t0 = time.perf_counter()
        a0, m0, x0 = STM.selftri_maps(ev["fwd"], ev["bwd"], K, hw, 1)
        t1 = time.perf_counter()
        log: list = []
        a1, m1, x1 = selftri_maps_par(ev["fwd"], ev["bwd"], K, hw, 1, pool_dir, pool_chunk, nch,
                                      log)
        t2 = time.perf_counter()
        t_ref.append(1000 * (t1 - t0))
        t_par.append(1000 * (t2 - t1))
        b0, b1 = switched_output(ev["fwd"], x0).tobytes(), switched_output(ev["fwd"], x1).tobytes()
        r = {"seq": Path(f).stem, "output_equal": b0 == b1,
             "arrays_equal": a0.keys() == a1.keys()
             and all(a0[k].tobytes() == a1[k].tobytes() for k in a0),
             "meta_equal": m0 == m1, "why_unusable": m0.get("why_unusable"),
             "recoverPose": log, "out_sha256_16": hashlib.sha256(b0).hexdigest()[:16]}
        rows.append(r)
        if not (r["output_equal"] and r["arrays_equal"] and r["meta_equal"]):
            print(f"EXACT_REFIT_MISMATCH {json.dumps(r)}", flush=True)
    ok = all(r["output_equal"] and r["arrays_equal"] and r["meta_equal"] for r in rows)
    summ = {"cv2": cv2.__version__, "numpy": np.__version__, "threads": nch,
            "n_items": len(rows), "all_equal": ok,
            "unusable": [[r["seq"], r["why_unusable"]] for r in rows if r["why_unusable"]],
            "recoverPose_paths": {k: sum(r["recoverPose"].count(k) for r in rows)
                                  for k in ("chunked", "fallback", "serial")},
            "median_ms_R": float(np.median(t_ref)), "median_ms_Rp": float(np.median(t_par))}
    print(f"EXACT_REFIT_DONE {json.dumps(summ)}", flush=True)
    if out:
        out.write_text(json.dumps({"summary": summ, "items": rows}, indent=1) + "\n")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--evidence-dir", required=True)
    c.add_argument("--threads", type=int, default=8)
    c.add_argument("--out")
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 64 if e.code else 0
    return check(Path(a.evidence_dir), a.threads, Path(a.out) if a.out else None)


if __name__ == "__main__":
    sys.exit(main())
