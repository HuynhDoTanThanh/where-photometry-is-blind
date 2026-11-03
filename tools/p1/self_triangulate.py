#!/usr/bin/env python3
"""self_triangulate.py: the prior-free teacher triangulated from U2Flow's own flow (plan rows
W2-10, W2-12; proposal eq. 'Tự tam giác hoá'), written as a geometry cache that render_frig.py
and refine_pose.py read like any other teacher's. CPU, multiprocess.

What it does, per KITTI sequence (the w03-11 evaluation windows: frame 10 with t-1 = 09 and
t+1 = 11; w10-18 drops t-1 on purpose, so it has no self-triangulation [P region protocol]):
    1. backward: E from the low-alpha, visible, in-frame correspondences p -> p + F_E(10->09)
       (findEssentialMat, RANSAC 1 px, p 0.999), (R_b, t_b) = recoverPose, |t_b| = 1;
    2. D_10(p) = triangulate(p, p + F_E(10->09)) for every visible p (linear DLT, both cameras
       in front), with the triangulation angle;
    3. forward: (R_f, t_f_hat) the same way from 10 -> 11, D^(f) triangulated with |t_f| = 1;
       s = median over V_both of D^(b) / D^(f); the forward motion is (R_f, s t_f_hat);
    4. the item: frames [09, 10, 11]; inv_depth only for frame 10 (the others invalid); poses
       c2w(09) = inv(M_b), c2w(10) = I, c2w(11) = inv(M_f); K the grid K of the calibration;
       conf = the triangulation angle (ranked); tri_angle_deg; low_parallax (< 1 deg) bits.
    F_rig^stri(p) = pi(K (R_f D_10(p) K^-1 p~ + s t_f_hat)) - p is then render_frig's job.
Evidence: dump_eval.py --source u2flow (flow_10to09, alpha, vis; flow_10to11, ...).

How to invoke (Mac, .venv/bin/python; or a VM CPU job):
    tools/p1/self_triangulate.py run --evidence-cache C/diag/u2flow --data D --out-dir O \
        [--procs 8] [--max-prio 0] [--tf-root R/derived/evalselftri/v2]
Output lines: as p1lib/jobrun.py, plus P1_WORKER, P1_WORKERS_EXIT.
Exit codes: 0 every window done (a failed triangulation is recorded in failures.csv) · 1 left ·
2 precondition · 3 stamp mismatch · 50 seal · 64 usage · 143 TERM.

Why it is built this way.
- It follows the proposal's equation step by step; the parts it leaves open are fixed here and
  tagged: the RANSAC threshold of findEssentialMat (1 px, the PnP threshold of the same
  proposal [E]), the alpha cut for the essential matrix (exp(alpha) <= 2, as for PnP [E]), and
  the 1 deg parallax flag [P] ('b_T tăng khi góc tam giác hoá < 1°'); how much b_T grows is not
  given, so the flag and the angle are stored and render_frig leaves b_T unchanged [U].
- K is the calibration's (metadata, not a label [P io.training_inputs]) zoomed to the grid.
- Depth is in units of |t_b| = 1: the flow is scale-free because pose and depth share it.
"""
from __future__ import annotations

import argparse
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
from p1lib import multiproc as MP  # noqa: E402

GRID = (256, 832)
ALPHA_MAX = float(np.log(2.0))
RANSAC_PX, PROB = 1.0, 0.999
LOW_PARALLAX_DEG = 1.0  # [P]
TOOL_FILES = [HERE / "self_triangulate.py", HERE / "rigid.py", HERE / "p1lib" / "geomio.py"]


def correspondences(ev: dict, use_alpha: bool = True):
    H, W = ev["flow"].shape[1:]
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    tx, ty = xs + ev["flow"][0], ys + ev["flow"][1]
    vis = ev.get("vis", np.ones((H, W), bool))
    inside = (tx >= 0) & (tx <= W - 1) & (ty >= 0) & (ty <= H - 1) & np.isfinite(tx)
    m_all = vis & inside
    m_e = m_all & (ev["alpha"] <= ALPHA_MAX) if (use_alpha and "alpha" in ev) else m_all
    return xs, ys, tx, ty, m_all, m_e


def relative_motion(p1: np.ndarray, p2: np.ndarray, K: np.ndarray, seed: int):
    """(R, t) with X_2 = R X_1 + t, |t| = 1, from correspondences p1 -> p2."""
    import cv2

    if p1.shape[0] < 8:
        raise JR.ItemFailed("essential", f"{p1.shape[0]} correspondences")
    cv2.setRNGSeed(int(seed) % 2**31)
    E, mask = cv2.findEssentialMat(p1, p2, K, method=cv2.RANSAC, prob=PROB,
                                   threshold=RANSAC_PX)
    if E is None or E.shape != (3, 3):
        raise JR.ItemFailed("essential", "findEssentialMat gave no single E")
    n, R, t, _ = cv2.recoverPose(E, p1, p2, K, mask=mask)
    if n < 8:
        raise JR.ItemFailed("recoverPose", f"{n} points in front")
    return R, t.reshape(3), int(n)


def triangulate(xs, ys, tx, ty, m, K, R, t) -> tuple[np.ndarray, np.ndarray]:
    """Linear (DLT) triangulation of p -> q under P1 = K[I|0], P2 = K[R|t]: z-depth in camera
    1 (nan where it fails or a camera sees the point behind it) and the parallax angle (deg)."""
    H, W = xs.shape
    D = np.full((H, W), np.nan)
    ang = np.full((H, W), np.nan)
    if not m.any():
        return D, ang
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t.reshape(3, 1)])
    u1, v1, u2, v2 = xs[m], ys[m], tx[m], ty[m]
    A = np.stack([u1[:, None] * P1[2] - P1[0], v1[:, None] * P1[2] - P1[1],
                  u2[:, None] * P2[2] - P2[0], v2[:, None] * P2[2] - P2[1]], 1)
    _, _, Vt = np.linalg.svd(A)
    Xh = Vt[:, -1, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        X = Xh[:, :3] / Xh[:, 3:4]
    z1 = X[:, 2]
    z2 = (X @ R.T + t)[:, 2]
    ok = np.isfinite(z1) & (z1 > 0) & (z2 > 0)
    c2 = -R.T @ t
    r1 = X / np.linalg.norm(X, axis=1, keepdims=True)
    r2 = X - c2
    r2 = r2 / np.linalg.norm(r2, axis=1, keepdims=True)
    a = np.degrees(np.arccos(np.clip((r1 * r2).sum(1), -1, 1)))
    D[m] = np.where(ok, z1, np.nan)
    ang[m] = np.where(ok, a, np.nan)
    return D, ang


def self_triangulate(ev_b: dict, ev_f: dict, K: np.ndarray, seed: int) -> dict:
    """The library entry (tests call it on exact synthetic flows)."""
    xs, ys, txb, tyb, mb_all, mb_e = correspondences(ev_b)
    p1 = np.stack([xs[mb_e], ys[mb_e]], 1)
    Rb, tb, nb = relative_motion(p1, np.stack([txb[mb_e], tyb[mb_e]], 1), K, seed)
    Db, angb = triangulate(xs, ys, txb, tyb, mb_all, K, Rb, tb)
    xs, ys, txf, tyf, mf_all, mf_e = correspondences(ev_f)
    Rf, tf, nf = relative_motion(np.stack([xs[mf_e], ys[mf_e]], 1),
                                 np.stack([txf[mf_e], tyf[mf_e]], 1), K, seed + 1)
    Df, _ = triangulate(xs, ys, txf, tyf, mf_all, K, Rf, tf)
    both = np.isfinite(Db) & np.isfinite(Df) & (Df > 0)
    if both.sum() < 8:
        raise JR.ItemFailed("scale", f"{int(both.sum())} pixels triangulated both ways")
    s = float(np.median(Db[both] / Df[both]))
    if not np.isfinite(s) or s <= 0:
        raise JR.ItemFailed("scale", f"scale {s}")
    return {"D": Db, "angle": angb, "M_b": rigid.rt_to_M(Rb, tb),
            "M_f": rigid.rt_to_M(Rf, s * tf), "s": s, "n_b": nb, "n_f": nf,
            "n_both": int(both.sum())}


def produce_fn(evi: CA.CacheIndex, data: Path, grid=GRID):
    def produce(row: dict):
        got = evi.get(GI.evidence_id(row))
        if got is None:
            raise JR.ItemFailed("input", "no evidence item")
        ea, em = got
        native = tuple(em.get("native_hw") or C.png_size(KT.frame_path(
            data, row["ds"], row["sub"], row["side"], row["seq"], 10)))
        K, _ = KT.calib_K(data, row["ds"], row["sub"], row["seq"], row["side"])
        Kg = rigid.zoom_K(K, native, grid)
        res = self_triangulate(GI.evidence(ea, "10to09", grid), GI.evidence(ea, "10to11", grid),
                               Kg, int(KT.parse_id(row["id"])["seq"]))
        D = res["D"]
        valid = np.isfinite(D) & (D > 0)
        inv = np.zeros((3, *grid))
        inv[1] = np.where(valid, 1.0 / np.where(valid, D, 1.0), 0.0)
        vstack = np.zeros((3, *grid), bool)
        vstack[1] = valid
        conf = np.zeros((3, *grid))
        conf[1] = np.nan_to_num(res["angle"])
        poses = np.stack([np.linalg.inv(res["M_b"]), np.eye(4), np.linalg.inv(res["M_f"])])
        arrays = {"inv_depth": inv.astype(np.float16), "valid": CA.pack_bits(vstack),
                  "conf": conf.astype(np.float16),
                  "conf_rank": rigid.rank_u8(conf, vstack), "pose_c2w": poses,
                  "K": np.stack([Kg] * 3), "inv_depth_f32": inv.astype(np.float32),
                  "tri_angle_deg": np.nan_to_num(res["angle"]).astype(np.float16),
                  "low_parallax": CA.pack_bits(valid & (np.nan_to_num(res["angle"])
                                                        < LOW_PARALLAX_DEG))}
        meta = {"frames": [9, 10, 11], "native_hw": list(native), "model_hw": list(grid),
                "shape": [3, *grid], "K_native": K.tolist(), "pair_idx": [1, 2], "tm1_idx": 0,
                "k15_half": row.get("k15_half"), "win": row["win"],
                "teacher": {"teacher": "selftri", "evidence": em.get("source_ident")},
                "scale_s": res["s"], "n_inliers": [res["n_b"], res["n_f"]],
                "n_both": res["n_both"], "valid_frac": float(valid.mean()),
                "units": "z-depth in |t_b| = 1 units; only frame 10 has depth"}
        return arrays, meta
    return produce


def export(a, job: JR.ItemJob, rc: int) -> int:
    if rc == C.EXIT_OK and a.tf_root:
        n = GI.export_teacher_cache(job.cache_dir, Path(a.tf_root), "selftri", "evalselftri")
        C.say("P1_EXPORT", f"{n} windows -> {a.tf_root} (teacher_cache format)")
    return rc


def args_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "worker"):
        p = sub.add_parser(name)
        p.add_argument("--evidence-cache", required=True)
        p.add_argument("--data")
        p.add_argument("--out-dir", required=True)
        p.add_argument("--plan", default=str(MW.PLAN))
        p.add_argument("--max-prio", type=int)
        p.add_argument("--work-dir")
        p.add_argument("--shard")
        p.add_argument("--shard-size", type=int, default=25)
        p.add_argument("--procs", type=int, default=1)
        p.add_argument("--limit", type=int)
        p.add_argument("--tf-root", help="also export to teacher_cache format here")
    return ap


def main(argv: list[str] | None = None) -> int:
    try:
        a = args_parser().parse_args(argv)
    except SystemExit as e:
        return C.EXIT_USAGE if e.code else C.EXIT_OK
    try:
        import cv2

        data = C.data_root(a.data)
        C.refuse_sealed(a.evidence_cache, a.out_dir, a.work_dir, a.tf_root)
        evi = CA.CacheIndex(Path(a.evidence_cache))
        plan = MW.load_plan(Path(a.plan))
        rows = [r for r in plan["eval_windows"] if r["win"] == "w03-11"]
        if a.max_prio is not None:
            rows = [r for r in rows if r["prio"] <= a.max_prio]
        settings = {"evidence_identity": evi.manifest.get("identity"), "cv2": cv2.__version__,
                    "evidence_host": evi.manifest.get("host"),
                    "ransac_px": RANSAC_PX, "prob": PROB, "alpha_max": "ln 2",
                    "low_parallax_deg": LOW_PARALLAX_DEG, "grid": list(GRID)}
        identity = {"settings": settings, "code": C.code_identity(TOOL_FILES)}
        out = Path(a.out_dir)
        work = Path(a.work_dir) if a.work_dir else out / "work" / "selftri"
        job = JR.ItemJob(kind="geometry-eval", source="selftri", items=rows,
                         cache_dir=out / "cache" / "diag" / "selftri", work_dir=work,
                         shard_size=a.shard_size, shard=a.shard, identity=identity,
                         settings=settings, allow_failures=True, limit=a.limit)
        if a.cmd == "worker" or a.procs <= 1:
            if a.cmd == "run":
                job.progress.resolved({"tool": "tools/p1/self_triangulate.py",
                                       "n_items": job.total})
            rc = job.run(produce_fn(evi, data))
            return export(a, job, rc) if a.cmd == "run" else rc
        C.check_stamp(work, identity)
        job.progress.resolved({"tool": "tools/p1/self_triangulate.py", "n_items": job.total,
                               "procs": a.procs})
        args = ["--evidence-cache", a.evidence_cache, "--out-dir", a.out_dir, "--plan", a.plan,
                "--work-dir", str(work), "--shard-size", str(a.shard_size)]
        if a.data:
            args += ["--data", a.data]
        if a.max_prio is not None:
            args += ["--max-prio", str(a.max_prio)]
        rc = MP.run_workers(Path(__file__), args, a.procs, work, set(job.ids), job.progress,
                            poll_s=1.0)
        job.write_manifest()
        return export(a, job, rc)
    except C.Stop as e:
        C.say("P1_ERROR", e.msg)
        return e.code


if __name__ == "__main__":
    sys.exit(main())
