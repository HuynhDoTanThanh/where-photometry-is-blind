#!/usr/bin/env python3
"""gpu_refit.py: PX-F6 (reports/PAPER_EXPERIMENTS.md; decision row before the run). The frozen
selftri rule with its refit moved where it is cheap: the two essential-matrix RANSACs (OpenCV,
CPU) run concurrently in two threads, and the dense DLT triangulation (one 4x4 SVD per pixel,
about 2/3 of the CPU refit's time on a Mac profile) runs in torch float64 on the GPU. The rest is
the CPU chain's own code: self_triangulate.correspondences / relative_motion, the scale rule, and
selftri_train_maps.selftri_maps (fp16 inverse depth, rigid.render_pair, the alpha > 2.75 and
|T - S| <= 100 px rule), so the rule is unchanged by construction.

How to invoke (a launch.sh job, tools/p1/specs/gpu_refit.spec.json, profile g0):
    python -u tools/p1/gpu_refit.py --data {data} --out-dir {out_dir} \
        --ckpt {data}/weights/u2flow/kitti.pth --run {run_id}
    1. the equality check on all 194 KITTI-12 training ids (tools/p1/specs/k12_items_K.json):
       per item, the CPU selftri_maps and selftri_maps_gpu on the same evidence; the switched
       masks, the max |flow difference| and the per-image Fl outlier counts (K15 rule) of both
       native outputs against the KITTI-12 GT -> OUT/gpu_refit_check.json;
    2. the PX-A5 timing protocol (time_arbitration.py, EV-0628: items 000000-000019, 3 warm-up,
       median of 3 passes, torch.cuda.synchronize) with the procedures S, FF, R, Rg ->
       OUT/timing.json.
    Mac smoke (code path only, never a number): --device cpu --n-check 2 --n-time 1
Exit codes: 0 ok · 2 precondition · 64 usage.

Why it is built this way
- The timing is a cost label; it is only usable if the GPU refit gives the same answers, so the
  check runs first on every KITTI-12 image, in the same job, on the same evidence.
- Nothing of the rule is re-implemented: selftri_maps is called with self_triangulate swapped for
  the version below, whose only differences are the thread pool and the torch SVD.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "eval"))

import numpy as np  # noqa: E402

ITEMS_K = HERE / "specs" / "k12_items_K.json"


def triangulate_torch(xs, ys, tx, ty, m, K, R, t, dev) -> tuple[np.ndarray, np.ndarray]:
    """self_triangulate.triangulate's math, line for line, in torch float64 on dev."""
    import torch

    H, W = xs.shape
    D = np.full((H, W), np.nan)
    ang = np.full((H, W), np.nan)
    if not m.any():
        return D, ang

    def to(a):
        return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float64, device=dev)

    P1 = to(K @ np.hstack([np.eye(3), np.zeros((3, 1))]))
    P2 = to(K @ np.hstack([R, t.reshape(3, 1)]))
    u1, v1, u2, v2 = to(xs[m]), to(ys[m]), to(tx[m]), to(ty[m])
    A = torch.stack([u1[:, None] * P1[2] - P1[0], v1[:, None] * P1[2] - P1[1],
                     u2[:, None] * P2[2] - P2[0], v2[:, None] * P2[2] - P2[1]], 1)
    _, _, Vh = torch.linalg.svd(A)
    Xh = Vh[:, -1, :]
    X = Xh[:, :3] / Xh[:, 3:4]
    Rt, tt = to(R), to(t.reshape(3))
    z1 = X[:, 2]
    z2 = (X @ Rt.T + tt)[:, 2]
    ok = torch.isfinite(z1) & (z1 > 0) & (z2 > 0)
    c2 = -Rt.T @ tt
    r1 = X / torch.linalg.norm(X, dim=1, keepdim=True)
    r2 = X - c2
    r2 = r2 / torch.linalg.norm(r2, dim=1, keepdim=True)
    a = torch.rad2deg(torch.arccos(torch.clamp((r1 * r2).sum(1), -1, 1)))
    nan = torch.tensor(float("nan"), dtype=torch.float64, device=dev)
    D[m] = torch.where(ok, z1, nan).cpu().numpy()
    ang[m] = torch.where(ok, a, nan).cpu().numpy()
    return D, ang


def make_self_triangulate(dev, pool):
    """self_triangulate.self_triangulate with the two RANSACs in pool and the DLT on dev."""
    import rigid
    import self_triangulate as ST
    from p1lib import jobrun as JR

    def self_triangulate_gpu(ev_b: dict, ev_f: dict, K: np.ndarray, seed: int) -> dict:
        xs, ys, txb, tyb, mb_all, mb_e = ST.correspondences(ev_b)
        xf, yf, txf, tyf, mf_all, mf_e = ST.correspondences(ev_f)
        fut_b = pool.submit(ST.relative_motion, np.stack([xs[mb_e], ys[mb_e]], 1),
                            np.stack([txb[mb_e], tyb[mb_e]], 1), K, seed)
        fut_f = pool.submit(ST.relative_motion, np.stack([xf[mf_e], yf[mf_e]], 1),
                            np.stack([txf[mf_e], tyf[mf_e]], 1), K, seed + 1)
        try:
            Rb, tb, nb = fut_b.result()     # the CPU chain raises the backward failure first
            Db, angb = triangulate_torch(xs, ys, txb, tyb, mb_all, K, Rb, tb, dev)
            Rf, tf, nf = fut_f.result()
        finally:
            fut_f.cancel()
            try:
                fut_f.result()
            except Exception:  # noqa: BLE001  (already raised above, or not needed)
                pass
        Df, _ = triangulate_torch(xf, yf, txf, tyf, mf_all, K, Rf, tf, dev)
        both = np.isfinite(Db) & np.isfinite(Df) & (Df > 0)
        if both.sum() < 8:
            raise JR.ItemFailed("scale", f"{int(both.sum())} pixels triangulated both ways")
        s = float(np.median(Db[both] / Df[both]))
        if not np.isfinite(s) or s <= 0:
            raise JR.ItemFailed("scale", f"scale {s}")
        return {"D": Db, "angle": angb, "M_b": rigid.rt_to_M(Rb, tb),
                "M_f": rigid.rt_to_M(Rf, s * tf), "s": s, "n_b": nb, "n_f": nf,
                "n_both": int(both.sum())}

    return self_triangulate_gpu


def selftri_maps_gpu(ev_fwd, ev_bwd, K_native, native_hw, seed, dev, pool):
    """selftri_train_maps.selftri_maps UNCHANGED, with its self_triangulate swapped."""
    import selftri_train_maps as STM

    orig = STM.ST.self_triangulate
    STM.ST.self_triangulate = make_self_triangulate(dev, pool)
    try:
        return STM.selftri_maps(ev_fwd, ev_bwd, K_native, native_hw, seed)
    finally:
        STM.ST.self_triangulate = orig


def evidence(model, images3, dev) -> dict:
    """kitti.pth evidence of one (09, 10, 11) item as gate_maps.Evidence forms it: fwd 10->11,
    bwd 10->09, flows and alphas cast to float64, vis = 1 - occlusion > 0.5. Shared by
    gpu_refit.check and fast_refit.check (time_arbitration.rule_on keeps its own copy: its body is
    the timed region of EV-0628/EV-0672 and is not changed)."""
    from p1lib import u2flow as UF

    ev = {}
    for name, pair, kf, ka, kv in (("fwd", images3[1:], "f12", "a12", "vis12"),
                                   ("bwd", images3[:-1], "b21", "a21", "vis21")):
        e = UF.bidirectional(model, pair, dev)
        ev[name] = {"flow": e[kf][0].float().cpu().numpy().astype(np.float64),
                    "alpha": e[ka][0, 0].float().cpu().numpy().astype(np.float64),
                    "vis": (e[kv][0, 0].cpu().numpy() > 0.5)}
    return ev


def fl_count(flow: np.ndarray, gt_png: Path) -> tuple[int, int]:
    """(outliers, valid) of the K15 rule on a KITTI GT png: EPE > 3 and > 5 % of |GT|."""
    import regions as rg

    g, v = rg.decode_kitti_flow(gt_png.read_bytes())
    val = v > 0
    epe = np.sqrt(((flow.transpose(1, 2, 0) - g) ** 2).sum(-1))
    mag = np.sqrt((g ** 2).sum(-1))
    out = (epe > 3.0) & (epe > 0.05 * mag) & val
    return int(out.sum()), int(val.sum())


def check(a, dev) -> dict:
    import selftri_train_maps as STM
    import torch
    from p1lib import common as C
    from p1lib import kitti as KT
    from p1lib import u2flow as UF

    from core.utils.flow_utils import resize_flow

    doc = json.loads(ITEMS_K.read_text("utf-8"))
    model, ckpt_sha = UF.load_checkpoint(Path(a.ckpt), "u2flow-kitti", dev)
    model.eval()
    data = Path(a.data)
    gt_dir = data / "kitti12" / "training" / "flow_occ"
    per, worst = [], 0.0
    items = doc["items"][:a.n_check] if a.n_check else doc["items"]
    with ThreadPoolExecutor(2) as pool:
        for it in items:
            seq, K = it["seq"], np.asarray(it["K_native"], np.float64)
            fl = [KT.frame_path(data, doc["dataset"], doc["sub"], doc["side"], seq, f)
                  for f in (9, 10, 11)]
            hw = C.png_size(fl[1])
            images3 = UF.frames_dataset(data, [fl])[0][0]
            ev = evidence(model, images3, dev)
            outs = {}
            for tag in ("cpu", "gpu"):
                if tag == "cpu":
                    _, meta, ex = STM.selftri_maps(ev["fwd"], ev["bwd"], K, hw, 1)
                else:
                    _, meta, ex = selftri_maps_gpu(ev["fwd"], ev["bwd"], K, hw, 1, dev, pool)
                f12, rule = ex["flow_12"], ex["rule"]
                o = ev["fwd"]["flow"] if f12 is None else np.where(
                    rule[None], np.nan_to_num(f12), ev["fwd"]["flow"])
                with torch.no_grad():
                    nat = resize_flow(torch.from_numpy(o[None]).float().to(dev), list(hw))
                outs[tag] = (f12, rule, nat[0].cpu().numpy(), meta.get("why_unusable"))
            (fc, rc, nc, wc), (fg, rg_, ng, wg) = outs["cpu"], outs["gpu"]
            same_mask = bool(np.array_equal(rc, rg_))
            dflow = 0.0
            if fc is not None and fg is not None:
                both = np.isfinite(fc).all(0) & np.isfinite(fg).all(0)
                dflow = float(np.abs(fc[:, both] - fg[:, both]).max()) if both.any() else 0.0
            dnat = float(np.abs(nc - ng).max())
            gt_png = gt_dir / f"{seq}_10.png"
            oc, nv = fl_count(nc, gt_png)
            og, _ = fl_count(ng, gt_png)
            worst = max(worst, dnat)
            per.append({"seq": seq, "why_cpu": wc, "why_gpu": wg, "same_mask": same_mask,
                        "n_switched_cpu": int(rc.sum()), "n_switched_gpu": int(rg_.sum()),
                        "max_abs_grid_flow_diff": dflow, "max_abs_native_diff": dnat,
                        "fl_out_cpu": oc, "fl_out_gpu": og, "n_valid": nv})
            print(f"GPU_REFIT_CHECK {seq} mask={same_mask} dnat={dnat:.3g} fl {oc}/{og}",
                  flush=True)
    eq = all(p["same_mask"] and p["fl_out_cpu"] == p["fl_out_gpu"]
             and p["why_cpu"] == p["why_gpu"] for p in per)
    return {"tool": "tools/p1/gpu_refit.py", "ckpt_sha256": ckpt_sha, "n_items": len(per),
            "equal": eq, "max_abs_native_diff": worst,
            "fl_cpu_px_pooled_pct": 100.0 * sum(p["fl_out_cpu"] for p in per)
            / max(1, sum(p["n_valid"] for p in per)),
            "fl_gpu_px_pooled_pct": 100.0 * sum(p["fl_out_gpu"] for p in per)
            / max(1, sum(p["n_valid"] for p in per)),
            "items": per}


def args_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--run", default=None)
    p.add_argument("--device", default="cuda", help="cpu: a Mac smoke of the code path only")
    p.add_argument("--n-check", type=int, default=0, help="smoke: the first N check items")
    p.add_argument("--n-time", type=int, default=0, help="smoke: the first N timing items")
    return p


def main(argv: list[str] | None = None) -> int:
    a = args_parser().parse_args(argv)
    import time_arbitration as TA
    import torch
    from p1lib import common as C

    torch.set_num_threads(1)
    if a.device == "cuda" and not torch.cuda.is_available():
        print("GPU_REFIT_FAIL no CUDA device", flush=True)
        return 2
    dev = torch.device(a.device)
    prog = C.Progress(1, "GPU_REFIT_PROGRESS")
    prog.resolved({"tool": "tools/p1/gpu_refit.py", "run": a.run, "device": a.device,
                   "n_check": a.n_check or 194, "timing": "time_arbitration.py, procs S,FF,R,Rg"})
    doc = check(a, dev)
    doc["smoke"] = bool(a.n_check)
    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    C.atomic_json(od / "gpu_refit_check.json", doc)
    print(f"GPU_REFIT_CHECK_DONE equal={doc['equal']} max_abs_native_diff="
          f"{doc['max_abs_native_diff']:.3g} fl {doc['fl_cpu_px_pooled_pct']:.4f}/"
          f"{doc['fl_gpu_px_pooled_pct']:.4f}", flush=True)
    targv = ["--data", a.data, "--out-dir", a.out_dir, "--ckpt", a.ckpt, "--device", a.device,
             "--procs", "S,FF,R,Rg"]
    if a.run:
        targv += ["--run", a.run]
    if a.n_time:
        targv += ["--n-items", str(a.n_time), "--warmup", "0", "--reps", "1"]
    rc = TA.main(targv)
    prog.tick(1)
    return rc


if __name__ == "__main__":
    sys.exit(main())
