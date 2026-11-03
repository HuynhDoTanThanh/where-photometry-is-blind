#!/usr/bin/env python3
"""time_arbitration.py: PX-A5 (reports/PAPER_EXPERIMENTS.md, decision EV-0580). Wall-clock time and
peak GPU memory per image of four procedures on one G4, using the code that made the numbers:

    S          U2Flow kitti.pth two-frame: model(x, only_fw=True) on (10, 11), then resize_flow to
               the native size (eval_kitti.predict_and_score's path)
    FF         the authors' +FF: p1lib.u2flow.ff_item (two forwards on (09,10) and (10,11), then
               train_fusion_kitti.train_and_run_tiny_model, iters 2000, thr 35, lr 0.01), resize
    R          the frozen selftri rule: kitti.pth bidirectional on (10, 11) and (10, 09)
               (gate_maps.Evidence's forwards), selftri_train_maps.selftri_maps UNCHANGED (the
               essential-matrix pose, DLT triangulation, rigid render, the alpha > 2.75 and
               |T - S| <= 100 px rule) on the CPU, the switch, then resize_flow
    FF+R       FF, then the same rule against the FF flow (R's forwards and refit included)
    Rg         PX-F6 (--procs S,FF,R,Rg; tools/p1/gpu_refit.py runs it): R with the refit's two
               RANSACs in two CPU threads and the DLT triangulation on the GPU
               (gpu_refit.selftri_maps_gpu); the rule's code is otherwise R's
    FF@N       PX-T1: FF with the authors' `iterations` argument set to N (ff_item(iterations=N)),
               everything else FF's
    Rp@N       PX-F11 (tools/p1/fast_refit.py runs it): R with exact_refit_par's CPU refit
               (recoverPose on the RANSAC inliers in N chunks, the DLT in 4N chunks, N threads,
               sides overlapped; outputs byte-identical to R's, checked in the same job)
    Rpg@N      PX-F11: Rp@N's pose with Rg's GPU DLT (exact_refit_par, dev = the GPU)
    FFnd       PX-F11 (supplement sensitivity row only): FF with the cuDNN flags the authors'
               train_fusion_kitti.py runs with (torch's defaults, deterministic and benchmark
               off: 152bb38 sets neither), where FF runs under p1lib.u2flow.deterministic_cudnn
               as for the paper's +FF numbers
    FF+X       X in Rg, Rp@N, Rpg@N: FF, then that refit against the FF flow (as FF+R)

How to invoke (a launch.sh job, tools/p1/specs/time_arbitration.spec.json):
    python -u tools/p1/time_arbitration.py --data {data} --out-dir {out_dir} \
        --ckpt {data}/weights/u2flow/kitti.pth [--reps 3] [--warmup 3]
    [--order-seed S]   a fresh procedure order per item: --procs' order shuffled by
                       random.Random(S * 1000 + k) for item k (no fixed predecessor); recorded
                       per item. Without it the order is --procs' order for every item, as before.
    [--deadline-unix T] stop before an item that would start after T (partial file, exit 3)
Output: OUT/timing.json (per item, per procedure: the 3 timed passes in ms and the sha256 of the
first pass's native output and of its evidence, hashed after the timer; per procedure: the median
over items of each item's median, the IQR, peak torch.cuda.max_memory_allocated in MiB; the GPU,
the host CPU model and topology, the cgroup CPU quota, the load average, the thread settings,
torch, cv2 and numpy versions). Exit codes: 0 ok · 2 precondition · 3 stopped at the deadline
(partial, not quotable) · 64 usage (bad --procs printed before anything loads).

Why it is built this way.
- A cost label for the paper (metric s8: +FF is never quoted at the student's inference cost), so
  each procedure runs the SAME functions that produced the cached flows the numbers came from;
  nothing is re-implemented. The CPU refit is timed as it runs (single thread, as on the VM).
- torch.cuda.synchronize() brackets every pass; 3 warm-up items absorb cudnn and allocator setup;
  the job runs alone on the GPU (the launch order puts it after the FF jobs on that session).
- K comes from tools/p1/specs/time_arbitration_items.json (read on the Mac): the g0 profile
  carries frames and weights, not the KITTI-12 calibration bundle.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

ITEMS = HERE / "specs" / "time_arbitration_items.json"
PROCS = ("S", "FF", "R", "FF+R")


def args_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--weights-key", default="u2flow-kitti")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--run", default=None)
    p.add_argument("--decision", default="EV-0580",
                   help="the decision row this timing serves (PX-A5: EV-0580; PX-T1: its row)")
    p.add_argument("--device", default="cuda", help="cpu: a Mac smoke of the code path only")
    p.add_argument("--n-items", type=int, default=None, help="smoke: the first N items")
    p.add_argument("--procs", default=",".join(PROCS),
                   help="procedures to time, from S, FF, R, FF+R, Rg (PX-F6), FF@N (PX-T1), "
                        "Rp@N, Rpg@N, FFnd, FF+Rg, FF+Rp@N, FF+Rpg@N (PX-F11)")
    p.add_argument("--order-seed", type=int, default=None,
                   help="shuffle the procedure order per item with random.Random(S * 1000 + k) "
                        "(PX-F11); default: the --procs order for every item")
    p.add_argument("--deadline-unix", type=float, default=None,
                   help="stop before an item that would start after this time (exit 3)")
    return p


def parse_procs(text: str) -> tuple[tuple, dict, dict]:
    """(procs, ff_n {FF@N: N}, par_n {refit Rp@N/Rpg@N: N}); ValueError names the bad entry."""
    procs = tuple(x for x in text.split(",") if x)
    if not procs or len(set(procs)) != len(procs):
        raise ValueError(f"{text!r}: empty or repeated")
    ff_n, par_n = {}, {}

    def refit_ok(r: str) -> bool:
        head, at, n = r.partition("@")
        if r in ("R", "Rg"):
            return True
        if head not in ("Rp", "Rpg") or not at or not n.isdigit() or int(n) < 1:
            return False
        par_n[r] = int(n)
        return True

    for x in procs:
        if x in ("S", "FF", "FFnd"):
            continue
        if x.startswith("FF@"):
            n = x[3:]
            if not n.isdigit() or int(n) < 20:
                raise ValueError(f"{x}: FF@N needs an integer N >= 20 (the lr schedule steps "
                                 "every N // 20 iterations)")
            ff_n[x] = int(n)
        elif x.startswith("FF+"):
            if not refit_ok(x[3:]):
                raise ValueError(f"{x}: FF+X needs X in R, Rg, Rp@N, Rpg@N (N >= 1)")
        elif not refit_ok(x):
            raise ValueError(f"{x}: not a procedure")
    return procs, ff_n, par_n


def host_record() -> dict:
    """What the thread counts are read against: CPU model and topology, cgroup quota, load."""
    rec: dict = {"cpu_model": None, "lscpu": {}, "cgroup": None,
                 "loadavg": list(os.getloadavg()) if hasattr(os, "getloadavg") else None}
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                rec["cpu_model"] = line.split(":", 1)[1].strip()
                break
    except OSError:
        rec["cpu_model"] = platform.processor() or platform.machine()
    try:
        import subprocess

        out = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=10).stdout
        for line in out.splitlines():
            k, _, v = line.partition(":")
            if k.strip() in ("CPU(s)", "Thread(s) per core", "Core(s) per socket", "Socket(s)",
                             "Model name"):
                rec["lscpu"][k.strip()] = v.strip()
    except (OSError, ValueError):
        pass
    try:                                       # the reader coord/g4_day1.json was made with
        sys.path.insert(0, str(HERE.parents[0] / "vm"))
        import bootstrap

        rec["cgroup"] = bootstrap.cgroup_limits()
    except Exception as e:  # noqa: BLE001  (recorded, not fatal)
        rec["cgroup"] = {"error": f"{type(e).__name__}: {e}"}
    try:
        import threadpoolctl

        rec["threadpools"] = [{k: d.get(k) for k in ("user_api", "internal_api", "num_threads",
                                                     "version")}
                              for d in threadpoolctl.threadpool_info()]
    except Exception:  # noqa: BLE001  (optional: not in every lock)
        rec["threadpools"] = None
    return rec


def main(argv: list[str] | None = None) -> int:
    a = args_parser().parse_args(argv)
    if not re.fullmatch(r"EV-\d{4}", a.decision or ""):
        print(f"TIMING_FAIL --decision {a.decision!r}: not a ledger id EV-NNNN", flush=True)
        return 64
    try:
        procs, ff_n, par_n = parse_procs(a.procs)
    except ValueError as e:
        print(f"TIMING_FAIL --procs {e}", flush=True)
        return 64
    host_start = host_record()
    import cv2
    import selftri_train_maps as STM
    import torch
    from p1lib import common as C
    from p1lib import kitti as KT
    from p1lib import u2flow as UF

    from core.utils.flow_utils import resize_flow

    torch.set_num_threads(1)
    if a.device == "cuda" and not torch.cuda.is_available():
        print("TIMING_FAIL no CUDA device", flush=True)
        return 2
    dev = torch.device(a.device)
    cuda = dev.type == "cuda"

    def sync():
        if cuda:
            torch.cuda.synchronize(dev)
    doc = json.loads(ITEMS.read_text("utf-8"))
    model, ckpt_sha = UF.load_checkpoint(Path(a.ckpt), a.weights_key, dev)
    model.eval()
    tfk = UF.ff_module()
    data = Path(a.data)

    def frames(seq):
        return [KT.frame_path(data, doc["dataset"], doc["sub"], doc["side"], seq, f)
                for f in (9, 10, 11)]

    def to_native(flow_t, hw):  # flow_t (1, 2, h, w) torch on dev
        return resize_flow(flow_t, list(hw))

    import hashlib
    import random
    from concurrent.futures import ThreadPoolExecutor

    pool = None
    if any(x in ("Rg", "FF+Rg") for x in procs) or par_n:
        import gpu_refit as GR

        pool = ThreadPoolExecutor(2)
    chunk_pools, rp_paths = {}, {x: [] for x in par_n}
    if par_n:
        import exact_refit_par as ERP

        chunk_pools = {n: ThreadPoolExecutor(n) for n in set(par_n.values())}

    last_ev: list = [None]                     # the last pass's evidence, hashed after the timer

    def rule_on(base_grid, seq, K, hw, images3, refit="R"):
        # kitti.pth evidence exactly as gate_maps.Evidence.compute forms it (fwd 10->11, bwd 10->09)
        x = images3[None].to(dev)
        ev = {}
        for name, pair, kf, ka, kv in (("fwd", x[:, 1:], "f12", "a12", "vis12"),
                                       ("bwd", x[:, :-1], "b21", "a21", "vis21")):
            e = UF.bidirectional(model, pair[0], dev)
            ev[name] = {"flow": e[kf][0].float().cpu().numpy().astype(np.float64),
                        "alpha": e[ka][0, 0].float().cpu().numpy().astype(np.float64),
                        "vis": (e[kv][0, 0].cpu().numpy() > 0.5)}
        if base_grid is not None:              # FF+R: the distance is to the flow replaced
            ev["fwd"]["flow"] = base_grid
        last_ev[0] = ev
        if refit == "Rg":
            _, _, extra = GR.selftri_maps_gpu(ev["fwd"], ev["bwd"], np.asarray(K), hw, 1, dev, pool)
        elif refit in par_n:
            n = par_n[refit]
            _, _, extra = ERP.selftri_maps_par(
                ev["fwd"], ev["bwd"], np.asarray(K), hw, 1, pool, chunk_pools[n], n,
                rp_paths[refit], dev if refit.startswith("Rpg@") else None)
        else:
            _, _, extra = STM.selftri_maps(ev["fwd"], ev["bwd"], np.asarray(K), hw, 1)
        f12, rule = extra["flow_12"], extra["rule"]
        out = ev["fwd"]["flow"] if f12 is None else np.where(rule[None], np.nan_to_num(f12),
                                                             ev["fwd"]["flow"])
        with torch.no_grad():
            return to_native(torch.from_numpy(out[None]).float().to(dev), hw)

    def run_proc(proc, images3, seq, K, hw):
        x = images3[None].to(dev)
        if proc == "S":
            with torch.no_grad():
                return to_native(model(x[:, 1:], only_fw=True)["flows_f12"][-1], hw)
        if proc == "R":
            return rule_on(None, seq, K, hw, images3)
        if proc == "Rg" or proc in par_n:
            return rule_on(None, seq, K, hw, images3, refit=proc)
        if proc == "FFnd":                     # the authors' flags (torch defaults; 152bb38)
            ff = UF.ff_item(model, images3, dev)["fused"].detach()
            return to_native(ff, hw)
        with UF.deterministic_cudnn():         # the TinyModel trains: no outer no_grad
            ff = UF.ff_item(model, images3, dev, iterations=ff_n.get(proc))["fused"].detach()
        if proc == "FF" or proc in ff_n:
            return to_native(ff, hw)
        return rule_on(ff[0].float().cpu().numpy().astype(np.float64), seq, K, hw, images3,
                       refit=proc[3:])

    def sha(arrs) -> str:
        h = hashlib.sha256()
        for v in arrs:
            h.update(np.ascontiguousarray(v).tobytes())
        return h.hexdigest()[:16]

    items = doc["items"][:a.n_items] if a.n_items else doc["items"]
    per = []
    peak = {p: 0 for p in procs}
    stopped = None
    for k, it in enumerate(items):
        if a.deadline_unix is not None and time.time() > a.deadline_unix:
            stopped = f"deadline before item {it['seq']}"
            print(f"TIMING_STOPPED {stopped}", flush=True)
            break
        seq, K = it["seq"], it["K_native"]
        fl = frames(seq)
        hw = C.png_size(fl[1])
        ds = UF.frames_dataset(data, [fl])
        images3 = ds[0][0]
        warm = k < a.warmup
        order = list(procs)
        if a.order_seed is not None:
            random.Random(a.order_seed * 1000 + k).shuffle(order)
        rec = {"seq": seq, "native_hw": list(hw), "warmup": warm, "order": list(order),
               "ms": {}, "out_sha16": {}, "ev_sha16": {}}
        for proc in order:
            times = []
            for r in range(1 if warm else a.reps):
                if cuda:
                    torch.cuda.reset_peak_memory_stats(dev)
                last_ev[0] = None
                sync()
                t0 = time.perf_counter()
                nat = run_proc(proc, images3, seq, K, hw)
                sync()
                times.append(1000.0 * (time.perf_counter() - t0))
                if cuda:
                    peak[proc] = max(peak[proc], torch.cuda.max_memory_allocated(dev))
                if r == 0:                     # after the timer: hashes of what was timed
                    rec["out_sha16"][proc] = sha([nat.detach().float().cpu().numpy()])
                    e = last_ev[0]
                    rec["ev_sha16"][proc] = None if e is None else sha(
                        [e[s][k2] for s in ("fwd", "bwd") for k2 in ("flow", "alpha", "vis")])
                nat = e = None                 # nothing kept across reset_peak_memory_stats
                last_ev[0] = None
            rec["ms"][proc] = times
        per.append(rec)
        print(f"TIMING_ITEM {seq} " + " ".join(f"{p}={statistics.median(rec['ms'][p]):.1f}ms"
                                                for p in procs), flush=True)
    timed = [r for r in per if not r["warmup"]] or per
    summ = {}
    for p in procs if per else ():
        med = sorted(statistics.median(r["ms"][p]) for r in timed)
        q = statistics.quantiles(med, n=4) if len(med) >= 2 else [med[0]] * 3
        summ[p] = {"median_ms": statistics.median(med), "iqr_ms": [q[0], q[2]],
                   "n_items": len(med), "peak_mem_mib": peak[p] / 2 ** 20}
    out = {"tool": "tools/p1/time_arbitration.py", "decision": a.decision, "run": a.run,
           "ckpt_sha256": ckpt_sha, "ff": {"iters": int(tfk.iters), "thr": float(tfk.thr),
                                           "iters_at": ff_n},
           "stopped": stopped,
           "protocol": {"reps": a.reps, "warmup_items": a.warmup, "sync": "torch.cuda.synchronize",
                        "stat": "median over items of each item's median pass",
                        "torch_threads": 1, "chunk_threads": par_n, "order_seed": a.order_seed,
                        "order": ("--procs order" if a.order_seed is None else
                                  "random.Random(order_seed * 1000 + k).shuffle per item k"),
                        "recoverPose_paths": {x: {k: v.count(k) for k in
                                                  ("chunked", "fallback", "serial")}
                                              for x, v in rp_paths.items()}},
           "host": {"gpu": torch.cuda.get_device_name(dev) if cuda else "cpu",
                    "cpu": platform.processor() or platform.machine(),
                    "cpu_count": os.cpu_count(), "torch": torch.__version__,
                    "cv2": cv2.__version__, "numpy": np.__version__,
                    "cv2_threads": cv2.getNumThreads(),
                    "thread_env": {k: os.environ.get(k) for k in
                                   ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
                    "sched_affinity": (len(os.sched_getaffinity(0))
                                       if hasattr(os, "sched_getaffinity") else None),
                    "cudnn_default_flags": {"deterministic": torch.backends.cudnn.deterministic,
                                            "benchmark": torch.backends.cudnn.benchmark},
                    "packed_with_env": os.environ.get("U2_PACKED_WITH"),
                    "start": host_start, "end": host_record()},
           "summary": summ, "items": per}
    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    C.atomic_json(od / "timing.json", out)
    if stopped:
        return 3
    print("TIMING_OK " + json.dumps({p: round(v["median_ms"], 1) for p, v in summ.items()}),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
