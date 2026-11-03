"""K worker processes over one item job: each worker owns `--shard j/K` of the shards, the parent
checks the stamp once, reports progress for all of them on the bus, forwards TERM and returns the
worst exit code (run_ff_shard.py --procs K, self_triangulate.py --procs K).

How to use (library only):
    rc = run_workers(argv_without_shard, k=8, work_dir=W, total=n, progress=job.progress,
                     env_extra={"U2_P1_WORKER_FRAC": "0.05"})
The workers are this same script with `worker` in place of `run` and `--shard j/K`.

Why it is built this way.
- The plan picks K_FF <= cpu_quota - 4 single-core processes, each < 1 GB VRAM [P W2-09]; one
  process per core keeps the FF algorithm unbatched [P W2-08 'KHÔNG batch'].
- Only the parent writes to the bus (U2_BUS is removed from the workers' environment), so the
  step count is one monotone series of finished items (runlib classify reads it for STALLED);
  the parent counts finished items on disk, which is also what a rerun would skip.
- Workers stay in the job's process group, so job.py's stop and its overdue kill reach them [V
  tools/vm/job.py run_job: the job's group is killed after the leader exits].
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from . import common as C


def count_done(work_dir: Path, ids: set[str]) -> tuple[int, int]:
    done = failed = 0
    for p in Path(work_dir).iterdir() if Path(work_dir).is_dir() else []:
        n = p.name
        if n.endswith(".npz") and n[:-4] in ids:
            done += 1
        elif n.endswith(".failed.json") and n[:-12] in ids:
            failed += 1
    return done + failed, failed


def run_workers(script: Path, args: list[str], k: int, work_dir: Path, ids: set[str],
                progress: C.Progress, env_extra: dict | None = None, poll_s: float = 10.0) -> int:
    env = dict(os.environ)
    env.pop("U2_BUS", None)
    env.update({kk: str(v) for kk, v in (env_extra or {}).items()})
    env.setdefault("OMP_NUM_THREADS", "1")
    procs = []
    for j in range(k):
        cmd = [sys.executable, "-u", str(script), "worker", *args, "--shard", f"{j}/{k}"]
        procs.append(subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL))
        C.say("P1_WORKER", f"{j}/{k} pid={procs[-1].pid}")
    stop = {"sig": None}

    def on_term(signum, frame):  # noqa: ARG001
        stop["sig"] = signum
        for p in procs:
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)

    old = signal.signal(signal.SIGTERM, on_term)
    try:
        while any(p.poll() is None for p in procs):
            done, failed = count_done(work_dir, ids)
            progress.tick(done, failed, procs=sum(p.poll() is None for p in procs))
            time.sleep(poll_s)
    finally:
        signal.signal(signal.SIGTERM, old)
    done, failed = count_done(work_dir, ids)
    progress.tick(done, failed, force=True, procs=0)
    rcs = [p.returncode for p in procs]
    C.say("P1_WORKERS_EXIT", " ".join(str(r) for r in rcs))
    if stop["sig"] is not None:
        return C.EXIT_TERM
    return max(rcs) if rcs else C.EXIT_OK
