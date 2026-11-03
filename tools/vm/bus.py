"""The telemetry bus of one VM run: what a job reports, in the one vocabulary predicates read.

What it does. A training or inference process appends one JSON record per call to
<run_dir>/bus.jsonl ({"step", "utc", field: value, ...}); tools/vm/job.py (the supervisor) adds
gpu_util samples to the same file and writes <run_dir>/heartbeat.json; tools/colab/poll.sh copies
new lines to the Mac, where `note.py check RUN --kind liveness --telemetry bus.jsonl` evaluates the
registered liveness predicate on them. The field names are exactly the telemetry enum of
tools/lib/predicate_schema.json: loss_total, loss_photo, loss_teacher, s_per_step, gpu_util,
finite_all, dev_epe, dev_fl. Free extras must start with `x_` (never read by a predicate).

How to invoke.
    In the job (job.py puts <code>/tools/vm on PYTHONPATH and sets U2_BUS):
        import bus
        bus.log(step, loss_total=..., loss_photo=..., loss_teacher=...)   # s_per_step and
                                                                         # finite_all derived
        bus.log(step, dev_epe=..., dev_fl=...)      # in-run DEV-half validation only
        bus.resolved_config({...})                  # once, after the config is resolved
    Anywhere (python >= 3.9):
        python tools/vm/bus.py check FILE           # exit 0 valid, 1 refused (reasons on stderr)
        python tools/vm/bus.py tail FILE --from-line N [--max 200]   # BUS_LINE <json> lines
Exit codes (CLI): 0 ok | 1 a record is refused | 64 usage.

Why it is built this way.
- One vocabulary. predicates.py validates liveness clauses against the schema's telemetry enum
  and note.py refuses a telemetry file without a numeric step; a second spelling ('loss', 'epe')
  would leave a liveness predicate at not-yet for ever, which is TODO under another name
  (predicates.py WHY). So an unknown field is an error here, at the writer, not a silent drop.
- Nothing held-out crosses the bus. The seal (AGENTS.md hard rule 4) forbids any held-out number
  before prereg-v3, and the all-200 aggregate contains the held-out half; a key that names
  heldout, held_out, full200, all200 or test is refused, extras included. dev_epe and dev_fl are
  in-run liveness signals on the DEV half only and are never a result (hard rule 1: never quote
  an in-run number; results enter only through note.py result --from-eval).
- 'A heartbeat proves a writer is alive, not that work happens' (colab-fleet section 10: idle VMs
  said 'training' for a day [T]). So the bus carries step-indexed work (losses, s/step, GPU
  utilisation), and finite_all is derived from the losses of the same record when the job does
  not state it: a NaN loss turns finite_all false at once.
- Appends are one write of one line under O_APPEND, so the supervisor's gpu_util samples and the
  job's records never interleave inside a line; readers skip a torn last line.
- No numpy/torch import: values are coerced with float(); a tensor must be .item()'d by the job.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import re
import sys
import time

FIELDS = ("loss_total", "loss_photo", "loss_teacher", "s_per_step", "gpu_util", "finite_all",
          "dev_epe", "dev_fl")
LOSSES = ("loss_total", "loss_photo", "loss_teacher")
META = ("step", "utc", "src")
SEALED_RX = re.compile(r"held[_-]?out|full[_-]?200|all[_-]?200|(^|_)test($|_)", re.I)
EXTRA_RX = re.compile(r"^x_[a-z0-9_]{1,40}$")
RESOLVED_NAME = "resolved_config.json"

_last: dict = {}


class BusError(ValueError):
    """A record the bus refuses: the message names the field and the rule."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()  # noqa: UP017


def _num(v: object, key: str) -> float:
    if isinstance(v, bool):
        raise BusError(f"{key}: a boolean is not a number")
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise BusError(f"{key}: {v!r} is not a number") from None


def make_record(step: object, fields: dict, src: str = "job", now: float | None = None,
                last: dict | None = None) -> dict:
    """Validate one record and derive s_per_step and finite_all; raise BusError on a refusal."""
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise BusError(f"step must be a non-negative int, not {step!r}")
    rec: dict = {"step": step, "utc": utc_now(), "src": src}
    for k, v in fields.items():
        if SEALED_RX.search(k):
            raise BusError(f"{k}: the bus never carries a held-out, all-200 or test number "
                           "(AGENTS.md hard rule 4)")
        if k in META:
            raise BusError(f"{k} is set by the bus")
        if k == "finite_all":
            if not isinstance(v, bool):
                raise BusError("finite_all must be true or false")
            rec[k] = v
        elif k in FIELDS:
            x = _num(v, k)
            rec[k] = x if math.isfinite(x) else str(x)  # JSON has no NaN; 'nan' stays visible
        elif EXTRA_RX.match(k):
            rec[k] = v if isinstance(v, (str, bool)) or v is None else _num(v, k)
        else:
            raise BusError(f"{k} is not a telemetry field ({', '.join(FIELDS)}) and not an x_ "
                           "extra (tools/lib/predicate_schema.json owns the vocabulary)")
    losses = [k for k in LOSSES if k in rec]
    if "finite_all" not in rec and losses:
        rec["finite_all"] = all(isinstance(rec[k], float) for k in losses)
    t = time.time() if now is None else now
    prev = _last if last is None else last
    if "s_per_step" not in rec and src == "job" and prev.get("step") is not None \
            and step > prev["step"]:
        rec["s_per_step"] = round((t - prev["t"]) / (step - prev["step"]), 6)
    if src == "job":
        prev.update(step=step, t=t)
    return rec


def bus_path(path: str | None = None) -> str:
    p = path or os.environ.get("U2_BUS")
    if not p:
        raise BusError("U2_BUS is not set (job.py sets it to <run_dir>/bus.jsonl)")
    return p


def append(rec: dict, path: str | None = None) -> None:
    line = (json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n").encode()
    fd = os.open(bus_path(path), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)


def log(step: int, path: str | None = None, **fields: object) -> dict:
    """The job's call: validate, derive, append; returns the record written."""
    rec = make_record(step, fields)
    append(rec, path)
    return rec


def resolved_config(cfg: dict, path: str | None = None) -> str:
    """Write <run_dir>/resolved_config.json once the job has resolved its config. job.py turns
    it into the RESOLVED_CONFIG sentinel that launch.sh waits for (10 min)."""
    run_dir = os.path.dirname(bus_path(path))
    out = os.path.join(run_dir, RESOLVED_NAME)
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"utc": utc_now(), "config": cfg}, f, indent=1, sort_keys=True, default=str)
    os.replace(tmp, out)
    print("RESOLVED_CONFIG " + out, flush=True)
    return out


def read(path: str) -> tuple[list[dict], list[str]]:
    """Records of a bus file, and the problems found (a torn last line is not a problem)."""
    recs, bad = [], []
    try:
        with open(path, errors="replace") as f:
            lines = f.read().split("\n")
    except OSError as e:
        return [], [f"{path}: {e}"]
    for n, ln in enumerate(lines, start=1):
        if not ln.strip():
            continue
        try:
            r = json.loads(ln)
        except ValueError:
            if n == len(lines):
                continue
            bad.append(f"line {n}: not JSON")
            continue
        if not isinstance(r, dict) or isinstance(r.get("step"), bool) \
                or not isinstance(r.get("step"), int):
            bad.append(f"line {n}: needs an integer step")
            continue
        for k in r:
            if k in META:
                continue
            if SEALED_RX.search(k) or not (k in FIELDS or EXTRA_RX.match(k)):
                bad.append(f"line {n}: field {k} is refused")
        recs.append(r)
    return recs, bad


def main(argv: list[str]) -> int:
    if len(argv) < 3 or argv[1] not in ("check", "tail"):
        print(__doc__.split("Exit codes")[0].split("How to invoke.")[1], file=sys.stderr)
        return 64
    path = argv[2]
    if argv[1] == "check":
        recs, bad = read(path)
        for b in bad:
            print("BUS_REFUSED " + b, file=sys.stderr)
        print(f"BUS_CHECK records={len(recs)} refused={len(bad)}")
        return 1 if bad else 0
    start, mx = 0, 200
    rest = argv[3:]
    for i, a in enumerate(rest):
        if a == "--from-line" and i + 1 < len(rest):
            start = int(rest[i + 1])
        if a == "--max" and i + 1 < len(rest):
            mx = int(rest[i + 1])
    try:
        with open(path, errors="replace") as f:
            lines = f.read().split("\n")
    except OSError:
        lines = []
    complete = lines[:-1]  # the text after the last newline may be a torn write
    for ln in complete[start:start + mx]:
        print("BUS_LINE " + ln)
    print(f"BUS_TAIL next={min(len(complete), start + mx)} total={len(complete)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
