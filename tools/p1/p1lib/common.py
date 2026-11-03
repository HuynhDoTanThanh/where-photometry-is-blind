"""Shared plumbing of the P1 tools: exit codes, atomic writes, the seal refusal, shard selection,
progress on the VM bus, settings stamps and failure lists.

How to use (library only):
    from p1lib import common as C        # tools/p1 on sys.path (every P1 script does this)
    C.refuse_sealed(path)              # Stop(50) for anything under U2FLOW_data/sealed
    k, n = C.parse_shard("0/4")        # --shard k/n
    prog = C.Progress(total=788); prog.resolved({...}); prog.tick(done, failed)
    C.check_stamp(work_dir, identity)  # Stop(3) when a dir holds another recipe's outputs

Exit codes shared by every P1 tool (each header repeats the ones it uses):
    0 ok · 1 items left or failed (run it again: done items are skipped) · 2 precondition
    (missing inputs, weights or code) · 3 refused (settings stamp or pin mismatch, a stub on a GPU
    host, an export of a held-out id) · 50 seal (a path under the seal, or GT asked for a
    non-dev id; a STOP, AGENTS.md section 4) · 64 usage · 143 stopped by TERM.

Why it is built this way.
- One copy of the rules every wrapper must keep (colab-fleet references/launcher.md sections 2-5):
  a job that is not train.py or eval_kitti.py must write resolved_config.json itself, or
  launch.sh waits 600 s for RESOLVED_CONFIG and fails the run [V tools/vm/job.py:926-931, the
  PORT_TRAINERS / RESOLVED_AT_START note]; progress goes through tools/vm/bus.py so STALLED and
  NO_TELEMETRY (runlib classify) see work, with only x_ extras beside the schema's fields.
- Resumable by construction: every item is written to a dot-named temp file and renamed, so a
  killed worker never leaves a half item that a rerun would skip (the SAM producer's rule,
  tools/vm/sam_masks.py header).
- Resume never mixes recipes: a settings stamp (.p1_stamp.json, a dot-file, never shipped) pins
  the method identity of a work dir, as sam_masks.py does with .sam_settings.json (Tier 1c review).
- Hard rule 4: the seal is refused by path here, before any read; GT is never read by these tools.
- Python >= 3.9 standard library at import (numpy only inside functions that need it).
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import json
import os
import re
import shutil
import struct
import sys
import time
from pathlib import Path

P1_DIR = Path(__file__).resolve().parents[1]
REPO = P1_DIR.parents[1]
EXIT_OK, EXIT_FAIL, EXIT_PRE, EXIT_REFUSED, EXIT_SEAL, EXIT_USAGE = 0, 1, 2, 3, 50, 64
EXIT_TERM = 143
STAMP = ".p1_stamp.json"
SEALED_RX = re.compile(r"u2flow_data/sealed|/sealed(/|$)|data_scene_flow", re.I)
SHARD_RX = re.compile(r"^(\d+)/(\d+)$")


class Stop(Exception):
    """A refusal or precondition failure with its exit code."""

    def __init__(self, code: int, msg: str):
        super().__init__(msg)
        self.code, self.msg = code, msg


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()  # noqa: UP017


def say(tag: str, msg: str = "") -> None:
    print(f"{tag} {msg}".rstrip(), flush=True)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def atomic_bytes(path: Path, data: bytes) -> None:
    """Write via a dot-named temp file in the same dir, then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def atomic_json(path: Path, doc: object) -> None:
    atomic_bytes(path, (json.dumps(doc, indent=1, sort_keys=True) + "\n").encode())


def load_json(path: Path | str) -> dict:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if not isinstance(d, dict):
        raise Stop(EXIT_PRE, f"{path}: not a JSON object")
    return d


def refuse_sealed(*paths: object) -> None:
    """Stop(50) when any path lies under the seal or names the KITTI-15 GT source archive."""
    for p in paths:
        if p is None:
            continue
        s = str(Path(str(p)).expanduser().resolve()) if str(p) else ""
        if SEALED_RX.search(s) or SEALED_RX.search(str(p)):
            raise Stop(EXIT_SEAL, f"{p} is under the seal (AGENTS.md hard rule 4): never an "
                                  "input or output of a P1 tool")


def data_root(arg: str | None) -> Path:
    v = arg or os.environ.get("U2_DATA")
    if not v:
        raise Stop(EXIT_USAGE, "--data (or U2_DATA) is needed")
    refuse_sealed(v)
    return Path(v)


def parse_shard(spec: str | None) -> tuple[int, int]:
    if not spec:
        return 0, 1
    m = SHARD_RX.match(spec)
    if not m or int(m.group(2)) < 1 or not 0 <= int(m.group(1)) < int(m.group(2)):
        raise Stop(EXIT_USAGE, f"--shard must be k/n with 0 <= k < n, not {spec!r}")
    return int(m.group(1)), int(m.group(2))


def shard_plan(ids: list[str], size: int) -> list[list[str]]:
    """Consecutive chunks of the (already ordered) id list; shard i is ids[i*size:(i+1)*size]."""
    size = max(1, int(size))
    return [ids[i:i + size] for i in range(0, len(ids), size)]


def mine(n_shards: int, k: int, n: int) -> list[int]:
    """The shard indices a --shard k/n worker owns: i % n == k (stable across reruns)."""
    return [i for i in range(n_shards) if i % n == k]


WEIGHTS_JSON = REPO / "data" / "manifest" / "weights.json"


def weights_pin(key: str) -> dict:
    """One entry of data/manifest/weights.json (data-bundles owns the file, teacher-factory the
    teacher entries); the P1 jobs verify every checkpoint's sha256 against it."""
    d = load_json(WEIGHTS_JSON)["weights"]
    if key not in d:
        raise Stop(EXIT_PRE, f"{key}: not in data/manifest/weights.json")
    return d[key]


def deps_dir(lock_rel: str | None, data: Path, pattern: str | None, what: str) -> str | None:
    """A teacher-factory install dir. With --deps-lock (env/teachers/<t>.txt, relative to the
    repo) the lock is installed here first by tools/teacher/teacher_deps.install: idempotent and
    content-addressed (U2_DATA/pydeps/teacher-<t>-<sha12>/), so a fresh session needs no separate
    install job; otherwise the --code-dir / --pydeps glob must match exactly one dir."""
    if not lock_rel:
        return resolve_dir(pattern, what)
    tdir = str(REPO / "tools" / "teacher")
    if tdir not in sys.path:
        sys.path.insert(0, tdir)
    import teacher_deps as td  # noqa: PLC0415

    lock = Path(lock_rel) if Path(lock_rel).is_absolute() else REPO / lock_rel
    try:
        return td.install(td.parse_lock(lock), Path(data))["dir"]
    except td.DepsError as e:
        raise Stop(e.code, f"--deps-lock {lock_rel}: {e.msg}") from None
    except OSError as e:
        raise Stop(EXIT_PRE, f"--deps-lock {lock_rel}: {e}") from None


def resolve_dir(pattern: str | None, what: str) -> str | None:
    """A directory argument that may be a glob (teacher-factory names its install dirs by the
    lock's sha12, e.g. {data}/pydeps/teacher-colmap-*): exactly one directory must match."""
    if not pattern:
        return None
    import glob

    hits = sorted(p for p in glob.glob(pattern) if os.path.isdir(p))
    if len(hits) != 1:
        raise Stop(EXIT_PRE, f"{what} {pattern}: {len(hits)} matching dirs, need exactly 1")
    return hits[0]


def gpu_host() -> bool:
    return shutil.which("nvidia-smi") is not None


def host_record() -> dict:
    """Where an artefact was made (provenance; prereg/drafts/GATE-1.json _row_conventions): the
    GPU of this host (nvidia-smi; None on a CPU host such as the Mac), the platform, and the
    libraries whose versions move results between hosts (numpy; cv2 for PnP, RANSAC and
    findEssentialMat, Mac 5.0.0 vs VM 4.14.0 [V env/lock_G0.txt:42])."""
    import platform
    import subprocess

    rec: dict = {"platform": platform.platform(), "machine": platform.machine(),
                 "python": sys.version.split()[0], "gpu_name": None, "numpy": None, "cv2": None}
    for mod in ("numpy", "cv2"):
        try:
            rec[mod] = __import__(mod).__version__
        except ImportError:
            pass
    if gpu_host():
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=30)
            names = [x.strip() for x in r.stdout.splitlines() if x.strip()]
            rec["gpu_name"] = names[0] if names else None
        except (OSError, subprocess.SubprocessError):
            pass
    return rec


def png_size(path: Path | str) -> tuple[int, int]:
    """(H, W) from a PNG's IHDR chunk, without decoding the image."""
    with open(path, "rb") as f:
        head = f.read(24)
    if len(head) < 24 or head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
        raise Stop(EXIT_PRE, f"{path}: not a PNG")
    w, h = struct.unpack(">II", head[16:24])
    return int(h), int(w)


def read_rgb(path: Path | str):
    """uint8 H x W x 3 RGB (cv2.imread, then BGR -> RGB)."""
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise Stop(EXIT_PRE, f"{path}: unreadable image")
    return img[:, :, ::-1].copy()


def code_identity(files: list[Path]) -> dict:
    """sha256 of each tool file that defines a method (a recipe change is a new stamp)."""
    return {str(Path(f).resolve().relative_to(REPO)): sha256_file(f) for f in files}


def check_stamp(work_dir: Path, identity: dict) -> None:
    """Write the stamp into a fresh dir; Stop(3) when the dir holds another recipe's outputs."""
    work_dir.mkdir(parents=True, exist_ok=True)
    p = work_dir / STAMP
    if p.is_file():
        have = load_json(p)
        if have.get("identity") != identity:
            diff = sorted(k for k in set(have.get("identity", {})) | set(identity)
                          if have.get("identity", {}).get(k) != identity.get(k))
            raise Stop(EXIT_REFUSED, f"{work_dir} holds outputs of another recipe (stamp keys "
                                     f"differ: {', '.join(diff)}); use a new --work-dir")
        return
    others = [q for q in work_dir.iterdir() if not q.name.startswith(".")]
    if others:
        raise Stop(EXIT_REFUSED, f"{work_dir} holds {len(others)} file(s) and no stamp: "
                                 "not resumable safely; use a new --work-dir")
    atomic_json(p, {"identity": identity, "utc": utc_now()})


def write_failures(path: Path, rows: list[dict]) -> None:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["item", "stage", "reason"])
    for r in sorted(rows, key=lambda r: (str(r.get("item")), str(r.get("stage")))):
        w.writerow([r.get("item"), r.get("stage"), str(r.get("reason", ""))[:300]])
    atomic_bytes(path, buf.getvalue().encode())


def read_failures(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


class Progress:
    """Progress on the VM bus (tools/vm/bus.py) when U2_BUS is set; plain lines otherwise.
    step = items finished so far (done + skipped + failed); x_ extras carry the split."""

    def __init__(self, total: int, tag: str = "P1_PROGRESS"):
        self.total, self.tag = int(total), tag
        self.bus = None
        self.last = 0.0
        if os.environ.get("U2_BUS"):
            sys.path.insert(0, str(REPO / "tools" / "vm"))
            import bus  # noqa: PLC0415  (tools/vm/bus.py; job.py also puts it on PYTHONPATH)

            self.bus = bus

    def resolved(self, cfg: dict) -> None:
        """resolved_config.json for launch.sh (written once the inputs are checked)."""
        if self.bus is not None:
            self.bus.resolved_config(cfg)
        else:
            say("P1_RESOLVED", json.dumps(cfg, sort_keys=True, default=str)[:2000])

    def tick(self, done: int, failed: int = 0, force: bool = False, **extra: object) -> None:
        now = time.time()
        if not force and now - self.last < 20:
            return
        self.last = now
        fields = {"x_items_total": self.total, "x_items_failed": int(failed)}
        fields.update({f"x_{k}": v for k, v in extra.items()})
        if self.bus is not None:
            self.bus.log(int(done), **fields)
        say(self.tag, f"done={done}/{self.total} failed={failed}")
