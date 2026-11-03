"""Append-only, hash-chained JSONL ledgers under experiments/, and the only code that writes them.

What it does:

    from ledger import Ledger
    L = Ledger()                                  # repo root, or $U2_REPO_ROOT
    L.ensure_files()                              # create missing empty ledgers (note.py init)
    row = L.append("evidence", {...})             # adds v, id, utc, prev, sha under one flock
    rows = L.read("evidence")                     # refuses a file whose chain is broken
    problems = L.verify(git_ref="HEAD")           # [] when every ledger is intact
    autocommit([paths], "ledger: ...")            # pathspec commit of exactly those paths

    python tools/lib/ledger.py verify             # exit 0 intact, 4 broken (tick treats 4 as STOP)

Errors: LedgerBroken (a chain or line is not what this module wrote: a STOP, exit 4 in note.py),
RowRefused (one row broke a rule, nothing written: exit 1), and the base LedgerError for a lock
timeout. Every writer goes through this module: tools/note.py (the CLI) and, from Tier 1,
tools/prereg/gate.py, tools/prereg/unseal.py and tools/eval/score_cached.py (the hook's B8 list).

Every row carries v (schema version), id (per-ledger sequence, e.g. EV-0007), utc (read from the
clock here, never supplied by the caller), prev (sha of the previous line in the same file; the
first line uses 64 zeros) and sha = sha256 of the row's canonical JSON without `sha`. A line is
stored in exactly its canonical form (sorted keys, no spaces, no NaN), so verify() can also refuse
a line that was re-serialised by something other than this module.

Why it is like this. Measured on copies of topaneu's ledgers on 2026-09-24 (blueprint key a2):
- "Append-only" was a convention. registry.csv and insights.jsonl were rewritten whole on every
  write, with no lock, so two writers (autorun, the agent, a second session) could lose rows, and
  a `sed -i` through Bash left no trace. Here one fcntl.flock on experiments/.lock serialises every
  read-check-append, and the hash chain makes an edit, insertion, deletion or reorder visible at
  the exact line. macOS lacks the flock(1) command, not the flock(2) syscall, so this works here.
- A chain alone cannot see a truncated tail (drop the last k lines and the rest still links), so
  verify(git_ref="HEAD") also requires the committed copy of each ledger to be a byte prefix of
  the working file. That is why every write is followed by autocommit().
- The same instant was stored as naive local time in evidence.ts ('2026-09-24T15:51:07') and as
  UTC in the idea log ('2026-09-24T08:51:07+00:00'): the class of error behind EXP-003's '~22:30'
  when the CLI history said 22:14. Rows get `utc` from the clock with an explicit +00:00, and any
  caller field that is a timestamp by name must carry an offset or the append is refused.
- evidence `kind` was free text: 67 kinds, 56 of them used once, so nothing could be queried by
  kind. EVIDENCE_KINDS below is the closed enum; detail goes in a free `subkind`.
- The project uses ONE Google account (AGENTS.md, Colab FAQ note). The earlier two-lane design
  put a `lane` label on runs and results; a stale writer that still does so is refused here
  instead of silently reintroducing the concept. Rows record `session` (the Colab session
  name) and packing instead.
- Readers take the same lock as writers: a reader that races an append can see a line without
  its trailing newline and would report a torn ledger that is not torn.

Generated views (registry.csv, RESULTS.md, ...) are written with atomic_write_text(): a tmp file
plus os.replace, so a crash never leaves half a view behind.

autocommit() uses `git commit -- <paths>` (git's --only mode), so whatever else the user has staged
stays staged and out of the commit. It is skipped, with a warning, during a rebase or merge, on a
detached HEAD, outside a git work tree, or when U2_NO_AUTOCOMMIT=1 (tests).
"""
from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
GENESIS = "0" * 64
RESERVED = frozenset({"v", "id", "utc", "prev", "sha"})

# name -> (file, id prefix, zero-pad width). INS keeps topaneu's three digits so INS-NNN ids read
# the same in both repos; the rest use four because topaneu's evidence reached 103 rows in a month.
KINDS: dict[str, tuple[str, str, int]] = {
    "registry": ("registry.jsonl", "REG", 4),
    "runs": ("runs.jsonl", "RUNEV", 4),
    "results": ("results.jsonl", "RES", 4),
    "evidence": ("evidence.jsonl", "EV", 4),
    "insights": ("insights.jsonl", "INS", 3),
    "ideas": ("ideas.jsonl", "IDEAEV", 4),
    "gates": ("gates.jsonl", "GATEEV", 4),
    "unseal": ("unseal.jsonl", "UNS", 4),
}

# Closed enum (blueprint tools/lib/ledger.py). Free detail goes in `subkind`.
EVIDENCE_KINDS = (
    "measurement", "kill-check", "liveness", "reproduction", "env", "cost", "timing", "launch",
    "data-integrity", "prediction-commit", "unseal", "gate", "literature", "review", "decision",
    "ops-incident",
)
# One vocabulary for how a number is known: topaneu's measured/sourced/inferred merged with the
# month plan's [V]/[T]/[U]/[E] tags (AGENTS.md hard rule 6).
STANDARDS = ("measured", "sourced-V", "sourced-T", "unverified-U", "estimated-E")

LOCK_TIMEOUT_S = float(os.environ.get("U2_LEDGER_LOCK_TIMEOUT", "60"))


class LedgerError(RuntimeError):
    """Base class: a ledger is unreadable or broken, a row was refused, or the lock timed out."""


class LedgerBroken(LedgerError):
    """A chain, a line or a committed prefix is not what ledger.py wrote. A STOP (exit 4): a
    decision taken on a tampered ledger is worse than no decision."""


class RowRefused(LedgerError):
    """One row broke a ledger rule (naive timestamp, 'lane', free-text kind, ...). Nothing was
    written. A refusal (exit 1), not a STOP: fix the row and retry."""


# --------------------------------------------------------------------------- time
def utc_now() -> str:
    """Aware ISO 8601 UTC, seconds precision, read from the clock: '2026-09-24T08:51:07+00:00'."""
    # timezone.utc, not datetime.UTC (3.11+): house style keeps tools importable on 3.10.
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")  # noqa: UP017


def parse_ts(value: str) -> dt.datetime:
    """Parse an ISO 8601 timestamp and refuse it unless it carries an explicit offset."""
    if not isinstance(value, str) or not value.strip():
        raise RowRefused(f"timestamp must be an ISO 8601 string, got {value!r}")
    text = value.strip()
    if text.endswith(("Z", "z")):  # fromisoformat() before 3.11 rejects the Z suffix
        text = text[:-1] + "+00:00"
    try:
        t = dt.datetime.fromisoformat(text)
    except ValueError as e:
        raise RowRefused(f"timestamp {value!r} is not ISO 8601: {e}") from None
    if t.tzinfo is None or t.utcoffset() is None:
        raise RowRefused(
            f"naive timestamp {value!r} refused: write the offset (+00:00 for UTC, +07:00 for "
            "this Mac). topaneu stored the same instant as naive local time and as UTC.")
    return t


def check_ts(value: str) -> str:
    """Return `value` unchanged if it is an aware ISO 8601 timestamp; raise LedgerError if not."""
    parse_ts(value)
    return value


def _is_ts_key(key: str) -> bool:
    return key in ("utc", "ts", "at") or key.endswith(("_utc", "_at", "_ts"))


def _check_timestamps(obj: Any, path: str = "") -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            where = f"{path}.{k}" if path else str(k)
            if isinstance(k, str) and _is_ts_key(k) and v is not None:
                try:
                    check_ts(v)
                except LedgerError as e:
                    raise RowRefused(f"field {where}: {e}") from None
            else:
                _check_timestamps(v, where)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _check_timestamps(v, f"{path}[{i}]")


def _lane_path(obj: Any, path: str = "") -> str | None:
    """Where a 'lane' key sits anywhere in a row (a nested data.lane is refused too)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            where = f"{path}.{k}" if path else str(k)
            if k == "lane":
                return where
            hit = _lane_path(v, where)
            if hit:
                return hit
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hit = _lane_path(v, f"{path}[{i}]")
            if hit:
                return hit
    return None


# ------------------------------------------------------------------ canonical form
def canonical(obj: Any) -> str:
    """The one serialisation that is hashed and stored. NaN and Infinity are refused."""
    try:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                          allow_nan=False)
    except ValueError as e:  # allow_nan=False raises ValueError for NaN/inf
        raise RowRefused(f"row is not strict JSON ({e}); NaN and Infinity cannot be stored") \
            from None
    except TypeError as e:
        raise RowRefused(f"row is not JSON-serialisable: {e}") from None


def row_sha(row: dict) -> str:
    body = {k: v for k, v in row.items() if k != "sha"}
    return hashlib.sha256(canonical(body).encode("utf-8")).hexdigest()


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def default_root() -> Path:
    env = os.environ.get("U2_REPO_ROOT")
    return Path(env).resolve() if env else Path(__file__).resolve().parents[2]


def atomic_write_text(path: Path | str, text: str) -> bool:
    """Write a generated file via tmp + os.replace. Returns False when the content is unchanged
    (the file is not touched, so its mtime and git status stay quiet)."""
    p = Path(path)
    if p.exists() and p.read_text(encoding="utf-8") == text:
        return False
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.tmp-{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, p)
    return True


# --------------------------------------------------------------------------- ledger
class Ledger:
    """All ledgers of one repo. One lock file guards every ledger, so cross-file checks
    (an id free in the registry AND in configs/exp) and the append that follows are atomic."""

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root).resolve() if root else default_root()
        self.dir = self.root / "experiments"
        self.lock_path = self.dir / ".lock"
        self._depth = 0
        self._fd: int | None = None

    def path(self, kind: str) -> Path:
        if kind not in KINDS:
            raise RowRefused(f"unknown ledger {kind!r}; known: {', '.join(KINDS)}")
        return self.dir / KINDS[kind][0]

    # ---------------------------------------------------------------- locking
    @contextlib.contextmanager
    def locked(self, timeout: float | None = None) -> Iterator[None]:
        """Exclusive flock on experiments/.lock. Re-entrant inside one Ledger object."""
        if self._depth:
            self._depth += 1
            try:
                yield
            finally:
                self._depth -= 1
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        limit = LOCK_TIMEOUT_S if timeout is None else timeout
        deadline = time.monotonic() + limit
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    os.close(fd)
                    raise LedgerError(
                        f"could not lock {self.lock_path} within {limit:.0f} s; another note.py, "
                        "tick or gate.py is writing. Retry; never delete the lock file.") from None
                time.sleep(0.05)
        self._fd, self._depth = fd, 1
        try:
            yield
        finally:
            self._depth = 0
            self._fd = None
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    # ---------------------------------------------------------------- reading
    def _scan(self, kind: str) -> tuple[list[dict], list[str]]:
        """Parse one ledger and check every link. Returns (rows, problems)."""
        p = self.path(kind)
        if not p.exists():
            return [], []
        raw = p.read_bytes()
        problems: list[str] = []
        rows: list[dict] = []
        if raw and not raw.endswith(b"\n"):
            problems.append(f"{p.name}: last line is torn (no trailing newline)")
        prev: str | None = GENESIS
        seen: set[str] = set()
        lines = raw.decode("utf-8", errors="replace").split("\n")
        for n, line in enumerate(lines, start=1):
            where = f"{p.name}:{n}"
            if line == "":
                if n != len(lines):  # the only legitimate empty piece is after the final newline
                    problems.append(f"{where}: blank line (written outside ledger.py)")
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                problems.append(f"{where}: not JSON ({e.msg})")
                prev = None  # the next row cannot be linked to garbage; its own checks still run
                continue
            if not isinstance(row, dict):
                problems.append(f"{where}: not a JSON object")
                continue
            try:
                if canonical(row) != line:
                    problems.append(f"{where}: not in canonical form (rewritten outside ledger.py)")
            except LedgerError as e:
                problems.append(f"{where}: {e}")
            if row.get("v") != SCHEMA_VERSION:
                problems.append(f"{where}: schema version {row.get('v')!r} != {SCHEMA_VERSION}")
            if row.get("sha") != row_sha(row):
                problems.append(f"{where}: sha does not match the row (row {row.get('id')} edited)")
            if prev is not None and row.get("prev") != prev:
                problems.append(f"{where}: prev does not match the previous line's sha "
                                "(a row was inserted, deleted or reordered above this one)")
            rid = row.get("id")
            if not isinstance(rid, str) or not rid:
                problems.append(f"{where}: missing id")
            elif rid in seen:
                problems.append(f"{where}: duplicate id {rid}")
            else:
                seen.add(rid)
            try:
                parse_ts(row.get("utc"))  # type: ignore[arg-type]
            except LedgerError as e:
                problems.append(f"{where}: utc {e}")
            prev = row.get("sha")
            rows.append(row)
        return rows, problems

    def read(self, kind: str) -> list[dict]:
        """Every row of one ledger, oldest first. Refuses (LedgerError) if the chain is broken:
        a decision taken on a tampered ledger is worse than no decision."""
        with self.locked():
            rows, problems = self._scan(kind)
        if problems:
            raise LedgerBroken("ledger broken: " + "; ".join(problems[:5])
                              + ("" if len(problems) <= 5 else f"; +{len(problems) - 5} more")
                              + ". Run `python tools/note.py verify` and STOP.")
        return rows

    def verify(self, kinds: Iterable[str] | None = None, git_ref: str | None = None) -> list[str]:
        """Walk every ledger; return every problem found ([] means intact).

        With git_ref, the committed copy of each ledger must also be a byte prefix of the
        working file, which is the only way to see a truncated tail."""
        problems: list[str] = []
        with self.locked():
            for kind in kinds or KINDS:
                problems += self._scan(kind)[1]
                if git_ref:
                    problems += self._git_prefix_problems(kind, git_ref)
        return problems

    def count(self, kind: str) -> int:
        with self.locked():
            return len(self._scan(kind)[0])

    def ensure_files(self, kinds: Iterable[str] | None = None) -> list[Path]:
        """Create each missing ledger as an empty file (mode 0644) under the lock; return the
        ones created. Idempotent: an existing ledger is never touched, so this cannot reset a
        chain. `note.py init` calls it; nothing may create experiments/*.jsonl any other way
        (the PreToolUse hook refuses shell and Write-tool writes there, rule B8)."""
        made: list[Path] = []
        with self.locked():
            for kind in kinds or KINDS:
                p = self.path(kind)
                try:
                    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
                except FileExistsError:
                    continue
                os.close(fd)
                made.append(p)
        return made

    def _git_prefix_problems(self, kind: str, git_ref: str) -> list[str]:
        p = self.path(kind)
        rel = p.relative_to(self.root).as_posix()
        try:
            r = subprocess.run(["git", "-C", str(self.root), "show", f"{git_ref}:{rel}"],
                               capture_output=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            return []
        if r.returncode != 0:  # not a repo, or the file is not in that commit yet
            return []
        committed = r.stdout
        working = p.read_bytes() if p.exists() else b""
        if not working.startswith(committed):
            return [f"{p.name}: the copy committed at {git_ref} is not a prefix of the working "
                    "file (rows removed or rewritten since the last commit)"]
        return []

    # ---------------------------------------------------------------- writing
    def _prepare(self, kind: str, row: dict) -> dict:
        if not isinstance(row, dict):
            raise RowRefused(f"a {kind} row must be a JSON object")
        clash = RESERVED & set(row)
        if clash:
            raise RowRefused(f"fields {sorted(clash)} are set by ledger.py, not by the caller "
                              "(utc comes from the clock; id, prev and sha from the chain)")
        lane = _lane_path(row)
        if lane:
            raise RowRefused(f"field '{lane}' refused: this project uses one Google account and "
                              "has no lanes. Record the Colab session name in 'session' (and "
                              "packing in 'packed_with'/'packing') instead")
        _check_timestamps(row)
        body = json.loads(canonical(row))  # normalises tuples and int keys, refuses NaN
        if kind == "evidence":
            if body.get("kind") not in EVIDENCE_KINDS:
                raise RowRefused(f"evidence kind {body.get('kind')!r} is not one of "
                                  f"{', '.join(EVIDENCE_KINDS)}; put detail in subkind")
            if not str(body.get("source") or "").strip():
                raise RowRefused("evidence without a source cannot be replayed")
        std = body.get("standard")
        if std is not None and std not in STANDARDS:
            raise RowRefused(f"standard {std!r} is not one of {', '.join(STANDARDS)}")
        return body

    def append(self, kind: str, row: dict, *, row_id: str | None = None) -> dict:
        """Append one row and return it as stored."""
        return self.append_many(kind, [row], row_ids=[row_id] if row_id else None)[0]

    def append_many(self, kind: str, rows: list[dict],
                    row_ids: list[str | None] | None = None) -> list[dict]:
        """Append several rows under one lock and one chain check (tools/eval writes hundreds)."""
        bodies = [self._prepare(kind, r) for r in rows]
        if not bodies:
            return []
        _, prefix, width = KINDS[kind]
        p = self.path(kind)
        with self.locked():
            existing = self.read(kind)  # refuses to extend a broken or torn chain
            ids = {r["id"] for r in existing}
            prev = existing[-1]["sha"] if existing else GENESIS
            seq = len(existing)
            out: list[dict] = []
            for i, body in enumerate(bodies):
                seq += 1
                rid = (row_ids[i] if row_ids else None) or f"{prefix}-{seq:0{width}d}"
                if rid in ids:
                    raise RowRefused(f"id {rid} already exists in {p.name}")
                ids.add(rid)
                row = {**body, "v": SCHEMA_VERSION, "id": rid, "utc": utc_now(), "prev": prev}
                row["sha"] = row_sha(row)
                prev = row["sha"]
                out.append(row)
            data = "".join(canonical(r) + "\n" for r in out).encode("utf-8")
            with open(p, "ab") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
        return out


# ------------------------------------------------------------------------ git
def _warn(msg: str) -> None:
    print(f"ledger: {msg}", file=sys.stderr)


def autocommit(paths: Iterable[Path | str], msg: str, root: Path | str | None = None,
               retries: int = 8) -> str:
    """Commit exactly `paths` (pathspec / --only), leaving anything else staged untouched.

    Returns 'committed', 'nothing', or 'skipped: <why>' / 'failed: <why>' (also printed). A
    failed commit never undoes the ledger write; `make ledger-verify` and the next write's commit
    pick the row up."""
    if os.environ.get("U2_NO_AUTOCOMMIT") == "1":
        return "skipped: U2_NO_AUTOCOMMIT=1"
    base = Path(root).resolve() if root else default_root()

    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-C", str(base), *args], capture_output=True, text=True,
                              timeout=120)

    try:
        top = git("rev-parse", "--show-toplevel")
    except (OSError, subprocess.TimeoutExpired) as e:
        _warn(f"autocommit skipped: git unavailable ({e})")
        return "skipped: git unavailable"
    if top.returncode != 0 or Path(top.stdout.strip()).resolve() != base:
        # An enclosing repo (e.g. a scratch root inside another checkout) is not ours to commit.
        return "skipped: not the top of a git work tree"
    gd = Path(git("rev-parse", "--git-dir").stdout.strip())
    gitdir = gd if gd.is_absolute() else base / gd
    for marker, why in (("rebase-merge", "a rebase is in progress"),
                        ("rebase-apply", "a rebase is in progress"),
                        ("MERGE_HEAD", "a merge is in progress")):
        if (gitdir / marker).exists():
            _warn(f"autocommit skipped: {why}; commit the ledgers after it finishes")
            return f"skipped: {why}"
    if git("symbolic-ref", "-q", "HEAD").returncode != 0:
        _warn("autocommit skipped: detached HEAD; check out a branch and re-run a note.py write")
        return "skipped: detached HEAD"
    rel = []
    for p in paths:
        pp = Path(p)
        pp = pp if pp.is_absolute() else base / pp
        rel.append(pp.resolve().relative_to(base).as_posix())
    rel = sorted(set(rel))
    if not rel:
        return "nothing"
    delay = 0.2
    last = ""
    for _ in range(retries):
        a = git("add", "--", *rel)
        if a.returncode == 0:
            c = git("commit", "-q", "-m", msg, "--", *rel)
            if c.returncode == 0:
                return "committed"
            text = c.stdout + c.stderr
            if "nothing to commit" in text or "no changes added" in text:
                return "nothing"
            last = text.strip()
        else:
            last = a.stderr.strip()
        if "index.lock" not in last:
            break
        time.sleep(delay)  # another git process holds the index; back off and retry
        delay = min(delay * 2, 2.0)
    _warn(f"autocommit failed ({last[:300]}); the row is written, commit it by hand")
    return f"failed: {last[:300]}"


# ------------------------------------------------------------------------ CLI
def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args[:1] != ["verify"]:
        print("usage: python tools/lib/ledger.py verify", file=sys.stderr)
        return 2
    led = Ledger()
    in_git = subprocess.run(["git", "-C", str(led.root), "rev-parse", "--verify", "-q", "HEAD"],
                            capture_output=True).returncode == 0
    problems = led.verify(git_ref="HEAD" if in_git else None)
    if problems:
        for p in problems:
            print(f"BROKEN {p}")
        return 4
    n = sum(led.count(k) for k in KINDS)
    print(f"ok: {n} rows across {len(KINDS)} ledgers, chains intact"
          + ("" if in_git else " (no git HEAD, tail truncation not checked)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
