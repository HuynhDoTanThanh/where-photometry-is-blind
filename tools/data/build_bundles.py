#!/usr/bin/env python3
"""Build the deterministic data bundles, and hold the one zip writer every data tool uses.

What it does
- Streams members out of source zips into role bundles, following the `select` rules in
  data/manifest/bundles.v1.json. Nothing is extracted to disk first (no 35 GB of temporary
  files). Stored and deflated source members both work (MPI-Sintel-complete.zip deflates 8,391
  of its 8,753 entries [M central directory, 2026-09-24]); the bundle always stores them.
- A select entry takes `src` and `prefix`, and at least one of:
  - `dirs`: directory prefixes (a '/' is appended);
  - `files`: exact member names, each of which must exist in the source;
  - `match`: a regular expression (re.search on the source name), which also narrows `dirs`
    and `files`.
  Optional keys: `exclude` (regexes that drop members), `strip` (a leading path removed before
  `prefix` is added, e.g. 'HD1K/'), and the KITTI-only `ids` and `frames` filters. Unknown keys,
  and a select that matches nothing (unless `allow_empty`), are refused.
- A source of kind `partial` (range-extracted by tools/data/range_extract.py: FlowFactor, the KITTI
  raw drives) is read from its `local_dir`, member by member, each checked against the size and
  CRC-32 in its range manifest (`local_manifest`), whose sha256 must be the one download_sources.sh
  recorded in src/SOURCES.json (`manifest_sha256`). A bundle with a `convert` step (the KITTI raw
  JPEG bundles) is built by tools/data/build_kraw.py instead.
- Each zip ends with its licence member(s) (LICENSE-KITTI.txt, or the `file` of each key in
  `licence`, which may be a list), FILES.sha256 and BUNDLE.json at the zip root, in that order.
  tools/data/fetch.py moves those to DATA_ROOT/.bundles/<name>/ when it unpacks; payload paths are
  relative to DATA_ROOT.
- Uploadable bundles go to DATA_ROOT/bundles/. HOLD bundles (not sealed, and `hold: true`,
  `uploadable: false`, wave 'hold' or an `origin`: Sintel, HD1K, Sintel raw, Spring, FlowFactor,
  whose licences are unclear or not yet approved for hosting) go to DATA_ROOT/hold/ (their
  `build_dir` must agree), are built only with --only or --hold, and never appear in an upload
  list.
- After a build it records bytes, sha256, n_files and the provenance in the manifest (merged
  under the manifest lock, update_manifest), and writes the per-file list to
  data/manifest/files/<name>.sha256.
- A built name is FROZEN (install_build): every build is written to a staging file first, and
  it replaces DATA_ROOT/<dir>/<file> only when nothing records the name yet, or when its bytes
  equal what is recorded (the manifest sha256, links.v1.json, files/<name>.sha256). Otherwise it
  is exit 1, 'NAME_FROZEN ... bump --version'. --replace-unregistered allows a different build
  only for a name that links.v1.json does not hold and that has no hardlink (st_nlink 1). A file
  with st_nlink > 1 (linked into upload/) is never replaced.
- On a VM, fetch.py calls rebuild_on_vm() when Drive answers 'Quota exceeded' for an IMAGE
  bundle. That call reads the KITTI S3 zip over HTTP range requests, and the result must be
  byte-identical to the Mac build, which is checked by sha256.

How to invoke (on the Mac, with the repo .venv)
  python tools/data/build_bundles.py --data-root /Volumes/Shin/U2FLOW_data --plan
  python tools/data/build_bundles.py --data-root /Volumes/Shin/U2FLOW_data --wave 1 --extract-gt
  python tools/data/build_bundles.py --data-root /Volumes/Shin/U2FLOW_data --only u2f-meta-v1
  python tools/data/build_bundles.py --data-root /Volumes/Shin/U2FLOW_data --hold
  python tools/data/build_bundles.py --data-root /Volumes/Shin/U2FLOW_data --upload-list --wave 1
  (--upload-list only lists the built files; the upload folder is assemble_upload.py's.)
Exit codes:
  0   ok
  1   a build or verification failed; a frozen name would change (NAME_FROZEN); or a HOLD
      refusal: a HOLD bundle aimed at bundles/ or upload/, a HOLD file found there, or a HOLD
      bundle in an upload list
  2   a precondition is missing: a source is absent or unverified (download_sources.sh), or the
      split is absent or uncommitted (tools/prereg/make_split.py)
  3   upload list incomplete: some bundles are not built yet
  50  seal refusal (including a sealed bundle's file found in bundles/ or upload/)
  64  usage

Why it is written this way
- Python's zipfile refuses to go past 2 GiB - 1 without ZIP64 (ZIP64_LIMIT = (1 << 31) - 1 and
  _writecheck raises LargeZipFile; checked on CPython 3.12.7). The bundles are 2.5-2.8 GB and
  must stay plain 32-bit zips (cap 3.9 GiB, no ZIP64). So this file writes the stored format
  itself: local header, data, central directory and end record, with no extra fields and no data
  descriptors. The size is known exactly before the first byte is written:
      22 + sum(76 + 2 * len(name) + size).
- Determinism: sorted names; DOS date 1980-01-01 00:00; Unix mode 0644; no timestamp or git sha
  inside the zip; metadata last in a fixed order. The same inputs give the same sha256 on the
  Mac and on a VM, which is what makes the S3 rebuild checkable.
- The held-out seal (AGENTS.md hard rule 4): a select on a source marked `heldout_guard` must
  carry an ids filter, and no held-out id may enter a non-sealed bundle. Building the sealed
  bundle and moving data_scene_flow.zip under sealed/ belong to tools/prereg/seal.py (prereg,
  Tier 1), the only program the hook lets touch the seal. This script refuses both (exit 50).
- On a VM, GT-bearing bundles and GT-bearing sources are never rebuilt (exit 50).
- HOLD bundles live outside bundles/ and upload/ because 'upload bundles/' is the easiest wrong
  drag, and register_links used to ignore unknown names (data inventory critique, 2026-09-24).
  The writer refuses any non-uploadable bundle whose output path has a bundles/ or upload/
  component, and the build and the upload list refuse to run while a HOLD or sealed file sits
  there.
- Range reads check status 206, the Content-Range span and the exact byte count, and stream the
  body: a server that ignores Range answers 200 with the whole file (SURFdrive does this for
  'bytes=0-...' [M data inventory, 2026-09-24]), and reading that body into memory first would
  pull 29.8 GB. DaRUS answers HEAD with 403 [M], so the size falls back to a 1-byte range read
  from byte 1.
- The frozen name (review 2026-09-25, blocker): a rebuild used to os.replace() a registered or
  hardlinked file with new bytes and record the new sha256, so Drive kept serving the old bytes
  under the name (every VM fetch then fails verify, exit 30) and upload/ kept the old inode
  (assemble_upload: a false seal STOP). A name never changes once built: bump --version.
- The manifest lock: build_bundles, build_kraw, pack_derived register, register_links,
  check_links and download_sources.sh each used to load a whole JSON file, work for minutes and
  save it back, so the last writer erased the other's sha256 (a lost update; review
  2026-09-25). update_json re-reads the file under an exclusive lock, merges only its own
  fields and writes a per-writer temp file. The lock is flock(2) through fcntl on a fixed path,
  $U2_STATE/locks/data-<file>-<hash>.lock, as tools/lib/ledger.py does: the kernel drops it when
  a writer dies, so no stale lock can block a build (macOS lacks only the flock(1) command).
- repo_file selects are limited to splits/<name>.json: a manifest edit must not be able to
  embed a checkpoint (no LICENSE) or an audit file in a public bundle.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import fcntl
import hashlib
import io
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
import zipfile
import zlib
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, BinaryIO

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "data/manifest/bundles.v1.json"
DATA_ROOT_MAC = Path("/Volumes/Shin/U2FLOW_data")

CAP_BYTES = 4_187_593_113  # 3.9 GiB: bundles.v1.json limits.cap_bytes
MAX_ENTRIES = 65_534  # 0xFFFF in the end record is the ZIP64 marker, so stay one below it
ZIP32_MAX = 0xFFFF_FFFE  # 0xFFFFFFFF in a size or offset field is the ZIP64 marker
DOS_DATE = (0 << 9) | (1 << 5) | 1  # 1980-01-01
DOS_TIME = 0
ZIP_VERSION = 20
MADE_BY = (3 << 8) | ZIP_VERSION  # 3 = Unix, so readers apply external_attr as a mode
EXT_ATTR = 0o100644 << 16  # regular file, rw-r--r--
LOCAL_FMT = "<IHHHHHIIIHH"  # 30 bytes
CENTRAL_FMT = "<IHHHHHHIIIHHHHHII"  # 46 bytes
EOCD_FMT = "<IHHHHIIH"  # 22 bytes
LOCAL_SIG, CENTRAL_SIG, EOCD_SIG = 0x04034B50, 0x02014B50, 0x06054B50
META_NAMES = ("FILES.sha256", "BUNDLE.json")
CHUNK = 1 << 20
ZIP_MAGIC = b"PK\x03\x04"
PUBLIC_DIRS = ("bundles", "upload")  # DATA_ROOT/bundles is the upload set; upload/ is its mirror
HOLD_DIR = "hold"  # DATA_ROOT/hold: built, sha-listed, pushed privately, never uploaded
LOCAL_METHODS = frozenset({zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED, zipfile.ZIP_BZIP2,
                           zipfile.ZIP_LZMA})  # what CPython's zipfile decodes
HTTP_METHODS = frozenset({zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED})  # decode_member below

_JUNK = re.compile(r"(^|/)(\._[^/]*|\.DS_Store|__MACOSX|\.AppleDouble|Thumbs\.db)(/|$)")
_ID_FRAME = re.compile(r"^(\d{6})(?:_(\d{2}))?\.[A-Za-z0-9]+$")
_FILES_LINE = re.compile(r"([0-9a-f]{64}) [ *](.+)")
_CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+|\*)")
_ZIP_SELECT = frozenset({"src", "prefix", "dirs", "files", "match", "exclude", "strip", "ids",
                         "frames", "allow_empty", "note"})
_REPO_SELECT = frozenset({"repo_file", "dest", "note"})
_REPO_FILE = re.compile(r"^splits/[\w.-]+\.json$")  # the only repo files a bundle may carry
STATE_DEFAULT = "/Users/thanhhdt/.u2flow-colab/state"
LOCK_TIMEOUT_S = 120.0


class BundleError(RuntimeError):
    """A build, verify or plan step failed. `code` is the process exit code it maps to."""

    code = 1


class Precondition(BundleError):
    code = 2


class VerifyError(BundleError):
    code = 1


class SealRefusal(BundleError):
    code = 50


class HoldRefusal(BundleError):
    """A HOLD (not uploadable) bundle aimed at, or found in, bundles/ or upload/."""

    code = 1


class RangeError(BundleError):
    """An HTTP range read failed: wrong status, span or length, or retries exhausted."""

    code = 1


class NameFrozen(BundleError):
    """A built, registered or hardlinked bundle name would get other bytes: bump --version."""

    code = 1


class ArgParser(argparse.ArgumentParser):
    """argparse exits 2 on a usage error, but 2 means 'precondition' in these tools and 64 means
    usage (the data-bundles skill's exit table; Makefile header). Every tools/data CLI uses it."""

    def error(self, message: str) -> None:  # type: ignore[override]
        self.print_usage(sys.stderr)
        self.exit(64, f"{self.prog}: error: {message}\n")


# --------------------------------------------------------------------------- small helpers
def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def tmp_name(path: Path, suffix: str = ".tmp") -> Path:
    """A per-writer temp name beside `path`: two processes (or threads) never share one."""
    path = Path(path)
    return path.with_name(f"{path.name}.{os.getpid()}-{threading.get_ident()}{suffix}")


def save_json(path: Path, obj: Any, *, sort_keys: bool = False) -> None:
    """Atomic rewrite (per-writer tmp + rename), 2-space indent, trailing newline. A document
    that other processes also write goes through update_json instead."""
    path = Path(path)
    tmp = tmp_name(path)
    try:
        tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=sort_keys) + "\n")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def state_dir() -> Path:
    return Path(os.environ.get("U2_STATE", STATE_DEFAULT))


def lock_path(path: Path) -> Path:
    """The fixed lock file of one JSON document: $U2_STATE/locks/data-<name>-<hash>.lock."""
    real = os.path.realpath(path)
    key = hashlib.sha256(real.encode()).hexdigest()[:16]
    return state_dir() / "locks" / f"data-{Path(real).name}-{key}.lock"


@contextlib.contextmanager
def json_lock(path: Path, timeout: float = LOCK_TIMEOUT_S) -> Iterator[None]:
    """Exclusive flock(2) on the document's lock file; released by the kernel if we die."""
    lp = lock_path(path)
    lp.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(lp, os.O_RDWR | os.O_CREAT, 0o600)
    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise BundleError(f"could not lock {lp} within {timeout:.0f} s: another data "
                                      "tool is writing that manifest; re-run") from None
                time.sleep(0.05)
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def update_json(path: Path, fn: Callable[[Any], None], *,
                default: Callable[[], Any] | None = None, sort_keys: bool = False) -> Any:
    """Read-modify-write one JSON document under its lock: re-read it, apply fn(doc) in place,
    write a per-writer temp file and rename it. An exception in fn writes nothing."""
    path = Path(path)
    with json_lock(path):
        try:
            doc = load_json(path)
        except FileNotFoundError:
            if default is None:
                raise
            doc = default()
        fn(doc)
        save_json(path, doc, sort_keys=sort_keys)
    return doc


def update_manifest(path: Path, section: str, name: str, fields: dict) -> dict:
    """Merge `fields` into manifest[section][name] (created when absent) under the lock; the
    other entries keep whatever other writers saved meanwhile. Returns the merged entry."""
    out: dict = {}

    def fn(doc: dict) -> None:
        block = doc.setdefault(section, {})
        entry = block.setdefault(name, {})
        entry.update(fields)
        out.update(entry)

    update_json(path, fn)
    return out


def section_of(doc: dict, name: str) -> str:
    """The manifest section that holds a bundle spec: 'bundles', or a separate 'hold'."""
    return "hold" if name in (doc.get("hold") or {}) else "bundles"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def queue_evidence(kind: str, subkind: str, standard: str, source: str, summary: dict) -> None:
    """Append one row to $U2_STATE/pending_evidence.jsonl for tick/note.py to ingest (Tier 1).

    The key is `observed_utc`, not `utc`: tools/lib/ledger.py reserves `utc` for its own stamp.
    Best effort: a missing state dir must never fail a data build.
    """
    state = Path(os.environ.get("U2_STATE", "/Users/thanhhdt/.u2flow-colab/state"))
    row = {"kind": kind, "subkind": subkind, "standard": standard, "observed_utc": utc_now(),
           "source": source, "summary": summary}
    try:
        state.mkdir(parents=True, exist_ok=True, mode=0o700)
        with open(state / "pending_evidence.jsonl", "a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    except OSError as e:
        print(f"WARN evidence not queued ({e}): {json.dumps(row, sort_keys=True)}", file=sys.stderr)


def git_sha(repo: Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def is_committed(repo: Path, rel: str) -> bool:
    """True when `rel` is tracked and identical to HEAD (no staged or unstaged change)."""
    try:
        a = subprocess.run(["git", "-C", str(repo), "ls-files", "--error-unmatch", rel],
                           capture_output=True, timeout=20)
        b = subprocess.run(["git", "-C", str(repo), "diff", "--quiet", "HEAD", "--", rel],
                           capture_output=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return False
    return a.returncode == 0 and b.returncode == 0


def is_junk(name: str) -> bool:
    """AppleDouble, .DS_Store, __MACOSX: what Finder and the external APFS volume add."""
    return bool(_JUNK.search(name))


def check_name(name: str) -> str:
    """Refuse any member name that could escape DATA_ROOT or carry Finder junk."""
    if not name or len(name) > 1024 or "\\" in name or "\x00" in name or name.startswith("/"):
        raise BundleError(f"unsafe member name {name!r}")
    try:
        name.encode("ascii")
    except UnicodeEncodeError:
        raise BundleError(f"non-ASCII member name {name!r} (no UTF-8 flag is set)") from None
    if any(p in ("", ".", "..") for p in name.split("/")):
        raise BundleError(f"unsafe member name {name!r}")
    if is_junk(name):
        raise BundleError(f"Finder junk in member name {name!r}")
    return name


def id_frame(member: str) -> tuple[str | None, int | None]:
    """'training/image_2/000123_09.png' -> ('000123', 9); 'training/calib/000030.txt' -> id only."""
    m = _ID_FRAME.match(member.rsplit("/", 1)[-1])
    if not m:
        return None, None
    return m.group(1), (int(m.group(2)) if m.group(2) is not None else None)


def builder_of(spec: dict) -> str:
    """Which tool builds a bundle: build_kraw for a bundle with a `convert` step (the KITTI raw
    JPEG bundles), else build_bundles, unless `builder` names one."""
    if spec.get("builder"):
        return str(spec["builder"])
    conv = spec.get("convert") or any("convert" in s for s in spec.get("select") or [])
    return "build_kraw" if conv else "build_bundles"


def is_hold(spec: dict) -> bool:
    """A HOLD bundle: built on the Mac into DATA_ROOT/hold/ and never uploaded. The same test as
    fetch.is_hold (uploadable=false, wave 'hold' or an origin), plus an explicit `hold: true`."""
    return not spec.get("sealed") and (spec.get("uploadable") is False
                                       or str(spec.get("wave")).lower() == "hold"
                                       or bool(spec.get("origin")) or spec.get("hold") is True)


def is_public(spec: dict) -> bool:
    """May this bundle sit in bundles/ and be uploaded?"""
    return bool(spec.get("uploadable")) and not spec.get("sealed") and not is_hold(spec)


def all_specs(doc: dict) -> dict[str, dict]:
    """Every bundle spec: the `bundles` section plus a separate `hold` section, if any."""
    out = dict(doc.get("bundles") or {})
    for n, s in (doc.get("hold") or {}).items():
        if n in out:
            raise BundleError(f"{n} is in both the bundles and the hold sections")
        out[n] = s
    return out


def spec_of(doc: dict, name: str) -> dict:
    """One bundle's spec (the dict inside the manifest, so updates to it are saved)."""
    specs = all_specs(doc)
    if name not in specs:
        raise BundleError(f"unknown bundle {name}")
    return specs[name]


def bundle_out_path(data_root: Path, spec: dict) -> Path:
    """DATA_ROOT/bundles/<file> for public bundles, DATA_ROOT/hold/<file> for HOLD ones."""
    f = str(spec.get("file") or "")
    if not f or "/" in f or "\\" in f or f.startswith("."):
        raise BundleError(f"{spec.get('name')}: file {f!r} must be a plain file name")
    if spec.get("sealed"):
        raise SealRefusal(f"{spec.get('name')} is sealed: tools/prereg/seal.py writes it")
    d = "bundles" if is_public(spec) else HOLD_DIR
    if spec.get("build_dir") not in (None, d):
        raise BundleError(f"{spec.get('name')}: build_dir {spec['build_dir']!r} disagrees with its "
                          f"HOLD status (it goes to {d}/)")
    return Path(data_root) / d / f


def check_out_path(spec: dict, out_path: Path) -> None:
    """Refuse to write a bundle that is not public under a bundles/ or upload/ directory."""
    if is_public(spec):
        return
    p = Path(out_path)
    parts = set(Path(os.path.abspath(p)).parent.parts) | set(p.resolve().parent.parts)
    hit = sorted(parts & set(PUBLIC_DIRS))
    if hit:
        cls = SealRefusal if spec.get("sealed") else HoldRefusal
        raise cls(f"{spec.get('name')} is not uploadable: refusing to write it under {hit[0]}/ "
                  f"({out_path}); HOLD bundles go to DATA_ROOT/{HOLD_DIR}/")


def public_leaks(doc: dict, data_root: Path) -> tuple[list[Path], list[Path]]:
    """(sealed, hold) bundle files found anywhere under DATA_ROOT/bundles or DATA_ROOT/upload."""
    specs = all_specs(doc).values()
    sealed = {b["file"] for b in specs if b.get("sealed") and b.get("file")}
    hold = {b["file"] for b in specs if not b.get("sealed") and not is_public(b) and b.get("file")}
    found_s: list[Path] = []
    found_h: list[Path] = []
    for d in PUBLIC_DIRS:
        root = Path(data_root) / d
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*")):
            if p.name in sealed:
                found_s.append(p)
            elif p.name in hold:
                found_h.append(p)
    return found_s, found_h


def refuse_public_leaks(doc: dict, data_root: Path) -> None:
    sealed, hold = public_leaks(doc, data_root)
    if sealed:
        raise SealRefusal(f"sealed bundle file(s) in a public directory: {sealed[0]} (STOP: "
                          "report it; nothing sealed may sit in bundles/ or upload/)")
    if hold:
        shown = ", ".join(map(str, hold[:3]))
        raise HoldRefusal(f"HOLD bundle file(s) in a public directory: {shown}; move them to "
                          f"DATA_ROOT/{HOLD_DIR}/ (and delete any Drive copy)")


# --------------------------------------------------------------------------- the zip writer
@dataclasses.dataclass
class Member:
    """One file to store. `opener` returns a binary stream of exactly `size` bytes.

    `crc` is the CRC-32 when the source already knows it (a source zip's central directory);
    otherwise write_zip reads the member once more to compute it. `sha256`, when given, is the
    expected digest: a mismatch aborts the zip (the VM rebuild checks every file this way).
    `fetch` is set by remote sources and lets rebuild_on_vm prefetch in parallel.
    """

    name: str
    size: int
    opener: Callable[[], BinaryIO]
    crc: int | None = None
    sha256: str | None = None
    fetch: Callable[[], bytes] | None = None


def bytes_member(name: str, data: bytes) -> Member:
    return Member(name, len(data), lambda d=data: io.BytesIO(d), zlib.crc32(data) & 0xFFFFFFFF)


def file_member(name: str, path: Path) -> Member:
    return Member(name, Path(path).stat().st_size, lambda p=path: open(p, "rb"))


def zip_size(entries: Iterable[tuple[str, int]]) -> int:
    """Exact size of a stored zip with no extras: local 30 + central 46 per member, end 22."""
    return 22 + sum(76 + 2 * len(n.encode("ascii")) + s for n, s in entries)


def files_sha256_len(names: Iterable[str]) -> int:
    """Byte length of FILES.sha256 ('<64 hex>  <name>\\n' per line) before any hash is known."""
    return sum(64 + 2 + len(n.encode("ascii")) + 1 for n in names)


@dataclasses.dataclass
class _Written:
    name: str
    size: int
    crc: int
    sha256: str
    offset: int


class _Out:
    """Write-through file wrapper that hashes the whole zip and tracks the offset itself."""

    def __init__(self, f: BinaryIO) -> None:
        self.f = f
        self.h = hashlib.sha256()
        self.pos = 0

    def write(self, b: bytes) -> None:
        self.f.write(b)
        self.h.update(b)
        self.pos += len(b)


_SOURCE_READ_ERRORS = (zipfile.BadZipFile, zlib.error, EOFError, NotImplementedError)


def decode_member(raw: bytes, method: int, size: int, name: str) -> bytes:
    """A member's bytes from its compressed data: stored as is, or raw-deflate (zlib wbits -15)
    inflated to exactly `size` bytes. The caller checks the CRC-32."""
    if method == zipfile.ZIP_STORED:
        data = raw
    elif method == zipfile.ZIP_DEFLATED:
        d = zlib.decompressobj(-15)
        try:
            data = d.decompress(raw, size + 1)
        except zlib.error as e:
            raise VerifyError(f"{name}: inflate failed ({e})") from None
        if not d.eof:
            raise VerifyError(f"{name}: deflate stream truncated or longer than {size} bytes")
    else:
        raise BundleError(f"{name}: compression method {method} is not supported here")
    if len(data) != size:
        raise VerifyError(f"{name}: {len(data)} bytes after decoding, {size} expected")
    return data


def _prepass(m: Member) -> tuple[int, str]:
    crc, h, n = 0, hashlib.sha256(), 0
    try:
        with m.opener() as f:
            for chunk in iter(lambda: f.read(CHUNK), b""):
                crc = zlib.crc32(chunk, crc)
                h.update(chunk)
                n += len(chunk)
    except _SOURCE_READ_ERRORS as e:
        raise VerifyError(f"{m.name}: the source could not be read ({e})") from None
    if n != m.size:
        raise BundleError(f"{m.name}: {n} bytes on read, {m.size} expected (did it change?)")
    return crc & 0xFFFFFFFF, h.hexdigest()


def local_header(name: bytes, crc: int, size: int) -> bytes:
    return struct.pack(LOCAL_FMT, LOCAL_SIG, ZIP_VERSION, 0, 0, DOS_TIME, DOS_DATE, crc, size,
                       size, len(name), 0)


def central_header(name: bytes, crc: int, size: int, offset: int) -> bytes:
    if offset > ZIP32_MAX or size > ZIP32_MAX:
        raise BundleError(f"offset {offset} or size {size} needs ZIP64, which bundles never use")
    return struct.pack(CENTRAL_FMT, CENTRAL_SIG, MADE_BY, ZIP_VERSION, 0, 0, DOS_TIME, DOS_DATE,
                       crc, size, size, len(name), 0, 0, 0, 0, EXT_ATTR, offset)


def _write_member(out: _Out, m: Member, crc: int) -> _Written:
    nb = m.name.encode("ascii")
    offset = out.pos
    if offset > ZIP32_MAX or m.size > ZIP32_MAX:
        raise BundleError(f"{m.name}: offset or size past 4 GiB would need ZIP64")
    out.write(local_header(nb, crc, m.size))
    out.write(nb)
    c, h, n = 0, hashlib.sha256(), 0
    try:
        with m.opener() as f:
            for chunk in iter(lambda: f.read(CHUNK), b""):
                n += len(chunk)
                if n > m.size:
                    raise BundleError(f"{m.name}: more than the {m.size} bytes announced")
                c = zlib.crc32(chunk, c)
                h.update(chunk)
                out.write(chunk)
    except _SOURCE_READ_ERRORS as e:  # a corrupt deflated source member: zipfile's own CRC/zlib
        raise VerifyError(f"{m.name}: the source could not be read ({e})") from None
    c &= 0xFFFFFFFF
    if n != m.size:
        raise BundleError(f"{m.name}: {n} bytes read, {m.size} announced")
    if c != crc:
        raise VerifyError(f"{m.name}: CRC-32 {c:08x}, the source says {crc:08x}")
    sha = h.hexdigest()
    if m.sha256 and sha != m.sha256:
        raise VerifyError(f"{m.name}: sha256 {sha[:12]}.., expected {m.sha256[:12]}..")
    return _Written(m.name, m.size, c, sha, offset)


def plan_zip(payload: list[Member], licences: list[tuple[str, bytes]], bundle_json: bytes,
             *, cap: int = CAP_BYTES, max_entries: int = MAX_ENTRIES) -> int:
    """Validate names and return the exact zip size. Raises before anything is written."""
    names = [check_name(m.name) for m in payload]
    for n in names:
        if "/" not in n:
            raise BundleError(f"payload member {n!r} must sit under a directory (the zip root "
                              "holds only the licence and the two metadata files)")
    if len(set(names)) != len(names):
        dup = sorted({n for n in names if names.count(n) > 1})
        raise BundleError(f"duplicate member names: {dup[:5]}")
    for n, _ in licences:
        if "/" in check_name(n) or n in META_NAMES:
            raise BundleError(f"bad licence/notice member name {n!r}")
    covered = names + [n for n, _ in licences]
    entries = ([(m.name, m.size) for m in payload] + [(n, len(d)) for n, d in licences]
               + [("FILES.sha256", files_sha256_len(covered)), ("BUNDLE.json", len(bundle_json))])
    if len(entries) > max_entries:
        raise BundleError(f"{len(entries)} entries > {max_entries}: split the bundle")
    total = zip_size(entries)
    if total > cap:
        raise BundleError(f"{total} bytes > cap {cap}: split the bundle")
    return total


def write_zip(out_path: Path, payload: list[Member], *, licences: list[tuple[str, bytes]] = (),
              bundle_json: bytes, cap: int = CAP_BYTES, max_entries: int = MAX_ENTRIES) -> dict:
    """Write a deterministic stored zip: payload sorted by name, then licences, FILES.sha256
    (covering payload and licences) and BUNDLE.json. Writes a per-writer <out>.<pid>-<tid>.part
    and renames it. A target with a second hardlink (it is in upload/) is never replaced."""
    payload = sorted(payload, key=lambda m: m.name)
    licences = list(licences)
    total = plan_zip(payload, licences, bundle_json, cap=cap, max_entries=max_entries)
    out_path = Path(out_path)
    refuse_linked(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tmp_name(out_path, ".part")
    written: list[_Written] = []
    try:
        with open(tmp, "wb") as f:
            out = _Out(f)
            for m in payload + [bytes_member(n, d) for n, d in licences]:
                crc = m.crc
                if crc is None:
                    crc, pre = _prepass(m)
                    m = dataclasses.replace(m, sha256=m.sha256 or pre)
                written.append(_write_member(out, m, crc))
            files_txt = "".join(f"{w.sha256}  {w.name}\n" for w in written).encode("ascii")
            for n, d in (("FILES.sha256", files_txt), ("BUNDLE.json", bundle_json)):
                written.append(_write_member(out, bytes_member(n, d), zlib.crc32(d) & 0xFFFFFFFF))
            cd_off = out.pos
            for w in written:
                nb = w.name.encode("ascii")
                out.write(central_header(nb, w.crc, w.size, w.offset))
                out.write(nb)
            cd_size = out.pos - cd_off
            out.write(struct.pack(EOCD_FMT, EOCD_SIG, 0, 0, len(written), len(written), cd_size,
                                  cd_off, 0))
            f.flush()
            os.fsync(f.fileno())
        if out.pos != total:
            raise BundleError(f"internal: wrote {out.pos} bytes, planned {total}")
        refuse_linked(out_path)  # a link may have appeared while we wrote
        os.replace(tmp, out_path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    body = written[:-2]
    return {"path": str(out_path), "bytes": total, "sha256": out.h.hexdigest(),
            "n_entries": len(written), "n_files": len(body), "n_payload": len(payload),
            "payload_bytes": sum(m.size for m in payload),
            "files": [(w.name, w.size, w.sha256) for w in body]}


# --------------------------------------------------------------------------- the frozen name
def refuse_linked(path: Path) -> None:
    """Never replace a file that has a second hardlink: the other link is the copy in upload/
    (assemble_upload.py), and a rename would leave it holding the old bytes."""
    try:
        st = Path(path).stat()
    except FileNotFoundError:
        return
    if st.st_nlink > 1:
        raise NameFrozen(f"NAME_FROZEN {Path(path).name}: it has {st.st_nlink} hardlinks (it is "
                         "in upload/), so it is never replaced: bump --version")


def files_text(res: dict) -> str:
    """The per-file list of a write_zip result, as data/manifest/files/<name>.sha256 holds it."""
    return "".join(f"{sha}  {name}\n" for name, _, sha in res["files"])


def links_entry(links_path: Path | None, name: str) -> dict | None:
    """links.v1.json's record of a name (it was uploaded and registered), or None."""
    if links_path is None or not Path(links_path).is_file():
        return None
    try:
        return (load_json(links_path).get("links") or {}).get(name)
    except (OSError, ValueError):
        return {"id": "?", "note": "links.v1.json is unreadable: treated as registered"}


def staging_path(out: Path) -> Path:
    """Where a build is written before install_build decides: beside the final file, hidden."""
    out = Path(out)
    return out.with_name(f".{out.name}.{os.getpid()}.staged")


def install_build(name: str, staged: Path, out: Path, res: dict, *, spec: dict,
                  files_path: Path, links_path: Path | None,
                  replace_unregistered: bool = False) -> str:
    """Move a staged build to its final name, unless the name is frozen and the bytes differ.

    Frozen: the manifest records a sha256, links.v1.json holds the name, files/<name>.sha256
    exists, or the final file has a second hardlink. Returns 'installed' or 'unchanged' (the
    final file already holds these bytes); raises NameFrozen and deletes the staged file when
    the bytes would change. --replace-unregistered lets a different build replace a name that
    is recorded only on this Mac (not in links.v1.json, no hardlink)."""
    staged, out = Path(staged), Path(out)
    new_sha, new_list = res["sha256"], files_text(res)
    link = links_entry(links_path, name)
    recorded = {s for s in (spec.get("sha256"), (link or {}).get("sha256")) if s}
    old_list = Path(files_path).read_text() if Path(files_path).is_file() else None
    nlink = out.stat().st_nlink if out.exists() else 0
    same_out = out.is_file() and sha256_file(out) == new_sha
    why = []
    if recorded - {new_sha}:
        why.append(f"recorded sha256 {sorted(recorded - {new_sha})[0][:16]}..")
    if old_list is not None and old_list != new_list:
        why.append(f"files/{Path(files_path).name} lists other files or bytes")
    if nlink > 1 and not same_out:
        why.append(f"{out.name} has {nlink} hardlinks (it is in upload/)")
    if why and not (replace_unregistered and link is None and nlink <= 1):
        staged.unlink(missing_ok=True)
        raise NameFrozen(f"NAME_FROZEN {name}: this build (sha256 {new_sha[:16]}..) differs: "
                         f"{'; '.join(why)}. A name never changes once built or uploaded: bump "
                         "--version" + ("" if link is not None or nlink > 1 else
                                        " (or --replace-unregistered: never uploaded)"))
    if same_out:
        staged.unlink(missing_ok=True)
        return "unchanged"
    refuse_linked(out)
    os.replace(staged, out)
    return "installed"


# --------------------------------------------------------------------------- unpack and verify
def member_dest(data_root: Path, bundle: str, arcname: str) -> Path:
    """Payload goes to DATA_ROOT/<arcname>; root-level members to DATA_ROOT/.bundles/<bundle>/."""
    check_name(arcname)
    if "/" not in arcname:
        return Path(data_root) / ".bundles" / bundle / arcname
    return Path(data_root) / arcname


def parse_files_sha256(text: str) -> list[tuple[str, str]]:
    """sha256sum format -> [(name, sha)]; refuses malformed lines."""
    rows = []
    for ln, line in enumerate(text.splitlines(), 1):
        m = _FILES_LINE.fullmatch(line)
        if not m:
            raise VerifyError(f"FILES.sha256 line {ln} is malformed: {line[:80]!r}")
        rows.append((m.group(2), m.group(1)))
    return rows


def extract_bundle(zip_path: Path, data_root: Path, bundle: str, *,
                   n_files: int | None = None) -> dict:
    """Unpack a bundle safely, then verify every FILES.sha256 entry FROM DISK.

    Directory entries are skipped and never counted (topaneu counted them and rebuilt for
    nothing). Symlinks, absolute paths and '..' are refused. Returns {'n_files', 'n_extracted'}.
    """
    data_root = Path(data_root)
    n_extracted = 0
    with zipfile.ZipFile(zip_path) as z:
        infos = [i for i in z.infolist() if not i.filename.endswith("/")]
        names = {i.filename for i in infos}
        missing = [n for n in META_NAMES if n not in names]
        if missing:
            raise VerifyError(f"{bundle}: {', '.join(missing)} missing at the zip root")
        for i in infos:
            if ((i.external_attr >> 16) & 0o170000) == 0o120000:
                raise VerifyError(f"{bundle}: symlink member {i.filename!r} refused")
            dest = member_dest(data_root, bundle, i.filename)
            dest.parent.mkdir(parents=True, exist_ok=True)
            # A per-writer temp name: two bundles may carry the same file (u2f-hd1k-a-v1 and -b
            # both list HD1K/hd1k_input/README.txt) and fetch.py unpacks two items in parallel.
            # The same pattern as fetch._write_payload.
            tmp = dest.with_name(f"{dest.name}.{os.getpid()}-{threading.get_ident()}.u2ftmp")
            try:
                with z.open(i) as src, open(tmp, "wb") as out:
                    shutil.copyfileobj(src, out, CHUNK)
                os.replace(tmp, dest)
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise
            n_extracted += 1
    return dict(verify_extracted(data_root, bundle, n_files=n_files), n_extracted=n_extracted)


def verify_extracted(data_root: Path, bundle: str, *, n_files: int | None = None) -> dict:
    meta = Path(data_root) / ".bundles" / bundle / "FILES.sha256"
    if not meta.is_file():
        raise VerifyError(f"{bundle}: {meta} missing")
    rows = parse_files_sha256(meta.read_text())
    bad = []
    for name, want in rows:
        p = member_dest(data_root, bundle, name)
        if not p.is_file():
            bad.append(f"missing {name}")
        elif sha256_file(p) != want:
            bad.append(f"sha256 differs: {name}")
    if bad:
        raise VerifyError(f"{bundle}: {len(bad)} of {len(rows)} files fail FILES.sha256, e.g. "
                          + "; ".join(bad[:3]))
    if n_files is not None and len(rows) != n_files:
        raise VerifyError(f"{bundle}: FILES.sha256 lists {len(rows)} files, manifest n_files "
                          f"is {n_files}")
    return {"n_files": len(rows)}


def check_zip(path: Path, *, size: int | None = None, sha256: str | None = None) -> str | None:
    """Return None when the zip on disk has the right bytes, magic and sha256; else the reason.

    Always reads from disk: gdown's 'Skipping already downloaded file' (download.py:898-901)
    and a truncated transfer both leave a file whose name looks right.
    """
    path = Path(path)
    if not path.is_file():
        return "absent"
    with open(path, "rb") as f:
        head = f.read(4)
    if head != ZIP_MAGIC:
        return f"no PK magic (starts {head!r})"
    st = path.stat().st_size
    if size is not None and st != size:
        return f"{st} bytes, manifest says {size}"
    if sha256 is not None and sha256_file(path) != sha256:
        return "sha256 differs from the manifest"
    return None


# --------------------------------------------------------------------------- sources
class LocalZipSource:
    """A source zip on local disk (the Mac build, and the tests). zipfile decodes stored,
    deflated, bzip2 and lzma members and checks each member's CRC-32 at its end."""

    methods = LOCAL_METHODS

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.z = zipfile.ZipFile(self.path)

    def infolist(self) -> list[zipfile.ZipInfo]:
        return self.z.infolist()

    def member(self, arcname: str, info: zipfile.ZipInfo, sha: str | None) -> Member:
        # ZipExtFile checks the CRC at EOF, and _write_member checks it again against info.CRC.
        return Member(arcname, info.file_size, lambda i=info: self.z.open(i), info.CRC, sha)


class PartialSource:
    """A range-extracted source (tools/data/range_extract.py): the files under `root`, listed
    by its range manifest with the remote zip's sizes and CRC-32s. The files on disk are already
    decoded, so every member is 'stored'; write_zip re-checks each CRC-32 while copying."""

    methods = frozenset({zipfile.ZIP_STORED})

    def __init__(self, root: Path, manifest: Path) -> None:
        self.root, self.manifest = Path(root), Path(manifest)
        doc = load_json(self.manifest)
        if doc.get("complete") is False:
            raise Precondition(f"{self.manifest}: range_extract.py reported failures; re-run it")
        self.doc = doc
        self._infos = []
        for name, rec in sorted((doc.get("members") or {}).items()):
            zi = zipfile.ZipInfo(check_name(name))
            zi.file_size = zi.compress_size = int(rec["size"])
            zi.CRC = int(rec["crc32"], 16)
            self._infos.append(zi)
        if not self._infos:
            raise Precondition(f"{self.manifest} lists no member")

    def infolist(self) -> list[zipfile.ZipInfo]:
        return list(self._infos)

    def path(self, info: zipfile.ZipInfo) -> Path:
        return self.root / check_name(info.filename)

    def missing(self) -> list[str]:
        """Listed files that are absent or have another size (a partial or stale extraction)."""
        out = []
        for i in self._infos:
            p = self.path(i)
            if not p.is_file() or p.stat().st_size != i.file_size:
                out.append(i.filename)
        return out

    def member(self, arcname: str, info: zipfile.ZipInfo, sha: str | None) -> Member:
        p = self.path(info)
        if not p.is_file() or p.stat().st_size != info.file_size:
            raise Precondition(f"{p} is absent or not {info.file_size} B: re-run range_extract.py")
        return Member(arcname, info.file_size, lambda q=p: open(q, "rb"), info.CRC, sha)


class _HttpRaw(io.RawIOBase):
    def __init__(self, src: HttpZipSource) -> None:
        self.src, self.pos = src, 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, off: int, whence: int = 0) -> int:
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.src.size + off
        return self.pos

    def readinto(self, b: Any) -> int:
        n = min(len(b), max(0, self.src.size - self.pos))
        if n == 0:
            return 0
        data = self.src.get(self.pos, self.pos + n - 1)
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


class HttpZipSource:
    """A remote zip read by HTTP range requests (KITTI S3 on a VM; range_extract.py on the Mac).

    Stored and deflated members. Every read must answer 206 with the exact Content-Range span
    and byte count; the body is streamed, so a server that ignores Range (200 with the whole
    file) is refused after its headers, before its body. Connection errors, 5xx, 408 and 429 are
    retried with back-off; 200 and other 4xx are refused at once. One range GET per member
    (header + data together); rebuild_on_vm prefetches 8 at a time, because a bundle's members
    are scattered through the source zip (its order is not sorted).
    """

    methods = HTTP_METHODS
    attempts = 5
    backoff = 1.0  # seconds before the 2nd attempt, doubling, at most 30 s (tests set 0)

    def __init__(self, url: str, *, timeout: float = 60, size: int | None = None) -> None:
        import requests  # lazy: only range readers need it

        self._requests = requests
        self.url, self.timeout = url, timeout
        self._local = threading.local()
        self.size = size if size is not None else self._probe_size()
        self.z = zipfile.ZipFile(io.BufferedReader(_HttpRaw(self), buffer_size=1 << 20))

    def _session(self) -> Any:
        s = getattr(self._local, "s", None)
        if s is None:
            s = self._local.s = self._requests.Session()
        return s

    def _range(self, start: int, end: int) -> Any:
        return self._session().get(self.url, headers={"Range": f"bytes={start}-{end}",
                                                      "Accept-Encoding": "identity"},
                                   timeout=self.timeout, stream=True)

    def _probe_size(self) -> int:
        """HEAD first. DaRUS answers HEAD 403 but serves ranges, so fall back to one 1-byte
        range from byte 1 (never byte 0, which SURFdrive ignores) and read the total from it."""
        try:
            r = self._session().head(self.url, timeout=self.timeout, allow_redirects=True)
            n = r.headers.get("Content-Length", "")
            if r.status_code == 200 and n.isdigit():
                return int(n)
            why = f"HEAD answered {r.status_code}"
        except self._requests.RequestException as e:
            why = f"HEAD failed ({type(e).__name__})"
        try:
            with self._range(1, 1) as r:
                m = _CONTENT_RANGE.fullmatch(r.headers.get("Content-Range", "").strip())
                if r.status_code == 206 and m and m.group(3) != "*":
                    return int(m.group(3))
                why += f"; a 1-byte range answered {r.status_code}"
        except self._requests.RequestException as e:
            why += f"; a 1-byte range failed ({type(e).__name__}: {e})"
        raise RangeError(f"cannot size {self.url}: {why}")

    def get(self, start: int, end: int) -> bytes:
        """Bytes [start, end], inclusive and clipped to the file."""
        end = min(end, self.size - 1)
        want = end - start + 1
        if want <= 0:
            return b""
        last: Exception | None = None
        for k in range(self.attempts):
            if k:
                time.sleep(min(30.0, self.backoff * 2 ** (k - 1)))
            try:
                with self._range(start, end) as r:
                    if r.status_code != 206:
                        last = RangeError(f"range {start}-{end}: HTTP {r.status_code}" + (
                            " (the server ignored Range)" if r.status_code == 200 else ""))
                        if r.status_code in (408, 429) or r.status_code >= 500:
                            continue
                        raise last
                    m = _CONTENT_RANGE.fullmatch(r.headers.get("Content-Range", "").strip())
                    if not m or (int(m.group(1)), int(m.group(2))) != (start, end):
                        last = RangeError(f"range {start}-{end}: Content-Range "
                                          f"{r.headers.get('Content-Range')!r}")
                        continue
                    buf = bytearray()
                    for block in r.iter_content(CHUNK):
                        buf += block
                        if len(buf) > want:
                            break
                    if len(buf) == want:
                        return bytes(buf)
                    last = RangeError(f"range {start}-{end}: {len(buf)} bytes, {want} expected")
            except self._requests.RequestException as e:
                last = e
        raise RangeError(f"range read failed after {self.attempts} attempts: {last}")

    def infolist(self) -> list[zipfile.ZipInfo]:
        return self.z.infolist()

    def read_member(self, info: zipfile.ZipInfo) -> bytes:
        if info.compress_type not in self.methods:
            raise BundleError(f"{info.filename}: compression method {info.compress_type} is not "
                              "range-read (stored and deflated only)")
        nlen = len(info.filename.encode())
        blob = self.get(info.header_offset, info.header_offset + 30 + nlen + 256
                        + info.compress_size - 1)
        if len(blob) < 30:
            raise VerifyError(f"{info.filename}: local header past the end of the file")
        sig, *_, n2, elen = struct.unpack(LOCAL_FMT, blob[:30])
        if sig != LOCAL_SIG:
            raise BundleError(f"{info.filename}: no local header at {info.header_offset}")
        start = 30 + n2 + elen
        raw = blob[start:start + info.compress_size]
        if len(raw) < info.compress_size:
            a = info.header_offset + start + len(raw)
            raw += self.get(a, info.header_offset + start + info.compress_size - 1)
        return decode_member(raw, info.compress_type, info.file_size, info.filename)

    def member(self, arcname: str, info: zipfile.ZipInfo, sha: str | None) -> Member:
        fetch = lambda i=info: self.read_member(i)  # noqa: E731
        return Member(arcname, info.file_size, lambda: io.BytesIO(fetch()), info.CRC, sha, fetch)


class _Prefetch:
    """Keep up to `window` remote members in flight, in writer order; memory stays bounded."""

    def __init__(self, fetchers: list[Callable[[], bytes]], window: int = 8) -> None:
        self.f = fetchers
        self.window = window
        self.pool = ThreadPoolExecutor(max_workers=window)
        self.fut: dict[int, Future] = {}

    def get(self, i: int) -> bytes:
        for j in range(i, min(i + self.window, len(self.f))):
            if j not in self.fut:
                self.fut[j] = self.pool.submit(self.f[j])
        return self.fut.pop(i).result()

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)


# --------------------------------------------------------------------------- selection
def resolve_ids(spec: dict | None, split: dict | None) -> set[str] | None:
    if spec is None:
        return None
    if "list" in spec:
        ids = sorted(spec["list"])
    else:
        key = {"split:dev": "dev_ids", "split:heldout": "heldout_ids"}.get(spec.get("from", ""))
        if key is None:
            raise BundleError(f"unknown ids source {spec!r}")
        if split is None:
            raise Precondition("this bundle needs the split (tools/prereg/make_split.py)")
        ids = sorted(split[key])
    if "first" in spec:
        ids = ids[: int(spec["first"])]
    return set(ids)


def uses_split(spec: dict) -> bool:
    return any(str((s.get("ids") or {}).get("from", "")).startswith("split:")
               or str(s.get("repo_file", "")).startswith("splits/")
               for s in spec.get("select") or [])


def _check_select(name: str, k: int, sel: dict) -> None:
    allowed = _REPO_SELECT if "repo_file" in sel else _ZIP_SELECT
    extra = sorted(x for x in sel if x not in allowed and not x.startswith("_"))
    if extra:
        raise BundleError(f"{name}: select #{k} has unknown key(s) {extra} "
                          f"(allowed: {', '.join(sorted(allowed))})")
    if "repo_file" in sel:
        if "dest" not in sel:
            raise BundleError(f"{name}: select #{k} (repo_file) needs dest")
        if not _REPO_FILE.match(str(sel["repo_file"])) or not _REPO_FILE.match(str(sel["dest"])):
            raise BundleError(f"{name}: select #{k}: a repo file must be splits/<name>.json (the "
                              f"committed split), not {sel['repo_file']!r} -> {sel['dest']!r}")
        return
    if "src" not in sel or "prefix" not in sel:
        raise BundleError(f"{name}: select #{k} needs src and prefix")
    if not any(sel.get(x) for x in ("dirs", "files", "match")):
        raise BundleError(f"{name}: select #{k} on {sel['src']} needs dirs, files or match")
    for x in ("dirs", "files", "exclude"):
        if not isinstance(sel.get(x, []), list):
            raise BundleError(f"{name}: select #{k}: {x} must be a list")


def select_members(doc: dict, name: str, *, sources: dict[str, Any], split: dict | None,
                   repo_root: Path | None, allow_sealed: bool = False, on_vm: bool = False,
                   expected: dict[str, str] | None = None) -> list[Member]:
    """Apply one bundle's `select` rules to its sources, with the seal checks first."""
    spec = spec_of(doc, name)
    if spec.get("sealed") and not allow_sealed:
        raise SealRefusal(f"{name} is sealed: only tools/prereg/seal.py builds it (prereg, Tier 1)")
    heldout = set(split["heldout_ids"]) if split else set()
    members: list[Member] = []
    for k, sel in enumerate(spec.get("select") or [], 1):
        _check_select(name, k, sel)
        if "repo_file" in sel:
            if on_vm or repo_root is None:
                raise BundleError(f"{name}: repo files ({sel['repo_file']}) are Mac-side only")
            p = Path(repo_root) / sel["repo_file"]
            if not p.is_file():
                raise Precondition(f"{name} needs {sel['repo_file']}")
            members.append(file_member(sel["dest"], p))
            continue
        src_name = sel["src"]
        if src_name not in doc["sources"]:
            raise BundleError(f"{name}: select #{k} names unknown source {src_name!r}")
        src_spec = doc["sources"][src_name]
        ids = resolve_ids(sel.get("ids"), split)
        if src_spec.get("heldout_guard") and not allow_sealed:
            if ids is None:
                raise SealRefusal(f"{name}: a select on {src_name} without an ids filter would "
                                  "take held-out GT")
            if split is None:
                raise Precondition(f"{name}: {src_name} needs the split to check the seal")
            leak = sorted(ids & heldout)
            if leak:
                raise SealRefusal(f"{name}: held-out ids {leak[:5]} selected from {src_name}")
        dirs = tuple(d.rstrip("/") + "/" for d in sel.get("dirs") or [])
        files = set(sel.get("files") or [])
        match = re.compile(sel["match"]) if sel.get("match") else None
        exclude = [re.compile(x) for x in sel.get("exclude") or []]
        strip = str(sel.get("strip") or "")
        frames = sel.get("frames")
        src = sources[src_name]
        methods = getattr(src, "methods", HTTP_METHODS)
        infos = src.infolist()
        absent = sorted(files - {i.filename for i in infos})
        if absent:
            raise BundleError(f"{name}: select #{k}: {src_name} has no {absent[:3]}")
        n0 = len(members)
        for info in infos:
            n = info.filename
            if n.endswith("/") or is_junk(n):
                continue
            if (dirs or files) and not (n in files or (dirs and n.startswith(dirs))):
                continue
            if match is not None and not match.search(n):
                continue
            if any(x.search(n) for x in exclude):
                continue
            sid, fr = id_frame(n)
            if ids is not None and sid not in ids:
                continue
            if frames is not None and (fr is None or not any(lo <= fr <= hi for lo, hi in frames)):
                continue
            if strip and not n.startswith(strip):
                raise BundleError(f"{name}: {src_name}:{n} does not start with strip {strip!r}")
            if info.compress_type not in methods:
                raise BundleError(f"{name}: {src_name}:{n} uses compression method "
                                  f"{info.compress_type}; this source reads {sorted(methods)}")
            arc = sel["prefix"] + n[len(strip):]
            want = None
            if expected is not None:
                want = expected.get(arc)
                if want is None:
                    raise VerifyError(f"{name}: {arc} is not in files/{name}.sha256 "
                                      "(manifest drift)")
            members.append(src.member(arc, info, want))
        if len(members) == n0 and not sel.get("allow_empty"):
            raise BundleError(f"{name}: select #{k} on {src_name} matched nothing")
    return members


def licence_keys(spec: dict) -> list[str]:
    """`licence` is one key of the manifest's `licences`, or a list of them."""
    lic = spec.get("licence")
    keys = [lic] if isinstance(lic, str) else list(lic or [])
    if not keys:
        raise BundleError(f"{spec.get('name')}: no licence named")
    return keys


def render_licence(lic: dict) -> bytes:
    """A licence member's text. `text` (a string or a list of lines) is used as is. Otherwise the
    structured fields, laid out exactly as v1 did for KITTI, so KITTI bundles keep their bytes;
    url, terms, citations and repackaging_note are optional. ASCII only, as in v1."""
    if lic.get("text") is not None:
        t = lic["text"]
        t = "\n".join(t) if isinstance(t, list) else str(t)
        return (t.rstrip("\n") + "\n").encode("ascii")
    head = f"{lic['spdx']} ({lic['url']})" if lic.get("url") else str(lic["spdx"])
    lines = [head, "", lic["notice"], f"Source: {lic['notice_source']}"]
    if lic.get("terms"):
        lines += ["", "Terms:"] + [f"- {t}" for t in lic["terms"]]
    if lic.get("citations"):
        lines += ["", "Cite:"] + [f"- {c}" for c in lic["citations"]]
    if lic.get("repackaging_note"):
        lines += ["", lic["repackaging_note"]]
    lines += [""]
    return "\n".join(lines).encode("ascii")


def licence_text(doc: dict, key: str) -> bytes:
    lic = (doc.get("licences") or {}).get(key)
    if lic is None:
        raise BundleError(f"licence {key!r} is not in the manifest's licences")
    try:
        return render_licence(lic)
    except KeyError as e:
        raise BundleError(f"licence {key}: field {e} missing") from None
    except UnicodeEncodeError as e:
        raise BundleError(f"licence {key}: non-ASCII text ({e.object[e.start:e.end]!r})") from None


def licence_members(doc: dict, spec: dict) -> list[tuple[str, bytes]]:
    out = []
    for k in licence_keys(spec):
        text = licence_text(doc, k)
        out.append((str(doc["licences"][k].get("file") or f"LICENSE-{k}.txt"), text))
    return out


def render_bundle_json(spec: dict, source_shas: dict[str, str | None], split_sha: str | None,
                       n_payload: int, payload_bytes: int) -> bytes:
    """BUNDLE.json holds nothing that differs between the Mac build and a VM rebuild."""
    used = sorted({s["src"] for s in spec.get("select") or [] if "src" in s})
    doc = {
        "schema": 1, "name": spec["name"], "version": spec["version"], "role": spec["role"],
        "licence": spec["licence"], "gt_bearing": bool(spec.get("gt_bearing")),
        "sources": {s: source_shas.get(s) for s in used},
        "split_sha256": split_sha if uses_split(spec) else None,
        "n_payload": n_payload, "payload_bytes": payload_bytes,
        "layout": ("payload paths are relative to DATA_ROOT; the licence, FILES.sha256 and "
                   "BUNDLE.json sit at the zip root and fetch.py moves them to "
                   "DATA_ROOT/.bundles/<name>/"),
        "format": "tools/data/build_bundles.py zip v1: stored, sorted, 1980-01-01, 0644, no ZIP64",
    }
    if is_hold(spec):
        doc["distribution"] = "hold: never in the public folder; private Mac push only"
    return (json.dumps(doc, indent=1, sort_keys=True) + "\n").encode("ascii")


def build_bundle(doc: dict, name: str, *, sources: dict[str, Any], split: dict | None,
                 split_sha: str | None, source_shas: dict[str, str | None], out_path: Path,
                 repo_root: Path | None, allow_sealed: bool = False, on_vm: bool = False,
                 expected: dict[str, str] | None = None, prefetch: int = 0) -> dict:
    spec = spec_of(doc, name)
    if builder_of(spec) != "build_bundles":
        raise BundleError(f"{name} is built by tools/data/{builder_of(spec)}.py")
    check_out_path(spec, out_path)
    members = select_members(doc, name, sources=sources, split=split, repo_root=repo_root,
                             allow_sealed=allow_sealed, on_vm=on_vm, expected=expected)
    members.sort(key=lambda m: m.name)
    lic = licence_members(doc, spec)
    if expected is not None:
        got = {m.name for m in members} | {n for n, _ in lic}
        if got != set(expected):
            raise VerifyError(f"{name}: {len(set(expected) - got)} expected files not in the "
                              f"source, {len(got - set(expected))} unexpected")
    pf = None
    if prefetch and all(m.fetch for m in members):
        pf = _Prefetch([m.fetch for m in members], window=prefetch)
        members = [dataclasses.replace(m, opener=lambda i=i: io.BytesIO(pf.get(i)))
                   for i, m in enumerate(members)]
    bj = render_bundle_json(spec, source_shas, split_sha, len(members),
                            sum(m.size for m in members))
    try:
        res = write_zip(out_path, members, licences=lic, bundle_json=bj)
    finally:
        if pf:
            pf.close()
    res["source_shas"] = {s: source_shas.get(s) for s in
                          sorted({x["src"] for x in spec["select"] if "src" in x})}
    res["split_sha256"] = split_sha if uses_split(spec) else None
    return res


def rebuild_on_vm(doc: dict, name: str, *, out_path: Path, files_sha256: Path,
                  source_factory: Callable[[str], Any] = HttpZipSource,
                  prefetch: int = 8) -> dict:
    """Rebuild an IMAGE bundle from the KITTI S3 zips; the zip must equal the Mac build.

    Refuses (SealRefusal, exit 50) sealed or GT-bearing bundles and GT-bearing sources: GT never
    crosses to a VM except as the Mac-built dev GT bundles.
    """
    spec = spec_of(doc, name)
    if spec.get("sealed") or spec.get("gt_bearing"):
        raise SealRefusal(f"{name} carries GT: never rebuilt on a VM (push it from the Mac)")
    if not spec.get("rebuildable_on_vm"):
        raise BundleError(f"{name} is not marked rebuildable_on_vm")
    for s in spec["select"]:
        if "repo_file" in s:
            raise BundleError(f"{name} includes repo files: not rebuildable on a VM")
        if doc["sources"][s["src"]].get("gt_bearing"):
            raise SealRefusal(f"{name} reads {s['src']}, which carries GT: refused on a VM")
    if not spec.get("sha256") or not spec.get("source_shas"):
        raise Precondition(f"{name} has no Mac build recorded (sha256, source_shas)")
    expected = {n: sha for n, sha in parse_files_sha256(Path(files_sha256).read_text())}
    srcs = {s["src"]: source_factory(doc["sources"][s["src"]]["url"]) for s in spec["select"]}
    res = build_bundle(doc, name, sources=srcs, split=None, split_sha=None,
                       source_shas=spec["source_shas"], out_path=out_path, repo_root=None,
                       on_vm=True, expected=expected, prefetch=prefetch)
    if res["sha256"] != spec["sha256"]:
        Path(out_path).unlink(missing_ok=True)
        raise VerifyError(f"{name}: the rebuilt zip's sha256 differs from the Mac build")
    return res


# --------------------------------------------------------------------------- Mac CLI
def _sources_record(data_root: Path) -> dict:
    p = data_root / "src" / "SOURCES.json"
    return (load_json(p).get("sources") or {}) if p.is_file() else {}


def open_sources(doc: dict, spec: dict, data_root: Path,
                 rec: dict | None = None) -> tuple[dict[str, Any], dict[str, str]]:
    """LocalZipSource and recorded sha256 for every source a bundle selects from; each must be
    on disk and verified in src/SOURCES.json (download_sources.sh), else Precondition."""
    rec = _sources_record(data_root) if rec is None else rec
    n = spec.get("name")
    srcs: dict[str, Any] = {}
    shas: dict[str, str] = {}
    for s in sorted({x["src"] for x in spec.get("select") or [] if "src" in x}):
        if s not in doc["sources"]:
            raise BundleError(f"{n}: unknown source {s!r}")
        ss = doc["sources"][s]
        r = rec.get(s) or {}
        if ss.get("kind") == "partial":
            srcs[s], shas[s] = open_partial(s, ss, data_root, r, n)
            continue
        path = data_root / ss["local"]
        if not path.is_file():
            raise Precondition(f"{n}: {s} is not at {path} (tools/data/download_sources.sh)")
        if not r.get("sha256") or r.get("bytes") != path.stat().st_size:
            raise Precondition(f"{n}: {s} is not verified in src/SOURCES.json")
        srcs[s], shas[s] = LocalZipSource(path), r["sha256"]
    return srcs, shas


def open_partial(s: str, ss: dict, data_root: Path, r: dict, n: object) -> tuple[Any, str]:
    """A range-extracted source, verified: its range manifest's sha256 must be the one
    download_sources.sh recorded (manifest_sha256), and every listed file must be on disk."""
    if not ss.get("local_dir") or not ss.get("local_manifest"):
        raise BundleError(f"{n}: partial source {s} needs local_dir and local_manifest")
    man = data_root / ss["local_manifest"]
    if not man.is_file():
        raise Precondition(f"{n}: {s}: no range manifest at {man} (download_sources.sh, which "
                           "runs tools/data/range_extract.py)")
    sha = sha256_file(man)
    if r.get("manifest_sha256") != sha:
        raise Precondition(f"{n}: {s} is not verified in src/SOURCES.json (its range manifest "
                           "changed or was never adopted: download_sources.sh --adopt)")
    src = PartialSource(data_root / ss["local_dir"], man)
    bad = src.missing()
    if bad:
        raise Precondition(f"{n}: {s}: {len(bad)} listed file(s) absent or resized under "
                           f"{src.root}, e.g. {bad[0]}; re-run range_extract.py")
    return src, sha


def _selected(doc: dict, only: list[str], wave: int | None, *, hold: bool = False,
              purpose: str = "build") -> list[str]:
    """Bundle names for a command. `purpose` is build, plan or upload: a build skips bundles
    that another tool builds (build_kraw); an upload list never takes a HOLD bundle."""
    bundles = all_specs(doc)
    for n in only:
        if n not in bundles:
            raise BundleError(f"unknown bundle {n!r}")
        if bundles[n].get("sealed"):
            raise SealRefusal(f"{n} is sealed: tools/prereg/seal.py (prereg, Tier 1) builds it")
    if only:
        names = list(only)
    elif hold:
        names = [n for n, b in bundles.items() if is_hold(b)]
    else:  # an explicit --wave takes that wave whatever month1 says (wave 3 is month1=false)
        names = [n for n, b in (doc.get("bundles") or {}).items()
                 if is_public(b) and (b.get("wave") == wave if wave is not None
                                      else b.get("month1"))]
    if purpose == "upload":
        held = [n for n in names if not is_public(bundles[n])]
        if held:
            raise HoldRefusal(f"{held[0]} is a HOLD bundle (uploadable=false): never uploaded")
    if purpose == "build":
        foreign = [n for n in names if builder_of(bundles[n]) != "build_bundles"]
        if only and foreign:
            raise BundleError(f"{foreign[0]} is built by "
                              f"tools/data/{builder_of(bundles[foreign[0]])}.py")
        for n in foreign:
            print(f"SKIP {n}: built by tools/data/{builder_of(bundles[n])}.py")
        names = [n for n in names if n not in foreign]
    return names


def _estimate(spec: dict) -> tuple[int | None, str]:
    p = spec.get("planned") or {}
    n = p.get("n_payload")
    b = p.get("payload_bytes") or p.get("payload_bytes_E")
    if n is None or b is None:
        return None, "unknown"
    # [E] average member name ~45 chars (kitti15/multi_view/training/image_2/000000_00.png)
    return 22 + n * (76 + 90) + b + 4096, "E"


def _load_split(repo: Path, split_path: Path, need: bool) -> tuple[dict | None, str | None]:
    if not split_path.is_file():
        if need:
            raise Precondition(f"{split_path} is missing: run tools/prereg/make_split.py first")
        return None, None
    rel = os.path.relpath(split_path, repo)
    if need and not is_committed(repo, rel):
        raise Precondition(f"{rel} must be committed before any GT bundle is built (prereg)")
    return load_json(split_path), sha256_file(split_path)


def cmd_plan(doc: dict, names: list[str], data_root: Path) -> int:
    rec = _sources_record(data_root)
    print(f"{'bundle':32} {'wave':>4} {'role':6} {'n_payload':>9} {'zip bytes':>14}  status")
    total = 0
    for n in names:
        spec = spec_of(doc, n)
        est, std = _estimate(spec)
        srcs = sorted({s["src"] for s in spec.get("select") or [] if "src" in s})
        missing = [s for s in srcs if not ((rec.get(s) or {}).get("sha256")
                                           or (rec.get(s) or {}).get("manifest_sha256"))]
        shown = ", ".join(missing[:3]) + (f" and {len(missing) - 3} more" if len(missing) > 3
                                          else "")
        status = "built" if spec.get("sha256") else ("needs " + shown if missing else "ready")
        if builder_of(spec) != "build_bundles":
            status += f" (tools/data/{builder_of(spec)}.py builds it)"
        if not is_public(spec):
            status += f"; HOLD -> {HOLD_DIR}/"
        if spec.get("bytes"):
            est, std = spec["bytes"], "measured"
        cap_ok = est is None or est <= CAP_BYTES
        total += est or 0
        print(f"{n:32} {spec.get('wave') or '-':>4} {spec['role']:6} "
              f"{(spec.get('planned') or {}).get('n_payload') or '?':>9} "
              f"{est if est is not None else '?':>14}  {std}; {status}"
              + ("" if cap_ok else "  OVER CAP"))
    print(f"total about {total / 1e9:.2f} GB; cap per bundle {CAP_BYTES} B (3.9 GiB, no ZIP64)")
    return 0


def cmd_upload_list(doc: dict, names: list[str], data_root: Path, wave: int | None) -> int:
    """The built files of a selection, with sha256. NOT an upload instruction: the only thing
    the user uploads is upload/u2flow-bundles-v1, which assemble_upload.py builds behind its
    guard (review 2026-09-25: this list used to say 'Upload ...' and point at bundles/)."""
    refuse_public_leaks(doc, data_root)
    missing, total = 0, 0
    w = wave if wave is not None else "<N>"
    print("Built files only (NOT an upload list: never drag bundles/). The one folder to upload is "
          f"{data_root / 'upload' / 'u2flow-bundles-v1'}, made by `make upload WAVE={w}` "
          f"(tools/data/assemble_upload.py --wave {w}) behind its fail-closed guard.")
    for n in names:
        b = spec_of(doc, n)
        path = bundle_out_path(data_root, b)
        if not b.get("sha256") or not path.is_file():
            print(f"NOT_BUILT {b['file']}")
            missing += 1
            continue
        total += b["bytes"]
        print(f"BUILT_FILE {path}  {b['bytes']} B  sha256 {b['sha256']}")
    print(f"total {total / 1e9:.2f} GB in {len(names) - missing} files; {missing} not built")
    return 3 if missing else 0


def unzip_test(path: Path, name: str) -> None:
    """Info-ZIP's own reader must accept the zip (it is how a user would check it)."""
    if shutil.which("unzip"):
        t = subprocess.run(["unzip", "-tq", str(path)], capture_output=True, text=True)
        if t.returncode != 0:
            Path(path).unlink(missing_ok=True)
            raise VerifyError(f"{name}: unzip -tq failed: {t.stdout[-300:]}{t.stderr[-300:]}")


def record_build(manifest: Path, doc: dict, name: str, res: dict, fields: dict) -> None:
    """Write files/<name>.sha256, then merge the build's fields into the manifest entry under
    the lock (update_manifest) and into the in-memory doc."""
    files_dir = Path(manifest).parent / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    save_text = files_text(res)
    p = files_dir / f"{name}.sha256"
    if not p.is_file() or p.read_text() != save_text:
        tmp = tmp_name(p)
        tmp.write_text(save_text)
        os.replace(tmp, p)
    update_manifest(manifest, section_of(doc, name), name, fields)
    spec_of(doc, name).update(fields)


def cmd_build(doc: dict, names: list[str], *, manifest: Path, data_root: Path, repo: Path,
              split_path: Path, extract_gt: bool, replace_unregistered: bool = False) -> int:
    refuse_public_leaks(doc, data_root)
    rec = _sources_record(data_root)
    files_dir = manifest.parent / "files"
    need_split = any(uses_split(spec_of(doc, n)) for n in names)
    split, split_sha = _load_split(repo, split_path, need_split)
    builder = git_sha(repo)
    rc = 0
    for n in names:
        spec = spec_of(doc, n)
        out = bundle_out_path(data_root, spec)
        hold = not is_public(spec)
        done = spec.get("sha256") and check_zip(out, size=spec.get("bytes"),
                                                sha256=spec["sha256"]) is None
        if done:
            print(f"BUILT {n} (already, sha256 {spec['sha256'][:12]}..)")
        else:
            srcs, shas = open_sources(doc, spec, data_root, rec)
            print(f"BUILD {n} -> {out}" + (" (HOLD: never uploaded)" if hold else ""), flush=True)
            staged = staging_path(out)
            try:
                res = build_bundle(doc, n, sources=srcs, split=split, split_sha=split_sha,
                                   source_shas=shas, out_path=staged, repo_root=repo)
                unzip_test(staged, n)
                how = install_build(n, staged, out, res, spec=spec,
                                    files_path=files_dir / f"{n}.sha256",
                                    links_path=manifest.parent / "links.v1.json",
                                    replace_unregistered=replace_unregistered)
            finally:
                staged.unlink(missing_ok=True)
            record_build(manifest, doc, n, res, dict(
                bytes=res["bytes"], sha256=res["sha256"], n_files=res["n_files"],
                n_payload=res["n_payload"], payload_bytes=res["payload_bytes"],
                source_shas=res["source_shas"], split_sha256=res["split_sha256"],
                builder_git_sha=builder, built_utc=utc_now(),
                path=os.path.relpath(out, data_root)))
            print(f"BUILT {n} {res['bytes']} B sha256 {res['sha256']} n_files {res['n_files']}"
                  + (" (the same bytes were already there)" if how == "unchanged" else ""))
            queue_evidence("data-integrity", "bundle-build", "measured",
                           "tools/data/build_bundles.py; " + str(out),
                           {"bundle": n, "bytes": res["bytes"], "sha256": res["sha256"],
                            "n_files": res["n_files"], "unzip_test": bool(shutil.which("unzip")),
                            "hold": hold})
        if extract_gt and spec.get("mac_extract_to"):
            dest = data_root / spec["mac_extract_to"]
            r = extract_bundle(out, dest, n, n_files=spec.get("n_files"))
            print(f"EXTRACTED {n} -> {dest} ({r['n_files']} files verified from disk)")
    return rc


def main(argv: list[str] | None = None) -> int:
    ap = ArgParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT_MAC)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--repo", type=Path, default=REPO)
    ap.add_argument("--split", type=Path, default=None,
                    help="default: <repo>/splits/k15_split_v1.json")
    sel = ap.add_mutually_exclusive_group()
    sel.add_argument("--wave", type=int, choices=(1, 2, 3))
    sel.add_argument("--hold", action="store_true",
                     help="the HOLD bundles (uploadable=false, not sealed), into DATA_ROOT/hold/")
    ap.add_argument("--only", action="append", default=[], metavar="NAME")
    ap.add_argument("--plan", action="store_true", help="print sizes and readiness; write nothing")
    ap.add_argument("--dry-run", action="store_true", help="same as --plan")
    ap.add_argument("--upload-list", action="store_true")
    ap.add_argument("--extract-gt", action="store_true",
                    help="also unpack bundles with mac_extract_to (dev and KITTI-12 GT) for the "
                         "Mac scorer")
    ap.add_argument("--replace-unregistered", action="store_true",
                    help="let a different build replace a recorded name that was never uploaded "
                         "(not in links.v1.json, no hardlink); otherwise NAME_FROZEN, exit 1")
    ap.add_argument("--finalize", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    try:
        if a.finalize:
            raise SealRefusal("moving data_scene_flow.zip under sealed/ is tools/prereg/seal.py's "
                              "job (prereg, Tier 1); the hook lets no other program touch the seal")
        doc = load_json(a.manifest)
        purpose = "plan" if a.plan or a.dry_run else "upload" if a.upload_list else "build"
        if purpose == "upload" and a.hold:
            raise HoldRefusal("HOLD bundles are never uploaded: there is no upload list for them")
        names = _selected(doc, a.only, a.wave, hold=a.hold, purpose=purpose)
        if not names:
            print("nothing selected", file=sys.stderr)
            return 64
        if purpose == "plan":
            return cmd_plan(doc, names, a.data_root)
        if purpose == "upload":
            return cmd_upload_list(doc, names, a.data_root, a.wave)
        return cmd_build(doc, names, manifest=a.manifest, data_root=a.data_root, repo=a.repo,
                         split_path=a.split or a.repo / "splits/k15_split_v1.json",
                         extract_gt=a.extract_gt, replace_unregistered=a.replace_unregistered)
    except BundleError as e:
        print(f"{type(e).__name__.upper()}: {e}", file=sys.stderr)
        return e.code


if __name__ == "__main__":
    sys.exit(main())
