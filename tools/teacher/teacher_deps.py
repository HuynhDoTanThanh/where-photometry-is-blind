"""Teacher dependencies and weights on a VM: hashed locks into a private pydeps dir, pinned HF
weights into U2_DATA/weights/hf, url weights verified (and unzipped) where fetch.py left them.

What it does (a library for tools/teacher/run_geometry.py; also a small CLI).
    parse_lock(path)      env/teachers/<name>.txt -> items (u2f-teacher-lock 1, see its header)
    install(lock, data)   download each item (resumable .part, 3 attempts), check sha256 and
                          bytes, unpack into DATA/pydeps/teacher-<name>-<sha12 of the lock>/
                          (wheels unzipped; archives: kept members, root stripped) or, for
                          target=torch_hub/DIR, into DATA/torch_home/hub/DIR; INSTALL.json last.
                          Returns the sys.path entries the worker must put first.
    fetch_hf(spec, data)  every pinned file of a registry entry's weights at its revision into
                          DATA/weights/hf/<repo>/<revision>/<file>, sha256-checked (exit 3 on a
                          difference); the base URL is https://huggingface.co unless
                          U2_TEACHER_HF_BASE says otherwise (tests use file://).
    url_weight(key, data) a weights.json url/gdown item fetch.py already put at DATA/<file>:
                          sha256-checked, its `extract` member or tree unpacked once.
    freeze_sha()          `pip freeze --all`, sorted, as tools/vm/bootstrap.py:217-228 computes
                          lock_G0's pip_freeze_sha256 (the job checks it is unchanged).
CLI (Mac or VM, python >= 3.9, standard library only):
    python tools/teacher/teacher_deps.py check            # every lock parses; names, hashes
    python tools/teacher/teacher_deps.py install --lock env/teachers/pi3.txt --data D
Exit codes: 0 ok · 2 precondition (a download failed, a lock line is malformed, a file is
missing) · 3 refused (sha256 or size differs from the pin, an item would shadow an image package,
an unsafe member path) · 64 usage.

Why it is built this way.
- The image is frozen: lock_G0's pip_freeze_sha256 is part of every comparison's protocol key
  (codebase section 7). sam_masks.py set the pattern: unpack pinned code under U2_DATA/pydeps
  and put it first on the job's own sys.path; never pip into site-packages [V tools/vm/
  sam_masks.py:535-598]. pi3's requirements.txt pins torch 2.5.1 / numpy 1.26.4 and
  MapAnything's pins opencv-python-headless 4.10 [V both repos, 2026-10-03]: a resolver would
  replace the image's sm_120 torch. So nothing here resolves, builds or runs pip install.
- An item that ships a top-level torch, torchvision, numpy, cv2, PIL, scipy, safetensors or
  huggingface_hub is refused (exit 3): it would shadow the image's copy from the front of
  sys.path, which is the same drift by another route.
- Content-addressed install dirs (sha12 of the lock file): a changed lock never reuses an old
  tree, and an unchanged one is reused (re-install under 10 minutes, plan W2-04 done_when).
- HF weights are fetched by revision with plain HTTPS, not huggingface_hub's cache: one file per
  pinned path, checked against weights.json, so a moved 'main' or a cache symlink can never hand
  the job other bytes (fetch.py leaves hf items to teacher-factory [V tools/data/fetch.py:971]).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LOCK_DIR = REPO / "env" / "teachers"
WEIGHTS_JSON = REPO / "data" / "manifest" / "weights.json"
HF_BASE_DEFAULT = "https://huggingface.co"
PROTECTED = frozenset({"torch", "torchvision", "torchaudio", "numpy", "cv2", "PIL", "scipy",
                       "safetensors", "huggingface_hub", "triton", "nvidia"})
LINE_RX = re.compile(r"^(wheel|archive)\s+(\S+)\s+(\S+)\s+(\S+)((?:\s+[a-z0-9_]+=\S+)+)\s*$")
SHA_RX = re.compile(r"^[0-9a-f]{64}$")
EXIT_OK, EXIT_PRE, EXIT_REFUSED, EXIT_USAGE = 0, 2, 3, 64


class DepsError(Exception):
    def __init__(self, code: int, msg: str):
        super().__init__(msg)
        self.code, self.msg = code, msg


def say(tag: str, msg: str = "") -> None:
    print(f"{tag} {msg}".rstrip(), flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def utc_now() -> str:
    import datetime as dt

    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()  # noqa: UP017


# ------------------------------------------------------------------------------- locks
def parse_lock(path: Path) -> dict:
    """{'name', 'path', 'sha256', 'items': [...]}; DepsError(2) on any malformed line."""
    text = Path(path).read_text()
    items = []
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = LINE_RX.match(line)
        if not m:
            raise DepsError(EXIT_PRE, f"{path}:{n}: not a u2f-teacher-lock line")
        kind, name, ref, url = m.group(1), m.group(2), m.group(3), m.group(4)
        kv = dict(x.split("=", 1) for x in m.group(5).split())
        unknown = set(kv) - {"sha256", "bytes", "root", "keep", "target", "path"}
        if unknown:
            raise DepsError(EXIT_PRE, f"{path}:{n}: unknown key(s) {sorted(unknown)}")
        if not SHA_RX.match(kv.get("sha256", "")) or not kv.get("bytes", "").isdigit():
            raise DepsError(EXIT_PRE, f"{path}:{n}: sha256=<64 hex> and bytes=<int> are required")
        if not url.startswith("https://"):
            raise DepsError(EXIT_PRE, f"{path}:{n}: only https origins are pinned")
        it = {"kind": kind, "name": name, "ref": ref, "url": url, "sha256": kv["sha256"],
              "bytes": int(kv["bytes"]), "target": kv.get("target"), "path": kv.get("path")}
        if kind == "archive":
            if not kv.get("root", "").endswith("/") or not kv.get("keep"):
                raise DepsError(EXIT_PRE, f"{path}:{n}: an archive needs root=PREFIX/ and keep=")
            it["root"], it["keep"] = kv["root"], kv["keep"].split(",")
        if it["target"] and not re.match(r"^torch_hub/[A-Za-z0-9_.-]+$", it["target"]):
            raise DepsError(EXIT_PRE, f"{path}:{n}: target must be torch_hub/<dir>")
        items.append(it)
    if not items:
        raise DepsError(EXIT_PRE, f"{path}: no items")
    return {"name": Path(path).stem, "path": str(path), "items": items,
            "sha256": hashlib.sha256(text.encode()).hexdigest()}


def all_locks() -> list[dict]:
    return [parse_lock(p) for p in sorted(LOCK_DIR.glob("*.txt"))]


# ------------------------------------------------------------------------------- download
def _base_url_override(url: str) -> str:
    """U2_TEACHER_URL_MAP='https://a/=file:///x/' rewrites origins (tests and the Mac only)."""
    rule = os.environ.get("U2_TEACHER_URL_MAP", "")
    if rule and "=" in rule:
        src, dst = rule.split("=", 1)
        if url.startswith(src):
            return dst + url[len(src):]
    return url


def download(url: str, dest: Path, sha256: str, size: int | None = None,
             attempts: int = 3) -> dict:
    """URL -> DEST via DEST.part (resumed with Range when the server allows it); sha256 and size
    checked before the rename. An existing DEST with the right sha256 is kept."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and sha256_file(dest) == sha256:
        return {"path": str(dest), "cached": True, "bytes": dest.stat().st_size}
    part = dest.with_name(dest.name + ".part")
    real = _base_url_override(url)
    t0, err = time.time(), None
    for attempt in range(1, attempts + 1):
        have = part.stat().st_size if part.is_file() else 0
        req = urllib.request.Request(real)
        if have and real.startswith("http"):
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                status = getattr(r, "status", 200)
                mode = "ab" if have and status == 206 else "wb"
                with open(part, mode) as f:
                    shutil.copyfileobj(r, f, 1 << 20)
            err = None
            break
        except OSError as e:  # URLError is an OSError
            err = e
            say("TEACHER_DL", f"attempt {attempt} of {real} failed: {e}")
            time.sleep(min(30, 5 * attempt))
    if err is not None:
        raise DepsError(EXIT_PRE, f"could not download {real}: {err}")
    got_size = part.stat().st_size
    if size is not None and got_size != size:
        part.unlink(missing_ok=True)
        raise DepsError(EXIT_REFUSED, f"{dest.name}: {got_size} bytes != {size} pinned")
    got = sha256_file(part)
    if got != sha256:
        part.unlink(missing_ok=True)
        raise DepsError(EXIT_REFUSED, f"{dest.name}: sha256 {got} != {sha256} pinned")
    os.replace(part, dest)
    secs = max(time.time() - t0, 1e-6)
    return {"path": str(dest), "cached": False, "bytes": got_size,
            "MBps": round(got_size / secs / 1e6, 2)}


# ------------------------------------------------------------------------------- unpack
def _safe(parts: list[str]) -> bool:
    return bool(parts) and all(p not in ("", ".", "..") for p in parts) and not parts[0] == ""


def _kept(rel: str, keep: list[str]) -> bool:
    if keep == ["*"]:
        return True
    return any(rel == k or (k.endswith("/") and rel.startswith(k)) for k in keep)


def _members(arc: Path):
    """(relative name, is_file, reader) for a .tar.gz or .zip archive or a wheel."""
    if zipfile.is_zipfile(arc):
        z = zipfile.ZipFile(arc)
        for i in z.infolist():
            yield i.filename, not i.is_dir(), (lambda i=i: z.read(i))
        z.close()
        return
    with tarfile.open(arc) as t:
        for m in t.getmembers():
            if m.issym() or m.islnk():
                continue
            f = t.extractfile(m) if m.isfile() else None
            yield m.name, m.isfile(), (lambda f=f: f.read() if f else b"")


def unpack(item: dict, arc: Path, dest: Path) -> list[str]:
    """Kept members of ARC into DEST; returns the top-level names written. Refuses unsafe paths
    and anything that would shadow an image package (exit 3)."""
    tops: set[str] = set()
    n = 0
    for name, is_file, read in _members(arc):
        if not is_file:
            continue
        rel = name
        if item["kind"] == "archive":
            if not rel.startswith(item["root"]):
                continue
            rel = rel[len(item["root"]):]
            if not _kept(rel, item["keep"]):
                continue
        elif ".data/" in rel.split("/", 1)[0] + "/":  # wheel scripts/headers: never needed
            continue
        parts = rel.split("/")
        if not _safe(parts):
            raise DepsError(EXIT_REFUSED, f"{arc.name}: unsafe member path {name!r}")
        top = parts[0].removesuffix(".py")
        if top in PROTECTED and not item.get("target"):
            raise DepsError(EXIT_REFUSED, f"{arc.name} ships a top-level {top!r}: it would shadow "
                                          "the image's copy from the front of sys.path")
        out = dest.joinpath(*parts)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(read())
        tops.add(parts[0])
        n += 1
    if n == 0:
        raise DepsError(EXIT_PRE, f"{arc.name}: nothing matched keep={item.get('keep')}")
    return sorted(tops)


def pydeps_dir(lock: dict, data: Path) -> Path:
    return data / "pydeps" / f"teacher-{lock['name']}-{lock['sha256'][:12]}"


def install(lock: dict, data: Path) -> dict:
    """Install every item of LOCK (idempotent). Returns {'dir', 'sys_path', 'torch_home', ...}."""
    target = pydeps_dir(lock, data)
    stamp = target / "INSTALL.json"
    torch_home = data / "torch_home"
    try:
        rec = json.loads(stamp.read_text())
    except (OSError, ValueError):
        rec = {}
    hub_ok = all((torch_home / "hub" / it["target"].split("/", 1)[1] / ".u2f_sha256").is_file()
                 and (torch_home / "hub" / it["target"].split("/", 1)[1] / ".u2f_sha256")
                 .read_text().strip() == it["sha256"]
                 for it in lock["items"] if it["target"])
    if rec.get("lock_sha256") != lock["sha256"] or not hub_ok:
        dl = data / "pydeps" / "_dl"
        stage = target.with_name(target.name + ".tmp")
        shutil.rmtree(stage, ignore_errors=True)
        stage.mkdir(parents=True)
        done = []
        for it in lock["items"]:
            fname = f"{it['name']}-{it['sha256'][:12]}" + (".whl" if it["kind"] == "wheel"
                                                           else ".arc")
            got = download(it["url"], dl / fname, it["sha256"], it["bytes"])
            if it["target"]:
                hub = torch_home / "hub" / it["target"].split("/", 1)[1]
                tmp = hub.with_name(hub.name + ".tmp")
                shutil.rmtree(tmp, ignore_errors=True)
                tmp.mkdir(parents=True)
                tops = unpack(it, Path(got["path"]), tmp)
                (tmp / ".u2f_sha256").write_text(it["sha256"] + "\n")
                shutil.rmtree(hub, ignore_errors=True)
                os.replace(tmp, hub)
            else:
                tops = unpack(it, Path(got["path"]), stage)
            done.append({"name": it["name"], "ref": it["ref"], "sha256": it["sha256"],
                         "tops": tops, "target": it["target"], "download": got})
        rec = {"lock": lock["path"], "lock_sha256": lock["sha256"], "items": done,
               "utc": utc_now(), "python": sys.version.split()[0]}
        (stage / "INSTALL.json").write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
        shutil.rmtree(target, ignore_errors=True)
        os.replace(stage, target)
    extra = [str(target / it["path"]) for it in lock["items"] if it.get("path")]
    say("TEACHER_DEPS", f"{lock['name']} {lock['sha256'][:12]} at {target}")
    return {"dir": str(target), "sys_path": [str(target), *extra], "torch_home": str(torch_home),
            "lock": lock["path"], "lock_sha256": lock["sha256"], "items": rec.get("items", [])}


# ------------------------------------------------------------------------------- weights
def weights_doc() -> dict:
    return json.loads(WEIGHTS_JSON.read_text())["weights"]


def fetch_hf(wspec: dict, data: Path) -> dict:
    """Every file of a registry `weights` block with origin hf, at its revision, checked against
    the block's pinned sha256 (the same values as weights.json)."""
    repo, rev = wspec.get("repo"), wspec.get("revision")
    if not (repo and rev and re.fullmatch(r"[0-9a-f]{40}", rev)):
        raise DepsError(EXIT_PRE, f"weights {wspec.get('key')}: no pinned HF repo and revision")
    base = os.environ.get("U2_TEACHER_HF_BASE", HF_BASE_DEFAULT).rstrip("/")
    out_dir = data / "weights" / "hf" / repo.replace("/", "__") / rev
    files = {}
    for fname, sha in sorted((wspec.get("files") or {}).items()):
        url = f"{base}/{repo}/resolve/{rev}/{fname}"
        files[fname] = download(url, out_dir / fname, sha, None)
        say("TEACHER_WEIGHT", f"{repo}@{rev[:10]} {fname} sha256 ok")
    return {"dir": str(out_dir), "repo": repo, "revision": rev, "files": files}


def url_weight(key: str, data: Path, wdoc: dict | None = None) -> dict:
    """A gdown/url weights.json item that fetch.py fetched to DATA/<file>: sha256-checked, and its
    `extract` member (or whole tree, `all_to`) unpacked once, the member checked when pinned."""
    w = (wdoc or weights_doc()).get(key)
    if not w:
        raise DepsError(EXIT_PRE, f"weights.json has no {key!r}")
    src = data / w["file"]
    if not src.is_file():
        raise DepsError(EXIT_PRE, f"{src} missing: the job's profile must fetch weight {key!r}")
    pin = w.get("sha256")
    got = sha256_file(src)
    if pin and got != pin:
        raise DepsError(EXIT_REFUSED, f"{src.name} sha256 {got} != {pin} (weights.json {key})")
    ex = w.get("extract") or {}
    rec = {"key": key, "file": str(src), "sha256": got, "pinned": bool(pin)}
    if ex.get("member"):
        dst = data / ex["to"]
        if not (dst.is_file() and (not ex.get("sha256") or sha256_file(dst) == ex["sha256"])):
            with zipfile.ZipFile(src) as z:
                blob = z.read(ex["member"])
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name("." + dst.name + ".tmp")
            tmp.write_bytes(blob)
            os.replace(tmp, dst)
        if ex.get("sha256") and sha256_file(dst) != ex["sha256"]:
            raise DepsError(EXIT_REFUSED, f"{dst.name} differs from the pinned member sha256")
        rec["extracted"] = str(dst)
    elif ex.get("all_to"):
        dst = data / ex["all_to"]
        marker = dst / ".u2f_sha256"
        if not (marker.is_file() and marker.read_text().strip() == got):
            dst.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(src) as z:
                for i in z.infolist():
                    parts = i.filename.split("/")
                    if i.is_dir() or not _safe(parts):
                        continue
                    (dst / parts[-1]).write_bytes(z.read(i))  # flat zips (Monodepth2)
            marker.write_text(got + "\n")
        rec["extracted"] = str(dst)
    return rec


# ------------------------------------------------------------------------------- image
def freeze_lines() -> list[str] | None:
    """`pip freeze --all` sorted, stripped, WARNING lines dropped: bootstrap.py:217-224's rule."""
    try:
        r = subprocess.run([sys.executable, "-m", "pip", "freeze", "--all",
                            "--disable-pip-version-check"], capture_output=True, text=True,
                           timeout=180)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    return sorted(ln.strip() for ln in r.stdout.splitlines()
                  if ln.strip() and not ln.startswith("WARNING"))


def freeze_sha(lines: list[str] | None = None) -> str:
    lines = freeze_lines() if lines is None else lines
    return "" if lines is None else hashlib.sha256("\n".join(lines).encode()).hexdigest()


# ------------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="teacher_deps.py", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    i = sub.add_parser("install")
    i.add_argument("--lock", required=True)
    i.add_argument("--data", default=os.environ.get("U2_DATA", "/content/data"))
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return EXIT_USAGE if e.code else EXIT_OK
    try:
        if a.cmd == "check":
            for lk in all_locks():
                say("TEACHER_LOCK", f"{lk['name']} sha12={lk['sha256'][:12]} items="
                                    + ",".join(f"{it['name']}@{it['ref'][:10]}"
                                               for it in lk["items"]))
            return EXIT_OK
        rec = install(parse_lock(Path(a.lock)), Path(a.data).resolve())
        say("TEACHER_DEPS_OK", json.dumps({"dir": rec["dir"], "sys_path": rec["sys_path"]}))
        return EXIT_OK
    except DepsError as e:
        say("TEACHER_REFUSED" if e.code == EXIT_REFUSED else "TEACHER_ERROR", e.msg)
        return e.code


if __name__ == "__main__":
    sys.exit(main())
