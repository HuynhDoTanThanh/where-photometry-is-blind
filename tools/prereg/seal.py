#!/usr/bin/env python3
"""The seal: move the KITTI-15 GT source under DATA_ROOT/sealed/, build the sealed held-out GT
bundle there, prove nothing held-out sits anywhere else, and stamp held-out result rows.

What it does
    plan      prints where the GT source zip is, what finalize would do, and the leak scan.
              Touches nothing.
    finalize  (--yes) the one-way step: checks the committed split and the source's recorded
              sha256, refuses while held-out GT sits outside gt/ or sealed/, moves
              src/data_scene_flow.zip to sealed/src/ (one rename, same volume), builds
              sealed/u2f-k15-gt-heldout-v1.zip with build_bundles' writer (allow_sealed), records
              its sha256 in data/manifest/bundles.v1.json and location=sealed in src/SOURCES.json.
              Idempotent: a second run verifies and exits 0.
    verify    [--deep] the seal holds: the source only under sealed/src/, the bundle's size (and
              with --deep its sha256 and member ids) as recorded, no held-out id under gt/, no
              sealed file in bundles/ or upload/, and (--deep) no KITTI-15 GT dir elsewhere.
    The deep scan (plan, finalize, verify --deep) also walks the extra scan roots: by default,
    with the real data root, /Volumes/Shin/U2FLOW_audit_private (the 2026-09-24 audit's
    quarantine) and the repo's output/ (scratch); --scan-root R (repeatable) replaces them.
    There a held-out GT dir, a copy of the source zip, or a per-pair result file
    (name matching 'per_pair') is a leak.
    Library (unseal.py, tools/eval/score_cached.py): heldout_members() yields held-out GT bytes
    in memory after an unseal of the batch is recorded; stamp_rows() writes the seal stamps.

How to invoke (Mac only; the hook's B6 rule lets only this program, unseal.py and
score_cached.py --g0-audit touch the seal)
    .venv/bin/python tools/prereg/seal.py plan
    .venv/bin/python tools/prereg/seal.py finalize --yes
    .venv/bin/python tools/prereg/seal.py verify [--deep]
    Common options: --data-root D (default /Volumes/Shin/U2FLOW_data), --repo R, --manifest M,
    --split S (default <repo>/splits/k15_split_v1.json), --scan-root R (repeatable).

Exit codes: 0 ok · 1 refused (no --yes, a malformed manifest) · 2 precondition (split not
    committed, source not recorded or absent, not sealed yet for verify) · 50 seal event, a
    STOP (AGENTS.md section 4): held-out GT outside the seal, two copies of the source, bytes
    that differ from the record, a sealed file in a public dir, or the devkit sample id held out
    · 64 usage.

Why it is like this
- AGENTS.md hard rule 4: the held-out GT lives only in U2FLOW_data/sealed on the Mac, never on a
  VM; no held-out number exists before prereg-v3 except the GATE-0 aggregate-only audit. The
  2026-09-24 audit extracted all 200 GT maps before any split existed (reports/PLAN.md section 4,
  2026-09-25); a seal that only moved the zip would leave such a copy behind, so finalize scans
  for KITTI-15 GT dirs by name and refuses while any held-out id sits outside gt/ or sealed/.
- The audit's own copy (all 200 GT maps and per-pair +FF rows) was moved OUT of the data root to
  /Volumes/Shin/U2FLOW_audit_private, where a data-root-only scan never looked: `finalize` and
  `verify --deep` reported a seal that did not hold (Tier 1 review, 2026-10-02). The deep scan
  therefore covers that quarantine and output/ too, and refuses (50) until the user moves the
  copy under sealed/ or deletes it.
- The sealed bundle is built with the same deterministic writer as every other bundle
  (build_bundles.write_zip), so its sha256 is reproducible from the source and the split, and
  note.py accepts it as a gt_pack_sha. Its per-file sha list stays inside sealed/, not in the
  committed manifest directory.
- devkit_scene_flow.zip carries sample GT of training id 000006 (data-bundles section 9). With
  the committed split v1 that id is dev [M splits/k15_split_v1.json]; if a future split held it
  out, sealing the devkit is not automated: finalize exits 50 for the user to decide.
- Held-out region masks and per-image values are never written to disk (metric regions.md,
  "Seal and caching"): heldout_members() yields bytes in memory, and only once unseal.py has
  recorded the batch.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("/Volumes/Shin/U2FLOW_data")
GT_ZIP = "data_scene_flow.zip"
DEVKIT_ZIP = "devkit_scene_flow.zip"
DEVKIT_SAMPLE_ID = "000006"
SEALED_BUNDLE = "u2f-k15-gt-heldout-v1"
RAW_GT_DIRS = ("flow_occ", "flow_noc", "disp_occ_0", "disp_occ_1", "disp_noc_0", "disp_noc_1",
               "obj_map")
# Outside the data root, scanned by default when --data-root is the real one (--scan-root
# replaces them): the audit's quarantine and the repo's scratch output/.
DEFAULT_EXTRA_ROOTS = (Path("/Volumes/Shin/U2FLOW_audit_private"), REPO / "output")
# A per-pair (per-image) result file under an extra root: an all-200 per-image number.
PER_PAIR_RX = re.compile(r"per[_-]?pair", re.I)
# Directories the leak scan never descends into: image trees (no GT), version control, the seal.
PRUNE = frozenset({".git", "image_02", "image_03", "image_2", "image_3", "multi_view",
                   "__pycache__", ".dl", ".bundles"})


class SealError(RuntimeError):
    code = 1


class Precondition(SealError):
    code = 2


class SealEvent(SealError):
    """A STOP: something held-out is where it must not be, or the bytes are not the record."""

    code = 50


def _bb():
    """tools/data/build_bundles.py: the one bundle writer and the manifest/SOURCES lock helpers."""
    spec = importlib.util.spec_from_file_location("u2f_build_bundles",
                                                  REPO / "tools" / "data" / "build_bundles.py")
    if spec is None or spec.loader is None:
        raise SealError("cannot load tools/data/build_bundles.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("u2f_build_bundles", mod)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------- layout
class Layout:
    def __init__(self, data_root: Path, repo: Path, manifest: Path, split: Path,
                 extra_roots: list[Path] | tuple[Path, ...] = ()) -> None:
        self.data_root, self.repo = Path(data_root), Path(repo)
        self.extra_roots = [Path(p) for p in extra_roots]
        self.manifest, self.split_path = Path(manifest), Path(split)
        self.src_zip = self.data_root / "src" / GT_ZIP
        self.sealed = self.data_root / "sealed"
        self.sealed_zip = self.sealed / "src" / GT_ZIP
        self.sources = self.data_root / "src" / "SOURCES.json"

    def bundle_spec(self, doc: dict) -> dict:
        spec = (doc.get("bundles") or {}).get(SEALED_BUNDLE)
        if not isinstance(spec, dict) or spec.get("sealed") is not True \
                or spec.get("uploadable") is not False:
            raise SealError(f"{self.manifest}: {SEALED_BUNDLE} must exist with sealed true and "
                            "uploadable false")
        f = str(spec.get("file") or "")
        if not f or "/" in f:
            raise SealError(f"{SEALED_BUNDLE}: file must be a plain name")
        return spec

    def bundle_path(self, doc: dict) -> Path:
        return self.sealed / self.bundle_spec(doc)["file"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def committed_split(lay: Layout) -> tuple[dict, str]:
    """The split, only if the working file equals its committed copy at HEAD."""
    try:
        raw = lay.split_path.read_bytes()
    except OSError:
        raise Precondition(f"{lay.split_path} is missing: the split comes first "
                           "(tools/prereg/make_split.py)") from None
    try:
        rel = lay.split_path.resolve().relative_to(lay.repo.resolve()).as_posix()
    except ValueError:
        raise Precondition(f"{lay.split_path} is outside the repo {lay.repo}") from None
    try:
        r = subprocess.run(["git", "-C", str(lay.repo), "show", f"HEAD:{rel}"],
                           capture_output=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        r = None
    if r is None or r.returncode != 0 or r.stdout != raw:
        raise Precondition(f"{rel} must be committed and unchanged before the seal")
    doc = json.loads(raw)
    dev, held = set(doc.get("dev_ids") or []), set(doc.get("heldout_ids") or [])
    if not dev or not held or dev & held:
        raise SealError(f"{rel}: dev and held-out ids must be non-empty and disjoint")
    return doc, hashlib.sha256(raw).hexdigest()


def source_record(lay: Layout) -> dict:
    try:
        rec = (json.loads(lay.sources.read_text()).get("sources") or {}).get(GT_ZIP)
    except (OSError, ValueError):
        rec = None
    if not isinstance(rec, dict) or not rec.get("sha256") or not rec.get("bytes"):
        raise Precondition(f"{GT_ZIP} is not recorded with sha256 and bytes in {lay.sources} "
                           "(tools/data/download_sources.sh --adopt)")
    return rec


def _ids_in(names: list[str]) -> set[str]:
    out = set()
    for n in names:
        stem = n.split("_", 1)[0].split(".", 1)[0]
        if len(stem) == 6 and stem.isdigit():
            out.add(stem)
    return out


def scan(root: Path, skip: set[Path], per_pair: list[Path] | None = None
         ) -> tuple[list[tuple[Path, set[str]]], list[Path]]:
    """One walk of ROOT (not under SKIP): every <...>/training/<GT dir> with the ids it holds,
    and every file named like the GT source zip. Image trees are pruned (no GT in them). With
    PER_PAIR (a list), file names matching PER_PAIR_RX are appended to it (names only)."""
    found, zips = [], []
    skip_r = {p.resolve() for p in skip}
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        if here.resolve() in skip_r:
            dirnames[:] = []
            continue
        dirnames[:] = sorted(d for d in dirnames if d not in PRUNE)
        zips += [here / f for f in filenames if f == GT_ZIP]
        if per_pair is not None:
            per_pair += [here / f for f in filenames if PER_PAIR_RX.search(f)]
        if here.name in RAW_GT_DIRS and here.parent.name == "training":
            if not _is_kitti12(here.parent):
                found.append((here, _ids_in(filenames)))
            dirnames[:] = []
    return found, zips


def _is_kitti12(training: Path) -> bool:
    """KITTI-12 reuses six-digit ids (000000-000193) and the names flow_occ/flow_noc, so its GT
    tree must not read as KITTI-15. KITTI-12 has disp_occ/colored_0 and no obj_map or
    disp_occ_0; a tree that shows neither signature counts as KITTI-15 (fail closed) unless
    its path names KITTI-12."""
    try:
        sib = {p.name for p in training.iterdir() if p.is_dir()}
    except OSError:
        sib = set()
    if sib & {"obj_map", "disp_occ_0", "disp_noc_0", "disp_occ_1", "disp_noc_1"}:
        return False
    if sib & {"disp_occ", "disp_noc", "colored_0", "colored_1"}:
        return True
    low = training.as_posix().lower()
    return "kitti12" in low or "stereo_flow" in low


def scan_gt_dirs(root: Path, skip: set[Path]) -> list[tuple[Path, set[str]]]:
    return scan(root, skip)[0]


def leaks(lay: Layout, doc: dict, heldout: set[str], deep: bool) -> list[str]:
    """Every place the seal does not hold, as text. Never opens a GT file: names only."""
    out = []
    gt = lay.data_root / "gt"
    if gt.is_dir():
        for d, ids in scan_gt_dirs(gt, set()):
            bad = sorted(ids & heldout)
            if bad:
                out.append(f"{d} holds {len(bad)} held-out id(s) (gt/ is dev and KITTI-12 only)")
    for pub in ("bundles", "upload"):
        base = lay.data_root / pub
        if base.is_dir():
            name = lay.bundle_spec(doc)["file"]
            out += [f"{p} is the sealed bundle in a public dir" for p in base.rglob(name)]
            out += [f"{p} is the GT source in a public dir" for p in base.rglob(GT_ZIP)]
    if deep:
        dirs, zips = scan(lay.data_root, {lay.sealed, gt})
        for d, ids in dirs:
            bad = sorted(ids & heldout)
            if bad:
                out.append(f"{d} holds {len(bad)} held-out GT file id(s) outside gt/ and sealed/")
        for p in zips:
            if p.resolve() != lay.src_zip.resolve():
                out.append(f"{p} is another copy of the GT source outside the seal")
        for extra in lay.extra_roots:
            if not extra.is_dir():
                continue
            pp: list[Path] = []
            dirs, zips = scan(extra, {lay.sealed}, pp)
            for d, ids in dirs:
                bad = sorted(ids & heldout)
                if bad:
                    out.append(f"{d} holds {len(bad)} held-out GT file id(s) outside the data "
                               "root (an extra scan root)")
            out += [f"{p} is a copy of the GT source outside the seal" for p in zips]
            out += [f"{p} is a per-pair result file (all-200 per-image numbers) outside the "
                    "seal" for p in pp]
    return out


def where_is_source(lay: Layout) -> str:
    a, b = lay.src_zip.is_file(), lay.sealed_zip.is_file()
    return "both" if a and b else "src" if a else "sealed" if b else "missing"


# ----------------------------------------------------------------------------- commands
def _load(lay: Layout) -> dict:
    try:
        return json.loads(lay.manifest.read_text())
    except (OSError, ValueError) as e:
        raise SealError(f"{lay.manifest}: {e}") from None


def cmd_plan(lay: Layout) -> int:
    doc = _load(lay)
    lay.bundle_spec(doc)
    split, split_sha = committed_split(lay)
    held = set(split["heldout_ids"])
    rec = source_record(lay)
    loc = where_is_source(lay)
    print(f"split: {len(split['dev_ids'])} dev / {len(held)} held-out, sha {split_sha[:12]}")
    print(f"source {GT_ZIP}: {loc}; recorded {rec['bytes']} B, sha {rec['sha256'][:12]}")
    print(f"devkit sample id {DEVKIT_SAMPLE_ID}: "
          f"{'HELD-OUT (finalize exits 50)' if DEVKIT_SAMPLE_ID in held else 'dev (no action)'}")
    bp = lay.bundle_path(doc)
    print(f"sealed bundle: {bp} {'exists' if bp.is_file() else 'not built'}")
    found = leaks(lay, doc, held, deep=True)
    for f in found:
        print(f"LEAK {f}")
    print("finalize would: " + ("verify only (already sealed)" if loc == "sealed" and
                                bp.is_file() else f"move {lay.src_zip} -> {lay.sealed_zip}, "
                                f"build {bp.name}, record it"))
    return SealEvent.code if found or loc == "both" else 0


def _check_member_ids(path: Path, held: set[str], dev: set[str]) -> int:
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.startswith("kitti15/")]
    ids = _ids_in([n.rsplit("/", 1)[-1] for n in names])
    if ids & dev or not ids <= held:
        raise SealEvent(f"{path.name}: member ids are not exactly held-out ids ("
                        f"{len(ids & dev)} dev id(s) inside)")
    return len(names)


def cmd_finalize(lay: Layout, yes: bool) -> int:
    if not yes:
        raise SealError("finalize moves the GT source for good: re-run with --yes "
                        "(`seal.py plan` shows what it does)")
    bb = _bb()
    doc = _load(lay)
    spec = lay.bundle_spec(doc)
    split, split_sha = committed_split(lay)
    held, dev = set(split["heldout_ids"]), set(split["dev_ids"])
    if DEVKIT_SAMPLE_ID in held:
        raise SealEvent(f"{DEVKIT_ZIP} holds sample GT of {DEVKIT_SAMPLE_ID}, which this split "
                        "holds out; sealing the devkit is the user's decision (data-bundles "
                        "section 9), not automated")
    rec = source_record(lay)
    loc = where_is_source(lay)
    if loc == "both":
        raise SealEvent(f"two copies of {GT_ZIP} ({lay.src_zip}, {lay.sealed_zip}): report it")
    if loc == "missing":
        raise Precondition(f"{GT_ZIP} is neither at {lay.src_zip} nor at {lay.sealed_zip}")
    found = leaks(lay, doc, held, deep=True)
    if found:
        raise SealEvent("the seal would not hold: " + "; ".join(found[:5]))
    cur = lay.src_zip if loc == "src" else lay.sealed_zip
    if cur.stat().st_size != int(rec["bytes"]):
        raise SealEvent(f"{cur} has {cur.stat().st_size} B, the record says {rec['bytes']}")
    sha = sha256_file(cur)
    if sha != rec["sha256"]:
        raise SealEvent(f"{cur} sha256 {sha[:12]} differs from the record {rec['sha256'][:12]}")
    if loc == "src":
        lay.sealed.mkdir(mode=0o700, exist_ok=True)
        (lay.sealed / "src").mkdir(mode=0o700, exist_ok=True)
        try:
            os.rename(lay.src_zip, lay.sealed_zip)
        except OSError as e:
            raise SealError(f"cannot move {GT_ZIP} into the seal by rename ({e}); both must be "
                            "on one volume, and nothing was copied") from None
        print(f"MOVED {lay.src_zip} -> {lay.sealed_zip}")

    def mark(d: dict) -> None:
        r = (d.setdefault("sources", {})).setdefault(GT_ZIP, {})
        r.update(location="sealed", dest=f"sealed/src/{GT_ZIP}",
                 sealed_by="tools/prereg/seal.py")
        r.setdefault("sealed_utc", bb.utc_now())

    bb.update_json(lay.sources, mark)
    out = lay.bundle_path(doc)
    if out.is_file() and spec.get("sha256") and sha256_file(out) == spec["sha256"]:
        print(f"SEALED_BUNDLE_OK {out.name} (already built, sha {spec['sha256'][:12]})")
    else:
        src = bb.LocalZipSource(lay.sealed_zip)
        res = bb.build_bundle(doc, SEALED_BUNDLE, sources={GT_ZIP: src}, split=split,
                              split_sha=split_sha, source_shas={GT_ZIP: sha}, out_path=out,
                              repo_root=lay.repo, allow_sealed=True)
        lst = lay.sealed / f"{SEALED_BUNDLE}.files.sha256"
        lst.write_text("".join(f"{s}  {n}\n" for n, _, s in res["files"]))
        bb.update_manifest(lay.manifest, "bundles", SEALED_BUNDLE, {
            "bytes": res["bytes"], "sha256": res["sha256"], "n_files": res["n_files"],
            "n_payload": res["n_payload"], "payload_bytes": res["payload_bytes"],
            "source_shas": {GT_ZIP: sha}, "split_sha256": split_sha,
            "builder_git_sha": bb.git_sha(lay.repo), "built_utc": bb.utc_now(),
            "path": f"sealed/{out.name}"})
        print(f"BUILT {out} ({res['n_payload']} files, sha {res['sha256'][:12]})")
    n = _check_member_ids(out, held, dev)
    bb.queue_evidence("data-integrity", "seal", "measured", "tools/prereg/seal.py finalize",
                      {"bundle": SEALED_BUNDLE, "n_members": n, "split_sha256": split_sha,
                       "source_sha256": sha, "location": "sealed"})
    print(f"SEAL_OK {SEALED_BUNDLE}: {n} held-out GT members; source under sealed/src/")
    return 0


def cmd_verify(lay: Layout, deep: bool) -> int:
    doc = _load(lay)
    spec = lay.bundle_spec(doc)
    split, split_sha = committed_split(lay)
    held, dev = set(split["heldout_ids"]), set(split["dev_ids"])
    loc = where_is_source(lay)
    if loc == "both":
        raise SealEvent(f"two copies of {GT_ZIP}")
    if loc != "sealed":
        raise Precondition(f"not sealed yet: {GT_ZIP} is {loc} (seal.py finalize --yes)")
    out = lay.bundle_path(doc)
    if not out.is_file() or not spec.get("sha256"):
        raise Precondition(f"{out.name} is not built or not recorded (seal.py finalize --yes)")
    if out.stat().st_size != spec.get("bytes"):
        raise SealEvent(f"{out} has {out.stat().st_size} B, the manifest says {spec.get('bytes')}")
    if spec.get("split_sha256") not in (None, split_sha):
        raise SealEvent(f"{out.name} was built from another split ({spec['split_sha256'][:12]})")
    found = leaks(lay, doc, held, deep=deep)
    if found:
        raise SealEvent("; ".join(found[:5]))
    if deep:
        if sha256_file(out) != spec["sha256"]:
            raise SealEvent(f"{out} sha256 differs from the manifest")
        _check_member_ids(out, held, dev)
    print(f"SEAL_VERIFIED {'deep' if deep else 'shallow'}: source sealed, bundle "
          f"{spec['sha256'][:12]}, no leak found")
    return 0


# ----------------------------------------------------------------------------- library
def unsealed_batches(unseal_rows: list[dict]) -> set[str]:
    return {str(r.get("batch")) for r in unseal_rows if r.get("batch")}


def heldout_members(lay: Layout, batch: str, unseal_rows: list[dict],
                    want: tuple[str, ...] = ("flow_occ", "flow_noc", "obj_map")
                    ) -> Iterator[tuple[str, bytes]]:
    """(arcname, bytes) of held-out GT, in memory, only after unseal.py recorded BATCH. Nothing
    is written to disk; the caller decodes and scores (score_cached), and never caches it."""
    if batch not in unsealed_batches(unseal_rows):
        raise SealError(f"batch {batch!r} is not unsealed (experiments/unseal.jsonl): "
                        "tools/prereg/unseal.py first, a human step")
    doc = _load(lay)
    spec = lay.bundle_spec(doc)
    out = lay.bundle_path(doc)
    if sha256_file(out) != spec.get("sha256"):
        raise SealEvent(f"{out} sha256 differs from the manifest: the seal does not hold")
    with zipfile.ZipFile(out) as z:
        for n in sorted(z.namelist()):
            parts = n.split("/")
            if len(parts) >= 3 and parts[0] == "kitti15" and parts[-2] in want:
                yield n, z.read(n)


def stamp_rows(rows: list[dict], primary: dict, batch: str,
               unseal_rows: list[dict]) -> list[dict]:
    """The seal stamps on held-out rows (metric skill section 6): confirmatory when the row is
    one PRIMARY.json declares in BATCH (a confirmatory batch), else exploratory post-unseal."""
    if batch not in unsealed_batches(unseal_rows):
        raise SealError(f"batch {batch!r} is not unsealed")
    spec = ((primary.get("unseal_batches") or {}).get(batch)) or {}
    declared = spec.get("rows") or []
    conf = spec.get("confirmatory") is True
    keys = ("arm", "metric", "region", "agg", "step")
    out = []
    for r in rows:
        if r.get("split") != "heldout":
            raise SealError(f"stamp_rows: {r.get('split')} row; only held-out rows are stamped")
        arm = str(r.get("run") or "").split("-")[2] if r.get("run") else r.get("arm")
        probe = {"arm": arm, **{k: r.get(k) for k in keys if k != "arm"}}
        hit = conf and any(all(d.get(k) in (None, probe.get(k)) for k in keys) for d in declared)
        out.append({**r, "unseal_batch": batch, "confirmatory": bool(hit),
                    "post_unseal": not hit})
    return out


# ----------------------------------------------------------------------------- CLI
class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # type: ignore[override]
        self.print_usage(sys.stderr)
        self.exit(64, f"{self.prog}: error: {message}\n")


def main(argv: list[str] | None = None) -> int:
    p = Parser(prog="seal.py", description=__doc__.splitlines()[0])
    p.add_argument("verb", choices=("plan", "finalize", "verify"))
    p.add_argument("--yes", action="store_true")
    p.add_argument("--deep", action="store_true")
    p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    p.add_argument("--repo", type=Path, default=REPO)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--split", type=Path, default=None)
    p.add_argument("--scan-root", type=Path, action="append", default=None,
                   help="an extra root for the deep leak scan (repeatable; replaces the defaults)")
    try:
        a = p.parse_args(argv)
    except SystemExit as e:
        return int(e.code or 0)
    if a.scan_root is not None:
        extra = list(a.scan_root)
    elif a.data_root.resolve() == DATA_ROOT.resolve():
        extra = list(DEFAULT_EXTRA_ROOTS)
    else:
        extra = []
    lay = Layout(a.data_root, a.repo, a.manifest or a.repo / "data/manifest/bundles.v1.json",
                 a.split or a.repo / "splits/k15_split_v1.json", extra)
    try:
        if a.verb == "plan":
            return cmd_plan(lay)
        if a.verb == "finalize":
            return cmd_finalize(lay, a.yes)
        return cmd_verify(lay, a.deep)
    except SealError as e:
        tag = "SEAL_STOP" if e.code == 50 else "PRECONDITION" if e.code == 2 else "REFUSED"
        print(f"{tag}: {e}", file=sys.stderr)
        return e.code
    except Exception as e:
        code: Any = getattr(e, "code", None)  # build_bundles' BundleError carries its exit code
        if isinstance(code, int) and type(e).__name__.endswith(("Refusal", "Error",
                                                                "Precondition")):
            print(f"{type(e).__name__.upper()}: {e}", file=sys.stderr)
            return code
        raise


if __name__ == "__main__":
    sys.exit(main())
