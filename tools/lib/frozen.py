"""What the pre-registration has frozen, read the same way by every tool that needs it.

What it does
    aggregate_allowlist(root)   the checkpoints whose all-200 aggregate may exist (hard rule 4's
                                GATE-0 exception): the released kitti.pth always, plus the
                                ckpt_sha256 list of prereg/inputs/aggregate_only_allowlist.json AS
                                FROZEN IN prereg-v1. Without that tag, kitti.pth only.
    gate0_state(root)           which prereg/GATE-0.json is in force: its sha256 and where it
                                comes from (tag:prereg-v1, HEAD, staged, none) and whether it is
                                committed. eval_kitti/score_cached stamp the sha into metrics.json,
                                score.json, the rows and the parity evidence; gate.py refuses a GO
                                when a row it uses was produced under another GATE-0.json.
    frozen_bytes(root, rel)     a file as prereg-v1 holds it, else as HEAD holds it, else as staged.

How to invoke: a library (tools/note.py, tools/eval/{eval_kitti,score_cached}.py,
tools/prereg/gate.py). Python >= 3.9, standard library only (eval_kitti also runs on a VM).
    python tools/lib/frozen.py [--root R]     # prints the allowlist and the GATE-0 state as JSON

Raises FrozenError when the working tree contradicts the frozen copy (an allowlist that differs
from prereg-v1's, or one created after the tag; a weights.json whose u2flow-kitti sha differs
from the frozen one). Callers turn it into their own refusal (exit 1 or 3).

Why (Tier 1 review, 2026-10-02)
- The allowlist decides which checkpoints may get an all-200 number, and all-200 minus dev is the
  held-out value. Read from the working tree, any file created after prereg-v1 (B10 freezes only
  files a tag holds) could add one of our own checkpoints. So it is read from the tag, the working
  tree must agree, and before the tag only the released kitti.pth counts. The kitti.pth sha comes
  from the frozen (tag or HEAD) weights.json, so editing that file in the working tree changes
  nothing.
- Reproduction numbers were gated on GATE-0.json being committed at HEAD only: commit, measure,
  look, edit the tolerances, recommit before the tag, and nothing flagged it. Stamping the sha of
  the GATE-0.json in force into every number lets gate.py see that the criteria changed after the
  number existed.
- One helper, so note.py, eval_kitti, score_cached and gate.py cannot drift apart (there were
  three copies of the allowlist rule).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

V1_TAG = "prereg-v1"
ALLOWLIST_REL = "prereg/inputs/aggregate_only_allowlist.json"
GATE0_REL = "prereg/GATE-0.json"
WEIGHTS_REL = "data/manifest/weights.json"


class FrozenError(RuntimeError):
    """The working tree contradicts what prereg froze."""


def _git(root: Path, *args: str) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None


def has_git(root: Path) -> bool:
    return (Path(root) / ".git").exists()


def tag_exists(root: Path, tag: str = V1_TAG) -> bool:
    r = _git(root, "rev-parse", "-q", "--verify", f"refs/tags/{tag}")
    return r is not None and r.returncode == 0


def git_blob(root: Path, ref: str, rel: str) -> bytes | None:
    r = _git(root, "show", f"{ref}:{rel}")
    return r.stdout if r is not None and r.returncode == 0 else None


def frozen_bytes(root: Path, rel: str) -> tuple[bytes | None, str]:
    """(bytes, source) of REL: prereg-v1's copy when the tag exists, else HEAD's when the root is
    a git work tree, else the file itself (a staged commit on a VM, a test root)."""
    root = Path(root)
    if has_git(root):
        if tag_exists(root):
            return git_blob(root, f"refs/tags/{V1_TAG}", rel), f"tag:{V1_TAG}"
        return git_blob(root, "HEAD", rel), "HEAD"
    p = root / rel
    return (p.read_bytes() if p.is_file() else None), "file"


def released_kitti_sha(root: Path) -> str | None:
    """The released kitti.pth sha256 from the frozen weights.json; the working tree must agree."""
    raw, src = frozen_bytes(root, WEIGHTS_REL)
    if raw is None:
        return None
    try:
        sha = ((json.loads(raw).get("weights") or {}).get("u2flow-kitti") or {}).get("sha256")
    except ValueError:
        return None
    wt = Path(root) / WEIGHTS_REL
    if sha and src != "file" and wt.is_file():
        try:
            now = ((json.loads(wt.read_text("utf-8")).get("weights") or {}).get("u2flow-kitti")
                   or {}).get("sha256")
        except ValueError:
            now = None
        if now != sha:
            raise FrozenError(f"{WEIGHTS_REL}: u2flow-kitti sha256 in the working tree differs "
                              f"from {src}'s; the released checkpoint is never re-pinned by hand")
    return sha if isinstance(sha, str) and len(sha) == 64 else None


def aggregate_allowlist(root: Path) -> set[str]:
    """The released kitti.pth, plus prereg-v1's frozen allowlist. Raises FrozenError when the
    working-tree allowlist differs from the tag's, or exists although the tag lacks it."""
    root = Path(root)
    out = set()
    k = released_kitti_sha(root)
    if k:
        out.add(k)
    wt = root / ALLOWLIST_REL
    if not (has_git(root) and tag_exists(root)):
        return out  # before prereg-v1 (or without git): kitti.pth only; a draft file counts not
    frozen = git_blob(root, f"refs/tags/{V1_TAG}", ALLOWLIST_REL)
    if frozen is None:
        if wt.exists():
            raise FrozenError(f"{ALLOWLIST_REL} exists but {V1_TAG} does not hold it: a file "
                              "created after the tag is not frozen (a prereg amendment)")
        return out
    if not wt.is_file() or wt.read_bytes() != frozen:
        raise FrozenError(f"{ALLOWLIST_REL} differs from {V1_TAG}'s copy (or is missing): an "
                          "edit after the tag is a prereg amendment")
    try:
        listed = json.loads(frozen).get("ckpt_sha256") or []
    except ValueError as e:
        raise FrozenError(f"{V1_TAG}:{ALLOWLIST_REL} is not JSON ({e})") from None
    return out | {s for s in listed if isinstance(s, str) and len(s) == 64}


def gate0_state(root: Path) -> dict:
    """{sha256, source, committed} of the GATE-0.json in force.
    - git work tree: prereg-v1's copy when the tag exists, else HEAD's; committed means the
      working file equals it (a draft edit after the commit is not in force).
    - no .git (a VM's staged commit): the file counts only when U2_GIT_SHA names the staged
      commit (stage_code.py packs `git archive <sha>`, so an untracked draft cannot be there)."""
    root = Path(root)
    p = root / GATE0_REL
    wt = p.read_bytes() if p.is_file() else None
    if has_git(root):
        raw, src = frozen_bytes(root, GATE0_REL)
        committed = raw is not None and wt == raw
        return {"sha256": hashlib.sha256(raw).hexdigest() if raw is not None else None,
                "source": src if raw is not None else "none", "committed": committed}
    staged = bool(os.environ.get("U2_GIT_SHA")) and wt is not None
    return {"sha256": hashlib.sha256(wt).hexdigest() if wt is not None else None,
            "source": "staged" if staged else ("file" if wt is not None else "none"),
            "committed": staged}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    a = ap.parse_args(argv)
    try:
        allow = sorted(aggregate_allowlist(a.root))
    except FrozenError as e:
        print(json.dumps({"error": str(e)}))
        return 1
    print(json.dumps({"aggregate_allowlist": allow, "gate0": gate0_state(a.root)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
