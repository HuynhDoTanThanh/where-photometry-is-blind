#!/usr/bin/env python3
"""p1_windows.py: the P1 item lists (evaluation windows, training windows, FF pairs) and the
committed plan file every P1 job checks its list against (plan row W2-05).

What it does.
    plan     builds the lists from the published file lists (data/manifest/files/*-mv-*.sha256;
             no image needed) and writes tools/p1/specs/windows_v1.json: the evaluation windows
             in full (one row each), and for the larger lists only their count and the sha256 of
             their id list. Prints the counts.
    check    rebuilds the lists from an extracted data tree (a VM's /content/data) and compares
             them with the plan file: exit 3 when anything differs.
    list     prints the ids of one list (eval, train, ff, ff-eval) from the plan + a data tree.
Jobs call load_items() in-process: the list comes from the data tree on the VM and must hash to
the committed plan, so a missing frame or a changed bundle is a refusal, never a silent change of
the evaluated set.

How to invoke (repo root).
    .venv/bin/python tools/p1/p1_windows.py plan [--files data/manifest/files] [--out F]
    python tools/p1/p1_windows.py check --data /content/data [--plan F]
    python tools/p1/p1_windows.py list --data D --kind eval [--max-prio 0]
Output lines: WINDOWS_PLAN, WINDOWS_COUNT, WINDOWS_CHECK_OK / WINDOWS_CHECK_DIFF, WINDOWS_ERROR.
Exit codes: 0 ok · 2 precondition (no manifest files, no split, no data tree) · 3 the data tree
differs from the plan · 64 usage.

Why it is built this way.
- The plan's '788 windows' [P W2-05] hold [M 2026-10-03], with 2 truncated: KITTI-12 training
  000031 and 000082 lack frames 00-04, so their w03-11 keeps frames 05-11 (flagged truncated,
  never shifted); the rule is teacher-factory's (tools/teacher/make_windows.py), so the P1 lists
  and every teacher cache name the same windows.
- Dev and KITTI-12 first: prio 0 rows are KITTI-15 dev ids (splits/k15_split_v1.json, whose
  sha256 the plan records) and KITTI-12; prio 1 is the other KITTI-15 half, whose IMAGES are run
  (outputs kept, never evaluated before unseal [P W2-06]). No GT is read here.
- The KITTI-15 half label is data, not a flag: argv never names the held-out half (launch.sh
  refuses such argv, runlib SEALED_RX), so jobs take `--max-prio 0` instead.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from p1lib import common as C  # noqa: E402
from p1lib import kitti as KT  # noqa: E402

PLAN = C.P1_DIR / "specs" / "windows_v1.json"
SPLIT = C.REPO / "splits" / "k15_split_v1.json"
FILES = C.REPO / "data" / "manifest" / "files"
KINDS = ("eval", "train", "ff", "ff-eval")
VERSION = "p1win_v1"


def ids_sha(rows: list[dict]) -> str:
    """sha256 over each row's id AND frames: a truncated window that loses one more frame
    changes the list even though its id does not."""
    return hashlib.sha256("\n".join(f"{r['id']}:{','.join(map(str, r.get('frames', [])))}"
                                     for r in rows).encode()).hexdigest()


def dev_ids(split: Path = SPLIT) -> list[str]:
    if not split.is_file():
        raise C.Stop(C.EXIT_PRE, f"{split}: no split file")
    return list(C.load_json(split)["dev_ids"])


def build(inv: dict, devs: list[str]) -> dict:
    ev, ev_absent = KT.eval_windows(inv, devs)
    tr, tr_absent = KT.train_windows(inv)
    return {"eval": ev, "eval_absent": ev_absent, "train": tr, "train_absent": tr_absent,
            "ff": KT.ff_pairs(inv), "ff-eval": KT.ff_eval_pairs(inv, devs)}


def counts(lists: dict) -> dict:
    ev = lists["eval"]
    by = {}
    for r in ev:
        k = f"{r['ds']}:{r['win']}:{r['k15_half'] or 'k12'}"
        by[k] = by.get(k, 0) + 1
    return {"eval": len(ev), "eval_absent": len(lists["eval_absent"]),
            "eval_truncated": sum(1 for r in ev if r["truncated"]),
            "train_truncated": sum(1 for r in lists["train"] if r["truncated"]),
            "eval_prio0": sum(1 for r in ev if r["prio"] == 0), "eval_by": by,
            "train": len(lists["train"]), "train_absent": len(lists["train_absent"]),
            "ff": len(lists["ff"]), "ff-eval": len(lists["ff-eval"])}


def plan_doc(lists: dict, source: dict) -> dict:
    return {
        "schema": 1, "version": VERSION, "tool": "tools/p1/p1_windows.py",
        "definitions": {"eval_windows": KT.EVAL_WINDOWS, "train_windows": KT.TRAIN_WINDOWS,
                        "eval_pair": list(KT.EVAL_PAIR), "excluded_frames": sorted(KT.EXCLUDED),
                        "doc": "tools/p1/p1lib/kitti.py header"},
        "source": source, "split_sha256": C.sha256_file(SPLIT), "counts": counts(lists),
        "eval_windows": lists["eval"], "eval_absent": lists["eval_absent"],
        "lists": {k: {"n": len(lists[k]), "ids_sha256": ids_sha(lists[k])}
                  for k in ("eval", "train", "ff", "ff-eval")},
        "train_absent": lists["train_absent"],
    }


def dump_plan(doc: dict) -> str:
    """Pretty, but one row per line for the long lists (reviewable diffs)."""
    head = {k: v for k, v in doc.items() if k not in ("eval_windows", "eval_absent",
                                                      "train_absent")}
    out = json.dumps(head, indent=1, sort_keys=True)[:-2]
    for k in ("eval_absent", "eval_windows", "train_absent"):
        rows = ",\n".join("  " + json.dumps(r, sort_keys=True) for r in doc[k])
        out += f',\n "{k}": [\n{rows}\n ]'
    return out + "\n}\n"


def load_plan(path: Path = PLAN) -> dict:
    if not path.is_file():
        raise C.Stop(C.EXIT_PRE, f"{path}: no plan file (p1_windows.py plan)")
    return C.load_json(path)


def load_items(kind: str, data: Path, plan_path: Path = PLAN, max_prio: int | None = None,
               inv: dict | None = None) -> list[dict]:
    """The rows of one list, rebuilt from the data tree and checked against the plan file."""
    if kind not in KINDS:
        raise C.Stop(C.EXIT_USAGE, f"--kind must be one of {', '.join(KINDS)}")
    plan = load_plan(plan_path)
    lists = build(inv if inv is not None else KT.inventory_from_data(data), dev_ids())
    want = plan["lists"][kind]
    have = {"n": len(lists[kind]), "ids_sha256": ids_sha(lists[kind])}
    if have != want:
        raise C.Stop(C.EXIT_REFUSED, f"{kind}: the data tree gives n={have['n']} "
                                     f"sha={have['ids_sha256'][:12]}, the plan n={want['n']} "
                                     f"sha={want['ids_sha256'][:12]} (missing frames?)")
    rows = lists[kind]
    if max_prio is not None:
        rows = [r for r in rows if r.get("prio", 0) <= max_prio]
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--files", default=str(FILES))
    p.add_argument("--out", default=str(PLAN))
    c = sub.add_parser("check")
    c.add_argument("--data")
    c.add_argument("--plan", default=str(PLAN))
    lp = sub.add_parser("list")
    lp.add_argument("--data")
    lp.add_argument("--plan", default=str(PLAN))
    lp.add_argument("--kind", required=True, choices=KINDS)
    lp.add_argument("--max-prio", type=int)
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return C.EXIT_USAGE if e.code else C.EXIT_OK
    try:
        if a.cmd == "plan":
            files = Path(a.files)
            srcs = sorted(f for f in files.glob("u2f-*.sha256") if "-mv-" in f.name)
            if not srcs:
                raise C.Stop(C.EXIT_PRE, f"{files}: no *-mv-*.sha256 file lists")
            inv = KT.inventory_from_manifests(files)
            lists = build(inv, dev_ids())
            source = {"kind": "manifests", "files": {f.name: C.sha256_file(f) for f in srcs}}
            doc = plan_doc(lists, source)
            C.atomic_bytes(Path(a.out), dump_plan(doc).encode())
            C.say("WINDOWS_PLAN", f"{a.out}")
            C.say("WINDOWS_COUNT", json.dumps(doc["counts"], sort_keys=True))
            return C.EXIT_OK
        data = C.data_root(a.data)
        if a.cmd == "check":
            plan = load_plan(Path(a.plan))
            lists = build(KT.inventory_from_data(data), dev_ids())
            diff = [k for k in plan["lists"] if plan["lists"][k] != {
                "n": len(lists[k]), "ids_sha256": ids_sha(lists[k])}]
            if lists["eval"] != plan["eval_windows"]:
                diff.append("eval_windows rows")
            if diff:
                C.say("WINDOWS_CHECK_DIFF", ", ".join(diff))
                return C.EXIT_REFUSED
            C.say("WINDOWS_CHECK_OK", json.dumps(counts(lists), sort_keys=True))
            return C.EXIT_OK
        for r in load_items(a.kind, data, Path(a.plan), a.max_prio):
            print(r["id"])
        return C.EXIT_OK
    except C.Stop as e:
        C.say("WINDOWS_ERROR", e.msg)
        return e.code


if __name__ == "__main__":
    sys.exit(main())
