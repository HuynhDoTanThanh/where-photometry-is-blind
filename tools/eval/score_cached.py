#!/usr/bin/env python3
"""Mac-side re-scoring of a run's dumped float32 flows against the Mac GT (U2FLOW_data/gt),
with parity to the producing run's metrics.json (link 2), and the results rows for note.py.

How to invoke (repo root, .venv/bin/python; the run dir arrived through colab-fleet's collect
path, sha256-checked against the VM's manifest):
    python tools/eval/score_cached.py --run-dir OUT                       # dev or k12: link 2
    python tools/eval/score_cached.py --run-dir OUT --meta meta.json --rows   # + eval_rows.json
    python tools/eval/score_cached.py --run-dir OUT --meta meta.json --rows --record
    python tools/eval/score_cached.py --run-dir OUT --g0-audit \
        --gt-zip /Volumes/Shin/U2FLOW_data/sealed/u2f-k15-gt-heldout-v1.zip   # all-200, Mac only
    (dev ids come from --gt-root, held-out ids from the zip, read in memory; rows need the zip's
    sha256 to equal the manifest's u2f-k15-gt-heldout-v1, which seal.py finalize records)
meta.json carries what the run, not the scorer, knows: {"exp", "run", "seed", "step",
"session", "packed_with", "protocol", "flags", "oracle"} (metric skill section 6). It may not
carry 'ff': a +FF row's {seed, iters, thr, lr} comes only from the producing run's metrics.json
(eval_kitti.py --ff-seed), and every row and parity evidence carries the run's gate0_sha256
(the GATE-0.json in force when the number was made; a different one now is refused, exit 3).
--record appends the rows through note.record_eval_rows (the same gate as the CLI, under the
ledger lock), then the parity links as evidence (kind reproduction, subkind parity-link0/1/2,
the data GATE-0 G0.2 reads, with the artefact path and sha256 it re-hashes); without --record
the exact call is:
    python tools/note.py result --from-eval OUT/mac/eval_rows.json
Outputs (--out, default OUT/mac): score.json, per_image.csv (dev, k12), parity_link2.json,
eval_rows.json. stdout: SCORE_OK (counts and shas; values only with --print-aggregate, and only
when allowed).

Exit codes: 0 SCORE_OK | 1 error | 3 refused by a gate (rows for a partial, dirty or unverified
run; all-200 before prereg/GATE-0.json or off the allowlist; note.py refused the rows) |
6 parity link 2 failed: no number may be quoted until the cause is fixed | 7 integrity: a shard,
a GT file, the GT bundle or the evaluator hash differs from what the run recorded | 50 seal: a
held-out split or id, all-200 without --g0-audit, or a GT path under the seal (a STOP) | 64 usage.

What it checks, in order
    1. the seal: split heldout is unseal.py's (prereg); dev ids must all be dev ids; full200
       needs --g0-audit, prereg/GATE-0.json and the checkpoint on the aggregate-only allowlist;
    2. eval_sha here equals the run's (the evaluator did not change between the two hosts);
    3. every shard's sha256 against PREDS.json, and pred_sha against metrics.json;
    4. the GT bundle: its manifest sha256 equals the run's gt_pack_sha, and every GT file read
       matches the bundle's FILES.sha256;
    5. link 2 (dev, k12): identical n/out(/out3) per image and region, |delta| <= 1e-5 on every
       per-image EPE and on every aggregate.
For full200 nothing per image is written or printed: the aggregate EPE (img-mean) and Fl
(px-pooled) over 200 images are the only outputs (hard rule 4; quoting.md).

Why it is built this way
- Held-out GT never reaches a VM, so the all-200 number can only be made here, and it is trusted
  because links 1 and 2 hold on dev (metric skill section 3).
- topaneu found 7 wrong numbers in one day, 3 copied from prose: rows are generated from this
  scorer's own output with every provenance field, never typed (hard rule 6, note.py --from-eval).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval_kitti as ek  # noqa: E402
import regions as rg  # noqa: E402

EXIT_OK, EXIT_ERR, EXIT_GATE, EXIT_PARITY, EXIT_INTEGRITY, EXIT_SEAL, EXIT_USAGE = (
    0, 1, 3, 6, 7, 50, 64)
TOL_LINK2 = 1e-5
SEALED_BUNDLE = "u2f-k15-gt-heldout-v1"   # built by tools/prereg/seal.py finalize
META_FIELDS = ("exp", "run", "seed", "step", "session", "packed_with", "protocol", "flags",
               "oracle")
_PAIRS = (("Fl", "px-pooled"), ("EPE", "px-pooled"), ("EPE", "img-mean"))
_K12_PAIRS = (("Fl(K15-rule)", "px-pooled"), ("Out3", "px-pooled"), ("EPE", "px-pooled"),
              ("EPE", "img-mean"))


class Fail(Exception):
    def __init__(self, code: int, msg: str) -> None:
        super().__init__(msg)
        self.code = code


def _sealed(p: Path | None) -> bool:
    return p is not None and "u2flow_data/sealed" in str(Path(p).resolve()).lower()


def check_seal(m: dict, g0_audit: bool, root: Path) -> None:
    split = m.get("split")
    if split == "heldout" or split not in ek.SPLITS:
        raise Fail(EXIT_SEAL, f"split {split!r}: held-out rows come only from unseal.py via "
                   "seal.py (prereg skill); score_cached never scores them")
    if split == "dev":
        dev, held = ek.split_ids(root)
        bad = [i for i in m["ids"] if i in set(held) or i not in set(dev)]
        if bad:
            raise Fail(EXIT_SEAL, f"{len(bad)} id(s) of this dev run are not dev ids")
    if split == "full200":
        if not g0_audit:
            raise Fail(EXIT_SEAL, "full200 is scored only as the GATE-0 aggregate-only audit "
                       "(--g0-audit)")
        if not ek.gate0_committed(root):
            raise Fail(EXIT_GATE, f"{ek.GATE0_REL} must be committed before the all-200 audit")
        try:
            allow = ek.aggregate_allowlist(root)
        except ek.Refused as e:
            raise Fail(EXIT_GATE, str(e)) from None
        if m["ckpt_sha"] not in allow:
            raise Fail(EXIT_GATE, f"checkpoint {m['ckpt_sha'][:12]} is not on the aggregate-only "
                       "allowlist (prereg-v1's, else kitti.pth only): no held-out number for our "
                       "own methods")
        if not m.get("gate0_sha256"):
            raise Fail(EXIT_GATE, "the all-200 run recorded no committed GATE-0.json sha "
                       "(metrics.json gate0_sha256): it predates the criteria")
        if m["ids"] != list(ek.K15_IDS):
            raise Fail(EXIT_SEAL, "an all-200 audit covers exactly ids 000000-000199")
    elif g0_audit:
        raise Fail(EXIT_USAGE, "--g0-audit belongs to a full200 run only")


def _close(a, b, tol: float) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return abs(float(a) - float(b)) <= tol


def compare_link2(mine: list[dict], m: dict, agg: dict) -> dict:
    fails = []
    theirs = m.get("per_image") or []
    if [p["id"] for p in mine] != [p["id"] for p in theirs]:
        fails.append("the image lists differ")
    worst = 0.0
    for a, b in zip(mine, theirs, strict=False):
        for reg, ra in a["r"].items():
            rb = b["r"].get(reg)
            if rb is None or any(ra[k] != rb.get(k) for k in ra if k != "epe_sum"):
                fails.append(f"{a['id']} {reg}: counts differ")
                continue
            if ra["n"]:
                d = abs(ra["epe_sum"] / ra["n"] - rb["epe_sum"] / rb["n"])
                worst = max(worst, d)
                if not d <= TOL_LINK2:
                    fails.append(f"{a['id']} {reg}: per-image EPE |delta| > {TOL_LINK2}")
        for k in ("u2f_noc", "u2f_occ"):
            if not _close(a.get(k), b.get(k), TOL_LINK2):
                fails.append(f"{a['id']}: {k} differs")
    theirs_agg = m.get("aggregate") or {}
    for k, v in agg["u2f"].items():
        if not _close(v, (theirs_agg.get("u2f") or {}).get(k), TOL_LINK2):
            fails.append(f"aggregate {k} differs")
    for reg, rec in agg["regions"].items():
        other = (theirs_agg.get("regions") or {}).get(reg) or {}
        for k, v in rec.items():
            tol = 0 if k in ("n_valid", "n_out", "n_images", "n_out3") else TOL_LINK2
            if not _close(v, other.get(k), tol):
                fails.append(f"region {reg} {k} differs")
    agg_d = []
    for k, v in agg["u2f"].items():
        o = (theirs_agg.get("u2f") or {}).get(k)
        if isinstance(v, float) and isinstance(o, float):
            agg_d.append(abs(v - o))
    same = not any("counts differ" in f or " n_" in f for f in fails)
    return {"ok": not fails, "fails": fails[:50], "n_fails": len(fails), "tolerance": TOL_LINK2,
            "max_abs_per_image_epe_delta": worst, "max_abs_diff": max([worst, *agg_d]),
            "outliers_equal": same}


def build_rows(m: dict, agg: dict, meta: dict) -> list[dict]:
    missing = [k for k in META_FIELDS if k not in meta]
    if missing:
        raise Fail(EXIT_USAGE, f"--meta lacks {missing}")
    if "ff" in meta:
        raise Fail(EXIT_USAGE, "--meta may not carry 'ff': the +FF seed and parameters come only "
                   "from the producing run's metrics.json (eval_kitti.py --ff-seed)")
    env = m["env"]
    base = {**{k: meta[k] for k in META_FIELDS}, "dataset": m["dataset"], "split": m["split"],
            "eval_sha": m["eval_sha"], "gt_pack_sha": agg["gt_pack_sha"],
            "pred_sha": m["pred_sha"], "ckpt_sha": m["ckpt_sha"], "gpu_name": env["gpu_name"],
            "cc": env["cc"], "driver": env["driver"], "lock_hash": env["lock_hash"],
            "git_sha": m["git_sha"], "standard": "measured"}
    if m.get("ff"):                       # +FF rows: {seed, iters, thr, lr, ...} (trap T26),
        base["ff"] = m["ff"]              # as the producing run recorded them, never typed
    if m.get("gate0_sha256"):             # the GATE-0.json in force when the number was made
        base["gate0_sha256"] = m["gate0_sha256"]
    rows = []
    if m["split"] == "full200":
        a = agg["regions"]["all"]
        for metric, ag, v in (("EPE", "img-mean", agg["u2f"]["epe"]),
                              ("Fl", "px-pooled", agg["u2f"]["fl"])):
            rows.append({**base, "region": "all", "metric": metric, "agg": ag, "value": v,
                         "n_images": agg["n_images"], "n_pixels": a["n_valid"],
                         "g0_audit": True})
        return rows
    pairs = _PAIRS if m["dataset"] == "kitti15" else _K12_PAIRS
    for reg, rec in agg["regions"].items():
        for metric, ag in pairs:
            v = {"Fl": rec["fl"], "Fl(K15-rule)": rec["fl"], "Out3": rec.get("out3"),
                 "EPE": rec["epe_pooled"] if ag == "px-pooled" else rec["epe_img_mean"]}[metric]
            if v is None or not rec["n_images"]:
                continue
            rows.append({**base, "region": reg, "metric": metric, "agg": ag, "value": v,
                         "n_images": rec["n_images"], "n_pixels": rec["n_valid"]})
    return rows


class SplitGT:
    """All-200 GT: dev ids from the verified Mac GT root, held-out ids from the zip (the sealed
    bundle, read in memory; nothing is extracted, written or cached)."""

    def __init__(self, dev: rg.GTSource, sealed: rg.GTSource, held: set[str]) -> None:
        self.dev, self.sealed, self.held = dev, sealed, held

    def read(self, img_id: str, need_obj: bool = True) -> rg.GTImage:
        return (self.sealed if img_id in self.held else self.dev).read(img_id, need_obj)

    def close(self) -> None:
        self.dev.close()
        self.sealed.close()


def _note():
    spec = importlib.util.spec_from_file_location("note", HERE.parent / "note.py")
    note = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(note)
    return note


def record_evidence(subkind: str, source: str, data: dict, summary: str,
                    exp: str | None) -> None:
    """A parity link as evidence kind 'reproduction' (GATE-0 G0.2 reads data.split, n_images,
    max_abs_diff, outliers_equal, gpu_name; its source starts with tools/eval/)."""
    argv = ["evidence", "--kind", "reproduction", "--subkind", subkind, "--standard",
            "measured", "--source", source, "--summary", summary,
            "--data", json.dumps(data, sort_keys=True)]
    if exp:
        argv += ["--exp", exp]
    if _note().main(argv) != 0:
        raise Fail(EXIT_GATE, f"note.py refused the {subkind} evidence")


def record(rows: list[dict], source: str) -> list[dict]:
    note = _note()
    try:
        return note.record_eval_rows(rows, source, root=ek.repo_root())
    except (note.Refused, note.pr.PredicateError, note.lg.LedgerError) as e:
        raise Fail(EXIT_GATE, f"note.py refused the rows: {e}") from None


def score(run_dir: Path, *, gt_root: Path | None, out: Path | None = None,
          g0_audit: bool = False, gt_zip: Path | None = None, gt_zip_prefix: str = "",
          root: Path | None = None) -> dict:
    root = root or ek.repo_root()
    m = json.loads((run_dir / "metrics.json").read_text("utf-8"))
    out = out or run_dir / "mac"
    check_seal(m, g0_audit, root)
    for p in (gt_root, out):
        if _sealed(p):
            raise Fail(EXIT_SEAL, f"{p} is under the seal; seal.py/unseal.py read it, not this")
    full = m["split"] == "full200"
    g0 = ek.frozen.gate0_state(root)
    if m.get("gate0_sha256") and g0["committed"] and g0["sha256"] != m["gate0_sha256"]:
        raise Fail(EXIT_GATE, "prereg/GATE-0.json changed between the run and this scoring "
                   f"(run {m['gate0_sha256'][:12]}, now {g0['sha256'][:12]}): the criteria "
                   "moved after the number existed; a STOP for the user, never re-scored")
    esha, _ = rg.eval_sha(rg.CODE_ROOT)
    if esha != m["eval_sha"]:
        raise Fail(EXIT_INTEGRITY, "eval_sha differs from the run's: the evaluator changed "
                   "between the producing host and this one (metric skill section 9)")
    pred_dir = run_dir / m.get("preds", "preds")
    shards = json.loads((pred_dir / "PREDS.json").read_text("utf-8"))["shards"]
    if ek.pred_sha(shards) != m["pred_sha"]:
        raise Fail(EXIT_INTEGRITY, "PREDS.json does not hash to the run's pred_sha")
    dataset = m["dataset"]
    if full:
        if gt_zip is None:
            raise Fail(EXIT_USAGE, "--gt-zip is required for the all-200 audit")
        if gt_root is None:
            raise Fail(EXIT_USAGE, "--gt-root (the dev GT) is required for the all-200 audit")
        b = ek.gt_bundle(dataset, gt_root, root)
        dev_src = rg.GTSource(dataset, root=gt_root, require_r4=True)
        held = set(ek.split_ids(root)[1])
        src = SplitGT(dev_src, rg.GTSource(dataset, zip_path=gt_zip, zip_prefix=gt_zip_prefix,
                                           verify=False, calib=dev_src.calib,
                                           require_r4=True), held)
        gt_sha = rg.sha256_file(gt_zip)
        man = json.loads((root / "data" / "manifest" / "bundles.v1.json").read_text("utf-8"))
        sealed_sha = ((man.get("bundles") or {}).get(SEALED_BUNDLE) or {}).get("sha256")
        integrity = ("verified" if gt_sha == sealed_sha and dev_src.verified
                     and b["integrity"] == "verified" else "unverified")
    else:
        if gt_root is None:
            raise Fail(EXIT_USAGE, "--gt-root is required")
        src = rg.GTSource(dataset, root=gt_root, require_r4=True)
        b = ek.gt_bundle(dataset, gt_root, root)
        if b["gt_pack_sha"] != m["gt_pack_sha"]:
            raise Fail(EXIT_INTEGRITY, f"GT bundle {b['name']}: the manifest sha256 differs "
                       "from the run's gt_pack_sha")
        gt_sha = b["gt_pack_sha"]
        integrity = "verified" if (src.verified and b["integrity"] == "verified") else "unverified"
    per_image = []
    try:
        seen = []
        for img_id, flow in ek.read_preds(pred_dir):
            seen.append(img_id)
            gt = src.read(img_id, need_obj=not full)
            per_image.append({"id": img_id, "H": gt.shape[0], "W": gt.shape[1],
                              **rg.score_image(flow, gt, dataset, only_all=full)})
    except rg.IntegrityError as e:
        raise Fail(EXIT_INTEGRITY, str(e)) from None
    finally:
        src.close()
    if seen != m["ids"]:
        raise Fail(EXIT_INTEGRITY, "the shards do not hold exactly the run's ids in order")
    agg = rg.aggregate(per_image, dataset, only_all=full)
    if full:
        agg = {"n_images": agg["n_images"], "n_pixels": agg["n_pixels"],
               "u2f": {"epe": agg["u2f"]["epe"], "fl": agg["u2f"]["fl"]},
               "regions": {"all": {k: agg["regions"]["all"][k]
                                   for k in ("n_valid", "n_out", "n_images")}},
               "aggregate_only": True}
        per_image = []
    agg.update(gt_pack_sha=gt_sha, gt_integrity=integrity)
    if full:
        agg["gt_packs"] = {"dev": b["gt_pack_sha"], "heldout": gt_sha}
    res = {"schema": 1, "tool": "tools/eval/score_cached.py", "run_dir": str(run_dir),
           "dataset": dataset, "split": m["split"], "n_images": len(m["ids"]),
           "partial": m["partial"], "eval_sha": esha, "pred_sha": m["pred_sha"],
           "ckpt_sha": m["ckpt_sha"], "aggregate": agg, "gate0_sha256": m.get("gate0_sha256"),
           "ff": m.get("ff"), "utc": ek.utc_now()}
    if not full:
        link2 = compare_link2(per_image, m, agg)
        ek.atomic_write(out / "parity_link2.json", json.dumps(
            {**link2, "link": 2, "split": m["split"], "n_images": len(m["ids"]),
             "partial": m["partial"], "gpu_name": m["env"]["gpu_name"], "eval_sha": esha,
             "pred_sha": m["pred_sha"], "utc": res["utc"]},
            indent=1, sort_keys=True) + "\n")
        res["parity_link2"] = {"ok": link2["ok"], "n_fails": link2["n_fails"]}
        res["per_image"] = per_image
        ek.write_per_image_csv(out / "per_image.csv", per_image)
    ek.atomic_write(out / "score.json", json.dumps(res, indent=1, sort_keys=True) + "\n")
    return res


def rows_allowed(m: dict, res: dict) -> str | None:
    """Why rows may not be written for this run, or None."""
    if m["partial"]:
        return "a partial run (--ids) is a smoke: no rows"
    if m.get("git_dirty") or not m.get("git_sha"):
        return "the run's evaluator files were not committed (git_dirty): a smoke, no rows"
    if res["aggregate"]["gt_integrity"] == "unverified" or m.get("gt_integrity") == "unverified":
        return "GT read without its FILES.sha256 check: no rows"
    if res.get("parity_link2", {"ok": True})["ok"] is not True:
        return "parity link 2 failed"
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True, help="eval_kitti.py --out, as collected")
    ap.add_argument("--gt-root", default="/Volumes/Shin/U2FLOW_data/gt")
    ap.add_argument("--out", help="default: RUN_DIR/mac")
    ap.add_argument("--split", help="optional: must equal the run's split")
    ap.add_argument("--g0-audit", action="store_true")
    ap.add_argument("--gt-zip", help="all-200 GT zip (held in memory, never extracted)")
    ap.add_argument("--gt-zip-prefix", default="kitti15/",
                    help="member prefix before 'training/' ('kitti15/' in the sealed bundle, "
                    "'' in the KITTI source zip)")
    ap.add_argument("--meta", help="run metadata JSON (see the header)")
    ap.add_argument("--rows", action="store_true", help="write eval_rows.json")
    ap.add_argument("--record", action="store_true", help="append the rows via note.py")
    ap.add_argument("--print-aggregate", action="store_true")
    a = ap.parse_args(argv)
    run_dir = Path(a.run_dir)
    out = Path(a.out) if a.out else run_dir / "mac"
    root = ek.repo_root()
    try:
        m = json.loads((run_dir / "metrics.json").read_text("utf-8"))
        if a.split and a.split != m.get("split"):
            raise Fail(EXIT_SEAL if a.split == "heldout" else EXIT_USAGE,
                       f"--split {a.split} but the run is {m.get('split')}")
        if (a.rows or a.record) and not a.meta:
            raise Fail(EXIT_USAGE, "--rows/--record need --meta")
        res = score(run_dir, gt_root=Path(a.gt_root), out=out,
                    g0_audit=a.g0_audit, gt_zip=Path(a.gt_zip) if a.gt_zip else None,
                    gt_zip_prefix=a.gt_zip_prefix, root=root)
        link = res.get("parity_link2", {}).get("ok")
        print(f"SCORE_OK split={res['split']} n={res['n_images']} partial={res['partial']} "
              f"parity_link2={'n/a' if link is None else ('ok' if link else 'FAIL')} "
              f"pred_sha={res['pred_sha'][:12]} eval_sha={res['eval_sha'][:12]} out={out}")
        if link is False:
            print(f"PARITY2_FAIL see {out / 'parity_link2.json'}")
            return EXIT_PARITY
        if a.print_aggregate:
            if not res["partial"] and (not m.get("released_ckpt")
                                       or ek.gate0_committed(root)):
                print("SCORE_AGG " + json.dumps(res["aggregate"]["u2f"], sort_keys=True))
            else:
                print("SCORE_AGG withheld: a partial run, or a released checkpoint before "
                      f"{ek.GATE0_REL} exists", file=sys.stderr)
        if a.rows or a.record:
            why = rows_allowed(m, res)
            if why:
                raise Fail(EXIT_GATE, why)
            rows = build_rows(m, res["aggregate"], json.loads(Path(a.meta).read_text("utf-8")))
            rows_path = out / "eval_rows.json"
            ek.atomic_write(rows_path, json.dumps({"rows": rows}, indent=1, sort_keys=True)
                            + "\n")
            print(f"ROWS {len(rows)} -> {rows_path}; record: python tools/note.py result "
                  f"--from-eval {rows_path}")
            if a.record:
                stored = record(rows, str(rows_path))
                print(f"RECORDED {len(stored)} row(s)"
                      + (f" {stored[0]['id']}..{stored[-1]['id']}" if stored else ""))
                exp = json.loads(Path(a.meta).read_text("utf-8")).get("exp")
                links = ((0, run_dir / "parity_link0.json", "tools/eval/eval_kitti.py "
                          "--parity-link0"),
                         (1, run_dir / "parity_link1.json", "tools/eval/eval_kitti.py "
                          "--parity-link1"),
                         (2, out / "parity_link2.json", "tools/eval/score_cached.py"))
                keep = ("split", "n_images", "max_abs_diff", "outliers_equal", "gpu_name", "ok",
                        "tolerance", "eval_sha", "pred_sha", "partial")
                for n, pj, cmd in links:
                    if not pj.is_file():
                        continue
                    d = json.loads(pj.read_text("utf-8"))
                    # the artefact and its sha: gate.py re-hashes it and reads the fields from
                    # it, so a typed evidence row cannot stand in for a measurement (GATE-0)
                    data = {**{k: d.get(k) for k in keep},
                            "gate0_sha256": m.get("gate0_sha256"),
                            "artifact": str(pj.resolve()), "artifact_sha256": rg.sha256_file(pj)}
                    record_evidence(f"parity-link{n}", f"{cmd}; {pj}", data,
                                    f"parity link {n}: {'ok' if d.get('ok') else 'FAIL'}", exp)
                    print(f"EVIDENCE parity-link{n} recorded")
        return EXIT_OK
    except Fail as e:
        print(f"score_cached: refused: {e}", file=sys.stderr)
        return e.code
    except rg.IntegrityError as e:
        print(f"score_cached: integrity: {e}", file=sys.stderr)
        return EXIT_INTEGRITY
    except (rg.RegionError, OSError, ValueError, KeyError) as e:
        print(f"score_cached: {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_ERR


if __name__ == "__main__":
    sys.exit(main())
