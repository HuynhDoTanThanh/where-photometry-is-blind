#!/usr/bin/env python3
"""composite_bound.py: the IDEA-002 teacher-complementarity bound. On the dev half of KITTI-15,
from CACHED flows only, how much could a per-pixel composite of a student S (U2Flow kitti.pth,
optionally the PF and M pilots) with the W3-02 rigid-teacher renders T lower px-pooled Fl? CPU, on
the Mac; it computes numbers, writes no ledger row (hard rule 7) and starts no VM.

What it computes (per image, then pooled; Fl = 100 * sum(out) / sum(n), KITTI-15 outlier rule
EPE > 3 px and EPE / |F_gt| > 0.05 from tools/eval/regions.py `outliers`, via region_table.py's
`pixel_maps`), on regions all (V_occ), R1 and R3 of regions_v1:
    S                     the student as it is
    oracle-blind-R3[T]    T on GT-static occluded-out-of-frame pixels (R3) where T is finite, S
                          elsewhere: the GATE-1 draft (b) rule ('teacher on GT-blind static
                          pixels, U2Flow elsewhere', prereg/drafts/GATE-1.json _row_conventions
                          .composite) with R2 DROPPED: regions.R2_NEAR_EMPTY is None, so R2 is
                          'withheld-S6' (regions.md S6; reports/p1_region/region_table_r4.json):
                          no R2 count may be used or shown before the S6 threshold is fixed
    ORACLE-pixel[T]       per pixel the better of S and T (T non-finite = S): an upper bound that
                          reads GT, never a method
    ORACLE-best-of-named  per pixel the best of S and every named RIGID teacher variant
                          (mapanything ref/native, pi3 ref/native, selftri; all windows)
    ORACLE-best-of-spec-nonrigid-incl
                          the same over every teacher entry of the W3-02 spec, which adds
                          COLMAP, Monodepth2 and RAFT C+T (a flow network, NOT a rigid teacher):
                          the loosest bound, context only, never the IDEA-002 kill read
The kill read (header.kill_read) takes the largest Fl-all gain of the rigid oracle rows only
(ORACLE-pixel[named] and ORACLE-best-of-named) against IDEA-002's 0.6 pp; best-of-spec is
reported beside it under its own label.
Each teacher variant is scored per window (w03-11, w10-18) and, where it has both, 'train-like'
= counts pooled over the two windows (each dev pair counted twice; S counted twice alike), the
GATE-1 draft's window convention. Every row reports Fl with an image-bootstrap CI and the gain
vs S in pp (Fl(S) - Fl(row)) with a PAIRED image-bootstrap CI (tools/eval/stats.py).
Exact shortcut: at a fixed GT pixel the outlier test is monotone in EPE (|F_gt| is the same for
S and T), so 'the lower-EPE flow is an outlier' == out_S AND out_T; the oracles are computed from
the outlier maps and are exact for Fl (tests/test_composite_bound.py proves it). EPE is not
reported here.

How to invoke (repo root, the Mac venv):
    .venv/bin/python tools/p1/composite_bound.py --seed 20261006 \
        [--spec tools/p1/specs/composite_bound.idea-002.json] \
        [--gt-root /Volumes/Shin/U2FLOW_data/gt] \
        [--B 10000] [--alpha 0.05] [--json-out reports/p1_region/composite_bound.json | --no-json]
    B and alpha come from prereg/PRIMARY.json primary_endpoint.ci unless given; the seed is a
    diagnostic one (bootstrap_seed is null until prereg-v3) and is recorded as such.
    A Markdown table goes to stdout; the JSON (header + rows) to --json-out, an allowlisted path
    under reports/p1_region/ (region_table.out_path) that this tool may overwrite only if it wrote
    it. Exit codes: region_table.py's (0 ok, 1 error, 2 precondition: U2Flow or the GT root
    missing, 7 integrity, 50 seal, 64 usage).

Why it is built this way.
- One kernel and one loader: GT, region masks, cache readers, seal checks and the outlier map all
  come from tools/p1/region_table.py (and through it tools/eval/regions.py, stats.py); this file
  adds only the composite rules. tools/eval/* is never edited.
- Dev half only: every input goes through region_table.open_entry (a held-out id, a non-dev run
  or a sealed path is exit 50 before a pixel is read); KITTI-12 is not scored here.
- All composite rows are oracle rows (they read GT to pick pixels); the student row S is the only
  non-oracle number, and a student run's S Fl-all is cross-checked against its own metrics.json.
- A teacher that lacks an image (selftri: 96/100) or a pixel (NaN = no depth) falls back to S, so
  every row is scored on the same 100 images and the gains are paired.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import region_table as RT  # noqa: E402

ek, rg, st, C = RT.ek, RT.rg, RT.st, RT.C
Fail = RT.Fail
TOOL = "tools/p1/composite_bound.py"
DEFAULT_SPEC = "tools/p1/specs/composite_bound.idea-002.json"
DEFAULT_JSON = "reports/p1_region/composite_bound.json"
REGIONS = ("all", "R1", "R3")
BLIND_REGION = "R3"          # R2 u R3 once regions.R2_NEAR_EMPTY is fixed (S6); R3 until then
DS = "kitti15"
KILL_PP = 0.6                # IDEA-002 kill / GATE-1 draft (b) bar, quoted for the read-out only


# ------------------------------------------------------------------------------ composite rules
def composite_blind(out_s: np.ndarray, fin_t: np.ndarray, out_t: np.ndarray,
                    blind: np.ndarray) -> np.ndarray:
    """oracle-blind: T's outlier map on blind & finite(T), S's elsewhere."""
    use = blind & fin_t
    return np.where(use, out_t, out_s)


def oracle_pixel(out_s: np.ndarray, teachers: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Per-pixel error oracle over S and the teachers: an outlier only if S and every teacher
    that covers the pixel are outliers (a non-finite teacher pixel cannot help)."""
    out = out_s.copy()
    for fin_t, out_t in teachers:
        out &= out_t | ~fin_t
    return out


def counts(out: np.ndarray, masks: dict, regions=REGIONS) -> dict:
    return {r: {"n": int(masks[r].sum()), "out": int((out & masks[r]).sum())} for r in regions}


# ------------------------------------------------------------------------------ loading
def _unpack(b: np.ndarray, shape) -> np.ndarray:
    return RT.unpack(b, shape)


def load_source(entry: dict, root: Path, dev: set, held: set, gtc: RT.GTCache) -> dict:
    """{'prov', 'pred_sha', 'maps': {id: (finite_bits, out_bits)}} for one spec entry (dev half
    only). Raises Fail(EXIT_PRE) when the path is not on disk."""
    e = {**entry, "datasets": [DS]}
    RT.check_entry(e)
    prov, streams = RT.open_entry(e, root, dev, held)
    if DS not in streams:
        raise Fail(RT.EXIT_PRE, f"{e['path']}: holds no {DS} predictions")
    factory, pred_sha, _ = streams[DS]
    maps = {}
    for seq, flow in factory():
        shape, flow_occ, _ = gtc.get(seq)
        fin, _, out = RT.pixel_maps(flow, shape, flow_occ, DS)
        maps[seq] = (np.packbits(fin.reshape(-1)), np.packbits((out & fin).reshape(-1)))
    return {"prov": prov, "pred_sha": pred_sha, "maps": maps}


def _resolve(spec: dict, root: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """(students, named teacher entries, every teacher entry of teachers_spec)."""
    tpath = Path(spec["teachers_spec"])
    tspec = json.loads((tpath if tpath.is_absolute() else root / tpath).read_text("utf-8"))
    all_t = [e for e in tspec["sources"] if e["role"] == "teacher"]
    named = []
    for want in spec["teachers"]:
        hit = [e for e in all_t if e["source"] == want["source"] and e["pose"] == want["pose"]]
        if not hit:
            raise Fail(RT.EXIT_USAGE, f"teacher {want} is not in {tpath}")
        named += hit
    students = []
    for s in spec["students"]:
        if s.get("from_teachers_spec"):
            hit = [e for e in tspec["sources"] if e["source"] == s["source"]]
            if len(hit) != 1:
                raise Fail(RT.EXIT_USAGE, f"student {s['source']}: not exactly once in {tpath}")
            students.append({**hit[0], "label": s.get("label", s["source"])})
        else:
            students.append(dict(s))
    return students, named, all_t


def tkey(e: dict) -> str:
    return f"{e['source']}/{e['pose']}/{e['window']}"


# ------------------------------------------------------------------------------ statistics
def _arr(per_image: list[dict], region: str) -> dict:
    return {"n": np.array([p[region]["n"] for p in per_image], np.int64),
            "out": np.array([p[region]["out"] for p in per_image], np.int64)}


def arm_stats(per_image: list[dict], base: list[dict], boot: dict, draws) -> dict:
    """{region: {fl, fl_ci, gain_pp, gain_ci, n_images, n_pixels}}; gain = Fl(base) - Fl(arm),
    paired on the same image draws (stats.paired_image_bootstrap)."""
    lo, hi = 100 * boot["alpha"] / 2, 100 * (1 - boot["alpha"] / 2)
    res = {}
    for r in REGIONS:
        a, b = _arr(per_image, r), _arr(base, r)
        n_px = int(a["n"].sum())
        rec = {"n_images": int((a["n"] > 0).sum()), "n_pixels": n_px, "agg": "px-pooled"}
        if n_px == 0:
            res[r] = {**rec, "status": "n=0", "fl": None}
            continue
        rec["fl"] = st.statistic(a, "fl")
        rec["fl_ci"] = None
        if draws is not None:
            reps = np.array([st.statistic(a, "fl", i) for i in draws], np.float64)
            reps = reps[~np.isnan(reps)]
            rec["fl_ci"] = [float(x) for x in np.percentile(reps, [lo, hi])] if len(reps) else None
        if boot["B"] > 0:
            g = st.paired_image_bootstrap(b, a, stat="fl", B=boot["B"], seed=boot["seed"],
                                          alpha=boot["alpha"])
            rec["gain_pp"], rec["gain_ci"] = g["estimate"], g["ci"]
        else:
            rec["gain_pp"], rec["gain_ci"] = st.statistic(b, "fl") - rec["fl"], None
        rec["status"] = "ok"
        res[r] = rec
    return res


def _pool(*arms: list[dict]) -> list[dict]:
    """Per-image counts summed over windows (train-like)."""
    return [{r: {"n": sum(a[i][r]["n"] for a in arms), "out": sum(a[i][r]["out"] for a in arms)}
             for r in REGIONS} for i in range(len(arms[0]))]


# ------------------------------------------------------------------------------ the bound
def build(students: dict, teachers: dict, named: list[str], ids: list[str], mask_of,
          boot: dict) -> list[dict]:
    """Rows for every student. students/teachers: {key: {'maps': {id: (fin_bits, out_bits)}}};
    named: the teacher keys of the named set; mask_of(id) -> (shape, {region: mask})."""
    n = len(ids)
    draws = (np.random.default_rng(boot["seed"]).integers(0, n, (boot["B"], n))
             if boot["B"] > 0 else None)
    rows = []
    for skey, s in students.items():
        per = {"S": []}
        for i in ids:
            shape, m = mask_of(i)
            fin_s, out_s = (_unpack(x, shape) for x in s["maps"][i])
            if not fin_s.all():
                raise Fail(RT.EXIT_ERR, f"student {skey} {i}: non-finite flow (a student covers "
                           "every pixel)")
            tmaps = {}
            for tk, t in teachers.items():
                got = t["maps"].get(i)
                tmaps[tk] = ((_unpack(got[0], shape), _unpack(got[1], shape)) if got is not None
                             else (np.zeros(shape, bool), np.zeros(shape, bool)))
            per["S"].append(counts(out_s, m))
            for tk in named:
                fin_t, out_t = tmaps[tk]
                per.setdefault(f"oracle-blind-{BLIND_REGION}[{tk}]", []).append(
                    counts(composite_blind(out_s, fin_t, out_t, m[BLIND_REGION]), m))
                per.setdefault(f"ORACLE-pixel[{tk}]", []).append(
                    counts(oracle_pixel(out_s, [tmaps[tk]]), m))
            per.setdefault("ORACLE-best-of-named", []).append(
                counts(oracle_pixel(out_s, [tmaps[k] for k in named]), m))
            per.setdefault("ORACLE-best-of-spec-nonrigid-incl", []).append(
                counts(oracle_pixel(out_s, list(tmaps.values())), m))
        # train-like = both windows pooled, for a (source, pose) that has both
        groups: dict = {}
        for tk in named:
            src, pose, win = tk.split("/")
            groups.setdefault(f"{src}/{pose}", {})[win] = tk
        for sp, wins in groups.items():
            if {"w03-11", "w10-18"} <= set(wins):
                for kind in (f"oracle-blind-{BLIND_REGION}", "ORACLE-pixel"):
                    per[f"{kind}[{sp}/train-like]"] = _pool(
                        per[f"{kind}[{wins['w03-11']}]"], per[f"{kind}[{wins['w10-18']}]"])
        base1, base2 = per["S"], _pool(per["S"], per["S"])
        for arm, pi in per.items():
            two = arm.endswith("/train-like]")
            inner = arm[arm.index("[") + 1:-1] if "[" in arm else ""
            window = inner.split("/")[2] if inner.count("/") == 2 else "n/a"
            rows.append({"student": skey, "arm": arm, "oracle": arm != "S", "window": window,
                         "regions": arm_stats(pi, base2 if two else base1, boot, draws)})
    return rows


# ------------------------------------------------------------------------------ output
def _fmt(rec: dict, key: str, ci: str) -> str:
    v = rec.get(key)
    if v is None:
        return "-"
    c = rec.get(ci)
    return f"{v:.2f}" + (f" [{c[0]:.2f}, {c[1]:.2f}]" if c else "")


def render(doc: dict) -> str:
    h = doc["header"]
    lvl = round(100 * (1 - h["statistics"]["alpha"]))
    out = [f"# IDEA-002 composite bound (dev half, KITTI-15, n={h['n_images']} images)", "",
           f"Fl px-pooled, %; gain = Fl(S) - Fl(row) in pp; {lvl}% image-bootstrap CIs "
           f"(B={h['statistics']['B']}, seed {h['statistics']['seed']}, "
           "diagnostic). ORACLE rows read GT: upper bounds, never methods. "
           f"{h['s6']['note']}", ""]
    for skey in h["students"]:
        out += [f"## student {skey}: {h['students'][skey]['label']}", "",
                "| row | Fl-all [CI] | gain all [CI] | Fl R1 | gain R1 | Fl R3 [CI] "
                "| gain R3 [CI] |",
                "|---|---|---|---|---|---|---|"]
        for r in doc["rows"]:
            if r["student"] != skey:
                continue
            a, r1, r3 = (r["regions"][k] for k in REGIONS)
            out.append(f"| {r['arm']} | {_fmt(a, 'fl', 'fl_ci')} | {_fmt(a, 'gain_pp', 'gain_ci')}"
                       f" | {_fmt(r1, 'fl', '')} | {_fmt(r1, 'gain_pp', '')} "
                       f"| {_fmt(r3, 'fl', 'fl_ci')} | {_fmt(r3, 'gain_pp', 'gain_ci')} |")
        out.append("")
    k = h.get("kill_read")
    if k:
        out += [f"Kill read (IDEA-002, user decides): {k['text']}", ""]
    return "\n".join(out)


def _is_ours(p: Path, suffix: str) -> bool:
    try:
        return json.loads(p.read_text("utf-8")).get("header", {}).get("tool") == TOOL
    except (OSError, ValueError, AttributeError):
        return False


RIGID_ORACLES = ("ORACLE-pixel[", "ORACLE-best-of-named")
NONRIGID_ORACLE = "ORACLE-best-of-spec-nonrigid-incl"


def kill_read(rows: list[dict], student: str) -> dict | None:
    """IDEA-002's kill on the RIGID oracle rows only (ORACLE-pixel over a named teacher, and
    ORACLE-best-of-named): it fires when their largest Fl-all gain is < 0.6 pp. The
    non-rigid-inclusive best-of-spec row is reported beside it as context and never decides."""
    def gain(r):
        return r["regions"]["all"].get("gain_pp")

    mine = [r for r in rows if r["student"] == student and gain(r) is not None]
    cand = [(gain(r), r["arm"]) for r in mine if r["arm"].startswith(RIGID_ORACLES)]
    if not cand:
        return None
    g, arm = max(cand)
    ctx = [gain(r) for r in mine if r["arm"] == NONRIGID_ORACLE]
    fires = g < KILL_PP
    text = (f"largest RIGID error-oracle Fl-all gain for {student} = {g:.2f} pp ({arm}) "
            f"{'<' if fires else '>='} {KILL_PP} pp: kill {'FIRES' if fires else 'does not fire'}")
    if ctx:
        text += (f" (context, not the kill: {NONRIGID_ORACLE} = {ctx[0]:.2f} pp, includes "
                 "RAFT C+T, Monodepth2, COLMAP)")
    return {"student": student, "rigid_max_gain_pp": g, "rigid_arm": arm, "threshold_pp": KILL_PP,
            "fires": fires, "nonrigid_incl_gain_pp": ctx[0] if ctx else None, "text": text}


def crosscheck_region_table(root: Path, pred_sha: str, s_row: dict, ds: str = DS,
                            names: dict | None = None, out3: bool = False,
                            source: str = "u2flow") -> dict:
    """S's R1 and R3 Fl against region_table.py's JSON for the SAME pred_sha (a cache student
    has no metrics.json). Fl-all cannot be cross-checked there: region_table has no 'all' row
    and its R2 is withheld-S6, so the partition cannot be summed. names maps region_table's
    region to the s_row region (KITTI-12: R1 -> K12-R1); out3 also checks Out3 (KITTI-12);
    source names the region_table source (u2flow_ff for the +FF cache, selftri_vs_ff.py)."""
    names = names or {"R1": "R1", "R3": "R3"}
    for name in ("region_table_r4.json", "region_table.json"):
        p = root / RT.OUT_DIR / name
        if not p.is_file():
            continue
        doc = json.loads(p.read_text("utf-8"))
        srcs = [s for s in doc["header"]["sources"] if s.get("source") == source
                and (s.get("datasets") or {}).get(ds, {}).get("pred_sha") == pred_sha]
        if not srcs:
            continue
        res = {"against": f"{RT.OUT_DIR}/{name}", "fl_all": "not cross-checkable (no 'all' row; "
               "R2 withheld-S6)"}
        for r, mine in names.items():
            (c,) = [c for c in doc["cells"] if c["source"] == source and c["dataset"] == ds
                    and c["region"] == r]
            ref, got = c["fl_failclosed"]["value"], s_row["regions"][mine]["fl"]
            if abs(ref - got) > 1e-6 or c["n_pixels"] != s_row["regions"][mine]["n_pixels"]:
                raise Fail(RT.EXIT_ERR, f"{source} {r} Fl {got} != {name} {ref}")
            res[r] = {"region_table_fl": ref, "abs_diff": abs(ref - got)}
            if out3:
                ref3, got3 = c["out3_failclosed"]["value"], s_row["regions"][mine]["out3"]
                if abs(ref3 - got3) > 1e-6:
                    raise Fail(RT.EXIT_ERR, f"{source} {r} Out3 {got3} != {name} {ref3}")
                res[r].update(region_table_out3=ref3, out3_abs_diff=abs(ref3 - got3))
        return res
    return {"against": None, "why": "no region_table JSON with this pred_sha: not cross-checked"}


def code_identity(root: Path, files: list[Path]) -> dict:
    """The commit and the sha256 of the files that produced this JSON; dirty = any of them
    differs from HEAD or is untracked (ek.git_info only looks at evaluator files)."""
    rel = [str(f.relative_to(root)) if f.is_relative_to(root) else str(f) for f in files]
    info = {"files": {r: C.sha256_file(f) for r, f in zip(rel, files, strict=True)}}
    try:
        info["git_head"] = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                          capture_output=True, text=True,
                                          check=True).stdout.strip()
        stt = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--", *rel],
                             capture_output=True, text=True, check=True).stdout
        info["dirty"] = bool(stt.strip())
    except (OSError, subprocess.CalledProcessError):
        info["git_head"], info["dirty"] = None, True
    return info


def run(a) -> int:
    root = ek.repo_root()
    gt_root = Path(a.gt_root)
    C.refuse_sealed(gt_root, a.spec)
    spec_p = Path(a.spec) if Path(a.spec).is_absolute() else root / a.spec
    spec = json.loads(spec_p.read_text("utf-8"))
    boot = RT.bootstrap_settings(a, root)
    json_out = None if a.no_json else RT.out_path(a.json_out, root, ".json", is_ours=_is_ours)
    if not gt_root.is_dir():
        raise Fail(RT.EXIT_PRE, f"{gt_root}: GT root not found")
    students_e, named_e, all_t = _resolve(spec, root)
    dev_l, held_l = ek.split_ids(root)
    dev, held = set(dev_l), set(held_l)
    ids = sorted(dev_l)
    gt = rg.GTSource(DS, root=gt_root)
    gtc = RT.GTCache(gt, RT.V1Provider(), DS, len(ids) + 1)
    if rg.R2_NEAR_EMPTY is not None:
        print("composite_bound: regions.R2_NEAR_EMPTY is now fixed; the blind rule still uses R3 "
              "only until this tool is updated to R2 u R3 deliberately", file=sys.stderr)
    try:
        def mask_of(i):
            shape, _, packed = gtc.get(i)
            return shape, {r: RT.unpack(packed[r], shape) for r in REGIONS}

        missing, students, teachers, src_prov = [], {}, {}, []
        for e in students_e:
            key = e["source"]
            try:
                s = load_source(e, root, dev, held, gtc)
            except Fail as ex:
                if ex.code != RT.EXIT_PRE:
                    raise
                if key == "u2flow":
                    raise
                missing.append({"source": key, "why": str(ex)})
                continue
            if e["format"] == "run":
                mj = Path(e["path"]) / "metrics.json"
                if not mj.is_file():
                    raise Fail(RT.EXIT_PRE, f"student {key}: {mj} missing (S cross-check)")
                s["ref_fl_all"] = json.loads(mj.read_text("utf-8"))["aggregate"]["regions"][
                    "all"]["fl"]
            got = sorted(s["maps"])
            if got != ids:
                raise Fail(RT.EXIT_ERR, f"student {key}: {len(got)} dev images, need all "
                           f"{len(ids)} (paired rows only)")
            students[key] = s
            src_prov.append({"key": key, "role": "student", "label": e.get("label"),
                             "path": e["path"], "format": e["format"], "pred_sha": s["pred_sha"],
                             "n_images": len(got), **{k: v for k, v in s["prov"].items()
                                                      if v is not None}})
        named = [tkey(e) for e in named_e]
        for e in {tkey(x): x for x in named_e + all_t}.values():
            k = tkey(e)
            try:
                t = load_source(e, root, dev, held, gtc)
            except Fail as ex:
                if ex.code != RT.EXIT_PRE or k in named:
                    raise
                missing.append({"source": k, "why": str(ex)})
                continue
            teachers[k] = t
            src_prov.append({"key": k, "role": "teacher", "named": k in named, "path": e["path"],
                             "format": e["format"], "pred_sha": t["pred_sha"],
                             "n_images": len(t["maps"]), **{kk: v for kk, v in t["prov"].items()
                                                            if v is not None}})
        rows = build(students, teachers, named, ids, mask_of, boot)
    finally:
        gt.close()
    # cross-check S against each run's own metrics.json (same GT, same kernel)
    checks = {}
    for key, s in students.items():
        got = next(r for r in rows if r["student"] == key and r["arm"] == "S")
        if "ref_fl_all" in s:
            ref = s["ref_fl_all"]
            d = abs(got["regions"]["all"]["fl"] - ref)
            checks[key] = {"against": "the run's metrics.json", "metrics_json_fl_all": ref,
                           "abs_diff": d}
            if d > 1e-6:
                raise Fail(RT.EXIT_ERR, f"{key}: S Fl-all {got['regions']['all']['fl']} != "
                           f"its metrics.json {ref}: not the same kernel or GT")
        elif key == "u2flow":
            checks[key] = crosscheck_region_table(root, s["pred_sha"], got)
    esha, _ = rg.eval_sha(rg.CODE_ROOT)
    try:
        gtb = ek.gt_bundle(DS, gt_root, root)
    except (OSError, ValueError, rg.IntegrityError) as ex:
        gtb = {"name": rg.GT_BUNDLE[DS], "gt_pack_sha": None, "integrity": f"unknown: {ex}"}
    header = {
        "tool": TOOL, "idea": "IDEA-002", "standard": "measured", "split": "dev",
        "dataset": DS, "n_images": len(ids), "agg": "px-pooled",
        "metric_version": rg.METRIC_VERSION, "regions_provider": RT.V1Provider.name,
        "eval_sha": esha, "gt": gtb, "git": ek.git_info(),
        "code": code_identity(root, [HERE / "composite_bound.py", HERE / "region_table.py",
                                     spec_p]),
        "spec": str(spec_p.relative_to(root)
                                                                   if spec_p.is_relative_to(root)
                                                                   else spec_p),
        "statistics": {"unit": "image", "fl_ci": "percentile, one (B x N) draw shared by all "
                       "rows", "gain_ci": "stats.paired_image_bootstrap (Fl(S) - Fl(row))",
                       **{k: boot[k] for k in ("B", "seed", "alpha", "seed_source",
                                               "settings_source")}},
        "s6": {"R2_NEAR_EMPTY": rg.R2_NEAR_EMPTY, "blind_region": BLIND_REGION,
               "note": "R2 is withheld-S6 (regions.R2_NEAR_EMPTY is None, regions.md S6): the "
                       "oracle-blind rule uses R3 only, and no R2 number is computed or shown; "
                       "Fl-all still counts every V_occ pixel."},
        "definitions": {
            "S": "the student's prediction (oracle=false)",
            f"oracle-blind-{BLIND_REGION}[T]": f"T on {BLIND_REGION} (GT-static, occluded, "
            "out-of-frame) where T is finite, S elsewhere; GATE-1 draft (b) rule minus R2",
            "ORACLE-pixel[T]": "per pixel the better of S and T (GT-chosen upper bound)",
            "ORACLE-best-of-named": "per pixel the best of S and every named RIGID teacher "
                                    "variant (the IDEA-002 kill reads these and ORACLE-pixel)",
            NONRIGID_ORACLE: "per pixel the best of S and every teacher entry of the "
                             "teachers_spec, incl. RAFT C+T (non-rigid), Monodepth2, COLMAP: "
                             "context only, never the kill read",
            "train-like": "counts pooled over w03-11 and w10-18 (each pair twice; S twice)",
            "teacher fallback": "a teacher's missing image or non-finite pixel uses S",
            "exactness": "outlier is monotone in EPE at fixed GT, so the per-pixel min-EPE "
                         "choice's outlier map is out_S AND out_T (exact for Fl)"},
        "students": {p["key"]: {"label": p.get("label")} for p in src_prov
                     if p["role"] == "student"},
        "named_teachers": named, "sources": src_prov, "missing": missing,
        "student_crosscheck": checks,
        "ledger": "no results row (hard rule 7); record as note.py evidence --kind measurement",
    }
    header["kill_read"] = kill_read(rows, "u2flow")
    doc = RT._jsonable({"header": header, "rows": rows})
    if json_out:
        ek.atomic_write(json_out, json.dumps(doc, indent=1) + "\n")
    sys.stdout.write(render(doc) + "\n")
    if json_out:
        print(f"COMPOSITE_BOUND rows={len(rows)} json={json_out}", file=sys.stderr)
    return RT.EXIT_OK


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--spec", default=DEFAULT_SPEC)
    ap.add_argument("--gt-root", default="/Volumes/Shin/U2FLOW_data/gt")
    ap.add_argument("--json-out", default=DEFAULT_JSON)
    ap.add_argument("--no-json", action="store_true")
    ap.add_argument("--B", type=int)
    ap.add_argument("--alpha", type=float)
    ap.add_argument("--seed", type=int)
    try:
        a = ap.parse_args(argv)
    except SystemExit as ex:
        return RT.EXIT_USAGE if ex.code else RT.EXIT_OK
    try:
        return run(a)
    except Fail as e:
        print(f"composite_bound: {'REFUSED (seal)' if e.code == RT.EXIT_SEAL else 'error'}: {e}",
              file=sys.stderr)
        return e.code
    except C.Stop as e:
        print(f"composite_bound: {e}", file=sys.stderr)
        return e.code
    except rg.IntegrityError as e:
        print(f"composite_bound: integrity: {e}", file=sys.stderr)
        return RT.EXIT_INTEGRITY
    except (OSError, ValueError, KeyError) as e:
        print(f"composite_bound: {type(e).__name__}: {e}", file=sys.stderr)
        return RT.EXIT_ERR


if __name__ == "__main__":
    sys.exit(main())
