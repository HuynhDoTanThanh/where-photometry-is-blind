#!/usr/bin/env python3
"""selftri_vs_ff.py: IDEA-005. Does the frozen IDEA-004 rule (EV-0402 / EV-0403) add anything on
top of U2Flow's own +FF fusion (arXiv:2604.10056 sec 3.4; teardown
research/teardowns/2026-10-08-uncertainty-gated-rigid-refit.md)? Cached flows only, CPU on the
Mac; KITTI-15 dev half and KITTI-12 training; it writes no ledger row (hard rule 7), starts no VM,
tunes nothing, and edits nothing in tools/eval.

Arms (per image, same GT, same region masks; every arm covers every pixel)
    S     U2Flow kitti.pth, two-frame (cache u2flow, native_10to11; W3-02 spec)
    FF    U2Flow +FF: the per-image TinyModel fusion of train_fusion_kitti.py (thr 35, iters
          2000, lr 0.01, base_seed 1; cache u2flow_ff, flow_key 'native'; the SAME cache the W3-02
          region table scored, tools/p1/specs/region_table.w3-02.json). '3-frame + per-image
          optimisation' (metric s8); never a two-frame method
    R     the frozen rule on S: arbitration_cv.frozen_image with S as the student. Must equal
          reports/p1_region/arbitration_frozen_k12.json (EV-0403; its K15 part is EV-0402) EXACTLY
          (values, gains, CIs, switched counts), and arbitration_cv.frozen_repro (EV-0402) again
    FF+R  the SAME frozen rule with the +FF flow as the student flow: switch to T = selftri (the
          render triangulated from kitti.pth's own 10->09 / 10->11 flows) iff T is finite AND
          kitti.pth's alpha_10to11 > 2.75 AND |T - FF| <= 100 px. alpha stays kitti.pth's (+FF
          has no uncertainty output); the distance is to the flow being replaced. FROZEN_*
          constants unchanged, no retuning: this is arbitration_cv.frozen_image called with
          {'flow': FF, 'alpha': alpha_S}
Comparisons (gain = base - arm, positive = the arm is better; paired image bootstrap,
tools/eval/stats.py through arbitration_cv.frozen_stats; every comparison uses the same seed and
n, so the same image draws)
    FF vs S, R vs S, FF+R vs FF (the IDEA-005 kill quantity), R vs FF (direct, paired)
Statistics: Fl px-pooled (KITTI-12: Fl(K15-rule)), KITTI-12 Out-All 3 px, EPE px-pooled and
img-mean; regions arbitration_cv.FROZEN_REGIONS (K15 all/R1/R3; K12 all/noc/occ/K12-R1/K12-R3).
Kill (IDEA-005): the Fl 'all' gain of FF+R over FF < 0.2 pp on KITTI-15 dev OR on KITTI-12, or its
CI includes 0, on either: the rule is subsumed by +FF. Out-All is reported, not part of the kill.
Selection: every KITTI-15 dev row is in-sample-OPTIMISTIC (EV-0402 chose 2.75 and the band on
those 100 images); KITTI-12 is the only out-of-sample (clean) test. Rows, kill and tables say so.

How to invoke (repo root, the Mac venv):
    .venv/bin/python tools/p1/selftri_vs_ff.py --seed 20261008 --B 10000 \
        [--json-out reports/p1_region/selftri_vs_ff.json | --no-json]
    Held-out ARB test (prereg PRIMARY.json; only after unseal.py recorded the batch):
    .venv/bin/python tools/p1/selftri_vs_ff.py --heldout-batch A --run rel=RUN --run arb=RUN \
        --run ff-s1=RUN ... --run ffarb-s3=RUN     (inputs default to the pinned paths)
    It refuses (50) before opening anything unless experiments/unseal.jsonl (read through
    ledger.py, chain verified) holds the batch AND prereg/PRIMARY.json is the file that row
    recorded AND the prereg-v3 tag holds those bytes at the row's tag_commit (no path option:
    fixed files only; git runs with GIT_* scrubbed). A batch already looked at (its output files,
    its committed .started marker, a held-out result row or an arb-heldout-<batch> evidence row)
    is refused (50): a repeat is a second look. The marker is created with O_EXCL and committed
    just before the first GT read. Then
    every pin must be a full sha256 and match (exit 7): code (HO_PINNED, frozen_rule), inputs
    (inputs_sha256), and prereg/arb_data_manifest.json with every file it lists; the --run ids
    must be pins.registration's. Before any contrast, the validity checks run (100 images, the
    rel level against the GATE-0 derivation); a failure writes the declared 'invalid' outcome
    and exits 7. Only the rows PRIMARY declares for the batch are computed; aggregates and
    stamped rows only, with every field note.py requires. It writes nothing to the ledgers:
    it prints the exact `note.py result --from-eval` and `note.py evidence` commands (EV-0615).
    --check (EV-0610) runs every gate, pin, coverage, id and run-id check on the real paths and
    stops before any GT read; steps not yet taken (unseal, the tag, registration) print PENDING.
Exit codes: region_table.py's (0 ok, 1 error or a failed reproduction, 2 precondition,
7 integrity, 50 seal, 64 usage).

Why it is built this way.
- One kernel: the switch, the per-image counts, the statistics, the student loader, the seal
  checks and the EV-0402 reproduction are arbitration_cv / composite_bound / region_table
  functions called as they are; this file only adds the FF arm and the comparisons.
- The reproduction checks make the R arm and the FF arm identical to what EV-0402/0403 and the
  W3-02 region table already measured, so a drift in any input shard or kernel stops the run.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import arbitration_cv as AC  # noqa: E402

CB, RT, ek, rg, st, C = AC.CB, AC.RT, AC.ek, AC.rg, AC.st, AC.C
Fail = RT.Fail
TOOL = "tools/p1/selftri_vs_ff.py"
IDEA = "IDEA-005"
DEFAULT_JSON = "reports/p1_region/selftri_vs_ff.json"
FF_SPEC = "tools/p1/specs/region_table.w3-02.json"
FF_SOURCE = "u2flow_ff"
EV0403_JSON = AC.FROZEN_JSON
EV0403_SHA = "8b0e72de81ec9af79b465fe8086036d54809837cc9a7f248cf286bc4122f7057"  # evidence.jsonl
KILL_PP = 0.2                                     # IDEA-005 kill (registered before any number)
ARMS = ("S", "FF", "R", "FF+R")
COMPARISONS = (("FF vs S", "S", "FF"), ("R vs S", "S", "R"),
               ("FF+R vs FF", "FF", "FF+R"), ("R vs FF", "FF", "R"))
KILL_CMP = "FF+R vs FF"
# Selection status of each dataset's rows (review SELFTRIFF finding 1, as 1ea5011 for
# arbitration_robust): the rule's alpha threshold and band were chosen on the 100 KITTI-15 dev
# images (EV-0402), so every K15 dev row is in-sample; KITTI-12 is the only out-of-sample test.
SELECTION = {"kitti15": f"in-sample-OPTIMISTIC: {AC.FROZEN_ALPHA_GT:g} and the band were chosen "
                        "on these 100 images (EV-0402)",
             "kitti12": "out-of-sample: the rule was frozen (IDEA-004) before any KITTI-12 number"}
SELECTION_TAG = {"kitti15": "IN-SAMPLE-OPTIMISTIC", "kitti12": "out-of-sample"}


# ------------------------------------------------------------------------------ per image
def image_arms(s: dict, ff: np.ndarray, flow_t, shape, flow_occ, masks: dict, ds: str,
               regions) -> tuple[dict, dict]:
    """({arm: counts}, {rule arm: switched-pixel counts}) of one image. s = {'flow', 'alpha'} of
    kitti.pth; ff the +FF flow; flow_t None = no selftri render (R = S, FF+R = FF)."""
    cs, cr, use_r = AC.frozen_image(s, flow_t, shape, flow_occ, masks, ds, regions)
    cf, cfr, use_fr = AC.frozen_image({"flow": ff, "alpha": s["alpha"]}, flow_t, shape, flow_occ,
                                      masks, ds, regions)
    return {"S": cs, "FF": cf, "R": cr, "FF+R": cfr}, {"R": use_r, "FF+R": use_fr}


# ------------------------------------------------------------------------------ loading
def ff_entry(root: Path) -> dict:
    spec = json.loads((root / FF_SPEC).read_text("utf-8"))
    hit = [e for e in spec["sources"] if e["source"] == FF_SOURCE]
    if len(hit) != 1:
        raise Fail(RT.EXIT_PRE, f"{FF_SOURCE}: not exactly once in {FF_SPEC}")
    return hit[0]


def ff_provenance(path: Path) -> dict:
    """The +FF producer's settings, read from the cache manifest (seed rule, FF script, ckpt)."""
    try:
        m = json.loads((path / "manifest.json").read_text("utf-8"))
    except (OSError, ValueError):
        return {"manifest": None}
    s = (m.get("identity") or {}).get("settings", {}).get("ff", {})
    return {"manifest": str(path / "manifest.json"), "base_seed": s.get("base_seed"),
            "seed_rule": s.get("seed_rule"), "ff": s.get("ff"), "ckpt_sha256": s.get("ckpt_sha256"),
            "gpu_name": (m.get("host") or {}).get("gpu_name"), "complete": m.get("complete")}


def dataset_run(ds: str, root: Path, gt_root: Path, s_entry: dict, f_entry: dict, t_entry: dict,
                boot: dict) -> dict:
    """The four arms on one dataset: KITTI-12 ids 0-193 or the KITTI-15 dev half (region_table's
    seal checks run inside open_entry before any pixel is read)."""
    dev_l, held_l = ek.split_ids(root)
    dev, held = set(dev_l), set(held_l)
    ids = sorted(dev_l) if ds == "kitti15" else list(ek.K12_IDS)
    regions = AC.FROZEN_REGIONS[ds]
    gt = rg.GTSource(ds, root=gt_root)
    gtc = RT.GTCache(gt, RT.V1Provider(), ds, 0)
    try:
        S = AC.load_student(s_entry, root, dev if ds == "kitti15" else set(ids), held, gtc, ds=ds,
                            with_fb=False)
        if sorted(S["maps"]) != ids:
            raise Fail(RT.EXIT_ERR, f"u2flow {ds}: {len(S['maps'])} images, need {len(ids)}")
        _, fstreams = RT.open_entry({**f_entry, "datasets": [ds]}, root, dev, held)
        if ds not in fstreams:
            raise Fail(RT.EXIT_PRE, f"{f_entry['path']}: holds no {ds} predictions")
        ffac, ff_sha, _ = fstreams[ds]
        FF = {}
        for seq, flow in ffac():
            if seq not in S["maps"]:
                raise Fail(RT.EXIT_ERR, f"+FF {ds} {seq}: not an id of the split")
            FF[seq] = np.asarray(flow, np.float32)
        if sorted(FF) != ids:
            raise Fail(RT.EXIT_ERR, f"+FF {ds}: {len(FF)} images, need {len(ids)}")
        _, tstreams = RT.open_entry({**t_entry, "datasets": [ds]}, root, dev, held)
        if ds not in tstreams:
            raise Fail(RT.EXIT_PRE, f"{t_entry['path']}: holds no {ds} predictions")
        tfac, t_sha, _ = tstreams[ds]
        per = {}

        def one(seq, flow_t):
            shape, flow_occ, packed = gtc.get(seq)
            m = {r: RT.unpack(packed[r], shape) for r in regions}
            per[seq] = image_arms(S["maps"][seq], FF.pop(seq), flow_t, shape, flow_occ, m, ds,
                                  regions)
            S["maps"][seq] = None

        for seq, flow_t in tfac():
            if seq not in S["maps"]:
                raise Fail(RT.EXIT_ERR, f"teacher {ds} {seq}: not an id of the split")
            one(seq, flow_t)
        no_t = [i for i in ids if i not in per]
        for seq in no_t:
            one(seq, None)
    finally:
        gt.close()
    arms = {a: [per[i][0][a] for i in ids] for a in ARMS}
    switched = {}
    for a in ("R", "FF+R"):
        use = {k: sum(per[i][1][a][k] for i in ids) for k in per[ids[0]][1][a]}
        use["n_valid"] = sum(p["all"]["n"] for p in arms["S"])
        switched[a] = {"valid_pct": 100.0 * use["sw_all"] / use["n_valid"],
                       "image_pct": 100.0 * use["sw_img"] / use["px"], "counts": use}
    stats = AC.FROZEN_STATS[ds]
    values = {}
    for a in ARMS:
        values[a] = {}
        for r in regions:
            arr = RT.region_arrays(arms[a], r)
            values[a][r] = {"n_images": int((arr["n"] > 0).sum()), "n_pixels": int(arr["n"].sum()),
                            **{k: st.statistic(arr, k) for k in stats}}
    comps = {name: AC.frozen_stats(arms[b], arms[x], regions, stats, boot, ds)
             for name, b, x in COMPARISONS}
    for c in comps.values():                       # frozen_stats names its two arms S / frozen
        for rec in c.values():
            for k in stats:
                if k in rec:
                    rec[k]["base"] = rec[k].pop("S")
                    rec[k]["arm"] = rec[k].pop("frozen")
    try:
        gtb = ek.gt_bundle(ds, gt_root, root)
    except (OSError, ValueError, rg.IntegrityError) as ex:
        gtb = {"name": rg.GT_BUNDLE[ds], "gt_pack_sha": None, "integrity": f"unknown: {ex}"}
    return {"dataset": ds, "split": RT.SPLIT_OF[ds], "selection": SELECTION[ds],
            "n_images": len(ids),
            "teacher_images": len(ids) - len(no_t), "teacher_missing_ids": no_t, "gt": gtb,
            "pred_sha": {"S": S["pred_sha"], "FF": ff_sha, "T": t_sha}, "values": values,
            "comparisons": comps, "switched": switched}


# ------------------------------------------------------------------------------ checks
def repro_ev0403(res: dict, ev: dict) -> dict:
    """The R arm (R vs S) must equal the frozen-mode JSON's results for this dataset EXACTLY:
    S and composite values, gains, CIs, n, switched counts, and the S / T shards."""
    ds = res["dataset"]
    (ref,) = [r for r in ev["results"] if r["dataset"] == ds]
    if (ref["S_pred_sha"], ref["T_pred_sha"]) != (res["pred_sha"]["S"], res["pred_sha"]["T"]):
        raise Fail(RT.EXIT_ERR, f"EV-0403 {ds}: S or teacher pred_sha differs from this run's")
    got = res["comparisons"]["R vs S"]
    n = 0
    for r, want in ref["regions"].items():
        mine = got[r]
        for k in ("n_images", "n_pixels"):
            if want[k] != mine[k]:
                raise Fail(RT.EXIT_ERR, f"EV-0403 {ds} {r} {k}: {mine[k]} != {want[k]}")
        for stat in AC.FROZEN_STATS[ds]:
            w, g = want[stat], mine[stat]
            pairs = [(w["S"], g["base"]), (w["frozen"], g["arm"]), (w["gain"], g["gain"]),
                     *zip(w["gain_ci"] or [], g["gain_ci"] or [], strict=True)]
            for x, y in pairs:
                if x != y:
                    raise Fail(RT.EXIT_ERR, f"EV-0403 {ds} {r} {stat}: {y} != {x} (reproduction)")
                n += 1
    if ref["switched"]["counts"] != res["switched"]["R"]["counts"]:
        raise Fail(RT.EXIT_ERR, f"EV-0403 {ds}: switched counts differ")
    return {"against": EV0403_JSON, "dataset": ds, "values_compared": n, "exact": True}


def ff_crosscheck(root: Path, res: dict) -> dict:
    """The FF arm's R1 / R3 Fl (and K12 Out3) against the W3-02 region table's u2flow_ff cells
    for the same pred_sha (composite_bound.crosscheck_region_table)."""
    ds = res["dataset"]
    names = {"R1": "R1", "R3": "R3"} if ds == "kitti15" else {"R1": "K12-R1", "R3": "K12-R3"}
    v = res["values"]["FF"]
    row = {"regions": {m: {"fl": v[m]["fl"], "n_pixels": v[m]["n_pixels"],
                           **({"out3": v[m]["out3"]} if ds == "kitti12" else {})}
                       for m in names.values()}}
    got = CB.crosscheck_region_table(root, res["pred_sha"]["FF"], row, ds=ds, names=names,
                                     out3=ds == "kitti12", source=FF_SOURCE)
    if got.get("against") is None:
        raise Fail(RT.EXIT_ERR, f"+FF {ds}: no W3-02 region table with pred_sha "
                   f"{res['pred_sha']['FF']}: the FF cache is not the one W3-02 scored")
    return got


def kill_read(results: list[dict]) -> dict:
    """IDEA-005's kill on the Fl 'all' gain of FF+R over FF: fires if < 0.2 pp, or its CI
    includes 0, on KITTI-15 dev OR on KITTI-12 (as registered). No CI (B = 0) and no gain below
    the bar = not evaluated (fires None). Each dataset carries its selection status: the K15 dev
    read is in-sample-OPTIMISTIC, KITTI-12 is the only clean (out-of-sample) test."""
    reasons, readable, per = [], True, {}
    for res in results:
        ds = res["dataset"]
        rec = res["comparisons"][KILL_CMP]["all"]["fl"]
        per[ds] = {"gain_pp": rec["gain"], "gain_ci": rec["gain_ci"], "label": rec["label"],
                   "selection": SELECTION[ds]}
        if rec["gain"] < KILL_PP:
            reasons.append(f"{ds} gain {rec['gain']:.3f} pp < {KILL_PP} pp")
        ci = rec["gain_ci"]
        if ci is None:
            readable = False
        elif ci[0] <= 0.0 <= ci[1]:
            reasons.append(f"{ds} CI [{ci[0]:.3f}, {ci[1]:.3f}] includes 0")
    fires = True if reasons else (False if readable else None)
    verdict = {True: "FIRES (" + "; ".join(reasons) + ")", False: "does not fire",
               None: "NOT EVALUATED (no CI: B = 0)"}[fires]
    text = ("Fl 'all' gain of FF+R over FF: "
            + ", ".join(f"{d} [{SELECTION_TAG[d]}] {p['gain_pp']:.3f} pp"
                        + (f" [{p['gain_ci'][0]:.3f}, {p['gain_ci'][1]:.3f}]"
                           if p["gain_ci"] else "")
                        for d, p in per.items())
            + f": kill {verdict}. KITTI-12 is the only clean (out-of-sample) test; the KITTI-15 "
            f"dev read is in-sample-OPTIMISTIC ({AC.FROZEN_ALPHA_GT:g} and the band were chosen "
            "on those 100 images, EV-0402)")
    return {"comparison": KILL_CMP, "threshold_pp": KILL_PP, "fires": fires, "readable": readable,
            "reasons": reasons, "per_dataset": per, "out_of_sample": "kitti12 only",
            "text": text}


# ------------------------------------------------------------------------------ output
def render(doc: dict) -> str:
    h = doc["header"]
    b = h["statistics"]
    out = [f"# IDEA-005 frozen selftri rule vs U2Flow +FF ({h['rule']['text_ff']})", "",
           f"Gain = base - arm (positive = arm better); paired image bootstrap "
           f"{round(100 * (1 - b['alpha']))}% CI (B={b['B']}, seed {b['seed']}).", ""]
    for res in doc["results"]:
        ds = res["dataset"]
        sw = res["switched"]
        out += [f"## {ds} [{SELECTION_TAG[ds]}] ({res['split']}, n={res['n_images']}, selftri on "
                f"{res['teacher_images']}); {SELECTION[ds]}; switched (% GT-valid px): R "
                f"{sw['R']['valid_pct']:.2f}, FF+R {sw['FF+R']['valid_pct']:.2f}", "",
                "| region | metric | " + " | ".join(ARMS) + " | "
                + " | ".join(c[0] for c in COMPARISONS) + " |",
                "|---" * (2 + len(ARMS) + len(COMPARISONS)) + "|"]
        stats = [k for k in AC.FROZEN_STATS[ds] if k in ("fl", "out3")]
        for r in AC.FROZEN_REGIONS[ds]:
            for k in stats:
                cells = [f"{res['values'][a][r][k]:.3f}" for a in ARMS]
                for name, _, _ in COMPARISONS:
                    e = res["comparisons"][name][r][k]
                    ci = e["gain_ci"]
                    cells.append(f"{e['gain']:.3f}" + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci
                                                       else ""))
                label = res["comparisons"]["R vs S"][r][k]["label"]
                out.append(f"| {r} | {label} | " + " | ".join(cells) + " |")
        out.append("")
    out += [f"Reproduction: {h['reproduction']['summary']}",
            f"Kill read (IDEA-005, user decides): {h['kill_read']['text']}", ""]
    return "\n".join(out)


def _is_ours(p: Path, suffix: str) -> bool:
    try:
        return json.loads(p.read_text("utf-8")).get("header", {}).get("tool") == TOOL
    except (OSError, ValueError, AttributeError):
        return False


def run(a) -> int:
    root = ek.repo_root()
    gt_root = Path(a.gt_root)
    C.refuse_sealed(gt_root, CB.DEFAULT_SPEC)
    boot = RT.bootstrap_settings(a, root)
    json_out = (None if a.no_json else
                RT.out_path(a.json_out or DEFAULT_JSON, root, ".json", is_ours=_is_ours))
    if not gt_root.is_dir():
        raise Fail(RT.EXIT_PRE, f"{gt_root}: GT root not found")
    ev_p = root / EV0403_JSON
    ev = json.loads(ev_p.read_text("utf-8"))
    eh = ev["header"]
    if eh.get("mode") != AC.FROZEN_MODE or (eh["rule"]["alpha_gt"], eh["rule"]["d_le_px"],
                                            eh["rule"]["teacher"]) != (
            AC.FROZEN_ALPHA_GT, AC.FROZEN_D_LE_PX, AC.FROZEN_TEACHER):
        raise Fail(RT.EXIT_ERR, f"{ev_p}: not the frozen IDEA-004 output")
    if (eh["statistics"]["B"], eh["statistics"]["seed"], eh["statistics"]["alpha"]) != (
            boot["B"], boot["seed"], boot["alpha"]):
        raise Fail(RT.EXIT_USAGE, f"B, seed and alpha must equal EV-0403's "
                   f"({eh['statistics']['B']}, {eh['statistics']['seed']}, "
                   f"{eh['statistics']['alpha']}) for the exact reproduction")
    spec_p = root / CB.DEFAULT_SPEC
    spec = json.loads(spec_p.read_text("utf-8"))
    students_e, named_e, _ = CB._resolve(spec, root)
    (t_entry,) = [e for e in named_e if CB.tkey(e) == AC.FROZEN_TEACHER]
    s_entry = next(e for e in students_e if e["source"] == "u2flow")
    f_entry = ff_entry(root)
    results = [dataset_run(ds, root, gt_root, s_entry, f_entry, t_entry, boot)
               for ds in ("kitti12", "kitti15")]
    k12, k15 = results
    repro = {"ev0403": [repro_ev0403(r, ev) for r in results]}
    k15_r = {r: k15["comparisons"]["R vs S"][r] for r in AC.REGIONS}
    k15_r = {r: {"fl": {"gain": v["fl"]["gain"], "gain_ci": v["fl"]["gain_ci"]}}
             for r, v in k15_r.items()}
    repro["ev0402"] = AC.frozen_repro(root, k15_r, k15["switched"]["R"]["counts"],
                                      k15["pred_sha"]["S"], k15["pred_sha"]["T"], boot)
    repro["ff_w3_02"] = [ff_crosscheck(root, r) for r in results]
    ev_sha = C.sha256_file(ev_p)
    repro["ev0403_json_sha256"] = {"here": ev_sha, "ev0403": EV0403_SHA,
                                   "equal": ev_sha == EV0403_SHA}
    repro["summary"] = (f"R arm equals EV-0403 exactly on both datasets "
                        f"({sum(x['values_compared'] for x in repro['ev0403'])} values) and "
                        f"EV-0402 (K15 dev Fl-all gain {repro['ev0402']['gain_pp_all']['here']:.4f}"
                        f" pp); FF arm R1/R3 Fl (K12 Out3) equal the W3-02 u2flow_ff cells; "
                        f"EV-0403 JSON sha256 {'matches' if ev_sha == EV0403_SHA else 'DIFFERS'}")
    esha, _ = rg.eval_sha(rg.CODE_ROOT)
    header = {
        "tool": TOOL, "idea": IDEA, "standard": "measured",
        "rule": {"teacher": AC.FROZEN_TEACHER, "alpha_gt": AC.FROZEN_ALPHA_GT,
                 "d_le_px": AC.FROZEN_D_LE_PX, "d_gt_px": None,
                 "source": "IDEA-004 frozen rule (EV-0402 in-sample choice), arbitration_cv "
                           "FROZEN_* unchanged; no tuning, no retuning for FF",
                 "text_s": f"R: T where T finite and alpha_S > {AC.FROZEN_ALPHA_GT:g} and "
                           f"|T - S| <= {AC.FROZEN_D_LE_PX:g} px, else S",
                 "text_ff": f"FF+R: T where T finite and alpha_S > {AC.FROZEN_ALPHA_GT:g} and "
                            f"|T - FF| <= {AC.FROZEN_D_LE_PX:g} px, else FF",
                 "ff_note": "alpha is kitti.pth's alpha_10to11 (+FF has no uncertainty); T = "
                            "selftri, triangulated from kitti.pth's own 10->09 and 10->11 flows "
                            "(not from +FF); the distance is to the flow being replaced (FF)"},
        "arms": {"S": {"source": "u2flow kitti.pth", "path": s_entry["path"],
                       "flow_key": s_entry.get("flow_key")},
                 "FF": {"source": FF_SOURCE, "spec": FF_SPEC, "path": f_entry["path"],
                        "flow_key": f_entry.get("flow_key"),
                        "label": "+FF: 3-frame + per-image optimisation (metric s8); not a "
                                 "two-frame method, not at the student's inference cost",
                        "producer": ff_provenance(Path(f_entry["path"]))},
                 "T": {"key": AC.FROZEN_TEACHER, "path": t_entry["path"],
                       "derived_from": AC.DERIVED_FROM["selftri/native"]},
                 "frames": "S: 10,11; FF: 09,10,11 (+ per-image optimisation); R: 09,10,11 "
                           "(selftri uses kitti.pth's 10->09); FF+R: 09,10,11"},
        "selection": {**SELECTION, "out_of_sample_rows": "kitti12 only (each result carries "
                                                          "'selection')"},
        "comparisons": {name: f"gain = {b} - {x} (positive = {x} better)"
                        for name, b, x in COMPARISONS},
        "metric_version": rg.METRIC_VERSION, "eval_sha": esha, "git": ek.git_info(),
        "regions_provider": RT.V1Provider.name,
        "regions_note": "as arbitration_cv frozen mode: K15 dev all, R1, R3 (R2 withheld-S6); "
                        "K12 all, noc, occ, K12-R1, K12-R3 (noc scored against flow_occ, "
                        "V1Provider; Out3 on noc is NOT the official Out-Noc)",
        "statistics": {"unit": "image", "gain": "base - arm, stats.paired_image_bootstrap via "
                       "arbitration_cv.frozen_stats; same seed and n for every comparison, so "
                       "the same image draws",
                       **{k: boot[k] for k in ("B", "seed", "alpha", "seed_source",
                                               "settings_source")}},
        "code": CB.code_identity(root, [HERE / "selftri_vs_ff.py", HERE / "arbitration_cv.py",
                                        HERE / "composite_bound.py", HERE / "region_table.py",
                                        spec_p, root / FF_SPEC]),
        "reproduction": repro,
        "inputs_never_read": "KITTI-15 held-out and all-200 (region_table seal checks)",
        "ledger": "no results row (hard rule 7); record as note.py evidence --kind measurement",
    }
    header["kill_read"] = kill_read(results)
    doc = RT._jsonable({"header": header, "results": results})
    if json_out:
        ek.atomic_write(json_out, json.dumps(doc, indent=1) + "\n")
    sys.stdout.write(render(doc) + "\n")
    return RT.EXIT_OK


# ------------------------------------------------------------------------------ held-out (ARB)
# The confirmatory ARB test (prereg PRIMARY.json, batch A): the frozen rule on the sealed
# KITTI-15 held-out half against the authors' kitti.pth and +FF. Scoped to ONE unsealed batch:
# only the rows PRIMARY declares for it are computed (critic EV-0573), and nothing runs before
# the batch's unseal row, PRIMARY's frozen status, and every code and input pin match (EV-0569,
# EV-0570). Aggregates only: no per-image value or held-out mask is written (seal.py docstring).
# The gate reads only fixed paths (code review ARB-HELDOUT B1): the unseal rows from
# experiments/unseal.jsonl through ledger.py's hash-chain check, PRIMARY from prereg/PRIMARY.json,
# which must be the bytes the unseal row recorded and the bytes the prereg-v3 tag holds.
REPO = HERE.parents[1]
DATA = Path("/Volumes/Shin/U2FLOW_data")
PRIMARY_REL = "prereg/PRIMARY.json"
DATA_MANIFEST_REL = "prereg/arb_data_manifest.json"
HO_TAG = "prereg-v3"
HO_DEFAULTS = {
    "s_preds": DATA / "runs/EXP-002-audit-s1/out/preds",
    "ff_preds": [DATA / f"runs/EXP-003-ff-s{s}/out/preds" for s in (1, 2, 3)],
    "t_preds": DATA / "work/p1render/heldout/selftri_native_w03-11/kitti15/preds",
    "alpha_cache": DATA / "runs/p1-dump-eval-u2flow-r1/out/cache/diag/u2flow",
    "selftri_cache": DATA / "derived/evalselftri/v2",
}
FF_SEEDS = ("1", "2", "3")
HO_ARMS = ("rel", "arb", "ff", "ffarb")
# every file whose bytes decide a held-out number; PRIMARY analysis_code.code_sha256 pins each
HO_PINNED = ("tools/p1/selftri_vs_ff.py", "tools/p1/arbitration_cv.py",
             "tools/p1/composite_bound.py", "tools/p1/region_table.py",
             "tools/p1/render_heldout.py", "tools/p1/render_frig.py", "tools/p1/rigid.py",
             "tools/p1/p1lib/geomio.py", "tools/p1/p1lib/cache.py", "tools/p1/p1lib/common.py",
             "tools/p1/p1lib/kitti.py",   # parse_id maps alpha items to ids (review minor 12)
             "tools/eval/stats.py", "tools/eval/regions.py", "tools/eval/eval_kitti.py",
             "core/utils/flow_utils.py", "tools/prereg/seal.py")
HO_RULE_PINNED = ("tools/p1/self_triangulate.py",)        # PRIMARY frozen_rule.code_sha256
HO_BUILT_ROWS = {("Fl", "all", "px-pooled", "final")}   # what batch A declares; B is not built
N_HELDOUT = 100
# the rel-level GT-integrity check (critic EV-0575): held-out Fl of rel = (all-200 - dev) by counts
REL_LEVEL_ROWS = {"full200": "RES-0032", "dev": "RES-0001"}
REL_LEVEL_TOL_PP = 1e-6
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _seal():
    import importlib.util  # noqa: PLC0415

    spec = importlib.util.spec_from_file_location("u2f_seal", REPO / "tools/prereg/seal.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ledger():
    sys.path.insert(0, str(HERE.parents[1] / "tools" / "lib"))
    import ledger  # noqa: PLC0415

    return ledger


class MemGT(rg.GTSource):
    """Held-out GT served from seal.heldout_members' in-memory bytes; never written anywhere.
    No disp_occ_0 or calibration is requested, so R4 stays unset (batch A scores 'all' only)."""

    def __init__(self, members: dict) -> None:   # {(gt dir, image id): png bytes}
        self.dataset, self.zip_prefix, self.root, self._zip = "kitti15", "", None, None
        self.listing, self.bundle_dir, self.verified = None, None, True
        self.require_r4, self.calib, self._mem = False, None, members

    def read_bytes(self, d: str, img_id: str) -> bytes:
        return self._mem[(d, img_id)]

    def close(self) -> None:
        self._mem.clear()


def _pred_sha(pred_dir: Path) -> str:
    return ek.pred_sha(json.loads((pred_dir / "PREDS.json").read_text("utf-8"))["shards"])


def _git(*args: str) -> subprocess.CompletedProcess | None:
    """git on REPO with GIT_* scrubbed from the environment, so GIT_DIR or GIT_WORK_TREE cannot
    point the tag checks at another repository (review minor 13)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    try:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, timeout=30,
                              env=env)
    except (OSError, subprocess.TimeoutExpired):
        return None


def heldout_gate(batch: str, check: bool = False) -> tuple[list[dict], dict, list[str]]:
    """(unseal rows, PRIMARY, pending) or Fail(50). Nothing the caller chooses decides it: the
    unseal rows come from the fixed ledger through ledger.py (a forged or edited file breaks its
    chain), and PRIMARY must be the bytes that row recorded AND the bytes the prereg-v3 tag holds,
    at the commit the row recorded. check=True (--check, which never reads GT) reports a step that
    has not happened yet (no unseal row, PRIMARY not frozen, no tag) as pending instead (EV-0610);
    a recorded row that disagrees still fails."""
    led = _ledger()
    pending: list[str] = []
    try:
        rows = led.Ledger(REPO).read("unseal")
    except led.LedgerError as e:
        raise Fail(RT.EXIT_SEAL, f"experiments/unseal.jsonl does not verify: {e}") from None
    mine = [r for r in rows if r.get("batch") == batch]
    row = mine[0] if mine else None
    if row is None:
        if not check:
            raise Fail(RT.EXIT_SEAL, f"batch {batch!r} is not unsealed (experiments/unseal.jsonl):"
                       " tools/prereg/unseal.py first")
        pending.append(f"no unseal row for batch {batch}")
    try:
        raw = (REPO / PRIMARY_REL).read_bytes()
        primary = json.loads(raw)
    except (OSError, ValueError) as e:
        raise Fail(RT.EXIT_SEAL, f"{PRIMARY_REL}: {e}") from None
    if row is not None and row.get("primary_sha") != hashlib.sha256(raw).hexdigest():
        raise Fail(RT.EXIT_SEAL, f"{PRIMARY_REL} is not the file unseal.py recorded for batch "
                   f"{batch} ({row.get('id')})")
    frozen = primary.get("status") == "frozen" and primary.get("frozen_by_tag") == HO_TAG
    if not frozen or (row is not None and row.get("tag") != HO_TAG):
        why = (f"{PRIMARY_REL} must be frozen by {HO_TAG}, and batch {batch} unsealed under it "
               f"(status {primary.get('status')!r}, frozen_by_tag "
               f"{primary.get('frozen_by_tag')!r}, unseal tag "
               f"{row.get('tag') if row is not None else None!r})")
        if not check or row is not None:
            raise Fail(RT.EXIT_SEAL, why)
        pending.append(why)
    r = _git("show", f"refs/tags/{HO_TAG}:{PRIMARY_REL}")
    if r is None or r.returncode != 0 or r.stdout != raw:
        why = f"tag {HO_TAG} does not hold this exact {PRIMARY_REL}"
        if not check or row is not None:
            raise Fail(RT.EXIT_SEAL, why)
        pending.append(why)
    elif row is not None:
        c = _git("rev-list", "-n", "1", f"refs/tags/{HO_TAG}")
        commit = c.stdout.decode().strip() if c is not None and c.returncode == 0 else ""
        if not commit or row.get("tag_commit") != commit:
            raise Fail(RT.EXIT_SEAL, f"the unseal row's tag_commit is not the commit {HO_TAG} "
                       f"points to ({row.get('id')}): the tag moved after the unseal")
    return rows, primary, pending


def _level(primary: dict, comp: str) -> float:
    if comp == "ARB":
        return float(primary["primary_endpoint"]["ci"]["level"])
    (sec,) = [s for s in primary["secondaries"] if s["id"] == comp]
    if "ci_level" not in sec:
        raise Fail(RT.EXIT_PRE, f"PRIMARY {comp}: no ci_level (a step-down method such as Holm "
                   "is not built here: declare per-comparison levels)")
    return float(sec["ci_level"])


def _hex_pins(pins: dict, where: str, bad: list) -> dict:
    """The pins of one block; a value that is not a full sha256 is itself a failed pin."""
    for k, v in pins.items():
        if not k.startswith("_") and not (isinstance(v, str) and HEX64.match(v)):
            bad.append(f"{where}[{k}]: {v!r} is not a sha256 (64 lowercase hex)")
    return {k: v for k, v in pins.items() if not k.startswith("_")}


def _check_data_manifest(primary: dict, a, inputs: dict, held: list[str], bad: list) -> None:
    """prereg/arb_data_manifest.json: its own sha is pinned in PRIMARY; every file it lists must
    match, and the selftri cache must hold exactly its listed held-out items (EV-0570, EV-0576)."""
    pin = ((primary["primary_endpoint"].get("pins") or {}).get("data_manifest") or {})
    want = pin.get("sha256")
    path = REPO / DATA_MANIFEST_REL
    if not (isinstance(want, str) and HEX64.match(want)):
        bad.append(f"pins.data_manifest.sha256 {want!r} is not a sha256")
        return
    if not path.is_file() or C.sha256_file(path) != want:
        bad.append(f"{DATA_MANIFEST_REL} missing or differs from pins.data_manifest.sha256")
        return
    man = json.loads(path.read_text("utf-8"))

    def need(p: Path, sha, what: str) -> None:
        if not (isinstance(sha, str) and HEX64.match(sha)):
            bad.append(f"manifest {what}: {sha!r} is not a sha256")
        elif not p.is_file() or C.sha256_file(p) != sha:
            bad.append(f"manifest {what}: {p} missing or differs")

    runs = man.get("prediction_runs") or {}
    srcs = {"EXP-002-audit-s1": (Path(a.s_preds), inputs["S"]),
            **{f"EXP-003-ff-s{s}": (Path(p), inputs["FF"][s])
               for s, p in zip(FF_SEEDS, a.ff_preds, strict=True)}}
    for run, (pd, psha) in srcs.items():
        m = runs.get(run) or {}
        need(pd / "PREDS.json", m.get("PREDS.json_sha256"), f"{run} PREDS.json")
        need(pd.parent / "metrics.json", m.get("metrics.json_sha256"), f"{run} metrics.json")
        if m.get("pred_sha") != psha:
            bad.append(f"manifest {run} pred_sha differs from inputs_sha256")
    tr = man.get("heldout_T_render") or {}
    need(Path(a.t_preds) / "PREDS.json", tr.get("PREDS.json_sha256"), "held-out T PREDS.json")
    if tr.get("pred_sha") != inputs["T"]:
        bad.append("manifest heldout_T_render pred_sha differs from inputs_sha256 T")
    dump = man.get("alpha_and_flow_dump") or {}
    acache = Path(a.alpha_cache)
    need(acache / "manifest.json", dump.get("cache_manifest.json_sha256"), "alpha manifest.json")
    run_dir = next((p.parent for p in acache.parents if p.name == "out"), None)
    if run_dir is None:     # runs/<run>/out/cache/...: the VM manifest sits beside out/ (round 2)
        bad.append(f"alpha cache {acache} is not under a run's out/ directory")
    else:
        need(run_dir / "MANIFEST.vm.json", dump.get("MANIFEST.vm.json_sha256"),
             "dump MANIFEST.vm.json")
    sc = man.get("selftri_cache") or {}
    root = Path(a.selftri_cache)
    need(root / ".teacher_settings.json", sc.get("teacher_settings_sha256"),
         "selftri .teacher_settings.json")
    items = sc.get("heldout_items") or {}
    missing = sc.get("heldout_missing") or {}
    for i in held:
        rel = f"kitti15/training/image_2/{i}_09_11.npz"
        if rel in items:
            need(root / rel, items[rel], f"selftri {rel}")
        elif i in missing:
            if (root / rel).exists():
                bad.append(f"selftri {rel}: declared missing but present (no later fill, EV-0576)")
        else:
            bad.append(f"selftri item for held-out {i}: neither listed nor declared missing")
    extra = sorted(set(items) - {f"kitti15/training/image_2/{i}_09_11.npz" for i in held})
    if extra:
        bad.append(f"manifest lists selftri items outside the held-out ids: {extra[:3]}")


def _check_pins(primary: dict, a, held: list[str]) -> dict:
    """Every pinned code file, input and data file must match PRIMARY before any held-out pixel.
    A pin that is not a full sha256 (a placeholder) fails like a differing one (EV-0569)."""
    bad: list[str] = []
    code = _hex_pins((primary.get("analysis_code") or {}).get("code_sha256") or {},
                     "analysis_code.code_sha256", bad)
    rule = _hex_pins(primary["primary_endpoint"]["frozen_rule"].get("code_sha256") or {},
                     "frozen_rule.code_sha256", bad)
    for rel in HO_PINNED:
        if rel not in code:
            bad.append(f"{rel}: not pinned in analysis_code.code_sha256")
    for rel in HO_RULE_PINNED:
        if rel not in rule:
            bad.append(f"{rel}: not pinned in frozen_rule.code_sha256")
    for block, pins in (("analysis_code", code), ("frozen_rule", rule)):
        for rel, want in pins.items():
            p = REPO / rel
            got = C.sha256_file(p) if p.is_file() else "missing"
            if got != want:
                bad.append(f"{block} {rel}: here {got}, PRIMARY {want}")
    esha, _ = rg.eval_sha(rg.CODE_ROOT)
    want_esha = (primary.get("analysis_code") or {}).get("eval_sha")
    if esha != want_esha:
        bad.append(f"eval_sha here {esha}, PRIMARY {want_esha}")
    inp = primary.get("inputs_sha256") or {}
    ff_pins = inp.get("FF") if isinstance(inp.get("FF"), dict) else {}
    _hex_pins({k: v for k, v in inp.items() if k != "FF"}, "inputs_sha256", bad)
    _hex_pins(ff_pins, "inputs_sha256.FF", bad)
    have = {"S": _pred_sha(Path(a.s_preds)), "T": _pred_sha(Path(a.t_preds)),
            "FF": {s: _pred_sha(Path(p)) for s, p in zip(FF_SEEDS, a.ff_preds, strict=True)},
            "alpha_manifest": C.sha256_file(Path(a.alpha_cache) / "manifest.json")}
    for k, v in have.items():
        if inp.get(k) != v:
            bad.append(f"input {k}: here {v}, PRIMARY {inp.get(k)}")
    if not bad:
        _check_data_manifest(primary, a, have, held, bad)
    if bad:
        raise Fail(RT.EXIT_INTEGRITY, "pins differ from PRIMARY.json: " + "; ".join(bad))
    return {"code": code, "rule": rule, "inputs": have, "eval_sha": esha}


def _check_runs(primary: dict, runs: dict, need: list[str]) -> dict:
    """{key: (exp, seed)}: each --run must be the registered run of its arm and seed
    (pins.registration; EXP-NNN-<arm>-s<seed>, rel and arb seed 1, ff/ffarb the FF seed)."""
    reg = (primary["primary_endpoint"].get("pins") or {}).get("registration") or {}
    out = {}
    for k in need:
        arm, _, sd = k.partition("-s")
        seed = int(sd) if sd else 1
        exp = reg.get(arm)
        if not (isinstance(exp, str) and _is_exp_id(exp)):
            raise Fail(RT.EXIT_PRE, f"pins.registration.{arm} is {exp!r}: not a registered id")
        want = f"{exp}-{arm}-s{seed}"
        if runs[k] != want:
            raise Fail(RT.EXIT_USAGE, f"--run {k}={runs[k]}: the registered run is {want}")
        out[k] = (exp, seed)
    return out


def _is_exp_id(s: str) -> bool:
    return bool(re.match(r"^EXP-\d{3}$", s))


def _outcome(templates: dict, g: float, ci: list[float]) -> dict:
    """PRIMARY's decision order: lo > 0, then hi < 0, else spans 0 (a CI end AT 0 spans)."""
    lo, hi = ci
    key = "ci_low_gt_0" if lo > 0 else "ci_high_lt_0" if hi < 0 else "ci_spans_0"
    nums = f"{abs(g) if key == 'ci_high_lt_0' else g:.3f} pp [{lo:.3f}, {hi:.3f}]"
    text = templates[key].replace("|G| pp [lo, hi]", nums).replace("G pp [lo, hi]", nums)
    return {"outcome": key, "sentence": text}


def _outcome_iu(templates: dict, cis: list[list[float]]) -> dict:
    key = ("all_ci_low_gt_0" if all(c[0] > 0 for c in cis) else
           "all_ci_high_lt_0" if all(c[1] < 0 for c in cis) else "otherwise")
    return {"outcome": key, "sentence": templates[key]}


def _invalid(primary: dict, reason: str) -> dict:
    """The declared 'invalid' outcome for ARB, S1 and S2 (EV-0579); the reason carries no number."""
    t = primary["primary_endpoint"]["outcomes"]["invalid"]
    s = {"outcome": "invalid", "sentence": t.replace("<reason>", reason)}
    return {"ARB": s, "S1": s, "S2": s}


def _rel_level(n_scored: int, fl_scored: float, results: list[dict], s_pred_sha: str) -> str | None:
    """None when the scored held-out rel level equals the one derived by counts from the GATE-0
    all-200 row and the dev row (EV-0575), else the reason (no number in it)."""
    by = {r.get("id"): r for r in results}
    a, d = by.get(REL_LEVEL_ROWS["full200"]), by.get(REL_LEVEL_ROWS["dev"])
    sel = ("metric", "agg", "region", "step")
    if not a or not d or a.get("split") != "full200" or d.get("split") != "dev" \
            or a.get("pred_sha") != s_pred_sha \
            or any(a.get(k) != v or d.get(k) != v for k, v in
                   zip(sel, ("Fl", "px-pooled", "all", "final"), strict=True)):
        return "the rel-level reference rows are not the GATE-0 all-200 and dev Fl rows"
    n_h = int(a["n_pixels"]) - int(d["n_pixels"])
    if n_h != n_scored:
        return "the held-out pixel count differs from n_all - n_dev"
    derived = (a["value"] * a["n_pixels"] - d["value"] * d["n_pixels"]) / n_h
    if abs(derived - fl_scored) > REL_LEVEL_TOL_PP:
        return "the scored rel level differs from the GATE-0 derivation (GT integrity)"
    return None


def _source_rows(results: list[dict], pred_sha: str) -> dict:
    """The GATE-0 all-200 row that scored this prediction set: provenance (GPU, session, lock,
    checkpoint, +FF settings) for the held-out rows of the same predictions."""
    hit = [r for r in results if r.get("pred_sha") == pred_sha and r.get("split") == "full200"
           and r.get("metric") == "Fl" and r.get("agg") == "px-pooled"]
    if not hit:
        raise Fail(RT.EXIT_PRE, f"no GATE-0 all-200 row scored pred_sha {pred_sha[:12]}")
    return hit[0]


def _composite_sha(*parts: str) -> str:
    """pred_sha of a composite arm (arb, ffarb) made on the Mac from pinned inputs: it has no
    shard file of its own, so it is the sha256 of its frozen recipe and its input pred_shas."""
    recipe = f"{AC.FROZEN_TEACHER}|alpha>{AC.FROZEN_ALPHA_GT}|d<={AC.FROZEN_D_LE_PX}"
    return hashlib.sha256("|".join((recipe, *parts)).encode("utf-8")).hexdigest()


def build_rows(declared: list[dict], runs: dict, run_ids: dict, values: dict, npx: dict,
               pins: dict, results: list[dict], gt_pack_sha: str, git_sha: str) -> list[dict]:
    """Rows with every field note.py's _check_eval_row needs (metric skill section 6), before
    seal.stamp_rows. Provenance of rel/ff is their GATE-0 producing run; arb/ffarb inherit S's or
    FF_s's and carry a composite pred_sha."""
    inp = pins["inputs"]
    rows = []
    for r in declared:
        keys = [r["arm"]] if r["arm"] in ("rel", "arb") else [f"{r['arm']}-s{s}" for s in FF_SEEDS]
        for k in keys:
            arm, _, sd = k.partition("-s")
            base = inp["S"] if arm in ("rel", "arb") else inp["FF"][sd]
            src = _source_rows(results, base)
            psha = base if arm in ("rel", "ff") else _composite_sha(base, inp["T"],
                                                                     inp["alpha_manifest"])
            exp, seed = run_ids[k]
            row = {"exp": exp, "run": runs[k], "seed": seed, "step": "final",
                   "dataset": "kitti15", "split": "heldout", "region": r["region"],
                   "metric": "Fl", "value": values[k], "agg": "px-pooled",
                   "n_images": N_HELDOUT, "n_pixels": npx[k], "eval_sha": pins["eval_sha"],
                   "gt_pack_sha": gt_pack_sha, "pred_sha": psha,
                   **{f: src[f] for f in ("ckpt_sha", "gpu_name", "cc", "driver", "lock_hash",
                                          "session")},
                   "git_sha": git_sha, "packed_with": [], "protocol": "FULL",
                   "flags": list(src.get("flags") or []), "oracle": False,
                   "standard": "measured"}
            if src.get("ff") is not None:
                row["ff"] = src["ff"]
            rows.append(row)
    return rows


def contrasts(primary: dict, per: dict, regions) -> dict:
    """ARB (G = rel - arb), S1 (H_s = ff_s - arb) and S2 (K_s = ff_s - ffarb_s), each at its own
    PRIMARY CI level (EV-0578), on the same image draws (one bootstrap seed and B)."""
    ci = primary["primary_endpoint"]["ci"]

    def cmp(base, arm, level):
        boot = {"B": int(ci["B"]), "seed": int(ci["bootstrap_seed"]), "alpha": 1.0 - level}
        e = AC.frozen_stats(per[base], per[arm], regions, ("fl",), boot, "kitti15")["all"]["fl"]
        return {"gain_pp": e["gain"], "ci": e["gain_ci"], "level": level, "base": base,
                "arm": arm}

    g = cmp("rel", "arb", _level(primary, "ARB"))
    res = {"ARB": {**g, **_outcome(primary["primary_endpoint"]["outcomes"], g["gain_pp"],
                                   g["ci"])}}
    for sid, mk in (("S1", lambda s: ("ff-s" + s, "arb")),
                    ("S2", lambda s: ("ff-s" + s, "ffarb-s" + s))):
        lv = _level(primary, sid)
        per_seed = {s: cmp(*mk(s), lv) for s in FF_SEEDS}
        (sec,) = [x for x in primary["secondaries"] if x["id"] == sid]
        res[sid] = {"per_ff_seed": per_seed,
                    **_outcome_iu(sec["outcomes"], [v["ci"] for v in per_seed.values()])}
    return res


def _check_registration(primary: dict, runs: dict, need: list[str], check: bool,
                        pending: list[str]) -> dict | None:
    """run ids of every arm and seed. --check with pins.registration still null (before the
    four ARB arms are registered at prereg-v2) reports it as pending; --run may be omitted in
    --check, which then checks the registered ids' own form."""
    reg = {k: v for k, v in ((primary["primary_endpoint"].get("pins") or {})
                             .get("registration") or {}).items() if not k.startswith("_")}
    if check and (not reg or any(reg.get(arm) is None for arm in HO_ARMS)):
        pending.append("pins.registration is not filled (the ARB arms are not registered)")
        return None
    if check and not runs:
        runs.update({k: f"{reg.get(k.partition('-s')[0])}-{k.partition('-s')[0]}-s"
                        f"{k.partition('-s')[2] or 1}" for k in need})
    miss = [k for k in need if k not in runs]
    if miss:
        raise Fail(RT.EXIT_USAGE, f"--run ARM=RUN_ID needed for {miss}")
    return _check_runs(primary, runs, need)


def _mark_started(path: Path, batch: str, head: str) -> None:
    """Create the batch's started marker exclusively and commit it, before any GT byte is read
    (review minor 17). A second scorer racing this one loses the O_EXCL create; deleting the
    marker afterwards shows in git. If the commit fails nothing has been read: the marker is
    removed and the run refuses."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        raise Fail(RT.EXIT_SEAL, f"{path.name} exists: batch {batch} was already started, a "
                   "second look (AGENTS.md section 4: STOP, the user's call)") from None
    utc = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(json.dumps({"batch": batch, "started_utc": utc, "head": head}) + "\n")
    rel = str(path.relative_to(REPO))
    for args in (("add", "--", rel),
                 ("commit", "-q", "-m", f"ARB held-out batch {batch}: scoring started", "--",
                  rel)):
        r = _git(*args)
        if r is None or r.returncode != 0:     # nothing read yet: take the marker back
            path.unlink()
            raise Fail(RT.EXIT_PRE, f"could not commit {rel} (git {args[0]}); no GT was read "
                       "and the marker was removed, so a rerun after fixing git is allowed")


def heldout_run(a) -> int:
    batch, check = a.heldout_batch, bool(a.check)
    unseal_rows, primary, pending = heldout_gate(batch, check)   # first: nothing read before it
    if a.alpha is not None:
        raise Fail(RT.EXIT_USAGE, "--alpha is not a held-out option (PRIMARY fixes the level)")
    if a.json_out or a.no_json:
        raise Fail(RT.EXIT_USAGE, "--json-out/--no-json are not held-out options (one fixed "
                   "output, so a repeat is seen)")
    out = REPO / f"reports/p1_region/arb_heldout_{batch}.json"
    ev_out = out.with_suffix(".evidence.json")
    started = out.with_suffix(".started")
    led = _ledger().Ledger(REPO)
    done = [r.get("id") for r in led.read("results")
            if r.get("split") == "heldout" and r.get("unseal_batch") == batch]
    # review minor 17: an invalid outcome leaves no result row, only its evidence row, and a run
    # killed after the GT read leaves only the started marker; each is a trace of a look
    done += [r.get("id") for r in led.read("evidence")
             if r.get("subkind") == f"arb-heldout-{batch}"]
    seen = [p.name for p in (out, ev_out, started) if p.exists()] + done[:3]
    if seen:                                   # review minor 15: a repeat is a second look
        raise Fail(RT.EXIT_SEAL, f"batch {batch} was already scored ({', '.join(seen)}): "
                   "running again re-reads held-out GT, a second look (AGENTS.md section 4: "
                   "STOP, the user's call)")
    seal = _seal()
    spec = (primary.get("unseal_batches") or {}).get(batch) or {}
    declared = spec.get("rows") or []
    for r in declared:
        if r["arm"] not in HO_ARMS or (r["metric"], r["region"], r["agg"], r["step"]) \
                not in HO_BUILT_ROWS:
            raise Fail(RT.EXIT_PRE, f"batch {batch} row {r}: not built in this scorer")
    if not declared:
        raise Fail(RT.EXIT_PRE, f"batch {batch}: PRIMARY declares no rows to score here")
    fr = primary["primary_endpoint"]["frozen_rule"]
    if (fr["teacher"], float(fr["alpha_gt"]), float(fr["d_le_px"])) != (
            AC.FROZEN_TEACHER, AC.FROZEN_ALPHA_GT, AC.FROZEN_D_LE_PX):
        raise Fail(RT.EXIT_PRE, "PRIMARY frozen_rule differs from arbitration_cv FROZEN_*")
    ci = primary["primary_endpoint"]["ci"]
    if (a.B is not None and a.B != int(ci["B"])) or \
            (a.seed is not None and a.seed != int(ci["bootstrap_seed"])):
        raise Fail(RT.EXIT_USAGE, "--B/--seed must equal PRIMARY's (or be omitted)")
    runs = dict(x.split("=", 1) for x in a.run or [])
    need = sorted({r["arm"] if r["arm"] in ("rel", "arb") else f"{r['arm']}-s{s}"
                   for r in declared for s in FF_SEEDS})
    run_ids = _check_registration(primary, runs, need, check, pending)

    root = ek.repo_root()
    dev_l, held_l = ek.split_ids(root)
    held = sorted(held_l)
    if len(held) != N_HELDOUT:
        raise Fail(RT.EXIT_PRE, f"{len(held)} held-out ids, PRIMARY declares {N_HELDOUT}")
    pins = _check_pins(primary, a, held)
    lay = seal.Layout(DATA, REPO, REPO / "data/manifest/bundles.v1.json",
                      REPO / "splits/k15_split_v1.json")
    gt_pack_sha = lay.bundle_spec(seal._load(lay))["sha256"]
    regions = tuple(sorted({r["region"] for r in declared}))
    S = {i: f for i, f in ek.read_preds(Path(a.s_preds)) if i in held}
    T = {i: f for i, f in ek.read_preds(Path(a.t_preds))}
    if not set(T) <= set(held) or sorted(S) != held:
        raise Fail(RT.EXIT_ERR, "S must cover the 100 held-out ids and T only held-out ids")
    n_t, no_t = len(T), [i for i in held if i not in T]
    # dev_evidence reads GT-free alpha for any id set; its parameter is named for the dev use,
    # and here it gets the held-out ids (no GT is touched by it)
    alpha = dict(AC.dev_evidence(Path(a.alpha_cache), set(held), ds="kitti15"))
    if sorted(alpha) != held:
        raise Fail(RT.EXIT_ERR, f"alpha for {len(alpha)} of {len(held)} held-out ids")
    FF = {s: {i: f for i, f in ek.read_preds(Path(p)) if i in held}
          for s, p in zip(FF_SEEDS, a.ff_preds, strict=True)}
    short = [s for s in FF_SEEDS if sorted(FF[s]) != held]
    if short:                                   # before any GT byte is decoded (review m7)
        raise Fail(RT.EXIT_ERR, f"+FF seed(s) {short} do not cover the 100 held-out ids")
    head = _git("rev-parse", "HEAD")
    head = head.stdout.decode().strip() if head is not None and head.returncode == 0 else ""
    if not re.match(r"^[0-9a-f]{40}$", head):   # rows need it (note.py): refuse before any GT
        raise Fail(RT.EXIT_PRE, f"{REPO} has no readable HEAD commit for the rows' git_sha")
    if check:                                   # EV-0610: every check above, no GT read
        print(f"CHECK_OK batch {batch}: gate, batch scope, rule, runs, {len(held)} ids, "
              f"{len(pins['code']) + len(pins['rule'])} code pins, inputs, data manifest, "
              f"S/T/alpha/FF coverage (T on {n_t}); no GT read")
        for x in pending:
            print(f"PENDING {x}")
        return RT.EXIT_OK
    _mark_started(started, batch, head)        # the last step before the first GT byte
    mem = {}
    for name, data in seal.heldout_members(lay, batch, unseal_rows):
        parts = name.split("/")
        mem[(parts[-2], parts[-1].split("_")[0])] = data
    gt = MemGT(mem)
    gtc = RT.GTCache(gt, RT.V1Provider(), "kitti15", 0)
    per = {k: [] for k in ("rel", "arb", *(f"ff-s{s}" for s in FF_SEEDS),
                           *(f"ffarb-s{s}" for s in FF_SEEDS))}
    try:
        for seq in held:
            shape, flow_occ, packed = gtc.get(seq)
            m = {r: RT.unpack(packed[r], shape) for r in regions}
            s = {"flow": np.asarray(S.pop(seq), np.float32),
                 "alpha": AC.alpha_native(alpha.pop(seq)["alpha_10to11"], shape)
                 .astype(np.float32)}
            t = T.get(seq)
            cs, cr, _ = AC.frozen_image(s, t, shape, flow_occ, m, "kitti15", regions)
            per["rel"].append(cs)
            per["arb"].append(cr)
            for sd in FF_SEEDS:
                f = {"flow": np.asarray(FF[sd].pop(seq), np.float32), "alpha": s["alpha"]}
                cf, cfr, _ = AC.frozen_image(f, t, shape, flow_occ, m, "kitti15", regions)
                per[f"ff-s{sd}"].append(cf)
                per[f"ffarb-s{sd}"].append(cfr)
    finally:
        gt.close()

    header = {"tool": TOOL, "mode": f"heldout batch {batch}", "primary": PRIMARY_REL,
              "pins": pins, "gt_pack_sha": gt_pack_sha, "n_images": len(held),
              "teacher_images": n_t, "teacher_missing": no_t,
              "per_image": "never written (seal)"}
    # validity first, before any contrast (PRIMARY validity_checks; EV-0575, EV-0579)
    arrays = {k: RT.region_arrays(v, "all") for k, v in per.items()}
    npx = {k: int(x["n"].sum()) for k, x in arrays.items()}
    values = {k: st.statistic(x, "fl") for k, x in arrays.items()}
    results = _ledger().Ledger(REPO).read("results")
    # an image counts as scored when it has GT-valid pixels; every arm must score all 100
    if any(len(v) != N_HELDOUT or int((arrays[k]["n"] > 0).sum()) != N_HELDOUT
           for k, v in per.items()):
        reason = "fewer than 100 held-out images scored"
    else:
        reason = _rel_level(npx["rel"], values["rel"], results, pins["inputs"]["S"])
    if reason:
        res, rows = _invalid(primary, reason), []
    else:
        res = contrasts(primary, per, regions)
        rows = build_rows(declared, runs, run_ids, values, npx, pins, results, gt_pack_sha,
                          head)
        rows = seal.stamp_rows(rows, primary, batch, unseal_rows)
    doc = RT._jsonable({"header": header, "comparisons": res, "rows": rows})
    ek.atomic_write(out, json.dumps(doc, indent=1) + "\n")
    urow = next(r for r in unseal_rows if r.get("batch") == batch)
    ev = RT._jsonable({"batch": batch, "unseal_row": urow.get("id"),
                       "primary_sha": urow.get("primary_sha"), "out": str(out.relative_to(REPO)),
        "comparisons": res, "n_images": len(held), "teacher_images": n_t})
    ek.atomic_write(ev_out, json.dumps(ev, indent=1) + "\n")
    for k in ("ARB", "S1", "S2"):
        print(f"{k}: {res[k]['sentence']}")
    # EV-0615: only note.py writes the ledgers (AGENTS.md hard rule 7); the exact commands
    src = f"tools/p1/selftri_vs_ff.py --heldout-batch {batch}; {out.relative_to(REPO)}"
    if rows:
        print(f"RECORD .venv/bin/python tools/note.py result --from-eval {out.relative_to(REPO)}")
    print(f"RECORD .venv/bin/python tools/note.py evidence --kind measurement --subkind "
          f"arb-heldout-{batch} --standard measured --source '{src}' --summary "
          f"'ARB held-out batch {batch}: {res['ARB']['outcome']}' "
          f"--data @{ev_out.relative_to(REPO)}")
    return RT.EXIT_INTEGRITY if reason else RT.EXIT_OK


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--heldout-batch", help="score this unsealed PRIMARY batch (the ARB test; "
                    "reads experiments/unseal.jsonl and prereg/PRIMARY.json, fixed paths)")
    ap.add_argument("--check", action="store_true", help="with --heldout-batch: every gate, pin, "
                    "coverage, id and run-id check on the real paths, then stop; reads no GT")
    ap.add_argument("--run", action="append", help="ARM=RUN_ID (rel, arb, ff-sN, ffarb-sN)")
    ap.add_argument("--s-preds", default=str(HO_DEFAULTS["s_preds"]))
    ap.add_argument("--ff-preds", nargs=3, default=[str(p) for p in HO_DEFAULTS["ff_preds"]])
    ap.add_argument("--t-preds", default=str(HO_DEFAULTS["t_preds"]))
    ap.add_argument("--alpha-cache", default=str(HO_DEFAULTS["alpha_cache"]))
    ap.add_argument("--selftri-cache", default=str(HO_DEFAULTS["selftri_cache"]))
    ap.add_argument("--gt-root", default="/Volumes/Shin/U2FLOW_data/gt")
    ap.add_argument("--json-out", help=f"default {DEFAULT_JSON}")
    ap.add_argument("--no-json", action="store_true")
    ap.add_argument("--B", type=int)
    ap.add_argument("--alpha", type=float)
    ap.add_argument("--seed", type=int)
    try:
        a = ap.parse_args(argv)
    except SystemExit as ex:
        return RT.EXIT_USAGE if ex.code else RT.EXIT_OK
    try:
        return heldout_run(a) if a.heldout_batch else run(a)
    except Fail as e:
        print(f"selftri_vs_ff: {'REFUSED (seal)' if e.code == RT.EXIT_SEAL else 'error'}: {e}",
              file=sys.stderr)
        return e.code
    except C.Stop as e:
        print(f"selftri_vs_ff: {e}", file=sys.stderr)
        return e.code
    except rg.IntegrityError as e:
        print(f"selftri_vs_ff: integrity: {e}", file=sys.stderr)
        return RT.EXIT_INTEGRITY
    except (OSError, ValueError, KeyError) as e:
        print(f"selftri_vs_ff: {type(e).__name__}: {e}", file=sys.stderr)
        return RT.EXIT_ERR


if __name__ == "__main__":
    sys.exit(main())
