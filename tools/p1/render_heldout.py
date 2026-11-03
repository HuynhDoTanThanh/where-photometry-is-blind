#!/usr/bin/env python3
"""render_heldout.py: the frozen selftri teacher T (selftri/native/w03-11) at native resolution
for the KITTI-15 HELD-OUT windows, for the ARB confirmatory test (prereg/drafts/
PRIMARY.arb-v3-draft.json, reports/prereg_arb_plan.md step 4a). Images-only, Mac-only, CPU.

What it does.
    1. Dev parity: renders T for EVERY dev window of the geometry cache with the same kernel as
       render_frig.py (window_K -> geometry_frame -> rigid.render_pair on the 256 x 832 grid ->
       render_frig.resize_flow_like_the_student), and requires each native flow to be
       bit-identical to the committed dev export (U2FLOW_data/work/p1render/exp/
       selftri_native_w03-11, pred_sha 7ab03140..., the T behind EV-0402/0403). Any difference
       stops the run (exit 1) before a held-out window is rendered.
    2. Renders the held-out windows the same way and writes them in eval_kitti.py's prediction
       format to OUT/kitti15/preds (PREDS.json + shards; NaN where the teacher has no depth),
       plus OUT/RENDER.json (code and input identities, parity result, failures, pred_sha).

How to invoke (repo root, .venv/bin/python):
    tools/p1/render_heldout.py [--geom-cache D] [--dev-export E] [--out-dir O]
Output: RENDER_HELDOUT {json}. Exit codes: 0 ok · 1 dev parity failed · 2 precondition ·
50 seal (an output or input path under the seal, or a public dir) · 64 usage.

Why it is built this way.
- render_frig.py's own guard stays as it is: render --native skips held-out rows (scorable) and
  export-preds exits 50 on a held-out id, so no general-purpose tool can make a held-out
  prediction by accident (critic EV-0568). This file is the one held-out path. It never calls
  render_frig's export, never edits render_frig.py (sha f42a1e75... is the code that rendered dev
  and KITTI-12), and reads no GT: a window's native size comes from the geometry cache's
  native_hw (the image), and T is fixed before prereg-v3, so it cannot depend on a held-out look.
- The kernel lines are those of render_frig.render_window's front half, called through the same
  functions; the dev parity check proves they give the committed dev bytes, so the held-out T is
  the T that was frozen (IDEA-004), not a re-implementation.
- Outputs stay on the Mac: a path under U2FLOW_data/{bundles,upload} or the seal is refused.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "tools" / "eval"))

import numpy as np  # noqa: E402
import p1_windows as MW  # noqa: E402
import render_frig as RF  # noqa: E402
import rigid  # noqa: E402
from p1lib import common as C  # noqa: E402
from p1lib import geomio as GI  # noqa: E402
from p1lib import kitti as KT  # noqa: E402

DATA = Path("/Volumes/Shin/U2FLOW_data")
GEOM = DATA / "derived" / "evalselftri" / "v2"
DEV_EXPORT = DATA / "work" / "p1render" / "exp" / "selftri_native_w03-11"
OUT = DATA / "work" / "p1render" / "heldout" / "selftri_native_w03-11"
WIN = "w03-11"                     # the frozen teacher's window (IDEA-004)
K_CHOICE = "teacher"               # as the committed render (manifest settings.k)
PUBLIC = (DATA / "bundles", DATA / "upload")
TOOL_FILES = [*RF.TOOL_FILES, HERE / "render_heldout.py"]


def native_t(geo: GI.GeomIndex, row: dict, k: str) -> np.ndarray:
    """T at the window's native resolution: render_frig.render_window's kernel (same calls, same
    order, pose 'native'), returning only the gate-row flow (native_12, float32, NaN = no depth)."""
    ga, gm = geo.get(row["id"])
    i10, i11 = gm["frames"].index(10), gm["frames"].index(11)
    if "pose_ok" in ga and not (ga["pose_ok"][i10] and ga["pose_ok"][i11]):
        raise ValueError("frame 10 or 11 has no teacher pose")
    M = rigid.relative_pose(ga["pose_c2w"][i10], ga["pose_c2w"][i11])
    K = GI.window_K(ga, gm, i10, k)
    g10 = GI.geometry_frame(ga, gm, i10, RF.GRID, K_model=K)
    g11 = GI.geometry_frame(ga, gm, i11, RF.GRID, K_model=K)
    has2 = bool(g11["valid_model"].any())
    out = rigid.render_pair(g10["inv_model"], g11["inv_model"] if has2 else None, K, M, RF.GRID,
                            valid1=g10["valid_model"],
                            valid2=g11["valid_model"] if has2 else None)
    return RF.resize_flow_like_the_student(out["flow_12"], tuple(gm["native_hw"]))


def run(a) -> int:
    import eval_kitti as ek  # noqa: PLC0415

    geom, dev_exp, out = Path(a.geom_cache), Path(a.dev_export), Path(a.out_dir)
    C.refuse_sealed(geom, dev_exp, out)
    if any(out.resolve().is_relative_to(p) for p in PUBLIC):
        raise C.Stop(C.EXIT_SEAL, f"{out}: a public dir; held-out T stays in a Mac work dir")
    if (out / "kitti15" / "preds" / "PREDS.json").exists():
        raise C.Stop(C.EXIT_PRE, f"{out}: already rendered (delete it to re-render)")
    geo = GI.GeomIndex(geom)
    k, _ = RF.resolve_k_of(geo, K_CHOICE)
    by_id = {r["id"]: r for r in MW.load_plan(MW.PLAN)["eval_windows"]}
    rows = [by_id[i] for i in geo.ids() if i in by_id and by_id[i]["win"] == WIN
            and by_id[i]["ds"] == "kitti15"]
    dev_rows = sorted((r for r in rows if r.get("k15_half") == "dev"), key=lambda r: r["id"])
    held_rows = sorted((r for r in rows if r.get("k15_half") == "heldout"), key=lambda r: r["id"])
    if not held_rows or not dev_rows:
        raise C.Stop(C.EXIT_PRE, f"{geom}: {len(dev_rows)} dev and {len(held_rows)} held-out "
                     f"{WIN} windows")

    # 1. dev parity against the committed dev export (the T of EV-0402/0403)
    ref_dir = dev_exp / "kitti15" / "preds"
    ref = dict(ek.read_preds(ref_dir))
    ref_sha = ek.pred_sha(json.loads((ref_dir / "PREDS.json").read_text("utf-8"))["shards"])
    bad, n_dev = [], 0
    for r in dev_rows:
        seq = KT.parse_id(r["id"])["seq"]
        if seq not in ref:
            continue                      # a dev window the committed render also failed
        try:
            t = native_t(geo, r, k)
        except ValueError:
            bad.append(f"{seq}: render failed here, not in the reference")
            continue
        n_dev += 1
        if not np.array_equal(t, ref[seq], equal_nan=True):
            bad.append(seq)
    parity = {"reference": str(ref_dir), "reference_pred_sha": ref_sha, "windows": n_dev,
              "reference_windows": len(ref), "bit_identical": not bad, "mismatches": bad}
    if bad or n_dev != len(ref):
        C.say("RENDER_HELDOUT", json.dumps({"parity": parity}))
        raise C.Stop(C.EXIT_FAIL, f"dev parity failed: {len(bad)} mismatches, {n_dev} of "
                     f"{len(ref)} reference windows compared")

    # 2. the held-out windows (images only)
    w = ek.ShardWriter(out / "kitti15" / "preds")
    failures, ids = [], []
    for r in held_rows:
        seq = KT.parse_id(r["id"])["seq"]
        try:
            w.add(seq, native_t(geo, r, k).astype(np.float32))
            ids.append(seq)
        except ValueError as e:
            failures.append({"id": seq, "why": str(e)})
    sha = w.close()
    doc = {"tool": "tools/p1/render_heldout.py", "teacher": "selftri/native/" + WIN, "k": k,
           "geom_cache": str(geom), "geom_identity": geo.manifest.get("identity"),
           "code": C.code_identity(TOOL_FILES), "constants": rigid.CONSTANTS,
           "dev_parity": parity, "heldout_windows": len(held_rows), "rendered": len(ids),
           "failures": failures, "pred_sha": sha, "gt_read": "none (images only)",
           "native_rule": "native_12 = resize_flow(grid flow_12), as render_frig's gate row"}
    C.atomic_json(out / "RENDER.json", doc)
    C.say("RENDER_HELDOUT", json.dumps({"rendered": len(ids), "failures": len(failures),
                                        "pred_sha": sha, "dev_parity_windows": n_dev}))
    return C.EXIT_OK


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--geom-cache", default=str(GEOM))
    ap.add_argument("--dev-export", default=str(DEV_EXPORT))
    ap.add_argument("--out-dir", default=str(OUT))
    try:
        a = ap.parse_args(argv)
    except SystemExit as ex:
        return C.EXIT_USAGE if ex.code else C.EXIT_OK
    try:
        return run(a)
    except C.Stop as ex:
        C.say("P1_ERROR", ex.msg)
        return ex.code


if __name__ == "__main__":
    sys.exit(main())
