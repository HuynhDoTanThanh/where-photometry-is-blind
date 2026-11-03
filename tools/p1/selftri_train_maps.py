#!/usr/bin/env python3
"""selftri_train_maps.py: the teacher maps of the M' arm (reports/design_Mprime_2026-10-08.md,
sections 2 and 3 row 1): per training pair t -> t+1, the selftri rigid render (U2Flow kitti.pth's
own three-frame rigid refit) where IDEA-004's frozen rule switches S to T. A launch.sh job (spec
tools/p1/specs/selftri_train_pilot.spec.json, job infer); core/u2f_providers.py
make_teacher_provider reads the result unchanged (u2f.gate_cache = $U2_DATA/derived/p1selftri).

What it does, per pair of gate_maps.pair_rows that has t-1 in its window (selftri needs it):
  evidence (GPU)  gate_maps.Evidence UNCHANGED: kitti.pth bidirectional on (t, t+1) and (t-1, t):
                  flow_fwd = t -> t+1, flow_bwd = t -> t-1, alpha and visibility of each (fp16 /
                  packbits, scratch files in --evidence-dir, the gate-map evidence format)
  selftri (CPU)   self_triangulate.self_triangulate UNCHANGED on (bwd, fwd) with the calibration K
                  zoomed to 256 x 832: depth of frame t (|t_b| = 1 units), the essential-matrix
                  pose t -> t+1 with the backward/forward scale (the 'native' pose of the eval
                  cache). Inverse depth is rounded to fp16 as self_triangulate.produce_fn stores
                  it, so the label equals the selftri/native render that EV-0402 scored
  render (CPU)    rigid.render_pair UNCHANGED (no target depth: selftri has none for t+1):
                  flow_12 (NaN where no depth) and its Laplace scale b_12
  rule            w = RIGID = 1[flow_12 finite AND alpha_fwd > 2.75 AND native-px
                  |flow_12 - flow_fwd| <= 100] (IDEA-004, tools/p1/arbitration_cv.py FROZEN_*;
                  the test pins the equality). Native px: u x W/832, v x H/256 (resize_flow's
                  scale, H x W = the frame's PNG size); alpha is the grid value (log grid-px^2),
                  thresholded as arbitration_cv thresholds its bilinear-resized copy
  maps            F_T = flow_12, b_T = 1.0 px (design D5, constant), w = RIGID = the rule; all 0
                  where w = 0. b_rigid = rigid.py's b_12 on the support, stored beside (unused by
                  the loss). A pair whose triangulation fails is stored with zero maps, usable
                  False and the reason (the manifest stays complete; the provider serves nothing)
Items (id = the loader pair id 'kitti15_testing_image_2_000012_05') go to
<cache-root>/gatemap (p1lib/cache.py), the format TeacherProvider reads:
    F_T fp16 (2, 256, 832) u then v, grid px, frame t -> t+1 . b_T fp16 . w fp16 . RIGID packbits
    b_rigid fp16 . meta_json (usable, why_unusable, counts, scale, inliers, native_hw, seed)
and the manifest says kind 'gatemap', source 'selftri'.

HO-tr (design D2, option a). Default: the subset band <= --max-band, KITTI-15 only, minus the
teacher-off list (prereg/drafts/ho-tr-teacher-off.v2-draft.json: every K15 training sequence), so
only KITTI-15 TESTING pairs; a KITTI-15 training or KITTI-12 row is refused (exit 3).
--allow-s-derived (options b/c, a prereg-v2 draft change, the user's decision) skips the list for
this S-derived label and allows --datasets kitti12; it is recorded in the manifest. Note that
TeacherProvider still serves KITTI-15 testing pairs only (design section 3 fallback).

Parity (design section 6 step 2; Mac only): `eval-windows` runs the SAME selftri_maps() on the
cached dev evidence of p1-dump-eval-u2flow-r1 (window w03-11, frame 10 -> 11, KITTI-15 dev half
only), carries the 256 x 832 mask to native by arbitration_cv.nearest_ac, takes T = resize_flow of
the grid flow_12 (render_frig's gate row) where the carried mask is set and T is finite, and scores
S vs the composite with arbitration_cv's kernel (load_student, frozen_counts, frozen_stats). PASS
iff the in-sample Fl-all gain is within 0.1 pp of 0.673 and the switched image share within 0.5 pp
of 4.09 % (tolerances fixed in the design before the run). It also reports the native rule on the
same T (must reproduce EV-0402) and |T - cached selftri/native export|.

How to invoke:
    VM:  python -u tools/p1/selftri_train_maps.py run --data D --out-dir O \
             --ckpt D/weights/u2flow/kitti.pth --subset prereg/drafts/pilot-subset.v2-draft.json \
             --subset-sha256 SHA --max-band 0 \
             --teacher-off prereg/drafts/ho-tr-teacher-off.v2-draft.json \
             --cache-root D/derived/p1selftri --procs 8 [--stage both|evidence|maps]
         python -u tools/p1/selftri_train_maps.py summary --work-dir W --out-dir O
    Mac: .venv/bin/python tools/p1/selftri_train_maps.py eval-windows [--evidence-cache E] \
             [--data CALIB_ROOT] [--B 1000 --seed 20261008] [--json-out F]
         (defaults: E = U2FLOW_data/runs/p1-dump-eval-u2flow-r1/out/cache/diag/u2flow,
          CALIB_ROOT = U2FLOW_data/work/p1data, F = reports/p1_region/selftri_train_parity.json)
Stages: both (default), evidence (GPU only), maps (CPU only, no torch: reads the evidence files).
Output lines: as p1lib/jobrun.py, plus SELFTRI_MAPS_SUMMARY {json}, SELFTRI_PARITY {json}.
Exit codes: 0 ok / PASS . 1 items left or failed / parity FAIL . 2 precondition . 3 refused
(HO-tr, checkpoint pin, stamp) . 50 seal . 64 usage . 143 TERM.

Why it is built this way.
- No threshold or geometry lives here: the rule is IDEA-004's frozen one (constants asserted equal
  to arbitration_cv.FROZEN_* by the tests), selftri and the renderer are the reviewed libraries.
- The evidence files are gate_maps' own (same class, checkpoint and grid check), so the default
  --evidence-dir is shared with gate_maps.py and either job reuses the other's GPU work.
- The grid rule differs from the native rule EV-0402 scored (alpha bilinear vs grid, a mask
  carried by nearest neighbour): the parity mode bounds that difference before any G4 hour.
- Held-out safety: training rows never touch frames 09-12 (pair_rows); the eval mode reads only
  KITTI-15 dev items (arbitration_cv.dev_evidence, region_table's seal checks); refuse_sealed
  guards every path.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gate_maps as GM  # noqa: E402
import numpy as np  # noqa: E402
import rigid  # noqa: E402
import self_triangulate as ST  # noqa: E402
from p1lib import cache as CA  # noqa: E402
from p1lib import common as C  # noqa: E402
from p1lib import geomio as GI  # noqa: E402
from p1lib import jobrun as JR  # noqa: E402
from p1lib import kitti as KT  # noqa: E402
from p1lib import multiproc as MP  # noqa: E402

sys.path.insert(0, str(C.REPO))
from core import u2f_subset as US  # noqa: E402  (stdlib only: no torch at import)

GRID = (256, 832)
VERSION = 1
ALPHA_GT = 2.75      # IDEA-004 / EV-0402 (arbitration_cv.FROZEN_ALPHA_GT; test pins equality)
D_LE_PX = 100.0      # native px (arbitration_cv.FROZEN_D_LE_PX)
B_T_PX = 1.0         # design D5: constant Laplace scale where w > 0
SOURCE = "selftri"
CACHE_SUBDIR = ("derived", "p1selftri")
TOOL_FILES = [HERE / "selftri_train_maps.py", HERE / "self_triangulate.py", HERE / "rigid.py",
              HERE / "gate_maps.py", HERE / "p1lib" / "geomio.py", HERE / "p1lib" / "cache.py"]
RULE_TEXT = (f"w = RIGID = 1[flow_12 finite and alpha_fwd > {ALPHA_GT:g} and native-px "
             f"|flow_12 - flow_fwd| <= {D_LE_PX:g}]; b_T = {B_T_PX:g} px where w > 0; "
             "F_T = b_T = 0 where w = 0")
# parity gate (design section 6 step 2; fixed before the run)
PARITY_GAIN_PP, PARITY_GAIN_TOL = 0.673, 0.1
PARITY_IMG_PCT, PARITY_IMG_TOL = 4.09, 0.5
PARITY_JSON = "reports/p1_region/selftri_train_parity.json"
DUMP_EVAL = Path("/Volumes/Shin/U2FLOW_data/runs/p1-dump-eval-u2flow-r1/out/cache/diag/u2flow")
MAC_P1DATA = Path("/Volumes/Shin/U2FLOW_data/work/p1data")


# ------------------------------------------------------------------------------ the library
def native_px_distance(flow_a: np.ndarray, flow_b: np.ndarray, native_hw, grid=GRID) -> np.ndarray:
    """|a - b| of two grid flows in native px: u x W/w, v x H/h (resize_flow's scale)."""
    H, W = (int(x) for x in native_hw)
    h, w = grid
    d = np.nan_to_num(np.asarray(flow_a, np.float64)) - np.asarray(flow_b, np.float64)
    return np.hypot(d[0] * (W / w), d[1] * (H / h))


def zero_maps(grid=GRID) -> dict:
    return {**GM.zero_maps(grid), "b_rigid": np.zeros(grid, np.float16)}


def selftri_maps(ev_fwd: dict, ev_bwd: dict, K_native: np.ndarray, native_hw, seed: int,
                 grid=GRID) -> tuple[dict, dict, dict]:
    """One pair. ev_fwd / ev_bwd: GI.evidence dicts on the grid (t -> t+1, t -> t-1); K_native the
    calibration K at native (H, W). Returns (arrays, meta, grid extras {flow_12 float64, rule}).
    A failed triangulation gives zero maps, usable False (extras: flow_12 None)."""
    grid = tuple(grid)
    Kg = rigid.zoom_K(np.asarray(K_native, np.float64), tuple(native_hw), grid)
    meta = {"version": VERSION, "grid": list(grid), "native_hw": [int(x) for x in native_hw],
            "K_native": np.asarray(K_native, np.float64).tolist(), "seed": int(seed),
            "rule": RULE_TEXT, "alpha_gt": ALPHA_GT, "d_le_px": D_LE_PX, "b_T_px": B_T_PX,
            "cv2": GM._cv2_version(), "usable": False, "why_unusable": None}
    try:
        res = ST.self_triangulate(ev_bwd, ev_fwd, Kg, int(seed))
    except JR.ItemFailed as e:
        meta.update(why_unusable=f"selftri {e.stage}: {e.reason}", why_class=f"selftri {e.stage}")
        return zero_maps(grid), meta, {"flow_12": None, "rule": np.zeros(grid, bool)}
    D = res["D"]
    valid = np.isfinite(D) & (D > 0)
    # the eval cache's storage (self_triangulate.produce_fn: inv fp16, poses c2w) and render_frig's
    # read of it (valid & inv > 0, relative_pose of the stored poses)
    inv = np.where(valid, 1.0 / np.where(valid, D, 1.0), 0.0).astype(np.float16).astype(np.float64)
    valid &= inv > 0
    M = rigid.relative_pose(np.eye(4), np.linalg.inv(res["M_f"]))
    rend = rigid.render_pair(np.where(valid, inv, 0.0), None, Kg, M, grid, valid1=valid)
    f12, b12 = rend["flow_12"], rend["b_12"]
    fin = np.isfinite(f12).all(axis=0)
    alpha = np.asarray(ev_fwd["alpha"], np.float64)
    a_ok = alpha > ALPHA_GT                       # NaN alpha never switches
    d = native_px_distance(f12, ev_fwd["flow"], native_hw, grid)
    d_ok = fin & (d <= D_LE_PX)
    rule = fin & a_ok & d_ok
    arrays = {"F_T": np.where(rule, np.nan_to_num(f12), 0.0).astype(np.float16),
              "b_T": np.where(rule, B_T_PX, 0.0).astype(np.float16),
              "w": rule.astype(np.float16), "RIGID": CA.pack_bits(rule),
              "b_rigid": np.where(rule & np.isfinite(b12), b12, 0.0).astype(np.float16)}
    n = int(rule.sum())
    meta.update({"scale_s": res["s"], "n_inliers": [res["n_b"], res["n_f"]],
                 "n_both": res["n_both"], "n_T_finite": int(fin.sum()),
                 "n_alpha_gt": int((fin & a_ok).sum()), "n_d_le": int(d_ok.sum()),
                 "n_w_pos": n, "support_frac": n / rule.size, "usable": n > 0})
    if not n:
        meta.update(why_unusable="no pixel passes the rule", why_class="no pixel with weight")
    return arrays, meta, {"flow_12": f12, "rule": rule}


# ------------------------------------------------------------------------------ the list
def select_rows(inv: dict, keys: frozenset, off: frozenset, datasets: tuple,
                allow_s_derived: bool) -> list[dict]:
    """Rows with t-1 in the window. Without allow_s_derived (HO-tr, option a): the teacher-off
    list applies and every row must be a KITTI-15 testing pair, else Stop(3). With it the list is
    skipped (the caller passes it anyway, for the record) and KITTI-12 may be asked for."""
    if not allow_s_derived and tuple(datasets) != ("kitti15",):
        raise C.Stop(C.EXIT_REFUSED, f"--datasets {','.join(datasets)}: KITTI-12 rows are "
                                     "teacher-off under HO-tr; --allow-s-derived is the user's "
                                     "decision (design D2)")
    rows = GM.pair_rows(inv, keys, frozenset() if allow_s_derived else off, tuple(datasets))
    rows = [r for r in rows if r["tm1"] is not None]
    if any({r["t"], r["t"] + 1, r["tm1"]} & KT.EXCLUDED for r in rows):
        raise C.Stop(C.EXIT_REFUSED, "a pair touches frames 09-12")
    bad = [r["id"] for r in rows if not (r["ds"] == "kitti15" and r["sub"] == "testing")]
    if bad and not allow_s_derived:
        raise C.Stop(C.EXIT_REFUSED, f"{len(bad)} row(s) outside the KITTI-15 testing sequences "
                                     f"(first {bad[0]}): a KITTI-15 training sequence missing "
                                     "from --teacher-off? HO-tr refuses it without "
                                     "--allow-s-derived")
    return rows


# ------------------------------------------------------------------------------ the job
class Producer(GM.Producer):
    """gate_maps.Producer's evidence part (get_evidence, produce_evidence, the checkpoint and grid
    check) with the selftri CPU part instead of the MapAnything gate."""

    def __init__(self, a, data: Path, evdir: Path, model_override=None):  # noqa: D107
        self.a, self.data, self.evdir = a, data, evdir
        self.geo, self.ap, self.sam_dir, self._win, self._ev = None, None, None, None, None
        self.prm = GM.GateParams(grid=tuple(a.test_shape))
        self.model_override = model_override
        self.meta_dir = Path(a.meta_dir) if getattr(a, "meta_dir", None) else None

    def produce(self, row: dict):
        ev = self.get_evidence(row, compute=self.a.stage != "maps")
        if ev is None:
            raise JR.ItemFailed("evidence", f"no evidence file for {row['id']} "
                                            "(run --stage evidence)")
        grid = self.prm.grid
        if "flow_bwd" not in ev:
            arrays, meta = zero_maps(grid), {"version": VERSION, "usable": False,
                                             "why_unusable": "no backward evidence",
                                             "why_class": "no evidence"}
        else:
            native = C.png_size(KT.frame_path(self.data, row["ds"], row["sub"], row["side"],
                                              row["seq"], row["t"]))
            K, _ = KT.calib_K(self.data, row["ds"], row["sub"], row["seq"], row["side"])
            arrays, meta, _ = selftri_maps(GI.evidence(ev, "fwd", grid),
                                           GI.evidence(ev, "bwd", grid), K, native,
                                           GM.item_seed(row["id"]), grid)
        meta.update(frames=[row["t"], row["t"] + 1], tm1=row["tm1"])
        if self.meta_dir is not None:
            C.atomic_json(self.meta_dir / f"{row['id']}.json", meta)
        return arrays, meta


def summarise(meta_dir: Path) -> dict:
    """Pooled counts over the sidecars, and design step 8's STOP read (usable/items < 0.5 or mean
    support outside [1 %, 10 %])."""
    n = usable = px = wpos = 0
    why, sup = {}, []
    for p in sorted(Path(meta_dir).glob("*.json")):
        m = json.loads(p.read_text())
        n += 1
        if m.get("usable"):
            usable += 1
            sup.append(float(m["support_frac"]))
            wpos += int(m["n_w_pos"])
            px += int(np.prod(m["grid"]))
        else:
            k = m.get("why_class") or "other"
            why[k] = why.get(k, 0) + 1
    mean_sup = float(np.mean(sup)) if sup else None
    stop = (n == 0 or usable / n < 0.5 or mean_sup is None or not (0.01 <= mean_sup <= 0.10))
    return {"items": n, "usable": usable, "unusable_by_class": why,
            "mean_support_usable": mean_sup,
            "pooled_support_usable": (wpos / px) if px else None,
            "step8_stop": bool(stop),
            "step8_rule": "STOP if usable/items < 0.5 or mean support outside [1 %, 10 %] "
                          "(reports/design_Mprime_2026-10-08.md section 6 step 8)"}


def setup(a, model_override=None):
    data = C.data_root(a.data)
    out = Path(a.out_dir)
    C.refuse_sealed(out, a.work_dir, a.ckpt, a.cache_root, a.evidence_dir, a.subset,
                    a.teacher_off)
    if not (a.subset and a.subset_sha256):
        raise C.Stop(C.EXIT_USAGE, "--subset and --subset-sha256 are required")
    if not a.teacher_off and not a.allow_s_derived:
        raise C.Stop(C.EXIT_USAGE, "--teacher-off is required (HO-tr)")
    try:
        sub = US.load_subset(Path(a.subset) if Path(a.subset).is_absolute()
                             else C.REPO / a.subset, a.subset_sha256)
    except (OSError, ValueError, US.SubsetError) as e:
        raise C.Stop(C.EXIT_REFUSED, f"--subset: {e}") from e
    off, off_sha = frozenset(), None
    if a.teacher_off:
        off, off_sha = GM.load_teacher_off(Path(a.teacher_off) if Path(a.teacher_off).is_absolute()
                                           else C.REPO / a.teacher_off)
    datasets = tuple(x for x in a.datasets.split(",") if x)
    rows = select_rows(KT.inventory_from_data(data), sub.keys(int(a.max_band)), off, datasets,
                       a.allow_s_derived)
    if not rows:
        raise C.Stop(C.EXIT_REFUSED, "no pair with t-1 (the subset minus the teacher-off list)")
    evdir = Path(a.evidence_dir) if a.evidence_dir else data / "derived" / "p1work" / "gm-evidence"
    if a.stage != "maps":
        evdir.mkdir(parents=True, exist_ok=True)
        need = int(len(rows) * GM.EV_BYTES_PER_PAIR * 1.3)
        free = shutil.disk_usage(evdir).free
        if free < need:
            raise C.Stop(C.EXIT_PRE, f"{evdir}: {free / 2**30:.1f} GiB free, the evidence files of "
                                     f"{len(rows)} pairs need about {need / 2**30:.1f} GiB")
    selection = {"file": Path(sub.path).name, "file_sha256": sub.sha256,
                 "table_sha256": sub.doc["table_sha256"], "max_band": int(a.max_band),
                 "datasets": list(datasets), "teacher_off_sha256": off_sha,
                 "teacher_off_n": len(off), "allow_s_derived": bool(a.allow_s_derived),
                 "n": len(rows),
                 "rule": ("subset band <= max_band with t-1 in the window; " + (
                     "S-derived exemption: the teacher-off list NOT applied (design D2 b/c)"
                     if a.allow_s_derived else
                     "KITTI-15 minus the teacher-off list (HO-tr): the K15 testing sequences"))}
    settings = {"kind": "gatemap", "teacher": SOURCE, "version": VERSION,
                "grid": list(a.test_shape),
                "rule": RULE_TEXT, "alpha_gt": ALPHA_GT, "d_le_px": D_LE_PX, "b_T_px": B_T_PX,
                "selftri": {"ransac_px": ST.RANSAC_PX, "prob": ST.PROB, "alpha_max": "ln 2",
                            "k": "calibration zoomed to the grid", "pose": "essential matrix "
                            "(native), scale = median D_b / D_f", "inv_depth": "fp16 as the "
                            "eval cache"},
                "render": {"constants": rigid.CONSTANTS, "target_depth": None},
                "selection": selection, "stage": a.stage, "evidence_dir": str(evdir),
                "weights_key": a.weights_key,
                "derived_from": "S-derived three-frame rigid refit of kitti.pth's own flows "
                                "t->t-1 and t->t+1 (not an external teacher)",
                "outputs": "F_T fp16 (2,256,832), b_T fp16, w fp16, RIGID packbits, b_rigid fp16 "
                           "(rigid.py b_12 on the support; unused by the loss)"}
    identity = {"settings": {k: v for k, v in settings.items() if k != "stage"},
                "code": C.code_identity(TOOL_FILES)}
    work = Path(a.work_dir) if a.work_dir else (
        data / "derived" / "p1work"
        / f"st-{C.sha256_bytes(repr(sorted(identity.items())).encode())[:12]}")
    root = Path(a.cache_root) if a.cache_root else data.joinpath(*CACHE_SUBDIR)
    C.refuse_sealed(root, work)
    ev_stage = a.stage == "evidence"
    cache_dir = (evdir / "_index_selftri") if ev_stage else root / GM.GM_DIRNAME
    if ev_stage:
        work = work.with_name(work.name + "-ev")
        identity = {"settings": {"kind": "selftri-evidence", "version": VERSION,
                                 "selection": selection, "weights_key": a.weights_key},
                    "code": C.code_identity([HERE / "gate_maps.py",
                                             HERE / "selftri_train_maps.py"])}
    a.meta_dir = str(work / ".meta")
    Path(a.meta_dir).mkdir(parents=True, exist_ok=True)
    prod = Producer(a, data, evdir, model_override)
    job = JR.ItemJob(kind="selftri-evidence" if ev_stage else "gatemap", source=SOURCE,
                     items=rows, cache_dir=cache_dir, work_dir=work,
                     shard_size=a.shard_size or 250, shard=a.shard, identity=identity,
                     settings=settings, limit=a.limit, selection=selection)
    return prod, job, rows, work, cache_dir


# ------------------------------------------------------------------------------ parity (Mac)
def eval_windows(a) -> int:
    """Design section 6 step 2 on the Mac (KITTI-15 dev, w03-11 only)."""
    import arbitration_cv as ACV  # noqa: PLC0415  (Mac side: torch, tools/eval)
    import render_frig as RF  # noqa: PLC0415

    CB, RT, ek, rg = ACV.CB, ACV.RT, ACV.ek, ACV.rg
    root = ek.repo_root()
    gt_root, data, evc = Path(a.gt_root), Path(a.data), Path(a.evidence_cache)
    C.refuse_sealed(gt_root, data, evc, a.json_out)
    boot = RT.bootstrap_settings(a, root)
    spec = json.loads((root / CB.DEFAULT_SPEC).read_text("utf-8"))
    students_e, named_e, _ = CB._resolve(spec, root)
    s_entry = dict(next(e for e in students_e if e["source"] == "u2flow"), path=str(evc))
    t_entry = next(e for e in named_e if CB.tkey(e) == ACV.FROZEN_TEACHER)
    dev_l, held_l = ek.split_ids(root)
    dev, held = set(dev_l), set(held_l)
    ids = sorted(dev_l)
    regions = ACV.REGIONS
    gt = rg.GTSource(ACV.DS, root=gt_root)
    gtc = RT.GTCache(gt, RT.V1Provider(), ACV.DS, len(ids) + 1)
    per = {k: {} for k in ("S", "grid", "native")}
    use = {k: {"sw_img": 0, "sw_all": 0, "px": 0, "n_valid": 0} for k in ("grid", "native")}
    tdiff = {"n_joint": 0, "max_px": 0.0, "n_gt_0.01px": 0, "n_gt_0.01px_switched": 0,
             "finite_mismatch_px": 0}
    failed, mask_agree = [], {"n": 0, "both": 0, "grid_only": 0, "native_only": 0}
    try:
        S = ACV.load_student(s_entry, root, dev, held, gtc, ds=ACV.DS, with_fb=False)
        if sorted(S["maps"]) != ids:
            raise ACV.Fail(RT.EXIT_ERR, f"u2flow: {len(S['maps'])} dev images, need {len(ids)}")
        _, streams = RT.open_entry({**t_entry, "datasets": [ACV.DS]}, root, dev, held)
        cached_t = dict(streams[ACV.DS][0]())
        for seq, arr in ACV.dev_evidence(evc, dev):
            s = S["maps"][seq]
            shape, flow_occ, packed = gtc.get(seq)
            m = {r: RT.unpack(packed[r], shape) for r in regions}
            K, _ = KT.calib_K(data, ACV.DS, "training", seq, "image_2")
            _, _, ex = selftri_maps(GI.evidence(arr, "10to11", GRID),
                                    GI.evidence(arr, "10to09", GRID), K, shape, int(seq))
            cs = ACV.frozen_counts(s["flow"], shape, flow_occ, m, ACV.DS, regions)
            per["S"][seq] = cs
            if ex["flow_12"] is None:
                failed.append(seq)
                sws = {"grid": np.zeros(shape, bool), "native": np.zeros(shape, bool)}
                T = None
            else:
                T = RF.resize_flow_like_the_student(ex["flow_12"], shape)
                fin_t = np.isfinite(T).all(axis=0)
                sws = {"grid": ACV.nearest_ac(ex["rule"], shape) & fin_t,
                       "native": ACV.frozen_switch(s["alpha"], s["flow"], T)}
                c = cached_t.get(seq)
                if c is not None:
                    fc = np.isfinite(c).all(axis=0)
                    j = fc & fin_t
                    tdiff["n_joint"] += int(j.sum())
                    tdiff["finite_mismatch_px"] += int((fc ^ fin_t).sum())
                    if j.any():
                        dd = np.where(j, np.abs(T - np.nan_to_num(c)).max(axis=0), 0.0)
                        tdiff["max_px"] = max(tdiff["max_px"], float(dd.max()))
                        tdiff["n_gt_0.01px"] += int((dd > 0.01).sum())
                        tdiff["n_gt_0.01px_switched"] += int(((dd > 0.01)
                                                              & sws["grid"]).sum())
            g, nn = sws["grid"], sws["native"]
            mask_agree["n"] += g.size
            mask_agree["both"] += int((g & nn).sum())
            mask_agree["grid_only"] += int((g & ~nn).sum())
            mask_agree["native_only"] += int((nn & ~g).sum())
            for k, sw in sws.items():
                comp = s["flow"] if T is None else np.where(sw[None], T, s["flow"])
                per[k][seq] = ACV.frozen_counts(comp, shape, flow_occ, m, ACV.DS, regions)
                use[k]["sw_img"] += int(sw.sum())
                use[k]["sw_all"] += int((sw & m["all"]).sum())
                use[k]["px"] += int(sw.size)
                use[k]["n_valid"] += int(m["all"].sum())
    finally:
        gt.close()
    if sorted(per["S"]) != ids:
        raise ACV.Fail(RT.EXIT_ERR, f"evidence for {len(per['S'])} dev ids, need {len(ids)}")
    per_s = [per["S"][i] for i in ids]
    res = {}
    for k in ("grid", "native"):
        stats = ACV.frozen_stats(per_s, [per[k][i] for i in ids], regions, ("fl",), boot,
                                 ACV.DS)
        res[k] = {"gain_pp": {r: stats[r]["fl"]["gain"] for r in regions},
                  "gain_ci_all": stats["all"]["fl"]["gain_ci"],
                  "S_fl_all": stats["all"]["fl"]["S"], "fl_all": stats["all"]["fl"]["frozen"],
                  "switched_image_pct": 100.0 * use[k]["sw_img"] / use[k]["px"],
                  "switched_valid_pct": 100.0 * use[k]["sw_all"] / use[k]["n_valid"],
                  "counts": use[k]}
    g = res["grid"]
    d_gain = abs(g["gain_pp"]["all"] - PARITY_GAIN_PP)
    d_img = abs(g["switched_image_pct"] - PARITY_IMG_PCT)
    verdict = "PASS" if (d_gain <= PARITY_GAIN_TOL and d_img <= PARITY_IMG_TOL) else "FAIL"
    esha, _ = rg.eval_sha(rg.CODE_ROOT)
    doc = {"header": {
        "tool": "tools/p1/selftri_train_maps.py eval-windows", "standard": "measured",
        "design": "reports/design_Mprime_2026-10-08.md section 6 step 2",
        "split": "dev", "dataset": ACV.DS, "window": "w03-11 (10 -> 11, t-1 = 09)",
        "n_images": len(ids), "selftri_failed_ids": failed, "evidence_cache": str(evc),
        "rule": RULE_TEXT, "eval_sha": esha, "git": ek.git_info(), "head": _head(),
        "code": C.code_identity(TOOL_FILES + [HERE / "arbitration_cv.py", HERE / "render_frig.py",
                                              HERE / "composite_bound.py",
                                              HERE / "region_table.py"]),
        "metrics": {"fl": "Fl px-pooled % (KITTI-15 rule: EPE > 3 px and > 5 % of |F_gt|), "
                          "regions_v1 all (= V_occ) / R1 / R3 (R2 withheld-S6), KITTI-15 dev "
                          f"half, n = {len(ids)} images, pooled outlier counts",
                    "gain_pp": "Fl(S) - Fl(composite) pp, same pixels, same images",
                    "gain_ci_all": "paired image bootstrap (tools/eval/stats.py) on 'all'",
                    "switched_image_pct": "switched px / all px of the n frames (native)",
                    "switched_valid_pct": "switched GT-valid px / GT-valid px"},
        "selection": "in-sample-OPTIMISTIC: tau 2.75 and d 100 were chosen on these dev images "
                     "(EV-0402); a parity check of the grid-gated producer, never a gain claim",
        "statistics": {k: boot[k] for k in ("B", "seed", "alpha", "seed_source")},
        "tolerances": {"gain_pp": [PARITY_GAIN_PP, PARITY_GAIN_TOL],
                       "switched_image_pct": [PARITY_IMG_PCT, PARITY_IMG_TOL],
                       "source": "fixed in the design before the run (EV-0402 in-sample row)"},
        "T": "resize_flow(grid flow_12) (render_frig's gate row), switched where the carried "
             "mask is set and T is finite",
        "T_vs_cached_selftri_native_export": tdiff,
        "T_vs_cached_note": "the cached render used the float32-rounded K of the teacher_cache "
                            "export; large differences sit at pixels with huge flows (z near 0), "
                            "which the |T - S| <= 100 px rule never switches",
        "mask_agreement_px": mask_agree,
        "ledger": "no results row (hard rule 7); record as note.py evidence --kind measurement"},
        "grid_rule_carried": res["grid"], "native_rule_same_T": res["native"],
        "verdict": verdict, "abs_diff": {"gain_pp": d_gain, "switched_image_pct": d_img}}
    doc = RT._jsonable(doc)
    if a.json_out:
        p = Path(a.json_out) if Path(a.json_out).is_absolute() else root / a.json_out
        ek.atomic_write(p, json.dumps(doc, indent=1) + "\n")
    C.say("SELFTRI_PARITY", json.dumps({"verdict": verdict,
                                        "grid_gain_all_pp": g["gain_pp"]["all"],
                                        "grid_switched_image_pct": g["switched_image_pct"],
                                        "native_gain_all_pp": res["native"]["gain_pp"]["all"],
                                        "native_switched_image_pct":
                                            res["native"]["switched_image_pct"],
                                        "T_vs_cached": tdiff, "failed": failed}))
    return C.EXIT_OK if verdict == "PASS" else 1


def _head() -> dict:
    """HEAD and whether this tool's own files differ from it (the parity JSON names its code)."""
    import subprocess

    def vcs(*args):
        r = subprocess.run(["git", *args], cwd=C.REPO, capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None

    files = [str(p.relative_to(C.REPO)) for p in TOOL_FILES]
    return {"sha": vcs("rev-parse", "HEAD"),
            "tool_files_dirty": bool(vcs("status", "--porcelain", "--", *files))}


def run_eval_windows(a) -> int:
    """eval_windows with arbitration_cv's error mapping (region_table exit codes)."""
    import arbitration_cv as ACV  # noqa: PLC0415

    try:
        return eval_windows(a)
    except ACV.Fail as e:
        print(f"selftri_train_maps: {'REFUSED (seal)' if e.code == ACV.RT.EXIT_SEAL else 'error'}"
              f": {e}", file=sys.stderr)
        return e.code
    except ACV.rg.IntegrityError as e:
        print(f"selftri_train_maps: integrity: {e}", file=sys.stderr)
        return ACV.RT.EXIT_INTEGRITY


# ------------------------------------------------------------------------------ CLI
def args_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "worker", "summary"):
        p = sub.add_parser(name)
        p.add_argument("--data")
        p.add_argument("--out-dir", required=True)
        p.add_argument("--ckpt")
        p.add_argument("--weights-key", default="u2flow-kitti")
        p.add_argument("--subset")
        p.add_argument("--subset-sha256")
        p.add_argument("--max-band", type=int, default=0)
        p.add_argument("--teacher-off")
        p.add_argument("--datasets", default="kitti15")
        p.add_argument("--allow-s-derived", action="store_true",
                       help="skip the teacher-off list for this S-derived label (design D2 b/c)")
        p.add_argument("--stage", default="both", choices=("both", "evidence", "maps"))
        p.add_argument("--evidence-dir")
        p.add_argument("--cache-root", help="where gatemap/ lives (default "
                                            "DATA/derived/p1selftri; stays on the VM)")
        p.add_argument("--work-dir")
        p.add_argument("--shard")
        p.add_argument("--shard-size", type=int)
        p.add_argument("--limit", type=int)
        p.add_argument("--device", default="cuda")
        p.add_argument("--test-shape", type=int, nargs=2, default=list(GRID))
        p.add_argument("--procs", type=int, default=1)
        p.add_argument("--run", default="")
    p = sub.add_parser("eval-windows")
    p.add_argument("--evidence-cache", default=str(DUMP_EVAL))
    p.add_argument("--data", default=str(MAC_P1DATA), help="calibration root")
    p.add_argument("--gt-root", default="/Volumes/Shin/U2FLOW_data/gt")
    p.add_argument("--json-out", default=PARITY_JSON)
    p.add_argument("--B", type=int)
    p.add_argument("--alpha", type=float)
    p.add_argument("--seed", type=int)
    return ap


def passthrough(a) -> list[str]:
    out = ["--out-dir", a.out_dir, "--weights-key", a.weights_key, "--max-band", str(a.max_band),
           "--datasets", a.datasets, "--stage", a.stage, "--device", a.device,
           "--test-shape", *[str(x) for x in a.test_shape]]
    if a.allow_s_derived:
        out.append("--allow-s-derived")
    for flag, v in (("--data", a.data), ("--ckpt", a.ckpt), ("--subset", a.subset),
                    ("--subset-sha256", a.subset_sha256), ("--teacher-off", a.teacher_off),
                    ("--evidence-dir", a.evidence_dir), ("--cache-root", a.cache_root),
                    ("--work-dir", a.work_dir), ("--shard-size", a.shard_size),
                    ("--limit", a.limit)):
        if v is not None:
            out += [flag, str(v)]
    return out


def main(argv: list[str] | None = None, model_override=None) -> int:
    try:
        a = args_parser().parse_args(argv)
    except SystemExit as e:
        return C.EXIT_USAGE if e.code else C.EXIT_OK
    try:
        if a.cmd == "eval-windows":
            return run_eval_windows(a)
        if a.cmd == "summary":
            if not a.work_dir:
                raise C.Stop(C.EXIT_USAGE, "summary needs --work-dir (it holds .meta/)")
            C.refuse_sealed(a.work_dir, a.out_dir)
            doc = summarise(Path(a.work_dir) / ".meta")
            C.atomic_json(Path(a.out_dir) / "selftri_maps_summary.json", doc)
            C.say("SELFTRI_MAPS_SUMMARY", json.dumps(doc, sort_keys=True))
            return C.EXIT_OK
        prod, job, rows, work, cache_dir = setup(a, model_override)
        produce = prod.produce_evidence if a.stage == "evidence" else prod.produce
        ident = {"tool": "tools/p1/selftri_train_maps.py", "run": a.run, "stage": a.stage,
                 "n_items": job.total, "procs": a.procs, "cache_dir": str(cache_dir),
                 "work_dir": str(work), "allow_s_derived": bool(a.allow_s_derived)}
        if a.cmd == "worker" or a.procs <= 1:
            if a.cmd == "run":
                job.progress.resolved(ident)
            rc = job.run(produce)
        else:
            if a.shard and a.shard != "0/1":
                raise C.Stop(C.EXIT_USAGE, "--procs K owns the sharding; drop --shard")
            C.check_stamp(work, job.identity)
            job.progress.resolved(ident)
            frac = os.environ.get("U2_MEM_FRAC")
            env = {"U2_P1_WORKER_FRAC": f"{float(frac) / a.procs:.4f}"} if frac else {}
            rc = MP.run_workers(Path(__file__), passthrough(a), a.procs, work, set(job.ids),
                                job.progress, env_extra=env)
            job.write_manifest()
        if a.cmd == "run" and (cache_dir / "manifest.json").is_file():
            C.atomic_json(Path(a.out_dir) / f"manifest_{job.kind}_{SOURCE}.json",
                          C.load_json(cache_dir / "manifest.json"))
            if a.stage != "evidence":
                doc = summarise(work / ".meta")
                C.atomic_json(Path(a.out_dir) / "selftri_maps_summary.json", doc)
                C.say("SELFTRI_MAPS_SUMMARY", json.dumps(doc, sort_keys=True))
        return rc
    except C.Stop as e:
        C.say("P1_ERROR", e.msg)
        return e.code


if __name__ == "__main__":
    sys.exit(main())
