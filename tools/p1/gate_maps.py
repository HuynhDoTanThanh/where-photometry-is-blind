#!/usr/bin/env python3
"""gate_maps.py: the per-pair teacher maps of the M pilot (plan rows W4-01, W4-02 for the pilot
subset): w (weight), RIGID, F_T (rigid flow from the MapAnything depth and a refined pose) and
b_T (its Laplace scale), for the teacher-ON training pairs. A launch.sh job (spec
tools/p1/specs/gate_maps_pilot.spec.json, job infer, run on the pilot VM that holds the private
MapAnything train cache); core/u2f_providers.py make_teacher_provider reads the result.

What it does, per training pair t -> t+1 of a KITTI-15 TESTING sequence of the declared subset
(HO-tr: the KITTI-15 training sequences are teacher-OFF, see --teacher-off; KITTI-12 is off):
  evidence  (GPU)  U2Flow kitti.pth, forward on (t, t+1) and on (t-1, t) when t-1 is in the window:
                   flow t->t+1, flow t->t-1, alpha (log variance) and visibility of each. NO flow is
                   kept as a result: only the evidence file below, which is scratch (--evidence-dir)
  geometry (CPU)   MapAnything depth and pose of frames t-1, t, t+1 from the cache (GeomIndex),
                   brought to the 256 x 832 grid with the window's ONE K (--k, default calib: the
                   W2-14 recommendation, decisions_2026-10-06.md, recorded in every item)
  E^P              e^alpha <= 2 and visible and valid_T and not OOF and not O_T, per direction, as
                   arbitrate_run.frame_inputs builds it
  pose             refine_pose.refine_pair on the forward pair (PnP on the teacher depth over E^P,
                   stratified by SAM leaf): T_ref and A3's pair_w (0 when PnP fails, the inlier
                   ratio < 0.5, < 300 inliers, or T_ref departs from the teacher pose by > 0.5 deg /
                   10 deg). pair_w = 0 makes the whole frame's weight 0
  gate             arbitrate.build_pair_evidence + gate (A1-A8) over the forward and the backward
                   evidence pairs, pair_w from above. A fold whose pose failed withholds its pixels
                   (fail closed, the reviewed rule); fold_poses_disagree is counted, and withheld
                   too with --disagree withhold (default report: the library only warns)
  maps             F_T, b_T = rigid.render_pair(depth_t, depth_t+1, K, T_ref): flow_12, b_12 on the
                   grid; w = gate weight x valid_T (b_T <= b_max, no front veto). OOF and O_T
                   pixels KEEP the segment's weight on purpose (proposal A8: the weight is applied
                   to every pixel of an accepted segment, occluded and out-of-frame ones too; they
                   are the blind pixels the teacher is for, and beta = 1 for FF); RIGID = the
                   pixel's segment is accepted by the gate (see RIGID below)
Items (one per pair, id = the loader pair id 'kitti15_testing_image_2_000012_05', the key the FF
provider uses) go to <cache-root>/gatemap in the p1lib/cache.py format:
    F_T   fp16 (2, 256, 832)   u then v, grid pixels, frame t -> t+1 (STORED order); 0 where w = 0
    b_T   fp16 (256, 832)      px, >= 0.5 where w > 0 (rigid.B0), 0 elsewhere
    w     fp16 (256, 832)      the A8 weight in [0, 1]
    RIGID packbits (256, 832)  bool
    meta_json: usable, why_unusable, pair_w + reasons, T_ref PnP numbers, the fold failure and
               pose-disagreement counts per pair, tree kind, K choice, counts, cv2 version
OOF, O_T and D_1 are NOT stored: the M loss (core/u2f_losses.py T_NAMES) reads F_T, b_T, w and
RIGID only (w keeps OOF and O_T pixels, see maps); K_aug is not requested either.
Not applied (disclosed in the manifest, settings.not_applied): c_T, the confidence-rank factor of
the proposal's w, and A2's sky and depth-edge tests; the teacher's own confidence is unused.

RIGID. The pixel's segment is accepted by the gate (gate.accept[leaf]), the pixel has a rigid flow
in every pair (no withheld fold) and a usable teacher pixel (valid_T), and the
pair passed A3 (pair_w = 1). It is segment-level on purpose: l_hg drops a segment id when
>= 50 % of its pixels are RIGID (core/u2f_losses.drop_rigid_ids); the 3 px border and the carved
outliers of A8 are zero in w but stay RIGID (their segment is rigid), so L_T (w > 0 AND RIGID) is
unchanged by it.

Trees. The sam:v1 pack on the VM holds the loaders' frames 00-08 and 13-20 (never 09-12), i.e.
every frame this job needs; its full_seg labels and key objects make a TWO-level tree, weaker
than the AMG hierarchy of A1 (arbitrate_run.sam_tree says the same). A frame without a mask file
falls back to the 32 x 32 grid tree: meta.tree = 'grid-fallback' (never silent: the summary and
the summary counts them, the manifest settings record the mode, --tree sam makes it a failure,
--tree grid forces the grid for a bench; the pilot spec uses --tree sam).

How to invoke (VM: python -u; repo root = the staged commit):
    python -u tools/p1/gate_maps.py run --data D --out-dir O --ckpt D/weights/u2flow/kitti.pth \
        --geom-cache D/derived/geommapa/v2 --sam-dir D/derived/sam/v1 \
        --cache-root D/derived/p1cache \
        --subset prereg/drafts/pilot-subset.v2-draft.json --subset-sha256 SHA --max-band 0 \
        --teacher-off prereg/drafts/ho-tr-teacher-off.v2-draft.json --procs 8 \
        [--stage both|evidence|gate]
    python -u tools/p1/gate_maps.py summary --cache-root D/derived/p1cache --out-dir O
Stages: both (default; each worker computes the evidence if its file is missing, then the CPU part),
evidence (GPU only, writes the evidence files), gate (CPU only, no torch: reads them; rerun with
another --disagree or Params without touching the GPU). The evidence files live in --evidence-dir
(default D/derived/p1work/gm-evidence), about 2-3 MB per pair [E], and are kept.
Output lines: as p1lib/jobrun.py, plus GATE_MAPS_SUMMARY {json}.
Exit codes: 0 ok . 1 items left or failed . 2 precondition (a window or frame missing) . 3 refused
(checkpoint pin, stamp, list) . 50 seal . 64 usage . 143 TERM.

Why it is built this way.
- The gate is a library (arbitrate.py) and this file only feeds it, as arbitrate_run.py does; no
  threshold lives here (every constant is the proposal's, arbitrate.Params / refine_pose.PAIR_W).
- Scale: F_T is rendered with T_ref, the PnP pose on the teacher's own depth, so depth and
  translation share a scale (refine_pose header); the gate's evidence masks use the cache's own
  pose, as the reviewed arbitrate_run does.
- Only the forward direction is produced: the hook carries one F_T / b_T / w per pair (u2f_losses
  header), and a temporally swapped pair is zeroed by the hook (allow_swap_zero, disclosed).
- Held-out safety: the list is the subset minus the teacher-off list, KITTI-15 testing only; no GT
  path is ever formed; refuse_sealed guards every path.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import arbitrate as AR  # noqa: E402
import numpy as np  # noqa: E402
import refine_pose as RP  # noqa: E402
import rigid  # noqa: E402
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
TOOL_FILES = [HERE / "gate_maps.py", HERE / "arbitrate.py", HERE / "refine_pose.py",
              HERE / "rigid.py", HERE / "p1lib" / "geomio.py", HERE / "p1lib" / "cache.py"]
GM_DIRNAME = "gatemap"
EV_BYTES_PER_PAIR = 3_000_000   # [E] evidence file, about 2.6 MB measured on arithmetic
_OFF_RX = re.compile(r"(kitti15|kitti12)/(training|testing)/\d{6}")


# ------------------------------------------------------------------------------ the list
def load_teacher_off(path: Path) -> tuple[frozenset, str]:
    """(keys, sha256 of the file bytes). The hook's own format: a JSON list of sequence keys."""
    raw = Path(path).read_bytes()
    keys = json.loads(raw)
    if not isinstance(keys, list) or not keys:
        raise C.Stop(C.EXIT_REFUSED, f"{path}: an empty or non-list teacher-off file is refused")
    for k in keys:
        if not isinstance(k, str) or not _OFF_RX.fullmatch(k):
            raise C.Stop(C.EXIT_REFUSED, f"{path}: not a sequence key: {k!r}")
    return frozenset(keys), hashlib.sha256(raw).hexdigest()


def pair_rows(inv: dict, keys: frozenset, off: frozenset,
              datasets: tuple = ("kitti15",)) -> list[dict]:
    """Teacher-ON training pairs: loader pairs (never frames 09-12) of the sequences in `keys`
    (the subset) that are not in `off` (HO-tr list) and belong to `datasets`. Row id = the loader
    pair id; window = w00-08 (t <= 7) or w13-20 (t >= 13); tm1 = t - 1 when it is in the same
    window (the backward evidence pair), else None."""
    rows = []
    for (ds, sub, side, seq), frames in sorted(inv.items()):
        key = f"{ds}/{sub}/{seq}"
        if ds not in datasets or key not in keys or key in off:
            continue
        have = set(frames)
        for t, _ in KT.training_pairs(frames):
            win = "w00-08" if t + 1 <= 8 else "w13-20"
            lo = 0 if win == "w00-08" else 13
            tm1 = t - 1 if (t - 1 >= lo and t - 1 in have) else None
            rows.append({"id": KT.pair_id(ds, sub, side, seq, t), "ds": ds, "sub": sub,
                         "side": side, "seq": seq, "t": t, "win": win,
                         "win_id": KT.win_id(ds, sub, side, seq, win), "tm1": tm1})
    return rows


def item_seed(item_id: str) -> int:
    return int(hashlib.sha256(f"gm:{item_id}".encode()).hexdigest()[:8], 16)


# ------------------------------------------------------------------------------ SAM trees
def sam_tree(sam_dir, row: dict, frame: int, grid_hw=GRID):
    """The pack's two-level tree of frame `frame` (full_seg labels, then key objects as larger
    masks) or None when the label PNG is absent. Key objects are read as .npy (the authors'
    format) or .npz (key 'key_objects', what tools/vm/sam_masks.py writes), as core/u2f_paths."""
    import cv2

    lab = GI.sam_labels(Path(sam_dir) if sam_dir else None, row, frame, grid_hw)
    if lab is None:
        return None
    masks = [lab == v for v in np.unique(lab) if v > 0]
    stem = Path(sam_dir) / f"{GI.SAM_DS[row['ds']]}_mv_sam1_key_objects" / row["sub"] / row["side"]
    arr = None
    npy, npz = stem / f"{row['seq']}_{frame:02d}.npy", stem / f"{row['seq']}_{frame:02d}.npz"
    if npy.is_file():
        arr = np.load(npy, allow_pickle=False)
    elif npz.is_file():
        with np.load(npz) as z:
            arr = z["key_objects"]
    H, W = grid_hw
    for k in range(arr.shape[2] if arr is not None and arr.ndim == 3 else 0):
        m = cv2.resize((arr[:, :, k] > 127).astype(np.uint8), (W, H),
                       interpolation=cv2.INTER_NEAREST) > 0
        if m.any() and not m.all():
            masks.append(m)
    return AR.build_tree_from_masks(masks, grid_hw)


def pick_tree(mode: str, sam_dir, row: dict, frame: int, grid_hw=GRID):
    """(tree, kind): 'sam2level' | 'grid-fallback' | 'grid'. mode auto falls back (labelled),
    sam refuses a frame without a mask, grid forces the 32 x 32 cells."""
    if mode != "grid":
        t = sam_tree(sam_dir, row, frame, grid_hw)
        if t is not None:
            return t, "sam2level"
        if mode == "sam":
            raise JR.ItemFailed("sam", f"no SAM mask for frame {frame} of {row['id']}")
        return AR.build_tree_from_masks([], grid_hw), "grid-fallback"
    return AR.build_tree_from_masks([], grid_hw), "grid"


# ------------------------------------------------------------------------------ the CPU part
@dataclasses.dataclass
class GateParams:
    k: str = "calib"                    # window K: calib (W2-14 recommendation) | teacher
    disagree: str = "report"            # report | withhold (fold_poses_disagree)
    grid: tuple = GRID


def zero_maps(grid_hw=GRID) -> dict:
    H, W = grid_hw
    return {"F_T": np.zeros((2, H, W), np.float16), "b_T": np.zeros((H, W), np.float16),
            "w": np.zeros((H, W), np.float16), "RIGID": CA.pack_bits(np.zeros((H, W), bool))}


def compute_item(row: dict, ga: dict, gm: dict, ev: dict, tree, tree_kind: str,
                 prm: GateParams | None = None, ap: AR.Params | None = None) -> tuple[dict, dict]:
    """The CPU part of one pair. ga, gm = the window item (GeomIndex.get); ev = the evidence arrays
    {'flow_fwd','alpha_fwd','vis_fwd'[, 'flow_bwd','alpha_bwd','vis_bwd']} (fp16 / packbits as
    stored); tree = the SAM (or grid) tree of frame t. Returns (arrays, meta). A pair that cannot
    be served (no pose, no evidence, the window lacks the frames) returns zero maps with usable
    False and the reason, so the manifest stays complete and the provider serves nothing."""
    prm = prm or GateParams()
    ap = ap or AR.Params()
    grid = tuple(prm.grid)
    t = row["t"]
    frames = [int(f) for f in gm["frames"]]
    meta = {"version": VERSION, "frames": [t, t + 1], "tm1": None, "grid": list(grid),
            "k": prm.k, "k_source": GI.k_source(gm), "tree": tree_kind, "usable": False,
            "why_unusable": None, "disagree_policy": prm.disagree, "cv2": _cv2_version(),
            "params": dataclasses.asdict(ap)}

    def unusable(why: str):
        meta["why_unusable"] = why
        meta["why_class"] = reason_class(why)
        return zero_maps(grid), meta

    if t not in frames or t + 1 not in frames:
        return unusable(f"frames {t}, {t + 1} not both in the window {frames}")
    i1, i2 = frames.index(t), frames.index(t + 1)
    K = GI.window_K(ga, gm, i1, prm.k)
    g1 = GI.geometry_frame(ga, gm, i1, grid, K_model=K)
    g2 = GI.geometry_frame(ga, gm, i2, grid, K_model=K)
    if not (g1["pose_ok"] and g2["pose_ok"]):
        return unusable("no teacher pose")
    if "flow_fwd" not in ev:
        return unusable("no forward evidence")
    seed = item_seed(row["id"])
    labels = tree.leaf.astype(np.int64) + 1
    depth = rigid.depth_from_inv(np.where(g1["valid"], g1["inv"], np.nan))

    pairs, names, ev_f = [], [], None
    dirs = [("fwd", t + 1, i2, g2)]
    tm1 = row.get("tm1")
    if tm1 is not None and tm1 in frames and "flow_bwd" in ev:
        i0 = frames.index(tm1)
        g0 = GI.geometry_frame(ga, gm, i0, grid, K_model=K)
        if g0["pose_ok"]:
            dirs.append(("bwd", tm1, i0, g0))
            meta["tm1"] = tm1
    for name, _other, _i, g2x in dirs:
        M = rigid.relative_pose(g1["pose_c2w"], g2x["pose_c2w"])
        rend = rigid.render_pair(g1["inv"], g2x["inv"], g1["K"], M, grid, valid1=g1["valid"],
                                 valid2=g2x["valid"])
        e = GI.evidence(ev, name, grid)
        # E^P (A4): which pixels give EVIDENCE for the pose fit and the segment statistics. It is
        # not the set of supervised pixels (w below keeps OOF and O_T pixels)
        ok = rend["valid_12"] & ~rend["oof_12"] & ~rend["occ_12"]
        ok &= e["alpha"] <= RP.ALPHA_MAX
        ok &= e["vis"]
        ok &= np.isfinite(e["flow"]).all(axis=0)
        pairs.append(AR.build_pair_evidence(depth, g1["K"], e["flow"], ok, tree,
                                            seed + len(pairs), name=name, p=ap))
        names.append(name)
        if name == "fwd":
            ev_f, M_native = e, M
    # A3: the pose refined on the forward pair, and its weight
    res = RP.refine_pair(g1, g2, ev_f, M_native, labels, seed)
    pw, why = RP.pair_weight(res)
    ref = res.get("ref")
    meta["pair_w"], meta["pair_w_reasons"] = int(pw), why
    meta["ref"] = None if ref is None else {k: ref[k] for k in ("n_in", "n", "rmse_px")}
    meta["pose_dev"] = res.get("dev")
    g = AR.gate(tree, pairs, pair_w=float(pw), p=ap)
    nop = np.zeros(grid, bool)
    for pe in pairs:
        nop |= ~np.isfinite(pe.rig).all(axis=0)
    disagree = len(g.fold_warnings)
    meta.update({"n_pairs": len(pairs), "pair_names": names, "n_folds": 2 * len(pairs),
                 "fold_failures": len(g.fold_failures), "fold_failure_list": g.fold_failures,
                 "fold_poses_disagree": disagree, "fold_warnings": g.fold_warnings,
                 "no_pose_px": int(g.no_pose_px), "carved_px": int(g.carved_px),
                 "road_accepted": bool(g.road["accepted"]), "n_leaf": int(tree.leaf.max()) + 1})
    withheld = prm.disagree == "withhold" and disagree > 0
    meta["withheld_disagree"] = bool(withheld)
    maps = zero_maps(grid)
    if pw and ref is not None and not withheld:
        rt = rigid.render_pair(g1["inv"], g2["inv"], g1["K"], ref["M"], grid, valid1=g1["valid"],
                               valid2=g2["valid"])
        # valid_T only: an OOF or O_T pixel keeps the accepted segment's weight (A8; decision of
        # the main agent, reports/decisions_2026-10-06.md Round 4); its rigid flow is defined
        mT = rt["valid_12"]
        mT &= np.isfinite(rt["flow_12"]).all(axis=0) & np.isfinite(rt["b_12"])
        w = np.where(mT, g.w, 0.0).astype(np.float32)
        rig = g.accept[tree.leaf] & mT & ~nop
        maps = {"F_T": np.where(w > 0, np.nan_to_num(rt["flow_12"], nan=0.0), 0.0)
                .astype(np.float16),
                "b_T": np.where(w > 0, rt["b_12"], 0.0).astype(np.float16),
                "w": w.astype(np.float16), "RIGID": CA.pack_bits(rig)}
        meta["n_rigid"] = int(rig.sum())
    meta["n_w_pos"] = int((maps["w"] > 0).sum())
    meta["w_sum"] = float(maps["w"].astype(np.float64).sum())
    meta["usable"] = meta["n_w_pos"] > 0
    meta["pair_w_classes"] = sorted({reason_class(x) for x in why})
    if not meta["usable"]:
        meta["why_unusable"] = ("pair_w 0: " + "; ".join(why)) if not pw else (
            "withheld: fold poses disagree" if withheld else "no pixel with weight")
        if not pw:
            meta["why_class"] = "pair_w zero"
        elif withheld:
            meta["why_class"] = "withheld: fold poses disagree"
        elif g.pair_w_eff == 0:
            meta["why_class"] = "road segment not accepted"
        else:
            meta["why_class"] = "no pixel with weight"
    return maps, meta


def reason_class(text: str) -> str:
    """A numberless class of a pair_w or unusable reason string (the summary counts classes; the
    sidecars keep the full strings)."""
    t = str(text)
    for key, cls in (("pnp failed", "pnp failed"), ("inlier ratio", "inlier ratio"),
                     ("inliers", "inliers"), ("translation direction", "translation direction"),
                     ("rotation", "rotation"), ("withheld", "withheld: fold poses disagree"),
                     ("road", "road segment not accepted"), ("no teacher pose", "no pose"),
                     ("evidence", "no evidence"),
                     ("not both in the window", "frames not in window"),
                     ("no pixel", "no pixel with weight")):
        if key in t:
            return cls
    return "other"



def _cv2_version() -> str | None:
    try:
        import cv2

        return cv2.__version__
    except ImportError:
        return None


# ------------------------------------------------------------------------------ the GPU part
class Evidence:
    """U2Flow's forward on (t, t+1) and (t-1, t): the arrays compute_item reads (fp16, packbits)."""

    def __init__(self, a, data: Path, model_override=None):
        import torch
        from p1lib import u2flow as UF  # noqa: PLC0415  (imports eval_kitti: torch)

        self.UF, self.data, self.grid = UF, data, tuple(a.test_shape)
        self.device = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available())
                                   else "cpu")
        frac = os.environ.get("U2_P1_WORKER_FRAC")
        if frac and self.device.type == "cuda":
            torch.cuda.set_per_process_memory_fraction(float(frac))
        torch.set_num_threads(max(1, int(os.environ.get("U2_P1_THREADS", "1"))))
        self.model, self.ckpt, self.key = model_override, a.ckpt, a.weights_key
        self.ckpt_sha = "test-model" if model_override is not None else UF.verify_checkpoint(
            Path(a.ckpt), a.weights_key)

    def net(self):
        if self.model is None:
            self.model, _ = self.UF.load_checkpoint(Path(self.ckpt), self.key, self.device)
        return self.model

    def compute(self, row: dict) -> dict:
        rows = [[row["t"], row["t"] + 1]] + ([[row["tm1"], row["t"]]] if row.get("tm1") is
                                               not None else [])
        fl = [[KT.frame_path(self.data, row["ds"], row["sub"], row["side"], row["seq"], f)
               for f in pair] for pair in rows]
        ds = self.UF.frames_dataset(self.data, fl, self.grid)
        out = {}
        for k, (name, kf, ka, kv) in enumerate((("fwd", "f12", "a12", "vis12"),
                                                ("bwd", "b21", "a21", "vis21"))):
            if k >= len(rows):
                break
            e = self.UF.bidirectional(self.net(), ds[k][0], self.device)
            f = e[kf][0].detach().float().cpu().numpy()
            if not np.isfinite(f).all():
                raise JR.ItemFailed("nonfinite", f"{name} evidence flow")
            out[f"flow_{name}"] = f.astype(np.float16)
            out[f"alpha_{name}"] = e[ka][0, 0].detach().float().cpu().numpy().astype(np.float16)
            out[f"vis_{name}"] = CA.pack_bits(e[kv][0, 0].detach().cpu().numpy() > 0.5)
        return out


# ------------------------------------------------------------------------------ the job
class Producer:
    def __init__(self, a, data: Path, geo: GI.GeomIndex, evdir: Path, model_override=None):
        self.a, self.data, self.geo, self.evdir = a, data, geo, evdir
        self.prm = GateParams(k=a.k, disagree=a.disagree, grid=tuple(a.test_shape))
        self.ap = AR.Params()
        self.sam_dir = Path(a.sam_dir) if a.sam_dir else None
        self.model_override = model_override
        self._ev = None
        self._win: tuple[str, tuple] | None = None
        self.meta_dir = Path(a.meta_dir) if getattr(a, "meta_dir", None) else None

    def evidence(self) -> Evidence:
        if self._ev is None:
            self._ev = Evidence(self.a, self.data, self.model_override)
        return self._ev

    def expected_ckpt(self) -> str:
        """The checkpoint the evidence must come from: the weights.json pin (verify_checkpoint
        refuses any other file), or the test model's tag."""
        if self.model_override is not None or os.environ.get("U2_P1_STUB") == "1":
            return "test-model"   # CPU tests only (the stub flag is refused on a GPU host)
        return C.weights_pin(self.a.weights_key)["sha256"]

    def ev_path(self, item_id: str) -> Path:
        return Path(self.evdir) / f"{item_id}.npz"

    def get_evidence(self, row: dict, compute: bool = True) -> dict | None:
        p = self.ev_path(row["id"])
        if p.is_file():
            arrays, meta = CA.read_item(p)
            want = self.expected_ckpt()
            if meta.get("ckpt_sha256") != want or meta.get("grid") != list(self.prm.grid):
                raise C.Stop(C.EXIT_REFUSED,
                             f"{p}: evidence made with another checkpoint or grid "
                             f"({str(meta.get('ckpt_sha256'))[:12]}, {meta.get('grid')}), expected "
                             f"{want[:12]}, {list(self.prm.grid)}: use another --evidence-dir")
            return arrays
        if not compute:
            return None
        arrays = self.evidence().compute(row)
        CA.write_item(Path(self.evdir), row["id"], arrays,
                      {"item": row["id"], "ckpt_sha256": self.evidence().ckpt_sha,
                       "grid": list(self.prm.grid), "frames": [row["tm1"], row["t"], row["t"] + 1]})
        return arrays

    def window(self, row: dict):
        if self._win is None or self._win[0] != row["win_id"]:
            got = self.geo.get(row["win_id"])
            if got is None:
                raise C.Stop(C.EXIT_PRE, f"{row['win_id']}: not in the geometry cache")
            self._win = (row["win_id"], got)
        return self._win[1]

    def produce_evidence(self, row: dict):
        self.get_evidence(row)
        return {"ok": np.ones(1, np.uint8)}, {"stage": "evidence"}

    def produce(self, row: dict):
        ev = self.get_evidence(row, compute=self.a.stage != "gate")
        if ev is None:
            raise JR.ItemFailed("evidence", f"no evidence file for {row['id']} "
                                            "(run --stage evidence)")
        ga, gm = self.window(row)
        tree, kind = pick_tree(self.a.tree, self.sam_dir, row, row["t"], self.prm.grid)
        arrays, meta = compute_item(row, ga, gm, ev, tree, kind, self.prm, self.ap)
        if self.meta_dir is not None:  # a small sidecar: the summary never opens the shards
            C.atomic_json(self.meta_dir / f"{row['id']}.json", {k: v for k, v in meta.items()
                                                              if k not in ("params",)})
        return arrays, meta


def summarise(meta_dir: Path) -> dict:
    """Pooled counts over the sidecars of a run (counts, never rates of rates)."""
    n = usable = 0
    tot = {"fold_failures": 0, "n_folds": 0, "fold_poses_disagree": 0, "n_pairs": 0,
           "no_pose_px": 0, "n_w_pos": 0}
    trees, why, pwc, k_src = {}, {}, {}, set()
    items_disagree = items_failfold = pw0 = 0
    for p in sorted(Path(meta_dir).glob("*.json")):
        m = json.loads(p.read_text())
        n += 1
        usable += int(bool(m.get("usable")))
        trees[m.get("tree")] = trees.get(m.get("tree"), 0) + 1
        k_src.add(f"{m.get('k')}/{m.get('k_source')}")
        for k in tot:
            tot[k] += int(m.get(k) or 0)
        items_disagree += int((m.get("fold_poses_disagree") or 0) > 0)
        items_failfold += int((m.get("fold_failures") or 0) > 0)
        pw0 += int(m.get("pair_w") == 0)
        if not m.get("usable"):
            w = m.get("why_class") or reason_class(m.get("why_unusable"))
            why[w] = why.get(w, 0) + 1
        for c in m.get("pair_w_classes") or []:   # an item can fail A3 for several reasons
            pwc[c] = pwc.get(c, 0) + 1
    return {"items": n, "usable": usable, "unusable_by_class": why, "pair_w_0_by_class": pwc,
            "tree_counts": trees,
            "k": sorted(k_src), "pooled": tot, "items_with_fold_failure": items_failfold,
            "items_with_pose_disagreement": items_disagree, "items_pair_w_0": pw0,
            "tree_note": "sam2level = the pack's full_seg + key objects, not the AMG hierarchy of "
                         "A1; grid-fallback = a frame without a SAM mask (32 x 32 cells)"}


# ------------------------------------------------------------------------------ setup / CLI
def setup(a, model_override=None):
    data = C.data_root(a.data)
    out = Path(a.out_dir)
    C.refuse_sealed(out, a.work_dir, a.ckpt, a.cache_root, a.evidence_dir, a.geom_cache,
                    a.sam_dir)
    if not (a.subset and a.subset_sha256 and a.teacher_off):
        raise C.Stop(C.EXIT_USAGE, "--subset, --subset-sha256 and --teacher-off are required")
    try:
        sub = US.load_subset(Path(a.subset) if Path(a.subset).is_absolute()
                             else C.REPO / a.subset, a.subset_sha256)
    except (OSError, ValueError, US.SubsetError) as e:
        raise C.Stop(C.EXIT_REFUSED, f"--subset: {e}") from e
    off, off_sha = load_teacher_off(Path(a.teacher_off) if Path(a.teacher_off).is_absolute()
                                    else C.REPO / a.teacher_off)
    keys = sub.keys(int(a.max_band))
    rows = pair_rows(KT.inventory_from_data(data), keys, off)
    if not rows:
        raise C.Stop(C.EXIT_REFUSED, "no teacher-on pair (the subset minus the teacher-off list)")
    if any(r["sub"] != "testing" or r["ds"] != "kitti15" for r in rows):
        raise C.Stop(C.EXIT_REFUSED, "a teacher-on pair outside the KITTI-15 testing sequences")
    if any({r["t"], r["t"] + 1} & KT.EXCLUDED for r in rows):
        raise C.Stop(C.EXIT_REFUSED, "a pair touches frames 09-12")
    geo = GI.GeomIndex(Path(a.geom_cache or data / "derived" / "geommapa" / "v2"))
    missing = sorted({r["win_id"] for r in rows if r["win_id"] not in set(geo.ids())})
    if missing:
        raise C.Stop(C.EXIT_PRE, f"{len(missing)} window(s) not in the geometry cache, first: "
                                 f"{missing[0]}")
    evdir = Path(a.evidence_dir) if a.evidence_dir else data / "derived" / "p1work" / "gm-evidence"
    if a.stage != "gate":   # about 2.6 MB per pair [E]: a full disk must be a refusal, not failures
        evdir.mkdir(parents=True, exist_ok=True)
        need = int(len(rows) * EV_BYTES_PER_PAIR * 1.3)
        free = shutil.disk_usage(evdir).free
        if free < need:
            raise C.Stop(C.EXIT_PRE, f"{evdir}: {free / 2**30:.1f} GiB free, the evidence files of "
                                     f"{len(rows)} pairs need about {need / 2**30:.1f} GiB")
    selection = {"file": Path(sub.path).name, "file_sha256": sub.sha256,
                 "table_sha256": sub.doc["table_sha256"], "max_band": int(a.max_band),
                 "teacher_off_sha256": off_sha, "teacher_off_n": len(off), "n": len(rows),
                 "rule": "subset band <= max_band, KITTI-15, minus the teacher-off list "
                         "(HO-tr: K15 training off; K12 off): the K15 testing sequences"}
    settings = {"kind": "gatemap", "version": VERSION, "k": a.k, "disagree": a.disagree,
                "tree": a.tree, "grid": list(a.test_shape),
                "params": dataclasses.asdict(AR.Params()),
                "geom_cache": {"path": str(geo.path), "identity": geo.manifest.get("identity")},
                "sam_dir": a.sam_dir, "selection": selection, "stage": a.stage,
                "evidence_dir": str(evdir), "weights_key": a.weights_key,
                "outputs": "F_T fp16 (2,256,832), b_T fp16, w fp16, RIGID packbits; "
                           "OOF, O_T, D_1 and K_aug not stored (M's loss does not read them); w "
                           "keeps OOF and O_T pixels (A8)",
                "not_applied": "c_T (confidence-rank factor of the proposal's w; the teacher's "
                               "confidence is unused), A2 sky and depth-edge tests, R1 relaxed "
                               "evidence; w = A8 gate weight x valid_T"}
    identity = {"settings": {k: v for k, v in settings.items() if k not in ("stage",)},
                "code": C.code_identity(TOOL_FILES)}
    work = Path(a.work_dir) if a.work_dir else (
        data / "derived" / "p1work"
        / f"gm-{C.sha256_bytes(repr(sorted(identity.items())).encode())[:12]}")
    root = Path(a.cache_root) if a.cache_root else out / "cache"
    C.refuse_sealed(root, work)
    ev_stage = a.stage == "evidence"
    cache_dir = (evdir / "_index") if ev_stage else root / GM_DIRNAME
    if ev_stage:
        work = work.with_name(work.name + "-ev")
        identity = {"settings": {"kind": "gatemap-evidence", "version": VERSION,
                                 "selection": selection, "weights_key": a.weights_key},
                    "code": C.code_identity([HERE / "gate_maps.py"])}
    a.meta_dir = str(work / ".meta")
    Path(a.meta_dir).mkdir(parents=True, exist_ok=True)
    prod = Producer(a, data, geo, evdir, model_override)
    job = JR.ItemJob(kind="gatemap-evidence" if ev_stage else "gatemap", source="mapanything",
                     items=rows, cache_dir=cache_dir, work_dir=work,
                     shard_size=a.shard_size or 250, shard=a.shard, identity=identity,
                     settings=settings, limit=a.limit, selection=selection)
    return prod, job, rows, work, cache_dir


def args_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "worker", "summary"):
        p = sub.add_parser(name)
        p.add_argument("--data")
        p.add_argument("--out-dir", required=True)
        p.add_argument("--ckpt")
        p.add_argument("--weights-key", default="u2flow-kitti")
        p.add_argument("--geom-cache", help="the MapAnything train cache (default "
                                            "DATA/derived/geommapa/v2)")
        p.add_argument("--sam-dir", help="sam:v1 root (DATA/derived/sam/v1)")
        p.add_argument("--subset")
        p.add_argument("--subset-sha256")
        p.add_argument("--max-band", type=int, default=0)
        p.add_argument("--teacher-off")
        p.add_argument("--k", default="calib", choices=GI.K_CHOICES)
        p.add_argument("--disagree", default="report", choices=("report", "withhold"))
        p.add_argument("--tree", default="auto", choices=("auto", "sam", "grid"))
        p.add_argument("--stage", default="both", choices=("both", "evidence", "gate"))
        p.add_argument("--evidence-dir")
        p.add_argument("--cache-root", help="where cache/gatemap lives (stays on the VM)")
        p.add_argument("--work-dir")
        p.add_argument("--shard")
        p.add_argument("--shard-size", type=int)
        p.add_argument("--limit", type=int)
        p.add_argument("--device", default="cuda")
        p.add_argument("--test-shape", type=int, nargs=2, default=list(GRID))
        p.add_argument("--procs", type=int, default=1)
        p.add_argument("--run", default="")
    return ap


def passthrough(a) -> list[str]:
    out = ["--out-dir", a.out_dir, "--weights-key", a.weights_key, "--max-band", str(a.max_band),
           "--k", a.k, "--disagree", a.disagree, "--tree", a.tree, "--stage", a.stage,
           "--device", a.device, "--test-shape", *[str(x) for x in a.test_shape]]
    for flag, v in (("--data", a.data), ("--ckpt", a.ckpt), ("--geom-cache", a.geom_cache),
                    ("--sam-dir", a.sam_dir), ("--subset", a.subset),
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
        if a.cmd == "summary":
            work = Path(a.work_dir) if a.work_dir else None
            if work is None:
                raise C.Stop(C.EXIT_USAGE, "summary needs --work-dir (it holds meta/)")
            doc = summarise(work / ".meta")
            C.atomic_json(Path(a.out_dir) / "gate_maps_summary.json", doc)
            C.say("GATE_MAPS_SUMMARY", json.dumps(doc, sort_keys=True))
            return C.EXIT_OK
        prod, job, rows, work, cache_dir = setup(a, model_override)
        produce = prod.produce_evidence if a.stage == "evidence" else prod.produce
        ident = {"tool": "tools/p1/gate_maps.py", "run": a.run, "stage": a.stage,
                 "n_items": job.total, "procs": a.procs, "cache_dir": str(cache_dir),
                 "work_dir": str(work), "k": a.k, "disagree": a.disagree, "tree": a.tree}
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
            C.atomic_json(Path(a.out_dir) / f"manifest_{job.kind}.json",
                          C.load_json(cache_dir / "manifest.json"))
            if a.stage != "evidence":
                doc = summarise(work / ".meta")
                C.atomic_json(Path(a.out_dir) / "gate_maps_summary.json", doc)
                C.say("GATE_MAPS_SUMMARY", json.dumps(doc, sort_keys=True))
        return rc
    except C.Stop as e:
        C.say("P1_ERROR", e.msg)
        return e.code


if __name__ == "__main__":
    sys.exit(main())
