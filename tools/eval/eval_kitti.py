#!/usr/bin/env python3
"""Run U2Flow on a KITTI split with the frozen validate_kitti_2015 semantics, dump the float32
flows in sha-pinned shards, and write metrics.json with per-image counts (dev and k12 only).

How to invoke (repo root; on a G4 VM through the colab-fleet launch path, or on the Mac on MPS):
    python tools/eval/eval_kitti.py --split dev --ckpt checkpoints/kitti.pth \
        --data-root /content/data --out runs/eval-dev            # VM: GT under DATA_ROOT
    python tools/eval/eval_kitti.py --split dev --data-root DIR --gt-root \
        /Volumes/Shin/U2FLOW_data/gt --ids 000001,000006 --out OUT   # Mac smoke (<= 10 ids)
    python tools/eval/eval_kitti.py --split k12 ...                  # KITTI-12 train, ids 0-193
    python tools/eval/eval_kitti.py --split full200 --g0-audit ...   # predictions only, no GT
    python tools/eval/eval_kitti.py --split dev --parity-link1 ...   # + the 152bb38 blob beside
    python tools/eval/eval_kitti.py --split dev --parity-link0 [--ref-dir D] ...
        # + the whole unported 152bb38 tree (dataset, network, autocast, evaluator) on the same
        #   pairs in a child process: max |delta flow| <= 1e-5 and identical outlier counts
    python tools/eval/eval_kitti.py --split full200 --g0-audit --ff-seed 1 ...
        # +FF (train_fusion_kitti.py's per-image TinyModel, as released) seeded with 1, frames
        #   09/10/11; metrics.json records ff = {seed, iters, thr, lr, ...}
Then score on the Mac: python tools/eval/score_cached.py --run-dir OUT (parity link 2).

Outputs under --out: metrics.json; per_image.csv (dev, k12); preds/shard-NNN.npz (zip, stored,
'<id>.npy' = (2, H, W) float32 u, v at GT resolution after resize_flow) and preds/PREDS.json
(the shard table; pred_sha = sha256 of its canonical 'shards' list); parity_link1.json;
parity_link0.json (and link0/ with the reference tree's flows).
stdout: EVAL_OK (paths, counts, shas; never a metric value unless --print-aggregate is allowed).

Exit codes: 0 EVAL_OK | 1 error | 3 refused by a gate (no prereg/GATE-0.json yet for a
reproduction number of a released checkpoint; all-200 off the aggregate-only allowlist) |
6 parity link 0 or 1 failed: no number may be quoted | 50 seal: a held-out id was asked for, or
full200 without --g0-audit (a STOP, AGENTS.md section 4) | 64 usage.

Semantics kept (metric skill section 1): the authors' KITTI2015/KITTI2012 dataset classes with
only their 'your_folder' paths rewritten per instance (image_2 frames _10 -> _11, /255, Zoom to
--test-shape, default 256x832); raft_2f_u built from configs/KITTI_MV with iters = 12 and
mixed_precision = True set explicitly (T24) [V evaluate.py:976-978]; model(x, only_fw=True)
['flows_f12'][-1]; resize_flow to each GT's (H, W) [V :540-543]; scoring by regions.py. The
autocast inside the net is torch.cuda.amp, so it is active only on CUDA: on MPS/CPU the run is
fp32 and metrics.json says so (params.autocast_effective); gpu_name keys the difference.
Device and path handling are ours: no authors' file is edited (codebase notes: none needed).

Why it is built this way
- T17: evaluate.py's __main__ hides the GPU (CUDA_VISIBLE_DEVICES='6'); importing functions does
  not run it, and nothing here runs it. T19: --ckpt with map_location, and ckpt_sha recorded.
- T22: no U2Flow PNG writer is a valid prediction (uint16, 1/64 px); only these float32 dumps
  are predictions, and score_cached re-hashes every shard against PREDS.json.
- Hard rule 4: held-out GT never reaches a VM, so full200 here only predicts; the all-200
  aggregate is score_cached --g0-audit's, on the Mac. A held-out id in dev is exit 50.
- PLAN Tier 1 (09-27): GATE-0.json is committed BEFORE any reproduction number, so a run of a
  released checkpoint over more than SMOKE_MAX_IDS ids refuses until prereg/GATE-0.json exists.
- Parity link 1 (metric skill section 3) runs validate_kitti_2015 and resize_flow as loaded from
  the 152bb38 blobs (ids checked) on the same dataset object and the same model outputs (a memo
  keyed by the input bytes, so every blob call must hit it), so only the evaluator differs.
  Outlier and valid counts must be identical. The blob's floats are float32 (torch .mean() per
  image, 100 * np.mean of a float32 array; trap T8), so they are compared with the same float32
  arithmetic applied here to the same images (max_abs_diff, expected 0, bound 1e-6); our
  published values are exact (float64 sums, Fl from counts) and their rounding gap to the blob
  is reported as *_exact_vs_blob (measured 2026-10-02 on 2 dev ids, MPS: <= 6e-7).
- The codebase port routes loader roots through core/u2f_paths.py and refuses held-out GT at
  construction; make_dataset works with both that and the 152bb38 constructors.
- Parity link 0 (Tier 1 review): link 1 feeds the blob the SAME cached model outputs, so it cannot
  see a change in the ported network forward (raft_2f_u's torch.amp autocast, .to(device)) or the
  ported dataset class. Link 0 runs the 152bb38 tree itself (staged by stage_code.py under
  _ref/152bb38, else `git archive 152bb38`) in a child process on the same image files, on CUDA
  with its own torch.cuda.amp fp16 autocast, and compares every resized flow with this run's
  shards. GATE-0's +-0.05 EPE band is about 20x the measured audit deviation, so only this check
  would catch a forward drift on the G4.
- +FF (GATE-0 G0.3; evaluator-traps T26): the released script scores inline against GT and is
  unseeded, so nothing produced +FF predictions. --ff-seed N runs its own
  train_and_run_tiny_model (imported, unchanged: iters, thr, lr from the module) seeded with N
  exactly where U2_FF_SEED seeds it (after the model is loaded), dumps the fused flows like any
  run, and records ff = {seed, iters, thr, lr, script, script_sha256} in metrics.json, the only
  place score_cached takes it from (a typed meta 'ff' is refused).
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import csv
import datetime as dt
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import time
import types
import warnings
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(CODE_ROOT), str(CODE_ROOT / "tools" / "vm"),
                str(CODE_ROOT / "tools" / "lib")]
sys.path.append(str(CODE_ROOT / "core"))   # core/Networks/corr.py imports 'utils.utils'
import frozen  # noqa: E402  (tools/lib: the allowlist and GATE-0.json as prereg froze them)
import regions as rg  # noqa: E402

EXIT_OK, EXIT_ERR, EXIT_GATE, EXIT_PARITY, EXIT_SEAL, EXIT_USAGE = 0, 1, 3, 6, 50, 64
SMOKE_MAX_IDS = 10
K12_IDS = tuple(f"{i:06d}" for i in range(194))       # [V flow_datasets.py:199]
K15_IDS = tuple(f"{i:06d}" for i in range(200))       # [V flow_datasets.py:276]
SPLITS = {"dev": "kitti15", "k12": "kitti12", "full200": "kitti15"}
ALLOWLIST_REL = "prereg/inputs/aggregate_only_allowlist.json"
GATE0_REL = "prereg/GATE-0.json"
SPLIT_REL = "splits/k15_split_v1.json"
TOL_LINK1 = 1e-6
TOL_LINK0 = 1e-5            # px, max |delta flow| (GATE-0 G0.2 link 0)
FF_SCRIPT = "train_fusion_kitti.py"
ZIP_DATE = (1980, 1, 1, 0, 0, 0)


class Refused(Exception):
    def __init__(self, code: int, msg: str) -> None:
        super().__init__(msg)
        self.code = code


def repo_root() -> Path:
    """Where splits/, prereg/ and data/manifest/ live: $U2_REPO_ROOT (tests) or this repo."""
    env = os.environ.get("U2_REPO_ROOT")
    return Path(env).resolve() if env else CODE_ROOT


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(text, "utf-8")
    os.replace(tmp, path)


# ------------------------------------------------------------------------------ split and gates
def split_ids(root: Path) -> tuple[list[str], list[str]]:
    doc = json.loads((root / SPLIT_REL).read_text("utf-8"))
    return list(doc["dev_ids"]), list(doc["heldout_ids"])


def resolve_ids(split: str, ids: list[str] | None, root: Path, g0_audit: bool) -> list[str]:
    """The ids to run, in split order. Held-out ids exist only under full200 --g0-audit."""
    if split not in SPLITS:
        raise Refused(EXIT_SEAL if split == "heldout" else EXIT_USAGE,
                      f"split {split!r}: held-out scoring is unseal.py's (prereg), never a run"
                      if split == "heldout" else f"unknown split {split!r}")
    if split == "full200":
        if not g0_audit:
            raise Refused(EXIT_SEAL, "full200 exists only as the GATE-0 aggregate-only audit "
                          "(--g0-audit): all-200 minus dev is the held-out value")
        if ids:
            raise Refused(EXIT_USAGE, "full200 runs all 200 ids; --ids is refused")
        return list(K15_IDS)
    universe = split_ids(root)[0] if split == "dev" else list(K12_IDS)
    if split == "dev":
        held = set(split_ids(root)[1])
        bad = [i for i in (ids or []) if i in held]
        if bad:
            raise Refused(EXIT_SEAL, f"{len(bad)} held-out id(s) asked for under split dev; the "
                          "held-out half is sealed (hard rule 4)")
    if not ids:
        return universe
    unknown = [i for i in ids if i not in universe]
    if unknown:
        raise Refused(EXIT_USAGE, f"ids not in split {split}: {unknown[:5]}")
    return [i for i in universe if i in set(ids)]


def released_ckpts(root: Path) -> set[str]:
    try:
        w = json.loads((root / "data" / "manifest" / "weights.json").read_text("utf-8"))
    except (OSError, ValueError):
        return set()
    return {v["sha256"] for k, v in (w.get("weights") or {}).items()
            if k.startswith("u2flow-") and isinstance(v, dict) and v.get("sha256")}


def aggregate_allowlist(root: Path) -> set[str]:
    """The released kitti.pth plus prereg-v1's FROZEN allowlist (tools/lib/frozen.py, shared with
    note.py and score_cached); a working-tree edit after the tag is refused (exit 3)."""
    try:
        return frozen.aggregate_allowlist(root)
    except frozen.FrozenError as e:
        raise Refused(EXIT_GATE, str(e)) from None


def gate0_committed(root: Path) -> bool:
    """The GATE-0.json in force is committed (prereg-v1's copy, else HEAD's, equal to the working
    file); on a VM only a staged commit (U2_GIT_SHA) counts (tools/lib/frozen.py gate0_state)."""
    return bool(frozen.gate0_state(root)["committed"])


def check_gates(split: str, ids: list[str], ckpt_sha: str, root: Path, partial: bool) -> None:
    if split == "full200":
        if not gate0_committed(root):
            raise Refused(EXIT_GATE, f"{GATE0_REL} must be committed before the all-200 audit")
        if ckpt_sha not in aggregate_allowlist(root):
            raise Refused(EXIT_GATE, f"checkpoint {ckpt_sha[:12]} is not on the aggregate-only "
                          f"allowlist ({ALLOWLIST_REL}; before prereg-v1 only kitti.pth)")
    if ckpt_sha in released_ckpts(root) and not gate0_committed(root) \
            and len(ids) > SMOKE_MAX_IDS:
        raise Refused(EXIT_GATE, f"a released checkpoint over {len(ids)} ids is a reproduction "
                      f"number: commit {GATE0_REL} first (PLAN Tier 1, 09-27), or run a smoke "
                      f"of <= {SMOKE_MAX_IDS} ids with --ids")


def gt_bundle(dataset: str, gt_root: Path | None, root: Path) -> dict:
    """The GT bundle this run reads: name, manifest sha256 (gt_pack_sha), integrity state."""
    name = rg.GT_BUNDLE[dataset]
    man = json.loads((root / "data" / "manifest" / "bundles.v1.json").read_text("utf-8"))
    sha = ((man.get("bundles") or {}).get(name) or {}).get("sha256")
    info = {"name": name, "gt_pack_sha": sha, "integrity": "unverified"}
    if gt_root is None:
        return info
    ok = gt_root / ".bundles" / f"{name}.ok"
    if ok.is_file() and sha and sha not in ok.read_text("utf-8", "replace"):
        raise rg.IntegrityError(f"{ok} does not record the manifest sha256 of {name}")
    if (gt_root / ".bundles" / name / "FILES.sha256").is_file():
        info["integrity"] = "verified"
    return info


# ------------------------------------------------------------------------------ environment
def pick_device(name: str):
    import torch

    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)


def _dists_sha() -> str:
    from importlib import metadata

    lines = sorted({f"{d.metadata['Name']}=={d.version}".lower() for d in metadata.distributions()
                    if d.metadata["Name"]})
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def env_info(device) -> dict:
    """gpu_name, cc, driver and lock_hash of the host that produces pred_sha. On a VM they come
    from bootstrap.py's status (vm_lock_hash, metric skill section 6); elsewhere from the same
    lock_lines/lock_hash_of_lines over this host's versions (lock_source 'host')."""
    import bootstrap
    import cv2
    import torch

    st = Path(os.environ.get("U2_VM_ROOT", "/content")) / "u2flow" / "bootstrap.status"
    versions = {"python": sys.version.split()[0], "torch": torch.__version__,
                "numpy": np.__version__, "cv2": cv2.__version__}
    if st.is_file():
        rec = json.loads(st.read_text("utf-8"))
        v = rec.get("versions") or {}
        cc = v.get("cc") or ""
        return {"gpu_name": v.get("gpu_name", ""),
                "cc": [int(x) for x in cc.split(".")] if cc else None,
                "driver": v.get("driver", ""), "lock_hash": rec.get("vm_lock_hash"),
                "lock_source": "vm-bootstrap", "versions": {**versions, **v}}
    if device.type == "cuda":
        gpu, cc = torch.cuda.get_device_name(device), list(torch.cuda.get_device_capability(device))
    else:
        gpu, cc = ("Apple MPS" if device.type == "mps" else "cpu"), None
    vals = {"cc": "" if cc is None else f"{cc[0]}.{cc[1]}", "cuda": torch.version.cuda or "",
            "cudnn": str(torch.backends.cudnn.version() or ""), "cv2": cv2.__version__,
            "driver": "", "gpu_name": gpu, "numpy": np.__version__,
            "pip_freeze_sha256": _dists_sha(), "python": versions["python"],
            "sm_120_in_arch_list": "false", "torch": torch.__version__,
            "torchvision": _version("torchvision"), "vm_requirements_sha256": ""}
    return {"gpu_name": gpu, "cc": cc, "driver": "",
            "lock_hash": bootstrap.lock_hash_of_lines(bootstrap.lock_lines(vals)),
            "lock_source": "host", "versions": vals}


def _version(mod: str) -> str:
    try:
        return __import__(mod).__version__
    except ImportError:
        return ""


def git_info() -> dict:
    """The commit the run launched from. A stager without .git passes U2_GIT_SHA (clean by
    construction: it staged that commit). Otherwise git; dirty = an evaluator file differs from
    HEAD or is untracked, which makes the run a smoke (score_cached writes no rows for it)."""
    env = os.environ.get("U2_GIT_SHA")
    if env:
        return {"git_sha": env, "git_dirty": False, "git_source": "U2_GIT_SHA"}
    try:
        sha = subprocess.run(["git", "-C", str(CODE_ROOT), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        st = subprocess.run(["git", "-C", str(CODE_ROOT), "status", "--porcelain", "--",
                             *rg.eval_files(CODE_ROOT)], capture_output=True, text=True,
                            check=True).stdout
        return {"git_sha": sha, "git_dirty": bool(st.strip()), "git_source": "git"}
    except (OSError, subprocess.CalledProcessError):
        return {"git_sha": None, "git_dirty": True, "git_source": "none"}


# ------------------------------------------------------------------------------ data and model
@contextlib.contextmanager
def _loader_roots(data_root: Path, gt_root: Path | None):
    """U2_DATA/U2_GT for the ported loaders (core/u2f_paths.py), restored afterwards; the 152bb38
    loaders ignore them. The instance's paths are rewritten below either way."""
    old = {k: os.environ.get(k) for k in ("U2_DATA", "U2_GT")}
    os.environ["U2_DATA"] = str(data_root)
    os.environ["U2_GT"] = str(gt_root or data_root)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def make_dataset(dataset: str, data_root: Path, gt_root: Path | None, ids: list[str],
                 test_shape: tuple[int, int], with_gt: bool = True, three_frame: bool = False):
    """The authors' dataset class, one instance, its paths rewritten to the bundle layout
    (DATA_ROOT/<ds>/multi_view/training/image_2, GT_ROOT/<ds>/training/flow_*). Works with the
    152bb38 constructors and with the codebase port's (ids=..., held-out GT refused at
    construction). Without GT (full200) the instance is built as 'testing', so no GT path is
    ever formed, and then pointed at the training frames."""
    import inspect

    from core import flow_datasets as fd

    cls = fd.KITTI2015 if dataset == "kitti15" else fd.KITTI2012
    kw: dict = {"test_shape": list(test_shape), "split": "training" if with_gt else "testing"}
    if three_frame:
        if dataset != "kitti15":
            raise ValueError("+FF here is KITTI-15 only (train_fusion_kitti.py)")
        kw["three_frame"] = True
    if with_gt and "ids" in inspect.signature(cls.__init__).parameters:
        kw["ids"] = [int(i) for i in ids]
    with _loader_roots(data_root, gt_root):
        ds = cls(**kw)
    img = data_root / dataset / "multi_view" / "training" / "image_2"
    frames = ("09", "10", "11") if three_frame else ("10", "11")
    ds.image_list = [[str(img / f"{i}_{f}.png") for f in frames] for i in ids]
    if with_gt:
        gt = gt_root / dataset / "training"
        ds.flow_list = [str(gt / "flow_occ" / f"{i}_10.png") for i in ids]
        ds.flow_noc = [str(gt / "flow_noc" / f"{i}_10.png") for i in ids]
    else:
        ds.flow_list, ds.flow_noc, ds.is_test = [], [], True
        ds.extra_info = [f"{i}_10.png" for i in ids]
    missing = [p for p in [q for pair in ds.image_list for q in pair] + ds.flow_list
               + ds.flow_noc if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} input file(s) missing, first: {missing[0]}")
    return ds


def load_model(ckpt: Path, device):
    import torch

    from configs.KITTI_MV import get_cfg
    from core.Networks import build_network

    cfg = get_cfg()
    cfg.mixed_precision = True
    cfg[cfg.network].mixed_precision = True
    cfg.raft_2f_u.iters = 12
    model = build_network(cfg)
    sd = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()


# ------------------------------------------------------------------------------ shards
class ShardWriter:
    """Deterministic shards: a stored zip of '<id>.npy' members with a fixed date, written to
    .part and renamed. Identical flows give identical bytes, so pred_sha is reproducible."""

    def __init__(self, pred_dir: Path, shard_size: int = 25) -> None:
        self.dir, self.size = pred_dir, max(1, shard_size)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.buf: list[tuple[str, np.ndarray]] = []
        self.shards: list[dict] = []

    def add(self, img_id: str, flow: np.ndarray) -> None:
        if flow.dtype != np.float32 or flow.ndim != 3 or flow.shape[0] != 2:
            raise ValueError(f"{img_id}: a prediction is (2, H, W) float32, got "
                             f"{flow.dtype} {flow.shape}")
        self.buf.append((img_id, flow))
        if len(self.buf) >= self.size:
            self._flush()

    def _flush(self) -> None:
        if not self.buf:
            return
        name = f"shard-{len(self.shards):03d}.npz"
        tmp = self.dir / (name + ".part")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as zf:
            for img_id, arr in self.buf:
                b = io.BytesIO()
                np.lib.format.write_array(b, np.ascontiguousarray(arr), allow_pickle=False)
                zi = zipfile.ZipInfo(f"{img_id}.npy", date_time=ZIP_DATE)
                zi.external_attr = 0o644 << 16
                zf.writestr(zi, b.getvalue())
        os.replace(tmp, self.dir / name)
        self.shards.append({"file": name, "sha256": rg.sha256_file(self.dir / name),
                            "ids": [i for i, _ in self.buf]})
        self.buf = []

    def close(self) -> str:
        self._flush()
        doc = {"schema": 1, "dtype": "float32",
               "layout": "(2, H, W): u, v at GT resolution after resize_flow",
               "shards": self.shards}
        atomic_write(self.dir / "PREDS.json", json.dumps(doc, indent=1, sort_keys=True) + "\n")
        return pred_sha(self.shards)


def pred_sha(shards: list[dict]) -> str:
    return rg.sha256_bytes(json.dumps(shards, sort_keys=True, separators=(",", ":")).encode())


def read_preds(pred_dir: Path, verify: bool = True):
    """Yield (id, flow) from PREDS.json's shards, re-hashing each shard first (link 2 rule)."""
    doc = json.loads((pred_dir / "PREDS.json").read_text("utf-8"))
    for sh in doc["shards"]:
        p = pred_dir / sh["file"]
        if verify and rg.sha256_file(p) != sh["sha256"]:
            raise rg.IntegrityError(f"{p.name}: sha256 differs from PREDS.json")
        with zipfile.ZipFile(p) as zf:
            for img_id in sh["ids"]:
                with zf.open(f"{img_id}.npy") as f:
                    yield img_id, np.lib.format.read_array(io.BytesIO(f.read()),
                                                           allow_pickle=False)


# ------------------------------------------------------------------------------ the run
def ff_module():
    """train_fusion_kitti.py, imported unchanged (its module-level print goes to stderr)."""
    with contextlib.redirect_stdout(sys.stderr):
        import train_fusion_kitti as tfk
    return tfk


def ff_info(seed: int) -> dict:
    tfk = ff_module()
    return {"seed": int(seed), "iters": int(tfk.iters), "thr": float(tfk.thr),
            "lr": float(tfk.lr), "script": FF_SCRIPT, "U2_FF_SEED": str(int(seed)),
            "script_sha256": rg.sha256_file(CODE_ROOT / FF_SCRIPT), "frames": ["09", "10", "11"],
            "procedure": "train_fusion_kitti.py __main__: model(10,11) and model(09,10), "
                         "occlusion masks, train_and_run_tiny_model, resize_flow"}


def ff_forward(model, images, device):
    """One +FF prediction exactly as train_fusion_kitti.py's __main__ makes it: the two-frame
    model on (10, 11) and on (09, 10), the bidirectional occlusion masks, then the TinyModel
    fitted per image (needs autograd, so it runs outside no_grad)."""
    import torch

    from core.utils.warp_utils import get_occu_mask_bidirection

    tfk = ff_module()
    x = images[None].to(device)
    with torch.no_grad():
        o12 = model(x[:, 1:])
        o01 = model(x[:, :-1])
    f12, f21 = o12["flows_f12"][-1], o12["flows_b21"][-1]
    f01, f10 = o01["flows_f12"][-1], o01["flows_b21"][-1]
    vis_f = 1 - get_occu_mask_bidirection(f12, f21)
    vis_b = 1 - get_occu_mask_bidirection(f10, f01)
    fused, _ = tfk.train_and_run_tiny_model(f12, f10, vis_f, vis_b, o12["uncertainty_f12"][-1],
                                            o01["uncertainty_b21"][-1],
                                            iterations=int(tfk.iters), device=device)
    return fused


def predict_and_score(model, ds, ids: list[str], dataset: str, device, writer: ShardWriter,
                      gt_src: rg.GTSource | None, only_all: bool = False,
                      ff: bool = False) -> list[dict]:
    """The validate_kitti_2015 loop: forward, resize_flow to the GT size, dump, score. The
    dataset's GT tensors (readFlowKITTI) must equal regions.py's verified decode. With FF the
    forward is train_fusion_kitti.py's (ff_forward)."""
    import torch

    from core.utils.flow_utils import resize_flow

    per_image = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*autocast.*")
        warnings.filterwarnings("ignore", message=".*CUDA is not available.*")
        for k, img_id in enumerate(ids):
            item = ds[k]
            if gt_src is None:
                images, _, h, w = item
            else:
                images, flow_gt, valid_gt, flownoc, nocmask = item
                _, h, w = flow_gt.size()
            if ff:
                flow_pred = ff_forward(model, images, device)
            else:
                with torch.no_grad():
                    flow_pred = model(images[None].to(device), only_fw=True)["flows_f12"][-1]
            with torch.no_grad():
                up = resize_flow(flow_pred, (h, w))[0].detach().cpu().numpy()
            up = up.astype(np.float32, copy=False)
            writer.add(img_id, up)
            if gt_src is None:
                continue
            gt = gt_src.read(img_id, need_obj=not only_all)
            same = (np.array_equal(gt.flow_occ, flow_gt.numpy())
                    and np.array_equal(gt.valid_occ, valid_gt.numpy())
                    and np.array_equal(gt.flow_noc, flownoc.numpy())
                    and np.array_equal(gt.valid_noc, nocmask.numpy()))
            if not same:
                raise rg.IntegrityError(f"{img_id}: regions.py's GT decode differs from "
                                        "readFlowKITTI's")
            rec = {"id": img_id, "H": int(h), "W": int(w),
                   **rg.score_image(up, gt, dataset, only_all=only_all)}
            if not only_all:
                rec["f32"] = blob_style_means(up, gt, dataset)
            per_image.append(rec)
    return per_image


def blob_style_means(pred: np.ndarray, gt: rg.GTImage, dataset: str) -> dict:
    """Per-image epe/noc/occ exactly as validate_kitti_2015/2012 compute them: torch float32
    `.mean().item()` on CPU, with their guards. Only parity link 1 uses these (like arithmetic
    on identical data, so the blob must match to the bit); every number we publish comes from
    the float64 epe_sum. None stands for the NaN of an unguarded empty mean (KITTI-12 noc)."""
    import torch

    p, fo, fn = (torch.from_numpy(np.ascontiguousarray(x)) for x in (pred, gt.flow_occ,
                                                                      gt.flow_noc))
    vo, vn = torch.from_numpy(gt.valid_occ), torch.from_numpy(gt.valid_noc)
    epe = torch.sum((p - fo) ** 2, dim=0).sqrt().view(-1)
    noc_e = torch.sum((p - fn) ** 2, dim=0).sqrt().view(-1)
    val, val_noc = vo.view(-1) >= 0.5, vn.view(-1) >= 0.5
    res = {"epe": epe[val].mean().item()}
    if dataset == "kitti15" and not torch.any(val_noc):
        res["noc"] = 0.0
    else:
        res["noc"] = noc_e[val_noc].mean().item()
    diff = vo - vn
    res["occ"] = 0.0 if (diff == 0).all() else epe[diff.view(-1) >= 0.5].mean().item()
    return {k: (None if v != v else v) for k, v in res.items()}


def write_per_image_csv(path: Path, per_image: list[dict]) -> None:
    regs = list(per_image[0]["r"])
    keys = list(per_image[0]["r"][regs[0]])
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["id", "H", "W", "u2f_noc", "u2f_occ"] + [f"{r}:{k}" for r in regs for k in keys])
    for p in per_image:
        w.writerow([p["id"], p["H"], p["W"], p.get("u2f_noc"), p.get("u2f_occ")]
                   + [repr(p["r"][r][k]) for r in regs for k in keys])
    atomic_write(path, buf.getvalue())


def run(*, split: str, ids: list[str], model, device, data_root: Path, gt_root: Path | None,
        out: Path, ckpt: str, ckpt_sha: str, test_shape=(256, 832), shard_size: int = 25,
        g0_audit: bool = False, root: Path | None = None, ff_seed: int | None = None) -> dict:
    """Everything after argument parsing; tests call it with a fake model. With FF_SEED, the
    +FF procedure seeded with it (seeded here, after the model is built: U2_FF_SEED's place)."""
    root = root or repo_root()
    dataset = SPLITS[split]
    if ff_seed is not None and dataset != "kitti15":
        raise Refused(EXIT_USAGE, "+FF here is KITTI-15 only (train_fusion_kitti.py)")
    full = resolve_ids(split, None, root, g0_audit)
    partial = list(ids) != list(full)
    check_gates(split, ids, ckpt_sha, root, partial)
    with_gt = split != "full200"
    if with_gt and gt_root is None:
        raise Refused(EXIT_USAGE, "--gt-root (or --data-root holding the GT bundle) is needed")
    for p in (data_root, gt_root, out):
        if p is not None and "u2flow_data/sealed" in str(Path(p).resolve()).lower():
            raise Refused(EXIT_SEAL, f"{p} is under the seal; never an eval input or output")
    ds = make_dataset(dataset, data_root, gt_root, ids, test_shape, with_gt=with_gt,
                      three_frame=ff_seed is not None)
    gt_src = rg.GTSource(dataset, root=gt_root, require_r4=True) if with_gt else None
    bundle = gt_bundle(dataset, gt_root, root) if with_gt else None
    if gt_src is not None and bundle["integrity"] == "verified" and not gt_src.verified:
        bundle["integrity"] = "unverified"
    esha, efiles = rg.eval_sha(CODE_ROOT)
    g0 = frozen.gate0_state(root)  # stamped: gate.py refuses a GO under other criteria
    t0 = time.time()
    writer = ShardWriter(out / "preds", shard_size)
    ff = None
    if ff_seed is not None:
        from core import u2f_runtime

        ff = ff_info(ff_seed)
        u2f_runtime.seed_everything(int(ff_seed))
    per_image = predict_and_score(model, ds, ids, dataset, device, writer, gt_src,
                                  ff=ff is not None)
    psha = writer.close()
    import torch

    metrics = {
        "schema": 1, "tool": "tools/eval/eval_kitti.py", "metric_version": rg.METRIC_VERSION,
        "regions_version": rg.REGIONS_VERSION, "dataset": dataset, "split": split,
        "ids": list(ids), "n_images": len(ids), "partial": partial, "g0_audit": g0_audit,
        "ckpt": str(ckpt), "ckpt_sha": ckpt_sha,
        "released_ckpt": ckpt_sha in released_ckpts(root),
        "gt_bundle": None if bundle is None else bundle["name"],
        "gt_pack_sha": None if bundle is None else bundle["gt_pack_sha"],
        "gt_integrity": "none" if bundle is None else bundle["integrity"],
        "eval_sha": esha, "eval_files": efiles, "pred_sha": psha, "preds": "preds",
        "gate0": g0, "gate0_sha256": g0["sha256"] if g0["committed"] else None, "ff": ff,
        "params": {"iters": 12, "mixed_precision": True, "test_shape": list(test_shape),
                   "only_fw": True, "iterate": "last", "device": str(device),
                   "autocast_effective": bool(device.type == "cuda"
                                              and torch.cuda.is_available())},
        "env": env_info(device), **git_info(), "utc": utc_now(),
        "seconds": round(time.time() - t0, 1),
    }
    if with_gt:
        metrics["per_image"] = per_image
        metrics["aggregate"] = rg.aggregate(per_image, dataset)
        metrics["checks"] = {"S1": "asserted", "S2": "asserted", "S3": "asserted",
                             "S4_noc_not_in_total": sum(p["checks"]["S4_noc_not_in"]
                                                        for p in per_image)}
        write_per_image_csv(out / "per_image.csv", per_image)
    else:
        metrics["aggregate"] = None
        metrics["note"] = ("predictions only: the all-200 aggregate is made on the Mac by "
                           "score_cached.py --g0-audit (held-out GT never reaches a VM)")
    atomic_write(out / "metrics.json", json.dumps(metrics, indent=1, sort_keys=True) + "\n")
    return metrics


def may_print(m: dict, root: Path) -> bool:
    return (m.get("aggregate") is not None and not m["partial"]
            and (not m["released_ckpt"] or gate0_committed(root)))


# ------------------------------------------------------------------------------ parity link 1
class Memo:
    """Model outputs keyed by the input bytes, so the blob and this tool score the same flows;
    a blob call that misses means the two fed different inputs."""

    def __init__(self, model) -> None:
        self.model, self.cache, self.hits, self.misses = model, {}, 0, 0

    def eval(self):
        self.model.eval()
        return self

    def __call__(self, images, only_fw=True):
        x = images.detach().cpu().contiguous()
        key = (tuple(x.shape), hashlib.sha256(x.numpy().tobytes()).hexdigest())
        if key in self.cache:
            self.hits += 1
        else:
            self.misses += 1
            self.cache[key] = self.model(images, only_fw=only_fw)["flows_f12"][-1].detach()
        return {"flows_f12": [self.cache[key].clone()]}


def load_blob(rel: str, blob_file: Path | None = None) -> str:
    if blob_file is not None:
        data = Path(blob_file).read_bytes()
    else:
        data = subprocess.run(["git", "-C", str(CODE_ROOT), "show", f"{rg.PARITY_REF}:{rel}"],
                              capture_output=True, check=True).stdout
    bid = rg.git_blob_id(data)
    if not bid.startswith(rg.BLOB_IDS[rel]):
        raise rg.IntegrityError(f"{rel}: blob {bid[:12]} is not {rg.PARITY_REF}'s "
                                f"{rg.BLOB_IDS[rel]}")
    return data.decode("utf-8")


def blob_function(src: str, name: str, namespace: dict):
    tree = ast.parse(src)
    node = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name), None)
    if node is None:
        raise rg.IntegrityError(f"{name} not found in the blob")
    code = ast.get_source_segment(src, node)
    exec(compile(code, f"<{rg.PARITY_REF}:{name}>", "exec"), namespace)  # noqa: S102
    return namespace[name]


class _NpSpy(types.SimpleNamespace):
    """numpy for the blob, recording each np.concatenate (the pooled outlier arrays)."""

    def __init__(self) -> None:
        super().__init__()
        self.concats: list[np.ndarray] = []

    def __getattr__(self, name):
        return getattr(np, name)

    def concatenate(self, arrs, *a, **k):
        r = np.concatenate(arrs, *a, **k)
        self.concats.append(r)
        return r


@contextlib.contextmanager
def _cuda_as(device):
    import torch

    if device.type == "cuda":
        yield
        return
    orig = torch.Tensor.cuda
    torch.Tensor.cuda = lambda self, *a, **k: self.to(device)
    try:
        yield
    finally:
        torch.Tensor.cuda = orig


def run_blob(model, ds, device, test_shape, blob_dir: Path | None = None) -> dict:
    """validate_kitti_2015 and resize_flow from the 152bb38 blobs, on `ds`, with `model`."""
    import torch

    # A staged VM tree has no .git, so `git show` fails there (measured on G4 2026-10-03,
    # EXP-001-rel-s1): read the blobs from the 152bb38 tree stage_code.py ships under _ref/,
    # each still checked against its git blob id in load_blob.
    if blob_dir is None and (CODE_ROOT / "_ref" / rg.PARITY_REF).is_dir():
        blob_dir = CODE_ROOT / "_ref" / rg.PARITY_REF

    def blob_path(rel):
        return None if blob_dir is None else Path(blob_dir) / rel

    ru = blob_function(load_blob("core/utils/flow_utils.py", blob_path("core/utils/flow_utils.py")),
                       "resize_flow", {"torch": torch})
    spy = _NpSpy()
    ns = {"torch": torch, "np": spy, "tqdm": lambda it, **k: it, "time": time, "os": os,
          "resize_flow": ru, "frame_utils": None,
          "flow_datasets": types.SimpleNamespace(KITTI2015=lambda test_shape=None, **k: ds)}
    fn = blob_function(load_blob("evaluate.py", blob_path("evaluate.py")),
                       "validate_kitti_2015", ns)
    buf = io.StringIO()
    with torch.no_grad(), _cuda_as(device), contextlib.redirect_stdout(buf), \
            warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = fn(model, test_shape=list(test_shape))
    line = [ln for ln in buf.getvalue().splitlines() if ln.startswith("Validation KITTI")][-1]
    nums = [float(x) for x in re.findall(r"-?\d+\.\d+|nan", line.split(":", 1)[1])]
    printed = dict(zip(["epe", "fl", "noc", "occ", "fl_noc", "fl_occ"], nums, strict=True))
    return {"returned": {k: float(v) for k, v in res.items()}, "printed": printed,
            "concats": [(int(c.size), int(c.sum())) for c in spy.concats]}


def fl_float32(n_out: int, n_valid: int) -> float:
    """The blob's Fl arithmetic on given counts: 100 * np.mean(float32 0/1 array). The float32
    sum of 0/1 values is exact below 2^24, np.mean divides in float32, and `100 *` follows this
    numpy's promotion (float32 under NEP 50, float64 before; trap T8)."""
    if not n_valid:
        return 0.0
    return float(100 * (np.float32(n_out) / np.float32(n_valid)))


def compare_link1(agg: dict, blob: dict, n_calls: int, hits: int,
                  per_image: list[dict] | None = None) -> dict:
    """Pass: identical outlier/valid counts (all, noc, occ); |delta| <= 1e-6 between the blob's
    epe, noc, occ, fl and printed fl_noc/fl_occ (%f) and the same float32 arithmetic applied by
    this tool to the same images (blob_style_means, fl_float32); every blob call a memo hit.
    Our published values are exact (float64 sums, Fl from integer counts), so they differ from
    the blob's float32 ones by rounding only (Fl: up to 1.9e-6 at Fl 16-32, trap T8); those gaps
    are reported as *_exact_vs_blob, never hidden in a tolerance."""
    r, u = agg["regions"], agg["u2f"]
    emu = dict(u)
    if per_image and all("f32" in p for p in per_image):
        for k in ("epe", "noc", "occ"):
            vals = [np.nan if p["f32"][k] is None else p["f32"][k] for p in per_image]
            emu[k] = float(np.mean(np.array(vals)))
    want = [(r["all"]["n_valid"], r["all"]["n_out"])]
    if r["noc"]["n_valid"]:
        want.append((r["noc"]["n_valid"], r["noc"]["n_out"]))
    if r["occ"]["n_valid"]:
        want.append((r["occ"]["n_valid"], r["occ"]["n_out"]))
    fails = []
    if [tuple(c) for c in blob["concats"]] != want:
        fails.append("outlier/valid counts differ (all, noc, occ)")
    f32 = {k: fl_float32(r[reg]["n_out"], r[reg]["n_valid"])
           for k, reg in (("fl", "all"), ("fl_noc", "noc"), ("fl_occ", "occ"))}
    deltas = {k: abs(emu[k] - blob["returned"][k]) for k in ("epe", "noc", "occ")}
    exact = {f"{k}_exact_vs_blob": abs(u[k] - blob["returned"][k])
             for k in ("epe", "noc", "occ", "fl")}
    deltas["fl"] = abs(f32["fl"] - blob["returned"]["fl"])
    # fl_noc/fl_occ exist only in the blob's print, formatted '%f': format ours the same way
    deltas.update({k: abs(float(f"{f32[k]:f}") - blob["printed"][k]) for k in ("fl_noc",
                                                                             "fl_occ")})
    fails += [f"|delta {k}| > {TOL_LINK1}" for k, d in deltas.items() if not d <= TOL_LINK1]
    if hits != n_calls:
        fails.append(f"{n_calls - hits} blob call(s) fed inputs this tool never fed the model")
    same = not any("counts" in f for f in fails)
    return {"ok": not fails, "fails": fails, "max_abs_delta": deltas, "tolerance": TOL_LINK1,
            "max_abs_diff": max(deltas.values()), "outliers_equal": same,
            **exact,
            "counts_identical": same, "ref": rg.PARITY_REF, "blob_ids": rg.BLOB_IDS}


# ------------------------------------------------------------------------------ parity link 0
LINK0_CHILD = r"""
import json, os, sys
import numpy as np
import torch
ref, args = sys.argv[1], json.loads(sys.argv[2])
os.chdir(ref)
sys.path[:0] = [ref, os.path.join(ref, "core")]
dev = torch.device(args["device"])
if dev.type != "cuda":  # 152bb38 calls .cuda(); on a Mac test the same device stands in
    torch.Tensor.cuda = lambda self, *a, **k: self.to(dev)
import evaluate as ev
from core import flow_datasets as fd
from configs.KITTI_MV import get_cfg
from core.Networks import build_network
ids, img, gt = args["ids"], args["img"], args["gt"]
_orig = fd.KITTI2015.__init__
def _init(self, *a, **k):
    _orig(self, *a, **k)
    self.image_list = [[f"{img}/{i}_10.png", f"{img}/{i}_11.png"] for i in ids]
    self.flow_list = [f"{gt}/flow_occ/{i}_10.png" for i in ids]
    self.flow_noc = [f"{gt}/flow_noc/{i}_10.png" for i in ids]
fd.KITTI2015.__init__ = _init
cfg = get_cfg()
cfg.mixed_precision = True
cfg[cfg.network].mixed_precision = True
cfg.raft_2f_u.iters = 12
model = torch.nn.DataParallel(build_network(cfg))
model.load_state_dict(torch.load(args["ckpt"], map_location="cpu", weights_only=True), strict=True)
model.to(dev)
model.eval()
flows = []
_rf = ev.resize_flow
def _rec(flow, size):
    up = _rf(flow, size)
    flows.append(up[0].detach().float().cpu().numpy().astype(np.float32))
    return up
ev.resize_flow = _rec
with torch.no_grad():
    r = ev.validate_kitti_2015(model=model.module, test_shape=args["test_shape"])
np.savez(args["out"], **{i: f for i, f in zip(ids, flows)})
print("LINK0_JSON " + json.dumps({"returned": {k: float(v) for k, v in r.items()},
                                  "n": len(flows)}))
"""


def ref_tree(ref_dir: Path | None, work: Path) -> tuple[Path, dict]:
    """The unported 152bb38 tree: --ref-dir, else <code>/_ref/152bb38 (stage_code.py stages it on
    a VM), else `git archive 152bb38` into WORK. Its evaluate.py and flow_utils.py must be
    152bb38's blobs; every evaluator file's blob id is returned for the record."""
    cand = Path(ref_dir) if ref_dir else CODE_ROOT / "_ref" / rg.PARITY_REF
    if not (cand / "evaluate.py").is_file():
        if ref_dir:
            raise Refused(EXIT_USAGE, f"--ref-dir {ref_dir} holds no evaluate.py")
        try:
            arc = subprocess.run(["git", "-C", str(CODE_ROOT), "archive", rg.PARITY_REF, "core",
                                  "configs", "evaluate.py"], capture_output=True,
                                 check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            raise Refused(EXIT_USAGE, "no 152bb38 tree: stage it (stage_code.py packs "
                          "_ref/152bb38) or pass --ref-dir") from None
        cand = work / f"ref{rg.PARITY_REF}"
        import tarfile

        with tarfile.open(fileobj=io.BytesIO(arc)) as tf:
            try:
                tf.extractall(cand, filter="data")
            except TypeError:  # a Python without extraction filters: git's own archive
                tf.extractall(cand)
    blobs = {}
    for rel in rg.EVAL_FILES:
        f = cand / rel
        if f.is_file():
            blobs[rel] = rg.git_blob_id(f.read_bytes())[:12]
    for rel, want in rg.BLOB_IDS.items():
        if not blobs.get(rel, "").startswith(want):
            raise rg.IntegrityError(f"{cand}/{rel}: blob {blobs.get(rel)} is not "
                                    f"{rg.PARITY_REF}'s {want}")
    return cand, blobs


def compare_link0(ours: dict, theirs: dict, gt_src: rg.GTSource, ids: list[str]) -> dict:
    """max |delta flow| over every pixel of every id, and identical per-region counts (n, out)
    when both flows are scored against the same GT."""
    fails, worst, same = [], 0.0, True
    for i in ids:
        a, b = ours.get(i), theirs.get(i)
        if a is None or b is None or a.shape != b.shape:
            fails.append(f"{i}: missing or shaped differently")
            same = False
            continue
        worst = max(worst, float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))))
        gt = gt_src.read(i, need_obj=True)
        ra, rb = rg.score_image(a, gt, "kitti15")["r"], rg.score_image(b, gt, "kitti15")["r"]
        if any(ra[k]["n"] != rb[k]["n"] or ra[k]["out"] != rb[k]["out"] for k in ra):
            same = False
            fails.append(f"{i}: outlier counts differ")
    if not worst <= TOL_LINK0:
        fails.append(f"max |delta flow| {worst:.3g} > {TOL_LINK0}")
    return {"ok": not fails, "fails": fails[:20], "n_fails": len(fails),
            "max_abs_flow_diff": worst, "tolerance": TOL_LINK0, "outliers_equal": same}


def run_link0(m: dict, *, ids: list[str], data_root: Path, gt_root: Path, out: Path,
              ckpt: Path, device, test_shape, ref_dir: Path | None = None,
              timeout_s: float = 7200) -> dict:
    work = out / "link0"
    work.mkdir(parents=True, exist_ok=True)
    ref, blobs = ref_tree(ref_dir, work)
    args = {"device": str(device), "ids": list(ids), "ckpt": str(Path(ckpt).resolve()),
            "img": str((data_root / "kitti15" / "multi_view" / "training" / "image_2").resolve()),
            "gt": str((gt_root / "kitti15" / "training").resolve()),
            "out": str((work / "ref_flows.npz").resolve()), "test_shape": list(test_shape)}
    env = {k: v for k, v in os.environ.items() if not k.startswith("U2_")}
    env["PYTHONPATH"] = ""
    r = subprocess.run([sys.executable, "-c", LINK0_CHILD, str(ref), json.dumps(args)],
                       cwd=ref, env=env, capture_output=True, text=True, timeout=timeout_s)
    line = [x for x in r.stdout.splitlines() if x.startswith("LINK0_JSON ")]
    if r.returncode != 0 or not line:
        raise rg.IntegrityError(f"the 152bb38 child failed (exit {r.returncode}): "
                                f"{r.stderr.strip()[-600:]}")
    child = json.loads(line[-1][len("LINK0_JSON "):])
    with np.load(args["out"]) as z:
        theirs = {k: z[k] for k in z.files}
    ours = dict(read_preds(out / "preds"))
    gt_src = rg.GTSource("kitti15", root=gt_root, require_r4=True)
    try:
        res = compare_link0(ours, theirs, gt_src, ids)
    finally:
        gt_src.close()
    doc = {**res, "link": 0, "split": m["split"], "n_images": len(ids), "partial": m["partial"],
           "device": str(device), "gpu_name": m["env"]["gpu_name"],
           "autocast_effective": m["params"]["autocast_effective"], "ref": rg.PARITY_REF,
           "ref_dir": str(ref), "ref_blobs": blobs, "ref_returned": child["returned"],
           "eval_sha": m["eval_sha"], "pred_sha": m["pred_sha"], "ckpt_sha": m["ckpt_sha"],
           "gate0_sha256": m.get("gate0_sha256"), "utc": utc_now()}
    atomic_write(out / "parity_link0.json", json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return doc


# ------------------------------------------------------------------------------ CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=("dev", "k12", "full200", "heldout"))
    ap.add_argument("--ckpt", default=str(CODE_ROOT / "checkpoints" / "kitti.pth"))
    ap.add_argument("--data-root", help="DATA_ROOT with <ds>/multi_view/training/image_2")
    ap.add_argument("--gt-root", help="GT root (default: --data-root); Mac: U2FLOW_data/gt")
    ap.add_argument("--out", help="output directory (created)")
    ap.add_argument("--ids", help="comma-separated subset of the split (a smoke: partial)")
    ap.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    ap.add_argument("--test-shape", nargs=2, type=int, default=[256, 832])
    ap.add_argument("--shard-size", type=int, default=25)
    ap.add_argument("--g0-audit", action="store_true")
    ap.add_argument("--parity-link1", action="store_true")
    ap.add_argument("--parity-link0", action="store_true",
                    help="also run the unported 152bb38 tree on the same pairs (dev only)")
    ap.add_argument("--ref-dir", help="the 152bb38 tree (default <code>/_ref/152bb38, else git)")
    ap.add_argument("--ff-seed", type=int, default=None,
                    help="+FF (train_fusion_kitti.py) seeded with this seed; KITTI-15 only")
    ap.add_argument("--blob-dir", help="staged 152bb38 files (evaluate.py, core/utils/"
                    "flow_utils.py) when the VM has no git history; ids are checked")
    ap.add_argument("--print-aggregate", action="store_true")
    ap.add_argument("--print-eval-sha", action="store_true")
    a = ap.parse_args(argv)
    if a.print_eval_sha:
        print(rg.eval_sha(CODE_ROOT)[0])
        return EXIT_OK
    if not (a.split and a.data_root and a.out):
        ap.print_usage(sys.stderr)
        print("eval_kitti: --split, --data-root and --out are required", file=sys.stderr)
        return EXIT_USAGE
    root = repo_root()
    try:
        ids = resolve_ids(a.split, [s for s in (a.ids or "").split(",") if s] or None, root,
                          a.g0_audit)
        if (a.parity_link1 or a.parity_link0) and a.split != "dev":
            raise Refused(EXIT_USAGE, "parity links 0 and 1 run on dev ids (metric skill "
                          "section 3)")
        if a.ff_seed is not None and (a.parity_link1 or a.parity_link0 or a.split == "k12"):
            raise Refused(EXIT_USAGE, "--ff-seed is KITTI-15 +FF; the parity links check the "
                          "plain evaluator")
        ckpt = Path(a.ckpt)
        ckpt_sha = rg.sha256_file(ckpt)
        check_gates(a.split, ids, ckpt_sha, root, True)
        device = pick_device(a.device)
        model = load_model(ckpt, device)
        memo = Memo(model) if a.parity_link1 else model
        data_root = Path(a.data_root)
        gt_root = Path(a.gt_root) if a.gt_root else data_root
        out = Path(a.out)
        m = run(split=a.split, ids=ids, model=memo, device=device, data_root=data_root,
                gt_root=gt_root, out=out, ckpt=str(ckpt), ckpt_sha=ckpt_sha,
                test_shape=tuple(a.test_shape), shard_size=a.shard_size, g0_audit=a.g0_audit,
                root=root, ff_seed=a.ff_seed)
        print(f"EVAL_OK split={a.split} n={m['n_images']} partial={m['partial']} "
              f"pred_sha={m['pred_sha'][:12]} eval_sha={m['eval_sha'][:12]} "
              f"ckpt_sha={ckpt_sha[:12]} out={out}")
        if a.print_aggregate:
            if may_print(m, root):
                print("EVAL_AGG " + json.dumps(m["aggregate"]["u2f"], sort_keys=True))
            else:
                print("EVAL_AGG withheld: a partial run, or a released checkpoint before "
                      f"{GATE0_REL} exists", file=sys.stderr)
        if a.parity_link1:
            hits0 = memo.hits
            ds = make_dataset("kitti15", data_root, gt_root, ids, tuple(a.test_shape))
            blob = run_blob(memo, ds, device, tuple(a.test_shape),
                            Path(a.blob_dir) if a.blob_dir else None)
            res = compare_link1(m["aggregate"], blob, len(ids), memo.hits - hits0,
                                m["per_image"])
            atomic_write(out / "parity_link1.json", json.dumps(
                {**res, "link": 1, "split": a.split, "n_images": len(ids),
                 "gpu_name": m["env"]["gpu_name"], "eval_sha": m["eval_sha"],
                 "pred_sha": m["pred_sha"], "partial": m["partial"], "utc": utc_now()},
                indent=1, sort_keys=True) + "\n")
            if not res["ok"]:
                print("PARITY1_FAIL " + "; ".join(res["fails"]))
                return EXIT_PARITY
            print("PARITY1_OK max_abs_delta=" + json.dumps(res["max_abs_delta"], sort_keys=True))
        if a.parity_link0:
            res0 = run_link0(m, ids=ids, data_root=data_root, gt_root=gt_root, out=out,
                             ckpt=ckpt, device=device, test_shape=tuple(a.test_shape),
                             ref_dir=Path(a.ref_dir) if a.ref_dir else None)
            if not res0["ok"]:
                print("PARITY0_FAIL " + "; ".join(res0["fails"][:5]))
                return EXIT_PARITY
            print(f"PARITY0_OK max_abs_flow_diff={res0['max_abs_flow_diff']:.3g} "
                  f"outliers_equal={res0['outliers_equal']}")
        return EXIT_OK
    except Refused as e:
        print(f"eval_kitti: refused: {e}", file=sys.stderr)
        return e.code
    except (rg.IntegrityError, rg.RegionError, FileNotFoundError, OSError, ValueError) as e:
        print(f"eval_kitti: {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_ERR


if __name__ == "__main__":
    sys.exit(main())
