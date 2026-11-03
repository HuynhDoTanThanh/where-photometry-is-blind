"""The SAM producer: U²Flow's full segmentation and key-object masks for every KITTI frame its
MV_Sem loaders read, written in exactly the loaders' layout, resumable, then packed as sam:v1.

What it does (a VM job; data-bundles references/derived.md section 7 is the contract).
    plan      lists the frames KITTI2015_MV_Sem and KITTI2012_MV_Sem read [V core/flow_datasets.py
              :217-254, :295-332 at bae8cf7]: in kitti15/ and kitti12/ multi_view/{testing,
              training}/image_{2,3}/, every frame of a consecutive pair that touches no frame 09-12
              (26,760 frames, 23,604 pairs [M 2026-10-03, from data/manifest/files/*-mv-*.sha256];
              --from-manifest counts them on the Mac, --data on a VM).
    run       the job: install the pinned segment_anything (env/vm-requirements-sam.txt) into
              U2_DATA/pydeps/, verify the SAM ViT-H checkpoint against data/manifest/weights.json,
              check the frame list against the real loader classes, write resolved_config.json,
              start --procs workers (static shards, staggered), report progress on the bus, then
              write the manifest and, with --pack, run tools/data/pack_derived.py pack.
    worker    one shard (started by run; never by hand).
    manifest  (re)writes OUT/sam_manifest.json and OUT/sam_files.sha256 from what is on disk.
Per frame <sub>/<side>/<name>.png the outputs are, under --sam-dir (default U2_SAM_DIR, else
U2_DATA/derived/sam/v1, the loaders' own default [V core/u2f_paths.py:77-78, 106-112]):
    <ds>_mv_sam1_key_objects/<sub>/<side>/<stem>.npz   uint8 H x W x K, 0/255, one channel per key
        object (np.savez_compressed, key 'key_objects'; core/u2f_paths.load_key_objects reads it
        when <stem>.npy is absent, codebase commit 8); --key-format npy writes the authors' .npy
    <ds>_mv_sam1_fullseg/<sub>/<side>/<name>.png       uint8 H x W label map, 0 = no mask
    (--rle) U2_DATA/derived/samrle/v1/<ds>_mv_sam1_rle/<sub>/<side>/<stem>.json.gz: SAM's own
        records (uncompressed RLE, area, bbox, predicted_iou, point_coords, stability_score,
        crop_box), gzip mtime 0; pycocotools.frPyObjects turns them into UnSAMFlow's coco_rle JSON
<ds> is kitti2015 or kitti2012. Frames 09-12 are never written (pack refuses them in sam).
--eval-frames (PX-F3, 2026-10-12): instead, frame 10 of every KITTI-15 (200) and KITTI-12 (194)
    training image_2, the frame the evaluators read [V core/flow_datasets.py:200-203], same recipe
    and file layout, into the private kind evalsam v1 (U2_DATA/derived/evalsam/v1, RLE records in
    the same dir with --rle). pack_derived allows frames 09-12 only in eval* kinds, which are never
    uploadable or fetched; no MV_Sem loader reads frame 10, so the loader check is not applicable.
    Images only: no GT is read (the held-out ids' GT stays sealed; their images are not).

How to invoke.
    Through the launcher (tools/vm/sam_v1.spec.json; job infer, profile sam):
        python -u tools/vm/sam_masks.py run --data {data} --out-dir {out_dir} --procs 4 \
            --pack --run {run_id}
    Mac:  .venv/bin/python tools/vm/sam_masks.py plan --from-manifest data/manifest
    VM:   python -u tools/vm/sam_masks.py plan --data /content/data
          python -u tools/vm/sam_masks.py run ... --limit 200 --procs 1 \
              --sam-dir /content/data/derived/sambench/p1     # a bench: s/frame, its own dir
          python -u tools/vm/sam_masks.py manifest --data /content/data --out-dir D
Output lines: SAM_PLAN, SAM_DEPS, SAM_WEIGHT (and PIN), SAM_LOADER_CHECK, SAM_STAMP, SAM_START,
SAM_PROGRESS, SAM_WORKER_EXIT, SAM_MANIFEST, SAM_PACK, SAM_DONE, SAM_REFUSED, SAM_ERROR.
Exit codes: 0 ok · 1 frames left or a worker failed (run it again: done frames are skipped) or an
unexpected file in the output tree · 2 precondition (missing image dirs, segment_anything archive
or checkpoint, a frame list that differs from the loaders', a --pack run whose loader check could
not import core) · 3 refused (checkpoint sha256 differs from the pin, a stub generator on a GPU
host, --pack of a non-faithful run or of another --sam-dir, a non-faithful run into the v1 dirs,
an output dir whose settings stamp differs from this run's or that holds outputs without one) ·
50 a seal refusal passed through from pack_derived.py (STOP, AGENTS.md section 4) · 64 usage ·
143 stopped by TERM.

Why it is built this way.
- Fidelity first. U²Flow's README says its masks come from the official segment-anything and
  UnSAMFlow's sam_inference.py [V U2FLOW README.md:33]. UnSAMFlow uses "the default ViT-H SAM
  model" [V facebookresearch/UnSAMFlow@0363aed README.md:31] with SamAutomaticMaskGenerator and no
  AMG flag set (get_amg_kwargs drops every None [V sam_inference.py:190-205]), so the generator
  defaults apply: points_per_side 32, points_per_batch 64, pred_iou_thresh 0.88,
  stability_score_thresh 0.95, stability_score_offset 1.0, box_nms_thresh 0.7, crop_n_layers 0,
  crop_nms_thresh 0.7, crop_overlap_ratio 512/1500, crop_n_points_downscale_factor 1,
  min_mask_region_area 0 [V facebookresearch/segment-anything@dca509fe
  automatic_mask_generator.py:39-50]. It runs fp32 (sam.to(device), no autocast [V
  sam_inference.py:211-215]); so does this producer by default, with torch's own precision
  defaults left untouched and recorded. Images: cv2.imread then BGR->RGB [V :278-282].
- Full segmentation [V sam_inference.py:321-358]: masks covering the whole frame are dropped,
  the rest sorted by area (np.argsort(area)[::-1], kept here verbatim: its tie order is numpy's),
  a background column of area H*W prepended, each pixel takes the first argmin of (area if
  covered else H*W+1), labels renumbered 0.. in sorted order, labels > 255 set to 0, uint8 PNG.
  full_seg() computes the same first-argmin by painting in that order with a strict '<' (no
  H x W x N float64 array); tests/test_sam_masks.py checks it against a literal transcription.
- Key objects [V sam_inference.py:389-418], in SAM's output order: bbox w in [50, 300] and h in
  [50, 200], area / (w*h) >= 0.5, and at least 6 masks (itself included) sharing a pixel with
  it; each kept mask is a 0/255 uint8 channel. A frame with none is an H x W x 0 array, which
  the loader's Zoom handles [V core/utils/transforms/sep_transforms.py:36-37].
- Compressed key objects: raw uint8 .npy are 13-53 GB [E pack_derived PACK_WARN], hours through
  `colab download` at 5.28 MB/s [M coord/g4_day1.json, drill D6]; np.savez_compressed holds the
  identical array, is byte-deterministic (zip dates 1980-01-01 [M numpy 2.5.3]) and the loader
  reads it already. The plan's np.packbits would need a loader change (codebase).
- Resumable per frame: each file is written to a dot-named temp file and renamed; the full-seg
  PNG is written last, so a frame counts as done only when every output exists. pack skips
  dot-files, so a temp file left by a killed worker never ships.
- Resume never mixes recipes (Tier 1c review): a done frame is skipped on its file's presence
  alone, so before any worker starts the output dir must carry `.sam_settings.json` (a dot-file:
  never packed) whose identity equals this run's: the method settings (every AMG value except
  points_per_batch, model, precision, generator, key format), the segment_anything archive
  sha256, the checkpoint sha256 and this file's sha256. A fresh dir gets the stamp; a different
  stamp, or outputs with no stamp, is exit 3. A non-faithful run never writes into
  DATA/derived/sam/v1, U2_SAM_DIR or samrle/v1 (exit 3, with or without --pack): a bf16 or
  threshold bench goes to its own --sam-dir. The stamp is copied into out/sam_manifest.json.
- No change to the image: segment_anything goes into U2_DATA/pydeps/ from a hash-checked archive
  (env/vm-requirements-sam.txt), never pip into site-packages, so lock_G0's pip freeze and
  lock_hash do not move (codebase section 7) and nothing else on the VM changes.
- Workers share one G4 (the plan's bench: 1 vs 4-6 processes; post-processing is CPU-bound
  [prereg/inputs/colab-plan-G4-full.json final W1-06]); each sets a VRAM fraction of
  U2_MEM_FRAC / procs. They stay in the job's process group, so job.py's stop reaches them.
- The masks live under U2_DATA/derived/, not in the run's out/: harvest.sh would otherwise pull
  ~2 GB through `colab download`; collect_derived.sh brings the packed shards (colab-fleet 11).
  out/ keeps the manifest (per-subset counts, sha256 of every file), progress and the pack log.
- Python >= 3.9 standard library at import (plan works on the Mac without numpy or torch).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KIND, VERSION, RLE_KIND = "sam", 1, "samrle"
EVAL_KIND = "evalsam"  # --eval-frames: frame 10, a private eval* kind (pack_derived EVAL_PREFIX)
EVAL_N = {"kitti2015": 200, "kitti2012": 194}  # training ids [V core/flow_datasets.py:200, 278]
DATASETS = (("kitti2015", "kitti15"), ("kitti2012", "kitti12"))  # loader name, U2_DATA dir
SUBS = ("testing", "training")  # the loaders' loop order [V core/flow_datasets.py:229, 307]
SIDES = ("image_2", "image_3")
REQS = REPO / "env" / "vm-requirements-sam.txt"
WEIGHTS_JSON = REPO / "data" / "manifest" / "weights.json"
WEIGHT_KEY = "sam-vit-h"
NOTICE = REPO / "tools" / "vm" / "notices" / "UnSAMFlow-CC-BY-NC.txt"
PACK = REPO / "tools" / "data" / "pack_derived.py"
MODEL_TYPE = "vit_h"
# SamAutomaticMaskGenerator's defaults [V segment-anything@dca509fe automatic_mask_generator.py
# :39-50]; UnSAMFlow passes none of them [V UnSAMFlow@0363aed sam_inference.py:190-215].
AMG_DEFAULTS = {"points_per_side": 32, "points_per_batch": 64, "pred_iou_thresh": 0.88,
                "stability_score_thresh": 0.95, "stability_score_offset": 1.0,
                "box_nms_thresh": 0.7, "crop_n_layers": 0, "crop_nms_thresh": 0.7,
                "crop_overlap_ratio": 512 / 1500, "crop_n_points_downscale_factor": 1,
                "min_mask_region_area": 0}
# points_per_batch only sizes the decoder batches; every threshold is per mask and the NMS runs
# after all batches, so it may change for speed without changing the method (recorded).
SPEED_ONLY = ("points_per_batch",)
SOURCES = {"unsamflow": "facebookresearch/UnSAMFlow@0363aed7b258ad7e659d82cbdb1766709f7b9429",
           "segment_anything": "facebookresearch/segment-anything@"
                               "dca509fe793f601edb92606367a655c15ac00fdf"}
EXIT_OK, EXIT_FAIL, EXIT_PRE, EXIT_REFUSED, EXIT_SEAL, EXIT_USAGE = 0, 1, 2, 3, 50, 64
EXIT_TERM = 143
NAME_RX = re.compile(r"^[0-9]{6}_[0-9]{2}\.png$")
PROGRESS_DIR, WORK_DIR = "progress", "work"
STAMP = ".sam_settings.json"  # a dot-file: pack_derived and write_manifest skip it


class Stop(Exception):
    def __init__(self, code: int, msg: str):
        super().__init__(msg)
        self.code, self.msg = code, msg


def utc_now() -> str:
    import datetime as dt

    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()  # noqa: UP017


def say(tag: str, msg: str = "") -> None:
    print(f"{tag} {msg}".rstrip(), flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def atomic_bytes(path: Path, data: bytes) -> None:
    """Write via a dot-named temp file in the same dir, then rename (pack skips dot-files)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def atomic_json(path: Path, doc: dict) -> None:
    atomic_bytes(path, (json.dumps(doc, indent=1, sort_keys=True) + "\n").encode())


# ------------------------------------------------------------------------------- the frames
def sam_dir_default(data: Path) -> Path:
    return Path(os.environ.get("U2_SAM_DIR") or data / "derived" / KIND / f"v{VERSION}")


def pairs_from_names(names: list[str]) -> list[tuple[str, str]]:
    """The loaders' pair rule on one sorted directory listing [V core/flow_datasets.py:233-246]:
    consecutive frame numbers, neither of them 9-12."""
    out = []
    for a, b in zip(names, names[1:], strict=False):
        ia, ib = int(a[-6:-4]), int(b[-6:-4])
        if ia != ib - 1 or 12 >= ia >= 9 or 12 >= ib >= 9:
            continue
        out.append((a, b))
    return out


def frame_rel(sub: str, side: str, name: str) -> str:
    return f"{sub}/{side}/{name}"


def plan_listing(listing: dict[tuple[str, str, str], list[str]]) -> list[dict]:
    """Frames, in the loaders' order, from {(ds, sub, side): sorted names}: each frame once."""
    frames, seen = [], set()
    for ds, _top in DATASETS:
        for sub in SUBS:
            for side in SIDES:
                names = listing.get((ds, sub, side))
                if names is None:
                    continue
                for pair in pairs_from_names(names):
                    for n in pair:
                        key = (ds, sub, side, n)
                        if key not in seen:
                            seen.add(key)
                            frames.append({"ds": ds, "sub": sub, "side": side, "name": n})
    return frames


def list_data(data: Path) -> dict[tuple[str, str, str], list[str]]:
    """The image listings the loaders make (os.listdir, sorted). A missing dir is exit 2: the
    loader would raise FileNotFoundError at construction."""
    out, missing = {}, []
    for ds, top in DATASETS:
        for sub in SUBS:
            for side in SIDES:
                d = data / top / "multi_view" / sub / side
                if not d.is_dir():
                    missing.append(str(d))
                    continue
                names = sorted(os.listdir(d))
                bad = [n for n in names if not NAME_RX.match(n)]
                if bad:
                    raise Stop(EXIT_PRE, f"{d} holds {len(bad)} file(s) the loader cannot parse "
                                         f"(e.g. {bad[0]!r}): int(name[-6:-4]) would fail")
                out[(ds, sub, side)] = names
    if missing:
        raise Stop(EXIT_PRE, f"{len(missing)} image dir(s) missing, e.g. {missing[0]}: fetch "
                             "profile 'sam' first (tools/data/fetch.py --profile sam)")
    return out


def list_manifest(man: Path) -> dict[tuple[str, str, str], list[str]]:
    """The same listings from the Mac's per-bundle file lists (data/manifest/files/)."""
    out: dict[tuple[str, str, str], list[str]] = {}
    tops = {top: ds for ds, top in DATASETS}
    rx = re.compile(r"^(kitti1[25])/multi_view/(training|testing)/(image_[23])/([^/]+)$")
    for f in sorted((man / "files").glob("u2f-*mv-*.sha256")):
        for line in f.read_text().splitlines():
            parts = line.split(None, 1)
            m = rx.match(parts[1].strip()) if len(parts) == 2 else None
            if m and m.group(4).endswith(".png"):
                out.setdefault((tops[m.group(1)], m.group(2), m.group(3)), []).append(m.group(4))
    return {k: sorted(set(v)) for k, v in out.items()}


def eval_frames(names: dict[str, set[str]] | None = None, data: Path | None = None) -> list[dict]:
    """--eval-frames: frame 10 of every training image_2, KITTI-15 then KITTI-12. NAMES is
    {ds: names present} (the Mac's manifest); otherwise DATA is checked. A missing image: exit 2."""
    frames, missing = [], []
    for ds, top in DATASETS:
        for i in range(EVAL_N[ds]):
            fr = {"ds": ds, "sub": "training", "side": "image_2", "name": f"{i:06d}_10.png"}
            ok = (fr["name"] in names.get(ds, set()) if names is not None else
                  (data / top / "multi_view" / "training" / "image_2" / fr["name"]).is_file())
            if not ok:
                missing.append(f"{top}/{fr['name']}")
            frames.append(fr)
    if missing:
        raise Stop(EXIT_PRE, f"{len(missing)} frame-10 image(s) missing, e.g. {missing[0]}: fetch "
                             "profile 'evalsam' first (tools/data/fetch.py --profile evalsam)")
    return frames


def frame_paths(fr: dict, data: Path, sam_dir: Path, key_format: str = "npz",
                rle_dir: Path | None = None) -> dict:
    top = dict(DATASETS)[fr["ds"]]
    rel = frame_rel(fr["sub"], fr["side"], fr["name"])
    stem = rel[:-4]
    p = {"image": data / top / "multi_view" / rel,
         "fullseg": sam_dir / f"{fr['ds']}_mv_sam1_fullseg" / rel,
         "key": sam_dir / f"{fr['ds']}_mv_sam1_key_objects" / f"{stem}.{key_format}",
         "key_npy": sam_dir / f"{fr['ds']}_mv_sam1_key_objects" / f"{stem}.npy"}
    if rle_dir is not None:
        p["rle"] = rle_dir / f"{fr['ds']}_mv_sam1_rle" / f"{stem}.json.gz"
    return p


def frame_done(p: dict) -> bool:
    return all(p[k].is_file() for k in ("fullseg", "key", "rle") if k in p)


def subset(fr: dict) -> str:
    return f"{fr['ds']}/{fr['sub']}/{fr['side']}"


def counts(frames: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for fr in frames:
        out[subset(fr)] = out.get(subset(fr), 0) + 1
    return dict(sorted(out.items()))


def n_pairs(listing: dict) -> int:
    return sum(len(pairs_from_names(v)) for v in listing.values())


def loader_check(data: Path, sam_dir: Path, frames: list[dict]) -> str:
    """Construct the real KITTI2015_MV_Sem and KITTI2012_MV_Sem with U2_DATA and U2_SAM_DIR set
    to ours and compare every path they would read with what this producer writes. Returns 'ok',
    'skipped: ...' (core not importable here) or raises Stop(2) on any difference."""
    env = {"U2_DATA": str(data), "U2_SAM_DIR": str(sam_dir)}
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        sys.path.insert(0, str(REPO))
        try:
            from core import flow_datasets as fd
        except Exception as e:  # noqa: BLE001 (no torch/torchvision/cv2 here: recorded)
            return f"skipped: import core.flow_datasets: {type(e).__name__}: {e}"
        want_img, want_seg, want_key = set(), set(), set()
        for cls in (fd.KITTI2015_MV_Sem, fd.KITTI2012_MV_Sem):
            ds = cls(None, None, None)
            for imgs, sems, segs in zip(ds.image_list, ds.sem_list, ds.full_seg_list,
                                        strict=True):
                want_img.update(imgs)
                want_key.update(sems)
                want_seg.update(segs)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    have_img, have_seg, have_key = set(), set(), set()
    for fr in frames:
        p = frame_paths(fr, data, sam_dir)
        have_img.add(str(p["image"]))
        have_seg.add(str(p["fullseg"]))
        have_key.add(str(p["key_npy"]))
    for what, a, b in (("images", have_img, want_img), ("fullseg", have_seg, want_seg),
                       ("key_objects", have_key, want_key)):
        if a != b:
            ex = sorted(a ^ b)[:2]
            raise Stop(EXIT_PRE, f"the frame list differs from the loaders' {what}: "
                                 f"{len(a - b)} extra, {len(b - a)} missing, e.g. {ex}")
    return "ok"


# ------------------------------------------------------------------- UnSAMFlow's two outputs
def full_seg(masks, areas, shape: tuple[int, int]):
    """UnSAMFlow main_mask_to_full_seg [V UnSAMFlow@0363aed sam_inference.py:321-358] on SAM's
    masks (N x H x W bool, SAM order) and their 'area' values."""
    import numpy as np

    h, w = shape
    areas = np.array(list(areas))
    if len(areas) == 0:
        return np.zeros((h, w), dtype=np.uint8)
    keep = areas < h * w  # drop a mask that equals the full frame [V :329-331]
    m, a = masks[keep], areas[keep]
    order = np.argsort(a)[::-1]  # largest first; numpy's tie order, as UnSAMFlow [V :333-336]
    m, a = m[order], a[order]
    # argmin over [background (area H*W, everywhere)] + masks of (area if covered else H*W+1),
    # first index on ties [V :338-347]: painting in that order with a strict '<'
    label = np.zeros((h, w), dtype=np.int64)
    cur = np.full((h, w), h * w, dtype=np.int64)
    for j in range(len(a)):
        upd = m[j] & (a[j] < cur)
        label[upd] = j + 1
        cur[upd] = a[j]
    uniq = np.unique(label)  # renumber 0.. in sorted order [V :349-353]
    mapping = np.zeros(uniq.max() + 1)
    mapping[uniq] = np.arange(len(uniq))
    new = mapping[label]
    new[new > 255] = 0  # 'almost not existent' [V :355-357]
    return new.astype(np.uint8)


def key_objects(masks, recs: list[dict], shape: tuple[int, int]):
    """UnSAMFlow main_mask_to_key_objects [V UnSAMFlow@0363aed sam_inference.py:389-418]."""
    import numpy as np

    h, w = shape
    n = len(recs)
    flat = masks.reshape(n, -1) if n else np.zeros((0, h * w), dtype=bool)
    out = []
    for i, r in enumerate(recs):
        bw, bh = r["bbox"][2:4]
        if not (50 <= bh <= 200 and 50 <= bw <= 300):
            continue
        if r["area"] / (bh * bw) < 0.5:
            continue
        idx = np.flatnonzero(flat[i])
        if int(flat[:, idx].any(axis=1).sum()) >= 6:  # masks sharing a pixel, itself included
            out.append(masks[i].astype(np.uint8) * 255)
    if not out:
        return np.zeros((h, w, 0), dtype=np.uint8)
    return np.stack(out, axis=-1)


def rle_encode(mask) -> dict:
    """Uncompressed column-major RLE, SAM's mask_to_rle_pytorch format (counts start with zeros)."""
    import numpy as np

    h, w = mask.shape
    v = mask.T.reshape(-1).astype(np.int8)
    change = np.flatnonzero(np.diff(v)) + 1
    edges = np.concatenate([[0], change, [v.size]])
    counts = np.diff(edges).tolist()
    if v.size and v[0] == 1:
        counts = [0] + counts
    return {"size": [h, w], "counts": counts}


def rle_decode(rle: dict):
    """SAM's rle_to_mask [V segment-anything@dca509fe utils/amg.py:138-149]."""
    import numpy as np

    h, w = rle["size"]
    mask = np.empty(h * w, dtype=bool)
    idx, parity = 0, False
    for c in rle["counts"]:
        mask[idx: idx + c] = parity
        idx += c
        parity ^= True
    return mask.reshape(w, h).transpose()


# ------------------------------------------------------------------------------ generators
class StubGenerator:
    """Test-only stand-in for SamAutomaticMaskGenerator(output_mode='uncompressed_rle'): a fixed,
    image-independent set of rectangles that exercises every rule (a full-frame mask, equal
    areas, a key object with five children). Refused on a host with nvidia-smi."""

    name = "stub"

    def generate(self, image) -> list[dict]:
        import numpy as np

        h, w = image.shape[:2]
        boxes = [(0, 0, w, h), (0, 0, w // 2, h)]
        if h >= 70 and w >= 80:
            boxes += [(4, 4, 60, 60)] + [(6 + 10 * k, 8, 9, 20) for k in range(5)]
        boxes += [(w - 20, h - 12, 10, 6), (w - 20, h - 12, 6, 10)]  # equal areas, overlapping
        recs = []
        for x, y, bw, bh in boxes:
            m = np.zeros((h, w), dtype=bool)
            m[y: y + bh, x: x + bw] = True
            recs.append({"segmentation": rle_encode(m), "area": int(m.sum()),
                         "bbox": [x, y, bw, bh], "predicted_iou": 0.9,
                         "point_coords": [[x + 0.5, y + 0.5]], "stability_score": 0.96,
                         "crop_box": [0, 0, w, h]})
        if int(image[0, 0, 0]) == 1:  # a frame with no mask at all (tests mark it so)
            return []
        return recs


class SamGenerator:
    """segment_anything's SamAutomaticMaskGenerator in 'uncompressed_rle' mode, fp32 by default."""

    name = "segment_anything"

    def __init__(self, checkpoint: Path | None, model_type: str, device: str, precision: str,
                 amg: dict, model=None):
        import torch
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

        if model is None:
            model = sam_model_registry[model_type](
                checkpoint=str(checkpoint) if checkpoint else None)
        model.to(device=device)
        self.device, self.precision = device, precision
        self.gen = SamAutomaticMaskGenerator(model, output_mode="uncompressed_rle", **amg)
        self.torch = torch

    def generate(self, image) -> list[dict]:
        if self.precision == "fp32":
            return self.gen.generate(image)
        dt = {"bf16": self.torch.bfloat16, "fp16": self.torch.float16}[self.precision]
        with self.torch.autocast(self.device.split(":")[0], dtype=dt):
            return self.gen.generate(image)


def read_rgb(path: Path):
    """cv2.imread then BGR->RGB, as UnSAMFlow [V sam_inference.py:278-282]; imageio if no cv2."""
    try:
        import cv2
    except ImportError:
        import imageio.v2 as imageio

        im = imageio.imread(str(path))
        return im[:, :, :3] if im.ndim == 3 else im[:, :, None].repeat(3, axis=2)
    im = cv2.imread(str(path))
    if im is None:
        raise OSError(f"cv2 could not read {path}")
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def encode_png(arr) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def encode_key(arr, fmt: str) -> bytes:
    import numpy as np

    buf = io.BytesIO()
    if fmt == "npz":
        np.savez_compressed(buf, key_objects=arr)
    else:
        np.save(buf, arr)
    return buf.getvalue()


def rle_doc_bytes(recs: list[dict]) -> bytes:
    """The --rle file of one frame: SAM's records as sorted compact JSON, gzip with mtime 0."""
    doc = json.dumps(recs, sort_keys=True, separators=(",", ":")).encode()
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0, compresslevel=6) as gz:
        gz.write(doc)
    return buf.getvalue()


def process_frame(gen, p: dict, key_format: str) -> dict:
    import numpy as np

    img = read_rgb(p["image"])
    h, w = img.shape[:2]
    recs = gen.generate(img)
    masks = (np.stack([rle_decode(r["segmentation"]) for r in recs]) if recs
             else np.zeros((0, h, w), dtype=bool))
    ko = key_objects(masks, recs, (h, w))
    fs = full_seg(masks, [r["area"] for r in recs], (h, w))
    if "rle" in p:
        atomic_bytes(p["rle"], rle_doc_bytes(recs))
    atomic_bytes(p["key"], encode_key(ko, key_format))
    atomic_bytes(p["fullseg"], encode_png(fs))  # last: its presence marks the frame done
    return {"n_masks": len(recs), "n_key": int(ko.shape[2]), "n_labels": int(fs.max()) + 1,
            "h": h, "w": w}


# ---------------------------------------------------------------------- deps and weights
def req_line() -> tuple[str, str, str]:
    """(url, sha256, sha12 of the requirements file) from env/vm-requirements-sam.txt."""
    text = REQS.read_text()
    url = re.search(r"^segment-anything @ (\S+)", text, re.M)
    sha = re.search(r"--hash=sha256:([0-9a-f]{64})", text)
    if not (url and sha):
        raise Stop(EXIT_PRE, f"{REQS} has no 'segment-anything @ URL --hash=sha256:...' line")
    return url.group(1), sha.group(1), hashlib.sha256(text.encode()).hexdigest()[:12]


def ensure_deps(data: Path, tarball: Path | None = None) -> Path:
    """segment_anything from the pinned archive into DATA/pydeps/sam-<sha12>/, first on sys.path."""
    url, want, tag = req_line()
    target = data / "pydeps" / f"sam-{tag}"
    stamp = target / "INSTALL.json"
    rec = {}
    try:
        rec = json.loads(stamp.read_text())
    except (OSError, ValueError):
        pass
    if rec.get("sha256") != want or not (target / "segment_anything" / "__init__.py").is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        arc = target.parent / f"segment-anything-{want[:12]}.tar.gz"
        if tarball is not None:
            if not tarball.is_file():
                raise Stop(EXIT_PRE, f"--sam-tarball {tarball}: no such file")
            shutil.copyfile(tarball, arc)
        elif not (arc.is_file() and sha256_file(arc) == want):
            part = arc.with_name(arc.name + ".part")
            for attempt in range(1, 4):
                try:
                    with urllib.request.urlopen(url, timeout=120) as r, open(part, "wb") as f:
                        shutil.copyfileobj(r, f, 1 << 20)
                    os.replace(part, arc)
                    break
                except OSError as e:
                    say("SAM_DEPS", f"download attempt {attempt} failed: {e}")
                    time.sleep(5 * attempt)
            else:
                raise Stop(EXIT_PRE, f"could not download {url}")
        got = sha256_file(arc)
        if got != want:
            arc.unlink(missing_ok=True)
            raise Stop(EXIT_PRE, f"segment-anything archive sha256 {got} != {want} pinned in "
                                 f"{REQS.name}: nothing installed")
        stage = target.with_name(target.name + ".tmp")
        shutil.rmtree(stage, ignore_errors=True)
        stage.mkdir(parents=True)
        with tarfile.open(arc) as t:
            n = 0
            for m in t.getmembers():
                parts = m.name.split("/")
                keep = (len(parts) >= 3 and parts[1] == "segment_anything"
                        and m.name.endswith(".py")) or (len(parts) == 2 and parts[1] == "LICENSE")
                if not keep or not m.isfile() or ".." in parts:
                    continue
                dest = stage.joinpath(*parts[1:])
                dest.parent.mkdir(parents=True, exist_ok=True)
                src = t.extractfile(m)
                assert src is not None
                dest.write_bytes(src.read())
                n += 1
        (stage / "INSTALL.json").write_text(json.dumps(
            {"sha256": want, "url": url, "files": n, "requirements": str(REQS.name),
             "utc": utc_now()}, indent=1, sort_keys=True) + "\n")
        shutil.rmtree(target, ignore_errors=True)
        os.replace(stage, target)
    if str(target) not in sys.path:
        sys.path.insert(0, str(target))
    for k in [k for k in sys.modules if k == "segment_anything" or
              k.startswith("segment_anything.")]:
        del sys.modules[k]  # a preinstalled copy never shadows the pinned one
    say("SAM_DEPS", f"segment_anything {want[:12]} at {target}")
    return target


def weight_spec() -> dict:
    try:
        return json.loads(WEIGHTS_JSON.read_text())["weights"][WEIGHT_KEY]
    except (OSError, ValueError, KeyError) as e:
        raise Stop(EXIT_PRE, f"{WEIGHTS_JSON}: no '{WEIGHT_KEY}' entry ({e})") from None


def check_weight(path: Path, spec: dict) -> dict:
    """sha256 of the checkpoint against the manifest pin (or its sourced-T value): exit 3 if it
    differs; an unpinned checkpoint is recorded and printed as PIN for weights.json."""
    if not path.is_file():
        raise Stop(EXIT_PRE, f"no checkpoint at {path}: fetch profile 'sam' (weight {WEIGHT_KEY})")
    got = sha256_file(path)
    pin = spec.get("sha256")
    ref = pin or spec.get("sha256_sourced_T")
    if ref and got != ref:
        raise Stop(EXIT_REFUSED, f"{path.name} sha256 {got} != {ref} "
                                 f"({'pinned' if pin else 'sourced-T'} in weights.json)")
    if not pin:
        say("PIN", f"name={WEIGHT_KEY} sha256={got} bytes={path.stat().st_size} "
                   "note=\"record it in data/manifest/weights.json\"")
    return {"file": str(path), "sha256": got, "bytes": path.stat().st_size,
            "matches": "pin" if pin else ("sourced-T" if ref else "unpinned")}


# ------------------------------------------------------------------------------- settings
def settings(a: argparse.Namespace) -> dict:
    amg = dict(AMG_DEFAULTS)
    for kv in a.amg or []:
        k, sep, v = kv.partition("=")
        if k not in AMG_DEFAULTS or not sep:
            raise Stop(EXIT_USAGE, f"--amg {kv}: not KEY=VALUE of a SamAutomaticMaskGenerator "
                                   "setting")
        amg[k] = int(v) if isinstance(AMG_DEFAULTS[k], int) else float(v)
    deviations = [f"amg.{k}={amg[k]}" for k in AMG_DEFAULTS
                  if amg[k] != AMG_DEFAULTS[k] and k not in SPEED_ONLY]
    if a.model_type != MODEL_TYPE:
        deviations.append(f"model_type={a.model_type}")
    if a.precision != "fp32":
        deviations.append(f"precision={a.precision}")
    if a.generator != "sam":
        deviations.append(f"generator={a.generator}")
    if a.checkpoint:
        deviations.append(f"checkpoint={a.checkpoint}")
    return {"amg": amg, "model_type": a.model_type, "precision": a.precision,
            "generator": a.generator, "key_format": a.key_format, "rle": bool(a.rle),
            "output_mode": "uncompressed_rle", "deviations": deviations,
            "faithful": not deviations, "sources": SOURCES,
            "kind": EVAL_KIND if getattr(a, "eval_frames", False) else KIND, "version": VERSION}


def pack_dir(data: Path, kind: str = KIND) -> Path:
    """Where pack_derived.py requires the masks of KIND v1: DATA/derived/<kind>/v1."""
    return data / "derived" / kind / f"v{VERSION}"


def v1_dirs(data: Path) -> set[Path]:
    """The directories v1 is read or packed from: no non-faithful output may land there."""
    return {Path(os.path.realpath(p)) for p in (pack_dir(data), sam_dir_default(data),
                                                data / "derived" / RLE_KIND / f"v{VERSION}")}


def stamp_identity(cfg: dict, seg_sha: str | None, ckpt_sha: str | None) -> dict:
    """What decides the bytes of a frame's outputs; two runs may share an output dir only when
    these are equal. points_per_batch (SPEED_ONLY), rle and the run's bookkeeping are left out."""
    method = {k: cfg[k] for k in ("model_type", "precision", "generator", "key_format",
                                  "output_mode", "sources")}
    method["amg"] = {k: v for k, v in cfg["amg"].items() if k not in SPEED_ONLY}
    return {"schema": 1, "kind": cfg["kind"], "version": VERSION, "method": method,
            "segment_anything_sha256": seg_sha, "checkpoint_sha256": ckpt_sha,
            "producer_sha256": sha256_file(Path(__file__).resolve())}


def has_outputs(root: Path) -> str | None:
    """The first non-dot file under ROOT (an output some run wrote), else None."""
    if not root.is_dir():
        return None
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in files:
            if not f.startswith("."):
                return str(Path(dirpath) / f)
    return None


def ensure_stamp(root: Path, ident: dict, cfg: dict, run: str | None) -> dict:
    """ROOT/.sam_settings.json: written on the first use of ROOT, compared on every later one.
    Exit 3 when it differs from IDENT, or when ROOT holds outputs but no stamp (unknown recipe)."""
    p = root / STAMP
    try:
        have = json.loads(p.read_text())
    except FileNotFoundError:
        have = None
    except (OSError, ValueError) as e:
        raise Stop(EXIT_REFUSED, f"{p} is unreadable ({e}): the recipe of the frames there is "
                                 "unknown; remove the directory or use another --sam-dir") from None
    if have is None:
        stray = has_outputs(root)
        if stray:
            raise Stop(EXIT_REFUSED, f"{root} holds outputs (e.g. {stray}) but no {STAMP}: their "
                                     "settings are unknown and would be reused as done frames; "
                                     "remove the directory or use another --sam-dir")
        doc = {"identity": ident, "faithful": cfg["faithful"], "deviations": cfg["deviations"],
               "first_written": {"utc": utc_now(), "run": run,
                                 "code_sha": os.environ.get("U2_GIT_SHA"),
                                 "session": os.environ.get("U2_SESSION")}}
        root.mkdir(parents=True, exist_ok=True)
        atomic_json(p, doc)
        say("SAM_STAMP", f"new {p}")
        return doc
    if not isinstance(have, dict) or have.get("identity") != ident:
        old = have.get("identity") if isinstance(have, dict) else None
        old = old if isinstance(old, dict) else {}
        diff = sorted(k for k in set(ident) | set(old) if old.get(k) != ident.get(k))
        first = have.get("first_written") if isinstance(have, dict) else None
        raise Stop(EXIT_REFUSED, f"{p} records other settings ({', '.join(diff) or 'format'} "
                                 f"differ; first written {first}): "
                                 "its done frames are not this recipe's; never mix them. Use "
                                 "the same settings and commit, another --sam-dir, or remove it")
    say("SAM_STAMP", f"same settings as {p}")
    return have


def numerics_record(device: str) -> dict:
    """What decides the numbers, recorded, never changed: torch's own defaults stay as they are
    (UnSAMFlow set none); cudnn is made deterministic (no effect on the result's definition)."""
    rec: dict = {"device": device, "python": sys.version.split()[0]}
    try:
        import numpy as np

        rec["numpy"] = np.__version__
    except ImportError:
        pass
    try:
        import cv2

        rec["cv2"] = cv2.__version__
    except ImportError:
        rec["cv2"] = None
    try:
        import PIL

        rec["pillow"] = PIL.__version__
    except ImportError:
        pass
    try:
        import torch
    except ImportError:
        return rec
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    rec.update(torch=torch.__version__, cuda=torch.version.cuda,
               cudnn=torch.backends.cudnn.version() if torch.backends.cudnn.is_available()
               else None, cudnn_deterministic=True, cudnn_benchmark=False)
    for k, f in (("matmul_fp32_precision", lambda: torch.backends.cuda.matmul.fp32_precision),
                 ("conv_fp32_precision", lambda: torch.backends.cudnn.conv.fp32_precision),
                 ("float32_matmul_precision", torch.get_float32_matmul_precision)):
        try:
            rec[k] = f()
        except Exception:  # noqa: BLE001 (an older torch: the key is absent)
            rec[k] = None
    if device.startswith("cuda") and torch.cuda.is_available():
        rec["gpu_name"] = torch.cuda.get_device_name(0)
        rec["cc"] = list(torch.cuda.get_device_capability(0))
    return rec


def pick_device(want: str) -> str:
    if want != "auto":
        return want
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def gpu_host() -> bool:
    return shutil.which("nvidia-smi") is not None


# ------------------------------------------------------------------------------- worker
_STOP = {"flag": False}


def _on_term(signum, frame) -> None:  # noqa: ARG001
    _STOP["flag"] = True


def cmd_worker(a: argparse.Namespace) -> int:
    signal.signal(signal.SIGTERM, _on_term)
    data, sam_dir = Path(a.data), Path(a.sam_dir)
    rle_dir = Path(a.rle_dir) if a.rle_dir else None
    job = json.loads(Path(a.frames_file).read_text())
    cfg = job["settings"]
    if cfg["generator"] == "stub":
        if gpu_host():
            raise Stop(EXIT_REFUSED, "the stub generator never runs on a GPU host")
        gen = StubGenerator()
    else:
        if a.pydeps:
            sys.path.insert(0, a.pydeps)
        import torch

        if os.environ.get("U2_TORCH_THREADS"):  # the spec's env; recorded in sam_config.json
            torch.set_num_threads(int(os.environ["U2_TORCH_THREADS"]))
        frac = float(os.environ.get("U2_MEM_FRAC") or 0) / max(1, int(job["procs"]))
        if frac and job["device"].startswith("cuda") and torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(frac)
        numerics_record(job["device"])  # sets cudnn deterministic / benchmark in this process
        gen = SamGenerator(Path(job["checkpoint"]) if job.get("checkpoint") else None,
                           cfg["model_type"], job["device"], cfg["precision"], cfg["amg"])
    prog = Path(a.progress)
    prog.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    for fr in job["frames"]:
        if _STOP["flag"]:
            break
        p = frame_paths(fr, data, sam_dir, cfg["key_format"], rle_dir)
        if frame_done(p):
            continue
        t0 = time.time()
        info = process_frame(gen, p, cfg["key_format"])
        line = dict(info, frame=f"{subset(fr)}/{fr['name']}", secs=round(time.time() - t0, 3),
                    utc=utc_now())
        with open(prog, "a") as f:
            f.write(json.dumps(line, sort_keys=True) + "\n")
        n += 1
    say("SAM_WORKER_DONE", f"k={a.k} frames={n} stopped={_STOP['flag']}")
    return EXIT_TERM if _STOP["flag"] else EXIT_OK


# ------------------------------------------------------------------------------- run
def _bus(step: int, **fields) -> None:
    if not os.environ.get("U2_BUS"):
        return
    try:
        sys.path.insert(0, str(REPO / "tools" / "vm"))
        import bus

        bus.log(step, **fields)
    except Exception as e:  # noqa: BLE001 (telemetry never stops the job)
        say("SAM_BUS_ERROR", f"{type(e).__name__}: {e}")


def _resolved(cfg: dict) -> None:
    if not os.environ.get("U2_BUS"):
        return
    sys.path.insert(0, str(REPO / "tools" / "vm"))
    import bus

    bus.resolved_config(cfg)


def _count_lines(paths: list[Path]) -> int:
    n = 0
    for p in paths:
        try:
            with open(p, "rb") as f:
                n += sum(1 for _ in f)
        except OSError:
            pass
    return n


def cmd_run(a: argparse.Namespace) -> int:
    # Absolute paths: the workers start with cwd=REPO (a relative --data broke them).
    data = Path(a.data).resolve()
    out = Path(a.out_dir).resolve()
    cfg = settings(a)
    ev = cfg["kind"] == EVAL_KIND
    sam_dir = (Path(a.sam_dir).resolve() if a.sam_dir else
               (pack_dir(data, EVAL_KIND) if ev else sam_dir_default(data)).resolve())
    rle_dir = (sam_dir if ev else data / "derived" / RLE_KIND / f"v{VERSION}") if a.rle else None
    stub_test = cfg["generator"] == "stub" and os.environ.get("U2_SAM_STUB") == "1"
    if cfg["generator"] == "stub":
        if gpu_host():
            raise Stop(EXIT_REFUSED, "the stub generator never runs on a GPU host")
        if not stub_test:
            raise Stop(EXIT_REFUSED, "--generator stub is for tests (U2_SAM_STUB=1)")
    if a.pack and not cfg["faithful"] and not stub_test:
        raise Stop(EXIT_REFUSED, f"--pack of a non-faithful run ({', '.join(cfg['deviations'])})"
                                 f": {KIND} v{VERSION} holds only the UnSAMFlow recipe; write a "
                                 "deviation to another --sam-dir and never pack it as v1")
    if not cfg["faithful"] and not stub_test and (
            Path(os.path.realpath(sam_dir)) in v1_dirs(data) | {Path(os.path.realpath(
                pack_dir(data, EVAL_KIND)))} or rle_dir is not None):
        raise Stop(EXIT_REFUSED, f"a non-faithful run ({', '.join(cfg['deviations'])}) never "
                                 f"writes into the v1 directories ({sam_dir}"
                                 f"{', samrle/v1' if rle_dir else ''}): a later faithful run "
                                 "would reuse its frames; give it its own --sam-dir, no --rle")
    kdir = pack_dir(data, cfg["kind"])
    if a.pack and Path(os.path.realpath(sam_dir)) != Path(os.path.realpath(kdir)):
        raise Stop(EXIT_REFUSED, f"--pack packs only {kdir} (pack_derived's src for "
                                 f"{cfg['kind']} v{VERSION}), not --sam-dir {sam_dir}")
    out.mkdir(parents=True, exist_ok=True)
    if ev:
        frames = eval_frames(data=data)
        say("SAM_PLAN", f"frames={len(frames)} eval-frames "
                        f"subsets={json.dumps(counts(frames), separators=(',', ':'))}")
        chk = "n/a: eval frames (no MV_Sem loader reads frame 10)"
    else:
        listing = list_data(data)
        frames = plan_listing(listing)
        say("SAM_PLAN", f"frames={len(frames)} pairs={n_pairs(listing)} "
                        f"subsets={json.dumps(counts(frames), separators=(',', ':'))}")
        chk = loader_check(data, sam_dir, frames)
    say("SAM_LOADER_CHECK", chk)
    if a.pack and cfg["generator"] == "sam" and chk != "ok" and not ev:
        raise Stop(EXIT_PRE, f"--pack needs the loader check against the real classes ({chk}): "
                             "a v1 that no loader was compared with is never packed")
    if a.limit:
        frames = frames[: a.limit]
    device = pick_device(a.device)
    rec = {"settings": cfg, "frames_planned": len(frames), "loader_check": chk,
           "sam_dir": str(sam_dir), "rle_dir": str(rle_dir) if rle_dir else None,
           "data": str(data), "procs": a.procs, "limit": a.limit, "run": a.run,
           "session": os.environ.get("U2_SESSION"), "code_sha": os.environ.get("U2_GIT_SHA"),
           "threads": {"U2_TORCH_THREADS": os.environ.get("U2_TORCH_THREADS"),
                       "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS")},
           "utc": utc_now()}
    pydeps, ckpt = None, None
    if cfg["generator"] == "sam":
        pydeps = str(ensure_deps(data, Path(a.sam_tarball) if a.sam_tarball else None))
        spec = weight_spec()
        ckpt = Path(a.checkpoint).resolve() if a.checkpoint else data / spec["file"]
        rec["weight"] = check_weight(ckpt, spec) if not a.checkpoint else {
            "file": str(ckpt), "sha256": sha256_file(ckpt), "matches": "override"}
        say("SAM_WEIGHT", json.dumps(rec["weight"], sort_keys=True))
        rec["segment_anything"] = json.loads((Path(pydeps) / "INSTALL.json").read_text())
    ident = stamp_identity(cfg, (rec.get("segment_anything") or {}).get("sha256"),
                           (rec.get("weight") or {}).get("sha256"))
    rec["stamp"] = ensure_stamp(sam_dir, ident, cfg, a.run)
    if rle_dir is not None:
        ensure_stamp(rle_dir, ident, cfg, a.run)
    rec["numerics"] = numerics_record(device) if cfg["generator"] == "sam" else {"device": device}
    atomic_json(out / "sam_config.json", rec)
    _resolved(rec)
    todo = [fr for fr in frames if not frame_done(frame_paths(fr, data, sam_dir,
                                                              cfg["key_format"], rle_dir))]
    done0 = len(frames) - len(todo)
    say("SAM_START", f"todo={len(todo)} done={done0} procs={a.procs} device={device} "
                     f"faithful={cfg['faithful']}")
    gen_id = str(int(time.time()))
    procs = max(1, min(a.procs, len(todo))) if todo else 0
    work, prog_files, kids = out / WORK_DIR, [], []
    stopped = {"sig": 0}

    def on_term(signum, frame) -> None:  # noqa: ARG001
        stopped["sig"] = signum
        for _k, pr in kids:
            if pr.poll() is None:
                pr.send_signal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGHUP, on_term)
    t0 = time.time()
    for k in range(procs):
        mine = todo[k::procs]
        ff = work / f"{gen_id}-w{k}.json"
        atomic_json(ff, {"frames": mine, "settings": cfg, "device": device, "procs": procs,
                         "checkpoint": str(ckpt) if ckpt else None})
        pf = out / PROGRESS_DIR / f"{gen_id}-w{k}.jsonl"
        prog_files.append(pf)
        argv = [sys.executable, "-u", str(Path(__file__).resolve()), "worker", "--data",
                str(data), "--sam-dir", str(sam_dir), "--frames-file", str(ff), "--progress",
                str(pf), "--k", str(k)]
        if rle_dir:
            argv += ["--rle-dir", str(rle_dir)]
        if pydeps:
            argv += ["--pydeps", pydeps]
        log = open(out / f"worker-{gen_id}-{k}.log", "ab")  # the child keeps it open
        kids.append((k, subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log,
                                         stderr=subprocess.STDOUT, cwd=str(REPO))))
        if k + 1 < procs and a.stagger_s > 0:
            end = time.time() + a.stagger_s
            while time.time() < end and not stopped["sig"]:
                time.sleep(0.2)
    last_bus, rcs = 0.0, {}
    while kids and any(pr.poll() is None for _k, pr in kids):
        time.sleep(min(2.0, a.bus_every_s))
        if time.time() - last_bus >= a.bus_every_s:
            last_bus = time.time()
            n = _count_lines(prog_files)
            spf = (time.time() - t0) / n if n else None
            fields = {"x_frames_total": len(frames), "x_procs": procs,
                      "x_workers_alive": sum(pr.poll() is None for _k, pr in kids)}
            if spf:
                fields["s_per_step"] = round(spf, 4)
            _bus(done0 + n, **fields)
            say("SAM_PROGRESS", f"done={done0 + n}/{len(frames)} s_per_frame="
                                f"{spf and round(spf, 3)}")
    for k, pr in kids:
        rcs[k] = pr.wait()
        say("SAM_WORKER_EXIT", f"k={k} rc={rcs[k]}")
    n = _count_lines(prog_files)
    _bus(done0 + n, x_frames_total=len(frames), x_procs=procs, x_workers_alive=0)
    if stopped["sig"]:
        say("SAM_STOPPED", f"signal {stopped['sig']}; {done0 + n}/{len(frames)} frames done")
        return EXIT_TERM
    man = write_manifest(data, sam_dir, out, frames, cfg, rle_dir, strict=not a.limit,
                         stamp=rec["stamp"])
    if man["missing"]:
        say("SAM_ERROR", f"{man['missing']} frame(s) not done; worker rcs {rcs}: run again "
                         "(done frames are skipped)")
        return EXIT_FAIL
    if a.limit:
        say("SAM_DONE", f"--limit {a.limit}: {len(frames)} frames, no pack")
        return EXIT_OK
    if a.pack:
        rc = pack(a, data, sam_dir, out, cfg["kind"])
        if rc == 0 and rle_dir and rle_dir != sam_dir:
            rc = pack(a, data, rle_dir, out, RLE_KIND)
        if rc != 0:
            return rc
    say("SAM_DONE", f"{len(frames)} frames; manifest {man['sha256']}")
    return EXIT_OK


# ------------------------------------------------------------------------------- manifest
def write_manifest(data: Path, sam_dir: Path, out: Path, frames: list[dict], cfg: dict,
                   rle_dir: Path | None = None, strict: bool = True,
                   stamp: dict | None = None) -> dict:
    """OUT/sam_manifest.json (per-subset counts, every file's bytes and sha256) and
    OUT/sam_files.sha256 (sha256sum lines with the payload paths pack writes into FILES.sha256);
    STAMP is the output dir's .sam_settings.json (the recipe of every frame listed).
    A non-dot file under the output tree that no planned frame explains is exit 1: pack would
    ship it."""
    roots = [(sam_dir, f"derived/{cfg['kind']}/v{VERSION}/")]
    if rle_dir is not None and rle_dir != sam_dir:
        roots.append((rle_dir, f"derived/{RLE_KIND}/v{VERSION}/"))
    want: dict[Path, str] = {}
    missing = 0
    for fr in frames:
        p = frame_paths(fr, data, sam_dir, cfg["key_format"], rle_dir)
        if not frame_done(p):
            missing += 1
            continue
        for k in ("key", "fullseg", "rle"):
            if k in p:
                want[p[k]] = subset(fr)
    extra = []
    for root, _pfx in roots:
        if not root.is_dir():
            continue
        for dirpath, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for f in files:
                fp = Path(dirpath) / f
                if not f.startswith(".") and fp not in want:
                    extra.append(str(fp))
    if extra and not missing and strict:
        raise Stop(EXIT_FAIL, f"{len(extra)} file(s) under the output tree belong to no planned "
                              f"frame (e.g. {extra[0]}): pack would ship them; remove them")
    files, per = [], {}
    for fp in sorted(want, key=str):
        root, pfx = next((r, x) for r, x in roots if fp.is_relative_to(r))
        rel = pfx + fp.relative_to(root).as_posix()
        files.append({"path": rel, "bytes": fp.stat().st_size, "sha256": sha256_file(fp)})
        s = per.setdefault(want[fp], {"frames": 0, "files": 0, "bytes": 0})
        s["files"] += 1
        s["bytes"] += files[-1]["bytes"]
    for fr in frames:
        if frame_done(frame_paths(fr, data, sam_dir, cfg["key_format"], rle_dir)):
            per.setdefault(subset(fr), {"frames": 0, "files": 0, "bytes": 0})["frames"] += 1
    listing = "".join(f"{f['sha256']}  {f['path']}\n" for f in files).encode()
    atomic_bytes(out / "sam_files.sha256", listing)
    doc = {"schema": 1, "kind": cfg["kind"], "version": VERSION, "settings": cfg,
           "frames_planned": len(frames), "frames_done": len(frames) - missing,
           "missing": missing, "subsets": dict(sorted(per.items())), "n_files": len(files),
           "bytes": sum(f["bytes"] for f in files),
           "files_sha256": hashlib.sha256(listing).hexdigest(), "files": files,
           "unexplained": extra[:20], "settings_stamp": stamp, "utc": utc_now()}
    atomic_json(out / "sam_manifest.json", doc)
    sha = sha256_file(out / "sam_manifest.json")
    say("SAM_MANIFEST", f"sha256={sha} files={len(files)} bytes={doc['bytes']} "
                        f"done={doc['frames_done']}/{len(frames)}")
    return {"sha256": sha, "missing": missing, "files": len(files)}


def cmd_manifest(a: argparse.Namespace) -> int:
    data = Path(a.data).resolve()
    cfg = settings(a)
    ev = cfg["kind"] == EVAL_KIND
    sam_dir = (Path(a.sam_dir).resolve() if a.sam_dir else
               (pack_dir(data, EVAL_KIND) if ev else sam_dir_default(data)).resolve())
    rle_dir = (sam_dir if ev else data / "derived" / RLE_KIND / f"v{VERSION}") if a.rle else None
    frames = eval_frames(data=data) if ev else plan_listing(list_data(data))
    try:
        stamp = json.loads((sam_dir / STAMP).read_text())
    except (OSError, ValueError):
        stamp = None
    m = write_manifest(data, sam_dir, Path(a.out_dir).resolve(), frames, cfg, rle_dir,
                       stamp=stamp)
    return EXIT_FAIL if m["missing"] else EXIT_OK


# ------------------------------------------------------------------------------- pack
def pack(a: argparse.Namespace, data: Path, src: Path, out: Path, kind: str) -> int:
    """tools/data/pack_derived.py pack for one kind; its PACK_SHARD lines go to OUT/sam_pack.json
    (collect_derived.sh then brings DATA/derived/out/u2f-derived-<kind>-v1-sNN.zip to the Mac)."""
    argv = [sys.executable, "-u", str(PACK), "pack", "--kind", kind, "--version", str(VERSION),
            "--dataset", "kitti", "--src", str(src), "--run", a.run or "unknown",
            "--data-root", str(data), "--out", str(data / "derived" / "out"),
            "--inputs-from", str(data / ".bundles"), "--notice", str(NOTICE), "--replace"]
    if os.environ.get("U2_SESSION"):
        argv += ["--session", os.environ["U2_SESSION"]]
    if os.environ.get("U2_GIT_SHA"):
        argv += ["--code-sha", os.environ["U2_GIT_SHA"]]
    if a.pack_manifest:
        argv += ["--manifest", a.pack_manifest]
    say("SAM_PACK", " ".join(argv[2:]))
    r = subprocess.run(argv, capture_output=True, text=True, cwd=str(REPO))
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    shards = []
    for ln in r.stdout.splitlines():
        if ln.startswith("PACK_SHARD|"):
            _, name, nbytes, sha, nfiles = ln.split("|")
            shards.append({"name": name, "bytes": int(nbytes), "sha256": sha,
                           "n_files": int(nfiles)})
    doc_path = out / "sam_pack.json"
    try:
        doc = json.loads(doc_path.read_text())
    except (OSError, ValueError):
        doc = {}
    doc[kind] = {"rc": r.returncode, "shards": shards, "out": str(data / "derived" / "out"),
                 "utc": utc_now()}
    atomic_json(doc_path, doc)
    if r.returncode != 0:
        say("SAM_ERROR", f"pack_derived.py pack --kind {kind} exited {r.returncode}")
        return EXIT_SEAL if r.returncode == 50 else (r.returncode or EXIT_FAIL)
    return EXIT_OK


# ------------------------------------------------------------------------------- plan
def cmd_plan(a: argparse.Namespace) -> int:
    if a.eval_frames:
        if a.from_manifest:
            names: dict[str, set[str]] = {}
            tops = {top: ds for ds, top in DATASETS}
            rx = re.compile(r"^(kitti1[25])/multi_view/training/image_2/([0-9]{6}_10\.png)$")
            for f in sorted((Path(a.from_manifest) / "files").glob("u2f-*mv-*.sha256")):
                for line in f.read_text().splitlines():
                    m = rx.match(line.split(None, 1)[-1].strip())
                    if m:
                        names.setdefault(tops[m.group(1)], set()).add(m.group(2))
            frames = eval_frames(names=names)
        else:
            frames = eval_frames(data=Path(a.data))
        doc = {"frames": len(frames), "kind": EVAL_KIND, "subsets": counts(frames),
               "outputs_per_frame": 2 + (1 if a.rle else 0)}
        if a.s_per_frame:
            doc["g4_h_E"] = round(len(frames) * a.s_per_frame / 3600, 2)
        say("SAM_PLAN", json.dumps(doc, sort_keys=True))
        return EXIT_OK
    if a.from_manifest:
        listing = list_manifest(Path(a.from_manifest))
        if len(listing) != len(DATASETS) * len(SUBS) * len(SIDES):
            raise Stop(EXIT_PRE, f"{a.from_manifest}/files lists {len(listing)} of 8 image dirs")
    else:
        listing = list_data(Path(a.data))
    frames = plan_listing(listing)
    doc = {"frames": len(frames), "pairs": n_pairs(listing), "subsets": counts(frames),
           "outputs_per_frame": 2 + (1 if a.rle else 0)}
    if a.s_per_frame:
        doc["g4_h_E"] = round(len(frames) * a.s_per_frame / 3600, 2)
    say("SAM_PLAN", json.dumps(doc, sort_keys=True))
    return EXIT_OK


# ------------------------------------------------------------------------------- CLI
def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data", default=os.environ.get("U2_DATA", "/content/data"))
    p.add_argument("--sam-dir", default=None, help="default U2_SAM_DIR or DATA/derived/sam/v1")
    p.add_argument("--key-format", choices=("npz", "npy"), default="npz")
    p.add_argument("--rle", action="store_true", help="also write samrle v1 (SAM's RLE records)")
    p.add_argument("--eval-frames", action="store_true",
                   help="frame 10 of KITTI-15/12 training image_2 into evalsam v1 (PX-F3)")
    p.add_argument("--generator", choices=("sam", "stub"), default="sam")
    p.add_argument("--model-type", default=MODEL_TYPE)
    p.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="fp32")
    p.add_argument("--amg", action="append", metavar="KEY=VALUE",
                   help="override one generator setting (anything but points_per_batch is a "
                        "deviation)")
    p.add_argument("--checkpoint", default=None, help="a deviation: another checkpoint file")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="sam_masks.py", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--data", default=os.environ.get("U2_DATA", "/content/data"))
    p.add_argument("--from-manifest", default=None, metavar="DIR",
                   help="count from DIR/files/u2f-*mv-*.sha256 (the Mac)")
    p.add_argument("--rle", action="store_true")
    p.add_argument("--eval-frames", action="store_true")
    p.add_argument("--s-per-frame", type=float, default=None, help="for a G4-h estimate [E]")
    r = sub.add_parser("run")
    _common(r)
    r.add_argument("--out-dir", required=True, help="the run's out/ ({out_dir})")
    r.add_argument("--procs", type=int, default=4)
    r.add_argument("--stagger-s", type=float, default=20.0)
    r.add_argument("--bus-every-s", type=float, default=60.0)
    r.add_argument("--limit", type=int, default=0, help="first N planned frames (a bench)")
    r.add_argument("--device", default="auto")
    r.add_argument("--pack", action="store_true")
    r.add_argument("--run", default=os.environ.get("U2_RUN_ID"))
    r.add_argument("--pack-manifest", default=None, help=argparse.SUPPRESS)
    r.add_argument("--sam-tarball", default=None,
                   help="a local copy of the pinned archive (checked against the same sha256)")
    w = sub.add_parser("worker")
    w.add_argument("--data", required=True)
    w.add_argument("--sam-dir", required=True)
    w.add_argument("--rle-dir", default=None)
    w.add_argument("--frames-file", required=True)
    w.add_argument("--progress", required=True)
    w.add_argument("--k", type=int, required=True)
    w.add_argument("--pydeps", default=None)
    m = sub.add_parser("manifest")
    _common(m)
    m.add_argument("--out-dir", required=True)
    return ap


def main(argv: list[str] | None = None) -> int:
    try:
        a = parser().parse_args(argv)
    except SystemExit as e:
        return EXIT_USAGE if e.code else EXIT_OK
    try:
        if a.cmd == "run" and a.procs < 1:
            raise Stop(EXIT_USAGE, "--procs must be >= 1")
        return {"plan": cmd_plan, "run": cmd_run, "worker": cmd_worker,
                "manifest": cmd_manifest}[a.cmd](a)
    except Stop as e:
        say("SAM_REFUSED" if e.code in (EXIT_REFUSED, EXIT_USAGE) else "SAM_ERROR", e.msg)
        return e.code


if __name__ == "__main__":
    sys.exit(main())
