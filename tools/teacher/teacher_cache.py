"""The teacher cache format: one compressed .npz per window, written atomically, read and checked
by the same module. The only interface between the teacher factory and its consumers (rigid.py,
refine_pose.py, arbitrate.py, the teacher loader of W4-03).

Layout (under a cache root, by default U2_DATA/derived/<kind>/v<N>/, kind from
tools/teacher/teachers.json cache_kinds: eval* for evaluation windows, geom*/teacher* for training
windows):
    <top>/<sub>/<side>/<seq>_<first:02d>_<last:02d>.npz      e.g. kitti15/training/image_2/
                                                             000012_03_11.npz
    .teacher_settings.json   the recipe stamp (a dot-file: pack_derived and manifests skip it)
The name carries the first and last frame on purpose: pack_derived.py finds frames 09-12 through
its NNNNNN(_FF)+ token [V tools/data/pack_derived.py:144, 315-321], so a window touching them is
refused in any kind but eval*.

Geometry window (kind 'geometry'; N frames, model grid h x w unless grid == 'train'):
    frames     int16 (N,)          KITTI frame numbers, increasing
    invdepth   float16 (N,h,w)     1 / z-depth in the teacher's own scale; 0 where invalid
    conf       float16 (N,h,w)     the teacher's own confidence (MapAnything conf; pi3
                                   sigmoid(logit); 1 for a teacher without one)
    conf_rank  uint8 (N,h,w)       percentile rank of conf over the window's valid pixels,
                                   0..255 (proposal io: 'confidence chuan hoa theo hang trong cua
                                   so'); TIES SHARE THEIR MEAN RANK (tools/p1/rigid.py rank_u8's
                                   rule), so a constant conf is 128 everywhere; 0 where invalid
    flags      uint8 (N,h,w)       bit 0 INVALID (non-finite or <= 0 depth), bit 1 TEACHER_MASK
                                   (the teacher's own validity mask says no: MapAnything
                                   non_ambiguous_mask, pi3 sigmoid(conf) <= 0.1)
    pose_c2w   float32 (N,4,4)     OpenCV camera-to-world, world = the window's first camera
    K          float32 (N,3,3)     intrinsics at the stored grid (predicted, fitted or given)
    K_calib    float32 (N,3,3)     KITTI P_rect K rescaled to the stored grid
    hw_orig    int32 (N,2)         original image height, width
    meta       0-d str             JSON: schema, teacher, kind, window id, grid, model_wh,
                                   scale ('metric' | 'relative'), k_source, stamp sha256;
                                   from cache v2: producer {gpu_name, device} (the GPU that ran
                                   the network: a teacher row's gpu_name), conf_source and
                                   conf_tie_max_frac (the largest tie group's share)
    invdepth_f32 float32 (N,h,w)   OPTIONAL, only in a W2-14 reference run (run_geometry.py
                                   --store-f32, never a canonical cache): the same inverse depth
                                   before fp16 rounding
Flow window (kind 'flow'; P consecutive pairs of the window, training grid):
    frames int16 (N,), pairs int16 (P,2), flow_fw float16 (P,2,H,W) (t -> t+1, pixels),
    flow_bw float16 (P,2,H,W) (t+1 -> t), hw_orig, meta.

No threshold is applied here (no sky, depth-edge or b_T rule): the cache carries the teacher's raw
evidence, and every hand-set threshold lives downstream where the method skill's SOTA check
applies. The rank transform has no parameter.

How to invoke: imported (numpy only). `python tools/teacher/teacher_cache.py check FILE...`
re-reads files and prints CACHE_OK / CACHE_BAD (exit 0 / 1); `estimate` prints sizes [E].
Exit codes: 0 ok · 1 a file fails the schema · 64 usage.

Why it is built this way.
- Per-window files make resume trivial (a file exists only once complete: dot-temp + rename) and
  keep the tens of thousands of files under pack_derived's 60,000-entry shard cap.
- np.savez_compressed is byte-deterministic for equal arrays (fixed zip dates) [M numpy 2.5.3,
  data-bundles references/derived.md section 7], so a re-run of a deterministic teacher hashes
  the same and a cross-VM difference is visible in the manifest.
- fp16 inverse depth: 1/z keeps resolution where flow is sensitive (near) and maps sky/far to
  ~0; fp16 has 11 significant bits, about 0.05 % relative at any depth [E].
- Ties in the rank (review 2026-10-03): ranking ties by raster order made a teacher without a
  confidence (Monodepth2: conf 1 everywhere) rank the window's first frame 0 and its last ~1
  [M: frame 0 mean c_T 0.0, frame 7 0.996], and re-ranked COLMAP's {1, 0.5, 0} by position;
  c_T enters the gate weight (eq. 10), so the mean rank of a tie group is the only answer that
  carries no position. Cache v2 (run_geometry.py VERSION) has it; v1 never existed on a G4.
"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

SCHEMA = 1
FLAG_INVALID, FLAG_TEACHER_MASK = 1, 2
GEOM_KEYS = {"frames": "int16", "invdepth": "float16", "conf": "float16", "conf_rank": "uint8",
             "flags": "uint8", "pose_c2w": "float32", "K": "float32", "K_calib": "float32",
             "hw_orig": "int32"}
GEOM_OPTIONAL = {"invdepth_f32": "float32"}  # a W2-14 reference run only (--store-f32)
FLOW_KEYS = {"frames": "int16", "pairs": "int16", "flow_fw": "float16", "flow_bw": "float16",
             "hw_orig": "int32"}
STAMP = ".teacher_settings.json"


class CacheError(Exception):
    pass


def window_rel(w: dict) -> str:
    """The cache path of window W relative to the cache root."""
    fr = w["frames"]
    return f"{w['top']}/{w['sub']}/{w['side']}/{w['seq']}_{fr[0]:02d}_{fr[-1]:02d}.npz"


def rank_uint8(conf, valid):
    """Percentile rank of CONF over VALID pixels of the whole window as uint8 (0 lowest, 255
    highest): ties share their mean rank, so equal confidences get equal ranks wherever they
    are (a constant conf gives 128); one valid pixel gives 255; invalid pixels 0. The same
    numbers as tools/p1/rigid.py rank_u8 (tests/test_teacher_factory.py checks it)."""
    import numpy as np

    out = np.zeros(conf.shape, dtype=np.uint8)
    v = np.asarray(valid, bool).reshape(-1)
    c = np.asarray(conf).reshape(-1)[v].astype(np.float64)
    if c.size == 0:
        return out
    flat = out.reshape(-1)
    if c.size == 1:
        flat[v] = 255
        return out
    order = np.argsort(c, kind="stable")
    ranks = np.empty(c.size, dtype=np.float64)
    ranks[order] = np.arange(c.size, dtype=np.float64)
    _uniq, inv = np.unique(c, return_inverse=True)
    mean_rank = np.bincount(inv, weights=ranks) / np.bincount(inv)
    flat[v] = np.rint(255.0 * mean_rank[inv] / (c.size - 1)).astype(np.uint8)
    return out


def tie_max_frac(conf, valid) -> float:
    """The share of the window's valid pixels in its largest group of equal confidences (1.0
    for a teacher without one; pi3's sigmoid saturating at 1.0 shows here) [W2-04 smoke]."""
    import numpy as np

    c = np.asarray(conf).reshape(-1)[np.asarray(valid, bool).reshape(-1)]
    if c.size == 0:
        return 0.0
    _u, counts = np.unique(c, return_counts=True)
    return float(counts.max() / c.size)


def scale_K(K, src_wh, dst_wh):
    """K for an image resized (no crop) from SRC_WH to DST_WH, pixel-centre convention:
    f' = s f, c' = s (c + 0.5) - 0.5 (proposal equations[2])."""
    import numpy as np

    K = np.asarray(K, dtype=np.float64)
    sx, sy = dst_wh[0] / src_wh[0], dst_wh[1] / src_wh[1]
    out = K.copy()
    out[0, 0], out[1, 1] = K[0, 0] * sx, K[1, 1] * sy
    out[0, 2], out[1, 2] = sx * (K[0, 2] + 0.5) - 0.5, sy * (K[1, 2] + 0.5) - 0.5
    out[0, 1] = K[0, 1] * sx
    return out


def relative_c2w(c2w):
    """Express camera-to-world poses in the frame of the first camera: inv(T_0) @ T_i."""
    import numpy as np

    c2w = np.asarray(c2w, dtype=np.float64)
    inv0 = np.linalg.inv(c2w[0])
    return np.einsum("ij,njk->nik", inv0, c2w)


def geometry_arrays(depth, conf, teacher_mask, pose_c2w, K, K_calib, frames, hw_orig,
                    store_f32: bool = False) -> dict:
    """The stored arrays from a teacher's raw per-window output (depth N x h x w float). A
    teacher without a confidence (conf None) stores conf 1: every valid pixel ties, so its rank
    is the constant 128 (rank_uint8). store_f32 adds invdepth_f32 (W2-14 reference runs)."""
    import numpy as np

    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    inv = np.zeros_like(depth)
    inv[valid] = 1.0 / depth[valid]
    conf = np.ones_like(depth) if conf is None else np.asarray(conf, dtype=np.float32)
    conf = np.where(np.isfinite(conf), conf, 0.0)
    flags = np.where(valid, 0, FLAG_INVALID).astype(np.uint8)
    if teacher_mask is not None:
        flags |= np.where(np.asarray(teacher_mask, dtype=bool), 0, FLAG_TEACHER_MASK).astype(
            np.uint8)
    out = {"frames": np.asarray(frames, dtype=np.int16),
           "invdepth": inv.astype(np.float16), "conf": conf.astype(np.float16),
           "conf_rank": rank_uint8(conf, valid), "flags": flags,
           "pose_c2w": relative_c2w(pose_c2w).astype(np.float32),
           "K": np.asarray(K, dtype=np.float32), "K_calib": np.asarray(K_calib, np.float32),
           "hw_orig": np.asarray(hw_orig, dtype=np.int32)}
    if store_f32:
        out["invdepth_f32"] = inv.astype(np.float32)
    return out


def encode(arrays: dict, meta: dict) -> bytes:
    import numpy as np

    buf = io.BytesIO()
    payload = dict(arrays)
    payload["meta"] = np.array(json.dumps(meta, sort_keys=True, separators=(",", ":")))
    np.savez_compressed(buf, **payload)
    return buf.getvalue()


def write(path: Path, arrays: dict, meta: dict) -> int:
    """Validate, then write PATH via a dot-named temp file and rename; returns bytes written."""
    validate(arrays, meta)
    data = encode(arrays, meta)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
    return len(data)


def read(path: Path) -> tuple[dict, dict]:
    """(arrays, meta) of one cache file, validated."""
    import numpy as np

    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files if k != "meta"}
        if "meta" not in z.files:
            raise CacheError(f"{path}: no meta")
        meta = json.loads(str(z["meta"]))
    validate(arrays, meta)
    return arrays, meta


def validate(arrays: dict, meta: dict) -> None:
    import numpy as np

    if meta.get("schema") != SCHEMA or meta.get("kind") not in ("geometry", "flow"):
        raise CacheError(f"meta schema/kind {meta.get('schema')}/{meta.get('kind')}")
    keys = GEOM_KEYS if meta["kind"] == "geometry" else FLOW_KEYS
    missing = set(keys) - set(arrays)
    if missing:
        raise CacheError(f"missing arrays {sorted(missing)}")
    for k, dt in keys.items():
        if str(arrays[k].dtype) != dt:
            raise CacheError(f"{k}: dtype {arrays[k].dtype} != {dt}")
    extra = set(arrays) - set(keys) - (set(GEOM_OPTIONAL) if meta["kind"] == "geometry"
                                       else set())
    if extra:
        raise CacheError(f"unknown arrays {sorted(extra)}")
    fr = arrays["frames"]
    n = len(fr)
    if n < 2 or not np.all(np.diff(fr.astype(int)) > 0):
        raise CacheError("frames must be >= 2, strictly increasing")
    if arrays["hw_orig"].shape != (n, 2):
        raise CacheError("hw_orig shape")
    if meta["kind"] == "geometry":
        shp = arrays["invdepth"].shape
        if len(shp) != 3 or shp[0] != n:
            raise CacheError(f"invdepth shape {shp}")
        for k in ("conf", "conf_rank", "flags"):
            if arrays[k].shape != shp:
                raise CacheError(f"{k} shape {arrays[k].shape} != {shp}")
        for k in ("pose_c2w", "K", "K_calib"):
            want = (n, 4, 4) if k == "pose_c2w" else (n, 3, 3)
            if arrays[k].shape != want:
                raise CacheError(f"{k} shape {arrays[k].shape} != {want}")
        if not np.all(np.isfinite(arrays["invdepth"].astype(np.float32))):
            raise CacheError("non-finite inverse depth")
        if "invdepth_f32" in arrays:
            f = arrays["invdepth_f32"]
            if str(f.dtype) != "float32" or f.shape != shp or not np.all(np.isfinite(f)):
                raise CacheError("invdepth_f32 must be finite float32 of invdepth's shape")
        if not np.all(np.isfinite(arrays["pose_c2w"])):
            raise CacheError("non-finite pose")
        if not np.allclose(arrays["pose_c2w"][0], np.eye(4), atol=1e-4):
            raise CacheError("pose_c2w[0] is not the identity (world = first camera)")
        if meta.get("grid") not in ("model", "train"):
            raise CacheError("meta.grid must be model or train")
    else:
        p = arrays["pairs"]
        if p.ndim != 2 or p.shape[1] != 2:
            raise CacheError("pairs shape")
        for k in ("flow_fw", "flow_bw"):
            s = arrays[k].shape
            if len(s) != 4 or s[0] != len(p) or s[1] != 2:
                raise CacheError(f"{k} shape {s}")
            if not np.all(np.isfinite(arrays[k].astype(np.float32))):
                raise CacheError(f"non-finite {k}")


def iter_files(root: Path):
    """Every cache file under ROOT (dot-files and dot-dirs skipped), sorted."""
    out = []
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        out += [Path(dirpath) / f for f in files if f.endswith(".npz") and not f.startswith(".")]
    return sorted(out)


def estimate(n_frames: int, h: int, w: int, raw_conf: bool = True) -> dict:
    """Uncompressed bytes per frame and in total for a geometry cache [E: zlib on fp16 depth
    saves little, on uint8 flags and ranks much more]."""
    per = h * w * (2 + 1 + 1 + (2 if raw_conf else 0)) + 4 * 16 + 2 * 4 * 9 + 8 + 2
    return {"bytes_per_frame": per, "total_GB": round(per * n_frames / 1e9, 2)}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in ("check", "estimate"):
        print("usage: teacher_cache.py check FILE... | estimate N_FRAMES H W", file=sys.stderr)
        return 64
    if argv[0] == "estimate":
        try:
            n, h, w = (int(x) for x in argv[1:4])
        except ValueError:
            return 64
        print("CACHE_ESTIMATE " + json.dumps(estimate(n, h, w)))
        return 0
    bad = 0
    for f in argv[1:]:
        try:
            _arr, meta = read(Path(f))
            print(f"CACHE_OK {f} {meta.get('teacher')} {meta.get('window')}")
        except (CacheError, OSError, ValueError, KeyError) as e:
            bad += 1
            print(f"CACHE_BAD {f} {e}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
