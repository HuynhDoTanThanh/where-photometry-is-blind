"""Readers shared by refine_pose.py, self_triangulate.py and render_frig.py: one geometry item, one
evidence item and one SAM label map, all brought to the training grid; and the bridge to the
teacher-factory cache format (tools/teacher/teacher_cache.py, cache schema 1).

How to use (library only):
    geo = GeomIndex(path)                         # a P1 cache dir (manifest.json + shards) OR a
                                                  # teacher_cache root (<top>/<sub>/<side>/*.npz)
    arrays, meta = geo.get("kitti15_training_image_2_000012_w03-11")
    K = window_K(arrays, meta, i10, "teacher")    # the window's ONE K (eq. 0): teacher | calib
    g = geometry_frame(arrays, meta, idx, grid, K_model=K)  # inverse depth, validity, K, pose
    ev = evidence(arrays, "10to11", grid)         # flow, alpha, vis (fp16 -> float64)
    lab = sam_labels(sam_dir, row, frame, grid)   # SAM full_seg, nearest-resized, or None
    export_teacher_cache(p1_cache_dir, root, teacher)  # P1 geometry items -> teacher_cache files

The two formats. teacher-factory owns the teacher cache (one validated .npz per window; its
skill's references/cache-format.md): MapAnything, pi3, Monodepth2 come from
tools/teacher/run_geometry.py in that format, and the COLMAP and self-triangulation teachers are
exported to it (kinds evalcolmap / evalselftri of tools/teacher/teachers.json), so the gate and
the loader read one format. The P1 format (p1lib/cache.py: tar shards + manifest) carries the P1
job outputs (evidence, FF, poses, F_rig). In the internal (arrays, meta) form both look alike:
inv_depth, valid (packbits; teacher_cache flags == 0, i.e. neither INVALID nor TEACHER_MASK
[E: a pixel the teacher itself masks is not used]), conf_rank, pose_c2w and K (float64).

The window's K. Equation 0 has one K, so every renderer takes ONE K per window (window_K): the
source frame's own teacher K ('teacher': MapAnything's predicted K, pi3's fitted K, or the
calibration for COLMAP, self-triangulation and Monodepth2) or the calibration rescaled to the
model grid ('calib'). k_source(meta) says which kind the cache's K is; a teacher whose K is not
the calibration needs an explicit choice (render_frig.py / refine_pose.py --k), recorded in their
identities, because the choice is W2-14's and moves every F_rig [M teacher-factory: MapAnything's
fy differs from the calibration by up to 14 %].

Why. The evidence (dump_eval) lives on U2Flow's 256 x 832 grid; teachers live at their own
resolution; SAM masks at native resolution. Everything meets on the grid by the same half-pixel
rule (rigid.to_grid); labels use the nearest source pixel, never an interpolated label [E].
SAM's full_seg layout is the loaders' own: <sam>/<kitti2015|kitti2012>_mv_sam1_fullseg/<sub>/
<side>/<seq>_<ff>.png [V core/u2f_paths.py sam_dirs; tools/vm/sam_masks.py header].
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import cache as CA
from . import common as C

SAM_DS = {"kitti15": "kitti2015", "kitti12": "kitti2012"}


K_CHOICES = ("teacher", "calib")


def k_source(meta: dict) -> str:
    """What the cache's K is: 'calibration' (COLMAP, self-triangulation, Monodepth2, the P1 caches
    whose items carry K_native), else the teacher's own ('predicted ...', 'fitted ...'), else
    'unknown' (an old or foreign cache: treated as not the calibration)."""
    ks = meta.get("k_source")
    if ks is None and "K_native" in meta:
        return "calibration"
    return str(ks) if ks else "unknown"


def resolve_k(meta: dict, k: str | None) -> str:
    """The --k choice of render_frig.py / refine_pose.py: required unless the cache's K is the
    calibration (then both choices are the same K and 'teacher' is used)."""
    if k is not None:
        if k not in K_CHOICES:
            raise C.Stop(C.EXIT_USAGE, f"--k must be one of {', '.join(K_CHOICES)}")
        return k
    if k_source(meta) == "calibration":
        return "teacher"
    raise C.Stop(C.EXIT_USAGE, f"this cache's K is '{k_source(meta)}', not the calibration: "
                               "give --k teacher|calib (the W2-14 decision, recorded in the "
                               "identity; teacher-factory SKILL.md section 7)")


def window_K(arrays: dict, meta: dict, i: int, k: str) -> np.ndarray:
    """The ONE K (model grid) of a window for eq. 0: frame i's own K ('teacher') or the
    calibration rescaled to the model grid ('calib'; a P1 item derives it from K_native)."""
    import rigid

    if k == "teacher":
        return np.asarray(arrays["K"][i], np.float64)
    if k != "calib":
        raise C.Stop(C.EXIT_USAGE, f"window_K: unknown choice {k!r}")
    if "K_calib" in arrays:
        return np.asarray(arrays["K_calib"][i], np.float64)
    if "K_native" in meta:
        return rigid.zoom_K(np.asarray(meta["K_native"]), tuple(meta["native_hw"]),
                            tuple(meta["shape"][1:]))
    raise C.Stop(C.EXIT_PRE, "this cache carries no calibration K (K_calib or K_native)")


def geometry_frame(arrays: dict, meta: dict, i: int, grid: tuple[int, int],
                   f32: bool = False, K_model=None) -> dict:
    """Frame i on the grid. K_model: the window's one K at the model grid (window_K); default
    frame i's own K, which only a single-frame consumer may use (rigid.py eq. 0)."""
    import rigid

    key = "inv_depth_f32" if f32 and "inv_depth_f32" in arrays else "inv_depth"
    inv = arrays[key][i].astype(np.float64)
    shape = tuple(meta["shape"])
    valid = CA.unpack_bits(arrays["valid"], shape)[i] & (inv > 0)
    g, ok = rigid.resize_bilinear(inv, grid, valid)
    pose_ok = bool(arrays["pose_ok"][i]) if "pose_ok" in arrays else True
    Km = np.asarray(arrays["K"][i] if K_model is None else K_model, np.float64)
    return {"inv_model": np.where(valid, inv, 0.0), "valid_model": valid,
            "inv": np.where(ok, g, 0.0), "valid": ok, "K_model": Km,
            "K": rigid.zoom_K(Km, inv.shape, grid), "pose_c2w": arrays["pose_c2w"][i],
            "pose_ok": pose_ok, "conf_rank": arrays["conf_rank"][i] if "conf_rank" in arrays
            else None}


def evidence(arrays: dict, d: str, grid: tuple[int, int]) -> dict:
    f = arrays[f"flow_{d}"].astype(np.float64)
    if f.shape[1:] != tuple(grid):
        raise C.Stop(C.EXIT_PRE, f"evidence grid {f.shape[1:]} is not {grid}")
    out = {"flow": f}
    if f"alpha_{d}" in arrays:
        out["alpha"] = arrays[f"alpha_{d}"].astype(np.float64)
    if f"vis_{d}" in arrays:
        out["vis"] = CA.unpack_bits(arrays[f"vis_{d}"], grid)
    return out


def resize_labels_nearest(labels: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    import rigid

    h, w = labels.shape
    H, W = grid
    xs = np.clip(np.rint(rigid.to_grid(np.arange(W), W, w)), 0, w - 1).astype(int)
    ys = np.clip(np.rint(rigid.to_grid(np.arange(H), H, h)), 0, h - 1).astype(int)
    return labels[np.ix_(ys, xs)]


def sam_path(sam_dir: Path, row: dict, frame: int) -> Path:
    return (Path(sam_dir) / f"{SAM_DS[row['ds']]}_mv_sam1_fullseg" / row["sub"] / row["side"]
            / f"{row['seq']}_{frame:02d}.png")


def sam_labels(sam_dir: Path | None, row: dict, frame: int, grid: tuple[int, int]):
    if sam_dir is None:
        return None
    p = sam_path(sam_dir, row, frame)
    if not p.is_file():
        return None
    import cv2

    lab = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if lab is None or lab.ndim != 2:
        raise C.Stop(C.EXIT_PRE, f"{p}: not a single-channel label map")
    return resize_labels_nearest(lab.astype(np.int32), grid)


def window_row(item_id: str, plan: dict) -> dict:
    for r in plan["eval_windows"]:
        if r["id"] == item_id:
            return r
    raise C.Stop(C.EXIT_PRE, f"{item_id}: not an evaluation window of the plan")


def evidence_id(row: dict) -> str:
    """The dump_eval item of a window: its sequence's frame 10."""
    from . import kitti as KT

    return KT.pair_id(row["ds"], row["sub"], row["side"], row["seq"], 10)


# ------------------------------------------------------------------------------ both formats
ROLE_WIN = {"eval-a": "w03-11", "eval-b": "w10-18", "train-a": "w00-08", "train-b": "w13-20",
            "ref": "w05-15"}


def teacher_cache_module():
    import sys

    sys.path.insert(0, str(C.REPO / "tools" / "teacher"))
    import teacher_cache  # noqa: PLC0415

    return teacher_cache


def tf_window_name(rel: str) -> str | None:
    """Our item id for a teacher_cache file name '<top>/<sub>/<side>/<seq>_<a>_<b>.npz', by its
    first and last frame (eval: last <= 11 is w03-11, first >= 10 is w10-18; train: last <= 8 is
    w00-08, first >= 13 is w13-20); None when ambiguous (then the file's meta.role decides)."""
    import re

    m = re.match(r"^(kitti1[25])/(training|testing)/(image_[23])/(\d{6})_(\d\d)_(\d\d)\.npz$",
                 rel)
    if not m:
        return None
    top, sub, side, seq, a, b = m.groups()
    a, b = int(a), int(b)
    if a <= 9 and b == 11:
        win = "w03-11"
    elif a == 10 and b >= 12:
        win = "w10-18"
    elif b <= 8:
        win = "w00-08"
    elif a >= 13:
        win = "w13-20"
    elif a >= 5 and b >= 12 and a <= 9:
        win = "w05-15"
    else:
        return None
    return f"{top}_{sub}_{side}_{seq}_{win}"


def from_teacher_cache(arrays: dict, meta: dict) -> tuple[dict, dict]:
    tc = teacher_cache_module()
    inv = arrays["invdepth"]
    valid = (arrays["flags"] == 0) & (inv.astype(np.float32) > 0)
    hw = arrays["hw_orig"]
    out = {"inv_depth": inv, "valid": CA.pack_bits(valid), "conf_rank": arrays["conf_rank"],
           "pose_c2w": arrays["pose_c2w"].astype(np.float64),
           "K": arrays["K"].astype(np.float64), "K_calib": arrays["K_calib"].astype(np.float64)}
    if "invdepth_f32" in arrays:  # a W2-14 reference run (run_geometry.py --store-f32)
        out["inv_depth_f32"] = arrays["invdepth_f32"]
    m = {"frames": [int(f) for f in arrays["frames"]], "native_hw": [int(hw[0, 0]),
                                                                     int(hw[0, 1])],
         "shape": list(inv.shape), "model_hw": list(inv.shape[1:]),
         "win": ROLE_WIN.get(meta.get("role", ""), None), "k15_half": meta.get("k15_half"),
         "teacher": {"teacher": meta.get("teacher"), "format": f"teacher_cache schema "
                     f"{tc.SCHEMA}", "stamp_sha256": meta.get("stamp_sha256")},
         "tf_window": meta.get("window"), "k_source": meta.get("k_source"),
         "producer": meta.get("producer")}
    return out, m


class GeomIndex:
    """Geometry items by our id, from a P1 cache dir or a teacher_cache root."""

    def __init__(self, path: Path):
        self.path = Path(path)
        C.refuse_sealed(self.path)
        if (self.path / "manifest.json").is_file():
            self.p1 = CA.CacheIndex(self.path)
            self.manifest = self.p1.manifest
            self.files = None
            return
        self.p1 = None
        tc = teacher_cache_module()
        self.files = {}
        for f in tc.iter_files(self.path):
            rel = f.relative_to(self.path).as_posix()
            name = tf_window_name(rel)
            if name is None:
                _, meta = tc.read(f)
                d = rel.split("/")
                name = f"{d[0]}_{d[1]}_{d[2]}_{d[3][:6]}_{ROLE_WIN[meta['role']]}"
            self.files[name] = f
        if not self.files:
            raise C.Stop(C.EXIT_PRE, f"{self.path}: neither a P1 cache (manifest.json) nor a "
                                     "teacher_cache root (*.npz)")
        stamp = self.path / tc.STAMP
        ident = C.load_json(stamp) if stamp.is_file() else {"root": str(self.path)}
        self.manifest = {"source": self.path.parts[-2] if self.path.name.startswith("v")
                         else self.path.name, "identity": ident, "format": "teacher_cache"}

    def ids(self) -> list[str]:
        return self.p1.ids() if self.p1 is not None else sorted(self.files)

    def get(self, item_id: str):
        if self.p1 is not None:
            return self.p1.get(item_id)
        f = self.files.get(item_id)
        if f is None:
            return None
        a, m = teacher_cache_module().read(f)
        if m.get("kind") != "geometry":
            raise C.Stop(C.EXIT_PRE, f"{f}: not a geometry window")
        arrays, meta = from_teacher_cache(a, m)
        meta["win"] = meta["win"] or item_id.rsplit("_", 1)[1]
        return arrays, meta


def producer_gpu(geo: GeomIndex, meta: dict) -> dict:
    """The GPU behind one geometry item, for a teacher row's gpu_name (prereg/drafts/GATE-1.json
    _row_conventions): the GPU that ran the teacher's network (run_geometry.py meta.producer,
    a P1 manifest's host), else, for self-triangulation, the GPU that ran U2Flow's evidence
    (settings.evidence_host); None when nothing recorded one (a gate selector then never
    matches: fail-closed, never a guessed GPU)."""
    if geo.p1 is not None:
        host = geo.manifest.get("host") or {}
        ev = (geo.manifest.get("settings") or {}).get("evidence_host") or {}
    else:
        host = meta.get("producer") or {}
        ev = {"gpu_name": host.get("evidence_gpu_name")}
    if host.get("gpu_name"):
        return {"gpu_name": host["gpu_name"], "gpu_source": "geometry"}
    if ev.get("gpu_name"):
        return {"gpu_name": ev["gpu_name"], "gpu_source": "evidence"}
    return {"gpu_name": None, "gpu_source": None}


def export_teacher_cache(p1_dir: Path, root: Path, teacher: str, kind: str) -> int:
    """Write every P1 geometry item of p1_dir as a teacher_cache window under root (validated by
    teacher_cache.write); the stamp records the P1 manifest's identity. Returns the count."""
    from . import kitti as KT

    tc = teacher_cache_module()
    idx = CA.CacheIndex(Path(p1_dir))
    root = Path(root)
    C.refuse_sealed(root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = root / tc.STAMP
    ident = {"exported_from": "tools/p1", "kind": kind, "teacher": teacher,
             "p1_identity": idx.manifest.get("identity")}
    if stamp.is_file() and C.load_json(stamp).get("p1_identity") != ident["p1_identity"]:
        raise C.Stop(C.EXIT_REFUSED, f"{root} holds another recipe's teacher cache")
    C.atomic_json(stamp, ident)
    host = idx.manifest.get("host") or {}
    ev_host = (idx.manifest.get("settings") or {}).get("evidence_host") or {}
    n = 0
    for item in idx.ids():
        a, m = idx.get(item)
        d = KT.parse_id(item)
        shape = tuple(m["shape"])
        # COLMAP may leave frames unregistered: they are dropped, so pose_c2w[0] = I (the
        # teacher_cache rule) is the first REGISTERED camera, never a placeholder
        keep = np.asarray(a["pose_ok"], bool) if "pose_ok" in a else np.ones(shape[0], bool)
        if keep.sum() < 2:
            continue
        valid = CA.unpack_bits(a["valid"], shape)[keep]
        inv = a["inv_depth"].astype(np.float64)[keep]
        with np.errstate(divide="ignore"):
            depth = np.where(valid & (inv > 0), 1.0 / np.where(inv > 0, inv, 1.0), np.nan)
        conf = (a["conf"] if "conf" in a else a["conf_rank"]).astype(np.float32)[keep]
        import rigid  # noqa: PLC0415

        nk = int(keep.sum())
        kcal = rigid.zoom_K(np.asarray(m["K_native"]), tuple(m["native_hw"]), shape[1:])
        fr = [f for f, k in zip(m["frames"], keep, strict=True) if k]
        arrays = tc.geometry_arrays(depth, conf, None, a["pose_c2w"][keep], a["K"][keep],
                                    [kcal] * nk, fr, [m["native_hw"]] * nk)
        wid = f"{d['ds']}/{d['sub']}/{d['side']}/{d['seq']}_{fr[0]:02d}_{fr[-1]:02d}"
        role = {v: k for k, v in ROLE_WIN.items()}.get(m["win"], m["win"])
        meta = {"schema": tc.SCHEMA, "kind": "geometry", "teacher": teacher, "window": wid,
                "role": role, "grid": "model", "model_wh": [shape[2], shape[1]],
                "scale": "relative", "k_source": "calibration", "k15_half": m.get("k15_half"),
                "stamp_sha256": C.sha256_bytes(repr(ident).encode()),
                "producer": {"gpu_name": host.get("gpu_name"), "device": "p1",
                             "evidence_gpu_name": ev_host.get("gpu_name")}}
        tc.write(root / f"{wid}.npz", arrays, meta)
        n += 1
    return n
