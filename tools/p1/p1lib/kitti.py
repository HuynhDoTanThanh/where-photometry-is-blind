"""KITTI multi-view frames, the P1 windows and pair lists, and the rectified calibration.

How to use (library only; tools/p1/p1_windows.py is the CLI):
    inv = inventory_from_manifests(REPO / "data/manifest/files")   # Mac, no image needed
    inv = inventory_from_data(Path("/content/data"))               # VM, from the extracted tree
    eval_windows(inv, dev_ids)  train_windows(inv)  ff_pairs(inv)  ff_eval_pairs(inv)
    K, P = calib_K(data, "kitti15", "training", "000000", "image_2")  # rectified 3x3 and 3x4
    frame_path(data, "kitti15", "training", "image_2", "000000", 10)

Definitions (each from the plan or the loaders; [M] counts are measured by p1_windows.py):
- Evaluation windows (W2-05, W2-06; proposal final.region_protocol 'Cửa sổ cho chẩn đoán G1'):
  w03-11 = frames 03..11, the scored pair 10 -> 11 at its end with t-1 = 09; w10-18 = frames
  10..18, the pair at its start, t-1 dropped on purpose (it mimics the training pair 13 -> 14,
  which has no t-1). Only image_2 of the training dirs (KITTI-15 ids 0-199, KITTI-12 ids 0-193):
  the scored pair is always _10 -> _11 of image_2 (metric regions.md). A window keeps the frames
  that exist inside its nominal range and is flagged truncated when some are missing (never
  shifted: a shifted window would put the pair at another position, which the G1 table reports
  separately [P]); it is absent with fewer than 2 frames or, for evaluation, without 10 and 11.
  This is tools/teacher/make_windows.py's rule (teacher-factory), checked by
  tests/test_p1_jobs.py, so every P1 tool and every teacher cache see the same windows.
- Training windows (W2-15): w00-08 and w13-20 over training and testing dirs, image_2 and
  image_3, both datasets (the 1,578 sequence-cameras of the plan [P]); never a frame 09-12
  (proposal 'Cửa sổ huấn luyện không bao giờ chứa frame KITTI 9-12').
- FF pairs (W2-11): the loaders' training pairs (consecutive frames, none of them 09-12 [V
  core/flow_datasets.py:233-246, as tools/vm/sam_masks.py pairs_from_names]) whose t-1 exists and
  is not 09-12, since FF needs (t-1, t, t+1) [V train_fusion_kitti.py __main__: model on (09, 10)
  and (10, 11)]. ff_eval_pairs: the diagnostic 10 -> 11 pair with t-1 = 09 (image_2, training
  dirs), tagged diag_only: it never enters a training cache.
- Calibration: KITTI-15 calib_cam_to_cam P_rect_02 / P_rect_03, KITTI-12 calib P2 / P3; K is the
  left 3x3 block of the rectified 3x4 P [V the meta bundle's calib files, read 2026-10-03]. That one
  per-sequence P covers every multi-view frame of the sequence is [U]: the devkit readme says
  nothing about multi_view calibration (read 2026-10-03); the frames are rectified images of
  the same camera, so it is expected [E]. A self-check: refine_pose's PnP residual on dev windows.

Why. The plan's count is '394 sequence (788 cửa sổ)' [P W2-05]; the manifests show that KITTI-12
training 000031 and 000082 lack frames 00-04, so their w03-11 holds frames 05-11 only: 788
windows, 2 truncated [M 2026-10-03]; training windows 3,156, 14 truncated [M]. The counts are
printed by p1_windows.py and recorded in the window file, never assumed.
"""
from __future__ import annotations

import re
from pathlib import Path

DATASETS = {"kitti15": "k15", "kitti12": "k12"}
SUBS = ("training", "testing")
SIDES = ("image_2", "image_3")
EVAL_WINDOWS = {"w03-11": (3, 11), "w10-18": (10, 18)}
TRAIN_WINDOWS = {"w00-08": (0, 8), "w13-20": (13, 20)}
EVAL_PAIR = (10, 11)
EXCLUDED = frozenset(range(9, 13))       # frames 09-12 never enter training
N_TRAINING = {"kitti15": 200, "kitti12": 194}  # [V core/flow_datasets.py:199, 276]
NAME_RX = re.compile(r"^(\d{6})_(\d{2})\.png$")
MANIFEST_RX = re.compile(r"^(kitti1[25])/multi_view/(training|testing)/(image_[23])/"
                         r"(\d{6})_(\d{2})\.png$")


def win_id(ds: str, sub: str, side: str, seq: str, win: str) -> str:
    return f"{ds}_{sub}_{side}_{seq}_{win}"


def pair_id(ds: str, sub: str, side: str, seq: str, t: int) -> str:
    return f"{ds}_{sub}_{side}_{seq}_{t:02d}"


def parse_id(item: str) -> dict:
    """Inverse of win_id / pair_id."""
    m = re.match(r"^(kitti1[25])_(training|testing)_(image_[23])_(\d{6})_(w\d\d-\d\d|\d\d)$",
                 item)
    if not m:
        raise ValueError(f"not a P1 item id: {item!r}")
    ds, sub, side, seq, tail = m.groups()
    d = {"ds": ds, "sub": sub, "side": side, "seq": seq}
    if tail.startswith("w"):
        d["win"] = tail
    else:
        d["t"] = int(tail)
    return d


def frame_path(data: Path, ds: str, sub: str, side: str, seq: str, frame: int) -> Path:
    return Path(data) / ds / "multi_view" / sub / side / f"{seq}_{frame:02d}.png"


def inventory_from_manifests(files_dir: Path) -> dict:
    """(ds, sub, side, seq) -> sorted frames, from data/manifest/files/*-mv-*.sha256 and the
    f0912 bundles (the published file lists; no image is read)."""
    inv: dict = {}
    for f in sorted(Path(files_dir).glob("u2f-*.sha256")):
        if "-mv-" not in f.name:
            continue
        for line in f.read_text().splitlines():
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            m = MANIFEST_RX.match(parts[1].strip())
            if m:
                ds, sub, side, seq, fr = m.groups()
                inv.setdefault((ds, sub, side, seq), set()).add(int(fr))
    return {k: sorted(v) for k, v in inv.items()}


def inventory_from_data(data: Path) -> dict:
    inv: dict = {}
    for ds in DATASETS:
        for sub in SUBS:
            for side in SIDES:
                d = Path(data) / ds / "multi_view" / sub / side
                if not d.is_dir():
                    continue
                for p in d.iterdir():
                    m = NAME_RX.match(p.name)
                    if m:
                        inv.setdefault((ds, sub, side, m.group(1)), set()).add(int(m.group(2)))
    return {k: sorted(v) for k, v in inv.items()}


def _window_rows(inv: dict, keys, wins: dict, extra,
                 need: frozenset = frozenset()) -> tuple[list[dict], list[dict]]:
    """A window keeps the frames that exist inside its nominal range (truncated, flagged); it is
    absent when fewer than 2 remain or a frame in `need` is missing. This is
    tools/teacher/make_windows.py build()'s rule, so both packages see the same windows."""
    rows, absent = [], []
    for key in sorted(keys):
        ds, sub, side, seq = key
        have = set(inv.get(key, ()))
        for w, (a, b) in wins.items():
            nominal = list(range(a, b + 1))
            frames = [f for f in nominal if f in have]
            miss = [f for f in nominal if f not in have]
            wid = win_id(ds, sub, side, seq, w)
            if len(frames) < 2 or not need <= set(frames):
                absent.append({"id": wid, "missing_frames": miss})
                continue
            rows.append({"id": wid, "ds": ds, "sub": sub, "side": side, "seq": seq, "win": w,
                         "frames": frames, "nominal": [a, b], "truncated": bool(miss),
                         **extra(ds, seq, w, frames)})
    return rows, absent


def eval_windows(inv: dict, dev_ids: list[str]) -> tuple[list[dict], list[dict]]:
    """The 10 -> 11 evaluation windows; prio 0 = KITTI-15 dev and KITTI-12 (run first), 1 = the
    other KITTI-15 half (images only: its outputs are kept, never evaluated before unseal)."""
    dev = set(dev_ids)
    keys = [(ds, "training", "image_2", f"{i:06d}") for ds, n in N_TRAINING.items()
            for i in range(n)]

    def extra(ds, seq, w, frames):
        half = None if ds == "kitti12" else ("dev" if seq in dev else "heldout")
        i10, i11 = frames.index(EVAL_PAIR[0]), frames.index(EVAL_PAIR[1])
        return {"pair_idx": [i10, i11],
                "tm1_idx": frames.index(9) if (w == "w03-11" and 9 in frames) else None,
                "k15_half": half, "prio": 1 if half == "heldout" else 0}

    rows, absent = _window_rows(inv, keys, EVAL_WINDOWS, extra, frozenset(EVAL_PAIR))
    rows.sort(key=lambda r: (r["prio"], r["id"]))
    return rows, absent


def train_windows(inv: dict) -> tuple[list[dict], list[dict]]:
    keys = [k for k in inv if k[1] in SUBS and k[2] in SIDES]

    def extra(ds, seq, w, frames):
        assert not EXCLUDED & set(frames)
        return {"prio": 0}

    rows, absent = _window_rows(inv, keys, TRAIN_WINDOWS, extra)
    rows.sort(key=lambda r: r["id"])
    return rows, absent


def training_pairs(frames: list[int]) -> list[tuple[int, int]]:
    """The loaders' pair rule on one sequence's frames: consecutive numbers, none of them 09-12."""
    s = set(frames)
    return [(t, t + 1) for t in sorted(s) if t + 1 in s and t not in EXCLUDED
            and t + 1 not in EXCLUDED]


def ff_pairs(inv: dict) -> list[dict]:
    rows = []
    for key in sorted(inv):
        ds, sub, side, seq = key
        have = set(inv[key])
        for t, _ in training_pairs(inv[key]):
            if t - 1 in have and t - 1 not in EXCLUDED:
                rows.append({"id": pair_id(ds, sub, side, seq, t), "ds": ds, "sub": sub,
                             "side": side, "seq": seq, "t": t, "frames": [t - 1, t, t + 1]})
    return rows


def ff_eval_pairs(inv: dict, dev_ids: list[str]) -> list[dict]:
    dev = set(dev_ids)
    rows = []
    for ds, n in N_TRAINING.items():
        for i in range(n):
            seq = f"{i:06d}"
            have = set(inv.get((ds, "training", "image_2", seq), ()))
            if {9, 10, 11} <= have:
                half = None if ds == "kitti12" else ("dev" if seq in dev else "heldout")
                rows.append({"id": pair_id(ds, "training", "image_2", seq, 10), "ds": ds,
                             "sub": "training", "side": "image_2", "seq": seq, "t": 10,
                             "frames": [9, 10, 11], "diag_only": True, "k15_half": half,
                             "prio": 1 if half == "heldout" else 0})
    rows.sort(key=lambda r: (r["prio"], r["id"]))
    return rows


# ------------------------------------------------------------------------------ calibration
def _read_kv(path: Path) -> dict:
    out = {}
    for line in Path(path).read_text().splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        try:
            out[k.strip()] = [float(x) for x in v.split()]
        except ValueError:
            continue
    return out


def calib_path(data: Path, ds: str, sub: str, seq: str) -> Path:
    if ds == "kitti15":
        return Path(data) / "kitti15" / "calib" / sub / "calib_cam_to_cam" / f"{seq}.txt"
    return Path(data) / "kitti12" / "calib" / sub / "calib" / f"{seq}.txt"


def calib_K(data: Path, ds: str, sub: str, seq: str, side: str):
    """(K 3x3, P 3x4) float64 of the rectified camera that took `side`, at native resolution."""
    import numpy as np

    kv = _read_kv(calib_path(data, ds, sub, seq))
    cam = side[-1]
    key = f"P_rect_0{cam}" if ds == "kitti15" else f"P{cam}"
    if key not in kv or len(kv[key]) != 12:
        raise KeyError(f"{calib_path(data, ds, sub, seq)}: no {key}")
    P = np.array(kv[key], dtype=np.float64).reshape(3, 4)
    return P[:, :3].copy(), P
