"""A tiny synthetic KITTI multi-view tree for the P1 CPU tests and Mac smokes: textured frames,
rectified calibration files in the official layouts, a fake file list and a windows plan; and
an analytic scene (Stub) whose geometry is written in teacher-factory's cache format.

How to use (library only):
    tree = build_tree(tmp, k15=("000000", "000001"), k12=("000000", "000031"), hw=(48, 160))
    tree["data"], tree["files"], tree["plan"]    # data root, manifest dir, windows plan path
    write_stub_teacher_cache(root, rows, data)   # one teacher_cache .npz per window (exact depth)
The stub refuses to run on a GPU host or without U2_P1_STUB=1: a stub can never produce a
cache on a G4 (tools/vm/sam_masks.py's rule).
KITTI-15 000001 is a dev id and 000000 a held-out id of splits/k15_split_v1.json [V 2026-10-03];
KITTI-12 000031 lacks frames 00-04 as in the real data [M data/manifest/files], so its w03-11
window is absent. No GT is written: no P1 tool reads GT.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import common as C
from . import kitti as KT


def k_for(hw: tuple[int, int]) -> list[float]:
    h, w = hw
    f = 700.0 * w / 1242.0
    return [f, 0.0, (w - 1) / 2 + 0.3, 0.0, 0.0, f, (h - 1) / 2 - 0.2, 0.0, 0.0, 0.0, 1.0, 0.0]


def write_png(path: Path, img: np.ndarray) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", img[:, :, ::-1])
    assert ok
    path.write_bytes(buf.tobytes())


def write_label_png(path: Path, labels: np.ndarray) -> None:
    """A single-channel uint8 PNG, the layout of SAM's full_seg maps."""
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", np.asarray(labels, np.uint8))
    assert ok
    path.write_bytes(buf.tobytes())


def build_tree(root: Path, k15=("000000", "000001"), k12=("000000", "000031"),
               hw: tuple[int, int] = (48, 160), frames=range(21), sides=("image_2",),  # noqa: B008
               subs=("training",)) -> dict:
    root = Path(root)
    data, files = root / "data", root / "files"
    files.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    lines = []
    for ds, seqs in (("kitti15", k15), ("kitti12", k12)):
        for sub in subs:
            for seq in seqs:
                P = " ".join(f"{v:e}" for v in k_for(hw))
                cp = KT.calib_path(data, ds, sub, seq)
                cp.parent.mkdir(parents=True, exist_ok=True)
                if ds == "kitti15":
                    cp.write_text(f"calib_time: synthetic\nP_rect_02: {P}\nP_rect_03: {P}\n")
                else:
                    cp.write_text(f"P0: {P}\nP1: {P}\nP2: {P}\nP3: {P}\n")
                for side in sides:
                    for f in frames:
                        if ds == "kitti12" and seq == "000031" and f < 5:
                            continue
                        img = rng.integers(0, 255, (*hw, 3), dtype=np.uint8)
                        p = KT.frame_path(data, ds, sub, side, seq, f)
                        write_png(p, img)
                        rel = p.relative_to(data).as_posix()
                        lines.append(f"{C.sha256_file(p)}  {rel}")
    (files / "u2f-synth-mv-v1.sha256").write_text("\n".join(lines) + "\n")
    plan = root / "windows_v1.json"
    import p1_windows as MW  # noqa: PLC0415  (tools/p1 on sys.path)

    rc = MW.main(["plan", "--files", str(files), "--out", str(plan)])
    assert rc == 0
    return {"data": data, "files": files, "plan": plan, "hw": hw}


# ------------------------------------------------------------------------------ the stub scene
class Stub:
    """An analytic scene: a ground plane, a far wall and a box, seen by a camera moving forward
    0.8 per frame with a slight yaw; frame f is at scene_pose(f - 10), so every stub product
    (teacher geometry, dump_eval's stub evidence, the fake COLMAP) describes one scene with
    exact depth and poses."""

    @staticmethod
    def scene_pose(k: int) -> np.ndarray:
        import rigid

        a = np.radians(0.5 * k)
        R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        return rigid.rt_to_M(R, [0.02 * k, 0.0, 0.8 * k])  # camera-to-world

    @staticmethod
    def scene_depth(K: np.ndarray, T_c2w: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
        """z-depth of the first surface hit: ground y = 1.5, wall z = 40, a box face z = 12 for
        x in [-1, 1], y in [-0.5, 1.5] (world, OpenCV axes)."""
        h, w = hw
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
        d_cam = np.linalg.inv(K) @ np.stack([xs, ys, np.ones_like(xs)]).reshape(3, -1)
        R, c = T_c2w[:3, :3], T_c2w[:3, 3]
        d = R @ d_cam  # world ray directions, z-depth parameter: point = c + s * d
        s = np.full(d.shape[1], np.inf)
        with np.errstate(divide="ignore", invalid="ignore"):
            sg = (1.5 - c[1]) / d[1]
            s = np.where((sg > 0), np.minimum(s, sg), s)
            sw = (40.0 - c[2]) / d[2]
            s = np.where(sw > 0, np.minimum(s, sw), s)
            sb = (12.0 - c[2]) / d[2]
            px, py = c[0] + sb * d[0], c[1] + sb * d[1]
            hit = (sb > 0) & (np.abs(px) <= 1.0) & (py >= -0.5) & (py <= 1.5)
            s = np.where(hit, np.minimum(s, sb), s)
        return s.reshape(h, w)  # d_cam has z = 1, so s is the z-depth


def require_stub_host() -> None:
    import os

    if C.gpu_host() or os.environ.get("U2_P1_STUB") != "1":
        raise C.Stop(C.EXIT_REFUSED, "the stub runs only in CPU tests (U2_P1_STUB=1, no "
                                     "nvidia-smi)")


def write_stub_teacher_cache(root: Path, rows: list[dict], data: Path,
                             model_hw: tuple[int, int] = (40, 128), teacher: str = "stub") -> int:
    """Each window as tools/teacher/run_geometry.py would write it (teacher_cache.write
    validates it), with the stub scene's exact depth and poses at model_hw."""
    import rigid

    from .geomio import ROLE_WIN, teacher_cache_module

    require_stub_host()
    tc = teacher_cache_module()
    role_of = {v: k for k, v in ROLE_WIN.items()}
    n = 0
    for r in rows:
        h0, w0 = C.png_size(KT.frame_path(data, r["ds"], r["sub"], r["side"], r["seq"],
                                          r["frames"][0]))
        K, _ = KT.calib_K(data, r["ds"], r["sub"], r["seq"], r["side"])
        Km = rigid.zoom_K(K, (h0, w0), model_hw)
        poses = [Stub.scene_pose(f - 10) for f in r["frames"]]
        depth = np.stack([Stub.scene_depth(Km, T, model_hw) for T in poses])
        conf = 1.0 / (1.0 + np.nan_to_num(depth, posinf=1e6))
        arrays = tc.geometry_arrays(depth, conf, None, poses, [Km] * len(poses),
                                    [Km] * len(poses), r["frames"], [[h0, w0]] * len(poses))
        fr = r["frames"]
        wid = f"{r['ds']}/{r['sub']}/{r['side']}/{r['seq']}_{fr[0]:02d}_{fr[-1]:02d}"
        meta = {"schema": tc.SCHEMA, "kind": "geometry", "teacher": teacher, "window": wid,
                "role": role_of[r["win"]], "grid": "model",
                "model_wh": [model_hw[1], model_hw[0]], "scale": "metric",
                "k_source": "calibration", "k15_half": r.get("k15_half"),
                "stamp_sha256": "stub", "producer": {"gpu_name": None, "device": "stub"}}
        tc.write(Path(root) / f"{wid}.npz", arrays, meta)
        n += 1
    return n
