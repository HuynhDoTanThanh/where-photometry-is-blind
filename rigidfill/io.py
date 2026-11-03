"""Reading and writing KITTI files: 16-bit flow PNGs, calibration files and images.

KITTI flow PNGs store ``u`` and ``v`` as ``64 * value + 2**15`` in the red and green channels
of a 16-bit PNG and a validity flag in the blue channel (KITTI 2012 and 2015 devkits).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np

PathLike = Union[str, Path]


def read_kitti_flow(path: PathLike) -> Tuple[np.ndarray, np.ndarray]:
    """A KITTI flow PNG as ``(flow, valid)``: float32 ``(2, H, W)`` and bool ``(H, W)``.

    Invalid pixels have flow 0.
    """
    import cv2

    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None or raw.ndim != 3 or raw.shape[2] != 3 or raw.dtype != np.uint16:
        raise ValueError(f"{path}: not a 16-bit, 3-channel KITTI flow PNG")
    bgr = raw.astype(np.float32)
    valid = raw[:, :, 0] > 0
    flow = (np.stack([bgr[:, :, 2], bgr[:, :, 1]], 0) - 2.0**15) / 64.0
    flow[:, ~valid] = 0.0
    return flow.astype(np.float32), valid


def write_kitti_flow(path: PathLike, flow, valid=None) -> None:
    """Write a ``(2, H, W)`` flow as a KITTI flow PNG.

    Values are rounded to the nearest 1/64 px and clipped to the format's range (about
    +-512 px). ``valid`` defaults to the pixels where both components are finite.
    """
    import cv2

    f = np.asarray(flow, dtype=np.float64)
    if f.ndim != 3 or f.shape[0] != 2:
        raise ValueError(f"flow must be (2, H, W), got {f.shape}")
    ok = np.isfinite(f).all(axis=0)
    v = ok if valid is None else (np.asarray(valid, dtype=bool) & ok)
    f = np.where(ok[None], f, 0.0)
    enc = np.clip(np.rint(f * 64.0 + 2.0**15), 0, 65535).astype(np.uint16)
    enc[:, ~v] = 2**15
    img = np.stack([v.astype(np.uint16), enc[1], enc[0]], -1)  # B = valid, G = v, R = u
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), img):
        raise OSError(f"{path}: could not write the PNG")


def read_calib_file(path: PathLike) -> Dict[str, List[float]]:
    """``key: numbers`` lines of a KITTI calibration file (non-numeric lines are skipped)."""
    out: Dict[str, List[float]] = {}
    for line in Path(path).read_text().splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        try:
            out[key.strip()] = [float(x) for x in val.split()]
        except ValueError:
            continue
    return out


def read_kitti_intrinsics(path: PathLike, camera: int = 2) -> np.ndarray:
    """``(3, 3)`` intrinsics K of a rectified KITTI camera at native resolution.

    K is the left 3 x 3 block of the rectified projection matrix: ``P_rect_0<camera>`` in
    KITTI 2015 ``calib_cam_to_cam`` and raw-data files, ``P<camera>`` in KITTI 2012 ``calib``
    files. Camera 2 is the left colour camera (``image_2``).
    """
    kv = read_calib_file(path)
    for key in (f"P_rect_0{camera}", f"P_rect_{camera:02d}", f"P{camera}"):
        if key in kv and len(kv[key]) == 12:
            return np.asarray(kv[key], dtype=np.float64).reshape(3, 4)[:, :3].copy()
    raise KeyError(f"{path}: no 3x4 projection matrix for camera {camera} "
                   f"(looked for P_rect_0{camera} and P{camera})")


def read_image(path: PathLike) -> np.ndarray:
    """An image file as RGB uint8 ``(H, W, 3)``."""
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"{path}: cannot read the image")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def image_size(path: PathLike) -> Tuple[int, int]:
    """``(H, W)`` of an image file."""
    return tuple(read_image(path).shape[:2])  # type: ignore[return-value]


def write_image(path: PathLike, rgb) -> None:
    """Write an RGB uint8 image (the format follows the file extension)."""
    import cv2

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(np.asarray(rgb, np.uint8), cv2.COLOR_RGB2BGR)):
        raise OSError(f"{path}: could not write the image")


__all__ = ["read_kitti_flow", "write_kitti_flow", "read_calib_file", "read_kitti_intrinsics",
           "read_image", "image_size", "write_image"]
