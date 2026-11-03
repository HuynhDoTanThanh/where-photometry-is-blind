#!/usr/bin/env python3
"""RigidFill on U2Flow's KITTI checkpoint, for one frame triple (t-1, t, t+1).

U2Flow (Sun et al., CVPR 2026) is not part of this repository: clone it and download its
released KITTI checkpoint first (https://github.com/sunzunyi/U2FLOW). This script imports the
network from your checkout and runs it as in the paper: frames resized to the 256 x 832 grid,
two bidirectional passes, on (t, t+1) and on (t-1, t), with U2Flow's uncertainty head and its own
forward-backward occlusion test; S is U2Flow's own upsampling of the forward flow.

    python examples/run_u2flow.py --u2flow ~/code/U2FLOW \
        --ckpt ~/code/U2FLOW/checkpoints/kitti.pth \
        --frames 000163_09.png 000163_10.png 000163_11.png --calib calib/000163.txt \
        [--gt flow_occ/000163_10.png] [--out outputs/u2flow] [--device cuda]

For KITTI 2012 the frames are training/image_2/<id>_09.png, _10.png, _11.png of the multi-view
set and the calibration is training/calib/<id>.txt; for KITTI 2015, image_2/<id>_09..11.png and
calib_cam_to_cam/<id>.txt (camera 2). Writes the corrected flow and U2Flow's flow as KITTI flow
PNGs and a figure; with --gt it prints Fl (KITTI 2015 outlier rule) and EPE of both.

Needs U2Flow's own dependencies (torch, torchvision, yacs, einops, matplotlib, scipy) besides
rigidfill. On a CPU the network runs in float32; the paper's GPU runs used U2Flow's fp16
autocast. RANSAC makes single images sensitive to that small difference (see README.md).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from a checkout too

import rigidfill  # noqa: E402
from rigidfill.metrics import endpoint_error, outliers  # noqa: E402
from rigidfill.viz import flow_to_color, outlier_overlay, overlay_mask, tile  # noqa: E402

GRID = (256, 832)  # U2Flow's KITTI input size (configs/KITTI_MV.py image_size)


def load_u2flow(root: Path, ckpt: Path, device):
    """U2Flow's network as its evaluation script builds it (KITTI_MV config, 12 iterations)."""
    import torch

    if not (root / "core" / "Networks" / "raft_2f_u.py").is_file():
        raise SystemExit(f"{root}: not a U2Flow checkout (core/Networks/raft_2f_u.py missing)")
    sys.path[:0] = [str(root), str(root / "core")]  # U2Flow imports both ways
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


def load_frame(path: Path):
    """U2Flow's test-time transform: RGB / 255, cv2.resize to the grid, CHW float tensor."""
    import cv2
    import torch

    img = rigidfill.read_image(path).astype(np.float32) / 255.0
    img = cv2.resize(img, (GRID[1], GRID[0]))
    return torch.from_numpy(img.transpose(2, 0, 1).copy()).float()


def bidirectional(model, a, b, device) -> dict:
    """One U2Flow pass on (a, b): both flows, both log-variances, both visibility masks."""
    import torch
    from core.utils.warp_utils import get_occu_mask_bidirection

    with torch.no_grad():
        o = model(torch.stack([a, b])[None].to(device))
        f12, b21 = o["flows_f12"][-1], o["flows_b21"][-1]
        vis12 = 1 - get_occu_mask_bidirection(f12, b21)
        vis21 = 1 - get_occu_mask_bidirection(b21, f12)

    def np_(x):
        return x[0].float().cpu().numpy()

    return {"f12": f12, "flow_f": np_(f12), "flow_b": np_(b21),
            "alpha_f": np_(o["uncertainty_f12"][-1])[0],
            "alpha_b": np_(o["uncertainty_b21"][-1])[0],
            "vis_f": np_(vis12)[0] > 0.5, "vis_b": np_(vis21)[0] > 0.5}


def scores(flow, gt, valid) -> str:
    n_out = int(outliers(flow, gt, valid).sum())
    return (f"Fl {100 * n_out / valid.sum():6.3f}% ({n_out:,} outliers)   "
            f"EPE {endpoint_error(flow, gt)[valid].mean():6.3f} px")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--u2flow", required=True, type=Path, help="U2Flow checkout")
    ap.add_argument("--ckpt", required=True, type=Path, help="U2Flow's kitti.pth")
    ap.add_argument("--frames", required=True, nargs=3, type=Path, metavar=("T-1", "T", "T+1"))
    ap.add_argument("--calib", required=True, type=Path, help="KITTI calibration file")
    ap.add_argument("--camera", type=int, default=2, help="rectified camera index (2 = left)")
    ap.add_argument("--gt", type=Path, help="optional KITTI flow PNG of frame t (flow_occ)")
    ap.add_argument("--out", type=Path, default=Path("outputs/u2flow"))
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    a = ap.parse_args()

    import warnings

    import torch

    # U2Flow's network asks for CUDA autocast, which PyTorch deprecates and disables on a CPU
    warnings.filterwarnings("ignore", message=".*autocast.*")
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_u2flow(a.u2flow.expanduser(), a.ckpt.expanduser(), device)
    from core.utils.flow_utils import resize_flow

    prev, cur, nxt = (load_frame(p) for p in a.frames)
    H, W = rigidfill.read_image(a.frames[1]).shape[:2]
    K = rigidfill.read_kitti_intrinsics(a.calib, a.camera)

    t0 = time.perf_counter()
    fw = bidirectional(model, cur, nxt, device)   # forward pass (t, t+1)
    bw = bidirectional(model, prev, cur, device)  # backward pass (t-1, t)
    with torch.no_grad():
        S = resize_flow(fw["f12"], (H, W))[0].float().cpu().numpy()  # U2Flow's output
    t1 = time.perf_counter()
    out, info = rigidfill.correct_flow(
        fw["flow_f"], bw["flow_b"], fw["alpha_f"], K, (H, W),
        vis_fw=fw["vis_f"], vis_bw=bw["vis_b"], alpha_bw=bw["alpha_b"], base_flow=S,
        gate="u2flow")
    t2 = time.perf_counter()

    print(f"U2Flow, two bidirectional passes: {1000 * (t1 - t0):.0f} ms on {device}")
    print(f"RigidFill: {1000 * (t2 - t1):.0f} ms on the CPU; "
          + (f"replaced {info.n_replaced:,} pixels ({100 * info.replaced_fraction:.2f}%)"
             if info.ok else f"no rigid flow ({info.failure}); output = U2Flow's flow"))
    if info.ok:
        g = info.geometry
        print(f"  RANSAC inliers {g.n_inliers_bw:,} (t->t-1), {g.n_inliers_fw:,} (t->t+1); "
              f"baseline ratio lambda {g.scale:.3f}")

    name = a.frames[1].stem
    a.out.mkdir(parents=True, exist_ok=True)
    rigidfill.write_kitti_flow(a.out / f"{name}_rigidfill.png", out)
    rigidfill.write_kitti_flow(a.out / f"{name}_u2flow.png", S)
    rgb = rigidfill.read_image(a.frames[1])
    vmax = float(np.percentile(np.hypot(*S), 99))
    if a.gt:
        gt, valid = rigidfill.read_kitti_flow(a.gt)
        print(f"  U2Flow            {scores(S, gt, valid)}")
        print(f"  U2Flow+RigidFill  {scores(out, gt, valid)}")
        second = flow_to_color(np.where(valid[None], gt, np.nan), vmax)
        rows = "frame t, replaced pixels in green | ground truth; U2Flow | RigidFill; " \
               "outliers of U2Flow | outliers of RigidFill, in red"
    else:
        second = flow_to_color(np.where(info.gate[None], out, np.nan), vmax)
        rows = "frame t, replaced pixels in green | the replaced flow; U2Flow | RigidFill"
    panels = [overlay_mask(rgb, info.gate), second, flow_to_color(S, vmax),
              flow_to_color(out, vmax)]
    if a.gt:
        panels += [outlier_overlay(rgb, outliers(S, gt, valid), valid),
                   outlier_overlay(rgb, outliers(out, gt, valid), valid)]
    rigidfill.write_image(a.out / f"{name}_rigidfill_vis.jpg", tile(panels, cols=2))
    print(f"wrote {a.out / (name + '_rigidfill.png')} and {name}_u2flow.png (KITTI flow PNGs), "
          f"{name}_rigidfill_vis.jpg ({rows})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
