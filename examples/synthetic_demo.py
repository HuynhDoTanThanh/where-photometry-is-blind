#!/usr/bin/env python3
"""RigidFill on a synthetic driving scene: no downloads, no network, CPU only.

A camera drives forward down a corridor. A simulated flow network gets every pixel right except
those that leave the next frame, where it predicts only 60% of the motion and reports a high
uncertainty (the failure of photometric training that RigidFill targets). RigidFill recovers the
camera motion and the depth from the network's own forward and backward flows and replaces the
flow of those pixels by the rigid flow.

    python examples/synthetic_demo.py [--out outputs/synthetic_demo]

Prints the error before and after, and writes a figure (network flow, RigidFill output, replaced
pixels, errors) and the corrected flow as a KITTI flow PNG.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from a checkout too

import rigidfill  # noqa: E402
from rigidfill.synthetic import make_scene  # noqa: E402
from rigidfill.viz import error_map, flow_to_color, overlay_mask, tile  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="outputs/synthetic_demo", help="output directory")
    ap.add_argument("--noise", type=float, default=0.1,
                    help="Gaussian noise (grid px) on the simulated flows")
    a = ap.parse_args()

    scene = make_scene(image_size=(375, 1242), grid=(256, 832), noise_px=a.noise)
    t0 = time.perf_counter()
    out, info = rigidfill.correct_flow(
        scene.flow_fw, scene.flow_bw, scene.alpha_fw, scene.K, scene.image_size,
        vis_fw=scene.vis_fw, vis_bw=scene.vis_bw, alpha_bw=scene.alpha_bw)
    dt = time.perf_counter() - t0

    truth, oof = scene.flow_fw_true_native, scene.oof_native
    e_net = np.hypot(*(info.base_flow - truth))
    e_out = np.hypot(*(out - truth))
    g = info.geometry
    print(f"RigidFill: {'ok' if info.ok else info.failure} in {1000 * dt:.0f} ms (CPU)")
    print(f"  forward motion |t| = {np.linalg.norm(g.t_fw):.3f} backward baselines "
          f"(true {scene.true_scale:.3f}); RANSAC inliers {g.n_inliers_bw:,} (backward), "
          f"{g.n_inliers_fw:,} (forward)")
    print(f"  replaced {info.n_replaced:,} of {oof.size:,} pixels "
          f"({100 * info.replaced_fraction:.1f}%); {int(oof.sum()):,} pixels leave the frame")
    print(f"  {'':24s}{'network':>10s}{'RigidFill':>11s}")
    for name, m in (("out-of-frame EPE (px)", oof), ("in-frame EPE (px)", ~oof),
                    ("all pixels EPE (px)", np.ones_like(oof))):
        print(f"  {name:24s}{e_net[m].mean():10.3f}{e_out[m].mean():11.3f}")

    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    vmax = float(np.nanpercentile(np.hypot(*truth), 99))
    panels = [flow_to_color(info.base_flow, vmax), flow_to_color(out, vmax),
              overlay_mask(flow_to_color(truth, vmax), info.gate),
              flow_to_color(truth, vmax),
              error_map(info.base_flow, truth, max_err=5.0), error_map(out, truth, max_err=5.0)]
    rigidfill.write_image(od / "synthetic_demo.png", tile(panels, cols=2))
    rigidfill.write_kitti_flow(od / "flow_rigidfill.png", out)
    print(f"wrote {od / 'synthetic_demo.png'} (rows: network | RigidFill; replaced pixels in "
          f"green | ground truth; error of the network | error of RigidFill, red = 5 px or more)")
    print(f"wrote {od / 'flow_rigidfill.png'} (KITTI flow PNG)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
