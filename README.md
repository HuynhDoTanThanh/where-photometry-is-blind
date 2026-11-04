# Where Photometry Is Blind: RigidFill

**Inference-time geometric correction of unsupervised optical flow in driving scenes**

Authors to be announced.

[Project page](https://huynhdotanthanh.github.io/where-photometry-is-blind/) &nbsp;|&nbsp; Paper: coming soon &nbsp;|&nbsp; [MIT licence](LICENSE)

![Frame t of KITTI-12 image 000163 with the labelled pixels that leave frame t+1 in orange; frame t+1; outlier pixels of U²Flow and of U²Flow with RigidFill in red](docs/assets/img/teaser.jpg)

Unsupervised optical flow learns from photometric consistency, so a pixel that leaves the next
frame gets no direct training signal, and such out-of-frame pixels carry a large share of the
errors of unsupervised networks. **RigidFill** corrects a frozen network at inference time,
without retraining: it estimates the camera motion from the network's own reliable
correspondences, triangulates the backward correspondences (frame t to t−1) to get the depth of
frame t, reprojects every pixel into camera t+1, and replaces the network's flow with this
rigid flow where the network is uncertain. It needs three consecutive frames and the camera
intrinsics. This repository is a small, pip-installable Python package (NumPy and OpenCV only)
with examples and tests.

## Installation

```bash
git clone https://github.com/HuynhDoTanThanh/where-photometry-is-blind.git
cd where-photometry-is-blind
pip install -e .                 # Python >= 3.9; installs numpy and opencv-python
pip install -e ".[test]" && pytest   # optional: run the tests (CPU, a few seconds)
```

RigidFill itself needs no GPU and no deep-learning framework. Running a flow network (for example
U²Flow, below) needs that network's own dependencies.

## Quick start

Try it without any download on a synthetic driving scene, where a simulated network
under-estimates the flow of the pixels that leave the frame:

```bash
python examples/synthetic_demo.py
```

```
RigidFill: ok in 1977 ms (CPU)
  forward motion |t| = 1.117 backward baselines (true 1.111); RANSAC inliers 190,503 (backward), 156,242 (forward)
  replaced 90,346 of 465,750 pixels (19.4%); 91,031 pixels leave the frame
                             network  RigidFill
  out-of-frame EPE (px)       28.722      1.459
  in-frame EPE (px)            0.163      0.156
  all pixels EPE (px)          5.745      0.411
```

With your own network's outputs:

```python
import rigidfill

# Outputs of the network on its input grid (h, w), e.g. 256 x 832 for U2Flow on KITTI.
# Pass 1 on frames (t, t+1): flow_fw (2, h, w), its uncertainty alpha_fw (h, w), visibility vis_fw
# Pass 2 on frames (t-1, t): flow_bw (2, h, w) = the flow t -> t-1, alpha_bw, vis_bw
K = rigidfill.read_kitti_intrinsics("calib/000163.txt")          # 3x3 at native resolution
out, info = rigidfill.correct_flow(flow_fw, flow_bw, alpha_fw, K, image_size=(375, 1242),
                                   vis_fw=vis_fw, vis_bw=vis_bw, alpha_bw=alpha_bw)
rigidfill.write_kitti_flow("000163_10.png", out)                 # (2, 375, 1242) float32
print(info.ok, info.n_replaced, info.failure)
```

`out` is the corrected forward flow at native resolution. `info` holds the replaced pixels
(`info.gate`), the rigid flow (`info.rigid_flow`), the network's upsampled flow
(`info.base_flow`), the camera motion and depth (`info.geometry`), and, when no rigid flow
can be computed (for example a stationary camera), `info.ok = False` with the reason, in which
case `out` is the network's flow.

Pass `threads=` (for example the number of CPU cores) to run the geometry in parallel; the
output is byte-identical to the default of one thread (see Runtime below).

## What RigidFill needs

| Input | Shape | Notes |
|---|---|---|
| `flow_fw` | (2, h, w) | forward flow t → t+1 on the network grid: the flow to correct |
| `flow_bw` | (2, h, w) | backward flow t → t−1, from a second pass on frames (t−1, t) |
| `alpha_fw`, `alpha_bw` | (h, w) | uncertainty of each flow: U²Flow's log-variance, or `rigidfill.fb_score` for networks without an uncertainty head |
| `vis_fw`, `vis_bw` | (h, w) bool | forward-backward visibility of each flow (`rigidfill.fb_consistency`) |
| `K` | (3, 3) | intrinsics of the native image; rescaled to the grid internally |
| `image_size` | (H, W) | native size of the output; omit if the flows are already native |
| `base_flow` | (2, H, W) | optional: the network's own native-resolution output; default: `rigidfill.upsample_flow(flow_fw, image_size)` |

All flows are `(u, v)` in pixels, channel first. Pixel (x, y) is column x and row y, and
integer coordinates are pixel centres.

## Using it with U²Flow

[U²Flow](https://github.com/sunzunyi/U2FLOW) (Sun et al., CVPR 2026) is not included here (its
repository has no licence). Clone it, download its released KITTI checkpoint (`kitti.pth`),
install its dependencies, and run:

```bash
python examples/run_u2flow.py --u2flow /path/to/U2FLOW --ckpt /path/to/U2FLOW/checkpoints/kitti.pth \
    --frames training/image_2/000163_09.png training/image_2/000163_10.png training/image_2/000163_11.png \
    --calib training/calib/000163.txt --gt training/flow_occ/000163_10.png --out outputs/u2flow
```

The script runs U²Flow as in the paper: frames resized to its 256 × 832 grid, two bidirectional
passes, on (t, t+1) and on (t−1, t), with its uncertainty head and its own forward-backward
occlusion test. It writes the corrected flow and U²Flow's flow as KITTI flow PNGs and a figure,
and with `--gt` prints Fl and EPE. On KITTI 2012 image 000163 (the teaser above), on a CPU:

```
U2Flow, two bidirectional passes: 3825 ms on cpu
RigidFill: 1737 ms on the CPU; replaced 28,004 pixels (6.01%)
  RANSAC inliers 179,866 (t->t-1), 138,353 (t->t+1); baseline ratio lambda 1.003
  U2Flow            Fl 10.054% (13,553 outliers)   EPE  1.906 px
  U2Flow+RigidFill  Fl  7.371% (9,936 outliers)   EPE  1.485 px
```

On a CPU the network runs in float32, while the paper's GPU runs used U²Flow's half-precision
autocast (which U²Flow's code switches on by itself on a CUDA GPU). The flows differ by about
0.001 px, but RANSAC can settle on a different pose after such tiny changes, so a single image
can change a lot: with the paper's GPU outputs, image 000163 has 7,027 outliers after RigidFill
instead of 9,936. Over the 194 KITTI 2012 images, this CPU pipeline lowers Fl from 4.129% to
3.716% (RigidFill improves 146 images and worsens 31), against 4.130% to 3.613% with the
paper's GPU outputs.

## Using it with other networks

For a network without an uncertainty head, use the squared forward-backward residual
`r` as the score, `alpha = ln(r + 1e-3)`, and the matching gate:

```python
# pass 1 on (t, t+1) gives flow_fw (t -> t+1) and flow_fw_rev (t+1 -> t)
# pass 2 on (t-1, t) gives flow_bw (t -> t-1) and flow_bw_rev (t-1 -> t)
_, vis_fw = rigidfill.fb_consistency(flow_fw, flow_fw_rev)
_, vis_bw = rigidfill.fb_consistency(flow_bw, flow_bw_rev)
alpha_fw = rigidfill.fb_score(flow_fw, flow_fw_rev)
alpha_bw = rigidfill.fb_score(flow_bw, flow_bw_rev)
out, info = rigidfill.correct_flow(flow_fw, flow_bw, alpha_fw, K, image_size,
                                   vis_fw=vis_fw, vis_bw=vis_bw, alpha_bw=alpha_bw,
                                   gate="arflow")      # or "sunraft", or GateParams(...)
```

The `"arflow"` and `"sunraft"` presets are the gates selected for those two networks on the
KITTI 2015 development half (their flows taken on the 256 × 832 grid). For another network,
select `tau`, `d_max` and `d_min` on labelled images of your own. Where a correspondence leaves
the frame the residual reduces to the squared flow magnitude, so this score then acts as a
motion threshold rather than a confidence.

## Parameters

| Quantity | Argument | Paper value |
|---|---|---|
| Network grid | shape of the flows | 256 × 832 (U²Flow on KITTI) |
| Pose cutoff on the uncertainty | `alpha_max` | α ≤ ln 2 (σ² ≤ 2) |
| Essential matrix (`cv2.findEssentialMat`, five-point RANSAC) | `ransac_threshold`, `ransac_confidence` | 1 grid px, 0.999, OpenCV's default iteration cap |
| Pose recovery (`cv2.recoverPose`) | – | OpenCV's default depth limit, 50 baselines |
| Minimum support | – | 8 correspondences, 8 inliers in front of both cameras, 8 pixels with both depths |
| Gate: uncertainty threshold, `alpha > tau` | `tau` | 2.75 (U²Flow, σ² > 15.6), 1.0 (ARFlow), 1.5 (Sun-RAFT) |
| Gate: upper bound, `‖T − S‖ ≤ d_max` (native px) | `d_max` | 100 (U²Flow, ARFlow), none (Sun-RAFT) |
| Gate: lower bound, `‖T − S‖ > d_min` (native px) | `d_min` | none (U²Flow), 6 (ARFlow), 8 (Sun-RAFT) |
| Visibility test | `rigidfill.fb_consistency` | ‖F + w‖² ≤ 0.01 (‖F‖² + ‖w‖²) + 0.5 grid px² |
| Depth storage before rendering | `half_precision_depth` | float16 inverse depth (`True`) |
| CPU threads for the geometry | `threads` | 24 in the paper's timing run (the output does not depend on it) |

Only `tau` and `d_max` (and the other networks' gates) were tuned, on the 100 KITTI 2015
development images; the other constants were never tuned. Presets: `rigidfill.PRESETS`.

### How it works

1. **Camera motion.** For each direction d ∈ {backward, forward}, the visible pixels whose
   correspondence stays in the frame and whose uncertainty is at most ln 2 fit an essential
   matrix (`rigidfill.estimate_pose`); its decomposition gives a rotation and a unit
   translation.
2. **Depth.** Every visible, in-frame correspondence is triangulated (`rigidfill.triangulate`);
   a depth is kept if the point is in front of both cameras. The scale λ is the median ratio of
   backward to forward depth over pixels with both (`rigidfill.fit_geometry`).
3. **Rigid flow.** Each pixel with a backward depth D_b is reprojected into camera t+1:
   T(p) = π(K [R_f D_b(p) K⁻¹ p̃ + λ t_f]) − p (`rigidfill.rigid_flow`). This works for points
   that leave the image, which have no forward match but were visible in frame t−1.
4. **Gate.** T, α and the network's flow S are brought to native resolution with U²Flow's
   bilinear upsampling (`rigidfill.upsample_flow`, `upsample_score`), and the output is T where
   T is defined, α > τ and d_min < ‖T − S‖ ≤ d_max, else S (`rigidfill.uncertainty_gate`).

The whole image keeps the network's flow if a pose or the scale cannot be estimated.

## Results (from the paper)

Fl is the KITTI outlier rate (%): the share of labelled pixels whose flow error exceeds both
3 px and 5% of the true flow magnitude. Gain = base Fl minus corrected Fl, in percentage
points (pp), with its 95% confidence interval.

| Evaluation | Fl, U²Flow | Fl, U²Flow + RigidFill | Gain (pp) |
|---|---:|---:|---|
| KITTI 2015, held-out half (100 images) | 7.384 | **6.949** | 0.434 [0.259, 0.633] |
| KITTI 2012 (194 images) | 4.130 | **3.613** | 0.517 |

On KITTI 2012, RigidFill also improves ARFlow by 0.99 pp and Sun-RAFT by 0.66 pp in Fl, each
with its own gate.

The held-out half is the second half of `data/kitti15_split.json`; RigidFill's thresholds were
selected on the first (development) half.

**Runtime.** Median per KITTI image in the paper, with all network passes included, on one
NVIDIA RTX PRO 6000 Blackwell GPU and its host CPU: U²Flow alone 22.2 ms, U²Flow with RigidFill
212.7 ms (its geometry on 24 CPU threads, outputs bit-identical to one thread on all 194 KITTI
2012 images), and +FF 2,607.0 ms. RigidFill thus takes 0.08 times the time of +FF and 9.6 times
that of the network. This package runs the same geometry on the CPU with the `threads` option.
On an Apple M4 (10 cores; OpenCV 5.0.0, NumPy 2.5.3), the median time per KITTI image (network
passes excluded, 194 KITTI 2012 images) is 1.83 s with one thread, 0.54 s with 4 and 0.35 s
with 10, and the outputs of 4 and 10 threads are byte-identical to one thread's on all 194
images. With one thread, the time goes about half to the per-pixel triangulation and half to
OpenCV's pose recovery. It suits offline or accuracy-first use, not real-time inference.

**Limitations.** No measured gain on moving objects: RigidFill assumes a static scene, and the
depths of independently moving points are wrong. On the non-rigid scenes of MPI Sintel the
unchanged gate increases the error. It needs three consecutive frames and calibrated
intrinsics, and its thresholds were selected on 100 labelled KITTI 2015 images.

## Agreement with the paper's implementation

We checked this package against the implementation behind the paper's numbers on the 194
KITTI 2012 images, feeding it the paper's stored U²Flow outputs: it reproduces the RANSAC
inlier counts, poses and scales of all 193 images that have a rigid flow (and the failure of
the 194th, image 000180), replaces the same 4,668,075 pixels, gives a rigid flow at native
resolution within 0.001 px of the stored one at all but 19 pixels (none of them replaced; at
most 0.063 px), and reaches Fl 3.613016% against the paper's 3.613020% (U²Flow: 4.130272%), a
difference of one pixel whose error lies on the 5% outlier threshold. The stored rigid flow was
rendered from single-precision intrinsics and poses, which explains the residual differences.
The results were the same with OpenCV 5.0.0 (Python 3.12 and 3.9) and OpenCV 4.14.0 (Python 3.9).

## Repository layout

```
rigidfill/                  the package
  pipeline.py               correct_flow: RigidFill end to end
  geometry.py               correspondences, estimate_pose, triangulate, fit_geometry, rigid_flow
  gate.py                   uncertainty_gate, GateParams, PRESETS
  resize.py                 upsample_flow, upsample_score (U²Flow's upsampling in NumPy)
  consistency.py            fb_consistency, fb_score (networks without an uncertainty head)
  io.py, metrics.py, viz.py KITTI flow PNGs and calibration files, Fl and EPE, colour coding
  synthetic.py              the synthetic scene of the demo and the tests
examples/                   synthetic_demo.py, run_u2flow.py
tests/                      pytest suite (CPU)
data/kitti15_split.json     the development and held-out halves of KITTI 2015 training
docs/                       the project page (GitHub Pages)
```

## Citation

```bibtex
@misc{rigidfill2026,
  title        = {Where Photometry Is Blind: Inference-Time Geometric Correction
                  of Unsupervised Optical Flow in Driving Scenes},
  author       = {Authors to be announced},
  year         = {2026},
  howpublished = {\url{https://huynhdotanthanh.github.io/where-photometry-is-blind/}}
}
```

## Licence

The code is released under the [MIT licence](LICENSE). U²Flow's code and checkpoints are not
part of this repository; they come from [github.com/sunzunyi/U2FLOW](https://github.com/sunzunyi/U2FLOW),
which ships no licence. Images in `docs/assets/img/` derived from the KITTI Vision Benchmark
Suite are shared under CC BY-NC-SA 3.0, as KITTI is. The fonts in `docs/assets/fonts/`
(Barlow, Source Serif 4) are under the SIL Open Font License 1.1.

## Use of AI tools

Coding agents based on large language models (Anthropic's Claude through Claude Code, and
OpenAI's Codex) were used in this work to write and review code, including this package, and
to run experiments; Claude also drafted and edited text, including this README. The authors
reviewed the results and take responsibility for them; the paper's declaration gives the
details.

## Acknowledgements

We thank the authors of the KITTI benchmarks (Geiger, Lenz and Urtasun, CVPR 2012; Menze and
Geiger, CVPR 2015), MPI Sintel (Butler et al., ECCV 2012) and HD1K (Kondermann et al., CVPR
Workshops 2016) for their datasets, and the authors of
[U²Flow](https://github.com/sunzunyi/U2FLOW) (Sun et al., CVPR 2026),
[ARFlow](https://github.com/lliuz/ARFlow) (Liu et al., CVPR 2020) and
[Sun-RAFT](https://github.com/cv-stuttgart/Sun-RAFT) (Jahedi et al., WACV 2026) for releasing
their code and checkpoints.
