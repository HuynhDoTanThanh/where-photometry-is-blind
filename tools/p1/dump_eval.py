#!/usr/bin/env python3
"""dump_eval.py: the evidence of a flow network around frame 10 of every evaluation sequence
(plan rows W2-08, W2-09): U2Flow kitti.pth's flows, alphas and visibility for the pairs 10 <-> 11
and 09 <-> 10, or RAFT C+T's flows, as a resumable, sharded cache (a launch.sh job).

What it does. Items are the 394 sequences of the 'ff-eval' list (KITTI-15 ids 0-199 and KITTI-12
0-193, image_2, frames 09/10/11; tools/p1/specs/windows_v1.json). Per item:
    u2flow  flow_<d> fp16 (2,256,832), alpha_<d> fp16 (256,832) (log variance), vis_<d>
            packbits, for d in 10to11, 11to10, 09to10, 10to09 (two forwards: (10, 11), (09, 10));
            native_10to11 float32 (2,H0,W0) after resize_flow, for prio-0 items only (KITTI-15
            dev and KITTI-12), the prediction W1-07 made for the same pair
    raftct  flow_<d> fp16 on the 256 x 832 grid (carried from native by eq. 2, rigid.carry_flow)
            and native_10to11 float32 for prio-0 items; RAFT's KITTI protocol: native
            resolution, InputPadder(mode='kitti'), 24 iterations, test_mode [V princeton-vl/RAFT
            @2888e15 evaluate.py:131-146, core/utils/utils.py:7-24, fetched 2026-10-03]
    arflow  PX-S1: ARFlow pwclite_ar.tar on the 256 x 832 grid; alpha = ln(fb error + 1e-3)
    sunraft PX-J5: Sun-RAFT_kitti-test.pth by the authors' KITTI protocol (native, 12 iterations),
            carried to the grid by eq. 2; alpha and vis formed on the grid as for ARFlow
    stub    analytic flows of the stub scene (CPU tests only)
Monodepth2 is a geometry teacher: it runs through tools/teacher/run_geometry.py (teacher-factory),
so its depth and pose reach the same renderer as every other source.
Cache: OUT/cache/diag/<u2flow|raftct|arflow|sunraft|stubflow>/shard_###.tar + manifest.json
(p1lib/cache.py).

How to invoke.
    Launcher (tools/p1/specs/dump_eval_<source>.spec.json; job infer):
        python -u tools/p1/dump_eval.py run --source u2flow --data {data} --out-dir {out_dir} \
            --ckpt {data}/weights/u2flow/kitti.pth [--shard k/n] [--max-prio 0]
        python -u tools/p1/dump_eval.py run --source arflow|sunraft --data {data} \
            --out-dir {out_dir} --deps-lock env/teachers/<arflow|sunraft>.txt --max-prio 0
            (PX-S1 / PX-J5; specs dump_eval_arflow / dump_eval_sunraft)
        python -u tools/p1/dump_eval.py run --source raftct --data {data} --out-dir {out_dir} \
            --ckpt {data}/weights/raft/raft-things.pth --deps-lock env/teachers/raft.txt
            (or --code-dir <RAFT checkout or teacher-factory install dir, glob ok>)
    Mac: python tools/p1/dump_eval.py compare --cache DIR --preds RUNDIR/preds [--tol 1e-5]
        (U2Flow's native_10to11 against an eval_kitti.py prediction dir, id by id)
Output lines: as p1lib/jobrun.py, plus DUMP_COMPARE {json}.
Exit codes: 0 ok · 1 items left or failed; compare: a difference above --tol · 2 precondition ·
3 refused (checkpoint pin, stamp, list) · 50 seal · 64 usage · 143 TERM.

Why it is built this way.
- The U2Flow path reuses the authors' loader, network and occlusion function through
  p1lib/u2flow.py, so its 10 -> 11 flow is W1-07's prediction for that pair (compare checks it with
  link 0's tolerance, 1e-5 px [V tools/eval/eval_kitti.py TOL_LINK0]). Bidirectional outputs
  come from one forward without only_fw, so the forward flow MAY differ from an only_fw run in
  the last bits (kernel choice by batch shape) [U until compare runs on the G4].
- RAFT imports its own `core` package, which shadows U2Flow's; this job never imports U2Flow
  code when --source raftct, and puts the RAFT checkout first on sys.path [E].
- Native predictions only for prio 0: a held-out id's prediction in a scorable form is not
  needed before unseal, so it is not written (hard rule 4 keeps its GT off the VM anyway).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import p1_windows as MW  # noqa: E402
import rigid  # noqa: E402
from p1lib import cache as CA  # noqa: E402
from p1lib import common as C  # noqa: E402
from p1lib import jobrun as JR  # noqa: E402
from p1lib import kitti as KT  # noqa: E402

GRID = (256, 832)
DIRS = ("10to11", "11to10", "09to10", "10to09")
TOOL_FILES = [HERE / "dump_eval.py", HERE / "rigid.py", HERE / "p1lib" / "u2flow.py",
              HERE / "p1lib" / "cache.py"]
TOL_LINK0 = 1e-5


def paths(data: Path, row: dict) -> dict:
    return {f: KT.frame_path(data, row["ds"], row["sub"], row["side"], row["seq"], f)
            for f in (9, 10, 11)}


# ------------------------------------------------------------------------------ U2Flow
class U2FlowSource:
    name = "u2flow"

    def __init__(self, a, data: Path):
        import torch
        from p1lib import u2flow as UF

        self.UF, self.data = UF, data
        UF.eval_kitti()  # puts the code root (core/) on sys.path, as for the real model
        self.grid = tuple(a.test_shape)
        self.device = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available())
                                   else "cpu")
        self.model = a.model_override
        if self.model is None:
            self.model, self.ckpt_sha = UF.load_checkpoint(Path(a.ckpt), a.weights_key,
                                                           self.device)
        else:
            self.ckpt_sha = "test-model"
        self.ident = {"source": "u2flow", "ckpt_sha256": self.ckpt_sha, "grid": list(self.grid),
                      "weights_key": a.weights_key, "autocast": "torch.cuda.amp fp16 in the net"
                      " (CUDA only)", "device": str(self.device)}

    def produce(self, row: dict):
        import torch

        from core.utils.flow_utils import resize_flow

        p = paths(self.data, row)
        ds = self.UF.frames_dataset(self.data, [[p[9], p[10]], [p[10], p[11]]], self.grid)
        (im01, _, h, w), (im12, _, _, _) = ds[0], ds[1]
        e01 = self.UF.bidirectional(self.model, im01, self.device)
        e12 = self.UF.bidirectional(self.model, im12, self.device)
        arrays = {}
        for d, ev, k_f, k_a, k_v in (("10to11", e12, "f12", "a12", "vis12"),
                                     ("11to10", e12, "b21", "a21", "vis21"),
                                     ("09to10", e01, "f12", "a12", "vis12"),
                                     ("10to09", e01, "b21", "a21", "vis21")):
            f = ev[k_f][0].float().cpu().numpy()
            if not np.isfinite(f).all():
                raise JR.ItemFailed("nonfinite", f"flow {d}")
            arrays[f"flow_{d}"] = f.astype(np.float16)
            arrays[f"alpha_{d}"] = ev[k_a][0, 0].float().cpu().numpy().astype(np.float16)
            arrays[f"vis_{d}"] = CA.pack_bits(ev[k_v][0, 0].cpu().numpy() > 0.5)
        if row.get("prio", 0) == 0:
            with torch.no_grad():
                up = resize_flow(e12["f12"], (int(h), int(w)))[0].detach().cpu().numpy()
            arrays["native_10to11"] = up.astype(np.float32)
        meta = {"grid": list(self.grid), "native_hw": [int(h), int(w)], "dirs": list(DIRS),
                "source_ident": self.ident, "k15_half": row.get("k15_half"),
                "alpha": "log variance (sigma^2 = exp(alpha))",
                "vis": "1 - get_occu_mask_bidirection(f, b) [V train_fusion_kitti.py]"}
        return arrays, meta


# ------------------------------------------------------------------------------ RAFT C+T
class RaftSource:
    name = "raftct"

    def __init__(self, a, data: Path):
        import argparse as ap

        import torch

        self.data, self.grid = data, tuple(a.test_shape)
        self.device = torch.device(a.device)
        code = Path(C.deps_dir(getattr(a, "deps_lock", None), data, a.code_dir,
                               "--code-dir") or "")
        if not (code / "core" / "raft.py").is_file():
            raise C.Stop(C.EXIT_PRE, "--code-dir must hold princeton-vl/RAFT's core/raft.py "
                                     "(a checkout, or teacher-factory's install dir; glob ok), "
                                     "or give --deps-lock env/teachers/raft.txt")
        sys.path[:0] = [str(code / "core"), str(code)]
        from raft import RAFT  # noqa: PLC0415
        from utils.utils import InputPadder  # noqa: PLC0415

        pin = C.weights_pin("raft-models")["extract"]["sha256"]
        sha = C.sha256_file(a.ckpt)
        if sha != pin:
            raise C.Stop(C.EXIT_REFUSED, f"{a.ckpt}: sha256 {sha[:12]} is not raft-things' pin")
        args = ap.Namespace(small=False, mixed_precision=False, alternate_corr=False, dropout=0)
        model = torch.nn.DataParallel(RAFT(args))
        model.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=True))
        self.model = model.module.to(self.device).eval()
        self.Padder = InputPadder
        self.ident = {"source": "raftct", "ckpt_sha256": sha, "iters": 24, "padder": "kitti",
                      "resolution": "native", "grid_by": "rigid.carry_flow (eq. 2)"}

    def flow(self, a, b):
        import torch

        t = [torch.from_numpy(x).permute(2, 0, 1).float()[None].to(self.device) for x in (a, b)]
        pad = self.Padder(t[0].shape, mode="kitti")
        x1, x2 = pad.pad(*t)
        with torch.no_grad():
            _, up = self.model(x1, x2, iters=24, test_mode=True)
        return pad.unpad(up[0]).cpu().numpy().astype(np.float64)

    def produce(self, row: dict):
        p = paths(self.data, row)
        im = {f: C.read_rgb(p[f]) for f in p}
        arrays = {}
        pairs = {"10to11": (10, 11), "11to10": (11, 10), "09to10": (9, 10), "10to09": (10, 9)}
        for d, (s, t) in pairs.items():
            f = self.flow(im[s], im[t])
            if not np.isfinite(f).all():
                raise JR.ItemFailed("nonfinite", f"flow {d}")
            arrays[f"flow_{d}"] = rigid.carry_flow(f, self.grid).astype(np.float16)
            if d == "10to11" and row.get("prio", 0) == 0:
                arrays["native_10to11"] = f.astype(np.float32)
        h, w = im[10].shape[:2]
        return arrays, {"grid": list(self.grid), "native_hw": [h, w], "dirs": list(DIRS),
                        "source_ident": self.ident, "k15_half": row.get("k15_half")}


# ------------------------------------------------------------------------------ ARFlow (PX-S1)
ARFLOW_CKPT_SHA = "763f1777345c64b27a2e47e95716d938ae5d676224b42a9f217079603f56c80b"
ARFLOW_EPS = 1e-3         # alpha = ln(e + eps), finite everywhere; below every finite A_GRID
                          # point (ln 1e-3 < -4.5), so it decides no switch (PX-S1 draft rev 2)


class ArflowSource:
    """PX-S1's second base model: ARFlow pwclite_ar.tar (KITTI-15, two-frame), through the
    authors' own model, transform and helpers at e92a8bb (env/teachers/arflow.txt). It writes the
    u2flow evidence format, so self_triangulate.py and the scorers read it unchanged. ARFlow has
    no uncertainty head; alpha_<d> is the declared label-free substitute (PX-S1 design draft):
    alpha = ln(e + 1e-3), e(p) = |f12(p) + warp(f21, f12)(p)|^2 in grid px^2 with ARFlow's
    zeros padding: exactly the squared forward-backward error that get_occu_mask_bidirection
    thresholds [V ARFlow utils/warp_utils.py:96-103], out-of-frame pixels included (there the
    warped f21 is 0, so e = |f12|^2, as the authors' occlusion test sees it); vis is
    1 - get_occu_mask_bidirection, as for U2Flow."""
    name = "arflow"

    def __init__(self, a, data: Path):
        import types

        import torch

        self.data, self.grid = data, tuple(a.test_shape)
        self.device = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available())
                                   else "cpu")
        code = Path(C.deps_dir(getattr(a, "deps_lock", None), data, a.code_dir,
                               "--code-dir") or "")
        if not (code / "models" / "pwclite.py").is_file():
            raise C.Stop(C.EXIT_PRE, "--code-dir must hold lliuz/ARFlow's models/pwclite.py, or "
                                     "give --deps-lock env/teachers/arflow.txt")
        sys.path.insert(0, str(code))
        # pwclite imports the CUDA extension's module; the authors' pure-torch twin stands in
        import models.correlation_native as cn  # noqa: PLC0415
        shim = types.ModuleType("models.correlation_package.correlation")
        shim.Correlation = cn.Correlation
        sys.modules["models.correlation_package"] = types.ModuleType(
            "models.correlation_package")
        sys.modules["models.correlation_package.correlation"] = shim
        from models.pwclite import PWCLite  # noqa: PLC0415
        from transforms import sep_transforms as sep  # noqa: PLC0415
        from utils import flow_utils as fu  # noqa: PLC0415
        from utils import warp_utils as wu  # noqa: PLC0415

        ckpt = Path(a.ckpt) if a.ckpt else code / "checkpoints" / "KITTI15" / "pwclite_ar.tar"
        sha = C.sha256_file(ckpt)
        if sha != ARFLOW_CKPT_SHA:
            raise C.Stop(C.EXIT_REFUSED, f"{ckpt}: sha256 {sha[:12]} is not pwclite_ar's pin")
        cfg = types.SimpleNamespace(upsample=True, n_frames=2, reduce_dense=True)
        model = PWCLite(cfg)
        st = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
        model.load_state_dict({k.replace("module.", "", 1): v for k, v in st.items()})
        self.model = model.to(self.device).eval()
        self.zoom, self.to_t = sep.Zoom(*self.grid), sep.ArrayToTensor()
        self.fu, self.wu = fu, wu
        self.ident = {"source": "arflow", "ckpt_sha256": sha, "code": "lliuz/ARFlow@e92a8bb",
                      "grid": list(self.grid), "transform": "imageio float32, sep_transforms."
                      "Zoom (skimage resize), ArrayToTensor, /255 (inference.py)",
                      "alpha": "ln(e + 1e-3), e = |f12 + warp_zeros(f21, f12)|^2 grid px^2 "
                      "(get_occu_mask_bidirection's error)", "device": str(self.device)}

    def _img(self, path: Path):
        import imageio.v2 as imageio

        x = imageio.imread(path).astype(np.float32)
        return self.to_t(self.zoom(x)) / 255.0

    def _alpha(self, f12, f21):
        import torch

        e = ((f12 + self.wu.flow_warp(f21, f12, pad="zeros")) ** 2).sum(1, keepdim=True)
        return torch.log(e + ARFLOW_EPS)

    def bidirectional(self, a, b) -> dict:
        import torch

        x = torch.cat([a, b], 0)[None].to(self.device)
        with torch.no_grad():
            o = self.model(x, with_bk=True)
        f12, f21 = o["flows_fw"][0], o["flows_bw"][0]
        occ = self.wu.get_occu_mask_bidirection
        return {"f12": f12, "b21": f21, "a12": self._alpha(f12, f21),
                "a21": self._alpha(f21, f12), "vis12": 1 - occ(f12, f21),
                "vis21": 1 - occ(f21, f12)}

    def produce(self, row: dict):
        import torch

        p = paths(self.data, row)
        im = {f: self._img(p[f]) for f in p}
        h, w = C.png_size(p[10])
        e01, e12 = self.bidirectional(im[9], im[10]), self.bidirectional(im[10], im[11])
        arrays = {}
        for d, ev, k_f, k_a, k_v in (("10to11", e12, "f12", "a12", "vis12"),
                                     ("11to10", e12, "b21", "a21", "vis21"),
                                     ("09to10", e01, "f12", "a12", "vis12"),
                                     ("10to09", e01, "b21", "a21", "vis21")):
            f = ev[k_f][0].float().cpu().numpy()
            if not np.isfinite(f).all():
                raise JR.ItemFailed("nonfinite", f"flow {d}")
            arrays[f"flow_{d}"] = f.astype(np.float16)
            arrays[f"alpha_{d}"] = ev[k_a][0, 0].float().cpu().numpy().astype(np.float16)
            arrays[f"vis_{d}"] = CA.pack_bits(ev[k_v][0, 0].cpu().numpy() > 0.5)
        if row.get("prio", 0) == 0:
            with torch.no_grad():
                up = self.fu.resize_flow(e12["f12"].clone(), (int(h), int(w)))
            arrays["native_10to11"] = up[0].cpu().numpy().astype(np.float32)
        meta = {"grid": list(self.grid), "native_hw": [int(h), int(w)], "dirs": list(DIRS),
                "source_ident": self.ident, "k15_half": row.get("k15_half"),
                "alpha": self.ident["alpha"],
                "vis": "1 - get_occu_mask_bidirection(f, b) [V ARFlow utils/warp_utils.py]"}
        return arrays, meta


# ------------------------------------------------------------------------------ Sun-RAFT (PX-J5)
SUNRAFT_CKPT = {"url": "https://github.com/cv-stuttgart/Sun-RAFT/releases/download/v1.0.0/"
                       "Sun-RAFT_kitti-test.pth",
                "sha256": "a53af7c26edf8553db429307a478dcc2d0b63eb144f2e1d67c33f82040848548",
                "bytes": 21103087}  # [M, the Mac 2026-10-13; reports/cycles/PX-J4.survey.md]
SUNRAFT_ITERS = 12        # the authors' KITTI evaluation (evaluate.py validate_kitti, --iters 12)
# The RAFT config SunraftSource builds: the keys of cpy_eval_args_to_config [V config_loader.py:
# 110-121 @912e2b4]; mixed_precision False is PX-J5 rev 2's declared fp32 (the authors' default
# is True). One constant, so a test can pin what the network is built with.
SUNRAFT_CFG = {"mixed_precision": False, "fnet_norm": "instance", "cnet_norm": "batch"}
SUNRAFT_EPS = ARFLOW_EPS  # the same declared floor as ARFlow's alpha (PX-S1)
OCC_SCALE, OCC_BIAS = 0.01, 0.5   # get_occu_mask_bidirection's constants, identical in U2Flow,
                                  # ARFlow and Sun-RAFT [V each repo's warp_utils.py]


def flow_warp_zeros(x, flow12):
    """x (B,C,H,W) sampled at p + flow12(p): bilinear, zeros padding, align_corners=True, the
    flow_warp(..., pad='zeros') of ARFlow and Sun-RAFT [V both warp_utils.py]; local so the CPU
    tests need no third-party tree (tests/test_sunraft.py checks it against Sun-RAFT's own)."""
    import torch

    b, _, h, w = flow12.shape
    ys, xs = torch.meshgrid(torch.arange(h, dtype=flow12.dtype),
                            torch.arange(w, dtype=flow12.dtype), indexing="ij")
    gx = 2.0 * (xs[None] + flow12[:, 0]) / max(w - 1, 1) - 1.0
    gy = 2.0 * (ys[None] + flow12[:, 1]) / max(h - 1, 1) - 1.0
    return torch.nn.functional.grid_sample(x, torch.stack([gx, gy], -1), mode="bilinear",
                                           padding_mode="zeros", align_corners=True)


def fb_alpha_vis(f12: np.ndarray, f21: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """On one grid: alpha = ln(e + 1e-3), e = |f12 + warp_zeros(f21, f12)|^2 (PX-S1's declared
    reliability substitute), and vis = not(e > 0.01 (|f12|^2 + |warped f21|^2) + 0.5), the
    occlusion test of get_occu_mask_bidirection without Sun-RAFT's extra out-of-image masking,
    so vis means the same for U2Flow, ARFlow and Sun-RAFT."""
    import torch

    a = torch.from_numpy(np.asarray(f12, np.float32))[None]
    b = torch.from_numpy(np.asarray(f21, np.float32))[None]
    bw = flow_warp_zeros(b, a)
    e = ((a + bw) ** 2).sum(1)[0]
    mag = (a ** 2).sum(1)[0] + (bw ** 2).sum(1)[0]
    vis = ~(e > OCC_SCALE * mag + OCC_BIAS)
    return torch.log(e + SUNRAFT_EPS).numpy(), vis.numpy()


class SunraftSource:
    """PX-J5's third base model: Sun-RAFT (Jahedi et al., WACV 2026), the KITTI checkpoint
    Sun-RAFT_kitti-test.pth (unsupervised Chairs, then KITTI-15 multi-view TESTING pairs only;
    trained to predict out-of-frame motion by crop self-supervision, no full-image warping),
    run by the authors' own KITTI protocol [V evaluate.py validate_kitti @912e2b4]: RGB uint8 as
    float, native resolution, InputPadder(mode='kitti', coarsest_scale=8), 12 iterations,
    test_mode, one forward giving both directions (the forward's bw=True default); fp32
    throughout (PX-J5 rev 2: the declared deviation from the authors' mixed-precision default,
    so the G4 runs the path tests/test_sunraft.py's parity test covers); CUDA required.
    Flows are carried to the 256 x 832 grid by eq. 2 (rigid.carry_flow, as RAFT C+T); alpha
    and vis are formed there (fb_alpha_vis), so the evidence format is the u2flow one and
    self_triangulate.py and second_base.py read it unchanged.
    Code from env/teachers/sunraft.txt; the checkpoint from its pinned release URL
    (SUNRAFT_CKPT, sha256-checked by teacher_deps.download), never redistributed."""
    name = "sunraft"

    def __init__(self, a, data: Path):
        import torch

        self.data, self.grid = data, tuple(a.test_shape)
        if a.device == "cuda" and not torch.cuda.is_available():   # no silent CPU fallback
            raise C.Stop(C.EXIT_PRE, "--device cuda: CUDA is not available (Sun-RAFT never "
                                     "falls back to the CPU)")
        self.device = torch.device(a.device)
        self.amp = SUNRAFT_CFG["mixed_precision"]   # fp32 throughout (PX-J5 rev 2, minor 4)
        self.model = getattr(a, "model_override", None)
        if self.model is not None:          # CPU tests: a stub with the same call signature
            self.Padder, sha = a.padder_override, "test-model"
        else:
            code = Path(C.deps_dir(getattr(a, "deps_lock", None), data, a.code_dir,
                                   "--code-dir") or "")
            if (code / "core" / "raft.py").is_file():   # the lock's install dir keeps core/
                code = code / "core"
            if not (code / "raft.py").is_file():
                raise C.Stop(C.EXIT_PRE, "--code-dir must hold Sun-RAFT's core/ (raft.py), or "
                                         "give --deps-lock env/teachers/sunraft.txt")
            sys.path.insert(0, str(code))
            from raft import RAFT  # noqa: PLC0415
            from utils.utils import InputPadder  # noqa: PLC0415

            ckpt = (Path(a.ckpt) if a.ckpt
                    else data / "weights" / "sunraft" / "Sun-RAFT_kitti-test.pth")
            if not ckpt.is_file() or C.sha256_file(ckpt) != SUNRAFT_CKPT["sha256"]:
                sys.path.insert(0, str(C.REPO / "tools" / "teacher"))
                import teacher_deps as td  # noqa: PLC0415
                try:
                    td.download(SUNRAFT_CKPT["url"], ckpt, SUNRAFT_CKPT["sha256"],
                                SUNRAFT_CKPT["bytes"])
                except td.DepsError as e:
                    raise C.Stop(e.code, f"Sun-RAFT checkpoint: {e.msg}") from None
            sha = C.sha256_file(ckpt)
            if sha != SUNRAFT_CKPT["sha256"]:
                raise C.Stop(C.EXIT_REFUSED, f"{ckpt}: sha256 {sha[:12]} is not the pin")
            model = torch.nn.DataParallel(RAFT(dict(SUNRAFT_CFG)))
            model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
            self.model = model.module.to(self.device).eval()
            self.Padder = InputPadder
        self.ident = {"source": "sunraft", "ckpt_sha256": sha,
                      "code": "cv-stuttgart/Sun-RAFT@912e2b44c600fc733363b57a8852e9e1801e4c1f",
                      "iters": SUNRAFT_ITERS, "padder": "kitti, coarsest_scale 8",
                      "mixed_precision": self.amp, "resolution": "native",
                      "grid_by": "rigid.carry_flow (eq. 2)", "grid": list(self.grid),
                      "alpha": "ln(e + 1e-3), e = |f12 + warp_zeros(f21, f12)|^2 grid px^2, on "
                               "the carried grid flows (PX-S1's substitute)",
                      "device": str(self.device)}

    def flows(self, a, b) -> tuple[np.ndarray, np.ndarray]:
        """(forward a->b, backward b->a) at native resolution, float64, as validate_kitti."""
        import torch

        t = [torch.from_numpy(x).permute(2, 0, 1).float()[None].to(self.device) for x in (a, b)]
        pad = self.Padder(t[0].shape, mode="kitti", coarsest_scale=8)
        x1, x2 = pad.pad(*t)
        with torch.no_grad():
            _, up = self.model(x1, x2, iters=SUNRAFT_ITERS, test_mode=True)
        up = pad.unpad(up).float().cpu().numpy().astype(np.float64)
        if up.shape[0] != 2:
            raise JR.ItemFailed("shape", f"expected forward and backward flows, got {up.shape}")
        return up[0], up[1]

    def produce(self, row: dict):
        p = paths(self.data, row)
        im = {f: C.read_rgb(p[f]) for f in p}
        arrays, grid_f = {}, {}
        for (s, t), (dfw, dbw) in (((10, 11), ("10to11", "11to10")),
                                   ((9, 10), ("09to10", "10to09"))):
            fw, bw = self.flows(im[s], im[t])
            for d, f in ((dfw, fw), (dbw, bw)):
                if not np.isfinite(f).all():
                    raise JR.ItemFailed("nonfinite", f"flow {d}")
                grid_f[d] = rigid.carry_flow(f, self.grid).astype(np.float32)
            if dfw == "10to11" and row.get("prio", 0) == 0:
                arrays["native_10to11"] = fw.astype(np.float32)
        for d, back in (("10to11", "11to10"), ("11to10", "10to11"), ("09to10", "10to09"),
                        ("10to09", "09to10")):
            alpha, vis = fb_alpha_vis(grid_f[d], grid_f[back])
            arrays[f"flow_{d}"] = grid_f[d].astype(np.float16)
            arrays[f"alpha_{d}"] = alpha.astype(np.float16)
            arrays[f"vis_{d}"] = CA.pack_bits(vis)
        h, w = im[10].shape[:2]
        return arrays, {"grid": list(self.grid), "native_hw": [h, w], "dirs": list(DIRS),
                        "source_ident": self.ident, "k15_half": row.get("k15_half"),
                        "alpha": self.ident["alpha"],
                        "vis": "not(e > 0.01 |f|^2 + 0.01 |warped b|^2 + 0.5) on the grid "
                               "(get_occu_mask_bidirection's test)"}


# ------------------------------------------------------------------------------ stub
class StubSource:
    name = "stub"

    def __init__(self, a, data: Path):
        from p1lib.synth import require_stub_host

        require_stub_host()
        self.data, self.grid = data, tuple(a.test_shape)
        self.ident = {"source": "stub", "grid": list(self.grid)}

    def produce(self, row: dict):
        from p1lib.synth import Stub

        p = paths(self.data, row)
        h0, w0 = C.png_size(p[10])
        K, _ = KT.calib_K(self.data, row["ds"], row["sub"], row["seq"], row["side"])
        Kg = rigid.zoom_K(K, (h0, w0), self.grid)
        T = {f: Stub.scene_pose(f - 10) for f in (9, 10, 11)}
        D = {f: Stub.scene_depth(Kg, T[f], self.grid) for f in T}
        arrays = {}
        for d, (s, t) in {"10to11": (10, 11), "11to10": (11, 10), "09to10": (9, 10),
                          "10to09": (10, 9)}.items():
            r = rigid.rigid_flow(D[s], Kg, rigid.relative_pose(T[s], T[t]))
            oof = rigid.oof_mask(r, self.grid)
            arrays[f"flow_{d}"] = np.nan_to_num(r.flow).astype(np.float16)
            arrays[f"alpha_{d}"] = np.where(oof, 3.0, -1.0).astype(np.float16)
            arrays[f"vis_{d}"] = CA.pack_bits(r.valid & ~oof)
        if row.get("prio", 0) == 0:
            Kn = K
            Dn = Stub.scene_depth(Kn, T[10], (h0, w0))
            arrays["native_10to11"] = np.nan_to_num(rigid.rigid_flow(
                Dn, Kn, rigid.relative_pose(T[10], T[11])).flow).astype(np.float32)
        return arrays, {"grid": list(self.grid), "native_hw": [h0, w0], "dirs": list(DIRS),
                        "source_ident": self.ident, "k15_half": row.get("k15_half")}


SOURCES = {"u2flow": U2FlowSource, "raftct": RaftSource, "arflow": ArflowSource,
           "sunraft": SunraftSource, "stub": StubSource}


# ------------------------------------------------------------------------------ compare
def compare(cache_dir: Path, preds_dir: Path, tol: float) -> dict:
    sys.path.insert(0, str(C.REPO / "tools" / "eval"))
    import eval_kitti as ek  # noqa: PLC0415

    C.refuse_sealed(cache_dir, preds_dir)
    idx = CA.CacheIndex(cache_dir)
    ours = {}
    for item in idx.ids():
        d = KT.parse_id(item)
        if d["ds"] == "kitti15":
            ours[d["seq"]] = item
    worst, n, missing = 0.0, 0, []
    for img_id, flow in ek.read_preds(Path(preds_dir)):
        item = ours.get(img_id)
        got = idx.get(item) if item else None
        if got is None or "native_10to11" not in got[0]:
            missing.append(img_id)
            continue
        diff = float(np.abs(got[0]["native_10to11"].astype(np.float64) - flow).max())
        worst, n = max(worst, diff), n + 1
    return {"n": n, "max_abs_px": worst, "tol": tol, "missing": missing,
            "ok": n > 0 and worst <= tol and not missing}


def args_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--source", required=True, choices=sorted(SOURCES))
    r.add_argument("--data")
    r.add_argument("--out-dir", required=True)
    r.add_argument("--ckpt")
    r.add_argument("--weights-key", default="u2flow-kitti")
    r.add_argument("--code-dir")
    r.add_argument("--deps-lock", help="install teacher-factory's lock first and use its dir "
                                       "(env/teachers/raft.txt); replaces --code-dir")
    r.add_argument("--device", default="cuda")
    r.add_argument("--plan", default=str(MW.PLAN))
    r.add_argument("--max-prio", type=int)
    r.add_argument("--work-dir")
    r.add_argument("--shard")
    r.add_argument("--shard-size", type=int, default=25)
    r.add_argument("--limit", type=int)
    r.add_argument("--test-shape", type=int, nargs=2, default=list(GRID))
    r.add_argument("--run", default="")
    c = sub.add_parser("compare")
    c.add_argument("--cache", required=True)
    c.add_argument("--preds", required=True)
    c.add_argument("--tol", type=float, default=TOL_LINK0)
    return ap


def main(argv: list[str] | None = None) -> int:
    try:
        a = args_parser().parse_args(argv)
    except SystemExit as e:
        return C.EXIT_USAGE if e.code else C.EXIT_OK
    try:
        if a.cmd == "compare":
            res = compare(Path(a.cache), Path(a.preds), a.tol)
            C.say("DUMP_COMPARE", __import__("json").dumps(res, sort_keys=True))
            return C.EXIT_OK if res["ok"] else C.EXIT_FAIL
        return run(a)
    except C.Stop as e:
        C.say("P1_ERROR", e.msg)
        return e.code


def run(a, model_override=None) -> int:
    a.model_override = model_override
    data = C.data_root(a.data)
    out = Path(a.out_dir)
    C.refuse_sealed(out, a.work_dir, a.ckpt, a.code_dir)
    rows = MW.load_items("ff-eval", data, Path(a.plan), a.max_prio)
    if a.source not in ("stub", "arflow", "sunraft") and not a.ckpt and model_override is None:
        raise C.Stop(C.EXIT_USAGE, "--ckpt is needed")
    src = SOURCES[a.source](a, data)
    settings = {"source": src.ident, "max_prio": a.max_prio,
                "plan_lists": MW.load_plan(Path(a.plan))["lists"]["ff-eval"]}
    identity = {"settings": settings, "code": C.code_identity(TOOL_FILES)}
    work = Path(a.work_dir) if a.work_dir else (
        data / "derived" / "p1work" / f"evidence-{a.source}-"
        f"{C.sha256_bytes(repr(sorted(identity.items())).encode())[:12]}")
    cache_dir = out / "cache" / "diag" / ("stubflow" if a.source == "stub" else a.source)
    job = JR.ItemJob(kind="evidence", source=a.source, items=rows, cache_dir=cache_dir,
                     work_dir=work, shard_size=a.shard_size, shard=a.shard, identity=identity,
                     settings=settings, limit=a.limit)
    job.progress.resolved({"tool": "tools/p1/dump_eval.py", "run": a.run, "source": src.ident,
                           "n_items": job.total, "shard": a.shard or "0/1",
                           "cache_dir": str(cache_dir), "work_dir": str(work)})
    return job.run(src.produce)


if __name__ == "__main__":
    sys.exit(main())
