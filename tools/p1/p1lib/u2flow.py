"""U2Flow as an evidence network and as the FF pseudo-labeller, through the authors' own code:
the eval loader, the network, get_occu_mask_bidirection and train_fusion_kitti's TinyModel
procedure, imported unchanged (dump_eval.py and run_ff_shard.py).

How to use (library only; on the VM the code dir is the staged commit, on the Mac the repo):
    ek = eval_kitti()                                   # tools/eval/eval_kitti.py, imported
    model, sha = load_checkpoint(ckpt, "u2flow-kitti", device)   # sha256 pinned (weights.json)
    ds = frames_dataset(data, [[p09, p10], [p10, p11]], test_shape=(256, 832))
    ev = bidirectional(model, ds[k][0], device)         # flows, alpha (log variance), vis
    ff = ff_item(model, images3, device)                # images3 = frames (t-1, t, t+1)
    ff["fused"], ff["mask_replaced"], ff["mask_fused"]   # F_FF, m_FF, the authors' fused mask
    seed_item(base_seed, item_id)                        # a per-item seed (shard-independent)

Why it is built this way.
- One transform: frames go through ImgSeqDataset's is_test path (imageio read, /255,
  Zoom(test_shape), ArrayToTensor) [V core/flow_datasets.py:54-63], the path eval_kitti.py's
  predictions take, so U2Flow's evidence for 10 -> 11 equals W1-07's prediction for the same
  pair before resize_flow [P W2-09 'Số U²Flow phải khớp W1-07'; dump_eval.py compare checks it].
  The class is constructed in 'testing' mode (no GT path is ever formed) and its image_list is
  replaced by the frames asked for; KITTI-12 frames go through the same base-class code.
- One FF: ff_item() feeds train_fusion_kitti.train_and_run_tiny_model exactly the inputs
  eval_kitti.ff_forward() gives it (model on (t, t+1) and on (t-1, t), the two bidirectional
  occlusion masks, the forward and backward uncertainties, iterations = the module's iters).
  tests/test_p1_jobs.py checks the fused flow equals ff_forward's on the same seed
  [P W2-08: 'KHÔNG đổi thuật toán, KHÔNG batch'].
- m_FF is the REPLACEMENT mask, not the returned one. The proposal defines m_FF as the pixels
  where FF replaced the forward flow (module ffl 'pixel mà FF thay flow thuận'; L_FF 'Pixel mù
  mà FF đã thay flow'); the function returns clamp(m_f + m_b), which also covers every
  confident-forward pixel, where F_FF IS kitti.pth's forward flow [V train_fusion_kitti.py:141],
  so L_FF on it would self-distil the released model into PF. The replacement mask is the
  authors' mask_backward_no_forward = (1 - m_f) m_b (1 - nan) [V :134-136], recomputed here on
  the same tensors with the same formula (m_* = exp(unc).clamp(1e-3, 200) < thr [V :103-108])
  without touching the FF algorithm: 0 everywhere when the function returned early
  (valid_mask.sum() < 1 [V :111-112]); the NaN guard is applied as 'the fused flow equals the
  forward flow exactly' (a NaN prediction leaves the forward flow there [V :138-140]; an exact
  float32 tie elsewhere is a no-op replacement), counted per item.
- Seeds: the released +FF is unseeded (traps T26) and eval_kitti seeds once per run, so an image's
  TinyModel init depends on the images before it. A sharded, resumable job needs each item's
  result to depend on the item only: seed_item() seeds core.u2f_runtime.seed_everything with
  sha256(base_seed:item_id) before each item [E]; the check against ff_forward uses the same seed.
"""
from __future__ import annotations

import contextlib
import hashlib
import sys
from pathlib import Path

from . import common as C

WEIGHT_KEYS = ("u2flow-kitti", "u2flow-sintel")


def code_root() -> Path:
    return C.REPO


def eval_kitti():
    sys.path.insert(0, str(C.REPO / "tools" / "eval"))
    import eval_kitti as ek  # noqa: PLC0415

    return ek


def verify_checkpoint(ckpt: Path, key: str) -> str:
    """sha256 of a released U2Flow checkpoint, refused unless it is weights.json's pin."""
    C.refuse_sealed(ckpt)
    if key not in WEIGHT_KEYS:
        raise C.Stop(C.EXIT_USAGE, f"--weights-key must be one of {', '.join(WEIGHT_KEYS)}")
    if not Path(ckpt).is_file():
        raise C.Stop(C.EXIT_PRE, f"{ckpt}: checkpoint missing")
    sha = C.sha256_file(ckpt)
    if sha != C.weights_pin(key)["sha256"]:
        raise C.Stop(C.EXIT_REFUSED, f"{ckpt}: sha256 {sha[:12]} is not {key}'s pin")
    return sha


def load_checkpoint(ckpt: Path, key: str, device):
    sha = verify_checkpoint(ckpt, key)
    return eval_kitti().load_model(Path(ckpt), device), sha


def frames_dataset(data: Path, frame_lists: list[list[Path]], test_shape=(256, 832)):
    """The authors' eval dataset over arbitrary frame lists (2 or 3 frames each)."""
    ek = eval_kitti()
    from core import flow_datasets as fd  # noqa: PLC0415

    with ek._loader_roots(Path(data), None):  # noqa: SLF001  (U2_DATA for the ported loaders)
        ds = fd.KITTI2015(test_shape=list(test_shape), split="testing")
    ds.image_list = [[str(p) for p in fl] for fl in frame_lists]
    ds.extra_info = [Path(fl[0]).name for fl in frame_lists]
    ds.flow_list, ds.flow_noc, ds.is_test = [], [], True
    missing = [p for fl in ds.image_list for p in fl if not Path(p).is_file()]
    if missing:
        raise C.Stop(C.EXIT_PRE, f"{len(missing)} frame(s) missing, first: {missing[0]}")
    return ds


def occ_bidirection(f12, f21):
    from core.utils.warp_utils import get_occu_mask_bidirection  # noqa: PLC0415

    return get_occu_mask_bidirection(f12, f21)


def bidirectional(model, images, device) -> dict:
    """One U2Flow forward on a 2-frame tensor (2, 3, H, W): both flows, both alphas (the
    log-variance heads; sigma^2 = exp(alpha) [V train_fusion_kitti.py:94-95]) and both
    visibility masks (1 - get_occu_mask_bidirection) [V train_fusion_kitti.py __main__]."""
    import torch

    x = images[None].to(device)
    with torch.no_grad():
        o = model(x)
    f12, b21 = o["flows_f12"][-1], o["flows_b21"][-1]
    return {"f12": f12, "b21": b21, "a12": o["uncertainty_f12"][-1],
            "a21": o["uncertainty_b21"][-1], "vis12": 1 - occ_bidirection(f12, b21),
            "vis21": 1 - occ_bidirection(b21, f12)}


def ff_module():
    return eval_kitti().ff_module()


def ff_masks(unc_f, unc_b, thr: float):
    """(m_f, m_b) exactly as train_and_run_tiny_model forms them [V train_fusion_kitti.py:
    103-108]: sigma^2 = exp(log variance) clamped to [1e-3, 200], confident when < thr."""
    m_f = (unc_f.exp().clamp(min=1e-3, max=200) < thr).float()
    m_b = (unc_b.exp().clamp(min=1e-3, max=200) < thr).float()
    return m_f, m_b


def ff_item(model, images3, device, iterations: int | None = None) -> dict:
    """train_fusion_kitti.py's +FF for frames (t-1, t, t+1) at the model's input grid:
    fused (1,2,H,W) F_FF; mask_replaced (1,1,H,W) the proposal's m_FF (where FF replaced the
    forward flow); mask_fused (1,1,H,W) the authors' returned clamp(m_f + m_b); m_f, m_b; and
    n_nan_guard (pixels of (1 - m_f) m_b whose fused flow stayed the forward flow).
    iterations: None = the module's iters (2000); otherwise passed to the authors' own
    `iterations` argument, whose lr schedule steps every iterations // 20 (PX-T1), so it must
    be >= 20."""
    import torch

    tfk = ff_module()
    n_it = int(tfk.iters) if iterations is None else int(iterations)
    if n_it < 20:
        raise ValueError(f"iterations {n_it} < 20: the schedule divides by iterations // 20")
    x = images3[None].to(device)
    with torch.no_grad():
        o12 = model(x[:, 1:])
        o01 = model(x[:, :-1])
    f12, f21 = o12["flows_f12"][-1], o12["flows_b21"][-1]
    f01, f10 = o01["flows_f12"][-1], o01["flows_b21"][-1]
    vis_f = 1 - occ_bidirection(f12, f21)
    vis_b = 1 - occ_bidirection(f10, f01)
    unc_f, unc_b = o12["uncertainty_f12"][-1], o01["uncertainty_b21"][-1]
    fused, mask = tfk.train_and_run_tiny_model(f12, f10, vis_f, vis_b, unc_f, unc_b,
                                               iterations=n_it, device=device)
    with torch.no_grad():
        m_f, m_b = ff_masks(unc_f, unc_b, float(tfk.thr))
        cand = (1 - m_f) * m_b
        if (m_f * m_b).sum() < 1:   # the function returned (flow_forward, m_f): nothing replaced
            cand = torch.zeros_like(cand)
        kept = (fused == f12).all(dim=1, keepdim=True).float()
        replaced = cand * (1 - kept)
    return {"fused": fused, "mask_fused": mask, "mask_replaced": replaced, "m_f": m_f,
            "m_b": m_b, "n_nan_guard": int((cand * kept).sum().item())}


def item_seed(base_seed: int, item_id: str) -> int:
    return int(hashlib.sha256(f"{int(base_seed)}:{item_id}".encode()).hexdigest()[:8], 16)


def seed_item(base_seed: int, item_id: str) -> int:
    from core import u2f_runtime  # noqa: PLC0415

    s = item_seed(base_seed, item_id)
    u2f_runtime.seed_everything(s)
    return s


@contextlib.contextmanager
def deterministic_cudnn():
    """cudnn deterministic, benchmark off, for the FF jobs (recorded): U2Flow's trainer runs the
    same way [V train.py:469-470]; it does not change the FF algorithm, only kernel choice."""
    import torch

    old = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    try:
        yield
    finally:
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = old
