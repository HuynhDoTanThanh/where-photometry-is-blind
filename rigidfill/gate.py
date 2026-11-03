"""The uncertainty gate: where to replace the network's flow S by the rigid flow T.

At native resolution, a pixel is replaced when T is defined there and

    alpha > tau    and    d_min < ||T - S||_2 <= d_max,

with ``alpha`` the network's (upsampled) uncertainty and the distance in native pixels.
The thresholds of the paper are available as presets.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Union

import numpy as np


@dataclass(frozen=True)
class GateParams:
    """Thresholds of the gate.

    Attributes:
        tau: replace only where ``alpha > tau`` (strict).
        d_max: and ``||T - S||_2 <= d_max`` native pixels (inclusive; ``inf`` = no upper bound).
        d_min: and ``||T - S||_2 > d_min`` native pixels (strict; ``-inf`` = no lower bound).
    """

    tau: float
    d_max: float = math.inf
    d_min: float = -math.inf


#: Gate thresholds of the paper, each selected on the 100 KITTI-15 development images:
#: ``"u2flow"`` for U2Flow's predicted log-variance; ``"arflow"`` and ``"sunraft"`` for
#: ``alpha = ln(r + 1e-3)`` from :func:`rigidfill.fb_score`.
PRESETS: Dict[str, GateParams] = {
    "u2flow": GateParams(tau=2.75, d_max=100.0),
    "arflow": GateParams(tau=1.0, d_max=100.0, d_min=6.0),
    "sunraft": GateParams(tau=1.5, d_max=math.inf, d_min=8.0),
}


def resolve_gate(gate: Union[str, GateParams, None] = "u2flow", **overrides) -> GateParams:
    """A :class:`GateParams` from a preset name or an instance, with ``tau``, ``d_max`` or
    ``d_min`` overridden where the keyword is not ``None``."""
    if gate is None:
        gate = "u2flow"
    if isinstance(gate, str):
        try:
            base = PRESETS[gate.lower()]
        except KeyError:
            raise ValueError(f"unknown gate preset {gate!r}; choose one of "
                             f"{', '.join(sorted(PRESETS))} or pass GateParams") from None
    else:
        base = gate
    vals = {k: v for k, v in overrides.items() if v is not None}
    return GateParams(**{**base.__dict__, **vals})


def uncertainty_gate(alpha, rigid, base, params: Union[str, GateParams] = "u2flow"
                     ) -> np.ndarray:
    """The pixels where RigidFill replaces the network's flow.

    Args:
        alpha: ``(H, W)`` uncertainty at native resolution (:func:`rigidfill.upsample_score`).
            NaN or ``-inf`` never passes.
        rigid: ``(2, H, W)`` rigid flow T at native resolution, NaN where undefined.
        base: ``(2, H, W)`` network flow S at native resolution.
        params: a preset name or :class:`GateParams`.

    Returns:
        bool ``(H, W)``.
    """
    p = resolve_gate(params)
    T = np.asarray(rigid, dtype=np.float32)
    S = np.asarray(base, dtype=np.float32)
    defined = np.isfinite(T).all(axis=0)
    d = np.hypot(*(np.nan_to_num(T) - S))
    a = np.asarray(alpha, dtype=np.float32)
    with np.errstate(invalid="ignore"):
        return defined & (a > p.tau) & (d <= p.d_max) & (d > p.d_min)


def apply_gate(gate, rigid, base) -> np.ndarray:
    """``O = g T + (1 - g) S`` as a selection (undefined values of T never enter O); float32."""
    g = np.asarray(gate, dtype=bool)
    return np.where(g[None], np.asarray(rigid, np.float32),
                    np.asarray(base, np.float32)).astype(np.float32)
