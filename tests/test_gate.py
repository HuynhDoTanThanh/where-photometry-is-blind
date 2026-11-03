"""The uncertainty gate and its presets."""
import math

import numpy as np
import pytest

import rigidfill
from rigidfill import GateParams, uncertainty_gate


def fields(n=6):
    S = np.zeros((2, 1, n), np.float32)
    T = np.zeros((2, 1, n), np.float32)
    return S, T


def test_alpha_is_strict_and_distance_inclusive():
    S, T = fields()
    T[0, 0] = [0, 50, 100, 100.001, 5, 5]
    alpha = np.array([[3, 3, 3, 3, 2.75, 2.76]], np.float32)
    g = uncertainty_gate(alpha, T, S, "u2flow")
    assert g.tolist() == [[True, True, True, False, False, True]]


def test_undefined_rigid_flow_never_switches():
    S, T = fields(3)
    T[:, 0, 1] = np.nan
    alpha = np.full((1, 3), 10.0, np.float32)
    assert uncertainty_gate(alpha, T, S).tolist() == [[True, False, True]]


def test_nonfinite_alpha_never_switches():
    S, T = fields(3)
    alpha = np.array([[np.nan, -np.inf, 5.0]], np.float32)
    assert uncertainty_gate(alpha, T, S).tolist() == [[False, False, True]]


def test_presets_and_lower_bound():
    assert rigidfill.PRESETS["u2flow"] == GateParams(2.75, 100.0)
    assert rigidfill.PRESETS["arflow"] == GateParams(1.0, 100.0, 6.0)
    assert rigidfill.PRESETS["sunraft"] == GateParams(1.5, math.inf, 8.0)
    S, T = fields(4)
    T[0, 0] = [6, 6.5, 8, 500]
    alpha = np.full((1, 4), 2.0, np.float32)
    assert uncertainty_gate(alpha, T, S, "arflow").tolist() == [[False, True, True, False]]
    assert uncertainty_gate(alpha, T, S, "sunraft").tolist() == [[False, False, False, True]]


def test_overrides_and_errors():
    p = rigidfill.resolve_gate("u2flow", tau=1.0, d_max=None)
    assert p == GateParams(1.0, 100.0)
    with pytest.raises(ValueError):
        rigidfill.resolve_gate("raft")


def test_apply_gate_is_a_selection():
    S = np.ones((2, 2, 2), np.float32)
    T = np.full((2, 2, 2), np.nan, np.float32)
    T[:, 0, 0] = 7
    g = np.array([[True, False], [False, False]])
    out = rigidfill.apply_gate(g, T, S)
    assert np.isfinite(out).all() and out[0, 0, 0] == 7 and out[0, 1, 1] == 1


def test_fb_score_and_visibility():
    f = np.zeros((2, 4, 6), np.float32)
    f[0] = 1.0                       # everything moves one pixel right
    b = np.zeros_like(f)
    b[0] = -1.0                      # consistent backward flow
    r, vis = rigidfill.fb_consistency(f, b)
    assert np.allclose(r[:, :-1], 0) and vis[:, :-1].all()
    # the last column leaves the grid: the zero-padded warp gives r = |F|^2 = 1
    assert np.allclose(r[:, -1], 1.0)
    alpha = rigidfill.fb_score(f, b)
    assert np.allclose(alpha[:, :-1], np.log(1e-3), atol=1e-6)
    b[0] = 3.0                       # inconsistent: r = 16 > 0.01 (1 + 9) + 0.5
    _, vis = rigidfill.fb_consistency(f, b)
    assert not vis[:, :-1].any()
