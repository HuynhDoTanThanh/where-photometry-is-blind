"""Upsampling matches U2Flow's resize_flow (PyTorch, aligned corners)."""
import numpy as np
import pytest

import rigidfill


def test_identity_and_corners():
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    np.testing.assert_array_equal(rigidfill.upsample_bilinear(x, (3, 4)), x)
    up = rigidfill.upsample_bilinear(x, (5, 7))
    assert up[0, 0] == x[0, 0] and up[-1, -1] == x[-1, -1]
    assert up[2, 3] == pytest.approx(x.mean())


def test_flow_scaling():
    f = np.ones((2, 4, 8), np.float32)
    up = rigidfill.upsample_flow(f, (6, 20))
    assert np.allclose(up[0], 20 / 8) and np.allclose(up[1], 6 / 4)


def test_nan_spreads_to_the_support():
    x = np.zeros((4, 4), np.float32)
    x[1, 1] = np.nan
    up = rigidfill.upsample_bilinear(x, (7, 7))
    assert np.isnan(up[2, 2]) and np.isnan(up[1, 1]) and not np.isnan(up[5, 5])
    s = rigidfill.upsample_score(x, (7, 7))
    assert s[2, 2] == -np.inf and np.isfinite(s[5, 5])


def test_matches_torch():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(1)
    f = rng.normal(0, 20, (2, 32, 104)).astype(np.float32)
    f[:, 5, 7] = np.nan
    want = torch.nn.functional.interpolate(torch.from_numpy(f)[None], (47, 155),
                                           mode="bilinear", align_corners=True)[0]
    want[0] /= 104 / 155.0
    want[1] /= 32 / 47.0
    got = rigidfill.upsample_flow(f, (47, 155))
    want = want.numpy()
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
    ok = np.isfinite(want)
    assert np.abs(got[ok] - want[ok]).max() < 1e-4
