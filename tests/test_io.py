"""KITTI files."""
import numpy as np
import pytest

import rigidfill


def test_kitti_flow_png_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    flow = rng.uniform(-300, 300, (2, 37, 53)).astype(np.float32)
    valid = rng.random((37, 53)) > 0.3
    flow[:, 0, 0] = np.nan
    p = tmp_path / "f.png"
    rigidfill.write_kitti_flow(p, flow, valid)
    back, v = rigidfill.read_kitti_flow(p)
    want = valid.copy()
    want[0, 0] = False
    assert np.array_equal(v, want)
    assert np.abs(back - flow)[:, v].max() <= 1 / 128 + 1e-6
    assert (back[:, ~v] == 0).all()


def test_kitti_flow_png_default_valid(tmp_path):
    flow = np.zeros((2, 4, 5), np.float32)
    flow[1, 2, 3] = np.inf
    p = tmp_path / "g.png"
    rigidfill.write_kitti_flow(p, flow)
    _, v = rigidfill.read_kitti_flow(p)
    assert v.sum() == 19 and not v[2, 3]


@pytest.mark.parametrize("text,camera", [
    ("P0: 1 0 2 0 0 3 4 0 0 0 1 0\nP2: 7.07e2 0 6.01e2 46.9 0 7.07e2 1.83e2 0.1 0 0 1 0\n", 2),
    ("calib_time: x\nP_rect_02: 7.07e2 0 6.01e2 46.9 0 7.07e2 1.83e2 0.1 0 0 1 0\n", 2),
])
def test_read_intrinsics(tmp_path, text, camera):
    p = tmp_path / "calib.txt"
    p.write_text(text)
    K = rigidfill.read_kitti_intrinsics(p, camera)
    assert np.allclose(K, [[707, 0, 601], [0, 707, 183], [0, 0, 1]])


def test_read_intrinsics_missing(tmp_path):
    p = tmp_path / "calib.txt"
    p.write_text("P0: 1 0 0 0 0 1 0 0 0 0 1 0\n")
    with pytest.raises(KeyError):
        rigidfill.read_kitti_intrinsics(p, 2)


def test_scale_intrinsics_pixel_centres():
    K = np.array([[700.0, 0, 600.0], [0, 700.0, 180.0], [0, 0, 1]])
    Kg = rigidfill.scale_intrinsics(K, (375, 1242), (256, 832))
    sx, sy = 832 / 1242, 256 / 375
    assert Kg[0, 0] == pytest.approx(700 * sx) and Kg[1, 1] == pytest.approx(700 * sy)
    assert Kg[0, 2] == pytest.approx(sx * 600.5 - 0.5)
    assert Kg[1, 2] == pytest.approx(sy * 180.5 - 0.5)
