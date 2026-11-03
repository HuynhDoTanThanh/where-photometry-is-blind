"""RigidFill on an exact synthetic rigid scene."""
import numpy as np
import pytest

import rigidfill
from rigidfill.synthetic import make_scene


def run(scene, **kw):
    return rigidfill.correct_flow(scene.flow_fw, scene.flow_bw, scene.alpha_fw, scene.K,
                                  scene.image_size, vis_fw=scene.vis_fw, vis_bw=scene.vis_bw,
                                  alpha_bw=scene.alpha_bw, **kw)


@pytest.fixture(scope="module")
def native_scene():
    return make_scene(image_size=(96, 320), grid=None)


def test_recovers_out_of_frame_flow(native_scene):
    s = native_scene
    assert s.oof.sum() > 500  # pixels that leave frame t+1
    out, info = run(s, half_precision_depth=False)
    assert info.ok and info.failure is None
    err_before = np.hypot(*(s.flow_fw - s.flow_fw_true))[s.oof]
    err_after = np.hypot(*(out - s.flow_fw_true))[s.oof]
    assert err_before.mean() > 2.0
    assert err_after.max() < 1e-3
    # exactly the out-of-frame pixels are replaced; in-frame pixels keep the network's flow
    assert np.array_equal(info.gate, s.oof)
    assert np.array_equal(out[:, ~s.oof], s.flow_fw[:, ~s.oof])


def test_geometry_is_exact(native_scene):
    s = native_scene
    _, info = run(s, half_precision_depth=False)
    g = info.geometry
    # exact up to the float32 storage of the simulated flows
    assert g.scale == pytest.approx(s.true_scale, rel=1e-3)
    T = info.rigid_flow_grid
    ok = np.isfinite(T).all(0)
    assert ok.mean() > 0.9
    assert np.abs(T - s.flow_fw_true)[:, ok].max() < 1e-3


def test_half_precision_depth_is_close(native_scene):
    s = native_scene
    out, info = run(s)  # the default stores depth as float16 inverse depth
    err = np.hypot(*(out - s.flow_fw_true))[s.oof]
    assert info.ok and err.max() < 0.2


def test_upsampling_to_native():
    s = make_scene(image_size=(150, 498), grid=(96, 320))
    out, info = run(s)
    assert info.ok and out.shape == (2, 150, 498)
    oof = s.oof_native & info.gate
    err_before = np.hypot(*(info.base_flow - s.flow_fw_true_native))[oof]
    err_after = np.hypot(*(out - s.flow_fw_true_native))[oof]
    assert oof.sum() > 0.8 * s.oof_native.sum()
    # U2Flow's upsampling (aligned corners) and the pixel-centre intrinsics differ slightly
    # near the border; S and T share that difference
    assert err_after.mean() < 0.5 and err_after.mean() < 0.05 * err_before.mean()


def test_noisy_flows_still_help():
    s = make_scene(image_size=(96, 320), grid=None, noise_px=0.1, seed=3)
    out, info = run(s)
    assert info.ok
    err_before = np.hypot(*(s.flow_fw - s.flow_fw_true))[s.oof].mean()
    err_after = np.hypot(*(out - s.flow_fw_true))[s.oof].mean()
    assert err_after < 0.2 * err_before


def test_low_level_pipeline_matches(native_scene):
    s = native_scene
    Kg = rigidfill.scale_intrinsics(s.K, s.image_size, s.grid)
    T, geo = rigidfill.rigid_flow_from_flows(s.flow_fw, s.flow_bw, Kg, vis_fw=s.vis_fw,
                                             vis_bw=s.vis_bw, alpha_fw=s.alpha_fw,
                                             alpha_bw=s.alpha_bw)
    _, info = run(s)
    np.testing.assert_array_equal(T, info.rigid_flow_grid)
    assert geo.n_inliers_fw == info.geometry.n_inliers_fw
