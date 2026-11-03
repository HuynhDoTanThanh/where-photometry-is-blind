"""Images without a rigid flow keep the network's flow."""
import numpy as np

import rigidfill
from rigidfill.synthetic import make_scene


def scene():
    return make_scene(image_size=(64, 208), grid=None)


def test_too_few_correspondences_falls_back():
    s = scene()
    vis = np.zeros_like(s.vis_bw)
    vis[0, :5] = True            # 5 usable backward correspondences < 8
    out, info = rigidfill.correct_flow(s.flow_fw, s.flow_bw, s.alpha_fw, s.K, vis_fw=s.vis_fw,
                                       vis_bw=vis)
    assert not info.ok and info.failure.startswith("pose")
    assert info.rigid_flow is None and not info.gate.any()
    np.testing.assert_array_equal(out, s.flow_fw)


def test_all_uncertain_pose_set_falls_back():
    s = scene()
    alpha_bw = np.full_like(s.alpha_bw, 5.0)   # nothing passes alpha <= ln 2
    out, info = rigidfill.correct_flow(s.flow_fw, s.flow_bw, s.alpha_fw, s.K, vis_fw=s.vis_fw,
                                       vis_bw=s.vis_bw, alpha_bw=alpha_bw)
    assert not info.ok and "0 correspondences" in info.failure
    np.testing.assert_array_equal(out, s.flow_fw)


def test_stationary_camera_falls_back():
    s = scene()
    zero = np.zeros_like(s.flow_fw)
    out, info = rigidfill.correct_flow(s.flow_fw, zero, s.alpha_fw, s.K, vis_fw=s.vis_fw)
    assert not info.ok and info.failure.startswith("pose")
    np.testing.assert_array_equal(out, s.flow_fw)


def test_fallback_keeps_given_base_flow():
    s = scene()
    base = np.full((2, 64, 208), 3.0, np.float32)
    vis = np.zeros_like(s.vis_bw)
    out, info = rigidfill.correct_flow(s.flow_fw, s.flow_bw, s.alpha_fw, s.K, vis_bw=vis,
                                       base_flow=base)
    assert not info.ok
    np.testing.assert_array_equal(out, base)


def test_low_uncertainty_changes_nothing():
    s = scene()
    alpha = np.minimum(s.alpha_fw, 2.75)        # not > tau anywhere; pose set unchanged
    out, info = rigidfill.correct_flow(s.flow_fw, s.flow_bw, alpha, s.K, vis_fw=s.vis_fw,
                                       vis_bw=s.vis_bw, alpha_bw=s.alpha_bw)
    assert info.ok and not info.gate.any()
    np.testing.assert_array_equal(out, s.flow_fw)
