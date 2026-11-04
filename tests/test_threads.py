"""threads > 1 must give exactly the result of one thread (rigidfill/geometry.py)."""
import numpy as np
import pytest

import rigidfill
from rigidfill.geometry import correspondences, estimate_pose, fit_geometry, triangulate
from rigidfill.synthetic import make_scene


@pytest.fixture(scope="module", params=[0.0, 0.3])
def scene(request):
    # noise 0.3 px: RANSAC has outliers to reject and the triangulation is not exact
    return make_scene(noise_px=request.param, seed=1)


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


@pytest.mark.parametrize("threads", [2, 3, 8])
def test_fit_geometry_bit_identical(scene, threads):
    kw = dict(vis_fw=scene.vis_fw, vis_bw=scene.vis_bw, alpha_fw=scene.alpha_fw,
              alpha_bw=scene.alpha_bw)
    Kg = rigidfill.scale_intrinsics(scene.K, scene.image_size, scene.flow_fw.shape[1:])
    g1 = fit_geometry(scene.flow_fw, scene.flow_bw, Kg, **kw)
    gn = fit_geometry(scene.flow_fw, scene.flow_bw, Kg, threads=threads, **kw)
    for f in ("R_fw", "t_fw", "R_bw", "t_bw", "depth_bw"):
        assert same(getattr(g1, f), getattr(gn, f)), f
    assert (g1.scale, g1.n_inliers_bw, g1.n_inliers_fw, g1.n_both) == \
        (gn.scale, gn.n_inliers_bw, gn.n_inliers_fw, gn.n_both)


@pytest.mark.parametrize("threads", [4])
def test_correct_flow_bit_identical(scene, threads):
    def run(**kw):
        return rigidfill.correct_flow(scene.flow_fw, scene.flow_bw, scene.alpha_fw, scene.K,
                                      scene.image_size, vis_fw=scene.vis_fw,
                                      vis_bw=scene.vis_bw, alpha_bw=scene.alpha_bw, **kw)
    o1, i1 = run()
    on, i_n = run(threads=threads)
    assert i1.ok and i_n.ok
    assert same(o1, on) and same(i1.gate, i_n.gate) and same(i1.rigid_flow, i_n.rigid_flow)


def test_pose_and_triangulation_parts(scene):
    Kg = rigidfill.scale_intrinsics(scene.K, scene.image_size, scene.flow_fw.shape[1:])
    c = correspondences(scene.flow_bw, scene.vis_bw, scene.alpha_bw)
    p1, p2 = c.points(c.pose_set)
    R1, t1, n1 = estimate_pose(p1, p2, Kg)
    R4, t4, n4 = estimate_pose(p1, p2, Kg, threads=4)
    assert same(R1, R4) and same(t1, t4) and n1 == n4
    q1, q2 = c.points(c.usable)
    assert same(triangulate(q1, q2, Kg, R1, t1), triangulate(q1, q2, Kg, R1, t1, threads=5))


def test_failure_is_the_same(scene):
    # a still camera: no pose; the backward failure is reported first with any thread count
    still = np.zeros_like(scene.flow_fw)
    Kg = rigidfill.scale_intrinsics(scene.K, scene.image_size, scene.flow_fw.shape[1:])
    msgs = []
    for threads in (1, 4):
        with pytest.raises(rigidfill.RigidFillError) as e:
            fit_geometry(still, still, Kg, threads=threads)
        msgs.append(str(e.value))
    assert msgs[0] == msgs[1]
