"""Perception for both cameras: detector size limits, env camera choice, estimator with/without joint inputs."""
import cv2
import numpy as np
import torch

from perception import detector
from perception.estimator import CAMERAS, Estimator, EstRunner, load_est
from sim.env import BalanceEnv


def disc(radius: int, hw=(192, 256)) -> np.ndarray:
    img = np.full((*hw, 3), 40, np.uint8)
    cv2.circle(img, (hw[1] // 2, hw[0] // 2), radius, (255, 128, 25), -1)  # the ball's orange
    return img


def test_detector_defaults_unchanged_and_limits_configurable():
    small = disc(3)  # ~28 px: below the wrist default minimum of 40
    assert detector.detect(small)[3] == 0
    assert detector.detect(small, min_area=10)[3] == 1
    big = disc(8)
    np.testing.assert_array_equal(detector.detect(big), detector.detect(big, detector.MIN_AREA, detector.MAX_AREA))


def test_round_selection_prefers_the_disc_over_a_bigger_box():
    img = np.full((192, 256, 3), 40, np.uint8)
    cv2.circle(img, (60, 60), 6, (255, 128, 25), -1)            # ball: ~113 px, round
    cv2.rectangle(img, (150, 120), (157, 135), (255, 128, 25), -1)  # handle-like stub: 8 x 16 = 128 px, elongated
    assert detector.detect(img, min_area=20, max_area=2000)[0] > 0       # largest blob: the box (right side)
    assert detector.detect(img, min_area=20, max_area=2000, select="round")[0] < 0  # roundest: the ball (left side)


def test_env_renders_the_chosen_camera_at_the_chosen_size():
    p = CAMERAS["ego"]
    env = BalanceEnv(obs_mode="image", camera="ego", image_hw=p["image_hw"], light_model=False, grip_rot_deg=-90)
    obs, _ = env.reset(seed=0)
    assert obs["image"].shape == (*p["image_hw"], 3)
    assert detector.detect(obs["image"], **p["det"])[3] == 1  # ball on the paddle is seen
    assert BalanceEnv(obs_mode="image", grip_rot_deg=-90).cam_id == env.m.camera("wrist").id  # default unchanged


def test_estimator_inputs_per_camera(tmp_path):
    assert CAMERAS["wrist"]["n_in"] == 12 and CAMERAS["ego"]["n_in"] == 17
    m = Estimator(n_in=17)
    path = tmp_path / "e.pt"
    torch.save(dict(model=m.state_dict(), arch=dict(n_in=17), camera="ego"), path)
    loaded = load_est(path)
    assert loaded.n_in == 17 and loaded.camera == "ego"
    run = EstRunner(loaded)
    out = run.update(np.array([0.1, 0.2, 0.8, 1.0], np.float32), joints=np.zeros(5, np.float32))
    assert out.shape == (4,)


def test_wrist_estimator_still_loads_without_joints():
    est = load_est("weights/estimator.pt")  # shipped checkpoint, saved before cameras existed
    assert est.n_in == 12 and est.camera == "wrist"
    assert EstRunner(est).update(np.array([0.1, 0.2, 0.8, 1.0], np.float32)).shape == (4,)
