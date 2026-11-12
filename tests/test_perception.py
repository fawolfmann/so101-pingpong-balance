"""Perception for both cameras: detector size limits, env camera choice, estimator with/without joint inputs."""
import cv2
import numpy as np
import torch

from perception import detector
from perception.estimator import CAMERAS, N_DET, Estimator, EstRunner, load_est, with_joint_history
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
    assert CAMERAS["wrist"]["n_in"] == 12 and CAMERAS["ego"]["n_in"] == 27
    m = Estimator(n_in=27)
    path = tmp_path / "e.pt"
    torch.save(dict(model=m.state_dict(), arch=dict(n_in=27), camera="ego"), path)
    loaded = load_est(path)
    assert loaded.n_in == 27 and loaded.camera == "ego"
    run = EstRunner(loaded)
    out = run.update(np.array([0.1, 0.2, 0.8, 1.0], np.float32), joints=np.zeros(5, np.float32))
    assert out.shape == (4,)


def test_joint_history_pairs_each_detection_with_its_pose():
    # two episodes: rows 0-2 (ep 7), rows 3-4 (ep 8); joints of row i are all i
    x = torch.zeros(5, N_DET + 5)
    x[:, N_DET:] = torch.arange(5.0)[:, None]
    h = with_joint_history(x, torch.tensor([7, 7, 7, 8, 8]), 3)
    assert h.shape == (5, N_DET + 15)
    oldest_to_newest = h[:, N_DET::5]  # first joint of each of the 3 frames
    np.testing.assert_array_equal(oldest_to_newest.numpy(), [[0, 0, 0], [0, 0, 1], [0, 1, 2], [3, 3, 3], [3, 3, 4]])


def test_runner_feeds_the_same_history_as_training():
    m = Estimator(n_in=27)
    seen = []
    m.forward = lambda d: (seen.append(d[0].numpy().copy()), torch.zeros(1, 4))[1]
    run = EstRunner(m)
    det = np.array([0.1, 0.2, 0.8, 1.0], np.float32)
    for i in range(3):
        run.update(det, joints=np.full(5, i, np.float32))
    np.testing.assert_array_equal(seen[-1][N_DET::5], [0, 1, 2])
    np.testing.assert_array_equal(seen[1][N_DET::5], [0, 0, 1])


def test_first_ego_version_with_current_joints_still_runs():
    run = EstRunner(Estimator(n_in=17))
    assert run.update(np.array([0.1, 0.2, 0.8, 1.0], np.float32), joints=np.zeros(5, np.float32)).shape == (4,)


def test_wrist_estimator_still_loads_without_joints():
    est = load_est("weights/estimator.pt")  # shipped checkpoint, saved before cameras existed
    assert est.n_in == 12 and est.camera == "wrist"
    assert EstRunner(est).update(np.array([0.1, 0.2, 0.8, 1.0], np.float32)).shape == (4,)


def test_profile_input_sizes():
    sizes = {k: (v["n_aux"], v["n_in"]) for k, v in CAMERAS.items()}
    assert sizes == {"wrist": (0, 12), "ego": (5, 27), "ego_pose": (12, 48), "ego_track": (7, 33)}


def test_tracking_camera_centres_the_ball_and_reports_its_aim():
    from perception.estimator import aux_inputs, make_env, observe
    p = CAMERAS["ego_track"]
    env = make_env("ego_track", seed=0)
    obs, _ = env.reset(seed=3)
    first = observe(env, obs, p)  # aimed at the blade centre; the ball sits a little off it
    for _ in range(4):  # steered only by its own detections
        obs, *_ = env.step(np.zeros(5))
        det = observe(env, obs, p)
    assert det[3] == 1 and np.abs(det[:2]).max() < 0.05 and np.abs(det[:2]).max() < np.abs(first[:2]).max()
    fixed, _ = make_env("ego", seed=0).reset(seed=3)
    assert abs(detector.detect(fixed["image"], **p["det"])[1]) > 0.3  # the fixed camera aims above the blade
    aux = aux_inputs(env, p)
    assert aux.shape == (7,) and np.abs(aux[5:]).max() > 0.1  # pan/tilt moved away from the nominal aim


def test_camera_mount_features_follow_the_randomized_offset():
    from perception.estimator import aux_inputs, make_env
    env = make_env("ego_pose", domain_rand=True, cam_rand=1.0)
    env.reset(seed=5)
    f = aux_inputs(env, CAMERAS["ego_pose"])[5:]
    n = env._nominal
    np.testing.assert_allclose(f[:3] * 0.004, env.m.cam_pos[env.cam_id] - n["cam_pos"], atol=1e-6)
    np.testing.assert_allclose(f[6] * 4.0, env.m.cam_fovy[env.cam_id] - n["cam_fovy"], atol=1e-4)
    assert np.abs(f).max() <= 1.6
    plain = make_env("ego_pose")  # no domain randomization: no offset
    plain.reset(seed=5)
    assert np.all(plain.cam_features() == 0)
