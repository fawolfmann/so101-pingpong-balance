"""Whole-ball candidates, IoU association, coasting, camera-motion compensation; the arm + camera policy env."""
import cv2
import numpy as np

from perception.detector import TRACK_DET, BallDetector, BallTracker, camera_shift, candidates, detect

HW = (192, 256)
ORANGE = (255, 128, 25)


def frame(*discs) -> np.ndarray:
    img = np.full((*HW, 3), 40, np.uint8)
    for cx, cy, r in discs:
        cv2.circle(img, (cx, cy), r, ORANGE, -1)
    return img


def test_candidates_keep_whole_balls_only():
    assert len(candidates(frame((128, 96, 6)), **TRACK_DET)) == 1
    assert candidates(frame((128, 194, 6)), **TRACK_DET) == []  # cut by the bottom border: a sliver
    assert candidates(frame((128, 96, 3)), **TRACK_DET) == []  # too small for the ball at this distance
    img = frame((128, 96, 6))
    cv2.rectangle(img, (20, 20), (60, 28), ORANGE, -1)  # a long stripe
    assert len(candidates(img, **TRACK_DET)) == 1


def test_tracker_follows_the_overlapping_blob_not_the_bigger_one():
    t = BallTracker(HW)
    t.update(candidates(frame((100, 96, 6)), **TRACK_DET))
    det = t.update(candidates(frame((104, 96, 6), (200, 40, 8)), **TRACK_DET))  # ball moved 4 px; a bigger blob elsewhere
    assert det[3] == 1 and abs(det[0] - (104 - 128) / 128) < 0.01


def test_tracker_coasts_on_the_prediction_then_drops():
    t = BallTracker(HW, max_coast=2)
    for x in (100, 104, 108):
        t.update(candidates(frame((x, 96, 6)), **TRACK_DET))
    first = t.update([])
    assert first[3] == 0 and first[0] > (108 - 128) / 128  # predicted further along its motion, flagged as not seen
    t.update([])
    assert np.all(t.update([]) == 0)  # lost for longer than max_coast: no track


def test_camera_turn_is_compensated():
    t = BallTracker(HW)
    det = candidates(frame((128, 96, 6)), **TRACK_DET)
    t.update(det)
    shift = camera_shift(np.array([0.1, 0.0]), 62.0, HW)  # the camera turned: a static ball jumps across the image
    moved = candidates(frame((128 + int(shift[0] * 128), 96, 6)), **TRACK_DET)
    assert t.update(moved, shift)[3] == 1
    t2 = BallTracker(HW, grow=1.0)
    t2.update(det)
    assert t2.update(moved)[3] == 0  # without the compensation the jump breaks the track


def test_cam_policy_env_simulated_and_rendered_agree():
    from sim.cam_policy_env import CamPolicyEnv, obs_dims
    a, full = obs_dims(3, 3)
    dets = []
    for render in (False, True):
        env = CamPolicyEnv(render=render, frames=3, act_hist=3)
        obs, _ = env.reset(seed=4)
        assert obs.shape == (full,) and env.obs_dim == a
        obs, *_ = env.step(np.r_[np.zeros(5), 0.5, 0.0])  # pan only
        dets.append(obs[:4])
    assert dets[0][3] == dets[1][3] == 1
    np.testing.assert_allclose(dets[0][:2], dets[1][:2], atol=0.03)


def test_one_detector_with_a_tracking_switch():
    img = frame((128, 194, 6), (60, 60, 6))  # a cut ball at the bottom border and a whole one
    plain = BallDetector(min_area=20, max_area=2000)
    np.testing.assert_array_equal(plain(img), detect(img, min_area=20, max_area=2000))  # track=False is detect()
    tracked = BallDetector(track=True, hw=HW)
    assert tracked(img)[0] < 0  # only the whole ball (left) is a candidate
