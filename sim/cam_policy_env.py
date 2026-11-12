"""One policy for the arm and the tracking camera: 7 actions (5 joint deltas + pan/tilt rates), sees the ball only through
the camera's detections. Asymmetric actor-critic: the critic also gets the true state (training only).

Observation (actor part, all available on the real robot), newest first:
    [0 : 4F]          last F detections (x, y, r, visible), as perception.detector.detect() reports them
    [4F : 6F]         camera pan/tilt of those frames, relative to the nominal aim, rad / 0.5 (the servo encoders)
    next 10           joint angles - home, / 0.2 rad; joint velocities * 0.1
    next 7 * act_hist past actions
Critic-only extra (appended): true ball (u, w, vu, vw) in controller units, true projection (4), camera mounting offset
(7, scaled), latency / 4; height_info=True (bouncing): + ball height above its resting position / 0.1 m and velocity
along the blade normal / 1 m/s. The actor is the same either way.
Action: [0:5] joint deltas as BalanceEnv; [5:7] pan/tilt rate, x CAM_TRACK_RATE rad per step. The camera command goes
through the same latency as the arm.

render=False: detections are simulated (sim.cam_model: projection + fitted errors), physics speed, for training.
render=True:  the ego camera is rendered and the detector runs (perception.detector.candidates), for evaluation.
Either way the per-frame detection goes through perception.detector.BallTracker (association with the predicted position,
coasting through short losses: visible = 0 with the predicted x, y).
"""
from __future__ import annotations

import numpy as np
import torch

from perception.detector import BallDetector, camera_shift
from sim.cam_model import DetNoise, occluded, project, whole_in_view
from sim.env import CAM_TRACK_RATE, BalanceEnv

EGO_HW = (192, 256)
N_PRIV = 16


def obs_dims(frames: int, act_hist: int, height_info: bool = False) -> tuple[int, int]:
    """(actor obs size, actor + critic-only obs size)."""
    a = 6 * frames + 10 + 7 * act_hist
    return a, a + N_PRIV + 2 * int(height_info)


def expand_critic(sd: dict, frames: int, act_hist: int) -> dict:
    """A checkpoint trained without height_info -> one with it: zero critic weights for the 2 new inputs (appended last).
    The actor is untouched, so the policy acts exactly as before."""
    _, f0 = obs_dims(frames, act_hist)
    w = sd["critic.0.weight"]
    if w.shape[1] != f0:
        raise ValueError(f"critic has {w.shape[1]} inputs, expected {f0} (frames={frames}, act_hist={act_hist})")
    return {**sd, "critic.0.weight": torch.cat([w, w.new_zeros((w.shape[0], 2))], dim=1)}


class CamPolicyEnv:
    act_dim = 7

    def __init__(self, render: bool = False, frames: int = 3, act_hist: int = 3, cam_smooth: float = 0.05,
                 det_noise: DetNoise | None = None, height_info: bool = False, track_gate: float = 0.0, **env_kw):
        self.render, self.frames, self.act_hist, self.cam_smooth = render, frames, act_hist, cam_smooth
        self.noise = det_noise if det_noise is not None else DetNoise.load()
        env_kw.setdefault("grip_rot_deg", -90.0)
        env_kw.pop("ball_noise", None)  # the critic needs the true ball state from controller_obs()
        if render:
            self.env = BalanceEnv(obs_mode="image", camera="ego", cam_track=True, image_hw=EGO_HW, light_model=False, **env_kw)
        else:
            self.env = BalanceEnv(obs_mode="controller", camera="ego", cam_track=True, image_hw=EGO_HW, **env_kw)
        self.height_info = height_info
        self.obs_dim, self.full_dim = obs_dims(frames, act_hist, height_info)
        self.detector = BallDetector(track=True, hw=EGO_HW, gate=track_gate)
        self.rng = np.random.default_rng()

    @property
    def _action_delay(self) -> int:
        return self.env._action_delay

    @property
    def max_steps(self) -> int:
        return self.env.max_steps

    def _sim_candidates(self) -> list:
        proj, frac = project(self.env, with_frac=True)
        hidden = occluded(self.env) or not whole_in_view(proj, frac, EGO_HW)
        det = self.noise.sample(proj, hidden, self._ep, self.rng)
        return [det] if det[3] else []

    def _detect(self, obs, d_aim=(0.0, 0.0)) -> np.ndarray:
        shift = camera_shift(np.asarray(d_aim), self.env._nominal["cam_fovy"], EGO_HW, pitch=float(self.env._aim[1]))
        if self.render:
            self.last_image = obs["image"]  # for viewers
            return self.detector(obs["image"], shift)
        return self.detector.update(self._sim_candidates(), shift)  # simulated detection, same tracker

    def _obs(self) -> np.ndarray:
        e = self.env
        c = e.controller_obs()  # BalanceEnv built without ball_noise: [0:4] is the true ball state
        actor = np.concatenate([np.ravel(self._dets), np.ravel(self._aims), c[4:14], np.ravel(self._acts)])
        priv = np.concatenate([c[:4], project(e), e.cam_features(), [e._action_delay / 4]])
        if self.height_info:
            lift, vn = e.ball_lift()
            priv = np.r_[priv, lift / 0.1, vn]
        return np.concatenate([actor, priv]).astype(np.float32)

    def reset(self, seed: int | None = None):
        obs, info = self.env.reset(seed=seed)
        self.rng = np.random.default_rng(None if seed is None else [seed, 0xCA])
        self._ep = self.noise.new_episode(self.rng)
        self.detector.reset()
        det = self._detect(obs)
        self._dets = np.tile(det, (self.frames, 1))
        self._aims = np.tile(self.env.cam_aim(), (self.frames, 1))
        self._acts = np.zeros((self.act_hist, 7))
        self._cam_buf = [np.zeros(2)] * self.env._action_delay
        self._prev_cam = np.zeros(2)
        return self._obs(), info

    def step(self, action):
        a = np.clip(np.asarray(action, np.float64), -1, 1)
        self._cam_buf.append(a[5:7])
        cam = self._cam_buf.pop(0)  # same latency as the arm (BalanceEnv delays a[:5] itself)
        e = self.env
        e._aim_want = e._aim + cam * CAM_TRACK_RATE  # reached during this step (BalanceEnv rate-limits to the same speed)
        aim0 = e._aim.copy()
        obs, r, te, tr, info = e.step(a[:5])
        r -= self.cam_smooth * float(np.sum((a[5:7] - self._prev_cam) ** 2))
        self._prev_cam = a[5:7]
        det = self._detect(obs, e._aim - aim0)  # the servo encoders give the camera's turn
        self._dets = np.roll(self._dets, 1, axis=0); self._dets[0] = det
        self._aims = np.roll(self._aims, 1, axis=0); self._aims[0] = e.cam_aim()
        self._acts = np.roll(self._acts, 1, axis=0); self._acts[0] = a
        info["det"] = det
        return self._obs(), float(r), te, tr, info
