"""Simulated ball detections for the tracking camera, without rendering: project the ball through the camera, then add
detector-like errors. Lets the camera-steering policy train at physics speed; the noise is fitted against the real
renderer + detector (sim/fit_cam_model.py), and the trained policy is checked with real rendering (sim/eval_cam_policy.py).

project(env)          -> [x, y, r, in_view] as perception.detector.detect() reports them, error free
DetNoise.sample(...)  -> the projection with fitted errors: per-frame jitter, a per-episode bias, missed detections
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

MODEL_PATH = Path(__file__).parent.parent / "outputs" / "cam_noise_ego.json"  # the fitted values are also the defaults below
MIN_AREA_PX = 20  # the ego detector's smallest blob (perception.estimator._EGO_DET)
MIN_FRAC = 0.3  # less of the disc inside the image than this: too thin a sliver for the detector's shape tests
_g = np.linspace(-1, 1, 15)
DISC = np.array([(a, b) for a in _g for b in _g if a * a + b * b <= 1.0])  # sample points on the unit disc


def project(env, with_frac: bool = False):
    """The ball as the detector would see it in env's camera, without errors: x, y in [-1, 1] (right, down), radius in
    px / 10, visible flag. A ball cut by the image border gives the centroid and size of its visible part, as the
    detector does. Uses the camera's true (randomized) pose and FOV, like the rendered image does.
    with_frac: also return the visible fraction of the disc (1 = whole ball in the image)."""
    m, d, cid = env.m, env.d, env.cam_id
    h, w = env.image_hw
    R = d.cam_xmat[cid].reshape(3, 3)
    v = R.T @ (d.xpos[env.ball_body] - d.cam_xpos[cid])
    depth = -v[2]
    none = (np.zeros(4, np.float32), 0.0) if with_frac else np.zeros(4, np.float32)
    if depth <= 1e-3:
        return none
    tv = np.tan(np.deg2rad(m.cam_fovy[cid]) / 2)
    x = v[0] / depth / (tv * w / h)
    y = -v[1] / depth / tv
    r = (h / 2) / tv * env.ball_radius / depth / 10
    rx, ry = r * 10 / (w / 2), r * 10 / (h / 2)  # disc radius in image units
    if abs(x) < 1 - rx and abs(y) < 1 - ry:
        out, frac = np.array([x, y, r, 1.0], np.float32), 1.0
    else:
        pts = np.c_[x + rx * DISC[:, 0], y + ry * DISC[:, 1]]
        inside = (np.abs(pts) < 1).all(1)
        frac = float(inside.mean())
        r_vis = r * np.sqrt(frac)
        if frac < MIN_FRAC or np.pi * (r_vis * 10) ** 2 < MIN_AREA_PX:
            return none
        cx, cy = pts[inside].mean(0)
        out = np.array([cx, cy, r_vis, 1.0], np.float32)
    return (out, frac) if with_frac else out


def whole_in_view(proj: np.ndarray, frac: float, hw: tuple[int, int], margin_px: float = 1.0) -> bool:
    """The whole ball inside the image with a pixel to spare: what perception.detector.candidates() can accept."""
    if proj[3] == 0 or frac < 1:
        return False
    h, w = hw
    return abs(proj[0]) + (proj[2] * 10 + margin_px) / (w / 2) < 1 and abs(proj[1]) + (proj[2] * 10 + margin_px) / (h / 2) < 1


def occluded(env) -> bool:
    """Something (arm, paddle) between the camera and the ball centre."""
    d, cid = env.d, env.cam_id
    p0 = d.cam_xpos[cid].copy()
    vec = d.xpos[env.ball_body] - p0
    dist = float(np.linalg.norm(vec))
    gid = np.array([-1], np.int32)
    hit = mujoco.mj_ray(env.m, d, p0, vec / dist, None, 1, int(env.m.cam_bodyid[cid]), gid)  # skip the camera housing
    return 0 <= hit < dist - 1.5 * env.ball_radius and gid[0] != env.ball_geom


@dataclass
class DetNoise:
    """Detector error model, all in detect() units. Fitted by sim/fit_cam_model.py; defaults are placeholders."""
    bias: list = field(default_factory=lambda: [[0.0] * 4] * 3)  # residual mean = bias @ [1, x, y, r] for x, y, r
    jitter: list = field(default_factory=lambda: [0.003, 0.003, 0.01])  # per-frame std of x, y, r
    ep_bias: list = field(default_factory=lambda: [0.0, 0.0, 0.0])  # per-episode offset std (lighting, colour)
    miss: float = 0.0  # chance of no detection with the ball in view and unoccluded
    miss_edge: float = 0.5  # same, with the ball cut by the image border
    scale: tuple = (0.5, 1.5)  # per-episode multiplier of all error sizes (robustness margin)

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "DetNoise":
        return cls(**json.loads(Path(path).read_text())) if Path(path).exists() else cls()

    def save(self, path: Path = MODEL_PATH) -> None:
        Path(path).write_text(json.dumps(self.__dict__, indent=1))

    def new_episode(self, rng: np.random.Generator) -> dict:
        k = rng.uniform(*self.scale)
        return dict(k=k, off=rng.normal(0, 1, 3) * np.asarray(self.ep_bias) * k)

    def sample(self, proj: np.ndarray, hidden: bool, ep: dict, rng: np.random.Generator, frac: float = 1.0) -> np.ndarray:
        """proj, frac: project(env, with_frac=True); hidden: occluded(env)."""
        if proj[3] == 0 or hidden:
            return np.zeros(4, np.float32)
        if rng.uniform() < (self.miss_edge if frac < 1 else self.miss) * ep["k"]:
            return np.zeros(4, np.float32)
        f = np.array([1.0, proj[0], proj[1], proj[2]])
        out = proj[:3] + np.asarray(self.bias) @ f + ep["off"] + rng.normal(0, 1, 3) * np.asarray(self.jitter) * ep["k"]
        return np.array([*np.clip(out[:2], -1, 1), max(out[2], 0.0), 1.0], np.float32)
