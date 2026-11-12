"""Live viewer: the arm balancing from a camera alone (tracking or fixed ego camera), with the camera's own view inset.

    uv run mjpython -m sim.view_tracking                                    # macOS (mjpython); Linux: python
    uv run mjpython -m sim.view_tracking --est outputs/estimator/ego_dc.pt  # fixed ego camera, for comparison
    uv run mjpython -m sim.view_tracking --nudge 2 --latency 2

The full vision loop runs every step: render the camera -> detect the ball -> (tracking camera) steer pan/tilt toward the
detection -> estimator -> controller. The controller never sees the true ball state. On screen:
  inset (bottom left)  what the camera sees, with the detection (green ring) and the image centre (white cross)
  red ray              the camera's line of sight, from the post top
  green sphere         the estimated ball position, drawn on the paddle
  text                 estimate error vs the true ball, pan/tilt angles
[ and ] switch to the model cameras: the "ego" camera turns with the tracker. Close the window to stop.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import mujoco
import mujoco.viewer
import numpy as np

from perception.estimator import CAMERAS, EstRunner, aux_inputs, load_est, make_env, observe
from sim.env import FdVel
from sim.policies import controller_action, load_controller

ROOT = Path(__file__).parent.parent
CONTROL_HZ = 30
AFTER_END_S = 1.0
INSET_SCALE = 2  # camera image upscaled this much in the inset


def inset_image(img: np.ndarray, det: np.ndarray) -> np.ndarray:
    """Camera frame with the detection ring and image-centre cross, upscaled (set_images takes top-down rows)."""
    h, w = img.shape[:2]
    out = cv2.resize(img, (w * INSET_SCALE, h * INSET_SCALE), interpolation=cv2.INTER_NEAREST)
    H, W = out.shape[:2]
    cv2.drawMarker(out, (W // 2, H // 2), (255, 255, 255), cv2.MARKER_CROSS, 16, 1)
    if det[3]:  # det: x, y in [-1, 1] (x right, y down), blob radius in px / 10
        c = (int((det[0] + 1) / 2 * W), int((det[1] + 1) / 2 * H))
        cv2.circle(out, c, int(det[2] * 10 * INSET_SCALE) + 5, (60, 230, 60), 2)
    return np.ascontiguousarray(out)


def add_sphere(scn: mujoco.MjvScene, pos: np.ndarray, radius: float, rgba) -> None:
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, radius), pos, np.eye(3).ravel(),
                        np.asarray(rgba, np.float32))
    scn.ngeom += 1


def add_ray(scn: mujoco.MjvScene, a: np.ndarray, b: np.ndarray, rgba) -> None:
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, 3.0, a, b)
    scn.ngeom += 1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--est", default=str(ROOT / "outputs/estimator/ego_track_dc.pt"), help="camera estimator (its profile picks the camera)")
    p.add_argument("--controller", default=str(ROOT / "outputs/weights/ppo_kicks_smooth.pt"))
    p.add_argument("--seconds", type=float, default=10.0, help="episode length")
    p.add_argument("--nudge", type=float, default=0.0, help="push the ball (0.25 m/s, random direction) every this many seconds")
    p.add_argument("--latency", type=int, default=0, help="control latency, steps of 33 ms")
    p.add_argument("--cam-rand", type=float, default=0.3, help="camera mounting error scale (as in the evaluation)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    for f in (a.est, a.controller):
        if not Path(f).exists():
            raise SystemExit(f"missing checkpoint {f}")

    ac = load_controller(a.controller)
    model = load_est(a.est)
    cam_name = next((k for k, v in CAMERAS.items() if v["aux"] and v["n_in"] == model.n_in), model.camera)
    prof = CAMERAS[cam_name]
    env = make_env(cam_name, domain_rand=True, act_hist=ac.act_hist, max_latency=max(a.latency, 1), servo_rand=True,
                   cam_rand=a.cam_rand, max_steps=round(a.seconds * CONTROL_HZ))
    est, fd = EstRunner(model), FdVel()
    rng = np.random.default_rng(a.seed)
    every = round(a.nudge * CONTROL_HZ) if a.nudge > 0 else 0

    def reset(seed: int):
        obs, _ = env.reset(seed=seed)
        env._action_delay, env._delay_buf = a.latency, [np.zeros(5)] * a.latency
        est.reset(); fd.reset()
        return obs

    seed, episodes, balanced = a.seed, 0, 0
    obs = reset(seed)
    end_step, status, err = None, "balancing", np.zeros(2)
    with mujoco.viewer.launch_passive(env.m, env.d, show_left_ui=False, show_right_ui=False) as v:
        v.cam.lookat[:] = [0.25, 0.0, 0.45]; v.cam.distance = 1.1; v.cam.azimuth = 135; v.cam.elevation = -20
        with v.lock():
            mujoco.mjv_defaultOption(v.opt)
        while v.is_running():
            t0 = time.time()
            img = obs["image"]
            det = observe(env, obs, prof)  # a tracking camera also gets its pan/tilt target here
            tobs = env.controller_obs()
            b = est.update(det, joints=aux_inputs(env, prof))
            b = fd.update(b[:2])
            err = 0.9 * err + 0.1 * np.abs(b[:2] - tobs[:2]) * env.blade_radius * 100  # smoothed, cm
            tobs[:4] = b
            act = controller_action(ac, tobs)
            if end_step is None and every and env.t > 0 and env.t % every == 0:
                ang = rng.uniform(0, 2 * np.pi)
                env.d.qvel[env.ball_vadr:env.ball_vadr + 2] += 0.25 * np.array([np.cos(ang), np.sin(ang)])
            obs, _, te, tr, info = env.step(act)

            # overlays: line of sight, estimated ball on the paddle, camera inset, text
            c, n, uu, ww = env.blade_frame()
            est_pos = c + (b[0] * uu + b[1] * ww) * env.blade_radius + n * env.ball_radius
            cid = env.cam_id
            cam_pos, fwd = env.d.cam_xpos[cid], -env.d.cam_xmat[cid].reshape(3, 3)[:, 2]
            with v.lock():
                v.user_scn.ngeom = 0
                add_ray(v.user_scn, cam_pos, cam_pos + fwd * np.linalg.norm(env.d.geom_xpos[env.blade_geom] - cam_pos), (1, 0.2, 0.2, 1))
                add_sphere(v.user_scn, cam_pos, 0.012, (1, 0.2, 0.2, 1))
                add_sphere(v.user_scn, est_pos, env.ball_radius * 0.6, (0.2, 0.9, 0.3, 0.8))
            vp = v.viewport
            if vp is not None:
                ins = inset_image(img, det)
                v.set_images((mujoco.MjrRect(vp.left + 10, vp.bottom + 10, ins.shape[1], ins.shape[0]), ins))
            aim = (np.rad2deg(env._aim - env._aim0) if env.cam_track else np.zeros(2))
            v.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                          "camera\nepisode\nstatus\ntime\nest error u / w\npan / tilt\nballs kept",
                          f"{cam_name} ({Path(a.est).name})\n{episodes + 1} (latency {a.latency})\n{status}\n"
                          f"{env.t / CONTROL_HZ:.1f} / {a.seconds:g} s\n{err[0]:.2f} / {err[1]:.2f} cm\n"
                          f"{aim[0]:+.1f} / {aim[1]:+.1f} deg\n{balanced}/{episodes}")])
            v.sync()

            if end_step is None and (te or tr):
                episodes += 1; balanced += int(info["is_success"])
                status = "balanced" if info["is_success"] else f"DROPPED at {env.t / CONTROL_HZ:.1f} s"
                print(f"episode {episodes}: {status} | {balanced}/{episodes}", flush=True)
                end_step = env.t + round(AFTER_END_S * CONTROL_HZ)
            if end_step is not None and env.t >= end_step:
                seed += 1; obs = reset(seed); end_step, status = None, "balancing"
            time.sleep(max(0.0, 1 / CONTROL_HZ - (time.time() - t0)))


if __name__ == "__main__":
    main()
