"""Live viewer: one policy moving the arm and the tracking camera, seeing the ball only through the camera.

    uv run mjpython -m sim.view_cam_policy                                   # macOS (mjpython); Linux: python
    uv run mjpython -m sim.view_cam_policy --nudge 2.5 --latency 2 --seconds 15

Every step: render the camera -> whole-ball candidates -> tracker -> policy -> 5 joint moves + pan/tilt. On screen:
  inset (bottom left)  the camera image; green ring = ball seen, yellow = tracker coasting on its prediction
  red ray              the camera's line of sight
  text                 detection state, pan/tilt angles and the policy's camera command
[ and ] switch to the model cameras ("ego" is the policy-steered camera). Close the window to stop.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import mujoco
import mujoco.viewer
import numpy as np

from sim.cam_policy_env import CamPolicyEnv
from sim.policies import controller_action, load_controller
from sim.view_tracking import INSET_SCALE, add_ray

ROOT = Path(__file__).parent.parent
CONTROL_HZ = 30
AFTER_END_S = 1.0


def inset(img: np.ndarray, det: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    out = cv2.resize(img, (w * INSET_SCALE, h * INSET_SCALE), interpolation=cv2.INTER_NEAREST)
    H, W = out.shape[:2]
    cv2.drawMarker(out, (W // 2, H // 2), (255, 255, 255), cv2.MARKER_CROSS, 16, 1)
    if det[2] > 0:  # seen, or coasting on the prediction
        c = (int((det[0] + 1) / 2 * W), int((det[1] + 1) / 2 * H))
        cv2.circle(out, c, int(det[2] * 10 * INSET_SCALE) + 5, (60, 230, 60) if det[3] else (240, 210, 40), 2)
    return np.ascontiguousarray(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default=str(ROOT / "outputs/controller/cam_policy/best.pt"))
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--nudge", type=float, default=0.0, help="push the ball (0.25 m/s) every this many seconds")
    p.add_argument("--latency", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    if not Path(a.policy).exists():
        raise SystemExit(f"missing checkpoint {a.policy}")
    ac = load_controller(a.policy)
    ta = ac.train_args
    env = CamPolicyEnv(render=True, frames=ta.get("frames", 3), act_hist=ta["act_hist"], domain_rand=True, servo_rand=True,
                       cam_rand=0.3, max_latency=max(a.latency, 1), max_steps=round(a.seconds * CONTROL_HZ))
    e = env.env
    rng = np.random.default_rng(a.seed)
    every = round(a.nudge * CONTROL_HZ) if a.nudge > 0 else 0

    def reset(seed: int):
        obs, _ = env.reset(seed=seed)
        e._action_delay, e._delay_buf, env._cam_buf = a.latency, [np.zeros(5)] * a.latency, [np.zeros(2)] * a.latency
        return obs

    seed, episodes, balanced = a.seed, 0, 0
    obs = reset(seed)
    end_step, status, act, det = None, "balancing", np.zeros(7), np.zeros(4)
    with mujoco.viewer.launch_passive(e.m, e.d, show_left_ui=False, show_right_ui=False) as v:
        v.cam.lookat[:] = [0.25, 0.0, 0.45]; v.cam.distance = 1.1; v.cam.azimuth = 135; v.cam.elevation = -20
        with v.lock():
            mujoco.mjv_defaultOption(v.opt)
        while v.is_running():
            t0 = time.time()
            act = controller_action(ac, obs[:ac.actor_dim])
            if end_step is None and every and e.t > 0 and e.t % every == 0:
                ang = rng.uniform(0, 2 * np.pi)
                e.d.qvel[e.ball_vadr:e.ball_vadr + 2] += 0.25 * np.array([np.cos(ang), np.sin(ang)])
            obs, _, te, tr, info = env.step(act)
            det = info["det"]
            img = env.last_image  # the frame the policy's detector just saw

            cid = e.cam_id
            cam_pos, fwd = e.d.cam_xpos[cid], -e.d.cam_xmat[cid].reshape(3, 3)[:, 2]
            with v.lock():
                v.user_scn.ngeom = 0
                add_ray(v.user_scn, cam_pos, cam_pos + fwd * np.linalg.norm(e.d.geom_xpos[e.blade_geom] - cam_pos), (1, 0.2, 0.2, 1))
            vp = v.viewport
            if vp is not None:
                ins = inset(img, det)
                v.set_images((mujoco.MjrRect(vp.left + 10, vp.bottom + 10, ins.shape[1], ins.shape[0]), ins))
            aim = np.rad2deg(e._aim - e._aim0)
            state = "seen" if det[3] else ("coasting" if det[2] > 0 else "lost")
            v.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                          "policy\nepisode\nstatus\ntime\nball\npan / tilt\ncam command\nballs kept",
                          f"{Path(a.policy).parent.name}/{Path(a.policy).name}\n{episodes + 1} (latency {a.latency})\n{status}\n"
                          f"{e.t / CONTROL_HZ:.1f} / {a.seconds:g} s\n{state}\n{aim[0]:+.1f} / {aim[1]:+.1f} deg\n"
                          f"{act[5]:+.2f} / {act[6]:+.2f}\n{balanced}/{episodes}")])
            v.sync()

            if end_step is None and (te or tr):
                episodes += 1; balanced += int(info["is_success"])
                status = "balanced" if info["is_success"] else f"DROPPED at {e.t / CONTROL_HZ:.1f} s"
                print(f"episode {episodes}: {status} | {balanced}/{episodes}", flush=True)
                end_step = e.t + round(AFTER_END_S * CONTROL_HZ)
            if end_step is not None and e.t >= end_step:
                seed += 1; obs = reset(seed); end_step, status = None, "balancing"
            time.sleep(max(0.0, 1 / CONTROL_HZ - (time.time() - t0)))


if __name__ == "__main__":
    main()
