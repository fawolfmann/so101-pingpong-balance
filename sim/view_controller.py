"""Watch the trained PPO controller balance the ball in the live MuJoCo viewer.
Usage: uv run python -m sim.view_controller [--controller weights/controller.pt] [--rand] [--latency N] [--seed N] [--nudge SECONDS] [--seconds S]
  --rand      domain randomization on, with the settings the controller was trained with (latency, soft servos, noisy ball state)
  --latency   force a control latency of N extra steps (33 ms each)
  --seconds S episode length in seconds (default 10, the training length)
  --nudge S   kick the ball sideways every S seconds (you can also Ctrl+right-drag the ball with the mouse)
The policy only gets the controller observation (ball state in the blade frame + joints + recent actions); the actions are the policy mean."""
import argparse
import time
from pathlib import Path

import mujoco.viewer
import numpy as np

from sim.env import BalanceEnv
from sim.policies import controller_action, load_controller

p = argparse.ArgumentParser()
p.add_argument("--controller", default=str(Path(__file__).parent.parent / "weights/controller.pt")); p.add_argument("--rand", action="store_true")
p.add_argument("--latency", type=int, default=None); p.add_argument("--seed", type=int, default=0); p.add_argument("--nudge", type=float, default=0.0)
p.add_argument("--grip-rot", type=float, default=-90.0); p.add_argument("--seconds", type=float, default=10.0, help="episode length (s)")
a = p.parse_args()

ac = load_controller(a.controller)
ta = ac.train_args if a.rand else {}
env = BalanceEnv(obs_mode="controller", domain_rand=a.rand, light_model=False, grip_rot_deg=a.grip_rot, act_hist=ac.act_hist,  # full visual model so the arm and paddle are drawn
                 max_latency=ta.get("max_latency", 2), servo_rand=ta.get("servo_rand", False), ball_noise=ta.get("ball_noise", 0.0),
                 max_steps=round(a.seconds * 30))  # 30 Hz control
seed = a.seed
obs, _ = env.reset(seed=seed)
if a.latency is not None:
    env._action_delay = a.latency; env._delay_buf = [np.zeros(5)] * a.latency
print(f"loaded {a.controller}; randomization {'on' if a.rand else 'off'}; latency {env._action_delay}")
rng = np.random.default_rng(seed)
with mujoco.viewer.launch_passive(env.m, env.d) as v:
    v.cam.lookat[:] = [0.41, 0.0, 0.30]; v.cam.distance = 0.9; v.cam.azimuth = 150; v.cam.elevation = -25
    last_nudge = time.time(); episodes = successes = 0
    while v.is_running():
        t0 = time.time()
        obs, r, te, tr, info = env.step(controller_action(ac, obs))
        if a.nudge and time.time() - last_nudge > a.nudge:
            ang = rng.uniform(0, 2 * np.pi); env.d.qvel[env.ball_vadr:env.ball_vadr + 2] += 0.25 * np.array([np.cos(ang), np.sin(ang)]); last_nudge = time.time()
        v.sync()
        if te or tr:
            episodes += 1; successes += int(info["is_success"])
            print(f"episode {episodes}: {'success' if info['is_success'] else 'FAILED'} (len {env.t}, final dist {info['dist']*1000:.1f} mm) | {successes}/{episodes}")
            seed += 1; obs, _ = env.reset(seed=seed)
            if a.latency is not None:
                env._action_delay = a.latency; env._delay_buf = [np.zeros(5)] * a.latency
        time.sleep(max(0, env.n_substeps * env.m.opt.timestep - (time.time() - t0)))
