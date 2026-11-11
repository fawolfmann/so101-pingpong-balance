"""Shared pieces for the stress eval and the critic-value analysis: env setup, ball kicks, one recorded episode.

Kicks are BalanceEnv's own (the same code training uses): `kick_speed` m/s of sideways velocity in a random direction,
one per `kick_every` seconds at a random time in each window, after 1 s, drawn from the episode seed. Here every kick
has the same strength and kicks may land until the very end (no recovery margin).
"""
from __future__ import annotations

import multiprocessing as mp

import numpy as np
import torch

from sim.env import CONTROL_HZ, BalanceEnv
from sim.policies import controller_action


def make_env(ac, max_steps: int = 300, light_model: bool | None = None) -> BalanceEnv:
    """The controller's training conditions: domain randomization, latency 0..max, soft servos, noisy ball state."""
    ta = ac.train_args
    return BalanceEnv(obs_mode="controller", domain_rand=True, grip_rot_deg=ta.get("grip_rot", -90.0), act_hist=ac.act_hist,
                      max_latency=ta.get("max_latency", 4), servo_rand=ta.get("servo_rand", False),
                      ball_noise=ta.get("ball_noise", 0.0), max_steps=max_steps,
                      light_model=light_model)


def warning_lead(values: np.ndarray, threshold: float) -> int:
    """Steps of the final run of values below `threshold` (0 if the last value is not below it).
    For an episode that ends in a drop, this is how many steps ahead the critic warned."""
    below = np.asarray(values) < threshold
    if not below.size or not below[-1]:
        return 0
    above = np.flatnonzero(~below)
    return int(below.size - (above[-1] + 1 if above.size else 0))


@torch.no_grad()
def run_episode(env: BalanceEnv, ac, seed: int, seconds: float, kick_speed: float, kick_every: float,
                record: bool = False) -> dict:
    """One episode with the policy mean. Returns success/length/final distance; with record=True also per-step
    critic value, ball distance from the centre and the kick steps."""
    env.max_steps = round(seconds * CONTROL_HZ)
    env.kick_speed, env.kick_every, env.kick_grace, env.kick_margin = (kick_speed, kick_speed), kick_every, 1.0, 0.0
    env.reset(seed=seed)
    kicks = sorted(env.kicks)
    values, dists = [], []
    info: dict = {}
    for t in range(env.max_steps):
        obs = env.controller_obs()
        if record:
            values.append(float(ac.value(torch.as_tensor(obs[None]))[0]))
        _, _, terminated, truncated, info = env.step(controller_action(ac, obs))
        if record:
            dists.append(info["dist"] * 1000)
        if terminated or truncated:
            break
    ep = dict(seed=seed, seconds=seconds, kick_speed=kick_speed, success=bool(info["is_success"]),
              dropped=bool(info.get("fell") or info.get("bad_qacc")), length=t + 1, dist_mm=float(info["dist"] * 1000))
    if record:
        ep.update(value=values, dist_trace=dists, kicks=kicks)
    return ep


# ---- process pool: one env + controller per worker ----
_G: dict = {}


def _init(controller_path: str) -> None:
    from sim.policies import load_controller

    torch.set_num_threads(1)
    _G["ac"] = load_controller(controller_path)
    _G["env"] = make_env(_G["ac"])


def _job(kw: dict) -> dict:
    return run_episode(_G["env"], _G["ac"], **kw)


def run_parallel(controller_path: str, jobs: list[dict], workers: int) -> list[dict]:
    """Run run_episode(**job) for each job across `workers` processes (spawn, so it works on macOS)."""
    with mp.get_context("spawn").Pool(workers, _init, (controller_path,)) as pool:
        return pool.map(_job, jobs, chunksize=1)
