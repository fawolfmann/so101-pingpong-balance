"""Shared pieces for the stress eval and the critic-value analysis: env setup, ball kicks, one recorded episode.

A kick adds `kick_speed` m/s of sideways velocity to the ball, in a random direction (same as eval_policy's nudges,
which use 0.25 m/s). Kicks come one per `kick_every` seconds, at a random time inside each window, after a grace period.
"""
from __future__ import annotations

import multiprocessing as mp

import numpy as np
import torch

from sim.env import BalanceEnv
from sim.policies import controller_action

CONTROL_HZ = 30


def make_env(ac, max_steps: int = 300, light_model: bool | None = None) -> BalanceEnv:
    """The controller's training conditions: domain randomization, latency 0..max, soft servos, noisy ball state."""
    ta = ac.train_args
    return BalanceEnv(obs_mode="controller", domain_rand=True, grip_rot_deg=ta.get("grip_rot", -90.0), act_hist=ac.act_hist,
                      max_latency=ta.get("max_latency", 4), servo_rand=ta.get("servo_rand", False),
                      ball_noise=ta.get("ball_noise", 0.0), max_steps=max_steps,
                      light_model=light_model)


def kick_schedule(rng: np.random.Generator, n_steps: int, every_s: float, grace_s: float = 1.0) -> dict[int, float]:
    """{step: direction angle}: one kick at a random step inside each `every_s` window after `grace_s`."""
    if every_s <= 0:
        return {}
    start, window = round(grace_s * CONTROL_HZ), max(1, round(every_s * CONTROL_HZ))
    return {int(rng.integers(lo, min(lo + window, n_steps))): float(rng.uniform(0, 2 * np.pi))
            for lo in range(start, n_steps, window)}


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
    env.reset(seed=seed)
    kicks = kick_schedule(np.random.default_rng(seed + 777), env.max_steps, kick_every) if kick_speed > 0 else {}
    values, dists = [], []
    info: dict = {}
    for t in range(env.max_steps):
        obs = env.controller_obs()
        if record:
            values.append(float(ac.value(torch.as_tensor(obs[None]))[0]))
        if t in kicks:
            env.d.qvel[env.ball_vadr:env.ball_vadr + 2] += kick_speed * np.array([np.cos(kicks[t]), np.sin(kicks[t])])
        _, _, terminated, truncated, info = env.step(controller_action(ac, obs))
        if record:
            dists.append(info["dist"] * 1000)
        if terminated or truncated:
            break
    ep = dict(seed=seed, seconds=seconds, kick_speed=kick_speed, success=bool(info["is_success"]),
              dropped=bool(info.get("fell") or info.get("bad_qacc")), length=t + 1, dist_mm=float(info["dist"] * 1000))
    if record:
        ep.update(value=values, dist_trace=dists, kicks=sorted(kicks))
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
