"""Evaluate the controller in the randomized sim on the true ball state; success = ball within 6 cm of the centre at 10 s.

    uv run python -m sim.eval_policy                          # weights/controller.pt, latency 0..4, its training randomization
    uv run python -m sim.eval_policy --latency 2 --nudges 2   # --clean: exact ball state, nominal servos

Same seeds for every checkpoint. The full camera pipeline in sim: python -m perception.estimator --eval.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from sim.ram import check_ram

ROOT = Path(__file__).parent.parent
_G: dict = {}


def _init(cfg):
    import torch

    from sim.env import BalanceEnv
    from sim.policies import load_controller

    torch.set_num_threads(1)
    _G["cfg"] = cfg
    ac = _G["controller"] = load_controller(cfg["controller"])
    ta = {} if cfg["clean"] else ac.train_args
    _G["env"] = BalanceEnv(obs_mode="image", domain_rand=True, light_model=False, grip_rot_deg=cfg["grip_rot"], act_hist=ac.act_hist,
                           max_latency=cfg["max_lat"], servo_rand=ta.get("servo_rand", False), ball_noise=ta.get("ball_noise", 0.0))


def _episode(job):
    from sim.policies import controller_action

    lat, seed = job
    cfg, env = _G["cfg"], _G["env"]
    obs, _ = env.reset(seed=seed)
    env._action_delay = lat
    env._delay_buf = [np.zeros(5)] * lat
    rng = np.random.default_rng(seed + 777)
    nudges = {int(t): rng.uniform(0, 2 * np.pi) for t in rng.integers(30, 280, cfg["nudges"])}
    for t in range(env.max_steps):
        a = controller_action(_G["controller"], env.controller_obs())
        if t in nudges:
            env.d.qvel[env.ball_vadr:env.ball_vadr + 2] += 0.25 * np.array([np.cos(nudges[t]), np.sin(nudges[t])])
        obs, r, te, tr, info = env.step(a)
        if te or tr:
            break
    return dict(lat=lat, seed=seed, success=bool(info["is_success"]), length=t + 1, dist_mm=float(info["dist"] * 1000))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--controller", default=str(ROOT / "weights/controller.pt"))
    p.add_argument("--episodes", type=int, default=30, help="per latency")
    p.add_argument("--latency", default="0,1,2,3,4", help="comma list of control latencies (extra 33 ms steps) to test")
    p.add_argument("--nudges", type=int, default=0, help="random sideways ball kicks per episode")
    p.add_argument("--clean", action="store_true", help="exact ball state and nominal servos instead of the training randomization")
    p.add_argument("--grip-rot", type=float, default=-90.0)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=10_000)
    p.add_argument("--json", default=None, help="also write the per-episode results here")
    a = p.parse_args()
    check_ram(1.5 * 450 * a.workers, "Use fewer --workers.")
    lats = [int(x) for x in str(a.latency).split(",")]
    jobs = [(lat, a.seed + i) for lat in lats for i in range(a.episodes)]
    cfg = dict(controller=a.controller, nudges=a.nudges, grip_rot=a.grip_rot, clean=a.clean, max_lat=max(lats))
    with mp.get_context("spawn").Pool(a.workers, _init, (cfg,)) as pool:
        res = pool.map(_episode, jobs, chunksize=1)
    print(f"\n{a.controller} | {a.episodes} episodes per row | nudges/episode {a.nudges} | {'clean' if a.clean else 'training randomization'}")
    print(f"{'latency':>8} {'success':>9} {'len':>5} {'final dist':>11}")
    for lat in lats + ([None] if len(lats) > 1 else []):
        rs = [r for r in res if lat is None or r["lat"] == lat]
        print(f"{'all' if lat is None else lat:>8} {sum(r['success'] for r in rs):>4}/{len(rs):<4} {np.mean([r['length'] for r in rs]):5.0f} "
              f"{np.mean([r['dist_mm'] for r in rs]):8.1f} mm")
    if a.json:
        Path(a.json).write_text(json.dumps(res))


if __name__ == "__main__":
    main()
