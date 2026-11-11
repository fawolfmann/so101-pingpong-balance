"""Train the controller with GRPO (no critic): groups of episodes from shared seeds, group-relative advantages.

    uv run python -m sim.train_grpo --name grpo_kicks --total-steps 29e6 --act-hist 3 --grip-rot -90 --max-latency 4 \\
        --home-penalty 0.1 --servo-rand --ball-noise 1.0 --episode-s 4 --kick-max 0.3 --kick-every 2

Each iteration: P = envs / group-size seeds, G episodes per seed, all run to the end; advantage per episode =
(return - group mean) / group std, given to every step; then the clipped update (sim/grpo.py).
Env flags match train_controller.py, so a PPO and a GRPO run can use identical conditions.
Writes outputs/controller/<name>/: args.json, log.csv, best.pt (best recent success), last.pt.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from sim.env import CONTROL_HZ
from sim.grpo import group_advantages, grpo_update
from sim.ppo import ActorCritic
from sim.train_controller import save
from sim.vec_env import EpisodicVecEnv

OUT = Path(__file__).parent.parent / "outputs" / "controller"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="grpo_dev")
    p.add_argument("--iters", type=int, default=500, help="iterations (ignored if --total-steps is set)")
    p.add_argument("--total-steps", type=float, default=None, help="stop after this many env steps (match a PPO budget)")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--envs-per-worker", type=int, default=8)
    p.add_argument("--group-size", type=int, default=8, help="G episodes per seed")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=8)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--kl-coef", type=float, default=0.0, help="KL penalty toward a reference policy (0 = off)")
    p.add_argument("--ref-every", type=int, default=50, help="refresh the reference policy every N iterations")
    # env: same flags and defaults as train_controller.py
    p.add_argument("--no-domain-rand", action="store_true")
    p.add_argument("--max-latency", type=int, default=2)
    p.add_argument("--act-hist", type=int, default=1)
    p.add_argument("--home-penalty", type=float, default=0.0)
    p.add_argument("--servo-rand", action="store_true")
    p.add_argument("--ball-noise", type=float, default=0.0)
    p.add_argument("--grip-rot", type=float, default=0.0)
    p.add_argument("--smooth-penalty", type=float, default=0.01, help="reward -= this * sum((a - previous a)^2): discourages jitter")
    p.add_argument("--episode-s", type=float, default=10.0)
    p.add_argument("--kick-min", type=float, default=0.0)
    p.add_argument("--kick-max", type=float, default=0.0)
    p.add_argument("--kick-every", type=float, default=4.0)
    p.add_argument("--kick-margin", type=float, default=1.0)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    if (a.workers * a.envs_per_worker) % a.group_size or a.group_size < 2:
        p.error("--group-size must be >= 2 and divide --workers x --envs-per-worker")
    if not 0.0 <= a.kick_min <= a.kick_max:
        p.error("need 0 <= --kick-min <= --kick-max")
    return a


@torch.no_grad()
def rollout(env: EpisodicVecEnv, ac: ActorCritic, seeds: list[int]) -> tuple[dict, list[dict]]:
    """Run one episode per env to the end. Returns the per-step batch (env index per row) and per-env results."""
    obs, _ = env.reset(seeds)
    rows = dict(obs=[], act=[], logp=[], env=[])
    active = np.ones(env.n_envs, bool)
    while active.any():
        act, logp, _ = ac.act(torch.as_tensor(obs))
        idx = np.flatnonzero(active)
        rows["obs"].append(obs[idx]); rows["act"].append(act.numpy()[idx]); rows["logp"].append(logp.numpy()[idx])
        rows["env"].append(idx)
        obs, _, done = env.step(act.numpy())
        active &= ~done
    return {k: np.concatenate(v) for k, v in rows.items()}, env.results()


def main() -> None:
    a = parse_args()
    torch.manual_seed(a.seed)
    torch.set_num_threads(2)
    run_dir = OUT / a.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(a), indent=1))

    env = EpisodicVecEnv(a.workers, a.envs_per_worker, domain_rand=not a.no_domain_rand, max_latency=a.max_latency,
                         act_hist=a.act_hist, grip_rot_deg=a.grip_rot, home_penalty=a.home_penalty, servo_rand=a.servo_rand,
                         ball_noise=a.ball_noise, max_steps=round(a.episode_s * CONTROL_HZ), kick_min=a.kick_min,
                         kick_max=a.kick_max, kick_every=a.kick_every, kick_margin=a.kick_margin,
                         smooth_penalty=a.smooth_penalty)
    G, P = a.group_size, env.n_envs // a.group_size
    ac = ActorCritic(env.obs_dim, env.act_dim)
    opt = torch.optim.Adam(list(ac.actor.parameters()) + [ac.log_std], lr=a.lr, eps=1e-5)
    ref = copy.deepcopy(ac).eval() if a.kl_coef > 0 else None
    budget = a.total_steps or float("inf")
    max_iters = a.iters if a.total_steps is None else 10**9

    recent = deque(maxlen=200)
    total, best, t0, writer = 0, -1.0, time.time(), None
    with open(run_dir / "log.csv", "w", newline="") as csv_f:
        for it in range(1, max_iters + 1):
            frac = min(total / budget, (it - 1) / max_iters) if a.total_steps else (it - 1) / max_iters
            for g in opt.param_groups:  # linear lr decay to 0 over the run
                g["lr"] = a.lr * (1 - frac)
            seeds = [a.seed * 10**7 + (it - 1) * P + p for p in range(P) for _ in range(G)]
            rows, res = rollout(env, ac, seeds)
            adv_ep = group_advantages(np.array([r["ret"] for r in res]), G)
            batch = dict(obs=torch.as_tensor(rows["obs"]), act=torch.as_tensor(rows["act"]),
                         logp=torch.as_tensor(rows["logp"]), adv=torch.as_tensor(adv_ep[rows["env"]]))
            s = grpo_update(ac, opt, batch, epochs=a.epochs, minibatches=a.minibatches, clip=a.clip, kl_coef=a.kl_coef, ref=ref)
            if ref is not None and it % a.ref_every == 0:
                ref.load_state_dict(ac.state_dict())
            total += len(rows["obs"])
            recent.extend(res)

            no_signal = float(np.mean(np.abs(adv_ep.reshape(P, G)).sum(1) == 0))
            row = dict(iter=it, steps=total, fps=int(total / (time.time() - t0)), std=float(ac.log_std.detach().exp().mean()),
                       ep_return=float(np.mean([r["ret"] for r in recent])), ep_len=float(np.mean([r["length"] for r in recent])),
                       success=float(np.mean([r["success"] for r in recent])), no_signal_groups=no_signal, **s)
            for L in range(a.max_latency + 1):
                es = [r["success"] for r in recent if r["latency"] == L]
                row[f"success_lat{L}"] = float(np.mean(es)) if es else float("nan")
            if writer is None:
                writer = csv.DictWriter(csv_f, fieldnames=list(row)); writer.writeheader()
            writer.writerow(row); csv_f.flush()
            if it % 10 == 0 or it == 1:
                print(f"it {it:5d} steps {total / 1e6:6.2f}M fps {row['fps']:5d} | ret {row['ep_return']:6.1f} "
                      f"len {row['ep_len']:5.0f} succ {row['success']:.2f} | kl {s['kl']:.4f} clip {s['clipfrac']:.2f} "
                      f"std {row['std']:.2f} no-signal groups {no_signal:.2f}", flush=True)
            done = total >= budget or it == max_iters
            if it % a.save_every == 0 or done:
                ck = dict(model=ac.state_dict(), args=vars(a), iter=it, steps=total)
                if len(recent) >= 100 and row["success"] >= best:
                    best = row["success"]; save(ck, run_dir / "best.pt")
                save(ck, run_dir / "last.pt")
            if done:
                break
    env.close()
    print(f"done: {total / 1e6:.2f}M steps in {(time.time() - t0) / 60:.1f} min, checkpoints in {run_dir}")


if __name__ == "__main__":
    main()
