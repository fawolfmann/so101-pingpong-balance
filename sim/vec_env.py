"""Parallel BalanceEnvs: n_workers processes, each stepping envs_per_worker envs per pipe message (an env step is only ~0.25 ms,
so one message per env would cost more than the physics).

step() auto-resets finished envs and returns (obs, rew, terminated, truncated, final_obs, episodes):
final_obs = the real last obs of a finished episode (for value bootstrapping at the time limit), episodes = stats of finished episodes.
"""
from __future__ import annotations

import multiprocessing as mp

import numpy as np

from sim.ram import check_ram


def _worker(conn, env_kwargs, seeds):
    from sim.env import BalanceEnv

    envs = [BalanceEnv(obs_mode="controller", **env_kwargs) for _ in seeds]
    ret = np.zeros(len(envs))
    length = np.zeros(len(envs), int)
    for e, s in zip(envs, seeds):
        e.reset(seed=s)
    while True:
        cmd, data = conn.recv()
        if cmd == "reset":
            conn.send(np.stack([e.reset()[0] for e in envs]))
        elif cmd == "step":
            obs, rew, term, trunc, final, eps = [], [], [], [], [], []
            for i, (e, a) in enumerate(zip(envs, data)):
                o, r, te, tr, info = e.step(a)
                ret[i] += r
                length[i] += 1
                fo = np.zeros_like(o)
                if te or tr:
                    fo = o
                    eps.append(dict(ret=ret[i], length=int(length[i]), success=bool(info["is_success"]), latency=e._action_delay))
                    ret[i], length[i] = 0.0, 0
                    o, _ = e.reset()
                obs.append(o); rew.append(r); term.append(te); trunc.append(tr); final.append(fo)
            conn.send((np.stack(obs), np.array(rew, np.float32), np.array(term), np.array(trunc), np.stack(final), eps))
        elif cmd == "close":
            conn.close()
            return


class VecBalanceEnv:
    def __init__(self, n_workers: int = 6, envs_per_worker: int = 4, seed: int = 0, **env_kwargs):
        # ~135 MB per env (measured): refuse to start rather than run out of RAM
        check_ram(1.5 * 135 * n_workers * envs_per_worker, "Use fewer envs (--workers / --envs-per-worker) and raise --steps to keep the batch size, see the README.")
        ctx = mp.get_context("spawn")  # fork is unsafe with torch/CUDA in the parent
        self.n_envs = n_workers * envs_per_worker
        self.conns, self.procs = [], []
        for w in range(n_workers):
            parent, child = ctx.Pipe()
            seeds = [seed + w * envs_per_worker + k for k in range(envs_per_worker)]
            p = ctx.Process(target=_worker, args=(child, env_kwargs, seeds), daemon=True)
            p.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(p)
        self.per = envs_per_worker
        self.obs_dim, self.act_dim = 14 + 5 * env_kwargs.get("act_hist", 1), 5

    def reset(self):
        for c in self.conns:
            c.send(("reset", None))
        return np.concatenate([c.recv() for c in self.conns])

    def step(self, actions):
        actions = np.asarray(actions, np.float32).reshape(self.n_envs, self.act_dim)
        for i, c in enumerate(self.conns):  # send all first, so workers run in parallel
            c.send(("step", actions[i * self.per : (i + 1) * self.per]))
        out = [c.recv() for c in self.conns]
        obs, rew, term, trunc, final = (np.concatenate([o[k] for o in out]) for k in range(5))
        episodes = [ep for o in out for ep in o[5]]
        return obs, rew, term, trunc, final, episodes

    def close(self):
        for c in self.conns:
            try:
                c.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=5)
