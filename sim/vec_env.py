"""Parallel BalanceEnvs: n_workers processes, each stepping envs_per_worker envs per pipe message (an env step is only ~0.25 ms,
so one message per env would cost more than the physics).

step() auto-resets finished envs and returns (obs, rew, terminated, truncated, final_obs, episodes):
final_obs = the real last obs of a finished episode (for value bootstrapping at the time limit), episodes = stats of finished episodes.
"""
from __future__ import annotations

import multiprocessing as mp

import numpy as np

from sim.ram import check_ram


def make_train_env(env_kwargs):
    """BalanceEnv in controller mode, or with cam_policy=True the arm + camera env (sim/cam_policy_env.py)."""
    kw = dict(env_kwargs)
    if kw.pop("cam_policy", False):
        from sim.cam_policy_env import CamPolicyEnv
        return CamPolicyEnv(**kw)
    for k in ("frames", "cam_smooth", "height_info", "track_gate", "juggle", "juggle_frac", "juggle_drop"):
        kw.pop(k, None)
    from sim.env import BalanceEnv
    return BalanceEnv(obs_mode="controller", **kw)


def env_dims(env_kwargs) -> tuple[int, int, int]:
    """(obs size, actor obs size, action size) of make_train_env(env_kwargs)."""
    if env_kwargs.get("cam_policy"):
        from sim.cam_policy_env import obs_dims
        actor, full = obs_dims(env_kwargs.get("frames", 3), env_kwargs.get("act_hist", 1), env_kwargs.get("height_info", False),
                               env_kwargs.get("juggle", False))
        return full, actor, 7
    d = 14 + 5 * env_kwargs.get("act_hist", 1)
    return d, d, 5


def _worker(conn, env_kwargs, seeds):
    envs = [make_train_env(env_kwargs) for _ in seeds]
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
                    eps.append(dict(ret=ret[i], length=int(length[i]), success=bool(info["is_success"]), latency=e._action_delay,
                                    kick_z=bool(info.get("kick_z", False)), juggle=bool(info.get("juggle", False)),
                                    juggle_s=info.get("juggle_s", 0.0), n_bounces=info.get("n_bounces", 0),
                                    stalled=bool(info.get("stalled", False))))
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
        self.obs_dim, self.actor_dim, self.act_dim = env_dims(env_kwargs)

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


def _episodic_worker(conn, env_kwargs, n_envs):
    """Like _worker, but resets with given seeds and never auto-resets: a finished env returns its last obs, reward 0
    and done=True until the next reset."""
    from sim.env import BalanceEnv

    envs = [BalanceEnv(obs_mode="controller", **env_kwargs) for _ in range(n_envs)]
    done = np.zeros(n_envs, bool)
    res: list[dict] = [{} for _ in envs]
    last = [None] * n_envs
    while True:
        cmd, data = conn.recv()
        if cmd == "reset":
            obs, kicks = [], []
            for i, (e, s) in enumerate(zip(envs, data)):
                o, _ = e.reset(seed=int(s))
                obs.append(o); last[i] = o
                kicks.append(tuple((t, round(float(v[0]), 9), round(float(v[1]), 9)) for t, v in sorted(e.kicks.items())))
                res[i] = dict(seed=int(s), ret=0.0, length=0, success=False, latency=e._action_delay)
            done[:] = False
            conn.send((np.stack(obs), kicks))
        elif cmd == "step":
            obs, rew = [], []
            for i, (e, a) in enumerate(zip(envs, data)):
                if done[i]:
                    obs.append(last[i]); rew.append(0.0)
                    continue
                o, r, te, tr, info = e.step(a)
                res[i]["ret"] += r; res[i]["length"] += 1
                if te or tr:
                    done[i] = True
                    res[i].update(success=bool(info["is_success"]), dropped=bool(info["fell"] or info["bad_qacc"]))
                obs.append(o); rew.append(r); last[i] = o
            conn.send((np.stack(obs), np.array(rew, np.float32), done.copy()))
        elif cmd == "results":
            conn.send([dict(r) for r in res])
        elif cmd == "close":
            conn.close()
            return


class EpisodicVecEnv:
    """Parallel BalanceEnvs for GRPO: reset(seeds) starts one episode per env from the given seeds (envs sharing a seed
    share the start state, randomization and kicks); step() runs them until every episode has ended (no auto-reset)."""

    def __init__(self, n_workers: int = 6, envs_per_worker: int = 8, **env_kwargs):
        check_ram(1.5 * 135 * n_workers * envs_per_worker, "Use fewer envs (--workers / --envs-per-worker).")
        ctx = mp.get_context("spawn")
        self.n_envs, self.per = n_workers * envs_per_worker, envs_per_worker
        self.conns, self.procs = [], []
        for _ in range(n_workers):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_episodic_worker, args=(child, env_kwargs, envs_per_worker), daemon=True)
            p.start(); child.close()
            self.conns.append(parent); self.procs.append(p)
        self.obs_dim, self.act_dim = 14 + 5 * env_kwargs.get("act_hist", 1), 5

    def _chunks(self, x):
        return [x[i * self.per:(i + 1) * self.per] for i in range(len(self.conns))]

    def reset(self, seeds) -> tuple[np.ndarray, list[tuple]]:
        """Returns (obs, kicks): kicks[i] = env i's kick schedule as ((step, vx, vy), ...)."""
        seeds = list(seeds)
        if len(seeds) != self.n_envs:
            raise ValueError(f"need {self.n_envs} seeds, got {len(seeds)}")
        for c, s in zip(self.conns, self._chunks(seeds)):
            c.send(("reset", s))
        out = [c.recv() for c in self.conns]
        return np.concatenate([o[0] for o in out]), [k for o in out for k in o[1]]

    def step(self, actions) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (obs, reward, done); done stays True for an env once its episode has ended."""
        actions = np.asarray(actions, np.float32).reshape(self.n_envs, self.act_dim)
        for c, a in zip(self.conns, self._chunks(actions)):
            c.send(("step", a))
        out = [c.recv() for c in self.conns]
        return tuple(np.concatenate([o[k] for o in out]) for k in range(3))

    def results(self) -> list[dict]:
        """Per env: seed, return, length, success, dropped, latency of the current episode."""
        for c in self.conns:
            c.send(("results", None))
        return [r for c in self.conns for r in c.recv()]

    def close(self):
        VecBalanceEnv.close(self)
