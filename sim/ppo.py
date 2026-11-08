"""PPO in plain PyTorch: actor-critic networks, GAE and the clipped update. The training loop is in train_controller.py."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def mlp(inp: int, out: int, hidden: int = 256, layers: int = 2, out_gain: float = 1.0) -> nn.Sequential:
    mods, d = [], inp
    for _ in range(layers):
        lin = nn.Linear(d, hidden)
        nn.init.orthogonal_(lin.weight, np.sqrt(2)); nn.init.zeros_(lin.bias)
        mods += [lin, nn.Tanh()]
        d = hidden
    last = nn.Linear(d, out)
    nn.init.orthogonal_(last.weight, out_gain); nn.init.zeros_(last.bias)
    return nn.Sequential(*mods, last)


class ActorCritic(nn.Module):
    """Separate actor and critic MLPs. Gaussian policy: the actor gives the mean, the std is a learned obs-independent parameter."""

    def __init__(self, obs_dim: int = 19, act_dim: int = 5, hidden: int = 256, init_std: float = 0.5):
        super().__init__()
        self.actor = mlp(obs_dim, act_dim, hidden, out_gain=0.01)  # near-zero first actions
        self.critic = mlp(obs_dim, 1, hidden, out_gain=1.0)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(np.log(init_std))))

    def dist(self, obs):
        return torch.distributions.Normal(self.actor(obs), self.log_std.exp())

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    @torch.no_grad()
    def act(self, obs):
        """Sample actions: (action, log_prob, value). Unclipped; the env clips."""
        d = self.dist(obs)
        a = d.sample()
        return a, d.log_prob(a).sum(-1), self.value(obs)


def compute_gae(rew, val, val_next, done, gamma=0.99, lam=0.95):
    """Generalized Advantage Estimation on (T, N) arrays. Returns (advantages, returns).
    val_next[t] is set by the caller: V(s_t+1) normally, 0 if the ball fell, V(last state) at the time limit.
    done[t] = 1 where an episode ended, so the advantage does not run across episodes."""
    T = rew.shape[0]
    delta = rew + gamma * val_next - val
    adv = np.zeros_like(rew)
    last = np.zeros_like(rew[0])
    for t in reversed(range(T)):
        last = delta[t] + gamma * lam * (1.0 - done[t]) * last
        adv[t] = last
    return adv, adv + val


def ppo_update(ac, opt, batch, epochs=10, minibatches=8, clip=0.2, vf_coef=0.5, ent_coef=0.0, max_grad=0.5, target_kl=0.03):
    """PPO update on a batch dict (obs, act, logp, adv, ret). Returns mean diagnostics."""
    n = batch["obs"].shape[0]
    mb = n // minibatches
    stats = dict(pg=0.0, vf=0.0, ent=0.0, kl=0.0, clipfrac=0.0, steps=0)
    for _ in range(epochs):
        perm = torch.randperm(n, device=batch["obs"].device)
        for i in range(minibatches):
            idx = perm[i * mb : (i + 1) * mb]
            d = ac.dist(batch["obs"][idx])
            logp = d.log_prob(batch["act"][idx]).sum(-1)
            log_ratio = logp - batch["logp"][idx]
            ratio = log_ratio.exp()  # pi_new / pi_old
            adv = batch["adv"][idx]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            pg = -torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()
            v = ac.value(batch["obs"][idx])
            vf = 0.5 * (v - batch["ret"][idx]).pow(2).mean()
            ent = d.entropy().sum(-1).mean()
            loss = pg + vf_coef * vf - ent_coef * ent
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(ac.parameters(), max_grad)
            opt.step()
            with torch.no_grad():
                kl = ((ratio - 1) - log_ratio).mean()  # approx KL(old || new)
                stats["pg"] += pg.item(); stats["vf"] += vf.item(); stats["ent"] += ent.item(); stats["kl"] += kl.item()
                stats["clipfrac"] += ((ratio - 1).abs() > clip).float().mean().item(); stats["steps"] += 1
        if stats["kl"] / stats["steps"] > target_kl:  # moved too far from the data-collecting policy: stop
            break
    k = stats.pop("steps")
    return {key: val / k for key, val in stats.items()}
