"""GRPO for the balancing controller: group-relative advantages and the clipped policy update. No critic.

A group = G episodes from the same seed (same start, randomization and kicks); they differ only by the policy's
sampling noise. Each episode's advantage is its return relative to its group, (R - mean) / std, and every step of
the episode gets that advantage. The update is PPO's clipped objective with an optional KL penalty toward a frozen
reference policy (as in DeepSeekMath's GRPO). The loss is averaged over all steps in the batch, so a short episode
(an early drop) does not weigh more per step than a long one. Training loop: train_grpo.py.

The network is the same ActorCritic as PPO, so checkpoints work with every eval tool; the critic is never trained.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def group_advantages(returns: np.ndarray, group_size: int) -> np.ndarray:
    """Episode returns laid out group by group ([g0e0, g0e1, ..., g1e0, ...]) -> (R - group mean) / group std.
    A group whose returns are all equal gives 0 (no learning signal) instead of dividing by zero."""
    r = np.asarray(returns, np.float64)
    if group_size < 2 or r.size % group_size:
        raise ValueError(f"{r.size} returns do not split into groups of {group_size} (need >= 2)")
    g = r.reshape(-1, group_size)
    std = g.std(axis=1, keepdims=True)
    adv = np.where(std > 1e-8, (g - g.mean(axis=1, keepdims=True)) / np.maximum(std, 1e-8), 0.0)
    return adv.reshape(-1).astype(np.float32)


def grpo_update(ac, opt, batch: dict, epochs: int = 4, minibatches: int = 8, clip: float = 0.2, kl_coef: float = 0.0,
                ref=None, max_grad: float = 0.5, target_kl: float = 0.03) -> dict:
    """Clipped policy update on a batch dict (obs, act, logp, adv), one row per step. kl_coef > 0 needs `ref`
    (a frozen policy): adds kl_coef * KL(new || ref), k3 estimator. Returns mean diagnostics."""
    if kl_coef > 0 and ref is None:
        raise ValueError("kl_coef > 0 needs a reference policy")
    n = batch["obs"].shape[0]
    mb = max(1, n // minibatches)
    actor_params = list(ac.actor.parameters()) + [ac.log_std]
    stats = dict(loss=0.0, kl=0.0, kl_ref=0.0, clipfrac=0.0, steps=0)
    for _ in range(epochs):
        perm = torch.randperm(n, device=batch["obs"].device)
        for i in range(0, n - mb + 1, mb):
            idx = perm[i:i + mb]
            obs, act, adv = batch["obs"][idx], batch["act"][idx], batch["adv"][idx]
            logp = ac.dist(obs).log_prob(act).sum(-1)
            log_ratio = logp - batch["logp"][idx]
            ratio = log_ratio.exp()
            loss = -torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()
            kl_ref = torch.zeros(())
            if kl_coef > 0:
                with torch.no_grad():
                    ref_logp = ref.dist(obs).log_prob(act).sum(-1)
                d = ref_logp - logp
                kl_ref = (d.exp() - d - 1).mean()  # k3: unbiased, always >= 0
                loss = loss + kl_coef * kl_ref
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(actor_params, max_grad)
            opt.step()
            with torch.no_grad():
                stats["loss"] += loss.item(); stats["kl"] += ((ratio - 1) - log_ratio).mean().item()
                stats["kl_ref"] += float(kl_ref); stats["clipfrac"] += ((ratio - 1).abs() > clip).float().mean().item()
                stats["steps"] += 1
        if stats["kl"] / stats["steps"] > target_kl:  # moved too far from the data-collecting policy: stop
            break
    k = stats.pop("steps")
    return {key: val / k for key, val in stats.items()}
