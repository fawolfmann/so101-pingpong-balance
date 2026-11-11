"""GRPO pieces: group-relative advantages, the clipped policy update, and seeded episodic rollouts."""
import numpy as np
import pytest
import torch

from sim.grpo import group_advantages, grpo_update
from sim.ppo import ActorCritic
from sim.vec_env import EpisodicVecEnv


def test_group_advantages_are_relative_within_each_group():
    returns = np.array([1.0, 2.0, 3.0, 10.0, 10.0, 10.0])
    adv = group_advantages(returns, group_size=3)
    np.testing.assert_allclose(adv[:3], [-1.2247, 0.0, 1.2247], atol=1e-3)  # (R - mean) / std
    np.testing.assert_allclose(adv[3:], 0.0)  # identical returns: no signal, not a division by zero


def test_group_advantages_rejects_bad_group_size():
    with pytest.raises(ValueError):
        group_advantages(np.zeros(5), group_size=2)


def test_grpo_update_moves_policy_toward_positive_advantage_actions():
    torch.manual_seed(0)
    ac = ActorCritic(obs_dim=4, act_dim=2)
    opt = torch.optim.Adam(ac.parameters(), lr=1e-2)
    obs = torch.zeros(64, 4)
    act = torch.cat([torch.full((32, 2), 0.5), torch.full((32, 2), -0.5)])
    adv = torch.cat([torch.ones(32), -torch.ones(32)])  # +0.5 was good, -0.5 was bad
    with torch.no_grad():
        logp = ac.dist(obs).log_prob(act).sum(-1)
    critic_before = [p.clone() for p in ac.critic.parameters()]
    for _ in range(5):
        stats = grpo_update(ac, opt, dict(obs=obs, act=act, logp=logp, adv=adv), epochs=4, minibatches=2)
    assert ac.actor(obs[:1])[0].mean() > 0.05  # mean action moved toward +0.5
    assert all(torch.equal(a, b) for a, b in zip(critic_before, ac.critic.parameters()))  # critic untouched
    assert {"loss", "kl", "clipfrac"} <= stats.keys()


@pytest.fixture(scope="module")
def venv():
    env = EpisodicVecEnv(n_workers=2, envs_per_worker=2, domain_rand=True, grip_rot_deg=-90, act_hist=3,
                         max_steps=45, kick_min=0.1, kick_max=0.3, kick_every=0.5, kick_grace=0.5, kick_margin=0.2)
    yield env
    env.close()


def test_same_seed_gives_same_start_and_kicks(venv):
    obs, kicks = venv.reset([11, 11, 11, 12])
    np.testing.assert_allclose(obs[0], obs[1]); np.testing.assert_allclose(obs[0], obs[2])
    assert kicks[0] == kicks[1] == kicks[2] and kicks[0] != kicks[3]


def test_finished_episodes_stay_finished_until_all_done(venv):
    venv.reset([1, 2, 3, 4])
    rng = np.random.default_rng(0)
    active = np.ones(4, bool)
    steps = 0
    while active.any():
        obs, rew, done = venv.step(rng.uniform(-1, 1, (4, 5)))
        assert not np.any(rew[~active])  # no reward after an env finished
        active &= ~done
        steps += 1
    assert steps <= 45
