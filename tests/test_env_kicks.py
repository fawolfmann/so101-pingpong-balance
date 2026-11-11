"""Mid-episode ball kicks in BalanceEnv: off by default, drawn from the seed, applied in the physics."""
import numpy as np
import pytest

from sim.env import BalanceEnv, kick_schedule

KICKS = dict(kick_min=0.1, kick_max=0.3, kick_every=1.0, kick_grace=0.5, kick_margin=1.0)


def make(**kw) -> BalanceEnv:
    return BalanceEnv(obs_mode="controller", domain_rand=True, grip_rot_deg=-90, act_hist=3, max_steps=120, **kw)


def test_kick_schedule_respects_grace_and_end_margin():
    rng = np.random.default_rng(0)
    ks = kick_schedule(rng, n_steps=300, every_s=2.0, grace_s=1.0, margin_s=1.0, speed=(0.1, 0.3))
    assert ks and all(30 <= t < 270 for t in ks)  # after 1 s, at least 1 s before the end
    speeds = [np.hypot(*v) for v in ks.values()]
    assert all(0.1 - 1e-9 <= s <= 0.3 + 1e-9 for s in speeds)


def test_kick_schedule_empty_when_off_or_no_room():
    rng = np.random.default_rng(0)
    assert kick_schedule(rng, 300, every_s=2.0, grace_s=1.0, margin_s=1.0, speed=(0.0, 0.0)) == {}
    assert kick_schedule(rng, 300, every_s=0.0, grace_s=1.0, margin_s=1.0, speed=(0.1, 0.3)) == {}
    assert kick_schedule(rng, 45, every_s=2.0, grace_s=1.0, margin_s=1.0, speed=(0.1, 0.3)) == {}


def test_no_kicks_by_default():
    env = make()
    env.reset(seed=3)
    assert env.kicks == {}


def test_same_seed_same_kicks_whatever_the_policy_does():
    a, b = make(**KICKS), make(**KICKS)
    a.reset(seed=7); b.reset(seed=7)
    assert a.kicks and a.kicks.keys() == b.kicks.keys()
    for t in a.kicks:
        np.testing.assert_allclose(a.kicks[t], b.kicks[t])
    sched = {t: v.copy() for t, v in a.kicks.items()}
    rng = np.random.default_rng(0)
    for _ in range(60):  # different actions must not change the schedule
        a.step(rng.uniform(-1, 1, 5)); b.step(np.zeros(5))
    assert a.kicks.keys() == sched.keys()


def test_different_seeds_and_unseeded_resets_give_different_kicks():
    env = make(**KICKS)
    env.reset(seed=1); k1 = dict(env.kicks)
    env.reset(seed=2); k2 = dict(env.kicks)
    env.reset(); k3 = dict(env.kicks)
    assert k1 and k2 and k3
    assert k1.keys() != k2.keys() or not all(np.allclose(k1[t], k2[t]) for t in k1)
    assert k3.keys() != k2.keys() or not all(np.allclose(k3[t], k2[t]) for t in k3)


def test_kick_changes_ball_velocity_by_the_kick_vector():
    env = make(kick_min=0.2, kick_max=0.2, kick_every=1.0, kick_grace=0.5, kick_margin=1.0)
    env.reset(seed=5)
    t_kick = min(env.kicks)
    while env.t < t_kick:
        env.step(np.zeros(5))
    vadr = env.ball_vadr
    before = env.d.qvel[vadr:vadr + 2].copy()
    env.step(np.zeros(5))
    delta = env.d.qvel[vadr:vadr + 2] - before
    kick = env.kicks[t_kick]
    assert np.hypot(*kick) == pytest.approx(0.2)
    cos = delta @ kick / (np.linalg.norm(delta) * np.linalg.norm(kick))
    # A kick makes the ball slide without spin; friction turns it into rolling within the step. A thin spherical shell
    # (I = 2/3 m r^2) rolls at 3/5 of the push speed, so the ball ends up ~0.12 m/s faster, in the kick direction.
    assert cos > 0.9
    assert np.linalg.norm(delta) == pytest.approx(0.6 * 0.2, rel=0.25)


def test_invalid_kick_range_rejected():
    with pytest.raises(ValueError):
        make(kick_min=0.3, kick_max=0.1)
    with pytest.raises(ValueError):
        make(kick_min=-0.1, kick_max=0.1)
