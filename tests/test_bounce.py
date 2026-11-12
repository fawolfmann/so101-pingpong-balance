"""Bouncing: upward kick components, the elastic contact, airborne-safe termination, starting from a balance policy."""
import mujoco
import numpy as np
import torch

from sim.env import BalanceEnv

KICKS = dict(kick_max=0.3, kick_every=1.0, max_steps=150)


def test_sideways_kicks_unchanged_when_bouncing_is_off():
    a, b = BalanceEnv(**KICKS), BalanceEnv(**KICKS, kick_z_max=0.0)
    a.reset(seed=3); b.reset(seed=3)
    assert a.kicks.keys() == b.kicks.keys() and all(np.array_equal(a.kicks[t], b.kicks[t]) for t in a.kicks)
    assert all(len(k) == 2 for k in a.kicks.values())


def test_upward_kicks_in_a_share_of_episodes():
    env = BalanceEnv(**KICKS, kick_z_min=0.5, kick_z_max=1.5, kick_z_frac=0.5)
    up = []
    for s in range(40):
        env.reset(seed=s)
        up.append(env.kick_z_on)
        if env.kick_z_on:
            assert all(len(k) == 3 and 0.5 <= k[2] <= 1.5 for k in env.kicks.values())
    assert 0.2 < np.mean(up) < 0.8


def rebound(env: BalanceEnv, drop: float = 0.1) -> float:
    """Restitution from a drop onto the paddle: sqrt(rebound height / drop height)."""
    env.reset(seed=0)
    m, d = env.m, env.d
    c, n, _, _ = env.blade_frame()
    d.qpos[env.ball_qadr:env.ball_qadr + 3] = c + n * (env.rest_height() + drop)
    d.qvel[env.ball_vadr:env.ball_vadr + 6] = 0
    mujoco.mj_forward(m, d)
    h, vz = [], []
    for _ in range(240):
        mujoco.mj_step(m, d)
        h.append(env.ball_lift()[0]); vz.append(d.qvel[env.ball_vadr + 2])
    vz = np.array(vz)
    k = np.flatnonzero((vz[:-1] < 0) & (vz[1:] >= 0))[0]
    return float(np.sqrt(max(h[k:k + 120]) / drop))


def test_elastic_contact_bounces_like_a_ping_pong_ball():
    assert rebound(BalanceEnv(grip_rot_deg=-90)) < 0.35  # original: the ball lands dead
    e = rebound(BalanceEnv(grip_rot_deg=-90, contact_damp=(30.0, 30.0)))
    assert 0.7 < e < 0.9


def test_ball_in_the_air_is_not_a_drop_when_bouncing():
    env = BalanceEnv(grip_rot_deg=-90, contact_damp=(30.0, 30.0))
    env.reset(seed=0)
    env.d.qvel[env.ball_vadr + 2] = 1.2  # straight up, ~7 cm
    for _ in range(4):
        _, _, term, _, info = env.step(np.zeros(5))
        assert not term
    assert info["ball_h"] > 0.08  # the balance-only rule would have ended the episode here
    plain = BalanceEnv(grip_rot_deg=-90)
    plain.reset(seed=0)
    plain.d.qvel[plain.ball_vadr + 2] = 1.2
    assert any(plain.step(np.zeros(5))[2] for _ in range(4))


def test_bounce_training_starts_from_the_same_policy():
    from sim.cam_policy_env import CamPolicyEnv, expand_critic, obs_dims
    from sim.ppo import ActorCritic
    a0, f0 = obs_dims(3, 3)
    _, f1 = obs_dims(3, 3, height_info=True)
    base = ActorCritic(f0, 7, actor_dim=a0)
    new = ActorCritic(f1, 7, actor_dim=a0)
    new.load_state_dict(expand_critic(base.state_dict(), 3, 3))
    obs = torch.randn(4, f1)
    with torch.no_grad():
        assert torch.equal(new.dist(obs).mean, base.dist(obs[:, :f0]).mean)  # same actions
        assert torch.allclose(new.value(obs), base.value(obs[:, :f0]))  # same values until training uses the new inputs
    env = CamPolicyEnv(height_info=True, contact_damp=(30.0, 30.0))
    o, _ = env.reset(seed=1)
    assert o.shape == (f1,) and env.obs_dim == a0
