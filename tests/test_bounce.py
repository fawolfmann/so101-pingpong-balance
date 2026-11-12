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


def test_juggling_counts_bounces_and_stops_a_held_ball():
    from sim.env import CONTROL_HZ, JUGGLE_STALL_S
    env = BalanceEnv(grip_rot_deg=-90, contact_damp=(30.0, 30.0), max_steps=150)
    env.reset(seed=0, options={"juggle": True, "drop": 0.10})
    assert env.kicks == {} and env.juggle
    rewards, info = [], {}
    for _ in range(150):
        _, r, term, trunc, info = env.step(np.zeros(5))
        rewards.append(r)
        if term or trunc:
            break
    assert info["n_bounces"] >= 1 and info["apex_mean"] >= 0.03  # the free bounces after the drop count
    assert info["stalled"] or info["fell"]  # a still arm cannot keep it going
    assert len(rewards) <= (JUGGLE_STALL_S + 1.5) * CONTROL_HZ
    plain = BalanceEnv(grip_rot_deg=-90)
    plain.reset(seed=0)
    assert not plain.juggle and "n_bounces" not in plain.step(np.zeros(5))[4]


def test_juggling_starts_from_the_bouncing_policy():
    from sim.cam_policy_env import CamPolicyEnv, expand_for_juggle, obs_dims
    from sim.ppo import ActorCritic
    a0, f0 = obs_dims(3, 3, height_info=True)
    a1, f1 = obs_dims(3, 3, height_info=True, juggle=True)
    base = ActorCritic(f0, 7, actor_dim=a0)
    new = ActorCritic(f1, 7, actor_dim=a1)
    new.load_state_dict(expand_for_juggle(base.state_dict(), 3, 3))
    old = torch.randn(4, f0)
    obs = torch.cat([old[:, :a0], torch.ones(4, 1), old[:, a0:], torch.rand(4, 1)], dim=1)  # mode = 1, any since-bounce
    with torch.no_grad():
        assert torch.allclose(new.dist(obs).mean, base.dist(old).mean)
        assert torch.allclose(new.value(obs), base.value(old))
    env = CamPolicyEnv(height_info=True, juggle=True, juggle_frac=1.0, contact_damp=(30.0, 30.0))
    o, _ = env.reset(seed=2)
    assert o.shape == (f1,) and o[a1 - 1] == 1.0 and env.env.juggle
    o, _ = env.reset(seed=2, juggle=False)
    assert o[a1 - 1] == 0.0 and not env.env.juggle


def test_height_command_starts_from_the_juggling_policy():
    from sim.cam_policy_env import CamPolicyEnv, expand_for_height_cmd, obs_dims
    from sim.ppo import ActorCritic
    a0, f0 = obs_dims(3, 3, height_info=True, juggle=True)
    a1, f1 = obs_dims(3, 3, height_info=True, juggle=True, height_cmd=True)
    base, new = ActorCritic(f0, 7, actor_dim=a0), ActorCritic(f1, 7, actor_dim=a1)
    new.load_state_dict(expand_for_height_cmd(base.state_dict(), 3, 3))
    old = torch.randn(4, f0)
    obs = torch.cat([old[:, :a0], torch.rand(4, 1), old[:, a0:]], dim=1)  # any commanded height
    with torch.no_grad():
        assert torch.allclose(new.dist(obs).mean, base.dist(old).mean)
        assert torch.allclose(new.value(obs), base.value(old))
    env = CamPolicyEnv(height_info=True, juggle=True, juggle_frac=1.0, contact_damp=(30.0, 30.0),
                       juggle_target=(0.04, 0.2), cam_track_w=0.3)
    o, _ = env.reset(seed=3)
    assert o.shape == (f1,) and 0.2 <= o[a1 - 1] <= 1.0  # target / 0.2 for a 4-20 cm command
    assert 0.04 <= env.env.target_apex <= 0.2
