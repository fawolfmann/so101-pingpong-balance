import numpy as np
import pytest

from sim.stress import kick_schedule, warning_lead, run_episode, make_env
from sim.policies import load_controller


def test_kick_schedule_one_kick_per_window_after_grace():
    rng = np.random.default_rng(0)
    kicks = kick_schedule(rng, n_steps=600, every_s=4.0, grace_s=1.0)
    steps = sorted(kicks)
    assert len(steps) == 5  # windows [30,150) [150,270) ... [510,600): 600 steps = 20 s
    assert all(30 <= t < 600 for t in steps)
    assert all(0 <= a < 2 * np.pi for a in kicks.values())


def test_kick_schedule_disabled():
    assert kick_schedule(np.random.default_rng(0), 300, every_s=0.0) == {}


def test_warning_lead_counts_final_run_below_threshold():
    values = np.array([10, 10, 2, 10, 10, 3, 2, 1.0])  # early dip recovers; final run starts at index 5
    assert warning_lead(values, threshold=5.0) == 3


def test_warning_lead_none_when_never_below():
    assert warning_lead(np.array([10.0, 9.0, 8.0]), threshold=5.0) == 0


@pytest.mark.parametrize("kick_speed", [0.0, 3.0])
def test_run_episode_records_values(kick_speed):
    ac = load_controller("weights/controller.pt")
    env = make_env(ac)
    ep = run_episode(env, ac, seed=1, seconds=2.0, kick_speed=kick_speed, kick_every=0.5, record=True)
    assert ep["length"] == len(ep["value"]) == len(ep["dist_trace"])
    assert ep["length"] <= 60
    if kick_speed == 0.0:
        assert ep["success"] and ep["length"] == 60
    else:  # 3 m/s kicks every 0.5 s knock the ball off
        assert not ep["success"] and ep["length"] < 60


def test_alarm_table_catches_drop_without_false_alarms():
    from sim.critic_plot import alarm_table

    ok = [dict(value=[50.0] * 100)]
    drops = [dict(value=[50.0] * 90 + [0.0] * 10)]  # value collapses 10 steps before the drop
    row = next(r for r in alarm_table(ok, drops) if r["pct"] == 1)
    assert row["caught"] == 1.0 and row["false_alarm"] == 0.0
    assert row["lead_s"] == 10 / 30
