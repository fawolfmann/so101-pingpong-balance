"""Evaluate an arm + camera policy (sim/cam_policy_env.py), by default with real rendering and the real detector.

    uv run python -m sim.eval_cam_policy --policy outputs/controller/cam_policy/best.pt --latency 0,2,4 --episodes 30 --nudges 2
    uv run python -m sim.eval_cam_policy --policy ... --sim-det      # simulated detections (what it was trained on)

Same test as perception.estimator --eval: seeds 10000+i, servo and camera randomization, latency forced per row,
--nudges pushes of 0.25 m/s at random steps. Bouncing: --nudge-up lo,hi adds an upward speed to every push (m/s),
--contact-damp lo,hi makes the ball/blade contact elastic (as in training with those flags).
Juggling: --juggle runs juggling episodes (ball dropped from --juggle-drop, no pushes, --seconds long) and reports the
time juggled, bounces, and how episodes ended (time up, stall, drop). Also reports how the camera was used: detection rate, ball offset from
the image centre, pan/tilt speed.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp

import numpy as np

_G: dict = {}


def _init(cfg):
    import torch
    from sim.cam_policy_env import CamPolicyEnv
    from sim.policies import load_controller
    torch.set_num_threads(1)
    ac = load_controller(cfg["policy"])
    ta = ac.train_args
    extra = dict(contact_damp=cfg["contact_damp"]) if cfg["contact_damp"] else {}
    if ta.get("juggle_frac", 0.0) > 0:  # the policy has the mode input
        extra.update(juggle=True, height_info=True, juggle_drop=cfg["juggle_drop"])
    if cfg["seconds"]:
        extra["max_steps"] = round(cfg["seconds"] * 30)
    _G.update(cfg=cfg, ac=ac, env=CamPolicyEnv(render=not cfg["sim_det"], frames=ta.get("frames", 3), act_hist=ta["act_hist"],
                                              domain_rand=True, servo_rand=True, cam_rand=0.3, max_latency=cfg["max_lat"],
                                              track_gate=ta.get("track_gate", 0.0), **extra))
    if cfg["nudge_up"]:
        _G["env"].env.airborne_ok = True  # the ball may leave the blade


def _episode(job):
    from sim.policies import controller_action
    lat, seed = job
    cfg, env, ac = _G["cfg"], _G["env"], _G["ac"]
    obs, _ = env.reset(seed=seed, **(dict(juggle=True) if cfg["juggle"] else {}))
    env.env._action_delay, env.env._delay_buf, env._cam_buf = lat, [np.zeros(5)] * lat, [np.zeros(2)] * lat
    rng = np.random.default_rng(seed + 777)  # the same nudges as perception.estimator's evaluation
    nudges = {int(t): rng.uniform(0, 2 * np.pi) for t in rng.integers(30, 280, cfg["nudges"])}
    up = np.random.default_rng(seed + 778)  # separate stream: the push times and directions match the sideways test
    seen, off, cam_speed = [], [], []
    for t in range(env.max_steps):
        a = controller_action(ac, obs[:ac.actor_dim])
        if t in nudges:
            vz = up.uniform(*cfg["nudge_up"]) if cfg["nudge_up"] else 0.0
            env.env.d.qvel[env.env.ball_vadr:env.env.ball_vadr + 3] += np.r_[0.25 * np.cos(nudges[t]), 0.25 * np.sin(nudges[t]), vz]
        obs, _, te, tr, info = env.step(a)
        det = info["det"]
        seen.append(det[3]); cam_speed.append(np.abs(a[5:7]).max())
        if det[3]:
            off.append(np.abs(det[:2]).max())
        if te or tr:
            break
    if cfg["juggle"]:
        return dict(lat=lat, success=bool(info["is_success"]), juggle_s=info["juggle_s"], n_bounces=info["n_bounces"],
                    stalled=bool(info["stalled"]), fell=bool(info["fell"]), apex=info["apex_mean"], seen=float(np.mean(seen)))
    return dict(lat=lat, success=bool(info["is_success"]), length=t + 1, dist_mm=float(info["dist"] * 1000), seen=float(np.mean(seen)),
                off=float(np.mean(off)) if off else 1.0, cam=float(np.mean(cam_speed)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--latency", default="0,2,4")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--nudges", type=int, default=0)
    p.add_argument("--sim-det", action="store_true", help="simulated detections instead of rendering")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--nudge-up", default="", help="upward speed range added to every push, 'lo,hi' m/s")
    p.add_argument("--contact-damp", default="", help="elastic ball/blade contact, damping range 'lo,hi'")
    p.add_argument("--juggle", action="store_true", help="juggling episodes (needs a policy trained with --juggle-frac)")
    p.add_argument("--juggle-drop", default="0.08,0.15", help="juggling: drop height range, m")
    p.add_argument("--seconds", type=float, default=0.0, help="episode length (default 10 s)")
    a = p.parse_args()
    pair = lambda v: tuple(float(x) for x in v.split(",")) if v else None
    lats = [int(v) for v in a.latency.split(",")]
    jobs = [(lat, 10_000 + i) for lat in lats for i in range(a.episodes)]
    cfg = dict(policy=a.policy, nudges=a.nudges, sim_det=a.sim_det, max_lat=max(lats), nudge_up=pair(a.nudge_up),
               contact_damp=pair(a.contact_damp), juggle=a.juggle, seconds=a.seconds,
               juggle_drop=pair(a.juggle_drop))
    if a.juggle and not a.contact_damp:
        cfg["contact_damp"] = (20.0, 40.0)  # juggling needs the elastic paddle
    with mp.get_context("spawn").Pool(a.workers, _init, (cfg,)) as pool:
        res = pool.map(_episode, jobs, chunksize=1)
    print(f"\n{a.policy} | {'simulated' if a.sim_det else 'rendered'} detections | {a.episodes} episodes per row | nudges {a.nudges}"
          + (f" | push up {a.nudge_up} m/s" if a.nudge_up else "") + (f" | contact damp {cfg['contact_damp']}" if cfg["contact_damp"] else "")
          + (f" | JUGGLING, drop {a.juggle_drop} m" if a.juggle else ""))
    if a.juggle:
        print(f"{'latency':>8} {'juggled to the end':>19} {'mean s':>7} {'median s':>9} {'bounces':>8} {'stalls':>7} {'drops':>6} {'apex cm':>8}")
        for lat in lats + ([None] if len(lats) > 1 else []):
            rs = [r for r in res if lat is None or r["lat"] == lat]
            js = np.array([r["juggle_s"] for r in rs])
            ap = [r["apex"] for r in rs if np.isfinite(r["apex"])]
            print(f"{'all' if lat is None else lat:>8} {sum(r['success'] for r in rs):>12}/{len(rs):<6} {js.mean():7.1f} {np.median(js):9.1f}"
                  f" {np.mean([r['n_bounces'] for r in rs]):8.1f} {sum(r['stalled'] for r in rs):7d} {sum(r['fell'] for r in rs):6d}"
                  f" {100 * np.mean(ap) if ap else float('nan'):8.1f}")
        return
    print(f"{'latency':>8} {'success':>9} {'len':>5} {'final dist':>11}  {'ball seen':>9} {'|offset|':>9} {'cam |a|':>8}")
    for lat in lats + ([None] if len(lats) > 1 else []):
        rs = [r for r in res if lat is None or r["lat"] == lat]
        m = lambda k: np.mean([r[k] for r in rs])
        print(f"{'all' if lat is None else lat:>8} {sum(r['success'] for r in rs):>4}/{len(rs):<4} {m('length'):5.0f} {m('dist_mm'):8.1f} mm"
              f"  {m('seen'):9.3f} {m('off'):9.3f} {m('cam'):8.2f}")


if __name__ == "__main__":
    main()
