"""Does bounce height follow the commanded height? Deterministic policy, simulated detections, 30 juggling episodes.

    uv run python -m sim.height_follow outputs/controller/juggle_height/last.pt
"""
import sys
import numpy as np
from sim.cam_policy_env import CamPolicyEnv
from sim.policies import load_controller, controller_action

ac = load_controller(sys.argv[1])
lo, hi = (float(v) for v in ac.train_args["juggle_target"].split(","))
env = CamPolicyEnv(render=False, frames=3, act_hist=3, domain_rand=True, servo_rand=True, cam_rand=0.3, max_latency=2,
                   max_steps=600, track_gate=0.3, juggle=True, height_info=True, contact_damp=(20., 40.), juggle_target=(lo, hi))
pairs, vz = [], []
for s in range(30):
    obs, _ = env.reset(seed=1000 + s, juggle=True); e = env.env
    for t in range(600):
        tgt, n = e.target_apex, e.n_bounces
        obs, _, te, tr, info = env.step(controller_action(ac, obs[:ac.actor_dim]))
        if e.n_bounces > n:
            pairs.append((tgt, e._apexes[-1]))
        if te or tr:
            break
p = np.array(pairs) * 100
print(f"{sys.argv[1]}: {len(p)} bounces | corr(target, apex) = {np.corrcoef(p[:, 0], p[:, 1])[0, 1]:.2f}")
edges = np.linspace(lo * 100, hi * 100, 5)
for a, b in zip(edges[:-1], edges[1:]):
    m = (p[:, 0] >= a) & (p[:, 0] < b)
    print(f"  target {a:4.1f}-{b:4.1f} cm: bounce height mean {p[m, 1].mean():.1f} cm (n={m.sum()})")
