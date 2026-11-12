"""Fit the simulated-detection error model (sim/cam_model.py) against the real renderer + detector, and check it.

    uv run python -m sim.fit_cam_model --frames 60000 --workers 6

Collects frames from the tracking camera with the controller balancing (true state) and the camera aimed off the ball by
a random drift, so the ball covers the whole image. Each frame gives the rendered detection and the error-free projection.
Fits on 80% of the episodes, then compares the model's samples with the real detector on the other 20%.
Writes outputs/cam_noise_ego.json and outputs/cam_pairs.npz.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np

from sim.cam_model import MODEL_PATH, DetNoise, occluded, project, whole_in_view

ROOT = Path(__file__).parent.parent
OFF_STD, OFF_A = 0.2, float(np.exp(-1 / 45))  # camera aim drift off the ball: rad, ~1.5 s correlation


def _collect(job):
    import torch
    from perception.estimator import CAMERAS, make_env, observe
    from perception.detector import TRACK_DET, candidates
    from sim.policies import controller_action, load_controller
    torch.set_num_threads(1)
    seed0, n_eps, controller = job
    ac = load_controller(controller)
    prof = CAMERAS["ego_track"]
    env = make_env("ego_track", domain_rand=True, cam_rand=0.3, servo_rand=True, act_hist=ac.act_hist, max_latency=2,
                   kick_max=0.3, kick_every=2.0)
    rows = []
    for e in range(n_eps):
        seed = seed0 + e
        obs, _ = env.reset(seed=seed)
        rng = np.random.default_rng(seed + 99)
        off = np.zeros(2)
        for t in range(env.max_steps):
            det = observe(env, obs, prof)  # rule target toward the detected ball...
            off = OFF_A * off + np.sqrt(1 - OFF_A ** 2) * rng.normal(0, OFF_STD, 2)
            if det[3] == 0:  # lost: back toward the blade (collection only; the policy has to find it itself)
                env._aim_want = env._yaw_pitch(env._parent_dir(env.d.geom_xpos[env.blade_geom]))
            env._aim_want = env._aim_want + off * min(1.0, t / 30)  # ...pushed off it by a slow random drift
            proj, frac = project(env, with_frac=True)
            cands = candidates(obs["image"], **TRACK_DET)
            whole = whole_in_view(proj, frac, env.image_hw)
            rows.append(np.r_[seed, cands[0] if cands else np.zeros(4), proj, float(occluded(env)), frac, float(whole)])
            obs, _, te, tr, _ = env.step(controller_action(ac, env.controller_obs()))
            if te or tr:
                break
    return np.array(rows, np.float32)


def collect(frames: int, workers: int, controller: str) -> np.ndarray:
    n_eps = frames // 250 + 1
    per = max(1, n_eps // (workers * 4))
    jobs = [(50_000 + s, per, controller) for s in range(0, n_eps, per)]  # disjoint seed ranges
    with mp.get_context("spawn").Pool(workers) as pool:
        return np.concatenate(pool.map(_collect, jobs, chunksize=1))


def split(rows):
    ep, det, proj, occ, frac, whole = rows[:, 0], rows[:, 1:5], rows[:, 5:9], rows[:, 9] > 0, rows[:, 10], rows[:, 11] > 0
    return ep, det, proj, occ & True, frac, whole


def fit(rows) -> tuple[DetNoise, dict]:
    ep, det, proj, occ, frac, whole = split(rows)
    vis = whole & ~occ  # the simulated detector may report the ball
    edge = (proj[:, 3] > 0) & ~whole  # in view but cut by (or touching) the border: never a whole-ball candidate
    inner = vis
    both = inner & (det[:, 3] > 0)
    res = det[:, :3] - proj[:, :3]
    good = both & (np.abs(res[:, :2]).max(1) < 0.1)  # wrong-blob outliers are counted, not fitted
    F = np.c_[np.ones(len(proj)), proj[:, :3]]
    bias = np.linalg.lstsq(F[good], res[good], rcond=None)[0].T  # 3 x 4
    rr = res - F @ bias.T
    eps = np.unique(ep[good])
    means = np.array([rr[good & (ep == e)].mean(0) for e in eps])
    within = np.concatenate([rr[good & (ep == e)] - m for e, m in zip(eps, means)])
    model = DetNoise(bias=bias.round(5).tolist(), jitter=within.std(0).round(5).tolist(), ep_bias=means.std(0).round(5).tolist(),
                     miss=round(float(1 - det[inner, 3].mean()), 5), miss_edge=1.0)
    stats = dict(frames=len(rows), whole_in_view=float(vis.mean()), occluded=float(((proj[:, 3] > 0) & occ).mean()),
                 cut_by_border=float(edge.mean()), outliers=float(1 - good.sum() / max(both.sum(), 1)),
                 seen_when_cut=float(det[edge, 3].mean()) if edge.any() else 0.0,
                 seen_when_out=float(det[proj[:, 3] == 0, 3].mean()) if (proj[:, 3] == 0).any() else 0.0)
    return model, stats


def check(model: DetNoise, rows) -> None:
    """Held-out episodes: the model's simulated detections vs the real detector, against the same projections."""
    ep, det, proj, occ, frac, whole = split(rows)
    rng = np.random.default_rng(0)
    sim, eps_state = np.zeros_like(det), {}
    for i in range(len(rows)):
        e = ep[i]
        if e not in eps_state:
            eps_state[e] = model.new_episode(rng)
        sim[i] = model.sample(proj[i], bool(occ[i]) or not whole[i], eps_state[e], rng)
    vis = whole & ~occ
    print(f"\nheld-out check: {len(rows)} frames, {len(np.unique(ep))} episodes")
    cut = (proj[:, 3] > 0) & ~whole
    for name, sel in (("whole ball in view", vis), ("ball cut by the border", cut), ("ball out of view", proj[:, 3] == 0)):
        if sel.any():
            print(f"  detection rate, {name:<23} real {det[sel, 3].mean():.4f}  model {sim[sel, 3].mean():.4f}  ({sel.mean():.3f} of frames)")
    both = vis & (det[:, 3] > 0) & (sim[:, 3] > 0)
    for k, name in enumerate(["x", "y", "r"]):
        er, es = det[both, k] - proj[both, k], sim[both, k] - proj[both, k]
        q = [50, 90, 99]
        print(f"  {name} error vs projection   real mean {er.mean():+.4f} |err| p50/p90/p99 " + "/".join(f"{np.percentile(np.abs(er), p):.4f}" for p in q)
              + f"   model mean {es.mean():+.4f} |err| " + "/".join(f"{np.percentile(np.abs(es), p):.4f}" for p in q))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int, default=60_000)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--controller", default=str(ROOT / "outputs/weights/ppo_kicks_smooth.pt"))
    p.add_argument("--reuse", action="store_true", help="refit from outputs/cam_pairs.npz")
    a = p.parse_args()
    pairs = ROOT / "outputs/cam_pairs.npz"
    rows = np.load(pairs)["rows"] if a.reuse else collect(a.frames, a.workers, a.controller)
    np.savez_compressed(pairs, rows=rows)
    eps = np.unique(rows[:, 0])
    test = np.isin(rows[:, 0], eps[np.random.default_rng(0).permutation(len(eps))[: len(eps) // 5]])
    model, stats = fit(rows[~test])
    print("collection:", {k: round(v, 4) if isinstance(v, float) else v for k, v in stats.items()})
    print("fitted model:", model.__dict__)
    model.save(MODEL_PATH)
    check(model, rows[test])
    print(f"saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
