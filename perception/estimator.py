"""Ball estimator: the last 3 detections (12 numbers) -> ball position and velocity on the blade, in controller units.
Only the position is used at run time (FdVel computes the velocity); the velocity output is just an extra training target.
Trained in sim only, used unchanged on the real arm.

Cameras (CAMERAS): "wrist" is rigid on the paddle, so the detections alone are enough (12 inputs). "ego" sits on a post
on the base and does not move with the paddle, so the same pixel means a different blade position for every arm pose:
its estimator also gets the 5 joint angles (controller units, (q - home) / 0.2), 17 inputs. A checkpoint stores its
camera, and evaluation / EstRunner follow it.

    uv run python -m perception.estimator --name my_estimator --frames 500000     # collect sim data + train
    uv run python -m perception.estimator --name ego_est --camera ego --controller outputs/weights/ppo_kicks_smooth.pt
    uv run python -m perception.estimator --eval weights/estimator.pt --latency 0,2,4 --workers 4 --servo-rand
        # the controller balances in sim on the estimate (camera -> detector -> estimator -> FdVel) instead of the true state
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from sim.ram import check_ram

ROOT = Path(__file__).parent.parent
N_DET = 12  # last 3 detections x (x, y, r, visible)

# Per camera: render size, detector settings (blob-size limits in px of that image), estimator inputs.
# ego: higher saturation + roundest blob, because it sees the paddle handle (wood-coloured in the randomization):
# 12.6% of ego detections hit the handle with the wrist settings, 0% with these (593 randomized frames).
CAMERAS = {
    "wrist": dict(image_hw=(96, 128), det=dict(min_area=40, max_area=1200), joints=False, n_in=N_DET),
    "ego": dict(image_hw=(192, 256), det=dict(min_area=20, max_area=2000, s_min=170, select="round"), joints=True, n_in=N_DET + 5),
}


class Estimator(nn.Module):
    def __init__(self, hidden: int = 256, n_in: int = N_DET):
        super().__init__()
        self.n_in, self.camera = n_in, "wrist" if n_in == N_DET else "ego"
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 4))

    def forward(self, d):
        return self.net(d)


def load_est(path) -> Estimator:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    m = Estimator(**ck.get("arch", {}))
    m.load_state_dict(ck["model"])
    m.camera = ck.get("camera", "wrist")  # checkpoints from before cameras existed are wrist estimators
    return m.eval()


class EstRunner:
    """Keeps the last 3 detections and returns the current estimate. Used in sim and on the robot."""

    def __init__(self, model: Estimator):
        self.model = model.eval()
        self.reset()

    def reset(self):
        self.dets: deque | None = None
        self.last = np.zeros(4, np.float32)

    @torch.no_grad()
    def update(self, det, joints=None) -> np.ndarray:
        """det: detector output; joints: the 5 joint angles in controller units (needed by an ego estimator)."""
        if self.dets is None:
            self.dets = deque([det] * 3, maxlen=3)
        else:
            self.dets.append(det)
        if det[3] == 0:  # ball not seen: hold the last estimate, decay its velocity
            self.last = self.last * np.array([1, 1, 0.8, 0.8], np.float32)
            return self.last
        x = np.ravel(list(self.dets))
        if self.model.n_in > N_DET:
            if joints is None:
                raise ValueError(f"the {self.model.camera} estimator needs the joint angles")
            x = np.r_[x, joints]
        self.last = self.model(torch.from_numpy(x.astype(np.float32))[None])[0].numpy()
        return self.last


def augment(d: torch.Tensor, g: torch.Generator, r_lo: float = 0.9, r_hi: float = 1.25) -> torch.Tensor:
    """Blob radius scale, a shared + per-frame position offset, and 3% dropped detections."""
    B, dev = len(d), d.device
    extra = d[:, N_DET:]  # joint angles (ego): exact, not augmented
    d = d[:, :N_DET].reshape(B, 3, 4).clone()
    vis = d[:, :, 3:4]
    rnd = lambda *s: torch.rand(*s, generator=g, device=dev)
    d[:, :, 2:3] *= r_lo + (r_hi - r_lo) * rnd(B, 1, 1)
    d[:, :, :2] += (0.02 * torch.randn(B, 1, 2, generator=g, device=dev) + 0.005 * torch.randn(B, 3, 2, generator=g, device=dev)) * vis
    d = d * (rnd(B, 3, 1) > 0.03)  # a miss = all four numbers 0
    return torch.cat([d.reshape(B, N_DET), extra], dim=1)


def _collect_worker(wid, out, frames, controller_ckpt, noise_max, seed, max_latency, start_radius, start_speed, cam_rand, camera):
    """The controller drives (true ball state + action noise, sigma ~ U(0, noise_max) per episode). Records detections + true ball state."""
    from perception.detector import detect
    from sim.env import BalanceEnv
    from sim.policies import load_controller, controller_action

    torch.set_num_threads(1)
    ctrl = load_controller(controller_ckpt)
    cam = CAMERAS[camera]
    env = BalanceEnv(obs_mode="image", camera=camera, image_hw=cam["image_hw"], domain_rand=True, light_model=False, grip_rot_deg=-90.0,
                     act_hist=ctrl.act_hist, max_latency=max_latency, cam_rand=cam_rand, seed=seed)
    rng = np.random.default_rng(seed + 12345)
    X, BALL, EP = (np.zeros((frames, k), np.float32) for k in (cam["n_in"], 4, 1))
    n, ep, stats = 0, wid * 1_000_000, []
    while n < frames:
        env.ball_start_radius, env.ball_start_speed = rng.uniform(0.0, start_radius), rng.uniform(0.0, start_speed)
        obs, _ = env.reset(seed=int(rng.integers(1 << 31)))
        dets = None
        sigma = rng.uniform(0, noise_max)
        nudges = {int(t): rng.uniform(0, 2 * np.pi) for t in rng.integers(30, 280, int(rng.integers(0, 3)))}
        ended = False
        for t in range(env.max_steps):
            det = detect(obs["image"], **cam["det"])
            if dets is None:
                dets = deque([det] * 3, maxlen=3)  # as EstRunner does
            else:
                dets.append(det)
            tobs = env.controller_obs()
            a = np.clip(controller_action(ctrl, tobs) + rng.normal(0, sigma, 5), -1, 1)
            X[n], BALL[n], EP[n] = np.r_[np.ravel(list(dets)), tobs[4:9]][:cam["n_in"]], tobs[:4], ep
            n += 1
            if t in nudges:
                env.d.qvel[env.ball_vadr:env.ball_vadr + 2] += 0.25 * np.array([np.cos(nudges[t]), np.sin(nudges[t])])
            obs, r, te, tr, info = env.step(a)
            if te or tr:
                ended = True
                break
            if n >= frames:
                break
        if ended:
            stats.append(dict(length=t + 1, success=bool(info["is_success"])))
        ep += 1
    np.savez(Path(out) / f"shard_{wid:02d}.npz", x=X[:n], ball=BALL[:n], ep=EP[:n, 0].astype(np.int64))
    (Path(out) / f"shard_{wid:02d}.json").write_text(json.dumps(stats))


def rollout(out: Path, frames, workers, controller_ckpt, noise_max, seed, max_latency, start_radius=0.04, start_speed=0.1, cam_rand=1.0, camera="wrist"):
    check_ram(1.5 * 450 * workers, "Use fewer --workers.")
    out.mkdir(parents=True, exist_ok=True)
    per = -(-frames // workers)
    ctx = mp.get_context("spawn")
    ps = [ctx.Process(target=_collect_worker, args=(w, str(out), per, str(controller_ckpt), noise_max, seed + 1000 * w, max_latency, start_radius, start_speed, cam_rand, camera))
          for w in range(workers)]
    for q in ps: q.start()
    for q in ps: q.join()
    if any(q.exitcode != 0 for q in ps):
        raise SystemExit(f"rollout workers failed: {[q.exitcode for q in ps]}")
    return [s for w in range(workers) for s in json.loads((out / f"shard_{w:02d}.json").read_text())]


def load_dets(dirs, dev):
    parts = [np.load(f) for d in dirs for f in sorted(Path(d).glob("shard_*.npz"))]
    x = torch.from_numpy(np.concatenate([p["x"] for p in parts]))
    y = torch.from_numpy(np.concatenate([p["ball"] for p in parts]))
    ep = torch.from_numpy(np.concatenate([p["ep"] + 10_000_000 * i for i, p in enumerate(parts)]))
    keep = x[:, 11] > 0  # only frames where the ball is visible (otherwise EstRunner holds the last estimate)
    val = ((ep % 20) == 0) & keep  # every 20th episode is validation
    return x.to(dev), y.to(dev), torch.nonzero(keep & ~val)[:, 0].to(dev), torch.nonzero(val)[:, 0].to(dev)


def train(a):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wide = Path(a.data_root) / f"{a.name}_wide"
    if not (wide / "shard_00.npz").exists():
        print(f"collecting {a.frames} wide frames -> {wide}", flush=True)
        st = rollout(wide, a.frames, a.workers, a.controller, a.noise_max, a.seed, 3, start_radius=0.065, start_speed=0.3, cam_rand=a.cam_rand, camera=a.camera)
        print(f"  {len(st)} episodes, mean length {np.mean([s['length'] for s in st]):.0f}", flush=True)
    else:
        print(f"reusing the data already collected in {wide} (use a new --name to collect again)", flush=True)
    x, y, tr, va = load_dets([wide], dev)
    print(f"{len(tr)} train / {len(va)} val frames", flush=True)
    if x.shape[1] != CAMERAS[a.camera]["n_in"]:
        raise SystemExit(f"{wide} holds {x.shape[1]}-input data, not {a.camera} data: use a new --name")
    m = Estimator(n_in=x.shape[1]).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=a.lr)
    g = torch.Generator(device=dev); g.manual_seed(a.seed)
    w = torch.tensor([1.0, 1.0, 0.25, 0.25], device=dev)  # position weighted over velocity
    for e in range(a.epochs):
        for pg in opt.param_groups:
            pg["lr"] = a.lr * (1 - e / a.epochs)
        m.train()
        perm = tr[torch.randperm(len(tr), generator=g, device=dev)]
        for i in range(0, len(perm) - a.batch + 1, a.batch):
            j = perm[i:i + a.batch]
            xa = augment(x[j], g, 1 - a.r_aug, 1 + a.r_aug)
            loss = (((m(xa) - y[j]) ** 2) * w).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if (e + 1) % 5 == 0 or e == a.epochs - 1:
            m.eval()
            with torch.no_grad():
                p = m(x[va]); t = y[va]
                d = t[:, :2].norm(dim=1) * 8
                pe = (p[:, :2] - t[:, :2]).abs() * 8
                ve = (p[:, 2:] - t[:, 2:]).abs() * 50
                msg = " | ".join(f"{lo}-{hi}cm u {pe[(d >= lo) & (d < hi), 0].mean():.2f} w {pe[(d >= lo) & (d < hi), 1].mean():.2f}" for lo, hi in [(0, 1), (1, 3), (3, 5), (5, 8)])
                print(f"epoch {e + 1}: pos err (cm) {msg} | vel err {ve.mean(0)[0]:.1f}/{ve.mean(0)[1]:.1f} cm/s", flush=True)
    out = ROOT / "outputs/estimator"; out.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model=m.cpu().state_dict(), arch=dict(n_in=m.n_in), camera=a.camera, args=vars(a)), out / f"{a.name}.pt")
    print(f"saved {out / f'{a.name}.pt'}")


_G: dict = {}


def _init(cfg):
    from sim.env import BalanceEnv
    from sim.policies import load_controller
    torch.set_num_threads(1)
    _G["cfg"], _G["controller"] = cfg, load_controller(cfg["controller"])
    model = load_est(cfg["est"])
    _G["cam"] = CAMERAS[model.camera]
    _G["env"] = BalanceEnv(obs_mode="image", camera=model.camera, image_hw=_G["cam"]["image_hw"], domain_rand=True, light_model=False,
                           grip_rot_deg=-90.0, act_hist=_G["controller"].act_hist, max_latency=cfg["max_lat"],
                           servo_rand=cfg["servo_rand"], cam_rand=cfg["cam_rand"])
    _G["est"] = EstRunner(model)
    from sim.env import FdVel
    _G["fd"] = FdVel()


def _episode(job):
    from perception.detector import detect
    from sim.policies import controller_action
    lat, seed = job
    cfg, env, est = _G["cfg"], _G["env"], _G["est"]
    obs, _ = env.reset(seed=seed)
    env._action_delay = lat
    env._delay_buf = [np.zeros(5)] * lat
    rng = np.random.default_rng(seed + 777)
    nudges = {int(t): rng.uniform(0, 2 * np.pi) for t in rng.integers(30, 280, cfg["nudges"])}
    est.reset(); _G["fd"].reset()
    errs = []
    for t in range(env.max_steps):
        cam = _G["cam"]
        det = detect(obs["image"], **cam["det"]); det[2] *= cfg["r_scale"]
        tobs = env.controller_obs()
        b = est.update(det, joints=tobs[4:9])
        b = _G["fd"].update(b[:2])  # velocity from FdVel, as in training and on the robot
        errs.append(np.r_[np.abs(b[:2] - tobs[:2]) * 8, np.abs(b[2:] - tobs[2:4]) * 50])
        tobs[:4] = b
        a = controller_action(_G["controller"], tobs)
        if t in nudges:
            env.d.qvel[env.ball_vadr:env.ball_vadr + 2] += 0.25 * np.array([np.cos(nudges[t]), np.sin(nudges[t])])
        obs, r, te, tr, info = env.step(a)
        if te or tr:
            break
    e = np.mean(errs, 0)
    return dict(lat=lat, success=bool(info["is_success"]), length=t + 1, dist_mm=float(info["dist"] * 1000), e=e.tolist())


def evaluate(a):
    if not Path(a.eval).exists():
        raise SystemExit(f"no checkpoint {a.eval}")
    check_ram(1.5 * 450 * a.workers, "Use fewer --workers.")
    lats =[int(v) for v in a.latency.split(",")]
    jobs = [(lat, 10_000 + i) for lat in lats for i in range(a.episodes)]
    with mp.get_context("spawn").Pool(a.workers, _init, (dict(est=a.eval, controller=a.controller, nudges=a.nudges, r_scale=a.r_scale, max_lat=max(lats), servo_rand=a.servo_rand, cam_rand=a.cam_rand),)) as pool:
        res = pool.map(_episode, jobs, chunksize=1)
    print(f"\n{Path(a.controller).name} on estimator {a.eval} ({load_est(a.eval).camera} camera) | servo_rand {a.servo_rand} cam_rand {a.cam_rand} | {a.episodes} episodes per row | nudges {a.nudges} | blob radius x{a.r_scale}")
    print(f"{'latency':>8} {'success':>9} {'len':>5} {'final dist':>11}   est err u/w (cm), du/dw (cm/s)")
    for lat in lats + ([None] if len(lats) > 1 else []):
        rs = [r for r in res if lat is None or r["lat"] == lat]
        e = np.mean([r["e"] for r in rs], 0)
        print(f"{'all' if lat is None else lat:>8} {sum(r['success'] for r in rs):>4}/{len(rs):<4} {np.mean([r['length'] for r in rs]):5.0f} "
              f"{np.mean([r['dist_mm'] for r in rs]):8.1f} mm   {e[0]:.2f} / {e[1]:.2f}, {e[2]:.1f} / {e[3]:.1f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="est_dev")
    p.add_argument("--frames", type=int, default=500_000, help="frames to collect")
    p.add_argument("--noise-max", type=float, default=0.3)
    p.add_argument("--cam-rand", type=float, default=0.3, help="camera randomization scale for the collected data")
    p.add_argument("--r-aug", type=float, default=0.07, help="blob radius augmentation +-this")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--camera", default="wrist", choices=sorted(CAMERAS), help="training: which camera's data (eval reads it from the checkpoint)")
    p.add_argument("--controller", default=str(ROOT / "weights/controller.pt"))
    p.add_argument("--data-root", default=str(ROOT / "data"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval", default=None, help="estimator checkpoint to evaluate (skips training)")
    p.add_argument("--latency", default="0,1,2,3")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--nudges", type=int, default=0)
    p.add_argument("--r-scale", type=float, default=1.0, help="eval: scale the detected blob radius")
    p.add_argument("--servo-rand", action="store_true", help="eval: randomized servos")
    a = p.parse_args()
    evaluate(a) if a.eval else train(a)


if __name__ == "__main__":
    main()
