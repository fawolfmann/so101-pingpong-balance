"""Train the PPO controller (true ball state, no camera). Recipe for the released weights: README.

    uv run python -m sim.train_controller --name my_controller --act-hist 3 --grip-rot -90

One iteration: rollout (steps x all envs) -> GAE -> PPO update -> log -> checkpoint. Writes outputs/controller/<name>/.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from sim.ppo import ActorCritic, compute_gae, ppo_update
from sim.vec_env import VecBalanceEnv

OUT = Path(__file__).parent.parent / "outputs" / "controller"


def save(obj, path: Path):
    """Write to a temp file, then rename: a crash mid-write cannot corrupt the checkpoint."""
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def bounce_kwargs(a) -> dict:
    """BalanceEnv settings for bouncing (empty when off, so older runs resume unchanged)."""
    kw = {}
    if getattr(a, "kick_z_max", 0.0) > 0:
        kw.update(kick_z_min=a.kick_z_min, kick_z_max=a.kick_z_max, kick_z_frac=a.kick_z_frac)
    if getattr(a, "contact_damp", ""):
        kw["contact_damp"] = tuple(float(v) for v in a.contact_damp.split(","))
    if getattr(a, "juggle_frac", 0.0) > 0 and getattr(a, "juggle_smooth", -1.0) >= 0:
        kw["juggle_smooth"] = a.juggle_smooth
    if getattr(a, "juggle_frac", 0.0) > 0:
        if getattr(a, "juggle_calm", -1.0) >= 0:
            kw["juggle_calm"] = a.juggle_calm
        kw["juggle_acc"] = getattr(a, "juggle_acc", 0.05)
        if getattr(a, "juggle_target", ""):
            kw["juggle_target"] = tuple(float(v) for v in a.juggle_target.split(","))
            kw["height_w"] = getattr(a, "height_w", 2.0)
        if getattr(a, "juggle_reward", "default") != "default":
            kw.update(juggle_reward=a.juggle_reward, dwell_w=a.dwell_w)
    return kw


def cam_bounce_kwargs(a) -> dict:
    """--cam-policy extras for bouncing / juggling: the critic's height inputs, the mode input, the tracker fallback."""
    kw = dict(height_info=True) if bounce_kwargs(a) else {}
    if getattr(a, "juggle_frac", 0.0) > 0:
        kw.update(juggle=True, juggle_frac=a.juggle_frac, juggle_drop=tuple(float(v) for v in a.juggle_drop.split(",")))
    if getattr(a, "track_gate", 0.0) > 0:
        kw["track_gate"] = a.track_gate
    if getattr(a, "cam_track_w", 0.0) > 0:
        kw["cam_track_w"] = a.cam_track_w
    if getattr(a, "cam_reversal_w", 0.0) > 0:
        kw["cam_reversal_w"] = a.cam_reversal_w
    if getattr(a, "cam_zone", 0.0) > 0:
        kw["cam_zone"] = a.cam_zone
    if getattr(a, "juggle_rest_frac", 0.0) > 0:
        kw["juggle_rest_frac"] = a.juggle_rest_frac
    return kw


def sd_actor_dim(a) -> int:
    """The actor input size these flags give (--cam-policy)."""
    from sim.cam_policy_env import obs_dims
    return obs_dims(a.frames, a.act_hist, bool(bounce_kwargs(a)), getattr(a, "juggle_frac", 0.0) > 0,
                    bool(getattr(a, "juggle_target", "")))[0]


def init_weights(a, critic_dim: int) -> dict:
    """--init-from: the checkpoint's weights; a --cam-policy one without the height inputs gets zero weights for them."""
    sd = torch.load(a.init_from, map_location="cpu", weights_only=True)["model"]
    if getattr(a, "juggle_target", "") and sd["actor.0.weight"].shape[1] + 1 == sd_actor_dim(a):
        from sim.cam_policy_env import expand_for_height_cmd  # juggling checkpoint -> + the height command input
        return expand_for_height_cmd(sd, a.frames, a.act_hist)
    if getattr(a, "juggle_frac", 0.0) > 0 and sd["critic.0.weight"].shape[1] + 2 == critic_dim:
        from sim.cam_policy_env import expand_for_juggle  # bouncing checkpoint -> + mode and time-since-bounce inputs
        return expand_for_juggle(sd, a.frames, a.act_hist)
    if sd["critic.0.weight"].shape[1] + 2 == critic_dim and getattr(a, "cam_policy", False):
        from sim.cam_policy_env import expand_critic
        sd = expand_critic(sd, a.frames, a.act_hist)
    return sd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="controller_dev")
    p.add_argument("--iters", type=int, default=None, help="total iterations (default 500; on --resume the saved value)")
    p.add_argument("--steps", type=int, default=128, help="steps per env per iteration")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--envs-per-worker", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--reward-scale", type=float, default=0.1, help="rewards are multiplied by this for learning (logs stay unscaled)")
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--no-domain-rand", action="store_true")
    p.add_argument("--max-latency", type=int, default=2, help="randomized control latency 0..N extra steps")
    p.add_argument("--act-hist", type=int, default=1, help="past actions in the obs")
    p.add_argument("--home-penalty", type=float, default=0.0, help="reward -= this * sum(((q - home)/20deg)^2): keeps the arm near home")
    p.add_argument("--servo-rand", action="store_true", help="randomize servo stiffness, damping and force limit per episode")
    p.add_argument("--ball-noise", type=float, default=0.0, help="estimator-like error on the ball obs, scale U(0.5, 1.5) x this per episode (1 = as measured)")
    p.add_argument("--grip-rot", type=float, default=0.0, help="paddle grip angle (real grip: -90 = blade seen edge-on by the wrist camera)")
    p.add_argument("--smooth-penalty", type=float, default=0.01, help="reward -= this * sum((a - previous a)^2): discourages jitter")
    p.add_argument("--episode-s", type=float, default=10.0, help="episode length in seconds (30 control steps each)")
    p.add_argument("--kick-min", type=float, default=0.0, help="ball kick speed range, m/s (see sim/env.py); --kick-max 0 = no kicks")
    p.add_argument("--kick-max", type=float, default=0.0)
    p.add_argument("--kick-every", type=float, default=4.0, help="one kick per this many seconds")
    p.add_argument("--kick-margin", type=float, default=1.0, help="no kicks in the last this-many seconds (time to recover)")
    p.add_argument("--cam-policy", action="store_true", help="one policy for the arm + tracking camera, ball seen only through "
                   "simulated detections (sim/cam_policy_env.py); the critic also gets the true state")
    p.add_argument("--frames", type=int, default=3, help="--cam-policy: detection frames in the obs")
    p.add_argument("--cam-smooth", type=float, default=0.05, help="--cam-policy: reward -= this * sum((cam a - previous)^2)")
    p.add_argument("--kick-z-min", type=float, default=0.0, help="bouncing: upward kick component range, m/s (0.5 -> ~1 cm "
                   "hop, 1.5 -> ~11 cm); --kick-z-max 0 = off")
    p.add_argument("--kick-z-max", type=float, default=0.0)
    p.add_argument("--kick-z-frac", type=float, default=1.0, help="share of episodes with upward kicks (the rest: sideways only)")
    p.add_argument("--contact-damp", default="", help="elastic ball/blade contact, damping range 'lo,hi' per episode "
                   "(sim/env.py BOUNCE_STIFF: 20 -> restitution ~0.86, 40 -> ~0.75); empty = the original dead contact")
    p.add_argument("--track-gate", type=float, default=0.0, help="--cam-policy: tracker distance fallback, image units")
    p.add_argument("--juggle-frac", type=float, default=0.0, help="--cam-policy: share of juggling episodes (adds the mode "
                   "input; sim/env.py JUGGLE_*); the rest are balance episodes")
    p.add_argument("--juggle-drop", default="0.08,0.15", help="juggling: drop height range above the resting ball, m")
    p.add_argument("--juggle-smooth", type=float, default=-1.0, help="smoothness penalty in juggling episodes (sharp hits); "
                   "< 0 = same as --smooth-penalty")
    p.add_argument("--juggle-calm", type=float, default=-1.0, help="smoothness penalty between hits in juggling episodes; "
                   "< 0 = --smooth-penalty")
    p.add_argument("--juggle-acc", type=float, default=0.05, help="joint-acceleration penalty between hits (juggling)")
    p.add_argument("--juggle-target", default="", help="juggling: commanded bounce height range 'lo,hi' m, redrawn every "
                   "1-3 bounces (adds the height command input); empty = off")
    p.add_argument("--cam-zone", type=float, default=0.0, help="--cam-policy: with --cam-track-w, no penalty while the ball "
                   "is within this radius of the image centre (image units, 1 = edge), linear penalty outside it")
    p.add_argument("--juggle-reward", default="default", choices=["default", "clean"],
                   help="juggling reward: 'clean' = survival + smoothness + dwell penalty only (sim/env.py)")
    p.add_argument("--dwell-w", type=float, default=1.0, help="clean juggling: penalty per step with the ball sitting on the blade")
    p.add_argument("--juggle-rest-frac", type=float, default=0.0, help="juggling: share of episodes starting with the ball at rest")
    p.add_argument("--height-w", type=float, default=2.0, help="juggling: bonus per bounce at the commanded height")
    p.add_argument("--cam-track-w", type=float, default=0.0, help="--cam-policy: reward for keeping the ball near the image centre")
    p.add_argument("--cam-reversal-w", type=float, default=0.0, help="--cam-policy: penalty for pan/tilt direction reversals "
                   "(sum over axes of max(0, -rate_t * rate_t-1)); fast moves in one direction are not penalised")
    p.add_argument("--anchor-coef", type=float, default=0.0, help="with --init-from and --juggle-frac: keep the actions in "
                   "balance episodes (mode 0) close to the starting policy (KL weight); 0 = off")
    p.add_argument("--init-from", default="", help="start from this checkpoint's weights (a --cam-policy checkpoint without "
                   "the bouncing critic inputs gets zero weights for them)")
    p.add_argument("--init-std", type=float, default=0.0, help="starting exploration std (0 = the default 0.5, or with --init-from the checkpoint's)")
    p.add_argument("--critic-warmup", type=int, default=0, help="with --init-from: train only the critic for this many "
                   "iterations (the actor stays frozen until the value estimates fit the new task)")
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-entity", default=None, help="wandb username or team (needed if the account has no default entity)")
    p.add_argument("--wandb-resume", action="store_true", help="on --resume, continue the existing wandb run instead of starting a new one")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true", help="continue from <name>/last.pt (same --name); keeps the saved hyperparameters")
    a = p.parse_args()
    if not 0.0 <= a.kick_min <= a.kick_max:
        p.error("need 0 <= --kick-min <= --kick-max")
    if a.episode_s <= 0:
        p.error("--episode-s must be > 0")

    torch.manual_seed(a.seed)
    torch.set_num_threads(2)  # the CPU cores belong to the simulators
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = OUT / a.name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = None
    if a.resume:
        if not (run_dir / "last.pt").exists():
            raise SystemExit(f"--resume: {run_dir / 'last.pt'} not found")
        ckpt = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
        keep = {"iters", "save_every", "wandb", "wandb_entity", "wandb_resume", "resume", "workers", "envs_per_worker", "name"}  # may differ from the first launch
        for k, v in ckpt["args"].items():
            if k not in keep:
                setattr(a, k, v)
        a.iters = a.iters or ckpt["args"]["iters"]
        if ckpt["iter"] >= a.iters:
            raise SystemExit(f"already at iteration {ckpt['iter']} of {a.iters}; pass a larger --iters to train further")
    a.iters = a.iters or 500
    start_it = ckpt["iter"] if ckpt else 0
    (run_dir / "args.json").write_text(json.dumps(vars(a), indent=1))

    env = VecBalanceEnv(a.workers, a.envs_per_worker, seed=a.seed * 1000 + start_it * 50, domain_rand=not a.no_domain_rand,
                        max_latency=a.max_latency, act_hist=a.act_hist, grip_rot_deg=a.grip_rot,
                        home_penalty=a.home_penalty, servo_rand=a.servo_rand, ball_noise=a.ball_noise,
                        max_steps=round(getattr(a, "episode_s", 10.0) * 30), kick_min=getattr(a, "kick_min", 0.0),
                        kick_max=getattr(a, "kick_max", 0.0), kick_every=getattr(a, "kick_every", 4.0),
                        kick_margin=getattr(a, "kick_margin", 1.0), smooth_penalty=getattr(a, "smooth_penalty", 0.01),  # getattr: --resume of runs saved before these flags
                        **bounce_kwargs(a),
                        **(dict(cam_policy=True, frames=a.frames, cam_smooth=a.cam_smooth, cam_rand=0.3,
                                **cam_bounce_kwargs(a)) if getattr(a, "cam_policy", False) else {}))
    N, T = env.n_envs, a.steps
    ac = ActorCritic(env.obs_dim, env.act_dim, actor_dim=env.actor_dim).to(dev)
    if getattr(a, "init_from", "") and not ckpt:
        ac.load_state_dict(init_weights(a, env.obs_dim))
        print(f"initialized from {a.init_from}", flush=True)
    if getattr(a, "init_std", 0.0) > 0 and not ckpt:  # the starting exploration noise, from scratch or a checkpoint
        with torch.no_grad():
            ac.log_std.fill_(float(np.log(a.init_std)))
    anchor = None
    if getattr(a, "anchor_coef", 0.0) > 0 and getattr(a, "init_from", ""):
        ref = ActorCritic(env.obs_dim, env.act_dim, actor_dim=env.actor_dim).to(dev)
        ref.load_state_dict(init_weights(a, env.obs_dim))
        ref.requires_grad_(False)
        mode_col = env.actor_dim - 1  # the mode input (1 juggling, 0 balance) is the actor's last input
        anchor = (lambda o: ref.dist(o).mean, lambda o: o[:, mode_col] < 0.5, a.anchor_coef)
        print(f"anchor: balance-episode actions kept close to {a.init_from} (coef {a.anchor_coef})", flush=True)
    opt = torch.optim.Adam(ac.parameters(), lr=a.lr, eps=1e-5)
    if ckpt:
        ac.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        print(f"resumed {a.name} at iteration {start_it} ({ckpt['steps']/1e6:.2f}M steps), training to iteration {a.iters}")
    if a.wandb:
        try:  # a wandb problem (or wandb not installed) must not kill the run: fall back to log.csv
            import wandb
            cont =bool(ckpt and a.wandb_resume)
            wandb.init(project="so101-balance", entity=a.wandb_entity, name=a.name if not ckpt or cont else a.name + "-resumed",
                       id=a.name if (cont or not ckpt) else None, resume="must" if cont else None, config=vars(a), dir=str(run_dir))
        except Exception as e:
            print(f"WARNING: wandb disabled ({type(e).__name__}: {str(e)[:200]}). Continuing with log.csv only.", flush=True)
            a.wandb = False
    log_path = run_dir / "log.csv"
    writer = None
    if ckpt and log_path.exists():  # resume: append with the same columns
        header = log_path.read_text().splitlines()[0].split(",")
        csv_f = open(log_path, "a", newline="")
        writer = csv.DictWriter(csv_f, fieldnames=header, extrasaction="ignore")
    else:
        csv_f = open(log_path, "w", newline="")

    obs = env.reset()
    recent = deque(maxlen=200)  # stats of the last finished episodes
    total0 = ckpt["steps"] if ckpt else 0
    t0, total = time.time(), total0
    best = ckpt.get("best", -1.0) if ckpt else -1.0
    for it in range(start_it + 1, a.iters + 1):
        for g in opt.param_groups:  # linear lr decay to 0
            g["lr"] = a.lr * (1 - (it - 1) / a.iters)

        # 1. rollout
        B = dict(obs=np.zeros((T, N, env.obs_dim), np.float32), act=np.zeros((T, N, env.act_dim), np.float32),
                 logp=np.zeros((T, N), np.float32), val=np.zeros((T, N), np.float32), rew=np.zeros((T, N), np.float32),
                 done=np.zeros((T, N), np.float32), vnext=np.zeros((T, N), np.float32))
        final_idx, final_obs = [], []
        for t in range(T):
            o = torch.as_tensor(obs, device=dev)
            act, logp, val = ac.act(o)
            act_np = act.cpu().numpy()
            nobs, rew, term, trunc, fobs, eps = env.step(act_np)
            B["obs"][t], B["act"][t], B["logp"][t], B["val"][t] = obs, act_np, logp.cpu().numpy(), val.cpu().numpy()
            B["rew"][t] = rew * a.reward_scale
            B["done"][t] = term | trunc
            for i in np.flatnonzero(trunc & ~term):  # time limit: bootstrap from the true last state
                final_idx.append((t, i)); final_obs.append(fobs[i])
            recent.extend(eps)
            obs = nobs
        total += T * N

        # 2. advantages
        with torch.no_grad():
            v_all = ac.value(torch.as_tensor(np.concatenate([B["obs"][1:], obs[None]]), device=dev)).cpu().numpy()
            B["vnext"][:] = np.where(B["done"] > 0, 0.0, v_all)  # ball fell -> 0; time-limit entries fixed below
            if final_obs:
                vf = ac.value(torch.as_tensor(np.stack(final_obs), device=dev)).cpu().numpy()
                for (t, i), v in zip(final_idx, vf):
                    B["vnext"][t, i] = v
        adv, ret = compute_gae(B["rew"], B["val"], B["vnext"], B["done"], a.gamma, a.lam)

        # 3. PPO update (during --critic-warmup only the critic learns)
        warm = it <= getattr(a, "critic_warmup", 0)
        for prm in [*ac.actor.parameters(), ac.log_std]:
            prm.requires_grad_(not warm)
        flat = lambda x: torch.as_tensor(x.reshape(T * N, *x.shape[2:]), device=dev)
        batch = dict(obs=flat(B["obs"]), act=flat(B["act"]), logp=flat(B["logp"]), adv=flat(adv), ret=flat(ret))
        s = ppo_update(ac, opt, batch, ent_coef=a.ent_coef, anchor=anchor)

        # 4. logging
        row = dict(iter=it, steps=total, fps=int((total - total0) / (time.time() - t0)), std=float(ac.log_std.exp().mean().detach()), **s)
        nan = float("nan")  # every column exists from the first row, even before an episode has finished
        row.update(ep_return=nan, ep_len=nan, success=nan, **{f"success_lat{L}": nan for L in range(a.max_latency + 1)})
        if getattr(a, "kick_z_max", 0.0) > 0:
            row.update(success_upkick=nan, success_sidekick=nan)
        if getattr(a, "juggle_frac", 0.0) > 0:
            row.update(juggle_s=nan, juggle_bounces=nan, juggle_stalled=nan, success_balance=nan, jitter_between=nan, hit_tilt_deg=nan,
                       apex_std_cm=nan, target_err_cm=nan, cam_flips_per_s=nan, apex_mean_cm=nan, on_blade_frac=nan)
        if recent:
            row.update(ep_return=float(np.mean([e["ret"] for e in recent])), ep_len=float(np.mean([e["length"] for e in recent])),
                       success=float(np.mean([e["success"] for e in recent])))
            for L in range(a.max_latency + 1):
                es = [e["success"] for e in recent if e["latency"] == L]
                row[f"success_lat{L}"] = float(np.mean(es)) if es else float("nan")
            if getattr(a, "juggle_frac", 0.0) > 0:
                js = [e for e in recent if e.get("juggle")]
                row["juggle_s"] = float(np.mean([e["juggle_s"] for e in js])) if js else float("nan")
                row["juggle_bounces"] = float(np.mean([e["n_bounces"] for e in js])) if js else float("nan")
                row["juggle_stalled"] = float(np.mean([e["stalled"] for e in js])) if js else float("nan")
                row["jitter_between"] = float(np.nanmean([e.get("jitter", np.nan) for e in js])) if js else float("nan")
                row["hit_tilt_deg"] = float(np.nanmean([e.get("tilt", np.nan) for e in js])) if js else float("nan")
                row["apex_std_cm"] = 100 * float(np.nanmean([e.get("apex_std", np.nan) for e in js])) if js else float("nan")
                row["target_err_cm"] = 100 * float(np.nanmean([e.get("target_err", np.nan) for e in js])) if js else float("nan")
                row["cam_flips_per_s"] = float(np.nanmean([e.get("cam_flips", np.nan) for e in js])) if js else float("nan")
                row["apex_mean_cm"] = 100 * float(np.nanmean([e.get("apex_mean", np.nan) for e in js])) if js else float("nan")
                row["on_blade_frac"] = float(np.nanmean([e.get("on_blade", np.nan) for e in js])) if js else float("nan")
                bs = [e["success"] for e in recent if not e.get("juggle")]
                row["success_balance"] = float(np.mean(bs)) if bs else float("nan")
            if getattr(a, "kick_z_max", 0.0) > 0:
                for name, z in (("success_upkick", True), ("success_sidekick", False)):
                    es = [e["success"] for e in recent if e.get("kick_z", False) == z]
                    row[name] = float(np.mean(es)) if es else float("nan")
        if writer is None:
            writer = csv.DictWriter(csv_f, fieldnames=list(row)); writer.writeheader()
        writer.writerow(row); csv_f.flush()
        if a.wandb:
            wandb.log(row, step=total)
        if it % 10 == 0 or it == 1:
            eta = (time.time() - t0) / (it - start_it) * (a.iters - it) / 60
            print(f"[{'#' * int(20 * it / a.iters):<20}] {100 * it / a.iters:3.0f}% eta {eta:4.0f} min | it {it:4d}/{a.iters} steps {total/1e6:5.2f}M fps {row['fps']:5d} | ret {row.get('ep_return', float('nan')):7.1f} len {row.get('ep_len', float('nan')):5.0f} "
                  f"succ {row.get('success', float('nan')):.2f} ({' '.join(f'lat{L} {row.get(f'success_lat{L}', float('nan')):.2f}' for L in range(a.max_latency + 1))}) "
                  f"| kl {s['kl']:.4f} clip {s['clipfrac']:.2f} std {row['std']:.2f}"
                  + (f" | up-kick {row['success_upkick']:.2f} side-kick {row['success_sidekick']:.2f}"
                     if getattr(a, "kick_z_max", 0.0) > 0 else "")
                  + (f" | juggle {row['juggle_s']:.1f} s {row['juggle_bounces']:.1f} bounces stall {row['juggle_stalled']:.2f}"
                     f" jitter {row['jitter_between']:.3f} tilt {row['hit_tilt_deg']:.1f}°"
                     f" apex {row['apex_mean_cm']:.1f}±{row['apex_std_cm']:.1f} cm on-blade {row['on_blade_frac']:.2f}"
                     f" cam flips {row['cam_flips_per_s']:.1f}/s"
                     f" balance {row['success_balance']:.2f}" if getattr(a, "juggle_frac", 0.0) > 0 else ""), flush=True)

        # 5. checkpoints
        if it % a.save_every == 0 or it == a.iters:
            if recent and row["success"] >= best and len(recent) >= 100:
                best = row["success"]
                save(dict(model=ac.state_dict(), args=vars(a), iter=it, steps=total), run_dir / "best.pt")
            save(dict(model=ac.state_dict(), opt=opt.state_dict(), args=vars(a), iter=it, steps=total, best=best), run_dir / "last.pt")
    env.close()
    print(f"done: {total/1e6:.2f}M steps in {(time.time()-t0)/60:.1f} min, checkpoints in {run_dir}")


if __name__ == "__main__":
    main()
