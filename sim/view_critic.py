"""Live viewer: the controller under kicks of different strengths, with the critic value plotted over the scene.

    uv run mjpython -m sim.view_critic                                  # macOS (mjpython); Linux: python
    uv run mjpython -m sim.view_critic --kicks 0.1,0.2,0.3 --kick-every 2 --seconds 20

Each episode kicks the ball with --kicks in order (m/s, then repeats), one every --kick-every seconds. The plot (last 10 s) shows:
  blue    critic value (expected future reward)
  orange  ball distance from the paddle centre, cm
  gray    the edge: past ~6.8 cm the ball is off
After a drop the sim runs 1 s more so you see the ball fall. Training randomization is on (latency, soft servos, noisy ball state). Close the window to stop.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch

from sim.policies import controller_action, load_controller
from sim.stress import CONTROL_HZ, make_env

WINDOW_S = 10.0
AFTER_END_S = 1.0  # keep simulating this long after the episode ends
LINES = [("critic value", (0.16, 0.47, 0.84)), ("ball dist (cm)", (0.92, 0.41, 0.20)), ("edge", (0.55, 0.55, 0.52))]


def make_figure(y_lo: float, y_hi: float) -> mujoco.MjvFigure:
    fig = mujoco.MjvFigure()
    mujoco.mjv_defaultFigure(fig)
    fig.title = "critic value vs ball distance"
    fig.xlabel = "seconds"
    fig.flg_extend = 0
    fig.flg_legend = 1
    fig.range[0] = [-WINDOW_S, 0]
    fig.range[1] = [y_lo, y_hi]
    fig.gridsize = [6, 7]
    fig.figurergba[3] = 0.6
    for i, (name, rgb) in enumerate(LINES):
        fig.linename[i] = name
        fig.linergb[i] = rgb
    fig.linewidth = 2.0
    return fig


def update_figure(fig: mujoco.MjvFigure, hist: list[tuple[float, float, float]], edge_cm: float) -> None:
    """hist: (time s, value, dist cm), newest last; plots the last WINDOW_S seconds against time-from-now."""
    if not hist:
        return
    now = hist[-1][0]
    pts = [h for h in hist if h[0] >= now - WINDOW_S][-mujoco.mjMAXLINEPNT:]
    xs = np.array([p[0] - now for p in pts])
    for i, ys in enumerate([np.array([p[1] for p in pts]), np.array([p[2] for p in pts]), np.full(len(pts), edge_cm)]):
        fig.linepnt[i] = len(pts)
        fig.linedata[i, 0:2 * len(pts):2] = xs
        fig.linedata[i, 1:2 * len(pts):2] = ys


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--controller", default=str(Path(__file__).parent.parent / "weights/controller.pt"))
    p.add_argument("--kicks", default="0.1,0.15,0.2,0.25", help="kick strengths to cycle through, m/s")
    p.add_argument("--kick-every", type=float, default=3.0)
    p.add_argument("--seconds", type=float, default=30.0, help="episode length")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    kicks = [float(k) for k in a.kicks.split(",")]

    ac = load_controller(a.controller)
    env = make_env(ac, max_steps=round(a.seconds * CONTROL_HZ), light_model=False)  # full visual model: arm + paddle drawn
    edge_cm = (env.blade_radius - 0.5 * env.ball_radius) * 100
    fig = make_figure(-2.0, 10.0)
    rng = np.random.default_rng(a.seed)
    seed, episodes, balanced, n_kicks = a.seed, 0, 0, 0
    last_kick = "none yet"
    env.reset(seed=seed)
    hist: list[tuple[float, float, float]] = []
    end_step = None  # set when the episode ends; the sim keeps running until then so you see the ball fall
    status = "balancing"

    # side panels hidden so the plot has the right side (Tab / Shift+Tab bring them back)
    with mujoco.viewer.launch_passive(env.m, env.d, show_left_ui=False, show_right_ui=False) as v:
        v.cam.lookat[:] = [0.41, 0.0, 0.30]; v.cam.distance = 0.9; v.cam.azimuth = 150; v.cam.elevation = -25
        while v.is_running():
            t0 = time.time()
            obs = env.controller_obs()
            with torch.no_grad():
                value = float(ac.value(torch.as_tensor(obs[None]))[0])
            every = round(a.kick_every * CONTROL_HZ)
            if end_step is None and env.t > 0 and env.t % every == 0:
                speed = kicks[n_kicks % len(kicks)]; ang = rng.uniform(0, 2 * np.pi); n_kicks += 1
                env.d.qvel[env.ball_vadr:env.ball_vadr + 2] += speed * np.array([np.cos(ang), np.sin(ang)])
                last_kick = f"{speed:g} m/s at {env.t / CONTROL_HZ:.1f} s"
            _, _, te, tr, info = env.step(controller_action(ac, obs))
            hist.append((env.t / CONTROL_HZ, value, info["dist"] * 100))

            vp = v.viewport
            if vp is not None:
                w, h = int(vp.width * 0.34), int(vp.height * 0.5)  # right side, vertically centred
                update_figure(fig, hist, edge_cm)
                v.set_figures((mujoco.MjrRect(vp.left + vp.width - w, vp.bottom + (vp.height - h) // 2, w, h), fig))
            v.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                         "episode\nstatus\ntime\nlast kick\nnext kick\nbalanced",
                         f"{episodes + 1} (latency {env._action_delay})\n{status}\n{env.t / CONTROL_HZ:.1f} / {a.seconds:g} s\n"
                         f"{last_kick}\n{kicks[n_kicks % len(kicks)]:g} m/s\n{balanced}/{episodes}"))
            v.sync()

            if end_step is None and (te or tr):
                episodes += 1; balanced += int(info["is_success"])
                status = "balanced" if info["is_success"] else f"DROPPED at {env.t / CONTROL_HZ:.1f} s"
                print(f"episode {episodes}: {status} (last kick {last_kick}) | {balanced}/{episodes}", flush=True)
                end_step = env.t + round(AFTER_END_S * CONTROL_HZ)  # keep simulating: watch the ball fall
            if end_step is not None and env.t >= end_step:
                seed += 1; env.reset(seed=seed); hist = []; n_kicks = 0; last_kick = "none yet"
                end_step, status = None, "balancing"
            time.sleep(max(0.0, env.n_substeps * env.m.opt.timestep - (time.time() - t0)))


if __name__ == "__main__":
    main()
