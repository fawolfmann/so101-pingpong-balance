"""Stress eval: success rate vs kick strength and episode length, under the controller's training randomization.

    uv run python -m sim.stress_eval                                   # 6 kick speeds x 10/20/30 s, 40 episodes per cell
    uv run python -m sim.stress_eval --speeds 0,0.5 --seconds 10 --episodes 20

One kick per --kick-every seconds (a random time in each window), each adding the kick speed to the ball's sideways
velocity in a random direction. The controller was trained on 10 s episodes with no kicks. Same seeds in every cell
and for every checkpoint, so results are comparable (e.g. PPO vs GRPO). Writes outputs/stress/<name>.json and .png.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sim.ram import check_ram
from sim.stress import CONTROL_HZ, run_parallel

ROOT = Path(__file__).parent.parent
INK, INK_2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # sequential, light -> dark


def summarize(res: list[dict], speeds: list[float], secs: list[float]) -> dict:
    """{(speed, seconds): (success rate, mean survival seconds, mean action jitter)}."""
    out = {}
    for s in speeds:
        for sec in secs:
            rs = [r for r in res if r["kick_speed"] == s and r["seconds"] == sec]
            out[(s, sec)] = (np.mean([r["success"] for r in rs]), np.mean([r["length"] for r in rs]) / CONTROL_HZ,
                             np.mean([r["jitter"] for r in rs]))
    return out


def plot(table: dict, speeds: list[float], secs: list[float], title: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    grid = np.array([[table[(s, sec)][0] * 100 for sec in secs] for s in speeds])
    fig, ax = plt.subplots(figsize=(1.6 + 1.5 * len(secs), 1.2 + 0.55 * len(speeds)), facecolor=SURFACE)
    ax.imshow(grid, cmap=LinearSegmentedColormap.from_list("blue", BLUE_RAMP), vmin=0, vmax=100, aspect="auto")
    for i, s in enumerate(speeds):
        for j, sec in enumerate(secs):
            rate, surv, jit = table[(s, sec)]
            ax.text(j, i, f"{rate * 100:.0f}%\n{surv:.1f} s · j {jit:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if rate * 100 > 55 else INK)
    ax.set_xticks(range(len(secs)), [f"{sec:g} s" for sec in secs], color=INK_2)
    ax.set_yticks(range(len(speeds)), [f"{s:g} m/s" for s in speeds], color=INK_2)
    ax.set_xlabel("episode length", color=INK_2); ax.set_ylabel("kick strength", color=INK_2)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.tick_params(length=0); [sp.set_visible(False) for sp in ax.spines.values()]
    fig.text(0.01, 0.01, "cell: success rate / mean time before drop or end · j = action jitter", color=INK_2, fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(path, dpi=150); plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--controller", default=str(ROOT / "weights/controller.pt"))
    p.add_argument("--speeds", default="0,0.05,0.1,0.15,0.2,0.25", help="kick strengths, m/s")
    p.add_argument("--seconds", default="10,20,30", help="episode lengths, s")
    p.add_argument("--kick-every", type=float, default=4.0, help="one kick per this many seconds")
    p.add_argument("--episodes", type=int, default=40, help="per cell")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=20_000)
    p.add_argument("--name", default=None, help="output file name (default: controller file name)")
    a = p.parse_args()
    speeds = [float(x) for x in a.speeds.split(",")]
    secs = [float(x) for x in a.seconds.split(",")]
    check_ram(400 * a.workers, "Use fewer --workers.")

    jobs = [dict(seed=a.seed + i, seconds=sec, kick_speed=s, kick_every=a.kick_every)
            for s in speeds for sec in secs for i in range(a.episodes)]
    res = run_parallel(a.controller, jobs, a.workers)
    table = summarize(res, speeds, secs)

    name = a.name or Path(a.controller).stem
    print(f"\n{a.controller} | {a.episodes} episodes per cell | 1 kick per {a.kick_every:g} s | training randomization")
    print("success rate (mean seconds before drop or end)")
    print(f"{'kick':>9} " + " ".join(f"{f'{sec:g} s':>14}" for sec in secs))
    for s in speeds:
        print(f"{s:>5g} m/s " + " ".join(f"{table[(s, sec)][0] * 100:>6.0f}% ({table[(s, sec)][1]:4.1f} s)" for sec in secs))
    jit = np.mean([r["jitter"] for r in res])
    print(f"action jitter (mean |a_t - a_t-1|, all cells): {jit:.3f}   (original controller ~0.03; visible shaking ~0.3)")

    out = ROOT / "outputs/stress"; out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(dict(args=vars(a), episodes=res)))
    plot(table, speeds, secs, f"{name}: success under kicks (1 per {a.kick_every:g} s)", out / f"{name}.png")
    print(f"wrote {out / name}.json and .png")


if __name__ == "__main__":
    main()
