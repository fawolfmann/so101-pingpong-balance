"""Critic value as a "drop risk" signal: does PPO's critic see a drop coming, and how early?

    uv run python -m sim.critic_plot                       # 120 episodes, 20 s, 0.2 m/s kick every 3 s
    uv run python -m sim.critic_plot --kick-speed 0.75 --episodes 200

The critic predicts the discounted future reward. A drop costs -5 and ends the reward stream, so the value should
fall before the ball does. An alarm = the value staying below a threshold until the drop. For each threshold
(a percentile of the values seen in balanced episodes) it reports:
  caught      share of drops with an alarm at least 1 step before the drop
  lead        median warning time before the drop
  false alarm share of balanced episodes whose value ever dipped below the threshold
Writes outputs/critic/<name>.png and .json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sim.ram import check_ram
from sim.stress import CONTROL_HZ, run_parallel, warning_lead

ROOT = Path(__file__).parent.parent
INK, INK_2, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb", "#e6e5e0"
OK, DROP = "#2a78d6", "#eb6834"  # categorical slots 1-2: balanced, dropped
PERCENTILES = (0.5, 1, 2, 5, 10)
PRE_DROP_S = 3.0


def alarm_table(ok: list[dict], drops: list[dict]) -> list[dict]:
    pool = np.concatenate([e["value"] for e in ok]) if ok else np.array([0.0])
    rows = []
    for pct in PERCENTILES:
        thr = float(np.percentile(pool, pct))
        leads = np.array([warning_lead(np.array(e["value"]), thr) for e in drops])
        rows.append(dict(pct=pct, threshold=thr, caught=float(np.mean(leads > 0)) if len(leads) else float("nan"),
                         lead_s=float(np.median(leads[leads > 0]) / CONTROL_HZ) if np.any(leads > 0) else 0.0,
                         false_alarm=float(np.mean([np.min(e["value"]) < thr for e in ok])) if ok else float("nan")))
    return rows


def _style(ax) -> None:
    ax.set_facecolor(SURFACE); ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)


def plot(ok: list[dict], drops: list[dict], rows: list[dict], title: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2), facecolor=SURFACE, gridspec_kw=dict(width_ratios=(1.25, 1)))
    # left: example episodes over time, kicks marked
    for e, c in [(x, OK) for x in ok[:2]] + [(x, DROP) for x in drops[:2]]:
        t = np.arange(len(e["value"])) / CONTROL_HZ
        a1.plot(t, e["value"], color=c, lw=2, alpha=0.9)
        a1.plot([k / CONTROL_HZ for k in e["kicks"] if k < len(t)], [e["value"][k] for k in e["kicks"] if k < len(t)],
                "v", color=c, ms=8, mec=SURFACE, mew=2)
        if not e["success"]:
            a1.plot(t[-1], e["value"][-1], "X", color=c, ms=10, mec=SURFACE, mew=2)
    a1.set_xlabel("time (s)   ▼ kick   ✕ drop", color=INK_2, fontsize=9); a1.set_ylabel("critic value", color=INK_2, fontsize=9)
    a1.set_title("Example episodes", color=INK, fontsize=10, loc="left")
    a1.plot([], [], color=OK, lw=2, label="balanced"); a1.plot([], [], color=DROP, lw=2, label="dropped")
    a1.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    # right: every drop aligned on the drop time, against the balanced band
    n = round(PRE_DROP_S * CONTROL_HZ)
    if ok:
        lo, hi = np.percentile(np.concatenate([e["value"] for e in ok]), [5, 95])
        a2.axhspan(lo, hi, color=OK, alpha=0.12, lw=0)
        a2.text(-PRE_DROP_S, hi, " balanced episodes, 5-95%", color=INK_2, fontsize=8, va="bottom")
    tail = [np.array(e["value"][-n:]) for e in drops]
    for v in tail:
        a2.plot(-np.arange(len(v))[::-1] / CONTROL_HZ, v, color=DROP, lw=1, alpha=0.25)
    full = [v for v in tail if len(v) == n]
    if full:
        a2.plot(-np.arange(n)[::-1] / CONTROL_HZ, np.median(full, axis=0), color=DROP, lw=2.5, label="dropped, median")
        a2.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="lower left")
    r = next(r for r in rows if r["pct"] == 1)
    a2.axhline(r["threshold"], color=MUTED, lw=1, ls="--")
    a2.text(-PRE_DROP_S, r["threshold"], f" alarm at 1st pct: catches {r['caught']:.0%}, median {r['lead_s']:.2f} s early, "
            f"{r['false_alarm']:.0%} false alarms", color=INK_2, fontsize=8, va="bottom")
    a2.set_xlabel("seconds before the drop", color=INK_2, fontsize=9)
    a2.set_title(f"Last {PRE_DROP_S:g} s before each drop (n={len(drops)})", color=INK, fontsize=10, loc="left")
    for ax in (a1, a2):
        _style(ax)
    fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--controller", default=str(ROOT / "weights/controller.pt"))
    p.add_argument("--episodes", type=int, default=120)
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--kick-speed", type=float, default=0.2)
    p.add_argument("--kick-every", type=float, default=3.0)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=30_000)
    p.add_argument("--name", default=None)
    a = p.parse_args()
    check_ram(400 * a.workers, "Use fewer --workers.")

    jobs = [dict(seed=a.seed + i, seconds=a.seconds, kick_speed=a.kick_speed, kick_every=a.kick_every, record=True)
            for i in range(a.episodes)]
    res = run_parallel(a.controller, jobs, a.workers)
    drops = [e for e in res if e["dropped"]]
    ok = [e for e in res if e["success"]]
    rows = alarm_table(ok, drops)

    name = a.name or Path(a.controller).stem
    print(f"\n{a.controller} | {a.episodes} episodes of {a.seconds:g} s | {a.kick_speed:g} m/s kick every {a.kick_every:g} s")
    print(f"balanced {len(ok)}, dropped {len(drops)}, other {len(res) - len(ok) - len(drops)} (ended off-centre)")
    print(f"{'threshold':>16} {'caught':>8} {'lead':>8} {'false alarm':>12}")
    for r in rows:
        print(f"{r['pct']:>5g} pct {r['threshold']:6.1f} {r['caught']:8.0%} {r['lead_s']:7.2f}s {r['false_alarm']:12.0%}")

    out = ROOT / "outputs/critic"; out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(dict(args=vars(a), alarms=rows, episodes=res)))
    plot(ok, drops, rows, f"{name}: critic value as drop warning ({a.kick_speed:g} m/s kick every {a.kick_every:g} s)",
         out / f"{name}.png")
    print(f"wrote {out / name}.png and .json")


if __name__ == "__main__":
    main()
