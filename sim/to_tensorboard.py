"""Collect every experiment under outputs/ into one TensorBoard log (re-run any time; it rebuilds from scratch).

    uv run --extra tensorboard python -m sim.to_tensorboard
    uv run --extra tensorboard tensorboard --logdir outputs/tensorboard

Runs (left panel), by prefix:
  controller/<run>   PPO / GRPO training curves from outputs/controller/<run>/log.csv (x = env steps); settings as text
  estimator/<name>   estimator training: position error per distance band and velocity error per epoch
  eval/<name>        closed-loop evaluations: success rate, episode length, final distance (x = latency, steps of 33 ms),
                     one tag group per number of nudges; the full table as text
  stress/<name>      kick stress test: success and time before drop (x = kick speed, cm/s) per episode length; plot image
  notes              the camera-model fits and the results sections of docs/ROADMAP.md, as text
Runs marked LEAKY used a tracker aimed from the true ball state; they are kept for the record, not for comparison.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
OUT = ROOT / "outputs"


def writer(logdir: Path, name: str):
    from torch.utils.tensorboard import SummaryWriter
    return SummaryWriter(str(logdir / name))


def code(text: str) -> str:
    """Text for TensorBoard's markdown view, kept as typed."""
    return "\n".join("    " + line for line in text.splitlines())


def controllers(logdir: Path) -> int:
    n = 0
    for run in sorted((OUT / "controller").glob("*/log.csv")):
        w = writer(logdir, f"controller/{run.parent.name}")
        with open(run) as f:
            for row in csv.DictReader(f):
                step = int(float(row["steps"]))
                for k, v in row.items():
                    if k in ("iter", "steps") or v in ("", None):
                        continue
                    x = float(v)
                    if math.isfinite(x):
                        group = "success_by_latency" if k.startswith("success_lat") else ("train" if k in ("ep_return", "ep_len", "success") else "ppo")
                        w.add_scalar(f"{group}/{k}", x, step)
        args = run.parent / "args.json"
        if args.exists():
            w.add_text("settings", code(json.dumps(json.loads(args.read_text()), indent=1)))
        w.close(); n += 1
    return n


EPOCH = re.compile(r"epoch (\d+): pos err \(cm\) (.*) \| vel err ([\d.]+)/([\d.]+)")
BAND = re.compile(r"(\d+-\d+cm) u ([\d.]+) w ([\d.]+)")


def estimators(logdir: Path) -> int:
    n = 0
    for log in sorted(OUT.glob("*.log")):
        lines = [EPOCH.search(l) for l in log.read_text(errors="ignore").splitlines()]
        lines = [m for m in lines if m]
        if not lines:
            continue
        w = writer(logdir, f"estimator/{log.stem}")
        for m in lines:
            ep = int(m.group(1))
            for band, u, wv in BAND.findall(m.group(2)):
                w.add_scalar(f"pos_err_cm_u/{band}", float(u), ep)
                w.add_scalar(f"pos_err_cm_w/{band}", float(wv), ep)
            w.add_scalar("vel_err_cm_s/u", float(m.group(3)), ep)
            w.add_scalar("vel_err_cm_s/w", float(m.group(4)), ep)
        w.close(); n += 1
    return n


ROW = re.compile(r"^\s+(\d+|all)\s+(\d+)/(\d+)\s+(\d+)\s+([\d.]+) mm(.*)$")


def eval_blocks(text: str):
    """(header, nudges, rows) per table in an evaluation log."""
    blocks, cur = [], None
    for line in text.splitlines():
        if "episodes per row" in line:
            nud = re.search(r"nudges (\d+)", line)
            cur = dict(header=line.strip(), nudges=int(nud.group(1)) if nud else 0, rows=[], lines=[line])
            blocks.append(cur)
        elif cur is not None and line.strip():
            cur["lines"].append(line)
            m = ROW.match(line)
            if m:
                cur["rows"].append(dict(lat=m.group(1), ok=int(m.group(2)), n=int(m.group(3)), length=float(m.group(4)),
                                        dist=float(m.group(5)), rest=m.group(6)))
    return blocks


def eval_name(log: Path, header: str) -> str:
    """Run name for one evaluation table: the policy (arm + camera policies) or the log, plus the test variant."""
    first = header.split(" | ")[0]
    name = Path(first).parent.name if first.startswith("outputs/controller/") else log.stem.removeprefix("eval_")
    name += "_simdet" if "simulated detections" in header else ""
    name += "_upkick" if "push up" in header else ""
    name += "_elastic" if "contact damp" in header else ""
    return name


def evals(logdir: Path) -> int:
    n = 0
    for log in sorted(OUT.glob("eval_*.log")):
        blocks = eval_blocks(log.read_text(errors="ignore"))
        if not blocks:
            continue
        runs: dict[str, list] = {}
        for b in blocks:  # one log can hold several variants (detections, upward pushes, contact) and policies
            runs.setdefault(eval_name(log, b["header"]), []).append(b)
        for name, bs in runs.items():
            w = writer(logdir, f"eval/{name}")
            for b in bs:
                tag = f"nudges{b['nudges']}"
                for r in b["rows"]:
                    if r["lat"] == "all":
                        w.add_scalar(f"overall/success_rate_{tag}", r["ok"] / r["n"], 0)
                        continue
                    lat = int(r["lat"])
                    w.add_scalar(f"{tag}/success_rate", r["ok"] / r["n"], lat)
                    w.add_scalar(f"{tag}/episode_length", r["length"], lat)
                    w.add_scalar(f"{tag}/final_dist_mm", r["dist"], lat)
                w.add_text(f"table_{tag}", code("\n".join(b["lines"])))
            w.close(); n += 1
    return n


def stress(logdir: Path) -> int:
    n = 0
    for f in sorted((OUT / "stress").glob("*.json")):
        d = json.loads(f.read_text())
        eps = d.get("episodes", [])
        if not eps:
            continue
        w = writer(logdir, f"stress/{f.stem}")
        for sec in sorted({e["seconds"] for e in eps}):
            for spd in sorted({e["kick_speed"] for e in eps}):
                es = [e for e in eps if e["seconds"] == sec and e["kick_speed"] == spd]
                x = int(round(spd * 100))
                w.add_scalar(f"{sec:g}s/success_rate", float(np.mean([e["success"] for e in es])), x)
                w.add_scalar(f"{sec:g}s/seconds_before_drop", float(np.mean([e["length"] for e in es])) / 30, x)
        w.add_text("settings", code(json.dumps(d.get("args", {}), indent=1)))
        png = f.with_suffix(".png")
        if png.exists():
            import cv2
            img = cv2.imread(str(png))[:, :, ::-1]
            w.add_image("plot", np.ascontiguousarray(img), dataformats="HWC")
        w.close(); n += 1
    return n


def notes(logdir: Path) -> None:
    w = writer(logdir, "notes")
    for log in ("fit_cam_model.log", "fit_cam_model_limits.log"):
        p = OUT / log
        if p.exists():
            w.add_text(f"camera_model/{p.stem}", code(p.read_text(errors="ignore")))
    road = (ROOT / "docs/ROADMAP.md").read_text()
    for sec in re.split(r"\n(?=### )", road):
        title = sec.splitlines()[0].lstrip("# ").strip()
        if "results" in title.lower():
            w.add_text(f"roadmap/{re.sub(r'[^a-z0-9]+', '_', title.lower()).strip('_')}", sec)
    w.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", default=str(OUT / "tensorboard"))
    a = p.parse_args()
    logdir = Path(a.logdir)
    if logdir.exists():
        shutil.rmtree(logdir)  # rebuilt from the source logs every time
    counts = dict(controller=controllers(logdir), estimator=estimators(logdir), eval=evals(logdir), stress=stress(logdir))
    notes(logdir)
    print(f"wrote {logdir}: " + ", ".join(f"{v} {k} runs" for k, v in counts.items()) + ", notes")
    print(f"view: uv run --extra tensorboard tensorboard --logdir {logdir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
