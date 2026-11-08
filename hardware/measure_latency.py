"""Measure the delay from a position command to (a) the joint reading moving and (b) the wrist image changing, in ms and 30 Hz steps.

    uv run python -m hardware.measure_latency PORT [CAMERA] [--trials 8] [--deg 6]

First clamp the paddle and go home with `uv run python -m hardware.run_real --grip-only`; no ball. wrist_flex steps down by --deg and back each trial. Torque stays on afterwards.
"""
import argparse
import threading
import time

import cv2
import numpy as np
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower

from hardware.camera import open_camera

STEP_MS = 1000 / 30


def cross(t, x, t_cmd, fracs=(0.1, 0.5)):
    """Time after t_cmd at which x has covered each fraction of its total change; None if it never clearly changed."""
    pre, end = x[t < t_cmd], x[t > t[-1] - 0.25]
    lvl, fin = np.median(pre), np.median(end)
    if abs(fin - lvl) < max(8 * pre.std(), 0.3):
        return [None] * len(fracs)
    y = (x - lvl) / (fin - lvl)
    out = []
    for f in fracs:
        hit = (t >= t_cmd) & (y >= f)
        out.append(float(t[hit][0] - t_cmd) if hit.any() else None)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("port"); p.add_argument("cam", nargs="?", default="/dev/video4")
    p.add_argument("--id", default="my_follower_arm", help="LeRobot calibration id of your arm")
    p.add_argument("--trials", type=int, default=8); p.add_argument("--deg", type=float, default=6.0)
    p.add_argument("--kp", type=int, default=32, help="servo P gain: keep it the same as in run_real")
    a = p.parse_args()

    cap = open_camera(a.cam)
    frames, stop = [], False

    def grab():
        while not stop:
            ok, f = cap.read(); t = time.perf_counter()
            if ok: frames.append((t, cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (80, 60), interpolation=cv2.INTER_AREA).astype(np.float32)))

    robot = SOFollower(SOFollowerRobotConfig(port=a.port, id=a.id, disable_torque_on_disconnect=False, position_p_coefficient=a.kp))
    robot.bus.connect()
    robot.bus.sync_write("Goal_Position", {k: float(v) for k, v in robot.bus.sync_read("Present_Position").items()})  # goal = present, so torque on does not jump
    robot.configure()  # gains (P=--kp), then torque ON
    key = "wrist_flex.pos"
    base = robot.get_observation()[key]
    th = threading.Thread(target=grab, daemon=True); th.start()
    time.sleep(1.0)
    print(f"holding {key} = {base:.1f} deg, stepping -{a.deg} deg and back, {a.trials} trials ...")
    res = []
    try:
        for i in range(a.trials):
            target = base - a.deg  # downward: upward ran into the joint limit at this pose on my arm
            time.sleep(1.2)
            n0 = len(frames)
            jt, jx = [], []
            t_end = time.perf_counter() + 0.5
            while time.perf_counter() < t_end:  # baseline before the command
                jt.append(time.perf_counter()); jx.append(robot.get_observation()[key])
            t_cmd = time.perf_counter(); robot.send_action({key: target})
            t_end = t_cmd + 1.0
            while time.perf_counter() < t_end:
                jt.append(time.perf_counter()); jx.append(robot.get_observation()[key])
            fr = frames[n0:]  # before the arm goes back: the image must end at the new position
            robot.send_action({key: base})
            time.sleep(0.8)
            ft = np.array([t for t, _ in fr]); ref = np.median(np.stack([f for t, f in fr if t < t_cmd][-5:]), axis=0)
            fx = np.array([np.abs(f - ref).mean() for _, f in fr])
            jt, jx = np.array(jt), np.array(jx)
            j10, j50 = cross(jt, jx, t_cmd); c10, c50 = cross(ft, fx, t_cmd)
            res.append((j10, j50, c10, c50))
            f = lambda v: f"{v*1000:5.0f}" if v is not None else "  nan"
            print(f"  trial {i+1}: joint 10% {f(j10)} / 50% {f(j50)} ms | image 10% {f(c10)} / 50% {f(c50)} ms | joint read rate {len(jt)/(jt[-1]-jt[0]):.0f} Hz")
    finally:
        stop = True; th.join(timeout=1); cap.release(); robot.disconnect()
    ok = [r for r in res if None not in r]
    if not ok:
        raise SystemExit("no usable trials (nothing moved?)")
    m = np.median(np.array(ok), axis=0) * 1000
    print(f"\nmedian over {len(ok)} usable trials (ms | 30 Hz steps):")
    for name, v in zip(["joint 10%", "joint 50%", "image 10%", "image 50%"], m):
        print(f"  command -> {name:9s}: {v:5.0f} ms | {v/STEP_MS:.1f} steps")
    print(f"  camera lag behind the joint reading: {m[2]-m[0]:.0f} ms at 10%, {m[3]-m[1]:.0f} ms at 50%")
    print("spread of image-50% over trials:", [f"{r[3]*1000:.0f}" for r in ok], "ms")
    print("NOTE: this includes the servo's own response, not only pure delay; the sim servo reaches 10% / 50% of a 6 deg step in 17 / 40 ms.")

if __name__ == "__main__":
    main()
