"""The controller balancing on the real arm, 30 Hz: wrist image -> detector -> estimator -> FdVel -> controller -> joint targets.

    uv run python -m hardware.run_real          # live: torque ON, the arm moves
    uv run python -m hardware.run_real --dry    # no motor writes, prints what it would do

Sequence: torque on holding the current pose -> keep or insert the paddle (prompt) -> squeeze -> move home -> wait until the ball is on the blade for 5 frames ->
control. If the ball is lost (--lost), the estimate leaves --max-ball or a target leaves --max-dev, it holds still and waits for the ball again.
Joint mapping: sim offset (rad) = radians(real_deg - home_deg); targets move by action * 0.04 rad per step, as in the sim.
Each run is logged to logs/real_<time>.npz.
"""
import argparse
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower

from hardware.camera import open_camera
from hardware.pose_guide import TARGET
from perception.detector import detect
from sim.env import FdVel
from perception.estimator import EstRunner, load_est
from sim.policies import load_controller, controller_action

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
HZ, ACTION_SCALE = 30, 0.04


def ask(msg):
    try:
        return input(msg).strip().lower()
    except KeyboardInterrupt:
        raise SystemExit("\nstopped: torque stays on.")


# Sim reference for --tilt-test: ball displacement (du, dw) in cm, 0.4 s after release on a blade held tilted +2 deg by one joint.
# Only the signs are compared. shoulder_pan is not tested: it cannot tilt a level blade (its sign is checked with --jog).
SIM_TILT = {"wrist_flex": (0.0, 1.64), "elbow_flex": (0.0, 1.64), "wrist_roll": (-1.63, 0.0)}


def tilt_test(a, robot, read, latest, est, home, bias, stop, cap, bus):
    """Joint sign check: you hold the ball at the centre, one joint tilts 3 deg, you release, the camera measures where it rolls."""
    import shutil, subprocess
    tts = shutil.which("spd-say") or shutil.which("say")  # Linux / macOS text-to-speech
    say = lambda txt: subprocess.Popen([tts, txt], stderr=subprocess.DEVNULL) if tts else print(f">> {txt.upper()}", flush=True)
    send = lambda deg_off: robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, home + bias + deg_off)})
    rows = []
    trials = [(j, d) for j in SIM_TILT for d in (3, -3)]
    try:
        for k, (jn, deg) in enumerate(trials):
            ji = JOINTS.index(jn); off = np.zeros(5)
            print(f"\ntrial {k + 1}/{len(trials)}: {jn} {deg:+d} deg. Hold the ball at the centre ...", flush=True)
            say(f"Trial {k + 1}. Hold the ball at the centre."); time.sleep(4)
            for s_ in np.linspace(0, 1, 10):
                off[ji] = s_ * deg; send(off); time.sleep(1 / 30)
            time.sleep(0.5)
            say("Release"); print("  RELEASE", flush=True)
            est.reset(); track = []; t0 = time.perf_counter()
            while time.perf_counter() - t0 < 2.0:
                img = cv2.cvtColor(cv2.resize(latest["f"], (128, 96), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
                det = detect(img); det[2] /= a.r_corr
                on_blade = det[3] > 0 and abs(det[0] - 0.2) < 0.25  # a ball on the edge-on blade is always near image x = +0.2
                b = est.update(det if on_blade else np.zeros(4, np.float32))
                if on_blade: track.append((time.perf_counter() - t0, b[0] * 8, b[1] * 8))
                time.sleep(1 / 30)
            for s_ in np.linspace(1, 0, 10):
                off[ji] = s_ * deg; send(off); time.sleep(1 / 30)
            say("Catch the ball")
            T = np.array(track) if track else np.zeros((0, 3))
            if len(T) < 6:
                print(f"  not enough ball frames on the blade ({len(T)}): this trial is skipped"); rows.append((jn, deg, np.nan, np.nan)); continue
            # motion = last 3 on-blade frames minus the first 3 after 0.4 s (skips the reaction delay)
            st = T[T[:, 0] > 0.4] if np.sum(T[:, 0] > 0.4) >= 4 else T
            du, dw = st[-3:, 1].mean() - st[:3, 1].mean(), st[-3:, 2].mean() - st[:3, 2].mean()
            print(f"  ball moved du {du:+.1f} dw {dw:+.1f} cm over {st[-1, 0] - st[0, 0]:.1f} s ({len(T)} frames)")
            rows.append((jn, deg, du, dw))
    finally:
        stop[0] = True; time.sleep(0.1); cap.release(); bus.disconnect(False)
    print("\nTILT TEST: real vs sim (the sign of the main axis must match)")
    for jn, deg, du, dw in rows:
        su, sw = np.array(SIM_TILT[jn]) * np.sign(deg)
        ax = 0 if abs(su) > abs(sw) else 1
        r_v, s_v = (du, dw)[ax], (su, sw)[ax]
        verdict = "skipped" if np.isnan(r_v) else ("MATCH" if np.sign(r_v) == np.sign(s_v) and abs(r_v) > 0.5 else ("too small" if abs(r_v) <= 0.5 else "FLIPPED"))
        print(f"  {jn:13s} {deg:+d} deg: real du {du:+5.1f} dw {dw:+5.1f} | sim moves {'u' if ax == 0 else 'w'} {'+' if s_v > 0 else '-'} | {verdict}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--est", default=str(ROOT / "weights/estimator.pt"))
    ap.add_argument("--controller", default=str(ROOT / "weights/controller.pt"))
    ap.add_argument("--home", default=str(ROOT / "hardware/home.json"), help="your level-paddle home pose (made by hardware.pose_guide)")
    ap.add_argument("--cam", default="/dev/video4"); ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--id", default="my_follower_arm", help="LeRobot calibration id of your arm")
    ap.add_argument("--seconds", type=float, default=0, help="total run time; 0 = until Ctrl+C")
    ap.add_argument("--r-corr", type=float, default=1.1, help="divide the real blob radius by this (the real ball looks ~10%% bigger than in sim)")
    ap.add_argument("--max-dev", type=float, default=25, help="stop if any joint target goes this many degrees from home")
    ap.add_argument("--max-ball", type=float, default=6.5, help="stop if the ball estimate is this many cm from the centre")
    ap.add_argument("--lost", type=int, default=10, help="hold still after this many frames without the ball")
    ap.add_argument("--kp", type=int, default=32, help="servo P gain (LeRobot default 16: my arm sagged 4-7 deg and ignored small commands)")
    ap.add_argument("--grip-load", type=int, default=100, help="gripper squeeze target: servo load in 0.1%% units (100 = 10%%; the overload protection trips at ~20%%); 0 = no squeeze")
    ap.add_argument("--grip-open", type=float, default=40, help="how far the gripper opens to insert the paddle, in %% of its calibrated range")
    ap.add_argument("--grip-only", action="store_true", help="only clamp the paddle and go home (if a home pose is saved), then exit; torque stays on")
    ap.add_argument("--jog", default=None, help="e.g. wrist_flex:5,elbow_flex:5: move each joint by that many degrees for 3 s and back (visual sign check)")
    ap.add_argument("--tilt-test", action="store_true", help="joint sign check against the sim (see tilt_test)")
    ap.add_argument("--dry", action="store_true", help="no motor writes at all (torque untouched)")
    ap.add_argument("--record", type=float, default=0, help="save the wrist camera (640x480) for this many seconds from the moment the arm is at home, "
                    "to logs/real_<time>.mp4; 0 = off")
    a = ap.parse_args()

    have_home = Path(a.home).exists()
    if not have_home and not a.grip_only:
        raise SystemExit(f"no home pose at {a.home}: clamp the paddle with `--grip-only`, then run `uv run python -m hardware.pose_guide` (format: hardware/home_example.json)")
    home = np.array([json.load(open(a.home))[f"{j}.pos"] for j in JOINTS] if have_home else TARGET)  # no saved home yet: the sim home
    controller = load_controller(a.controller)
    est = EstRunner(load_est(a.est))
    fd = FdVel()
    cap = None if a.grip_only and not a.dry else open_camera(a.cam)  # clamping the paddle needs no camera
    latest, stop = {}, [False]

    def grab():
        while not stop[0]:
            ok, f = cap.read()
            if ok: latest["f"] = f

    if cap is not None:
        threading.Thread(target=grab, daemon=True).start()
    t_cam = time.perf_counter()
    while cap is not None and "f" not in latest:  # before anything moves
        if time.perf_counter() - t_cam > 5:
            stop[0] = True; time.sleep(0.1); cap.release()
            raise SystemExit(f"no image from camera {a.cam} after 5 s (wrong --cam?). Nothing was moved.")
        time.sleep(0.01)
    robot = SOFollower(SOFollowerRobotConfig(port=a.port, id=a.id, disable_torque_on_disconnect=False, max_relative_target=10.0, position_p_coefficient=a.kp))
    bus = robot.bus
    bus.connect()
    def read():
        d = bus.sync_read("Present_Position")  # one bus read: all joints from the same moment
        return np.array([d[j] for j in JOINTS])

    def to_home():  # straight line in joint space, 4 s
        q = read()
        print(f"moving to home in 2 s (off by {np.round(q - home, 1)} deg) ...", flush=True); time.sleep(2)
        for s in np.linspace(0, 1, 120):
            robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, q + s * (home - q))}); time.sleep(1 / 30)
        time.sleep(1.0)

    q0 = read()
    far = np.any(np.abs(q0 - home) > 3)
    if a.dry and far:
        print(f"(dry) arm is off home by {np.round(q0 - home, 1)} deg; a live run would first move it home")
    if not a.dry:
        g0 = bus.sync_read("Present_Position")["gripper"]  # include the gripper: a stale goal could open it
        bus.sync_write("Goal_Position", {**{j: float(v) for j, v in zip(JOINTS, q0)}, "gripper": float(g0)})  # torque still off: goal = present, so no jump
        robot.configure()  # gains (P=--kp, D=32), then torque ON holding q0
        print(f"torque ON (P={a.kp}), holding the current pose")
        if a.grip_load > 0:
            insert = ask("Paddle already in? Enter to keep it. To insert it, type o + Enter: ") == "o"
            if insert:
                to_home()
                g = float(bus.sync_read("Present_Position")["gripper"])
                for v in np.linspace(g, max(g, a.grip_open), 30):
                    bus.write("Goal_Position", "gripper", float(v)); time.sleep(1 / 30)
                ask("Hold the paddle in the jaws by its handle, then press Enter to close: ")
            # Squeeze by load: close in 3-tick steps until the gripper's load reaches --grip-load, then hold. Commanding "closed" instead
            # stalls the servo above its overload limit (~20% for 2 s on my servos) and it drops off the bus. Torque_Limit is a second cap.
            bus.write("Torque_Limit", "gripper", 300)
            raw0 = bus.read("Present_Position", "gripper", normalize=False)
            load = lambda: bus.read("Present_Load", "gripper", normalize=False) & 0x3FF  # bit 10 = direction
            travel = raw0 - bus.calibration["gripper"].range_min if insert else 90  # inserting: from open all the way to contact
            sq, ld, stalls, prev, touching = 0, 0, 0, raw0, not insert
            while sq < travel:
                sq += 3 if touching else 8  # faster until the jaws touch the paddle
                bus.write("Goal_Position", "gripper", int(raw0 - sq), normalize=False)
                time.sleep(0.08)
                ld = load()
                if touching:
                    if ld >= a.grip_load:
                        break
                else:  # load alone is not contact (it spikes while the jaws just move): they must also have stopped
                    pos = bus.read("Present_Position", "gripper", normalize=False)
                    stalls = stalls + 1 if ld >= a.grip_load / 3 and prev - pos <= 2 else 0
                    prev = pos
                    if stalls >= 2:
                        touching, sq = True, raw0 - pos  # goal back to where the jaws stopped, then squeeze as usual
            time.sleep(0.3)
            ld = load()
            print(f"gripper: closed {sq} ticks (~{sq * 360 / 4096:.1f} deg), holding load {ld / 10:.0f}% (target {a.grip_load / 10:.0f}%, servo trips at 20%)")
            if ld >= 180:
                bus.disable_torque("gripper")
                raise SystemExit("gripper load too close to the overload trip: gripper torque OFF. Use a smaller --grip-load.")
        q0 = read()
        far = np.any(np.abs(q0 - home) > 3)
        if far and have_home:
            to_home()
        if a.grip_only:
            bus.disconnect(False)
            print("paddle clamped, torque stays ON. Release: uv run python -m hardware.torque_off gripper")
            return
    bias = np.zeros(5)  # extra command that holds the arm at home against gravity sag
    if not a.dry:
        goal = home.copy() if far else q0.copy()
        for _ in range(60):  # 2 s: adjust the command until the joints sit at home
            q = read()
            goal = home + np.clip(goal - home + 0.25 * (home - q), -12, 12)
            robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, goal)}); time.sleep(1 / 30)
        bias = goal - home
        q0 = read()
        print(f"at home: joints off by {np.round(q0 - home, 1)} deg, holding with sag compensation {np.round(bias, 1)} deg")

    if a.jog:
        try:
            for item in a.jog.split(","):
                jn, deg = item.split(":"); ji = JOINTS.index(jn); deg = float(deg); off = np.zeros(5)
                print(f"JOG {jn} {deg:+.0f} deg: watch the paddle TIP (up or down?)", flush=True); time.sleep(1.5)
                for s_ in np.linspace(0, 1, 30):
                    off[ji] = s_ * deg; robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, home + bias + off)}); time.sleep(1 / 30)
                time.sleep(3)
                for s_ in np.linspace(1, 0, 30):
                    off[ji] = s_ * deg; robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, home + bias + off)}); time.sleep(1 / 30)
                time.sleep(1.5)
        finally:
            stop[0] = True; cap.release(); bus.disconnect(False)
        return
    if a.tilt_test:
        return tilt_test(a, robot, read, latest, est, home, bias, stop, cap, bus)

    tgt = np.radians(q0 - home)  # target offset from home in rad, as in the sim
    hist = np.zeros((controller.act_hist, 5), np.float32)
    q_prev, t_prev, qd = q0, time.perf_counter(), np.zeros(5)
    seen_run, lost, armed, why = 0, 0, True, "time"
    episodes = []  # (seconds balanced, why it stopped)
    log = {k: [] for k in ("t", "q", "tgt", "det", "est", "act", "mode")}
    out = ROOT / "logs"; out.mkdir(exist_ok=True)
    rec = None
    if a.record > 0:
        h, w = latest["f"].shape[:2]
        rec_path = out / f"real_{time.strftime('%H%M%S')}.mp4"
        rec = cv2.VideoWriter(str(rec_path), cv2.VideoWriter_fourcc(*"mp4v"), HZ, (w, h))
        print(f"recording the first {a.record:.0f} s to {rec_path}")
    print("ARMED: put the ball on the blade centre (it starts as soon as it sees the ball 5 frames in a row). Runs until Ctrl+C.")
    t_start = t0 = last_print = time.perf_counter()
    try:
        while a.seconds <= 0 or time.perf_counter() - t_start < a.seconds:
            tick = time.perf_counter()
            q = read(); dt = tick - t_prev
            qd = 0.7 * qd + 0.3 * np.radians(q - q_prev) / max(dt, 1e-3); q_prev, t_prev = q, tick
            frame = latest["f"]
            img = cv2.cvtColor(cv2.resize(frame, (128, 96), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            det = detect(img); det[2] /= a.r_corr
            b = fd.update(est.update(det)[:2])
            if rec is not None:
                if tick - t_start < a.record:
                    rec.write(frame)  # one frame per tick = real time at 30 fps
                else:
                    rec.release(); rec = None; print(f"\nrecording saved: {rec_path}", flush=True)
            act = np.zeros(5, np.float32)
            if armed:
                on_blade = det[3] > 0 and abs(det[0] - 0.2) < 0.25  # ignores a ball held elsewhere in view
                seen_run = seen_run + 1 if on_blade else 0
                if seen_run >= 5:
                    armed, t0, lost = False, time.perf_counter(), 0
                    print(f"\nCONTROL ON  (ball at u {b[0]*8:+.1f} w {b[1]*8:+.1f} cm)", flush=True)
            else:
                lost = 0 if det[3] else lost + 1
                stop_why = None
                if lost >= a.lost: stop_why = f"ball not seen for {lost} frames"
                elif np.hypot(*b[:2]) * 8 > a.max_ball: stop_why = f"ball estimate {np.hypot(*b[:2])*8:.1f} cm from centre"
                else:
                    obs = np.concatenate([b, np.radians(q - home) / 0.2, qd * 0.1, hist.ravel()]).astype(np.float32)
                    act = controller_action(controller, obs)
                    new_tgt = tgt + act * ACTION_SCALE
                    if np.any(np.abs(np.degrees(new_tgt)) > a.max_dev):
                        stop_why = f"joint target would exceed {a.max_dev} deg: {np.round(np.degrees(new_tgt), 1)}"; act[:] = 0
                    else:
                        tgt = new_tgt
                        if not a.dry:
                            robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, home + bias + np.degrees(tgt))})
                        hist = np.roll(hist, 1, axis=0); hist[0] = act
                if stop_why:  # hold the last target, wait for the ball again
                    episodes.append((time.perf_counter() - t0, stop_why))
                    print(f"\nHOLD after {episodes[-1][0]:.1f} s: {stop_why}. Arm holds still. Put the ball back on the centre to restart (Ctrl+C to quit).", flush=True)
                    armed, seen_run, hist = True, 0, np.zeros((controller.act_hist, 5), np.float32)
                    est.reset(); fd.reset()
            for k, v in zip(log, (tick - t_start, q, np.degrees(tgt), det, b, act, armed)): log[k].append(v)
            if tick - last_print > (2.0 if armed else 0.5):
                last_print = tick
                print(f"{'ARMED ' if armed else ''}{tick - t0:6.1f}s | ball {'seen' if det[3] else 'LOST'} u {b[0]*8:+5.1f} w {b[1]*8:+5.1f} cm "
                      f"| target off home {np.round(np.degrees(tgt), 1)} | act {np.round(act, 2)}", flush=True)
            time.sleep(max(0, 1 / HZ - (time.perf_counter() - tick)))
    except KeyboardInterrupt:
        why = "Ctrl+C"
    except Exception as e:  # e.g. a motor dropped off the bus: still keep the log and the summary
        why = f"error: {type(e).__name__}: {e}"
    finally:
        stop[0] = True; time.sleep(0.1); cap.release()
        if rec is not None:  # stopped early: keep what was recorded
            rec.release(); print(f"recording saved (cut short): {rec_path}")
        try:
            bus.disconnect(False)  # torque stays on: the arm holds the last target
        except Exception as e:
            print(f"could not close the motor bus: {e}")
    if not armed:
        episodes.append((time.perf_counter() - t0, why))
    print(f"\nSTOPPED: {why}." + ("" if a.dry else " Torque is still ON (arm holds; release: uv run python -m hardware.torque_off)."))
    for i, (dur, w) in enumerate(episodes):
        print(f"  attempt {i + 1}: balanced {dur:6.1f} s, ended by: {w}")
    f = out / f"real_{time.strftime('%H%M%S')}{'_dry' if a.dry else ''}.npz"
    np.savez(f, **{k: np.array(v) for k, v in log.items()}, why=why)
    print(f"log: {f}")


if __name__ == "__main__":
    main()
