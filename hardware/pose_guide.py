"""Save your home pose: the arm goes limp (the gripper keeps its grip), you move it by hand until the paddle is level, then Ctrl+C.

    uv run python -m hardware.run_real --grip-only   # first, to clamp the paddle
    uv run python -m hardware.pose_guide

Prints how far each joint is from the sim home, in degrees. That is only a guide and the numbers need not reach 0: get roughly close,
then level the paddle with wrist_flex (mine ended 14 deg off). Ctrl+C saves the pose you are holding to --save (default hardware/home.json), and the arm stays there.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower

ROOT = Path(__file__).parent.parent
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
PARTS = ["base turn", "upper arm", "forearm", "wrist tilt", "wrist twist"]
TARGET = [0.0, -37.2, 8.6, 28.6, 92.8]  # sim home in real readings (wrist_roll: real = sim + 90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--id", default="my_follower_arm", help="LeRobot calibration id of your arm")
    ap.add_argument("--save", default=str(ROOT / "hardware/home.json"))
    a = ap.parse_args()
    robot = SOFollower(SOFollowerRobotConfig(port=a.port, id=a.id, disable_torque_on_disconnect=False))
    robot.bus.connect()
    tgt = np.array(TARGET)
    print("        " + "  ".join(f"{p:>13s}" for p in PARTS))
    print("        " + "  ".join(f"{j:>13s}" for j in JOINTS))
    print("target  " + "  ".join(f"{v:13.1f}" for v in tgt))
    print("\nThe numbers are only a guide and do not need to reach 0. Get roughly close, then tilt the paddle with wrist_flex until it is level.")
    print("Ctrl+C saves the pose you are holding, and the arm stays there.\n")
    d = {}
    try:
        input("Hold the arm, then press Enter: it goes limp and the gripper keeps its grip. ")
        robot.bus.disable_torque(JOINTS)
        while True:
            d = robot.bus.sync_read("Present_Position")
            q = np.array([d[j] for j in JOINTS])
            off = q - tgt
            print("\roff by  " + "  ".join(f"{o:+13.1f}" for o in off), end="", flush=True)
            time.sleep(0.15)
    except KeyboardInterrupt:
        pass
    try:
        if d:  # d is the last complete reading: a Ctrl+C in the middle of a motor read cannot lose the pose or drop the arm
            Path(a.save).write_text(json.dumps({f"{k}.pos": float(v) for k, v in d.items()}, indent=1))
            print(f"\nsaved {a.save}: " + ", ".join(f"{k} {v:.1f}" for k, v in d.items()))
            robot.bus.sync_write("Goal_Position", {j: float(d[j]) for j in JOINTS})  # goal = pose, so torque on does not jump
            robot.bus.enable_torque(JOINTS, num_retry=5)
    finally:
        robot.bus.disconnect(False)


if __name__ == "__main__":
    main()
