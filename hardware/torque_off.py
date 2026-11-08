"""Switch torque OFF on all motors (arm goes limp and falls under gravity: HOLD IT). Also releases the gripper squeeze.

    uv run python -m hardware.torque_off [--port PORT]            # all motors
    uv run python -m hardware.torque_off [--port PORT] gripper    # only the gripper (arm keeps holding)
"""
import argparse
import time

from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower

ap = argparse.ArgumentParser()
ap.add_argument("motors", nargs="*"); ap.add_argument("--port", default="/dev/ttyACM0")
ap.add_argument("--id", default="my_follower_arm", help="LeRobot calibration id of your arm")
a = ap.parse_args()
motors = a.motors or None
robot = SOFollower(SOFollowerRobotConfig(port=a.port, id=a.id, disable_torque_on_disconnect=False))
robot.bus.connect()
print(f"torque OFF in 2 s on {motors or 'ALL motors'}" + ("" if motors else ": hold the arm, it will drop"), flush=True)
time.sleep(2)
robot.bus.disable_torque(motors)
robot.bus.disconnect(False)
print("done")
