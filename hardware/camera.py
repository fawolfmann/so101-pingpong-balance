"""Open the wrist camera at 640x480, 30 fps. `cam` is a device path (Linux: /dev/video4) or an index ("0", "1", ...)."""
import sys

import cv2


def open_camera(cam: str) -> cv2.VideoCapture:
    src = int(cam) if str(cam).isdigit() else cam
    cap = cv2.VideoCapture(src, cv2.CAP_V4L2) if sys.platform.startswith("linux") else cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480); cap.set(cv2.CAP_PROP_FPS, 30); cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {cam} (wrong --cam, or another program is using it?)")
    return cap
