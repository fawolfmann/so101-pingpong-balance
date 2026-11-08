"""Orange-ball colour detector, the same code in sim and on the real camera.

detect(rgb uint8 (96, 128, 3)) -> float32 [x, y, r, visible]
    x, y     blob centre, -1..1 across the image
    r        sqrt(area / pi) / 10 px (~0.8 with the ball at the blade centre)
    visible  1 if a ball-like blob was found, else 0 (and x = y = r = 0)
"""
from __future__ import annotations

import cv2
import numpy as np

LO, HI = np.array([3, 110, 110], np.uint8), np.array([22, 255, 255], np.uint8)  # HSV, OpenCV units (H 0-180)
KERNEL = np.ones((3, 3), np.uint8)
MIN_AREA, MAX_AREA = 40, 1200  # px; the ball is ~180 px at the blade centre. 40 rejects small orange specks in the background


def mask(rgb: np.ndarray) -> np.ndarray:
    """Orange pixels, with a 3x3 opening to remove speckle."""
    return cv2.morphologyEx(cv2.inRange(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV), LO, HI), cv2.MORPH_OPEN, KERNEL)


def detect(rgb: np.ndarray) -> np.ndarray:
    n, _, st, cen = cv2.connectedComponentsWithStats(mask(rgb), connectivity=8)
    if n <= 1:
        return np.zeros(4, np.float32)
    a, bw, bh = st[1:, cv2.CC_STAT_AREA], st[1:, cv2.CC_STAT_WIDTH], st[1:, cv2.CC_STAT_HEIGHT]
    # ball-like: size in range, fills its bounding box (a disc fills pi/4), not a long stripe (e.g. the red band on the gripper)
    ok = (a >= MIN_AREA) & (a <= MAX_AREA) & (a / (bw * bh) > 0.45) & (np.maximum(bw, bh) <= 2.5 * np.minimum(bw, bh))
    if not ok.any():
        return np.zeros(4, np.float32)
    k = 1 + int(np.argmax(np.where(ok, a, -1)))  # largest ball-like blob
    area = st[k, cv2.CC_STAT_AREA]
    h, w = rgb.shape[:2]
    cx, cy = cen[k]
    return np.array([(cx - w / 2) / (w / 2), (cy - h / 2) / (h / 2), np.sqrt(area / np.pi) / 10, 1.0], np.float32)
