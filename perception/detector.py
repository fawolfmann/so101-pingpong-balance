"""Orange-ball colour detector, the same code in sim and on the real camera.

detect(rgb uint8 (H, W, 3), min_area, max_area) -> float32 [x, y, r, visible]   (wrist camera: 96x128, default limits)
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


def mask(rgb: np.ndarray, s_min: int = int(LO[1])) -> np.ndarray:
    """Orange pixels (saturation >= s_min), with a 3x3 opening to remove speckle."""
    lo = LO.copy(); lo[1] = s_min
    return cv2.morphologyEx(cv2.inRange(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV), lo, HI), cv2.MORPH_OPEN, KERNEL)


def detect(rgb: np.ndarray, min_area: int = MIN_AREA, max_area: int = MAX_AREA, s_min: int = int(LO[1]),
           select: str = "area") -> np.ndarray:
    """Blob size limits are in pixels of this image; the defaults fit the wrist camera at 96x128 (see estimator.CAMERAS).
    select: which ball-like blob wins, "area" (largest) or "round" (roundest, weighted by size). The ego camera uses a
    higher s_min and "round": it sees the paddle handle, which can be wood-coloured and is picked otherwise."""
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask(rgb, s_min), connectivity=8)
    if n <= 1:
        return np.zeros(4, np.float32)
    a, bw, bh = st[1:, cv2.CC_STAT_AREA], st[1:, cv2.CC_STAT_WIDTH], st[1:, cv2.CC_STAT_HEIGHT]
    # ball-like: size in range, fills its bounding box (a disc fills pi/4), not a long stripe (e.g. the red band on the gripper)
    ok = (a >= min_area) & (a <= max_area) & (a / (bw * bh) > 0.45) & (np.maximum(bw, bh) <= 2.5 * np.minimum(bw, bh))
    if not ok.any():
        return np.zeros(4, np.float32)
    if select == "round":
        score = np.full(len(a), -1.0)
        for i in np.flatnonzero(ok):
            cnt, _ = cv2.findContours((lab == i + 1).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            score[i] = 4 * np.pi * a[i] / max(cv2.arcLength(cnt[0], True), 1.0) ** 2 * np.sqrt(a[i])  # circularity x size
        k = 1 + int(np.argmax(score))
    else:
        k = 1 + int(np.argmax(np.where(ok, a, -1)))  # largest ball-like blob
    area = st[k, cv2.CC_STAT_AREA]
    h, w = rgb.shape[:2]
    cx, cy = cen[k]
    return np.array([(cx - w / 2) / (w / 2), (cy - h / 2) / (h / 2), np.sqrt(area / np.pi) / 10, 1.0], np.float32)
