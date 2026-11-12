"""Orange-ball colour detector, the same code in sim and on the real camera.

detect(rgb uint8 (H, W, 3), min_area, max_area) -> float32 [x, y, r, visible]   (wrist camera: 96x128, default limits)
    x, y     blob centre, -1..1 across the image
    r        sqrt(area / pi) / 10 px (~0.8 with the ball at the blade centre)
    visible  1 if a ball-like blob was found, else 0 (and x = y = r = 0)
Single frame, no memory; what the estimators were trained on (unchanged).
BallDetector(track=False | True): the same, or with the tracking below, behind one call: det(rgb, shift).

For a moving camera (the arm + camera policy), frame-to-frame tracking on top of the same colour mask:
candidates(rgb, ...)  orange blobs that look like a whole ball: roughly round (aspect <= 1.5, fills >= 55% of its
                      enclosing circle), big enough, not touching the image border (a ball cut by the border is a
                      sliver whose centre and size are wrong; 84% of cut balls touch the border, 3% of whole ones)
BallTracker.update(cands, shift)
    predicts where the ball is now from the last frames (constant velocity in the image, plus the shift the camera's
    own pan/tilt caused, known from its servos), and keeps the candidate whose box overlaps the predicted box most (IoU).
    No overlapping candidate: reports the prediction with visible = 0 for up to max_coast frames, then drops the track;
    with no track, the largest candidate starts one. Simulated detections (sim/cam_model.py) go through the same
    BallTracker, so training and rendered evaluation share the association logic.
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


# --- tracking for a moving camera ---
MAX_ASPECT, MIN_FILL = 1.5, 0.55
# ego camera at 192x256: the ball is 5-7.5 px in radius from the post; min_area 60 (4.4 px) rejects a small orange part
# at the arm base (~3 px) that the camera sees when it points down
TRACK_DET = dict(min_area=60, max_area=2000, s_min=170)


def candidates(rgb: np.ndarray, min_area: int = 20, max_area: int = 2000, s_min: int = 170, **_) -> list[np.ndarray]:
    """Whole-ball blobs as [x, y, r, 1] (detect() units), largest first. Extra keys (select=...) are ignored."""
    h, w = rgb.shape[:2]
    n, _, st, cen = cv2.connectedComponentsWithStats(mask(rgb, s_min), connectivity=8)
    out = []
    for i in range(1, n):
        x0, y0, bw, bh, a = st[i]
        if not min_area <= a <= max_area:
            continue
        if x0 == 0 or y0 == 0 or x0 + bw == w or y0 + bh == h:
            continue
        if max(bw, bh) / min(bw, bh) > MAX_ASPECT or a / (np.pi * (max(bw, bh) / 2) ** 2) < MIN_FILL:
            continue
        cx, cy = cen[i]
        out.append((a, np.array([(cx - w / 2) / (w / 2), (cy - h / 2) / (h / 2), np.sqrt(a / np.pi) / 10, 1.0], np.float32)))
    return [c for _, c in sorted(out, key=lambda t: -t[0])]


def box(det: np.ndarray, hw: tuple[int, int], grow: float = 1.0) -> np.ndarray:
    """Square box around a detection, in image units (x0, y0, x1, y1)."""
    h, w = hw
    rx, ry = det[2] * 10 / (w / 2) * grow, det[2] * 10 / (h / 2) * grow
    return np.array([det[0] - rx, det[1] - ry, det[0] + rx, det[1] + ry])


def iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class BallTracker:
    def __init__(self, hw: tuple[int, int] = (192, 256), iou_min: float = 0.05, grow: float = 2.0, max_coast: int = 5):
        self.hw, self.iou_min, self.grow, self.max_coast = hw, iou_min, grow, max_coast
        self.reset()

    def reset(self) -> None:
        self.pos: np.ndarray | None = None  # last position (x, y), current image coordinates
        self.vel = np.zeros(2)
        self.r = 0.0
        self.lost = 0

    def predict(self) -> np.ndarray | None:
        return None if self.pos is None else self.pos + self.vel

    def update(self, cands: list[np.ndarray], shift=(0.0, 0.0)) -> np.ndarray:
        """cands: candidates() of this frame (or a simulated detection); shift: image motion of a static point since the
        last frame, caused by the camera turning."""
        shift = np.asarray(shift, np.float64)
        if self.pos is None:
            if not cands:
                return np.zeros(4, np.float32)
            c = cands[0]
            self.pos, self.vel, self.r, self.lost = c[:2].astype(np.float64), np.zeros(2), float(c[2]), 0
            return c.copy()
        pred = self.pos + self.vel + shift
        pbox = box(np.array([*pred, self.r]), self.hw, self.grow)
        scores = [iou(pbox, box(c, self.hw, self.grow)) for c in cands]
        k = int(np.argmax(scores)) if scores else -1
        if k >= 0 and scores[k] >= self.iou_min:
            c = cands[k]
            new = c[:2].astype(np.float64)
            self.vel = 0.5 * self.vel + 0.5 * (new - (self.pos + shift))  # smoothed image velocity, camera motion removed
            self.pos, self.r, self.lost = new, float(c[2]), 0
            return c.copy()
        self.lost += 1
        if self.lost > self.max_coast:
            self.reset()
            return self.update(cands, shift) if cands else np.zeros(4, np.float32)
        self.pos = pred
        return np.array([*np.clip(pred, -1.5, 1.5), self.r, 0.0], np.float32)  # coasting: predicted, not seen


def camera_shift(d_aim: np.ndarray, fovy_deg: float, hw: tuple[int, int], pitch: float = 0.0) -> np.ndarray:
    """Image motion of a static point near the image centre when the camera turns by d_aim = (d yaw, d pitch) rad about
    its mount (yaw about the vertical): small-angle, nominal FOV; a camera tilted by `pitch` sees yaw scaled by cos(pitch)."""
    h, w = hw
    tv = np.tan(np.deg2rad(fovy_deg) / 2)
    return np.array([d_aim[0] * np.cos(pitch) / (tv * w / h), d_aim[1] / tv])


class BallDetector:
    """One entry point for both modes.

        det = BallDetector()                              # = detect(): single frame, no memory
        det = BallDetector(track=True, hw=(192, 256))     # whole-ball candidates + BallTracker
        x, y, r, visible = det(rgb, shift)                # shift: camera_shift() of the camera's own turn (track only)
        det.reset()                                       # new episode

    track=False takes detect()'s keywords (min_area, max_area, s_min, select); track=True takes candidates()' keywords
    (default TRACK_DET) and BallTracker's (iou_min, grow, max_coast). update(cands, shift) feeds the tracker candidates
    from elsewhere (simulated detections in training)."""

    def __init__(self, track: bool = False, hw: tuple[int, int] = (192, 256), **kw):
        self.track = track
        tk = {k: kw.pop(k) for k in ("iou_min", "grow", "max_coast") if k in kw}
        self.kw = kw if (kw or not track) else dict(TRACK_DET)
        self.tracker = BallTracker(hw, **tk) if track else None

    def reset(self) -> None:
        if self.tracker is not None:
            self.tracker.reset()

    def update(self, cands: list[np.ndarray], shift=(0.0, 0.0)) -> np.ndarray:
        return self.tracker.update(cands, shift)

    def __call__(self, rgb: np.ndarray, shift=(0.0, 0.0)) -> np.ndarray:
        if not self.track:
            return detect(rgb, **self.kw)
        return self.tracker.update(candidates(rgb, **self.kw), shift)
