"""Passive presentation-attack detection (anti-spoof) from a single RGB feed.

A printed photo or a still screen shows almost no landmark motion and never
blinks; a live face does both. Per track we keep two cheap signals:

1. landmark motion — mean frame-to-frame displacement of the 5 YuNet
   landmarks, normalized by face size (scale- and distance-invariant);
2. eye-region texture energy — Laplacian variance of the two eye crops
   normalized by the whole-face variance (lighting-invariant). A blink
   makes both eye energies drop sharply below the personal baseline and
   recover.

decide() combines them into live / unknown / spoof:

   live   -> motion above floor, plus (blink seen or motion clearly strong)
   spoof  -> motion below floor AND no blink in the window
   unknown-> everything else (insufficient evidence either way)

Honest limits (documented in docs/SECURITY.md): a high-quality video replay
of a blinking person can pass passive checks — that is what the optional
challenge-response mode (VC_CHALLENGE) is for. The module never claims to
be a certified PAD; it removes the trivial attacks (photo, static screen).
"""

import cv2
import numpy as np

# How much landmark jitter (fraction of sqrt(face area)) counts as "moving".
DEFAULT_MIN_MOTION = 0.004
# Eye energy below baseline * BLINK_DROP is an eye-closed candidate.
BLINK_DROP = 0.72
BLINK_RECOVER = 0.85      # ...and back above baseline * this = a real blink
BLINK_RECOVER_FRAMES = 6  # recovery must happen within this many frames
HISTORY_MAX = 45          # ~3 s of observations at 15 fps


def eye_energy(gray, face) -> tuple:
    """(left, right) texture-energy ratios for the two eye regions.

    Ratio of Laplacian variance in a small crop around each eye to the
    variance of the whole face box; dividing by the face keeps the metric
    stable when lighting changes. Closed eyes => smooth eyelid => low ratio.
    """
    x, y, w, h = (int(v) for v in face[:4])
    h_img, w_img = gray.shape[:2]
    x, y, w, h = max(0, x), max(0, y), max(1, w), max(1, h)
    face_roi = gray[y:y + h, x:x + w]
    if face_roi.size == 0:
        return 0.0, 0.0
    face_var = float(cv2.Laplacian(face_roi, cv2.CV_64F).var()) + 1e-6

    radius = max(2, int(0.16 * w))
    out = []
    for ex, ey in (face[4:6], face[6:8]):       # YuNet: right eye, left eye
        cx, cy = int(ex), int(ey)
        x1, y1 = max(0, cx - radius), max(0, cy - radius)
        x2, y2 = min(w_img, cx + radius), min(h_img, cy + radius)
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            out.append(0.0)
            continue
        energy = float(cv2.Laplacian(roi, cv2.CV_64F).var())
        out.append(energy / face_var)
    return out[0], out[1]


def normalized_landmarks(face) -> np.ndarray:
    """The 5 landmarks (10 values) relative to the face box, scaled by
    sqrt(area) so the motion metric does not depend on distance to camera."""
    x, y, w, h = face[:4]
    scale = max(np.sqrt(max(w * h, 1.0)), 1.0)
    pts = np.asarray(face[4:14], dtype=np.float64).reshape(5, 2)
    pts = pts - np.array([x, y], dtype=np.float64)
    return (pts / scale).reshape(-1)


def motion_score(history) -> float:
    """Mean per-frame landmark displacement over the stored window."""
    if len(history) < 2:
        return 0.0
    arr = np.asarray(history[-HISTORY_MAX:], dtype=np.float64)
    deltas = np.abs(np.diff(arr, axis=0))
    return float(deltas.mean())


def blink_count(energy_history) -> int:
    """Number of closed->open eye transitions in the (left, right) history."""
    if len(energy_history) < 3:
        return 0
    hist = energy_history[-HISTORY_MAX:]
    combined = [(l + r) / 2.0 for l, r in hist]
    baseline = float(np.median(combined))
    if baseline <= 0:
        return 0

    blinks, i = 0, 0
    while i < len(combined):
        if combined[i] < baseline * BLINK_DROP:
            # look for recovery within the allowed window
            for j in range(i + 1, min(i + 1 + BLINK_RECOVER_FRAMES, len(combined))):
                if combined[j] >= baseline * BLINK_RECOVER:
                    blinks += 1
                    i = j + 1
                    break
            else:
                i += 1
        else:
            i += 1
    return blinks


def decide(motion: float, blinks: int, min_motion: float = DEFAULT_MIN_MOTION) -> str:
    """Combine the two signals into 'live' | 'unknown' | 'spoof'."""
    if motion >= min_motion and (blinks >= 1 or motion >= 2 * min_motion):
        return "live"
    if motion < min_motion and blinks == 0:
        return "spoof"
    return "unknown"


def entry_label(mode: str, state: str) -> str:
    """How a liveness state is recorded on an entry, per enforcement mode.

    enforced: live -> 'live', spoof -> 'spoof', unknown -> 'unverified'
    monitor : always recorded but never blocks a MATCHED entry
    off     : no label at all
    """
    if mode == "off":
        return None
    return {"live": "live", "spoof": "spoof", "unknown": "unverified"}[state]
