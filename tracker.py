"""Multi-face tracking with per-visitor evidence accumulation.

The old pipeline made a decision from a single frame — one lucky or blurred
frame could log a wrong identity. A track follows each face across frames
(IoU matching) and accumulates:

- identity votes: best_match results per frame; a visitor is only "matched"
  after CONFIRM consecutive frames agree (N-frame confirmation),
- liveness observations: landmark motion + eye-energy history for the
  passive anti-spoof check (see liveness.py).

One frame of noise can no longer create a log entry; a stranger pacing in
and out of frame keeps the same track id until they leave the view.
"""

import numpy as np

from liveness import (DEFAULT_MIN_MOTION, HISTORY_MAX, blink_count,
                      decide, eye_energy, motion_score, normalized_landmarks)


def iou(a, b) -> float:
    """Intersection-over-union of two (x, y, w, h) boxes."""
    ax, ay, aw, ah = a[:4]
    bx, by, bw, bh = b[:4]
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class Track:
    def __init__(self, track_id: int, face, frame_idx: int, gray=None):
        self.id = track_id
        self.face = None
        self.box = None
        self.first_frame = frame_idx
        self.last_frame = frame_idx
        self.hits = 0
        self.misses = 0
        # identity evidence
        self.top_person = None
        self.top_streak = 0
        self.unknown_streak = 0
        self.last_person = None
        self.last_score = 0.0
        self.logged = False          # a MATCHED/UNKNOWN entry was emitted
        self.hold_frames = 0         # frames spent waiting for liveness
        # liveness evidence
        self.landmarks = []          # normalized landmark vectors
        self.energies = []           # (left, right) eye-energy ratios
        self.observe(face, frame_idx, gray)

    # ------------------------------------------------------------ frames

    def observe(self, face, frame_idx: int, gray=None):
        """Feed one detection frame into this track."""
        self.face = face
        self.box = tuple(int(v) for v in face[:4])
        self.last_frame = frame_idx
        self.hits += 1
        self.misses = 0
        if gray is not None:
            self.landmarks.append(normalized_landmarks(face))
            self.landmarks = self.landmarks[-HISTORY_MAX:]
            self.energies.append(eye_energy(gray, face))
            self.energies = self.energies[-HISTORY_MAX:]

    def miss(self, frame_idx: int) -> int:
        self.misses += 1
        self.last_frame = frame_idx
        return self.misses

    # --------------------------------------------------- identity votes

    def record(self, person_id, score: float):
        """Record this frame's match result; keep the running streak."""
        self.last_person, self.last_score = person_id, score
        if person_id == self.top_person:
            self.top_streak += 1
        else:
            self.top_person = person_id
            self.top_streak = 1
        if person_id is None:
            self.unknown_streak += 1
        else:
            self.unknown_streak = 0

    def ready_to_log(self, confirm_frames: int) -> bool:
        """True when the same identity won enough consecutive frames."""
        return (not self.logged
                and self.top_person is not None
                and self.top_streak >= confirm_frames)

    def ready_unknown(self, unknown_frames: int) -> bool:
        """True when no identity passed threshold for enough consecutive frames."""
        return (not self.logged
                and self.top_person is None
                and self.unknown_streak >= unknown_frames)

    # -------------------------------------------------------- liveness

    @property
    def motion(self) -> float:
        return motion_score(self.landmarks)

    @property
    def blinks(self) -> int:
        return blink_count(self.energies)

    def liveness_state(self, min_motion: float = DEFAULT_MIN_MOTION) -> str:
        """'live' | 'unknown' | 'spoof' from this track's observations.

        'spoof' needs at least ~10 frames (about a second) of evidence so a
        genuinely still visitor is never branded an attack on 3 frames."""
        state = "unknown"
        if len(self.landmarks) >= 4:
            state = decide(self.motion, self.blinks, min_motion)
        if state == "spoof" and len(self.landmarks) < 10:
            return "unknown"
        return state


class MultiTracker:
    """Greedy IoU tracker: detections -> stable track ids across frames."""

    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 6):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.tracks = {}
        self._next_id = 1

    def update(self, faces, frame_idx: int, gray=None):
        """Attach this frame's detections to existing tracks (highest IoU
        first), spawn tracks for new faces, retire long-missed ones.
        `gray` (optional) feeds the liveness observations.
        Returns the list of currently active tracks."""
        detections = [] if faces is None else [faces[i] for i in range(len(faces))]

        # greedy matching: score every (track, detection) IoU pair, best first
        pairs = []
        for tid, tr in self.tracks.items():
            for i, det in enumerate(detections):
                score = iou(tr.box, det)
                if score >= self.iou_threshold:
                    pairs.append((score, tid, i))
        pairs.sort(key=lambda p: -p[0])

        used_t, used_d = set(), set()
        for score, tid, i in pairs:
            if tid in used_t or i in used_d:
                continue
            used_t.add(tid)
            used_d.add(i)
            self.tracks[tid].observe(detections[i], frame_idx, gray)

        for i, det in enumerate(detections):
            if i in used_d:
                continue
            tid = self._next_id
            self._next_id += 1
            self.tracks[tid] = Track(tid, det, frame_idx, gray)

        for tid in list(self.tracks):
            tr = self.tracks[tid]
            if tid not in used_t and tr.last_frame != frame_idx:
                if tr.miss(frame_idx) > self.max_missed:
                    del self.tracks[tid]

        return list(self.tracks.values())

    def reset(self):
        self.tracks.clear()
        self._next_id = 1
