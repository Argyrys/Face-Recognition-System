"""Frame sources: webcam index, video file, or directory of images.

The app targets machines that may not have a webcam, so every path of the
system also works from recorded video or still images.
"""

import glob
import os
import threading
import time

import cv2

_IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
_IMAGE_INTERVAL = 0.1  # pace still-image sources at ~10 fps


class FrameSource:
    def __init__(self, spec="0"):
        self.spec = str(spec)
        self._lock = threading.Lock()
        self._images = None
        self._cap = None
        self._idx = 0
        self._loop = True

        if os.path.isdir(spec):
            files = []
            for ext in _IMAGE_EXTS:
                files.extend(glob.glob(os.path.join(spec, ext)))
            if not files:
                raise ValueError(f"no images found in directory: {spec}")
            self._images = sorted(files)
            self.label = f"images:{os.path.basename(os.path.normpath(spec))} ({len(self._images)})"
        elif os.path.isfile(spec):
            self._cap = cv2.VideoCapture(spec)
            if not self._cap.isOpened():
                raise RuntimeError(f"cannot open video file: {spec}")
            self._loop = False
            self.label = f"video:{os.path.basename(spec)}"
        else:
            index = int(spec)
            self._cap = cv2.VideoCapture(index)
            if not self._cap.isOpened():
                raise RuntimeError(
                    f"cannot open camera index {index} (no webcam on this machine?)"
                )
            self.label = f"webcam:{index}"

    def read(self):
        """Next BGR frame. None = source exhausted (video file only;
        image directories cycle forever)."""
        with self._lock:
            if self._images is not None:
                frame = cv2.imread(self._images[self._idx])
                self._idx = (self._idx + 1) % len(self._images)
                if frame is None:
                    raise RuntimeError(f"failed to read image: {self._images[self._idx]}")
                time.sleep(_IMAGE_INTERVAL)
                return frame
            ok, frame = self._cap.read()
            if not ok or frame is None:
                if self._loop:
                    # camera dropped a frame; try to reopen once
                    self._cap.release()
                    self._cap = cv2.VideoCapture(self.spec)
                    ok, frame = self._cap.read()
                    if not ok or frame is None:
                        return None
                    return frame
                return None
            return frame

    def release(self):
        with self._lock:
            if self._cap is not None:
                self._cap.release()
