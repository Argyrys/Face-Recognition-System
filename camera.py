"""Frame sources: webcam index, video file, directory of images, or a
network video stream URL (e.g. a phone camera served by IP Webcam /
DroidCam). The app targets machines that may not have a webcam, so every
path of the system also works from recorded video, still images, or a
phone camera over the network.
"""

import base64
import glob
import os
import re
import threading
import time
import urllib.request
from urllib.parse import unquote, urlsplit

import cv2
import numpy as np

_IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
_IMAGE_INTERVAL = 0.1  # pace still-image sources at ~10 fps


def _is_stream(spec: str) -> bool:
    return spec.startswith(("http://", "https://", "rtsp://", "rtmp://"))


class MjpegStream:
    """Direct reader for `multipart/x-mixed-replace` MJPEG streams.

    OpenCV's FFMPEG backend buffers HTTP streams heavily -- a phone camera
    that serves ~15 fps comes out of cv2.VideoCapture at well under 1 fps,
    which makes live recognition unusable. Parsing the multipart stream
    ourselves keeps latency at a frame or two.
    """

    def __init__(self, url: str, timeout: float = 10.0, retries: int = 5):
        self.url = url
        self.timeout = timeout
        self.retries = retries
        self._resp = None
        self._buf = b""
        self._boundary = b"--frame"

    def _open(self):
        parts = urlsplit(self.url)
        req = urllib.request.Request(self.url)
        if parts.username:
            token = base64.b64encode(
                f"{unquote(parts.username)}:{unquote(parts.password or '')}".encode()
            ).decode()
            req.add_header("Authorization", "Basic " + token)
        self._resp = urllib.request.urlopen(req, timeout=self.timeout)
        ctype = self._resp.headers.get("Content-Type", "")
        if "boundary=" in ctype:
            self._boundary = ("--" + ctype.split("boundary=", 1)[1].strip()).encode()
        self._buf = b""

    def _fill(self):
        chunk = self._resp.read(65536)
        if not chunk:
            raise RuntimeError("stream closed by server")
        self._buf += chunk

    def _next_frame(self):
        if self._resp is None:
            self._open()
        while True:
            idx = self._buf.find(self._boundary)
            if idx == -1:
                if len(self._buf) > 16 * 1024 * 1024:
                    raise RuntimeError("no frame boundary in stream")
                self._fill()
                continue

            head_end = self._buf.find(b"\r\n\r\n", idx)
            if head_end == -1:
                self._fill()
                continue

            headers = self._buf[idx:head_end]
            m = re.search(rb"content-length:\s*(\d+)", headers, re.I)
            if not m:
                self._buf = self._buf[head_end + 4:]
                continue
            length = int(m.group(1))
            start = head_end + 4
            while len(self._buf) < start + length:
                self._fill()

            data = self._buf[start:start + length]
            self._buf = self._buf[start + length:]
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                return img

    def read(self):
        """Next frame, reconnecting on dropouts. None if the stream is dead."""
        for attempt in range(self.retries):
            try:
                return self._next_frame()
            except Exception:
                self.close()
                if attempt + 1 < self.retries:
                    time.sleep(0.5)
        return None

    def close(self):
        if self._resp is not None:
            try:
                self._resp.close()
            except Exception:
                pass
            self._resp = None


class FrameSource:
    def __init__(self, spec="0"):
        self.spec = str(spec)
        self._lock = threading.Lock()
        self._images = None
        self._cap = None
        self._stream = None
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
        elif _is_stream(self.spec):
            # Phone / network camera (IP Webcam, DroidCam, RTSP, ...).
            self._stream = MjpegStream(self.spec)
            probe = self._stream.read()
            if probe is None:
                raise RuntimeError(
                    f"cannot open video stream: {self.spec} "
                    "(is the camera app running, reachable, and allowed?)"
                )
            self.label = f"stream:{self.spec}"
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
            if self._stream is not None:
                return self._stream.read()
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
            if self._stream is not None:
                self._stream.close()
            if self._cap is not None:
                self._cap.release()
