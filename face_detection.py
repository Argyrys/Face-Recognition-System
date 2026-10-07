"""YuNet face detection wrapper (OpenCV 5.x)."""

import os

import cv2
import numpy as np

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
    "face_detection_yunet_2023mar.onnx",
)


class FaceDetector:
    """Detect faces; returns YuNet rows: x, y, w, h + 5 landmarks (10) + score."""

    def __init__(self, score_threshold=0.6, nms_threshold=0.3, top_k=50000):
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self._detector = None
        self._size = None

    def detect(self, image: np.ndarray):
        """Return (N, 15) ndarray of faces or None."""
        h, w = image.shape[:2]
        if self._detector is None or self._size != (w, h):
            self._detector = cv2.FaceDetectorYN.create(
                MODEL_PATH,
                "",
                (w, h),
                self.score_threshold,
                self.nms_threshold,
                self.top_k,
            )
            self._size = (w, h)
        _, faces = self._detector.detect(image)
        return faces

    @staticmethod
    def best(faces):
        """Highest-scoring face row or None."""
        if faces is None or len(faces) == 0:
            return None
        return faces[int(np.argmax(faces[:, -1]))]

    @staticmethod
    def box(face) -> tuple:
        """(x, y, w, h) ints for a face row."""
        x, y, w, h = face[:4]
        return int(x), int(y), int(w), int(h)
