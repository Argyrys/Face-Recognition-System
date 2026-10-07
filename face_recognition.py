"""SFace face embedding and cosine matching (OpenCV 5.x)."""

import os

import cv2
import numpy as np

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
    "face_recognition_sface_2021dec.onnx",
)

# OpenCV SFace operating point: cosine similarity >= 0.363 => same person.
COSINE_THRESHOLD = 0.363


class FaceRecognizer:
    def __init__(self):
        self._rec = cv2.FaceRecognizerSF.create(MODEL_PATH, "")

    def embed(self, image: np.ndarray, face) -> np.ndarray:
        """L2-normalized (128,) float32 embedding for a face row."""
        aligned = self._rec.alignCrop(image, face)
        feat = np.asarray(self._rec.feature(aligned), dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm
        return feat.astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two L2-normalized embeddings."""
    return float(np.dot(a, b))


def best_match(embedding: np.ndarray, gallery):
    """Match against [(person_id, embedding), ...].

    Returns (person_id or None, score). None when below COSINE_THRESHOLD
    or the gallery is empty -- never guesses a closest identity.
    """
    if not gallery:
        return None, 0.0
    ids = [g[0] for g in gallery]
    matrix = np.stack([g[1] for g in gallery])
    scores = matrix @ embedding
    i = int(np.argmax(scores))
    score = float(scores[i])
    if score >= COSINE_THRESHOLD:
        return ids[i], score
    return None, score
