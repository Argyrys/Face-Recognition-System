"""Enroll consented users from photos (single image or averaged multi-image)."""

import argparse
import os

import cv2
import numpy as np

from config import Config
from database import Database
from face_detection import FaceDetector
from face_recognition import FaceRecognizer

SNAPSHOT_DIR = Config.SNAPSHOT_DIR


def _load_image(source):
    """Load from path or raw bytes (uploads)."""
    if isinstance(source, (bytes, bytearray)):
        arr = np.frombuffer(source, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(source)
    if img is None:
        raise ValueError(f"cannot read image: {source if isinstance(source, str) else 'upload'}")
    return img


def enroll(db: Database, detector: FaceDetector, recognizer: FaceRecognizer,
           images, name: str, role: str, department: str, external_id: str,
           consent: bool) -> dict:
    """Enroll a person from one or more images containing their face.

    Multiple images: embeddings are averaged then re-normalized.
    Returns the created person id and snapshot filename.
    """
    if role not in ("Faculty", "Student"):
        raise ValueError("role must be 'Faculty' or 'Student'")
    if not consent:
        raise ValueError("consent is required for biometric enrollment")

    embeddings = []
    best_face = None
    best_img = None
    best_score = -1.0
    for src in images:
        img = _load_image(src)
        faces = detector.detect(img)
        face = detector.best(faces)
        if face is None:
            raise ValueError("no face detected in one of the images")
        embeddings.append(recognizer.embed(img, face))
        score = float(faces[:, -1].max())
        if score > best_score:
            best_score, best_face, best_img = score, face, img

    mean = np.mean(np.stack(embeddings), axis=0)
    norm = np.linalg.norm(mean)
    if norm > 0:
        mean = mean / norm
    mean = mean.astype(np.float32)

    person_id = db.add_person(name, role, department, external_id, mean, consent)

    # store an aligned crop of the clearest photo for the record
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    aligned = recognizer._rec.alignCrop(best_img, best_face)
    snapshot = f"person_{person_id}.jpg"
    cv2.imwrite(os.path.join(SNAPSHOT_DIR, snapshot), aligned)

    return {"id": person_id, "snapshot": snapshot, "images": len(embeddings)}


def main():
    parser = argparse.ArgumentParser(
        description="Enroll a consented person from one or more photos."
    )
    parser.add_argument("--image", action="append", required=True,
                        help="path to face photo (repeatable for averaging)")
    parser.add_argument("--name", required=True)
    parser.add_argument("--role", required=True, choices=["Faculty", "Student"])
    parser.add_argument("--department", required=True)
    parser.add_argument("--id", required=True, help="Employee ID / Student ID")
    parser.add_argument("--no-consent", action="store_true",
                        help="refuse enrollment without consent (default: consent given)")
    args = parser.parse_args()

    db = Database()
    result = enroll(
        db, FaceDetector(), FaceRecognizer(), args.image,
        args.name, args.role, args.department, args.id,
        consent=not args.no_consent,
    )
    print(f"Enrolled #{result['id']}: {args.name} ({args.role}, {args.department}) "
          f"from {result['images']} image(s)")


if __name__ == "__main__":
    main()
