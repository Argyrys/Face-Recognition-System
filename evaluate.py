"""Evaluate recognition quality: TP/FP/FN/TN, accuracy, precision, recall, F1.

Modes:
  --self-test                 re-identify sample_data images against itself;
                              every correct re-match is a TP, a match to the
                              wrong person is FP, a miss is FN.
  --gallery D --probe D --label X
                              probe images (true identity X) against a gallery
                              built from enrolled images in D.

Results are written to reports/evaluation.csv.
"""

import argparse
import csv
import glob
import os
import sys

import cv2

from database import Database
from face_detection import FaceDetector
from face_recognition import FaceRecognizer, best_match

REPORT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "reports", "evaluation.csv")
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def _images(directory):
    return [p for p in sorted(glob.glob(os.path.join(directory, "*")))
            if p.lower().endswith(IMAGE_EXT)]


def evaluate(gallery_dir, probe_dir, label=None):
    """Return (rows, summary). label=None -> self-identification test."""
    db = Database(path=":memory:")
    det, rec = FaceDetector(), FaceRecognizer()
    known = set()

    for path in _images(gallery_dir):
        img = cv2.imread(path)
        if img is None:
            continue
        face = det.best(det.detect(img))
        if face is None:
            continue
        name = os.path.basename(path)
        db.add_person(name, "Student", "Eval", path, rec.embed(img, face), True)
        known.add(name)

    gallery = db.gallery()
    if not gallery:
        sys.exit(f"no faces found in gallery dir: {gallery_dir}")

    rows, tp, fp, fn, tn = [], 0, 0, 0, 0
    for path in _images(probe_dir):
        img = cv2.imread(path)
        if img is None:
            continue
        actual = label if label is not None else os.path.basename(path)
        face = det.best(det.detect(img))
        if face is None:
            rows.append((os.path.basename(path), actual, "NO_FACE", "", "FN"))
            fn += 1
            continue
        person_id, score = best_match(rec.embed(img, face), gallery)
        if person_id is None:
            if actual in known:
                fn += 1
                verdict = "FN"
            else:
                tn += 1
                verdict = "TN"
            predicted = "UNKNOWN"
        else:
            predicted = db.person(person_id)["name"]
            if predicted == actual:
                tp += 1
                verdict = "TP"
            else:
                fp += 1
                verdict = "FP"
        rows.append((os.path.basename(path), actual, predicted,
                     f"{score:.4f}", verdict))

    n = tp + tn + fn + fp
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * recall / (prec + recall) if (prec + recall) else 0.0
    summary = {
        "samples": n, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "accuracy": round(acc, 4), "precision": round(prec, 4),
        "recall": round(recall, 4), "f1": round(f1, 4),
    }
    return rows, summary


def write_report(rows, summary):
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "actual", "predicted", "cosine_score", "verdict"])
        w.writerows(rows)
        w.writerow([])
        w.writerow(["metric", "value"])
        for k, v in summary.items():
            w.writerow([k, v])
    return REPORT


def main():
    parser = argparse.ArgumentParser(description="Evaluate face recognition quality")
    parser.add_argument("--gallery", help="dir of enrolled-person photos")
    parser.add_argument("--probe", help="dir of probe/test photos")
    parser.add_argument("--label", help="true identity of probe images")
    parser.add_argument("--self-test", action="store_true",
                        help="re-identify sample_data images against itself")
    args = parser.parse_args()

    if args.self_test:
        sample = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "sample_data")
        if not os.path.isdir(sample) or not _images(sample):
            sys.exit("sample_data is empty — add a few face images first")
        rows, summary = evaluate(sample, sample)
    elif args.gallery and args.probe and args.label:
        rows, summary = evaluate(args.gallery, args.probe, args.label)
    else:
        parser.error("need --self-test or --gallery/--probe/--label")

    path = write_report(rows, summary)
    print("evaluation:", summary)
    print("report ->", path)


if __name__ == "__main__":
    main()
