"""Threshold sweep: turn the SFace cosine threshold into an evidenced point.

0.363 is the vendor default for SFace — a reasonable start, but not a
measurement on YOUR camera, YOUR enrolled photos, or YOUR lighting. This
tool measures score distributions and sweeps the threshold:

  positives (genuine pairs) : same identity, original vs augmented variants
                              (flip, rotation, brightness, blur, scale)
  negatives (imposter pairs): every cross-identity embedding pair

For each threshold it reports FAR (false accept rate), FRR, precision,
recall and F1, and recommends two operating points:

  balanced : the threshold with the best F1
  security : the highest recall achievable under FAR <= --max-far
             (default 1%) — the setting for an access-control appliance

Usage:
    python threshold_sweep.py                 # sample_data, with augmentation
    python threshold_sweep.py --dir my_photos --max-far 0.01

Outputs reports/threshold_sweep.csv and reports/threshold_sweep.md.

Honest limitation: augmented variants are a surrogate for real cross-session
captures. Re-run this on probe photos taken with the actual installation
camera before freezing the threshold (noted in the generated report too).
"""

import argparse
import csv
import glob
import os
import sys

import cv2
import numpy as np

from face_detection import FaceDetector
from face_recognition import COSINE_THRESHOLD, FaceRecognizer, cosine

REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp")
DEFAULT_THRESHOLDS = [round(0.20 + i * 0.01, 2) for i in range(51)]  # 0.20-0.70


# ------------------------------------------------------------------ inputs

def augment(img):
    """Cross-condition variants of one photo (surrogate probe session)."""
    h, w = img.shape[:2]
    out = []
    out.append(("hflip", cv2.flip(img, 1)))
    for angle in (-8, 8):
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        out.append((f"rot{angle:+d}", cv2.warpAffine(
            img, m, (w, h), borderMode=cv2.BORDER_REFLECT)))
    for factor in (0.7, 1.3):
        out.append((f"bright{factor}", np.clip(
            img.astype(np.float32) * factor, 0, 255).astype(np.uint8)))
    out.append(("blur1.5", cv2.GaussianBlur(img, (5, 5), 1.5)))
    small = cv2.resize(img, (max(8, w // 2), max(8, h // 2)))
    out.append(("scale0.5", cv2.resize(small, (w, h))))
    return out


def collect_scores(image_dir, do_augment=True, max_images=0):
    """Embed every identity's original + variants; return genuine/imposter
    cosine lists. Identity = image filename (each file is one person)."""
    det, rec = FaceDetector(), FaceRecognizer()
    paths = [p for p in sorted(glob.glob(os.path.join(image_dir, "*")))
             if p.lower().endswith(IMAGE_EXT)]
    if max_images:
        paths = paths[:max_images]
    if not paths:
        sys.exit(f"no images found in: {image_dir}")

    per_identity = {}       # name -> [embedding, ...]
    used = skipped = 0
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            continue
        name = os.path.basename(path)
        embeds = []
        variants = [(("original"), img)]
        if do_augment:
            variants += augment(img)
        for tag, var in variants:
            face = det.best(det.detect(var))
            if face is None:
                skipped += 1
                continue
            embeds.append(rec.embed(var, face))
            used += 1
        if embeds:
            per_identity[name] = embeds

    if len(per_identity) < 1:
        sys.exit("no faces detected in the input set")

    pos, neg = [], []
    names = sorted(per_identity)
    for name in names:
        embeds = per_identity[name]
        for i in range(len(embeds)):
            for j in range(i + 1, len(embeds)):
                pos.append(cosine(embeds[i], embeds[j]))
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            for ea in per_identity[names[a]]:
                for eb in per_identity[names[b]]:
                    neg.append(cosine(ea, eb))

    print(f"identities={len(names)} embeddings={used} "
          f"(skipped variants: {skipped}) "
          f"genuine_pairs={len(pos)} imposter_pairs={len(neg)}")
    return pos, neg


# ------------------------------------------------------------ pure scoring

def summarize(pos, neg, thresholds=None, max_far=0.01):
    """Sweep thresholds over the score lists; importable for tests (no models).

    Returns {"rows": [...], "balanced": row, "security": row, "vendor": row}.
    """
    thresholds = thresholds or DEFAULT_THRESHOLDS
    rows = []
    n_pos, n_neg = len(pos), len(neg)
    for t in thresholds:
        tp = sum(1 for s in pos if s >= t)
        fn = n_pos - tp
        fp = sum(1 for s in neg if s >= t)
        tn = n_neg - fp
        far = fp / n_neg if n_neg else 0.0
        frr = fn / n_pos if n_pos else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append({
            "threshold": t, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "far": round(far, 4), "frr": round(frr, 4),
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4),
        })

    # Perfectly separated data ties many thresholds: report the widest
    # proven-safe interval and pick its midpoint (max margin from BOTH the
    # nearest imposter and the nearest genuine score).
    def _midpoint(pool):
        if not pool:
            return None
        return pool[len(pool) // 2]

    secure_pool = [r for r in rows if r["far"] <= max_far]
    if secure_pool:
        best_recall = max(r["recall"] for r in secure_pool)
        tied = [r for r in secure_pool if r["recall"] == best_recall]
        security = _midpoint(tied)
        security_interval = (tied[0]["threshold"], tied[-1]["threshold"])
    else:
        security = min(rows, key=lambda r: r["far"])
        security_interval = (security["threshold"], security["threshold"])

    best_f1 = max(r["f1"] for r in rows)
    balanced = _midpoint([r for r in rows if r["f1"] == best_f1])

    vendor = next((r for r in rows
                   if abs(r["threshold"] - COSINE_THRESHOLD) < 1e-9), None)
    return {"rows": rows, "balanced": balanced, "security": security,
            "vendor": vendor, "max_far": max_far,
            "security_interval": security_interval,
            "n_pos": n_pos, "n_neg": n_neg}


# ----------------------------------------------------------------- reports

def _fmt(row):
    return (f"| {row['threshold']:.3f} | {row['far']:.4f} | {row['frr']:.4f} | "
            f"{row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} |")


def write_reports(result, out_dir=REPORT_DIR, source="", augmented=True):
    os.makedirs(out_dir, exist_ok=True)
    rows = result["rows"]
    bal, sec = result["balanced"], result["security"]
    lo, hi = result["security_interval"]
    ven = result["vendor"]
    probes = ("augmented cross-condition variants "
              "(flip/rotate/brightness/blur/scale)"
              if augmented else "original images only")

    csv_path = os.path.join(out_dir, "threshold_sweep.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md_path = os.path.join(out_dir, "threshold_sweep.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(f"""# Face matching threshold sweep

- Source: {source}
- Genuine pairs: {result['n_pos']} · Imposter pairs: {result['n_neg']}
- Probes: {probes}
- Sweep: {rows[0]['threshold']:.2f} – {rows[-1]['threshold']:.2f} step 0.01

## Recommended operating points

| Use case | Threshold | FAR | FRR | Recall | F1 |
|---|---|---|---|---|---|
| **Security** (FAR <= {result['max_far']:.0%}) | **{sec['threshold']:.3f}** | {sec['far']:.4f} | {sec['frr']:.4f} | {sec['recall']:.3f} | {sec['f1']:.3f} |
| **Balanced** (best F1) | **{bal['threshold']:.3f}** | {bal['far']:.4f} | {bal['frr']:.4f} | {bal['recall']:.3f} | {bal['f1']:.3f} |
""")
        if ven:
            fh.write(f"| Vendor default (SFace) | {ven['threshold']:.3f} | "
                     f"{ven['far']:.4f} | {ven['frr']:.4f} | "
                     f"{ven['recall']:.3f} | {ven['f1']:.3f} |\n")
        fh.write(f"""
**Proven-safe interval: [{lo:.2f}, {hi:.2f}]** — every threshold in this
range achieved FAR <= {result['max_far']:.0%} at the maximum measured recall.
The recommended value is the interval midpoint (largest margin from both the
nearest imposter and the nearest genuine score). The vendor default
({COSINE_THRESHOLD}) is inside the interval if that row above shows
acceptable numbers.

## Sweep table (selected rows)

| threshold | FAR | FRR | precision | recall | F1 |
|---|---|---|---|---|---|
""")
        for i, row in enumerate(rows):
            if i % 5 == 0:                     # every 0.05 step
                fh.write(_fmt(row) + "\n")
        fh.write("""
## How to read this

- **FAR** (false accept rate): fraction of *imposter* pairs that would be
  accepted — the security metric. For an access appliance you want this
  near zero; it is bounded by the number of imposter pairs measured.
- **FRR** (false reject rate): fraction of *genuine* pairs refused — the
  convenience metric.

## Honest limitations

Augmented variants stand in for a second capture session (different day,
camera distance, expression). Before freezing the threshold for real use,
re-run this sweep with probe photos taken by the actual installation camera:

    python threshold_sweep.py --dir path/to/real_probes
""")
    print(f"reports -> {csv_path}\n         {md_path}")
    return csv_path, md_path


def main():
    parser = argparse.ArgumentParser(description="Sweep the cosine match threshold")
    parser.add_argument("--dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "sample_data"),
        help="directory of face images, one identity per file")
    parser.add_argument("--no-augment", action="store_true",
                        help="use original images only (thin statistics)")
    parser.add_argument("--max-far", type=float, default=0.01,
                        help="FAR ceiling for the security point (default %(default)s)")
    parser.add_argument("--max-images", type=int, default=0,
                        help="limit input images (debugging)")
    args = parser.parse_args()

    pos, neg = collect_scores(args.dir, do_augment=not args.no_augment,
                              max_images=args.max_images)
    result = summarize(pos, neg, max_far=args.max_far)
    write_reports(result, source=args.dir, augmented=not args.no_augment)

    bal, sec = result["balanced"], result["security"]
    print(f"security point : t={sec['threshold']:.3f} FAR={sec['far']:.4f} "
          f"recall={sec['recall']:.3f}")
    print(f"balanced point : t={bal['threshold']:.3f} F1={bal['f1']:.3f}")
    if result["vendor"]:
        print(f"vendor default : t={result['vendor']['threshold']:.3f} "
              f"FAR={result['vendor']['far']:.4f} "
              f"recall={result['vendor']['recall']:.3f}")


if __name__ == "__main__":
    main()
