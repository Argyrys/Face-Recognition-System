# Face matching threshold sweep

- Source: C:\Users\rs987\Documents\Default Project\VC_Face_Recognition\sample_data
- Genuine pairs: 56 · Imposter pairs: 64
- Probes: augmented cross-condition variants (flip/rotate/brightness/blur/scale)
- Sweep: 0.20 – 0.70 step 0.01

## Recommended operating points

| Use case | Threshold | FAR | FRR | Recall | F1 |
|---|---|---|---|---|---|
| **Security** (FAR <= 1%) | **0.450** | 0.0000 | 0.0000 | 1.000 | 1.000 |
| **Balanced** (best F1) | **0.450** | 0.0000 | 0.0000 | 1.000 | 1.000 |

**Proven-safe interval: [0.21, 0.69]** — every threshold in this
range achieved FAR <= 1% at the maximum measured recall.
The recommended value is the interval midpoint (largest margin from both the
nearest imposter and the nearest genuine score). The vendor default
(0.363) is inside the interval if that row above shows
acceptable numbers.

## Sweep table (selected rows)

| threshold | FAR | FRR | precision | recall | F1 |
|---|---|---|---|---|---|
| 0.200 | 0.0156 | 0.0000 | 0.983 | 1.000 | 0.991 |
| 0.250 | 0.0000 | 0.0000 | 1.000 | 1.000 | 1.000 |
| 0.300 | 0.0000 | 0.0000 | 1.000 | 1.000 | 1.000 |
| 0.350 | 0.0000 | 0.0000 | 1.000 | 1.000 | 1.000 |
| 0.400 | 0.0000 | 0.0000 | 1.000 | 1.000 | 1.000 |
| 0.450 | 0.0000 | 0.0000 | 1.000 | 1.000 | 1.000 |
| 0.500 | 0.0000 | 0.0000 | 1.000 | 1.000 | 1.000 |
| 0.550 | 0.0000 | 0.0000 | 1.000 | 1.000 | 1.000 |
| 0.600 | 0.0000 | 0.0000 | 1.000 | 1.000 | 1.000 |
| 0.650 | 0.0000 | 0.0000 | 1.000 | 1.000 | 1.000 |
| 0.700 | 0.0000 | 0.0357 | 1.000 | 0.964 | 0.982 |

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
