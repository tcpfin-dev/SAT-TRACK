# Validation Prediction Panels - Selection Notes

Source: `runs/obb/yolo26m-predict/batch/labels/` (2,505 prediction files, `class x1 y1 x2 y2 x3 y3 x4 y4 conf`).

## How the frames were scored

Every val frame was matched class-aware, greedy, highest confidence first against
`yolo_obb_dataset/labels/val/`, with quad IoU from `cv2.intersectConvexConvex` and a 0.5 IoU threshold.
Per frame that gives TP / FP / FN, precision, recall, F1, mean IoU of the true positives, and the count
of false positives that overlap a differently labelled aircraft (misclassifications).

Pool result over 2,511 val images: mean image F1 **0.856**, median **1.000**, 1,450 frames (57.7%) with
zero FP and zero FN, 1,492 at F1 >= 0.95, 111 frames with no true positive at all.

## Selection rule for the thirteen panels

- Zero FP, zero FN, zero misclassifications, F1 = 1.00 (all thirteen qualify)
- At least 7 aircraft, so the boxes have to survive overlap
- No two panels showing the same scene type, and together they cover all 20 classes
- Spread over scene density: 21, 20, 16, 16, 16, 14, 14, 12, 12, 12, 11, 10, 7 aircraft

| # | Frame | Aircraft | Classes | TP | mIoU | Mean conf |
|---|-------|----------|---------|----|------|-----------|
| 01 | `1198` | 20 | F/A-18 | 20 | 0.906 | 0.84 |
| 02 | `3467` | 21 | C-5 | 21 | 0.857 | 0.90 |
| 03 | `1439` | 16 | SU-24 x9, SU-34 x4, SU-35 x3 | 16 | 0.846 | 0.80 |
| 04 | `3727` | 12 | TU-95 x6, TU-160 x5, TU-22 x1 | 12 | 0.869 | 0.88 |
| 05 | `3216` | 11 | F-15 x6, C-130 x2, KC-135 x2, B-52 x1 | 11 | 0.855 | 0.89 |
| 06 | `2958` | 7 | KC-10 x3, KC-135 x2, C-17 x1, C-5 x1 | 7 | 0.858 | 0.78 |
| 07 | `2168` | 10 | E-3 | 10 | 0.915 | 0.90 |
| 08 | `242` | 14 | B-52 | 14 | 0.891 | 0.89 |
| 09 | `2810` | 12 | F-22 | 12 | 0.838 | 0.88 |
| 10 | `3300` | 16 | F-16 | 16 | 0.878 | 0.86 |
| 11 | `1374` | 12 | E-8 | 12 | 0.922 | 0.93 |
| 12 | `2065` | 14 | B-1B | 14 | 0.862 | 0.92 |
| 13 | `1180` | 16 | P-3C | 16 | 0.881 | 0.90 |

Panels 01-08 are the scene-type spread; 09-13 are the hard-class frames and were added so the section shows
every class the per-class tables argue about. They carry the weakest class in the repo (F-22, 0.615 mAP@0.5)
next to two of the strongest (B-1B 0.989, P-3C 0.975), at the same zero-FP / zero-FN standard as the rest.

All 20 classes are covered across the thirteen panels: SU-35, C-130, C-17, C-5, F-16, TU-160, E-3, B-52,
P-3C, B-1B, E-8, TU-22, F-15, KC-135, F-22, F/A-18, TU-95, KC-10, SU-34, SU-24.

## Caveats worth keeping in the README text

- These are hand-picked best frames, not a random sample. The aggregate 0.916 mAP@0.5 stays the headline;
  a caption should say the panels are deliberately the cleanest frames.
- 6 val images have no prediction file at all: `420`, `421`, `904`, `1537`, `1728`, `3293`. They score as
  zero and are included in the pool stats above.
- 111 frames (4.4%) have geometry right and every class wrong, which is the F-22/F-16/F-15 confusion the
  README already explains. `93` (9 C-5 + 2 KC-10 all called B-1B) is the clearest example if you ever want
  an error panel; it is not in this selection.

## Files

- `contact_sheet.jpg` - all thirteen panels, 2 columns
- `panel_01..13_*.jpg` - combined panels, ground truth left, predictions right; these are what the README's
  Validation Predictions table embeds
- `split/val_<id>_labels.jpg`, `split/val_<id>_pred.jpg` - the same thirteen frames as separate images, for
  the older two-column Ground Truth | Predictions layout if it is ever preferred
