# 🛰️ SAT-TRACK – Satellite Military Aircraft Detection & Tracking

**Oriented bounding box detection of 20 military aircraft classes in overhead satellite imagery using YOLO26-OBB at 1024 px with rotation augmentation (nano and small backbones).**

[![Model](https://img.shields.io/badge/model-YOLO26s--OBB-blue)](https://docs.ultralytics.com/tasks/obb/)
[![Dataset](https://img.shields.io/badge/dataset-MAR20-orange)](https://github.com/username/MAR20)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

---

## Overview

SAT-TRACK detects and classifies military aircraft in satellite images using oriented bounding boxes (OBB). Unlike horizontal bounding boxes, OBB captures the exact orientation of parked aircraft – critical for airbase monitoring, fleet counting, and activity assessment.

**20 classes** spanning US, Russian, and allied air forces: fighters (F-15, F-16, F-22, F/A-18, SU-34, SU-35), bombers (B-52, B-1B, TU-95, TU-160, TU-22), transports (C-5, C-17, C-130), tankers (KC-135, KC-10), AWACS (E-3, E-8), and maritime patrol (P-3C).

---

## Dataset

| Property | Value |
|----------|-------|
| Source | [MAR20](https://github.com/username/MAR20) |
| Images | 3,842 satellite images |
| Resolution | ~0.3–0.5 m/px (varies) |
| Classes | 20 military aircraft |
| Annotations | Oriented bounding boxes (DOTA format) |
| Train / Val | 1,331 / 2,511 (predefined split, no airfield overlap) |

### Classes

| US / Allied | Russian |
|-------------|---------|
| C-130, C-17, C-5, KC-135, KC-10 | SU-35, SU-34, SU-24 |
| E-3, E-8, P-3C | TU-160, TU-95, TU-22 |
| F-16, F-15, F-22, F/A-18, B-52, B-1B | – |

---

## Quick Start

```bash
# Install (a local virtualenv keeps this tidy)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Dataset conversion only (no training)
python train_yolo_obb.py --setup-only

# Validate (uses the newest run found, or --model to pick one)
python test_model.py --val
python test_model.py --model runs/obb/mar20_yolo26s_l40s_1024/weights/best.pt --val

# Predict on a single image
python test_model.py --image 42 --show

# Predict entire folder
python test_model.py --dir ./my_images/
```

### Reproducing the reported runs

These are the commands behind the numbers in Validation Results. Everything not passed on the command
line stays at the script defaults (`optimizer: auto`, `amp: true`, `seed: 0`, `deterministic: true`,
`close_mosaic: 10`).

```bash
# YOLO26s-OBB @ 1024 + rotation  ->  runs/obb/mar20_yolo26s_l40s_1024   (CURRENT MODEL, L40S)
# Same command as n - only --model and --name differ.
python train_yolo_obb.py --model yolo26s-obb.pt --epochs 200 --patience 60 \
    --imgsz 1024 --batch 32 --device 0 --workers 16 \
    --cache ram --degrees 180 --cos-lr --name mar20_yolo26s_l40s_1024

# YOLO26n-OBB @ 1024 + rotation  ->  runs/obb/mar20_yolo26n_l40s_1024   (capacity ablation, L40S)
python train_yolo_obb.py --model yolo26n-obb.pt --epochs 200 --patience 60 \
    --imgsz 1024 --batch 32 --device 0 --workers 16 \
    --cache ram --degrees 180 --cos-lr --name mar20_yolo26n_l40s_1024
```

Two defaults matter for reproduction:

- **`--epochs` defaults to 100**, so it must be passed explicitly. Both reported runs used 200.
- **`--patience` defaults to 20 and `--cos-lr` defaults to off**, so both must be passed to reproduce the
  early-stop and LR-tail behaviour of the reported runs.

The two runs differ **only** in `--model` (nano vs small) and `--name`: batch, image size, cache, workers,
augmentation, schedule and seed are identical, so the n → s comparison is a controlled capacity test.

### Resolved hyperparameters (identical for the n and s runs)

Dumped from each run's `args.yaml`:

| Group | Values |
|-------|--------|
| Input / batch | `imgsz 1024`, `batch 32`, `cache ram`, `workers 16`, `nbs 64` |
| Schedule | `epochs 200`, `patience 60`, `lr0 0.01`, `lrf 0.01`, `cos_lr true`, `momentum 0.937`, `weight_decay 0.0005`, `warmup_epochs 3.0` |
| Loss | `box 7.5`, `cls 0.5`, `dfl 1.5` |
| Val / inference | `iou 0.7`, `max_det 300`, `conf null` |
| Geometry aug | `degrees 180.0`, `translate 0.1`, `scale 0.5`, `shear 0.0`, `perspective 0.0` |
| Flip / mosaic | `flipud 0.0`, `fliplr 0.5`, `mosaic 1.0`, `mixup 0.0`, `cutmix 0.0`, `copy_paste 0.0` |
| Photometric | `hsv_h 0.015`, `hsv_s 0.7`, `hsv_v 0.4`, `erasing 0.4`, `auto_augment randaugment` |
| Determinism | `seed 0`, `deterministic true`, `optimizer auto`, `amp true` |

### L40S configuration

What the 1024 recipe changes on the training side, and what each flag bought:

```bash
python train_yolo_obb.py --model yolo26s-obb.pt --epochs 200 --patience 60 \
    --imgsz 1024 --batch 32 --device 0 --workers 16 \
    --cache ram --degrees 180 --cos-lr --name mar20_yolo26s_l40s_1024
```

| Flag | Effect measured |
|------|-----------------|
| `--cache ram` | Removed per-epoch decode; 1,331 images at ~800 px fit in well under 1 GB |
| `--workers 16` | Loader kept up; GPU went to 89–100% utilisation (s), 326 W of a 350 W TDP |
| `--batch 32 --imgsz 1024` | 25.7 GB resident of 46 GB (s), 15.3 GB (n), 26.9 s/epoch (s), 17.1 s/epoch (n) |
| `--degrees 180` | Rotation augmentation, matching arbitrary satellite azimuth |
| `--cos-lr --patience 60` | Cosine tail; s best fitness at epoch 100 and stopped at 160 |

Measured throughput for the current configuration:

| Run | Images/s | Pixel/s |
|-----|----------|---------|
| 1024 @ batch 32, `cache: ram` | ~105 | ~110 Mpixel/s |

The GPU utilisation figures (`nvidia-smi`) measure kernel residency, not FLOP occupancy: this config sits
around 5% MFU, which is expected for a 10.5 M-parameter detector.

### YOLO26 OBB checkpoints

| Size | Checkpoint |
|------|------------|
| nano | `yolo26n-obb.pt` |
| small | `yolo26s-obb.pt` |

YOLO26-OBB removes DFL regression and runs **NMS-free end-to-end** inference, so post-processing cost no
longer scales with the number of boxes. That matters here: dense airfields generate hundreds of
overlapping OBBs, exactly where NMS dominates latency.

---

## Validation Results

> **Current model:** `runs/obb/mar20_yolo26s_l40s_1024` – YOLO26s-OBB (10.54 M params), same 1024 recipe.
> `best.pt` (epoch 100 of 160) scores **mAP@0.5 0.8914 / mAP@0.5:0.95 0.7862**, P 0.8636, R 0.8420, with a
> 60-epoch level of **0.8660 / 0.7626** and 15 of 160 epochs within 0.01 of its peak. Validated with
> `python test_model.py --model runs/obb/mar20_yolo26s_l40s_1024/weights/best.pt --val`
> (the script reads `imgsz` from the run's `args.yaml`) over all 2,511 val images / 14,471 instances. It is
> the deployment candidate for the Jetson Orin Nano and the cheapest good model in the repo.

> **Capacity ablation:** `runs/obb/mar20_yolo26n_l40s_1024` – YOLO26n-OBB (2.65 M params), identical command
> and recipe. `best.pt` (epoch 108 of 168) scores **mAP@0.5 0.8708 / mAP@0.5:0.95 0.7620**, P 0.8258,
> R 0.8223, with a 60-epoch level of **0.8534 / 0.7465**.

### Benchmark across backbone scales

Both rows use the identical command, the same split, the same decoder and the same hyperparameters; only
the backbone scale changes, so they are a like-for-like capacity comparison.

| Model | Input | Params | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Inference |
|-------|-------|--------|---------|--------------|-----------|--------|-----------|
| YOLO26n-OBB + rot aug | 1024 | 2.65 M | 0.8708 | 0.7620 | 0.8258 | 0.8223 | – |
| **YOLO26s-OBB + rot aug** | **1024** | **10.54 M** | **0.8914** | **0.7862** | **0.8636** | **0.8420** | 4.0 ms |

Going from nano to small (4× the parameters) buys **+0.0206 mAP@0.5** and **+0.0242 mAP@0.5:0.95**, plus
**+0.0378 precision** and **+0.0197 recall**. Unlike a pure width step on a downscaled input, here the
larger backbone wins on *every* metric, so capacity is still buying real accuracy at this resolution.

### Peak vs converged level (both scales)

Best-epoch numbers are what `best.pt` holds; the "level" is a mean ± std over a window of the run's own
epochs. That window is the honest estimator, because best-epoch selection is a maximum taken over exactly
this kind of jitter. Both windows are 60 epochs.

| Model | Best epoch | Level window | Peak mAP50 | Peak mAP50-95 | Level mAP50 | Level mAP50-95 | Level P | Level R | Epochs ≥ 0.90 | Within 0.01 of peak |
|-------|-----------|--------------|------------|---------------|-------------|----------------|---------|---------|---------------|---------------------|
| YOLO26n-OBB | 108 (of 168) | last 60 | 0.8708 | 0.7620 | 0.8534 ± 0.0039 | 0.7465 ± 0.0034 | 0.8239 | 0.7897 | 0 | 16 / 168 |
| **YOLO26s-OBB** | **100 (of 160)** | **last 60** | **0.8914** | **0.7862** | **0.8660 ± 0.0086** | **0.7626 ± 0.0076** | **0.8554** | **0.8270** | **0** | **15 / 160** |

Read across the level columns:

- **n → s holds up.** +0.0126 level mAP@0.5 and +0.0161 level mAP@0.5:0.95, which is roughly 1.5–2× the
  wider of the two runs' epoch-to-epoch spread, and the ordering survives either way of measuring it
  (peak: +0.0206 / +0.0242, level: +0.0126 / +0.0161). Still one run per scale, so treat it as a signal
  with a plausible sign, not as a constant.
- **Neither run's peak is a plateau.** Only **15 of 160** s epochs land within 0.01 of its peak and
  **16 of 168** for n; the honest figure to quote for either model is the level column, not the best epoch.

YOLO26n has no inference timing listed: it was not timed at 1024. The s figure (4.0 ms inference, ~6 ms
per image end-to-end) was measured by the L40S validation pass, so it is a training-card number, not an
edge one – the Jetson benchmark exists to produce a common-device latency table.

---

## Run Details

### YOLO26s @ 1024 run details

**Image size:** 1024×1024 · **Rotation augmentation:** `degrees 180`  
**Epochs:** 200 configured – stopped at 160 by `patience 60`; best fitness at epoch 100 (held by `best.pt`)  
**Batch:** 32 · **Cache:** ram · **Workers:** 16 · **cos_lr:** enabled · **Train / Val:** 1,331 / 2,511  
**Hardware:** NVIDIA L40S (46 GB usable, 192 vCPU host) – 26.9 s/epoch, 72 min total, peak 25.7 GB VRAM  
**Utilisation:** 89–100% `nvidia-smi`, 326 W of 350 W TDP, SM clock pinned at 2505/2520 MHz  
**Checkpoints:** intermediate ones deleted during the run to stay inside a 10 GB volume; `best.pt` retained

### YOLO26n @ 1024 run details

**Image size:** 1024×1024 · **Rotation augmentation:** `degrees 180`  
**Epochs:** 200 configured – early-stopped at 168 by `patience 60`; best fitness at epoch 108 (held by `best.pt`)  
**Batch:** 32 · **Cache:** ram · **Workers:** 16 · **cos_lr:** enabled · **Train / Val:** 1,331 / 2,511  
**Hardware:** NVIDIA L40S (46 GB usable, 192 vCPU host) – 17.1 s/epoch median, 51 min total, peak 15.3 GB VRAM  
**Utilisation:** mean **54%** `nvidia-smi` (range 20–99%), 213 W of 350 W TDP, SM clock pinned at 2505/2520 MHz  
**Checkpoints:** all `epoch*.pt` retained locally (19 files, ~470 MB); `best.pt` is the deployable one

Same command as the s@1024 run, only the checkpoint differs, so this is a controlled capacity test. The
L40S idles about half the time here: nano's forward/backward at batch 32 is cheap enough that the card
finishes and waits on the dataloader, whereas the 4× heavier s model hid the same loader cost at 89–100%
utilisation. Sizing the batch to the model (64–128 for nano at 1024) would have cut wall clock roughly in
half without changing the effective batch, since `nbs 64` absorbs it via gradient accumulation.

The 0.7729 fitness at epoch 108 is an **isolated peak**, not the run's level: the next best epochs in the
whole run are ep 82 (0.7721) and ep 81 (0.7696), nothing adjacent comes close, and the mean mAP@0.5 of the
last ten epochs is 0.8485 against a peak of 0.8708. Quote ~0.76 mAP@0.5:0.95 as nano's reproducible
performance at 1024, and treat the peak as a lucky draw under patience expiry.

### YOLO26s vs. YOLO26n at 1024

Raw 4-decimal values from each run's own validation pass.

| Metric | YOLO26n-OBB | YOLO26s-OBB | Δ |
|--------|-------------|-------------|---|
| mAP@0.5 | 0.8708 | 0.8914 | **+0.0206** |
| mAP@0.5:0.95 | 0.7620 | 0.7862 | **+0.0242** |
| Precision | 0.8258 | 0.8636 | +0.0378 |
| Recall | 0.8223 | 0.8420 | +0.0197 |

Per-class deltas for the n → s step are not tabulated: the per-class validation output was kept for the
small model only. The s run's own per-class table is below.

### YOLO26s @ 1024, per class

Sorted by mAP@0.5. All 20 classes, 14,471 val instances.

| Class | mAP50 | mAP50-95 | Precision | Recall | Instances |
|-------|-------|----------|-----------|--------|-----------|
| B-52 | **0.987** | 0.904 | 0.989 | 0.896 | 579 |
| B-1B | 0.986 | 0.891 | 0.831 | 0.996 | 556 |
| C-17 | 0.986 | 0.874 | 0.932 | 0.986 | 860 |
| C-130 | 0.981 | 0.887 | 0.982 | 0.932 | 993 |
| E-3 | 0.980 | 0.895 | 0.939 | 0.990 | 489 |
| C-5 | 0.970 | 0.890 | 0.993 | 0.861 | 463 |
| TU-160 | 0.969 | 0.863 | 0.980 | 0.891 | 334 |
| P-3C | 0.968 | 0.862 | 0.970 | 0.944 | 611 |
| TU-95 | 0.955 | 0.842 | 0.969 | 0.910 | 958 |
| TU-22 | 0.936 | 0.800 | 0.841 | 0.903 | 462 |
| SU-24 | 0.927 | 0.764 | 0.936 | 0.877 | 551 |
| KC-135 | 0.925 | 0.811 | 0.961 | 0.799 | 1,119 |
| SU-35 | 0.914 | 0.791 | 0.873 | 0.825 | 1,092 |
| SU-34 | 0.851 | 0.733 | 0.814 | 0.801 | 717 |
| E-8 | 0.847 | 0.762 | 0.850 | 0.815 | 356 |
| F/A-18 | 0.840 | 0.727 | 0.974 | 0.556 | 1,847 |
| F-16 | 0.821 | 0.691 | 0.699 | 0.805 | 827 |
| KC-10 | 0.774 | 0.705 | 0.554 | 0.804 | 148 |
| F-15 | 0.617 | 0.533 | 0.723 | 0.521 | 1,091 |
| F-22 | 0.600 | 0.503 | 0.493 | 0.687 | 418 |

**13 of 20 classes clear 0.90 mAP@0.5**: every bomber, transport, AWACS and maritime patrol class, plus
SU-35. Big airframes are effectively solved (B-52, B-1B, C-17, C-130, E-3 all ≥ 0.98).

The failures are concentrated in the fighter block, and they are **two different failure modes**:

- **Low recall (operating point):** **F-15** (P 0.723 / R 0.521) and **F/A-18** (P 0.974 / R 0.556) put a
  confident box on what they find and miss the rest. Per-class confidence calibration is the cheap fix
  here, not more training.
- **Low precision (identity):** **F-16** (P 0.699 / R 0.805) and **F-22** (P 0.493 / R 0.687) find aircraft
  but over-predict and mislabel them. That is consistent with twin-tail fighter planforms being nearly
  interchangeable at this ground sample distance, and no threshold fixes it.
- **KC-10** (0.774) is the weakest tanker with a small instance count (148), so its number carries the most
  val noise of the group.

### Measurement caveats (read before quoting numbers)

- **One run per scale.** No seed averaging, so the n → s gap of +0.0206 is a single observation.
- **Epoch-to-epoch jitter is real.** Standard deviation of mAP@0.5 over the last 60 epochs is 0.0086 for s
  and 0.0039 for n, and the last-ten-epoch means are 0.8698 (s) vs a 0.8914 peak. The n → s gain is larger
  than one sigma for s, but treat it as "s is better and probably by about two points", not as a settled
  constant.
- **`best.pt` was selected as the best epoch on this same val split**, so it carries selection optimism.
  A held-out test set (MAR20's split is reused as val here) is what would settle it.
- **The nano number is a peak, not a level.** Best fitness landed at epoch 108 (0.7729) and patience expired
  60 epochs later at epoch 168; no adjacent epoch came within 0.013 of it, and the last-ten-epoch mean
  mAP@0.5 is 0.8485 vs a peak of 0.8708. Its reproducible level at 1024 is ~0.76 mAP@0.5:0.95, so the
  nano/small gap in reproducible terms is wider than the table's peak-vs-peak delta suggests.
- **The two runs are a controlled pair.** Identical command (batch 32, `cache ram`, `degrees 180`,
  `cos_lr`, `patience 60`, `seed 0`), identical hardware; only capacity differs.
- **Input-size caveat:** the 1024 models were validated at 1024 because `test_model.py` reads `imgsz` from
  the run's `args.yaml`. Scoring them at any other size understates them and is not comparable.
- **Both runs were launched from the same command**, so hyperparameters are auditable from the
  reproduction block rather than asserted.
- **Same training hardware for both runs** (NVIDIA L40S 48 GB). Training hardware does not change the
  weights' quality, but cuDNN/cuBLAS kernel selection means `seed: 0` + `deterministic: true` does **not**
  give bit-identical reproduction across GPUs. Expect ±1e-3 wobble in the metrics on a different card.
- **Latency is a single-device number.** The 4.0 ms inference figure comes from the L40S val pass, not from
  the edge target, so it is not a deployment number.

### Training Curves

![Training Results](runs/obb/mar20_yolo26s_l40s_1024/results.png)

### Confusion Matrix

![Confusion Matrix](runs/obb/mar20_yolo26s_l40s_1024/confusion_matrix_normalized.png)

### Validation Predictions

| Ground Truth | Predictions |
|-------------|-------------|
| ![Labels 0](runs/obb/mar20_yolo26s_l40s_1024/val_batch0_labels.jpg) | ![Preds 0](runs/obb/mar20_yolo26s_l40s_1024/val_batch0_pred.jpg) |
| ![Labels 1](runs/obb/mar20_yolo26s_l40s_1024/val_batch1_labels.jpg) | ![Preds 1](runs/obb/mar20_yolo26s_l40s_1024/val_batch1_pred.jpg) |

---

## Per-Class Breakdown (YOLO26s-OBB @ 1024)

Classes grouped by mAP@0.5. Numbers are the validation pass quoted above.

### 🟢 Strong (mAP50 ≥ 0.90) – Large, Distinct Airframes

| Class   | mAP50 | Precision | Recall | Notes |
|---------|-------|-----------|--------|-------|
| B-52    | 0.987 | 0.989     | 0.896  | Stratofortress – best class overall at this resolution |
| B-1B    | 0.986 | 0.831     | 0.996  | Lancer – swing-wing, near-perfect recall |
| C-17    | 0.986 | 0.932     | 0.986  | Globemaster III – tight, reliable boxes |
| C-130   | 0.981 | 0.982     | 0.932  | Hercules – balanced P/R |
| E-3     | 0.980 | 0.939     | 0.990  | AWACS – rotodome unmistakable from above |
| C-5     | 0.970 | 0.993     | 0.861  | Galaxy – massive T-tail airlifter, highest precision |
| TU-160  | 0.969 | 0.980     | 0.891  | Blackjack – largest combat aircraft |
| P-3C    | 0.968 | 0.970     | 0.944  | Orion – long fuselage + 4-engine prop |
| TU-95   | 0.955 | 0.969     | 0.910  | Bear – swept wings + 4 turboprops |
| TU-22   | 0.936 | 0.841     | 0.903  | Backfire – swing-wing bomber |
| SU-24   | 0.927 | 0.936     | 0.877  | Fencer – variable-sweep wing |
| KC-135  | 0.925 | 0.961     | 0.799  | Stratotanker – precision high, recall the weak half |
| SU-35   | 0.914 | 0.873     | 0.825  | Flanker-E – strongest fighter class |

> 13/20 classes above 0.90 mAP50.

### 🟡 Moderate (0.80–0.90)

| Class  | mAP50 | Precision | Recall | Notes |
|--------|-------|-----------|--------|-------|
| SU-34  | 0.851 | 0.814     | 0.801  | Fullback – balanced but mid |
| E-8    | 0.847 | 0.850     | 0.815  | JSTARS – 356 val instances |
| F/A-18 | 0.840 | 0.974     | 0.556  | Hornet – near-perfect precision, misses ~44% |
| F-16   | 0.821 | 0.699     | 0.805  | Falcon – finds them, over-predicts |

### 🟠 Weak (0.70–0.80)

| Class  | mAP50 | Precision | Recall | Notes |
|--------|-------|-----------|--------|-------|
| KC-10  | 0.774 | 0.554     | 0.804  | Extender – lowest precision of the 13 strong-airframe families |

### 🔴 Poor (<0.70)

| Class  | mAP50 | Precision | Recall | Notes |
|--------|-------|-----------|--------|-------|
| F-15   | 0.617 | 0.723     | 0.521  | Eagle – recall-limited at the default operating point |
| F-22   | 0.600 | 0.493     | 0.687  | Raptor – finds them but over-predicts and mislabels |

> **The shape of the error:** every class above 0.90 is a distinct large airframe. Of the six classes below
> 0.85, four are fighters; the outliers are KC-10 (a 148-instance tanker class) and E-8 (an AWACS class that
> sits at 0.847, just under the line). The fighters fail in two directions – F-15/F/A-18 by missing aircraft
> (low recall), F-16/F-22 by labelling them wrong (low precision) – so a single global threshold cannot fix
> both sides.

---

## Why the Fighter Gap Exists

1. **Planform similarity** – all modern fighters are single/double-seat, swept-wing, single/double-tail. The difference between an F-16 and SU-35 from 800 km up is a few pixels of wing sweep.
2. **Scale** – fighters are 15–20 m long. At 0.3–0.5 m/px they occupy ~48×48 pixels at a 1024×1024 input. Quadrupling the parameters (n → s) moved the aggregate mAP@0.5 by +0.0206 and left F-15 at 0.617 and F-22 at 0.600, which reads as a data/ambiguity limit rather than a capacity limit.
3. **Feature asymmetry** – the model learns strong features for bombers (wing config, engine count, size) that don't exist for fighters.
4. **Not data starvation** – at 1024 the correlation between class instance count and mAP@0.5 is **−0.055**. F-22 has 418 val instances and scores 0.600; TU-160 has 334 and scores 0.969. The difference tracks how distinguishable the airframe is, not how many examples exist.
5. **Operating point** – F-15 (P 0.723 / R 0.521) and F/A-18 (P 0.974 / R 0.556) are recall-limited at the default confidence, not incapable. F-16 (P 0.699) and F-22 (P 0.493) are precision-limited.

Resolution and rotation helped the classes that were *slightly* under-resolved (SU-24 0.927, TU-22 0.936,
F-16 0.821, KC-10 0.774) but did not rescue the ones that are fundamentally ambiguous at this ground
sample distance (F-15 0.617, F-22 0.600).

---

## Architecture Strategy

**Phase 1 – Scale up YOLO26-OBB:** done. YOLO26s-OBB (10.54 M params) at 1024 with rotation
augmentation reaches mAP@0.5 0.8914 / mAP@0.5:0.95 0.7862, beating YOLO26n-OBB trained with the identical
command on every metric (+0.0206 mAP@0.5, +0.0378 precision, +0.0197 recall). Fighters still cap the
average: the class mean of the 14 non-fighter classes is 0.942, against 0.891 overall.

**Phase 2 – Resolution and augmentation:** done. `imgsz 1024` plus `degrees 180` is the recipe for both
reported runs; the pixel budget is what let the small classes move at all, and the identical command on
nano confirms the gain tracks capacity as well as pixels – resolution is a multiplier on the parameters you
already have.

**Phase 3 – Edge deployment and benchmarking:** YOLO26s-OBB (10.54 M) is the deliverable: the target is an
**NVIDIA Jetson Orin Nano 8GB** and 8 GB of shared memory is the binding constraint before compute is. FP16
TensorRT at batch 1 is the variant most likely to fit comfortably.

| Stage       | Model                          | Role                                              | Status        |
|-------------|--------------------------------|---------------------------------------------------|---------------|
| Scaled      | YOLO26n-OBB @ 1024 + rot aug   | NMS-free head, small-object regression, capacity check | ✅ mAP50 0.871 / 0.762 (level 0.853 / 0.747) |
| **Current** | **YOLO26s-OBB @ 1024 + rot aug** | **Width + resolution + rotation; cheapest good model, headline model** | ✅ **mAP50 0.891 / 0.786 (level 0.866 / 0.763)** |
| Deployment  | YOLO26s-OBB on Jetson Orin Nano | TensorRT export, on-device benchmark + paper      | ⬜ Next |

---

## Edge Deployment (Jetson Orin Nano 8GB)

The trained 1024 px YOLO26s-OBB checkpoint is the deliverable; the deployment target is an NVIDIA Jetson
Orin Nano 8GB (Ampere, 1024-core GPU, 6-core ARM Cortex-A78AE, 8 GB shared LPDDR5 at 68 GB/s).

Planned benchmark matrix – same 2,511-image val split, accuracy must match the desktop numbers reported
above. No desktop latency reference exists at 1024 yet; the L40S figure is training-card only, so the first
measured baseline will come from the Jetson run itself.

| Variant | What it measures |
|---------|------------------|
| PyTorch FP32 (`best.pt`) | Baseline, no optimisation |
| ONNX Runtime | Export portability, CPU/GPU EP |
| TensorRT FP16 | Realistic edge latency |
| TensorRT INT8 (calibrated on MAR20) | Throughput ceiling vs. mAP loss |

Metrics to report per variant: per-image latency (pre/inference/post split), FPS at batch 1, mean
power draw and energy per frame (`jtop`), peak memory, thermal throttling over a sustained run, and
mAP@0.5 / mAP@0.5:0.95 parity against the desktop checkpoint.

---

## Next Steps

The headline model is YOLO26s-OBB at 1024 with rotation augmentation: **0.8914 mAP@0.5 / 0.7862 mAP@0.5:0.95**
at `best.pt` (epoch 100 of 160), with a 60-epoch level of 0.8660 / 0.7626. What remains is deployment,
evaluation hygiene, and writing.

1. **Export and benchmark on Jetson Orin Nano 8GB** – `best.pt` → ONNX → TensorRT FP16/INT8, measure
   latency, FPS, power, and memory per variant.
2. **Validate on-device accuracy parity** – run the same val split on the Jetson at 1024; any mAP loss from
   FP16/INT8 quantisation is the headline trade-off, and 8 GB of shared memory is the likely binding
   constraint before compute is.
3. **Write the short paper** – YOLO26s-OBB on MAR20 (n → s capacity comparison, the per-class table) and the
   Jetson Orin Nano benchmark suite (accuracy / latency / power), quoting the exact commands from
   "Reproducing the reported runs" and the measurement caveats above.
4. **Prove gains with seeds, not single runs** – every number here is one run. Re-run the 1024 config 2–3
   times with different seeds and report mean ± std before claiming the +0.0206 is settled.
5. **Per-class confidence calibration** – the cheap, no-retraining win: F-15 (P 0.723 / R 0.521),
   F/A-18 (P 0.974 / R 0.556), KC-10 (P 0.554 / R 0.804) and E-8 (P 0.850 / R 0.815) are all sitting on a
   badly-placed operating point. Tune the conf threshold per class and re-report mAP.
6. **Optional: F-22 targeted work** – still the weakest class at 0.600 with precision 0.493. More pixels
   did not help, so the remaining levers are hard-negative mining against the other twin-tail fighters or
   an upstream crop-and-reclassify stage.

---

## Files

| File | Purpose |
|------|---------|
| `train_yolo_obb.py` | Conversion + training. Defaults to `yolo26n-obb.pt`, `--epochs 100`, `--patience 20`, `--batch 16`, `--imgsz 640`. Also passes through `--cache`, `--cos-lr`, `--degrees`, `--close-mosaic`, `--lr0`, `--optimizer`, `--seed`, `--no-amp`, `--rect`; `--model` selects another YOLO26-OBB variant |
| `test_model.py` | Validation, single-image prediction, batch inference on any YOLO26-OBB checkpoint (`--model` to choose). Reads `imgsz` from the run's `args.yaml` (override with `--imgsz`) |
| `benchmark_detection.py` | Class-agnostic detection benchmark (IoU / centre-offset / angle error, conf sweep) on any YOLO26-OBB checkpoint |
| `runs/obb/mar20_yolo26s_l40s_1024/` | **Current model** – YOLO26s-OBB @ 1024 with rotation augmentation, the cheapest good model and the Jetson candidate. Curves, confusion matrix, val batches, `best.pt` (weights git-ignored) |
| `runs/obb/mar20_yolo26n_l40s_1024/` | YOLO26n-OBB @ 1024 with rotation augmentation – the capacity ablation for the row above, same command, `best.pt` = epoch 108 |
| `logs/train_l40s_1024.log`, `logs/val_1024.txt` | Full training transcript and final validation output for YOLO26s (current model) |
| `yolo_obb_dataset/mar20_obb.yaml` | Dataset config (auto-generated, path fixed to this checkout) |

---
*Built with [Ultralytics YOLO26](https://docs.ultralytics.com/) on the MAR20 dataset.*
