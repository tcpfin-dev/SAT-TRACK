"""
Class-agnostic detection benchmark for YOLO-OBB checkpoints.

Ultralytics metrics (mAP) couple detection to classification: a perfectly placed box with the
wrong class scores zero. This script ignores the predicted class entirely and asks only two
questions per ground-truth aircraft:

    1. did the model put ANY box on it?          -> detection rate at IoU 0.5 / 0.75 / 0.9
    2. how far off is that box?                  -> IoU, centre offset (px), size ratio, angle error

Matching is greedy by oriented-box IoU (probabilistic IoU), highest pair first, one prediction per
ground-truth instance. Unmatched predictions are reported as spurious boxes so a permissive
threshold cannot look good just by spraying boxes everywhere.

Usage:
    python benchmark_detection.py                          # best model, val split, conf sweep
    python benchmark_detection.py --model runs/obb/mar20_yolo26s_l40s_1024/weights/best.pt
    python benchmark_detection.py --confs 0.25 0.1 0.01 --limit 300
    python benchmark_detection.py --classes F-A-18 SU-34   # subclass report only
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_YAML = BASE_DIR / "yolo_obb_dataset" / "mar20_obb.yaml"
LABELS_DIR = BASE_DIR / "yolo_obb_dataset" / "labels" / "val"
IMAGES_DIR = BASE_DIR / "yolo_obb_dataset" / "images" / "val"

CANDIDATE_WEIGHTS = [
    BASE_DIR / "runs" / "obb" / "mar20_yolo26s_l40s_1024" / "weights" / "best.pt",
    BASE_DIR / "runs" / "obb" / "mar20_yolo26n_l40s_1024" / "weights" / "best.pt",
]


def resolve_model(cli_model: str | None) -> Path:
    if cli_model:
        return Path(cli_model).expanduser().resolve()
    for cand in CANDIDATE_WEIGHTS:
        if cand.exists():
            return cand
    return CANDIDATE_WEIGHTS[0]


def resolve_imgsz(model_path: Path, cli_imgsz: int | None) -> int:
    if cli_imgsz:
        return cli_imgsz
    args_yaml = model_path.parent.parent / "args.yaml"
    if args_yaml.exists():
        for line in args_yaml.read_text().splitlines():
            if line.startswith("imgsz:"):
                try:
                    return int(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
    return 640


def load_names() -> list[str]:
    """Class names, from the dataset yaml so a retrained model with a different order still works."""
    text = DATA_YAML.read_text()
    m = re.search(r"names:\s*\[(.*?)\]", text, re.S)
    if not m:
        raise SystemExit(f"could not parse names from {DATA_YAML}")
    return [n.strip().strip("'\"").strip() for n in m.group(1).split(",")]


def load_gt(labels_dir: Path, images_dir: Path, names: list[str]):
    """{(stem): [(xyxyxyxy_px, class_name), ...]} and the image size for each stem."""
    import cv2
    import numpy as np

    gt, sizes = {}, {}
    for lbl in sorted(labels_dir.glob("*.txt")):
        stem = lbl.stem
        rows = [r.split() for r in lbl.read_text().splitlines() if r.strip()]
        if not rows:
            gt[stem] = []
            continue
        img_path = images_dir / f"{stem}.jpg"
        h, w = cv2.imread(str(img_path)).shape[:2]
        sizes[stem] = (w, h)
        boxes = []
        for r in rows:
            cid = int(r[0])
            pts = np.array([float(v) for v in r[1:9]], dtype=np.float32).reshape(4, 2)
            pts *= np.array([w, h], dtype=np.float32)
            boxes.append((pts, names[cid]))
        gt[stem] = boxes
    return gt, sizes


def greedy_match(iou_mat, iou_thresh: float):
    """Return [(gt_idx, det_idx), ...] one-to-one, highest IoU first."""
    import numpy as np

    pairs, used_g, used_d = [], set(), set()
    work = iou_mat.copy()
    while True:
        if work.size == 0:
            break
        flat = work.argmax()
        gi, di = divmod(int(flat), work.shape[1])
        if work[gi, di] < iou_thresh:
            break
        pairs.append((gi, di))
        used_g.add(gi)
        used_d.add(di)
        work[gi, :] = -1
        work[:, di] = -1
    return pairs, used_g, used_d


def angle_err_deg(a: float, b: float) -> float:
    d = abs(math.degrees(a) - math.degrees(b)) % 180.0
    return min(d, 180.0 - d)


def pct(vals, p):
    import numpy as np
    return float(np.percentile(vals, p)) if len(vals) else float("nan")


def run(args):
    import numpy as np
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.metrics import batch_probiou
    from ultralytics.utils.ops import xyxyxyxy2xywhr

    model_path = resolve_model(args.model)
    imgsz = resolve_imgsz(model_path, args.imgsz)
    names = load_names()
    gt, sizes = load_gt(LABELS_DIR, IMAGES_DIR, names)

    stems = sorted(gt)
    if args.classes:
        want = set(args.classes)
        stems = [s for s in stems if any(n in want for _, n in gt[s])]
    if args.limit:
        stems = stems[: args.limit]

    print(f"model   : {model_path}")
    print(f"imgsz   : {imgsz}")
    print(f"images  : {len(stems)} val images, "
          f"{sum(len(gt[s]) for s in stems)} ground-truth aircraft")
    print(f"conf    : lowest {args.probe_conf} for the single inference pass, "
          f"swept afterwards: {args.confs}\n")

    model = YOLO(str(model_path))
    paths = [str(IMAGES_DIR / f"{s}.jpg") for s in stems]

    def iou_whr(pts_a, pts_b):
        a = torch.from_numpy(pts_a).float()
        b = torch.from_numpy(pts_b).float()
        return batch_probiou(xyxyxyxy2xywhr(a), xyxyxyxy2xywhr(b)).numpy()

    def blank():
        return {"n_gt": 0, "n_pred": 0, "n_matched": 0, "hit": {0.5: 0, 0.75: 0, 0.9: 0},
                "ious": [], "cdist": [], "size_ratio": [], "ang": [],
                "per_cls": defaultdict(lambda: {"gt": 0, "hit": 0, "ious": []})}

    # Every threshold is scored from the same detections. Inference runs in fixed-size chunks and
    # each chunk's Results are dropped before the next one: a single streaming generator over the
    # whole split still OOMs a 4 GB cgroup (ultralytics keeps per-image state alive), so the chunk
    # size, not the split size, is what bounds resident memory.
    acc = {conf: blank() for conf in args.confs}

    for start in range(0, len(paths), args.chunk):
        chunk_stems = stems[start:start + args.chunk]
        chunk_results = model.predict(source=paths[start:start + args.chunk], imgsz=imgsz,
                                      conf=args.probe_conf, device=args.device,
                                      max_det=args.max_det, batch=args.batch, stream=False,
                                      verbose=False, plots=False)

        for stem, r in zip(chunk_stems, chunk_results):
            g = gt[stem]
            if r.obb is not None and len(r.obb.cls):
                p_pts = r.obb.xyxyxyxy.float().cpu().numpy().astype(np.float32)
                p_conf = r.obb.conf.float().cpu().numpy().astype(np.float32)
            else:
                p_pts, p_conf = np.zeros((0, 8), np.float32), np.zeros(0, np.float32)

            gt_pts = np.stack([b for b, _ in g]) if g else np.zeros((0, 8), np.float32)
            gt_whr = xyxyxyxy2xywhr(torch.from_numpy(gt_pts).float()).numpy() if g else None
            iou_all = iou_whr(gt_pts, p_pts) if (len(g) and len(p_pts)) else None
            w_img, h_img = sizes[stem]

            for conf, a in acc.items():
                keep = p_conf >= conf
                a["n_gt"] += len(g)
                a["n_pred"] += int(keep.sum())
                for _, cn in g:
                    a["per_cls"][cn]["gt"] += 1
                if not g or not keep.any():
                    continue

                iou = iou_all[:, keep]
                pd_whr = xyxyxyxy2xywhr(torch.from_numpy(p_pts[keep]).float()).numpy()
                pairs, _, _ = greedy_match(iou, 0.5)
                a["n_matched"] += len(pairs)

                for gi, cn in enumerate([c for _, c in g]):
                    best = float(iou[gi].max())
                    for t in a["hit"]:
                        if best >= t:
                            a["hit"][t] += 1
                    if best >= 0.5:
                        a["per_cls"][cn]["hit"] += 1
                        a["per_cls"][cn]["ious"].append(best)

                for gi, di in pairs:
                    a["ious"].append(float(iou[gi, di]))
                    a["cdist"].append(float(np.hypot(*(gt_whr[gi][:2] - pd_whr[di][:2])))
                                      / max(w_img, h_img) * 100.0)
                    a["size_ratio"].append(float(np.sqrt(
                        (pd_whr[di][2] * pd_whr[di][3]) / (gt_whr[gi][2] * gt_whr[gi][3]))))
                    a["ang"].append(angle_err_deg(float(gt_whr[gi][4]), float(pd_whr[di][4])))

        del chunk_results
        print(f"  {min(start + args.chunk, len(paths))}/{len(paths)} images", flush=True)

    report = {"model": str(model_path), "imgsz": imgsz, "images": len(stems),
              "gt_total": sum(len(gt[s]) for s in stems), "sweep": []}

    for conf in args.confs:
        a = acc[conf]
        n_gt, n_pred, n_matched = a["n_gt"], a["n_pred"], a["n_matched"]
        ious, cdist, size_ratio, ang = a["ious"], a["cdist"], a["size_ratio"], a["ang"]
        row = {
            "conf": conf,
            "gt": n_gt,
            "pred": n_pred,
            "matched_iou50": n_matched,
            "detect_rate": {str(t): (a["hit"][t] / n_gt if n_gt else 0.0) for t in a["hit"]},
            "spurious": n_pred - n_matched,
            "iou_mean": float(np.mean(ious)) if ious else 0.0,
            "iou_p05": pct(ious, 5),
            "center_err_pct": {
                "mean": float(np.mean(cdist)) if cdist else 0.0,
                "p50": pct(cdist, 50),
                "p95": pct(cdist, 95),
            },
            "size_ratio": {"mean": float(np.mean(size_ratio)) if size_ratio else 0.0,
                           "p05": pct(size_ratio, 5), "p95": pct(size_ratio, 95)},
            "angle_err_deg": {"mean": float(np.mean(ang)) if ang else 0.0,
                              "p95": pct(ang, 95)},
            "per_class": {k: {"gt": v["gt"], "hit": v["hit"],
                             "rate": (v["hit"] / v["gt"] if v["gt"] else 0.0),
                             "iou_mean": float(np.mean(v["ious"])) if v["ious"] else 0.0}
                          for k, v in sorted(a["per_cls"].items())},
        }
        report["sweep"].append(row)

        print(f"conf {conf:<6} gt={n_gt:5d} pred={n_pred:6d} matched={row['matched_iou50']:5d} "
              f"spurious={row['spurious']:6d} | "
              f"det@.5={row['detect_rate']['0.5']:.3f} det@.75={row['detect_rate']['0.75']:.3f} "
              f"det@.9={row['detect_rate']['0.9']:.3f} | IoU {row['iou_mean']:.3f} | "
              f"centre err {row['center_err_pct']['mean']:.2f}% "
              f"(p95 {row['center_err_pct']['p95']:.2f}%) | "
              f"size x{row['size_ratio']['mean']:.3f} | ang {row['angle_err_deg']['mean']:.1f}deg")

    out = BASE_DIR / "runs" / "obb" / "benchmarks" / "class_agnostic.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nsaved {out}")

    print("\nper-class detection (class-agnostic, last conf in sweep):")
    last = report["sweep"][-1]
    print(f"{'class':9s} {'gt':>5s} {'hit':>5s} {'rate':>6s} {'mean_iou':>9s}")
    for k, v in last["per_class"].items():
        print(f"{k:9s} {v['gt']:5d} {v['hit']:5d} {v['rate']:6.3f} {v['iou_mean']:9.3f}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=str, default=None, help="OBB checkpoint (default: newest run)")
    ap.add_argument("--imgsz", type=int, default=None,
                    help="inference size (default: from run's args.yaml, else 640)")
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--confs", type=float, nargs="+", default=[0.25, 0.1, 0.05, 0.01],
                    help="confidence thresholds to sweep in the report")
    ap.add_argument("--probe-conf", type=float, default=0.01,
                    help="conf used for the single inference pass; must be <= min(confs)")
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=200,
                    help="images per inference chunk; bounds RAM, lower it if the box OOMs")
    ap.add_argument("--limit", type=int, default=0, help="only the first N images (0 = all)")
    ap.add_argument("--classes", type=str, nargs="+", default=None,
                    help="restrict to images containing these classes")
    args = ap.parse_args()
    if min(args.confs) < args.probe_conf:
        print("--probe-conf must be <= every value in --confs")
        sys.exit(1)
    run(args)
