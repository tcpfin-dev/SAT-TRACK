"""
Test a trained YOLO-OBB model on MAR20 images.

Works with any YOLO26-OBB checkpoint (YOLO26n-OBB by default).

Usage:
    python test_model.py                           # batch predict on JPEGImages/
    python test_model.py --image 123               # predict on a single image
    python test_model.py --image 123 --show        # predict and display
    python test_model.py --val                     # run ultralytics val mode
    python test_model.py --model runs/obb/mar20_yolo26n_l40s_1024/weights/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "JPEGImages"
DATA_YAML = BASE_DIR / "yolo_obb_dataset" / "mar20_obb.yaml"

# Candidate weights, newest run first.
CANDIDATE_WEIGHTS = [
    BASE_DIR / "runs" / "obb" / "mar20_yolo26s_l40s_1024" / "weights" / "best.pt",
    BASE_DIR / "runs" / "obb" / "mar20_yolo26n_l40s_1024" / "weights" / "best.pt",
]

CLASS_NAMES = [
    "SU-35", "C-130", "C-17", "C-5", "F-16",
    "TU-160", "E-3", "B-52", "P-3C", "B-1B",
    "E-8", "TU-22", "F-15", "KC-135", "F-22",
    "F-A-18", "TU-95", "KC-10", "SU-34", "SU-24",
]


def resolve_model(cli_model: str | None) -> Path:
    if cli_model:
        return Path(cli_model).expanduser().resolve()
    for cand in CANDIDATE_WEIGHTS:
        if cand.exists():
            return cand
    return CANDIDATE_WEIGHTS[0]


def load_model(model_path: Path):
    from ultralytics import YOLO
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        sys.exit(1)
    return YOLO(str(model_path))


def resolve_imgsz(model_path: Path, cli_imgsz: int | None) -> int:
    """Eval at the size the checkpoint was trained at, unless told otherwise.

    The training run dir keeps an args.yaml next to weights/, so a 1024-trained model is
    scored at 1024 by default. Scoring it at 640 understates it badly.
    """
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


def val(model_path: Path, device: int | str, imgsz: int = 640):
    model = load_model(model_path)
    metrics = model.val(data=str(DATA_YAML), split="val", imgsz=imgsz, batch=16, device=device)
    return metrics


def predict_image(model_path: Path, img_path: Path, conf: float = 0.25, show: bool = False,
                  imgsz: int = 640):
    model = load_model(model_path)
    names = model.names or {i: n for i, n in enumerate(CLASS_NAMES)}
    results = model.predict(
        str(img_path),
        imgsz=imgsz,
        conf=conf,
        device=0,
        save=not show,
        show=show,
        project=str(BASE_DIR / "runs" / "obb" / "predict"),
        name="test",
        exist_ok=True,
    )
    for r in results:
        if r.obb is not None:
            for i in range(len(r.obb.cls)):
                cls_id = int(r.obb.cls[i].item())
                conf_val = float(r.obb.conf[i].item())
                name = names[cls_id] if isinstance(names, dict) else names[cls_id]
                x1, y1, x2, y2, x3, y3, x4, y4 = r.obb.xyxyxyxy[i].tolist()
                print(f"  {name}: {conf_val:.2%}  box=[({x1:.0f},{y1:.0f}) ({x2:.0f},{y2:.0f}) "
                      f"({x3:.0f},{y3:.0f}) ({x4:.0f},{y4:.0f})]")


def predict_batch(model_path: Path, directory: Path, conf: float = 0.25, imgsz: int = 640):
    model = load_model(model_path)

    images = sorted(directory.glob("*.jpg"))
    if not images:
        print(f"No .jpg files in {directory}")
        return

    results = model.predict(
        source=str(directory),
        imgsz=imgsz,
        conf=conf,
        device=0,
        save_txt=True,
        save_conf=True,
        project=str(BASE_DIR / "runs" / "obb" / "predict"),
        name="batch",
        exist_ok=True,
    )

    total_dets = 0
    for r in results:
        if r.obb is not None:
            total_dets += len(r.obb.cls)
    print(f"\n{len(images)} images processed, {total_dets} detections")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        help="Path to OBB checkpoint (default: newest run found)")
    parser.add_argument("--device", type=str, default="0", help="'0' for GPU, 'cpu' otherwise")
    parser.add_argument("--val", action="store_true", help="Run validation on test set")
    parser.add_argument("--image", type=str, help="Image ID (no extension) to predict")
    parser.add_argument("--dir", type=str, help="Directory of images to predict")
    parser.add_argument("--show", action="store_true", help="Show image with predictions")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="Inference size (default: read from the run's args.yaml, else 640)")
    args = parser.parse_args()

    device: int | str = int(args.device) if args.device.isdigit() else args.device
    model_path = resolve_model(args.model)
    imgsz = resolve_imgsz(model_path, args.imgsz)
    print(f"Using model: {model_path}")
    print(f"Image size: {imgsz}")

    if args.val:
        val(model_path, device, imgsz=imgsz)
    elif args.image:
        img_path = IMAGES_DIR / f"{args.image}.jpg"
        if not img_path.exists():
            print(f"Image not found: {img_path}")
            sys.exit(1)
        predict_image(model_path, img_path, conf=args.conf, show=args.show, imgsz=imgsz)
    elif args.dir:
        predict_batch(model_path, Path(args.dir), conf=args.conf, imgsz=imgsz)
    else:
        predict_batch(model_path, IMAGES_DIR, conf=args.conf, imgsz=imgsz)
