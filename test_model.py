"""
Test trained YOLO11-OBB model on MAR20 images.
Usage:
    python test_model.py                           # run on test set, save results
    python test_model.py --image 123               # predict on single image
    python test_model.py --image 123 --show        # predict and display
    python test_model.py --val                     # run ultralytics val mode
"""

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "JPEGImages"
MODEL_PATH = BASE_DIR / "runs" / "obb" / "mar20_train" / "weights" / "best.pt"

CLASS_NAMES = [
    "SU-35", "C-130", "C-17", "C-5", "F-16",
    "TU-160", "E-3", "B-52", "P-3C", "B-1B",
    "E-8", "TU-22", "F-15", "KC-135", "F-22",
    "F-A-18", "TU-95", "KC-10", "SU-34", "SU-24",
]


def val():
    from ultralytics import YOLO
    yaml_path = BASE_DIR / "yolo_obb_dataset" / "mar20_obb.yaml"
    model = YOLO(str(MODEL_PATH))
    model.val(data=str(yaml_path), split="val", imgsz=640, batch=16, device=0)


def predict_image(img_path: Path, conf: float = 0.25, show: bool = False):
    from ultralytics import YOLO
    model = YOLO(str(MODEL_PATH))
    results = model.predict(
        str(img_path),
        imgsz=640,
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
                name = CLASS_NAMES[cls_id]
                x1, y1, x2, y2, x3, y3, x4, y4 = r.obb.xyxyxyxy[i].tolist()
                print(f"  {name}: {conf_val:.2%}  box=[({x1:.0f},{y1:.0f}) ({x2:.0f},{y2:.0f}) ({x3:.0f},{y3:.0f}) ({x4:.0f},{y4:.0f})]")


def predict_batch(directory: Path, conf: float = 0.25):
    from ultralytics import YOLO

    images = sorted(directory.glob("*.jpg"))
    if not images:
        print(f"No .jpg files in {directory}")
        return

    model = YOLO(str(MODEL_PATH))
    results = model.predict(
        source=str(directory),
        imgsz=640,
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
    parser.add_argument("--val", action="store_true", help="Run validation on test set")
    parser.add_argument("--image", type=str, help="Image ID (no extension) to predict")
    parser.add_argument("--dir", type=str, help="Directory of images to predict")
    parser.add_argument("--show", action="store_true", help="Show image with predictions")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        print(f"Model not found: {MODEL_PATH}")
        sys.exit(1)

    if args.val:
        val()
    elif args.image:
        img_path = IMAGES_DIR / f"{args.image}.jpg"
        if not img_path.exists():
            print(f"Image not found: {img_path}")
            sys.exit(1)
        predict_image(img_path, conf=args.conf, show=args.show)
    elif args.dir:
        predict_batch(Path(args.dir), conf=args.conf)
    else:
        predict_batch(IMAGES_DIR, conf=args.conf)
