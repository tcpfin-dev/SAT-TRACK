"""
Train a YOLO oriented-bounding-box (OBB) detector on the MAR20 dataset.

Defaults to **YOLO26** (``yolo26n-obb.pt``) and targets only the YOLO26 OBB family. The script converts
PASCAL-VOC OBB annotations to YOLO OBB format, (re)generates the dataset yaml with a path that is valid
on *this* machine, and trains.

Usage:
    pip install ultralytics
    python train_yolo_obb.py                        # YOLO26n-OBB, 100 epochs
    python train_yolo_obb.py --model yolo26s-obb.pt --epochs 200
    python train_yolo_obb.py --setup-only           # dataset only, no training
    python train_yolo_obb.py --no-setup             # train on existing dataset
"""

from __future__ import annotations

import argparse
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ANNOTATIONS_DIR = BASE_DIR / "Annotations" / "Oriented Bounding Boxes"
IMAGES_DIR = BASE_DIR / "JPEGImages"
IMAGESETS_DIR = BASE_DIR / "ImageSets" / "Main"
DATASET_DIR = BASE_DIR / "yolo_obb_dataset"
RUNS_DIR = BASE_DIR / "runs" / "obb"

DEFAULT_MODEL = "yolo26n-obb.pt"

CLASS_NAMES = [
    "SU-35", "C-130", "C-17", "C-5", "F-16",
    "TU-160", "E-3", "B-52", "P-3C", "B-1B",
    "E-8", "TU-22", "F-15", "KC-135", "F-22",
    "F-A-18", "TU-95", "KC-10", "SU-34", "SU-24",
]

CLASS_MAP = {f"A{i}": i - 1 for i in range(1, 21)}


def load_split(split_name: str) -> set[str]:
    path = IMAGESETS_DIR / f"{split_name}.txt"
    if not path.exists():
        return set()
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def convert_xml_to_obb(xml_path: Path, img_width: int, img_height: int) -> list[str]:
    """Convert one PASCAL VOC OBB XML to list of YOLO OBB label lines (normalized)."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    lines = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        if name not in CLASS_MAP:
            continue
        cls_id = CLASS_MAP[name]

        rb = obj.find("robndbox")
        x1 = float(rb.find("x_left_top").text) / img_width
        y1 = float(rb.find("y_left_top").text) / img_height
        x2 = float(rb.find("x_right_top").text) / img_width
        y2 = float(rb.find("y_right_top").text) / img_height
        x3 = float(rb.find("x_right_bottom").text) / img_width
        y3 = float(rb.find("y_right_bottom").text) / img_height
        x4 = float(rb.find("x_left_bottom").text) / img_width
        y4 = float(rb.find("y_left_bottom").text) / img_height

        lines.append(f"{cls_id} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f}")

    return lines


def get_image_size(xml_path: Path, img_path: Path) -> tuple[int, int]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)
    if w > 0 and h > 0:
        return w, h
    # Some XMLs have 0×0 dimensions — fall back to actual image
    from PIL import Image
    with Image.open(img_path) as im:
        return im.size


def raw_dataset_available() -> bool:
    """True when the original MAR20 VOC-style tree is present and non-empty."""
    return any(ANNOTATIONS_DIR.glob("*.xml")) and any(IMAGES_DIR.glob("*.jpg"))


def dataset_available() -> bool:
    """True when the converted YOLO OBB dataset is already on disk."""
    return any((DATASET_DIR / "images" / "train").glob("*.jpg"))


def setup_dataset() -> None:
    """Build YOLO OBB dataset directory structure from MAR20."""
    if not raw_dataset_available():
        raise FileNotFoundError(
            f"Raw MAR20 not found ({ANNOTATIONS_DIR} / {IMAGES_DIR}). "
            "Drop the original MAR20 tree next to this script, or point "
            "--dataset at a pre-converted yolo_obb_dataset."
        )

    train_ids = load_split("train")
    val_ids = load_split("test")

    for split, ids in [("train", train_ids), ("val", val_ids)]:
        img_dir = DATASET_DIR / "images" / split
        lbl_dir = DATASET_DIR / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_id in sorted(ids):
            xml_path = ANNOTATIONS_DIR / f"{img_id}.xml"
            img_src = IMAGES_DIR / f"{img_id}.jpg"

            if not xml_path.exists() or not img_src.exists():
                print(f"  SKIP: {img_id} — missing XML or image")
                continue

            width, height = get_image_size(xml_path, img_src)
            obb_lines = convert_xml_to_obb(xml_path, width, height)

            shutil.copy2(img_src, img_dir / f"{img_id}.jpg")
            with open(lbl_dir / f"{img_id}.txt", "w") as f:
                f.write("\n".join(obb_lines) + ("\n" if obb_lines else ""))

    print(f"\nDataset ready: {DATASET_DIR}")
    print(f"  Train images: {len(list((DATASET_DIR / 'images' / 'train').glob('*.jpg')))}")
    print(f"  Val images:   {len(list((DATASET_DIR / 'images' / 'val').glob('*.jpg')))}")


def create_yaml(force: bool = True) -> Path:
    """(Re)write the dataset yaml so its ``path`` is correct for this checkout."""
    yaml_path = DATASET_DIR / "mar20_obb.yaml"
    if yaml_path.exists() and not force:
        return yaml_path

    # POSIX-style path works on Linux/WSL; ultralytics accepts str paths as-is.
    dataset_path = Path(os.path.abspath(DATASET_DIR)).as_posix()
    content = f"""# MAR20 YOLO OBB dataset config (auto-generated — safe to regenerate)
path: {dataset_path}
train: images/train
val: images/val

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        f.write(content)
    return yaml_path


def default_run_name(model_name: str) -> str:
    """e.g. yolo26n-obb.pt -> mar20_yolo26n_train"""
    stem = Path(model_name).stem.replace("-obb", "")  # yolo26n
    return f"mar20_{stem}_train"


def train(
    model_name: str = DEFAULT_MODEL,
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str | int | None = 0,
    patience: int = 20,
    name: str | None = None,
    project: Path = RUNS_DIR,
    fraction: float = 1.0,
    workers: int = 4,
    extra: dict | None = None,
):
    from ultralytics import YOLO

    yaml_path = create_yaml()

    model = YOLO(model_name)
    run_name = name or default_run_name(model_name)
    model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        patience=patience,
        save=True,
        save_period=10 if epochs >= 10 else 0,
        fraction=fraction,
        workers=workers,
        project=str(project),
        name=run_name,
        exist_ok=True,
        **(extra or {}),
    )
    # ultralytics sets .trainer during train(); prefer it over the results object
    save_dir = Path(str(getattr(model.trainer, "save_dir", project / run_name)))
    print(f"\nTraining complete. Run dir: {save_dir}")
    print(f"Best weights: {save_dir / 'weights' / 'best.pt'}")
    return save_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLO OBB on MAR20 (YOLO26 by default).")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"Ultralytics YOLO26 OBB checkpoint (default: {DEFAULT_MODEL}).")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0", help="'0'..'N' GPU, 'cpu', or 'mps'")
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--name", default=None, help="Run name (default: derived from --model)")
    p.add_argument("--fraction", type=float, default=1.0, help="Fraction of training data")
    p.add_argument("--workers", type=int, default=4,
                   help="Dataloader workers (keep low on WSL/low-RAM boxes to avoid OOM)")
    p.add_argument("--setup-only", action="store_true", help="Only build the dataset + yaml")
    p.add_argument("--no-setup", action="store_true", help="Skip dataset conversion")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = args.device
    if isinstance(device, str) and device.isdigit():
        device = int(device)

    if args.setup_only:
        setup_dataset()
        create_yaml()
        print("Setup done. Ready for training.")
        return

    if args.no_setup:
        if not dataset_available():
            raise FileNotFoundError(
                f"--no-setup was given but no converted dataset exists at {DATASET_DIR}. "
                "Run without --no-setup to convert the raw MAR20 tree."
            )
        print(f"--no-setup: using existing dataset at {DATASET_DIR}")
    elif raw_dataset_available():
        print("Setting up MAR20 YOLO OBB dataset...")
        setup_dataset()
    elif dataset_available():
        print(f"Raw MAR20 not found — using existing dataset at {DATASET_DIR}")
    else:
        raise FileNotFoundError(
            "No raw MAR20 tree and no converted dataset found. "
            "Run with --setup-only after placing the MAR20 files next to this script."
        )

    create_yaml()

    print(f"\nStarting {args.model} training...")
    train(
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        patience=args.patience,
        name=args.name,
        fraction=args.fraction,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
