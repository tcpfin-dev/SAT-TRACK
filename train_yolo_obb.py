"""
Train YOLO11-OBB on the MAR20 dataset.

Converts PASCAL VOC oriented bounding box annotations to YOLO OBB format,
sets up the dataset structure, and trains YOLO11-OBB.

Usage:
    pip install ultralytics
    python train_yolo_obb.py
"""

import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ANNOTATIONS_DIR = BASE_DIR / "Annotations" / "Oriented Bounding Boxes"
IMAGES_DIR = BASE_DIR / "JPEGImages"
IMAGESETS_DIR = BASE_DIR / "ImageSets" / "Main"
DATASET_DIR = BASE_DIR / "yolo_obb_dataset"

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


def setup_dataset() -> None:
    """Build YOLO OBB dataset directory structure from MAR20."""
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


def create_yaml() -> Path:
    yaml_path = DATASET_DIR / "mar20_obb.yaml"
    content = f"""# MAR20 YOLO OBB dataset config
path: {DATASET_DIR}
train: images/train
val: images/val

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
    with open(yaml_path, "w") as f:
        f.write(content)
    return yaml_path


def train() -> None:
    from ultralytics import YOLO

    yaml_path = create_yaml()

    model = YOLO("yolo11n-obb.pt")
    results = model.train(
        data=str(yaml_path),
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        patience=20,
        save=True,
        save_period=10,
        project=str(BASE_DIR / "runs" / "obb"),
        name="mar20_train",
        exist_ok=True,
    )
    print(f"\nTraining complete. Best model: {results.save_dir}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--setup-only":
        setup_dataset()
        create_yaml()
        print("Setup done. Ready for training.")
    else:
        print("Setting up MAR20 YOLO OBB dataset...")
        setup_dataset()
        print("\nStarting YOLO11-OBB training...")
        train()
