"""
Train a YOLO oriented-bounding-box (OBB) detector on the MAR20 dataset.

Defaults to **YOLO26** (``yolo26n-obb.pt``) and targets only the YOLO26 OBB family. The script converts
PASCAL-VOC OBB annotations to YOLO OBB format, (re)generates the dataset yaml with a path that is valid
on *this* machine, and trains.

``--model`` accepts three forms, and the checkpoint is fetched automatically when it is not on disk:

    "yolo26n-obb.pt"          official Ultralytics asset name -> downloaded into ``weights/``
    "https://.../some.pt"     downloaded straight from that URL
    "weights/best.pt"         any local file, used as-is

Usage:
    pip install ultralytics
    python train_yolo_obb.py                        # YOLO26n-OBB, 100 epochs
    python train_yolo_obb.py --model yolo26s-obb.pt --epochs 200
    python train_yolo_obb.py --setup-only           # dataset only, no training
    python train_yolo_obb.py --no-setup             # train on existing dataset
    python train_yolo_obb.py --model yolo26m-obb.pt --weights-dir /data/weights
    python train_yolo_obb.py --resume --batch 16    # continue the newest run's last.pt

--resume restores the optimizer, scaler, EMA, best fitness and early-stopping state from
the checkpoint, so it continues the run instead of restarting it. Ultralytics only lets a resume
change a small set of arguments (batch, imgsz, device, workers, cache, patience, close_mosaic,
save_period) - that set exists precisely so a crashed run can be resumed with a smaller batch.
Anything else on the command line is ignored by ultralytics, and so is anything you do not pass.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote, urlparse

BASE_DIR = Path(__file__).resolve().parent
ANNOTATIONS_DIR = BASE_DIR / "Annotations" / "Oriented Bounding Boxes"
IMAGES_DIR = BASE_DIR / "JPEGImages"
IMAGESETS_DIR = BASE_DIR / "ImageSets" / "Main"
DATASET_DIR = BASE_DIR / "yolo_obb_dataset"
RUNS_DIR = BASE_DIR / "runs" / "obb"
WEIGHTS_DIR = BASE_DIR / "weights"

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


MIN_PT_BYTES = 100_000  # anything smaller is a git-lfs pointer / truncated download


def _checkpoint_candidates(src: str, weights_dir: Path = WEIGHTS_DIR) -> list[Path]:
    """Local places a bare model name may already live, in search order."""
    raw = Path(src).expanduser()
    candidates = [raw, BASE_DIR / raw.name, Path(weights_dir) / raw.name]
    return [p.resolve() for p in candidates]


def resolve_model(src: str, weights_dir: Path = WEIGHTS_DIR, download: bool = True) -> Path:
    """Resolve a model spec to a local ``.pt``, fetching the checkpoint when it is missing.

    Accepts an official Ultralytics asset name (``yolo26n-obb.pt``), a direct URL, or a
    local path. Returns the verified local path.
    """
    from ultralytics import YOLO
    from ultralytics.utils.downloads import attempt_download_asset, safe_download

    src = str(src).strip()
    weights_dir = Path(weights_dir).expanduser().resolve()
    # pathlib mangles "://" into ":/", so repair that form before anything else
    if src.lower().startswith(("http:/", "https:/")) and "://" not in src:
        src = src.replace(":/", "://", 1)
    is_url = src.lower().startswith(("http://", "https://"))

    if is_url:
        name = unquote(urlparse(src).path).rsplit("/", 1)[-1] or "model.pt"
        target = weights_dir / name
        if target.is_file() and target.stat().st_size >= MIN_PT_BYTES:
            print(f"Using cached checkpoint: {target}")
        else:
            if not download:
                raise FileNotFoundError(f"{target} not found and --no-download was given")
            weights_dir.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {src}\n  -> {target}")
            safe_download(url=src, file=str(target), min_bytes=1e5)
        model_path = target
    else:
        existing = next(
            (p for p in _checkpoint_candidates(src, weights_dir) if p.is_file()), None
        )
        if existing and existing.stat().st_size >= MIN_PT_BYTES:
            model_path = existing
            print(f"Using local checkpoint: {model_path}")
        else:
            name = Path(src).name
            if not name.endswith(".pt"):
                name = f"{name}-obb.pt"
            target = weights_dir / name
            if target.is_file() and target.stat().st_size >= MIN_PT_BYTES:
                model_path = target
                print(f"Using cached checkpoint: {model_path}")
            else:
                if not download:
                    raise FileNotFoundError(
                        f"No checkpoint '{name}' found in {BASE_DIR} or {weights_dir}, "
                        "and --no-download was given."
                    )
                weights_dir.mkdir(parents=True, exist_ok=True)
                print(f"Fetching Ultralytics asset '{name}' -> {target}")
                # attempt_download_asset() returns quietly when the name is not a known
                # asset, so the size check below is what actually proves the fetch worked.
                attempt_download_asset(str(target))
                model_path = target

    if not model_path.is_file() or model_path.stat().st_size < MIN_PT_BYTES:
        raise FileNotFoundError(
            f"Checkpoint missing or truncated: {model_path}. Check the name against "
            "https://github.com/ultralytics/assets/releases (it must be a *-obb.pt file), "
            "or pass a direct URL with --model."
        )

    probe = YOLO(str(model_path))
    if getattr(probe, "task", None) not in (None, "obb"):
        raise ValueError(
            f"{model_path.name} is a '{probe.task}' model - training needs an OBB checkpoint"
        )
    params = sum(p.numel() for p in getattr(probe.model, "parameters", lambda: [])())
    sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    print(f"  task:   {probe.task}")
    print(f"  params: {params / 1e6:.2f}M")
    print(f"  size:   {model_path.stat().st_size / 1e6:.1f} MB")
    print(f"  sha256: {sha}")
    return model_path


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
    weights_dir: Path = WEIGHTS_DIR,
    download: bool = True,
):
    from ultralytics import YOLO

    yaml_path = create_yaml()

    model_path = resolve_model(model_name, weights_dir=weights_dir, download=download)
    model = YOLO(str(model_path))
    run_name = name or default_run_name(model_path.name)
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


def resolve_resume_ckpt(spec: str) -> Path:
    """Resolve ``--resume`` to a checkpoint that still holds optimizer state.

    ``latest`` (also ``true``/empty) picks the most recently written ``runs/obb/*/weights/last.pt``.
    A path is used as-is, relative paths are resolved against the script directory.
    """
    if str(spec).strip().lower() in {"latest", "true", "1", ""}:
        candidates = [p for p in RUNS_DIR.glob("*/weights/last.pt") if p.is_file()]
        if not candidates:
            raise FileNotFoundError(f"--resume latest: no runs under {RUNS_DIR}")
        newest = max(candidates, key=lambda p: p.stat().st_mtime)
        print(f"--resume latest -> {newest}")
        return newest

    path = Path(spec).expanduser()
    if not path.is_absolute():
        path = (BASE_DIR / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    return path


NBS = 64  # ultralytics nominal batch; any batch dividing it keeps the effective batch AND weight decay identical


def resume_plan(ckpt_path: Path) -> dict:
    """Print what a resume will actually do, before ultralytics is allowed to start.

    On resume the checkpoint - not the command line - owns data/project/name/epochs/batch. So a box
    whose absolute paths differ from the one that trained does not continue the run: ultralytics
    recreates the recorded save_dir, finds no results.csv, and starts a fresh one there. Check the
    paths here, while it is still a message instead of an hour of wasted compute.
    """
    import torch

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ta = dict(ckpt.get("train_args") or {})
    done = int(ckpt.get("epoch", -1)) + 1  # csv epoch column is 1-based, the checkpoint stores 0-based
    next_epoch = done + 1
    total = int(ta.get("epochs") or 0)
    patience = int(ta.get("patience") or 0)
    stopper = dict(ckpt.get("stopper") or {})
    best_epoch = stopper.get("best_epoch")
    best = ckpt.get("best_fitness", stopper.get("best_fitness"))
    data = ta.get("data")
    save_dir = Path(str(ta.get("save_dir") or Path(str(ta.get("project", "runs"))) / str(ta.get("name", "train"))))
    batch = int(ta.get("batch") or 0)
    nbs = int(ta.get("nbs") or NBS)

    print("\n--- resume plan -------------------------------------------------")
    print(f"checkpoint     : {ckpt_path}")
    print(f"epochs done    : {done} checkpointed of {total}  ->  reruns epoch {next_epoch} "
          f"(the interrupted epoch was never saved)")
    print(f"best fitness   : {best} at epoch {best_epoch}")
    print(f"trained with   : model={ta.get('model')} batch={batch} imgsz={ta.get('imgsz')} "
          f"optimizer={ta.get('optimizer')} nbs={nbs}")
    print(f"data (baked)   : {data}")
    print(f"save_dir(baked): {save_dir}")
    if patience:
        stop_at = (best_epoch or done) + patience
        print(f"early stopping : patience {patience} counts from best epoch {best_epoch} "
              f"-> stops at epoch {min(stop_at, total)} unless fitness beats {best} (patience 0 = never)")
    if batch and nbs and batch > nbs:
        print(f"[warn] batch {batch} > nbs {nbs}: accumulation collapses to 1 and weight decay is rescaled")
    elif batch and nbs and nbs % batch:
        print(f"[warn] batch {batch} does not divide nbs {nbs}: effective batch "
              f"{batch * max(round(nbs / batch), 1)} and weight decay x{batch * max(round(nbs / batch), 1) / nbs:.3f}")
    if ckpt.get("optimizer") is None:
        print(f"[warn] {ckpt_path.name} carries no optimizer/scaler state (a stripped best.pt?) - "
              f"resuming from it restarts AdamW moments, so the LR ramp will lurch. Prefer last.pt.")
    if not data or not Path(str(data)).is_file():
        print(f"[warn] dataset yaml NOT found at the baked path - this box cannot continue the run as-is. "
              f"Mirror the checkout (and yolo_obb_dataset/) at that exact path, or edit the dataset path in a copy.")
    if not save_dir.exists():
        print(f"[warn] run dir {save_dir} does not exist on this box: ultralytics will CREATE it and start "
              f"a fresh results.csv there, losing the {done}-epoch history. Mirror the run dir "
              f"(args.yaml + results.csv + weights/last.pt) at that exact path.")
    else:
        csv_path = save_dir / "results.csv"
        rows = len(csv_path.read_text().splitlines()) - 1 if csv_path.is_file() else 0
        print(f"run dir        : exists, results.csv has {rows} epoch rows")
    print("-----------------------------------------------------------------\n")
    return {"done": done, "next_epoch": next_epoch, "total": total, "best": best,
            "best_epoch": best_epoch, "data": data, "save_dir": save_dir}


def resume_train(ckpt_path: Path, explicit: set[str], args: argparse.Namespace) -> Path:
    """Continue an interrupted run from ``ckpt_path``.

    Only arguments the user actually typed are forwarded: everything else keeps the value
    recorded in the checkpoint, which is what makes this a resume and not a restart. Ultralytics
    silently clamps what it accepts on resume to batch/imgsz/device/workers/cache/patience/
    close_mosaic/save_period, so passing anything more would just produce warnings.
    """
    from ultralytics import YOLO

    kwargs: dict = {"resume": str(ckpt_path)}
    forward = {
        "--batch": "batch",
        "--imgsz": "imgsz",
        "--workers": "workers",
        "--patience": "patience",
        "--close-mosaic": "close_mosaic",
    }
    for flag, key in forward.items():
        if flag in explicit:
            kwargs[key] = getattr(args, key)
    if "--device" in explicit:
        device = args.device
        kwargs["device"] = int(device) if str(device).isdigit() else device
    if "--cache" in explicit and args.cache != "false":
        kwargs["cache"] = args.cache

    print(f"Resuming from {ckpt_path}")
    print("resume overrides: " + ", ".join(f"{k}={v}" for k, v in kwargs.items() if k != "resume"))
    resume_plan(ckpt_path)

    model = YOLO(str(ckpt_path))
    model.train(**kwargs)

    save_dir = Path(str(getattr(model.trainer, "save_dir", ckpt_path.parents[1])))
    print(f"\nResume complete. Run dir: {save_dir}")
    print(f"Best weights: {save_dir / 'weights' / 'best.pt'}")
    return save_dir


def _explicit_flags(argv: list[str] | None) -> set[str]:
    """Flags genuinely present on the command line - argparse defaults hide that distinction."""
    tokens = list(sys.argv[1:] if argv is None else argv)
    return {t.split("=", 1)[0] for t in tokens if t.startswith("--")}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLO OBB on MAR20 (YOLO26 by default).")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"Ultralytics YOLO26 OBB checkpoint: asset name, URL, or local .pt "
                        f"(default: {DEFAULT_MODEL}; downloaded automatically if not found)")
    p.add_argument("--weights-dir", default=str(WEIGHTS_DIR), type=Path,
                   help=f"Where fetched checkpoints are stored (default: {WEIGHTS_DIR})")
    p.add_argument("--no-download", action="store_true",
                   help="Never fetch the checkpoint; fail if it is not already on disk")
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
    p.add_argument("--resume", nargs="?", const="latest", default=None, metavar="CKPT",
                   help="Resume an interrupted run from its weights/last.pt (a full epochN.pt also "
                        "works). Pass --resume alone for the newest run. Only batch/imgsz/device/"
                        "workers/cache/patience/close-mosaic are taken from the command line on "
                        "resume; everything else keeps the values recorded in the checkpoint")

    # --- pass-through training options (only applied when given, so defaults stay untouched) ---
    t = p.add_argument_group("training options (passed straight to Ultralytics)")
    t.add_argument("--cache", default="false", choices=["false", "ram", "disk"],
                   help="Cache images: 'ram' for datasets that fit in memory (recommended on cloud)")
    t.add_argument("--cos-lr", action="store_true",
                   help="Enable cosine LR schedule (uses the full --epochs span)")
    t.add_argument("--degrees", type=float, default=None,
                   help="Rotation augmentation range in degrees (overhead imagery wants ~180)")
    t.add_argument("--close-mosaic", type=int, default=None,
                   help="Disable mosaic for the last N epochs (Ultralytics default: 10)")
    t.add_argument("--lr0", type=float, default=None, help="Initial learning rate")
    t.add_argument("--optimizer", default=None, help="auto|SGD|Adam|AdamW|NAdam|RAdam|RMSProp")
    t.add_argument("--seed", type=int, default=None, help="Global RNG seed")
    t.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    t.add_argument("--rect", action="store_true", help="Rectangular training batches")
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

    if args.resume:
        explicit = _explicit_flags(argv)
        for flag in ("--model", "--name"):
            if flag in explicit:
                print(f"[warn] --resume ignores {flag}: the checkpoint and its run dir win")
        resume_train(resolve_resume_ckpt(args.resume), explicit, args)
        return

    print(f"\nStarting {args.model} training...")

    # Only forward options the user actually set. Keys that train() already passes explicitly
    # (data/epochs/imgsz/batch/device/patience/name/fraction/workers/project/exist_ok) must NOT
    # appear here or Python raises "got multiple values for keyword argument".
    extra: dict = {}
    if args.cache != "false":
        extra["cache"] = args.cache
    if args.cos_lr:
        extra["cos_lr"] = True
    if args.degrees is not None:
        extra["degrees"] = args.degrees
    if args.close_mosaic is not None:
        extra["close_mosaic"] = args.close_mosaic
    if args.lr0 is not None:
        extra["lr0"] = args.lr0
    if args.optimizer is not None:
        extra["optimizer"] = args.optimizer
    if args.seed is not None:
        extra["seed"] = args.seed
    if args.no_amp:
        extra["amp"] = False
    if args.rect:
        extra["rect"] = True
    if extra:
        print("extra training args: " + ", ".join(f"{k}={v}" for k, v in extra.items()))

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
        weights_dir=args.weights_dir,
        download=not args.no_download,
        extra=extra or None,
    )


if __name__ == "__main__":
    main()
