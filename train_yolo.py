"""
Fine-tune a **YOLOv8** detector (Ultralytics). Training runs forward and backward passes inside
``model.train()``. Use ``--experiment-id`` to separate runs when comparing preprocessing / splits / subsets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    p = argparse.ArgumentParser(description="Fine-tune YOLOv8 on Manga109 YOLO export")
    p.add_argument(
        "--data",
        default="data/processed/manga109_yolo/dataset.yaml",
        help="Path to dataset yaml from convert_manga109_to_yolo.py",
    )
    p.add_argument("--weights", default="yolov8s.pt", help="Checkpoint to start from")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument(
        "--device",
        default=None,
        help="e.g. 0 / 0,1 / cpu (omit to let Ultralytics auto-pick GPU if available)",
    )
    # Ultralytics stores under ``{project}/{task}/{name}/`` (e.g. ``runs/detect/<name>/``).
    p.add_argument("--project", default="runs/detect")
    p.add_argument("--name", default="manga109")
    p.add_argument("--exist-ok", action="store_true", help="Allow overwriting existing run folder")
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Dataloader workers; use 0 on some macOS setups to avoid multiprocessing errors",
    )
    args = p.parse_args()

    if not Path(args.data).is_file():
        raise SystemExit(f"dataset yaml missing: {args.data} — run convert_manga109_to_yolo.py first")

    kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        flipud=0.0,
        fliplr=0.5,
        degrees=5.0,
        mosaic=0.8,
        workers=args.workers,
    )
    if args.device is not None:
        kwargs["device"] = args.device

    model = YOLO(args.weights)
    model.train(**kwargs)


if __name__ == "__main__":
    main()
