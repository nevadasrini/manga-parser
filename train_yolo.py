"""
Fine-tune a **YOLOv8** detector (Ultralytics). Training runs forward and backward passes inside
``model.train()``. Use distinct ``--name`` values when comparing preprocessing / backbone / augmentation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from yolo_training_utils import (
    attach_epoch_logging_callback,
    default_dataloader_workers,
    pick_yolo_device_for_mac_or_gpu,
    print_ultralytics_run_summary,
)


def normalize_ultralytics_project_name(project: str, name: str) -> tuple[str, str]:
    """
    Ultralytics 8 builds ``save_dir`` as::

        SETTINGS["runs_dir"] / task / project / name   (e.g. runs/detect/<project>/<name>/)

    so ``project`` must **not** be ``runs`` or ``runs/detect`` — use ``project=""`` for a plain
    ``runs/detect/<name>/weights/best.pt`` tree. Strip legacy ``detect/`` prefixes from ``name``.
    """
    p, n = (project or "").strip(), name.strip()
    if p in ("runs/detect", "runs"):
        p = ""
    if n.startswith("detect/"):
        n = n.split("/", 1)[1]
    return p or "", n


def main() -> None:
    p = argparse.ArgumentParser(description="Fine-tune YOLOv8 on Manga109 YOLO export")
    p.add_argument(
        "--data",
        default="data/processed/manga109_yolo/dataset.yaml",
        help="Path to dataset yaml from convert_manga109_to_yolo.py",
    )
    p.add_argument("--weights", default="yolov8s.pt", help="Checkpoint to start from")
    p.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs (raise for full comparisons; smoke runs can use --epochs 1)",
    )
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument(
        "--device",
        default=None,
        help=(
            "Training device: omit or 'auto' — Apple Silicon prefers mps (Metal), "
            "else CUDA 0 if available, else cpu. Explicit: mps, cpu, 0."
        ),
    )
    p.add_argument(
        "--project",
        default="",
        help='Subdirectory under runs/detect/ (default "" → runs/detect/<--name>/).',
    )
    p.add_argument("--name", default="manga109")
    p.add_argument("--exist-ok", action="store_true", help="Allow overwriting existing run folder")
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Dataloader workers; default is 0 on macOS else 4 (override if needed)",
    )
    p.add_argument(
        "--mosaic",
        type=float,
        default=0.8,
        help="Ultralytics mosaic augmentation probability (dense manga pages often benefit from higher values)",
    )
    p.add_argument(
        "--tensorboard",
        action="store_true",
        help="Enable TensorBoard logging under the run directory (if supported by your Ultralytics build)",
    )
    p.add_argument(
        "--epoch-log",
        type=Path,
        default=None,
        help="Append one line of validation metrics per epoch to this file (also printed)",
    )
    args = p.parse_args()

    if not Path(args.data).is_file():
        raise SystemExit(f"dataset yaml missing: {args.data} — run convert_manga109_to_yolo.py first")

    dev, dev_tag = pick_yolo_device_for_mac_or_gpu(args.device)
    work = default_dataloader_workers(args.workers)
    proj, run_name = normalize_ultralytics_project_name(args.project, args.name)
    if (proj, run_name) != (args.project.strip(), args.name.strip()):
        print(f"[train_yolo] normalized project/name: {args.project!r}/{args.name!r} → {proj!r}/{run_name!r}")
    kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=proj,
        name=run_name,
        exist_ok=args.exist_ok,
        flipud=0.0,
        fliplr=0.5,
        degrees=5.0,
        mosaic=args.mosaic,
        workers=work,
        device=dev,
    )
    print(f"[train_yolo] device={dev} (tag={dev_tag})  workers={work}")
    if args.tensorboard:
        kwargs["tensorboard"] = True

    model = YOLO(args.weights)
    if args.epoch_log is not None:
        attach_epoch_logging_callback(model, log_path=args.epoch_log)
    try:
        model.train(**kwargs)
    except TypeError as e:
        if args.tensorboard and "tensorboard" in str(e).lower():
            kwargs.pop("tensorboard", None)
            print("[warn] tensorboard= not supported by this Ultralytics build; retrying without it.")
            model.train(**kwargs)
        else:
            raise

    save_dir = getattr(getattr(model, "trainer", None), "save_dir", None)
    if save_dir:
        print_ultralytics_run_summary(Path(save_dir))
    else:
        print("\n(training finished; trainer.save_dir unavailable — check runs/detect/ for your --name)")


if __name__ == "__main__":
    main()
