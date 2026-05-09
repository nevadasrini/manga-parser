#!/usr/bin/env python3
"""
Evaluate panel (YOLO class **frame**) detections vs Manga109 **<frame>** ground truth.

Usage:

    python evaluate_yolo_panels.py \\
        --weights runs/detect/runs/manga109_smoke/weights/best.pt \\
        --yolo-root data/processed/manga109_yolo_smoke \\
        --split val \\
        --data-root data/raw/manga109/Manga109_released_2023_12_07
"""

from __future__ import annotations

import argparse
from pathlib import Path

from panel_eval_lib import evaluate_panel_split


def main() -> int:
    ap = argparse.ArgumentParser(description="Panel detection vs Manga109 frame GT")
    ap.add_argument("--weights", type=Path, required=True, help="Trained ``.pt``")
    ap.add_argument("--yolo-root", type=Path, default=None, help="Export root containing images/<split>/")
    ap.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Direct folder of .jpg pages (alternative to --yolo-root)",
    )
    ap.add_argument("--split", choices=("val", "test", "train"), default="val")
    ap.add_argument("--data-root", type=Path, default=Path("data/raw/manga109/Manga109_released_2023_12_07"))
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--panel-class", type=int, default=0)
    ap.add_argument("--experiment-id", default="", help="Label for logs / manifests")
    args = ap.parse_args()

    yr = args.yolo_root or Path(".")
    if args.yolo_root is None and args.images_dir is None:
        print("ERROR: supply --yolo-root or --images-dir")
        return 1

    try:
        r = evaluate_panel_split(
            weights=args.weights,
            yolo_root=yr,
            split=args.split,
            data_root=args.data_root,
            images_dir=args.images_dir,
            iou=args.iou,
            conf=args.conf,
            panel_class=args.panel_class,
            experiment_id=args.experiment_id or args.weights.stem,
            show_progress=True,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1

    print("\n=== Panel (=frame class) detection vs Manga109 <frame> GT ===")
    print(f"  experiment      : {r.experiment_id}")
    print(f"  split           : {r.split}")
    print(f"  images          : {r.n_images}")
    print(f"  skipped         : {r.skipped}")
    print(f"  IoU threshold   : {args.iou}")
    print(f"  conf threshold  : {args.conf}")
    print(f"  TP / FP / FN    : {r.tp} / {r.fp} / {r.fn}")
    print(f"  Precision       : {r.precision:.4f}")
    print(f"  Recall          : {r.recall:.4f}")
    print(f"  F1              : {r.f1:.4f}")
    print("\nGreedy IoU matching per image.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
