#!/usr/bin/env python3
"""
Evaluate speech-bubble detection (YOLO class **text** / id **1**) vs Manga109 **<text>** GT.

Same greedy IoU protocol as ``evaluate_yolo_panels.py``. Use the **same**
``--yolo-root`` export and ``--data-root`` as for panel eval so filenames line up.

    python evaluate_yolo_bubbles.py \\
        --weights runs/detect/manga109_bubbles/weights/best.pt \\
        --yolo-root data/processed/manga109_yolo \\
        --split val \\
        --data-root data/raw/manga109/Manga109_released_2023_12_07
"""

from __future__ import annotations

import argparse
from pathlib import Path

from panel_eval_lib import evaluate_bubble_split


def main() -> int:
    ap = argparse.ArgumentParser(description="Bubble (=text class) detection vs Manga109 <text> GT")
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--yolo-root", type=Path, default=None)
    ap.add_argument("--images-dir", type=Path, default=None)
    ap.add_argument("--split", choices=("val", "test", "train"), default="val")
    ap.add_argument("--data-root", type=Path, default=Path("data/raw/manga109/Manga109_released_2023_12_07"))
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--bubble-class", type=int, default=1)
    ap.add_argument("--experiment-id", default="")
    args = ap.parse_args()

    yr = args.yolo_root or Path(".")
    if args.yolo_root is None and args.images_dir is None:
        print("ERROR: supply --yolo-root or --images-dir")
        return 1

    try:
        r = evaluate_bubble_split(
            weights=args.weights,
            yolo_root=yr,
            split=args.split,
            data_root=args.data_root,
            images_dir=args.images_dir,
            iou=args.iou,
            conf=args.conf,
            bubble_class=args.bubble_class,
            experiment_id=args.experiment_id or args.weights.stem,
            show_progress=True,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1

    print("\n=== Bubble (=text class) detection vs Manga109 <text> GT ===")
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
