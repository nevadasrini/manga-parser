#!/usr/bin/env python3
"""
Export Manga109 to YOLO layout once per :data:`pipeline_yolo_export.IMAGE_VARIANT_CHOICES`.

Each variant gets its own ``output_dir`` (default: ``<base>/manga109_yolo_<variant>``) so you can
train separate YOLO runs and compare with ``information/plot_training_comparisons.py``.

Usage (from ``manga-parser/``):

    python export_all_image_variants.py --data-root ... --output-base data/processed
    python export_all_image_variants.py --data-root ... --variants raw preprocess_only --max-books 10
"""

from __future__ import annotations

import argparse

from convert_manga109_to_yolo import convert
from pipeline_yolo_export import IMAGE_VARIANT_CHOICES


def main() -> int:
    ap = argparse.ArgumentParser(description="YOLO exports for each pipeline image variant")
    ap.add_argument(
        "--data-root",
        default="data/raw/manga109/Manga109_released_2023_12_07",
        help="Path to Manga109 release directory",
    )
    ap.add_argument(
        "--output-base",
        default="data/processed",
        help="Parent directory; each variant is written to ``<base>/manga109_yolo_<variant>``",
    )
    ap.add_argument(
        "--variants",
        nargs="*",
        default=list(IMAGE_VARIANT_CHOICES),
        choices=IMAGE_VARIANT_CHOICES,
        help="Subset of variants to export (default: all)",
    )
    ap.add_argument("--train_ratio", type=float, default=0.70)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--split-mode",
        choices=("random", "stratified"),
        default="random",
    )
    ap.add_argument("--max-books", type=int, default=None)
    args = ap.parse_args()

    for v in args.variants:
        out_dir = f"{args.output_base.rstrip('/')}/manga109_yolo_{v}"
        suffix = "_smoke" if args.max_books else ""
        eid = (
            f"{args.split_mode}_seed{args.seed}_{v}"
            + ("" if args.max_books is None else f"_b{args.max_books}")
            + suffix
        )
        print(f"\n=== variant={v} -> {out_dir} ===")
        convert(
            data_root=args.data_root,
            output_dir=out_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            copy_images=True,
            seed=args.seed,
            split_mode=args.split_mode,
            max_books=args.max_books,
            experiment_id=eid,
            image_variant=v,
        )

    print(
        """
Train (example — 100 epochs each; adjust --epochs / --batch / --device):

  for v in raw preprocess_only preprocess_edges preprocess_repair_edges edges_repair_only; do
    python train_yolo.py --data data/processed/manga109_yolo_${v}/dataset.yaml \\
      --name manga109_${v} --epochs 100 --exist-ok
  done

Or:  bash information/train_all_image_variants.sh

Then point information/experiments_for_plots.yaml at each
  weights:   runs/detect/manga109_<variant>/weights/best.pt
  yolo_root: data/processed/manga109_yolo_<variant>
and run:  python information/plot_training_comparisons.py
""".strip()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
