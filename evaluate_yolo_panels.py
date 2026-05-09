#!/usr/bin/env python3
"""
Evaluate panel (YOLO class **frame**) detections vs Manga109 **<frame>** ground truth.

Maps each image filename ``{book}_{page:03d}.jpg`` back to XML, runs the trained model,
and reports micro-averaged precision / recall / F1 at an IoU threshold (greedy matching).

Usage:

    python evaluate_yolo_panels.py \\
        --weights runs/manga109_smoke/weights/best.pt \\
        --yolo-root data/processed/manga109_yolo_smoke \\
        --split val \\
        --data-root data/raw/manga109/Manga109_released_2023_12_07 \\
        --iou 0.5
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Sequence, Tuple

from tqdm import tqdm
from ultralytics import YOLO

from manga109_loader import Manga109Loader


STEM_RE = re.compile(r"^(.+)_(\d{3})$")


def xyxy_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def gt_frame_boxes_xyxy(page) -> List[Tuple[float, float, float, float]]:
    out: List[Tuple[float, float, float, float]] = []
    for ann in page.frames:
        b = ann.bbox
        out.append((float(b.xmin), float(b.ymin), float(b.xmax), float(b.ymax)))
    return out


def find_page(loader: Manga109Loader, book: str, page_index: int):
    bk = loader.books.get(book)
    if bk is None:
        return None
    for p in bk.pages:
        if p.page_index == page_index:
            return p
    return None


def match_greedy(
    gts: List[Tuple[float, float, float, float]],
    preds: List[Tuple[float, float, float, float]],
    iou_thresh: float,
) -> Tuple[int, int, int]:
    """Returns (tp, fp, fn) for one image (panel class only)."""
    if not gts and not preds:
        return 0, 0, 0
    if not gts:
        return 0, len(preds), 0
    if not preds:
        return 0, 0, len(gts)

    used_pred = set()
    tp = 0
    for gt in gts:
        best_j = -1
        best_iou = iou_thresh
        for j, pr in enumerate(preds):
            if j in used_pred:
                continue
            iou = xyxy_iou(gt, pr)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0:
            tp += 1
            used_pred.add(best_j)
    fn = len(gts) - tp
    fp = len(preds) - tp
    return tp, fp, fn


def main() -> int:
    ap = argparse.ArgumentParser(description="Panel detection vs Manga109 frame GT")
    ap.add_argument("--weights", type=Path, required=True, help="Trained ``.pt`` (e.g. best.pt)")
    ap.add_argument("--yolo-root", type=Path, help="Converted dataset root containing images/<split>")
    ap.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Override: folder of images (defaults to yolo-root/images/<split>).",
    )
    ap.add_argument("--split", choices=("val", "test", "train"), default="val")
    ap.add_argument("--data-root", type=Path, default=Path("data/raw/manga109/Manga109_released_2023_12_07"))
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--panel-class", type=int, default=0, help="YOLO class id for <frame>")
    args = ap.parse_args()

    img_root = args.images_dir or (args.yolo_root / "images" / args.split)
    if not img_root.is_dir():
        print(f"ERROR: image folder not found: {img_root}")
        return 1

    jpgs = sorted(img_root.glob("*.jpg"))
    if not jpgs:
        print(f"ERROR: no .jpg files in {img_root}")
        return 1

    print("Loading Manga109 for GT (frame boxes only)...")
    loader = Manga109Loader(str(args.data_root), categories={"frame"})

    model = YOLO(str(args.weights))

    tp = fp = fn = 0
    skipped = 0
    for img_path in tqdm(jpgs, desc=args.split):
        m = STEM_RE.match(img_path.stem)
        if not m:
            skipped += 1
            continue
        book = m.group(1)
        page_index = int(m.group(2))
        page = find_page(loader, book, page_index)
        if page is None:
            skipped += 1
            continue

        gts = gt_frame_boxes_xyxy(page)
        res = model.predict(str(img_path), conf=args.conf, verbose=False)
        preds: List[Tuple[float, float, float, float]] = []
        if res and len(res):
            boxes = res[0].boxes
            if boxes is not None and len(boxes):
                xyxy = boxes.xyxy.cpu().numpy()
                cls = boxes.cls.cpu().numpy().astype(int)
                for row, cl in zip(xyxy, cls):
                    if cl == args.panel_class:
                        preds.append(tuple(map(float, row)))

        a, b, c = match_greedy(gts, preds, args.iou)
        tp += a
        fp += b
        fn += c

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    print("\n=== Panel (=frame class) detection vs Manga109 <frame> GT ===")
    print(f"  split           : {args.split}")
    print(f"  images          : {len(jpgs)}")
    print(f"  skipped (stem/GT): {skipped}")
    print(f"  IoU threshold    : {args.iou}")
    print(f"  conf threshold   : {args.conf}")
    print(f"  TP / FP / FN     : {tp} / {fp} / {fn}")
    print(f"  Precision        : {prec:.4f}")
    print(f"  Recall           : {rec:.4f}")
    print(f"  F1               : {f1:.4f}")
    print(f"\nGreedy matching per image; order-dependent. Use fixed val split + manifest for comparisons.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
