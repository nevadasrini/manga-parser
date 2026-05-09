#!/usr/bin/env python3
"""
Run a fine-tuned YOLOv8 model on a manga page (or folder) and return **speech bubble** boxes.

YOLO dataset class **1** = ``text`` (see ``convert_manga109_to_yolo.py``). Other classes are ignored.

Examples::

    python detect_bubbles.py --weights runs/detect/manga109/weights/best.pt --image path/to/page.jpg
    python detect_bubbles.py -w best.pt -i data/processed/manga109_yolo/images/val/ --save-vis outputs/bubble_vis
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class BubbleBox:
    """Axis-aligned bubble in pixel coordinates (xyxy) + confidence."""

    xyxy: Tuple[float, float, float, float]
    conf: float
    cls: int


def detect_bubbles(
    model: YOLO,
    image_path: str | Path,
    *,
    conf: float = 0.25,
    imgsz: int | None = None,
    device: str | int | None = None,
    bubble_class: int = 1,
) -> List[BubbleBox]:
    """
    Predict on one image; keep only detections with ``cls == bubble_class``.
    """
    kw: dict = dict(conf=conf, verbose=False)
    if imgsz is not None:
        kw["imgsz"] = imgsz
    if device is not None:
        kw["device"] = device

    res = model.predict(str(image_path), **kw)
    out: List[BubbleBox] = []
    if not res or not len(res):
        return out
    boxes = res[0].boxes
    if boxes is None or len(boxes) == 0:
        return out

    xyxy_b = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    scores = boxes.conf.cpu().numpy()
    for row, cl, sc in zip(xyxy_b, cls, scores):
        if int(cl) != bubble_class:
            continue
        out.append(BubbleBox(xyxy=tuple(map(float, row)), conf=float(sc), cls=int(cl)))
    return out


def draw_bubbles_bgr(
    bgr: np.ndarray,
    bubbles: Sequence[BubbleBox],
    *,
    color: Tuple[int, int, int] = (0, 200, 0),
    thickness: int = 2,
) -> np.ndarray:
    for b in bubbles:
        x1, y1, x2, y2 = map(int, b.xyxy)
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            bgr,
            f"{b.conf:.2f}",
            (x1, max(0, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return bgr


def iter_images(folder: Path) -> Iterator[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def main() -> int:
    ap = argparse.ArgumentParser(description="Speech-bubble detection (YOLO class 1 / text)")
    ap.add_argument("--weights", "-w", type=Path, required=True)
    ap.add_argument("--image", "-i", type=Path, required=True, help="Single image or directory")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--bubble-class", type=int, default=1)
    ap.add_argument(
        "--save-vis",
        type=Path,
        default=None,
        help="Optional folder; overlay boxes on copies of inputs",
    )
    args = ap.parse_args()

    if not args.weights.is_file():
        print(f"Missing weights: {args.weights}")
        return 1

    path = args.image.resolve()
    if not path.exists():
        print(f"Not found: {path}")
        return 1

    model = YOLO(str(args.weights))
    imgs: List[Path] = []
    if path.is_dir():
        imgs = list(iter_images(path))
        if not imgs:
            print(f"No images in {path}")
            return 1
    else:
        imgs = [path]

    if args.save_vis is not None:
        args.save_vis.mkdir(parents=True, exist_ok=True)

    total_boxes = 0
    for img_path in imgs:
        bubbles = detect_bubbles(
            model,
            img_path,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            bubble_class=args.bubble_class,
        )
        total_boxes += len(bubbles)
        print(f"{img_path.name}: {len(bubbles)} bubble(s)")
        for k, bb in enumerate(bubbles):
            x1, y1, x2, y2 = bb.xyxy
            print(f"  [{k}] xyxy=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}) conf={bb.conf:.4f}")

        if args.save_vis is not None:
            bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if bgr is not None:
                vis = draw_bubbles_bgr(bgr, bubbles)
                out_p = args.save_vis / f"bubbles_{img_path.stem}.jpg"
                cv2.imwrite(str(out_p), vis)
                print(f"  wrote {out_p}")

    print(f"\nDone. {len(imgs)} image(s), {total_boxes} bubble(s) total.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
