#!/usr/bin/env python3
"""
Visualize **where a trained YOLO model places panel boxes (class 0)** after each classical
pipeline stage used for **training images**.

**What this is not**

- ``preview_manga109_pages.py`` only compares raw binarization (no detector).
- The **training export** in ``pipeline_yolo_export.py`` uses ``use_contours=False`` — YOLO
  never sees contour *fills* in that path; it sees BGR images (raw, grayscale-from-preprocess,
  or Canny edge maps). The model outputs **axis-aligned rectangles**, not contour interiors.
- Optional ** rectangle fill** here is a **semi-transparent overlay** for readability — not the
  classical contour polygon fill from ``panel_pipeline``.

**Typical weights**

- One checkpoint per variant from ``run_panel_variant_benchmark.py``: pass ``--benchmark-id``
  (the ``bm_ep…`` prefix) so each column loads ``runs/detect/<id>_<variant>/weights/best.pt``.
- Or a single ``--weights`` for every column (same net on different-looking inputs).

Usage (from ``manga-parser/``)::

    python preview_manga109_yolo_panels.py --book ARMS --page 80 --benchmark-id bm_ep60_mps_s42_rand_b3_0508_2159
    python preview_manga109_yolo_panels.py --book ARMS --page 3 --weights runs/detect/manga109/weights/best.pt
    python preview_manga109_yolo_panels.py --book ARMS --page 3 --weights-yaml weights_by_variant.yaml

YAML example (paths relative to manga-parser or absolute)::

    raw: runs/detect/my_run_raw/weights/best.pt
    preprocess_only: runs/detect/my_run_preprocess_only/weights/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

from manga109_loader import Manga109Loader, Manga109Page
from pipeline_yolo_export import (
    IMAGE_VARIANT_CHOICES,
    RAW,
    VARIANT_LABELS,
    bgr_for_yolo_variant,
)


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Optional defaults (same idea as preview_manga109_pages.py)
SINGLE_TEST_BOOK: Optional[str] = "ARMS"
SINGLE_TEST_PAGE: Optional[int] = 3

_VARIANT_HEADLINE = {
    RAW: "Original page",
    "preprocess_only": "After preprocess",
    "preprocess_edges": "Preprocess + edges",
    "preprocess_repair_edges": "Pre + edge repair + edges",
    "edges_repair_only": "Edge repair + edges (no pre)",
}

_COLOR_BOX = (40, 180, 255)  # BGR orange-ish
_COLOR_BOX_ALT = (60, 80, 220)


def _find_best_weights_path(benchmark_id: str, variant: str) -> Path:
    """Same search order as ``run_panel_variant_benchmark._find_best_weights_path``."""
    run = f"{benchmark_id}_{variant}"
    candidates = [
        ROOT / "runs" / "detect" / run / "weights" / "best.pt",
        ROOT / "runs" / "detect" / "runs" / "detect" / run / "weights" / "best.pt",
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    found = sorted(
        ROOT.glob(f"runs/**/{run}/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if found:
        return found[0].resolve()
    return candidates[0].resolve()


def _find_page(loader: Manga109Loader, book: str, page_index: int) -> Optional[Manga109Page]:
    if book not in loader.books:
        return None
    for p in loader.books[book].pages:
        if p.page_index == page_index:
            return p
    return None


def _resize_fixed_height(bgr: np.ndarray, target_h: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    if h == 0:
        return bgr
    scale = target_h / float(h)
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(bgr, (new_w, target_h), interpolation=cv2.INTER_AREA)


def draw_panel_boxes(
    bgr: np.ndarray,
    boxes_xyxy: Sequence[Tuple[float, float, float, float]],
    *,
    fill_alpha: float = 0.0,
    thickness: int = 2,
) -> np.ndarray:
    """Draw panel rectangles; optional translucent fill then crisp outlines."""
    out = bgr
    if fill_alpha > 0.0:
        overlay = bgr.copy()
        for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
            c = _COLOR_BOX if i % 2 == 0 else _COLOR_BOX_ALT
            cv2.rectangle(
                overlay,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                c,
                -1,
            )
        out = cv2.addWeighted(overlay, fill_alpha, bgr, 1.0 - fill_alpha, 0)
    for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
        c = _COLOR_BOX if i % 2 == 0 else _COLOR_BOX_ALT
        cv2.rectangle(
            out,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            c,
            thickness,
            lineType=cv2.LINE_AA,
        )
    return out


def _banner(bgr: np.ndarray, line1: str, line2: str) -> np.ndarray:
    h, w = bgr.shape[:2]
    pad = 52
    canvas = np.zeros((h + pad, w, 3), dtype=np.uint8)
    canvas[:] = (38, 38, 38)
    canvas[pad:, :] = bgr
    cv2.putText(canvas, line1, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(canvas, line2, (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return canvas


def predict_panels_xyxy(model, bgr: np.ndarray, *, conf: float, panel_class: int, device, predict_imgsz: Optional[int]):
    kw: Dict = dict(conf=conf, verbose=False)
    if device is not None:
        kw["device"] = device
    if predict_imgsz is not None:
        kw["imgsz"] = predict_imgsz
    res = model.predict(bgr, **kw)
    if not res or not len(res):
        return []
    boxes = res[0].boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    out: List[Tuple[float, float, float, float]] = []
    for row, cl in zip(xyxy, cls):
        if int(cl) != panel_class:
            continue
        out.append(tuple(map(float, row)))
    return out


def resolve_weights_path(
    variant: str,
    *,
    single: Optional[Path],
    benchmark_id: Optional[str],
    yaml_map: Optional[Dict[str, str]],
) -> Path:
    if yaml_map is not None:
        if variant not in yaml_map:
            raise SystemExit(f"Weights YAML has no key {variant!r}")
        p = Path(yaml_map[variant].strip())
        return p.resolve() if p.is_absolute() else (ROOT / p).resolve()
    if benchmark_id:
        return _find_best_weights_path(benchmark_id, variant)
    if single is not None:
        return single.resolve()
    raise SystemExit("Provide --weights, --benchmark-id, or --weights-yaml")


def main() -> int:
    ap = argparse.ArgumentParser(description="YOLO panel boxes on each pipeline image variant")
    ap.add_argument("--data-root", default="data/raw/manga109/Manga109_released_2023_12_07")
    ap.add_argument("--book", default=SINGLE_TEST_BOOK)
    ap.add_argument("--page", type=int, default=SINGLE_TEST_PAGE)
    ap.add_argument("--weights", type=Path, default=None, help="Single .pt used for every column if no benchmark/yaml map")
    ap.add_argument("--benchmark-id", default=None, metavar="ID", help="e.g. bm_ep60_mps_s42_rand_b3_0508_2159 — per-variant checkpoints")
    ap.add_argument("--weights-yaml", type=Path, default=None, help="YAML mapping variant name -> weights path")
    ap.add_argument(
        "--variants",
        nargs="+",
        default=["raw", "preprocess_only", "preprocess_edges", "preprocess_repair_edges"],
        choices=list(IMAGE_VARIANT_CHOICES),
        help="Pipeline stages as used in convert_manga109_to_yolo --image-variant",
    )
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--panel-class", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--predict-imgsz", type=int, default=None, help="Forward pass size (default: Ultralytics auto)")
    ap.add_argument("--row-height", type=int, default=640, help="Resize each column to this height for the grid")
    ap.add_argument(
        "--fill-alpha",
        type=float,
        default=0.12,
        help="0 = outline only; ~0.1–0.2 = light translucent fill inside each box",
    )
    ap.add_argument("--out", type=Path, default=None, help="Output PNG (default: outputs/panel_yolo_{book}_{page}.png)")
    args = ap.parse_args()

    for v in args.variants:
        if v not in IMAGE_VARIANT_CHOICES:
            print(f"Unknown variant {v}")
            return 1

    ymap: Optional[Dict[str, str]] = None
    if args.weights_yaml is not None:
        if not args.weights_yaml.is_file():
            print(f"Missing {args.weights_yaml}")
            return 1
        ymap = yaml.safe_load(args.weights_yaml.read_text())
        if not isinstance(ymap, dict):
            print("weights-yaml must be a mapping variant -> path")
            return 1
        ymap = {str(k): str(v) for k, v in ymap.items()}

    root = Path(args.data_root)
    if not root.is_dir():
        print(f"Not a directory: {root}")
        return 1
    if args.book is None or args.page is None:
        print("Need --book and --page")
        return 1

    loader = Manga109Loader(str(root))
    page = _find_page(loader, args.book, args.page)
    if page is None:
        print(f"Page not found: {args.book} index {args.page}")
        return 1

    img_path = Path(page.image_path)
    if not img_path.is_file():
        print(f"Missing image {img_path}")
        return 1

    bgr0 = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr0 is None:
        print(f"Could not read {img_path}")
        return 1

    try:
        from ultralytics import YOLO
    except ImportError as e:
        print("Install ultralytics: pip install ultralytics", e)
        return 1

    model_cache: Dict[str, object] = {}

    def get_model(wpath: Path):
        key = str(wpath.resolve())
        if key not in model_cache:
            if not wpath.is_file():
                print(f"Missing weights: {wpath}")
                raise SystemExit(1)
            model_cache[key] = YOLO(key)
        return model_cache[key]

    columns: List[np.ndarray] = []
    for variant in args.variants:
        wpath = resolve_weights_path(
            variant,
            single=args.weights,
            benchmark_id=args.benchmark_id,
            yaml_map=ymap,
        )
        mod = get_model(wpath)
        viz_in = bgr_for_yolo_variant(bgr0, variant) if variant != RAW else bgr0.copy()
        preds = predict_panels_xyxy(
            mod,
            viz_in,
            conf=args.conf,
            panel_class=args.panel_class,
            device=args.device,
            predict_imgsz=args.predict_imgsz,
        )
        titled = VARIANT_LABELS.get(variant, variant)
        head = _VARIANT_HEADLINE.get(variant, titled)
        drawn = draw_panel_boxes(viz_in, preds, fill_alpha=max(0.0, float(args.fill_alpha)))
        col = _resize_fixed_height(drawn, args.row_height)
        col = _banner(
            col,
            f"{head}",
            f"n={len(preds)} panels  |  conf≥{args.conf}  |  {Path(wpath).name}",
        )
        columns.append(col)

    max_w = max(c.shape[1] for c in columns)
    padded = []
    for c in columns:
        h, w = c.shape[:2]
        if w < max_w:
            pad = max_w - w
            left = pad // 2
            right = pad - left
            c = cv2.copyMakeBorder(c, 0, 0, left, right, cv2.BORDER_CONSTANT, value=(28, 28, 28))
        padded.append(c)
    grid = np.hstack(padded)

    out = args.out
    if out is None:
        safe_book = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in page.book_title)
        out = Path("outputs") / f"panel_yolo_{safe_book}_{page.page_index:03d}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), grid)

    print(f"Saved: {out.resolve()}")
    print(f"  source: {img_path}")
    print(f"  variants: {args.variants}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
