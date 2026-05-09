"""Greedy IoU evaluation: YOLO detections vs Manga109 GT (panels / ``<frame>`` or bubbles / ``<text>``)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from manga109_loader import Manga109Loader, Manga109Page

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


def gt_frame_boxes_xyxy(page: Manga109Page) -> List[Tuple[float, float, float, float]]:
    out: List[Tuple[float, float, float, float]] = []
    for ann in page.frames:
        b = ann.bbox
        out.append((float(b.xmin), float(b.ymin), float(b.xmax), float(b.ymax)))
    return out


def gt_text_boxes_xyxy(page: Manga109Page) -> List[Tuple[float, float, float, float]]:
    """Manga109 ``<text>`` regions (speech bubbles / captions)."""
    out: List[Tuple[float, float, float, float]] = []
    for ann in page.texts:
        b = ann.bbox
        out.append((float(b.xmin), float(b.ymin), float(b.xmax), float(b.ymax)))
    return out


def find_page(loader: Manga109Loader, book: str, page_index: int) -> Optional[Manga109Page]:
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


def confusion_counts(tp: int, fp: int, fn: int) -> Tuple[float, float, float, float]:
    """
    2×2 layout (sklearn-style) for exhaustive instance matching:

                   Pred +    Pred −
      Act +         TP       FN
      Act −         FP       TN*

    ``TN*`` is not counted for bbox detection (no background instances); reported as NaN or 0
    depending on plotting code.
    """
    tn_placeholder = 0.0
    return float(tp), float(fp), float(fn), tn_placeholder


@dataclass
class PanelEvalResult:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    n_images: int
    skipped: int
    split: str
    experiment_id: str


def evaluate_detection_split(
    *,
    weights: Path,
    yolo_root: Path,
    split: str,
    data_root: Path,
    images_dir: Optional[Path] = None,
    iou: float = 0.5,
    conf: float = 0.25,
    yolo_class_id: int = 0,
    loader_categories: Optional[set] = None,
    gt_boxes_xyxy: Callable[[Manga109Page], List[Tuple[float, float, float, float]]] = gt_frame_boxes_xyxy,
    experiment_id: str = "",
    show_progress: bool = False,
    model=None,
    loader=None,
) -> PanelEvalResult:
    """
    Greedy IoU matching for one YOLO ``class id`` vs GT boxes extracted from XML via ``gt_boxes_xyxy``.
    """
    img_root = images_dir if images_dir is not None else (yolo_root / "images" / split)
    if not img_root.is_dir():
        raise FileNotFoundError(f"Missing image folder: {img_root}")

    from tqdm import tqdm
    from ultralytics import YOLO

    jpgs = sorted(img_root.glob("*.jpg"))
    if not jpgs:
        raise FileNotFoundError(f"No JPGs under {img_root}")

    cats = loader_categories if loader_categories is not None else {"frame"}
    loader = loader or Manga109Loader(str(data_root), categories=cats)
    model = model or YOLO(str(weights))

    iterable = tqdm(jpgs, desc=f"{experiment_id}:{split}", leave=False) if show_progress else jpgs

    tp_total = fp_total = fn_total = 0
    skipped = 0
    for img_path in iterable:
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

        gts = gt_boxes_xyxy(page)
        res = model.predict(str(img_path), conf=conf, verbose=False)
        preds: List[Tuple[float, float, float, float]] = []
        if res and len(res):
            boxes = res[0].boxes
            if boxes is not None and len(boxes):
                xyxy_b = boxes.xyxy.cpu().numpy()
                cls = boxes.cls.cpu().numpy().astype(int)
                for row, cl in zip(xyxy_b, cls):
                    if int(cl) == yolo_class_id:
                        preds.append(tuple(map(float, row)))

        nt, nf, nfl = match_greedy(gts, preds, iou)
        tp_total += nt
        fp_total += nf
        fn_total += nfl

    prec = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    rec = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return PanelEvalResult(
        tp=tp_total,
        fp=fp_total,
        fn=fn_total,
        precision=prec,
        recall=rec,
        f1=f1,
        n_images=len(jpgs),
        skipped=skipped,
        split=split,
        experiment_id=experiment_id or weights.stem,
    )


def evaluate_panel_split(
    *,
    weights: Path,
    yolo_root: Path,
    split: str,
    data_root: Path,
    images_dir: Optional[Path] = None,
    iou: float = 0.5,
    conf: float = 0.25,
    panel_class: int = 0,
    experiment_id: str = "",
    show_progress: bool = False,
    model=None,
    loader=None,
) -> PanelEvalResult:
    """YOLO class ``panel_class`` vs Manga109 ``<frame>`` panels."""
    return evaluate_detection_split(
        weights=weights,
        yolo_root=yolo_root,
        split=split,
        data_root=data_root,
        images_dir=images_dir,
        iou=iou,
        conf=conf,
        yolo_class_id=panel_class,
        loader_categories={"frame"},
        gt_boxes_xyxy=gt_frame_boxes_xyxy,
        experiment_id=experiment_id,
        show_progress=show_progress,
        model=model,
        loader=loader,
    )


def evaluate_bubble_split(
    *,
    weights: Path,
    yolo_root: Path,
    split: str,
    data_root: Path,
    images_dir: Optional[Path] = None,
    iou: float = 0.5,
    conf: float = 0.25,
    bubble_class: int = 1,
    experiment_id: str = "",
    show_progress: bool = False,
    model=None,
    loader=None,
) -> PanelEvalResult:
    """YOLO class ``bubble_class`` (default ``1`` = ``text``) vs Manga109 ``<text>`` bubbles."""
    return evaluate_detection_split(
        weights=weights,
        yolo_root=yolo_root,
        split=split,
        data_root=data_root,
        images_dir=images_dir,
        iou=iou,
        conf=conf,
        yolo_class_id=bubble_class,
        loader_categories={"text"},
        gt_boxes_xyxy=gt_text_boxes_xyxy,
        experiment_id=experiment_id,
        show_progress=show_progress,
        model=model,
        loader=loader,
    )


def result_to_row_dict(r: PanelEvalResult) -> Dict[str, float | int | str]:
    return {
        "experiment_id": r.experiment_id,
        "split": r.split,
        "tp": r.tp,
        "fp": r.fp,
        "fn": r.fn,
        "precision": r.precision,
        "recall": r.recall,
        "f1": r.f1,
        "n_images": r.n_images,
        "skipped": r.skipped,
    }
