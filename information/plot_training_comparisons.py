#!/usr/bin/env python3
"""
Build comparison plots for YOLO experiments vs Manga109 GT (panels ``<frame>`` or bubbles ``<text>``).

Reads ``experiments_for_plots.yaml`` (or ``--config``), runs evaluation per experiment row,
writes PNGs into ``information/figures/``.

Each output figure uses **three subplots across** (Precision, Recall, F1). Inside each subplot
you get **one bar per entry in YAML ``splits``** (e.g. ``train`` / ``val`` / ``test``), shown in the legend.
Benchmark configs should list ``train`` under ``splits`` if you want training-set precision there.

Usage (from ``manga-parser/``; produces PNG + JSON under ``information/figures/``)::

    python information/plot_training_comparisons.py
    python information/plot_training_comparisons.py --config information/experiments_for_plots.yaml
    python information/plot_training_comparisons.py \\
        --config information/benchmark_runs/bm_ep1_mps_s42_rand_b3_0508_2159_experiments.yaml \\
        --output-suffix bm_ep1_mps_s42_rand_b3_0508_2159

Optional YAML field per experiment: ``label: my short name`` overrides default abbreviations on the graphs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_plot_helpers import plot_metrics_comparison, plot_tp_fp_fn_matrices, short_experiment_label
from manga109_loader import Manga109Loader
from panel_eval_lib import PanelEvalResult, evaluate_bubble_split, evaluate_panel_split
from ultralytics import YOLO

FIG_DIR = ROOT / "information" / "figures"


def _split_heading_word(code: str) -> str:
    return {"train": "training", "val": "validation", "test": "test"}.get(code, code)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=Path,
        default=ROOT / "information" / "experiments_for_plots.yaml",
    )
    ap.add_argument(
        "--output-suffix",
        default=None,
        help="Override YAML output_suffix for figure filenames (default: YAML or task name).",
    )
    args = ap.parse_args()

    if not args.config.is_file():
        print(f"Missing {args.config} — copy experiments_for_plots.example.yaml")
        return 1

    cfg = yaml.safe_load(args.config.read_text())
    data_root = Path(cfg["data_root"])
    iou = float(cfg.get("iou", 0.5))
    conf = float(cfg.get("conf", 0.25))
    panel_class = int(cfg.get("panel_class", 0))
    bubble_class = int(cfg.get("bubble_class", 1))
    task = str(cfg.get("task", "panel")).strip().lower()
    if task not in ("panel", "bubble"):
        print("task must be 'panel' or 'bubble'")
        return 1
    splits = list(cfg.get("splits", ["train", "val", "test"]))
    out_suffix = str(args.output_suffix or cfg.get("output_suffix") or task)
    cats = {"frame"} if task == "panel" else {"text"}

    experiment_labels: dict[str, str] = {}
    for exp in cfg.get("experiments", []):
        eid = str(exp["id"])
        raw = exp.get("label")
        if raw is not None and str(raw).strip():
            experiment_labels[eid] = str(raw).strip()
        elif eid not in experiment_labels:
            experiment_labels[eid] = short_experiment_label(eid, task=task)

    results: list[PanelEvalResult] = []
    summary_rows = []

    for exp in cfg["experiments"]:
        eid = exp["id"]
        weights = ROOT / exp["weights"]
        yolo_root = ROOT / exp["yolo_root"]
        if not weights.is_file():
            print(f"[skip] weights missing: {weights}")
            continue
        if not (yolo_root / "images").is_dir():
            print(f"[skip] yolo_root missing images/: {yolo_root}")
            continue

        loader = None
        model = None
        for sp in splits:
            img_dir = yolo_root / "images" / sp
            if not img_dir.is_dir() or not list(img_dir.glob("*.jpg")):
                print(f"[skip] {eid} {sp}: no images")
                continue

            if loader is None:
                loader = Manga109Loader(str(data_root), categories=cats)
                model = YOLO(str(weights))

            if task == "panel":
                r = evaluate_panel_split(
                    weights=weights,
                    yolo_root=yolo_root,
                    split=sp,
                    data_root=data_root,
                    iou=iou,
                    conf=conf,
                    panel_class=panel_class,
                    experiment_id=eid,
                    show_progress=True,
                    model=model,
                    loader=loader,
                )
            else:
                r = evaluate_bubble_split(
                    weights=weights,
                    yolo_root=yolo_root,
                    split=sp,
                    data_root=data_root,
                    iou=iou,
                    conf=conf,
                    bubble_class=bubble_class,
                    experiment_id=eid,
                    show_progress=True,
                    model=model,
                    loader=loader,
                )

            results.append(r)
            summary_rows.append(
                {
                    "task": task,
                    "experiment_id": r.experiment_id,
                    "split": r.split,
                    "precision": round(r.precision, 4),
                    "recall": round(r.recall, 4),
                    "f1": round(r.f1, 4),
                    "tp": r.tp,
                    "fp": r.fp,
                    "fn": r.fn,
                }
            )

    if not results:
        print("No results to plot.")
        return 1

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    metrics_png = FIG_DIR / f"metrics_val_test_comparison_{out_suffix}.png"
    matrix_png = FIG_DIR / f"tp_fp_fn_matrix_{out_suffix}.png"
    pos_label = "panel" if task == "panel" else "bubble"
    task_heading = "Panel boxes" if task == "panel" else "Speech bubbles"
    split_parts = [_split_heading_word(s) for s in splits]
    split_suptitle = ", ".join(split_parts)
    plot_metrics_comparison(
        results,
        metrics_png,
        splits_order=splits,
        metric_title_suffix="IoU overlap with annotated boxes",
        suptitle=f"{task_heading}: {split_suptitle}",
        experiment_labels=experiment_labels,
        task=task,
    )
    plot_tp_fp_fn_matrices(
        results,
        matrix_png,
        splits_order=splits,
        positive_class_label=pos_label,
        suptitle=f"{task_heading}: detection counts per split",
        experiment_labels=experiment_labels,
        task=task,
    )

    json_path = FIG_DIR / f"eval_summary_{out_suffix}.json"
    json_path.write_text(json.dumps(summary_rows, indent=2))
    print(f"Wrote {metrics_png}")
    print(f"Wrote {matrix_png}")
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
