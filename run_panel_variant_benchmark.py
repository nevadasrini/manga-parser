#!/usr/bin/env python3
"""
End-to-end panel-input benchmark: export all pipeline image variants, train one YOLO model per
variant with the **same epoch budget**, validate on val (during training), evaluate val+test vs
``<frame>`` GT, and write comparison figures with a **run-specific filename suffix**.

**Mac / device:** On Apple Silicon, training defaults to **MPS (Metal)** — usually much faster than
CPU. Intel Macs fall back to CPU; Linux/Windows with NVIDIA use CUDA device ``0``.

Default epoch budget: **60** per model (see :func:`yolo_training_utils.pick_benchmark_epoch_default`).

Usage (from ``manga-parser/``)::

    python run_panel_variant_benchmark.py
    python run_panel_variant_benchmark.py --epochs 40 --max-books 15
    python run_panel_variant_benchmark.py --device cpu --skip-export  # data already built
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from convert_manga109_to_yolo import convert  # noqa: E402
from pipeline_yolo_export import IMAGE_VARIANT_CHOICES  # noqa: E402
from yolo_training_utils import (  # noqa: E402
    pick_benchmark_epoch_default,
    pick_yolo_device_for_mac_or_gpu,
)


def _rel_under_manga_parser(path: Path) -> str:
    path = path.resolve()
    root = ROOT.resolve()
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _default_batch(dev_tag: str) -> int:
    if dev_tag == "mps":
        return 12
    if dev_tag == "cuda":
        return 16
    return 8


def _build_benchmark_id(
    *,
    epochs: int,
    dev_tag: str,
    seed: int,
    split_mode: str,
    max_books: Optional[int],
    stamp: str,
) -> str:
    parts = [f"ep{epochs}", dev_tag, f"s{seed}", split_mode[:4]]
    if max_books is not None:
        parts.append(f"b{max_books}")
    parts.append(stamp)
    return "bm_" + "_".join(parts)


def _find_best_weights_path(benchmark_id: str, variant: str) -> Path:
    """
    Canonical layout is ``runs/detect/<run_name>/weights/best.pt``.
    Older Ultralytics / trainer configs could nest ``runs/detect/runs/detect/``; glob as fallback.
    """
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
    return candidates[0]


def _write_experiments_yaml(
    path: Path,
    *,
    data_root: str,
    benchmark_id: str,
    yolo_parent: Path,
    output_suffix: str,
    weights_by_variant: Optional[Dict[str, str]] = None,
) -> None:
    experiments: List[Dict[str, Any]] = []
    for v in IMAGE_VARIANT_CHOICES:
        eid = {
            "raw": "panel_input_raw",
            "preprocess_only": "panel_input_preprocess_only",
            "preprocess_edges": "panel_input_preprocess_edges",
            "preprocess_repair_edges": "panel_input_preprocess_repair_edges",
            "edges_repair_only": "panel_input_edges_repair_only",
        }[v]
        if weights_by_variant and weights_by_variant.get(v):
            wrel = weights_by_variant[v]
        else:
            wrel = _rel_under_manga_parser(_find_best_weights_path(benchmark_id, v))
        experiments.append(
            {
                "id": eid,
                "weights": wrel,
                "yolo_root": _rel_under_manga_parser(yolo_parent / f"yolo_{v}"),
            }
        )
    cfg = {
        "task": "panel",
        "output_suffix": output_suffix,
        "data_root": data_root,
        "iou": 0.5,
        "conf": 0.25,
        "panel_class": 0,
        "splits": ["train", "val", "test"],
        "experiments": experiments,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))


def main() -> int:
    ap = argparse.ArgumentParser(description="Export → train → eval/plot for all panel image variants")
    ap.add_argument(
        "--data-root",
        default="data/raw/manga109/Manga109_released_2023_12_07",
        help="Manga109 release root",
    )
    ap.add_argument(
        "--output-base",
        default="data/processed",
        help="Parent directory for this benchmark's YOLO exports",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=f"Epochs per variant (default: {pick_benchmark_epoch_default()})",
    )
    ap.add_argument("--batch", type=int, default=None, help="Override auto batch (MPS/CUDA/CPU)")
    ap.add_argument(
        "--device",
        default=None,
        help="auto (default) | mps | cpu | 0 — passed to train_yolo / Ultralytics",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.70)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--split-mode", choices=("random", "stratified"), default="random")
    ap.add_argument("--max-books", type=int, default=None)
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-plots", action="store_true")
    args = ap.parse_args()

    epochs = int(args.epochs) if args.epochs is not None else pick_benchmark_epoch_default()
    dev, dev_tag = pick_yolo_device_for_mac_or_gpu(args.device)
    batch = args.batch if args.batch is not None else _default_batch(dev_tag)
    stamp = datetime.now().strftime("%m%d_%H%M")
    benchmark_id = _build_benchmark_id(
        epochs=epochs,
        dev_tag=dev_tag,
        seed=args.seed,
        split_mode=args.split_mode,
        max_books=args.max_books,
        stamp=stamp,
    )

    benchmark_root = (ROOT / args.output_base).resolve() / f"panel_benchmark_{benchmark_id}"
    figures_dir = ROOT / "information" / "figures"
    bench_meta_dir = ROOT / "information" / "benchmark_runs"
    cfg_path = bench_meta_dir / f"{benchmark_id}_experiments.yaml"
    suffix = benchmark_id

    manifest = {
        "benchmark_id": benchmark_id,
        "epochs_per_variant": epochs,
        "device_requested": args.device,
        "device_used": dev,
        "device_tag": dev_tag,
        "batch": batch,
        "seed": args.seed,
        "split_mode": args.split_mode,
        "max_books": args.max_books,
        "data_root": str((ROOT / args.data_root).resolve()),
        "benchmark_root": str(benchmark_root),
        "figures_out": {
            "metrics": _rel_under_manga_parser(figures_dir / f"metrics_val_test_comparison_{suffix}.png"),
            "matrix": _rel_under_manga_parser(figures_dir / f"tp_fp_fn_matrix_{suffix}.png"),
            "summary_json": _rel_under_manga_parser(figures_dir / f"eval_summary_{suffix}.json"),
        },
        "experiment_yaml": _rel_under_manga_parser(cfg_path),
    }

    print("=== Panel variant benchmark ===")
    print(f"  benchmark_id   : {benchmark_id}")
    print(f"  epochs/model   : {epochs}")
    print(f"  device         : {dev} ({dev_tag}); batch={batch}")
    print(f"  export root    : {benchmark_root}")
    print(f"  plot suffix    : {suffix}")

    benchmark_root.mkdir(parents=True, exist_ok=True)
    bench_meta_dir.mkdir(parents=True, exist_ok=True)
    (benchmark_root / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2))

    if not args.skip_export:
        for v in IMAGE_VARIANT_CHOICES:
            ydir = benchmark_root / f"yolo_{v}"
            print(f"\n--- export variant={v} -> {ydir} ---")
            convert(
                data_root=str((ROOT / args.data_root).resolve()),
                output_dir=ydir,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                copy_images=True,
                seed=args.seed,
                split_mode=args.split_mode,
                max_books=args.max_books,
                experiment_id=f"{benchmark_id}_{v}",
                image_variant=v,
            )
    else:
        print("(skip-export) reuse existing folders under benchmark_root")

    env = os.environ.copy()
    env["PYTHONWARNINGS"] = env.get("PYTHONWARNINGS", "ignore")

    if not args.skip_train:
        for v in IMAGE_VARIANT_CHOICES:
            ydir = benchmark_root / f"yolo_{v}"
            ds = ydir / "dataset.yaml"
            if not ds.is_file():
                print(f"ERROR: missing {ds}; run without --skip-export")
                return 1
            name = f"{benchmark_id}_{v}"
            print(f"\n--- train {name} ({epochs} epochs) ---")
            cmd = [
                sys.executable,
                str(ROOT / "train_yolo.py"),
                "--data",
                str(ds.relative_to(ROOT)),
                "--epochs",
                str(epochs),
                "--batch",
                str(batch),
                "--name",
                name,
                "--exist-ok",
                "--device",
                dev,
            ]
            subprocess.check_call(cmd, cwd=str(ROOT), env=env)
    else:
        print("(skip-train)")

    weights_by_variant: Dict[str, str] = {}
    for v in IMAGE_VARIANT_CHOICES:
        wt = _find_best_weights_path(benchmark_id, v)
        weights_by_variant[v] = _rel_under_manga_parser(wt)
        if not wt.is_file():
            print(f"[warn] missing weights for {v}: expected under runs/detect/... ({wt})")

    _write_experiments_yaml(
        cfg_path,
        data_root=args.data_root,
        benchmark_id=benchmark_id,
        yolo_parent=benchmark_root,
        output_suffix=suffix,
        weights_by_variant=weights_by_variant,
    )
    print(f"\nWrote experiment config → {cfg_path}")

    if args.skip_plots:
        print("(skip-plots) done.")
        return 0

    print("\n--- validation-style metrics during training ---")
    print("    See each run's results.png and results.csv under runs/detect/<run_name>/.")

    plot_cmd = [
        sys.executable,
        str(ROOT / "information/plot_training_comparisons.py"),
        "--config",
        str(cfg_path.relative_to(ROOT)),
        "--output-suffix",
        suffix,
    ]
    print(f"\n--- eval val+test + plots ---\n    {' '.join(plot_cmd)}")
    subprocess.check_call(plot_cmd, cwd=str(ROOT), env=env)

    marker = ROOT / "information" / "figures" / "LATEST_PANEL_BENCHMARK_SUFFIX.txt"
    marker.write_text(suffix + "\n")
    print(f"\nMarked latest benchmark suffix → {marker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
