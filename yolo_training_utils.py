"""Post-training summaries and optional epoch logging for Ultralytics YOLOv8."""

from __future__ import annotations

import csv
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def pick_yolo_device_for_mac_or_gpu(explicit: Optional[str] = None) -> Tuple[str, str]:
    """
    Choose an Ultralytics ``device`` string and a short tag for run IDs.

    On **Apple Silicon macOS**, **MPS** (Metal) is usually much faster than CPU for YOLO;
    Ultralytics expects ``device=\"mps\"``. Intel-only Macs have no CUDA and no MPS; use CPU.
    **NVIDIA GPUs** → ``\"0\"`` (first CUDA device).

    Override with explicit values such as ``\"mps\"``, ``\"cpu\"``, ``\"0\"``, ``\"0,1\"``.
    Pass ``explicit=\"auto\"`` or ``None`` for automatic selection.
    """
    exp = None if explicit in (None, "", "auto") else explicit.strip()

    def tag_for_device(dev: str) -> str:
        d = dev.lower()
        if d == "mps":
            return "mps"
        if "cpu" in d:
            return "cpu"
        if d.startswith("cuda") or d in {"0", "1", "2"} or "," in dev:
            return "cuda"
        return "custom"

    if exp is None:
        try:
            import torch

            mps_ok = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            if platform.system() == "Darwin" and mps_ok:
                return ("mps", "mps")
            if torch.cuda.is_available():
                return ("0", "cuda")
        except Exception:
            pass
        return ("cpu", "cpu")

    if exp is not None and exp.lower() == "cuda" and torch_available_cuda():
        return ("0", "cuda")

    return exp, tag_for_device(exp)


def torch_available_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def default_dataloader_workers(cli_workers: Optional[int]) -> int:
    """macOS multiprocessing loaders often glitch; ``0`` is the safe default."""
    if cli_workers is not None:
        return int(cli_workers)
    return 0 if platform.system() == "Darwin" else 4


def pick_benchmark_epoch_default() -> int:
    """Single default epoch budget shared across variant models in automatic benchmarks."""
    return 60


def read_results_csv_tail_rows(csv_path: Path, n: int = 3) -> List[Dict[str, str]]:
    if not csv_path.is_file():
        return []
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-n:] if rows else []


def print_ultralytics_run_summary(save_dir: Path, *, last_epochs: int = 1) -> None:
    """
    Print the last epoch line(s) from ``results.csv`` and point to ``results.png``.

    Ultralytics prints rich metrics during training; this echoes the numerical final epoch
    in one block after ``model.train()`` returns.
    """
    save_dir = Path(save_dir)
    csv_path = save_dir / "results.csv"
    png_path = save_dir / "results.png"

    print("\n=== Training run summary ===")
    print(f"  save_dir   : {save_dir.resolve()}")
    if png_path.is_file():
        print(f"  curves PNG : {png_path}")

    tail = read_results_csv_tail_rows(csv_path, n=last_epochs)
    if not tail:
        if not csv_path.is_file():
            print("  results.csv not found yet (training may export it only after first epoch completes).")
        return

    preferred = (
        "epoch",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "train/box_loss",
        "val/box_loss",
    )
    print(f"  Last {len(tail)} epoch row(s) from results.csv (key metrics):")
    for row in tail:
        parts = []
        for k in preferred:
            if k in row and row[k] not in (None, ""):
                parts.append(f"{k}={row[k]}")
        if not parts:
            parts = [f"{k}={v}" for k, v in sorted(row.items()) if v not in (None, "")]
        print("   ", " ".join(parts))


def flatten_trainer_metrics(trainer: Any) -> Dict[str, float]:
    """Best-effort map of trainer.metric keys → float for a single validation snapshot."""
    out: Dict[str, float] = {}
    metrics = getattr(trainer, "metrics", None)
    if metrics is None:
        return out
    if hasattr(metrics, "results_dict"):
        d = metrics.results_dict  # ultralytics Metrics object
        for k, v in d.items():
            try:
                out[str(k)] = float(v)
            except (TypeError, ValueError):
                continue
        return out
    if isinstance(metrics, dict):
        for k, v in metrics.items():
            try:
                out[str(k)] = float(v)
            except (TypeError, ValueError):
                continue
    return out


def epoch_end_log_line(epoch: int, trainer: Any) -> str:
    m = flatten_trainer_metrics(trainer)
    keys = sorted(m.keys())
    short = ", ".join(f"{k}={m[k]:.4f}" for k in keys[:12])
    extra = "..." if len(keys) > 12 else ""
    return f"epoch {epoch} val-metrics: {short} {extra}".rstrip()


def attach_epoch_logging_callback(model: Any, log_path: Path) -> None:
    """Append one line per epoch (after validation) to ``log_path`` and mirror to stdout."""

    log_path = Path(log_path)

    def on_train_epoch_end(trainer: Any) -> None:
        epoch = int(getattr(trainer, "epoch", -1))
        line = epoch_end_log_line(epoch, trainer)
        print(line, flush=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    model.add_callback("on_train_epoch_end", on_train_epoch_end)
