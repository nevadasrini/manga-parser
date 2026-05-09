"""Shared matplotlib plots for greedy IoU detection evaluation (panels or bubbles)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Protocol

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np


class HasEvalMetrics(Protocol):
    experiment_id: str
    split: str
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float


def humanize_experiment_id(eid: str, *, max_len: int = 32) -> str:
    """Turn ``snake_case`` ids into short, readable titles (no config required)."""
    s = eid.replace("_", " ").strip()
    if len(s) > max_len:
        return s[: max_len - 1] + "…"
    return s


# Short x-axis titles for grouped bar plots (YAML ``label:`` overrides per row).
_PANEL_EXPERIMENT_SHORT_LABELS: Dict[str, str] = {
    "panel_input_raw": "raw",
    "panel_input_preprocess_only": "pre",
    "panel_input_preprocess_edges": "pre+edge",
    "panel_input_preprocess_repair_edges": "pre+rep+edge",
    "panel_input_edges_repair_only": "rep+edge",
}
_BUBBLE_EXPERIMENT_SHORT_LABELS: Dict[str, str] = {
    # add keys when bubble benchmark ids stabilize
}


def short_experiment_label(eid: str, *, task: str = "panel") -> str:
    """Compact label for matplotlib (avoids clipped long snake_case ids)."""
    tbl = _BUBBLE_EXPERIMENT_SHORT_LABELS if task == "bubble" else _PANEL_EXPERIMENT_SHORT_LABELS
    if eid in tbl:
        return tbl[eid]
    if len(eid) <= 14:
        return eid
    return eid[:12] + "…"


def split_display_name(split_code: str) -> str:
    return {"val": "Validation", "test": "Test", "train": "Training"}.get(split_code, split_code.title())


_CELL_OUTLINE = [pe.withStroke(linewidth=4.0, foreground="white", alpha=0.92)]


def plot_metrics_comparison(
    results: List[HasEvalMetrics],
    out_path: Path,
    *,
    splits_order: List[str],
    suptitle: str = "Validation vs test",
    metric_title_suffix: str = "Greedy IoU vs ground truth",
    experiment_labels: Optional[Dict[str, str]] = None,
    task: str = "panel",
) -> None:
    """Grouped bars per experiment: one bar per split in ``splits_order`` (e.g. train / val / test)."""
    if not splits_order:
        splits_order = ["val", "test"]
    exp_ids = sorted({r.experiment_id for r in results})
    el = experiment_labels or {}
    labels = [el.get(eid) or short_experiment_label(eid, task=task) for eid in exp_ids]
    x = np.arange(len(exp_ids))
    n_sp = len(splits_order)
    bar_w = min(0.26, 0.75 / max(n_sp, 1))
    offsets = (np.arange(n_sp) - (n_sp - 1) / 2.0) * bar_w
    metric_cfg = [
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1 score"),
    ]
    fig_w = 11.0 + 1.8 * max(0, n_sp - 2)
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, 4.2), sharey=True)
    for ax, (mkey, mlabel) in zip(axes, metric_cfg):
        for si, split in enumerate(splits_order):
            ys = []
            for eid in exp_ids:
                row = next(
                    (r for r in results if r.experiment_id == eid and r.split == split),
                    None,
                )
                ys.append(getattr(row, mkey, 0.0) if row else 0.0)
            leg = split_display_name(split)
            ax.bar(x + offsets[si], ys, bar_w, label=leg)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(mlabel)
        ax.set_title(f"{mlabel}\n({metric_title_suffix})", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.24)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_tp_fp_fn_matrices(
    results: List[HasEvalMetrics],
    out_path: Path,
    *,
    splits_order: List[str],
    positive_class_label: str = "object",
    suptitle: str = "Detections matched to ground truth (counts per cell)",
    experiment_labels: Optional[Dict[str, str]] = None,
    task: str = "panel",
) -> None:
    """2×2 style matrix ``[[TP, FN],[FP, 0]]`` per experiment × each split."""
    if not splits_order:
        splits_order = ["val", "test"]
    exp_ids = sorted({r.experiment_id for r in results})
    splits = list(splits_order)
    n_exp = len(exp_ids)
    n_sp = len(splits)
    fig, axes = plt.subplots(n_exp, n_sp, figsize=(4.2 + 5.0 * n_sp * 0.28, 3.9 * n_exp), squeeze=False)
    vmax_data = float(max((max(r.tp, r.fp, r.fn, 1) for r in results), default=1.0))
    cmap = plt.get_cmap("Blues")
    color_vmax = max(vmax_data, 1.0)

    col_labels = ["Model says\nyes", "Model says\nno"]
    if positive_class_label == "panel":
        row_labels = ["Panel in\nannotations", "Extra box\n(no match)"]
    elif positive_class_label == "bubble":
        row_labels = ["Bubble in\nannotations", "Extra box\n(no match)"]
    else:
        row_labels = ["In label\nlist", "Extra box\n(no match)"]

    el = experiment_labels or {}
    for i, eid in enumerate(exp_ids):
        disp = el.get(eid) or short_experiment_label(eid, task=task)
        for j, sp in enumerate(splits):
            ax = axes[i][j]
            r = next((x for x in results if x.experiment_id == eid and x.split == sp), None)
            if r is None:
                ax.axis("off")
                continue
            mat = np.array([[r.tp, r.fn], [r.fp, 0.0]], dtype=float)
            im = ax.imshow(mat, vmin=0, vmax=color_vmax, cmap=cmap, aspect="equal")
            ax.set_xticks([0, 1])
            ax.set_xticklabels(col_labels, fontsize=9)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(row_labels, fontsize=9)
            for (yy, xx), val in np.ndenumerate(mat):
                ax.text(
                    xx,
                    yy,
                    f"{int(val)}",
                    ha="center",
                    va="center",
                    fontsize=13,
                    fontweight="bold",
                    color="#0c0c0c",
                    path_effects=_CELL_OUTLINE,
                )
            sp_name = split_display_name(sp)
            ax.set_title(
                f"{disp}\n{sp_name} · P={r.precision:.2f}  R={r.recall:.2f}  F1={r.f1:.2f}",
                fontsize=10,
            )
            cbar = plt.colorbar(im, ax=ax, fraction=0.046)
            cbar.ax.set_ylabel("Count", rotation=270, labelpad=14, fontsize=9)

    fig.suptitle(suptitle, fontsize=11.5, y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
