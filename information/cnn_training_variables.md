# CNN / YOLOv8 training variables — what they are & why they matter

This project fine-tunes a **convolutional neural network detector** (**YOLOv8 small**, Ultralytics): it predicts **bounding boxes + class** for panels (`frame`) and bubbles (`text`). Training uses **lots of knobs** so you must **freeze** unrelated ones when you compare “combinations.”

Below: **dataset / split variables** (experiment design), **`train_yolo.py` CLI** (CNN training), **`evaluate_yolo_panels.py` / plotting** (test vs annotations), plus **limitations** vs a tiny fully-connected (“MLP”) toy network.

---

## 1. Dataset & split variables (comparison design)

Stored in **`experiment_manifest.json`** when you convert Manga109 to YOLO.

| Variable | Meaning | Why it matters |
|----------|---------|----------------|
| **`split_mode`** | `random` vs `stratified` assignment of **whole books** to train/val/test | Avoids leakage (same volume in train **and** test). Stratified tries to equalize **average panels-per-page** across splits when you fear some books are much denser than others. |
| **`seed`** | RNG seed for **`random`** shuffling | Reproduces the **same books** in each split across runs → fair comparisons. Change seed only when you intentionally study split variance / confidence intervals. |
| **`train_ratio` / `val_ratio` / `test_ratio`** | Fraction of **books** in each bucket | Larger train → potentially better fit but less reliable val/test estimates. Typical 70 / 15 / 15 keeps enough held-out data. |
| **`max_books`** | Restrict to first *N* sorted titles | **Smoke tests** only; biases toward alphabetical titles, **not** a fair sample of full Manga109. Omit for thesis-scale numbers. |
| **`experiment_id`** | Human-readable label | Tie each trained checkpoint + tables + figures to **one** manifest JSON. |

**Why split by book, not page?**  
Pages from one volume share art style / layout shortcuts. Mixing volumes across train/test **inflates** reported accuracy.

---

## 2. Convolutional backbone & detection head (conceptual CNN pieces)

YOLO is not “front pass vs backward pass” as separate things you toggle — PyTorch computes **forward** (image → feature maps → box logits) and **backward** (gradient of loss w.r.t. weights) automatically.

| Piece | Role | Why it shapes results |
|-------|------|----------------------|
| **Backbone CNN** (`yolov8s.pt` …) | Strided conv layers extract **spatial features** (edges, textures, regions) | **Pretraining** on COCO initializes generic object detectors; manga fine-tuning adapts kernels to toner + gutters + panels. Starting from pretrained weights dominates training success on small annotated sets. |
| **Neck & head** | Multi-scale fusion + predictive layers for boxes/classes | Localization at different scales (small bubbles vs large panels). |
| **Parameters** (~11 M for YOLOv8s) vs an “academic toy” MLP | MLP collapses spatial structure unless you reshape huge vectors; detectors **reuse translation-equivariant convolution** crucial for layouts. |

**Backpropagation**: one **scalar loss** blends box regression + classification (+ distribution focal terms in Ultralytics implementation). Same loss drives all conv layers receiving gradients (**end-to-end**).

---

## 3. Variables in `train_yolo.py` (direct training controls)

These map to **`model.train(...)`** in Ultralytics.

| Variable | Typical value | Meaning & why |
|----------|----------------|---------------|
| **`--weights`** | `yolov8s.pt` | **Initialization**. Fine-tuning = start here; “from scratch” is rarely used without massive data. |
| **`--data`** | path to **`dataset.yaml`** | Declares classes, paths to train/val images (**must align** with the manifest experiment). |
| **`--epochs`** | 50–100+ prod; 1 for smoke | How many passes over train set — more may overfit unless regularized. |
| **`--batch`** | 16 (GPU-dependent) | Stochastic minibatch gradient noise + memory tradeoff — smaller batches are noisier but fit on smaller GPUs. |
| **`--imgsz`** | 640 typical; 320 smoke | Resize longer side — trades **spatial detail vs speed / memory**. |
| **`--device`** | `0` CUDA / `cpu` | Compute device; GPU strongly affects wall-clock throughput. |
| **`--workers`** | 4 (often `0` on macOS quirks) | DataLoader parallelism; `0` = main process loads images (fewer multiprocessing crashes). |
| **`--project`, `--name`, `--exist_ok`** | e.g. `runs/detect`, `exp_v1`, … | **Where checkpoints live** — **one `--name` per comparison** so you never overwrite unrelated runs. |
| **Augmentation (currently fixed in code)** | mosaic=0.8, fliplr=0.5, degrees=5 | Artificial diversity — reduces overfitting **if** augmentation matches domain (watch for destructive flips / rotations on asymmetric panels). |

**What you tune for “CNN vs preprocessing pipeline” comparisons:**  
The **CNN** learns from **whatever pixels** are inside `dataset.yaml`. If preprocessing ever changes exported images separately, rebuild a **different** `output_dir` + **`dataset.yaml`** + manifest — that becomes a controlled variable.

---

## 4. Evaluation variables (measuring panels vs annotations)

|`evaluate_yolo_panels.py` / `panel_eval_lib.py`|Purpose|
|`--iou`|Minimum **Intersection-over-Union** for a predicted box vs a **\<frame\>** box to count as hit — standard 0.5 is “loose”; 0.75 is strict.|
|`--conf`|Minimum **confidence** for YOLO boxes — lowers recall (↑FP), raises ↑FN.|
|**Greedy matching**|Each GT assigns at most one predicted box (highest IoU first per implementation order) → **approximate**, not Hungarian optimum — document when reporting numbers.|
|**Class filter**|Only **`panel_class`** (usually **0 = frame**) contributes — speech bubbles (**1 = text**) ignored for **panel IoU**.|

**Greedy detection metrics vs Ultralytics mAP:**  
YOLO trainer reports **Average Precision curves** aggregated differently. Numeric **precision/recall** from our script ≠ `mAP@0.5` — **do not** mix without relabeling table columns.

**Confusion-style matrix `[ [TP, FN], [FP, TN] ]`:**  
Fine-grained localization does **not** define a TN count for free-floating negatives; we plot **TN = 0** as a bookkeeping cell (see `information/figures/tp_fp_fn_matrix_by_experiment.png` caption).

---

## 5. Figures: validation vs held-out **test**, across experiments

Regenerate plots after updating **`information/experiments_for_plots.yaml`**:

```bash
cd manga-parser
source .venv/bin/activate   # optional
pip install matplotlib pyyaml
python information/plot_training_comparisons.py
```

Artifacts (saved under **`information/figures/`**):

| File | Content |
|------|---------|
| **`metrics_val_test_comparison.png`** | Bar groups: **Precision**, **Recall**, **F1** — **val** vs **test** per **`experiment.id`**. Same metric definition everywhere: greedy IoU on **\<frame\>** only. |
| **`tp_fp_fn_matrix_by_experiment.png`** | Heatmaps of aggregated **TP / FP / FN** (and TN placeholder) — one panel per (**experiment**, split). Lets you eyeball false positives vs missed panels. |
| **`eval_summary.json`** | Numeric table for reproducibility — paste into appendix / notebook. |

**Embedding in Markdown** (these paths are repo-relative):

![Validation vs test P/R/F1](figures/metrics_val_test_comparison.png)

![TP / FN / FP matrix by experiment × split](figures/tp_fp_fn_matrix_by_experiment.png)

### Snapshot (`eval_summary.json` — regenerate with plot script)

| experiment_id | split | Precision | Recall | F1 | TP | FP | FN |
|---------------|-------|-----------|--------|------|-----|-----|-----|
| smoke_random_b10_epoch1 | val | 0.541 | 0.789 | 0.642 | 761 | 646 | 203 |
| smoke_random_b10_epoch1 | test | 0.623 | 0.940 | 0.749 | 1494 | 906 | 95 |

*Smoke run only (10 books, 1 epoch, imgsz 320)—for thesis runs, overwrite YAML with full-data checkpoints and rerun `plot_training_comparisons.py`; numbers will drift.*

*If PNGs are missing locally, run the script below (needs trained checkpoints + exported YOLO folders).*

---

## 6. Checklist before claiming “comparison across methods”

- [ ] Identical **`data_root`** and **evaluation** thresholds (`iou`, `conf`) across checkpoints.  
- [ ] Separate **`dataset.yaml` + manifest` per preprocessing / split tweak.  
- [ ] Separate **`--name` / checkpoints** — never recycle the same weights folder accidentally.  
- [ ] Prefer **test** metrics for headline results; reserve **val** for early stopping / hype tuning — our plots show **both**.

---

## 7. Why not toggle “forward” / “backward” separately?

Backward pass is driven by whatever **loss** Ultralytics configures for detection. Turning it **off** is not meaningful for training a standard CNN—you would only “forward-eval” (**inference**) to measure a frozen baseline.

---

*Maintainers: regenerate figures after new training experiments; bump `experiment.id` strings in YAML and keep manifests under `experiment_manifest.json`.*
