# manga-parser

Tools to load **Manga109** from a local folder (XML + images), convert **panels (`frame`) and speech bubbles (`text`)** to **YOLO** labels, and train **Ultralytics YOLOv8** on that dataset.

This repo does **not** call `datasets.load_dataset` itself. You **download** data with the Hugging Face CLI (or copy an official release), then point scripts at the directory that contains **`images/`** and **`annotations/`**.

---

## 1. Hugging Face CLI install and login

**This is not a Mac limitation.** If `pip install huggingface` fails, it is usually because **there is no PyPI package with that exact name** for the Hub tools. Install **`huggingface_hub`** instead (it provides the **`hf`** command).

Use a **virtual environment** so `pip` and `hf` go to the same Python (macOS and Windows).

### macOS (Terminal, bash/zsh)

```bash
cd manga-parser
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -U huggingface_hub
hf auth login
hf auth whoami
```

### Windows (PowerShell)

```powershell
cd manga-parser
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -U huggingface_hub
hf auth login
hf auth whoami
```

If script execution is blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use **Command Prompt**: `.\.venv\Scripts\activate.bat` instead of `Activate.ps1`.

### If install still fails

- Use **`python -m pip install ...`** (not bare `pip`) so you install into the interpreter you think you are using.
- Upgrade pip: `python3 -m pip install -U pip` (Mac) / `python -m pip install -U pip` (Windows).
- Optional (only if you load data with the **`datasets`** library elsewhere): `pip install datasets` — still **not** the package name `huggingface`.

Paste your Hub token when `hf auth login` prompts (input may be hidden).

---

## 2. What to download (dataset files)

Download the **dataset repo** into this project (example path matches the defaults in `manga109_loader.py` / `convert_manga109_to_yolo.py` if you rename the folder to match, or pass `--data_root`).

From the **`manga-parser`** directory (with the venv **activated** so `hf` is available):

**macOS / Linux**

```bash
mkdir -p data/raw/manga109
hf download hal-utokyo/Manga109 --repo-type dataset --local-dir data/raw/manga109/Manga109_released_2023_12_07
```

**Windows (PowerShell)**

```powershell
New-Item -ItemType Directory -Force -Path data\raw\manga109 | Out-Null
hf download hal-utokyo/Manga109 --repo-type dataset --local-dir data\raw\manga109\Manga109_released_2023_12_07
```

That directory must end up with at least:

- **`images/<book_title>/<page>.jpg`** — page images  
- **`annotations/<book_title>.xml`** — per-book XML (frames, text, faces, bodies, etc.)

**Hub layout note:** `hf download` often creates an **extra inner folder** (e.g. `.../Manga109_released_2023_12_07/Manga109_released_2023_12_07/` where `annotations/` actually lives). **`Manga109Loader`** detects that: you can still pass the **outer** `--local-dir` path; it will print a one-line hint and use the nested root when needed.

**Cover page (`000.jpg`):** Manga109 uses **`page_index == 0`** → **`000.jpg`**. The XML has **no** explicit “cover” flag; many books use an **empty** `<page index="0" .../>` (no `<frame>` / `<text>`), which is a **heuristic** only. This repo treats **`Manga109Page.is_cover_page`** as index 0, skips it in **`convert_manga109_to_yolo.py`**, skips it by default in **`preview_manga109_pages.py`** (use **`--include-cover`** to print it), and exposes **`iter_story_pages(loader)`** for pipelines. In browse mode, preprocess **never** picks the cover as the first sample; with **`--book` / `--page`** you can still run on index **0** if you want an explicit cover test.

If your Hub folder name differs, either rename it or pass **`--data_root /path/to/folder`** to `convert_manga109_to_yolo.py` (and change paths in the small test scripts).

**Alternative:** install the official Manga109 release from [manga109.org](http://www.manga109.org/en/download.html) under the same layout; then set `data_root` to that folder.

---

## 3. Python dependencies

No `requirements.txt` is checked in; install what the code imports:

```bash
cd manga-parser
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install ultralytics opencv-python pyyaml tqdm matplotlib torch
```

- **GPU training:** install a CUDA-enabled `torch` build for your system; `train_yolo.py` uses `device=0`.  
- **CPU only:** change `device=0` to `device="cpu"` in `train_yolo.py` (training will be slow).

---

## 4. Commands to run (typical flow)

| Step | Command |
|------|--------|
| Inspect loader + stats | `python manga109_loader.py` *or* `python manga109_loader.py /path/to/manga109_root` |
| Print first N pages + preprocess compare PNG | `python preview_manga109_pages.py` / `-n 25` — default save **`outputs/preprocess_{Book}_{page:03d}.png`**. Single page: **`--book ARMS --page 80`** (or set **`SINGLE_TEST_BOOK` / `SINGLE_TEST_PAGE`** at top of that file). Optional **`--out path.png`**. Needs `opencv-python`; preprocess uses **`../cv/src/preprocess.py`**. |
| Convert to YOLO | `python convert_manga109_to_yolo.py --data_root data/raw/manga109/Manga109_released_2023_12_07 --output_dir data/processed/manga109_yolo` |
| Train YOLOv8 | `python train_yolo.py` *(after conversion; expects `data/processed/manga109_yolo/dataset.yaml`)* |

### Training model (what you already have)

**`train_yolo.py`** fine-tunes **YOLOv8** (`yolov8s.pt`). That **is** a standard academic/industry **deep neural network** (CNN). During training, PyTorch / Ultralytics runs **forward** passes (input → predictions) and **backward** passes (compute gradients and update weights) for you — you do not code “forward vs backward” manually.

There is **no separate from-scratch university demo MLP** in this repo; for detection on manga panels, a **pretrained detector** like YOLO is the usual baseline. A simple MLP would not map naturally to full-page bounding-box detection without heavy engineering.

### Train / validation / test split (reducing bias)

- **Always split by *book* (volume), not by page.** Your loader already does this: all pages from one manga stay in one split. That **prevents leakage** (near-duplicate pages from the same book in train and val) and is more important than random page splits.

- You **do not** need hand labels for “regular vs irregular” panels to get a reasonable split:
  - **Default:** `convert_manga109_to_yolo.py` uses **`--split-mode random`** (shuffle books, fixed `--seed`).
  - **Optional:** **`--split-mode stratified`** assigns books using **mean panel (`<frame>`) count per story page** per volume (from XML), so train/val/test each get a mix of panel-sparse and panel-dense books — a **proxy** when you do not know layout categories a priori.

```bash
python convert_manga109_to_yolo.py --split-mode stratified --seed 42 ...
```

Quick convert from Python (same as `test_convert.py`):

```bash
python test_convert.py
```

---

## 5. What each file is for

| File | Purpose |
|------|--------|
| **`manga109_loader.py`** | Core loader: reads **`annotations/*.xml`**, maps each page to **`images/<book>/<idx>.jpg`**, parses **`frame`** (panel), **`text`** (bubble), **`face`**, **`body`**. **`get_split()`** (random book shuffle), **`get_split_stratified_by_panel_density()`**, **`book_panel_density_proxy()`**, **`pages_for_split()`**. Run as script for stats + sample. |
| **`convert_manga109_to_yolo.py`** | Load Manga109 → **book-level** train/val/test → YOLO folders + **`dataset.yaml`**. **`--split-mode random|stratified`**. **Skips `000.jpg` (cover).** Other CLI: `--data_root`, `--output_dir`, `--no_copy_images`, ratios, `--seed`. |
| **`train_yolo.py`** | Fine-tunes **`yolov8s.pt`** on **`data/processed/manga109_yolo/dataset.yaml`** (epochs, batch, augmentations, `runs/manga109`). |
| **`tester.py`** | Minimal smoke test: constructs `Manga109Loader` on the default `data_root` and prints **`print_stats()`**. |
| **`test_convert.py`** | Calls **`convert()`** with fixed `data_root` / `output_dir` (same as a one-shot conversion). |
| **`test_labels.py`** | Prints a few **train** label `.txt` files and first line (sanity check after conversion). |
| **`test_visualize.py`** | Loads one **ARMS** page, draws **panel (blue)** and **text (green)** boxes, saves **`sample_annotations.png`**. |
| **`test.py`** | Tiny CUDA probe: prints whether **torch** sees a GPU and the device name. |
| **`preview_manga109_pages.py`** | Browse: first **N** story pages (`DEFAULT_NUM_PAGES`, **`-n`**). Single: **`--book` + `--page`** or **`SINGLE_TEST_*`** constants. Default output **`outputs/preprocess_{Book}_{page:03d}.png`**; override with **`--out`**. **`../cv/src/preprocess.py`** when `cv` sits next to `manga-parser`. |
| **`.gitignore`** | Ignores **`data/`**, **`runs/`**, **`outputs/`**, downloaded **`.pt`** weights, `__pycache__/`. |

---

## 6. License reminder

Manga109 is **gated** and **academically licensed**. Use only under the terms you agreed to on the Hub (no redistribution, proper attribution, etc.). Do not commit **`hf_` tokens** or other secrets into git.
