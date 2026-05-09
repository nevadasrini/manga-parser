"""
convert_manga109_to_yolo.py

Converts Manga109 annotations to YOLO format for YOLOv8 fine-tuning.

Output structure:
    data/processed/manga109_yolo/
        images/
            train/   val/   test/
        labels/
            train/   val/   test/
        dataset.yaml

YOLO label format (one .txt per image, one line per annotation):
    <class_id> <x_center> <y_center> <width> <height>
    All values normalized to [0, 1].

Class mapping:
    0 → frame  (panel boundary)
    1 → text   (speech bubble / text region)

The split is done at the book level to prevent data leakage between
pages of the same volume appearing in both train and val.
"""

import shutil
import yaml
from pathlib import Path
from tqdm import tqdm

from manga109_loader import Manga109Loader, Manga109Page


CLASS_MAP = {
    "frame": 0,
    "text":  1,
}

CLASS_NAMES = ["frame", "text"]


def page_to_yolo_lines(page: Manga109Page) -> list[str]:
    """
    Convert annotations on a page to YOLO label lines.
    Skip annotations with zero area or out-of-bounds coordinates.
    """
    lines = []
    for ann in page.annotations:
        class_id = CLASS_MAP.get(ann.category)
        if class_id is None:
            continue 

        xc, yc, w, h = ann.bbox.to_yolo(page.width, page.height)

        if w <= 0 or h <= 0:
            continue

        lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return lines


def write_split(
    pages: list[Manga109Page],
    split_name: str,
    output_dir: Path,
    copy_images: bool = True,
) -> int:
    """
    Writes images and label files for one split (train/val/test).
    Returns number of pages successfully written.
    """
    img_dir = output_dir / "images" / split_name
    lbl_dir = output_dir / "labels" / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for page in tqdm(pages, desc=f"  {split_name}", leave=False):
        src = Path(page.image_path)
        if not src.exists():
            continue

        stem = f"{page.book_title}_{page.page_index:03d}"
        dst_img = img_dir / f"{stem}.jpg"
        dst_lbl = lbl_dir / f"{stem}.txt"

        if copy_images:
            shutil.copy2(src, dst_img)

        lines = page_to_yolo_lines(page)
        dst_lbl.write_text("\n".join(lines))
        written += 1

    return written


def write_dataset_yaml(output_dir: Path):
    config = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    return yaml_path


def convert(
    data_root: str,
    output_dir: str,
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    copy_images: bool  = True,
    seed: int = 42,
):
    """
    Pipeline: load Manga109 → split → write YOLO dataset.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Manga109 annotations...")
    loader = Manga109Loader(
        data_root,
        categories={"frame", "text"},  # this what we need for YOLO
    )
    loader.print_stats()

    print("\nSplitting by book...")
    test_ratio = 1.0 - train_ratio - val_ratio
    train_titles, val_titles, test_titles = loader.get_split(
        ratio=(train_ratio, val_ratio, test_ratio),
        seed=seed,
    )
    print(f"  train books : {len(train_titles)}")
    print(f"  val   books : {len(val_titles)}")
    print(f"  test  books : {len(test_titles)}")

    train_pages = loader.pages_for_split(train_titles)
    val_pages   = loader.pages_for_split(val_titles)
    test_pages  = loader.pages_for_split(test_titles)
    print(f"\n  train pages : {len(train_pages)}")
    print(f"  val   pages : {len(val_pages)}")
    print(f"  test  pages : {len(test_pages)}")

    print(f"\nWriting YOLO dataset to {output_dir} ...")
    n_train = write_split(train_pages, "train", output_dir, copy_images)
    n_val   = write_split(val_pages,   "val",   output_dir, copy_images)
    n_test  = write_split(test_pages,  "test",  output_dir, copy_images)

    yaml_path = write_dataset_yaml(output_dir)

    print(f"\nDone.")
    print(f"  train : {n_train} pages")
    print(f"  val   : {n_val} pages")
    print(f"  test  : {n_test} pages")
    print(f"  dataset.yaml → {yaml_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert Manga109 annotations to YOLO format"
    )
    parser.add_argument(
        "--data_root",
        default="data/raw/manga109/Manga109_released_2023_12_07",
        help="Path to Manga109 release directory",
    )
    parser.add_argument(
        "--output_dir",
        default="data/processed/manga109_yolo",
        help="Output directory for YOLO dataset",
    )
    parser.add_argument(
        "--no_copy_images",
        action="store_true",
        help="Skip copying images (write labels only)",
    )
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio",   type=float, default=0.15)
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    convert(
        data_root   = args.data_root,
        output_dir  = args.output_dir,
        train_ratio = args.train_ratio,
        val_ratio   = args.val_ratio,
        copy_images = not args.no_copy_images,
        seed        = args.seed,
    )