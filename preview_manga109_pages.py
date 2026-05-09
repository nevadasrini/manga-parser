#!/usr/bin/env python3
"""
Print the first N Manga109 *pages* (loader records) and run binarization on one image.

Data loading lives in manga109_loader.py (class Manga109Loader). Each row is a page:
paths to a JPG on disk + parsed XML boxes — images are not embedded in memory until
we cv2.imread them.

Preprocessing + side-by-side image reuse ../cv/src/preprocess.py when the IntroCV
layout has ``IntroCV/cv`` next to ``IntroCV/manga-parser``.

Usage (from manga-parser/):

    python preview_manga109_pages.py
    python preview_manga109_pages.py -n 25
    python preview_manga109_pages.py --book ARMS --page 80
    python preview_manga109_pages.py --book ARMS --page 80 --out outputs/my_run.png
"""

from __future__ import annotations

import argparse
import itertools
import re
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import cv2

from manga109_loader import Manga109Loader, Manga109Page, iter_story_pages

# Default when you run without ``-n`` / ``--n`` (change here or pass ``-n`` on the command line).
DEFAULT_NUM_PAGES = 1

# One-page pipeline test without CLI: set both, then ``python preview_manga109_pages.py``.
# Same as: ``--book ARMS --page 80`` (page = dataset index → file ``080.jpg``).
SINGLE_TEST_BOOK: Optional[str] = "ARMS"  # e.g. "ARMS"
SINGLE_TEST_PAGE: Optional[int] = 3  # e.g. 80


def _safe_filename_stem(book: str) -> str:
    """Book title safe for use inside a filename."""
    s = re.sub(r"[^\w.\-]+", "_", book, flags=re.UNICODE)
    return s.strip("_") or "book"


def default_preprocess_output_path(book: str, page_index: int) -> Path:
    """Default side-by-side PNG: ``outputs/preprocess_{Book}_{page:03d}.png``."""
    stem = _safe_filename_stem(book)
    return Path("outputs") / f"preprocess_{stem}_{page_index:03d}.png"


def _import_preprocess_from_cv_sibling() -> Tuple[Optional[Callable[..., Any]], Optional[Callable[..., Any]]]:
    """Return (preprocess_binary, side_by_side) from sibling cv/src, or (None, None)."""
    here = Path(__file__).resolve().parent
    cv_src = here.parent / "cv" / "src"
    prep = cv_src / "preprocess.py"
    if not prep.is_file():
        return None, None
    sys.path.insert(0, str(cv_src))
    from preprocess import preprocess_binary, side_by_side  # noqa: E402

    return preprocess_binary, side_by_side


def _find_page(loader: Manga109Loader, book: str, page_index: int) -> Optional[Manga109Page]:
    if book not in loader.books:
        keys = sorted(loader.books.keys())
        preview = ", ".join(keys[:12])
        more = f" (+{len(keys) - 12} more)" if len(keys) > 12 else ""
        print(f"ERROR: unknown book {book!r}. Examples: {preview}{more}")
        return None
    for p in loader.books[book].pages:
        if p.page_index == page_index:
            return p
    print(f"ERROR: no page index {page_index} in book {book!r}")
    return None


def main() -> int:
    p = argparse.ArgumentParser(description="Preview first pages + optional preprocess compare.")
    p.add_argument(
        "--data_root",
        default="data/raw/manga109/Manga109_released_2023_12_07",
        help="Manga109 root with images/ and annotations/",
    )
    p.add_argument(
        "-n",
        "--n",
        "--num-pages",
        type=int,
        default=DEFAULT_NUM_PAGES,
        metavar="N",
        dest="n",
        help=f"How many pages to print (default: {DEFAULT_NUM_PAGES} from DEFAULT_NUM_PAGES in this file).",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output PNG path. Default: outputs/preprocess_{Book}_{page:03d}.png for the page you preprocess.",
    )
    p.add_argument(
        "--include-cover",
        action="store_true",
        help="Include 000.jpg (page_index 0); default is to skip cover for previews/preprocess.",
    )
    p.add_argument(
        "--book",
        default=SINGLE_TEST_BOOK,
        help="With --page: run the pipeline on exactly this book (overrides SINGLE_TEST_BOOK in file).",
    )
    p.add_argument(
        "--page",
        type=int,
        default=SINGLE_TEST_PAGE,
        metavar="INDEX",
        help="With --book: dataset page index (000.jpg → 0, 080.jpg → 80). Overrides SINGLE_TEST_PAGE in file.",
    )
    args = p.parse_args()

    root = Path(args.data_root)
    if not root.is_dir():
        print(f"ERROR: data_root is not a directory: {root.resolve()}")
        return 1

    book_arg = args.book
    page_arg = args.page
    single_mode = book_arg is not None and page_arg is not None
    if (book_arg is None) ^ (page_arg is None):
        print("ERROR: specify both --book and --page for a single-page run, or neither for browse mode.")
        return 1

    print("Loading with manga109_loader.Manga109Loader ...")
    loader = Manga109Loader(str(root))
    print(f"(Loaded {len(loader)} pages from {len(loader.books)} books.)\n")

    first_readable: Optional[Path] = None
    first_page_for_preprocess: Optional[Manga109Page] = None

    if single_mode:
        page = _find_page(loader, book_arg, page_arg)
        if page is None:
            return 1
        img_path = Path(page.image_path)
        print("=" * 72)
        print("Single-page mode (--book / --page or SINGLE_TEST_* constants in this file)")
        print("=" * 72)
        print(f"  book_title   : {page.book_title}")
        print(f"  page_index   : {page.page_index}  is_cover_page={page.is_cover_page}")
        print(f"  image_path   : {img_path}")
        print(f"  file_exists  : {img_path.is_file()}")
        print(f"  width,height : {page.width} x {page.height}")
        print(f"  #frames      : {len(page.frames)}  #texts: {len(page.texts)}")
        if page.is_cover_page:
            print("  [note] cover page — preprocess still runs if the file exists.")
        if img_path.is_file():
            first_readable = img_path
            first_page_for_preprocess = page
    else:
        page_iter = iter(loader) if args.include_cover else iter_story_pages(loader)
        scope = "pages (000.jpg cover skipped)" if not args.include_cover else "pages (cover included)"

        print("=" * 72)
        print(f"First {args.n} {scope} — image = file on disk at image_path, not bytes in XML")
        print("=" * 72)

        for i, page in enumerate(itertools.islice(page_iter, args.n)):
            img_path = Path(page.image_path)
            exists = img_path.is_file()
            print(f"\n--- row {i} in this preview (dataset page_index below) ---")
            print(f"  book_title   : {page.book_title}")
            print(f"  page_index   : {page.page_index}  is_cover_page={page.is_cover_page}")
            print(f"  image_path   : {img_path}")
            print(f"  file_exists  : {exists}")
            print(f"  width,height : {page.width} x {page.height}")
            print(f"  #annotations : {len(page.annotations)}")
            print(f"  #frames      : {len(page.frames)}  #texts: {len(page.texts)}  "
                  f"#faces: {len(page.faces)}  #bodies: {len(page.bodies)}")
            if exists and first_readable is None and not page.is_cover_page:
                first_readable = img_path
                first_page_for_preprocess = page

        print("\n" + "=" * 72)
        print("Where to SEE an image: open any `image_path` above in Preview / an image viewer,")
        print("or run test_visualize.py (saves sample_annotations.png with drawn boxes).")
        print("=" * 72)

    if first_readable is None or first_page_for_preprocess is None:
        print("\nNo image to preprocess (missing file or only cover in range).")
        return 0

    preprocess_binary, side_by_side = _import_preprocess_from_cv_sibling()
    if preprocess_binary is None or side_by_side is None:
        print(
            "\nSkipping preprocess: could not find ../cv/src/preprocess.py "
            "(clone IntroCV with both `cv/` and `manga-parser/` siblings)."
        )
        return 0

    bgr = cv2.imread(str(first_readable))
    if bgr is None:
        print(f"\nCould not cv2.imread: {first_readable}")
        return 1

    result = preprocess_binary(bgr, blur_ksize=3)
    collage = side_by_side(result.grayscale, result.binary, height=800, binarize_visual=True)

    out_path = (
        Path(args.out)
        if args.out
        else default_preprocess_output_path(
            first_page_for_preprocess.book_title,
            first_page_for_preprocess.page_index,
        )
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), collage)

    print(f"\nPreprocess (adaptive threshold + opening):")
    print(f"  source : {first_readable}")
    print(f"  book   : {first_page_for_preprocess.book_title} page {first_page_for_preprocess.page_index}")
    print(f"  saved  : {out_path.resolve()}")
    print(f"  polarity_flipped: {result.polarity_flipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
