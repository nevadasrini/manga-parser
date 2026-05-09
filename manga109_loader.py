"""
manga109_loader.py

Loads the manga109 dataset

Annotation schema:
    <frame>  — panel boundary        
    <text>   — speech bubble / text  
    <face>   — character face bounding box
    <body>   — character body bounding box

All elements use xmin/ymin/xmax/ymax (absolute pixels, xyxy format).

Cover pages:
    Image files are named ``{page_index:03d}.jpg``. Index **0** is the volume cover
    (**``000.jpg``**). The Manga109 XML has **no** dedicated ``is_cover`` attribute.
    In practice ``<page index="0" .../>`` is often **empty** (no ``<frame>`` / ``<text>``
    children), which is a useful heuristic but **not guaranteed** for every book—so
    code here treats **``page_index == 0``** as the cover by convention.
"""

# Image filename ``000.jpg`` — volume cover; skip for training / preprocessing pipelines.
MANGA109_COVER_PAGE_INDEX = 0

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple


@dataclass
class BBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def width(self) -> int:
        return self.xmax - self.xmin

    @property
    def height(self) -> int:
        return self.ymax - self.ymin

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_xywh(self) -> Tuple[int, int, int, int]:
        """Convert to (x, y, w, h) COCO format."""
        return (self.xmin, self.ymin, self.width, self.height)

    def to_yolo(self, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
        """Convert to normalized YOLO (xc, yc, w, h) format."""
        xc = (self.xmin + self.xmax) / 2 / img_w
        yc = (self.ymin + self.ymax) / 2 / img_h
        w  = self.width  / img_w
        h  = self.height / img_h
        return (
            max(0.0, min(1.0, xc)),
            max(0.0, min(1.0, yc)),
            max(0.0, min(1.0, w)),
            max(0.0, min(1.0, h)),
        )


@dataclass
class Manga109Annotation:
    ann_id: str
    category: str     
    bbox: BBox
    character_id: Optional[str] = None   


@dataclass
class Manga109Page:
    book_title: str
    page_index: int     
    image_path: str
    width: int
    height: int
    annotations: List[Manga109Annotation] = field(default_factory=list)

    @property
    def frames(self) -> List[Manga109Annotation]:
        """Panel boundary annotations."""
        return [a for a in self.annotations if a.category == "frame"]

    @property
    def texts(self) -> List[Manga109Annotation]:
        """Speech bubble / text region annotations."""
        return [a for a in self.annotations if a.category == "text"]

    @property
    def faces(self) -> List[Manga109Annotation]:
        return [a for a in self.annotations if a.category == "face"]

    @property
    def bodies(self) -> List[Manga109Annotation]:
        return [a for a in self.annotations if a.category == "body"]

    @property
    def image_name(self) -> str:
        return f"{self.page_index:03d}.jpg"

    @property
    def is_cover_page(self) -> bool:
        """Volume cover (``000.jpg``). Manga109 XML does not mark this explicitly."""
        return self.page_index == MANGA109_COVER_PAGE_INDEX


def has_panel_or_text_annotations(page: Manga109Page) -> bool:
    """
    True if the page has at least one panel or speech-bubble annotation.

    Covers often have **neither** (empty ``<page>`` in XML), but an empty page is
    not always a cover—use together with :py:attr:`Manga109Page.is_cover_page` when needed.
    """
    return bool(page.frames or page.texts)


def iter_story_pages(loader: "Manga109Loader"):
    """Iterate all pages except the volume cover (``000.jpg`` / ``page_index == 0``)."""
    for page in loader:
        if not page.is_cover_page:
            yield page


@dataclass
class Manga109Book:
    title: str
    pages: List[Manga109Page] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.pages)

    def __iter__(self):
        return iter(self.pages)

def resolve_manga109_root(data_root: Path) -> Path:
    """
    If ``data_root/annotations`` is missing, use the first immediate subdirectory that
    contains ``annotations/``. Hugging Face ``hf download`` often adds one extra folder
    level (e.g. ``.../Manga109_released_2023_12_07/Manga109_released_2023_12_07/``).
    """
    root = Path(data_root)
    try:
        root = root.resolve()
    except OSError:
        pass
    if (root / "annotations").is_dir():
        return root
    if not root.is_dir():
        return root
    for child in sorted(root.iterdir()):
        if child.is_dir() and not child.name.startswith("."):
            if (child / "annotations").is_dir():
                found = child.resolve()
                print(
                    f"[Manga109Loader] no annotations under {Path(data_root)!s}; "
                    f"using nested root {found}"
                )
                return found
    return root


class Manga109Loader:
    """
    loads manga109 annotations and maps them to image paths.
    """

    ALL_CATEGORIES = {"frame", "text", "face", "body"}

    def __init__(
        self,
        data_root: str,
        books: Optional[List[str]] = None,
        annotation_dir: str = "annotations",
        categories: Optional[set] = None,
    ):
        self.data_root     = resolve_manga109_root(Path(data_root))
        self.image_dir     = self.data_root / "images"
        self.ann_dir       = self.data_root / annotation_dir
        self.categories    = categories or self.ALL_CATEGORIES

        self.books: Dict[str, Manga109Book] = {}
        self._pages: List[Manga109Page] = []   

        available = self._discover_books()
        to_load   = books if books else available

        for title in to_load:
            if title not in available:
                print(f"  [warn] book not found, skipping: {title}")
                continue
            book = self._load_book(title)
            self.books[title] = book
            self._pages.extend(book.pages)

    def _discover_books(self) -> List[str]:
        """Return sorted list of book titles found in the annotations directory."""
        if not self.ann_dir.exists():
            raise FileNotFoundError(
                f"Annotation directory not found: {self.ann_dir}"
            )
        return sorted(
            p.stem for p in self.ann_dir.glob("*.xml")
        )

    def _parse_bbox(self, elem) -> BBox:
        return BBox(
            xmin=int(elem.attrib["xmin"]),
            ymin=int(elem.attrib["ymin"]),
            xmax=int(elem.attrib["xmax"]),
            ymax=int(elem.attrib["ymax"]),
        )

    def _load_book(self, title: str) -> Manga109Book:
        xml_path = self.ann_dir / f"{title}.xml"
        tree = ET.parse(xml_path)
        root = tree.getroot()

        book = Manga109Book(title=title)
        pages_elem = root.find("pages")
        if pages_elem is None:
            return book

        for page_elem in pages_elem.findall("page"):
            index  = int(page_elem.attrib["index"])
            width  = int(page_elem.attrib["width"])
            height = int(page_elem.attrib["height"])

            # Image path: images/<title>/<index:03d>.jpg
            image_path = str(self.image_dir / title / f"{index:03d}.jpg")

            page = Manga109Page(
                book_title=title,
                page_index=index,
                image_path=image_path,
                width=width,
                height=height,
            )

            for category in self.ALL_CATEGORIES:
                if category not in self.categories:
                    continue
                for ann_elem in page_elem.findall(category):
                    ann = Manga109Annotation(
                        ann_id=ann_elem.attrib.get("id", ""),
                        category=category,
                        bbox=self._parse_bbox(ann_elem),
                        character_id=ann_elem.attrib.get("character"),
                    )
                    page.annotations.append(ann)

            book.pages.append(page)

        return book


    def __len__(self) -> int:
        return len(self._pages)

    def __iter__(self):
        return iter(self._pages)

    def get_book(self, title: str) -> Manga109Book:
        return self.books[title]

    def get_split(
        self,
        ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        seed: int = 42,
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Split **book titles** (not pages) into train/val/test — avoids leakage (pages from the
        same volume appearing in both train and val).
        """
        import random
        titles = sorted(self.books.keys())
        rng = random.Random(seed)
        rng.shuffle(titles)

        n = len(titles)
        train_end = int(n * ratio[0])
        val_end   = train_end + int(n * ratio[1])

        return titles[:train_end], titles[train_end:val_end], titles[val_end:]

    def book_panel_density_proxy(self, title: str) -> float:
        """
        Proxy for “how panel-heavy” a volume is: mean ``<frame>`` count per **story** page
        (excludes cover). Use for **stratified** book splits when you do not have manual
        layout-category labels — balances easy vs dense books across train/val/test.
        """
        book = self.books.get(title)
        if not book:
            return 0.0
        counts: List[int] = []
        for p in book.pages:
            if p.is_cover_page:
                continue
            counts.append(len(p.frames))
        return float(sum(counts) / len(counts)) if counts else 0.0

    def get_split_stratified_by_panel_density(
        self,
        ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        seed: int = 42,
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Same book-level train/val/test sizes as :meth:`get_split`, but assigns books using a
        **greedy fill** on titles sorted by :meth:`book_panel_density_proxy` so each split
        gets a similar mix of low- and high-density volumes (reduces bias vs one random shuffle
        when some books are much more panel-rich than others).
        """
        titles = sorted(self.books.keys())
        n = len(titles)
        if n == 0:
            return [], [], []

        # Same counts as :meth:`get_split` (integer truncation).
        train_n = int(n * ratio[0])
        val_n = int(n * ratio[1])
        test_n = n - train_n - val_n

        scored = [(self.book_panel_density_proxy(t), t) for t in titles]
        scored.sort(key=lambda x: (x[0], x[1]))

        train: List[str] = []
        val: List[str] = []
        test: List[str] = []
        splits = (train, val, test)
        targets = (train_n, val_n, test_n)
        counts = [0, 0, 0]

        for _proxy, title in scored:
            deficits = [targets[i] - counts[i] for i in range(3)]
            j = max(range(3), key=lambda i: deficits[i])
            splits[j].append(title)
            counts[j] += 1

        return train, val, test

    def pages_for_split(
        self,
        split_titles: List[str],
    ) -> List[Manga109Page]:
        pages = []
        for title in split_titles:
            if title in self.books:
                pages.extend(self.books[title].pages)
        return pages

    # print book stats
    def print_stats(self):
        total_pages  = len(self._pages)
        total_frames = sum(len(p.frames) for p in self._pages)
        total_texts  = sum(len(p.texts)  for p in self._pages)
        total_faces  = sum(len(p.faces)  for p in self._pages)
        total_bodies = sum(len(p.bodies) for p in self._pages)

        print(f"Manga109 Dataset Statistics")
        print(f"{'─' * 40}")
        print(f"  Books loaded   : {len(self.books)}")
        print(f"  Pages total    : {total_pages}")
        print(f"  Frames (panels): {total_frames}")
        print(f"  Text regions   : {total_texts}")
        print(f"  Faces          : {total_faces}")
        print(f"  Bodies         : {total_bodies}")
        avg = total_frames / total_pages if total_pages else 0
        print(f"  Avg panels/page: {avg:.1f}")



if __name__ == "__main__":
    import sys

    data_root = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "data/raw/manga109/Manga109_released_2023_12_07"
    )

    loader = Manga109Loader(data_root)
    loader.print_stats()

    # show samples from book (skip 000.jpg cover)
    for book in loader.books.values():
        page = next((p for p in book.pages if not p.is_cover_page), book.pages[0])
        print(f"\nSample — {book.title}, page {page.page_index} (cover skipped if present)")
        print(f"  Image path : {page.image_path}")
        print(f"  Dimensions : {page.width} x {page.height}")
        print(f"  Panels     : {len(page.frames)}")
        print(f"  Text       : {len(page.texts)}")
        if page.frames:
            f = page.frames[0]
            print(f"  First panel: xmin={f.bbox.xmin} ymin={f.bbox.ymin} "
                  f"xmax={f.bbox.xmax} ymax={f.bbox.ymax}")
        break

    # train/val test split
    train, val, test = loader.get_split()
    print(f"\nSplit (by book): train={len(train)}, val={len(val)}, test={len(test)}")