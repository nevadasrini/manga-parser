"""
manga109_loader.py

Loads the manga109 dataset

Annotation schema:
    <frame>  — panel boundary        
    <text>   — speech bubble / text  
    <face>   — character face bounding box
    <body>   — character body bounding box

All elements use xmin/ymin/xmax/ymax (absolute pixels, xyxy format).
"""

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


@dataclass
class Manga109Book:
    title: str
    pages: List[Manga109Page] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.pages)

    def __iter__(self):
        return iter(self.pages)

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
        self.data_root     = Path(data_root)
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
        Split book titles into train/val/test sets and return (train_titles, val_titles, test_titles)
        """
        import random
        titles = sorted(self.books.keys())
        rng = random.Random(seed)
        rng.shuffle(titles)

        n = len(titles)
        train_end = int(n * ratio[0])
        val_end   = train_end + int(n * ratio[1])

        return titles[:train_end], titles[train_end:val_end], titles[val_end:]

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

    # show samples from book
    for book in loader.books.values():
        page = book.pages[0]
        print(f"\nSample — {book.title}, page {page.page_index}")
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