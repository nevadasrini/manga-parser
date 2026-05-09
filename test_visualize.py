import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from manga109_loader import Manga109Loader

loader = Manga109Loader("data/raw/manga109/Manga109_released_2023_12_07")

# grab one page from a book
page = loader.books["ARMS"].pages[5]
image = cv2.imread(page.image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

fig, ax = plt.subplots(1, 1, figsize=(10, 14))
ax.imshow(image)

for ann in page.frames:
    b = ann.bbox
    rect = patches.Rectangle(
        (b.xmin, b.ymin), b.width, b.height,
        linewidth=2, edgecolor="blue", facecolor="none", label="frame"
    )
    ax.add_patch(rect)

for ann in page.texts:
    b = ann.bbox
    rect = patches.Rectangle(
        (b.xmin, b.ymin), b.width, b.height,
        linewidth=2, edgecolor="green", facecolor="none", label="text"
    )
    ax.add_patch(rect)

ax.axis("off")
ax.set_title(f"{page.book_title} — page {page.page_index}  |  blue=panel  green=text")
plt.tight_layout()
plt.savefig("sample_annotations.png", dpi=150)
plt.show()
print("saved to sample_annotations.png")