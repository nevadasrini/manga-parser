from ultralytics import YOLO
from pathlib import Path
import random

if __name__ == '__main__':
    model = YOLO("runs/detect/runs/manga109/weights/best.pt")

    test_dir = Path("data/processed/manga109_yolo/images/test")
    test_images = list(test_dir.glob("*.jpg"))
    sample = random.sample(test_images, min(10, len(test_images)))

    model.predict(
        source=sample,
        conf=0.35,
        iou=0.45,
        save=True,
        project="results",
        name="predictions",
        exist_ok=True,
        line_width=2,
    )

    print("Done — check results/predictions/")