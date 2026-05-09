from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov8s.pt")

    model.train(
        data="data/processed/manga109_yolo/dataset.yaml",
        epochs=50,
        batch=16,
        imgsz=640,
        device=0,
        project="runs",
        name="manga109",
        exist_ok=True,
        flipud=0.0,
        fliplr=0.5,
        degrees=5.0,
        mosaic=0.8,
        workers=4,
    )