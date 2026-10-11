from pathlib import Path

from ultralytics import YOLO


def main():
    dataset_path = Path("data/yolo_ycbv/dataset.yaml").resolve()

    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    model = YOLO("yolov8n.pt")

    model.train(
        data=str(dataset_path),
        epochs=50,
        patience=10,
        imgsz=640,
        batch=8,
        device=0,
        workers=2,
        cache=False,
        project=str(Path("runs/ycbv").resolve()),
        name="baseline_50",
        seed=42,
        plots=True,
        save_period=-1,
        close_mosaic=10,
    )

if __name__ == "__main__":
    main()
