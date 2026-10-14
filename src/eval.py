from ultralytics import YOLO

model = YOLO("runs/ycbv/baseline_50/weights/best.pt")

metrics = model.val(
    data="data/yolo_ycbv/dataset.yaml",
    imgsz=640,
    batch=8,
    device=0,
    plots=True,
)

print("mAP@0.50:", metrics.box.map50)
print("mAP@0.50:0.95:", metrics.box.map)
