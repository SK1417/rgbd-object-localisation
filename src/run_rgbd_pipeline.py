from pathlib import Path
import json

import cv2
import numpy as np
from ultralytics import YOLO

from rgbd_geometry import localise_box


def main():
    scene_dir = Path("data/raw/ycbv/test/000048")
    frame_id = 1
    frame_stem = f"{frame_id:06d}"

    weights_path = Path(
        "runs/ycbv/coco_head_only_9800/weights/best.pt"
    )
    color_path = scene_dir / "rgb" / f"{frame_stem}.png"
    depth_path = scene_dir / "depth" / f"{frame_stem}.png"

    if not weights_path.exists():
        raise FileNotFoundError(weights_path)

    color_bgr = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

    if color_bgr is None or depth_raw is None:
        raise FileNotFoundError("Could not load RGB or depth.")

    if color_bgr.shape[:2] != depth_raw.shape:
        raise ValueError("RGB and depth dimensions do not match.")

    camera_data = json.loads(
        (scene_dir / "scene_camera.json").read_text()
    )
    camera = camera_data[str(frame_id)]

    camera_matrix = np.asarray(
        camera["cam_K"], dtype=np.float64
    ).reshape(3, 3)

    depth_m = (
        depth_raw.astype(np.float64)
        * float(camera["depth_scale"])
        / 1000.0
    )

    model = YOLO(str(weights_path))
    prediction = model.predict(
        source=color_bgr,
        imgsz=640,
        conf=0.25,
        iou=0.50,
        device=0,
        verbose=False,
    )[0]

    annotated_bgr = color_bgr.copy()
    records = []

    print("Detections:", len(prediction.boxes))

    for detection_index, detection in enumerate(prediction.boxes):
        class_id = int(detection.cls.item())
        object_id = class_id + 1
        confidence = float(detection.conf.item())

        # Ultralytics returns these coordinates in original-image pixels.
        box_xyxy = detection.xyxy[0].cpu().numpy()

        geometry = localise_box(
            box_xyxy,
            depth_m,
            camera_matrix,
        )

        record = {
            "detection_index": detection_index,
            "object_id": object_id,
            "class_id": class_id,
            "confidence": confidence,
            "predicted_box_xyxy": box_xyxy.tolist(),
            "geometry": geometry,
        }
        records.append(record)

        print(
            f"\nDetection {detection_index}: "
            f"object {object_id}, confidence {confidence:.3f}"
        )
        print("  Geometry status:", geometry["status"])
        print("  XYZ, metres:", geometry["xyz_m"])
        print(
            "  Valid depth fraction:",
            round(geometry["valid_fraction"], 3),
        )

        if geometry["status"] != "ok":
            print("  Reason:", geometry["reason"])

        left, top, right, bottom = geometry["box_xyxy"]
        color = (
            (0, 255, 0)
            if geometry["status"] == "ok"
            else (0, 165, 255)
        )

        cv2.rectangle(
            annotated_bgr,
            (left, top),
            (right - 1, bottom - 1),
            color,
            2,
        )

        if geometry["xyz_m"] is None:
            position_text = "XYZ unavailable"
        else:
            x_m, y_m, z_m = geometry["xyz_m"]
            position_text = f"XYZ {x_m:.2f},{y_m:.2f},{z_m:.2f}m"

        label = (
            f"ID: {object_id} | conf: {confidence:.2f} | {position_text}"
        )
        cv2.putText(
            annotated_bgr,
            label,
            (left, max(15, top - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            color,
            1,
        )

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    image_output = output_dir / "rgbd_predictions.png"
    json_output = output_dir / "rgbd_predictions.json"

    if not cv2.imwrite(str(image_output), annotated_bgr):
        raise RuntimeError(f"Could not save {image_output}")

    json_output.write_text(
        json.dumps(
            {
                "scene": scene_dir.name,
                "frame_id": frame_id,
                "coordinate_frame": "camera: X right, Y down, Z forward",
                "position_definition": (
                    "coordinate-wise median of valid depth points "
                    "inside the predicted box"
                ),
                "detections": records,
            },
            indent=2,
        ) + "\n"
    )

    print("\nSaved image:", image_output.resolve())
    print("Saved results:", json_output.resolve())


if __name__ == "__main__":
    main()
