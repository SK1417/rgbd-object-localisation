from pathlib import Path
import json

import cv2
import numpy as np


def box_iou(box_a, box_b):
    left = max(box_a[0], box_b[0])
    top = max(box_a[1], box_b[1])
    right = min(box_a[2], box_b[2])
    bottom = min(box_a[3], box_b[3])

    intersection = max(0.0, right - left) * max(0.0, bottom - top)

    area_a = (
        max(0.0, box_a[2] - box_a[0])
        * max(0.0, box_a[3] - box_a[1])
    )
    area_b = (
        max(0.0, box_b[2] - box_b[0])
        * max(0.0, box_b[3] - box_b[1])
    )
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def mask_surface_position(mask_path, depth_m, camera_matrix):
    mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask_raw is None:
        raise FileNotFoundError(mask_path)

    if mask_raw.shape != depth_m.shape:
        raise ValueError("Mask and depth dimensions do not match.")

    selected = (mask_raw > 0) & np.isfinite(depth_m) & (depth_m > 0)
    rows, cols = np.nonzero(selected)

    if len(rows) == 0:
        return None

    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    z_m = depth_m[rows, cols]
    x_m = (cols - cx) * z_m / fx
    y_m = (rows - cy) * z_m / fy

    return np.median(
        np.column_stack((x_m, y_m, z_m)),
        axis=0,
    )


prediction_path = Path("outputs/rgbd_predictions.json")
prediction_data = json.loads(prediction_path.read_text())

scene_dir = Path("data/raw/ycbv/test") / prediction_data["scene"]
frame_id = int(prediction_data["frame_id"])
frame_key = str(frame_id)
frame_stem = f"{frame_id:06d}"

annotations = json.loads(
    (scene_dir / "scene_gt.json").read_text()
)[frame_key]
annotation_info = json.loads(
    (scene_dir / "scene_gt_info.json").read_text()
)[frame_key]

if len(annotations) != len(annotation_info):
    raise ValueError("Annotation and annotation-info counts differ.")

camera = json.loads(
    (scene_dir / "scene_camera.json").read_text()
)[frame_key]

camera_matrix = np.asarray(
    camera["cam_K"], dtype=np.float64
).reshape(3, 3)

depth_path = scene_dir / "depth" / f"{frame_stem}.png"
depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

if depth_raw is None:
    raise FileNotFoundError(depth_path)

depth_m = (
    depth_raw.astype(np.float64)
    * float(camera["depth_scale"])
    / 1000.0
)

# Only visibly annotated instances are targets for this diagnostic.
targets = []

for annotation_index, (annotation, info) in enumerate(
    zip(annotations, annotation_info)
):
    if int(info["px_count_visib"]) <= 0:
        continue

    left, top, width, height = map(float, info["bbox_visib"])

    targets.append({
        "annotation_index": annotation_index,
        "object_id": int(annotation["obj_id"]),
        "box_xyxy": [left, top, left + width, top + height],
    })

predictions = sorted(
    prediction_data["detections"],
    key=lambda record: record["confidence"],
    reverse=True,
)

iou_threshold = 0.50
matched_indices = set()
true_positives = 0
false_positives = 0
surface_gaps_cm = []
report_rows = []

print(f"Evaluation IoU threshold: {iou_threshold:.2f}")
print("Predictions:", len(predictions))
print("Visible annotated instances:", len(targets))

for prediction in predictions:
    object_id = prediction["object_id"]
    predicted_box = prediction["predicted_box_xyxy"]

    best_target = None
    best_iou = 0.0

    for target in targets:
        if target["annotation_index"] in matched_indices:
            continue
        if target["object_id"] != object_id:
            continue

        overlap = box_iou(predicted_box, target["box_xyxy"])

        if overlap > best_iou:
            best_iou = overlap
            best_target = target

    row = {
        "detection_index": prediction["detection_index"],
        "object_id": object_id,
        "confidence": prediction["confidence"],
        "best_unmatched_same_class_iou": best_iou,
        "matched_annotation_index": None,
        "surface_gap_cm": None,
    }

    if best_target is not None and best_iou >= iou_threshold:
        true_positives += 1
        annotation_index = best_target["annotation_index"]
        matched_indices.add(annotation_index)

        row["result"] = "TP"
        row["matched_annotation_index"] = annotation_index

        predicted_xyz = prediction["geometry"]["xyz_m"]
        mask_path = (
            scene_dir
            / "mask_visib"
            / f"{frame_stem}_{annotation_index:06d}.png"
        )
        reference_xyz = mask_surface_position(
            mask_path, depth_m, camera_matrix
        )

        if predicted_xyz is not None and reference_xyz is not None:
            gap_cm = float(
                np.linalg.norm(
                    np.asarray(predicted_xyz) - reference_xyz
                ) * 100
            )
            row["surface_gap_cm"] = gap_cm
            surface_gaps_cm.append(gap_cm)
    else:
        false_positives += 1
        row["result"] = "FP"

    report_rows.append(row)

    gap_text = (
        f"{row['surface_gap_cm']:.2f} cm"
        if row["surface_gap_cm"] is not None
        else "unavailable"
    )

    print(
        f"\nDetection {row['detection_index']}: "
        f"obj {object_id}, confidence {row['confidence']:.3f}"
    )
    print("  Result:", row["result"])
    print(f"  Best unmatched same-class IoU: {best_iou:.3f}")

    if row["result"] == "TP":
        print("  Surface-estimate disagreement:", gap_text)

missed_targets = [
    target for target in targets
    if target["annotation_index"] not in matched_indices
]
false_negatives = len(missed_targets)

precision = (
    true_positives / (true_positives + false_positives)
    if true_positives + false_positives > 0
    else None
)
recall = (
    true_positives / (true_positives + false_negatives)
    if true_positives + false_negatives > 0
    else None
)

print("\n--- Frame summary ---")
print("TP:", true_positives)
print("FP:", false_positives)
print("FN:", false_negatives)
print(
    "Precision:",
    f"{precision:.3f}" if precision is not None else "undefined",
)
print(
    "Recall:",
    f"{recall:.3f}" if recall is not None else "undefined",
)
print(
    "Missed object IDs:",
    [target["object_id"] for target in missed_targets],
)
print("Matched detections with available 3D comparison:", len(surface_gaps_cm))

report = {
    "scene": prediction_data["scene"],
    "frame_id": frame_id,
    "iou_threshold": iou_threshold,
    "tp": true_positives,
    "fp": false_positives,
    "fn": false_negatives,
    "precision": precision,
    "recall": recall,
    "detections": report_rows,
    "missed_targets": missed_targets,
}

report_path = Path("outputs/frame_evaluation.json")
report_path.write_text(json.dumps(report, indent=2) + "\n")
print("Saved evaluation:", report_path.resolve())
