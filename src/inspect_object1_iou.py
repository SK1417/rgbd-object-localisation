from pathlib import Path
import json


def box_iou(box_a, box_b):
    """IoU for [left, top, right, bottom] boxes."""
    intersection_left = max(box_a[0], box_b[0])
    intersection_top = max(box_a[1], box_b[1])
    intersection_right = min(box_a[2], box_b[2])
    intersection_bottom = min(box_a[3], box_b[3])

    intersection_width = max(
        0.0, intersection_right - intersection_left
    )
    intersection_height = max(
        0.0, intersection_bottom - intersection_top
    )
    intersection_area = intersection_width * intersection_height

    area_a = (
        max(0.0, box_a[2] - box_a[0])
        * max(0.0, box_a[3] - box_a[1])
    )
    area_b = (
        max(0.0, box_b[2] - box_b[0])
        * max(0.0, box_b[3] - box_b[1])
    )

    union_area = area_a + area_b - intersection_area

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


# Read the saved predictions so we inspect exactly the same run.
prediction_data = json.loads(
    Path("outputs/rgbd_predictions.json").read_text()
)

scene_dir = (
    Path("data/raw/ycbv/test") / prediction_data["scene"]
)
frame_key = str(prediction_data["frame_id"])

annotations = json.loads(
    (scene_dir / "scene_gt.json").read_text()
)[frame_key]

annotation_info = json.loads(
    (scene_dir / "scene_gt_info.json").read_text()
)[frame_key]

object_id = 1

matching_indices = [
    index
    for index, annotation in enumerate(annotations)
    if int(annotation["obj_id"]) == object_id
]

if len(matching_indices) != 1:
    raise ValueError(
        "This exercise expects exactly one annotated object-1 instance."
    )

annotation_index = matching_indices[0]

left, top, width, height = map(
    float,
    annotation_info[annotation_index]["bbox_visib"],
)

# Convert BOP [left, top, width, height] to corner coordinates.
ground_truth_box = [
    left,
    top,
    left + width,
    top + height,
]

object_predictions = [
    record
    for record in prediction_data["detections"]
    if record["object_id"] == object_id
]

if len(object_predictions) != 2:
    raise ValueError(
        f"Expected two object-1 predictions; found "
        f"{len(object_predictions)}."
    )

print("Annotated visible box:", ground_truth_box)

for record in object_predictions:
    predicted_box = record["predicted_box_xyxy"]
    overlap = box_iou(predicted_box, ground_truth_box)

    print(f"\nDetection {record['detection_index']}")
    print(f"Confidence: {record['confidence']:.3f}")
    print("Predicted box:", [round(value, 2) for value in predicted_box])
    print(f"IoU with annotation: {overlap:.3f}")
    print("Meets evaluation IoU >= 0.50:", overlap >= 0.50)

mutual_iou = box_iou(
    object_predictions[0]["predicted_box_xyxy"],
    object_predictions[1]["predicted_box_xyxy"],
)

print(f"\nIoU between predictions: {mutual_iou:.3f}")
