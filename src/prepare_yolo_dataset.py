from collections import Counter 
from pathlib import Path 
import json 
import shutil
import cv2

raw_root = Path("data/raw/ycbv")
dataset_root = Path("data/yolo_ycbv")
preview_root = Path("outputs/label_previews")

manifest = json.loads(
    (raw_root / "training_subset_manifest.json").read_text()
)

split_scenes = {
    "train": manifest["train_scenes"],
    "val": manifest["val_scenes"]
}

class_counts = {}
image_counts = {}

for split, scene_ids in split_scenes.items():
    image_dir = dataset_root / "images" / split 
    label_dir = dataset_root / "labels" / split

    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    preview_root.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    image_count = 0
    empty_count = 0

    for scene_id in scene_ids:
        scene_dir = raw_root / "train_pbr" / scene_id

        annotations = json.loads(
            (scene_dir / "scene_gt.json").read_text()
        )
        annotations_info = json.loads(
            (scene_dir / "scene_gt_info.json").read_text()
        )

        frame_ids = manifest["frames_by_scene"][scene_id]

        for frame_index, frame_id in enumerate(frame_ids):
            frame_stem = f"{frame_id:06d}"
            frame_key = str(frame_id)
            source_image = scene_dir / "rgb" / f"{frame_stem}.jpg"

            image_bgr = cv2.imread(str(source_image))
            if image_bgr is None:
                raise FileNotFoundError(f"Could not read {source_image}")

            image_height, image_width = image_bgr.shape[:2]
            frame_annotations = annotations[frame_key]
            frame_info = annotations_info[frame_key]

            if len(frame_annotations) != len(frame_info):
                raise ValueError("")

            label_lines = []
            preview_bgr = image_bgr.copy()

            for annotation, info in zip(frame_annotations, frame_info):
                object_id = int(annotation["obj_id"])
                if not 1 <= object_id <= 21:
                    raise ValueError("")

                if int(info["px_count_visib"]) <= 0:
                    continue

                left, top, box_width, box_height = map(
                    float, info["bbox_visib"]
                )

                right = min(float(image_width), left + box_width)
                bottom = min(float(image_height), top + box_height)
                left = max(0.0, left)
                top = max(0.0, top)

                if right <= left or bottom <= top:
                    continue 

                class_id = object_id - 1
                centre_x = ((left + right) / 2) / image_width
                centre_y = ((top + bottom) / 2) / image_height 
                width_normalized = (right - left) / image_width 
                height_normalized = (bottom - top) / image_height

                normalized_box = [
                    centre_x,
                    centre_y, 
                    width_normalized,
                    height_normalized
                ]

                if not all(0 <= value <= 1 for value in normalized_box):
                    raise ValueError("")

                label_lines.append(
                    f"{class_id} "
                    f"{centre_x:.8f} "
	            f"{centre_y:.8f} "
                    f"{width_normalized:.8f} "
		    f"{height_normalized:.8f} "
                )
                counts[class_id] += 1

                cv2.rectangle(
                    preview_bgr,
                    (int(left), int(top)),
                    (int(right) - 1, int(bottom) - 1),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    preview_bgr, 
                    f"obj {object_id} / class {class_id}",
                    (int(left), max(15, int(top) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    1
                )

            output_stem = f"{scene_id}_{frame_stem}"
            destination_image = image_dir / f"{output_stem}.jpg"
            shutil.copy2(source_image, destination_image)

            label_path = label_dir / f"{output_stem}.txt"
            label_path.write_text(
                "\n".join(label_lines) + ("\n" if label_lines else "")
            )

            if not label_lines:
                empty_count += 1

            if frame_index in {0, len(frame_ids) // 2, len(frame_ids) - 1}:
                preview_path = (preview_root / f"{split}_{output_stem}.jpg")

            image_count += 1

    class_counts[split] = counts
    image_counts[split] = image_count
    print(
        f"{split}: {image_count} images, "
        f"{sum(counts.values())} boxes, "
        f"{empty_count} empty-label images"
    )

yaml_lines = [
    f"path: {json.dumps(str(dataset_root.resolve()))}",
    "train: images/train",
    "val: images/val",
    "names:",
]
yaml_lines.extend(
    f"  {class_id}: obj_{class_id + 1:06d}"
    for class_id in range(21)
)

yaml_path = dataset_root / "dataset.yaml"
yaml_path.write_text("\n".join(yaml_lines) + "\n")

print("\nObject ID | YOLO class | Train boxes | Val boxes")
print("-" * 52)

for class_id in range(21):
    print(
        f"{class_id + 1:9d} | "
        f"{class_id:10d} | "
        f"{class_counts['train'][class_id]:11d} | "
        f"{class_counts['val'][class_id]:9d}"
    )

missing_train = [
    class_id + 1 for class_id in range(21)
    if class_counts["train"][class_id] == 0
]
missing_val = [
    class_id + 1 for class_id in range(21)
    if class_counts["val"][class_id] == 0
]

print("\nObjects missing from training:", missing_train)
print("Objects missing from validation:", missing_val)
print("Dataset configuration:", yaml_path.resolve())
print("Label previews:", preview_root.resolve())
