from collections import Counter
from pathlib import Path
import json

import cv2


source_root = Path("data/raw/ycbv/test")
dataset_root = Path("data/yolo_ycbv_real")

image_dir = dataset_root / "images" / "val"
label_dir = dataset_root / "labels" / "val"
image_dir.mkdir(parents=True, exist_ok=True)
label_dir.mkdir(parents=True, exist_ok=True)

scene_files = sorted(source_root.glob("*/scene_gt.json"))
if not scene_files:
    raise FileNotFoundError(f"No scenes found under {source_root}")

class_counts = Counter()
image_count = 0
scene_count = 0

for annotation_path in scene_files:
    scene_dir = annotation_path.parent
    annotations = json.loads(annotation_path.read_text())
    annotation_info = json.loads(
        (scene_dir / "scene_gt_info.json").read_text()
    )

    rgb_paths = sorted(
        path for path in (scene_dir / "rgb").iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )

    if not rgb_paths:
        continue

    scene_count += 1

    for image_path in rgb_paths:
        frame_key = str(int(image_path.stem))
        image_bgr = cv2.imread(str(image_path))

        if image_bgr is None:
            raise FileNotFoundError(f"Could not read {image_path}")

        image_height, image_width = image_bgr.shape[:2]
        frame_annotations = annotations[frame_key]
        frame_info = annotation_info[frame_key]

        if len(frame_annotations) != len(frame_info):
            raise ValueError(f"Annotation mismatch: {image_path}")

        label_lines = []

        for annotation, info in zip(frame_annotations, frame_info):
            if int(info["px_count_visib"]) <= 0:
                continue

            object_id = int(annotation["obj_id"])
            if not 1 <= object_id <= 21:
                raise ValueError(f"Unexpected object ID: {object_id}")

            left, top, width, height = map(float, info["bbox_visib"])

            right = min(float(image_width), left + width)
            bottom = min(float(image_height), top + height)
            left = max(0.0, left)
            top = max(0.0, top)

            if right <= left or bottom <= top:
                continue

            class_id = object_id - 1
            centre_x = (left + right) / (2 * image_width)
            centre_y = (top + bottom) / (2 * image_height)
            normalized_width = (right - left) / image_width
            normalized_height = (bottom - top) / image_height

            # Explicit separators prevent joined-number label errors.
            label_lines.append(" ".join([
                str(class_id),
                f"{centre_x:.8f}",
                f"{centre_y:.8f}",
                f"{normalized_width:.8f}",
                f"{normalized_height:.8f}",
            ]))
            class_counts[class_id] += 1

        output_stem = f"{scene_dir.name}_{image_path.stem}"
        destination = image_dir / f"{output_stem}{image_path.suffix}"

        # Link to the image instead of copying its contents.
        if not destination.is_symlink() and not destination.exists():
            destination.symlink_to(image_path.resolve())

        if destination.resolve() != image_path.resolve():
            raise RuntimeError(f"Unexpected existing file: {destination}")

        label_path = label_dir / f"{output_stem}.txt"
        label_path.write_text(
            "\n".join(label_lines) + ("\n" if label_lines else "")
        )
        image_count += 1

if image_count == 0:
    raise ValueError("No real images were found.")

yaml_lines = [
    f"path: {json.dumps(str(dataset_root.resolve()))}",
    "train: images/val",
    "val: images/val",
    "names:",
]
yaml_lines.extend(
    f"  {class_id}: obj_{class_id + 1:06d}"
    for class_id in range(21)
)

yaml_path = dataset_root / "dataset.yaml"
yaml_path.write_text("\n".join(yaml_lines) + "\n")

# Force a fresh label scan if this preparation script is rerun.
for cache_path in label_dir.parent.glob("*.cache"):
    cache_path.unlink()

print("Real scenes:", scene_count)
print("Real images:", image_count)
print("Labelled boxes:", sum(class_counts.values()))

for class_id in range(21):
    print(f"Object {class_id + 1:2d}: {class_counts[class_id]} boxes")

print("Configuration:", yaml_path.resolve())
