from pathlib import Path
import json
import shutil
import zipfile


archive_path = Path("data/downloads/ycbv_train_pbr.zip")
output_root = Path("data/raw/ycbv")

train_scenes = ["000000", "000001", "000002", "000003"]
val_scenes = ["000004"]
sample_stride = 5

metadata_names = {
    "scene_camera.json",
    "scene_gt.json",
    "scene_gt_info.json",
}

selected_scenes = set(train_scenes + val_scenes)
selected_entries = []
selected_frames = {scene_id: [] for scene_id in selected_scenes}

with zipfile.ZipFile(archive_path) as archive:
    for entry in archive.infolist():
        if entry.is_dir():
            continue

        parts = Path(entry.filename).parts

        # Expected paths:
        # train_pbr/SCENE/scene_gt.json
        # train_pbr/SCENE/rgb/FRAME.jpg
        if len(parts) < 3 or parts[0] != "train_pbr":
            continue

        scene_id = parts[1]
        if scene_id not in selected_scenes:
            continue

        is_metadata = (
            len(parts) == 3
            and parts[2] in metadata_names
        )

        is_selected_rgb = (
            len(parts) == 4
            and parts[2] == "rgb"
            and Path(parts[3]).suffix.lower() in {".jpg", ".png", ".jpeg"}
            and Path(parts[3]).stem.isdigit()
            and int(Path(parts[3]).stem) % sample_stride == 0
        )

        if is_metadata or is_selected_rgb:
            selected_entries.append(entry)

        if is_selected_rgb:
            selected_frames[scene_id].append(
                int(Path(parts[3]).stem)
            )

    # Validate the selection before writing anything.
    selected_names = {entry.filename for entry in selected_entries}

    for scene_id in sorted(selected_scenes):
        if len(selected_frames[scene_id]) != 200:
            raise ValueError(
                f"Expected 200 RGB images in scene {scene_id}; "
                f"found {len(selected_frames[scene_id])}."
            )

        for metadata_name in metadata_names:
            expected_path = f"train_pbr/{scene_id}/{metadata_name}"
            if expected_path not in selected_names:
                raise FileNotFoundError(
                    f"Archive is missing {expected_path}"
                )

    required_bytes = sum(entry.file_size for entry in selected_entries)

    print("Selected RGB images:", sum(map(len, selected_frames.values())))
    print("Selected files including metadata:", len(selected_entries))
    print(f"Uncompressed size: {required_bytes / 1024**2:.1f} MiB")

    # Only selected files are decompressed.
    for entry in selected_entries:
        destination = output_root / entry.filename
        destination.parent.mkdir(parents=True, exist_ok=True)

        # Avoid silently overwriting an existing file.
        if destination.exists():
            if destination.stat().st_size != entry.file_size:
                raise RuntimeError(
                    f"Existing file has an unexpected size: {destination}"
                )
            continue

        with archive.open(entry) as source:
            with destination.open("wb") as target:
                shutil.copyfileobj(source, target)

    manifest = {
        "source_archive": str(archive_path),
        "train_scenes": train_scenes,
        "val_scenes": val_scenes,
        "sample_stride": sample_stride,
        "frames_by_scene": {
            scene_id: sorted(frame_ids)
            for scene_id, frame_ids in sorted(selected_frames.items())
        },
    }

    manifest_path = output_root / "training_subset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

print("Extraction complete.")
print("Manifest:", manifest_path.resolve())

for scene_id in sorted(selected_scenes):
    split = "train" if scene_id in train_scenes else "validation"
    print(
        f"Scene {scene_id}: "
        f"{len(selected_frames[scene_id])} images → {split}"
    )
