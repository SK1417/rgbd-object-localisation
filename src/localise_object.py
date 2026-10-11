from pathlib import Path
import json

import cv2
import numpy as np
import open3d as o3d


# Select the same frame we already inspected.
scene_dir = Path("data/raw/ycbv/test/000048")
frame_id = 1
annotation_index = 0
frame_stem = f"{frame_id:06d}"

# Load images.
color_path = scene_dir / "rgb" / f"{frame_stem}.png"
depth_path = scene_dir / "depth" / f"{frame_stem}.png"

color_bgr = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

if color_bgr is None or depth_raw is None:
    raise FileNotFoundError("Could not load the RGB or depth image.")

if color_bgr.shape[:2] != depth_raw.shape:
    raise ValueError("RGB and depth dimensions do not match.")

color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

# Load camera calibration.
camera_data = json.loads(
    (scene_dir / "scene_camera.json").read_text()
)
camera = camera_data[str(frame_id)]

camera_matrix = np.asarray(
    camera["cam_K"], dtype=np.float64
).reshape(3, 3)

fx = camera_matrix[0, 0]
fy = camera_matrix[1, 1]
cx = camera_matrix[0, 2]
cy = camera_matrix[1, 2]

depth_scale = float(camera["depth_scale"])
depth_m = depth_raw.astype(np.float64) * depth_scale / 1000.0

annotations = json.loads((scene_dir / "scene_gt.json").read_text())
frame_annotations = annotations[str(frame_id)]

annotation = frame_annotations[annotation_index]
object_id = int(annotation["obj_id"])

mask_path = (scene_dir / "mask_visib" / f"{frame_stem}_{annotation_index:06d}.png")
mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

object_mask = mask_raw > 0
mask_pixel_count = int(object_mask.sum())

valid_depth = np.isfinite(depth_m) & (depth_m > 0)
object_valid_mask = object_mask & valid_depth 

valid_pixel_count = int(object_valid_mask.sum())
valid_fraction = valid_pixel_count / mask_pixel_count

print("Object ID: ", object_id)
print("Annotation Index: ", annotation_index)
print("Visible mask pixels: ", mask_pixel_count)
print("Pixels with valid depth: ", valid_pixel_count)
print(f"Valid depth within visible mask: {100 * valid_fraction:.1f}%")

rows, cols = np.nonzero(object_valid_mask)
z_m = depth_m[rows, cols]
x_m = (cols - cx) * z_m/fx
y_m = (rows - cy) * z_m/fy

object_points = np.column_stack((x_m, y_m, z_m))

mean_surface_xyz = np.mean(object_points, axis=0)
median_surface_xyz = np.median(object_points, axis=0)

print(
    "Mean visible-surface XYZ, metres:",
    np.round(mean_surface_xyz, 4),
)
print(
    "Median visible-surface XYZ, metres:",
    np.round(median_surface_xyz, 4),
)
print(
    "Depth percentiles [5, 50, 95], metres:",
    np.round(np.percentile(z_m, [5, 50, 95]), 4),
)

output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)

object_cloud = o3d.geometry.PointCloud()
object_cloud.points = o3d.utility.Vector3dVector(object_points)
object_cloud.colors = o3d.utility.Vector3dVector(
    color_rgb[rows, cols].astype(np.float64) / 255.0
)

cloud_path = output_dir / f"object_{object_id:06d}_surface.ply"
if not o3d.io.write_point_cloud(str(cloud_path), object_cloud):
    raise RuntimeError(f"Could not save: {cloud_path}")

# Save an RGB overlay showing the selected visible mask.
overlay_bgr = color_bgr.copy()
highlight_bgr = np.array([0, 255, 0], dtype=np.float64)

overlay_bgr[object_mask] = (
    0.6 * color_bgr[object_mask].astype(np.float64)
    + 0.4 * highlight_bgr
).astype(np.uint8)

overlay_path = output_dir / f"object_{object_id:06d}_mask.png"
if not cv2.imwrite(str(overlay_path), overlay_bgr):
    raise RuntimeError(f"Could not save: {overlay_path}")

print("Saved object cloud:", cloud_path.resolve())
print("Saved mask overlay:", overlay_path.resolve())

#### What if only bounding box is available and not the object mask?

mask_rows, mask_cols = np.nonzero(object_mask)

left = int(mask_cols.min())
right = int(mask_cols.max()) + 1
top = int(mask_rows.min())
bottom = int(mask_rows.max()) + 1

box_mask = np.zeros_like(object_mask, dtype=bool)
box_mask[top:bottom, left:right] = True

box_valid_mask = box_mask & valid_depth
box_pixel_count = int(box_mask.sum())
box_valid_count = int(box_valid_mask.sum())

centre_u = (left + right - 1) // 2
centre_v = (top + bottom - 1) // 2

print("\n--- Bounding-box comparison ---")
print("Box [left, top, right, bottom]:", [left, top, right, bottom])
print("Box centre [u, v]:", [centre_u, centre_v])
print(
    "Box centre belongs to visible object mask:",
    bool(object_mask[centre_v, centre_u]),
)
print(
    "Visible object occupies this percentage of the box:",
    round(100 * mask_pixel_count / box_pixel_count, 1),
)
print(
    "Valid depth within box:",
    f"{100 * box_valid_count / box_pixel_count:.1f}%",
)

# Method 1: a single centre pixel.
if valid_depth[centre_v, centre_u]:
    centre_z = float(depth_m[centre_v, centre_u])
    centre_xyz = np.array([
        (centre_u - cx) * centre_z / fx,
        (centre_v - cy) * centre_z / fy,
        centre_z,
    ])
    print("Centre-pixel XYZ, metres:", np.round(centre_xyz, 4))
else:
    print("Centre-pixel estimate unavailable: invalid depth.")

# Method 2: coordinate-wise median of all valid box points.
if box_valid_count > 0:
    box_rows, box_cols = np.nonzero(box_valid_mask)
    box_z = depth_m[box_rows, box_cols]
    box_x = (box_cols - cx) * box_z / fx
    box_y = (box_rows - cy) * box_z / fy

    box_points = np.column_stack((box_x, box_y, box_z))
    box_median_xyz = np.median(box_points, axis=0)

    print("Box median XYZ, metres:", np.round(box_median_xyz, 4))

    difference_m = np.linalg.norm(
        box_median_xyz - median_surface_xyz
    )
    print(
        "Box-versus-mask median difference, centimetres:",
        round(float(difference_m * 100), 2),
    )

# Method 3: our existing mask-based estimate.
print(
    "Mask median XYZ, metres:",
    np.round(median_surface_xyz, 4),
)

# Save an image showing the box and the sampled centre pixel.
comparison_bgr = overlay_bgr.copy()
cv2.rectangle(
    comparison_bgr,
    (left, top),
    (right - 1, bottom - 1),
    (255, 0, 0),
    2,
)
cv2.circle(
    comparison_bgr,
    (centre_u, centre_v),
    4,
    (0, 0, 255),
    -1,
)

comparison_path = output_dir / "object_box_comparison.png"
if not cv2.imwrite(str(comparison_path), comparison_bgr):
    raise RuntimeError(f"Could not save: {comparison_path}")

print("Saved box comparison:", comparison_path.resolve())

print("\n--- Consistency checks ---")

check_rows, check_cols = np.nonzero(object_valid_mask)
check_z = depth_m[check_rows, check_cols]

check_points = np.column_stack((
    (check_cols - cx) * check_z / fx,
    (check_rows - cy) * check_z / fy,
    check_z,
))

print("Intrinsics [fx, fy, cx, cy]:", [fx, fy, cx, cy])
print("Object points shape:", object_points.shape)
print("Fresh points shape:", check_points.shape)

print(
    "Valid object row range:",
    [int(check_rows.min()), int(check_rows.max())],
)
print(
    "Valid object column range:",
    [int(check_cols.min()), int(check_cols.max())],
)
print(
    "Valid object pixel median [u, v]:",
    [float(np.median(check_cols)), float(np.median(check_rows))],
)

print("Fresh mean XYZ:", np.round(check_points.mean(axis=0), 6))
print("Fresh median XYZ:", np.round(np.median(check_points, axis=0), 6))

print(
    "Original points match fresh points:",
    object_points.shape == check_points.shape
    and np.allclose(object_points, check_points),
)
print(
    "Stored median matches fresh median:",
    np.allclose(
        median_surface_xyz,
        np.median(check_points, axis=0),
    ),
)

print(
    "All valid object pixels are inside box:",
    bool(np.all(box_mask[object_valid_mask])),
)
print("Valid box pixel count:", int(box_valid_mask.sum()))
print("Valid object pixel count:", int(object_valid_mask.sum()))
