from pathlib import Path 
import json 
import cv2 
import numpy as np 

scene_dir = Path("data/raw/ycbv/test/000048")
frame_id = 1
frame_stem = f"{frame_id:06d}"

depth_path = scene_dir / "depth" / f"{frame_stem}.png"
depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

camera_data = json.loads((scene_dir / "scene_camera.json").read_text())
camera = camera_data[str(frame_id)]

camera_matrix = np.asarray(camera["cam_K"], dtype=np.float64).reshape(3,3)

fx = camera_matrix[0, 0]
fy = camera_matrix[1, 1]
cx = camera_matrix[0, 2]
cy = camera_matrix[1, 2]

depth_m = (depth_raw.astype(np.float64)
           * float(camera["depth_scale"])
           / 1000.0
)
valid_depth = np.isfinite(depth_m) & (depth_m > 0)

annotations = json.loads(
    (scene_dir / "scene_gt.json").read_text()
)[str(frame_id)]

def median_surface_position(selection_mask):
    selected_valid = selection_mask & valid_depth
    rows, cols = np.nonzero(selected_valid)

    if len(rows) == 0:
        return None

    z_m = depth_m[rows,cols]
    x_m = (cols - cx) * z_m / fx
    y_m = (rows - cy) * z_m / fy

    points_xyz = np.column_stack((x_m, y_m, z_m))
    return np.median(points_xyz, axis=0)

print(
    "Object | Mask valid % | Box occupancy % | "
    "Centre on object | Centre depth valid | Box-mask gap cm"
)
print("-" * 105)

for annotation_index, annotation in enumerate(annotations):
    object_id = int(annotation["obj_id"])
    mask_path = (
        scene_dir 
        / "mask_visib" 
        / f"{frame_stem}_{annotation_index:06d}.png"
    )

    mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    object_mask = mask_raw > 0
    mask_count = int(object_mask.sum())

    if mask_count == 0:
        print(f"{object_id:6d} | No visible mask pixels")
        continue

    mask_rows, mask_cols = np.nonzero(object_mask)

    left = int(mask_cols.min())
    right = int(mask_cols.max()) + 1
    top = int(mask_rows.min())
    bottom = int(mask_rows.max()) + 1

    box_mask = np.zeros_like(object_mask)
    box_mask[top:bottom, left:right] = True

    centre_u = (left+right - 1) // 2
    centre_v = (top+bottom - 1) // 2

    mask_valid_percent = (
        100 * np.count_nonzero(object_mask & valid_depth) / mask_count
    )
    occupancy_percent = 100 * mask_count / int(box_mask.sum())

    centre_on_object = bool(object_mask[centre_v, centre_u])
    centre_depth_valid = bool(valid_depth[centre_v, centre_u])

    mask_xyz = median_surface_position(object_mask)
    box_xyz = median_surface_position(box_mask)

    if mask_xyz is None or box_xyz is None:
        gap_text = "unavailable"
    else:
        gap_cm = float(np.linalg.norm(box_xyz - mask_xyz) * 100)
        gap_text = f"{gap_cm:2f}"

    print(
        f"{object_id:6d} | "
        f"{mask_valid_percent:12.1f} | "
        f"{occupancy_percent:15.1f} | "
        f"{str(centre_on_object):16s} | "
        f"{str(centre_depth_valid):18s} | "
        f"{gap_text}"
    )

from rgbd_geometry import localise_box

print("\n --- Reusable function: obj 1 ---")

mask_path = scene_dir / "mask_visib" / f"{frame_stem}_000000.png"
mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) 

if mask_raw is None:
    raise FileNotFoundError(f"Could not load: {mask_path}")
    
object_mask = mask_raw > 0
mask_rows, mask_cols = np.nonzero(object_mask)

box_xyxy = [
    int(mask_cols.min()),
    int(mask_rows.min()),
    int(mask_cols.max()) + 1, 
    int(mask_rows.max()) + 1
]
result = localise_box(box_xyxy, depth_m, camera_matrix)

print("Status:", result["status"])
print("Reason:", result["reason"])
print("XYZ, metres:", result["xyz_m"])
print("Valid pixels:", result["valid_pixels"])
print("Valid depth fraction:", round(result["valid_fraction"], 4))
print("Depth percentiles, metres:", result["depth_percentiles_m"])

missing_depth = np.zeros_like(depth_m)
missing_result = localise_box(
    box_xyxy,
    missing_depth,
    camera_matrix,
)

print("\n--- Missing-depth example ---")
print("Status:", missing_result["status"])
print("Reason:", missing_result["reason"])
print("XYZ:", missing_result["xyz_m"])