import numpy as np 

def localise_box(
        box_xyxy,
        depth_m,
        camera_matrix,
        min_valid_pixels = 30,
        min_valid_fraction = 0.10
):
    depth_m = np.asarray(depth_m)
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
    box = np.asarray(box_xyxy, dtype=np.float64)

    if depth_m.ndim != 2:
        raise ValueError("depth_m must be a 2D array.")

    if camera_matrix.shape != (3, 3):
        raise ValueError("camera_matrix must have shape (3, 3).")

    if not np.isfinite(camera_matrix).all():
        raise ValueError("Camera intrinsics must be finite.")

    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError("Box must contain four finite coordinates.")

    if min_valid_pixels < 1:
        raise ValueError("min_valid_pixels must be at least 1.")

    if not 0 <= min_valid_fraction <= 1:
        raise ValueError("min_valid_fraction must be between 0 and 1.")

    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    if fx <= 0 or fy <= 0:
        raise ValueError("Focal lengths must be positive.")

    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("Box must have positive width and height.")

    height, width = depth_m.shape

    # Round outward, then clip to image boundaries.
    left = int(np.clip(np.floor(box[0]), 0, width))
    top = int(np.clip(np.floor(box[1]), 0, height))
    right = int(np.clip(np.ceil(box[2]), 0, width))
    bottom = int(np.clip(np.ceil(box[3]), 0, height))

    result = {
        "status": "unavailable",
        "reason": None,
        "box_xyxy": [left, top, right, bottom],
        "xyz_m": None,
        "valid_pixels": 0,
        "valid_fraction": 0.0,
        "depth_percentiles_m": None,
    }

    if right <= left or bottom <= top:
        result["reason"] = "Box does not overlap the image"
        return result 

    depth_crop = depth_m[top:bottom, left:right]
    valid = np.isfinite(depth_crop) & (depth_crop > 0)

    valid_count = int(valid.sum())
    valid_fraction = valid_count / depth_crop.size

    result["valid_pixels"] = valid_count
    result["valid_fraction"] = float(valid_fraction)

    if valid_count < min_valid_pixels:
        result["reason"] = "Too few valid depth pixels"
        return result

    if valid_fraction < min_valid_fraction:
        result['reason'] = 'Valid depth fraction is below the threshold'
        return result 

    crop_rows, crop_cols = np.nonzero(valid)

    image_rows = crop_rows + top 
    image_cols = crop_cols + left

    z_m = depth_crop[crop_rows, crop_cols]
    x_m = (image_cols - cx) * z_m / fx
    y_m = (image_rows - cy) * z_m / fy

    points_xyz = np.column_stack((x_m, y_m, z_m))

    result["status"] = "ok"
    result['xyz_m'] = np.median(points_xyz, axis=0).tolist()
    result['depth_percentiles_m'] = np.percentile(z_m, [5, 50, 95]).tolist()

    return result 

