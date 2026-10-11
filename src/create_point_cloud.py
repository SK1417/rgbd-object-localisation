from pathlib import Path 
import json 
import cv2
import numpy as np 
import open3d as o3d
import plotly.graph_objects as go

scene_dir = Path("data/raw/ycbv/test/000048")
frame_id = 1
frame_stem = f"{frame_id:06d}"

color_path = scene_dir / "rgb" / f"{frame_stem}.png"
depth_path = scene_dir / "depth" / f"{frame_stem}.png"

color_bgr = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

camera_data = json.loads((scene_dir / "scene_camera.json").read_text())
camera = camera_data[str(frame_id)]

camera_matrix = np.asarray(camera["cam_K"], dtype=np.float64).reshape(3,3)

fx = camera_matrix[0, 0]
fy = camera_matrix[1, 1]
cx = camera_matrix[0, 2]
cy = camera_matrix[1, 2]

depth_scale = float(camera["depth_scale"])
depth_m = depth_raw.astype(np.float64) * depth_scale / 1000.0

#select only usable pizels
max_depth_m = 2.0
valid_depth = np.isfinite(depth_m) & (depth_m > 0)
keep_points = valid_depth & (depth_m <= max_depth_m)

print("Total Pixels: ", depth_raw.size)
print("Points retained within 2m: ", int(keep_points.sum()))

# backprojection 
rows, cols = np.nonzero(keep_points)

z_m = depth_m[rows, cols]
x_m = (cols - cx) * z_m/fx
y_m = (rows - cy) * z_m/fy

points_xyz = np.column_stack((x_m, y_m, z_m))
point_colors = color_rgb[rows, cols].astype(np.float64) / 255.0

point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points_xyz)
point_cloud.colors = o3d.utility.Vector3dVector(point_colors)

print("Point array shape: ", points_xyz.shape)
print("Min XYZ, metres: ", np.round(points_xyz.min(axis=0), 3))
print("Max XYZ, metres: ", np.round(points_xyz.max(axis=0), 3))

output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)
cloud_path = output_dir / "scene_000048_frame_000001.ply"

saved = o3d.io.write_point_cloud(str(cloud_path), point_cloud)

print("Saved camera-frame cloud: ", cloud_path.resolve())

camera_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1,
	origin=[0,0,0])

rng = np.random.default_rng(42)
display_count = min(30_000, len(points_xyz))
display_indices = rng.choice(len(points_xyz), size=display_count, replace=False)

display_points = points_xyz[display_indices]
display_colors = np.clip(
    point_colors[display_indices] * 255,
    0,
    255,
).astype(np.uint8)

color_strings = [
    f"rgb({int(red)},{int(green)},{int(blue)})"
    for red, green, blue in display_colors
]

figure = go.Figure(
    data=[
        go.Scatter3d(
            x=display_points[:, 0],
            y=display_points[:, 1],
            z=display_points[:, 2],
            mode="markers",
            marker=dict(size=1.5, color=color_strings),
            hovertemplate=(
                "X: %{x:.3f} m<br>"
                "Y: %{y:.3f} m<br>"
                "Z: %{z:.3f} m<extra></extra>"
            ),
        )
    ]
)

figure.update_layout(
    title="YCB-Video point cloud — camera coordinates",
    scene=dict(
        xaxis_title="X: right (m)",
        yaxis_title="Y: down (m)",
        zaxis_title="Z: forward (m)",
        aspectmode="data",
    ),
    margin=dict(l=0, r=0, b=0, t=45),
)

html_path = output_dir / "point_cloud.html"
figure.write_html(str(html_path), include_plotlyjs=True)
print("Saved interactive view:", html_path.resolve())
