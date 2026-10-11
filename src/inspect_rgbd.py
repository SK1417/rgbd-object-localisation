from pathlib import Path 
import matplotlib 
matplotlib.use("Agg")
import matplotlib.pyplot as plt 
import numpy as np 
import cv2
import json

scene_dir = Path("data/raw/ycbv/test/000048")
frame_id = 1
frame_stem = f"{frame_id:06d}"

color_path = scene_dir / "rgb" / f"{frame_stem}.png"
depth_path = scene_dir / "depth" / f"{frame_stem}.png"

output_dir = Path("outputs")
output_dir.mkdir(parents = True, exist_ok=True)

color_bgr = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

camera_data = json.loads((scene_dir / "scene_camera.json").read_text())
camera = camera_data[str(frame_id)]

camera_matrix = np.asarray(camera["cam_K"], dtype = np.float64).reshape(3,3)
fx = camera_matrix[0,0]
fy = camera_matrix[1,1]
cx = camera_matrix[0,2]
cy = camera_matrix[1,2]

depth_scale = float(camera["depth_scale"])
depth_m = depth_raw.astype(np.float64) * depth_scale/1000.0
valid_depth = np.isfinite(depth_m) & (depth_m > 0)

# picking the valid pixel closest to principal point
valid_rows, valid_cols = np.nonzero(valid_depth)

pixel_distance_squared = ((valid_cols - cx)**2 + (valid_rows - cy)**2)
selected_index = np.argmin(pixel_distance_squared)

u = int(valid_cols[selected_index])
v = int(valid_rows[selected_index])

# backprojection of the pixel 

z_m = float(depth_m[v,u])
x_m = (u-cx)*z_m/fx
y_m = (v-cy)*z_m/fy

print(f"Selected pixel: u={u}, v={v}")
print(f"Stored depth: {depth_raw[v,u]}")
print(f"Camera XYZ, metres: [{x_m:.6f}, {y_m:.6f}, {z_m:.6f}]")

distance_m = np.linalg.norm([x_m, y_m, z_m])
print(f"Straight-line distance, metres: {distance_m:.6f}")

#Projecting back to image as a check 
u_check = fx * x_m / z_m + cx
v_check = fy * y_m / z_m + cy

print(f"Reprojected pixel: u={u_check:.6f}, v={v_check:.6f}")

depth_display = np.ma.masked_where(~valid_depth, depth_m)

depth_cmap = plt.get_cmap("viridis").copy()
depth_cmap.set_bad("black")

display_min, display_max = np.percentile(depth_m[valid_depth], [5,95])

fig, axes = plt.subplots(1,2,figsize=(12,5))
axes[0].imshow(color_rgb)
axes[0].set_title("Color Image")

depth_plot = axes[1].imshow(depth_display, cmap=depth_cmap, vmin=display_min, 
		vmax = display_max,
	)
axes[1].set_title("Depth - black means missing")

for axis in axes: 
	axis.scatter([u],[v],s=90,facecolors="none",edgecolors="red",linewidths = 2,)
	axis.set_xlabel("u - column")
	axis.set_ylabel("v - row")
fig.colorbar(depth_plot, ax=axes[1], label = "Depth (metres)")
fig.tight_layout()

figure_path = output_dir / "rgbd_inspection.png"
fig.savefig(figure_path, dpi=100)
plt.close(fig)

print("Saved: ", figure_path.resolve())


