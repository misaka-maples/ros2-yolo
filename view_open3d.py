import view_open3d as o3d
import numpy as np
import cv2
# ----------------------------- # 假设你已经有彩色和深度图 
color_path = "/home/maple/project/save_one/color.png" 
depth_path = "/home/maple/project/save_one/depth.npy" 

fx, fy = 386.005, 386.005 
cx, cy = 318.554, 235.491
# --- 读取彩色图 ---
color_raw = cv2.imread(color_path)
color_raw = cv2.cvtColor(color_raw, cv2.COLOR_BGR2RGB)
color_o3d = o3d.geometry.Image(color_raw)

# --- 读取深度图 ---
depth_raw = np.load(depth_path).astype(np.float32)  # 确保是 float32
depth_raw /= 1000.0  # mm -> m
depth_o3d = o3d.geometry.Image(depth_raw)

# --- 创建 RGBD 图像 ---
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_o3d,
    depth_o3d,
    depth_scale=1.0,       # 已经转为米
    depth_trunc=3.0,       # 最大深度3米
    convert_rgb_to_intensity=False
)

# --- 相机内参 ---
camera_intrinsics = o3d.camera.PinholeCameraIntrinsic()
camera_intrinsics.set_intrinsics(
    width=depth_raw.shape[1],
    height=depth_raw.shape[0],
    fx=fx,
    fy=fy,
    cx=cx,
    cy=cy
)

# --- 生成点云 ---
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image,
    camera_intrinsics
)

# --- 可视化 ---
o3d.visualization.draw_geometries([pcd], window_name="RGB-D PointCloud")
