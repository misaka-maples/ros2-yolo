from ultralytics import YOLO
import numpy as np
import cv2
# 加载YOLOv8模型
model = YOLO("yolov8s-seg.pt")

# 读取RGB图像
image = cv2.imread("test.jpg")

# YOLO 检测
results = model(image)

# 获取检测框
boxes = results[0].boxes.xyxy  # 获取每个物体的bounding box，格式为 [x1, y1, x2, y2]
print(boxes)



class DepthProcessor:
    def __init__(self, fx, fy, cx, cy):
        self.fx = fx  # 相机的焦距（像素单位）
        self.fy = fy
        self.cx = cx  # 主点坐标（通常为图像中心）
        self.cy = cy

    def depth_to_3d(self, u, v, depth_map):
        # 获取该像素的深度值
        Z = depth_map[v, u] / 1000.0  # 假设深度图的单位为毫米，转换为米
        X = (u - self.cx) * Z / self.fx
        Y = (v - self.cy) * Z / self.fy
        return np.array([X, Y, Z])  # 返回 3D 坐标 (X, Y, Z)

# 读取深度图（假设是一个灰度图，其中每个像素的值代表深度）
depth_image = cv2.imread("test_depth.png", cv2.IMREAD_UNCHANGED)

# 相机内参（示例）
fx, fy = 386.005, 386.005 
cx, cy = 318.554, 235.491
# 创建 DepthProcessor 实例
depth_processor = DepthProcessor(fx, fy, cx, cy)

# 假设我们使用第一个 mask 的中心作为抓取点
grasp_point_2d = np.array([int((boxes[0][0] + boxes[0][2]) / 2), int((boxes[0][1] + boxes[0][3]) / 2)])

# 将 2D 坐标转换为 3D 坐标
grasp_point_3d = depth_processor.depth_to_3d(grasp_point_2d[0], grasp_point_2d[1], depth_image)

print("抓取点的3D坐标: ", grasp_point_3d)
