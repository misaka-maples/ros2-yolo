import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import time
import numpy as np
from ultralytics import YOLO

class YOLODepthNode(Node):
    def __init__(self):
        super().__init__('yolo_depth_node')

        # YOLO模型加载
        self.model = YOLO("yolov8s-seg.pt")
        self.bridge = CvBridge()

        # 相机内参默认值
        self.fx = 607.0194091796875
        self.fy = 606.4129638671875
        self.cx = 323.3750915527344
        self.cy = 255.79656982421875
        self.camera_info_received = False
        self.depth_image = None
        
        # 【新增】深度图平滑核大小 (必须是奇数)
        self.depth_filter_size = 5 

        # 点击相关
        self.last_click_time = 0
        self.click_cooldown = 0.3
        
        # 存储当前显示的框，用于鼠标点击交互
        self.visible_detections = [] 

        # 安全退出标志
        self.quit_flag = False

        # ROS2话题订阅
        self.color_sub = self.create_subscription(
            Image, "/camera/camera/color/image_raw", self.color_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, "/camera/camera/depth/image_rect_raw", self.depth_callback, 10)
        self.info_sub = self.create_subscription(
            CameraInfo, "/camera/camera/color/camera_info", self.info_callback, 10)

        # 创建窗口并绑定鼠标点击事件
        cv2.namedWindow("YOLO detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("YOLO detection", 720, 720)
        cv2.setMouseCallback("YOLO detection", self.mouse_callback)

        # 定时器用于检查窗口关闭
        self.create_timer(0.03, self.timer_callback)

        self.get_logger().info("YOLO+Depth node started! (Depth filter enabled)")

    def timer_callback(self):
        if cv2.getWindowProperty("YOLO detection", cv2.WND_PROP_VISIBLE) < 1:
            self.get_logger().info("Window closed, requesting shutdown...")
            self.quit_flag = True

    def info_callback(self, msg: CameraInfo):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        self.camera_info_received = True
        self.get_logger().info(f"Camera info received: fx={self.fx}, fy={self.fy}")
        self.destroy_subscription(self.info_sub)

    def depth_callback(self, msg):
        # 转换为opencv格式 (mm)，通常是 CV_16UC1 (np.uint16)
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

        # ====== 核心修改: 深度图平滑 ======
        # 仅对有效的 16位深度图进行中值滤波
        if self.depth_image is not None and self.depth_image.dtype == np.uint16:
            # 使用中值滤波去除椒盐噪声和孤立的无效深度值，提高中心点采样的鲁棒性
            self.depth_image = cv2.medianBlur(self.depth_image, self.depth_filter_size)
        # ==================================

    def calculate_iou_contains(self, box_small, box_large):
        """
        计算小框有多少比例在大框内部。
        """
        x1_s, y1_s, x2_s, y2_s = box_small
        x1_l, y1_l, x2_l, y2_l = box_large

        # 计算交集坐标
        xi1 = max(x1_s, x1_l)
        yi1 = max(y1_s, y1_l)
        xi2 = min(x2_s, x2_l)
        yi2 = min(y2_s, y2_l)

        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height

        small_area = (x2_s - x1_s) * (y2_s - y1_s)
        
        if small_area == 0: return 0
        return inter_area / small_area

    def color_callback(self, msg):
        if self.depth_image is None or not self.camera_info_received:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # 限制最大检测框数量为 20 个
        results = self.model(frame, verbose=False, max_det=20) 
        
        # 获取YOLO结果
        boxes = results[0].boxes.xyxy.cpu().numpy()
        cls_ids = results[0].boxes.cls.cpu().numpy()
        names = results[0].names
        
        raw_detections = []

        # 1. 第一步：收集所有检测信息并计算面积
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            area = (x2 - x1) * (y2 - y1)
            cls_name = names[int(cls_ids[i])]
            
            raw_detections.append({
                'box': (x1, y1, x2, y2),
                'area': area,
                'cls_name': cls_name,
                'index': i,
                'hide': False
            })

        # 2. 第二步：过滤逻辑 (包含关系抑制)
        raw_detections.sort(key=lambda x: x['area'])

        for i in range(len(raw_detections)):
            if raw_detections[i]['hide']: continue
            
            for j in range(i + 1, len(raw_detections)):
                if raw_detections[j]['hide']: continue

                ratio = self.calculate_iou_contains(raw_detections[i]['box'], raw_detections[j]['box'])
                
                if ratio > 0.8:
                    raw_detections[j]['hide'] = True

        # 清空上一帧的可见列表
        self.visible_detections = []

        # 3. 第三步：绘制筛选后的框并过滤深度无效的框
        for det in raw_detections:
            if det['hide']:
                continue

            x1, y1, x2, y2 = det['box']
            cls_name = det['cls_name']

            # 计算中心点
            cx2d = int((x1 + x2) / 2)
            cy2d = int((y1 + y2) / 2)

            # 边界检查
            h, w = self.depth_image.shape
            if not (0 <= cx2d < w and 0 <= cy2d < h):
                self.get_logger().warn(f"Center point ({cx2d}, {cy2d}) out of bounds. Skipping detection.")
                continue

            # 获取深度并转换为米
            depth_mm = self.depth_image[cy2d, cx2d]
            depth_m = depth_mm / 1000.0 

            # 过滤深度无效的检测框 (Z <= 0)
            if depth_m <= 0.001: 
                continue

            Z = depth_m
            X = (cx2d - self.cx) * Z / self.fx
            Y = (cy2d - self.cy) * Z / self.fy

            # 保存到可见列表供鼠标点击使用
            self.visible_detections.append({
                'box': (x1, y1, x2, y2),
                'coords': (X, Y, Z),
                'cls': cls_name
            })

            # 绘图
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.circle(frame, (cx2d, cy2d), 3, (0, 0, 255), -1)
            
            # 文字标签 
            label = f"{cls_name} [{X:.2f}, {Y:.2f}, {Z:.2f}]m"
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(frame, (x1, y1), c2, (0, 255, 0), -1, cv2.LINE_AA) 
            cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)

        cv2.imshow("YOLO detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info("Quit key pressed, requesting shutdown...")
            self.quit_flag = True

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            now = time.time()
            if now - self.last_click_time < self.click_cooldown:
                return
            self.last_click_time = now

            clicked = False
            for item in self.visible_detections:
                x1, y1, x2, y2 = item['box']
                if x1 <= x <= x2 and y1 <= y <= y2:
                    X, Y, Z = item['coords']
                    name = item['cls']
                    self.get_logger().info(f"Clicked [{name}] -> 3D: X:{X:.3f} Y:{Y:.3f} Z:{Z:.3f}")
                    clicked = True
                    break 
            
            if not clicked:
                self.get_logger().info(f"Clicked at ({x}, {y}) - No object detected")

def main(args=None):
    rclpy.init(args=args)
    node = YOLODepthNode()
    try:
        while rclpy.ok() and not node.quit_flag:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == "__main__":
    main()