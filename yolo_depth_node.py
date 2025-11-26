import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import time
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

        # 点击相关
        self.last_click_time = 0
        self.click_cooldown = 0.3
        self.current_boxes = []
        self.current_coords = []

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

        self.get_logger().info("YOLO+Depth node started!")

    def timer_callback(self):
        # 检查窗口是否关闭
        if cv2.getWindowProperty("YOLO detection", cv2.WND_PROP_VISIBLE) < 1:
            self.get_logger().info("Window closed, requesting shutdown...")
            self.quit_flag = True

    def info_callback(self, msg: CameraInfo):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        self.camera_info_received = True
        self.get_logger().info("Camera info received.")
        self.destroy_subscription(self.info_sub)  # 只读取一次
        print(self.fx, self.fy, self.cx, self.cy)

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    def color_callback(self, msg):
        if self.depth_image is None or not self.camera_info_received:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        results = self.model(frame, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        self.current_boxes = []
        self.current_coords = []

        for i, box in enumerate(boxes[:10]):  # 只处理前10个
            x1, y1, x2, y2 = map(int, box[:4])
            cx2d = int((x1 + x2) / 2)
            cy2d = int((y1 + y2) / 2)

            depth = self.depth_image[cy2d, cx2d] / 1000.0  # mm -> m
            Z = depth
            X = (cx2d - self.cx) * Z / self.fx
            Y = (cy2d - self.cy) * Z / self.fy

            self.current_boxes.append((x1, y1, x2, y2))
            self.current_coords.append((X, Y, Z))

            # 绘制缩小的框和文字
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.circle(frame, (cx2d, cy2d), 3, (0, 0, 255), -1)
            cv2.putText(frame, f"[{X:.2f}, {Y:.2f}, {Z:.2f}]",
                        (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

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

            for idx, (x1, y1, x2, y2) in enumerate(self.current_boxes):
                if x1 <= x <= x2 and y1 <= y <= y2:
                    X, Y, Z = self.current_coords[idx]
                    self.get_logger().info(f"Clicked box {idx} -> 3D: X:{X:.3f} Y:{Y:.3f} Z:{Z:.3f}")
                    break

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
