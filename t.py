import rclpy
from rclpy.node import Node
import cv2
import numpy as np

WINDOW_NAME = "Test Window"

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

class WindowTestNode(Node):
    def __init__(self):
        super().__init__('window_test_node')
        self.get_logger().info("Node started, opening window...")
        self.create_timer(0.03, self.timer_callback)
        self.quit_flag = False  # 类成员

    def timer_callback(self):
        # 检查窗口是否关闭
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            self.get_logger().info("Window closed, requesting shutdown...")
            self.quit_flag = True
            return

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info("Quit key pressed, requesting shutdown...")
            self.quit_flag = True

def main(args=None):
    rclpy.init(args=args)
    node = WindowTestNode()
    try:
        while rclpy.ok() and not node.quit_flag:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
