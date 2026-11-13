import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from my_car_interfaces.msg import ObstacleInfo
from cv_bridge import CvBridge


class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_detector')
        self.bridge = CvBridge()

        # Listening to the camera
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # Publishing the AI output using the custom interface
        self.publisher_ = self.create_publisher(
            ObstacleInfo,
            '/yolo_obstacles',
            10
        )

    def image_callback(self, msg):
        # 1. Convert ROS Image to OpenCV format
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'passthrough')

        # ==========================================
        # ⚠️ KUNAL: WRITE YOLO CODE HERE ⚠️
        # Task:
        # - Load the YOLO model
        # - Run inference on cv_image
        # - Detect relevant objects
        # - Estimate object distance/height
        # - Decide whether the object is passable
        # - Publish the results using ObstacleInfo
        # ==========================================

        # 2. Publish dummy data to keep the ROS 2 graph happy
        #    until the real YOLO code is implemented.
        msg_out = ObstacleInfo()
        msg_out.object_label = "Waiting_For_YOLO"
        msg_out.distance = 0.0
        msg_out.height = 0.0
        msg_out.is_passable = True

        self.publisher_.publish(msg_out)


def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()