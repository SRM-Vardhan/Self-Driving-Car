import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge

from my_car_lane_detection.lane_detector import detect_lane


class LaneDetectionNode(Node):
    def __init__(self):
        super().__init__('lane_detector')

        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.publisher_ = self.create_publisher(
            Float32,
            '/lane_offset',
            10
        )

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            result = detect_lane(cv_image)

            lane_offset = result["lane_offset"]

            if lane_offset is None:
                return

            msg_out = Float32()
            msg_out.data = float(lane_offset)
            self.publisher_.publish(msg_out)

        except Exception as e:
            self.get_logger().error(f"Lane detection error: {e}")


def main(args=None):
    rclpy.init(args=args)

    node = LaneDetectionNode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()