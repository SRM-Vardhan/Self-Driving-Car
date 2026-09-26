import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import numpy as np


class SimpleSLAM(Node):
   def _init_(self):
         super()._init_('slam_node')
         self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)

        # Publisher
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 10)

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Store latest pose
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0

        # Create simple empty map
        self.map_width = 100
        self.map_height = 100
        self.map_resolution = 0.1

        self.map_data = np.zeros((self.map_width, self.map_height), dtype=np.int8)

    def odom_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

    def scan_callback(self, msg):

        # VERY SIMPLE MAPPING LOGIC (Educational)
        for i, distance in enumerate(msg.ranges):
            if distance < msg.range_max:
                angle = msg.angle_min + i * msg.angle_increment

                x = self.robot_x + distance * np.cos(angle)
                y = self.robot_y + distance * np.sin(angle)

                map_x = int(x / self.map_resolution)
                map_y = int(y / self.map_resolution)

                if 0 <= map_x < self.map_width and 0 <= map_y < self.map_height:
                    self.map_data[map_x][map_y] = 100  # Mark obstacle

        self.publish_map()
        self.publish_tf()

    def publish_map(self):
        map_msg = OccupancyGrid()
        map_msg.header.stamp = self.get_clock().now().to_msg()
        map_msg.header.frame_id = "map"

        map_msg.info.resolution = self.map_resolution
        map_msg.info.width = self.map_width
        map_msg.info.height = self.map_height

        map_msg.data = self.map_data.flatten().tolist()

        self.map_pub.publish(map_msg)

    def publish_tf(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "map"
        t.child_frame_id = "odom"

        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = SimpleSLAM()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if _name_ == '_main_':
    main()