import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import TransformStamped, Pose

from tf2_ros import TransformBroadcaster
from tf2 import Transformations  # ROS 2 Humble+
# OR compute quaternion manually for simpler cases

import numpy as np
from math import cos, sin, atan2


class SimpleSLAM(Node):

    def __init__(self):
        super().__init__('slam_node')

        # =========================
        # Subscribers
        # =========================

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        # =========================
        # Publisher
        # =========================

        self.map_pub = self.create_publisher(
            OccupancyGrid,
            '/map',
            10
        )

        # =========================
        # TF Broadcaster
        # =========================

        self.tf_broadcaster = TransformBroadcaster(self)

        # =========================
        # Robot Pose from Odometry
        # =========================

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0

        # =========================
        # Map Parameters
        # =========================

        self.map_width = 100
        self.map_height = 100
        self.map_resolution = 0.1

        self.map_origin_x = -(self.map_width * self.map_resolution) / 2.0
        self.map_origin_y = -(self.map_height * self.map_resolution) / 2.0

        self.map_data = np.full(
            (self.map_height, self.map_width),
            -1,
            dtype=np.int8
        )

        self.get_logger().info("Simple SLAM node started")

    def odom_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        # Correct quaternion to yaw conversion
        self.robot_theta = atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

    def scan_callback(self, msg):
        for i, distance in enumerate(msg.ranges):
            if not np.isfinite(distance):
                continue
            if distance < msg.range_min or distance > msg.range_max:
                continue

            laser_angle = msg.angle_min + i * msg.angle_increment

            # Laser coordinates relative to robot
            laser_x = distance * cos(laser_angle)
            laser_y = distance * sin(laser_angle)

            # Transform to world coordinates
            world_x = self.robot_x + laser_x * cos(self.robot_theta) - laser_y * sin(self.robot_theta)
            world_y = self.robot_y + laser_x * sin(self.robot_theta) + laser_y * cos(self.robot_theta)

            # World to map coordinates
            map_x = int((world_x - self.map_origin_x) / self.map_resolution)
            map_y = int((world_y - self.map_origin_y) / self.map_resolution)

            if 0 <= map_x < self.map_width and 0 <= map_y < self.map_height:
                self.map_data[map_y, map_x] = 100

        self.publish_map()

    def publish_map(self):
        map_msg = OccupancyGrid()
        map_msg.header.stamp = self.get_clock().now().to_msg()
        map_msg.header.frame_id = "map"

        map_msg.info.resolution = self.map_resolution
        map_msg.info.width = self.map_width
        map_msg.info.height = self.map_height

        map_msg.info.origin.position.x = self.map_origin_x
        map_msg.info.origin.position.y = self.map_origin_y
        map_msg.info.origin.position.z = 0.0
        map_msg.info.origin.orientation.w = 1.0

        map_msg.data = self.map_data.flatten().tolist()
        self.map_pub.publish(map_msg)

    def publish_tf(self):
        # For simple SLAM, you typically DON'T need to broadcast map->odom
        # as it's static. Instead, broadcast odom->base_link if needed.
        # This function can be removed for a minimal example.
        pass


def main(args=None):
    rclpy.init(args=args)
    node = SimpleSLAM()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()