import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import numpy as np
from tf_transformations import quaternion_from_euler  # For quaternion handling


class SimpleSLAM(Node):

    def _init_(self):
        super()._init_('slam_node')

        # Subscribers
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
        # FIX 1: Extract position AND orientation from odometry
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        
        # Convert quaternion to euler to get theta
        orientation_q = msg.pose.pose.orientation
        # Simple approximation for 2D (assuming rotation around z-axis only)
        self.robot_theta = np.arctan2(2.0 * (orientation_q.w * orientation_q.z + orientation_q.x * orientation_q.y),
                                       1.0 - 2.0 * (orientation_q.y * orientation_q.y + orientation_q.z * orientation_q.z))

    def scan_callback(self, msg):
        # FIX 2: Handle invalid ranges (inf and NaN)
        for i, distance in enumerate(msg.ranges):
            # Skip invalid readings
            if not np.isfinite(distance) or distance > msg.range_max or distance < msg.range_min:
                continue
                
            angle = msg.angle_min + i * msg.angle_increment

            x = self.robot_x + distance * np.cos(angle)
            y = self.robot_y + distance * np.sin(angle)

            # FIX 3: Properly convert world coordinates to map indices
            # Map origin is at bottom-left, so we need to offset and flip y-axis
            map_x = int((x + self.map_width * self.map_resolution / 2) / self.map_resolution)
            map_y = int((y + self.map_height * self.map_resolution / 2) / self.map_resolution)

            if 0 <= map_x < self.map_width and 0 <= map_y < self.map_height:
                self.map_data[map_x, map_y] = 100  # Mark obstacle (use comma for 2D indexing)

        self.publish_map()
        self.publish_tf()

    def publish_map(self):
        map_msg = OccupancyGrid()
        map_msg.header.stamp = self.get_clock().now().to_msg()
        map_msg.header.frame_id = "map"

        map_msg.info.resolution = self.map_resolution
        map_msg.info.width = self.map_width
        map_msg.info.height = self.map_height

        # FIX 4: Set map origin (center of map at 0,0)
        from geometry_msgs.msg import Pose
        map_msg.info.origin = Pose()
        map_msg.info.origin.position.x = -self.map_width * self.map_resolution / 2
        map_msg.info.origin.position.y = -self.map_height * self.map_resolution / 2
        map_msg.info.origin.position.z = 0.0
        map_msg.info.origin.orientation.w = 1.0

        # FIX 5: Flatten in correct order (row-major for OccupancyGrid)
        map_msg.data = self.map_data.flatten(order='C').tolist()

        self.map_pub.publish(map_msg)

    def publish_tf(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "map"
        # FIX 6: child_frame_id should be 'odom' for map->odom transform
        # OR 'base_link' if publishing odom->base_link
        t.child_frame_id = "odom"

        # FIX 7: Use actual robot position, not hardcoded zeros
        t.transform.translation.x = self.robot_x
        t.transform.translation.y = self.robot_y
        t.transform.translation.z = 0.0
        
        # FIX 8: Set proper rotation from robot's orientation
        from geometry_msgs.msg import Quaternion
        quat = quaternion_from_euler(0, 0, self.robot_theta)
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = SimpleSLAM()
    
    # FIX 9: Use spin_once in a loop for proper callback processing
    while rclpy.ok():
        rclpy.spin_once(node)
    
    node.destroy_node()
    rclpy.shutdown()


if _name_ == '_main_':
    main()