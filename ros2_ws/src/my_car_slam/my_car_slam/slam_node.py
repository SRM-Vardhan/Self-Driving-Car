import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry, OccupancyGrid

class SlamNode(Node):
    def __init__(self):
        super().__init__('slam_node')
        
        # Listening to LiDAR and Odometry
        self.scan_sub = self.create_subscription(PointCloud2, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        # Publishing the 2D Map
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 10)

    def scan_callback(self, msg):
        # ==========================================
        # ⚠️ SHUKSHAM: WRITE SLAM LOGIC HERE ⚠️
        # Task: Process the LiDAR scan data into a 2D map array.
        points = point_cloud2.read_points(
            msg,
            field_names=('x', 'y', 'z'),
            skip_nans=True
        )

        # Process every LiDAR point
        for point in points:

            lidar_x = point[0]
            lidar_y = point[1]
            lidar_z = point[2]

            # Ignore points that are too high
            if lidar_z > 2.0:
                continue

            # Convert LiDAR position to map position
            x = self.robot_x + lidar_x
            y = self.robot_y + lidar_y

            # Convert metres to map cells
            map_x = int(
                x / self.map_resolution
                + self.map_width / 2
            )

            map_y = int(
                y / self.map_resolution
                + self.map_height / 2
            )

            # Check whether point is inside map
            if (
                0 <= map_x < self.map_width
                and 0 <= map_y < self.map_height
            ):

                # Convert 2D coordinates into
                # one-dimensional array index
                index = (
                    map_y * self.map_width
                    + map_x
                )

                # Mark obstacle
                self.map_data[index] = 100

        
        # ==========================================
        # pass
def odom_callback(self, msg):
        # ==========================================
        # ⚠️ SHUKSHAM: WRITE ODOMETRY LOGIC HERE ⚠️
        # Task: Track the car's movement to update the map accurately.
        self.robot_x = msg.pose.pose.position.x
<<<<<<< HEAD
        self.robot_y = msg.pose.pose.position.y
=======
self.robot_y = msg.pose.pose.position.y
>>>>>>> 4707adc (Update my_car_slam package)
        # ==========================================
        # pass

def main(args=None):
    rclpy.init(args=args)
    node = SlamNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()