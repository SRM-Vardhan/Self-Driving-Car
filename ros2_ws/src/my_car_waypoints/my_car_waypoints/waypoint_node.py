import math

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import Point


class WaypointNode(Node):

    def __init__(self):
        super().__init__('waypoint_node')

        # ==========================================
        # PARAMETERS
        # ==========================================

        # For the first Gazebo integration, use a
        # local map-frame destination directly.
        #
        # These coordinates are in METERS and must
        # correspond to the /map coordinate system.
        self.declare_parameter('goal_x', float('nan'))
        self.declare_parameter('goal_y', float('nan'))

        # ==========================================
        # GNSS INPUT
        #
        # Kept because GNSS is part of Ayush's module.
        # It can be used later for GNSS-based waypoints.
        # ==========================================
        self.gps_sub = self.create_subscription(
            NavSatFix,
            '/gnss/fix',
            self.gps_callback,
            10
        )

        # ==========================================
        # TARGET WAYPOINT OUTPUT
        # ==========================================
        self.target_pub = self.create_publisher(
            Point,
            '/target_waypoint',
            10
        )

        # ==========================================
        # STATE
        # ==========================================
        self.current_lat = None
        self.current_lon = None

        self.goal_x = None
        self.goal_y = None

        self.load_goal()

        # Publish periodically so that the Global
        # Planner can start before or after this node.
        self.publish_timer = self.create_timer(
            1.0,
            self.publish_target
        )

        self.get_logger().info(
            'Waypoint Node started.'
        )

    # ==========================================
    # LOAD LOCAL GOAL
    # ==========================================
    def load_goal(self):

        goal_x = self.get_parameter(
            'goal_x'
        ).value

        goal_y = self.get_parameter(
            'goal_y'
        ).value

        if not math.isfinite(goal_x) or not math.isfinite(goal_y):
            self.get_logger().error(
                'No valid destination configured.'
            )
            self.get_logger().error(
                'Set goal_x and goal_y in map coordinates.'
            )
            return

        self.goal_x = float(goal_x)
        self.goal_y = float(goal_y)

        self.get_logger().info(
            f'Destination set to '
            f'x={self.goal_x:.3f} m, '
            f'y={self.goal_y:.3f} m '
            f'(map frame)'
        )

    # ==========================================
    # GNSS CALLBACK
    # ==========================================
    def gps_callback(self, msg):

        if not math.isfinite(msg.latitude):
            return

        if not math.isfinite(msg.longitude):
            return

        self.current_lat = msg.latitude
        self.current_lon = msg.longitude

    # ==========================================
    # PUBLISH TARGET WAYPOINT
    # ==========================================
    def publish_target(self):

        # No valid goal configured.
        if self.goal_x is None or self.goal_y is None:
            return

        out_msg = Point()

        out_msg.x = self.goal_x
        out_msg.y = self.goal_y
        out_msg.z = 0.0

        self.target_pub.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)

    node = WaypointNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()