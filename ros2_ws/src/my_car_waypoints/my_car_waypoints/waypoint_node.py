import math

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import Point


class WaypointNode(Node):

    def __init__(self):
        super().__init__('waypoint_node')

        # ==========================================================
        # PARAMETERS
        # ==========================================================
        #
        # For the current Gazebo integration:
        #
        # /target_waypoint is expressed directly in the MAP frame.
        #
        # x, y -> meters
        #
        # We are intentionally not converting the live GNSS fix
        # into the target. The live GNSS fix is the vehicle's
        # current position, NOT the destination.
        #
        self.declare_parameter('goal_x', 0.0)
        self.declare_parameter('goal_y', 0.0)
        self.declare_parameter('goal_enabled', False)

        # ==========================================================
        # GNSS INPUT
        # ==========================================================
        #
        # Kept as part of the waypoint module.
        #
        # Later, this can support:
        # GNSS destination -> local/map coordinates.
        #
        self.gps_sub = self.create_subscription(
            NavSatFix,
            '/gnss/fix',
            self.gps_callback,
            10
        )

        # ==========================================================
        # TARGET WAYPOINT OUTPUT
        # ==========================================================
        #
        # Coordinate convention:
        #
        #   x = target X in map frame, meters
        #   y = target Y in map frame, meters
        #   z = unused
        #
        self.target_pub = self.create_publisher(
            Point,
            '/target_waypoint',
            10
        )

        # ==========================================================
        # STATE
        # ==========================================================
        self.current_lat = None
        self.current_lon = None

        self.goal_x = None
        self.goal_y = None
        self.goal_enabled = False

        self.load_goal()

        # Publish periodically.
        #
        # This ensures the Global Planner receives the target even
        # if it starts after the Waypoint Node.
        self.publish_timer = self.create_timer(
            1.0,
            self.publish_target
        )

        self.get_logger().info(
            'Waypoint Node started.'
        )

    # ==============================================================
    # LOAD DESTINATION
    # ==============================================================

    def load_goal(self):

        self.goal_x = float(
            self.get_parameter('goal_x').value
        )

        self.goal_y = float(
            self.get_parameter('goal_y').value
        )

        self.goal_enabled = bool(
            self.get_parameter('goal_enabled').value
        )

        if not self.goal_enabled:
            self.get_logger().warn(
                'No destination configured. '
                'Set goal_enabled:=true and provide '
                'goal_x and goal_y.'
            )
            return

        if not math.isfinite(self.goal_x):
            self.get_logger().error(
                'Invalid goal_x.'
            )
            self.goal_enabled = False
            return

        if not math.isfinite(self.goal_y):
            self.get_logger().error(
                'Invalid goal_y.'
            )
            self.goal_enabled = False
            return

        self.get_logger().info(
            'Destination configured: '
            f'x={self.goal_x:.3f} m, '
            f'y={self.goal_y:.3f} m '
            '(map frame)'
        )

    # ==============================================================
    # GNSS CALLBACK
    # ==============================================================

    def gps_callback(self, msg):

        if not math.isfinite(msg.latitude):
            return

        if not math.isfinite(msg.longitude):
            return

        self.current_lat = msg.latitude
        self.current_lon = msg.longitude

    # ==============================================================
    # PUBLISH DESTINATION
    # ==============================================================

    def publish_target(self):

        if not self.goal_enabled:
            return

        out_msg = Point()

        out_msg.x = self.goal_x
        out_msg.y = self.goal_y
        out_msg.z = 0.0

        self.target_pub.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)

    node = WaypointNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()