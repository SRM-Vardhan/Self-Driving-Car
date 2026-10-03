import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point, PointStamped


class WaypointNode(Node):

    def __init__(self):
        super().__init__('waypoint_node')

        # ==========================================================
        # CONFIGURATION
        # ==========================================================

        # Current simulation destination source.
        #
        # True:
        #   Receive destination from RViz /clicked_point.
        #
        # False:
        #   No destination source is active.
        #
        # There is intentionally NO default destination.
        self.declare_parameter(
            'use_rviz_destination',
            True
        )

        self.use_rviz_destination = bool(
            self.get_parameter(
                'use_rviz_destination'
            ).value
        )

        # How frequently the currently selected destination
        # is republished.
        #
        # This prevents the Global Planner from missing a goal
        # because it started after the original click.
        self.declare_parameter(
            'publish_rate',
            1.0
        )

        self.publish_rate = float(
            self.get_parameter(
                'publish_rate'
            ).value
        )

        if self.publish_rate <= 0.0:
            raise ValueError(
                'publish_rate must be greater than 0.'
            )

        # ==========================================================
        # DESTINATION STATE
        # ==========================================================

        # None means that no destination has been selected yet.
        self.goal_x = None
        self.goal_y = None

        # ==========================================================
        # INPUT: RVIZ DESTINATION
        # ==========================================================

        self.clicked_point_sub = None

        if self.use_rviz_destination:

            self.clicked_point_sub = self.create_subscription(
                PointStamped,
                '/clicked_point',
                self.clicked_point_callback,
                10
            )

        # ==========================================================
        # OUTPUT: TARGET WAYPOINT
        # ==========================================================

        # Global Planner interface:
        #
        # /target_waypoint
        # geometry_msgs/msg/Point
        #
        # Project convention:
        #
        #   x = map X coordinate [m]
        #   y = map Y coordinate [m]
        #   z = 0
        #
        self.target_pub = self.create_publisher(
            Point,
            '/target_waypoint',
            10
        )

        # ==========================================================
        # REPUBLISH TIMER
        # ==========================================================

        self.publish_timer = self.create_timer(
            1.0 / self.publish_rate,
            self.publish_target
        )

        # ==========================================================
        # STARTUP INFORMATION
        # ==========================================================

        self.get_logger().info(
            'Waypoint / Destination Manager started.'
        )

        if self.use_rviz_destination:

            self.get_logger().info(
                'Destination source: RViz /clicked_point'
            )

            self.get_logger().info(
                'Waiting for a destination in the map frame...'
            )

        else:

            self.get_logger().warn(
                'No destination source is enabled.'
            )

    # ==============================================================
    # RVIZ DESTINATION CALLBACK
    # ==============================================================

    def clicked_point_callback(self, msg):
        """
        Receive a destination selected in RViz.

        Expected input:
            /clicked_point
            geometry_msgs/msg/PointStamped

        Required frame:
            map
        """

        # ----------------------------------------------------------
        # Check coordinate frame
        # ----------------------------------------------------------

        if msg.header.frame_id != 'map':

            self.get_logger().warn(
                'Ignoring destination because it is not in '
                f'the map frame. Received frame: '
                f'"{msg.header.frame_id}"'
            )

            return

        # ----------------------------------------------------------
        # Extract coordinates
        # ----------------------------------------------------------

        x = float(msg.point.x)
        y = float(msg.point.y)

        # ----------------------------------------------------------
        # Validate coordinates
        # ----------------------------------------------------------

        if not math.isfinite(x):

            self.get_logger().warn(
                'Ignoring destination: X is not finite.'
            )

            return

        if not math.isfinite(y):

            self.get_logger().warn(
                'Ignoring destination: Y is not finite.'
            )

            return

        # ----------------------------------------------------------
        # Store new destination
        # ----------------------------------------------------------

        self.goal_x = x
        self.goal_y = y

        self.get_logger().info(
            'New destination selected: '
            f'X={self.goal_x:.4f} m, '
            f'Y={self.goal_y:.4f} m '
            '(map frame)'
        )

        # Publish immediately rather than waiting for the timer.
        self.publish_target()

    # ==============================================================
    # TARGET WAYPOINT PUBLISHER
    # ==============================================================

    def publish_target(self):
        """
        Publish the currently selected destination.

        Nothing is published until a valid destination has been
        selected.
        """

        # ----------------------------------------------------------
        # No destination yet
        # ----------------------------------------------------------

        if self.goal_x is None or self.goal_y is None:
            return

        # ----------------------------------------------------------
        # Create target message
        # ----------------------------------------------------------

        target = Point()

        target.x = self.goal_x
        target.y = self.goal_y

        # The global planner is currently 2D.
        target.z = 0.0

        # ----------------------------------------------------------
        # Publish
        # ----------------------------------------------------------

        self.target_pub.publish(target)

    # ==============================================================
    # CURRENT DESTINATION INFORMATION
    # ==============================================================

    def get_current_goal(self):
        """
        Return the currently selected destination.

        Returns:
            (x, y) if a destination exists
            None otherwise
        """

        if self.goal_x is None or self.goal_y is None:
            return None

        return self.goal_x, self.goal_y


# ==================================================================
# MAIN
# ==================================================================

def main(args=None):

    rclpy.init(args=args)

    node = WaypointNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main() 