import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point, PointStamped


class WaypointNode(Node):

    def __init__(self):
        super().__init__('waypoint_node')

        # ==========================================================
        # SUBSCRIBER
        # ==========================================================

        # RViz publishes a PointStamped when the user clicks
        # the "Publish Point" tool.
        #
        # We only accept clicked points expressed in the map frame.
        self.clicked_point_sub = self.create_subscription(
            PointStamped,
            '/clicked_point',
            self.clicked_point_callback,
            10
        )

        # ==========================================================
        # PUBLISHER
        # ==========================================================

        # Global Planner expects:
        # /target_waypoint -> geometry_msgs/msg/Point
        #
        # Project convention:
        # x, y are coordinates in the map frame, in meters.
        self.target_pub = self.create_publisher(
            Point,
            '/target_waypoint',
            10
        )

        # ==========================================================
        # STATE
        # ==========================================================

        self.goal_x = None
        self.goal_y = None

        self.get_logger().info(
            'Waypoint Node started.'
        )

        self.get_logger().info(
            'Waiting for destination from RViz /clicked_point...'
        )

    # ==============================================================
    # RVIZ CLICK CALLBACK
    # ==============================================================

    def clicked_point_callback(self, msg):

        # ----------------------------------------------------------
        # Frame validation
        # ----------------------------------------------------------

        if msg.header.frame_id != 'map':

            self.get_logger().warn(
                'Ignoring clicked point: '
                f'frame_id="{msg.header.frame_id}". '
                'Expected frame_id="map".'
            )

            return

        # ----------------------------------------------------------
        # Read coordinates
        # ----------------------------------------------------------

        x = float(msg.point.x)
        y = float(msg.point.y)

        # ----------------------------------------------------------
        # Validate coordinates
        # ----------------------------------------------------------

        if not math.isfinite(x) or not math.isfinite(y):

            self.get_logger().warn(
                'Ignoring clicked point because X or Y is invalid.'
            )

            return

        # ----------------------------------------------------------
        # Store destination
        # ----------------------------------------------------------

        self.goal_x = x
        self.goal_y = y

        self.get_logger().info(
            f'New destination selected: '
            f'X={self.goal_x:.4f} m, '
            f'Y={self.goal_y:.4f} m '
            f'(map frame)'
        )

        # ----------------------------------------------------------
        # Publish destination
        # ----------------------------------------------------------

        self.publish_target()

    # ==============================================================
    # TARGET PUBLISHER
    # ==============================================================

    def publish_target(self):

        # This should only be called after a valid click.
        if self.goal_x is None or self.goal_y is None:
            return

        target = Point()

        target.x = self.goal_x
        target.y = self.goal_y
        target.z = 0.0

        self.target_pub.publish(target)

        self.get_logger().info(
            f'Published /target_waypoint: '
            f'X={target.x:.4f}, '
            f'Y={target.y:.4f}'
        )


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