import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import Point


# WGS-84 Earth semi-major axis
EARTH_RADIUS = 6378137.0


def global_to_local(lat, lon, ref_lat, ref_lon):
    """
    Convert global GPS coordinates (latitude, longitude)
    into local East-North coordinates in meters.
    """

    # Convert degrees to radians
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)

    ref_lat_rad = math.radians(ref_lat)
    ref_lon_rad = math.radians(ref_lon)

    # Difference from reference point
    dlat = lat_rad - ref_lat_rad
    dlon = lon_rad - ref_lon_rad

    # Convert angular differences to meters
    north = EARTH_RADIUS * dlat
    east = EARTH_RADIUS * math.cos(ref_lat_rad) * dlon

    return east, north


class WaypointNode(Node):
    def __init__(self):
        super().__init__('waypoint_node')

        # Listening to GNSS
        self.gps_sub = self.create_subscription(
            NavSatFix,
            '/gnss/fix',
            self.gps_callback,
            10
        )

        # Publishing target X, Y coordinates
        self.target_pub = self.create_publisher(
            Point,
            '/target_waypoint',
            10
        )

        # ==========================================
        # REFERENCE / ORIGIN POINT
        # ==========================================
        # Ayush can update these later according to
        # the selected simulation/real-world origin.
        self.ref_lat = 28.600000
        self.ref_lon = 77.200000

    def gps_callback(self, msg):

        # ==========================================
        # ⚠️ AYUSH: WAYPOINT LOGIC
        #
        # Convert GNSS latitude/longitude into
        # local X-Y coordinates in meters.
        #
        # Output:
        #   x = local X coordinate
        #   y = local Y coordinate
        #   z = unused
        #
        # ==========================================

        current_lat = msg.latitude
        current_lon = msg.longitude

        east, north = global_to_local(
            current_lat,
            current_lon,
            self.ref_lat,
            self.ref_lon
        )

        # ==========================================
        # TEMPORARY
        #
        # This currently publishes the converted
        # GNSS position.
        #
        # The actual DESTINATION waypoint logic
        # still needs to be completed by Ayush.
        # ==========================================

        out_msg = Point()
        out_msg.x = east
        out_msg.y = north
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