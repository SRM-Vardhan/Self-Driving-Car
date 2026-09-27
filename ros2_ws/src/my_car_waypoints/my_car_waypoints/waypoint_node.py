import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import Point

class WaypointNode(Node):
    def __init__(self):
        super().__init__('waypoint_node')
        
        # Listening to GPS
        self.gps_sub = self.create_subscription(NavSatFix, '/gnss/fix', self.gps_callback, 10)
        # Publishing X, Y coordinates
        self.target_pub = self.create_publisher(Point, '/target_waypoint', 10)

    def gps_callback(self, msg):
        #import math

# WGS-84 Earth semi-major axis
EARTH_RADIUS = 6378137.0


def global_to_local(lat, lon, ref_lat, ref_lon):
    """
    Convert global GPS coordinates (latitude, longitude)
    into local East-North coordinates in meters.

    Parameters:
        lat     : Current GPS latitude in degrees
        lon     : Current GPS longitude in degrees
        ref_lat : Reference/origin latitude in degrees
        ref_lon : Reference/origin longitude in degrees

    Returns:
        east    : Local East coordinate in meters
        north   : Local North coordinate in meters
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


# --------------------------------------------------
# REFERENCE / ORIGIN POINT
# --------------------------------------------------
# Set this to the starting point of your vehicle
# obtained from your GNSS receiver.

REF_LAT = 28.600000
REF_LON = 77.200000


# --------------------------------------------------
# EXAMPLE LIVE GNSS DATA
# --------------------------------------------------
# Replace these values with latitude and longitude
# received from your actual GNSS module.

current_lat = 28.600100
current_lon = 77.200100


# Convert global GPS -> local coordinates
east, north = global_to_local(
    current_lat,
    current_lon,
    REF_LAT,
    REF_LON
)


# Output local coordinates
print(f"Local East  : {east:.2f} m")
print(f"Local North : {north:.2f} m") ==========================================
        # ⚠️ AYUSH: WRITE WAYPOINT LOGIC HERE ⚠️
        #
        # Task:
        # Convert GNSS latitude/longitude into
        # local X-Y coordinates for the planner.
        #
        # Output convention:
        #   x = target X coordinate in meters
        #   y = target Y coordinate in meters
        #   z = unused
        #
        # The X-Y frame must be consistent with
        # the SLAM/map coordinate frame.
        # ==========================================
        target_x = 0.0
        target_y = 0.0
        
        out_msg = Point()
        out_msg.x = target_x
        out_msg.y = target_y
        self.target_pub.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = WaypointNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()