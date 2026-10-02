

import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

from nav_msgs.msg import Odometry, OccupancyGrid

import tf2_ros


class SimpleSLAM(Node):

    def __init__(self):
        super().__init__('simple_slam')

        # ============================================================
        # PARAMETERS
        # ============================================================

        self.declare_parameter('use_sim_time', True)

        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('odom_topic', '/odom')

        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')

        # 5 cm resolution
        self.declare_parameter('resolution', 0.05)

        # Initial map: 20m x 20m
        self.declare_parameter('initial_width', 400)
        self.declare_parameter('initial_height', 400)

        # Expand map when robot reaches the edge
        self.declare_parameter('expansion_margin', 5.0)

        # LiDAR limits
        self.declare_parameter('min_range', 0.15)
        self.declare_parameter('max_range', 12.0)

        # Point-cloud filtering
        self.declare_parameter('min_z', -0.5)
        self.declare_parameter('max_z', 1.5)

        # Reduce computation
        self.declare_parameter('point_skip', 2)
        self.declare_parameter('max_points', 500)

        # Scan matching
        self.declare_parameter('max_match_points', 150)

        # ============================================================
        # READ PARAMETERS
        # ============================================================

        self.scan_topic = self.get_parameter(
            'scan_topic'
        ).value

        self.odom_topic = self.get_parameter(
            'odom_topic'
        ).value

        self.map_frame = self.get_parameter(
            'map_frame'
        ).value

        self.odom_frame = self.get_parameter(
            'odom_frame'
        ).value

        self.base_frame = self.get_parameter(
            'base_frame'
        ).value

        self.resolution = float(
            self.get_parameter('resolution').value
        )

        self.expansion_margin = float(
            self.get_parameter('expansion_margin').value
        )

        self.min_range = float(
            self.get_parameter('min_range').value
        )

        self.max_range = float(
            self.get_parameter('max_range').value
        )

        self.min_z = float(
            self.get_parameter('min_z').value
        )

        self.max_z = float(
            self.get_parameter('max_z').value
        )

        self.point_skip = int(
            self.get_parameter('point_skip').value
        )

        self.max_points = int(
            self.get_parameter('max_points').value
        )

        self.max_match_points = int(
            self.get_parameter('max_match_points').value
        )

        # ============================================================
        # MAP
        # ============================================================

        width = int(
            self.get_parameter('initial_width').value
        )

        height = int(
            self.get_parameter('initial_height').value
        )

        # -1 = unknown
        #  0 = free
        # 100 = occupied

        self.map_data = np.full(
            (height, width),
            -1,
            dtype=np.int8
        )

        # Put (0,0) near the center of the initial map
        self.origin_x = -(width * self.resolution) / 2.0
        self.origin_y = -(height * self.resolution) / 2.0

        # ============================================================
        # PUBLISH MAP
        # ============================================================

        self.map_pub = self.create_publisher(
            OccupancyGrid,
            '/map',
            1
        )

        # ============================================================
        # ODOM
        # ============================================================

        self.odom_x = None
        self.odom_y = None
        self.odom_yaw = None

        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            20
        )

        # ============================================================
        # TF
        # ============================================================

        self.tf_buffer = tf2_ros.Buffer()

        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer,
            self
        )

        self.tf_broadcaster = tf2_ros.TransformBroadcaster(
            self
        )

        # ============================================================
        # MAP -> ODOM CORRECTION
        # ============================================================

        self.map_odom_x = 0.0
        self.map_odom_y = 0.0
        self.map_odom_yaw = 0.0

        # ============================================================
        # ROS BAG /SCAN
        # ============================================================

        self.scan_sub = self.create_subscription(
            PointCloud2,
            self.scan_topic,
            self.scan_callback,
            10
        )

        self.last_scan_stamp = None
        self.first_scan = True

        # ============================================================
        # MAP PUBLISH TIMER
        # ============================================================

        self.map_timer = self.create_timer(
            0.5,
            self.publish_map
        )

        self.get_logger().info(
            '========================================'
        )
        self.get_logger().info(
            'ROS BAG ONLY SimpleSLAM'
        )
        self.get_logger().info(
            'Reading: /scan PointCloud2'
        )
        self.get_logger().info(
            'Reading: /odom Odometry'
        )
        self.get_logger().info(
            'Reading: /tf and /tf_static'
        )
        self.get_logger().info(
            'Publishing: /map'
        )
        self.get_logger().info(
            '========================================'
        )

    # ================================================================
    # ODOMETRY
    # ================================================================

    def odom_callback(self, msg):

        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation

        self.odom_yaw = self.quaternion_to_yaw(
            q.x,
            q.y,
            q.z,
            q.w
        )

    # ================================================================
    # QUATERNION TO YAW
    # ================================================================

    def quaternion_to_yaw(
        self,
        x,
        y,
        z,
        w
    ):

        siny = 2.0 * (w * z + x * y)
        cosy = 1.0 - 2.0 * (y * y + z * z)

        return math.atan2(
            siny,
            cosy
        )

    # ================================================================
    # NORMALIZE ANGLE
    # ================================================================

    def normalize_angle(self, angle):

        return math.atan2(
            math.sin(angle),
            math.cos(angle)
        )

    # ================================================================
    # POINT CLOUD CALLBACK
    # ================================================================

    def scan_callback(self, msg):

        # We need odometry before processing scans
        if self.odom_x is None:
            return

        points = []

        try:

            for p in point_cloud2.read_points(
                msg,
                field_names=('x', 'y', 'z'),
                skip_nans=True
            ):

                x = float(p[0])
                y = float(p[1])
                z = float(p[2])

                distance = math.sqrt(
                    x * x + y * y
                )

                # Range filter
                if distance < self.min_range:
                    continue

                if distance > self.max_range:
                    continue

                # Height filter
                if z < self.min_z:
                    continue

                if z > self.max_z:
                    continue

                points.append(
                    [x, y, z]
                )

        except Exception as e:

            self.get_logger().error(
                'PointCloud2 error: %s' % str(e)
            )

            return

        if len(points) == 0:
            return

        points = np.asarray(
            points,
            dtype=np.float32
        )

        # Downsample
        if self.point_skip > 1:

            points = points[
                ::self.point_skip
            ]

        # Limit number of points
        if len(points) > self.max_points:

            indexes = np.linspace(
                0,
                len(points) - 1,
                self.max_points
            ).astype(int)

            points = points[indexes]

        # ==========================================================
        # SENSOR FRAME -> BASE FRAME
        # ==========================================================

        points = self.sensor_to_base(
            points,
            msg.header.frame_id,
            msg.header.stamp
        )

        if points is None:
            return

        self.process_scan(
            points,
            msg.header.stamp
        )

    # ================================================================
    # SENSOR -> BASE TRANSFORMATION USING BAG TF
    # ================================================================

    def sensor_to_base(
        self,
        points,
        source_frame,
        stamp
    ):

        if not source_frame:
            return points

        if source_frame == self.base_frame:
            return points

        try:

            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                source_frame,
                Time.from_msg(stamp),
                timeout=Duration(
                    seconds=0.2
                )
            )

        except Exception:

            self.get_logger().warn(
                'Waiting for TF: %s -> %s' %
                (
                    source_frame,
                    self.base_frame
                ),
                throttle_duration_sec=2.0
            )

            return None

        t = transform.transform.translation
        q = transform.transform.rotation

        qx = q.x
        qy = q.y
        qz = q.z
        qw = q.w

        # Quaternion -> rotation matrix
        R = np.array(
            [
                [
                    1 - 2*(qy*qy + qz*qz),
                    2*(qx*qy - qz*qw),
                    2*(qx*qz + qy*qw)
                ],
                [
                    2*(qx*qy + qz*qw),
                    1 - 2*(qx*qx + qz*qz),
                    2*(qy*qz - qx*qw)
                ],
                [
                    2*(qx*qz - qy*qw),
                    2*(qy*qz + qx*qw),
                    1 - 2*(qx*qx + qy*qy)
                ]
            ],
            dtype=np.float32
        )

        translation = np.array(
            [
                t.x,
                t.y,
                t.z
            ],
            dtype=np.float32
        )

        return points @ R.T + translation

    # ================================================================
    # PROCESS SCAN
    # ================================================================

    def process_scan(
        self,
        points,
        stamp
    ):

        self.last_scan_stamp = stamp

        # Current estimated robot pose from odometry
        predicted_x, predicted_y, predicted_yaw = \
            self.get_predicted_pose()

        # ============================================================
        # FIRST SCAN
        # ============================================================

        if self.first_scan:

            corrected_x = predicted_x
            corrected_y = predicted_y
            corrected_yaw = predicted_yaw

            self.first_scan = False

        else:

            # Use reduced scan for matching
            match_points = points

            if len(match_points) > self.max_match_points:

                indexes = np.linspace(
                    0,
                    len(match_points) - 1,
                    self.max_match_points
                ).astype(int)

                match_points = match_points[
                    indexes
                ]

            corrected_x, corrected_y, corrected_yaw = \
                self.scan_match(
                    match_points,
                    predicted_x,
                    predicted_y,
                    predicted_yaw
                )

        # ============================================================
        # UPDATE MAP -> ODOM
        # ============================================================

        self.update_map_odom(
            corrected_x,
            corrected_y,
            corrected_yaw
        )

        # ============================================================
        # BASE -> MAP
        # ============================================================

        map_points = self.base_to_map(
            points,
            corrected_x,
            corrected_y,
            corrected_yaw
        )

        # ============================================================
        # DYNAMIC MAP EXPANSION
        # ============================================================

        self.expand_map(
            map_points,
            corrected_x,
            corrected_y
        )

        # ============================================================
        # ROBOT CELL
        # ============================================================

        robot_cell = self.world_to_map(
            corrected_x,
            corrected_y
        )

        if robot_cell is None:
            return

        robot_mx, robot_my = robot_cell

        # ============================================================
        # RAY TRACE EACH LASER BEAM
        # ============================================================

        for point in map_points:

            px = float(point[0])
            py = float(point[1])

            obstacle_cell = self.world_to_map(
                px,
                py
            )

            if obstacle_cell is None:
                continue

            obstacle_mx, obstacle_my = obstacle_cell

            # Mark cells between robot and obstacle as FREE
            self.raytrace(
                robot_mx,
                robot_my,
                obstacle_mx,
                obstacle_my
            )

            # Mark endpoint as OCCUPIED
            if (
                0 <= obstacle_mx < self.map_data.shape[1]
                and
                0 <= obstacle_my < self.map_data.shape[0]
            ):

                self.map_data[
                    obstacle_my,
                    obstacle_mx
                ] = 100

        # ============================================================
        # PUBLISH MAP -> ODOM
        # ============================================================

        self.publish_map_odom_tf()

    # ================================================================
    # GET PREDICTED POSE
    # ================================================================

    def get_predicted_pose(self):

        c = math.cos(
            self.map_odom_yaw
        )

        s = math.sin(
            self.map_odom_yaw
        )

        x = (
            c * self.odom_x
            - s * self.odom_y
            + self.map_odom_x
        )

        y = (
            s * self.odom_x
            + c * self.odom_y
            + self.map_odom_y
        )

        yaw = self.normalize_angle(
            self.odom_yaw
            + self.map_odom_yaw
        )

        return x, y, yaw

    # ================================================================
    # BASE -> MAP
    # ================================================================

    def base_to_map(
        self,
        points,
        robot_x,
        robot_y,
        robot_yaw
    ):

        c = math.cos(robot_yaw)
        s = math.sin(robot_yaw)

        x = points[:, 0]
        y = points[:, 1]

        map_x = (
            c * x
            - s * y
            + robot_x
        )

        map_y = (
            s * x
            + c * y
            + robot_y
        )

        return np.column_stack(
            (
                map_x,
                map_y,
                points[:, 2]
            )
        )

    # ================================================================
    # SCAN MATCHING
    # ================================================================

    def scan_match(
        self,
        points,
        initial_x,
        initial_y,
        initial_yaw
    ):

        # Need enough occupied map information
        occupied_cells = np.count_nonzero(
            self.map_data >= 50
        )

        if occupied_cells < 30:

            return (
                initial_x,
                initial_y,
                initial_yaw
            )

        best_x = initial_x
        best_y = initial_y
        best_yaw = initial_yaw
        best_score = -1.0

        # Search around odometry prediction
        for dx in np.arange(
            -0.30,
            0.31,
            0.10
        ):

            for dy in np.arange(
                -0.30,
                0.31,
                0.10
            ):

                for da in np.deg2rad(
                    np.arange(
                        -10.0,
                        10.1,
                        4.0
                    )
                ):

                    x = initial_x + dx
                    y = initial_y + dy
                    yaw = initial_yaw + da

                    score = self.match_score(
                        points,
                        x,
                        y,
                        yaw
                    )

                    if score > best_score:

                        best_score = score
                        best_x = x
                        best_y = y
                        best_yaw = yaw

        return (
            best_x,
            best_y,
            self.normalize_angle(best_yaw)
        )

    # ================================================================
    # SCAN MATCH SCORE
    # ================================================================

    def match_score(
        self,
        points,
        robot_x,
        robot_y,
        robot_yaw
    ):

        c = math.cos(robot_yaw)
        s = math.sin(robot_yaw)

        x = points[:, 0]
        y = points[:, 1]

        wx = (
            c * x
            - s * y
            + robot_x
        )

        wy = (
            s * x
            + c * y
            + robot_y
        )

        mx = np.floor(
            (wx - self.origin_x)
            / self.resolution
        ).astype(np.int32)

        my = np.floor(
            (wy - self.origin_y)
            / self.resolution
        ).astype(np.int32)

        valid = (
            (mx >= 0)
            &
            (my >= 0)
            &
            (mx < self.map_data.shape[1])
            &
            (my < self.map_data.shape[0])
        )

        if not np.any(valid):
            return 0.0

        values = self.map_data[
            my[valid],
            mx[valid]
        ]

        # Only known cells
        known = values >= 0

        if not np.any(known):
            return 0.0

        values = values[known]

        # Occupied cells are the strongest match
        occupied = np.count_nonzero(
            values >= 50
        )

        return occupied / max(
            len(values),
            1
        )

    # ================================================================
    # MAP -> ODOM CORRECTION
    # ================================================================

    def update_map_odom(
        self,
        map_x,
        map_y,
        map_yaw
    ):

        c = math.cos(
            map_yaw - self.odom_yaw
        )

        s = math.sin(
            map_yaw - self.odom_yaw
        )

        correction_yaw = self.normalize_angle(
            map_yaw - self.odom_yaw
        )

        correction_x = (
            map_x
            - (
                c * self.odom_x
                - s * self.odom_y
            )
        )

        correction_y = (
            map_y
            - (
                s * self.odom_x
                + c * self.odom_y
            )
        )

        self.map_odom_x = correction_x
        self.map_odom_y = correction_y
        self.map_odom_yaw = correction_yaw

    # ================================================================
    # DYNAMIC MAP EXPANSION
    # ================================================================

    def expand_map(
        self,
        points,
        robot_x,
        robot_y
    ):

        if len(points) == 0:
            return

        min_x = min(
            float(np.min(points[:, 0])),
            robot_x
        )

        max_x = max(
            float(np.max(points[:, 0])),
            robot_x
        )

        min_y = min(
            float(np.min(points[:, 1])),
            robot_y
        )

        max_y = max(
            float(np.max(points[:, 1])),
            robot_y
        )

        current_min_x = self.origin_x

        current_max_x = (
            self.origin_x
            + self.map_data.shape[1]
            * self.resolution
        )

        current_min_y = self.origin_y

        current_max_y = (
            self.origin_y
            + self.map_data.shape[0]
            * self.resolution
        )

        if (
            min_x >= current_min_x
            and
            max_x <= current_max_x
            and
            min_y >= current_min_y
            and
            max_y <= current_max_y
        ):
            return

        new_min_x = min(
            current_min_x,
            min_x - self.expansion_margin
        )

        new_max_x = max(
            current_max_x,
            max_x + self.expansion_margin
        )

        new_min_y = min(
            current_min_y,
            min_y - self.expansion_margin
        )

        new_max_y = max(
            current_max_y,
            max_y + self.expansion_margin
        )

        new_width = int(
            math.ceil(
                (new_max_x - new_min_x)
                / self.resolution
            )
        )

        new_height = int(
            math.ceil(
                (new_max_y - new_min_y)
                / self.resolution
            )
        )

        new_map = np.full(
            (new_height, new_width),
            -1,
            dtype=np.int8
        )

        offset_x = int(
            round(
                (self.origin_x - new_min_x)
                / self.resolution
            )
        )

        offset_y = int(
            round(
                (self.origin_y - new_min_y)
                / self.resolution
            )
        )

        old_height, old_width = \
            self.map_data.shape

        new_map[
            offset_y:offset_y + old_height,
            offset_x:offset_x + old_width
        ] = self.map_data

        self.map_data = new_map

        self.origin_x = new_min_x
        self.origin_y = new_min_y

        self.get_logger().info(
            'Map expanded: %.1f m x %.1f m' %
            (
                new_width * self.resolution,
                new_height * self.resolution
            )
        )

    # ================================================================
    # WORLD -> MAP
    # ================================================================

    def world_to_map(
        self,
        x,
        y
    ):

        mx = int(
            math.floor(
                (x - self.origin_x)
                / self.resolution
            )
        )

        my = int(
            math.floor(
                (y - self.origin_y)
                / self.resolution
            )
        )

        if (
            mx < 0
            or my < 0
            or mx >= self.map_data.shape[1]
            or my >= self.map_data.shape[0]
        ):
            return None

        return mx, my

    # ================================================================
    # BRESENHAM RAY TRACING
    # ================================================================

    def raytrace(
        self,
        x0,
        y0,
        x1,
        y1
    ):

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)

        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1

        error = dx - dy

        x = x0
        y = y0

        while True:

            # Do not overwrite occupied cells
            if (
                0 <= x < self.map_data.shape[1]
                and
                0 <= y < self.map_data.shape[0]
            ):

                if self.map_data[y, x] != 100:
                    self.map_data[y, x] = 0

            if x == x1 and y == y1:
                break

            e2 = 2 * error

            if e2 > -dy:

                error -= dy
                x += sx

            if e2 < dx:

                error += dx
                y += sy

    # ================================================================
    # PUBLISH OCCUPANCY GRID
    # ================================================================

    def publish_map(self):

        msg = OccupancyGrid()

        if self.last_scan_stamp is not None:

            msg.header.stamp = \
                self.last_scan_stamp

        else:

            msg.header.stamp = \
                self.get_clock().now().to_msg()

        msg.header.frame_id = self.map_frame

        msg.info.resolution = float(
            self.resolution
        )

        msg.info.width = int(
            self.map_data.shape[1]
        )

        msg.info.height = int(
            self.map_data.shape[0]
        )

        msg.info.origin.position.x = \
            float(self.origin_x)

        msg.info.origin.position.y = \
            float(self.origin_y)

        msg.info.origin.position.z = 0.0

        msg.info.origin.orientation.x = 0.0
        msg.info.origin.orientation.y = 0.0
        msg.info.origin.orientation.z = 0.0
        msg.info.origin.orientation.w = 1.0

        msg.data = \
            self.map_data.flatten().tolist()

        self.map_pub.publish(msg)

    # ================================================================
    # MAP -> ODOM TF
    # ================================================================

    def publish_map_odom_tf(self):

        transform = \
            tf2_ros.TransformStamped()

        transform.header.stamp = \
            self.get_clock().now().to_msg()

        transform.header.frame_id = \
            self.map_frame

        transform.child_frame_id = \
            self.odom_frame

        transform.transform.translation.x = \
            self.map_odom_x

        transform.transform.translation.y = \
            self.map_odom_y

        transform.transform.translation.z = 0.0

        transform.transform.rotation.x = 0.0
        transform.transform.rotation.y = 0.0

        transform.transform.rotation.z = \
            math.sin(
                self.map_odom_yaw / 2.0
            )

        transform.transform.rotation.w = \
            math.cos(
                self.map_odom_yaw / 2.0
            )

        self.tf_broadcaster.sendTransform(
            transform
        )


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
