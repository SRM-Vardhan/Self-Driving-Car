

import math
from collections import deque

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

from nav_msgs.msg import Odometry, OccupancyGrid

from geometry_msgs.msg import TransformStamped

import tf2_ros
from tf2_ros import TransformException


class SimpleSLAM(Node):

    def __init__(self):
        super().__init__('simple_slam')

        # ============================================================
        # PARAMETERS
        # ============================================================

        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('odom_topic', '/odom')

        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')

        self.declare_parameter('resolution', 0.05)

        self.declare_parameter('initial_map_size', 400)

        self.declare_parameter('max_range', 8.0)
        self.declare_parameter('min_range', 0.10)

        self.declare_parameter('log_every_n_scans', 20)

        # Scan matching
        self.declare_parameter('use_scan_matching', True)
        self.declare_parameter('match_every_n_scans', 5)

        self.scan_topic = self.get_parameter(
            'scan_topic').value

        self.odom_topic = self.get_parameter(
            'odom_topic').value

        self.map_frame = self.get_parameter(
            'map_frame').value

        self.odom_frame = self.get_parameter(
            'odom_frame').value

        self.base_frame = self.get_parameter(
            'base_frame').value

        self.resolution = float(
            self.get_parameter('resolution').value)

        self.initial_map_size = int(
            self.get_parameter('initial_map_size').value)

        self.max_range = float(
            self.get_parameter('max_range').value)

        self.min_range = float(
            self.get_parameter('min_range').value)

        self.log_every_n_scans = int(
            self.get_parameter('log_every_n_scans').value)

        self.use_scan_matching = bool(
            self.get_parameter('use_scan_matching').value)

        self.match_every_n_scans = int(
            self.get_parameter('match_every_n_scans').value)

        # ============================================================
        # MAP
        # ============================================================

        self.map_width = self.initial_map_size
        self.map_height = self.initial_map_size

        self.map_origin_x = -(
            self.map_width * self.resolution / 2.0)

        self.map_origin_y = -(
            self.map_height * self.resolution / 2.0)

        # -1 = UNKNOWN
        #  0 = FREE
        # 100 = OCCUPIED
        self.grid = np.full(
            (self.map_height, self.map_width),
            -1,
            dtype=np.int8
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
        # SUBSCRIBERS
        # ============================================================

        self.scan_sub = self.create_subscription(
            PointCloud2,
            self.scan_topic,
            self.scan_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            10
        )

        # ============================================================
        # PUBLISHERS
        # ============================================================

        self.map_pub = self.create_publisher(
            OccupancyGrid,
            '/map',
            1
        )

        # ============================================================
        # ODOM STATE
        # ============================================================

        self.last_odom_pose = None

        # map -> odom correction
        self.map_to_odom_x = 0.0
        self.map_to_odom_y = 0.0
        self.map_to_odom_yaw = 0.0

        self.scan_count = 0

        # Prevent excessive map publishing
        self.last_map_publish = self.get_clock().now()

        self.get_logger().info(
            '==========================================='
        )
        self.get_logger().info(
            'SimpleSLAM started'
        )
        self.get_logger().info(
            f'Scan topic : {self.scan_topic}'
        )
        self.get_logger().info(
            f'Odom topic : {self.odom_topic}'
        )
        self.get_logger().info(
            f'Map frame  : {self.map_frame}'
        )
        self.get_logger().info(
            f'Odom frame : {self.odom_frame}'
        )
        self.get_logger().info(
            f'Base frame : {self.base_frame}'
        )
        self.get_logger().info(
            f'Resolution  : {self.resolution} m/cell'
        )
        self.get_logger().info(
            'map -> odom is published ONLY by this node'
        )
        self.get_logger().info(
            '==========================================='
        )

    # ================================================================
    # ODOM CALLBACK
    # ================================================================

    def odom_callback(self, msg):

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation

        yaw = self.quaternion_to_yaw(
            q.x,
            q.y,
            q.z,
            q.w
        )

        self.last_odom_pose = (
            x,
            y,
            yaw
        )

    # ================================================================
    # SCAN CALLBACK
    # ================================================================

    def scan_callback(self, msg):

        self.scan_count += 1

        if self.last_odom_pose is None:
            self.get_logger().warn(
                'Waiting for /odom...'
            )
            return

        # ------------------------------------------------------------
        # Verify LiDAR -> base_link TF
        # ------------------------------------------------------------

        laser_frame = msg.header.frame_id

        if laser_frame == '':
            self.get_logger().warn(
                '/scan has empty frame_id'
            )
            return

        try:

            laser_to_base = self.tf_buffer.lookup_transform(
                self.base_frame,
                laser_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.2)
            )

        except TransformException as ex:

            self.get_logger().warn(
                f'No TF {laser_frame} -> '
                f'{self.base_frame}: {ex}'
            )

            return

        # Log TF once
        if self.scan_count == 1:

            self.get_logger().info(
                f'LiDAR frame detected: {laser_frame}'
            )

            self.get_logger().info(
                f'Using TF: {laser_frame} -> '
                f'{self.base_frame}'
            )

        # ------------------------------------------------------------
        # Read PointCloud2
        # ------------------------------------------------------------

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
                    x * x +
                    y * y +
                    z * z
                )

                if distance < self.min_range:
                    continue

                if distance > self.max_range:
                    continue

                points.append(
                    (x, y, z)
                )

        except Exception as ex:

            self.get_logger().error(
                f'PointCloud2 reading failed: {ex}'
            )

            return

        if len(points) == 0:
            return

        points = np.asarray(
            points,
            dtype=np.float32
        )

        # ------------------------------------------------------------
        # Transform LiDAR points -> base_link
        # ------------------------------------------------------------

        t = laser_to_base.transform.translation

        q = laser_to_base.transform.rotation

        tx = t.x
        ty = t.y
        tz = t.z

        qx = q.x
        qy = q.y
        qz = q.z
        qw = q.w

        points_base = self.transform_points(
            points,
            tx,
            ty,
            tz,
            qx,
            qy,
            qz,
            qw
        )

        # ------------------------------------------------------------
        # ODOM pose
        # ------------------------------------------------------------

        odom_x, odom_y, odom_yaw = (
            self.last_odom_pose
        )

        # ------------------------------------------------------------
        # Convert odom pose -> map pose
        # ------------------------------------------------------------

        map_x, map_y, map_yaw = (
            self.odom_to_map_pose(
                odom_x,
                odom_y,
                odom_yaw
            )
        )

        # ------------------------------------------------------------
        # Basic scan matching
        # ------------------------------------------------------------

        if (
            self.use_scan_matching
            and
            self.scan_count %
            self.match_every_n_scans == 0
        ):

            (
                map_x,
                map_y,
                map_yaw
            ) = self.scan_match(
                points_base,
                map_x,
                map_y,
                map_yaw
            )

            # Update map -> odom
            self.update_map_to_odom(
                map_x,
                map_y,
                map_yaw,
                odom_x,
                odom_y,
                odom_yaw
            )

        # ------------------------------------------------------------
        # Transform LiDAR points into MAP frame
        # ------------------------------------------------------------

        points_map = self.transform_points_2d(
            points_base,
            map_x,
            map_y,
            map_yaw
        )

        # ------------------------------------------------------------
        # Ray tracing
        # ------------------------------------------------------------

        self.update_map(
            map_x,
            map_y,
            points_map
        )

        # ------------------------------------------------------------
        # Publish map
        # ------------------------------------------------------------

        self.publish_map(
            msg.header.stamp
        )

        # ------------------------------------------------------------
        # Publish map -> odom TF
        # ------------------------------------------------------------

        self.publish_map_odom_tf(
            msg.header.stamp
        )

        # ------------------------------------------------------------
        # Statistics
        # ------------------------------------------------------------

        if (
            self.scan_count %
            self.log_every_n_scans == 0
        ):

            unknown = np.count_nonzero(
                self.grid == -1
            )

            free = np.count_nonzero(
                self.grid == 0
            )

            occupied = np.count_nonzero(
                self.grid == 100
            )

            total = self.grid.size

            self.get_logger().info(
                f'Scan #{self.scan_count} | '
                f'points={len(points)} | '
                f'unknown={unknown} '
                f'({unknown / total * 100:.1f}%) | '
                f'free={free} '
                f'({free / total * 100:.1f}%) | '
                f'occupied={occupied} '
                f'({occupied / total * 100:.1f}%) | '
                f'map={self.map_width}x'
                f'{self.map_height}'
            )

    # ================================================================
    # TRANSFORM POINTS
    # ================================================================

    def transform_points(
        self,
        points,
        tx,
        ty,
        tz,
        qx,
        qy,
        qz,
        qw
    ):

        # Rotation matrix from quaternion

        R = np.array([
            [
                1 - 2 * (qy*qy + qz*qz),
                2 * (qx*qy - qz*qw),
                2 * (qx*qz + qy*qw)
            ],
            [
                2 * (qx*qy + qz*qw),
                1 - 2 * (qx*qx + qz*qz),
                2 * (qy*qz - qx*qw)
            ],
            [
                2 * (qx*qz - qy*qw),
                2 * (qy*qz + qx*qw),
                1 - 2 * (qx*qx + qy*qy)
            ]
        ])

        return (
            points @ R.T
            +
            np.array(
                [tx, ty, tz],
                dtype=np.float32
            )
        )

    # ================================================================
    # 2D TRANSFORMATION
    # ================================================================

    def transform_points_2d(
        self,
        points,
        x,
        y,
        yaw
    ):

        c = math.cos(yaw)
        s = math.sin(yaw)

        px = points[:, 0]
        py = points[:, 1]

        mx = (
            c * px
            -
            s * py
            +
            x
        )

        my = (
            s * px
            +
            c * py
            +
            y
        )

        return np.column_stack(
            (mx, my)
        )

    # ================================================================
    # ODOM -> MAP
    # ================================================================

    def odom_to_map_pose(
        self,
        x,
        y,
        yaw
    ):

        c = math.cos(
            self.map_to_odom_yaw
        )

        s = math.sin(
            self.map_to_odom_yaw
        )

        mx = (
            c * x
            -
            s * y
            +
            self.map_to_odom_x
        )

        my = (
            s * x
            +
            c * y
            +
            self.map_to_odom_y
        )

        myaw = self.normalize_angle(
            yaw +
            self.map_to_odom_yaw
        )

        return (
            mx,
            my,
            myaw
        )

    # ================================================================
    # SCAN MATCHING
    # ================================================================

    def scan_match(
        self,
        points,
        x,
        y,
        yaw
    ):

        # Keep computation reasonable

        if len(points) > 100:

            indices = np.linspace(
                0,
                len(points) - 1,
                100,
                dtype=int
            )

            points = points[indices]

        best_x = x
        best_y = y
        best_yaw = yaw

        best_score = -1

        translation_steps = [
            -0.10,
            0.0,
            0.10
        ]

        rotation_steps = [
            math.radians(-4),
            0.0,
            math.radians(4)
        ]

        for dx in translation_steps:

            for dy in translation_steps:

                for da in rotation_steps:

                    test_x = x + dx
                    test_y = y + dy
                    test_yaw = (
                        yaw + da
                    )

                    transformed = (
                        self.transform_points_2d(
                            points,
                            test_x,
                            test_y,
                            test_yaw
                        )
                    )

                    score = self.score_scan(
                        transformed
                    )

                    if score > best_score:

                        best_score = score

                        best_x = test_x
                        best_y = test_y
                        best_yaw = test_yaw

        return (
            best_x,
            best_y,
            best_yaw
        )

    # ================================================================
    # SCAN SCORE
    # ================================================================

    def score_scan(
        self,
        points
    ):

        score = 0

        for p in points:

            gx, gy = self.world_to_grid(
                p[0],
                p[1]
            )

            if (
                0 <= gx < self.map_width
                and
                0 <= gy < self.map_height
            ):

                if self.grid[gy, gx] == 100:

                    score += 1

        return score

    # ================================================================
    # UPDATE MAP
    # ================================================================

    def update_map(
        self,
        robot_x,
        robot_y,
        points
    ):

        # Make sure robot is inside map

        self.expand_map_if_needed(
            robot_x,
            robot_y
        )

        robot_gx, robot_gy = (
            self.world_to_grid(
                robot_x,
                robot_y
            )
        )

        for p in points:

            end_x = float(p[0])
            end_y = float(p[1])

            self.expand_map_if_needed(
                end_x,
                end_y
            )

            end_gx, end_gy = (
                self.world_to_grid(
                    end_x,
                    end_y
                )
            )

            # --------------------------------------------------------
            # RAY TRACE
            # --------------------------------------------------------

            cells = self.bresenham(
                robot_gx,
                robot_gy,
                end_gx,
                end_gy
            )

            if len(cells) == 0:
                continue

            # All cells except final = FREE

            for gx, gy in cells[:-1]:

                if (
                    0 <= gx < self.map_width
                    and
                    0 <= gy < self.map_height
                ):

                    # Do not overwrite occupied cells

                    if self.grid[gy, gx] != 100:

                        self.grid[gy, gx] = 0

            # Final cell = OCCUPIED

            gx, gy = cells[-1]

            if (
                0 <= gx < self.map_width
                and
                0 <= gy < self.map_height
            ):

                self.grid[gy, gx] = 100

    # ================================================================
    # BRESENHAM RAY TRACING
    # ================================================================

    def bresenham(
        self,
        x0,
        y0,
        x1,
        y1
    ):

        cells = []

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)

        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1

        err = dx - dy

        while True:

            cells.append(
                (x0, y0)
            )

            if (
                x0 == x1
                and
                y0 == y1
            ):
                break

            e2 = 2 * err

            if e2 > -dy:

                err -= dy
                x0 += sx

            if e2 < dx:

                err += dx
                y0 += sy

        return cells

    # ================================================================
    # DYNAMIC MAP EXPANSION
    # ================================================================

    def expand_map_if_needed(
        self,
        x,
        y
    ):

        margin = 20

        gx, gy = self.world_to_grid(
            x,
            y
        )

        expand_left = gx < margin
        expand_right = (
            gx >= self.map_width - margin
        )

        expand_bottom = gy < margin
        expand_top = (
            gy >= self.map_height - margin
        )

        if not (
            expand_left
            or expand_right
            or expand_bottom
            or expand_top
        ):
            return

        old_grid = self.grid

        old_width = self.map_width
        old_height = self.map_height

        new_width = old_width
        new_height = old_height

        shift_x = 0
        shift_y = 0

        if expand_left:

            new_width += old_width // 2
            shift_x += old_width // 2

        if expand_right:

            new_width += old_width // 2

        if expand_bottom:

            new_height += old_height // 2
            shift_y += old_height // 2

        if expand_top:

            new_height += old_height // 2

        new_grid = np.full(
            (new_height, new_width),
            -1,
            dtype=np.int8
        )

        new_grid[
            shift_y:
            shift_y + old_height,
            shift_x:
            shift_x + old_width
        ] = old_grid

        self.grid = new_grid

        self.map_width = new_width
        self.map_height = new_height

        self.map_origin_x -= (
            shift_x *
            self.resolution
        )

        self.map_origin_y -= (
            shift_y *
            self.resolution
        )

        self.get_logger().info(
            f'Map expanded to '
            f'{new_width}x{new_height}'
        )

    # ================================================================
    # WORLD -> GRID
    # ================================================================

    def world_to_grid(
        self,
        x,
        y
    ):

        gx = int(
            math.floor(
                (x - self.map_origin_x)
                /
                self.resolution
            )
        )

        gy = int(
            math.floor(
                (y - self.map_origin_y)
                /
                self.resolution
            )
        )

        return gx, gy

    # ================================================================
    # MAP -> ODOM UPDATE
    # ================================================================

    def update_map_to_odom(
        self,
        map_x,
        map_y,
        map_yaw,
        odom_x,
        odom_y,
        odom_yaw
    ):

        correction_yaw = (
            map_yaw -
            odom_yaw
        )

        correction_yaw = (
            self.normalize_angle(
                correction_yaw
            )
        )

        c = math.cos(
            correction_yaw
        )

        s = math.sin(
            correction_yaw
        )

        correction_x = (
            map_x -
            (
                c * odom_x
                -
                s * odom_y
            )
        )

        correction_y = (
            map_y -
            (
                s * odom_x
                +
                c * odom_y
            )
        )

        self.map_to_odom_x = (
            correction_x
        )

        self.map_to_odom_y = (
            correction_y
        )

        self.map_to_odom_yaw = (
            correction_yaw
        )

    # ================================================================
    # PUBLISH MAP
    # ================================================================

    def publish_map(
        self,
        stamp
    ):

        msg = OccupancyGrid()

        msg.header.stamp = stamp
        msg.header.frame_id = (
            self.map_frame
        )

        msg.info.resolution = (
            self.resolution
        )

        msg.info.width = (
            self.map_width
        )

        msg.info.height = (
            self.map_height
        )

        msg.info.origin.position.x = (
            self.map_origin_x
        )

        msg.info.origin.position.y = (
            self.map_origin_y
        )

        msg.info.origin.position.z = 0.0

        msg.info.origin.orientation.w = 1.0

        msg.data = (
            self.grid.flatten()
            .tolist()
        )

        self.map_pub.publish(msg)

    # ================================================================
    # PUBLISH MAP -> ODOM TF
    # ================================================================

    def publish_map_odom_tf(
        self,
        stamp
    ):

        transform = TransformStamped()

        transform.header.stamp = stamp

        transform.header.frame_id = (
            self.map_frame
        )

        transform.child_frame_id = (
            self.odom_frame
        )

        transform.transform.translation.x = (
            self.map_to_odom_x
        )

        transform.transform.translation.y = (
            self.map_to_odom_y
        )

        transform.transform.translation.z = 0.0

        qx, qy, qz, qw = (
            self.yaw_to_quaternion(
                self.map_to_odom_yaw
            )
        )

        transform.transform.rotation.x = qx
        transform.transform.rotation.y = qy
        transform.transform.rotation.z = qz
        transform.transform.rotation.w = qw

        self.tf_broadcaster.sendTransform(
            transform
        )

    # ================================================================
    # QUATERNION -> YAW
    # ================================================================

    def quaternion_to_yaw(
        self,
        x,
        y,
        z,
        w
    ):

        siny_cosp = (
            2.0 *
            (w * z + x * y)
        )

        cosy_cosp = (
            1.0 -
            2.0 *
            (y * y + z * z)
        )

        return math.atan2(
            siny_cosp,
            cosy_cosp
        )

    # ================================================================
    # YAW -> QUATERNION
    # ================================================================

    def yaw_to_quaternion(
        self,
        yaw
    ):

        qz = math.sin(
            yaw / 2.0
        )

        qw = math.cos(
            yaw / 2.0
        )

        return (
            0.0,
            0.0,
            qz,
            qw
        )

    # ================================================================
    # NORMALIZE ANGLE
    # ================================================================

    def normalize_angle(
        self,
        angle
    ):

        while angle > math.pi:
            angle -= 2.0 * math.pi

        while angle < -math.pi:
            angle += 2.0 * math.pi

        return angle


# ====================================================================
# MAIN
# ====================================================================

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
