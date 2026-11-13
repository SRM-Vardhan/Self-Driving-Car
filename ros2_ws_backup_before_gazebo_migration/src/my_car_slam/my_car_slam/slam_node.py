import math
from collections import deque

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

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

        self.declare_parameter('min_range', 0.10)
        self.declare_parameter('max_range', 8.0)

        # STEP 1: Angular Binning Resolution
        self.declare_parameter('angular_resolution_deg', 1.0)

        # STEP 3: Reversible occupancy evidence
        self.declare_parameter('occupied_hit_log_odds', 0.95)
        self.declare_parameter('free_miss_log_odds', 0.10)
        self.declare_parameter('occupied_threshold', 1.0)
        self.declare_parameter('free_threshold', -1.0)
        self.declare_parameter('log_odds_min', -5.0)
        self.declare_parameter('log_odds_max', 5.0)

        # 3D PointCloud Z filtering
        self.declare_parameter('min_z', -0.20)
        self.declare_parameter('max_z', 0.50)

        # Dynamic map expansion margin
        self.declare_parameter('map_margin_cells', 20)

        # Scan matching disabled for testing map stability
        self.declare_parameter('use_scan_matching', False)
        self.declare_parameter('match_every_n_scans', 5)

        self.declare_parameter('min_occupied_cells_for_matching', 50)
        self.declare_parameter('min_scan_match_score', 5)
        self.declare_parameter('log_every_n_scans', 20)

        # Queue Parameters
        self.declare_parameter('max_scan_queue_size', 20)
        self.declare_parameter('tf_wait_timeout_sec', 0.5)

        # ============================================================
        # PARAMETERS -> VARIABLES
        # ============================================================

        self.scan_topic = self.get_parameter('scan_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.map_frame = self.get_parameter('map_frame').value
        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value

        self.resolution = float(self.get_parameter('resolution').value)
        self.initial_map_size = int(self.get_parameter('initial_map_size').value)
        self.min_range = float(self.get_parameter('min_range').value)
        self.max_range = float(self.get_parameter('max_range').value)
        
        self.angular_resolution_deg = float(self.get_parameter('angular_resolution_deg').value)

        self.occupied_hit_log_odds = float(self.get_parameter('occupied_hit_log_odds').value)
        self.free_miss_log_odds = float(self.get_parameter('free_miss_log_odds').value)
        self.occupied_threshold = float(self.get_parameter('occupied_threshold').value)
        self.free_threshold = float(self.get_parameter('free_threshold').value)
        self.log_odds_min = float(self.get_parameter('log_odds_min').value)
        self.log_odds_max = float(self.get_parameter('log_odds_max').value)

        self.min_z = float(self.get_parameter('min_z').value)
        self.max_z = float(self.get_parameter('max_z').value)
        self.map_margin_cells = int(self.get_parameter('map_margin_cells').value)

        self.use_scan_matching = bool(self.get_parameter('use_scan_matching').value)
        self.match_every_n_scans = int(self.get_parameter('match_every_n_scans').value)
        self.min_occupied_cells_for_matching = int(self.get_parameter('min_occupied_cells_for_matching').value)
        self.min_scan_match_score = int(self.get_parameter('min_scan_match_score').value)
        self.log_every_n_scans = int(self.get_parameter('log_every_n_scans').value)

        self.max_scan_queue_size = int(self.get_parameter('max_scan_queue_size').value)
        
        self.tf_wait_timeout_ns = int(self.get_parameter('tf_wait_timeout_sec').value * 1e9)

        # ============================================================
        # MAP
        # ============================================================

        self.map_width = self.initial_map_size
        self.map_height = self.initial_map_size

        self.map_origin_x = -(self.map_width * self.resolution / 2.0)
        self.map_origin_y = -(self.map_height * self.resolution / 2.0)

        # -1 = UNKNOWN, 0 = FREE, 100 = OCCUPIED
        self.grid = np.full((self.map_height, self.map_width), -1, dtype=np.int8)
        self.log_odds = np.zeros((self.map_height, self.map_width), dtype=np.float32)

        # ============================================================
        # TF
        # ============================================================

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # ============================================================
        # QUEUE & SUBSCRIBERS
        # ============================================================

        self.scan_queue = deque()

        self.scan_sub = self.create_subscription(
            PointCloud2,
            self.scan_topic,
            self.queue_scan_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            10
        )

        self.queue_timer = self.create_timer(0.02, self.process_scan_queue)

        # ============================================================
        # MAP QoS
        # ============================================================

        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        self.map_pub = self.create_publisher(OccupancyGrid, '/map', map_qos)

        # ============================================================
        # STATE
        # ============================================================

        self.last_odom_pose = None

        self.map_to_odom_x = 0.0
        self.map_to_odom_y = 0.0
        self.map_to_odom_yaw = 0.0

        self.scan_count = 0
        self._last_warning_times = {}

        self.last_match_score = 0
        self.last_match_valid = False
        self.last_occupied_count = 0

        # ============================================================
        # STARTUP LOGGING
        # ============================================================

        self.get_logger().info('==========================================')
        self.get_logger().info('SimpleSLAM started - FULLY OPTIMIZED')
        self.get_logger().info(f'/scan       : {self.scan_topic}')
        self.get_logger().info('==========================================')

    # ================================================================
    # ODOM CALLBACK
    # ================================================================

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = self.quaternion_to_yaw(q.x, q.y, q.z, q.w)
        self.last_odom_pose = (x, y, yaw)

    # ================================================================
    # SCAN QUEUEING
    # ================================================================

    def queue_scan_callback(self, msg):
        stamp_is_zero = (msg.header.stamp.sec == 0 and msg.header.stamp.nanosec == 0)
        
        if stamp_is_zero:
            self.warn_throttled(
                'zero_scan_stamp',
                'PointCloud2 stamp is zero; dropping to prevent TF mismatch.',
                interval_sec=5.0
            )
            return

        if not msg.header.frame_id:
            self.get_logger().warn('/scan has empty frame_id, dropping')
            return

        self.scan_queue.append(msg)

        if len(self.scan_queue) > self.max_scan_queue_size:
            self.scan_queue.popleft()
            self.warn_throttled(
                'queue_overflow',
                f'Scan queue exceeded {self.max_scan_queue_size}. Dropping oldest scan. Check TF.',
                interval_sec=2.0
            )

    # ================================================================
    # QUEUE PROCESSOR TIMER
    # ================================================================

    def process_scan_queue(self):
        if self.last_odom_pose is None:
            self.warn_throttled('wait_odom', 'Waiting for /odom...', interval_sec=2.0)
            return

        now_ns = self.get_clock().now().nanoseconds

        while self.scan_queue:
            msg = self.scan_queue[0]
            scan_time = Time.from_msg(msg.header.stamp)
            laser_frame = msg.header.frame_id

            try:
                laser_to_base = self.tf_buffer.lookup_transform(
                    self.base_frame, laser_frame, scan_time, timeout=Duration(seconds=0.0)
                )

                odom_to_base = self.tf_buffer.lookup_transform(
                    self.odom_frame, self.base_frame, scan_time, timeout=Duration(seconds=0.0)
                )

                self.scan_queue.popleft()
                self.process_scan(msg, laser_to_base, odom_to_base)

            except TransformException as ex:
                msg_ns = scan_time.nanoseconds
                
                if (now_ns - msg_ns) > self.tf_wait_timeout_ns:
                    self.scan_queue.popleft()
                    self.get_logger().debug(f'Scan timeout dropped: {ex}')
                    
                    if (now_ns - msg_ns) > 10_000_000_000:
                        self.warn_throttled(
                            'sim_time_warn',
                            'Massive time discrepancy detected. If using Gazebo, did you forget --ros-args -p use_sim_time:=true?',
                            interval_sec=5.0
                        )
                else:
                    break

    # ================================================================
    # SCAN PROCESSING LOGIC
    # ================================================================

    def process_scan(self, msg, laser_to_base, odom_to_base):
        self.scan_count += 1
        scan_time = Time.from_msg(msg.header.stamp)

        # ------------------------------------------------------------
        # READ POINTCLOUD
        # ------------------------------------------------------------
        raw_points = []
        try:
            for p in point_cloud2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True):
                x, y, z = float(p[0]), float(p[1]), float(p[2])
                distance = math.sqrt(x * x + y * y + z * z)

                if distance < self.min_range or distance > self.max_range:
                    continue

                raw_points.append((x, y, z))
        except Exception as ex:
            self.get_logger().error(f'PointCloud2 error: {ex}')
            return

        if not raw_points:
            return

        points = np.asarray(raw_points, dtype=np.float32)

        # ------------------------------------------------------------
        # LASER -> BASE_LINK
        # ------------------------------------------------------------
        t = laser_to_base.transform.translation
        q = laser_to_base.transform.rotation

        points_base = self.transform_points(
            points, t.x, t.y, t.z, q.x, q.y, q.z, q.w
        )

        # ------------------------------------------------------------
        # Z FILTER
        # ------------------------------------------------------------
        z_mask = (points_base[:, 2] >= self.min_z) & (points_base[:, 2] <= self.max_z)
        
        # Apply mask to BOTH base frame and original laser frame points
        points_base = points_base[z_mask]
        points_laser = points[z_mask]

        if len(points_base) == 0:
            return

        points_xy = points_base[:, :2]

        # ------------------------------------------------------------
        # STEP 1: ANGULAR BINNING (From original LiDAR frame)
        # ------------------------------------------------------------
        # Calculate ranges and angles from the actual sensor origin
        ranges = np.linalg.norm(points_laser[:, :2], axis=1)
        angles = np.arctan2(points_laser[:, 1], points_laser[:, 0])
        angular_res = math.radians(self.angular_resolution_deg)
        
        # Calculate integer bin indices based on angle
        bin_indices = np.floor((angles + math.pi) / angular_res).astype(np.int32)
        
        # Sort points by range (closest first)
        sort_idx = np.argsort(ranges)
        sorted_bins = bin_indices[sort_idx]
        sorted_points_xy = points_xy[sort_idx] # Sort the mapped points based on laser distances
        
        # Extract the first (closest) point per unique bin
        _, unique_idx = np.unique(sorted_bins, return_index=True)
        points_xy = sorted_points_xy[unique_idx]

        # ------------------------------------------------------------
        # ODOM POSE
        # ------------------------------------------------------------
        odom_translation = odom_to_base.transform.translation
        odom_rotation = odom_to_base.transform.rotation

        odom_x = odom_translation.x
        odom_y = odom_translation.y
        odom_yaw = self.quaternion_to_yaw(
            odom_rotation.x, odom_rotation.y, odom_rotation.z, odom_rotation.w
        )

        map_x, map_y, map_yaw = self.odom_to_map_pose(odom_x, odom_y, odom_yaw)

        # ------------------------------------------------------------
        # SCAN MATCHING
        # ------------------------------------------------------------
        occupied_count = int(np.count_nonzero(self.grid == 100))
        self.last_occupied_count = occupied_count

        match_score = 0
        match_valid = False

        if self.use_scan_matching and self.scan_count % self.match_every_n_scans == 0:
            matched_x, matched_y, matched_yaw, match_score, match_valid = self.scan_match(
                points_xy, map_x, map_y, map_yaw, occupied_count
            )

            if match_valid:
                map_x, map_y, map_yaw = matched_x, matched_y, matched_yaw
                self.update_map_to_odom(map_x, map_y, map_yaw, odom_x, odom_y, odom_yaw)

        self.last_match_score = match_score
        self.last_match_valid = match_valid

        # ------------------------------------------------------------
        # STEP 2: TRACE FROM ACTUAL LIDAR ORIGIN
        # ------------------------------------------------------------
        c_map = math.cos(map_yaw)
        s_map = math.sin(map_yaw)
        
        # Laser translation in base_link frame
        lx_base = t.x
        ly_base = t.y
        
        # Laser origin in global map frame
        laser_map_x = map_x + (c_map * lx_base - s_map * ly_base)
        laser_map_y = map_y + (s_map * lx_base + c_map * ly_base)

        # ------------------------------------------------------------
        # TRANSFORM WHOLE SCAN INTO MAP FRAME
        # ------------------------------------------------------------
        points_map = self.transform_points_2d(points_xy, map_x, map_y, map_yaw)
        
        # Expand map bounds based on laser origin and points
        self.expand_map_for_scan(laser_map_x, laser_map_y, points_map)

        # Convert laser origin to grid coordinates for Bresenham
        laser_gx, laser_gy = self.world_to_grid(laser_map_x, laser_map_y)
        self.update_map(laser_gx, laser_gy, points_map)

        # ------------------------------------------------------------
        # PUBLISH MAP & TF
        # ------------------------------------------------------------
        self.publish_map(msg.header.stamp)
        self.publish_map_odom_tf(msg.header.stamp)

        if self.scan_count % self.log_every_n_scans == 0:
            unknown = int(np.count_nonzero(self.grid == -1))
            total = self.grid.size
            self.get_logger().info(
                f'SCAN #{self.scan_count} | '
                f'binned_points={len(points_xy)} | '
                f'queue_size={len(self.scan_queue)} | '
                f'unknown={unknown} ({unknown / total * 100:.1f}%) | '
                f'match_score={match_score}'
            )

    # ================================================================
    # POINT TRANSFORMATION
    # ================================================================

    def transform_points(self, points, tx, ty, tz, qx, qy, qz, qw):
        R = np.array([
            [1 - 2 * (qy*qy + qz*qz), 2 * (qx*qy - qz*qw), 2 * (qx*qz + qy*qw)],
            [2 * (qx*qy + qz*qw), 1 - 2 * (qx*qx + qz*qz), 2 * (qy*qz - qx*qw)],
            [2 * (qx*qz - qy*qw), 2 * (qy*qz + qx*qw), 1 - 2 * (qx*qx + qy*qy)]
        ])
        translation = np.array([tx, ty, tz], dtype=np.float32)
        return points @ R.T + translation

    # ================================================================
    # 2D TRANSFORMATION
    # ================================================================

    def transform_points_2d(self, points, x, y, yaw):
        c = math.cos(yaw)
        s = math.sin(yaw)
        px = points[:, 0]
        py = points[:, 1]
        mx = c * px - s * py + x
        my = s * px + c * py + y
        return np.column_stack((mx, my))

    # ================================================================
    # ODOM -> MAP
    # ================================================================

    def odom_to_map_pose(self, x, y, yaw):
        c = math.cos(self.map_to_odom_yaw)
        s = math.sin(self.map_to_odom_yaw)
        mx = c * x - s * y + self.map_to_odom_x
        my = s * x + c * y + self.map_to_odom_y
        myaw = self.normalize_angle(yaw + self.map_to_odom_yaw)
        return mx, my, myaw

    # ================================================================
    # SCAN MATCHING
    # ================================================================

    def scan_match(self, points, x, y, yaw, occupied_count):
        if occupied_count < self.min_occupied_cells_for_matching:
            return x, y, yaw, 0, False

        # Points are already culled by the binning, but cap at 100 for matcher speed
        if len(points) > 100:
            indices = np.linspace(0, len(points) - 1, 100, dtype=int)
            points = points[indices]

        best_x, best_y, best_yaw = x, y, yaw
        best_score = 0

        translation_steps = [-0.10, 0.0, 0.10]
        rotation_steps = [math.radians(-4), 0.0, math.radians(4)]

        for dx in translation_steps:
            for dy in translation_steps:
                for da in rotation_steps:
                    test_x, test_y, test_yaw = x + dx, y + dy, yaw + da
                    transformed = self.transform_points_2d(points, test_x, test_y, test_yaw)
                    score = self.score_scan(transformed)

                    if score > best_score:
                        best_score = score
                        best_x, best_y, best_yaw = test_x, test_y, test_yaw

        if best_score < self.min_scan_match_score:
            return x, y, yaw, best_score, False

        return best_x, best_y, best_yaw, best_score, True

    # ================================================================
    # SCAN SCORE
    # ================================================================

    def score_scan(self, points):
        score = 0
        for p in points:
            gx, gy = self.world_to_grid(p[0], p[1])
            if 0 <= gx < self.map_width and 0 <= gy < self.map_height:
                if self.grid[gy, gx] == 100:
                    score += 1
        return score

    # ================================================================
    # DYNAMIC MAP EXPANSION
    # ================================================================

    def expand_map_for_scan(self, robot_x, robot_y, points):
        min_x, max_x = robot_x, robot_x
        min_y, max_y = robot_y, robot_y

        if len(points) > 0:
            min_x = min(min_x, float(np.min(points[:, 0])))
            max_x = max(max_x, float(np.max(points[:, 0])))
            min_y = min(min_y, float(np.min(points[:, 1])))
            max_y = max(max_y, float(np.max(points[:, 1])))

        margin = self.map_margin_cells * self.resolution
        required_min_x, required_max_x = min_x - margin, max_x + margin
        required_min_y, required_max_y = min_y - margin, max_y + margin

        current_min_x = self.map_origin_x
        current_min_y = self.map_origin_y
        current_max_x = self.map_origin_x + self.map_width * self.resolution
        current_max_y = self.map_origin_y + self.map_height * self.resolution

        needs_expansion = (
            required_min_x < current_min_x or required_max_x > current_max_x or
            required_min_y < current_min_y or required_max_y > current_max_y
        )

        if not needs_expansion:
            return

        new_min_x = min(current_min_x, required_min_x)
        new_max_x = max(current_max_x, required_max_x)
        new_min_y = min(current_min_y, required_min_y)
        new_max_y = max(current_max_y, required_max_y)

        new_width = int(math.ceil((new_max_x - new_min_x) / self.resolution))
        new_height = int(math.ceil((new_max_y - new_min_y) / self.resolution))

        new_grid = np.full((new_height, new_width), -1, dtype=np.int8)
        new_log_odds = np.zeros((new_height, new_width), dtype=np.float32)

        offset_x = int(round((self.map_origin_x - new_min_x) / self.resolution))
        offset_y = int(round((self.map_origin_y - new_min_y) / self.resolution))

        new_grid[offset_y:offset_y + self.map_height, offset_x:offset_x + self.map_width] = self.grid
        new_log_odds[offset_y:offset_y + self.map_height, offset_x:offset_x + self.map_width] = self.log_odds

        self.grid = new_grid
        self.log_odds = new_log_odds
        self.map_width = new_width
        self.map_height = new_height
        self.map_origin_x = new_min_x
        self.map_origin_y = new_min_y

    # ================================================================
    # MAP UPDATE / RAY TRACING
    # ================================================================

    def update_map(self, laser_gx, laser_gy, points):
        free_cells = set()
        hit_cells = set()

        for p in points:
            end_gx, end_gy = self.world_to_grid(float(p[0]), float(p[1]))
            cells = self.bresenham(laser_gx, laser_gy, end_gx, end_gy)
            
            if not cells:
                continue

            for gx, gy in cells[:-1]:
                if 0 <= gx < self.map_width and 0 <= gy < self.map_height:
                    free_cells.add((gx, gy))

            gx, gy = cells[-1]
            if 0 <= gx < self.map_width and 0 <= gy < self.map_height:
                hit_cells.add((gx, gy))

        free_cells.difference_update(hit_cells)

        for gx, gy in free_cells:
            self.log_odds[gy, gx] = max(
                self.log_odds_min, self.log_odds[gy, gx] - self.free_miss_log_odds
            )
            self.update_cell_state(gx, gy)

        for gx, gy in hit_cells:
            self.log_odds[gy, gx] = min(
                self.log_odds_max, self.log_odds[gy, gx] + self.occupied_hit_log_odds
            )
            self.update_cell_state(gx, gy)

    def update_cell_state(self, gx, gy):
        evidence = self.log_odds[gy, gx]
        if evidence >= self.occupied_threshold:
            self.grid[gy, gx] = 100
        elif evidence <= self.free_threshold:
            self.grid[gy, gx] = 0

    def warn_throttled(self, key, message, interval_sec=2.0):
        now = self.get_clock().now().nanoseconds / 1e9
        last = self._last_warning_times.get(key, -1e9)
        if now - last >= interval_sec:
            self.get_logger().warning(message)
            self._last_warning_times[key] = now

    # ================================================================
    # BRESENHAM
    # ================================================================

    def bresenham(self, x0, y0, x1, y1):
        cells = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            cells.append((x0, y0))
            if x0 == x1 and y0 == y1:
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
    # WORLD -> GRID
    # ================================================================

    def world_to_grid(self, x, y):
        gx = int(math.floor((x - self.map_origin_x) / self.resolution))
        gy = int(math.floor((y - self.map_origin_y) / self.resolution))
        return gx, gy

    # ================================================================
    # UPDATE map -> odom
    # ================================================================

    def update_map_to_odom(self, map_x, map_y, map_yaw, odom_x, odom_y, odom_yaw):
        correction_yaw = self.normalize_angle(map_yaw - odom_yaw)
        c = math.cos(correction_yaw)
        s = math.sin(correction_yaw)
        correction_x = map_x - (c * odom_x - s * odom_y)
        correction_y = map_y - (s * odom_x + c * odom_y)

        self.map_to_odom_x = correction_x
        self.map_to_odom_y = correction_y
        self.map_to_odom_yaw = correction_yaw

    # ================================================================
    # PUBLISH MAP
    # ================================================================

    def publish_map(self, stamp):
        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = self.map_frame
        msg.info.resolution = self.resolution
        msg.info.width = self.map_width
        msg.info.height = self.map_height
        msg.info.origin.position.x = self.map_origin_x
        msg.info.origin.position.y = self.map_origin_y
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        msg.data = self.grid.flatten().tolist()
        self.map_pub.publish(msg)

    # ================================================================
    # PUBLISH map -> odom TF
    # ================================================================

    def publish_map_odom_tf(self, stamp):
        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = self.map_frame
        transform.child_frame_id = self.odom_frame
        transform.transform.translation.x = self.map_to_odom_x
        transform.transform.translation.y = self.map_to_odom_y
        transform.transform.translation.z = 0.0
        qx, qy, qz, qw = self.yaw_to_quaternion(self.map_to_odom_yaw)
        transform.transform.rotation.x = qx
        transform.transform.rotation.y = qy
        transform.transform.rotation.z = qz
        transform.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(transform)

    # ================================================================
    # QUATERNION -> YAW
    # ================================================================

    def quaternion_to_yaw(self, x, y, z, w):
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    # ================================================================
    # YAW -> QUATERNION
    # ================================================================

    def yaw_to_quaternion(self, yaw):
        return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)

    # ================================================================
    # NORMALIZE ANGLE
    # ================================================================

    def normalize_angle(self, angle):
        while angle > math.pi: angle -= 2.0 * math.pi
        while angle < -math.pi: angle += 2.0 * math.pi
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