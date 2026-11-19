"""
MPC-Based Local Planner for Autonomous Vehicles

Features:
1. MPC path tracking with kinematic bicycle model (do-mpc / IPOPT).
2. Obstacle-aware left/right trajectory selection.
3. LiDAR coordinate transformation using TF2.
4. Adaptive speed and curvature control.
5. Goal approach, stopping and goal-state latch.
6. Safety validation and zero-velocity fallback.
7. Lane offset is optional: NaN / stale / missing offset is ignored and
   the planner keeps running on the global path + obstacle layers.

ROS 2 Humble
"""

import time
import numpy as np
import casadi as ca
import do_mpc

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Float32

from tf2_ros import Buffer, TransformListener, TransformException


# ================================================================
# GENERAL HELPERS
# ================================================================

def quaternion_to_yaw(q):
    return float(np.arctan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    ))


def normalize_angle(angle):
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def distance_2d(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def quaternion_rotation_matrix(q):
    """Convert a quaternion into a 3D rotation matrix."""

    x, y, z, w = q.x, q.y, q.z, q.w

    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ], dtype=float)


def resample_path(xy, spacing, max_length):
    """
    Resample a polyline at a fixed spacing and rebuild the heading
    from the path geometry (does not trust the pose orientations).

    Returns a list of {'x','y','theta'} or None if the path is too short.
    """

    xy = np.asarray(xy, dtype=float)

    if len(xy) < 2:
        return None

    d = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))

    # Remove duplicate points.
    keep = np.concatenate(([True], d > 1e-3))
    xy = xy[keep]

    if len(xy) < 2:
        return None

    d = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
    s = np.concatenate(([0.0], np.cumsum(d)))

    total = min(float(s[-1]), float(max_length))
    count = max(2, int(total / spacing) + 1)

    s_new = np.linspace(0.0, total, count)

    x = np.interp(s_new, s, xy[:, 0])
    y = np.interp(s_new, s, xy[:, 1])

    theta = np.arctan2(np.gradient(y), np.gradient(x))

    return [
        {'x': float(x[i]), 'y': float(y[i]), 'theta': float(theta[i])}
        for i in range(count)
    ]


# ================================================================
# OBSTACLE-AWARE PATH PLANNER
# ================================================================

class ObstaclePathPlanner:

    def __init__(self, node, params):

        self.node = node
        self.logger = node.get_logger()

        self.max_offset = params['max_avoidance_offset']
        self.vehicle_radius = params['vehicle_half_width']
        self.clearance = params['min_clearance']

        self.emergency_distance = params['emergency_stop_distance']
        self.lookahead = params['obstacle_lookahead']

        self.last_selected_offset = 0.0

    def choose_safe_path(self, trajectory, obstacles):

        """
        Evaluate candidate lateral offsets.

        Positive offset = left of path.
        Negative offset = right of path.

        Returns:
            modified trajectory, selected offset, success
        """

        if not trajectory:
            return None, 0.0, False

        if obstacles is None or len(obstacles) == 0:
            self.last_selected_offset = 0.0
            return trajectory, 0.0, True

        candidates = [
            0.0,
            -0.15, 0.15,
            -0.30, 0.30,
            -0.45, 0.45,
            -0.60, 0.60,
            -0.75, 0.75,
            -self.max_offset, self.max_offset
        ]

        candidates = sorted(set(
            float(np.clip(v, -self.max_offset, self.max_offset))
            for v in candidates
        ))

        tx = np.array([p['x'] for p in trajectory], dtype=float)
        ty = np.array([p['y'] for p in trajectory], dtype=float)
        th = np.array([p['theta'] for p in trajectory], dtype=float)

        # Cumulative distance along the reference trajectory.
        seg = np.hypot(np.diff(tx), np.diff(ty))
        s = np.concatenate(([0.0], np.cumsum(seg)))

        # Smoothly introduce the lateral offset before reaching near obstacles.
        ramp = np.clip((s - 0.10) / 0.75, 0.0, 1.0)
        ramp = ramp * ramp * (3.0 - 2.0 * ramp)

        # Check clearance ahead of the front axle so side-adjacent pillars
        # already being cleared do not lock every candidate path.
        fwd_idx = s >= 0.25
        if not np.any(fwd_idx):
            fwd_idx = np.ones_like(s, dtype=bool)

        ox = np.asarray(obstacles)[:, 0][None, :]
        oy = np.asarray(obstacles)[:, 1][None, :]

        collision_radius = self.vehicle_radius + self.clearance

        best = None
        best_offset = 0.0
        best_score = float('inf')

        for offset in candidates:

            applied = offset * ramp

            px = tx - applied * np.sin(th)
            py = ty + applied * np.cos(th)

            # (forward path points x obstacle points) distance matrix.
            d = np.hypot(px[fwd_idx, None] - ox, py[fwd_idx, None] - oy)
            min_distance = float(d.min())

            # Reject paths that violate the safety clearance.
            if min_distance < collision_radius:
                continue

            # Prefer the smallest safe offset and avoid flipping sides.
            score = (
                abs(offset)
                + 0.25 * abs(offset - self.last_selected_offset)
                - 0.03 * min(min_distance, 2.0)
            )

            if score < best_score:
                best_score = score
                best = (px, py, applied)
                best_offset = offset

        if best is None:

            self.logger.error(
                "No collision-free avoidance trajectory found.",
                throttle_duration_sec=1.0
            )

            return None, 0.0, False

        px, py, applied = best

        best_path = [
            {
                'x': float(px[i]),
                'y': float(py[i]),
                'theta': float(th[i]),
                'offset': float(applied[i])
            }
            for i in range(len(px))
        ]

        if abs(best_offset) > 0.05:

            direction = "LEFT" if best_offset > 0 else "RIGHT"

            self.logger.warning(
                f"Obstacle avoidance: TURN {direction}, "
                f"offset={best_offset:.2f}m",
                throttle_duration_sec=1.0
            )

        self.last_selected_offset = best_offset

        return best_path, best_offset, True


# ================================================================
# SAFETY AND SPEED CONTROLLER
# ================================================================

class SafetyController:

    def __init__(self, node, params):

        self.node = node
        self.logger = node.get_logger()

        self.max_velocity = params['max_velocity']
        self.max_acceleration = params['max_acceleration']

        self.wheelbase = params['wheelbase']
        self.max_steering = params['max_steering_angle']

        self.mu = params['friction_coefficient']

        self.emergency_distance = params['emergency_stop_distance']
        self.vehicle_radius = params['vehicle_half_width']
        self.clearance = params['min_clearance']

    def curvature_speed(self, path, index, target_speed):

        """
        Lowest safe cornering speed over the whole local path.
        Curvature is measured on 3 points spaced ~0.5 m apart
        so waypoint noise does not create false sharp curves.
        """

        step = 5

        if len(path) < 2 * step + 1:
            return target_speed

        pts = np.array([[p['x'], p['y']] for p in path], dtype=float)

        p0 = pts[:-2 * step]
        p1 = pts[step:-step]
        p2 = pts[2 * step:]

        a = np.linalg.norm(p1 - p0, axis=1)
        b = np.linalg.norm(p2 - p1, axis=1)
        c = np.linalg.norm(p2 - p0, axis=1)

        cross = np.abs(
            (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
            - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
        )

        valid = (a > 1e-4) & (b > 1e-4) & (c > 1e-4)

        if not np.any(valid):
            return target_speed

        curvature = 2.0 * cross[valid] / (
            a[valid] * b[valid] * c[valid] + 1e-9
        )

        max_curvature = float(np.max(curvature))

        if max_curvature < 1e-5:
            return target_speed

        lateral_acceleration = 0.6 * self.mu * 9.81

        curve_speed = np.sqrt(lateral_acceleration / max_curvature)

        return float(min(target_speed, curve_speed))

    def obstacle_speed(self, obstacles, trajectory):

        """
        Check obstacles against the selected avoidance trajectory.
        """

        if obstacles is None or len(obstacles) == 0:
            return self.max_velocity

        if not trajectory:
            return 0.0

        tx = np.array([p['x'] for p in trajectory], dtype=float)
        ty = np.array([p['y'] for p in trajectory], dtype=float)

        seg = np.hypot(np.diff(tx), np.diff(ty))
        s = np.concatenate(([0.0], np.cumsum(seg)))
        fwd_idx = s >= 0.25
        if not np.any(fwd_idx):
            fwd_idx = np.ones_like(s, dtype=bool)

        obs = np.asarray(obstacles)

        d = np.hypot(
            tx[fwd_idx, None] - obs[:, 0][None, :],
            ty[fwd_idx, None] - obs[:, 1][None, :]
        )

        min_distance = float(d.min())

        safe_radius = self.vehicle_radius + self.clearance

        if min_distance < safe_radius:
            return 0.0

        # Gradually reduce speed as clearance becomes smaller.
        if min_distance < safe_radius + 1.0:

            ratio = np.clip(
                (min_distance - safe_radius) / 1.0,
                0.0,
                1.0
            )

            return float(self.max_velocity * (0.25 + 0.75 * ratio))

        return self.max_velocity

    def compute_speed(
        self,
        path,
        index,
        obstacles,
        target_speed,
        goal_distance
    ):

        curve_speed = self.curvature_speed(path, index, target_speed)

        obstacle_speed = self.obstacle_speed(obstacles, path)

        # Braking speed based on remaining distance.
        braking_speed = np.sqrt(
            max(0.0, 1.0 * self.max_acceleration * max(0.0, goal_distance - 0.15))
        )

        return max(
            0.0,
            min(
                target_speed,
                curve_speed,
                obstacle_speed,
                braking_speed
            )
        )


# ================================================================
# MPC CONTROLLER
# ================================================================

class MPCController:

    def __init__(self, node, params):

        self.node = node
        self.logger = node.get_logger()

        self.horizon = params['horizon']
        self.dt = params['dt']

        self.wheelbase = params['wheelbase']
        self.max_velocity = params['max_velocity']
        self.max_acceleration = params['max_acceleration']
        self.max_steering = params['max_steering_angle']

        self._setup_mpc()

    def _setup_mpc(self):

        model = do_mpc.model.Model('continuous')

        # States
        x = model.set_variable('_x', 'x')
        y = model.set_variable('_x', 'y')
        theta = model.set_variable('_x', 'theta')
        v = model.set_variable('_x', 'v')

        # Inputs
        a = model.set_variable('_u', 'a')
        delta = model.set_variable('_u', 'delta')

        # Time-varying references
        x_ref = model.set_variable('_tvp', 'x_ref')
        y_ref = model.set_variable('_tvp', 'y_ref')
        theta_ref = model.set_variable('_tvp', 'theta_ref')
        v_ref = model.set_variable('_tvp', 'v_ref')

        # Kinematic bicycle model
        model.set_rhs('x', v * ca.cos(theta))
        model.set_rhs('y', v * ca.sin(theta))
        model.set_rhs('theta', v * ca.tan(delta) / self.wheelbase)
        model.set_rhs('v', a)

        model.setup()

        mpc = do_mpc.controller.MPC(model)

        mpc.set_param(
            n_horizon=self.horizon,
            t_step=self.dt,
            n_robust=0,
            store_full_solution=True,
            nlpsol_opts={
                'ipopt.print_level': 0,
                'print_time': 0
            }
        )

        # Cost weights
        Qx = 15.0
        Qy = 15.0
        Qtheta = 8.0
        Qv = 3.0

        Ra = 0.5
        Rdelta = 0.8

        heading_error = ca.atan2(
            ca.sin(theta - theta_ref),
            ca.cos(theta - theta_ref)
        )

        tracking_cost = (
            Qx * (x - x_ref)**2
            + Qy * (y - y_ref)**2
            + Qtheta * heading_error**2
            + Qv * (v - v_ref)**2
        )

        control_cost = (
            Ra * a**2
            + Rdelta * delta**2
        )

        mpc.set_objective(
            mterm=tracking_cost,
            lterm=tracking_cost + control_cost
        )

        mpc.set_rterm(a=0.5, delta=1.0)

        # Physical constraints
        mpc.bounds['lower', '_u', 'a'] = -self.max_acceleration
        mpc.bounds['upper', '_u', 'a'] = self.max_acceleration

        mpc.bounds['lower', '_u', 'delta'] = -self.max_steering
        mpc.bounds['upper', '_u', 'delta'] = self.max_steering

        mpc.bounds['lower', '_x', 'v'] = 0.0
        mpc.bounds['upper', '_x', 'v'] = self.max_velocity

        # TVP template (filled every control step)
        self.tvp_template = mpc.get_tvp_template()

        def tvp_function(t_now):
            return self.tvp_template

        mpc.set_tvp_fun(tvp_function)

        mpc.setup()

        self.mpc = mpc
        self.initialized = False

        self.logger.info("MPC setup completed.")

    def compute_control(self, state, trajectory, target_speed):

        if not trajectory:
            return None

        try:

            state = np.asarray(state, dtype=float).reshape(4, 1)

            if not np.all(np.isfinite(state)):
                return None

            self.mpc.x0 = state

            if not self.initialized:
                self.mpc.set_initial_guess()
                self.initialized = True

            # Build the reference by DISTANCE, not by waypoint index:
            # step k should be where the vehicle will be after k*dt.
            xs = np.array([p['x'] for p in trajectory], dtype=float)
            ys = np.array([p['y'] for p in trajectory], dtype=float)
            ths = np.unwrap(
                np.array([p['theta'] for p in trajectory], dtype=float)
            )

            seg = np.hypot(np.diff(xs), np.diff(ys))
            s = np.concatenate(([0.0], np.cumsum(seg)))

            ref_speed = max(float(target_speed), 0.1)

            for k in range(self.horizon + 1):

                sk = min(k * self.dt * ref_speed, float(s[-1]))

                self.tvp_template['_tvp', k, 'x_ref'] = float(
                    np.interp(sk, s, xs)
                )
                self.tvp_template['_tvp', k, 'y_ref'] = float(
                    np.interp(sk, s, ys)
                )
                self.tvp_template['_tvp', k, 'theta_ref'] = float(
                    np.interp(sk, s, ths)
                )
                self.tvp_template['_tvp', k, 'v_ref'] = float(
                    target_speed
                )

            # Solve
            self.mpc.make_step(state)

            # Reject the result if IPOPT did not converge.
            try:
                solved = bool(np.asarray(self.mpc.data['success'])[-1])
            except Exception:
                solved = True

            if not solved:
                self.logger.warning(
                    "MPC solver did not converge.",
                    throttle_duration_sec=1.0
                )
                return None

            def extract_state(name):
                return np.asarray(
                    self.mpc.data.prediction(('_x', name), t_ind=-1)
                ).reshape(-1)

            def extract_input(name):
                return np.asarray(
                    self.mpc.data.prediction(('_u', name), t_ind=-1)
                ).reshape(-1)

            return {
                'x': extract_state('x'),
                'y': extract_state('y'),
                'theta': extract_state('theta'),
                'v': extract_state('v'),
                'a': extract_input('a'),
                'delta': extract_input('delta')
            }

        except Exception as e:

            self.logger.error(f"MPC calculation failed: {e}")

            return None


# ================================================================
# MAIN LOCAL PLANNER NODE
# ================================================================

class LocalPlannerNode(Node):

    PATH_SPACING = 0.1   # metres between resampled path points

    def __init__(self):

        super().__init__('local_planner_node')

        # --------------------------------------------------------
        # PARAMETERS
        # --------------------------------------------------------

        self.declare_parameters(
            namespace='',
            parameters=[

                ('mpc.horizon', 10),
                ('mpc.dt', 0.1),

                ('vehicle.wheelbase', 0.50),
                ('vehicle.max_velocity', 0.8),
                ('vehicle.max_acceleration', 0.5),
                ('vehicle.max_steering_angle', 30.0),
                ('vehicle.half_width', 0.22),

                ('frames.base_link', 'base_link'),

                ('safety.min_clearance', 0.08),
                ('safety.emergency_stop_distance', 0.32),
                ('safety.obstacle_lookahead', 2.0),
                ('safety.max_avoidance_offset', 1.0),
                ('safety.friction_coefficient', 0.6),

                ('lane.max_offset', 1.0),

                ('goal.position_tolerance', 0.30),
                ('goal.reset_distance', 0.50),

                ('topics.path', '/global_path'),
                ('topics.odom', '/ekf/odom'),
                ('topics.scan', '/scan'),
                ('topics.lane_offset', '/lane_offset'),
                ('topics.cmd_vel', '/cmd_vel')
            ]
        )

        def get(name):
            return self.get_parameter(name).value

        self.dt = get('mpc.dt')

        self.base_frame = get('frames.base_link')

        self.params = {

            'horizon': get('mpc.horizon'),
            'dt': self.dt,

            'wheelbase': get('vehicle.wheelbase'),
            'max_velocity': get('vehicle.max_velocity'),
            'max_acceleration': get('vehicle.max_acceleration'),

            'max_steering_angle': np.radians(
                get('vehicle.max_steering_angle')
            ),

            'vehicle_half_width': get('vehicle.half_width'),

            'min_clearance': get('safety.min_clearance'),
            'emergency_stop_distance': get(
                'safety.emergency_stop_distance'
            ),

            'obstacle_lookahead': get('safety.obstacle_lookahead'),

            'max_avoidance_offset': get('safety.max_avoidance_offset'),

            'friction_coefficient': get('safety.friction_coefficient')
        }

        self.max_lane_offset = get('lane.max_offset')

        self.goal_tolerance = get('goal.position_tolerance')
        self.goal_reset_distance = get('goal.reset_distance')

        # --------------------------------------------------------
        # TF2
        # --------------------------------------------------------

        self.tf_buffer = Buffer()

        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --------------------------------------------------------
        # SUBSCRIPTIONS
        # --------------------------------------------------------

        self.path_sub = self.create_subscription(
            Path, get('topics.path'), self.path_callback, 10
        )

        self.odom_sub = self.create_subscription(
            Odometry, get('topics.odom'), self.odom_callback, 10
        )

        self.scan_sub = self.create_subscription(
            PointCloud2, get('topics.scan'), self.scan_callback, 10
        )

        self.lane_sub = self.create_subscription(
            Float32, get('topics.lane_offset'), self.lane_callback, 10
        )

        # --------------------------------------------------------
        # PUBLISHER
        # --------------------------------------------------------

        self.cmd_pub = self.create_publisher(
            Twist, get('topics.cmd_vel'), 10
        )

        # --------------------------------------------------------
        # STATE
        # --------------------------------------------------------

        self.current_state = None
        self.reference_path = None

        self.obstacles_base = None

        self.last_odom_time = None
        self.last_scan_time = None

        self.odom_frame = ''

        self.lane_offset = 0.0
        self.lane_time = None

        self.goal_reached = False
        self.last_goal = None

        self.control_count = 0

        # --------------------------------------------------------
        # CONTROLLERS
        # --------------------------------------------------------

        self.mpc = MPCController(self, self.params)

        self.obstacle_planner = ObstaclePathPlanner(self, self.params)

        self.safety = SafetyController(self, self.params)

        self.timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info("MPC Local Planner initialized.")

    # ============================================================
    # CALLBACKS
    # ============================================================

    def path_callback(self, msg):

        if not msg.poses:

            self.reference_path = None
            return

        new_goal = np.array([
            msg.poses[-1].pose.position.x,
            msg.poses[-1].pose.position.y
        ])

        # Only reset the goal latch if a genuinely new destination
        # has been received.
        if self.last_goal is None:

            self.goal_reached = False

        elif np.linalg.norm(new_goal - self.last_goal) > (
            self.goal_reset_distance
        ):

            self.goal_reached = False

            self.get_logger().info(
                "New destination received. Goal latch reset."
            )

        self.last_goal = new_goal
        self.reference_path = msg

    def odom_callback(self, msg):

        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        yaw = quaternion_to_yaw(q)

        velocity = msg.twist.twist.linear.x

        self.current_state = np.array([
            p.x,
            p.y,
            yaw,
            max(0.0, velocity)
        ], dtype=float)

        self.odom_frame = msg.header.frame_id

        self.last_odom_time = time.monotonic()

    def lane_callback(self, msg):

        # NaN / inf means lane detection failed -> ignore the offset
        # and let the planner run on the global path + obstacle layers.
        if np.isfinite(msg.data):

            self.lane_offset = float(np.clip(
                msg.data,
                -self.max_lane_offset,
                self.max_lane_offset
            ))
            self.lane_time = time.monotonic()

        else:

            self.lane_time = None
            self.lane_offset = 0.0

    # ============================================================
    # PATH TRANSFORMATION
    # ============================================================

    def transform_path_to_odom(self, path):

        """Return the path as an (N, 2) array in the odom frame."""

        if not path.header.frame_id:
            raise RuntimeError("Global path has no frame_id.")

        if not self.odom_frame:
            raise RuntimeError("Odometry has no frame_id.")

        xy = np.array([
            [p.pose.position.x, p.pose.position.y]
            for p in path.poses
        ], dtype=float)

        if path.header.frame_id == self.odom_frame:
            return xy

        transform = self.tf_buffer.lookup_transform(
            self.odom_frame,
            path.header.frame_id,
            Time()
        )

        t = transform.transform

        yaw = quaternion_to_yaw(t.rotation)

        c = np.cos(yaw)
        s = np.sin(yaw)

        rotation = np.array([[c, -s], [s, c]])

        translation = np.array([
            t.translation.x,
            t.translation.y
        ])

        return xy @ rotation.T + translation

    # ============================================================
    # LIDAR TRANSFORMATION
    # ============================================================

    def scan_callback(self, msg):

        self.last_scan_time = time.monotonic()

        try:

            if not msg.header.frame_id:
                raise RuntimeError("LiDAR message has empty frame_id.")

            try:
                transform = self.tf_buffer.lookup_transform(
                    self.base_frame,
                    msg.header.frame_id,
                    Time.from_msg(msg.header.stamp)
                )
            except TransformException:
                # Stamp not available yet -> use the latest transform.
                transform = self.tf_buffer.lookup_transform(
                    self.base_frame,
                    msg.header.frame_id,
                    Time()
                )

            points = pc2.read_points(
                msg,
                field_names=['x', 'y', 'z'],
                skip_nans=True
            )

            # Humble returns a structured array; older versions a generator.
            if isinstance(points, np.ndarray) and points.dtype.names:
                xyz = np.column_stack([
                    points['x'], points['y'], points['z']
                ]).astype(float)
            else:
                xyz = np.array(list(points), dtype=float).reshape(-1, 3)

            xyz = xyz[np.all(np.isfinite(xyz), axis=1)]

            if xyz.shape[0] == 0:

                self.obstacles_base = np.empty((0, 2))
                return

            rotation = quaternion_rotation_matrix(
                transform.transform.rotation
            )

            translation = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ])

            xyz_base = (rotation @ xyz.T).T + translation

            # Retain obstacles within useful vehicle height.
            height_mask = (
                (xyz_base[:, 2] >= 0.22)
                & (xyz_base[:, 2] <= 1.80)
            )

            xyz_base = xyz_base[height_mask]

            # Only forward obstacles ahead of the front bumper in the local planning area.
            forward_mask = (
                (xyz_base[:, 0] > 0.20)
                & (xyz_base[:, 0] < self.params['obstacle_lookahead'])
            )

            xy = xyz_base[forward_mask][:, :2]

            # Keep one point per 10 cm cell so planning stays fast.
            if xy.shape[0] > 0:
                cells = np.unique(
                    np.round(xy / 0.10).astype(np.int32), axis=0
                )
                xy = cells.astype(float) * 0.10

            if xy.shape[0] > 1500:
                xy = xy[::int(np.ceil(xy.shape[0] / 1500))]

            self.obstacles_base = xy

        except Exception as e:

            self.obstacles_base = None

            self.get_logger().warning(
                f"LiDAR transformation failed: {e}",
                throttle_duration_sec=2.0
            )

    def obstacles_in_world(self):

        if self.obstacles_base is None:
            return None

        x, y, theta, _ = self.current_state

        c = np.cos(theta)
        s = np.sin(theta)

        points = self.obstacles_base[:, :2]

        rotation = np.array([
            [c, -s],
            [s, c]
        ])

        return (rotation @ points.T).T + np.array([x, y])

    # ============================================================
    # PATH HELPERS
    # ============================================================

    def nearest_path_index(self, path_xy):

        x, y = self.current_state[:2]

        distances = np.hypot(path_xy[:, 0] - x, path_xy[:, 1] - y)

        return int(np.argmin(distances))

    def make_local_trajectory(self, path_xy, index):

        # Short, evenly spaced path segment ahead of the vehicle.
        return resample_path(
            path_xy[index:],
            self.PATH_SPACING,
            self.params['obstacle_lookahead'] + 1.0
        )

    # ============================================================
    # GOAL CONTROL
    # ============================================================

    def check_goal(self, goal_xy):

        """goal_xy must be in the same frame as current_state (odom)."""

        x, y = self.current_state[:2]

        goal_distance = float(np.hypot(goal_xy[0] - x, goal_xy[1] - y))

        if goal_distance <= self.goal_tolerance:
            self.goal_reached = True

        return self.goal_reached, goal_distance

    # ============================================================
    # EMERGENCY SAFETY
    # ============================================================

    def emergency_obstacle_check(self):

        if self.obstacles_base is None:
            return True

        if len(self.obstacles_base) == 0:
            return False

        distances = np.hypot(
            self.obstacles_base[:, 0],
            self.obstacles_base[:, 1]
        )

        nearest = float(np.min(distances))

        if nearest < self.params['emergency_stop_distance']:

            self.get_logger().error(
                f"Emergency obstacle detected at {nearest:.2f}m",
                throttle_duration_sec=1.0
            )

            return True

        return False

    # ============================================================
    # MPC OUTPUT VALIDATION
    # ============================================================

    def validate_solution(self, solution):

        if solution is None:
            return False

        required = ['x', 'y', 'theta', 'v', 'a', 'delta']

        for key in required:

            if key not in solution:
                return False

            values = np.asarray(solution[key])

            if values.size == 0:
                return False

            if not np.all(np.isfinite(values)):
                return False

        if np.max(solution['v']) > self.params['max_velocity'] + 0.05:
            return False

        if np.min(solution['v']) < -0.05:
            return False

        if np.max(np.abs(solution['delta'])) > (
            self.params['max_steering_angle'] + 0.01
        ):
            return False

        return True

    # ============================================================
    # CONTROL LOOP
    # ============================================================

    def control_loop(self):

        self.control_count += 1

        try:

            # ----------------------------------------------------
            # 1. CHECK INPUT AVAILABILITY
            # ----------------------------------------------------

            if self.current_state is None:
                self.publish_stop("Waiting for odometry")
                return

            if time.monotonic() - self.last_odom_time > 0.5:
                self.publish_stop("Stale odometry")
                return

            if self.last_scan_time is None:
                self.publish_stop("Waiting for LiDAR")
                return

            if time.monotonic() - self.last_scan_time > 0.5:
                self.publish_stop("Stale LiDAR")
                return

            if self.obstacles_base is None:
                self.publish_stop("Invalid LiDAR transformation")
                return

            if self.reference_path is None:
                self.publish_stop("Waiting for global path")
                return

            # ----------------------------------------------------
            # 2. STOP IF GOAL ALREADY REACHED
            # ----------------------------------------------------

            if self.goal_reached:
                self.publish_stop("Final destination reached")
                return

            # ----------------------------------------------------
            # 3. TRANSFORM PATH INTO ODOM FRAME
            # ----------------------------------------------------

            try:

                path_xy = self.transform_path_to_odom(self.reference_path)

            except TransformException as e:

                self.publish_stop(f"Path TF unavailable: {e}")
                return

            if len(path_xy) == 0:
                self.publish_stop("Empty path")
                return

            # ----------------------------------------------------
            # 4. CHECK FINAL DESTINATION (odom frame)
            # ----------------------------------------------------

            goal_reached, goal_distance = self.check_goal(path_xy[-1])

            if goal_reached:
                self.publish_stop("Final destination reached")
                return

            # ----------------------------------------------------
            # 5. FIND NEAREST WAYPOINT, BUILD LOCAL TRAJECTORY
            # ----------------------------------------------------

            nearest_index = self.nearest_path_index(path_xy)

            local_trajectory = self.make_local_trajectory(
                path_xy,
                nearest_index
            )

            if not local_trajectory:
                self.publish_stop("Remaining path too short")
                return

            # ----------------------------------------------------
            # 6. LANE OFFSET (optional - NaN/stale means ignored)
            # ----------------------------------------------------

            if (
                self.lane_time is not None
                and time.monotonic() - self.lane_time < 0.5
            ):

                offset = -self.lane_offset

                adjusted = []

                for ref in local_trajectory:

                    theta = ref['theta']

                    adjusted.append({
                        'x': ref['x'] - offset * np.sin(theta),
                        'y': ref['y'] + offset * np.cos(theta),
                        'theta': theta
                    })

                local_trajectory = adjusted
            # ----------------------------------------------------
            # 7. IMMEDIATE OBSTACLE STOP
            # ----------------------------------------------------

            if self.emergency_obstacle_check():
                self.publish_stop("Emergency obstacle distance")
                return

            # ----------------------------------------------------
            # 8. CONVERT OBSTACLES INTO WORLD FRAME
            # ----------------------------------------------------

            obstacles_world = self.obstacles_in_world()

            # ----------------------------------------------------
            # 9. SELECT SAFE AVOIDANCE TRAJECTORY
            # ----------------------------------------------------

            safe_trajectory, selected_offset, success = (
                self.obstacle_planner.choose_safe_path(
                    local_trajectory,
                    obstacles_world
                )
            )

            if not success:
                self.publish_stop("No safe obstacle avoidance trajectory")
                return

            # ----------------------------------------------------
            # 10. ADAPTIVE SPEED
            # ----------------------------------------------------

            target_speed = self.params['max_velocity']

            safe_speed = self.safety.compute_speed(
                safe_trajectory,
                0,
                obstacles_world,
                target_speed,
                goal_distance
            )

            if safe_speed <= 0.02:
                self.publish_stop("Safe speed limit reached")
                return

            # ----------------------------------------------------
            # 11. COMPUTE MPC CONTROL
            # ----------------------------------------------------

            solution = self.mpc.compute_control(
                self.current_state,
                safe_trajectory,
                safe_speed
            )

            if not self.validate_solution(solution):
                self.publish_stop("MPC solution validation failed")
                return

            # ----------------------------------------------------
            # 12. CONVERT MPC OUTPUT TO TWIST
            # ----------------------------------------------------

            delta = float(solution['delta'][0])

            # v[0] is the CURRENT measured speed. The command must be
            # the next predicted speed, otherwise a stopped vehicle
            # would be told to stay stopped forever.
            v_pred = solution['v']

            velocity = float(v_pred[1] if len(v_pred) > 1 else v_pred[0])

            velocity = float(np.clip(velocity, 0.0, safe_speed))

            # Bicycle-model yaw rate: omega = v * tan(delta) / wheelbase
            yaw_rate = (
                velocity * np.tan(delta) / self.params['wheelbase']
            )

            cmd = Twist()

            cmd.linear.x = velocity

            cmd.angular.z = float(np.clip(yaw_rate, -1.0, 1.0))

            # ----------------------------------------------------
            # 13. FINAL SAFETY CHECK
            # ----------------------------------------------------

            if (
                not np.isfinite(cmd.linear.x)
                or not np.isfinite(cmd.angular.z)
            ):
                self.publish_stop("Invalid final control command")
                return

            # ----------------------------------------------------
            # 14. PUBLISH
            # ----------------------------------------------------

            self.cmd_pub.publish(cmd)

            if self.control_count % 20 == 0:

                self.get_logger().info(
                    f"Speed={cmd.linear.x:.2f} m/s | "
                    f"Steering={np.degrees(delta):.1f} deg | "
                    f"Yaw rate={cmd.angular.z:.2f} rad/s | "
                    f"Goal distance={goal_distance:.2f}m | "
                    f"Avoidance offset={selected_offset:.2f}m"
                )

        except Exception as e:

            self.publish_stop(f"Control loop exception: {e}")

            self.get_logger().error(f"Planner exception: {e}")

    # ============================================================
    # STOP COMMAND
    # ============================================================

    def publish_stop(self, reason):

        self.cmd_pub.publish(Twist())

        if self.control_count % 10 == 1:

            self.get_logger().warning(f"VEHICLE STOPPED: {reason}")

    # ============================================================
    # SHUTDOWN
    # ============================================================

    def destroy_node(self):

        self.get_logger().info("Shutting down local planner.")

        try:
            self.cmd_pub.publish(Twist())
        except Exception:
            pass

        super().destroy_node()


# ================================================================
# MAIN
# ================================================================

def main(args=None):

    rclpy.init(args=args)

    planner = None

    try:

        planner = LocalPlannerNode()

        rclpy.spin(planner)

    except KeyboardInterrupt:

        pass

    except Exception as e:

        print(f"Planner startup failed: {e}")

    finally:

        if planner is not None:

            planner.destroy_node()

        if rclpy.ok():

            rclpy.shutdown()


if __name__ == '__main__':
    main()