import math
import numpy as np
from typing import Tuple, Optional, list

import rclpy
from rclpy.node import Node
from rclpy.timer import Timer

from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Twist,PoseStamped, Point
from std_msgs.msg import Float32, Bool
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

from my_car_interfaces.msg import ObstacleInfo
# DO-MPC IMPORTS
try:
    import do_mpc
    from do_mpc.data import Data
    DOMPC_AVAILABLE = True
except ImportError:
    DOMPC_AVAILABLE = False
    print("WARNING: do-mpc not installed. Install with: pip install do-mpc")
# MPC CONTROLLER CLASS
class MPCController:
    """
    Model Predictive Control for vehicle local planning.
    
    Uses a kinematic bicycle model with MPC to compute optimal
    velocity commands for path tracking with constraints.
    
    State vector: [x, y, theta, v]
    - x, y: position in local frame
    - theta: heading angle
    - v: linear velocity
    
    Control vector: [a, delta]
    - a: acceleration (linear velocity change rate)
    - delta: steering angle
    """
    
    def __init__(self, 
                 horizon: int = 10,
                 dt: float = 0.1,
                 wheelbase: float = 2.5,
                 max_velocity: float = 2.0,
                 max_acceleration: float = 1.0,
                 max_steering_angle: float = math.radians(30.0),
                 max_steering_rate: float = math.radians(60.0)):
        """
        Initialize MPC Controller.
        
        Args:
            horizon: Prediction horizon (number of steps)
            dt: Time step duration (seconds)
            wheelbase: Vehicle wheelbase (meters)
            max_velocity: Maximum forward velocity (m/s)
            max_acceleration: Maximum acceleration magnitude (m/s²)
            max_steering_angle: Maximum steering angle (radians)
            max_steering_rate: Maximum steering rate (rad/s)
        """
        self.horizon = horizon
        self.dt = dt
        self.wheelbase = wheelbase
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.max_steering_angle = max_steering_angle
        self.max_steering_rate = max_steering_rate
        # Tuning weights
        self.Q_x = 10.0      # Cross-track error weight
        self.Q_theta = 5.0   # Heading error weight
        self.Q_v = 1.0       # Velocity tracking weight
        self.R_a = 0.5       # Acceleration effort weight
        self.R_delta = 0.3   # Steering effort weight
        # MPC solver
        self.model = None
        self.mpc = None
        self.simulator = None
        
        if DOMPC_AVAILABLE:
            self._setup_mpc()
        else:
            print("WARNING: MPC controller not initialized (do-mpc unavailable)")
    
    def _setup_mpc(self):
        """Setup MPC model and controller."""
        # STEP-1 CREATE MODEL
        self.model = do_mpc.model.Model(model_type='continuous')
        
        # State variables
        x = self.model.set_variable('_x', 'x', shape=(1,))      # Global X position
        y = self.model.set_variable('_x', 'y', shape=(1,))      # Global Y position
        theta = self.model.set_variable('_x', 'theta', shape=(1,))  # Heading
        v = self.model.set_variable('_x', 'v', shape=(1,))      # Linear velocity
        
        # Control variables
        a = self.model.set_variable('_u', 'a', shape=(1,))      # Acceleration
        delta = self.model.set_variable('_u', 'delta', shape=(1,))  # Steering angle
        
        # Reference trajectory (to track)
        x_ref = self.model.set_variable('_tvp', 'x_ref')
        y_ref = self.model.set_variable('_tvp', 'y_ref')
        v_ref = self.model.set_variable('_tvp', 'v_ref')
        
        # ====================================================================
        # STEP 2: DEFINE KINEMATIC BICYCLE MODEL
        # ====================================================================
        # dx/dt = v * cos(theta)
        # dy/dt = v * sin(theta)
        # dtheta/dt = v * tan(delta) / L
        # dv/dt = a
        
        self.model.set_rhs('x', v * do_mpc.tools.castools.cos(theta))
        self.model.set_rhs('y', v * do_mpc.tools.castools.sin(theta))
        self.model.set_rhs('theta', v * do_mpc.tools.castools.tan(delta) / self.wheelbase)
        self.model.set_rhs('v', a)
        # Build model
        self.model.setup()
        
        # ====================================================================
        # STEP 3: CREATE MPC CONTROLLER
        # ====================================================================
        self.mpc = do_mpc.controller.MPC(self.model)
        
        # Setup MPC
        mpc_settings = {
            'nlp_solver_path_opts': {'ipopt.print_level': 0, 'print_time': 0},
            'max_iter': 200,
            'store_full_solution': True,
        }
        self.mpc.set_param(**mpc_settings)
        
        # Prediction horizon
        self.mpc.bounds['lower', '_u', 'a'] = -self.max_acceleration
        self.mpc.bounds['upper', '_u', 'a'] = self.max_acceleration
        self.mpc.bounds['lower', '_u', 'delta'] = -self.max_steering_angle
        self.mpc.bounds['upper', '_u', 'delta'] = self.max_steering_angle
        
        self.mpc.bounds['lower', '_x', 'v'] = 0.0
        self.mpc.bounds['upper', '_x', 'v'] = self.max_velocity
        
        # Cost function: tracking + effort
        lterm = (
            self.Q_x * (x - x_ref)**2 +
            self.Q_x * (y - y_ref)**2 +
            self.Q_theta * (theta)**2 +
            self.Q_v * (v - v_ref)**2
        )
        
        mterm = (
            self.Q_x * (x - x_ref)**2 +
            self.Q_x * (y - y_ref)**2 +
            self.Q_theta * (theta)**2 +
            self.Q_v * (v - v_ref)**2
        )
        
        self.mpc.set_objective(mterm=mterm, lterm=lterm)
        
        # Control effort penalty
        self.mpc.set_objective(
            mterm=self.R_a * a**2 + self.R_delta * delta**2,
            lterm=self.R_a * a**2 + self.R_delta * delta**2,
            ineq_fun=None,
            ineq_gb=None,
        )
# Setup MPC
        self.mpc.set_rterm(a=self.R_a, delta=self.R_delta)
        
        # Setup solver
        self.mpc.setup()
        
        # ====================================================================
        # STEP 4: CREATE SIMULATOR (FOR PREDICTION VALIDATION)
        # ====================================================================
        self.simulator = do_mpc.simulator.Simulator(self.model)
        self.simulator.set_param(t_sync=self.dt)
        self.simulator.setup()
        
    def compute_control(self,
                       current_state: np.ndarray,
                       reference_trajectory: np.ndarray,
                       reference_velocities: np.ndarray) -> Tuple[float, float]:
        """
        Compute MPC control action.
        
        Args:
            current_state: [x, y, theta, v] - Current vehicle state
            reference_trajectory: (N, 2) - Reference path points [x, y]
            reference_velocities: (N,) - Reference velocities at each point
            
        Returns:
            (acceleration, steering_angle) - Control commands
        """
        
        if not DOMPC_AVAILABLE or self.mpc is None:
            return self._fallback_control(current_state, reference_trajectory)
        
        try:
            # Set initial state
            self.mpc.x0 = current_state
            self.simulator.x0 = current_state
            
            # Prepare time-varying reference trajectory
            tvp_template = self.mpc.get_tvp_template()
            
            for k in range(self.horizon):
                if k < len(reference_trajectory):
                    tvp_template['_tvp', k, 'x_ref'] = reference_trajectory[k, 0]
                    tvp_template['_tvp', k, 'y_ref'] = reference_trajectory[k, 1]
                    tvp_template['_tvp', k, 'v_ref'] = reference_velocities[k]
                else:
                    # Extend with last known reference
                    tvp_template['_tvp', k, 'x_ref'] = reference_trajectory[-1, 0]
                    tvp_template['_tvp', k, 'y_ref'] = reference_trajectory[-1, 1]
                    tvp_template['_tvp', k, 'v_ref'] = reference_velocities[-1]
            
            self.mpc.set_tvp_fun(lambda t: tvp_template)
            # Solve MPC
            try:
                u0 = self.mpc.make_step(current_state)
                acceleration = float(u0[0, 0])
                steering_angle = float(u0[1, 0])
                
                # Clamp values to safety limits
                acceleration = np.clip(acceleration, 
                                      -self.max_acceleration, 
                                      self.max_acceleration)
                steering_angle = np.clip(steering_angle,
                                        -self.max_steering_angle,
                                        self.max_steering_angle)
                
                return acceleration, steering_angle
                
            except Exception as e:
                print(f"MPC solver failed: {e}. Using fallback control.")
                return self._fallback_control(current_state, reference_trajectory)
        
        except Exception as e:
            print(f"MPC computation error: {e}")
            return self._fallback_control(current_state, reference_trajectory)
    
    def _fallback_control(self, 
                         current_state: np.ndarray,
                         reference_trajectory: np.ndarray) -> Tuple[float, float]:
        """
        Fallback pure pursuit controller when MPC is unavailable.
        
        Args:
            current_state: [x, y, theta, v]
            reference_trajectory: (N, 2) reference path
            
        Returns:
            (acceleration, steering_angle)
        """
        
        if len(reference_trajectory) == 0:
            return 0.0, 0.0
        
        # Current pose
        x, y, theta, v = current_state
        
        # Find nearest reference point
        distances = np.sqrt((reference_trajectory[:, 0] - x)**2 + 
                           (reference_trajectory[:, 1] - y)**2)
        nearest_idx = np.argmin(distances)
        
        # Lookahead point
        lookahead_idx = min(nearest_idx + 3, len(reference_trajectory) - 1)
        ref_x, ref_y = reference_trajectory[lookahead_idx]
        
        # Cross-track error
        cte = (ref_y - y) * np.cos(theta) - (ref_x - x) * np.sin(theta)
        
        # Heading to lookahead point
        dx = ref_x - x
        dy = ref_y - y
        desired_theta = np.arctan2(dy, dx)
        heading_error = desired_theta - theta
        
        # Normalize angle
        while heading_error > np.pi:
            heading_error -= 2 * np.pi
        while heading_error < -np.pi:
            heading_error += 2 * np.pi
        
        # Simple steering control
        k_p = 0.5  # Proportional gain
        steering_angle = np.clip(k_p * heading_error, 
                                -self.max_steering_angle,
                                self.max_steering_angle)
        
        # Speed control
        max_speed = 2.0
        acceleration = 0.5 * (max_speed - v)
        
        return acceleration, steering_angle
#LOCAL PLANNER NODE
class LocalPlannerNode(Node):
    """
    ROS2 Node for MPC-based local planning.
    
    Subscribes to:
      - /global_path: Global trajectory from global planner
      - /lane_offset: Lane offset for compensation
      - /odom: Vehicle odometry
      - /scan: LiDAR point cloud for obstacle detection
      - /yolo_obstacles: Optional YOLO detections
      
    Publishes to:
      - /cmd_vel: Velocity commands (Twist)
      - /mpc_status: MPC computation status
    """
    def __init__(self):
        super().__init__('local_planner')

        # ==========================================
        # GLOBAL PATH
        # ==========================================
        self.path_sub = self.create_subscription(
            Path,
            '/global_path',
            self.path_callback,
            10
        )

        # ==========================================
        # LANE OFFSET
        # Unit: meters
        # +ve -> right
        # -ve -> left
        #  0  -> centered
        # ==========================================
        self.lane_sub = self.create_subscription(
            Float32,
            '/lane_offset',
            self.lane_callback,
            10
        )

        # ==========================================
        # VEHICLE ODOMETRY
        # ==========================================
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        # ==========================================
        # 3D LiDAR
        #
        # This is the PRIMARY obstacle-safety input
        # for the system when YOLO is unavailable.
        # ==========================================
        self.lidar_sub = self.create_subscription(
            PointCloud2,
            '/scan',
            self.lidar_callback,
            10
        )

        # ==========================================
        # YOLO OBSTACLES
        #
        # OPTIONAL:
        # The vehicle does NOT depend on YOLO.
        # This subscription only provides semantic
        # information when the YOLO node is available.
        # ==========================================
        self.yolo_sub = self.create_subscription(
            ObstacleInfo,
            '/yolo_obstacles',
            self.yolo_callback,
            10
        )

        # ==========================================
        # VELOCITY COMMAND
        # ==========================================
        self.cmd_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        # ==========================================
        # CURRENT STATE
        # ==========================================
        self.current_path = None

        self.lane_offset = 0.0

        self.current_pose = None
        self.current_linear_velocity = 0.0
        self.current_angular_velocity = 0.0

        # ==========================================
        # YOLO STATE
        #
        # None means YOLO is not currently providing
        # semantic obstacle information.
        # ==========================================
        self.yolo_available = False
        self.detected_object = None
        self.distance_to_object = None
        self.object_height = None
        self.is_object_passable = True

        # ==========================================
        # LiDAR OBSTACLE STATE
        # ==========================================
        self.lidar_obstacle_detected = False
        self.lidar_nearest_distance = float('inf')

        # Front safety limits
        self.lidar_min_range = 0.30
        self.lidar_max_range = 12.0

        # Only consider points in the forward region.
        # x > 0 means in front of base_link.
        self.front_angle_limit = math.radians(45.0)

        # Simple safety distance for the prototype.
        self.safety_distance = 1.0
        self.emergency_stop_distance = 0.5

        self.get_logger().info(
            'Local Planner started. '
            '3D LiDAR is available as the primary '
            'obstacle-safety source. YOLO is optional.'
        )
        # VEHICLE PARAMETERS
        # ====================================================================
        self.wheelbase = 2.5
        self.track_width = 1.8
        
        # Max control limits
        self.max_velocity = 2.0
        self.max_acceleration = 1.0
        self.max_steering_angle = math.radians(30.0)
        
        # ====================================================================
        # FAIL-SAFE STATE
        # ====================================================================
        self.emergency_stop_active = False
        self.obstacle_avoidance_active = False
        self.path_tracking_active = False
        
        # ====================================================================
        # MPC CONTROLLER
        # ====================================================================
        self.mpc = MPCController(
            horizon=10,
            dt=0.1,
            wheelbase=self.wheelbase,
            max_velocity=self.max_velocity,
            max_acceleration=self.max_acceleration,
            max_steering_angle=self.max_steering_angle
        )
        
        # ====================================================================
        # CONTROL TIMER (100 Hz)
        # ====================================================================
        self.control_timer = self.create_timer(
            0.01,  # 10ms = 100 Hz
            self.control_loop
        )
        
        # Previous steering angle for rate limiting
        self.prev_steering_angle = 0.0
        self.max_steering_rate = math.radians(60.0)  # rad/s
        
        self.get_logger().info(
            'MPC Local Planner started.\n'
            '  - MPC Horizon: 10 steps\n'
            '  - Control Frequency: 100 Hz\n'
            '  - Vehicle Wheelbase: 2.5 m\n'
            '  - Lane Offset Convention: +X = right, -X = left\n'
            '  - Safety Distance: 1.0 m\n'
            '  - Emergency Stop Distance: 0.5 m'
        )

    # ==========================================
    # LANE CALLBACK
    # ==========================================
    def lane_callback(self, msg:Float32):
        """
        Update lane offset.
        
        Convention:
          +ve: offset to the right
          -ve: offset to the left
          0: centered in lane
        """
        self.lane_offset = msg.data

    # ==========================================
    # ODOMETRY CALLBACK
    # ==========================================
    def odom_callback(self, msg: Odometry):
        """Update vehicle odometry (pose and velocity)."""
        self.current_pose = msg.pose.pose

        self.current_linear_velocity = (
            msg.twist.twist.linear.x
        )

        self.current_angular_velocity = (
            msg.twist.twist.angular.z
        )

    # ==========================================
    # 3D LiDAR CALLBACK
    # ==========================================
    def lidar_callback(self, msg):
        """
        Read the PointCloud2 data and find the nearest
        valid obstacle point in the forward 45-degree
        sector.

        This is intentionally a simple safety layer.
        It is NOT the final obstacle avoidance algorithm.
        """

        nearest_distance = float('inf')

        try:
            points = point_cloud2.read_points(
                msg,
                field_names=('x', 'y', 'z'),
                skip_nans=True
            )

            for point in points:
                x, y, z = point

                # Ignore points too close or too far.
                distance = math.sqrt(
                    x * x + y * y + z * z
                )

                if distance < self.lidar_min_range:
                    continue

                if distance > self.lidar_max_range:
                    continue

                # Ignore points behind the vehicle.
                if x <= 0.0:
                    continue

                # Horizontal angle relative to vehicle forward axis.
                angle = math.atan2(y, x)

                if abs(angle) > self.front_angle_limit:
                    continue

                # Ignore points too far below the vehicle.
                # This prevents ground points from being treated
                # as obstacles in the simple prototype safety layer.
                if z < -0.5:
                    continue

                nearest_distance = min(
                    nearest_distance,
                    distance
                )

            self.lidar_nearest_distance = nearest_distance
            # Obstacle detection thresholds
            if nearest_distance < self.emergency_stop_distance:
                self.emergency_stop_active = True
            elif nearest_distance < self.safety_distance:
                self.obstacle_avoidance_active = True
            else:
                self.obstacle_avoidance_active = False
                if nearest_distance > self.safety_distance + 0.5:
                    self.emergency_stop_active = False
        
        except Exception as e:
            self.get_logger().warn(f'LiDAR processing error: {e}')

    # ==========================================
    # YOLO CALLBACK
    # ==========================================
    def yolo_callback(self, msg):
        self.yolo_available = True

        self.detected_object = msg.object_label
        self.distance_to_object = msg.distance
        self.object_height = msg.height
        self.is_object_passable = msg.is_passable

        # ==========================================
        # YOLO DATA RECEIVED
        #
        # YOLO is an optional perception source.
        # The local planner can use this information
        # when YOLO is available.
        # ==========================================

    # ==========================================
    # GLOBAL PATH CALLBACK
    # ==========================================
    def path_callback(self, msg: Path):
        """Store global path for tracking."""
        self.current_path = msg

# CONTROL LOOP (100 Hz)
    # ========================================================================
    def control_loop(self):
        """
        Main control loop executed at 100 Hz.
        
        Sequence:
          1. Check fail-safe conditions
          2. Extract current state
          3. Compute local reference trajectory
          4. Apply MPC control
          5. Implement safety constraints
          6. Publish velocity commands
        """
        
        # ====================================================================
        # STEP 1: FAIL-SAFE CHECK
        # ====================================================================
        if not self._check_system_health():
            self._publish_command(0.0, 0.0)
            return
        
        # ====================================================================
        # STEP 2: EXTRACT CURRENT STATE
        # ====================================================================
        if self.current_pose is None or self.current_path is None:
            self._publish_command(0.0, 0.0)
            return
        
        # Current state vector: [x, y, theta, v]
        current_state = np.array([
            self.current_pose.position.x,
            self.current_pose.position.y,
            self._get_heading_from_quaternion(self.current_pose.orientation),
            self.current_linear_velocity
        ])
        # ====================================================================
        # STEP 3: COMPUTE LOCAL REFERENCE TRAJECTORY
        # ====================================================================
        ref_trajectory, ref_velocities = self._extract_local_trajectory(
            current_state,
            self.current_path
        )
        
        if len(ref_trajectory) == 0:
            self._publish_command(0.0, 0.0)
            return
        
        # ====================================================================
        # STEP 4: MPC CONTROL COMPUTATION
        # ====================================================================
        try:
            acceleration, steering_angle = self.mpc.compute_control(
                current_state,
                ref_trajectory,
                ref_velocities
            )
        except Exception as e:
            self.get_logger().error(f'MPC computation failed: {e}')
            self._publish_command(0.0, 0.0)
            return
        
        # ====================================================================
        # STEP 5: SAFETY CONSTRAINTS & FAIL-SAFE LOGIC
        # ====================================================================
        acceleration, steering_angle = self._apply_safety_constraints(
            acceleration,
            steering_angle,
            current_state
        )
        
        # ====================================================================
        # STEP 6: VELOCITY COMMAND GENERATION
        # ====================================================================
        linear_velocity = current_state[3] + acceleration * 0.01  # Integrate
        angular_velocity = self._compute_angular_velocity(
            linear_velocity,
            steering_angle
        )
        
        # ====================================================================
        # STEP 7: PUBLISH COMMAND
        # ====================================================================
        self._publish_command(linear_velocity, angular_velocity)
    
    # ========================================================================
    # HELPER: SYSTEM HEALTH CHECK
    # ========================================================================
    def _check_system_health(self) -> bool:
        """
        Check overall system health for safe operation.
        
        Returns:
            True if system is healthy, False if emergency stop needed
        """
        
        # Check for critical LiDAR detection
        if self.emergency_stop_active:
            self.get_logger().warn('EMERGENCY STOP: Critical obstacle detected')
            return False
        
        # Check for valid odometry
        if self.current_pose is None:
            self.get_logger().warn('Health check failed: No odometry')
            return False
        
        # Check for valid path
        if self.current_path is None or len(self.current_path.poses) == 0:
            self.get_logger().warn('Health check failed: No global path')
            return False
        
        return True
# HELPER: EXTRACT LOCAL TRAJECTORY
    # ========================================================================
    def _extract_local_trajectory(self,
                                  current_state: np.ndarray,
                                  global_path: Path,
                                  lookahead_distance: float = 5.0,
                                  num_points: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract local reference trajectory from global path.
        
        Also applies lane offset compensation:
          - Positive offset: shift trajectory to the right
          - Negative offset: shift trajectory to the left
        
        Args:
            current_state: [x, y, theta, v]
            global_path: Global path message
            lookahead_distance: How far ahead to look (meters)
            num_points: Number of trajectory points to extract
            
        Returns:
            (trajectory_points, reference_velocities)
              trajectory_points: (N, 2) array of [x, y] positions
              reference_velocities: (N,) array of velocities
        """
        
        x, y, theta, v = current_state
        
        # Find nearest path point
        path_points = np.array([
            [pose.position.x, pose.position.y]
            for pose in global_path.poses
        ])
        
        distances = np.sqrt((path_points[:, 0] - x)**2 + 
                           (path_points[:, 1] - y)**2)
        nearest_idx = np.argmin(distances)
        
        # Extract lookahead points
        lookahead_idx = nearest_idx
        trajectory = []
        velocities = []
        accumulated_distance = 0.0
        
        for i in range(nearest_idx, len(global_path.poses)):
            if accumulated_distance > lookahead_distance:
                break
            
            pose = global_path.poses[i]
            px, py = pose.position.x, pose.position.y
            
            # Lane offset compensation
            # Perpendicular direction to heading
            offset_x = self.lane_offset * (-np.sin(theta))
            offset_y = self.lane_offset * (np.cos(theta))
            
            px_adjusted = px + offset_x
            py_adjusted = py + offset_y
            
            trajectory.append([px_adjusted, py_adjusted])
            
            # Reference velocity (constant for now, can be modified)
            velocities.append(self.max_velocity * 0.8)
            
            if i > nearest_idx:
                accumulated_distance += distances[i] - distances[i-1]
        
        if len(trajectory) < 2:
            # Add at least current pose and one lookahead point
            trajectory.append([x, y])
            trajectory.append([x + self.max_velocity * self.mpc.dt * np.cos(theta),
                             y + self.max_velocity * self.mpc.dt * np.sin(theta)])
            velocities = [v, self.max_velocity * 0.8]
        
        return np.array(trajectory), np.array(velocities)
    
    # ========================================================================
    # HELPER: SAFETY CONSTRAINTS
    # ========================================================================
    def _apply_safety_constraints(self,
                                  acceleration: float,
                                  steering_angle: float,
                                  current_state: np.ndarray) -> Tuple[float, float]:
        """
        Apply comprehensive safety constraints:
          1. Velocity limits
          2. Steering rate limiting
          3. Obstacle avoidance
          4. Emergency braking
          5. Heading stability
          
        Args:
            acceleration: Desired acceleration (m/s²)
            steering_angle: Desired steering angle (radians)
            current_state: [x, y, theta, v]
            
        Returns:
            (safe_acceleration, safe_steering_angle)
        """
        
        x, y, theta, v = current_state
        
        # ====================================================================
        # CONSTRAINT 1: Velocity limits
        # ====================================================================
        if v >= self.max_velocity:
            acceleration = min(acceleration, 0.0)  # Cannot accelerate beyond max
        
        if v <= 0.0 and acceleration < 0.0:
            acceleration = 0.0  # Cannot reverse        
# CONSTRAINT 2: Steering rate limiting
        # ====================================================================
        max_steering_change = self.max_steering_rate * 0.01  # 10ms timestep
        steering_angle = np.clip(
            steering_angle,
            self.prev_steering_angle - max_steering_change,
            self.prev_steering_angle + max_steering_change
        )
        self.prev_steering_angle = steering_angle
        
        # ====================================================================
        # CONSTRAINT 3: Obstacle avoidance
        # ====================================================================
        if self.obstacle_avoidance_active:
            # Reduce speed when obstacles detected
            if self.lidar_nearest_distance < self.safety_distance:
                deceleration = -0.5 * (self.safety_distance - self.lidar_nearest_distance)
                acceleration = min(acceleration, deceleration)
                self.get_logger().warn(
                    f'Obstacle avoidance active. Distance: {self.lidar_nearest_distance:.2f}m'
                )
        
        # ====================================================================
        # CONSTRAINT 4: Emergency braking
        # ====================================================================
        if self.emergency_stop_active:
            acceleration = -self.max_acceleration  # Maximum deceleration
            steering_angle = self.prev_steering_angle  # Maintain heading
            self.get_logger().error('EMERGENCY STOP ACTIVATED')
        
        # ====================================================================
        # CONSTRAINT 5: Acceleration limits
        # ====================================================================
        acceleration = np.clip(acceleration, 
                              -self.max_acceleration,
                              self.max_acceleration)
        
        # ====================================================================
        # CONSTRAINT 6: Steering angle limits
        # ====================================================================
        steering_angle = np.clip(steering_angle,
                                -self.max_steering_angle,
                                self.max_steering_angle)
        
        return acceleration, steering_angle
    
    # ========================================================================
    # HELPER: COMPUTE ANGULAR VELOCITY FROM STEERING
    # ========================================================================
    def _compute_angular_velocity(self,
                                  linear_velocity: float,
                                  steering_angle: float) -> float:
        """
        Compute angular velocity from linear velocity and steering angle.
        
        Using bicycle model:
        w = v * tan(delta) / L
        
        Args:
            linear_velocity: Forward velocity (m/s)
            steering_angle: Steering angle (radians)
            
        Returns:
            Angular velocity (rad/s)
        """
        
        if abs(steering_angle) < 0.001:  # Avoid division issues
            return 0.0
        
        # Bicycle model kinematics
        angular_velocity = (
            linear_velocity * np.tan(steering_angle) / self.wheelbase
        )
        
        # Limit angular velocity for stability
        max_angular_velocity = self.max_velocity / (self.wheelbase * 0.5)
        angular_velocity = np.clip(angular_velocity,
                                  -max_angular_velocity,
                                  max_angular_velocity)
        
        return float(angular_velocity)
    
    # ========================================================================
    # HELPER: EXTRACT HEADING FROM QUATERNION
    # ========================================================================
    def _get_heading_from_quaternion(self, orientation) -> float:
        """
        Convert quaternion to yaw angle (heading).
        
        Args:
            orientation: geometry_msgs/Quaternion
            
        Returns:
            Yaw angle in radians
        """
        
        x = orientation.x
        y = orientation.y
        z = orientation.z
        w = orientation.w
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x*x + y*y)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            sinp = np.copysign(1, sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y*y + z*z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return float(yaw)
    
    # ========================================================================
    # HELPER: PUBLISH VELOCITY COMMAND
    # ========================================================================
    def _publish_command(self, linear_velocity: float, angular_velocity: float):
        """
        Publish velocity command to vehicle.
        
        Args:
            linear_velocity: Forward velocity (m/s)
            angular_velocity: Angular velocity (rad/s)
        """
        
        cmd = Twist()
        cmd.linear.x = float(np.clip(linear_velocity, -self.max_velocity, self.max_velocity))
        cmd.angular.z = float(angular_velocity)
        
        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)

    node = LocalPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Local Planner shutting down...')
    finally:
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()





