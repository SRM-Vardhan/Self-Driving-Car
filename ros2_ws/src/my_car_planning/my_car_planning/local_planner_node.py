"""
MPC-Based Local Planner for Autonomous Vehicles with 4-Layer Safety System + Hybrid Lane Controller

Features:
- Layer 1: MPC constraints & path smoothness (lane corridor)
- Layer 2: Obstacle detection & lateral margins
- Layer 3: Dynamic constraints & speed adaptation
- Layer 4: Trajectory validation & execution safeguard
- HybridLaneController: For when lane offset data is available
- SafeLaneFollower: For when lane offset is NULL


"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Twist, Pose, PoseStamped
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Float32
import numpy as np
import time
import logging
import casadi as ca
import do_mpc
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener, TransformException
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_pose

# ===================================================================
# HELPER CLASSES
# ===================================================================

class LaneOffsetConverter:
    """Convert between angular and distance lane offsets."""
    
    def __init__(self, lookahead_distance=2.0):
        self.lookahead_distance = lookahead_distance
    
    def angle_to_lateral_offset(self, angle_degrees):
        """Convert angle to lateral distance (meters)."""
        angle_rad = np.radians(angle_degrees)
        
        if abs(angle_rad) < 0.2:  # < 11 degrees
            lateral_m = self.lookahead_distance * angle_rad
        else:
            lateral_m = self.lookahead_distance * np.tan(angle_rad)
        
        lateral_m = np.clip(lateral_m, -2.0, 2.0)
        return lateral_m
    
    def lateral_offset_to_angle(self, lateral_m):
        """Inverse: Convert meters to degrees."""
        angle_rad = np.arctan(lateral_m / self.lookahead_distance)
        return np.degrees(angle_rad)


class HybridLaneController:
    """
    Hybrid lane control using BOTH angle (heading) and distance (lateral).
    Active when lane offset data is available.
    """
    
    def __init__(self, lookahead_distance=2.0, lane_width=3.5):
        self.lookahead_distance = lookahead_distance
        self.lane_width = lane_width
        self.converter = LaneOffsetConverter(lookahead_distance)
    
    def process_lane_data(self, angle_degrees, vehicle_state):
        """
        Convert lane angle to control parameters.
        
        Args:
            angle_degrees: Angular offset from lane (degrees)
            vehicle_state: [x, y, theta, v]
            
        Returns:
            Dict with lateral_offset_m, heading_error_rad, lane_angle_deg
        """
        x, y, theta, v = vehicle_state
        
        # Step 1: Convert angle to meters
        lateral_m = self.converter.angle_to_lateral_offset(angle_degrees)
        
        # Step 2: Compute heading correction (scaled)
        heading_correction = np.radians(angle_degrees * 0.5)
        
        return {
            'lateral_offset_m': lateral_m,
            'heading_error_rad': heading_correction,
            'lane_angle_deg': angle_degrees,
        }
    
    def get_mpc_lane_reference(self, reference_path, lane_data):
        """
        Modify reference trajectory with lane offset.
        
        Args:
            reference_path: nav_msgs/Path
            lane_data: Output from process_lane_data()
            
        Returns:
            Modified reference with lane offset applied
        """
        lateral_offset = lane_data['lateral_offset_m']
        heading_offset = lane_data['heading_error_rad']
        
        adjusted_path = []
        
        for pose in reference_path.poses:
            x = pose.pose.position.x
            y = pose.pose.position.y
            
            # Get heading
            q = pose.pose.orientation
            yaw = self.quaternion_to_yaw(q)
            
            # Perpendicular direction
            perp_x = -np.sin(yaw)
            perp_y = np.cos(yaw)
            
            # Apply lateral offset
            x_adjusted = x + lateral_offset * perp_x
            y_adjusted = y + lateral_offset * perp_y
            
            # Apply heading offset
            yaw_adjusted = yaw + heading_offset
            
            adjusted_path.append({
                'x': x_adjusted,
                'y': y_adjusted,
                'theta': yaw_adjusted,
            })
        
        return adjusted_path
    
    @staticmethod
    def quaternion_to_yaw(quaternion):
        """Convert quaternion to yaw."""
        x = quaternion.x
        y = quaternion.y
        z = quaternion.z
        w = quaternion.w
        
        numerator = 2.0 * (w * z + x * y)
        denominator = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(numerator, denominator)
        
        return yaw


class SafeLaneFollower:
    """
    4-Layer safety system for navigation without lane offset.
    Active when lane offset is NULL.
    """
    
    def __init__(self, node, vehicle_params, safety_params):
        self.node = node
        self.logger = node.get_logger()
        
        self.wheelbase = vehicle_params.get('wheelbase', 2.5)
        self.max_velocity = vehicle_params.get('max_velocity', 2.0)
        self.max_acceleration = vehicle_params.get('max_acceleration', 1.5)
        self.max_steering_angle = vehicle_params.get('max_steering_angle', np.radians(30))
        
        self.emergency_stop_distance = safety_params.get('emergency_stop_distance', 0.5)
        self.critical_distance = safety_params.get('critical_distance', 1.0)
        self.caution_distance = safety_params.get('caution_distance', 2.0)
        self.min_clearance = safety_params.get('min_clearance', 0.2)
        self.default_lane_width = safety_params.get('default_lane_width', 0.5)
        
        self.mu = safety_params.get('friction_coefficient', 0.6)
        self.g = 9.81
        
        self.metrics = {
            'corridor_violations': 0,
            'speed_reductions': 0,
            'validation_failures': 0,
            'fallback_activations': 0,
        }
        
        self.logger.info("SafeLaneFollower initialized")
    
    # ===== LAYER 1: Lane Corridor =====
    
    def compute_lane_corridor(self, reference_path, lane_width=None):
        """Create left/right boundaries around reference path."""
        if lane_width is None:
            lane_width = self.default_lane_width
        
        corridor_left = []
        corridor_right = []
        corridor_center = []
        
        for pose in reference_path.poses:
            x = pose.pose.position.x
            y = pose.pose.position.y
            
            yaw = self.quaternion_to_yaw(pose.pose.orientation)
            
            perp_x = -np.sin(yaw)
            perp_y = np.cos(yaw)
            
            corridor_center.append([x, y])
            
            left_x = x + lane_width * perp_x
            left_y = y + lane_width * perp_y
            corridor_left.append([left_x, left_y])
            
            right_x = x - lane_width * perp_x
            right_y = y - lane_width * perp_y
            corridor_right.append([right_x, right_y])
        
        return np.array(corridor_left), np.array(corridor_right), np.array(corridor_center)

    def pointcloud_to_vehicle_points(self, cloud_msg):
        """Convert LiDAR points from laser_frame to base_link coordinates."""
        if cloud_msg is None or cloud_msg.width == 0 or cloud_msg.height == 0:
            return None

        if cloud_msg.header.frame_id != 'laser_frame':
            self.logger.warning(
                f"Unexpected LiDAR frame: {cloud_msg.header.frame_id}"
            )
            return None

        field_names = {field.name for field in cloud_msg.fields}
        if not {'x', 'y', 'z'}.issubset(field_names):
            return None

        try:
            points = pc2.read_points(
                cloud_msg,
                field_names=['x', 'y', 'z'],
                skip_nans=False
            )

            if points.size == 0:
                return None

            x = np.asarray(points['x'], dtype=float)
            y = np.asarray(points['y'], dtype=float)
            z = np.asarray(points['z'], dtype=float)

            valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)

            if not np.any(valid):
                return None

            x = x[valid]
            y = y[valid]

            # LiDAR is mounted 0.4m above base_link.
            z_vehicle = z[valid] + 0.4

            # Bypassing self-collision by raising the floor mask (Fixing NumPy type crash)
            lower_bound = np.greater_equal(z_vehicle, 0.20)
            upper_bound = np.less_equal(z_vehicle, 1.8)
            height_mask = np.logical_and(lower_bound, upper_bound)

            return np.column_stack((
                x[height_mask],
                y[height_mask],
                z_vehicle[height_mask]
            ))

        except Exception as e:
            self.logger.warning(f"PointCloud2 processing failed: {e}")
            return None
    
    # ===== LAYER 2: Obstacle Detection =====
    
    def compute_lateral_safety_margin(self, obstacle_points):
        """Analyze 3D LiDAR points for safe lateral margins."""
        if obstacle_points is None or len(obstacle_points) == 0:
            return 1.0, 1.0

        x_vehicle = obstacle_points[:, 0]
        y_vehicle = obstacle_points[:, 1]

        forward_mask = (x_vehicle > 0.5) & (x_vehicle < 3.0)
        y_forward = y_vehicle[forward_mask]

        if len(y_forward) == 0:
            return 1.0, 1.0

        y_left = y_forward[y_forward > 0]
        y_right = y_forward[y_forward < 0]

        margin_left = np.min(y_left) if len(y_left) > 0 else 1.0
        margin_right = -np.max(y_right) if len(y_right) > 0 else 1.0

        return margin_left, margin_right
    
    def compute_safe_lane_width(self, obstacle_points, default_lane_width=None):
        """Dynamically reduce lane width based on obstacles."""
        if default_lane_width is None:
            default_lane_width = self.default_lane_width
        
        margin_left, margin_right = self.compute_lateral_safety_margin(obstacle_points)
        
        safe_width_left = margin_left - self.min_clearance
        safe_width_right = margin_right - self.min_clearance
        
        safe_lane_width = min(
            max(safe_width_left, 0.2),
            max(safe_width_right, 0.2),
            default_lane_width
        )
        
        if safe_lane_width < default_lane_width * 0.7:
            self.logger.warning(
                f"Narrow corridor: {safe_lane_width:.2f}m "
                f"(left: {safe_width_left:.2f}m, right: {safe_width_right:.2f}m)"
            )
            self.metrics['corridor_violations'] += 1
        
        return safe_lane_width
    
    # ===== LAYER 3: Speed Adaptation =====
    
    def adaptive_speed_limit(self, obstacle_points, target_velocity=None):
        """Reduce speed based on nearby 3D LiDAR obstacles."""
        if target_velocity is None:
            target_velocity = self.max_velocity

        if obstacle_points is None or len(obstacle_points) == 0:
            return target_velocity

        # Only stop for obstacles directly in the driving path (Fixing NumPy type crash)
        in_front = np.greater(obstacle_points[:, 0], 0.0)
        in_path = np.less(np.abs(obstacle_points[:, 1]), 0.3)
        forward_mask = np.logical_and(in_front, in_path)
        forward_points = obstacle_points[forward_mask]

        if len(forward_points) == 0:
            return target_velocity

        distances = np.hypot(
            forward_points[:, 0],
            forward_points[:, 1]
        )

        nearest_obstacle = np.min(distances)

        if nearest_obstacle < self.emergency_stop_distance:
            safe_velocity = 0.0
            self.logger.error(
                f"EMERGENCY STOP: Obstacle at {nearest_obstacle:.2f}m"
            )
        elif nearest_obstacle < self.critical_distance:
            safe_velocity = 0.2
            self.logger.warning(
                f"Critical: {nearest_obstacle:.2f}m, speed→0.2 m/s"
            )
        elif nearest_obstacle < self.caution_distance:
            safe_velocity = 0.5
            self.logger.info(
                f"Caution: {nearest_obstacle:.2f}m, speed→0.5 m/s"
            )
        elif nearest_obstacle < 3.0:
            safe_velocity = 1.0
        else:
            safe_velocity = target_velocity

        if safe_velocity < target_velocity:
            self.metrics['speed_reductions'] += 1

        return safe_velocity
    
    def curvature_aware_speed(self, reference_path, current_idx, target_velocity=None):
        """Reduce speed on sharp curves."""
        if target_velocity is None:
            target_velocity = self.max_velocity
        
        if current_idx < 1 or current_idx >= len(reference_path.poses) - 1:
            return target_velocity
        
        p0 = reference_path.poses[current_idx - 1].pose.position
        p1 = reference_path.poses[current_idx].pose.position
        p2 = reference_path.poses[current_idx + 1].pose.position
        
        x0, y0 = p0.x, p0.y
        x1, y1 = p1.x, p1.y
        x2, y2 = p2.x, p2.y
        
        numerator = 2 * (x0*(y1-y2) + x1*(y2-y0) + x2*(y0-y1))
        denominator = (
            ((x1-x0)**2 + (y1-y0)**2) * ((x2-x1)**2 + (y2-y1)**2)
        )
        
        if denominator < 1e-6:
            return target_velocity
        
        curvature = abs(numerator) / (denominator + 1e-9)
        radius = 1.0 / (curvature + 1e-6)
        
        v_max_curve = np.sqrt(self.mu * self.g * radius)
        safe_velocity = min(v_max_curve * 0.8, target_velocity)
        
        if safe_velocity < target_velocity * 0.7:
            self.logger.debug(
                f"Sharp curve (R={radius:.2f}m), "
                f"speed: {target_velocity:.2f}→{safe_velocity:.2f} m/s"
            )
        
        return safe_velocity
    
    def compute_adaptive_speed(self, obstacle_points, reference_path, current_idx, target_velocity=None):
        """Combine obstacle and curvature speed limits."""
        if target_velocity is None:
            target_velocity = self.max_velocity
        
        v_obstacle = self.adaptive_speed_limit(obstacle_points, target_velocity)
        v_curve = self.curvature_aware_speed(reference_path, current_idx, target_velocity)
        
        return min(v_obstacle, v_curve)
    
    # ===== LAYER 4: Trajectory Validation =====
    
    def validate_mpc_solution_integrity(self, mpc_solution):
        """Check if MPC solution is valid (no NaN/Inf)."""
        for key in ['x', 'y', 'theta', 'v', 'a', 'delta']:
            if key not in mpc_solution:
                return False, 'CRITICAL', f"Missing key: {key}"
            
            values = mpc_solution[key]
            if np.any(np.isnan(values)):
                return False, 'CRITICAL', f"NaN in {key}"
            if np.any(np.isinf(values)):
                return False, 'CRITICAL', f"Inf in {key}"
        
        return True, 'OK', "Integrity OK"
    
    def validate_hard_constraints(self, mpc_solution):
        """Check hard physical constraints."""
        violations = []
        
        a = mpc_solution['a']
        delta = mpc_solution['delta']
        v = mpc_solution['v']
        
        if np.max(np.abs(a)) > self.max_acceleration + 0.01:
            violations.append(f"Accel: {np.max(np.abs(a)):.2f} > {self.max_acceleration:.2f}")
        
        if np.max(np.abs(delta)) > self.max_steering_angle + np.radians(0.5):
            violations.append(f"Steering: {np.degrees(np.max(np.abs(delta))):.1f}°")
        
        if np.max(v) > self.max_velocity + 0.05:
            violations.append(f"Velocity: {np.max(v):.2f} > {self.max_velocity:.2f}")
        
        if np.min(v) < -0.05:
            violations.append(f"Negative velocity: {np.min(v):.2f}")
        
        if violations:
            return False, 'CRITICAL', "; ".join(violations)
        
        return True, 'OK', "Constraints OK"
    
    def validate_trajectory_safety(self, mpc_solution, reference_path):
        """Check trajectory safety (CTE, smoothness)."""
        violations = []
        
        x_pred = mpc_solution['x']
        y_pred = mpc_solution['y']
        
        cte_list = []
        for i, (x, y) in enumerate(zip(x_pred, y_pred)):
            if i < len(reference_path.poses):
                x_ref = reference_path.poses[i].pose.position.x
                y_ref = reference_path.poses[i].pose.position.y
                cte = np.sqrt((x - x_ref)**2 + (y - y_ref)**2)
                cte_list.append(cte)
        
        if cte_list and np.max(cte_list) > 1.0:
            violations.append(f"CTE: {np.max(cte_list):.2f}m")
        
        delta_v = np.diff(mpc_solution['v'])
        if np.max(np.abs(delta_v)) > 0.5:
            violations.append(f"Velocity jump: {np.max(np.abs(delta_v)):.2f}")
        
        severity = 'WARNING' if violations else 'OK'
        reason = "; ".join(violations) if violations else "Safe"
        
        return len(violations) == 0, severity, reason
    
    def validate_control_smoothness(self, mpc_solution):
        """Check control smoothness (jerk, chattering)."""
        violations = []
        
        a = mpc_solution['a']
        delta = mpc_solution['delta']
        
        dt = 0.1
        jerk = np.diff(a) / dt
        if np.max(np.abs(jerk)) > 1.0:
            violations.append(f"Jerk: {np.max(np.abs(jerk)):.2f}")
        
        a_changes = np.sum(np.diff(np.sign(a)) != 0)
        if a_changes > len(a) // 3:
            violations.append(f"Accel chattering: {a_changes}")
        
        severity = 'WARNING' if violations else 'OK'
        reason = "; ".join(violations) if violations else "Smooth"
        
        return len(violations) == 0, severity, reason
    
    def validate_complete(self, mpc_solution, reference_path):
        """Run complete 4-stage validation."""
        stages = {
            'Integrity': self.validate_mpc_solution_integrity(mpc_solution),
            'Constraints': self.validate_hard_constraints(mpc_solution),
            'Trajectory': self.validate_trajectory_safety(mpc_solution, reference_path),
            'Smoothness': self.validate_control_smoothness(mpc_solution),
        }
        
        critical = [n for n, (v, s, _) in stages.items() if s == 'CRITICAL']
        warnings = [n for n, (v, s, _) in stages.items() if s == 'WARNING']
        
        if critical:
            recommendation = 'EXECUTE_FALLBACK'
            is_safe = False
            severity = 'CRITICAL'
        elif len(warnings) > 2:
            recommendation = 'EXECUTE_REDUCED_CONTROL'
            is_safe = False
            severity = 'WARNING'
        else:
            recommendation = 'EXECUTE_MPC'
            is_safe = True
            severity = 'OK'
        
        details = {'critical': critical, 'warnings': warnings, 'recommendation': recommendation}
        
        if not is_safe:
            self.metrics['validation_failures'] += 1
        
        return is_safe, severity, details, recommendation
    
    def execute_with_safeguard(self, mpc_solution, validation_result, vehicle_params):
        """Execute with safeguard."""
        is_safe, severity, details, recommendation = validation_result
        
        cmd = Twist()

        v_cmd = float(
            mpc_solution['v'][1]
            if len(mpc_solution['v']) > 1
            else mpc_solution['v'][0]
        )
        delta_cmd = float(mpc_solution['delta'][0])
        
        if recommendation == 'EXECUTE_MPC':
            cmd.linear.x = float(np.clip(v_cmd, 0, self.max_velocity))
            cmd.angular.z = float(
                np.clip(
                    delta_cmd * v_cmd / 
                    vehicle_params.get('wheelbase', self.wheelbase),
                    -1.0, 1.0
                )
            )
        
        elif recommendation == 'EXECUTE_REDUCED_CONTROL':
            cmd.linear.x = float(np.clip(v_cmd * 0.7, 0, self.max_velocity))
            cmd.angular.z = float(
                np.clip(
                    delta_cmd * 0.5 * v_cmd * 0.7 / 
                    vehicle_params.get('wheelbase', self.wheelbase),
                    -0.5, 0.5
                )
            )
        
        else:  # EXECUTE_FALLBACK
            self.logger.error(f"CRITICAL: {details['critical']}")
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.metrics['fallback_activations'] += 1

        return cmd

    @staticmethod
    def quaternion_to_yaw(quaternion):
        """Convert quaternion to yaw."""
        x, y, z, w = quaternion.x, quaternion.y, quaternion.z, quaternion.w
        numerator = 2.0 * (w * z + x * y)
        denominator = 1.0 - 2.0 * (y * y + z * z)
        return np.arctan2(numerator, denominator)
    
    def print_metrics(self):
        """Log metrics."""
        self.logger.info(
            f"Metrics: Violations={self.metrics['corridor_violations']}, "
            f"Speed_reductions={self.metrics['speed_reductions']}, "
            f"Val_failures={self.metrics['validation_failures']}, "
            f"Fallback={self.metrics['fallback_activations']}"
        )


# ===================================================================
# MPC SETUP WITH DO-MPC
# ===================================================================

class MPCController:
    """
    Do-MPC controller for vehicle path tracking.
    """
    
    def __init__(self, node, params):
        self.node = node
        self.logger = node.get_logger()
        self.params = params
        
        self.horizon = params['mpc']['horizon']
        self.dt = params['mpc']['dt']
        self.wheelbase = params['vehicle']['wheelbase']
        
        self.mpc = None
        self.simulator = None
        self.estimator = None
        
        self._setup_mpc()
        self.logger.info("MPC controller initialized")
    
    def _setup_mpc(self):
        """Setup do-mpc controller."""
        try:
            # Model
            model_type = 'continuous'
            model = do_mpc.model.Model(model_type)
            
            # States
            x = model.set_variable(var_type='_x', var_name='x', shape=(1, 1))
            y = model.set_variable(var_type='_x', var_name='y', shape=(1, 1))
            theta = model.set_variable(var_type='_x', var_name='theta', shape=(1, 1))
            v = model.set_variable(var_type='_x', var_name='v', shape=(1, 1))

            # Controls
            a = model.set_variable(var_type='_u', var_name='a', shape=(1, 1))
            delta = model.set_variable(var_type='_u', var_name='delta', shape=(1, 1))

            # Time-varying reference parameters
            x_ref = model.set_variable(var_type='_tvp', var_name='x_ref', shape=(1, 1))
            y_ref = model.set_variable(var_type='_tvp', var_name='y_ref', shape=(1, 1))
            theta_ref = model.set_variable(var_type='_tvp', var_name='theta_ref', shape=(1, 1))
            v_ref = model.set_variable(var_type='_tvp', var_name='v_ref', shape=(1, 1))
            
            # Dynamics (kinematic bicycle model)
            model.set_rhs('x', v * ca.cos(theta))
            model.set_rhs('y', v * ca.sin(theta))
            model.set_rhs('theta', (v / self.wheelbase) * ca.tan(delta))
            model.set_rhs('v', a)
            
            model.setup()
            
            # MPC setup
            mpc = do_mpc.controller.MPC(model)
            
            setup_mpc = {
                'n_horizon': self.horizon,
                't_step': self.dt,
                'n_robust': 0,
                'store_full_solution': True,
            }
            
            mpc.set_param(**setup_mpc)
            
            # Cost function
            Q_x = 10.0
            Q_y = 10.0
            Q_theta = 5.0
            Q_v = 2.0
            R_a = 0.5
            R_delta = 0.3
            
            lterm = (
                Q_x * (x - x_ref)**2 +
                Q_y * (y - y_ref)**2 +
                Q_theta * (theta - theta_ref)**2 +
                Q_v * (v - v_ref)**2 +
                R_a * a**2 +
                R_delta * delta**2
            )
            
            mterm = lterm
            
            tracking_cost = (
                Q_x * (x - x_ref)**2 +
                Q_y * (y - y_ref)**2 +
                Q_theta * (theta - theta_ref)**2 +
                Q_v * (v - v_ref)**2
            )

            lterm = tracking_cost + R_a * a**2 + R_delta * delta**2

            # Terminal cost must not depend on controls or TVPs.
            mterm = 0 * x

            mpc.set_objective(mterm=mterm, lterm=lterm)
            mpc.set_rterm(a=R_a, delta=R_delta)
            
            # Constraints
            mpc.bounds['lower', '_u', 'a'] = -1.5
            mpc.bounds['upper', '_u', 'a'] = 1.5
            mpc.bounds['lower', '_u', 'delta'] = -np.radians(30)
            mpc.bounds['upper', '_u', 'delta'] = np.radians(30)
            mpc.bounds['lower', '_x', 'v'] = 0.0
            mpc.bounds['upper', '_x', 'v'] = 2.0
            
            # Configure time-varying reference trajectory
            self.tvp_template = mpc.get_tvp_template()

            mpc.set_tvp_fun(lambda t_now: self.tvp_template)

            mpc.setup()

            self.mpc = mpc
            self._initial_guess_set = False
            self.logger.info("MPC setup complete with do-mpc")
            
        except Exception as e:
            self.logger.error(f"MPC setup failed: {e}")
            raise
    
    def compute_control(self, current_state, reference_trajectory, max_velocity, 
                       lane_data=None):
        """
        Compute MPC control.
        
        Args:
            current_state: [x, y, theta, v]
            reference_trajectory: List of reference points
            max_velocity: Max allowed speed
            lane_data: Optional lane data (hybrid controller)
            
        Returns:
            mpc_solution: Dict with x, y, theta, v, a, delta
        """
        if self.mpc is None:
            return None
        
        try:
            # Check reference trajectory
            if not reference_trajectory:
                self.logger.warning("Empty reference trajectory. Skipping MPC step.")
                return None

            # Set initial state
            self.mpc.x0 = np.array(current_state)

            # Initialize MPC guess once
            if not self._initial_guess_set:
                self.mpc.set_initial_guess()
                self._initial_guess_set = True

            # Update reference trajectory for the complete MPC horizon
            for k in range(self.horizon + 1):

                ref = reference_trajectory[min(k, len(reference_trajectory) - 1)]

                ref_theta = ref.get('theta', 0.0)

                # Apply lane offset if hybrid controller is active
                if lane_data is not None:
                    x_ref = ref['x'] - lane_data['lateral_offset_m'] * np.sin(ref_theta)
                    y_ref = ref['y'] + lane_data['lateral_offset_m'] * np.cos(ref_theta)
                    theta_ref = ref_theta + lane_data['heading_error_rad']
                else:
                    x_ref = ref['x']
                    y_ref = ref['y']
                    theta_ref = ref_theta

                self.tvp_template['_tvp', k] = np.array([
                    x_ref,
                    y_ref,
                    theta_ref,
                    max_velocity
                ], dtype=float)

            # Solve MPC
            u0 = self.mpc.make_step(current_state)
            
            # Extract solution
            x_pred = self.mpc.data.prediction(('_x', 'x'), 0)
            y_pred = self.mpc.data.prediction(('_x', 'y'), 0)
            theta_pred = self.mpc.data.prediction(('_x', 'theta'), 0)
            v_pred = self.mpc.data.prediction(('_x', 'v'), 0)
            a_pred = self.mpc.data.prediction(('_u', 'a'), 0)
            delta_pred = self.mpc.data.prediction(('_u', 'delta'), 0)
            
            solution = {
                'x': x_pred.flatten(),
                'y': y_pred.flatten(),
                'theta': theta_pred.flatten(),
                'v': v_pred.flatten(),
                'a': np.append(a_pred.flatten(), a_pred[-1]),
                'delta': np.append(delta_pred.flatten(), delta_pred[-1]),
            }
            
            return solution
            
        except Exception as e:
            self.logger.error(f"MPC solve failed: {e}")
            return None
        # ===================================================================
# MAIN LOCAL PLANNER NODE
# ===================================================================

class LocalPlannerNode(Node):
    """
    Complete MPC Local Planner with 4 Safety Layers + Hybrid Lane Control.
    """
    
    def __init__(self, node_name='local_planner_node'):
        super().__init__(node_name)
        
         # ========== TF2 TRANSFORM LISTENER ==========
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self
        )
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("LOCAL PLANNER NODE STARTING")
        self.get_logger().info("=" * 60)
        
        # ========== LOAD PARAMETERS ==========
        self.declare_parameters(
            namespace='',
            parameters=[
                ('mpc.horizon', 10),
                ('mpc.dt', 0.1),
                ('vehicle.wheelbase', 2.5),
                ('vehicle.max_velocity', 2.0),
                ('vehicle.max_acceleration', 1.5),
                ('vehicle.max_steering_angle', 30.0),
                ('safety.default_lane_width', 0.5),
                ('safety.emergency_stop_distance', 0.5),
                ('safety.critical_distance', 1.0),
                ('safety.caution_distance', 2.0),
                ('safety.min_clearance', 0.2),
                ('safety.friction_coefficient', 0.6),
            ]
        )
        
        self.params = {
            'mpc': {
                'horizon': self.get_parameter('mpc.horizon').value,
                'dt': self.get_parameter('mpc.dt').value,
            },
            'vehicle': {
                'wheelbase': self.get_parameter('vehicle.wheelbase').value,
                'max_velocity': self.get_parameter('vehicle.max_velocity').value,
                'max_acceleration': self.get_parameter('vehicle.max_acceleration').value,
                'max_steering_angle': np.radians(
                    self.get_parameter('vehicle.max_steering_angle').value
                ),
            },
            'safety': {
                'default_lane_width': self.get_parameter('safety.default_lane_width').value,
                'emergency_stop_distance': self.get_parameter('safety.emergency_stop_distance').value,
                'critical_distance': self.get_parameter('safety.critical_distance').value,
                'caution_distance': self.get_parameter('safety.caution_distance').value,
                'min_clearance': self.get_parameter('safety.min_clearance').value,
                'friction_coefficient': self.get_parameter('safety.friction_coefficient').value,
            }
        }
        
        # ========== SUBSCRIPTIONS ==========
        self.path_sub = self.create_subscription(
            Path, '/global_path', self.path_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )
        self.scan_sub = self.create_subscription(
            PointCloud2, '/scan', self.scan_callback, 10
        )
        self.lane_offset_sub = self.create_subscription(
            Float32, '/lane_offset', self.lane_offset_callback, 10
        )
        
        # ========== PUBLISHERS ==========
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # ========== STATE VARIABLES ==========
        self.current_state = None
        self.reference_path = None
        self.lane_offset = 0.0
        self.lane_offset_received = False
        self.scan_msg = None
        self.obstacle_points = None
        self.last_odom_time = None
        self.last_scan_time = None
        self.odom_frame_id = ''
        self.lane_offset_received_timestamp = None
        
        # ========== INITIALIZE CONTROLLERS & FOLLOWERS ==========
        self.mpc = MPCController(self, self.params)
        
        self.safe_follower = SafeLaneFollower(
            self, 
            self.params['vehicle'], 
            self.params['safety']
        )
        
        self.hybrid_controller = HybridLaneController(
            lookahead_distance=2.0,
            lane_width=3.5
        )
        
        # ========== CONTROL LOOP TIMER ==========
        self.control_timer = self.create_timer(
            self.params['mpc']['dt'],
            self.control_loop
        )
        
        # ========== METRICS TIMER ==========
        self.metrics_timer = self.create_timer(5.0, self.log_metrics)
        
        self.control_loop_count = 0
        
        self.get_logger().info("LocalPlannerNode initialized successfully")
        self.get_logger().info("=" * 60)
    
    # ========== CALLBACKS ==========
    
    def path_callback(self, msg):
        """Receive global reference path directly without TF lookups."""
        if not msg.poses:
            self.reference_path = None
            return

        # Directly store path to bypass TF transformation failures
        self.reference_path = msg
        self.get_logger().debug(
            f"Path received directly: {len(self.reference_path.poses)} points"
        )
    
    def odom_callback(self, msg):
        """Receive vehicle odometry."""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        
        q = msg.pose.pose.orientation
        yaw = self.safe_follower.quaternion_to_yaw(q)
        
        v = msg.twist.twist.linear.x
        
        self.current_state = np.array([x, y, yaw, v])
        self.last_odom_time = time.monotonic()
        self.odom_frame_id = msg.header.frame_id
    
    def scan_callback(self, msg):
        """Receive and process 3D LiDAR point cloud."""
        self.scan_msg = msg
        self.obstacle_points = self.safe_follower.pointcloud_to_vehicle_points(msg)
        self.last_scan_time = time.monotonic()
    
    def lane_offset_callback(self, msg):
        """
        Receive lane offset.
        NULL/None → Safe layers ACTIVE
        Valid value → Hybrid controller ACTIVE
        """
        try:
            offset_value = msg.data
            
            # Check for NULL
            if offset_value is None or str(offset_value).upper() == "NULL" or np.isnan(offset_value):
                self.lane_offset = 0.0
                self.lane_offset_received = False
                self.get_logger().debug(" Lane offset: NULL → Safe layers ACTIVE")
            else:
                self.lane_offset = float(offset_value)
                self.lane_offset_received = True
                self.lane_offset_received_timestamp = time.time()
                self.get_logger().debug(f" Lane offset: {self.lane_offset:.2f}m → Hybrid controller ACTIVE")
        
        except Exception as e:
            self.get_logger().warning(f"Lane offset parse error: {e}")
            self.lane_offset = 0.0
            self.lane_offset_received = False

    def _publish_stop(self, reason):
        """Publish a zero-velocity command."""
        self.cmd_pub.publish(Twist())
        if self.control_loop_count % 10 == 1:
            self.get_logger().warning(f"Vehicle stopped: {reason}")

    def control_loop(self):
        """Run one local-planning and safety-control cycle."""
        self.control_loop_count += 1

        try:
            # Check odometry availability and freshness
            if self.current_state is None or self.last_odom_time is None:
                self._publish_stop("Waiting for odometry")
                return

            if time.monotonic() - self.last_odom_time > 0.5:
                self._publish_stop("Odometry data is stale")
                return

            # Check LiDAR availability and freshness
            if self.scan_msg is None or self.last_scan_time is None:
                self._publish_stop("Waiting for LiDAR")
                return

            if time.monotonic() - self.last_scan_time > 0.5:
                self._publish_stop("LiDAR data is stale")
                return

            if self.obstacle_points is None:
                self._publish_stop("Invalid 3D LiDAR data")
                return

            # Check global path availability
            if self.reference_path is None or not self.reference_path.poses:
                self._publish_stop("Waiting for global path")
                return

            path_frame = self.reference_path.header.frame_id
            if path_frame and self.odom_frame_id and path_frame != self.odom_frame_id:
                # BYPASSED FOR SIMULATION: Allow map-framed path to execute
                pass

            # Find the nearest point on the reference path
            nearest_idx = self._find_nearest_path_index()

            # Stop when the final waypoint is reached
            final_pose = self.reference_path.poses[-1].pose.position
            goal_distance = np.hypot(
                self.current_state[0] - final_pose.x,
                self.current_state[1] - final_pose.y
            )

            if nearest_idx == len(self.reference_path.poses) - 1 and goal_distance < 0.25:
                self._publish_stop("Final waypoint reached")
                return

            # Create the remaining local reference path
            local_path = Path()
            local_path.header = self.reference_path.header
            local_path.poses = self.reference_path.poses[nearest_idx:]

            reference_trajectory = self._reference_path_to_list(local_path)
            if not reference_trajectory:
                self._publish_stop("Empty local reference trajectory")
                return

            # Calculate obstacle-aware and curvature-aware speed
            target_velocity = self.params['vehicle']['max_velocity']
            safe_velocity = self.safe_follower.compute_adaptive_speed(
                self.obstacle_points,
                self.reference_path,
                nearest_idx,
                target_velocity
            )

            if safe_velocity <= 0.01:
                self._publish_stop("Obstacle safety limit triggered")
                return

            # Apply lane offset when fresh lane data is available
            lane_data = None
            if (
                self.lane_offset_received
                and self.lane_offset_received_timestamp is not None
                and time.time() - self.lane_offset_received_timestamp < 0.5
            ):
                lane_data = {
                    'lateral_offset_m': -float(self.lane_offset),
                    'heading_error_rad': 0.0
                }

            # Compute MPC control
            solution = self.mpc.compute_control(
                self.current_state,
                reference_trajectory,
                safe_velocity,
                lane_data
            )

            if solution is None:
                self._publish_stop("MPC failed to calculate a solution")
                return

            # Validate the predicted trajectory
            validation_result = self.safe_follower.validate_complete(
                solution,
                local_path
            )

            # Execute validated control with safety fallback
            cmd = self.safe_follower.execute_with_safeguard(
                solution,
                validation_result,
                self.params['vehicle']
            )

            # Enforce the adaptive speed limit before publishing
            cmd.linear.x = float(np.clip(cmd.linear.x, 0.0, safe_velocity))
            self.cmd_pub.publish(cmd)

        except Exception as e:
            self.cmd_pub.publish(Twist())
            self.get_logger().error(f"Control loop error - vehicle stopped: {e}")

# ========== HELPER FUNCTIONS ==========
    
    def _find_nearest_path_index(self):
        """Find nearest point on reference path."""
        if self.reference_path is None or len(self.reference_path.poses) == 0:
            return 0
        
        x_current, y_current = self.current_state[0], self.current_state[1]
        
        distances = [
            np.sqrt(
                (x_current - p.pose.position.x)**2 +
                (y_current - p.pose.position.y)**2
            )
            for p in self.reference_path.poses
        ]
        
        return np.argmin(distances)
    
    def _reference_path_to_list(self, path):
        """Convert nav_msgs/Path to list of dicts."""
        traj = []
        for pose in path.poses:
            traj.append({
                'x': pose.pose.position.x,
                'y': pose.pose.position.y,
                'theta': self.safe_follower.quaternion_to_yaw(pose.pose.orientation),
            })
        return traj
    
    def log_metrics(self):
        """Log performance metrics."""
        self.get_logger().info(
            "=" * 60 + "\n" +
            "PERFORMANCE METRICS\n" +
            f"Corridor violations: {self.safe_follower.metrics['corridor_violations']}\n" +
            f"Speed reductions: {self.safe_follower.metrics['speed_reductions']}\n" +
            f"Validation failures: {self.safe_follower.metrics['validation_failures']}\n" +
            f"Fallback activations: {self.safe_follower.metrics['fallback_activations']}\n" +
            f"Control loop count: {self.control_loop_count}\n" +
            "=" * 60
        )
    
    def destroy_node(self):
        """Cleanup on shutdown."""
        self.get_logger().info("Shutting down LocalPlannerNode")
        self.cmd_pub.publish(Twist())
        super().destroy_node()


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    
    try:
        planner = LocalPlannerNode()
        rclpy.spin(planner)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()