import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, NavSatFix, JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import numpy as np

from my_car_localization.eskf_math import FullErrorStateEKF, GeodeticToENU


class EKFNode(Node):
    WHEEL_RADIUS = 0.06  # Exact wheel_radius from my_car URDF/Xacro

    def __init__(self):
        super().__init__('ekf_node')

        self.ekf = FullErrorStateEKF()
        self.ekf.sigma_a = 0.12
        self.ekf.R_wheel = np.diag([0.02, 0.02, 0.05]) ** 2
        self.ekf.P = np.diag([
            0.10, 0.10, 0.10,       # position [m^2]
            0.05, 0.05, 0.05,       # velocity [m^2/s^2]
            0.002, 0.002, 0.002,    # attitude [rad^2]
            0.005, 0.005, 0.005,    # accel bias
            0.001, 0.001, 0.001     # gyro bias
        ])

        self.geo2enu = None
        self.last_imu_time = None
        self.last_joint_time = None
        self.last_wheel_pos = None
        self._init_accels = []
        self._imu_calibrated = False

        self.imu_sub = self.create_subscription(
            Imu, '/imu', self.imu_callback, qos_profile_sensor_data
        )
        self.gps_sub = self.create_subscription(
            NavSatFix, '/gps/fix', self.gps_callback, qos_profile_sensor_data
        )
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, qos_profile_sensor_data
        )

        self.odom_pub = self.create_publisher(Odometry, '/ekf/odom', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.get_logger().info("Error-State EKF Node Initialized (IMU + GPS + Wheel Encoders).")

    def imu_callback(self, msg):
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        accel = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])
        gyro = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])

        if not self._imu_calibrated:
            self._init_accels.append(accel)
            self.last_imu_time = current_time
            if len(self._init_accels) >= 10:
                mean_accel = np.mean(self._init_accels, axis=0)
                self.ekf.align_roll_pitch_from_accel(mean_accel)
                Rm = self.ekf.get_state()["rotmat"]
                if abs(np.linalg.norm(mean_accel) - 9.81) < 0.5:
                    self.ekf.ba = Rm.T @ (Rm @ mean_accel + self.ekf.g)
                self._imu_calibrated = True
                self.get_logger().info("IMU attitude and static accel bias calibrated.")
            return

        dt = current_time - self.last_imu_time
        self.last_imu_time = current_time

        self.ekf.predict(accel, gyro, dt)
        self.publish_state(msg.header.stamp)

    def joint_callback(self, msg):
        if not self._imu_calibrated or not msg.name or not msg.position:
            return
        try:
            idx_l = msg.name.index('left_rear_wheel_joint')
            idx_r = msg.name.index('right_rear_wheel_joint')
            pos_l = float(msg.position[idx_l])
            pos_r = float(msg.position[idx_r])
        except (ValueError, IndexError):
            return

        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_joint_time is None:
            self.last_joint_time = current_time
            self.last_wheel_pos = (pos_l, pos_r)
            return

        dt = current_time - self.last_joint_time
        if dt < 0.005:
            return
        if dt > 0.5:
            self.last_joint_time = current_time
            self.last_wheel_pos = (pos_l, pos_r)
            return

        v_left = ((pos_l - self.last_wheel_pos[0]) / dt) * self.WHEEL_RADIUS
        v_right = ((pos_r - self.last_wheel_pos[1]) / dt) * self.WHEEL_RADIUS
        self.last_joint_time = current_time
        self.last_wheel_pos = (pos_l, pos_r)

        # Reject wheel-slip spikes when one tire spins against an obstacle
        if abs(v_left) > 1.2 or abs(v_right) > 1.2 or abs(v_left - v_right) > 0.45:
            return

        self.ekf.update_wheel_velocity(v_left, v_right, gate=None)

    def gps_callback(self, msg):
        if msg.status.status < 0:
            return

        if self.geo2enu is None:
            self.geo2enu = GeodeticToENU(msg.latitude, msg.longitude, msg.altitude)
            self.get_logger().info("GPS Origin Locked. Fusing /gps/fix position data.")
            return

        enu_pos = self.geo2enu(msg.latitude, msg.longitude, msg.altitude)
        self.ekf.update_gps_position(enu_pos, cov=[0.08, 0.08, 0.16])

    def publish_state(self, stamp):
        state = self.ekf.get_state()
        p = state["position"]
        q = state["quaternion"]
        v_body = state["rotmat"].T @ state["velocity"]

        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = float(p[0])
        t.transform.translation.y = float(p[1])
        t.transform.translation.z = float(p[2])
        t.transform.rotation.x = float(q[1])
        t.transform.rotation.y = float(q[2])
        t.transform.rotation.z = float(q[3])
        t.transform.rotation.w = float(q[0])
        self.tf_broadcaster.sendTransform(t)

        odom_msg = Odometry()
        odom_msg.header = t.header
        odom_msg.child_frame_id = t.child_frame_id
        odom_msg.pose.pose.position.x = float(p[0])
        odom_msg.pose.pose.position.y = float(p[1])
        odom_msg.pose.pose.position.z = float(p[2])
        odom_msg.pose.pose.orientation = t.transform.rotation
        odom_msg.twist.twist.linear.x = float(v_body[0])
        odom_msg.twist.twist.linear.y = float(v_body[1])
        odom_msg.twist.twist.linear.z = float(v_body[2])
        self.odom_pub.publish(odom_msg)


def main(args=None):
    rclpy.init(args=args)
    node = EKFNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()