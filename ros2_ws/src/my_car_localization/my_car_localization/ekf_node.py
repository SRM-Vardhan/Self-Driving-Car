import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, NavSatFix
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import numpy as np

from my_car_localization.eskf_math import FullErrorStateEKF, GeodeticToENU


class EKFNode(Node):
    def __init__(self):
        super().__init__('ekf_node')

        self.ekf = FullErrorStateEKF()
        self.geo2enu = None
        self.last_imu_time = None

        self.imu_sub = self.create_subscription(
            Imu, '/imu', self.imu_callback, qos_profile_sensor_data
        )
        self.gps_sub = self.create_subscription(
            NavSatFix, '/gps/fix', self.gps_callback, qos_profile_sensor_data
        )

        self.odom_pub = self.create_publisher(Odometry, '/ekf/odom', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.get_logger().info("Error-State EKF Node Initialized. Waiting for /imu and /gps/fix...")

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

        if self.last_imu_time is None:
            self.last_imu_time = current_time
            self.ekf.align_roll_pitch_from_accel(accel)
            return

        dt = current_time - self.last_imu_time
        self.last_imu_time = current_time

        self.ekf.predict(accel, gyro, dt)
        self.publish_state(msg.header.stamp)

    def gps_callback(self, msg):
        if msg.status.status < 0:
            return

        if self.geo2enu is None:
            self.geo2enu = GeodeticToENU(msg.latitude, msg.longitude, msg.altitude)
            self.get_logger().info("GPS Origin Locked. Fusing /gps/fix position data.")
            return

        enu_pos = self.geo2enu(msg.latitude, msg.longitude, msg.altitude)
        self.ekf.update_gps_position(enu_pos, fix="single")

    def publish_state(self, stamp):
        state = self.ekf.get_state()
        p = state["position"]
        q = state["quaternion"]
        v = state["velocity"]

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
        odom_msg.twist.twist.linear.x = float(v[0])
        odom_msg.twist.twist.linear.y = float(v[1])
        odom_msg.twist.twist.linear.z = float(v[2])
        self.odom_pub.publish(odom_msg)


def main(args=None):
    rclpy.init(args=args)
    node = EKFNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()