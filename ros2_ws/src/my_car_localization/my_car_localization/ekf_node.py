import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, NavSatFix
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import numpy as np

# Import the electronics team's math engine
from my_car_localization.eskf_math import FullErrorStateEKF, GeodeticToENU

class EKFNode(Node):
    def __init__(self):
        super().__init__('ekf_node')
        
        # Initialize EKF Math Engine
        self.ekf = FullErrorStateEKF()
        self.geo2enu = GeodeticToENU()
        self.enu_initialized = False
        
        self.last_imu_time = None
        
        # Subscribers (Ensure these topic names match your Gazebo outputs)
        self.imu_sub = self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.gps_sub = self.create_subscription(NavSatFix, '/gps/fix', self.gps_callback, 10)
        
        # Publishers & TF Broadcaster
        self.odom_pub = self.create_publisher(Odometry, '/ekf/odom', 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        
        self.get_logger().info("Error-State EKF Node Initialized. Waiting for IMU and GPS...")

    def imu_callback(self, msg):
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        # Extract raw IMU data
        accel = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        gyro = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        
        if self.last_imu_time is None:
            self.last_imu_time = current_time
            # Initialize EKF attitude from the very first IMU gravity reading
            self.ekf.align_roll_pitch_from_accel(accel)
            return
            
        dt = current_time - self.last_imu_time
        self.last_imu_time = current_time
        
        # 1. PREDICT STEP (Runs fast, at IMU frequency)
        self.ekf.predict(accel, gyro, dt)
        
        # 2. PUBLISH FUSED ODOMETRY & TF
        self.publish_state(msg.header.stamp)

    def gps_callback(self, msg):
        # Ignore invalid GPS fixes
        if msg.status.status < 0:
            return
            
        # Initialize local Cartesian map origin on first GPS lock
        if not self.enu_initialized:
            # The __init__ in GeodeticToENU acts as the origin setter in their script
            self.geo2enu.__init__(msg.latitude, msg.longitude, msg.altitude)
            self.enu_initialized = True
            self.get_logger().info("GPS Origin Locked. Fusing position data.")
            return
            
        # Convert Lat/Lon to local X/Y/Z meters (ENU format)
        enu_pos = self.geo2enu.__call__(msg.latitude, msg.longitude, msg.altitude)
        
        # 3. UPDATE STEP (Runs slower, at GPS frequency)
        self.ekf.update_gps_position(enu_pos, fix="single")

    def publish_state(self, stamp):
        state = self.ekf.get_state()
        p = state["position"]
        q = state["quaternion"] # The script uses Hamilton [w, x, y, z]
        v = state["velocity"]
        
        # Publish the spatial TF tree link (odom -> base_link)
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = p[0]
        t.transform.translation.y = p[1]
        t.transform.translation.z = p[2]
        
        # ROS 2 expects [x, y, z, w] for quaternions
        t.transform.rotation.x = q[1]
        t.transform.rotation.y = q[2]
        t.transform.rotation.z = q[3]
        t.transform.rotation.w = q[0]
        self.tf_broadcaster.sendTransform(t)
        
        # Publish the Odometry message for monitoring
        odom_msg = Odometry()
        odom_msg.header = t.header
        odom_msg.child_frame_id = t.child_frame_id
        odom_msg.pose.pose.position = t.transform.translation
        odom_msg.pose.pose.orientation = t.transform.rotation
        odom_msg.twist.twist.linear.x = v[0]
        odom_msg.twist.twist.linear.y = v[1]
        odom_msg.twist.twist.linear.z = v[2]
        self.odom_pub.publish(odom_msg)

def main(args=None):
    rclpy.init(args=args)
    node = EKFNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()