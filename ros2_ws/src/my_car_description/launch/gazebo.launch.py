import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
import xacro

def generate_launch_description():
    pkg_name = 'my_car_description'
    pkg_share = get_package_share_directory(pkg_name)

    tb3_gazebo_share = get_package_share_directory('turtlebot3_gazebo')
    world_path = os.path.join(tb3_gazebo_share, 'worlds', 'turtlebot3_world.world')

    # 1. Process the URDF
    xacro_file = os.path.join(pkg_share, 'urdf', 'my_car.urdf.xacro')
    robot_description_raw = xacro.process_file(xacro_file).toxml()

    # 2. Launch Gazebo (The Physics Engine)
    gazebo = ExecuteProcess(
    cmd=[
    'gazebo',
    '--verbose',
    '-s', 'libgazebo_ros_init.so',
    '-s', 'libgazebo_ros_factory.so',
    world_path
    ],
        output='screen'
    )

    # 3. Spawn the Robot (The "Birth" of the car)
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description',
                '-entity', 'my_car',
                '-x', '-2.0',  # Safe distance from the center pillars
                '-y', '-0.5',  # Slightly offset to avoid the back wall
                '-z', '0.5'],  # Drop from 50cm to let gravity settle it
        output='screen'
    )

    # 4. Robot State Publisher (The "Brain" broadcaster)
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description_raw,
                     'use_sim_time': True}] # IMPORTANT: Tells ROS to use Gazebo time, not Laptop time
    )

    return LaunchDescription([
        gazebo,
        node_robot_state_publisher,
        spawn_entity
    ])