import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node

import xacro


def generate_launch_description():

    pkg_share = get_package_share_directory('my_car_description')

    # Fortress launch file and a clean test world
    gz_share = get_package_share_directory('ros_gz_sim')

    gz_launch_file = os.path.join(
        gz_share, 'launch', 'gz_sim.launch.py'
    )

    world_path = os.path.join(
        gz_share, 'worlds', 'empty.sdf'
    )

    # Process the robot Xacro
    xacro_file = os.path.join(
        pkg_share, 'urdf', 'my_car.urdf.xacro'
    )

    robot_description = xacro.process_file(xacro_file).toxml()

    # Launch Gazebo Fortress
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gz_launch_file),
        launch_arguments={
            'gz_args': f'-r -v 4 {world_path}'
        }.items()
    )

    # Publish the robot description and TF
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[
            {
                'robot_description': robot_description,
                'use_sim_time': True
            }
        ],
        output='screen'
    )

    # Spawn the car after Gazebo has started
    spawn_car = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-world', 'empty',
            '-topic', 'robot_description',
            '-name', 'my_car',
            '-x', '-2.0',
            '-y', '-0.5',
            '-z', '0.5'
        ],
        output='screen'
    )

    return LaunchDescription([
        robot_state_publisher,
        gazebo,
        TimerAction(
            period=5.0,
            actions=[spawn_car]
        )
    ])


