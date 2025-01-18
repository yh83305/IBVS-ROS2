from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():

    # velocity_publisher_node = Node(
    #     package='mpc_ibvs',
    #     executable='velocity_publisher',
    #     name='velocity_publisher',
    #     output='screen',
    # )

    realsense_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('realsense2_camera'),
                'launch',
                'rs_launch.py'
            ])
        ])
    )

    mpc_detect_node = Node(
        package='mpc_ibvs',
        executable='mpc_detect',
        name='mpc_detect',
        output='screen',
    )

    mpc_control_node = Node(
        package='mpc_ibvs',
        executable='mpc_control',
        name='mpc_control',
        output='screen',
    )

    return LaunchDescription([
        # velocity_publisher_node,
        realsense_node,
        mpc_detect_node,
        mpc_control_node
    ])