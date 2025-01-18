from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():

    ir_mpc_detect_node = Node(
        package='mpc_ibvs',
        executable='ir_mpc_detect',
        name='ir_mpc_detect',
        output='screen',
    )

    ir_mpc_control_node = Node(
        package='mpc_ibvs',
        executable='ir_mpc_control',
        name='ir_mpc_control',
        output='screen',
    )

    return LaunchDescription([
        ir_mpc_detect_node,
        ir_mpc_control_node
    ])