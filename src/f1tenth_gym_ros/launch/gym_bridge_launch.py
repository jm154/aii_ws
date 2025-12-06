# MIT License
# ... (라이선스 헤더는 동일) ...

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import Command
from ament_index_python.packages import get_package_share_directory
import os
import yaml
# from launch_ros.actions import StaticTransformBroadcaster <-- Node를 직접 사용

def generate_launch_description():
    ld = LaunchDescription()
    config = os.path.join(
        get_package_share_directory('f1tenth_gym_ros'),
        'config',
        'sim.yaml'
        )
    config_dict = yaml.safe_load(open(config, 'r'))
    has_opp = config_dict['bridge']['ros__parameters']['num_agent'] > 1
    teleop = config_dict['bridge']['ros__parameters']['kb_teleop']

    bridge_node = Node(
        package='f1tenth_gym_ros',
        executable='gym_bridge',
        name='bridge',
        parameters=[config]
        # use_sim_time 제거
    )
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz',
        arguments=['-d', os.path.join(get_package_share_directory('f1tenth_gym_ros'), 'launch', 'gym_bridge.rviz')]
        # use_sim_time 제거
    )
    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        parameters=[{'yaml_filename': config_dict['bridge']['ros__parameters']['map_path'] + '.yaml'},
                    {'topic': 'map'},
                    {'frame_id': 'map'},
                    {'output': 'screen'}]
                    # use_sim_time 제거
    )
    nav_lifecycle_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        parameters=[{'autostart': True},
                    {'node_names': ['map_server']}]
                    # use_sim_time 제거
    )
    ego_robot_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='ego_robot_state_publisher',
        parameters=[{'robot_description': Command(['xacro ', os.path.join(get_package_share_directory('f1tenth_gym_ros'), 'launch', 'ego_racecar.xacro')])}],
        remappings=[('/robot_description', 'ego_robot_description')]
        # use_sim_time 제거
    )
    opp_robot_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='opp_robot_state_publisher',
        parameters=[{'robot_description': Command(['xacro ', os.path.join(get_package_share_directory('f1tenth_gym_ros'), 'launch', 'opp_racecar.xacro')])}],
        remappings=[('/robot_description', 'opp_robot_description')]
        # use_sim_time 제거
    )

    # --- TF 연결고리 노드 정의 ---
    ego_map_to_odom_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='ego_map_to_odom_broadcaster',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'ego_racecar/odom']
        # use_sim_time 제거
    )

    # map -> opp_racecar/odom 정적 변환 발행
    opp_map_to_odom_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='opp_map_to_odom_broadcaster',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'opp_racecar/odom']
        # use_sim_time 제거
    )
    # --- 여기까지 ---

    # finalize
    ld.add_action(rviz_node)
    ld.add_action(bridge_node)
    ld.add_action(nav_lifecycle_node)
    ld.add_action(map_server_node)
    ld.add_action(ego_robot_publisher)

    ld.add_action(ego_map_to_odom_tf)

    if has_opp:
        ld.add_action(opp_robot_publisher)
        ld.add_action(opp_map_to_odom_tf)

    return ld
