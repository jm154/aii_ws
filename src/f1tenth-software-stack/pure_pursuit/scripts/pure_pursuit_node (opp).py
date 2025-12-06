#!/usr/bin/env python3

import numpy as np
from scipy.spatial import distance, transform
import os

import rclpy
from rclpy.node import Node
from rclpy.time import Time, Duration
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PointStamped, QuaternionStamped, TransformStamped, PoseStamped
import tf2_ros


class PurePursuit(Node):
    """ 
    파라미터로 토픽 이름을 받는 단순 Pure Pursuit
    """
    def __init__(self):
        super().__init__('pure_pursuit_node')

        # --- ROS 파라미터 선언 ---
        self.declare_parameter('waypoints_path', '')
        self.declare_parameter('odom_topic', '/ego_racecar/odom')
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('viz_topic', '/visualization_marker_array')

        # --- 파라미터 값 읽어오기 ---
        waypoints_file = self.get_parameter('waypoints_path').get_parameter_value().string_value
        odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value
        visualization_topic = self.get_parameter('viz_topic').get_parameter_value().string_value
        
        # --- Pure Pursuit 파라미터 ---
        self.L = 1.0  # 룩어헤드 거리 (m)
        self.steering_gain = 0.5
        
        # --- Waypoint 파일 로드 ---
        # (self.waypoints 등을 먼저 정의해야 함)
        self.waypoints = np.array([[0.0, 0.0]]) # 임시 초기화
        self.numWaypoints = 0
        
        if not os.path.exists(waypoints_file):
            self.get_logger().error(f"!!! Waypoint 파일 없음: {waypoints_file}")
            return # self.waypoints가 없어서 pose_callback이 실패할 것임

        try:
            # 쉼표(,) 구분자로 먼저 시도
            csv_data = np.loadtxt(waypoints_file, delimiter=',', skiprows=1)
            self.get_logger().info("CSV 로드 (쉼표 구분자) 성공.")
        except Exception:
            try:
                # 세미콜론(;) 구분자로 재시도
                self.get_logger().warn("쉼표(,) 로드 실패. 세미콜론(;) 재시도...")
                csv_data = np.loadtxt(waypoints_file, delimiter=';', skiprows=0)
                self.get_logger().info("CSV 로드 (세미콜론 구분자) 성공.")
            except Exception as e:
                self.get_logger().error(f"CSV 파일 로드 최종 실패: {waypoints_file}. 에러: {e}")
                return # self.waypoints가 없어서 pose_callback이 실패할 것임

        # --- CSV 데이터 파싱 (오직 XY 좌표만) ---
        if csv_data.ndim == 1: # CSV에 줄이 하나만 있을 경우
             csv_data = csv_data.reshape(1, -1)

        if csv_data.shape[1] >= 2: # 최소 2개 열이 있는지 확인
            self.waypoints = csv_data[:, :2] # 오직 앞 2개 열(XY)만 사용
        else:
            self.get_logger().error("CSV 파일에 XY 좌표가 없습니다.")
            return

        self.numWaypoints = self.waypoints.shape[0]

        # --- 구독/발행 설정 ---
        # (파일 로드 성공 후에 구독/발행 설정)
        self.sub_pose = self.create_subscription(Odometry, odom_topic, self.pose_callback, 1)
        self.pub_drive = self.create_publisher(AckermannDriveStamped, drive_topic, 1)
        self.drive_msg = AckermannDriveStamped()
        self.pub_vis = self.create_publisher(MarkerArray, visualization_topic, 1)
        self.markerArray = MarkerArray()

        self.visualization_init() # 마커 초기화
        self.get_logger().info(f"PP 노드 초기화 완료. {self.numWaypoints}개 웨이포인트 로드. 구독: {odom_topic}")

    def pose_callback(self, pose_msg):
        self.currX = pose_msg.pose.pose.position.x
        self.currY = pose_msg.pose.pose.position.y
        self.currPos = np.array([self.currX, self.currY]).reshape((1, 2))

        quat = pose_msg.pose.pose.orientation
        quat = [quat.x, quat.y, quat.z, quat.w]
        R = transform.Rotation.from_quat(quat)
        self.rot = R.as_matrix()

        self.distances = distance.cdist(self.currPos, self.waypoints, 'euclidean').reshape((self.numWaypoints))
        self.closest_index = np.argmin(self.distances)
        self.closestPoint = self.waypoints[self.closest_index]

        targetPoint = self.get_closest_point_beyond_lookahead_dist(self.L)

        p_world = np.array([targetPoint[0], targetPoint[1], 0.0])
        p_robot_in_world = np.array([self.currX, self.currY, 0.0])
        p_local = self.rot.T @ (p_world - p_robot_in_world)
        
        y = p_local[1]
        gamma = self.steering_gain * (2 * y / self.L**2)
        gamma = np.clip(gamma, -0.35, 0.35)

        # --- 주행 명령 발행 ---
        self.drive_msg.drive.steering_angle = gamma
        # self.drive_msg.drive.speed = self.ref_speed_data[self.closest_index] # CSV 속도 사용
        self.drive_msg.drive.speed = 2.0  # ★★★ 속도 2.0으로 강제 고정 ★★★
        self.pub_drive.publish(self.drive_msg)

        # --- 시각화 마커 발행 ---
        self.targetMarker.points = [Point(x = targetPoint[0], y = targetPoint[1], z = 0.0)]
        self.closestMarker.points = [Point(x = self.closestPoint[0], y = self.closestPoint[1], z = 0.0)]
        self.markerArray.markers = [self.waypointMarker, self.targetMarker, self.closestMarker]
        self.pub_vis.publish(self.markerArray)
        
    def get_closest_point_beyond_lookahead_dist(self, threshold):
        point_index = self.closest_index
        dist = self.distances[point_index]
        while dist < threshold:
            point_index = (point_index + 1) % self.numWaypoints
            dist = self.distances[point_index]
        return self.waypoints[point_index]

    def visualization_init(self):
        # (시각화 초기화 코드는 오리지널 버전과 동일하게 사용)
        self.waypointMarker = Marker()
        self.waypointMarker.header.frame_id = 'map'
        self.waypointMarker.type = Marker.POINTS
        self.waypointMarker.color.g = 0.75
        self.waypointMarker.color.a = 1.0
        self.waypointMarker.scale.x = 0.05
        self.waypointMarker.scale.y = 0.05
        self.waypointMarker.id = 0
        self.waypointMarker.points = [Point(x = wpt[0], y = wpt[1], z = 0.0) for wpt in self.waypoints]

        self.targetMarker = Marker()
        self.targetMarker.header.frame_id = 'map'
        self.targetMarker.type = Marker.POINTS
        self.targetMarker.color.r = 0.75
        self.targetMarker.color.a = 1.0
        self.targetMarker.scale.x = 0.2
        self.targetMarker.scale.y = 0.2
        self.targetMarker.id = 1

        self.closestMarker = Marker()
        self.closestMarker.header.frame_id = 'map'
        self.closestMarker.type = Marker.POINTS
        self.closestMarker.color.b = 0.75
        self.closestMarker.color.a = 1.0
        self.closestMarker.scale.x = 0.2
        self.closestMarker.scale.y = 0.2
        self.closestMarker.id = 2


def main(args=None):
    rclpy.init(args=args)
    pure_pursuit_node = PurePursuit()
    rclpy.spin(pure_pursuit_node)
    pure_pursuit_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
