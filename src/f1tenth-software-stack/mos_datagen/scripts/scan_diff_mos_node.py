#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.duration import Duration
import numpy as np
import math
import tf2_ros
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import std_msgs.msg
from sklearn.neighbors import KDTree
import scipy.ndimage

class ScanDiffMosNode(Node):
    def __init__(self):
        super().__init__('scan_diff_mos_node')

        self.declare_parameter('diff_threshold', 0.2)
        self.threshold = self.get_parameter('diff_threshold').value
        self.filter_kernel_size = 9
        
        self.prev_points = None 
        self.prev_robot_pose = None 
        self.angles = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)
        self.create_subscription(LaserScan, '/ego_racecar/scan', self.scan_callback, qos)
        
        self.marker_pub = self.create_publisher(Marker, '/scan_diff_mos/markers', 10)
        
        # [로그] 시작 알림
        self.get_logger().info(f"=== Scan Diff MOS Node Started === Threshold: {self.threshold}m")

    def get_yaw_from_quat(self, q):
        return math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))

    def scan_callback(self, scan_msg):
        ranges = np.array(scan_msg.ranges)
        if self.angles is None or len(self.angles) != len(ranges):
            self.angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(ranges))
            
        valid = (ranges > scan_msg.range_min) & (ranges < scan_msg.range_max)
        
        x_curr = ranges[valid] * np.cos(self.angles[valid])
        y_curr = ranges[valid] * np.sin(self.angles[valid])
        curr_points = np.stack([x_curr, y_curr], axis=1) 

        # 2. TF 조회
        try:
            trans = self.tf_buffer.lookup_transform(
                'map', scan_msg.header.frame_id, rclpy.time.Time()) 
        except Exception as e:
            # [로그] TF 조회 실패 시
            self.get_logger().warn(f"TF Error: {e}", throttle_duration_sec=1.0)
            return

        # 3. 현재 로봇 위치
        tx = trans.transform.translation.x
        ty = trans.transform.translation.y
        yaw = self.get_yaw_from_quat(trans.transform.rotation)
        curr_robot_pose = (tx, ty, yaw)

        if self.prev_robot_pose is None:
            self.prev_points = curr_points
            self.prev_robot_pose = curr_robot_pose
            # [로그] 첫 프레임 저장
            self.get_logger().info("First frame saved. Waiting for next scan...", throttle_duration_sec=5.0)
            return

        # 4. 모션 보정
        px, py, pyaw = self.prev_robot_pose
        cp, sp = math.cos(pyaw), math.sin(pyaw)
        x_global = self.prev_points[:, 0] * cp - self.prev_points[:, 1] * sp + px
        y_global = self.prev_points[:, 0] * sp + self.prev_points[:, 1] * cp + py
        
        cx, cy, cyaw = curr_robot_pose
        cc, sc = math.cos(cyaw), math.sin(cyaw)
        dx = x_global - cx
        dy = y_global - cy
        x_comp = dx * cc + dy * sc
        y_comp = -dx * sc + dy * cc
        prev_points_comp = np.stack([x_comp, y_comp], axis=1)

        # 5. 비교 (Difference)
        if len(prev_points_comp) > 0:
            tree = KDTree(prev_points_comp)
            dists, _ = tree.query(curr_points)
            
            is_dynamic_raw = (dists.flatten() > self.threshold)
            
            is_dynamic_filtered = scipy.ndimage.binary_opening(
                is_dynamic_raw, 
                structure=np.ones(self.filter_kernel_size)
            )
            
            is_dynamic_final = scipy.ndimage.binary_dilation(
                is_dynamic_filtered, 
                structure=np.ones(self.filter_kernel_size)
            )
            
            # [로그] 동적 포인트 발견 시 알림
            num_dynamic = np.sum(is_dynamic_final)
            if num_dynamic > 0:
                self.get_logger().info(f"🔥 Dynamic Points: {num_dynamic}", throttle_duration_sec=0.5)
            
            # 6. 시각화
            viz_header = scan_msg.header
            viz_header.stamp = self.get_clock().now().to_msg()
            self.publish_markers(viz_header, curr_points, is_dynamic_final)
        
        self.prev_points = curr_points
        self.prev_robot_pose = curr_robot_pose

    def publish_markers(self, header, points, is_dynamic):
        marker = Marker()
        marker.header = header
        marker.ns = "scan_diff_mos"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        
        c_dynamic = std_msgs.msg.ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0) 
        
        dynamic_points = points[is_dynamic]
        
        for p_xy in dynamic_points:
            p = Point()
            p.x = float(p_xy[0])
            p.y = float(p_xy[1])
            p.z = 0.0
            
            marker.points.append(p)
            marker.colors.append(c_dynamic)
        
        # [로그] 마커 발행 확인
        # self.get_logger().info(f"Markers Published: {len(dynamic_points)} points", throttle_duration_sec=2.0)
        
        self.marker_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = ScanDiffMosNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
