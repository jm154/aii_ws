#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import numpy as np
import math
import tf2_ros
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import std_msgs.msg
import scipy.ndimage # 맵 팽창용

class GeometricMosNode(Node):
    def __init__(self):
        super().__init__('geometric_mos_node')

        # --- 파라미터 ---
        self.declare_parameter('dilation_size', 9) 
        self.dilation_size = self.get_parameter('dilation_size').value
        
        # --- 변수 ---
        self.map_data = None
        self.map_info = None
        self.dilated_map = None
        self.angles = None

        # --- TF ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --- QoS 설정 ---
        map_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )
        
        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)

        # --- Sub/Pub ---
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, map_qos)
        self.create_subscription(LaserScan, '/ego_racecar/scan', self.scan_callback, sensor_qos)
        
        self.marker_pub = self.create_publisher(Marker, '/geometric_mos/markers', 10)
        
        self.get_logger().info("Geometric MOS Node Started. Waiting for /map...")

    def map_callback(self, msg):
        self.get_logger().info("Map received! Processing...")
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y
        
        grid = np.array(msg.data, dtype=np.int8).reshape((height, width))
        walls = (grid >= 50)
        
        structure = np.ones((self.dilation_size, self.dilation_size), dtype=bool)
        self.dilated_map = scipy.ndimage.binary_dilation(walls, structure=structure)
        
        self.map_info = {
            'res': resolution, 'w': width, 'h': height, 'ox': origin_x, 'oy': origin_y
        }
        self.get_logger().info("Map processing complete.")

    def scan_callback(self, scan_msg):
        if self.dilated_map is None: return

        try:
            trans = self.tf_buffer.lookup_transform('map', scan_msg.header.frame_id, rclpy.time.Time())
        except Exception: return

        ranges = np.array(scan_msg.ranges)
        if self.angles is None or len(self.angles) != len(ranges):
            self.angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(ranges))
            
        valid = (ranges > scan_msg.range_min) & (ranges < scan_msg.range_max)
        x_local = ranges[valid] * np.cos(self.angles[valid])
        y_local = ranges[valid] * np.sin(self.angles[valid])
        
        tx = trans.transform.translation.x
        ty = trans.transform.translation.y
        q = trans.transform.rotation
        yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))
        
        c, s = math.cos(yaw), math.sin(yaw)
        x_map = (x_local * c - y_local * s) + tx
        y_map = (x_local * s + y_local * c) + ty
        
        res = self.map_info['res']
        ox = self.map_info['ox']
        oy = self.map_info['oy']
        
        u = ((x_map - ox) / res).astype(int)
        v = ((y_map - oy) / res).astype(int)
        
        h, w = self.dilated_map.shape
        in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        
        is_dynamic = np.zeros_like(x_local, dtype=bool)
        valid_indices = np.where(in_bounds)[0]
        u_valid = u[valid_indices]
        v_valid = v[valid_indices]
        
        # 맵 값이 False(빈공간)인 경우 -> Dynamic
        is_empty_space = ~self.dilated_map[v_valid, u_valid]
        is_dynamic[valid_indices] = is_empty_space
        
        self.publish_markers(scan_msg.header, x_local, y_local, is_dynamic)

    def publish_markers(self, header, x, y, is_dynamic):
        marker = Marker()
        marker.header = header 
        marker.ns = "geometric_mos"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        
        # 점 크기 (잘 보이게)
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        
        # 빨간색 (Dynamic Only)
        c_dynamic = std_msgs.msg.ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0) 
        
        # ⭐️ [수정] 동적(True)인 점만 루프 돌며 추가
        # numpy boolean indexing을 사용하여 동적 점만 골라냄
        dynamic_indices = np.where(is_dynamic)[0]
        
        for i in dynamic_indices:
            p = Point()
            p.x = float(x[i])
            p.y = float(y[i])
            p.z = 0.2 # 바닥에서 살짝 띄움
            
            marker.points.append(p)
            marker.colors.append(c_dynamic)
                
        self.marker_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = GeometricMosNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
