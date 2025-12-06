#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import os
import math
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

class SimpleDataLogger(Node):
    def __init__(self):
        super().__init__('simple_data_logger')
        
        # 파라미터 설정
        self.declare_parameter('ego_vehicle_name', 'ego_racecar')
        self.declare_parameter('dynamic_vehicle_name', 'opp_racecar')
        self.declare_parameter('save_path', '~/f1tenth_dataset_velocity') 
        
        self.ego_name = self.get_parameter('ego_vehicle_name').value
        self.dynamic_name = self.get_parameter('dynamic_vehicle_name').value
        self.save_path = os.path.expanduser(self.get_parameter('save_path').value)
        
        os.makedirs(self.save_path, exist_ok=True)
        
        # 구독 (토픽 이름 확인!)
        self.create_subscription(Odometry, f'/{self.ego_name}/odom', self.ego_cb, 10)
        self.create_subscription(Odometry, f'/{self.dynamic_name}/odom', self.opp_cb, 10)
        self.create_subscription(LaserScan, f'/{self.ego_name}/scan', self.scan_cb, 10)
        
        self.ego_odom = None
        self.opp_odom = None
        self.data_buffer = []
        self.save_count = 0
        
        self.get_logger().info("✅ Simple Logger Ready. Waiting for data...")

    def ego_cb(self, msg): self.ego_odom = msg
    def opp_cb(self, msg): self.opp_odom = msg

    def get_yaw(self, q):
        return math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))

    def scan_cb(self, scan_msg):
        if self.ego_odom is None or self.opp_odom is None:
            return

        # 1. Ego 정보 (속도, 위치, 헤딩)
        ego_v_x = self.ego_odom.twist.twist.linear.x
        ego_v_y = self.ego_odom.twist.twist.linear.y
        ego_w_z = self.ego_odom.twist.twist.angular.z
        
        ego_pos_x = self.ego_odom.pose.pose.position.x
        ego_pos_y = self.ego_odom.pose.pose.position.y
        ego_yaw = self.get_yaw(self.ego_odom.pose.pose.orientation)

        # 2. Opponent 정보
        opp_v_x = self.opp_odom.twist.twist.linear.x
        opp_v_y = self.opp_odom.twist.twist.linear.y
        opp_w_z = self.opp_odom.twist.twist.angular.z
        
        opp_pos_x = self.opp_odom.pose.pose.position.x
        opp_pos_y = self.opp_odom.pose.pose.position.y
        opp_yaw = self.get_yaw(self.opp_odom.pose.pose.orientation)

        # 3. [핵심] 속도 벡터 World Frame 변환
        # Ego (Body -> World)
        v_ego_world_x = ego_v_x * math.cos(ego_yaw) - ego_v_y * math.sin(ego_yaw)
        v_ego_world_y = ego_v_x * math.sin(ego_yaw) + ego_v_y * math.cos(ego_yaw)
        
        # Opp (Body -> World)
        v_opp_world_x = opp_v_x * math.cos(opp_yaw) - opp_v_y * math.sin(opp_yaw)
        v_opp_world_y = opp_v_x * math.sin(opp_yaw) + opp_v_y * math.cos(opp_yaw)

        # 4. 상대 속도 계산 (World Frame)
        v_rel_world_x = v_opp_world_x - v_ego_world_x
        v_rel_world_y = v_opp_world_y - v_ego_world_y

        # 5. [핵심] 상대 속도 Ego Frame으로 회전 (라벨용)
        # World -> Ego (Rotate by -ego_yaw)
        c, s = math.cos(ego_yaw), math.sin(ego_yaw)
        v_rel_ego_x = v_rel_world_x * c + v_rel_world_y * s
        v_rel_ego_y = -v_rel_world_x * s + v_rel_world_y * c

        # 6. 벽 속도 Ego Frame (내 속도의 반대)
        v_wall_ego_x = -(v_ego_world_x * c + v_ego_world_y * s) # == -ego_v_x (if vy is small)
        v_wall_ego_y = -(-v_ego_world_x * s + v_ego_world_y * c) 

        # 7. 포인트 라벨링 (거리 기준)
        ranges = np.array(scan_msg.ranges, dtype=np.float32)
        labels = np.zeros_like(ranges, dtype=np.uint8)
        point_velocities = np.zeros((len(ranges), 2), dtype=np.float32)
        
        # 라이다 포인트 좌표 계산 (Local -> World)
        angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(ranges))
        valid = (ranges > 0.01) & (ranges < 30.0)
        
        x_local = ranges * np.cos(angles)
        y_local = ranges * np.sin(angles)
        
        # Local -> World 변환 (수동 계산)
        x_world = (x_local * c - y_local * s) + ego_pos_x
        y_world = (x_local * s + y_local * c) + ego_pos_y
        
        # 거리 계산
        dists_sq = (x_world - opp_pos_x)**2 + (y_world - opp_pos_y)**2
        
        # Dynamic (Radius < 1.0m) - 넉넉하게
        is_dynamic = (dists_sq < 1.0**2) & valid
        
        labels[is_dynamic] = 1
        point_velocities[is_dynamic, 0] = v_rel_ego_x
        point_velocities[is_dynamic, 1] = v_rel_ego_y
        
        labels[~is_dynamic] = 0
        point_velocities[~is_dynamic, 0] = v_wall_ego_x
        point_velocities[~is_dynamic, 1] = v_wall_ego_y

        # 8. 저장 (모든 필드 포함)
        data = {
            'ranges': ranges,
            'labels': labels,
            'point_velocities': point_velocities,
            
            # Ego Twist (Body Frame)
            'ego_twist': np.array([ego_v_x, ego_v_y, ego_w_z]),
            
            # Ego Pose
            'ego_pose': np.array([ego_pos_x, ego_pos_y, ego_yaw]),
            
            # ⭐️ [추가됨] Dynamic Twist (Body Frame) - Sanity Check용
            'dynamic_twist': np.array([opp_v_x, opp_v_y, opp_w_z]),
            
            'timestamps': scan_msg.header.stamp.sec + scan_msg.header.stamp.nanosec * 1e-9
        }
        self.data_buffer.append(data)
        
        if len(self.data_buffer) >= 100:
            self.save()
            
    def save(self):
        filepath = os.path.join(self.save_path, f"data_{self.save_count:04d}.npz")
        keys = self.data_buffer[0].keys()
        save_dict = {k: np.stack([d[k] for d in self.data_buffer]) for k in keys}
        
        try:
            np.savez_compressed(filepath, **save_dict)
            self.get_logger().info(f"Saved {filepath}")
            self.data_buffer.clear()
            self.save_count += 1
        except Exception as e:
            self.get_logger().error(f"Save failed: {e}")

def main():
    rclpy.init()
    node = SimpleDataLogger()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()
