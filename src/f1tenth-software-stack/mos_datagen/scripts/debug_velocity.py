#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import math
import numpy as np

class VelocityDebugger(Node):
    def __init__(self):
        super().__init__('velocity_debugger')
        
        # 토픽 이름 확인 필수!
        self.create_subscription(Odometry, '/ego_racecar/odom', self.ego_callback, 10)
        self.create_subscription(Odometry, '/opp_racecar/odom', self.opp_callback, 10)
        
        self.ego_twist = None
        self.ego_pose = None
        self.opp_twist = None
        self.opp_pose = None
        
        self.timer = self.create_timer(0.5, self.print_status) # 0.5초마다 출력

    def ego_callback(self, msg):
        self.ego_twist = msg.twist.twist
        self.ego_pose = msg.pose.pose

    def opp_callback(self, msg):
        self.opp_twist = msg.twist.twist
        self.opp_pose = msg.pose.pose

    def get_yaw(self, q):
        return math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))

    def print_status(self):
        if not self.ego_twist or not self.opp_twist:
            print("Waiting for odom...")
            return

        # 1. Raw Speed (속력, 스칼라)
        ego_speed = self.ego_twist.linear.x
        opp_speed = self.opp_twist.linear.x
        
        # 2. Yaw (Heading)
        ego_yaw = self.get_yaw(self.ego_pose.orientation)
        opp_yaw = self.get_yaw(self.opp_pose.orientation)

        # 3. World Frame Velocity 계산
        # (시뮬레이터 twist는 Body Frame 기준이라고 가정 -> 회전해서 World로 변환)
        v_ego_world_x = ego_speed * math.cos(ego_yaw)
        v_ego_world_y = ego_speed * math.sin(ego_yaw)
        
        v_opp_world_x = opp_speed * math.cos(opp_yaw)
        v_opp_world_y = opp_speed * math.sin(opp_yaw)
        
        # 4. Relative Velocity (World Frame)
        v_rel_world_x = v_opp_world_x - v_ego_world_x
        v_rel_world_y = v_opp_world_y - v_ego_world_y
        
        # 5. Relative Velocity (Ego Frame) -> 우리가 라벨로 쓰는 값!
        # World 벡터를 Ego의 역방향(-ego_yaw)으로 회전
        c, s = math.cos(ego_yaw), math.sin(ego_yaw)
        v_rel_ego_x = v_rel_world_x * c + v_rel_world_y * s
        v_rel_ego_y = -v_rel_world_x * s + v_rel_world_y * c
        
        # 6. Wall Velocity (Ego Frame) -> 벽의 속도 라벨
        # 벽은 가만히 있으므로 내 속도의 반대
        v_wall_x = -ego_speed
        
        # --- 출력 ---
        print("-" * 50)
        print(f"🚗 My Speed:  {ego_speed:.2f} m/s")
        print(f"🚙 Opp Speed: {opp_speed:.2f} m/s")
        print("-" * 20)
        print(f"🧱 Wall Label (Expected): {v_wall_x:.2f} m/s")
        print(f"🎯 Car Label (Calculated): {v_rel_ego_x:.2f} m/s")
        print("-" * 20)
        
        # 7. 검증 로직
        diff = abs(v_wall_x - v_rel_ego_x)
        if diff < 0.5:
            print(f"⚠️ [WARNING] 차와 벽의 속도 차이가 거의 없음 ({diff:.2f})")
            print("   -> 상대 차가 멈춰있거나 매우 느림!")
        else:
            print(f"✅ [OK] 차와 벽이 확실히 구분됨 (차이: {diff:.2f})")
            print("   -> 수학적으로 문제 없음.")

def main():
    rclpy.init()
    node = VelocityDebugger()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()
