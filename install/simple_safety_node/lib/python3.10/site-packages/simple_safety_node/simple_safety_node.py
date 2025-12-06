#!/usr/bin/env python3
# 이 스크립트는 Python 3 인터프리터를 사용하여 실행됩니다.

import rclpy
# ROS 2 클라이언트 라이브러리
from rclpy.node import Node
# ROS 2 노드 클래스

import numpy as np
# 수학 및 과학 계산을 위한 라이브러리

from sensor_msgs.msg import LaserScan
# 레이저 스캔 메시지 타입
from nav_msgs.msg import Odometry
# 주행 정보 메시지 타입
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
# Ackermann 주행 메시지 타입

class SafetyNode(Node):
    # SafetyNode 클래스는 ROS 2 노드를 상속받아 정의합니다.

    def __init__(self):
        super().__init__('simple_safety_node')
        # 노드 이름을 'aeb_node'로 설정하고 부모 클래스(Node)를 초기화합니다.
        self.speed = 0.  # 현재 속도를 저장하는 변수 (m/s)
        self.desired_speed = 2.0  # 원하는 기본 속도 설정 (m/s)
        self.thres = 0.8  # Time-To-Collision 임계값 (s)
        self.ackermann_msg = AckermannDriveStamped()
        # Ackermann 주행 메시지 생성

        # ROS 구독자와 퍼블리셔 생성
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        # '/scan' 토픽을 구독하고, LaserScan 메시지를 수신할 때 scan_callback을 호출
        self.sub_odom = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        # '/ego_racecar/odom' 토픽을 구독하고, Odometry 메시지를 수신할 때 odom_callback을 호출
        self.pub_drive = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        # '/drive' 토픽에 퍼블리시할 퍼블리셔 생성

        # 파라미터 설정
        self.angle_min = -10  # 레이저 스캔의 최소 각도 (degrees)
        self.angle_max = 10   # 레이저 스캔의 최대 각도 (degrees)
        self.scan_min_index = 539 + int(self.angle_min * 4)
        # 최소 각도에 해당하는 레이저 스캔 배열의 인덱스 계산
        self.scan_max_index = 539 + int(self.angle_max * 4)
        # 최대 각도에 해당하는 레이저 스캔 배열의 인덱스 계산

    def odom_callback(self, odom_msg):
        # 주행 정보를 수신할 때 호출되는 콜백 함수
        # 현재 속도를 업데이트
        self.speed = odom_msg.twist.twist.linear.x

    def scan_callback(self, scan_msg):
        # 레이저 스캔 데이터를 수신할 때 호출되는 콜백 함수

        # Time-To-Collision (TTC) 계산
        r = np.array(scan_msg.ranges)[self.scan_min_index:self.scan_max_index]
        # 레이저 스캔 거리 데이터를 배열로 변환하고, 지정된 각도 범위 내의 거리 데이터만 선택
        theta = np.linspace(self.angle_min, self.angle_max, self.scan_max_index - self.scan_min_index)
        # 지정된 각도 범위에 대한 각도 배열 생성
        r_dot = self.speed * np.cos(theta)
        # 현재 속도와 각도를 사용하여 각 지점에서의 속도 성분 계산
        ttc = r / np.clip(r_dot, a_min=0.001, a_max=None)
        # TTC 계산 (0.001로 나누기 방지)
        min_ttc = np.min(np.clip(ttc, 0.0, 60.0))
        # TTC 값을 0~60초 사이로 클리핑하고 최소값 선택

        # 비상 제동 명령을 퍼블리시
        if (self.speed > 0 and min_ttc < self.thres) or (self.speed < 0 and min_ttc < (self.thres + 0.8)):
            # 속도가 양수이고 min_ttc가 임계값보다 작거나, 속도가 음수이고 min_ttc가 임계값 + 0.8보다 작으면 비상 제동
            print('min_ttc is {}, brake!!!!'.format(round(min_ttc, 2)))
            # 비상 제동 메시지 출력
            self.desired_speed = 0.0
            # 원하는 속도를 0으로 설정하여 정지

        self.ackermann_msg.drive.speed = self.desired_speed
        # Ackermann 메시지의 속도 설정
        self.pub_drive.publish(self.ackermann_msg)
        # 비상 제동 명령 퍼블리시

def main(args=None):
    rclpy.init(args=args)
    # ROS 2 초기화
    safety_node = SafetyNode()
    # SafetyNode 인스턴스 생성
    rclpy.spin(safety_node)
    # 노드를 스핀하여 콜백이 호출되도록 유지

    safety_node.destroy_node()
    # 노드를 명시적으로 파괴
    rclpy.shutdown()
    # ROS 2 종료

if __name__ == '__main__':
    main()
    # 스크립트가 직접 실행될 때 main() 함수 호출
