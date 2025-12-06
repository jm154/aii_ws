#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rclpy, time
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from builtin_interfaces.msg import Time as TimeMsg

class SimClockPub(Node):
    def __init__(self):
        super().__init__('sim_clock_pub')
        # 파라미터: 시작 시각, 배속
        self.declare_parameter('start_sec', 0.0)
        self.declare_parameter('sim_speed', 1.0)   # 1.0=실시간, 2.0=2배속 …

        self.start_sec = float(self.get_parameter('start_sec').value)
        self.sim_speed = float(self.get_parameter('sim_speed').value)

        self.pub = self.create_publisher(Clock, '/clock', 10)

        # 100 Hz 타이머 (0.01 s)
        self.dt = 0.01
        self.t_sim = self.start_sec
        self.t0_wall = time.monotonic()

        self.timer = self.create_timer(self.dt, self.on_timer)
        self.get_logger().info(f"Publishing /clock at 100 Hz, start={self.start_sec}s, speed={self.sim_speed}x")

    def on_timer(self):
        # 벽시계 기준 배속 적용: t_sim = start + sim_speed * (now - t0)
        now_wall = time.monotonic()
        self.t_sim = self.start_sec + self.sim_speed * (now_wall - self.t0_wall)

        sec = int(self.t_sim)
        nsec = int((self.t_sim - sec) * 1e9)
        msg = Clock()
        msg.clock = TimeMsg(sec=sec, nanosec=nsec)
        self.pub.publish(msg)

def main():
    rclpy.init()
    node = SimClockPub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

