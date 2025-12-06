#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

import numpy as np

from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from builtin_interfaces.msg import Duration
import std_msgs.msg

from sklearn.neighbors import KDTree, NearestNeighbors
import scipy.ndimage


class ScanMatchingMosNode(Node):
    def __init__(self):
        super().__init__('scan_matching_mos_node')

        # ----------------------
        # 파라미터
        # ----------------------
        self.declare_parameter('diff_threshold', 0.3)
        self.declare_parameter('downsample_rate', 1)

        self.threshold = float(self.get_parameter('diff_threshold').value)
        self.downsample_rate = int(self.get_parameter('downsample_rate').value)

        # ----------------------
        # ICP 설정
        # ----------------------
        self.icp_max_iter = 5000
        self.icp_tolerance = 0.0001

        self.prev_points = None
        self.angles = None

        # ----------------------
        # 로그 throttle용 카운터
        # ----------------------
        self.icp_log_counter = 0
        self.icp_log_skip = 2          # ICP 수렴 로그는 10번 중 1번만
        self.dynamic_log_counter = 0
        self.dynamic_log_skip = 2      # 동적 포인트 로그도 10번 중 1번만

        # ----------------------
        # QoS / Subscriber / Publisher
        # ----------------------
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)
        self.create_subscription(LaserScan, '/ego_racecar/scan', self.scan_callback, qos)

        self.marker_pub = self.create_publisher(Marker, '/scan_matching_mos/markers', 10)

        self.get_logger().info(
            f"=== Scan Matching MOS Started === "
            f"Threshold: {self.threshold} m, downsample_rate: {self.downsample_rate}"
        )

    # =========================================================
    #  ICP 2D
    # =========================================================
    def icp_2d(self, source: np.ndarray, target: np.ndarray):
        """
        source: (N, 2), 이전 프레임 포인트들
        target: (M, 2), 현재 프레임 포인트들
        """
        src = source.copy()
        total_R = np.eye(2)
        total_t = np.zeros(2)

        # Nearest Neighbor는 target 기준으로 한 번만 fit
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(target)

        for i in range(self.icp_max_iter):
            distances, indices = nbrs.kneighbors(src)

            # 너무 멀리 떨어진 점들은 매칭에서 제외
            valid_mask = (distances.flatten() < 1.0)
            if np.sum(valid_mask) < 10:
                # 매칭되는 점이 너무 적으면 ICP 실패로 보고 중단
                self.get_logger().warn(
                    f"ICP Lost: Not enough matched points ({np.sum(valid_mask)})"
                )
                break

            src_valid = src[valid_mask]
            tgt_valid = target[indices[valid_mask].flatten()]

            # 중심 제거 후 SVD 기반 최적 R, t 계산
            src_mean = np.mean(src_valid, axis=0)
            tgt_mean = np.mean(tgt_valid, axis=0)

            H = (src_valid - src_mean).T @ (tgt_valid - tgt_mean)
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T

            # 반사 방지
            if np.linalg.det(R) < 0:
                Vt[1, :] *= -1
                R = Vt.T @ U.T

            t = tgt_mean - (R @ src_mean)

            # source 업데이트
            src = (src @ R.T) + t

            # 누적 변환
            total_R = R @ total_R
            total_t = (R @ total_t) + t

            # 이동량 및 회전량 계산
            delta_trans = np.linalg.norm(t)
            delta_rot = np.arccos(
                np.clip((np.trace(R) - 1) / 2.0, -1.0, 1.0)
            )

            # 수렴 조건: 평행이동 + 회전 둘 다 충분히 작으면 종료
            if delta_trans < self.icp_tolerance and delta_rot < 1e-3:
                # 카운터 기반 throttle
                if self.icp_log_counter % self.icp_log_skip == 0:
                    self.get_logger().info(
                        f"✅ ICP Converged: Iter {i + 1}/{self.icp_max_iter} "
                        f"(dL={delta_trans:.4f}, dθ={delta_rot:.6f} rad)"
                    )
                self.icp_log_counter += 1
                break
        else:
            # 최대 반복에 도달 (완전히 수렴하지 않았을 수 있음)
            if self.icp_log_counter % self.icp_log_skip == 0:
                self.get_logger().warn(
                    f"⚠️ ICP Max Iter Reached ({self.icp_max_iter}) - "
                    f"May not be fully converged"
                )
            self.icp_log_counter += 1

        return src, total_R, total_t

    # =========================================================
    #  LaserScan 콜백
    # =========================================================
    def scan_callback(self, scan_msg: LaserScan):
        ranges = np.array(scan_msg.ranges, dtype=np.float32)

        # 각도 배열 한 번만 생성 (길이 변화 시 다시 생성)
        if self.angles is None or len(self.angles) != len(ranges):
            self.angles = np.linspace(
                scan_msg.angle_min,
                scan_msg.angle_max,
                len(ranges),
                dtype=np.float32
            )

        # 유효 range 필터링
        valid = (ranges > scan_msg.range_min) & (ranges < scan_msg.range_max)
        if not np.any(valid):
            # 유효한 포인트가 없으면 그냥 리턴
            return

        x_curr = ranges[valid] * np.cos(self.angles[valid])
        y_curr = ranges[valid] * np.sin(self.angles[valid])
        curr_points = np.stack([x_curr, y_curr], axis=1)

        # 첫 프레임이면 prev_points 초기화
        if self.prev_points is None:
            self.prev_points = curr_points
            self.get_logger().info("First frame saved. Starting ICP loop...")
            return

        # -----------------------------
        # 1. ICP 수행 (downsample 포함)
        # -----------------------------
        if self.downsample_rate > 1:
            source_down = self.prev_points[::self.downsample_rate]
            target_down = curr_points[::self.downsample_rate]
        else:
            source_down = self.prev_points
            target_down = curr_points

        if len(source_down) < 10 or len(target_down) < 10:
            # 다운샘플링 후 포인트가 너무 적으면 ICP 생략
            self.prev_points = curr_points
            return

        _, R_icp, t_icp = self.icp_2d(source_down, target_down)

        # 로봇(센서) 이동량 대략 확인
        move_dist = np.linalg.norm(t_icp)

        # -----------------------------
        # 2. 변환 적용 (이전 프레임 포인트를 현재 좌표계로 보정)
        # -----------------------------
        prev_points_comp = (self.prev_points @ R_icp.T) + t_icp

        # -----------------------------
        # 3. 거리 기반 Difference (동적 포인트 검출)
        # -----------------------------
        if len(prev_points_comp) > 0:
            tree = KDTree(prev_points_comp)
            dists, _ = tree.query(curr_points)  # curr_points 기준에서 이전 포인트까지 거리

            is_dynamic = (dists.flatten() > self.threshold)

            # 노이즈 제거 (binary morphological operations)
            is_dynamic = scipy.ndimage.binary_opening(is_dynamic, structure=np.ones(3))
            is_dynamic = scipy.ndimage.binary_dilation(is_dynamic, structure=np.ones(3))

            num_dynamic = int(np.sum(is_dynamic))

            # 동적 포인트 로그 (카운터 기반 throttle)
            if num_dynamic > 0:
                if self.dynamic_log_counter % self.dynamic_log_skip == 0:
                    self.get_logger().info(
                        f"🔥 Dynamic Points: {num_dynamic} | Move: {move_dist:.3f} m"
                    )
                self.dynamic_log_counter += 1

            # -----------------------------
            # 4. 시각화
            # -----------------------------
            self.publish_markers(scan_msg.header, curr_points, is_dynamic)

        # -----------------------------
        # 5. 현재 포인트를 다음 프레임의 prev_points로 저장
        # -----------------------------
        self.prev_points = curr_points

    # =========================================================
    #  Marker 시각화
    # =========================================================
    def publish_markers(self, header, points: np.ndarray, is_dynamic: np.ndarray):
        marker = Marker()
        marker.header = header
        marker.ns = "scan_matching_mos"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD

        # 유효한 쿼터니언 (w=1)
        marker.pose.orientation.w = 1.0

        # POINTS 타입에서 점 크기
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.0  # 2D니까 0으로 두어도 됨

        # lifetime 설정 (0.1초 정도만 유지)
        marker.lifetime = Duration(sec=0, nanosec=int(0.1 * 1e9))

        # 동적 포인트 색상 (빨간색)
        c_dynamic = std_msgs.msg.ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)

        dynamic_points = points[is_dynamic]

        for p_xy in dynamic_points:
            p = Point()
            p.x = float(p_xy[0])
            p.y = float(p_xy[1])
            p.z = 0.2
            marker.points.append(p)
            marker.colors.append(c_dynamic)

        self.marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = ScanMatchingMosNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

