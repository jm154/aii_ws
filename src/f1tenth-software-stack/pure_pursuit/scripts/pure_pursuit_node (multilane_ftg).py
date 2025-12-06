#!/usr/bin/env python3

import numpy as np
from scipy.spatial import distance, transform
from scipy.ndimage import binary_dilation
import os

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry, OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PointStamped, QuaternionStamped, TransformStamped, PoseStamped

class MultiLanePurePursuit(Node):
    def __init__(self):
        super().__init__('multi_lane_pure_pursuit_node')

        # ===== Parameters =====
        self.is_real = False
        self.map_name = 'E1_out2_refined'
        self.steering_gain = 0.5
        self.speed_reducing_rate = 0.6
        self.max_sight = 4.0
        self.steering_limit = 0.35  # radians

        # Speed-dependent lookahead
        self.lookahead_norm = 2.0 # Lookahead for normal speed
        self.lookahead_slow = 1.8 # Lookahead for decelerated speed
        self.wheelbase = self.lookahead_norm  # [m]  # (원본 유지)

        # Multi-lane parameters
        self.lane_offsets = [-0.8, -0.4, 0.0, 0.4, 0.8]
        self.lanes = []
        self.current_lane_idx = 2  # Start with the center lane
        self.hysteresis_counter = 0
        self.hysteresis_threshold = 0
        self.opposite_lane_penalty = 1.0  # Penalty for choosing opposite lane
        self.lane_switch_timer = 0
        self.lane_switch_cooldown = 30 # cycles

        # ===== Local Occupancy Grid Parameters (LiDAR-based) =====
        self.grid_res = 0.05  # [m/cell]
        self.grid_forward = 3.0  # [m]
        self.grid_side = 1.0  # [m]
        self.inflate_radius_m = 0.3  # [m]
        self.inflate_iters = max(1, int(self.inflate_radius_m / self.grid_res))

        self.grid_w = int(self.grid_forward / self.grid_res)
        self.grid_h = int(self.grid_side / self.grid_res)
        self.grid_y_offset = self.grid_h // 2

        # ===== FTG (Follow-the-Gap) params =====
        # 전방 좌/우 15°~45° 구간에서 1.0m 이내 장애물 감지 시 FTG 전환
        self.ftg_active = False
        self.ftg_proc_ranges = None
        self.ftg_downsample_gap = 10
        self.ftg_max_sight = self.max_sight
        self.ftg_max_gap_safe_dist = 1.2
        self.ftg_trigger_dist = 1.0
        self.ftg_sector_deg_min = 15.0
        self.ftg_sector_deg_max = 45.0

        # ===== Topics/Publishers =====
        odom_topic = '/pf/viz/inferred_pose' if self.is_real else '/ego_racecar/odom'
        self.sub_pose = self.create_subscription(PoseStamped if self.is_real else Odometry, odom_topic, self.pose_callback, 1)

        self.pub_drive = self.create_publisher(AckermannDriveStamped, '/drive', 1)
        self.drive_msg = AckermannDriveStamped()

        # Waypoints
        map_path = os.path.abspath(os.path.join('src', "f1tenth-software-stack", 'csv_data'))
        csv_data = np.loadtxt(f"{map_path}/{self.map_name}.csv", delimiter=';', skiprows=1)
        self.waypoints = csv_data[:, 1:3]
        if self.is_real:
            self.ref_speed = csv_data[:, 5] * 0.6
        else:
            self.ref_speed = 4.0
        self.numWaypoints = self.waypoints.shape[0]
        self._generate_lanes()
        self.active_waypoints = self.lanes[self.current_lane_idx]

        # Viz
        visualization_topic = '/visualization_marker_array'
        self.pub_vis = self.create_publisher(MarkerArray, visualization_topic, 1)
        self.markerArray = MarkerArray()

        self.visualization_init()

        # LiDAR
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        # Current state
        self.currX = 0.0
        self.currY = 0.0
        self.rot = np.eye(3)
        self.have_pose = False
        self.centerline_closest_index = 0

        # Latest local grid
        self.local_grid = np.zeros((self.grid_h, self.grid_w), dtype=np.uint8)

    def visualization_init(self):
        # Colors
        self.colors = [
            (0.0, 1.0, 0.0, 1.0),  # Green
            (0.5, 0.5, 1.0, 1.0),  # Light Blue
            (1.0, 1.0, 1.0, 1.0),  # White
            (1.0, 0.5, 0.5, 1.0),  # Light Red
            (1.0, 0.0, 0.0, 1.0),  # Red
        ]

        # Lane markers
        for i, lane in enumerate(self.lanes):
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.type = Marker.POINTS
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = self.colors[i]
            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.id = i
            marker.points = [Point(x=p[0], y=p[1], z=0.0) for p in lane]
            self.markerArray.markers.append(marker)

        # Active lane marker
        self.active_lane_marker = Marker()
        self.active_lane_marker.header.frame_id = 'map'
        self.active_lane_marker.type = Marker.POINTS
        self.active_lane_marker.color.r, self.active_lane_marker.color.g, self.active_lane_marker.color.b, self.active_lane_marker.color.a = (1.0, 1.0, 0.0, 1.0) # Yellow
        self.active_lane_marker.scale.x = 0.1
        self.active_lane_marker.scale.y = 0.1
        self.active_lane_marker.id = len(self.lanes)
        self.markerArray.markers.append(self.active_lane_marker)

        # Target marker
        self.targetMarker = Marker()
        self.targetMarker.header.frame_id = 'map'
        self.targetMarker.type = Marker.POINTS
        self.targetMarker.color.r = 1.0
        self.targetMarker.color.a = 1.0
        self.targetMarker.scale.x = 0.2
        self.targetMarker.scale.y = 0.2
        self.targetMarker.id = len(self.lanes) + 1
        self.markerArray.markers.append(self.targetMarker)

    # ============ Pure Pursuit / FTG 스위치 =============
    def pose_callback(self, pose_msg):
        self.currX = pose_msg.pose.position.x if self.is_real else pose_msg.pose.pose.position.x
        self.currY = pose_msg.pose.position.y if self.is_real else pose_msg.pose.pose.position.y
        quat = pose_msg.pose.orientation if self.is_real else pose_msg.pose.pose.orientation
        quat = [quat.x, quat.y, quat.z, quat.w]
        self.rot = transform.Rotation.from_quat(quat).as_matrix()
        self.have_pose = True

        # Find closest waypoint on centerline for speed profile
        curr_pos = np.array([self.currX, self.currY]).reshape((1, 2))
        distances = distance.cdist(curr_pos, self.waypoints, 'euclidean').reshape((-1))
        self.centerline_closest_index = int(np.argmin(distances))

        # FTG 우선: 전방 위험(15°~45°/좌우, 1m 이내)이면 FTG 제어 수행 후 반환
        if self.ftg_active and self.ftg_proc_ranges is not None:
            self.ftg_do_control(self.ftg_proc_ranges)
            return

        # 평소: 멀티레인 + Pure Pursuit
        decelerate = self._select_best_lane()
        self.drive_to_target_pure_pursuit(decelerate)

    def drive_to_target_pure_pursuit(self, decelerate=False):
        # Determine lookahead distance and velocity
        if self.is_real:
            base_velocity = self.ref_speed[self.centerline_closest_index]
        else:
            base_velocity = self.ref_speed

        if decelerate:
            lookahead_dist = self.lookahead_slow
            velocity = base_velocity * self.speed_reducing_rate
            print("[Deceleration] Obstacle prompted lane change, reducing speed.")
        else:
            lookahead_dist = self.lookahead_norm
            velocity = base_velocity

        # Find the target point
        target_point = self.get_target_point(lookahead_dist)

        # Transform the target point to the car's coordinate frame
        translated_target_point = self.translate_point(target_point)

        # Calculate curvature/steering angle (원본 유지)
        y = translated_target_point[1]
        gamma = self.steering_gain * (2 * y / self.wheelbase**2)
        angle = np.clip(gamma, -self.steering_limit, self.steering_limit)

        self.drive_msg.drive.steering_angle = angle
        self.drive_msg.drive.speed = float(velocity)
        self.pub_drive.publish(self.drive_msg)
        print(f"[Pure Pursuit] steer={round(angle, 3)}, speed={float(velocity):.2f}, lookahead={lookahead_dist:.1f}")

        # Update and publish visualization markers
        now = self.get_clock().now().to_msg()
        for marker in self.markerArray.markers:
            marker.header.stamp = now
        self.active_lane_marker.points = [Point(x=p[0], y=p[1], z=0.0) for p in self.active_waypoints]
        self.targetMarker.points = [Point(x=float(target_point[0]), y=float(target_point[1]), z=0.0)]
        self.pub_vis.publish(self.markerArray)

    def get_target_point(self, lookahead_dist):
        curr_pos = np.array([self.currX, self.currY]).reshape((1, 2))
        distances = distance.cdist(curr_pos, self.active_waypoints, 'euclidean').reshape((-1))
        closest_index = int(np.argmin(distances))

        point_index = closest_index
        dist = float(distances[point_index])

        while dist < lookahead_dist:
            point_index = (point_index + 1) % len(self.active_waypoints)
            dist = float(distances[point_index])

        return self.active_waypoints[point_index]

    def translate_point(self, target_point):
        # Create a 4x4 homogeneous transformation matrix
        H = np.zeros((4, 4))
        H[0:3, 0:3] = self.rot.T  # Transpose of rotation matrix for world to car frame
        H[0:3, 3] = -self.rot.T @ np.array([self.currX, self.currY, 0.0])
        H[3, 3] = 1.0

        # Convert target point to homogeneous coordinates
        target_homogeneous = np.array([target_point[0], target_point[1], 0.0, 1.0])

        # Apply the transformation
        transformed_target = H @ target_homogeneous

        return transformed_target[0:3]

    # ========================= Lane Generation and Selection =========================
    def _generate_lanes(self):
        """
        Generates parallel lanes based on the centerline waypoints.
        """
        self.lanes = []
        for offset in self.lane_offsets:
            lane = []
            for i in range(self.numWaypoints):
                p1 = self.waypoints[i]
                p2 = self.waypoints[(i + 1) % self.numWaypoints]

                # Tangent vector
                tangent = p2 - p1
                tangent_norm = np.linalg.norm(tangent)
                if tangent_norm > 0:
                    tangent = tangent / tangent_norm

                # Normal vector
                normal = np.array([-tangent[1], tangent[0]])

                new_point = p1 + offset * normal
                lane.append(new_point)
            self.lanes.append(np.array(lane))

    def _select_best_lane(self):
        """
        Selects the best lane based on collision checking and a cost function.
        Returns whether to decelerate.
        """
        original_lane_idx = self.current_lane_idx

        if self.lane_switch_timer > 0:
            self.lane_switch_timer -= 1

        # Check for collisions on all lanes
        lane_is_colliding = [self._check_lane_for_collision(lane) for lane in self.lanes]

        # If current lane is not colliding and we are in a cooldown period, stay in the lane.
        if not lane_is_colliding[self.current_lane_idx] and self.lane_switch_timer > 0:
            self.active_waypoints = self.lanes[self.current_lane_idx]
            return False

        best_lane_idx = self.current_lane_idx
        min_cost = float('inf')

        for i, lane in enumerate(self.lanes):
            if not lane_is_colliding[i]:
                cost = abs(self.lane_offsets[i])  # Simple cost: prefer center lane

                # Add penalty for choosing an opposite lane in a short period (원본 유지)
                is_opposite_side = (self.current_lane_idx < 2 and i > 2) or \
                                   (self.current_lane_idx > 2 and i < 2)
                if is_opposite_side:
                    cost += self.opposite_lane_penalty

                if cost < min_cost:
                    min_cost = cost
                    best_lane_idx = i

        # Hysteresis and lane change logic
        lane_changed = False
        if best_lane_idx != self.current_lane_idx:
            self.hysteresis_counter += 1
            if self.hysteresis_counter >= self.hysteresis_threshold:
                self.current_lane_idx = best_lane_idx
                print(f"[Lane Change] Switched to lane {self.current_lane_idx} (offset: {self.lane_offsets[self.current_lane_idx]}m)")
                self.hysteresis_counter = 0
                lane_changed = True
                self.lane_switch_timer = self.lane_switch_cooldown  # Reset cooldown on lane change
        else:
            self.hysteresis_counter = 0

        self.active_waypoints = self.lanes[self.current_lane_idx]

        # Determine if deceleration is needed
        # Decelerate if we changed lanes because the old lane was blocked.
        decelerate = lane_changed and lane_is_colliding[original_lane_idx]
        return decelerate

    def _check_lane_for_collision(self, lane):
        """
        Checks if a given lane collides with obstacles in the local grid.
        """
        # Find the closest point on the lane to the car
        dists = distance.cdist(np.array([[self.currX, self.currY]]), lane[:, 0:2]).reshape(-1)
        closest_idx = int(np.argmin(dists))

        # Check a horizon ahead
        check_horizon_m = 4.0
        check_horizon_indices = int(check_horizon_m / 0.18)  # Assuming waypoints are ~0.1m apart

        for i in range(check_horizon_indices):
            point_idx = (closest_idx + i) % len(lane)
            point = lane[point_idx]

            # Transform point to local frame
            local_point = self._map_point_to_base_local(point)

            # Check if the point is within the grid
            gx, gy, in_bounds = self._local_point_to_grid(local_point[0], local_point[1])

            if in_bounds and self.local_grid[gy, gx] > 0:
                return True  # Collision

        return False  # No collision

    # ========================= FTG (Follow-the-Gap) helpers =========================
    def _min_range_in_sector(self, scan: LaserScan, ang_min_rad: float, ang_max_rad: float) -> float:
        """
        [ang_min_rad, ang_max_rad] 구간(라디안)의 라이다 최소거리 반환.
        스캔 각도 기준은 scan.angle_min ~ scan.angle_max, 증분은 scan.angle_increment.
        """
        a0 = scan.angle_min
        da = scan.angle_increment
        n = len(scan.ranges)

        i_start = int(np.ceil((ang_min_rad - a0) / da))
        i_end = int(np.floor((ang_max_rad - a0) / da))
        i_start = max(0, min(n - 1, i_start))
        i_end = max(0, min(n - 1, i_end))
        if i_end < i_start:
            i_start, i_end = i_end, i_start

        seg = np.asarray(scan.ranges[i_start:i_end + 1], dtype=np.float32)
        seg = np.where(np.isfinite(seg), seg, np.inf)
        seg = np.where(seg > 0.0, seg, np.inf)
        return float(np.min(seg)) if seg.size > 0 else float('inf')

    def ftg_preprocess_lidar(self, ranges: np.ndarray) -> np.ndarray:
        """FTG 전처리: 다운샘플 + 시야 클리핑"""
        if self.ftg_downsample_gap <= 1:
            proc = ranges.copy()
        else:
            n = len(ranges) // self.ftg_downsample_gap
            proc = np.zeros(n, dtype=np.float32)
            for i in range(n):
                s = i * self.ftg_downsample_gap
                e = s + self.ftg_downsample_gap
                proc[i] = np.mean(ranges[s:e])
        proc = np.clip(proc, 0.0, self.ftg_max_sight)
        return proc

    def ftg_bubble_danger_zone(self, proc_ranges: np.ndarray, radius: int = 2) -> np.ndarray:
        """가장 가까운 장애물 주변 구간을 버블로 0 처리"""
        out = proc_ranges.copy()
        if out.size == 0:
            return out
        min_idx = int(np.argmin(out))
        start = max(min_idx - radius, 0)
        end = min(min_idx + radius, len(out) - 1)
        out[start:end + 1] = 0.0
        return out

    def ftg_find_max_gap(self, ranges: np.ndarray):
        """안전 임계 이상으로 연속 비어있는 최장 gap 구간 탐색"""
        longest = 0
        start = end = 0
        curr = 0
        N = len(ranges)
        while curr < N:
            if ranges[curr] > self.ftg_max_gap_safe_dist:
                s = curr
                while curr < N and ranges[curr] > self.ftg_max_gap_safe_dist:
                    curr += 1
                if curr - s > longest:
                    longest = curr - s
                    start, end = s, curr
            curr += 1
        return start, end

    def ftg_find_best_point(self, start: int, end: int, ranges: np.ndarray) -> int:
        """gap의 중앙 인덱스를 타깃으로 선택(기본 방식)"""
        if end <= start:
            return start
        return int((start + end) / 2)

    def ftg_do_control(self, proc_ranges: np.ndarray):
        """
        FTG 제어: 최장 gap 중앙을 향해 조향, 위험 가까우면 속도 저감.
        스티어링 한계는 기존 steering_limit을 따름.
        """
        start, end = self.ftg_find_max_gap(proc_ranges)
        best_i = self.ftg_find_best_point(start, end, proc_ranges)

        # 인덱스를 각도로 맵핑(간단한 휴리스틱)
        steering_angle = np.deg2rad(best_i * (self.ftg_downsample_gap / 4.0) - 90.0)
        steering_angle = float(np.clip(steering_angle, -self.steering_limit, self.steering_limit))

        # 속도 정책: 가까운 위험 존재 시 감속
        if np.min(proc_ranges) < self.ftg_trigger_dist:
            base_velocity = self.ref_speed[self.centerline_closest_index] if self.is_real else self.ref_speed
            velocity = float(base_velocity * self.speed_reducing_rate)
        else:
            velocity = float(self.ref_speed if not self.is_real else self.ref_speed[self.centerline_closest_index])

        self.drive_msg.drive.steering_angle = steering_angle
        self.drive_msg.drive.speed = velocity
        self.pub_drive.publish(self.drive_msg)
        print(f"[FTG] steer={round(steering_angle, 3)}, speed={velocity:.2f}")

    # ========================= Local Grid + FTG trigger =========================
    def scan_callback(self, scan_msg: LaserScan):
        # 기존 로컬 그리드 생성
        self._build_local_grid_from_scan(scan_msg)

        # ===== FTG 전환 로직 (각도 기반) =====
        # 전방 좌/우 15°~45° 섹터의 최소 거리 계산 → 1.0m보다 작으면 FTG on
        pos_min = self._min_range_in_sector(
            scan_msg,
            np.deg2rad(self.ftg_sector_deg_min),
            np.deg2rad(self.ftg_sector_deg_max),
        )
        neg_min = self._min_range_in_sector(
            scan_msg,
            -np.deg2rad(self.ftg_sector_deg_max),
            -np.deg2rad(self.ftg_sector_deg_min),
        )
        front_min = min(pos_min, neg_min)
        self.ftg_active = bool(front_min < self.ftg_trigger_dist)

        # FTG 제어용 라이다 전처리(항상 준비, FTG off면 그냥 보관 X)
        raw = np.array(scan_msg.ranges, dtype=np.float32)
        raw = np.where(np.isfinite(raw), raw, self.ftg_max_sight + 1.0)
        raw = np.where(raw > 0.0, raw, self.ftg_max_sight + 1.0)
        raw = np.clip(raw, 0.0, self.ftg_max_sight)
        proc = self.ftg_preprocess_lidar(raw)
        proc = self.ftg_bubble_danger_zone(proc, radius=2)
        self.ftg_proc_ranges = proc if self.ftg_active else None

    def _build_local_grid_from_scan(self, scan: LaserScan):
        if not self.have_pose:
            return

        grid = np.zeros((self.grid_h, self.grid_w), dtype=np.uint8)

        angle = scan.angle_min
        for r in scan.ranges:
            if np.isinf(r) or np.isnan(r) or r <= 0.0 or r > self.max_sight:
                angle += scan.angle_increment
                continue

            x = r * np.cos(angle)
            y = r * np.sin(angle)
            gi, gj, inb = self._local_point_to_grid(x, y)
            if inb:
                grid[gj, gi] = 1  # hit

            angle += scan.angle_increment

        grid = binary_dilation(grid, iterations=self.inflate_iters).astype(np.uint8) * 100
        self.local_grid = grid

    def _local_point_to_grid(self, x_local, y_local):
        if x_local < 0.0 or x_local > self.grid_forward:
            return 0, 0, False
        if abs(y_local) > (self.grid_side / 2.0):
            return 0, 0, False
        i = int(x_local / self.grid_res)
        j = int(self.grid_y_offset - (y_local / self.grid_res))
        inb = (0 <= i < self.grid_w) and (0 <= j < self.grid_h)
        return i, j, inb

    def _map_point_to_base_local(self, pt_xy):
        R_wb = self.rot  # body->world
        R_bw = R_wb.T  # world->body
        p_w = np.array([pt_xy[0] - self.currX, pt_xy[1] - self.currY, 0.0])
        p_b = R_bw @ p_w
        return np.array([p_b[0], p_b[1]])

def main(args=None):
    rclpy.init(args=args)
    node = MultiLanePurePursuit()
    print("[INFO] Multi Lane Pure Pursuit Node initialized")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

