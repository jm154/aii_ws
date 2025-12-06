#!/usr/bin/env python3

import numpy as np
from scipy.spatial import distance, transform
from scipy.ndimage import binary_dilation
import os

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


class PurePursuit(Node):
    def __init__(self):
        super().__init__('pure_pursuit_node')

        # ===== Parameters =====
        self.is_real = False
        self.map_name = 'E1_out2_refined'
        self.L = 1.0
        self.steering_gain = 0.5
        self.ref_speed = 2.0
        self.downsample_gap = 10
        self.max_sight = 4.0
        self.max_gap_safe_dist = 1.2

        # ===== Vehicle/Obstacle/Arc Params =====
        self.vehicle_width = 0.27     # [m]
        self.safety_margin_grid = 0.15     # [m] detect collision
        self.safety_margin_arc = 0.5 # [m] arc margin
        self.obs_search_radius = 0.5 # [m]
        self.arc_step = 0.10          # [m]
        self.arc_speed_scale = 0.6   # 아크 구간 감속 배율
        self.is_arc_mask = np.zeros(0, dtype=bool)  # track_waypoints와 길이 동일한 마스크


        # ===== Local Occupancy Grid Params =====
        self.grid_res = 0.05   # [m/cell]
        self.grid_forward = 6.0
        self.grid_side = 4.0

        self.grid_w = int(self.grid_forward / self.grid_res)
        self.grid_h = int(self.grid_side / self.grid_res)
        self.grid_y_offset = self.grid_h // 2
        
        # ===== Collision lookahead (extra) =====
        self.collision_check_extra = 1.0  # [m]  - 룩어헤드 L에 더해, 충돌 검사를 더 멀리

        # Buffer (for bubbled occupancy used in collision test)
        self.grid_vehicle_buffer_m = 0.5 * self.vehicle_width + self.safety_margin_grid
        self.grid_vehicle_buffer_cells = max(1, int(np.ceil(self.grid_vehicle_buffer_m / self.grid_res)))

        # ===== Topics/Publishers =====
        odom_topic = '/pf/viz/inferred_pose' if self.is_real else '/ego_racecar/odom'
        self.sub_pose = self.create_subscription(Odometry, odom_topic, self.pose_callback, 1)

        self.pub_drive = self.create_publisher(AckermannDriveStamped, '/drive', 1)
        self.drive_msg = AckermannDriveStamped()

        # Waypoints (original loop)
        map_path = os.path.abspath(os.path.join('src', "f1tenth-software-stack", 'csv_data'))
        '''
        self.waypoints = np.loadtxt(f"{map_path}/{self.map_name}.csv", delimiter=',', skiprows=1)
        self.numWaypoints = self.waypoints.shape[0]

        # ACTIVE TRACK (we splice arcs here; always exactly one list we mutate)
        self.track_waypoints = self.waypoints.copy()
        self.numTrackWaypoints = self.track_waypoints.shape[0]'''
        
        ''''''
        data = np.loadtxt(f"{map_path}/{self.map_name}.csv", delimiter=',', skiprows=1)

        # 로우가 1개인 경우에도 (1, M)으로 맞춤
        if data.ndim == 1:
            data = data[None, :]

        # 앞의 두 열만 XY로 사용 (나머지 열은 무시)
        self.waypoints = data[:, :2].astype(float)

        self.numWaypoints = self.waypoints.shape[0]
        self.track_waypoints = self.waypoints.copy()          # 항상 (N,2)
        self.numTrackWaypoints = self.track_waypoints.shape[0]
        self.is_arc_mask = np.zeros(self.numTrackWaypoints, dtype=bool)
        ''''''

        # Visualization
        visualization_topic = '/visualization_marker_array'
        self.pub_vis = self.create_publisher(MarkerArray, visualization_topic, 1)
        self.markerArray = MarkerArray()

        self.ref_speed = 6.0
        self.visualization_init()

        # LiDAR
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        # Current state
        self.currX = 0.0
        self.currY = 0.0
        self.rot = np.eye(3)
        self.have_pose = False

        # Latest grids
        self.local_grid = np.zeros((self.grid_h, self.grid_w), dtype=np.uint8)   # raw occupancy (0/100)
        self.occ_bubbled = np.zeros((self.grid_h, self.grid_w), dtype=np.uint8)  # dilated occupancy (0/100)

        # Raw LiDAR hits (local frame)
        self.last_hits = np.zeros((0, 2), dtype=np.float32)

        # Candidate arcs for viz + last spliced arc
        self.cand_left_world = None
        self.cand_right_world = None
        self.spliced_arc_world = None

        # FTG one-shot flag (set when both arcs collide)
        self.run_ftg_once = False

    # ========================= Scan / Grid =========================
    def scan_callback(self, scan_msg: LaserScan):
        """LiDAR 콜백: 로컬 occupancy/bubbled 갱신 + (필요시) FTG one-shot 실행"""
        self._build_local_grid_from_scan(scan_msg)

    # ========================= Pose / PP (single mode) =========================
    def pose_callback(self, msg):
        # Update pose
        self.currX = msg.pose.pose.position.x
        self.currY = msg.pose.pose.position.y
        quat = msg.pose.pose.orientation
        quat = [quat.x, quat.y, quat.z, quat.w]
        self.rot = transform.Rotation.from_quat(quat).as_matrix()
        self.have_pose = True

        # FTG 단발 예약 중이면 PP/타깃 갱신은 잠시 건너뛴다 (고정처럼 보이는 현상 방지)
        if self.run_ftg_once:
            return

        # Build PP target from ACTIVE TRACK
        if self.numTrackWaypoints < 3:
            # safety: fallback original
            self.track_waypoints = self.waypoints.copy()
            self.numTrackWaypoints = self.track_waypoints.shape[0]

        currPos = np.array([[self.currX, self.currY]])
        dists = distance.cdist(currPos, self.track_waypoints, 'euclidean').reshape((self.numTrackWaypoints))
        closest_idx = int(np.argmin(dists))

        # === 변경: 누적거리(arc-length) 기반 룩어헤드 ===
        target = self._get_target_ahead_track(closest_idx, self.L)
        if target is None:
            return
        t_local = self._map_point_to_base_local(target)

        # Collision check to the chosen target on ACTIVE TRACK (버블 격자 기준)
        target_idx = int(np.argmin(np.linalg.norm(self.track_waypoints - target, axis=1)))
        if len(self.is_arc_mask) == self.numTrackWaypoints and self.is_arc_mask[target_idx]:
            speed_cmd = self.ref_speed * self.arc_speed_scale
        else:
            speed_cmd = self.ref_speed
            
        check_len = float(self.L) + float(self.collision_check_extra)
        coll_idx = self._first_colliding_waypoint_idx_track_by_arclength(closest_idx, check_len)

        # If blocked -> plan arc, then splice it into ACTIVE TRACK (replace inner-of-circle wpts)
        if coll_idx is not None:
            coll_wpt = self.track_waypoints[coll_idx]
            coll_local = self._map_point_to_base_local(coll_wpt)
            planned = self._plan_and_splice_arc_from_collision(coll_local)
            if planned:
                # Recompute target on the NEW track immediately (누적거리 룩어헤드)
                dists = distance.cdist(currPos, self.track_waypoints, 'euclidean').reshape((self.numTrackWaypoints))
                closest_idx = int(np.argmin(dists))
                target = self._get_target_ahead_track(closest_idx, self.L)
                if target is None:
                    return
                t_local = self._map_point_to_base_local(target)
                self.get_logger().info("[SPLICE] Replaced inner waypoints by ARC")
            else:
                # both arcs collided -> schedule FTG once; 이 사이클은 PP 출력/마커 생략
                self.run_ftg_once = True
                return

        # Pure Pursuit control (always on ACTIVE TRACK)
        y = t_local[1]
        gamma = np.clip(self.steering_gain * (2.0 * y / (self.L ** 2)), -0.35, 0.35)

        self.drive_msg.drive.steering_angle = gamma
        self.drive_msg.drive.speed = float(speed_cmd)
        self.pub_drive.publish(self.drive_msg)

        print(f"[PP-TRACK] steer={round(gamma, 3)}, speed={float(speed_cmd):.2f}, "
              f"target_local=({t_local[0]:.2f},{t_local[1]:.2f})")

        # Visualization (ACTIVE TRACK, target, last spliced arc, candidate arcs)
        now = self.get_clock().now().to_msg()
        self.waypointMarker.header.stamp = now
        self.waypointMarker.points = [Point(x=float(w[0]), y=float(w[1]), z=0.0) for w in self.track_waypoints]

        self.targetMarker.header.stamp = now
        self.targetMarker.points = [Point(x=float(target[0]), y=float(target[1]), z=0.0)]

        markers = [self.waypointMarker, self.targetMarker]

        if self.spliced_arc_world is not None and len(self.spliced_arc_world) > 1:
            self.arcMarker.header.stamp = now
            self.arcMarker.points = [Point(x=float(p[0]), y=float(p[1]), z=0.0) for p in self.spliced_arc_world]
            markers.append(self.arcMarker)

        self._publish_candidate_arcs()  # updates left/right markers timestamps & points
        markers.extend([self.arcCandLeftMarker, self.arcCandRightMarker])

        self.markerArray.markers = markers
        self.pub_vis.publish(self.markerArray)

    # ========================= Arc Planning + Splicing =========================
    def _plan_and_splice_arc_from_collision(self, coll_local):
        """
        1) Estimate obstacle circle (C,r) near collision (local).
        2) Create clearance radius Rclr.
        3) Pick two ORIGINAL loop waypoints nearest to the circle with angle separation.
        4) Build CCW/CW arcs in local -> classify left/right -> choose feasible shortest.
        5) Transform best arc to world and **splice** into ACTIVE TRACK:
           drop all original waypoints strictly inside the circle, and replace the forward segment between the two
           selected original waypoint indices by arc points. If both arcs collide => return False.
        """
        # 1) circle near collision (prefer raw hits)
        C, obs_radius = self._estimate_obstacle_near_local_point(coll_local)
        if C is None:
            self.get_logger().warn("[ARC] obstacle estimation failed near collision point")
            return False
        Rclr = obs_radius + 0.5 * self.vehicle_width + self.safety_margin_arc

        # 2) nearest two waypoints on ORIGINAL loop (get indices too)
        A_w, B_w, idxA, idxB = self._two_nearest_wpts_to_circle_with_index(C, Rclr)
        if A_w is None or B_w is None:
            self.get_logger().warn("[ARC] need two nearest waypoints to circle but failed")
            return False

        A_l = self._map_point_to_base_local(A_w)
        B_l = self._map_point_to_base_local(B_w)

        # 3) build CCW/CW arcs on circle
        ccw_pts, cw_pts, ccw_len, cw_len = self._arcs_from_two_wpts(C, Rclr, A_l, B_l)
        if (ccw_pts is None) or (cw_pts is None):
            self.get_logger().warn("[ARC] failed to build CCW/CW arcs from two waypoints")
            return False

        # 4) classify left/right for consistent viz
        (left_name, left_local), (right_name, right_local) = self._classify_left_right_by_chord(A_l, B_l, ccw_pts, cw_pts)
        left_len  = ccw_len if left_local is ccw_pts else cw_len
        right_len = cw_len  if right_local is cw_pts else ccw_len

        # viz candidates
        self.cand_left_world  = self._local_path_to_world(left_local)
        self.cand_right_world = self._local_path_to_world(right_local)

        # 5) pick feasible (buffered occupancy)
        left_len_poly  = self._polyline_length(left_local)
        right_len_poly = self._polyline_length(right_local)

        feasible = []
        if self._path_collision_free_local_buffered(left_local):
            feasible.append(("left",  left_local,  left_len_poly))
        if self._path_collision_free_local_buffered(right_local):
            feasible.append(("right", right_local, right_len_poly))

        if not feasible:
            self.get_logger().warn("[ARC] both arcs collide; will run FTG once")
            return False

        best_name, best_local, best_len = min(feasible, key=lambda x: x[2])  # ← polyline 실길이 기준

        arc_world = self._local_path_to_world(best_local)

        # splice into ACTIVE TRACK using ORIGINAL indices (idxA -> idxB, forward wrapping)
        self._splice_arc_into_track(idxA, idxB, self._world_from_local_point(C), Rclr, arc_world)

        # save arc for viz
        self.spliced_arc_world = arc_world
        self.get_logger().info(f"[ARC] Spliced {best_name} arc (len={best_len:.2f}) between original idx {idxA}->{idxB}")
        return True

    def _splice_arc_into_track(self, idxA_orig, idxB_orig, Cw, R, arc_world):
        N = self.numWaypoints
        if N < 3:
            self.get_logger().warn("[SPLICE] original track too short")
            return

        fw_len = (idxB_orig - idxA_orig) % N
        bw_len = (idxA_orig - idxB_orig) % N
        if bw_len < fw_len:
            idxA_orig, idxB_orig = idxB_orig, idxA_orig
            arc_world = arc_world[::-1]
            fw_len = bw_len
        if fw_len == 0:
            self.get_logger().warn("[SPLICE] A and B are identical; skip splicing")
            return

        new_track = []
        new_mask = []  # ← 여기 추가
        inserted_arc = False

        for k in range(N):
            if k == idxA_orig and not inserted_arc:
                # 아크 삽입: 포인트마다 True
                for p in arc_world:
                    new_track.append(np.array([float(p[0]), float(p[1])]))
                    new_mask.append(True)
                inserted_arc = True
                continue

            if self._in_forward_span(k, idxA_orig, idxB_orig, N):
                continue

            w = self.waypoints[k]
            if np.linalg.norm(w - Cw) >= R:
                new_track.append(w)
                new_mask.append(False)

        if len(new_track) < 3:
            self.get_logger().warn("[SPLICE] new track too short after splicing; reverting to original")
            self.track_waypoints = self.waypoints.copy()
            self.numTrackWaypoints = self.track_waypoints.shape[0]
            self.is_arc_mask = np.zeros(self.numTrackWaypoints, dtype=bool)  # ← 전부 False
        else:
            self.track_waypoints = np.array(new_track, dtype=float)
            self.numTrackWaypoints = self.track_waypoints.shape[0]
            self.is_arc_mask = np.array(new_mask, dtype=bool)  # ← 마스크 저장


    # ========================= Local Grid / Scan =========================
    def _build_local_grid_from_scan(self, scan: LaserScan):
        if not self.have_pose:
            return

        grid = np.zeros((self.grid_h, self.grid_w), dtype=np.uint8)
        hits = []

        angle = scan.angle_min
        idx_step = max(1, self.downsample_gap // 2)
        for i in range(0, len(scan.ranges), idx_step):
            r = scan.ranges[i]
            if np.isinf(r) or np.isnan(r) or r <= 0.0 or r > self.max_sight:
                angle += scan.angle_increment * idx_step
                continue

            x = r * np.cos(angle)
            y = r * np.sin(angle)
            gi, gj, inb = self._local_point_to_grid(x, y)
            if inb:
                grid[gj, gi] = 1
            hits.append((x, y))
            angle += scan.angle_increment * idx_step

        # Raw occupancy (0/100)
        self.local_grid = (grid > 0).astype(np.uint8) * 100

        # Bubbled occupancy for waypoint/arc collision test (0/100)
        occ = (grid > 0).astype(np.uint8)
        occ_b = binary_dilation(occ, iterations=self.grid_vehicle_buffer_cells).astype(np.uint8)
        self.occ_bubbled = occ_b * 100

        # Keep raw hits
        self.last_hits = np.array(hits, dtype=np.float32) if hits else np.zeros((0, 2), dtype=np.float32)

        # If FTG one-shot requested, do it here (so we have the latest scan)
        if self.run_ftg_once:
            self._run_ftg_once(scan)

    # ========================= Target selection over ACTIVE TRACK =========================
    def _get_target_ahead_track(self, closest_idx, lookahead):
        """
        경로 누적거리(arc-length) 기준으로 L 이상 앞의 타깃을 반환.
        세그먼트 끝점(b)을 반환(필요시 보간으로 바꿀 수 있음).
        """
        idx = closest_idx
        acc = 0.0
        for _ in range(self.numTrackWaypoints):
            a = self.track_waypoints[idx]
            b = self.track_waypoints[(idx + 1) % self.numTrackWaypoints]
            seg = np.linalg.norm(b - a)
            acc += seg
            if acc >= lookahead:
                return b
            idx = (idx + 1) % self.numTrackWaypoints
        # 못 찾으면 한 칸 전진
        return self.track_waypoints[(closest_idx + 1) % self.numTrackWaypoints]

    def _first_colliding_waypoint_idx_track_by_arclength(self, start_idx, max_len_m):
        """start_idx에서 arc-length 누적이 max_len_m를 넘지 않는 범위에서
           버블과 최초로 충돌하는 웨이포인트 인덱스를 반환. 없으면 None."""
        i = start_idx
        acc = 0.0
        for _ in range(self.numTrackWaypoints):
            # 현재 점 검사
            wpt = self.track_waypoints[i]
            p_local = self._map_point_to_base_local(wpt)
            gi, gj, inb = self._local_point_to_grid(p_local[0], p_local[1])
            if inb and self.occ_bubbled[gj, gi] > 0:
                return i
            # 다음 세그먼트로 진행
            a = self.track_waypoints[i]
            b = self.track_waypoints[(i + 1) % self.numTrackWaypoints]
            acc += float(np.linalg.norm(b - a))
            if acc > max_len_m:
                break
            i = (i + 1) % self.numTrackWaypoints
        return None


    # ========================= Obstacle Estimation =========================
    def _estimate_obstacle_near_local_point(self, p_local):
        C, r = self._estimate_obstacle_center_from_hits(p_local, self.obs_search_radius)
        if C is not None:
            return C, r
        gi, gj, inb = self._local_point_to_grid(p_local[0], p_local[1])
        if not inb:
            return None, None
        return self._estimate_obstacle_at_cell(gi, gj)

    def _estimate_obstacle_center_from_hits(self, p_local, search_radius=None):
        if self.last_hits is None or len(self.last_hits) == 0:
            return None, None
        R = search_radius if search_radius is not None else self.obs_search_radius
        d = np.linalg.norm(self.last_hits - p_local[None, :], axis=1)
        neigh = self.last_hits[d <= R]
        if len(neigh) < 5:
            return None, None
        C = np.mean(neigh, axis=0)
        r = np.max(np.linalg.norm(neigh - C[None, :], axis=1))
        r = max(float(r), 0.10)
        return C, r

    def _estimate_obstacle_at_cell(self, gi, gj):
        win_r = max(1, int(self.obs_search_radius / self.grid_res))
        xs, ys = [], []
        for dj in range(-win_r, win_r + 1):
            j = gj + dj
            if j < 0 or j >= self.grid_h:
                continue
            row = self.local_grid[j]
            i_start = max(0, gi - win_r)
            i_end = min(self.grid_w - 1, gi + win_r)
            occ_cols = np.where(row[i_start:i_end + 1] > 0)[0]
            if occ_cols.size == 0:
                continue
            for dc in occ_cols:
                i = i_start + dc
                x = (i + 0.5) * self.grid_res
                y = (self.grid_y_offset - j + 0.5) * self.grid_res
                xs.append(x)
                ys.append(y)
        if len(xs) < 5:
            return None, None
        pts = np.vstack([xs, ys]).T
        c = np.mean(pts, axis=0)
        r = np.max(np.linalg.norm(pts - c, axis=1)) + 0.5 * self.grid_res
        r = max(r, 0.10)
        return c, r

    # ========================= Collision check helpers =========================
    def _path_collision_free_local_buffered(self, pts_local, step=None, out_of_bounds_blocks=True):
        """
        로컬 좌표계 polyline(pts_local)이 팽창 점유(self.occ_bubbled)와 충돌하는지 검사.
        - step: 샘플 간격[m], None이면 grid_res/2
        - out_of_bounds_blocks: 격자 밖을 충돌로 볼지 여부
        """
        if pts_local is None or len(pts_local) == 0:
            return False  # 경로가 없으면 실패

        step = step or (0.5 * self.grid_res)

        prev = pts_local[0]
        if not self._point_free_in_bubbled(prev[0], prev[1], out_of_bounds_blocks):
            return False

        for i in range(1, len(pts_local)):
            cur = pts_local[i]
            seg = cur - prev
            seg_len = float(np.linalg.norm(seg))
            if seg_len <= 1e-6:
                prev = cur
                continue
            n = max(1, int(np.ceil(seg_len / step)))
            for k in range(1, n + 1):
                p = prev + seg * (k / n)
                if not self._point_free_in_bubbled(p[0], p[1], out_of_bounds_blocks):
                    return False
            prev = cur
        return True

    def _point_free_in_bubbled(self, x_local, y_local, out_of_bounds_blocks=True):
        gi, gj, inb = self._local_point_to_grid(x_local, y_local)
        if not inb:
            return (not out_of_bounds_blocks)
        return self.occ_bubbled[gj, gi] == 0

    # ========================= Candidate arcs viz =========================
    def _publish_candidate_arcs(self):
        """Update candidate arc markers (left/right) for visualization."""
        now = self.get_clock().now().to_msg()

        # Left candidate (orange)
        self.arcCandLeftMarker.header.stamp = now
        if self.cand_left_world is not None and len(self.cand_left_world) > 1:
            self.arcCandLeftMarker.points = [
                Point(x=float(p[0]), y=float(p[1]), z=0.0)
                for p in self.cand_left_world
            ]
        else:
            self.arcCandLeftMarker.points = []

        # Right candidate (yellow)
        self.arcCandRightMarker.header.stamp = now
        if self.cand_right_world is not None and len(self.cand_right_world) > 1:
            self.arcCandRightMarker.points = [
                Point(x=float(p[0]), y=float(p[1]), z=0.0)
                for p in self.cand_right_world
            ]
        else:
            self.arcCandRightMarker.points = []

    # ========================= Arc Geometry =========================
    def _compose_arc_path(self, C, R, phi_in, phi_out, ccw=True):
        def wrap(a):
            return (a + 2*np.pi) % (2*np.pi)
        a0 = wrap(phi_in)
        a1 = wrap(phi_out)
        if ccw:
            if a1 < a0:
                a1 += 2*np.pi
            arc_len = (a1 - a0) * R
            n = max(3, int(arc_len / self.arc_step))
            angs = np.linspace(a0, a1, n)
        else:
            if a1 > a0:
                a1 -= 2*np.pi
            arc_len = (a0 - a1) * R
            n = max(3, int(arc_len / self.arc_step))
            angs = np.linspace(a0, a1, n)
        arc_pts = np.column_stack([C[0] + R*np.cos(angs), C[1] + R*np.sin(angs)])
        return arc_pts

    def _arcs_from_two_wpts(self, C, R, A_local, B_local):
        def ang(p):
            return np.arctan2(p[1] - C[1], p[0] - C[0])
        a0 = ang(A_local)
        a1 = ang(B_local)
        def normalize(a): return (a + 2*np.pi) % (2*np.pi)
        a0n, a1n = normalize(a0), normalize(a1)
        # CCW
        a0_ccw, a1_ccw = a0n, a1n
        if a1_ccw < a0_ccw: a1_ccw += 2.0*np.pi
        if abs(a1_ccw - a0_ccw) < np.deg2rad(3.0): a1_ccw = a0_ccw + np.deg2rad(3.0)
        arc_len_ccw = (a1_ccw - a0_ccw)*R
        n_ccw = max(3, int(arc_len_ccw / self.arc_step))
        angs_ccw = np.linspace(a0_ccw, a1_ccw, n_ccw)
        ccw_pts = np.column_stack([C[0] + R*np.cos(angs_ccw), C[1] + R*np.sin(angs_ccw)])
        # CW
        a0_cw, a1_cw = a0n, a1n
        if a1_cw > a0_cw: a1_cw -= 2.0*np.pi
        if abs(a0_cw - a1_cw) < np.deg2rad(3.0): a1_cw = a0_cw - np.deg2rad(3.0)
        arc_len_cw = (a0_cw - a1_cw)*R
        n_cw = max(3, int(arc_len_cw / self.arc_step))
        angs_cw = np.linspace(a0_cw, a1_cw, n_cw)
        cw_pts = np.column_stack([C[0] + R*np.cos(angs_cw), C[1] + R*np.sin(angs_cw)])
        return ccw_pts, cw_pts, float(arc_len_ccw), float(arc_len_cw)

    def _classify_left_right_by_chord(self, A_l, B_l, ccw_pts, cw_pts):
        def side(Q):
            AB = B_l - A_l
            AQ = Q - A_l
            return AB[0]*AQ[1] - AB[1]*AQ[0]
        ccw_mid = ccw_pts[len(ccw_pts)//2]
        cw_mid  = cw_pts[len(cw_pts)//2]
        s_ccw = side(ccw_mid); s_cw = side(cw_mid)
        if s_ccw * s_cw < 0:
            if s_ccw > 0: return ("left", ccw_pts), ("right", cw_pts)
            else:         return ("left", cw_pts), ("right", ccw_pts)
        else:
            return ("left", ccw_pts), ("right", cw_pts)
            
    def _polyline_length(self, pts):
        if pts is None or len(pts) < 2:
            return float('inf')
        d = np.diff(pts, axis=0)
        return float(np.sum(np.linalg.norm(d, axis=1)))


    # ========================= Grid utils / frames =========================
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
        R_wb = self.rot
        R_bw = R_wb.T
        p_w = np.array([pt_xy[0] - self.currX, pt_xy[1] - self.currY, 0.0])
        p_b = R_bw @ p_w
        return np.array([p_b[0], p_b[1]])

    def _local_path_to_world(self, pts_local):
        R_wb = self.rot
        t = np.array([self.currX, self.currY, 0.0])
        pts3 = np.column_stack([pts_local, np.zeros(len(pts_local))])
        out = (R_wb @ pts3.T).T + t
        return out[:, :2]

    def _world_from_local_point(self, p_local_xy):
        R_wb = self.rot
        t = np.array([self.currX, self.currY, 0.0])
        p3 = np.array([p_local_xy[0], p_local_xy[1], 0.0])
        return (R_wb @ p3 + t)[:2]

    # ========================= FTG one-shot =========================
    def _run_ftg_once(self, scan_msg: LaserScan):
        ranges = np.array(scan_msg.ranges[180:900])
        proc = self._preprocess_lidar(ranges)
        proc = self._bubble(proc, radius=2)
        s, e = self._find_max_gap(proc)
        if e <= s:
            # no gap -> stop
            self.drive_msg.drive.steering_angle = 0.0
            self.drive_msg.drive.speed = 0.0
            self.pub_drive.publish(self.drive_msg)
            self.get_logger().warn("[FTG] No valid gap; stopping")
        else:
            best = (s + e) // 2
            angle = np.deg2rad(best * self.downsample_gap / 4.0 - 90.0)
            speed = 1.0 if np.min(proc) < 1.0 else 2.0
            self.drive_msg.drive.steering_angle = angle
            self.drive_msg.drive.speed = speed
            self.pub_drive.publish(self.drive_msg)
            print(f"[FTG-ONCE] steer={round(angle, 3)}, speed={speed:.2f}, gap=({s},{e}), min_proc={np.min(proc):.2f}")
        # clear flag so PP resumes next cycles
        self.run_ftg_once = False

    def _preprocess_lidar(self, ranges):
        proc = np.zeros(len(ranges) // self.downsample_gap)
        for i in range(len(proc)):
            proc[i] = np.mean(ranges[i * self.downsample_gap:(i + 1) * self.downsample_gap])
        return np.clip(proc, 0.0, self.max_sight)

    def _bubble(self, proc, radius=2):
        k = int(np.argmin(proc))
        s = max(k - radius, 0)
        e = min(k + radius, len(proc) - 1)
        proc[s:e + 1] = 0.0
        return proc

    def _find_max_gap(self, proc):
        best_len, best_s, best_e = 0, 0, 0
        i = 0
        while i < len(proc):
            if proc[i] > self.max_gap_safe_dist:
                s = i
                while i < len(proc) and proc[i] > self.max_gap_safe_dist:
                    i += 1
                if i - s > best_len:
                    best_len, best_s, best_e = i - s, s, i
            i += 1
        return best_s, best_e

    # ========================= Helpers for circle/indices =========================
    def _two_nearest_wpts_to_circle_with_index(self, C_local, R, min_angle_sep_deg=10.0):
        """
        Return (A_world, B_world, idxA, idxB) on ORIGINAL loop.
        """
        best = []
        for k in range(self.numWaypoints):
            w = self.waypoints[k]
            wl = self._map_point_to_base_local(w)
            d = np.linalg.norm(wl - C_local)
            err = abs(d - R)
            ang = np.arctan2(wl[1] - C_local[1], wl[0] - C_local[0])
            best.append((err, k, w, wl, ang))
        best.sort(key=lambda x: x[0])
        if not best:
            return None, None, None, None

        A = best[0]
        A_ang = A[4]
        for cand in best[1:]:
            ang = cand[4]
            d_ang = abs(((ang - A_ang + np.pi) % (2 * np.pi)) - np.pi)
            if d_ang >= np.deg2rad(min_angle_sep_deg):
                return A[2], cand[2], A[1], cand[1]
        return None, None, None, None
        
    @staticmethod
    def _in_forward_span(k, a, b, N):
        """Return True if index k is on forward wrap from a to b inclusive over modulo-N circle."""
        if a <= b:
            return a <= k <= b
        else:
            return k >= a or k <= b


    # ========================= Visualization init =========================
    def visualization_init(self):
        # ACTIVE TRACK points (green)
        self.waypointMarker = Marker()
        self.waypointMarker.header.frame_id = 'map'
        self.waypointMarker.type = Marker.POINTS
        self.waypointMarker.color.g = 0.75
        self.waypointMarker.color.a = 1.0
        self.waypointMarker.scale.x = 0.05
        self.waypointMarker.scale.y = 0.05
        self.waypointMarker.id = 0
        self.waypointMarker.points = [Point(x=wpt[0], y=wpt[1], z=0.0) for wpt in self.track_waypoints]

        # Lookahead target (red)
        self.targetMarker = Marker()
        self.targetMarker.header.frame_id = 'map'
        self.targetMarker.type = Marker.POINTS
        self.targetMarker.color.r = 0.9
        self.targetMarker.color.g = 0.1
        self.targetMarker.color.b = 0.1
        self.targetMarker.color.a = 1.0
        self.targetMarker.scale.x = 0.15
        self.targetMarker.scale.y = 0.15
        self.targetMarker.id = 1

        # Last spliced arc (purple) as POINTS
        self.arcMarker = Marker()
        self.arcMarker.header.frame_id = 'map'
        self.arcMarker.type = Marker.POINTS
        self.arcMarker.color.r = 0.5
        self.arcMarker.color.b = 0.8
        self.arcMarker.color.a = 1.0
        self.arcMarker.scale.x = 0.06
        self.arcMarker.scale.y = 0.06
        self.arcMarker.id = 3

        # Candidate arcs for preview
        self.arcCandLeftMarker = Marker()
        self.arcCandLeftMarker.header.frame_id = 'map'
        self.arcCandLeftMarker.type = Marker.LINE_STRIP
        self.arcCandLeftMarker.color.r = 1.0
        self.arcCandLeftMarker.color.g = 0.5
        self.arcCandLeftMarker.color.b = 0.0
        self.arcCandLeftMarker.color.a = 1.0
        self.arcCandLeftMarker.scale.x = 0.05
        self.arcCandLeftMarker.id = 4

        self.arcCandRightMarker = Marker()
        self.arcCandRightMarker.header.frame_id = 'map'
        self.arcCandRightMarker.type = Marker.LINE_STRIP
        self.arcCandRightMarker.color.r = 1.0
        self.arcCandRightMarker.color.g = 1.0
        self.arcCandRightMarker.color.b = 0.0
        self.arcCandRightMarker.color.a = 1.0
        self.arcCandRightMarker.scale.x = 0.05
        self.arcCandRightMarker.id = 5


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuit()
    print("[INFO] Pure Pursuit with obstacle-arc splicing (local grid) initialized")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

