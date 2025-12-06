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
from geometry_msgs.msg import Point, PoseStamped


class PurePursuit(Node):
    def __init__(self):
        super().__init__('pure_pursuit_node')

        # ===== Parameters =====
        self.is_real = False
        self.map_name = 'E1_out2_refined'
        self.L = 1.0
        self.steering_gain = 0.5
        self.downsample_gap = 10
        self.max_sight = 4.0
        self.max_gap_safe_dist = 1.2

        # ===== Vehicle / Obstacle / Arc =====
        self.vehicle_width = 0.27
        self.safety_margin_grid = 0.15   # for grid dilation (collision check)
        self.safety_margin_arc = 0.5     # extra clearance on arcs
        self.obs_search_radius = 0.5
        self.arc_step = 0.10
        self.arc_speed_scale = 0.6
        self.is_arc_mask = np.zeros(0, dtype=bool)  # same length as track_waypoints

        # Smooth bridging between arc and original waypoints
        self.join_gap_thresh = 0.15
        self.bridge_max_sweep_deg = 60.0

        # Active arcs registry (index-based restore, supports multiple arcs)
        # Each item: {'start_i','end_i','idxA_orig','idxB_orig','entered'}
        self.active_arcs = []

        # ===== Local Occupancy Grid =====
        self.grid_res = 0.05
        self.grid_forward = 6
        self.grid_side = 4

        self.grid_w = int(self.grid_forward / self.grid_res)
        self.grid_h = int(self.grid_side / self.grid_res)
        self.grid_y_offset = self.grid_h // 2

        # Extra distance beyond L for collision lookahead
        self.collision_check_extra = 1.0

        # Dilation buffer (cells) for vehicle width + margin
        self.grid_vehicle_buffer_m = 0.5 * self.vehicle_width + self.safety_margin_grid
        self.grid_vehicle_buffer_cells = max(1, int(np.ceil(self.grid_vehicle_buffer_m / self.grid_res)))

        # ===== Topics / Publishers =====
        odom_topic = '/pf/viz/inferred_pose' if self.is_real else '/ego_racecar/odom'
        self.sub_pose = self.create_subscription(PoseStamped if self.is_real else Odometry, odom_topic, self.pose_callback, 1)

        self.pub_drive = self.create_publisher(AckermannDriveStamped, '/drive', 1)
        self.drive_msg = AckermannDriveStamped()

        # ===== Waypoints (ORIGINAL loop) =====
        map_path = os.path.abspath(os.path.join('src', "f1tenth-software-stack", 'csv_data'))
        data = np.loadtxt(f"{map_path}/{self.map_name}.csv", delimiter=';', skiprows=1)
        if data.ndim == 1:
            data = data[None, :]
        self.waypoints = data[:, 1:3].astype(float)

        # optional ref-speed column (safe)
        if data.shape[1] > 5:
            self.ref_speed_array = data[:, 5].astype(float)
        else:
            # 없으면 빈 배열로 두고 기본 속도를 사용
            self.ref_speed_array = np.array([], dtype=float)

        # fallback base speed (m/s) — CSV에 없을 때 사용
        self.ref_speed_default = 6.0

        # ACTIVE TRACK
        self.numWaypoints = self.waypoints.shape[0]
        self.track_waypoints = self.waypoints.copy()
        self.numTrackWaypoints = self.track_waypoints.shape[0]
        self.is_arc_mask = np.zeros(self.numTrackWaypoints, dtype=bool)

        # ===== Visualization =====
        visualization_topic = '/visualization_marker_array'
        self.pub_vis = self.create_publisher(MarkerArray, visualization_topic, 1)
        self.markerArray = MarkerArray()

        self.visualization_init()

        # ===== LiDAR =====
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        # ===== Current state =====
        self.currX = 0.0
        self.currY = 0.0
        self.rot = np.eye(3)
        self.have_pose = False

        # ===== Grids =====
        self.local_grid = np.zeros((self.grid_h, self.grid_w), dtype=np.uint8)   # 0/100 occupancy
        self.occ_bubbled = np.zeros((self.grid_h, self.grid_w), dtype=np.uint8)  # 0/100 dilated

        # Raw LiDAR points in local frame
        self.last_hits = np.zeros((0, 2), dtype=np.float32)

        # Arc visualization
        self.cand_left_world = None
        self.cand_right_world = None
        self.spliced_arc_world = None

        # FTG one-shot flag
        self.run_ftg_once = False

    # ========================= Scan / Grid =========================
    def scan_callback(self, scan_msg: LaserScan):
        """Build local occupancy and dilated grid. If FTG one-shot is scheduled, run it here."""
        self._build_local_grid_from_scan(scan_msg)

    def _build_local_grid_from_scan(self, scan: LaserScan):
        """Convert LaserScan to local occupancy grid and its dilated version."""
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

        # raw occupancy (0/100)
        self.local_grid = (grid > 0).astype(np.uint8) * 100

        # dilated occupancy (0/100)
        occ = (grid > 0).astype(np.uint8)
        occ_b = binary_dilation(occ, iterations=self.grid_vehicle_buffer_cells).astype(np.uint8)
        self.occ_bubbled = occ_b * 100

        # keep raw hits
        self.last_hits = np.array(hits, dtype=np.float32) if hits else np.zeros((0, 2), dtype=np.float32)

        # run FTG once if requested
        if self.run_ftg_once:
            self._run_ftg_once(scan)

    # ========================= Pose / PP =========================
    def pose_callback(self, msg):
        # Update pose
        if self.is_real:
            self.currX = msg.pose.position.x
            self.currY = msg.pose.position.y
            quat = msg.pose.orientation
        else:
            self.currX = msg.pose.pose.position.x
            self.currY = msg.pose.pose.position.y
            quat = msg.pose.pose.orientation
        quat = [quat.x, quat.y, quat.z, quat.w]
        self.rot = transform.Rotation.from_quat(quat).as_matrix()
        self.have_pose = True

        # Ensure ACTIVE TRACK has enough points
        if self.numTrackWaypoints < 3:
            self.track_waypoints = self.waypoints.copy()
            self.numTrackWaypoints = self.track_waypoints.shape[0]
            self.is_arc_mask = np.zeros(self.numTrackWaypoints, dtype=bool)
            self.active_arcs = []

        # Find closest waypoint on ACTIVE TRACK
        currPos = np.array([[self.currX, self.currY]])
        dists = distance.cdist(currPos, self.track_waypoints, 'euclidean').reshape((self.numTrackWaypoints))
        closest_idx = int(np.argmin(dists))

        # Index-based arc restoration (unconditional after passing)
        self._update_and_restore_arcs_by_index(closest_idx)

        # Skip PP update if FTG one-shot is scheduled
        if self.run_ftg_once:
            return

        # Lookahead target on ACTIVE TRACK
        target = self._get_target_ahead_track(closest_idx, self.L)
        if target is None:
            return
        t_local = self._map_point_to_base_local(target)

        # Speed scaling if target lies on an arc
        target_idx = int(np.argmin(np.linalg.norm(self.track_waypoints - target, axis=1)))
        base_speed = self._speed_at_track_index(closest_idx)
        if len(self.is_arc_mask) == self.numTrackWaypoints and self.is_arc_mask[target_idx]:
            speed_cmd = base_speed * self.arc_speed_scale
        else:
            speed_cmd = base_speed


        # Collision prediction within L + extra
        check_len = float(self.L) + float(self.collision_check_extra)
        coll_idx = self._first_colliding_waypoint_idx_track_by_arclength(closest_idx, check_len)

        # Plan and splice if blocked
        planned = False
        on_arc = False
        if coll_idx is not None:
            coll_wpt = self.track_waypoints[coll_idx]
            coll_local = self._map_point_to_base_local(coll_wpt)
            on_arc = (len(self.is_arc_mask) == self.numTrackWaypoints and self.is_arc_mask[coll_idx])
            planned = self._plan_and_splice_arc_from_collision(coll_local, prefer_active=on_arc)

            if planned:
                # Recompute closest and target on the NEW ACTIVE TRACK
                dists = distance.cdist(currPos, self.track_waypoints, 'euclidean').reshape((self.numTrackWaypoints))
                closest_idx = int(np.argmin(dists))
                target = self._get_target_ahead_track(closest_idx, self.L)
                if target is None:
                    return
                t_local = self._map_point_to_base_local(target)
                self.get_logger().info(f"[SPLICE] Replaced forward segment by ARC (active={on_arc})")
            else:
                # Both arcs collided → run FTG once
                self.run_ftg_once = True
                return

        # Pure Pursuit control
        y = t_local[1]
        gamma = np.clip(self.steering_gain * (2.0 * y / (self.L ** 2)), -0.35, 0.35)

        self.drive_msg.drive.steering_angle = gamma
        self.drive_msg.drive.speed = float(speed_cmd)
        self.pub_drive.publish(self.drive_msg)

        print(f"[PP-TRACK] steer={round(gamma, 3)}, speed={float(speed_cmd):.2f}, "
              f"target_local=({t_local[0]:.2f},{t_local[1]:.2f})")

        # Visualization
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

        self._publish_candidate_arcs()
        markers.extend([self.arcCandLeftMarker, self.arcCandRightMarker])

        self.markerArray.markers = markers
        self.pub_vis.publish(self.markerArray)
    

    # ========================= Arc Planning + Splicing =========================
    def _plan_and_splice_arc_from_collision(self, coll_local, prefer_active=False):
        """
        Build two candidate arcs (CCW/CW) around an estimated obstacle and splice the best feasible one.
        If prefer_active=True, pick endpoints on ACTIVE TRACK; else pick on ORIGINAL loop.
        """
        C, obs_radius = self._estimate_obstacle_near_local_point(coll_local)
        if C is None:
            self.get_logger().warn("[ARC] obstacle estimation failed near collision point")
            return False
        Rclr = obs_radius + 0.5 * self.vehicle_width + self.safety_margin_arc

        # Endpoints near the circle
        if prefer_active:
            A_w, B_w, idxA, idxB = self._two_nearest_wpts_to_circle_on_active_with_index(C, Rclr)
        else:
            A_w, B_w, idxA, idxB = self._two_nearest_wpts_to_circle_with_index(C, Rclr)
        if A_w is None or B_w is None:
            self.get_logger().warn("[ARC] need two nearest waypoints but failed")
            return False

        A_l = self._map_point_to_base_local(A_w)
        B_l = self._map_point_to_base_local(B_w)

        # CCW/CW arcs
        ccw_pts, cw_pts, _, _ = self._arcs_from_two_wpts(C, Rclr, A_l, B_l)
        if (ccw_pts is None) or (cw_pts is None):
            self.get_logger().warn("[ARC] failed to build CCW/CW arcs")
            return False

        (left_name, left_local), (right_name, right_local) = self._classify_left_right_by_chord(
            A_l, B_l, ccw_pts, cw_pts
        )

        # 후보 아크 미리보기 저장
        try:
            self.cand_left_world  = self._local_path_to_world(left_local)   if left_local  is not None else None
            self.cand_right_world = self._local_path_to_world(right_local)  if right_local is not None else None
        except Exception as e:
            self.get_logger().warn(f"[ARC-VIZ] candidate arcs to world failed: {e}")
            self.cand_left_world, self.cand_right_world = None, None

        # --- 선택 로직: '웨이포인트 개수' 기준, 동수면 오른쪽 ---
        feasible = []
        if self._path_collision_free_local_buffered(left_local):
            feasible.append(("left", left_local, len(left_local)))
        if self._path_collision_free_local_buffered(right_local):
            feasible.append(("right", right_local, len(right_local)))

        if not feasible:
            self.get_logger().warn("[ARC] both arcs collide; will run FTG once")
            return False

        if len(feasible) == 2:
            # 개수 비교 (적은 쪽), 동수면 'right'
            n_left  = feasible[0][2] if feasible[0][0] == "left"  else feasible[1][2]
            n_right = feasible[0][2] if feasible[0][0] == "right" else feasible[1][2]
            if n_left < n_right:
                best = ("left",  left_local,  n_left)
            elif n_right < n_left:
                best = ("right", right_local, n_right)
            else:
                best = ("right", right_local, n_right)  # tie → 오른쪽
        else:
            best = feasible[0]

        best_name, best_local, best_npts = best
        arc_world = self._local_path_to_world(best_local)
        Cw = self._world_from_local_point(C)

        if prefer_active:
            # splice into ACTIVE track
            self._splice_arc_into_active_track(idxA, idxB, Cw, Rclr, arc_world)
            # register the just-inserted arc range for index-based restore
            ia = int(np.argmin(np.linalg.norm(self.waypoints - arc_world[0], axis=1)))
            ib = int(np.argmin(np.linalg.norm(self.waypoints - arc_world[-1], axis=1)))
            self._register_spliced_arc(Cw, Rclr, ia, ib)
        else:
            # splice into ORIGINAL loop copy (also gets registered)
            self._splice_arc_into_track(idxA, idxB, Cw, Rclr, arc_world)

        self.spliced_arc_world = arc_world
        self.get_logger().info(
            f"[ARC] Spliced {best_name} arc (npts={best_npts}) "
            f"{'(ACTIVE)' if prefer_active else '(ORIGINAL)'} between idx {idxA}->{idxB}"
        )
        return True

    # ---------- Circle projection & arc-bridging ----------
    @staticmethod
    def _project_to_circle(C, R, p):
        v = p - C
        n = np.linalg.norm(v)
        if n < 1e-9:
            return np.array([C[0] + R, C[1]], dtype=float)
        return C + R * (v / n)

    def _bridge_arc_points(self, Cw, R, p_from, p_to, gap_thresh=None, max_sweep_deg=None):
        """Add short arc samples on the same circle to smoothly connect two points."""
        gap_thresh = self.join_gap_thresh if gap_thresh is None else gap_thresh
        max_sweep_deg = self.bridge_max_sweep_deg if max_sweep_deg is None else max_sweep_deg
        if np.linalg.norm(p_to - p_from) <= gap_thresh:
            return []

        a = self._project_to_circle(Cw, R, p_from)
        b = self._project_to_circle(Cw, R, p_to)

        def ang(p): return np.arctan2(p[1] - Cw[1], p[0] - Cw[0])

        a0 = ang(a)
        a1 = ang(b)
        d = (a1 - a0 + np.pi) % (2*np.pi) - np.pi   # shortest signed angle
        sweep = min(abs(d), np.deg2rad(max_sweep_deg))
        ccw = d > 0.0

        n = max(1, int((sweep * R) / max(self.arc_step, 1e-4)))
        if ccw:
            angs = np.linspace(a0, a0 + sweep, n + 1)[1:]
        else:
            angs = np.linspace(a0, a0 - sweep, n + 1)[1:]
        pts = np.column_stack([Cw[0] + R*np.cos(angs), Cw[1] + R*np.sin(angs)])
        return [np.array([float(x), float(y)]) for x, y in pts]

    # ---------- Splicing into ORIGINAL loop copy ----------
    def _splice_arc_into_track(self, idxA_orig, idxB_orig, Cw, R, arc_world):
        """Splice an arc into ORIGINAL loop copy → active track, remove points inside circle."""
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

        new_track, new_mask = [], []
        inserted_arc = False
        post_arc_bridge_done = False

        for k in range(N):
            if k == idxA_orig and not inserted_arc:
                # pre-bridge
                if len(new_track) > 0:
                    pre = self._bridge_arc_points(Cw, R, new_track[-1], arc_world[0])
                    for p in pre:
                        new_track.append(p); new_mask.append(True)
                # arc samples
                for p in arc_world:
                    new_track.append(np.array([float(p[0]), float(p[1])]))
                    new_mask.append(True)
                inserted_arc = True
                continue

            # skip A->B forward span
            if self._in_forward_span(k, idxA_orig, idxB_orig, N):
                continue

            # keep points outside circle
            w = self.waypoints[k]
            if np.linalg.norm(w - Cw) >= R:
                # post-bridge once
                if inserted_arc and not post_arc_bridge_done and len(new_track) > 0:
                    post = self._bridge_arc_points(Cw, R, new_track[-1], w)
                    for p in post:
                        new_track.append(p); new_mask.append(True)
                    post_arc_bridge_done = True
                new_track.append(w)
                new_mask.append(False if len(self.is_arc_mask) != N else bool(self.is_arc_mask[k]))

        if len(new_track) < 3:
            self.get_logger().warn("[SPLICE] new track too short; reverting to original")
            self.track_waypoints = self.waypoints.copy()
            self.numTrackWaypoints = self.track_waypoints.shape[0]
            self.is_arc_mask = np.zeros(self.numTrackWaypoints, dtype=bool)
            self.active_arcs = []
        else:
            self.track_waypoints = np.array(new_track, dtype=float)
            self.numTrackWaypoints = self.track_waypoints.shape[0]
            self.is_arc_mask = np.array(new_mask, dtype=bool)
            # register inserted arc range (find run on new mask)
            self._register_spliced_arc_after_mask_build()

    # ---------- Splicing into ACTIVE track ----------
    def _two_nearest_wpts_to_circle_on_active_with_index(self, C_local, R, min_angle_sep_deg=10.0):
        best = []
        for k in range(self.numTrackWaypoints):
            w = self.track_waypoints[k]
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

    def _splice_arc_into_active_track(self, idxA_trk, idxB_trk, Cw, R, arc_world):
        """Replace forward span A->B on ACTIVE track by arc samples; keep mask continuity."""
        N = self.numTrackWaypoints
        if N < 3:
            self.get_logger().warn("[SPLICE-ACT] track too short")
            return

        fw_len = (idxB_trk - idxA_trk) % N
        bw_len = (idxA_trk - idxB_trk) % N
        if bw_len < fw_len:
            idxA_trk, idxB_trk = idxB_trk, idxA_trk
            arc_world = arc_world[::-1]
            fw_len = bw_len
        if fw_len == 0:
            self.get_logger().warn("[SPLICE-ACT] A and B identical; skip")
            return

        new_track, new_mask = [], []
        inserted_arc = False
        post_arc_bridge_done = False

        for k in range(N):
            if k == idxA_trk and not inserted_arc:
                if len(new_track) > 0:
                    pre = self._bridge_arc_points(Cw, R, new_track[-1], arc_world[0])
                    for p in pre:
                        new_track.append(p); new_mask.append(True)
                for p in arc_world:
                    new_track.append(np.array([float(p[0]), float(p[1])]))
                    new_mask.append(True)
                inserted_arc = True
                continue

            if self._in_forward_span(k, idxA_trk, idxB_trk, N):
                continue

            w = self.track_waypoints[k]
            if np.linalg.norm(w - Cw) >= R:
                if inserted_arc and not post_arc_bridge_done and len(new_track) > 0:
                    post = self._bridge_arc_points(Cw, R, new_track[-1], w)
                    for p in post:
                        new_track.append(p); new_mask.append(True)
                    post_arc_bridge_done = True
                new_track.append(w)
                if len(self.is_arc_mask) == N:
                    new_mask.append(bool(self.is_arc_mask[k]))
                else:
                    new_mask.append(False)

        if len(new_track) < 3:
            self.get_logger().warn("[SPLICE-ACT] new track too short; revert")
            return
        else:
            self.track_waypoints = np.array(new_track, dtype=float)
            self.numTrackWaypoints = self.track_waypoints.shape[0]
            self.is_arc_mask = np.array(new_mask, dtype=bool)

    # ---------- Arc registry & index-based restore ----------
    def _register_spliced_arc(self, Cw, R, idxA_orig, idxB_orig):
        """Find the current arc run (True-run in is_arc_mask) nearest to Cw and register."""
        runs = self._collect_true_runs(self.is_arc_mask)
        if not runs:
            return
        best = None
        best_d = 1e9
        for (s, e) in runs:
            mid = (s + e) // 2
            d = np.linalg.norm(self.track_waypoints[mid] - Cw)
            if d < best_d:
                best_d = d
                best = (s, e)
        if best is None:
            return
        s, e = best
        self.active_arcs.append({
            'start_i': int(s),
            'end_i': int(e),
            'idxA_orig': int(idxA_orig),
            'idxB_orig': int(idxB_orig),
            'entered': False
        })

    def _register_spliced_arc_after_mask_build(self):
        """Used when splicing from ORIGINAL loop copy: pick any True-run (likely just created)."""
        runs = self._collect_true_runs(self.is_arc_mask)
        if not runs:
            return
        # take the longest True-run
        lengths = [self._forward_steps(s, e, self.numTrackWaypoints) + 1 for (s, e) in runs]
        j = int(np.argmax(lengths))
        s, e = runs[j]
        # estimate nearest original indices by endpoints matching
        ia = int(np.argmin(np.linalg.norm(self.waypoints - self.track_waypoints[s], axis=1)))
        ib = int(np.argmin(np.linalg.norm(self.waypoints - self.track_waypoints[e], axis=1)))
        self.active_arcs.append({
            'start_i': int(s),
            'end_i': int(e),
            'idxA_orig': int(ia),
            'idxB_orig': int(ib),
            'entered': False
        })

    def _update_and_restore_arcs_by_index(self, closest_idx):
        """Unconditionally restore original waypoints once we have passed an arc (index-based).
        Supports multiple arcs by iteratively restoring one-by-one if needed."""
        if not self.active_arcs:
            return

        changed = True
        while changed:
            changed = False
            # work on a snapshot; after any restore, break to recompute
            for i, arc in enumerate(list(self.active_arcs)):
                s = arc['start_i']
                e = arc['end_i']
                N = self.numTrackWaypoints
                inside = self._index_in_run(closest_idx, s, e, N)
                if inside:
                    arc['entered'] = True
                    continue
                if not arc['entered']:
                    continue
                # exited after entering: passed end?
                dist_to_curr = self._forward_steps(s, closest_idx, N)
                dist_to_end = self._forward_steps(s, e, N)
                if dist_to_curr >= dist_to_end:
                    # restore this arc segment
                    self._restore_arc_segment_by_indices(s, e, arc['idxA_orig'], arc['idxB_orig'])
                    # clear and rebuild registry from mask (remaining arcs still True-runs)
                    self.active_arcs.clear()
                    # re-register any remaining arcs (mask still has them)
                    for s2, e2 in self._collect_true_runs(self.is_arc_mask):
                        ia = int(np.argmin(np.linalg.norm(self.waypoints - self.track_waypoints[s2], axis=1)))
                        ib = int(np.argmin(np.linalg.norm(self.waypoints - self.track_waypoints[e2], axis=1)))
                        self.active_arcs.append({
                            'start_i': int(s2),
                            'end_i': int(e2),
                            'idxA_orig': int(ia),
                            'idxB_orig': int(ib),
                            'entered': False if not (s2 <= closest_idx <= e2) else True
                        })
                    changed = True
                    break  # track mutated; recompute next loop

    def _restore_arc_segment_by_indices(self, s, e, idxA_orig, idxB_orig):
        """Replace track[s..e] (True-run) with original waypoints [idxA_orig..idxB_orig] forward."""
        Nw = self.numWaypoints
        # build original segment forward from A to B (modulo loop)
        orig_seg = []
        i = idxA_orig
        while True:
            orig_seg.append(self.waypoints[i])
            if i == idxB_orig:
                break
            i = (i + 1) % Nw
        orig_seg = np.array(orig_seg, dtype=float)

        Nt = self.numTrackWaypoints
        new_track, new_mask = [], []
        for idx in range(Nt):
            in_arc = self._index_in_run(idx, s, e, Nt)
            if in_arc:
                if idx == s:
                    for p in orig_seg:
                        new_track.append(np.array([float(p[0]), float(p[1])]))
                        new_mask.append(False)
                continue
            new_track.append(self.track_waypoints[idx])
            new_mask.append(False)

        if len(new_track) >= 3:
            self.track_waypoints = np.array(new_track, dtype=float)
            self.numTrackWaypoints = len(new_track)
            self.is_arc_mask = np.array(new_mask, dtype=bool)
            

    # ========= utilities for runs/indices =========
    @staticmethod
    def _collect_true_runs(mask_bool):
        """Collect contiguous True runs (closed intervals) on a linearized array."""
        runs = []
        N = len(mask_bool)
        i = 0
        while i < N:
            if mask_bool[i]:
                s = i
                while i + 1 < N and mask_bool[i + 1]:
                    i += 1
                e = i
                runs.append((s, e))
            i += 1
        return runs

    @staticmethod
    def _forward_steps(a, b, N):
        """Forward modular distance a→b in steps [0..N-1]."""
        return (b - a) % N

    @staticmethod
    def _index_in_run(i, s, e, N):
        """Check if i is inside closed run [s..e] on a circular index set size N."""
        return (s <= e and s <= i <= e) or (s > e and (i >= s or i <= e))

    # ========================= Target on ACTIVE TRACK =========================
    def _get_target_ahead_track(self, closest_idx, lookahead):
        """Return the first waypoint whose cumulative arc-length from closest_idx exceeds L."""
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
        return self.track_waypoints[(closest_idx + 1) % self.numTrackWaypoints]

    def _first_colliding_waypoint_idx_track_by_arclength(self, start_idx, max_len_m):
        """Walk along ACTIVE TRACK up to max_len_m and return first colliding waypoint index."""
        i = start_idx
        acc = 0.0
        for _ in range(self.numTrackWaypoints):
            wpt = self.track_waypoints[i]
            p_local = self._map_point_to_base_local(wpt)
            gi, gj, inb = self._local_point_to_grid(p_local[0], p_local[1])
            if inb and self.occ_bubbled[gj, gi] > 0:
                return i
            a = self.track_waypoints[i]
            b = self.track_waypoints[(i + 1) % self.numTrackWaypoints]
            acc += float(np.linalg.norm(b - a))
            if acc > max_len_m:
                break
            i = (i + 1) % self.numTrackWaypoints
        return None
        
        # ========================= Speed lookup =========================
    def _speed_at_track_index(self, track_idx: int) -> float:
        """
        Map the ACTIVE-track index to the nearest ORIGINAL waypoint index
        and return its reference speed. If the CSV had no speed column,
        return a sane default (self.ref_speed_default).
        """
        # 기본값: CSV에 속도 정보가 없거나 비정상인 경우
        if not hasattr(self, "ref_speed_array") or self.ref_speed_array.size == 0:
            return float(getattr(self, "ref_speed_default", 6.0))

        # 안전 가드
        if self.numTrackWaypoints == 0 or self.numWaypoints == 0:
            return float(getattr(self, "ref_speed_default", 6.0))

        # 인덱스 클램프
        track_idx = int(np.clip(track_idx, 0, self.numTrackWaypoints - 1))

        # ACTIVE 포인트 → 원본 웨이포인트 최근접 인덱스
        p = self.track_waypoints[track_idx]  # (2,)
        k = int(np.argmin(np.linalg.norm(self.waypoints - p[None, :], axis=1)))

        # 경계/길이 방어
        k = int(np.clip(k, 0, len(self.ref_speed_array) - 1))

        v = float(self.ref_speed_array[k])
        if not np.isfinite(v) or v <= 0.0:
            # 이상치면 기본값
            return float(getattr(self, "ref_speed_default", 6.0))
        return v

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
                xs.append(x); ys.append(y)
        if len(xs) < 5:
            return None, None
        pts = np.vstack([xs, ys]).T
        c = np.mean(pts, axis=0)
        r = np.max(np.linalg.norm(pts - c, axis=1)) + 0.5 * self.grid_res
        r = max(r, 0.10)
        return c, r

    # ========================= Collision helpers =========================
    def _path_collision_free_local_buffered(self, pts_local, step=None, out_of_bounds_blocks=True):
        """Sample a polyline against dilated occupancy grid."""
        if pts_local is None or len(pts_local) == 0:
            return False
        step = step or (0.5 * self.grid_res)
        prev = pts_local[0]
        if not self._point_free_in_bubbled(prev[0], prev[1], out_of_bounds_blocks):
            return False
        for i in range(1, len(pts_local)):
            cur = pts_local[i]
            seg = cur - prev
            seg_len = float(np.linalg.norm(seg))
            if seg_len <= 1e-6:
                prev = cur; continue
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

        self.arcCandLeftMarker.header.stamp = now
        if self.cand_left_world is not None and len(self.cand_left_world) > 1:
            self.arcCandLeftMarker.points = [Point(x=float(p[0]), y=float(p[1]), z=0.0) for p in self.cand_left_world]
        else:
            self.arcCandLeftMarker.points = []

        self.arcCandRightMarker.header.stamp = now
        if self.cand_right_world is not None and len(self.cand_right_world) > 1:
            self.arcCandRightMarker.points = [Point(x=float(p[0]), y=float(p[1]), z=0.0) for p in self.cand_right_world]
        else:
            self.arcCandRightMarker.points = []

    # ========================= Arc Geometry =========================
    def _arcs_from_two_wpts(self, C, R, A_local, B_local):
        def ang(p): return np.arctan2(p[1] - C[1], p[0] - C[0])
        a0 = ang(A_local); a1 = ang(B_local)
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

    # ========================= Helpers for circle / indices =========================
    def _two_nearest_wpts_to_circle_with_index(self, C_local, R, min_angle_sep_deg=10.0):
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
        """True if index k is on forward wrap from a to b inclusive on modulo-N circle."""
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

        # Candidate arcs (preview)
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
    print("[INFO] Pure Pursuit with index-based arc restore initialized")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

