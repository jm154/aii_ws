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
from visualization_msgs.msg import Marker, MarkerArray ###
from geometry_msgs.msg import Point ###

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
        self.use_ftg = False

        # ===== Local Occupancy Grid Parameters (LiDAR-based) =====
        self.grid_res = 0.05  # [m/cell]
        self.grid_forward = 6.0  # [m]
        self.grid_side = 4.0  # [m]
        self.inflate_radius_m = 0.30  # [m]
        self.inflate_iters = max(1, int(self.inflate_radius_m / self.grid_res))

        self.grid_w = int(self.grid_forward / self.grid_res)
        self.grid_h = int(self.grid_side / self.grid_res)
        self.grid_y_offset = self.grid_h // 2

        # ===== Topics/Publishers =====
        odom_topic = '/pf/viz/inferred_pose' if self.is_real else '/ego_racecar/odom'
        self.sub_pose = self.create_subscription(Odometry, odom_topic, self.pose_callback, 1)

        self.pub_drive = self.create_publisher(AckermannDriveStamped, '/drive', 1)
        self.drive_msg = AckermannDriveStamped()

        # Waypoints
        map_path = os.path.abspath(os.path.join('src', "f1tenth-software-stack", 'csv_data'))
        self.waypoints = np.loadtxt(f"{map_path}/{self.map_name}.csv", delimiter=',', skiprows=1)
        self.numWaypoints = self.waypoints.shape[0]

        # Viz
        ###self.pub_vis = self.create_publisher(MarkerArray, '/visualization_marker_array', 1)
        ###self.visualization_init()
        
        ### Publish to visualization
        visualization_topic = '/visualization_marker_array'
        self.pub_vis = self.create_publisher(MarkerArray, visualization_topic, 1)
        self.markerArray = MarkerArray()

        map_path = os.path.abspath(os.path.join('src', "f1tenth-software-stack", 'csv_data'))
        data = np.loadtxt(f"{map_path}/{self.map_name}.csv", delimiter=',', skiprows=1)

        # 로우가 1개인 경우에도 (1, M)으로 맞춤
        if data.ndim == 1:
            data = data[None, :]

        # 앞의 두 열만 XY로 사용 (나머지 열은 무시)
        self.waypoints = data[:, :2].astype(float)

        self.numWaypoints = self.waypoints.shape[0]
        self.track_waypoints = self.waypoints.copy()          # 항상 (N,2)
        self.numTrackWaypoints = self.track_waypoints.shape[0]
        ###

        # self.ref_speed = csv_data[:, 5] * 0.6  # max speed for levine 2nd - real is 2m/s
        self.ref_speed = 2.0  # csv_data[:, 5]  # max speed - sim is 10m/s

        self.visualization_init()

        # LiDAR
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        # Current state
        self.currX = 0.0
        self.currY = 0.0
        self.rot = np.eye(3)
        self.have_pose = False

        # Latest local grid
        self.local_grid = np.zeros((self.grid_h, self.grid_w), dtype=np.uint8)

    # ========================= Pure Pursuit =========================
    def pose_callback(self, msg):
        self.currX = msg.pose.pose.position.x
        self.currY = msg.pose.pose.position.y
        quat = msg.pose.pose.orientation
        quat = [quat.x, quat.y, quat.z, quat.w]
        self.rot = transform.Rotation.from_quat(quat).as_matrix()
        self.have_pose = True

        if self.use_ftg:
            return

        currPos = np.array([[self.currX, self.currY]])
        dists = distance.cdist(currPos, self.waypoints, 'euclidean').reshape((self.numWaypoints))
        closest_idx = np.argmin(dists)
        target = self._get_target_ahead(closest_idx, dists, self.L)

        t_local = self._map_point_to_base_local(target)
        gi, gj, in_bounds = self._local_point_to_grid(t_local[0], t_local[1])

        if in_bounds and self.local_grid[gj, gi] > 0:  # note: row=gj(y), col=gi(x)
            if not self.use_ftg:
                self.use_ftg = True
                self.get_logger().info("[SWITCH] PP -> FTG (lookahead cell blocked)")
            return

        y = t_local[1]
        gamma = np.clip(self.steering_gain * (2 * y / self.L**2), -0.35, 0.35)

        self.drive_msg.drive.steering_angle = gamma
        self.drive_msg.drive.speed = self.ref_speed
        self.pub_drive.publish(self.drive_msg)
        print(f"[PP] steer={round(gamma, 3)}, speed={self.ref_speed:.2f} target_local=({t_local[0]:.2f},{t_local[1]:.2f}) "
              f"grid_blocked={in_bounds and self.local_grid[gj, gi] > 0}")
              
        ### Visualizing points
        closest_pt = self.waypoints[closest_idx]  # world frame
        self.targetMarker.header.stamp = self.get_clock().now().to_msg()
        self.closestMarker.header.stamp = self.get_clock().now().to_msg()
        self.waypointMarker.header.stamp = self.get_clock().now().to_msg()

        self.targetMarker.points = [Point(x=float(target[0]), y=float(target[1]), z=0.0)]
        self.closestMarker.points = [Point(x=float(closest_pt[0]), y=float(closest_pt[1]), z=0.0)]

        self.markerArray.markers = [self.waypointMarker, self.targetMarker, self.closestMarker]
        self.pub_vis.publish(self.markerArray)

        ###

    # ========================= FTG =========================
    def scan_callback(self, scan_msg: LaserScan):
        self._build_local_grid_from_scan(scan_msg)

        if not self.use_ftg:
            return

        ranges = np.array(scan_msg.ranges[180:900])
        proc = self._preprocess_lidar(ranges)
        proc = self._bubble(proc, radius=2)
        s, e = self._find_max_gap(proc)
        best = (s + e) // 2
        angle = np.deg2rad(best * self.downsample_gap / 4.0 - 90.0)
        speed = self.ref_speed * 0.6 if np.min(proc) < 1.0 else self.ref_speed

        self.drive_msg.drive.steering_angle = angle
        self.drive_msg.drive.speed = speed
        self.pub_drive.publish(self.drive_msg)
        print(f"[FTG] steer={round(angle, 3)}, speed={speed:.2f}, gap=({s},{e}), min_proc={np.min(proc):.2f}")

        self.use_ftg = False

    # ========================= Local Grid Helpers =========================
    def _build_local_grid_from_scan(self, scan: LaserScan):
        if not self.have_pose:
            return

        grid = np.zeros((self.grid_h, self.grid_w), dtype=np.uint8)

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
                grid[gj, gi] = 1  # hit

            angle += scan.angle_increment * idx_step

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

    # ========================= PP Helpers =========================
    def _get_target_ahead(self, closest_idx, dists, lookahead):
        idx = closest_idx
        while dists[idx] < lookahead:
            idx = (idx + 1) % self.numWaypoints
        return self.waypoints[idx]

    # ========================= FTG Helpers =========================
    def _preprocess_lidar(self, ranges):
        proc = np.zeros(len(ranges) // self.downsample_gap)
        for i in range(len(proc)):
            proc[i] = np.mean(ranges[i * self.downsample_gap:(i + 1) * self.downsample_gap])
        return np.clip(proc, 0.0, self.max_sight)

    def _bubble(self, proc, radius=2):
        k = int(np.argmin(proc))
        s = max(k - radius, 0); e = min(k + radius, len(proc) - 1)
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

    # ========================= Viz (Optional) =========================
    def visualization_init(self):
        # Green
        self.waypointMarker = Marker()
        self.waypointMarker.header.frame_id = 'map'
        self.waypointMarker.type = Marker.POINTS
        self.waypointMarker.color.g = 0.75
        self.waypointMarker.color.a = 1.0
        self.waypointMarker.scale.x = 0.05
        self.waypointMarker.scale.y = 0.05
        self.waypointMarker.id = 0
        self.waypointMarker.points = [Point(x=wpt[0], y=wpt[1], z=0.0) for wpt in self.waypoints]

        # Red
        self.targetMarker = Marker()
        self.targetMarker.header.frame_id = 'map'
        self.targetMarker.type = Marker.POINTS
        self.targetMarker.color.r = 0.75
        self.targetMarker.color.a = 1.0
        self.targetMarker.scale.x = 0.2
        self.targetMarker.scale.y = 0.2
        self.targetMarker.id = 1

        # Blue
        self.closestMarker = Marker()
        self.closestMarker.header.frame_id = 'map'
        self.closestMarker.type = Marker.POINTS
        self.closestMarker.color.b = 0.75
        self.closestMarker.color.a = 1.0
        self.closestMarker.scale.x = 0.2
        self.closestMarker.scale.y = 0.2
        self.closestMarker.id = 2


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuit()
    print("[INFO] Pure Pursuit Node with FTG (local grid) initialized")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
