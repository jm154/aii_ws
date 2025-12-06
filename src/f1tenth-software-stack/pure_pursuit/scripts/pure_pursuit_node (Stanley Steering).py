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
from geometry_msgs.msg import Point

class PurePursuit(Node):
    def __init__(self):
        super().__init__('pure_pursuit_node')

        # ===== Parameters =====
        self.is_real = False
        self.map_name = 'pnu'
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

        # Viz (publisher + marker array)
        visualization_topic = '/visualization_marker_array'
        self.pub_vis = self.create_publisher(MarkerArray, visualization_topic, 1)
        self.markerArray = MarkerArray()

        # (duplicate load kept to avoid touching other functionality in user's code)
        map_path = os.path.abspath(os.path.join('src', "f1tenth-software-stack", 'csv_data'))
        csv_data = np.loadtxt(map_path + '/' + self.map_name + '.csv', delimiter=',', skiprows=1)
        self.waypoints = csv_data[:, :]
        self.numWaypoints = self.waypoints.shape[0]

        self.ref_speed = 2.0

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

    # ========================= Pose / Stanley =========================
    def pose_callback(self, msg):
        self.currX = msg.pose.pose.position.x
        self.currY = msg.pose.pose.position.y
        quat = msg.pose.pose.orientation
        quat = [quat.x, quat.y, quat.z, quat.w]
        self.rot = transform.Rotation.from_quat(quat).as_matrix()
        self.have_pose = True

        if self.use_ftg:
            return

        # closest waypoint in world
        currPos = np.array([[self.currX, self.currY]])
        dists = distance.cdist(currPos, self.waypoints, 'euclidean').reshape((self.numWaypoints))
        closest_idx = np.argmin(dists)
        closest_pt = self.waypoints[closest_idx]

        # next waypoint (for path heading) — lightweight preview instead of PP lookahead
        next_idx = (closest_idx + 1) % self.numWaypoints
        next_pt = self.waypoints[next_idx]

        # map path heading (world)
        path_vec = next_pt - closest_pt
        path_heading = np.arctan2(path_vec[1], path_vec[0])

        # vehicle yaw from quaternion
        yaw = transform.Rotation.from_quat(quat).as_euler('xyz', degrees=False)[2]
        heading_error = self._wrap_to_pi(path_heading - yaw)

        # cross-track error (closest point in base_link)
        closest_local = self._map_point_to_base_local(closest_pt)
        e_y = closest_local[1]

        # Stanley control
        K_e = self.steering_gain
        v = max(self.ref_speed, 1e-3)
        gamma = heading_error + np.arctan2(K_e * e_y, v)
        gamma = np.clip(gamma, -0.35, 0.35)

        # obstacle gate using local grid at the (former) PP look point — keep logic as-is
        # (map the "next" point to base to check grid; lightweight safety)
        t_local = self._map_point_to_base_local(next_pt)
        gi, gj, in_bounds = self._local_point_to_grid(t_local[0], t_local[1])
        if in_bounds and self.local_grid[gj, gi] > 0:
            if not self.use_ftg:
                self.use_ftg = True
                self.get_logger().info("[SWITCH] Stanley -> FTG (forward cell blocked)")
            return

        # publish drive
        self.drive_msg.drive.steering_angle = gamma
        self.drive_msg.drive.speed = self.ref_speed
        self.pub_drive.publish(self.drive_msg)
        print(f"[Stanley] steer={round(gamma, 3)}, speed={self.ref_speed:.2f} closest_local=({closest_local[0]:.2f},{closest_local[1]:.2f}) "+
              f"grid_blocked={in_bounds and self.local_grid[gj, gi] > 0}")

        # ===== Visualization (waypoints + closest point only) =====
        self.closestMarker.header.stamp = self.get_clock().now().to_msg()
        self.waypointMarker.header.stamp = self.get_clock().now().to_msg()
        self.closestMarker.points = [Point(x=float(closest_pt[0]), y=float(closest_pt[1]), z=0.0)]
        self.markerArray.markers = [self.waypointMarker, self.closestMarker]
        self.pub_vis.publish(self.markerArray)

    # ========================= Angle Helper =========================
    def _wrap_to_pi(self, ang):
        # wrap angle to [-pi, pi]
        return (ang + np.pi) % (2 * np.pi) - np.pi

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
        speed = 1.0 if np.min(proc) < 1.0 else 2.0

        self.drive_msg.drive.steering_angle = angle
        self.drive_msg.drive.speed = speed
        self.pub_drive.publish(self.drive_msg)
        print(f"[FTG] steer={round(angle, 3)}, speed={speed:.2f}, gap=({s},{e}), min_proc={np.min(proc):.2f}")

        self.use_ftg = False

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
        # Waypoints (green)
        self.waypointMarker = Marker()
        self.waypointMarker.header.frame_id = 'map'
        self.waypointMarker.type = Marker.POINTS
        self.waypointMarker.color.g = 0.75
        self.waypointMarker.color.a = 1.0
        self.waypointMarker.scale.x = 0.05
        self.waypointMarker.scale.y = 0.05
        self.waypointMarker.id = 0
        self.waypointMarker.points = [Point(x=wpt[0], y=wpt[1], z=0.0) for wpt in self.waypoints]

        # Closest point (blue)
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
    print("[INFO] Stanley Node with FTG (local grid) initialized")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

