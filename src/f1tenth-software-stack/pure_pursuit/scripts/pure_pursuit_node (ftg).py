#!/usr/bin/env python3

import numpy as np
from scipy.spatial import distance, transform
import os

import rclpy
from rclpy.node import Node
from rclpy.time import Time, Duration
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PointStamped, QuaternionStamped, TransformStamped, PoseStamped
import tf2_ros


from sensor_msgs.msg import LaserScan


class PurePursuit(Node):
    """
    Implement Pure Pursuit on the car
    """
    def __init__(self):
        super().__init__('pure_pursuit_node')

        self.is_real = False
        self.is_ascending = True  # waypoint indices are ascending during tracking
        self.map_name = 'E1_out2_refined'
        # Topics & Subs, Pubs
        drive_topic = '/drive'
        odom_topic = '/pf/viz/inferred_pose' if self.is_real else '/ego_racecar/odom'
        visualization_topic = '/visualization_marker_array'

        # Subscribe to POSE
        self.sub_pose = self.create_subscription(PoseStamped if self.is_real else Odometry, odom_topic, self.pose_callback, 1)
        # Publish to drive
        self.pub_drive = self.create_publisher(AckermannDriveStamped, drive_topic, 1)
        self.drive_msg = AckermannDriveStamped()
        # Publish to visualization
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

        # self.ref_speed = csv_data[:, 5] * 0.6  # max speed for levine 2nd - real is 2m/s
        self.ref_speed = 2.0  # csv_data[:, 5]  # max speed - sim is 10m/s

        self.visualization_init()

        # sim params
        self.L = 1.0
        self.steering_gain = 0.5

        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.lidar_ranges = None
        self.use_ftg = False
        self.downsample_gap = 10
        self.max_sight = 4.0
        self.max_gap_safe_dist = 1.2

    def pose_callback(self, pose_msg):
        self.currX = msg.pose.pose.position.x
        self.currY = msg.pose.pose.position.y
        quat = msg.pose.pose.orientation
        quat = [quat.x, quat.y, quat.z, quat.w]
        self.rot = transform.Rotation.from_quat(quat).as_matrix()
        self.have_pose = True
        if self.use_ftg:
            return
            
        # Get current pose
        self.currX = pose_msg.pose.position.x if self.is_real else pose_msg.pose.pose.position.x
        self.currY = pose_msg.pose.position.y if self.is_real else pose_msg.pose.pose.position.y
        self.currPos = np.array([self.currX, self.currY]).reshape((1, 2))

        # Transform quaternion pose message to rotation matrix
        quat = pose_msg.pose.orientation if self.is_real else pose_msg.pose.pose.orientation
        quat = [quat.x, quat.y, quat.z, quat.w]
        R = transform.Rotation.from_quat(quat)
        self.rot = R.as_matrix()

        # Find closest waypoint to where we are
        self.distances = distance.cdist(self.currPos, self.waypoints, 'euclidean').reshape((self.numWaypoints))
        self.closest_index = np.argmin(self.distances)
        self.closestPoint = self.waypoints[self.closest_index]

        # Find target point
        targetPoint = self.get_closest_point_beyond_lookahead_dist(self.L)

        # Homogeneous transformation
        translatedTargetPoint = self.translatePoint(targetPoint)

        # calculate curvature/steering angle
        y = translatedTargetPoint[1]
        gamma = self.steering_gain * (2 * y / self.L**2)

        # publish drive message, don't forget to limit the steering angle.
        gamma = np.clip(gamma, -0.35, 0.35)
        self.drive_msg.drive.steering_angle = gamma
        self.drive_msg.drive.speed = (-1.0 if self.is_real else 1.0) * self.ref_speed
        self.pub_drive.publish(self.drive_msg)
        print("steering = {}, speed = {}".format(round(self.drive_msg.drive.steering_angle, 2), round(self.drive_msg.drive.speed, 2)))

        # Visualizing points
        self.targetMarker.points = [Point(x=targetPoint[0], y=targetPoint[1], z=0.0)]
        
        self.markerArray.markers = [self.waypointMarker, self.targetMarker]
        self.pub_vis.publish(self.markerArray)

    def get_closest_point_beyond_lookahead_dist(self, threshold):
        point_index = self.closest_index
        dist = self.distances[point_index]

        while dist < threshold:
            if self.is_ascending:
                point_index += 1
                if point_index >= len(self.waypoints):
                    point_index = 0
                dist = self.distances[point_index]
            else:
                point_index -= 1
                if point_index < 0:
                    point_index = len(self.waypoints) - 1
                dist = self.distances[point_index]

        point = self.waypoints[point_index]
        return point

    def translatePoint(self, targetPoint):
        H = np.zeros((4, 4))
        H[0:3, 0:3] = np.linalg.inv(self.rot)
        H[0, 3] = self.currX
        H[1, 3] = self.currY
        H[3, 3] = 1.0
        pvect = targetPoint - self.currPos
        convertedTarget = (H @ np.array((pvect[0, 0], pvect[0, 1], 0, 0))).reshape((4))
        return convertedTarget

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

    def scan_callback(self, scan_msg):
        ranges = np.array(scan_msg.ranges[180:899])
        proc_ranges = self.preprocess_lidar(ranges)
        proc_ranges = self.bubble_danger_zone(proc_ranges)

        if np.min(proc_ranges[15:45]) < 1.0:
            self.use_ftg = True
            self.do_ftg_control(proc_ranges)
        else:
            self.use_ftg = False

    def preprocess_lidar(self, ranges):
        proc_ranges = np.zeros(int(720 / self.downsample_gap))
        for i in range(len(proc_ranges)):
            proc_ranges[i] = np.mean(ranges[i * self.downsample_gap:(i + 1) * self.downsample_gap])
        proc_ranges = np.clip(proc_ranges, 0.0, self.max_sight)
        return proc_ranges

    def bubble_danger_zone(self, proc_ranges, radius=2):
        min_idx = np.argmin(proc_ranges)
        start = max(min_idx - radius, 0)
        end = min(min_idx + radius, len(proc_ranges) - 1)
        proc_ranges[start:end + 1] = 0.0
        return proc_ranges

    def find_max_gap(self, ranges):
        longest = 0
        start = end = 0
        curr = 0
        while curr < len(ranges):
            if ranges[curr] > self.max_gap_safe_dist:
                s = curr
                while curr < len(ranges) and ranges[curr] > self.max_gap_safe_dist:
                    curr += 1
                if curr - s > longest:
                    longest = curr - s
                    start, end = s, curr
            curr += 1
        return start, end

    def find_best_point(self, start, end, ranges):
        return int((start + end) / 2)

    def do_ftg_control(self, proc_ranges):
        start, end = self.find_max_gap(proc_ranges)
        best_i = self.find_best_point(start, end, proc_ranges)
        steering_angle = np.deg2rad(best_i * self.downsample_gap / 4.0 - 90.0)

        if np.min(proc_ranges) < 1.0:
            velocity = self.ref_speed * 0.6
        else:
            velocity = self.ref_speed

        self.drive_msg.drive.steering_angle = steering_angle
        self.drive_msg.drive.speed = velocity
        self.pub_drive.publish(self.drive_msg)

        print(f"[FTG] steer={round(steering_angle, 2)}, speed={velocity}")


def main(args=None):
    rclpy.init(args=args)
    print("PurePursuit Initialized")
    pure_pursuit_node = PurePursuit()
    rclpy.spin(pure_pursuit_node)
    pure_pursuit_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
