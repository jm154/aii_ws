#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.duration import Duration
from rclpy.time import Time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import tf2_ros 
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry 
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree
import colorsys
from collections import deque

# ---------------- CONFIG ----------------
TARGET_DT = 0.04  # 10 frames * (1/250Hz) = 0.04s. (Ideal comparison window)
FRAME_SKIP = 10   # 10프레임 전과 비교 (학습 데이터의 시간 창)
MAX_SPEED_LIMIT = 20.0 
# ----------------------------------------

# ==========================================
# 0. Helper Functions
# ==========================================
def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def rotation_matrix(yaw):
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array([[c, -s], [s, c]])

def rotate_world_to_body(vec_world, yaw):
    R = rotation_matrix(yaw)
    if vec_world.ndim == 1:
        return R.T.dot(vec_world)
    else:
        return (R.T @ vec_world.T).T

# ==========================================
# 1. Model Definition
# ==========================================
class ClusterFlowNet(nn.Module):
    def __init__(self):
        super(ClusterFlowNet, self).__init__()
        in_channels = 4 
        self.conv1 = nn.Conv1d(in_channels, 64, 1); self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 1); self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, 1); self.bn3 = nn.BatchNorm1d(256)
        self.ego_mlp = nn.Sequential(
            nn.Linear(4, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU()
        )
        self.fc1 = nn.Linear(640, 256); self.bn4 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128); self.bn5 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 2) 

    def forward_one_branch(self, x):
        x = F.relu(self.bn1(self.conv1(x))); x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x))); x = torch.max(x, 2, keepdim=False)[0] 
        return x

    def forward(self, curr_cluster, prev_patch, ego_vector, raw_ego_vel):
        feat_curr = self.forward_one_branch(curr_cluster)
        feat_prev = self.forward_one_branch(prev_patch)
        feat_ego = self.ego_mlp(ego_vector) 
        combined = torch.cat([feat_curr, feat_prev, feat_ego], dim=1) 
        x = F.relu(self.bn4(self.fc1(combined))); x = F.relu(self.bn5(self.fc2(x)))
        v_obj_pred = self.fc3(x) 
        pred_vel_relative = v_obj_pred - raw_ego_vel
        return pred_vel_relative, v_obj_pred

# ==========================================
# 2. Inference Node
# ==========================================
class MosInferNode(Node):
    def __init__(self):
        super().__init__('mos_infer_node')
        
        self.set_parameters([rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])

        self.declare_parameter('model_path', '/home/ugrp/aii_ws/src/f1tenth-software-stack/mos_datagen/cluster_flow_net.pth')
        self.declare_parameter('scan_topic', '/ego_racecar/scan')
        self.declare_parameter('odom_topic', '/ego_racecar/odom') 

        model_path = self.get_parameter('model_path').value
        scan_topic = self.get_parameter('scan_topic').value
        odom_topic = self.get_parameter('odom_topic').value

        self.get_logger().info(f"[INIT] Model: {model_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_points = 64
        self.dbscan = DBSCAN(eps=0.5, min_samples=3, algorithm='kd_tree')
        self.num_beams = 1080
        self.fov = 4.71238898
        self.angles = np.linspace(-self.fov/2, self.fov/2, self.num_beams)
        self.angles_norm = self.angles / (self.fov/2)

        self.model = ClusterFlowNet().to(self.device)
        self.load_model(model_path)
        
        self.history = deque(maxlen=FRAME_SKIP + 1)  
        self.prev_time_sec = None 
        self.vel_history = deque(maxlen=5) 
        self.latest_odom = None 

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # 마커 발행 주기 제어 변수 (250Hz -> 25Hz)
        self.marker_publish_counter = 0
        self.marker_publish_skip = 10 

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=1)
        
        self.create_subscription(Odometry, odom_topic, self.odom_callback, 10) 
        self.create_subscription(LaserScan, scan_topic, self.scan_callback, qos)
        self.marker_pub = self.create_publisher(MarkerArray, '/mos/markers', 10)

    def load_model(self, path):
        try:
            ckpt = torch.load(path, map_location=self.device)
            state = ckpt.get('model_state_dict', ckpt.get('model_state', ckpt))
            new_state = {k.replace('module.', ''): v for k, v in state.items()}
            self.model.load_state_dict(new_state)
            self.model.eval()
            self.get_logger().info(f"[MODEL] Loaded successfully")
        except Exception as e:
            self.get_logger().error(f"[MODEL] Failed to load: {e}")
            exit(1)

    def get_yaw(self, q):
        return math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))

    def odom_callback(self, msg):
        self.latest_odom = msg

    def polar_to_xy(self, ranges):
        valid = (ranges > 0.01) & (ranges < 30.0)
        x = ranges * np.cos(self.angles)
        y = ranges * np.sin(self.angles)
        return np.stack([x[valid], y[valid]], axis=1), valid

    def normalize_cluster(self, points, center, residuals, angles):
        pts_centered = points - center
        num_pts = len(pts_centered)
        if num_pts >= self.num_points:
            choice = np.random.choice(num_pts, self.num_points, replace=False)
        else:
            choice = np.random.choice(num_pts, self.num_points, replace=True)
        return pts_centered[choice], residuals[choice], angles[choice]

    def compute_residual(self, curr_ranges, prev_ranges, pose_curr, pose_prev):
        c, s = math.cos(pose_curr[2]), math.sin(pose_curr[2])
        H_c = np.array([[c, -s, pose_curr[0]], [s, c, pose_curr[1]], [0, 0, 1]])
        c, s = math.cos(pose_prev[2]), math.sin(pose_prev[2])
        H_p = np.array([[c, -s, pose_prev[0]], [s, c, pose_prev[1]], [0, 0, 1]])
        H_rel = np.linalg.inv(H_c) @ H_p

        valid_p = (prev_ranges > 0.01) & (prev_ranges < 30.0)
        r_p = prev_ranges[valid_p]
        th_p = self.angles[valid_p]
        x_p = r_p * np.cos(th_p)
        y_p = r_p * np.sin(th_p)
        ones = np.ones_like(x_p)
        pts_prev_homo = np.stack([x_p, y_p, ones], axis=0)

        if pts_prev_homo.shape[1] == 0: return np.zeros_like(curr_ranges)

        pts_prev_in_curr = H_rel @ pts_prev_homo
        x_pc = pts_prev_in_curr[0, :]
        y_pc = pts_prev_in_curr[1, :]

        r_warped = np.sqrt(x_pc**2 + y_pc**2)
        th_warped = np.arctan2(y_pc, x_pc)

        pred_ranges = np.full(self.num_beams, np.inf)
        angle_res = self.fov / (self.num_beams - 1)
        idxs = ((th_warped + self.fov/2) / angle_res).round().astype(int)
        valid_proj = (idxs >= 0) & (idxs < self.num_beams)
        
        for i, dist in zip(idxs[valid_proj], r_warped[valid_proj]):
            if dist < pred_ranges[i]:
                pred_ranges[i] = dist
        
        valid_curr = (curr_ranges > 0.01) & (curr_ranges < 30.0)
        residual = np.zeros_like(curr_ranges)
        mask = valid_curr & (pred_ranges != np.inf)
        residual[mask] = np.tanh(np.abs(curr_ranges[mask] - pred_ranges[mask]))
        return residual

    def scan_callback(self, scan_msg):
        current_time = Time.from_msg(scan_msg.header.stamp)
        current_sec = current_time.nanoseconds / 1e9
        
        # ⭐️ 1. Pose 취득 (Odom/Map)
        if self.latest_odom is None: return
        
        # 1a. Velocity Calculation Pose (Stable Odom)
        odom = self.latest_odom
        v_tx = odom.pose.pose.position.x
        v_ty = odom.pose.pose.position.y
        v_yaw = self.get_yaw(odom.pose.pose.orientation)
        curr_pose_v = (v_tx, v_ty, v_yaw) # Stable Pose for Velocity Diff

        # 1b. Map Alignment Pose (TF Lookup for Global Correction)
        try:
            # TF 요청: Time(0)을 사용해 버퍼의 가장 최신 변환을 요청
            trans = self.tf_buffer.lookup_transform(
                'map',  
                scan_msg.header.frame_id, 
                rclpy.time.Time(), 
                timeout=Duration(seconds=0.05)
            )
            map_tx = trans.transform.translation.x
            map_ty = trans.transform.translation.y
            map_yaw = self.get_yaw(trans.transform.rotation)
            curr_pose_map = (map_tx, map_ty, map_yaw)
        except Exception as e:
            # TF 실패 시, Odom Pose를 Map Pose로 임시 사용 (Local 시각화는 영향 없지만, Global 계산 정확도 저하)
            self.get_logger().debug(f"TF Lookup failed (Latest stamp requested): {e}. Using Odom pose for map alignment.")
            curr_pose_map = curr_pose_v

        curr_ranges = np.array(scan_msg.ranges)

        # 2. Points Local -> Global Map (Map-corrected Pose 사용)
        c_map, s_map = math.cos(curr_pose_map[2]), math.sin(curr_pose_map[2])
        curr_pts_local, valid_mask = self.polar_to_xy(curr_ranges)
        x_glob = curr_pts_local[:,0]*c_map - curr_pts_local[:,1]*s_map + curr_pose_map[0]
        y_glob = curr_pts_local[:,0]*s_map + curr_pts_local[:,1]*c_map + curr_pose_map[1]
        curr_points_map = np.stack([x_glob, y_glob], axis=1)

        # ⭐️ History 저장 (250Hz 속도로 Deque 채움)
        self.history.append({
            'ranges': curr_ranges,
            'pose': curr_pose_v,       
            'map_pose': curr_pose_map, 
            'time': current_sec,
            'points_map': curr_points_map 
        })

        # 데이터가 덜 쌓였으면 대기 (10프레임이 찼을 때만 처리)
        if len(self.history) <= FRAME_SKIP: return

        # ⭐️ 10프레임 전 데이터와 비교
        prev_data = self.history[0]  
        prev_pose_v = prev_data['pose']      
        prev_pose_map = prev_data['map_pose'] 
        prev_ranges = prev_data['ranges']
        prev_points_map = prev_data['points_map']
        prev_time = prev_data['time']

        # ⭐️ 속도 역산 (Odom Pose Diff 사용)
        dt = current_sec - prev_time # Actual time difference (Should be ~0.04s)
        # ⚠️ dt 폴백 로직 수정 (0.04s로 설정)
        if dt < 0.001: dt = TARGET_DT
        
        dx_global = curr_pose_v[0] - prev_pose_v[0] 
        dy_global = curr_pose_v[1] - prev_pose_v[1]
        dyaw = wrap_angle(curr_pose_v[2] - prev_pose_v[2])

        vec_world = np.array([dx_global, dy_global])
        vec_local = rotate_world_to_body(vec_world, prev_pose_v[2])

        vx_calc = vec_local[0] / dt
        vy_calc = vec_local[1] / dt
        omega_calc = dyaw / dt

        # 속도 튐 방지
        if abs(vx_calc) > MAX_SPEED_LIMIT:
            if len(self.vel_history) > 0:
                vx_calc = self.vel_history[-1][0]
                vy_calc = self.vel_history[-1][1]
            else:
                vx_calc = 0.0; vy_calc = 0.0

        self.vel_history.append(np.array([vx_calc, vy_calc]))
        avg_v = np.mean(self.vel_history, axis=0)
        vx_final, vy_final = avg_v[0], avg_v[1]

        # Model Inputs
        norm_vx = vx_final / 10.0
        norm_vy = vy_final / 10.0
        ego_vector_np = np.array([norm_vx, norm_vy, omega_calc, dt], dtype=np.float32)
        raw_ego_vel_np = np.array([vx_final, vy_final], dtype=np.float32)

        # Residual (Map Pose Diff 사용)
        residual_full = self.compute_residual(curr_ranges, prev_ranges, curr_pose_map, prev_pose_map)

        # Cluster data preparation
        curr_pts_local, valid_mask = self.polar_to_xy(curr_ranges)
        if len(curr_pts_local) < 10: return
        curr_residuals = residual_full[valid_mask]
        curr_angles_norm = self.angles_norm[valid_mask]

        labels = self.dbscan.fit_predict(curr_pts_local)
        unique_labels = set(labels)
        if -1 in unique_labels: unique_labels.remove(-1)

        if len(unique_labels) == 0: return

        curr_batch_list = []; prev_batch_list = []; cluster_centers = []; valid_label_list = []
        
        prev_tree = KDTree(prev_points_map)

        c, s = math.cos(curr_pose_map[2]), math.sin(curr_pose_map[2])

        for label in unique_labels:
            mask = (labels == label)
            if np.sum(mask) < 5: continue

            cluster_local = curr_pts_local[mask]
            cluster_res = curr_residuals[mask] # FIX: cluster_res 할당
            cluster_ang = curr_angles_norm[mask]
            center_local = np.mean(cluster_local, axis=0)
            
            center_map_x = center_local[0]*c - center_local[1]*s + curr_pose_map[0]
            center_map_y = center_local[0]*s + center_local[1]*c + curr_pose_map[1]
            center_map = np.array([center_map_x, center_map_y])

            prev_patch_local = np.empty((0, 2)); prev_patch_ang = np.empty((0,))
            
            idxs = prev_tree.query_radius([center_map], r=1.5)[0]
            if len(idxs) > 5:
                pts_map_patch = prev_points_map[idxs]
                dx_p = pts_map_patch[:,0] - curr_pose_map[0]
                dy_p = pts_map_patch[:,1] - curr_pose_map[1]
                x_loc = dx_p*c + dy_p*s
                y_loc = -dx_p*s + dy_p*c
                prev_patch_local = np.stack([x_loc, y_loc], axis=1)
                prev_patch_ang = np.arctan2(y_loc, x_loc) / (self.fov/2)

            if len(prev_patch_local) < 5:
                prev_patch_local = cluster_local
                prev_patch_ang = cluster_ang
            
            c_pts, c_res, c_ang = self.normalize_cluster(cluster_local, center_local, cluster_res, cluster_ang)
            curr_feat = np.stack([c_pts[:,0], c_pts[:,1], c_res, c_ang], axis=0)

            p_pts, _, p_ang = self.normalize_cluster(prev_patch_local, np.mean(prev_patch_local, axis=0), 
                                                     np.zeros(len(prev_patch_local)), prev_patch_ang)
            prev_feat = np.stack([p_pts[:,0], p_pts[:,1], np.zeros_like(p_ang), p_ang], axis=0)

            curr_batch_list.append(curr_feat); prev_batch_list.append(prev_feat)
            cluster_centers.append(center_local); valid_label_list.append(label)

        velocities = []
        
        if len(curr_batch_list) > 0:
            curr_tensor = torch.tensor(np.array(curr_batch_list), dtype=torch.float32).to(self.device)
            prev_tensor = torch.tensor(np.array(prev_batch_list), dtype=torch.float32).to(self.device)
            B = len(curr_batch_list)
            ego_tensor = torch.tensor(ego_vector_np, dtype=torch.float32).unsqueeze(0).repeat(B, 1).to(self.device)
            raw_ego_tensor = torch.tensor(raw_ego_vel_np, dtype=torch.float32).unsqueeze(0).repeat(B, 1).to(self.device)

            with torch.no_grad():
                pred_rel, _ = self.model(curr_tensor, prev_tensor, ego_tensor, raw_ego_tensor)
                velocities = pred_rel.cpu().numpy()

        # ⭐️ 마커 발행 주기 제어 로직 (25Hz로 제한)
        self.marker_publish_counter += 1
        if len(curr_batch_list) > 0 and self.marker_publish_counter >= self.marker_publish_skip:
            self.marker_publish_counter = 0
            
            viz_header = scan_msg.header
            viz_header.stamp = self.get_clock().now().to_msg() 
            # ⭐️ 시각화 프레임을 Local (라이다 센서) 프레임으로 설정하여 TF 에러 회피
            viz_header.frame_id = scan_msg.header.frame_id 

            # Local 좌표계 데이터만 전달
            self.publish_markers(viz_header, curr_pts_local, labels, cluster_centers, velocities, valid_label_list)


    def publish_markers(self, header, points, labels, centers, velocities, valid_labels):
        # ⚠️ 이 함수는 Local 좌표계(header.frame_id 기준)로만 마커를 발행합니다.
        markers = MarkerArray()

        # Clusters (Local Viz)
        pts_marker = Marker()
        pts_marker.header = header; pts_marker.ns = "clusters"; pts_marker.id = 0
        pts_marker.type = Marker.POINTS; pts_marker.action = Marker.ADD
        pts_marker.scale.x = 0.05; pts_marker.scale.y = 0.05
        pts_marker.lifetime = Duration(seconds=0.15).to_msg()
        
        for i, pt in enumerate(points):
            # Local 좌표를 그대로 사용
            p = Point(x=float(pt[0]), y=float(pt[1]), z=0.0) 
            label = labels[i]
            c_rgba = ColorRGBA(a=1.0)
            if label == -1: c_rgba.r, c_rgba.g, c_rgba.b = 0.2, 0.2, 0.2
            else:
                hue = (label * 0.618) % 1.0
                r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
                c_rgba.r, c_rgba.g, c_rgba.b = r, g, b
            pts_marker.points.append(p); pts_marker.colors.append(c_rgba)
        markers.markers.append(pts_marker)

        # Velocity Arrows (Local Viz)
        for i, label in enumerate(valid_labels):
            v_rel = velocities[i]
            center = centers[i]
            speed_rel = np.linalg.norm(v_rel)
            if speed_rel < 0.5: continue 
            
            # Local Position과 Local Velocity를 그대로 사용하여 화살표 생성
            cx_l = center[0]
            cy_l = center[1]
            vx_l = v_rel[0]
            vy_l = v_rel[1]

            arrow = Marker()
            arrow.header = header; arrow.ns = "vel_rel"; arrow.id = int(label) + 1000
            arrow.type = Marker.ARROW; arrow.action = Marker.ADD
            arrow.lifetime = Duration(seconds=0.2).to_msg()
            arrow.scale.x = 0.05; arrow.scale.y = 0.1; arrow.scale.z = 0.1
            arrow.color.r = 0.0; arrow.color.g = 1.0; arrow.color.b = 0.0; arrow.color.a = 0.8
            
            start = Point(x=float(cx_l), y=float(cy_l), z=0.2)
            end = Point(x=float(cx_l + vx_l), y=float(cy_l + vy_l), z=0.2)
            arrow.points = [start, end]
            markers.markers.append(arrow)

            # Text Marker (Local Viz)
            text = Marker()
            text.header = header; text.ns = "info"; text.id = int(label) + 2000
            text.type = Marker.TEXT_VIEW_FACING; text.action = Marker.ADD
            text.lifetime = Duration(seconds=0.2).to_msg()
            text.scale.z = 0.3; text.color.r = 1.0; text.color.g = 1.0; text.color.b = 1.0; text.color.a = 1.0
            text.pose.position.x = float(cx_l); text.pose.position.y = float(cy_l); text.pose.position.z = 0.5
            text.text = f"Rel: {speed_rel:.1f}m/s"
            markers.markers.append(text)

        self.marker_pub.publish(markers)

def main():
    rclpy.init()
    node = MosInferNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()
