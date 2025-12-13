import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree
import math
import random

# ==========================================
# 0. Configuration (설정)
# ==========================================
DATA_PATH = 'dataset_l/2.53ms/data_0020.npz'  # 경로 확인 필요
MODEL_PATH = 'cluster_flow_net.pth'

# ⭐️ 프레임 간격 설정 (몇 프레임 전 데이터와 비교할지) ⭐️
FRAME_SKIP = 10  

# 기본 DT (혹시 타임스탬프가 없을 때 대비용)
DT_DEFAULT = 0.04 

# LiDAR Sensor Specs
FOV = 4.71238898
NUM_BEAMS = 1080
RANGE_LIMIT = 30.0

# ==========================================
# 1. Helper Functions
# ==========================================
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

def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

# ==========================================
# 2. Model Definition
# ==========================================
class ClusterFlowNet(nn.Module):
    def __init__(self):
        super(ClusterFlowNet, self).__init__()
        self.conv1 = nn.Conv1d(4, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.bn3 = nn.BatchNorm1d(256)

        self.ego_mlp = nn.Sequential(
            nn.Linear(4, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU()
        )

        self.fc1 = nn.Linear(640, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 2)

    def forward_one_branch(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=False)[0]
        return x

    def forward(self, curr_cluster, prev_patch, ego_vector, raw_ego_vel):
        feat_curr = self.forward_one_branch(curr_cluster)
        feat_prev = self.forward_one_branch(prev_patch)
        feat_ego = self.ego_mlp(ego_vector)
        combined = torch.cat([feat_curr, feat_prev, feat_ego], dim=1)
        x = F.relu(self.bn4(self.fc1(combined)))
        x = F.relu(self.bn5(self.fc2(x)))
        
        v_obj_pred = self.fc3(x)
        pred_vel_relative = v_obj_pred - raw_ego_vel
        return pred_vel_relative, v_obj_pred

# ==========================================
# 3. Offline Tester Class
# ==========================================
class MOSOfflineTester:
    def __init__(self, data_path, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading data from {data_path}...")
        
        data = np.load(data_path)
        self.ranges = data['ranges']
        raw_poses = data['ego_pose']
        self.times = data['timestamps']
        
        # Pose 전처리 (7D -> 3D 변환 포함)
        if raw_poses.shape[1] == 7:
            self.poses = self.convert_pose_7d_to_3d(raw_poses)
        else:
            self.poses = raw_poses

        self.model = ClusterFlowNet().to(self.device)
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            new_state = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.model.load_state_dict(new_state)
            self.model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            exit()

        self.angles = np.linspace(-FOV/2, FOV/2, NUM_BEAMS)
        self.angles_norm = self.angles / (FOV/2)
        self.dbscan = DBSCAN(eps=0.1, min_samples=3)

    def convert_pose_7d_to_3d(self, poses_7d):
        x = poses_7d[:, 0]
        y = poses_7d[:, 1]
        qx, qy, qz, qw = poses_7d[:, 3], poses_7d[:, 4], poses_7d[:, 5], poses_7d[:, 6]
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return np.stack([x, y, yaw], axis=1)

    def polar_to_xy(self, ranges):
        valid = (ranges > 0.01) & (ranges < RANGE_LIMIT)
        x = ranges * np.cos(self.angles)
        y = ranges * np.sin(self.angles)
        return np.stack([x[valid], y[valid]], axis=1), valid, ranges[valid], self.angles_norm[valid]

    def normalize_cluster(self, points, center, residuals, angles):
        pts_centered = points - center
        num_pts = len(pts_centered)
        target_num = 64
        if num_pts >= target_num:
            choice = np.random.choice(num_pts, target_num, replace=False)
        else:
            choice = np.random.choice(num_pts, target_num, replace=True)
        return pts_centered[choice], residuals[choice], angles[choice]

    def run_random_test(self):
        # 1. 프레임 선택 (FRAME_SKIP 만큼의 여유 필요)
        if len(self.ranges) <= FRAME_SKIP:
            print("Not enough frames in data.")
            return
            
        idx = random.randint(FRAME_SKIP, len(self.ranges) - 1)
        prev_idx = idx - FRAME_SKIP # 10프레임 전
        
        print(f"\n--- Testing Frame Pair: {prev_idx} -> {idx} (Gap: {FRAME_SKIP}) ---")

        # 2. 데이터 추출
        ranges_curr = self.ranges[idx]
        ranges_prev = self.ranges[prev_idx] # 10프레임 전 데이터 사용
        
        p_curr = self.poses[idx]
        p_prev = self.poses[prev_idx]
        
        t_curr = self.times[idx]
        t_prev = self.times[prev_idx]

        # =========================================================
        # ⚡️ [핵심] Pose 변화량으로 직접 속도 계산 (DT 고정)
        # =========================================================
        dt = t_curr - t_prev
        
        # 타임스탬프가 비정상이면 기본값 * 프레임 수로 추정
        if dt <= 1e-6: 
            dt = DT_DEFAULT * FRAME_SKIP
            print(f"⚠️ Warning: dt is too small, using default {DT_DEFAULT} * {FRAME_SKIP} = {dt}s")

        # Global Frame 이동량
        dx_global = p_curr[0] - p_prev[0]
        dy_global = p_curr[1] - p_prev[1]
        dyaw = wrap_angle(p_curr[2] - p_prev[2])

        # Rotation Matrix (World -> Prev Body Frame)
        vec_world = np.array([dx_global, dy_global])
        vec_local = rotate_world_to_body(vec_world, p_prev[2])
        
        dx_local = vec_local[0]
        dy_local = vec_local[1]

        # 속도 계산 (Local Frame)
        vx_calc = dx_local / dt
        vy_calc = dy_local / dt
        omega_calc = dyaw / dt

        print(f"[Calculated Info] dt: {dt:.5f}s")
        print(f"[Calculated Info] Ego Velocity: vx={vx_calc:.3f}, vy={vy_calc:.3f}, w={omega_calc:.3f}")
        
        # 속도 튀는 경우 보정
        if abs(vx_calc) > 15.0: 
             print(f"⚠️ Velocity extremely high. Forcing DT based on {DT_DEFAULT}")
             dt = DT_DEFAULT * FRAME_SKIP
             vx_calc = dx_local / dt
             vy_calc = dy_local / dt
             omega_calc = dyaw / dt
             print(f"[Re-Calculated] Ego Velocity: vx={vx_calc:.3f}, vy={vy_calc:.3f}")

        # =========================================================

        # 3. 모델 입력 및 잔차 계산 준비
        c_c, s_c = math.cos(p_curr[2]), math.sin(p_curr[2])
        H_c = np.array([[c_c, -s_c, p_curr[0]], [s_c, c_c, p_curr[1]], [0, 0, 1]])
        c_p, s_p = math.cos(p_prev[2]), math.sin(p_prev[2])
        H_p = np.array([[c_p, -s_p, p_prev[0]], [s_p, c_p, p_prev[1]], [0, 0, 1]])
        H_rel = np.linalg.inv(H_c) @ H_p

        # 이전 포인트 클라우드 투영 (Warping)
        pts_prev_local, _, _, _ = self.polar_to_xy(ranges_prev)
        # Local to Global
        pts_prev_glob_x = pts_prev_local[:,0]*c_p - pts_prev_local[:,1]*s_p + p_prev[0]
        pts_prev_glob_y = pts_prev_local[:,0]*s_p + pts_prev_local[:,1]*c_p + p_prev[1]
        pts_prev_map = np.stack([pts_prev_glob_x, pts_prev_glob_y], axis=1)

        ones = np.ones((len(pts_prev_local), 1))
        pts_prev_homo = np.hstack([pts_prev_local, ones])
        pts_prev_in_curr = (H_rel @ pts_prev_homo.T).T
        
        r_warped = np.sqrt(pts_prev_in_curr[:,0]**2 + pts_prev_in_curr[:,1]**2)
        th_warped = np.arctan2(pts_prev_in_curr[:,1], pts_prev_in_curr[:,0])
        
        pred_ranges = np.full(NUM_BEAMS, np.inf)
        angle_res = FOV / (NUM_BEAMS - 1)
        beam_idxs = ((th_warped + FOV/2) / angle_res).round().astype(int)
        valid_proj = (beam_idxs >= 0) & (beam_idxs < NUM_BEAMS)
        
        for i, dist in zip(beam_idxs[valid_proj], r_warped[valid_proj]):
            if dist < pred_ranges[i]: pred_ranges[i] = dist
            
        residual_full = np.zeros_like(ranges_curr)
        mask = (ranges_curr > 0.01) & (pred_ranges != np.inf)
        residual_full[mask] = np.tanh(np.abs(ranges_curr[mask] - pred_ranges[mask]))

        # 4. 클러스터링 및 배치 생성
        pts_curr_local, valid_curr, _, ang_curr = self.polar_to_xy(ranges_curr)
        residuals = residual_full[valid_curr]
        labels = self.dbscan.fit_predict(pts_curr_local)
        unique_labels = set(labels) - {-1}

        curr_batch, prev_batch, centers = [], [], []
        prev_tree = KDTree(pts_prev_map)
        
        pts_curr_glob_x = pts_curr_local[:,0]*c_c - pts_curr_local[:,1]*s_c + p_curr[0]
        pts_curr_glob_y = pts_curr_local[:,0]*s_c + pts_curr_local[:,1]*c_c + p_curr[1]
        pts_curr_map = np.stack([pts_curr_glob_x, pts_curr_glob_y], axis=1)

        for label in unique_labels:
            mask = (labels == label)
            cluster_pts = pts_curr_local[mask]
            
            center_local = np.mean(cluster_pts, axis=0)
            center_map = np.mean(pts_curr_map[mask], axis=0)
            
            idxs = prev_tree.query_radius([center_map], r=1.5)[0]
            if len(idxs) > 5:
                patch_map = pts_prev_map[idxs]
                dx_p = patch_map[:,0] - p_curr[0]
                dy_p = patch_map[:,1] - p_curr[1]
                x_loc = dx_p*c_c + dy_p*s_c
                y_loc = -dx_p*s_c + dy_p*c_c
                patch_local = np.stack([x_loc, y_loc], axis=1)
                patch_ang = np.arctan2(y_loc, x_loc) / (FOV/2)
            else:
                patch_local = cluster_pts
                patch_ang = ang_curr[mask]

            c_p, c_r, c_a = self.normalize_cluster(cluster_pts, center_local, residuals[mask], ang_curr[mask])
            curr_feat = np.stack([c_p[:,0], c_p[:,1], c_r, c_a], axis=0)
            p_p, _, p_a = self.normalize_cluster(patch_local, np.mean(patch_local, axis=0), np.zeros(len(patch_local)), patch_ang)
            prev_feat = np.stack([p_p[:,0], p_p[:,1], np.zeros_like(p_a), p_a], axis=0)
            
            curr_batch.append(curr_feat)
            prev_batch.append(prev_feat)
            centers.append(center_local)

        if not curr_batch:
            print("No valid clusters found.")
            return

        # 5. 모델 추론
        curr_tensor = torch.tensor(np.array(curr_batch), dtype=torch.float32).to(self.device)
        prev_tensor = torch.tensor(np.array(prev_batch), dtype=torch.float32).to(self.device)
        
        # [입력 벡터 구성]
        ego_vec = np.array([vx_calc/10.0, vy_calc/10.0, omega_calc, dt], dtype=np.float32)
        ego_tensor = torch.tensor(ego_vec).unsqueeze(0).repeat(len(curr_batch), 1).to(self.device)
        
        raw_ego = torch.tensor(np.array([vx_calc, vy_calc]), dtype=torch.float32).unsqueeze(0).repeat(len(curr_batch), 1).to(self.device)
        
        with torch.no_grad():
            rel_vel, abs_vel = self.model(curr_tensor, prev_tensor, ego_tensor, raw_ego)
            
        rel_vel = rel_vel.cpu().numpy()
        abs_vel = abs_vel.cpu().numpy()

        self.visualize_result(pts_curr_local, labels, centers, rel_vel, abs_vel, vx_calc, vy_calc)

    def visualize_result(self, points, labels, centers, rel_vels, abs_vels, ego_vx, ego_vy):
        plt.figure(figsize=(10, 10))
        plt.title(f"MOS Verification (Ego: vx={ego_vx:.2f})")
        
        unique_labels = set(labels) - {-1}
        for lbl in unique_labels:
            mask = (labels == lbl)
            plt.scatter(points[mask, 0], points[mask, 1], s=5, alpha=0.5)
        
        plt.scatter(points[labels == -1, 0], points[labels == -1, 1], s=1, c='gray', alpha=0.1)

        # Ego Arrow (Red)
        plt.arrow(0, 0, ego_vx, ego_vy, color='red', width=0.08, head_width=0.2, label='Ego Vel')

        # Object Arrows (Green: Relative Only)
        for i, (center, v_rel) in enumerate(zip(centers, rel_vels)):
            # Green: Relative (Model Output)
            plt.arrow(center[0], center[1], v_rel[0], v_rel[1], 
                      color='green', width=0.04, head_width=0.15, alpha=0.8, label='Relative' if i==0 else "")
            plt.text(center[0], center[1], f"Rel: ({v_rel[0]:.1f}, {v_rel[1]:.1f})", color='black', fontsize=8)

        plt.xlabel("Robot X (m)")
        plt.ylabel("Robot Y (m)")
        plt.grid(True)
        plt.axis('equal')
        plt.legend(loc='upper right')
        plt.plot(0, 0, 'k^', markersize=10)
        plt.show()

if __name__ == "__main__":
    tester = MOSOfflineTester(DATA_PATH, MODEL_PATH)
    tester.run_random_test()
