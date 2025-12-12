import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import sys
import math
import random
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree
from matplotlib.lines import Line2D

# --- 설정 ---
DATA_DIR = os.path.expanduser("./dataset_l/5ms") # 경로 확인
MODEL_PATH = "cluster_flow_net.pth"
FRAME_SKIP = 10
NUM_POINTS = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Definition (속도 전용) ---
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
        self.fc3 = nn.Linear(128, 2) # Only Velocity

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
        return pred_vel_relative

# --- Helper Functions ---
def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def rotate_world_to_body(vec_world, yaw):
    c = math.cos(yaw); s = math.sin(yaw)
    x = vec_world[0] * c + vec_world[1] * s
    y = -vec_world[0] * s + vec_world[1] * c
    return np.array([x, y])

class OfflineTester:
    def __init__(self):
        # 1. Model Load
        self.model = ClusterFlowNet().to(DEVICE)
        if os.path.exists(MODEL_PATH):
            state = torch.load(MODEL_PATH, map_location=DEVICE)
            # Remove 'module.' prefix if needed
            new_state = {k.replace('module.', ''): v for k, v in state.items()}
            # Remove unexpected keys (like segmentation head) for compatibility
            filtered_state = {k: v for k, v in new_state.items() if k in self.model.state_dict()}
            
            self.model.load_state_dict(filtered_state, strict=False)
            self.model.eval()
            print(f"✅ Model loaded: {MODEL_PATH}")
        else:
            print(f"❌ Model not found: {MODEL_PATH}")
            sys.exit()

        # 2. Data Load
        self.files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
        if not self.files:
            print(f"❌ No data found in {DATA_DIR}")
            sys.exit()
        
        print(f"📂 Found {len(self.files)} files.")
        self.dbscan = DBSCAN(eps=0.5, min_samples=3)
        self.fov = 4.71238898
        self.num_beams = 1080
        self.angles = np.linspace(-self.fov/2, self.fov/2, self.num_beams)
        self.angles_norm = self.angles / (self.fov/2)

        # 초기 설정
        self.file_idx = 0
        self.frame_idx = 10
        self.load_file()

        # 3. GUI Setup
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        self.run_inference()
        plt.show()

    def load_file(self):
        self.data = np.load(self.files[self.file_idx])
        self.total_frames = len(self.data['ranges'])
        self.filename = os.path.basename(self.files[self.file_idx])
        print(f"📄 Loading File [{self.file_idx+1}/{len(self.files)}]: {self.filename} ({self.total_frames} frames)")

    def on_key(self, event):
        if event.key == 'right':
            self.frame_idx = min(self.frame_idx + 1, self.total_frames - 1)
            self.run_inference()
        elif event.key == 'left':
            self.frame_idx = max(self.frame_idx - 1, 10)
            self.run_inference()
        elif event.key == 'down':
            self.file_idx = (self.file_idx + 1) % len(self.files)
            self.frame_idx = 10
            self.load_file()
            self.run_inference()
        elif event.key == 'up':
            self.file_idx = (self.file_idx - 1) % len(self.files)
            self.frame_idx = 10
            self.load_file()
            self.run_inference()
        elif event.key == 'q':
            plt.close()

    def polar_to_xy(self, ranges):
        valid = (ranges > 0.01) & (ranges < 30.0)
        x = ranges[valid] * np.cos(self.angles[valid])
        y = ranges[valid] * np.sin(self.angles[valid])
        return np.stack([x, y], axis=1), valid

    def normalize_cluster(self, points, center, residuals, angles):
        pts = points - center
        num = len(pts)
        if num >= NUM_POINTS:
            idx = np.random.choice(num, NUM_POINTS, replace=False)
        else:
            idx = np.random.choice(num, NUM_POINTS, replace=True)
        return pts[idx], residuals[idx], angles[idx]

    def run_inference(self):
        t = self.frame_idx
        t_prev = t - 10
        
        # Data Extraction
        ranges = self.data['ranges'][t]
        prev_ranges = self.data['ranges'][t_prev]
        pose_curr = self.data['ego_pose'][t]
        pose_prev = self.data['ego_pose'][t_prev]
        
        # GT (비교용)
        labels_pt = self.data['labels'][t]
        vel_pt = self.data['point_velocities'][t]

        # Ego Motion Calc
        if 'timestamps' in self.data:
            dt = self.data['timestamps'][t] - self.data['timestamps'][t_prev]
            if dt < 0.001: dt = 0.4
        else:
            dt = 0.4
            
        dx_g = pose_curr[0] - pose_prev[0]
        dy_g = pose_curr[1] - pose_prev[1]
        dyaw = wrap_angle(pose_curr[2] - pose_prev[2])
        
        vec_world = np.array([dx_g, dy_g])
        vec_local = rotate_world_to_body(vec_world, pose_prev[2])
        
        vx = vec_local[0] / dt
        vy = vec_local[1] / dt
        wz = dyaw / dt
        
        ego_vec = np.array([vx/10.0, vy/10.0, wz, dt], dtype=np.float32)
        raw_ego = np.array([vx, vy], dtype=np.float32)

        # Preprocessing
        valid = (ranges > 0.01) & (ranges < 30.0)
        x = ranges[valid] * np.cos(self.angles[valid])
        y = ranges[valid] * np.sin(self.angles[valid])
        points_xy = np.stack([x, y], axis=1)
        angles_valid = self.angles_norm[valid]
        residuals = np.zeros(len(points_xy))

        # Clustering
        if len(points_xy) < 5: return
        labels = self.dbscan.fit_predict(points_xy)
        unique_labels = set(labels) - {-1}

        curr_list, prev_list = [], []
        centers = []
        indices_list = []
        cluster_gt_labels = []

        # GT Labels for valid points
        labels_valid = labels_pt[valid]
        
        # 유효한 포인트의 GT 속도
        vel_valid = vel_pt[valid]

        for l in unique_labels:
            mask = (labels == l)
            cluster = points_xy[mask]
            center = np.mean(cluster, axis=0)
            
            # 클러스터의 대표 GT 라벨 (Static/Dynamic/New)
            cls_votes = labels_valid[mask]
            if len(cls_votes) > 0:
                cluster_gt_labels.append(np.argmax(np.bincount(cls_votes)))
            else:
                cluster_gt_labels.append(0)

            # Normalize
            pts_norm, res_norm, ang_norm = self._normalize(cluster, center, residuals[mask], angles_valid[mask])
            
            curr_feat = np.stack([pts_norm[:,0], pts_norm[:,1], res_norm, ang_norm], axis=0)
            prev_feat = curr_feat.copy(); prev_feat[2,:] = 0
            
            curr_list.append(curr_feat)
            prev_list.append(prev_feat)
            centers.append(center)
            indices_list.append(mask)

        # Inference
        pred_vels = []

        if len(curr_list) > 0:
            b_curr = torch.tensor(np.array(curr_list), dtype=torch.float32).to(DEVICE)
            b_prev = torch.tensor(np.array(prev_list), dtype=torch.float32).to(DEVICE)
            b_ego = torch.tensor(ego_vec, dtype=torch.float32).unsqueeze(0).repeat(len(curr_list), 1).to(DEVICE)
            b_raw = torch.tensor(raw_ego, dtype=torch.float32).unsqueeze(0).repeat(len(curr_list), 1).to(DEVICE)
            
            with torch.no_grad():
                pred_vel = self.model(b_curr, b_prev, b_ego, b_raw)
                pred_vels = pred_vel.cpu().numpy()

        # Visualization
        self.plot_result(points_xy, labels, vel_valid, centers, pred_vels, cluster_gt_labels, indices_list, vx, vy)

    def _normalize(self, points, center, residuals, angles):
        pts = points - center
        num = len(pts)
        if num >= NUM_POINTS:
            idx = np.random.choice(num, NUM_POINTS, replace=False)
        else:
            idx = np.random.choice(num, NUM_POINTS, replace=True)
        return pts[idx], residuals[idx], angles[idx]

    def plot_result(self, points, labels, gt_vel, centers, pred_vels, gt_labels, indices_list, evx, evy):
        self.ax.clear()
        
        # 1. 배경 노이즈 (검정)
        clustered_mask = np.zeros(len(points), dtype=bool)
        for m in indices_list: clustered_mask |= m
        self.ax.scatter(points[~clustered_mask, 0], points[~clustered_mask, 1], s=1, c='black', alpha=0.1, label='Noise')

        # 2. 클러스터별 그리기
        for i, (center, vel, gt_cls) in enumerate(zip(centers, pred_vels, gt_labels)):
            mask = indices_list[i]
            pts = points[mask]
            
            # 클러스터 점 (파란색으로 통일 - 예측 클래스 없으므로)
            self.ax.scatter(pts[:,0], pts[:,1], s=10, c='blue', alpha=0.3)

            # --- 화살표 그리기 (속도 0.5 이상만) ---
            speed = np.linalg.norm(vel)
            if speed > 0.5:
                # 🟢 Pred Vel
                self.ax.quiver(center[0], center[1], vel[0], vel[1], color='green', 
                               angles='xy', scale_units='xy', scale=1, width=0.015, alpha=0.9, 
                               label='Pred Vel' if i==0 else "")
                # P:값
                self.ax.text(center[0], center[1]+0.5, f"P:{speed:.1f}", color='green', fontsize=9, fontweight='bold')
            
            # 🟡 GT Vel (비교용)
            # New(2)가 아닌 경우에만 GT가 존재함
            if gt_cls != 2:
                gt_v_cluster = np.mean(gt_vel[mask], axis=0)
                speed_gt = np.linalg.norm(gt_v_cluster)
                self.ax.quiver(center[0], center[1], gt_v_cluster[0], gt_v_cluster[1], color='orange', 
                               angles='xy', scale_units='xy', scale=1, width=0.01, alpha=0.6, 
                               label='GT Vel' if i==0 else "")
                # G:값
                self.ax.text(center[0], center[1]-0.5, f"G:{speed_gt:.1f}", color='orange', fontsize=9)
            
            # GT 클래스 표시 (참고용)
            cls_map = {0:'S', 1:'D', 2:'N'}
            self.ax.text(center[0]-1.0, center[1], f"GT:{cls_map[gt_cls]}", color='black', fontsize=8)

        self.ax.set_title(f"Velocity Test (Ego: {evx:.2f} m/s)")
        self.ax.plot(0, 0, 'k^', markersize=15, label='Ego')
        self.ax.grid(True)
        self.ax.axis('equal')
        self.ax.set_xlim(-15, 15)
        self.ax.set_ylim(-15, 15)
        
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', label='Cluster', markersize=10),
            Line2D([0], [0], color='green', lw=2, label='Pred Vel'),
            Line2D([0], [0], color='orange', lw=2, label='GT Vel'),
        ]
        self.ax.legend(handles=legend_elements, loc='upper right')
        self.fig.canvas.draw()

if __name__ == "__main__":
    tester = OfflineTester()
    tester.run_random_test()
