#!/usr/bin/env python3
import os
import numpy as np
from torch.utils.data import Dataset
import torch
import warnings
import math
import glob

# ⭐️ [설정] 몇 프레임 전 데이터와 비교할지 설정
FRAME_SKIP = 10 

# --- Helper Functions ---
def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def rotate_world_to_body(vec_world, yaw):
    c = np.cos(yaw)
    s = np.sin(yaw)
    # Rotation Matrix Transpose: [[c, s], [-s, c]]
    x = vec_world[0] * c + vec_world[1] * s
    y = -vec_world[0] * s + vec_world[1] * c
    return np.array([x, y])

# ✅ [추가] Twist 적분 함수
def integrate_twist(pose_prev, twist, dt):
    """Twist를 이용하여 Pose를 한 스텝 적분합니다 (오일러 적분)."""
    x, y, yaw = pose_prev
    vx, vy, w = twist
    
    # Body Frame 속도를 World Frame으로 변환
    dx_world = vx * np.cos(yaw) - vy * np.sin(yaw)
    dy_world = vx * np.sin(yaw) + vy * np.cos(yaw)
    
    # 위치 및 자세 업데이트
    x_new = x + dx_world * dt
    y_new = y + dy_world * dt
    yaw_new = wrap_angle(yaw + w * dt)
    
    return np.array([x_new, y_new, yaw_new])

class ClusterDataset(Dataset):
    """
    ... (주석 생략)
    """

    def __init__(self, root: str, split: str = "train", num_points: int = 64):
        super().__init__()
        self.root = root
        self.split = split
        self.num_points = num_points 
        
        # F1TENTH Lidar Params
        self.num_beams = 1080
        self.fov = 4.71238898
        self.angles = np.linspace(-self.fov/2, self.fov/2, self.num_beams)
        self.angles_norm = self.angles / (self.fov/2)

        self.files = sorted(glob.glob(os.path.join(root, "**", "*.npz"), recursive=True))
        self.index_map = []
        
        # 인덱싱 생성
        for fi, p in enumerate(self.files):
            try:
                with np.load(p, allow_pickle=True) as d:
                    if 'ranges' not in d: continue
                    T = len(d['ranges'])
                    for t in range(T):
                        self.index_map.append((fi, t))
            except Exception as e:
                print(f"[Dataset] Error reading {p}: {e}")

    def __len__(self):
        return len(self.index_map)

    def _normalize_cluster(self, points, residuals, angles, center):
        # ... (변경 없음)
        # 1. Centering
        pts_centered = points - center
        
        # 2. Sampling / Padding to num_points
        num_pts = len(pts_centered)
        if num_pts == 0:
            return np.zeros((self.num_points, 2)), np.zeros(self.num_points), np.zeros(self.num_points)
            
        if num_pts >= self.num_points:
            choice = np.random.choice(num_pts, self.num_points, replace=False)
        else:
            choice = np.random.choice(num_pts, self.num_points, replace=True)
            
        return pts_centered[choice], residuals[choice], angles[choice]

    def _compute_residual(self, curr_ranges, prev_ranges, pose_curr, pose_prev):
        # ... (이 함수는 pose_curr, pose_prev를 입력받아 Residual 계산, 내부 로직 변경 없음)
        def get_mat(p):
            x, y, th = p
            c, s = np.cos(th), np.sin(th)
            return np.array([[c, -s, x], [s, c, y], [0, 0, 1]])
        
        H_c = get_mat(pose_curr)
        H_p = get_mat(pose_prev)
        H_rel = np.linalg.inv(H_c) @ H_p

        valid_p = (prev_ranges > 0.01) & (prev_ranges < 30.0)
        r_p = prev_ranges[valid_p]
        th_p = self.angles[valid_p]
        x_p = r_p * np.cos(th_p)
        y_p = r_p * np.sin(th_p)
        
        if len(x_p) == 0:
            return np.zeros_like(curr_ranges)

        ones = np.ones_like(x_p)
        pts_prev_homo = np.stack([x_p, y_p, ones], axis=0)

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
        diff = np.abs(curr_ranges[mask] - pred_ranges[mask])
        residual[mask] = np.tanh(diff) 
        
        return residual

    def __getitem__(self, idx):
        file_idx, frame_idx = self.index_map[idx]
        
        # [FRAME_SKIP 체크] 과거 데이터 부족 시 NaN 반환
        if frame_idx < FRAME_SKIP:
            return (torch.zeros(4, self.num_points), 
                    torch.zeros(4, self.num_points), 
                    torch.zeros(4), 
                    torch.zeros(2),
                    torch.full((2,), float('nan')))

        path = self.files[file_idx]
        with np.load(path, allow_pickle=True) as d:
            ranges_all = d['ranges']
            ego_pose_all = d['ego_pose']
            ego_twist_all = d['ego_twist'] # Twist 로드
            timestamps = d.get('timestamps', None)
            point_vels = d.get('point_velocities', None)
            seg_ids_all = d.get('segment_id_per_point', None)

            # Load Data
            curr_ranges = np.array(ranges_all[frame_idx], dtype=float)
            prev_idx = frame_idx - FRAME_SKIP
            prev_ranges = np.array(ranges_all[prev_idx], dtype=float)
            
            # ✅ 수정: Residual 계산에 사용할 Pose를 Twist 적분으로 대체하기 위해
            #      Twist와 dt를 계산합니다. (Ego Vector 계산 로직 재사용)
            dt = 0.04 
            if timestamps is not None:
                dt_all = np.diff(timestamps)
                # dt는 FRAME_SKIP 간격의 총 합이 필요합니다.
                dt_total = timestamps[frame_idx] - timestamps[prev_idx]
                if dt_total <= 0.0001: dt_total = 0.04
                dt = dt_total
            else:
                dt_all = np.full(len(ranges_all) - 1, 0.04) # 기본 dt 

            # ---------------------------------------------------------
            # ⚡️ [핵심] Residual 계산에 사용할 Pose를 Twist 적분으로 재구성
            # ---------------------------------------------------------
            
            # 1. 초기 Pose 설정 (Twist 적분의 기준점)
            # Twist 적분은 누적 오차를 발생시키므로, 초기 Pose는 원본을 사용합니다.
            pose_ref = ego_pose_all[prev_idx] 
            
            # 2. Twist 적분 실행 (ref_idx 부터 curr_idx까지)
            poses_integrated = [pose_ref]
            
            # prev_idx는 이미 ego_pose_all[prev_idx]로 시작했으므로, 
            # prev_idx 부터 frame_idx - 1 까지 순차 적분
            for t in range(prev_idx, frame_idx):
                twist = ego_twist_all[t]
                
                # 해당 프레임 간격 dt (t to t+1)
                if timestamps is not None:
                    try:
                        step_dt = timestamps[t+1] - timestamps[t]
                        if step_dt <= 0.0001: step_dt = 0.04
                    except IndexError: # 마지막 프레임 처리 (이 로직에서는 발생하지 않아야 함)
                        step_dt = 0.04
                else:
                    step_dt = 0.04
                
                new_pose = integrate_twist(poses_integrated[-1], twist, step_dt)
                poses_integrated.append(new_pose)

            # Twist 적분 결과 추출
            pose_prev_twist_integrated = poses_integrated[0] # ego_pose_all[prev_idx]와 동일해야 함
            pose_curr_twist_integrated = poses_integrated[-1]

            # Residual 계산은 Twist 적분 Pose 사용
            residual_full = self._compute_residual(curr_ranges, prev_ranges, 
                                                   pose_curr_twist_integrated, pose_prev_twist_integrated)
            
            # Ego Vector 계산은 파일 twist 그대로 사용 (원래 로직 유지)
            twist_curr = np.array(ego_twist_all[frame_idx], dtype=float)
            # ---------------------------------------------------------

            # 2. Local Cartesian
            valid_mask = (curr_ranges > 0.01) & (curr_ranges < 30.0)
            x_c = curr_ranges * np.cos(self.angles)
            y_c = curr_ranges * np.sin(self.angles)
            points_local = np.stack([x_c, y_c], axis=1)

            # 3. Cluster Sampling
            target_seg_id = -1
            if seg_ids_all is not None:
                seg_ids_frame = seg_ids_all[frame_idx]
                unique_ids = np.unique(seg_ids_frame)
                unique_ids = unique_ids[unique_ids != -1]
                if len(unique_ids) > 0:
                    target_seg_id = np.random.choice(unique_ids)
            
            mask = (seg_ids_frame == target_seg_id) & valid_mask if target_seg_id != -1 else valid_mask

            if np.sum(mask) < 3:
                return (torch.zeros(4, self.num_points), torch.zeros(4, self.num_points), 
                        torch.zeros(4), torch.zeros(2), torch.full((2,), float('nan')))

            # 4. Data Extraction & Normalization
            cluster_pts = points_local[mask]
            cluster_res = residual_full[mask]
            cluster_ang = self.angles_norm[mask]
            center = np.mean(cluster_pts, axis=0)

            pts_norm, res_norm, ang_norm = self._normalize_cluster(cluster_pts, cluster_res, cluster_ang, center)

            # 5. Tensor Construction
            # [x, y, residual, angle]
            input_tensor = np.stack([pts_norm[:,0], pts_norm[:,1], res_norm, ang_norm], axis=0)
            prev_input_tensor = input_tensor.copy()
            prev_input_tensor[2, :] = 0.0 

            # 6. Ego Vector Handling (with Noise Injection)
            vx_raw = twist_curr[0]
            vy_raw = twist_curr[1]
            w_raw  = twist_curr[2]

            # 학습 시 Data Augmentation (노이즈 추가)
            if self.split == 'train':
                noise_vx = (np.random.randn() * np.abs(vx_raw) * 0.1) + (np.random.randn() * 0.05)
                noise_vy = (np.random.randn() * np.abs(vy_raw) * 0.1) + (np.random.randn() * 0.05)
                
                sim_vx = vx_raw + noise_vx
                sim_vy = vy_raw + noise_vy
            else:
                sim_vx = vx_raw
                sim_vy = vy_raw

            # Network Input (Normalized)
            norm_vx = sim_vx / 10.0
            norm_vy = sim_vy / 10.0
            ego_vector = np.array([norm_vx, norm_vy, w_raw, dt], dtype=np.float32) 
            
            # Physics Shortcut Input (Original Scale)
            raw_ego_vel = np.array([sim_vx, sim_vy], dtype=np.float32)

            # 7. Target Velocity (Ground Truth)
            if point_vels is not None:
                vels_frame = point_vels[frame_idx]
                vels_cluster = vels_frame[mask]
                if np.isnan(vels_cluster).all():
                    target_vel = np.array([np.nan, np.nan])
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        target_vel = np.nanmean(vels_cluster, axis=0)
            else:
                target_vel = np.array([np.nan, np.nan])

            if np.isnan(target_vel).any():
                 target_vel = np.array([np.nan, np.nan])

            return (torch.from_numpy(input_tensor.astype(np.float32)), 
                    torch.from_numpy(prev_input_tensor.astype(np.float32)), 
                    torch.from_numpy(ego_vector), 
                    torch.from_numpy(raw_ego_vel),
                    torch.from_numpy(target_vel.astype(np.float32)))
