#!/usr/bin/env python3
import os
import numpy as np
from torch.utils.data import Dataset
import torch
import warnings

class ClusterDataset(Dataset):
    """
    Returns per-sample:
        1. input_tensor: (4, N) float32 -> [x, y, residual, angle_norm]
        2. prev_input_tensor: (4, N) float32 -> [x, y, 0, angle_norm] (Interaction branch)
        3. ego_vector: (4,) float32 -> [vx_norm, vy_norm, omega, dt] (Network Input)
        4. raw_ego_vel: (2,) float32 -> [vx, vy] (m/s, CLEAN) (Physics Shortcut)
        5. target_vel_tensor: (2,) float32 -> [vx, vy] (Ground Truth Object Velocity)
        6. label: (1,) long -> 0:Static, 1:Dynamic (Weighted Loss용)
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

        # 파일 목록 로드
        if not os.path.exists(root):
            print(f"[Dataset] Warning: Root directory {root} does not exist.")
            self.files = []
        else:
            self.files = sorted([os.path.join(root, f) for f in os.listdir(root) if f.endswith(".npz")])
        
        self.index_map = []
        
        # Frame Skip 설정
        self.FRAME_SKIP = 10 
        
        # 데이터 인덱싱 생성
        print(f"[Dataset] Indexing {len(self.files)} files in {root}...")
        for fi, p in enumerate(self.files):
            try:
                with np.load(p, allow_pickle=True) as d:
                    if 'ranges' not in d: continue
                    T = len(d['ranges'])
                    # FRAME_SKIP 이전 프레임은 과거 데이터가 없으므로 건너뛰거나 더미 반환 (여기선 인덱스 포함하되 getitem에서 처리)
                    for t in range(T):
                        self.index_map.append((fi, t))
            except Exception as e:
                print(f"[Dataset] Error reading {p}: {e}")
        
        print(f"[Dataset] Total samples: {len(self.index_map)}")

    def __len__(self):
        return len(self.index_map)

    def _normalize_cluster(self, points, residuals, angles, center):
        pts_centered = points - center
        num_pts = len(pts_centered)
        
        if num_pts == 0:
            return np.zeros((self.num_points, 2)), np.zeros(self.num_points), np.zeros(self.num_points)
            
        # Point Sampling (Fixed Size)
        if num_pts >= self.num_points:
            choice = np.random.choice(num_pts, self.num_points, replace=False)
        else:
            choice = np.random.choice(num_pts, self.num_points, replace=True)
            
        return pts_centered[choice], residuals[choice], angles[choice]

    def _compute_residual(self, curr_ranges, prev_ranges, pose_curr, pose_prev):
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
        
        # Project previous points to current frame
        idxs = ((th_warped + self.fov/2) / angle_res).round().astype(int)
        
        valid_proj = (idxs >= 0) & (idxs < self.num_beams)
        for i, dist in zip(idxs[valid_proj], r_warped[valid_proj]):
            if dist < pred_ranges[i]:
                pred_ranges[i] = dist
                
        valid_curr = (curr_ranges > 0.01) & (curr_ranges < 30.0)
        residual = np.zeros_like(curr_ranges)
        
        # Compare current range vs predicted (warped) range
        mask = valid_curr & (pred_ranges != np.inf)
        diff = np.abs(curr_ranges[mask] - pred_ranges[mask])
        residual[mask] = np.tanh(diff) 
        
        return residual

    def __getitem__(self, idx):
        file_idx, frame_idx = self.index_map[idx]
        
        # [FRAME_SKIP 체크]
        if frame_idx < self.FRAME_SKIP:
            # 반환값 개수를 6개로 맞춤 (마지막에 라벨 -1 추가)
            return (torch.zeros(4, self.num_points), 
                    torch.zeros(4, self.num_points), 
                    torch.zeros(4), 
                    torch.zeros(2),
                    torch.full((2,), float('nan')),
                    torch.tensor(-1, dtype=torch.long)) # Dummy Label

        path = self.files[file_idx]
        with np.load(path, allow_pickle=True) as d:
            ranges_all = d['ranges']
            ego_pose_all = d['ego_pose']
            ego_twist_all = d['ego_twist']
            timestamps = d.get('timestamps', None)
            point_vels = d.get('point_velocities', None)
            seg_ids_all = d.get('segment_id_per_point', None)
            
            # ⭐️ [추가] 라벨 정보 로드 (0:Static, 1:Dynamic, 2:New)
            labels_all = d.get('labels', None) 

            # Load Data
            curr_ranges = np.array(ranges_all[frame_idx], dtype=float)
            prev_idx = frame_idx - self.FRAME_SKIP
            prev_ranges = np.array(ranges_all[prev_idx], dtype=float)
            
            pose_curr = ego_pose_all[frame_idx]
            pose_prev = ego_pose_all[prev_idx] 
            twist_curr = ego_twist_all[frame_idx]

            # DT Calc
            dt = 0.04 
            if timestamps is not None:
                dt = timestamps[frame_idx] - timestamps[prev_idx]
            if dt <= 0.0001: dt = 0.04

            # 1. Residual Calculation
            residual_full = self._compute_residual(curr_ranges, prev_ranges, pose_curr, pose_prev)

            # 2. Local Cartesian
            valid_mask = (curr_ranges > 0.01) & (curr_ranges < 30.0)
            x_c = curr_ranges * np.cos(self.angles)
            y_c = curr_ranges * np.sin(self.angles)
            points_local = np.stack([x_c, y_c], axis=1)

            # 3. Cluster Sampling (With Balancing)
            target_seg_id = -1
            if seg_ids_all is not None:
                seg_ids_frame = seg_ids_all[frame_idx]
                unique_ids = np.unique(seg_ids_frame)
                unique_ids = unique_ids[unique_ids != -1]
                
                if len(unique_ids) > 0:
                    # 기본적으로는 랜덤 선택
                    target_seg_id = np.random.choice(unique_ids)

                    # ✅ [수정] 학습 시 동적 객체 밸런싱 (Balancing) 적용
                    # Dynamic(1) 객체가 존재하면 50% 확률로 강제 선택
                    if self.split == 'train' and labels_all is not None:
                        labels_frame = labels_all[frame_idx]
                        
                        # 라벨이 1(Dynamic)인 포인트의 세그먼트 ID 찾기
                        # (벡터 연산으로 빠르게 처리)
                        dyn_mask = (labels_frame == 1)
                        if dyn_mask.any():
                            dyn_ids = np.unique(seg_ids_frame[dyn_mask])
                            dyn_ids = dyn_ids[dyn_ids != -1]
                            
                            if len(dyn_ids) > 0 and np.random.rand() < 0.5:
                                target_seg_id = np.random.choice(dyn_ids)
            
            mask = (seg_ids_frame == target_seg_id) & valid_mask if target_seg_id != -1 else valid_mask

            if np.sum(mask) < 3:
                return (torch.zeros(4, self.num_points), torch.zeros(4, self.num_points), 
                        torch.zeros(4), torch.zeros(2), torch.full((2,), float('nan')),
                        torch.tensor(-1, dtype=torch.long))

            # 4. Data Extraction & Normalization
            cluster_pts = points_local[mask]
            cluster_res = residual_full[mask]
            cluster_ang = self.angles_norm[mask]
            center = np.mean(cluster_pts, axis=0)

            pts_norm, res_norm, ang_norm = self._normalize_cluster(cluster_pts, cluster_res, cluster_ang, center)

            # 5. Tensor Construction
            input_tensor = np.stack([pts_norm[:,0], pts_norm[:,1], res_norm, ang_norm], axis=0)
            prev_input_tensor = input_tensor.copy()
            prev_input_tensor[2, :] = 0.0 

            # 6. Ego Vector Handling
            vx_raw = twist_curr[0]
            vy_raw = twist_curr[1]
            w_raw  = twist_curr[2]

            # 학습 시 Data Augmentation
            if self.split == 'train':
                noise_vx = (np.random.randn() * np.abs(vx_raw) * 0.1) + (np.random.randn() * 0.05)
                noise_vy = (np.random.randn() * np.abs(vy_raw) * 0.1) + (np.random.randn() * 0.05)
                
                sim_vx = vx_raw + noise_vx
                sim_vy = vy_raw + noise_vy
            else:
                sim_vx = vx_raw
                sim_vy = vy_raw

            # Network Input: 노이즈가 섞인 값 사용
            norm_vx = sim_vx / 10.0
            norm_vy = sim_vy / 10.0
            ego_vector = np.array([norm_vx, norm_vy, w_raw, dt], dtype=np.float32) 
            
            # Physics Shortcut Input: 노이즈 없는 깨끗한 원본 사용
            raw_ego_vel = np.array([vx_raw, vy_raw], dtype=np.float32)

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

            # ⭐️ [추가] 라벨 결정 로직
            final_label = 0 # Default Static
            if labels_all is not None:
                labels_frame = labels_all[frame_idx]
                cluster_labels = labels_frame[mask]
                if len(cluster_labels) > 0:
                    # 다수결(Mode)로 라벨 결정
                    counts = np.bincount(cluster_labels.astype(int))
                    final_label = np.argmax(counts)
                    
                    # New(2) 라벨은 Static(0)으로 취급
                    if final_label == 2: final_label = 0 

            # 8. Return (6개 반환)
            return (torch.from_numpy(input_tensor.astype(np.float32)), 
                    torch.from_numpy(prev_input_tensor.astype(np.float32)), 
                    torch.from_numpy(ego_vector), 
                    torch.from_numpy(raw_ego_vel),
                    torch.from_numpy(target_vel.astype(np.float32)),
                    torch.tensor(final_label, dtype=torch.long))
