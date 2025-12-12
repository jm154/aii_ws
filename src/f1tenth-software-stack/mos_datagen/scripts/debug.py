#!/usr/bin/env python3
import os
import numpy as np
from torch.utils.data import Dataset
import torch
import warnings
import math
import sys
import traceback

# ⭐️ [설정] 몇 프레임 전 데이터와 비교할지 설정 (원본 코드와 동일)
FRAME_SKIP = 10 

# ⚠️ [설정] 데이터 경로 설정 (사용자 요청 경로)
DATA_PATH = "../dataset_l/4ms" 

# --- Helper Functions ---
def wrap_angle(angle):
    """각도를 [-pi, pi] 범위로 래핑"""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def rotate_world_to_body(vec_world, yaw):
    """World 프레임 벡터를 Body 프레임으로 회전"""
    c = np.cos(yaw)
    s = np.sin(yaw)
    # Rotation Matrix Transpose: [[c, s], [-s, c]]
    x = vec_world[0] * c + vec_world[1] * s
    y = -vec_world[0] * s + vec_world[1] * c
    return np.array([x, y])

class ClusterDataset(Dataset):
    """
    Returns per-sample:
        input_tensor: (4, N) float32 -> [x, y, residual, angle_norm]
        prev_input_tensor: (4, N) float32 -> [x, y, 0, angle_norm] (Interaction branch)
        ego_vector: (4,) float32 -> [vx_norm, vy_norm, omega, dt] (Network Input)
        raw_ego_vel: (2,) float32 -> [vx, vy] (m/s, with noise) (Physics Shortcut)
        target_vel_tensor: (2,) float32 -> [vx, vy] (Ground Truth Object Velocity)
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

        self.files = sorted([os.path.join(root, f) for f in os.listdir(root) if f.endswith(".npz")])
        self.index_map = []
        
        # 인덱싱 생성
        for fi, p in enumerate(self.files):
            try:
                with np.load(p, allow_pickle=True) as d:
                    if 'ranges' not in d: continue
                    T = len(d['ranges'])
                    for t in range(T):
                        # FRAME_SKIP에 필요한 최소 프레임 인덱스 확인
                        if t >= FRAME_SKIP: 
                            self.index_map.append((fi, t))
            except Exception as e:
                print(f"[Dataset] Error reading {p}: {e}")

    def __len__(self):
        return len(self.index_map)

    def _normalize_cluster(self, points, residuals, angles, center):
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
        def get_mat(p):
            x, y, th = p
            c, s = np.cos(th), np.sin(th)
            return np.array([[c, -s, x], [s, c, y], [0, 0, 1]])
        
        H_c = get_mat(pose_curr)
        H_p = get_mat(pose_prev)
        # H_rel: Prev -> Curr 변환 행렬
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

        # 이전 스캔 포인트를 현재 좌표계로 변환 (Ego-Motion 적용)
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
        
        # Residual은 tanh를 사용하여 0~1 사이로 스케일링
        residual[mask] = np.tanh(diff) 
        
        return residual

    def __getitem__(self, idx):
        file_idx, frame_idx = self.index_map[idx]
        
        # __init__에서 frame_idx >= FRAME_SKIP를 이미 필터링했으므로 이 부분은 필요 없음
        # if frame_idx < FRAME_SKIP:
        #     ... (NaN 반환)
        #     pass 

        path = self.files[file_idx]
        with np.load(path, allow_pickle=True) as d:
            ranges_all = d['ranges']
            ego_pose_all = d['ego_pose']
            timestamps = d.get('timestamps', None)
            point_vels = d.get('point_velocities', None)
            seg_ids_all = d.get('segment_id_per_point', None)

            # Load Data
            curr_ranges = np.array(ranges_all[frame_idx], dtype=float)
            prev_idx = frame_idx - FRAME_SKIP
            prev_ranges = np.array(ranges_all[prev_idx], dtype=float)
            
            pose_curr = ego_pose_all[frame_idx]
            pose_prev = ego_pose_all[prev_idx] 

            # ---------------------------------------------------------
            # ⚡️ [핵심] Ego-Motion (속도) 직접 계산 (Pose Diff)
            # ---------------------------------------------------------
            dt = 0.04 # 기본값 (250Hz * 10 = 0.04s 가정)
            if timestamps is not None:
                dt = timestamps[frame_idx] - timestamps[prev_idx]
            
            # dt 안전장치
            if dt <= 0.0001: dt = 0.04

            # Global Frame 이동량
            dx_global = pose_curr[0] - pose_prev[0]
            dy_global = pose_curr[1] - pose_prev[1]
            dyaw = wrap_angle(pose_curr[2] - pose_prev[2])

            # Rotation Matrix (World -> Prev Body Frame)
            # 로봇이 10프레임 전 바라보던 방향 기준으로 이동량 분해
            vec_world = np.array([dx_global, dy_global])
            vec_local = rotate_world_to_body(vec_world, pose_prev[2])

            vx_calc = vec_local[0] / dt
            vy_calc = vec_local[1] / dt
            w_calc = dyaw / dt
            
            twist_curr = np.array([vx_calc, vy_calc, w_calc])
            # ---------------------------------------------------------

            # 1. Residual Calculation (10프레임 전과 비교)
            residual_full = self._compute_residual(curr_ranges, prev_ranges, pose_curr, pose_prev)

            # 2. Local Cartesian
            valid_mask = (curr_ranges > 0.01) & (curr_ranges < 30.0)
            x_c = curr_ranges * np.cos(self.angles)
            y_c = curr_ranges * np.sin(self.angles)
            points_local = np.stack([x_c, y_c], axis=1)

            # 3. Cluster Sampling
            target_seg_id = -1
            if seg_ids_all is not None and len(seg_ids_all) > frame_idx:
                seg_ids_frame = seg_ids_all[frame_idx]
                unique_ids = np.unique(seg_ids_frame)
                unique_ids = unique_ids[unique_ids != -1]
                if len(unique_ids) > 0:
                    target_seg_id = np.random.choice(unique_ids)
            
            mask = (seg_ids_frame == target_seg_id) & valid_mask if target_seg_id != -1 and 'seg_ids_frame' in locals() else valid_mask

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
            target_vel = np.array([np.nan, np.nan])
            if point_vels is not None and len(point_vels) > frame_idx:
                vels_frame = point_vels[frame_idx]
                vels_cluster = vels_frame[mask]
                
                if not np.all(np.isnan(vels_cluster)):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        target_vel = np.nanmean(vels_cluster, axis=0)

            if np.isnan(target_vel).any():
                target_vel = np.array([np.nan, np.nan])

            return (torch.from_numpy(input_tensor.astype(np.float32)), 
                    torch.from_numpy(prev_input_tensor.astype(np.float32)), 
                    torch.from_numpy(ego_vector), 
                    torch.from_numpy(raw_ego_vel),
                    torch.from_numpy(target_vel.astype(np.float32)))

# --- 검증 함수 ---
def test_cluster_dataset():
    """ClusterDataset의 주요 계산 결과(Ego-Motion, Residual)를 검증하는 함수."""
    
    print("--- 📚 ClusterDataset 계산 검증 시작 ---")
    print(f"데이터 경로: {DATA_PATH}")
    
    if not os.path.exists(DATA_PATH):
        print(f"\n[❌ 오류] 지정된 경로 '{DATA_PATH}'를 찾을 수 없습니다.")
        print("경로를 실제 데이터가 있는 폴더로 수정하거나 경로를 확인해주세요.")
        return

    try:
        # 데이터셋 인스턴스 생성 (학습 모드: 노이즈 추가 확인용)
        dataset = ClusterDataset(root=DATA_PATH, split="train", num_points=64)
        print(f"로드된 전체 샘플 수 (FRAME_SKIP 적용 후): {len(dataset)}")
        
        if len(dataset) == 0:
            print("[⚠️ 경고] 데이터셋에 유효한 샘플이 없습니다. .npz 파일이 경로에 있는지 확인하세요.")
            return

        # 유효한 첫 번째 샘플 찾기
        sample = None
        sample_idx = 0
        max_attempts = 100
        
        # NaN이 아닌 유효한 데이터를 가진 샘플을 찾습니다.
        while sample_idx < min(len(dataset), max_attempts):
            sample = dataset[sample_idx]
            
            # target_vel이 nan이 아니고, input_tensor에 데이터가 있는 경우 유효
            if not torch.isnan(sample[4]).all() and torch.sum(sample[0]) > 0.001:
                break
            
            sample_idx += 1
            sample = None

        if sample is None:
            print(f"[❌ 실패] 유효한 샘플을 {min(len(dataset), max_attempts)}회 시도 내에 찾지 못했습니다. (Target Velocity가 모두 NaN이거나 클러스터 포인트가 부족)")
            return

        # ----------------------------------------------------
        # 1. 계산된 Ego-Motion (네트워크 입력) 확인
        # ----------------------------------------------------
        
        # 반환 값 언팩
        input_tensor, prev_input_tensor, ego_vector, raw_ego_vel, target_vel_tensor = sample
        
        # ego_vector: [vx_norm, vy_norm, omega, dt]
        vx_norm, vy_norm, omega, dt = ego_vector.tolist()
        
        # raw_ego_vel: [vx, vy] (노이즈 포함)
        vx_raw, vy_raw = raw_ego_vel.tolist()
        
        # 계산된 값 (정규화 스케일 복원: 10.0으로 나누었으므로 10.0을 곱함)
        vx_calc = vx_norm * 10.0
        vy_calc = vy_norm * 10.0
        
        # Vx와 Vy 값 비교 (노이즈 때문에 완전히 일치하지는 않지만 근접해야 함)
        vx_diff = np.abs(vx_raw - vx_calc)
        vy_diff = np.abs(vy_raw - vy_calc)
        
        print(f"\n--- 🚗 Ego-Motion (자차 속도) 검증 (샘플 ID: {sample_idx}) ---")
        print(f"프레임 간 시간 간격 (dt): {dt:.4f} 초")
        print(f"각속도 (omega): {omega:.3f} rad/s")
        print(f"Vx (계산 기반): {vx_calc:.3f} m/s | Vx (노이즈 적용): {vx_raw:.3f} m/s | 차이: {vx_diff:.3f}")
        print(f"Vy (계산 기반): {vy_calc:.3f} m/s | Vy (노이즈 적용): {vy_raw:.3f} m/s | 차이: {vy_diff:.3f}")
        
        # 노이즈 허용 범위 설정 (예: 0.1 m/s)
        if vx_diff < 0.15 and vy_diff < 0.15: 
            print("[✅ 성공] 계산된 Ego-Motion 값과 노이즈 적용된 값이 근접합니다.")
        else:
            print("[⚠️ 경고] Vx/Vy 값의 차이가 예상보다 큽니다. 계산 로직을 다시 확인하거나 노이즈 허용 범위를 조정해보세요.")
            
        # ----------------------------------------------------
        # 2. Residual (잔차) 확인
        # ----------------------------------------------------
        
        # input_tensor: [x, y, residual, angle_norm]
        residuals = input_tensor[2, :].numpy()
        mean_residual = np.mean(residuals)
        max_residual = np.max(residuals)
        
        print("\n--- 💥 Residual (잔차) 검증 ---")
        print(f"샘플 Residual 평균: {mean_residual:.4f}")
        print(f"샘플 Residual 최대: {max_residual:.4f} (잔차는 tanh(diff)이므로 최대 1.0)")
        
        if 0.0 <= mean_residual <= 1.0 and 0.0 <= max_residual <= 1.0:
            print("[✅ 성공] Residual 값이 유효 범위(0.0 ~ 1.0) 내에 있습니다.")
        else:
            print("[❌ 실패] Residual 값이 유효 범위를 벗어났습니다.")
            
        # ----------------------------------------------------
        # 3. Target Velocity (정답) 확인
        # ----------------------------------------------------

        target_vx, target_vy = target_vel_tensor.tolist()
        
        print("\n--- 🎯 Target Object Velocity (Ground Truth) ---")
        print(f"Target Vx: {target_vx:.3f} m/s")
        print(f"Target Vy: {target_vy:.3f} m/s")
        
        if not np.isnan(target_vx) and not np.isnan(target_vy):
             print("[✅ 성공] 유효한 Target Velocity가 존재합니다.")
        else:
             print("[⚠️ 경고] Target Velocity가 NaN입니다. 이는 해당 클러스터에 GT 속도 정보가 없거나 유효하지 않음을 의미합니다.")
        # ----------------------------------------------------
        
    except Exception as e:
        print(f"\n[❌ 예외 발생] 테스트 중 오류가 발생했습니다: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    if DATA_PATH == "../dataset_l/4ms" and not os.path.exists(DATA_PATH):
        print("\n*** [경고] DATA_PATH가 기본값이며 해당 경로를 찾을 수 없습니다. ***")
        print(f"현재 경로 기준으로 '{DATA_PATH}'에 .npz 파일이 있는지 확인해주세요.")
        
    test_cluster_dataset()
