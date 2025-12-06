#!/usr/bin/env python3
import os
import glob
import numpy as np
import yaml
from PIL import Image
import math
import logging
import scipy.ndimage
from sklearn.cluster import DBSCAN
import torch  # PyTorch 사용

# ---------------- CONFIG ----------------
CFG = {
    "MAP_YAML_PATH": "/home/ugrp/aii_ws/src/f1tenth_gym_ros/maps/E1_out2_obs24.yaml",
    "ORIGINAL_NPZ_DIR": "../f1tenth_dataset",
    "OUTPUT_NPZ_DIR": "../dataset_vel_label_gpu", 
    "NUM_BEAMS": 1080,
    "LIDAR_FOV": 4.71238898,   
    "CLUSTER_EPS": 0.5,
    "CLUSTER_MIN_SAMPLES": 3,
    "BALLOON_RADIUS": 0.1,      
    "SEGMENT_OVERLAP_FRAC": 0.2, 
    "VEL_MATCH_DIST_MAX": 1.0, 
    "NEW_SEGMENT_DYNAMIC_ONLY_FIRST": True, 
    "DILATION_PIXELS": 9,
    # ⭐️ [수정됨] 250Hz 기준 기본값
    "DT_DEFAULT": 0.004 
}
# ----------------------------------------

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using Device: {device}")

# --- PyTorch Helper Functions ---
def wrap_angle_torch(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def calculate_rigid_velocity_torch(points_local, ego_twist):
    """GPU Tensor 기반 Rigid Velocity 계산"""
    if len(points_local) == 0:
        return torch.zeros((0, 2), device=device)
    
    vx = ego_twist[0]
    vy = ego_twist[1]
    wz = ego_twist[2]
    
    x = points_local[:, 0]
    y = points_local[:, 1]
    
    v_rot_x = -wz * y
    v_rot_y =  wz * x
    
    v_x = -(vx + v_rot_x)
    v_y = -(vy + v_rot_y)
    return torch.stack([v_x, v_y], dim=1)

class Labeler:
    def __init__(self, cfg):
        self.cfg = cfg
        self.load_map(cfg["MAP_YAML_PATH"], cfg["DILATION_PIXELS"])
        self.num_beams = cfg["NUM_BEAMS"]
        
        # Angles를 GPU 텐서로 미리 올려둠
        angles_np = np.linspace(-cfg["LIDAR_FOV"]/2, cfg["LIDAR_FOV"]/2, self.num_beams)
        self.angles = torch.tensor(angles_np, dtype=torch.float32, device=device)
        self.cos_angles = torch.cos(self.angles)
        self.sin_angles = torch.sin(self.angles)

        self.prev_points_world = None # Tensor (N, 2)
        self.prev_valid_beam_idxs = None
        self.prev_timestamp = None
        
        self.prev_segments = {}
        self.next_segment_gid = 1
        self.frame_counter = 0

    def load_map(self, map_yaml_path, dilation_pixels):
        with open(map_yaml_path, 'r') as f:
            meta = yaml.safe_load(f)
        self.resolution = meta['resolution']
        self.origin_x = meta['origin'][0]
        self.origin_y = meta['origin'][1]
        map_image_path = meta['image']
        if not os.path.isabs(map_image_path):
            map_image_path = os.path.join(os.path.dirname(map_yaml_path), map_image_path)
        map_img = np.array(Image.open(map_image_path).convert('L'))
        self.map_h, self.map_w = map_img.shape
        occ_thresh = 255 * (1.0 - meta['occupied_thresh'])
        binary_walls = (map_img < occ_thresh)
        struct = np.ones((dilation_pixels, dilation_pixels), dtype=bool)
        self.grid_map = scipy.ndimage.binary_dilation(binary_walls, structure=struct)
        logging.info(f"Map loaded: res={self.resolution}, size={self.map_w}x{self.map_h}")

    def scan_to_local_torch(self, ranges_tensor):
        valid = (ranges_tensor > 0.01) & (ranges_tensor < 30.0)
        r = ranges_tensor[valid]
        x = r * self.cos_angles[valid]
        y = r * self.sin_angles[valid]
        pts = torch.stack([x, y], dim=1)
        return pts, valid

    def local_to_world_torch(self, points_local, ego_pose):
        if len(points_local) == 0:
            return torch.zeros((0, 2), device=device)
        
        px, py, yaw = ego_pose
        c = torch.cos(yaw)
        s = torch.sin(yaw)
        
        x = points_local[:, 0]
        y = points_local[:, 1]
        
        x_new = x * c - y * s + px
        y_new = x * s + y * c + py
        
        return torch.stack([x_new, y_new], dim=1)

    def rotate_body_to_world_torch(self, vec_body, yaw):
        c = torch.cos(yaw)
        s = torch.sin(yaw)
        
        x = vec_body[0]
        y = vec_body[1]
        
        x_new = x * c - y * s
        y_new = x * s + y * c
        return torch.stack([x_new, y_new])
    
    def rotate_world_to_body_torch(self, vec_world, yaw):
        # R.T
        c = torch.cos(yaw)
        s = torch.sin(yaw)
        
        x = vec_world[:, 0]
        y = vec_world[:, 1]
        
        x_new = x * c + y * s
        y_new = -x * s + y * c
        return torch.stack([x_new, y_new], dim=1)

    def is_free_mask(self, pts_world):
        # Tensor/Numpy 호환 처리
        if isinstance(pts_world, torch.Tensor):
            pts_cpu = pts_world.cpu().numpy()
        else:
            pts_cpu = pts_world
            
        if len(pts_cpu) == 0:
            return np.zeros((0,), dtype=bool)
        
        u = ((pts_cpu[:, 0] - self.origin_x) / self.resolution).astype(int)
        v = (self.map_h - 1 - (pts_cpu[:, 1] - self.origin_y) / self.resolution).astype(int)
        
        valid = (u >= 0) & (u < self.map_w) & (v >= 0) & (v < self.map_h)
        is_wall = np.zeros(len(pts_cpu), dtype=bool)
        if np.any(valid):
            is_wall[valid] = self.grid_map[v[valid], u[valid]]
        return ~is_wall 

    def seed_prev_from_input_last(self, npz_path):
        try:
            with np.load(npz_path, allow_pickle=True) as d:
                ranges = torch.tensor(d['ranges'], device=device, dtype=torch.float32)
                ego_pose = torch.tensor(d['ego_pose'], device=device, dtype=torch.float32)
        except Exception:
            return
        if len(ranges) == 0: return
        
        last_idx = len(ranges) - 1
        last_ranges = ranges[last_idx]
        last_pose = ego_pose[last_idx]
        
        pts_local, valid_mask = self.scan_to_local_torch(last_ranges)
        pts_world = self.local_to_world_torch(pts_local, last_pose)
        
        if len(pts_world) > 0:
            beam_idxs = torch.where(valid_mask)[0].cpu().numpy()
            self.prev_points_world = pts_world.clone() 
            self.prev_valid_beam_idxs = beam_idxs
            
            gid = self.next_segment_gid; self.next_segment_gid += 1
            self.prev_segments[gid] = {
                'points': pts_world.clone(), 
                'centroid': pts_world.mean(dim=0).cpu().numpy(), 
                'last_seen': self.frame_counter
            }
            logging.info(f"Seeded prev points from {npz_path}")

    def process_npz_file(self, in_path, out_path):
        logging.info(f"Processing: {in_path} -> {out_path}")
        with np.load(in_path, allow_pickle=True) as data:
            original = dict(data)

        ranges_all = torch.tensor(original['ranges'], device=device, dtype=torch.float32)
        ego_pose_all = torch.tensor(original['ego_pose'], device=device, dtype=torch.float32)
        
        timestamps = original.get('timestamps', None)
        input_point_vels = original.get('point_velocities', None)

        T = ranges_all.shape[0]
        BEAMS = self.num_beams

        recalc_twist_all = np.zeros((T, 3), dtype=float)
        labels_out = np.zeros((T, BEAMS), dtype=np.uint8)
        vel_out = np.full((T, BEAMS, 2), np.nan, dtype=float)
        segid_out = np.full((T, BEAMS), -1, dtype=int)

        for t in range(T):
            ranges = ranges_all[t]
            ego_pose = ego_pose_all[t]
            ts = timestamps[t] if timestamps is not None else None

            # --- [STEP 1] Ego Velocity 역산 ---
            dt = CFG["DT_DEFAULT"]
            if t > 0:
                prev_pose = ego_pose_all[t-1]
                prev_ts = timestamps[t-1] if timestamps is not None else (ts - CFG["DT_DEFAULT"] if ts else None)
                if (ts is not None) and (prev_ts is not None):
                    dt_calc = ts - prev_ts
                    if dt_calc > 1e-6: dt = dt_calc

                dx_global = ego_pose[0] - prev_pose[0]
                dy_global = ego_pose[1] - prev_pose[1]
                dyaw = (ego_pose[2] - prev_pose[2] + np.pi) % (2 * np.pi) - np.pi

                yaw_prev = prev_pose[2]
                c = torch.cos(yaw_prev); s = torch.sin(yaw_prev)
                dx_local = dx_global * c + dy_global * s
                dy_local = -dx_global * s + dy_global * c

                vx_temp = (dx_local / dt).item()
                
                # [수정됨] 10m/s 넘게 튀면 dt가 너무 작은 것으로 간주하고 0.004로 고정 재계산
                if abs(vx_temp) > 10.0:
                    dt = 0.004 # 250Hz 강제
                    vx_calc = (dx_local / dt).item()
                    vy_calc = (dy_local / dt).item()
                    w_calc = (dyaw / dt).item()
                else:
                    vx_calc = vx_temp
                    vy_calc = (dy_local / dt).item()
                    w_calc = (dyaw / dt).item()
            else:
                vx_calc, vy_calc, w_calc = 0.0, 0.0, 0.0

            ego_twist = torch.tensor([vx_calc, vy_calc, w_calc], device=device, dtype=torch.float32)
            recalc_twist_all[t] = [vx_calc, vy_calc, w_calc]

            if (self.prev_timestamp is None) or (ts is None):
                pass 

            pts_local, valid_mask = self.scan_to_local_torch(ranges)
            beam_idxs = torch.where(valid_mask)[0].cpu().numpy() 
            N = len(pts_local)

            if N == 0:
                labels_out[t,:] = 0
                vel_out[t,:,:] = np.nan
                segid_out[t,:] = -1
                self.prev_points_world = None
                self.prev_valid_beam_idxs = None
                self.prev_timestamp = ts
                self.frame_counter += 1
                continue

            pts_world = self.local_to_world_torch(pts_local, ego_pose)

            # --- [STEP 2] Balloon Check ---
            is_within_balloon = np.zeros(N, dtype=bool)
            dists_to_prev = torch.full((N,), float('inf'), device=device)
            idxs_prev = torch.full((N,), -1, dtype=torch.long, device=device)

            if (self.prev_points_world is not None) and (len(self.prev_points_world) > 0):
                dists = torch.cdist(pts_world, self.prev_points_world) 
                min_dists, min_idxs = torch.min(dists, dim=1)
                
                dists_to_prev = min_dists
                idxs_prev = min_idxs
                is_within_balloon = (min_dists.cpu().numpy() <= CFG["BALLOON_RADIUS"])
            else:
                is_within_balloon[:] = False
            
            is_temp_new = ~is_within_balloon

            # --- [STEP 3] Clustering & Splitting ---
            pts_world_cpu = pts_world.cpu().numpy()
            seg_labels = DBSCAN(eps=CFG["CLUSTER_EPS"], min_samples=CFG["CLUSTER_MIN_SAMPLES"]).fit_predict(pts_world_cpu)
            unique_segs = np.unique(seg_labels)

            max_gid = np.max(unique_segs) if len(unique_segs) > 0 else 0
            new_seg_labels = seg_labels.copy()
            
            for seg in unique_segs:
                if seg == -1: continue
                mask = (seg_labels == seg)
                new_flags_in_seg = is_temp_new[mask]
                if np.any(new_flags_in_seg) and np.any(~new_flags_in_seg):
                    idxs_new_in_seg = np.where(mask & is_temp_new)[0]
                    max_gid += 1
                    new_seg_labels[idxs_new_in_seg] = max_gid
            
            seg_labels = new_seg_labels
            unique_segs = np.unique(seg_labels)

            # --- [STEP 4] Segment Stats ---
            seg_is_dynamic_dict = {}
            for seg in unique_segs:
                if seg == -1: continue
                mask = (seg_labels == seg)
                seg_pts_cpu = pts_world_cpu[mask]
                # Numpy 배열로 넘김 (AttributeError 해결)
                free_mask = self.is_free_mask(seg_pts_cpu)
                frac_free = np.mean(free_mask)
                seg_is_dynamic_dict[seg] = (frac_free >= 0.5)
            
            seg_is_dynamic_dict[-1] = False

            # --- [STEP 5] Segment Matching ---
            prev_gids = list(self.prev_segments.keys())
            seg_gid_map = {} 
            
            if len(prev_gids) > 0 and len(unique_segs) > 0:
                for i_cur, seg in enumerate(unique_segs):
                    if seg == -1: continue
                    mask = (seg_labels == seg)
                    if not np.any(mask): continue
                    pts_cur_gpu = pts_world[mask]
                    
                    best_frac = 0.0
                    best_gid = -1
                    
                    for gid in prev_gids:
                        pts_prev_gpu = self.prev_segments[gid]['points']
                        if len(pts_prev_gpu) == 0: continue
                        
                        dists = torch.cdist(pts_cur_gpu, pts_prev_gpu)
                        min_d, _ = torch.min(dists, dim=1)
                        
                        count = torch.sum(min_d <= CFG["BALLOON_RADIUS"]).item()
                        frac = count / len(pts_cur_gpu)
                        
                        if frac > best_frac:
                            best_frac = frac
                            best_gid = gid
                            
                    if best_frac >= CFG["SEGMENT_OVERLAP_FRAC"]:
                        seg_gid_map[seg] = best_gid

            for seg in unique_segs:
                if seg == -1: continue
                if seg not in seg_gid_map:
                    gid = self.next_segment_gid; self.next_segment_gid += 1
                    seg_gid_map[seg] = gid
                
                gid = seg_gid_map[seg]
                mask = (seg_labels == seg)
                pts_gpu = pts_world[mask]
                centroid = pts_gpu.mean(dim=0).cpu().numpy() if len(pts_gpu)>0 else np.array([np.nan, np.nan])
                self.prev_segments[gid] = {
                    'points': pts_gpu.clone(), 
                    'centroid': centroid, 
                    'last_seen': self.frame_counter
                }

            # --- [STEP 6] Velocity & Label Calculation (Vectorized GPU) ---
            rigid_vels = calculate_rigid_velocity_torch(pts_local, ego_twist).cpu().numpy()
            vel_frame = np.full((BEAMS, 2), np.nan, dtype=float)
            labels_this = np.zeros(BEAMS, dtype=np.uint8)

            idxs_prev_cpu = idxs_prev.cpu().numpy() 

            for k, beam_idx in enumerate(beam_idxs):
                seg = seg_labels[k]
                is_map_dynamic = seg_is_dynamic_dict.get(seg, False)
                is_matched = is_within_balloon[k]

                if is_matched:
                    if not is_map_dynamic:
                        labels_this[beam_idx] = 0 # Static
                        vel_frame[beam_idx, :] = rigid_vels[k]
                    else:
                        labels_this[beam_idx] = 1 # Dynamic
                        if (idxs_prev_cpu[k] >= 0) and (dt > 1e-6):
                            # GPU 연산
                            p_prev_gpu = self.prev_points_world[idxs_prev_cpu[k]]
                            p_curr_gpu = pts_world[k]
                            v_world_gpu = (p_curr_gpu - p_prev_gpu) / dt
                            
                            # 🛠️ [수정됨] float() 제거하고 Tensor 그대로 사용
                            ego_v_body = torch.tensor([ego_twist[0], ego_twist[1]], device=device)
                            V_ego_world = self.rotate_body_to_world_torch(ego_v_body, ego_pose[2])
                            
                            v_rel_world = v_world_gpu - V_ego_world
                            v_rel_ego = self.rotate_world_to_body_torch(v_rel_world.unsqueeze(0), ego_pose[2])[0]
                            
                            vel_frame[beam_idx, :] = v_rel_ego.cpu().numpy()
                        else:
                            vel_frame[beam_idx, :] = np.nan
                else:
                    # New Points
                    if not is_map_dynamic:
                        labels_this[beam_idx] = 0 
                        vel_frame[beam_idx, :] = rigid_vels[k]
                    else:
                        labels_this[beam_idx] = 2 
                        vel_frame[beam_idx, :] = np.nan

            # --- [STEP 7] Save ---
            segid_full = np.array([seg_gid_map.get(s, -1) for s in seg_labels], dtype=int)

            labels_out[t] = labels_this
            vel_out[t] = vel_frame
            segid_out[t] = segid_full

            self.prev_points_world = pts_world.clone() 
            self.prev_valid_beam_idxs = beam_idxs 
            self.prev_timestamp = ts if ts is not None else (self.prev_timestamp + dt if self.prev_timestamp is not None else None)
            self.frame_counter += 1

            if (t % 50) == 0:
                logging.debug(f"frame {t}/{T} valid={N}")

        original['ego_twist'] = recalc_twist_all
        original['labels'] = labels_out
        original['point_velocities'] = vel_out
        original['segment_id_per_point'] = segid_out
        
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez_compressed(out_path, **original)
        logging.info(f"Saved: {out_path}")

def main():
    os.makedirs(CFG["OUTPUT_NPZ_DIR"], exist_ok=True)
    files = sorted(glob.glob(os.path.join(CFG["ORIGINAL_NPZ_DIR"], "*.npz")))
    if len(files) == 0:
        logging.warning("No input files")
        return
    labeler = Labeler(CFG)
    for i, f in enumerate(files):
        if i > 0:
            labeler.seed_prev_from_input_last(files[i-1])
        out_path = os.path.join(CFG["OUTPUT_NPZ_DIR"], os.path.basename(f))
        labeler.process_npz_file(f, out_path)

if __name__ == "__main__":
    main()
