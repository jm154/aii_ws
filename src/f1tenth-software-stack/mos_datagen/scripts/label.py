#!/usr/bin/env python3
import os
import glob
import numpy as np
import yaml
from PIL import Image
import math
import logging
import scipy.ndimage
from sklearn.neighbors import KDTree
from sklearn.cluster import DBSCAN

# ---------------- CONFIG ----------------
CFG = {
    "MAP_YAML_PATH": "/home/ugrp/aii_ws/src/f1tenth_gym_ros/maps/E1_out2_obs24.yaml",
    "ORIGINAL_NPZ_DIR": "../f1tenth_dataset",
    "OUTPUT_NPZ_DIR": "../dataset_vel_label",
    "NUM_BEAMS": 1080,
    "LIDAR_FOV": 4.71238898,   
    "CLUSTER_EPS": 0.5,
    "CLUSTER_MIN_SAMPLES": 3,
    "BALLOON_RADIUS": 0.1,      
    "SEGMENT_OVERLAP_FRAC": 0.2, 
    "MATCH_DISTANCE_SEGMENT": 1.0,  
    "VEL_MATCH_DIST_MAX": 1.0, 
    "NEW_SEGMENT_DYNAMIC_ONLY_FIRST": True, 
    "DILATION_PIXELS": 9,
    "DT_DEFAULT": 0.004
}
# ----------------------------------------

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def rotation_matrix(yaw):
    c = math.cos(yaw); s = math.sin(yaw)
    return np.array([[c, -s],[s, c]])

def rotate_world_to_body(vec_world, yaw):
    R = rotation_matrix(yaw)
    if vec_world.ndim == 1:
        return R.T.dot(vec_world)
    else:
        return (R.T @ vec_world.T).T

def rotate_body_to_world(vec_body, yaw):
    R = rotation_matrix(yaw)
    if vec_body.ndim == 1:
        return R.dot(vec_body)
    else:
        return (R @ vec_body.T).T

def calculate_rigid_velocity(points_local, ego_twist):
    if points_local.shape[0] == 0:
        return np.zeros((0,2), dtype=float)
    vx = float(ego_twist[0]); vy = float(ego_twist[1]); wz = float(ego_twist[2]) if len(ego_twist) > 2 else 0.0
    x = points_local[:,0]; y = points_local[:,1]
    v_rot_x = -wz * y
    v_rot_y =  wz * x
    v_x = -(vx + v_rot_x)
    v_y = -(vy + v_rot_y)
    return np.stack([v_x, v_y], axis=1)

class Labeler:
    def __init__(self, cfg):
        self.cfg = cfg
        self.load_map(cfg["MAP_YAML_PATH"], cfg["DILATION_PIXELS"])
        self.num_beams = cfg["NUM_BEAMS"]
        self.angles = np.linspace(-cfg["LIDAR_FOV"]/2, cfg["LIDAR_FOV"]/2, self.num_beams)
        self.prev_points_world = None
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

    def scan_to_local(self, ranges):
        valid = (ranges > 0.01) & (ranges < 30.0)
        r = ranges[valid]
        theta = self.angles[valid]
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        pts = np.stack([x,y], axis=1)
        return pts, valid

    def local_to_world(self, points_local, ego_pose):
        if points_local.shape[0] == 0:
            return np.zeros((0,2), dtype=float)
        px, py, yaw = ego_pose
        R = rotation_matrix(yaw)
        return (R @ points_local.T).T + np.array([px, py])

    def world_to_pixel(self, pts_world):
        u = ((pts_world[:,0] - self.origin_x) / self.resolution).astype(int)
        v = (self.map_h - 1 - (pts_world[:,1] - self.origin_y) / self.resolution).astype(int)
        return u, v

    def is_free_mask(self, pts_world):
        if pts_world.shape[0] == 0:
            return np.zeros((0,), dtype=bool)
        u,v = self.world_to_pixel(pts_world)
        valid = (u>=0)&(u<self.map_w)&(v>=0)&(v<self.map_h)
        is_wall = np.zeros(len(pts_world), dtype=bool)
        if np.any(valid):
            is_wall[valid] = self.grid_map[v[valid], u[valid]]
        return ~is_wall

    def seed_prev_from_input_last(self, npz_path):
        try:
            with np.load(npz_path, allow_pickle=True) as d:
                ranges = d['ranges']
                ego_pose = d['ego_pose']
        except Exception:
            return
        if len(ranges) == 0:
            return
        last_idx = len(ranges) - 1
        last_ranges = ranges[last_idx]
        last_pose = ego_pose[last_idx]
        pts_local, valid_mask = self.scan_to_local(last_ranges)
        pts_world = self.local_to_world(pts_local, last_pose)
        if pts_world.shape[0] > 0:
            beam_idxs = np.where(valid_mask)[0]
            self.prev_points_world = pts_world.copy()
            self.prev_valid_beam_idxs = beam_idxs.copy()
            gid = self.next_segment_gid; self.next_segment_gid += 1
            self.prev_segments[gid] = {'points': pts_world.copy(), 'centroid': pts_world.mean(axis=0), 'last_seen': self.frame_counter}
            logging.info(f"Seeded prev points and prev_segment gid={gid} from {npz_path} (N={len(pts_world)})")

    def process_npz_file(self, in_path, out_path):
        logging.info(f"Processing: {in_path} -> {out_path}")
        with np.load(in_path, allow_pickle=True) as data:
            original = dict(data)

        ranges_all = original['ranges']
        ego_pose_all = original['ego_pose']
        ego_twist_all = original['ego_twist']
        timestamps = original.get('timestamps', None)
        input_point_vels = original.get('point_velocities', None)

        T = ranges_all.shape[0]
        BEAMS = self.num_beams

        labels_out = np.zeros((T, BEAMS), dtype=np.uint8)
        vel_out = np.full((T, BEAMS, 2), np.nan, dtype=float)
        segid_out = np.full((T, BEAMS), -1, dtype=int)

        for t in range(T):
            ranges = ranges_all[t]
            ego_pose = ego_pose_all[t]
            ego_twist = ego_twist_all[t]
            ts = timestamps[t] if timestamps is not None else None

            if (self.prev_timestamp is not None) and (ts is not None):
                dt = ts - self.prev_timestamp
                if dt <= 0: dt = CFG["DT_DEFAULT"]
            else:
                dt = CFG["DT_DEFAULT"]

            pts_local, valid_mask = self.scan_to_local(ranges)
            beam_idxs = np.where(valid_mask)[0]
            N = pts_local.shape[0]
            if N == 0:
                labels_out[t,:] = 0
                vel_out[t,:,:] = np.nan
                segid_out[t,:] = -1
                self.prev_points_world = None
                self.prev_valid_beam_idxs = None
                self.prev_timestamp = ts
                self.frame_counter += 1
                continue

            pts_world = self.local_to_world(pts_local, ego_pose)

            # 1) Balloon check against previous points
            # (이 결과는 나중에 New 판단에 쓰이지만, 지금 클러스터 분할에도 활용합니다)
            is_within_balloon = np.zeros(N, dtype=bool)
            if (self.prev_points_world is not None) and (self.prev_points_world.shape[0] > 0):
                tree_prev = KDTree(self.prev_points_world)
                dists_prev, _ = tree_prev.query(pts_world, k=1)
                is_within_balloon = (dists_prev.flatten() <= CFG["BALLOON_RADIUS"])
            else:
                is_within_balloon[:] = False
            
            # Balloon check 실패한 점들 = 잠재적 New 점들
            is_temp_new = ~is_within_balloon

            # 2) cluster current points in world
            seg_labels = DBSCAN(eps=CFG["CLUSTER_EPS"], min_samples=CFG["CLUSTER_MIN_SAMPLES"]).fit_predict(pts_world)
            unique_segs = np.unique(seg_labels)

            # ----------------------------------------------------------------
            # ⭐️ [핵심 수정] 클러스터 쪼개기 (New와 Not-New가 섞여있으면 분리)
            # ----------------------------------------------------------------
            max_gid = np.max(unique_segs) if len(unique_segs) > 0 else 0
            new_seg_labels = seg_labels.copy()
            
            for seg in unique_segs:
                if seg == -1: continue
                
                mask = (seg_labels == seg)
                # 해당 클러스터 내에 New 점들과 Not-New 점들이 혼재하는지 확인
                new_flags_in_seg = is_temp_new[mask]
                
                if np.any(new_flags_in_seg) and np.any(~new_flags_in_seg):
                    # 섞여 있다면 New 점들만 골라서 새로운 ID 발급
                    idxs_new_in_seg = np.where(mask & is_temp_new)[0]
                    max_gid += 1
                    new_seg_labels[idxs_new_in_seg] = max_gid
            
            # 업데이트된 라벨 적용
            seg_labels = new_seg_labels
            unique_segs = np.unique(seg_labels)
            # ----------------------------------------------------------------

            # compute per-seg world pts & centroid & dynamic flag
            seg_pts_world_map = {}
            seg_centroids = {}
            seg_is_dynamic = {}
            for seg in unique_segs:
                idxs_local = np.where(seg_labels == seg)[0]
                if len(idxs_local) == 0:
                    continue
                seg_pts = pts_world[idxs_local]
                seg_pts_world_map[seg] = seg_pts
                seg_centroids[seg] = seg_pts.mean(axis=0)
                free_mask = self.is_free_mask(seg_pts)
                frac_free = float(np.sum(free_mask)) / float(len(idxs_local))
                seg_is_dynamic[seg] = (frac_free >= 0.5)

            # ===================== NEW: SEGMENT-OVERLAP MATCHING =====================
            # Build list of previous segments (gids and their points)
            prev_gids = list(self.prev_segments.keys())
            prev_points_list = [self.prev_segments[g]['points'] for g in prev_gids] if len(prev_gids)>0 else []
            # We'll compute overlap matrix: rows=current segs, cols=prev_gids
            cur_segs = [s for s in unique_segs]
            overlap_matrix = np.zeros((len(cur_segs), len(prev_gids)), dtype=float) if len(prev_gids)>0 else np.zeros((len(cur_segs),0))
            for i_cur, seg in enumerate(cur_segs):
                pts_cur = seg_pts_world_map.get(seg, np.zeros((0,2)))
                if pts_cur.shape[0]==0:
                    continue
                for j_prev, pgid in enumerate(prev_gids):
                    pts_prev = prev_points_list[j_prev]
                    if pts_prev.shape[0] == 0:
                        overlap_matrix[i_cur, j_prev] = 0.0
                        continue
                    # KDTree from prev segment points
                    tree_pp = KDTree(pts_prev)
                    d, _ = tree_pp.query(pts_cur, k=1)
                    # fraction of current points within balloon radius of prev segment points
                    frac = float(np.sum(d.flatten() <= CFG["BALLOON_RADIUS"])) / float(pts_cur.shape[0])
                    overlap_matrix[i_cur, j_prev] = frac
            # Greedy matching
            seg_gid_map = {}  # seg_label -> (gid, seen_before_bool)
            if overlap_matrix.size > 0:
                I,J = np.where(overlap_matrix>0)
                pairs = [(i,j, overlap_matrix[i,j]) for i,j in zip(I,J)]
                pairs.sort(key=lambda x: x[2], reverse=True)
                used_prev = set()
                used_cur = set()
                for i_cur, j_prev, frac in pairs:
                    if frac < CFG["SEGMENT_OVERLAP_FRAC"]:
                        continue
                    if j_prev in used_prev or i_cur in used_cur:
                        continue
                    cur_seg = cur_segs[i_cur]
                    matched_gid = prev_gids[j_prev]
                    seg_gid_map[cur_seg] = (matched_gid, True)
                    used_prev.add(j_prev); used_cur.add(i_cur)
                # Remaining unmatched current segments -> assign new gids
                for i_cur, seg in enumerate(cur_segs):
                    if i_cur in used_cur:
                        continue
                    # new segment
                    gid = self.next_segment_gid; self.next_segment_gid += 1
                    seg_gid_map[seg] = (gid, False)
                    self.prev_segments[gid] = {'points': np.zeros((0,2)), 'centroid': seg_centroids.get(seg, np.array([np.nan,np.nan])), 'last_seen': self.frame_counter}
            else:
                for seg in cur_segs:
                    gid = self.next_segment_gid; self.next_segment_gid += 1
                    seg_gid_map[seg] = (gid, False)
                    self.prev_segments[gid] = {'points': np.zeros((0,2)), 'centroid': seg_centroids.get(seg, np.array([np.nan,np.nan])), 'last_seen': self.frame_counter}

            # After matching, update prev_segments entries
            for seg in cur_segs:
                gid, seen_before = seg_gid_map[seg]
                seg_pts = seg_pts_world_map.get(seg, np.zeros((0,2)))
                self.prev_segments[gid]['points'] = seg_pts.copy()
                self.prev_segments[gid]['centroid'] = seg_centroids.get(seg, np.array([np.nan,np.nan]))
                self.prev_segments[gid]['last_seen'] = self.frame_counter

            # ===================== END SEGMENT-OVERLAP MATCHING =====================

            # 4) per-point "new" assignment combining balloon + dynamic-first rule
            is_new_point = np.zeros(N, dtype=bool)
            for local_idx in range(N):
                seg = seg_labels[local_idx]
                gid, seen_before = seg_gid_map.get(seg, (None, False))
                dynamic_seg = seg_is_dynamic.get(seg, False)
                if dynamic_seg and (not seen_before) and CFG["NEW_SEGMENT_DYNAMIC_ONLY_FIRST"]:
                    is_new_point[local_idx] = True
                else:
                    if not is_within_balloon[local_idx]:
                        is_new_point[local_idx] = True
                    else:
                        is_new_point[local_idx] = False

            # 5) velocities
            rigid_vels = calculate_rigid_velocity(pts_local, ego_twist)
            vel_frame = np.full((BEAMS, 2), np.nan, dtype=float)

            if (self.prev_points_world is not None) and (self.prev_points_world.shape[0] > 0):
                tree_prev_pts = KDTree(self.prev_points_world)
                dists_to_prev, idxs_prev = tree_prev_pts.query(pts_world, k=1)
                dists_to_prev = dists_to_prev.flatten()
                idxs_prev = idxs_prev.flatten()
            else:
                dists_to_prev = np.full(N, np.inf)
                idxs_prev = np.full(N, -1, dtype=int)

            for k, beam_idx in enumerate(beam_idxs):
                if is_new_point[k]:
                    vel_frame[beam_idx, :] = np.nan
                    continue
                used = False
                if input_point_vels is not None:
                    try:
                        v_in = input_point_vels[t][beam_idx]
                        if not np.any(np.isnan(v_in)):
                            vel_frame[beam_idx,:] = v_in
                            used = True
                    except Exception:
                        pass
                if used:
                    continue
                prev_idx = idxs_prev[k]
                if (prev_idx >= 0) and (dists_to_prev[k] <= CFG["VEL_MATCH_DIST_MAX"]) and (dt > 1e-6):
                    p_prev = self.prev_points_world[prev_idx]
                    p_curr = pts_world[k]
                    v_world = (p_curr - p_prev) / dt
                    ego_v_body = np.array([float(ego_twist[0]), float(ego_twist[1])])
                    ego_yaw = float(ego_pose[2])
                    V_ego_world = rotate_body_to_world(ego_v_body, ego_yaw).flatten()
                    v_rel_world = v_world - V_ego_world
                    v_rel_ego = rotate_world_to_body(v_rel_world, ego_yaw)
                    vel_frame[beam_idx,:] = v_rel_ego
                    continue
                vel_frame[beam_idx,:] = rigid_vels[k]

            # 6) final labels
            labels_this = np.zeros(BEAMS, dtype=np.uint8)
            for k, beam_idx in enumerate(beam_idxs):
                if is_new_point[k]:
                    labels_this[beam_idx] = 2
                else:
                    if seg_is_dynamic.get(seg_labels[k], False):
                        labels_this[beam_idx] = 1
                    else:
                        labels_this[beam_idx] = 0

            segid_full = np.full(BEAMS, -1, dtype=int)
            for k, beam_idx in enumerate(beam_idxs):
                segid_full[beam_idx] = int(seg_gid_map.get(seg_labels[k], (-1, False))[0])

            labels_out[t] = labels_this
            vel_out[t] = vel_frame
            segid_out[t] = segid_full

            # update prev_points_world (global list) for ballooning next frame
            self.prev_points_world = pts_world.copy()
            self.prev_valid_beam_idxs = beam_idxs.copy()
            self.prev_timestamp = ts if ts is not None else (self.prev_timestamp + dt if self.prev_timestamp is not None else None)
            self.frame_counter += 1

            if (t % 50) == 0:
                logging.debug(f"frame {t}/{T} valid={N} new={int(np.sum(is_new_point))}")

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
