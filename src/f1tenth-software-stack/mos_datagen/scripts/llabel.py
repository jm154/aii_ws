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

# --- 설정 ---
MAP_YAML_PATH = "/home/ugrp/aii_ws/src/f1tenth_gym_ros/maps/E1_out2_obs2.yaml" 
INPUT_DIR = "../dataset_vel_label"          # 원본 데이터 (DataLogger + 1차 라벨링 결과)
OUTPUT_DIR = "../dataset_vel_label_final"   # 최종 저장 폴더

# 파라미터
DILATION_PIXELS = 9       # 벽 두께 (약 0.75m)
FORCE_THRESH = 255        # 맵 임계값 (필요하면 조정)
NEW_POINT_THRESHOLD = 0.5 # 스캔 매칭 임계값 (m)
DT_DEFAULT = 0.1

logging.basicConfig(level=logging.INFO, format='%(message)s')


class LabelRefiner:
    def __init__(self):
        self.load_map()
        # LiDAR 각도 (ego local 기준) – 실제 센서와 맞춰줘야 함
        self.angles = np.linspace(-2.35619, 2.35619, 1080)
        
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            logging.info(f"📁 새로운 폴더 생성: {OUTPUT_DIR}")

    # ---------------- Map 로드 ----------------
    def load_map(self):
        logging.info(f"🗺️ 맵 로드 중: {MAP_YAML_PATH}")
        with open(MAP_YAML_PATH, 'r') as f:
            map_info = yaml.safe_load(f)

        self.res = map_info['resolution']
        self.origin_x = map_info['origin'][0]
        self.origin_y = map_info['origin'][1]
        
        img_path = map_info['image']
        if not os.path.isabs(img_path):
            img_path = os.path.join(os.path.dirname(MAP_YAML_PATH), img_path)
            
        map_img = np.array(Image.open(img_path).convert('L'))
        self.h, self.w = map_img.shape
        
        # [주의] FORCE_THRESH = 255 이면 "255보다 작은 모든 픽셀"이 벽으로 간주됨
        binary_walls = (map_img < FORCE_THRESH)
        structure = np.ones((DILATION_PIXELS, DILATION_PIXELS), dtype=bool)
        self.dilated_map = scipy.ndimage.binary_dilation(binary_walls, structure=structure)
        logging.info(f"✅ 맵 준비 완료 (Dilation: {DILATION_PIXELS} px)")

    # ---------------- 공용 함수들 ----------------
    def compensate_motion(self, points, twist, dt):
        """
        이전 프레임의 점들을 '현재 ego 좌표계'로 보정하기 위한 motion compensation.
        twist: 현재 프레임 ego_twist [vx, vy, wz]
        dt   : 현재 프레임과 이전 프레임 사이 시간
        """
        vx, vy, wz = twist[0], twist[1], twist[2]
        dx, dy, dth = vx * dt, vy * dt, wz * dt

        c, s = np.cos(-dth), np.sin(-dth)
        R = np.array([[c, -s], [s, c]])

        # 회전 + 병진 보정
        points_rot = points @ R.T
        points_trans = points_rot - np.array([dx, dy])
        return points_trans

    def lidar_to_local(self, ranges):
        """
        LiDAR range를 ego local (x, y)로 변환
        """
        valid = (ranges > 0.01) & (ranges < 30.0)
        x = ranges * np.cos(self.angles)
        y = ranges * np.sin(self.angles)
        return np.stack([x, y], axis=1), valid

    # ---------------- 전체 폴더 처리 ----------------
    def process_folder(self):
        files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.npz")))
        if not files:
            logging.error("❌ 원본 파일이 없습니다.")
            return
        logging.info(f"📂 총 {len(files)}개 파일 처리 시작...")
        for f in files:
            self.refine_file(f)
        logging.info(f"🎉 모든 작업 완료! 저장 위치: {OUTPUT_DIR}")

    # ---------------- 개별 파일 처리 ----------------
    def refine_file(self, filepath):
        try:
            data = dict(np.load(filepath))
        except Exception as e:
            logging.error(f"파일 로드 에러: {e}")
            return

        ranges = data['ranges']            # (T, 1080)
        ego_pose = data['ego_pose']        # (T, 3)  [x, y, yaw]
        ego_twist = data['ego_twist']      # (T, 3)  [vx, vy, wz]
        labels = data['labels']            # (T, 1080) 0=static,1=dynamic,2=new(이전 단계 기준)
        velocities = data['point_velocities']  # (T, 1080, 2)
        timestamps = data.get('timestamps', None)

        num_frames = len(ranges)
        modified_static = 0   # 맵 때문에 static으로 바뀐 개수
        modified_new = 0      # 스캔 매칭 때문에 new로 새로 찍힌 개수
        
        # 스캔 비교용 '이전 프레임 local points'
        prev_points_valid = None   # (N_prev, 2)

        for i in range(num_frames):
            scan = ranges[i]
            pose = ego_pose[i]
            twist = ego_twist[i]
            
            # dt 계산
            dt = DT_DEFAULT
            if timestamps is not None and i > 0:
                dt = timestamps[i] - timestamps[i - 1]
                if dt <= 0:
                    dt = DT_DEFAULT

            # 1) LiDAR local 좌표 변환
            points_local, valid_mask = self.lidar_to_local(scan)
            valid_indices = np.where(valid_mask)[0]    # 유효 빔 인덱스
            points_valid = points_local[valid_mask]    # (N_valid, 2)

            # ---------------------------------------------------
            # Step A: [Scan-based New]  프레임 간 매칭으로만 New(2) 정의
            # ---------------------------------------------------
            if prev_points_valid is not None and len(points_valid) > 0:
                # 이전 프레임 점들을 "현재 프레임 ego 좌표계"로 보정
                prev_aligned = self.compensate_motion(prev_points_valid, twist, dt)

                if len(prev_aligned) > 0:
                    tree = KDTree(prev_aligned)
                    dists, _ = tree.query(points_valid)

                    # 이전 프레임에 근접한 점이 없으면 "새로 나타난 점"으로 간주
                    is_new = (dists.flatten() > NEW_POINT_THRESHOLD)
                    new_indices = valid_indices[is_new]

                    for idx in new_indices:
                        # 🔹 여기서 "New" 여부는 오직 스캔 매칭 기준으로만 결정
                        #    (맵에 있는지 없는지는 고려하지 않음)
                        if labels[i][idx] == 0:
                            labels[i][idx] = 2  # New
                            velocities[i][idx, :] = 0.0
                            modified_new += 1

            # 다음 프레임을 위한 스캔 저장 (현재 프레임 local 좌표 그대로 저장)
            if len(points_valid) > 0:
                prev_points_valid = points_valid.copy()
            else:
                prev_points_valid = None

            # ---------------------------------------------------
            # Step B: [Map-based Wall Filtering]
            #        맵 상에서 벽인 픽셀 ↔ Static으로 강제
            #        단, "New(2)"는 스캔 기준 정의를 존중해서 건드리지 않음.
            # ---------------------------------------------------
            # ego pose 기준 world 좌표로 투영
            c, s = math.cos(pose[2]), math.sin(pose[2])
            x_map = (points_local[:, 0] * c - points_local[:, 1] * s) + pose[0]
            y_map = (points_local[:, 0] * s + points_local[:, 1] * c) + pose[1]

            # world → map pixel (u, v)
            u = ((x_map - self.origin_x) / self.res).astype(int)
            v = (self.h - 1 - (y_map - self.origin_y) / self.res).astype(int)

            in_map = (u >= 0) & (u < self.w) & (v >= 0) & (v < self.h)

            is_wall_pixel = np.zeros_like(valid_mask, dtype=bool)
            check_mask = valid_mask & in_map

            if np.any(check_mask):
                is_wall_pixel[check_mask] = self.dilated_map[v[check_mask], u[check_mask]]

            # ➤ "벽 픽셀"이면서, 라벨이 static(0) 또는 dynamic(1)인 경우만 static으로 덮어쓰기
            #    ✅ 라벨 2(New)는 스캔 기반 정의를 유지하기 위해 보호
            to_fix = is_wall_pixel & ((labels[i] == 0) | (labels[i] == 1))

            if np.any(to_fix):
                count = np.sum(to_fix)
                modified_static += count

                labels[i][to_fix] = 0

                # 벽은 world 기준 정지 → ego 기준 속도는 -ego 속도
                v_ego_x = twist[0]
                v_ego_y = twist[1]
                velocities[i][to_fix, 0] = -v_ego_x
                velocities[i][to_fix, 1] = -v_ego_y

        # ---------------- 저장 ----------------
        data['labels'] = labels
        data['point_velocities'] = velocities

        filename = os.path.basename(filepath)
        save_path = os.path.join(OUTPUT_DIR, filename)
        np.savez_compressed(save_path, **data)
        logging.info(f" -> {filename}: Wall Fix {modified_static}, New Created {modified_new}")


if __name__ == "__main__":
    refiner = LabelRefiner()
    refiner.process_folder()

