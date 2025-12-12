import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import sys
import math
import colorsys

# --- 설정 ---
DATA_DIR = os.path.expanduser("../dataset_l/5ms") 
FRAME_STEP = 10           # 프레임 간격 (t-10, t-20, ...)
HISTORY_DEPTH = 5        # 누적할 과거 프레임 개수 (t-10 ~ t-50)
MIN_START_FRAME = HISTORY_DEPTH * FRAME_STEP # 최소 시작 프레임 (50)

class CumulativeAlignmentViewer:
    def __init__(self, data_dir):
        # 1. 파일 로드
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if not self.files:
            print(f"❌ 경로에 .npz 파일이 없습니다: {data_dir}")
            sys.exit(0)
        
        print(f"📂 총 {len(self.files)}개의 파일을 찾았습니다.")
        
        # 라이다 설정
        self.num_beams = 1080
        self.fov = 4.71238898
        self.angles = np.linspace(-self.fov/2, self.fov/2, self.num_beams)

        # 상태 변수
        self.file_idx = 0
        self.frame_idx = MIN_START_FRAME 
        
        # 플롯 설정
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        self.load_file()
        self.update_plot()
        plt.show()

    def load_file(self):
        self.data = np.load(self.files[self.file_idx])
        self.total_frames = len(self.data['ranges'])
        self.filename = os.path.basename(self.files[self.file_idx])
        
        self.frame_idx = max(MIN_START_FRAME, min(self.frame_idx, self.total_frames - 1))
        print(f"📄 Loading: {self.filename} ({self.total_frames} frames)")

    def on_key(self, event):
        if event.key == 'right':
            self.frame_idx = min(self.frame_idx + 1, self.total_frames - 1)
            self.update_plot()
        elif event.key == 'left':
            self.frame_idx = max(self.frame_idx - 1, MIN_START_FRAME)
            self.update_plot()
        elif event.key == 'down':
            self.file_idx = (self.file_idx + 1) % len(self.files)
            self.frame_idx = MIN_START_FRAME
            self.load_file()
            self.update_plot()
        elif event.key == 'up':
            self.file_idx = (self.file_idx - 1) % len(self.files)
            self.frame_idx = MIN_START_FRAME
            self.load_file()
            self.update_plot()
        elif event.key == 'q':
            plt.close()

    def polar_to_xy(self, ranges):
        valid = (ranges > 0.01) & (ranges < 30.0)
        x = ranges[valid] * np.cos(self.angles[valid])
        y = ranges[valid] * np.sin(self.angles[valid])
        return np.stack([x, y], axis=1), valid

    def get_transform_matrix(self, pose_curr, pose_prev):
        """t_prev -> t_curr 변환 행렬 계산"""
        def get_mat(p):
            x, y, th = p
            c, s = np.cos(th), np.sin(th)
            return np.array([[c, -s, x], [s, c, y], [0, 0, 1]])
        
        H_c = get_mat(pose_curr)
        H_p = get_mat(pose_prev)
        return np.linalg.inv(H_c) @ H_p 

    def warp_points(self, ranges_prev, H_rel):
        """변환 행렬을 사용하여 점들을 워핑"""
        valid_p = (ranges_prev > 0.01) & (ranges_prev < 30.0)
        r_p = ranges_prev[valid_p]
        th_p = self.angles[valid_p]
        x_p = r_p * np.cos(th_p)
        y_p = r_p * np.sin(th_p)
        
        if len(x_p) == 0: return np.zeros((0, 2))

        ones = np.ones_like(x_p)
        pts_prev_homo = np.stack([x_p, y_p, ones], axis=0)
        
        pts_prev_in_curr = H_rel @ pts_prev_homo
        
        return pts_prev_in_curr[:2, :].T

    def update_plot(self):
        self.ax.clear()
        t = self.frame_idx
        
        # 1. 현재 프레임 데이터 로드 및 기준 포즈 설정
        ranges_curr = self.data['ranges'][t]
        pose_curr = self.data['ego_pose'][t]
        
        # 2. 현재 점 (t) 시각화
        points_curr, _ = self.polar_to_xy(ranges_curr)
        self.ax.scatter(points_curr[:, 0], points_curr[:, 1], 
                        c='black', s=5, alpha=1.0, label='Current t')
        
        # 3. 과거 프레임들을 누적
        for i in range(1, HISTORY_DEPTH + 1):
            t_prev = t - i * FRAME_STEP
            
            ranges_prev = self.data['ranges'][t_prev]
            pose_prev = self.data['ego_pose'][t_prev]
            
            # T_rel = Pose(t) <- Pose(t_prev)
            H_rel = self.get_transform_matrix(pose_curr, pose_prev)
            points_warped = self.warp_points(ranges_prev, H_rel)
            
            # 색상: i=1 (t-10)은 진한 Cyan, i=5 (t-50)은 연한 색으로 설정
            hue = 0.5 - (i * 0.05) # 색조
            saturation = 1.0 - (i * 0.1) # 채도 (멀수록 흐리게)
            r, g, b = colorsys.hsv_to_rgb(0.5, 1.0, saturation) # 0.5는 Cyan 계열
            
            self.ax.scatter(points_warped[:, 0], points_warped[:, 1], 
                            c=[(r, g, b)], s=3, alpha=0.3, 
                            label=f'Warped t-{i*FRAME_STEP}' if i == 1 else "")

        # --- 시각화 설정 ---
        self.ax.set_title(f"Cumulative Alignment: File={self.filename} | Base Frame={t} (Total Depth: {HISTORY_DEPTH*FRAME_STEP} frames)")
        self.ax.plot(0, 0, 'k^', markersize=15, label='Ego')
        self.ax.grid(True)
        self.ax.axis('equal')
        self.ax.set_xlim(-15, 15)
        self.ax.set_ylim(-15, 15)
        self.ax.legend(loc='upper right')
        
        self.fig.canvas.draw()

if __name__ == "__main__":
    viewer = CumulativeAlignmentViewer(DATA_DIR)
