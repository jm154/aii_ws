import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import sys

from sklearn.cluster import DBSCAN
from matplotlib import cm

# ⭐️ 라벨링된 데이터 경로
DATA_DIR = os.path.expanduser("../VALIDATION/validation") # 경로 확인 필요

# ⭐️ 화살표 길이 스케일 (작을수록 화살표가 짧아짐)
V_SCALE = 0.5

# DBSCAN (세그먼트 재생성 시 사용)
DBSCAN_EPS = 0.2
DBSCAN_MIN_SAMPLES = 3

class LabelViewer:
    def __init__(self, data_dir):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if not self.files:
            print(f"경로에 .npz 파일이 없습니다: {data_dir}")
            sys.exit(0)

        print(f"총 {len(self.files)}개의 파일을 찾았습니다.")

        self.file_idx = 0
        self.frame_idx = 0
        self.angles = np.linspace(-2.35619, 2.35619, 1080)

        # view mode: 'point' or 'segment'
        self.view_mode = 'segment'

        # dynamic visual params
        self.v_scale = V_SCALE

        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        self.load_current_file()
        self.update_plot()
        plt.show()

    def load_current_file(self):
        target_file = self.files[self.file_idx]
        filename = os.path.basename(target_file)
        print(f"🔄 Loading: {filename}")

        try:
            data = np.load(target_file, allow_pickle=True)
            # ranges: (T, 1080)
            self.ranges = data['ranges']
            # labels might be (T,1080) per-point state (0/1/2) OR already per-segment labels.
            # We'll expect per-point state named 'labels' (common), and optional segment ids
            self.labels = data['labels'] if 'labels' in data else None

            # segment id per point if saved by labeling pipeline (preferred)
            # Try common key names.
            seg_key = None
            for k in ('segment_id_per_point', 'segment_ids', 'segment_id'):
                if k in data:
                    seg_key = k
                    break
            self.segment_ids_saved = data[seg_key] if seg_key is not None else None

            if 'point_velocities' in data:
                self.velocities = data['point_velocities']
            else:
                # default zeros: shape (T,1080,2)
                self.velocities = np.zeros((len(self.ranges), 1080, 2))

            self.total_frames = len(self.ranges)
            self.frame_idx = 0

        except Exception as e:
            print(f"🚨 로드 실패: {e}")
            self.ranges = np.zeros((0,1080))
            self.labels = None
            self.velocities = np.zeros((0,1080,2))
            self.total_frames = 0

    def on_key(self, event):
        key = event.key
        if key == 'right':
            self.frame_idx = (self.frame_idx + 1) % self.total_frames
            self.update_plot()
        elif key == 'left':
            self.frame_idx = (self.frame_idx - 1) % self.total_frames
            self.update_plot()
        elif key == 'down':
            self.file_idx = (self.file_idx + 1) % len(self.files)
            self.load_current_file()
            self.update_plot()
        elif key == 'up':
            self.file_idx = (self.file_idx - 1) % len(self.files)
            self.load_current_file()
            self.update_plot()
        elif key in ('escape', 'q'):
            plt.close()
        elif key == 's':
            # toggle view mode
            self.view_mode = 'point' if self.view_mode == 'segment' else 'segment'
            print(f"🔁 view_mode -> {self.view_mode}")
            self.update_plot()
        elif key == '+':
            self.v_scale = max(0.01, self.v_scale - 0.05)
            self.update_plot()
        elif key == '-':
            self.v_scale = self.v_scale + 0.05
            self.update_plot()

    def infer_segments_if_needed(self, points_valid):
        """
        If the .npz does not contain segment ids, cluster the valid points
        and return an array of per-valid-point segment labels.
        """
        if points_valid.shape[0] == 0:
            return np.array([], dtype=int)
        clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(points_valid)
        return clustering.labels_.astype(int)  # -1 indicates noise

    def update_plot(self):
        self.ax.clear()
        if self.total_frames == 0:
            self.ax.set_title("No frames")
            self.fig.canvas.draw()
            return

        scan = self.ranges[self.frame_idx]
        label = self.labels[self.frame_idx] if (self.labels is not None) else None
        vel = self.velocities[self.frame_idx]

        valid = (scan > 0.01) & (scan < 30.0)
        idx_valid = np.where(valid)[0]
        x_all = scan * np.cos(self.angles)
        y_all = scan * np.sin(self.angles)
        x = x_all[valid]
        y = y_all[valid]
        v = vel[valid]  # (N_valid, 2)
        l = label[valid] if label is not None else None

        # --- segment ids: prefer saved ones, else infer ---
        if self.segment_ids_saved is not None:
            seg_all = self.segment_ids_saved[self.frame_idx]
            seg_valid = seg_all[valid]
        else:
            seg_valid = self.infer_segments_if_needed(np.stack([x, y], axis=1))

        if len(seg_valid) != len(x):
            seg_valid = self.infer_segments_if_needed(np.stack([x, y], axis=1))

        unique_segs = np.unique(seg_valid)
        unique_segs_sorted = np.sort(unique_segs)
        cmap = cm.get_cmap('tab20', max(2, len(unique_segs_sorted)))
        seg_to_color = {sg: cmap(i % cmap.N) for i, sg in enumerate(unique_segs_sorted)}

        # compute per-segment majority state if labels present
        seg_state = {}
        if l is not None:
            for sg in unique_segs_sorted:
                mask = (seg_valid == sg)
                if np.sum(mask) == 0:
                    seg_state[sg] = -1
                else:
                    vals, counts = np.unique(l[mask], return_counts=True)
                    seg_state[sg] = vals[np.argmax(counts)]
        else:
            for sg in unique_segs_sorted:
                seg_state[sg] = 1 if sg != -1 else 0

        # --- Count diagnostics (prints) ---
        n_new = int(np.sum(l == 2)) if l is not None else 0
        n_dyn = int(np.sum(l == 1)) if l is not None else 0
        n_stat = int(np.sum(l == 0)) if l is not None else 0
        print(f"[File {self.file_idx} Frame {self.frame_idx}] Static:{n_stat} Dynamic:{n_dyn} New:{n_new} Segments:{len(unique_segs_sorted)}")

        # --- DRAWING ---
        if self.view_mode == 'point':
            # ... 기존 Point 모드 코드 유지 ...
            if l is None:
                for sg in unique_segs_sorted:
                    mask = seg_valid == sg
                    col = seg_to_color[sg]
                    self.ax.scatter(x[mask], y[mask], s=6, color=col, alpha=0.7, label=f"seg {sg}")
            else:
                static_mask = (l == 0)
                dynamic_mask = (l == 1)
                new_mask = (l == 2)

                if np.any(static_mask):
                    self.ax.scatter(x[static_mask], y[static_mask], s=2, c='gray', label='Static', alpha=0.3)
                    self.ax.quiver(x[static_mask][::20], y[static_mask][::20],
                                   v[static_mask][::20, 0], v[static_mask][::20, 1],
                                   color='black', alpha=0.3,
                                   angles='xy', scale_units='xy', scale=(1.0/self.v_scale), width=0.002)

                if np.any(dynamic_mask):
                    self.ax.scatter(x[dynamic_mask], y[dynamic_mask], s=20, c='red', label='Dynamic')
                    self.ax.quiver(x[dynamic_mask][::5], y[dynamic_mask][::5],
                                   v[dynamic_mask][::5, 0], v[dynamic_mask][::5, 1],
                                   color='green',
                                   angles='xy', scale_units='xy', scale=(1.0/self.v_scale), width=0.005)

                if np.any(new_mask):
                    self.ax.scatter(x[new_mask], y[new_mask], s=40, c='blue', label='New', alpha=0.95, edgecolors='white', linewidths=0.6, zorder=10)

        else:
            # ⭐️ [수정됨] Segment View Mode: 텍스트 추가
            for sg in unique_segs_sorted:
                mask = seg_valid == sg
                if np.sum(mask) == 0:
                    continue
                col = seg_to_color[sg]
                state = seg_state.get(sg, -1)
                
                # Style based on State
                if state == 0: # Static
                    edge = 'k'; alpha = 0.25; z = 1
                elif state == 1: # Dynamic
                    edge = 'r'; alpha = 0.9; z = 3
                elif state == 2: # New
                    edge = 'b'; alpha = 0.9; z = 4
                else:
                    edge = 'gray'; alpha = 0.7; z = 1

                # Draw Points
                self.ax.scatter(x[mask], y[mask], s=8, color=col, alpha=0.6, edgecolors=edge, linewidths=0.6, zorder=z)
                
                # Calculate Centroid & Average Velocity
                cx = np.mean(x[mask]); cy = np.mean(y[mask])
                
                # NaN Velocity Handling (for New points)
                v_cluster = v[mask]
                if np.isnan(v_cluster).any():
                    avg_v = np.array([0.0, 0.0])
                    speed = 0.0
                    is_nan_vel = True
                else:
                    avg_v = np.mean(v_cluster, axis=0) if v_cluster.shape[0] > 0 else np.array([0.0, 0.0])
                    speed = np.linalg.norm(avg_v)
                    is_nan_vel = False

                # Draw Velocity Arrow (Centroid)
                self.ax.plot(cx, cy, marker='o', markersize=6, color='k', markeredgecolor='w', zorder=5)
                if not is_nan_vel and speed > 0.01:
                    self.ax.quiver(cx, cy, avg_v[0], avg_v[1],
                                   angles='xy', scale_units='xy', scale=(1.0/self.v_scale),
                                   width=0.006, zorder=6, color='black')

                # ⭐️ [Text Label] ID, State, Speed
                state_str = {0: 'S', 1: 'D', 2: 'N'}.get(state, '?')
                
                if is_nan_vel:
                    vel_text = "Vel: NaN"
                else:
                    vel_text = f"{speed:.2f} m/s\n({avg_v[0]:.1f}, {avg_v[1]:.1f})"
                
                info_text = f"ID:{sg} [{state_str}]\n{vel_text}"

                self.ax.text(cx, cy, info_text, fontsize=8, zorder=10, 
                             ha='center', va='center',
                             bbox=dict(facecolor='white', alpha=0.7, pad=2, edgecolor='gray'))

            # Overlay New Points
            if l is not None:
                new_mask = (l == 2)
                if np.any(new_mask):
                    self.ax.scatter(x[new_mask], y[new_mask], s=60, c='cyan', edgecolors='k', linewidths=0.8, zorder=12, label='New (overlay)')

        # cosmetics
        self.ax.set_title(f"File: {self.file_idx} | Frame: {self.frame_idx} | Mode: {self.view_mode}")
        self.ax.set_aspect('equal')
        self.ax.grid(True, linestyle='--', alpha=0.5)
        self.ax.set_xlim(-10, 10); self.ax.set_ylim(-10, 10)
        self.ax.plot(0, 0, 'k^', markersize=10, label='Ego')

        # legend
        if self.view_mode == 'point' and l is not None:
            self.ax.legend(loc='upper right')
        
        self.fig.canvas.draw()


if __name__ == "__main__":
    # 데이터 경로를 실제 경로로 수정해서 사용하세요
    viewer = LabelViewer(DATA_DIR)
