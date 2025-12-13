import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import sys

# -------- 설정 --------
DATA_DIR = os.path.expanduser("../dataset_l/2.5ms")
FRAME_STEP = 10
NUM_BEAMS = 1080
FOV = 4.71238898
ANGLES = np.linspace(-FOV/2, FOV/2, NUM_BEAMS)


# ==============================
#  Utility
# ==============================
def polar_to_xy(ranges):
    valid = (ranges > 0.01) & (ranges < 30.0)
    xs = ranges[valid] * np.cos(ANGLES[valid])
    ys = ranges[valid] * np.sin(ANGLES[valid])
    return np.stack([xs, ys], axis=1), valid


# ---------------------------
# 정확한 SE2 변환 (Dataset 동일)
# ---------------------------
def warp_prev_to_curr(prev_ranges, pose_prev, pose_curr):
    x_p, y_p, th_p = pose_prev
    x_c, y_c, th_c = pose_curr

    c_p, s_p = np.cos(th_p), np.sin(th_p)
    R_p = np.array([[c_p, -s_p],
                    [s_p,  c_p]])

    c_c, s_c = np.cos(th_c), np.sin(th_c)
    R_cT = np.array([[ c_c, s_c],
                     [-s_c, c_c]])

    valid = (prev_ranges > 0.01) & (prev_ranges < 30.0)
    if not np.any(valid):
        return np.zeros((0, 2)), valid, np.zeros(0)

    r = prev_ranges[valid]
    ang = ANGLES[valid]

    pts_prev = np.stack([r * np.cos(ang), r * np.sin(ang)], axis=1)
    pts_world = (R_p @ pts_prev.T).T + np.array([x_p, y_p])
    pts_curr = (R_cT @ (pts_world - np.array([x_c, y_c])).T).T

    ang_curr = np.arctan2(pts_curr[:, 1], pts_curr[:, 0])
    ang_curr_norm = ang_curr / (FOV / 2)

    return pts_curr, valid, ang_curr_norm


# ----------------- ego_twist 적분 -----------------
def wrap_angle(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi

def integrate_ego_twist(pose_prev, twist, dt):
    x, y, yaw = pose_prev
    vx, vy, w = twist
    dx_world = vx * np.cos(yaw) - vy * np.sin(yaw)
    dy_world = vx * np.sin(yaw) + vy * np.cos(yaw)
    x_new = x + dx_world * dt
    y_new = y + dy_world * dt
    yaw_new = wrap_angle(yaw + w * dt)
    return np.array([x_new, y_new, yaw_new])


# ==============================
#  MAIN VIEWER
# ==============================
class ModelInputViewer:
    def __init__(self, data_dir):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if not self.files:
            print("❌ no .npz files found")
            sys.exit(0)

        print(f"📂 Found {len(self.files)} files")

        self.file_idx = 0
        self.load_file()

        self.frame_idx = FRAME_STEP

        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.update_plot()
        plt.show()

    def load_file(self):
        self.data = np.load(self.files[self.file_idx])
        self.ranges = self.data["ranges"]
        self.ego_twist = self.data["ego_twist"]
        pose_init = self.data["ego_pose"][0]

        # twist 적분으로 pose 계산
        self.poses = [pose_init]
        for t in range(1, len(self.ranges)):
            dt = 0.04
            if "timestamps" in self.data:
                dt = max(self.data["timestamps"][t] - self.data["timestamps"][t-1], 1e-4)
            self.poses.append(integrate_ego_twist(self.poses[-1], self.ego_twist[t-1], dt))
        self.poses = np.stack(self.poses, axis=0)
        self.total_frames = len(self.ranges)
        print(f"📄 Loaded {self.files[self.file_idx]} (pose integrated from twist)")

    def on_key(self, event):
        if event.key == "right":
            self.frame_idx = min(self.frame_idx + 1, self.total_frames - 1)
        elif event.key == "left":
            self.frame_idx = max(self.frame_idx - 1, FRAME_STEP)
        elif event.key == "down":
            self.file_idx = (self.file_idx + 1) % len(self.files)
            self.load_file()
            self.frame_idx = FRAME_STEP
        elif event.key == "up":
            self.file_idx = (self.file_idx - 1) % len(self.files)
            self.load_file()
            self.frame_idx = FRAME_STEP
        elif event.key == "q":
            plt.close()
            return
        self.update_plot()

    def update_plot(self):
        self.ax.clear()
        t = self.frame_idx
        tp = t - FRAME_STEP
        if tp < 0:
            return

        curr_xy, _ = polar_to_xy(self.ranges[t])

        warped_xy, valid_prev, ang_norm_prev = warp_prev_to_curr(
            self.ranges[tp],
            self.poses[tp],
            self.poses[t]
        )

        r_curr = self.ranges[t][valid_prev]
        r_prev_warp = np.linalg.norm(warped_xy, axis=1)
        residual = np.abs(r_curr - r_prev_warp)
        res_norm = (residual - residual.min()) / (residual.ptp() + 1e-6)

        # plot current
        self.ax.scatter(curr_xy[:, 0], curr_xy[:, 1],
                        s=5, c="black", alpha=1.0,
                        label=f"Curr t={t}")

        # plot warped prev
        self.ax.scatter(warped_xy[:, 0], warped_xy[:, 1],
                        s=8, c=res_norm, cmap="coolwarm",
                        alpha=0.9, label="Prev→Curr warped")

        # optional arrows
        for (x, y), a in zip(warped_xy[::40], ang_norm_prev[::40]):
            self.ax.arrow(x, y, 0.2*np.cos(a*np.pi), 0.2*np.sin(a*np.pi),
                          width=0.01, color="green", alpha=0.4)

        self.ax.set_title(
            f"Model Input Visualization (SE2 Correct, twist integrated) | "
            f"{os.path.basename(self.files[self.file_idx])} | Frame {t}"
        )
        self.ax.grid(True)
        self.ax.axis("equal")
        self.ax.set_xlim(-15, 15)
        self.ax.set_ylim(-15, 15)
        self.ax.legend(loc="upper right")
        self.fig.canvas.draw()


if __name__ == "__main__":
    ModelInputViewer(DATA_DIR)

