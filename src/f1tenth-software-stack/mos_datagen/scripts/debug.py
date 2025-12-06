import numpy as np
import os
import glob

# --- 경로 설정 ---
DIR_BEFORE = "../dataset_vel_label"        # 수정 전 폴더
DIR_AFTER  = "../dataset_vel_label_final"  # 수정 후 폴더

def compare():
    files = sorted(glob.glob(os.path.join(DIR_BEFORE, "*.npz")))
    if not files: print("❌ 원본 파일 없음"); return

    print(f"🔍 데이터 비교 시작 ({len(files)}개 파일)")
    print("-" * 60)
    print(f"{'Filename':<15} | {'Changed Total':<15} | {'1->0 (Wall Fix)':<15} | {'0->2 (New)':<15}")
    print("-" * 60)

    for f_before in files:
        filename = os.path.basename(f_before)
        f_after = os.path.join(DIR_AFTER, filename)
        
        if not os.path.exists(f_after):
            print(f"⚠️ {filename}: 수정 후 파일이 없음")
            continue
            
        # 데이터 로드
        data_b = np.load(f_before)
        data_a = np.load(f_after)
        
        lb_b = data_b['labels']
        lb_a = data_a['labels']
        
        # 1. 전체 바뀐 개수
        diff_mask = (lb_b != lb_a)
        total_diff = np.sum(diff_mask)
        
        # 2. Dynamic(1) -> Static(0) (벽 보정)
        wall_fix = np.sum((lb_b == 1) & (lb_a == 0))
        
        # 3. Static(0) -> New(2) (새로운 점 감지)
        new_created = np.sum((lb_b == 0) & (lb_a == 2))
        
        print(f"{filename:<15} | {total_diff:<15} | {wall_fix:<15} | {new_created:<15}")

if __name__ == "__main__":
    compare()
