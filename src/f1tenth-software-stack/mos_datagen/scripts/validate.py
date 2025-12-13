#!/usr/bin/env python3
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

# model.py와 dataset.py가 같은 폴더(혹은 python path)에 있어야 합니다.
from model import ClusterFlowNet
from dataset import ClusterDataset

# ---------------- hyperparams ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
NUM_WORKERS = 4
# ---------------------------------------------

def evaluate(model, dataloader):
    model.eval()
    
    # 저장할 리스트들
    static_errors = []
    dynamic_errors = []
    
    static_preds = []
    dynamic_preds = []
    
    static_gts = []
    dynamic_gts = []

    print("🚀 검증 시작 (정지한 Dynamic 객체 제외)...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Inference")):
            # 1. 데이터 언패킹
            curr_in = batch[0].to(DEVICE)
            prev_in = batch[1].to(DEVICE)
            ego_vector = batch[2].to(DEVICE)
            raw_ego_vel = batch[3].to(DEVICE) # Clean Ego Velocity
            target_vel = batch[4].to(DEVICE)  # GT Relative Velocity
            class_label = batch[5].to(DEVICE) # 0: Static, 1: Dynamic

            # 2. 유효 데이터 마스킹 (NaN 제거)
            mask = ~torch.isnan(target_vel).any(dim=1)
            if mask.sum() == 0: continue

            curr_in = curr_in[mask]
            prev_in = prev_in[mask]
            ego_vector = ego_vector[mask]
            valid_raw_ego = raw_ego_vel[mask]
            valid_target_rel = target_vel[mask]
            
            # 라벨 1차원화
            valid_labels = class_label[mask].view(-1) 

            # 3. 모델 추론
            output = model(curr_in, prev_in, ego_vector, valid_raw_ego)
            
            if isinstance(output, tuple):
                pred_rel = output[0]
                if len(output) >= 2:
                    pred_abs = output[1]
                else:
                    pred_abs = pred_rel + valid_raw_ego
            else:
                pred_rel = output
                pred_abs = pred_rel + valid_raw_ego

            # 4. 비교 대상: 절대 속도 (Absolute Velocity)
            gt_abs = valid_target_rel + valid_raw_ego
            
            # 5. 속도 및 오차 계산 (L2 Norm)
            speed_error = torch.norm(pred_abs - gt_abs, dim=1) # 오차
            pred_speed = torch.norm(pred_abs, dim=1)           # 예측 속력
            gt_speed = torch.norm(gt_abs, dim=1)               # 정답 속력

            # 6. 리스트로 변환 및 분리 저장
            speed_error = speed_error.cpu().numpy()
            pred_speed = pred_speed.cpu().numpy()
            gt_speed = gt_speed.cpu().numpy()
            labels = valid_labels.cpu().numpy()

            for i in range(len(labels)):
                err = speed_error[i]
                pred = pred_speed[i]
                gt = gt_speed[i]
                
                if labels[i] > 0.5: # Dynamic (라벨상 동적 객체)
                    # ⭐️ [핵심 수정] GT 속도가 거의 0인(0.1 미만) 경우 통계에서 제외
                    # "멈춰 있는 차" 때문에 그래프가 왜곡되는 것을 방지
                    if gt < 0.1:
                        continue

                    dynamic_errors.append(err)
                    dynamic_preds.append(pred)
                    dynamic_gts.append(gt)
                else: # Static (벽, 기둥 등)
                    static_errors.append(err)
                    static_preds.append(pred)
                    static_gts.append(gt)

    return (np.array(static_errors), np.array(dynamic_errors), 
            np.array(static_preds), np.array(dynamic_preds),
            np.array(static_gts), np.array(dynamic_gts))

def visualize_results(static_err, dynamic_err, static_preds, dynamic_preds, static_gts, dynamic_gts):
    print("\n" + "="*80)
    print("📊 Validation Statistics (Moving Dynamic Objects Only)")
    print("="*80)

    # --- Static Statistics ---
    print(f"[Static Objects] (Count: {len(static_err)})")
    if len(static_err) > 0:
        print(f"  - GT Speed Mean : {np.mean(static_gts):.4f} m/s")
        print(f"  - Pred Speed Mean: {np.mean(static_preds):.4f} m/s")
        print(f"  - Speed Error Mean: {np.mean(static_err):.4f} m/s")
        print(f"  - Speed Error Std : {np.std(static_err):.4f} m/s")
    else:
        print("  - No samples.")

    print("-" * 80)

    # --- Dynamic Statistics ---
    print(f"[Dynamic Objects (Moving > 0.1m/s)] (Count: {len(dynamic_err)})")
    if len(dynamic_err) > 0:
        print(f"  - GT Speed Mean : {np.mean(dynamic_gts):.4f} m/s")
        print(f"  - Pred Speed Mean: {np.mean(dynamic_preds):.4f} m/s")
        print(f"  - Speed Error Mean: {np.mean(dynamic_err):.4f} m/s")
        print(f"  - Speed Error Std : {np.std(dynamic_err):.4f} m/s")
    else:
        print("  - No samples.")
    print("="*80)

    # --- Visualization (2x2 Grid) ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Static: GT vs Pred Speed Distribution
    ax = axes[0, 0]
    if len(static_gts) > 0:
        bins = np.linspace(0, max(np.max(static_gts), np.max(static_preds), 1.0), 50)
        ax.hist(static_gts, bins=bins, alpha=0.5, label='GT Speed', color='gray', density=True)
        ax.hist(static_preds, bins=bins, alpha=0.5, label='Pred Speed', color='royalblue', density=True)
        ax.set_title('Static Objects: Speed Distribution\n(GT should be 0)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Speed (m/s)')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(alpha=0.3)

    # 2. Dynamic: GT vs Pred Speed Distribution
    ax = axes[0, 1]
    if len(dynamic_gts) > 0:
        max_val = max(np.max(dynamic_gts), np.max(dynamic_preds))
        bins = np.linspace(0, max_val + 0.5, 50)
        ax.hist(dynamic_gts, bins=bins, alpha=0.5, label='GT Speed', color='gray', density=True)
        ax.hist(dynamic_preds, bins=bins, alpha=0.5, label='Pred Speed', color='crimson', density=True)
        ax.set_title('Moving Dynamic Objects: Speed Distribution\n(Excluding Stopped Objects)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Speed (m/s)')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(alpha=0.3)

    # 3. Static: Error Distribution
    ax = axes[1, 0]
    if len(static_err) > 0:
        ax.hist(static_err, bins=50, color='royalblue', alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(static_err), color='red', linestyle='--', linewidth=2, label=f'Mean Error: {np.mean(static_err):.2f}')
        ax.set_title('Static Objects: Error Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Absolute Velocity Error (m/s)')
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(alpha=0.3)

    # 4. Dynamic: Error Distribution
    ax = axes[1, 1]
    if len(dynamic_err) > 0:
        ax.hist(dynamic_err, bins=50, color='crimson', alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(dynamic_err), color='blue', linestyle='--', linewidth=2, label=f'Mean Error: {np.mean(dynamic_err):.2f}')
        ax.set_title('Moving Dynamic Objects: Error Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Absolute Velocity Error (m/s)')
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    
    save_path = "validation_analysis_full.png"
    plt.savefig(save_path)
    print(f"\n📈 상세 분석 그래프 저장됨: {os.path.abspath(save_path)}")
    # plt.show() 

def main():
    parser = argparse.ArgumentParser(description="Evaluate ClusterFlowNet with detailed speed analysis.")
    
    parser.add_argument("--data-root", type=str, default="../VALIDATION/validation", 
                        help="Path to the validation dataset root")
    parser.add_argument("--checkpoint", type=str, default="../cluster_flow_net.pth", 
                        help="Path to the model checkpoint .pth file")
    
    args = parser.parse_args()

    # 1. 데이터셋 로드
    if not os.path.exists(args.data_root):
        print(f"❌ Error: Dataset path not found: {args.data_root}")
        return

    print(f"📂 Loading validation data from: {args.data_root}")
    val_dataset = ClusterDataset(root=args.data_root, split="val", num_points=64)
    
    if len(val_dataset) == 0:
        print("❌ Error: Dataset is empty.")
        return

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # 2. 모델 로드
    if not os.path.exists(args.checkpoint):
        print(f"❌ Error: Checkpoint file not found: {args.checkpoint}")
        return

    print(f"🤖 Loading model from: {args.checkpoint}")
    model = ClusterFlowNet().to(DEVICE)
    
    try:
        checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        print("✅ Checkpoint loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return

    # 3. 평가 실행
    static_err, dynamic_err, static_preds, dynamic_preds, static_gts, dynamic_gts = evaluate(model, val_loader)

    # 4. 결과 분석 및 시각화
    visualize_results(static_err, dynamic_err, static_preds, dynamic_preds, static_gts, dynamic_gts)

if __name__ == "__main__":
    main()
