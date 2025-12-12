#!/usr/bin/env python3
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# ⚠️ 주의: model.py와 dataset.py가 다음 조건을 만족해야 합니다.
# 1. model.py: ClusterFlowNet의 forward가 3가지 출력(pred_rel, pred_abs, aux_ego_pred)을 반환
# 2. dataset.py: ClusterDataset의 __init__에서 glob.glob을 사용하여 멀티 디렉토리를 지원
from model import ClusterFlowNet
from dataset import ClusterDataset

# ---------------- hyperparams ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
NUM_EPOCHS = 50
LR = 1e-4
NUM_WORKERS = 4

# ✅ 손실 가중치 설정
STATIC_WEIGHT = 3.0  # 정적 객체 손실에 3배 가중치 부여 (Ego-Bias 제거 목적)
AUX_EGO_WEIGHT = 0.5 # 보조 Ego-Motion 예측 손실 가중치
VEL_THRESHOLD = 0.1  # 이 값 이하의 target_vel은 정적 객체로 간주 (m/s)
L2_MIX_WEIGHT = 0.5  # ✅ Dynamic Loss에 L2를 혼합하는 비율
# ---------------------------------------------

def train_one_epoch(model, dataloader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    running_main_loss = 0.0
    running_aux_loss = 0.0
    seen_steps = 0

    for batch_idx, batch in enumerate(dataloader):
        # 1. 데이터 로드 및 GPU 전송
        curr_in = batch[0].to(DEVICE)
        prev_in = batch[1].to(DEVICE)
        ego_vector = batch[2].to(DEVICE)
        raw_ego_vel = batch[3].to(DEVICE)
        target_vel = batch[4].to(DEVICE) # GT Relative Velocity

        optimizer.zero_grad()
        
        # 2. Forward 
        pred_rel, pred_abs, aux_ego_pred = model(curr_in, prev_in, ego_vector, raw_ego_vel) 

        # 3. 유효 샘플 마스크 (NaN 제거)
        mask = ~torch.isnan(target_vel).any(dim=1)
        if mask.sum() == 0: continue
            
        valid_pred_rel = pred_rel[mask]
        valid_target = target_vel[mask]
        valid_raw_ego = raw_ego_vel[mask] 
        valid_ego_vector = ego_vector[mask]
        
        # --- A. 메인 손실 계산 (Weighted L1/L2 Hybrid Loss) ---
        
        # 4. 정적/동적 객체 마스크 정의
        is_static_mask = (valid_target.norm(dim=1) < VEL_THRESHOLD).float()  
        is_dynamic_mask = 1.0 - is_static_mask

        # 5a. L1 손실 (방향 및 크기에 대한 직접적인 페널티)
        loss_L1_per_sample = F.l1_loss(valid_pred_rel, valid_target, reduction='none').mean(dim=1)
        
        # 5b. L2 손실 (동적 객체의 방향/크기 오차 강조를 위해 사용)
        loss_L2_per_sample = F.mse_loss(valid_pred_rel, valid_target, reduction='none').mean(dim=1)

        # 6. Hybrid 손실 정의
        
        # 정적 손실: L1 (가중치 적용, Ego-Bias 강력 억제)
        static_loss_weighted = loss_L1_per_sample * is_static_mask * STATIC_WEIGHT
        
        # 동적 손실: L1 + L2 혼합 (방향 민감도를 높임)
        # L1은 작은 오차에, L2는 큰 오차(방향 틀어짐)에 페널티를 부여
        dynamic_hybrid_loss = (loss_L1_per_sample + L2_MIX_WEIGHT * loss_L2_per_sample) * is_dynamic_mask
        
        # 7. 최종 메인 손실
        main_loss = (static_loss_weighted + dynamic_hybrid_loss).mean()


        # --- B. 보조 손실 계산 (Auxiliary Ego-Motion Loss) ---
        
        # 8. GT Ego-Motion 벡터 생성 [vx, vy, omega]
        omega_gt = valid_ego_vector[:, 2].unsqueeze(1) 
        raw_v_gt = valid_raw_ego 
        ego_gt_vector = torch.cat([raw_v_gt, omega_gt], dim=1) 
        
        # 9. 보조 손실 (L1 사용)
        aux_loss = F.l1_loss(aux_ego_pred[mask], ego_gt_vector)
        
        
        # 10. 최종 손실 합산
        loss = main_loss + AUX_EGO_WEIGHT * aux_loss

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_main_loss += main_loss.item()
        running_aux_loss += aux_loss.item()
        seen_steps += 1

        if batch_idx % 100 == 0:
            print(f"[Epoch {epoch}] Batch {batch_idx}: Total Loss={loss.item():.4f}, "
                  f"Main Hybrid Loss={main_loss.item():.4f}, Aux L1 Loss={aux_loss.item():.4f}")

    epoch_loss = (running_loss / seen_steps) if seen_steps > 0 else 0.0
    epoch_main_loss = (running_main_loss / seen_steps) if seen_steps > 0 else 0.0
    epoch_aux_loss = (running_aux_loss / seen_steps) if seen_steps > 0 else 0.0
    
    print(f"Epoch {epoch} Metrics: Avg Main Hybrid Loss={epoch_main_loss:.6f}, Avg Aux L1 Loss={epoch_aux_loss:.6f}")
    return epoch_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="../dataset_l")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    args = parser.parse_args()

    print(f"Loading data from {args.data_root}...")
    train_dataset = ClusterDataset(root=args.data_root, split="train", num_points=64)
    
    if len(train_dataset) == 0:
         print("[❌ 오류] 데이터셋에 유효한 샘플이 없습니다. 경로와 파일 구조를 확인하세요.")
         return

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    
    print(f"Initializing Model on {DEVICE}...")
    model = ClusterFlowNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print("Starting Training...")
    for epoch in range(1, args.epochs + 1):
        epoch_loss = train_one_epoch(model, train_loader, optimizer, epoch)
        print(f"Epoch {epoch} Finished. Avg Total Loss: {epoch_loss:.6f}")

        if epoch % 5 == 0:
            ckpt_path = f"cluster_flow_net_checkpoint_epoch_{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

if __name__ == "__main__":
    main()
