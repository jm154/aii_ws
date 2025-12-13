#!/usr/bin/env python3
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2  # OpenCV 멀티스레딩 충돌 방지 (필수)
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau # ⭐️ 스케줄러 추가

from model import ClusterFlowNet
from dataset import ClusterDataset

# ---------------- hyperparams ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# RTX 4070 Ti 최적화 설정
BATCH_SIZE = 64     
NUM_EPOCHS = 100    # 스케줄러 동작을 위해 넉넉하게 설정
LR = 1e-4
NUM_WORKERS = 4     
# ---------------------------------------------

# DataLoader worker 충돌 방지
cv2.setNumThreads(0)

def train_one_epoch(model, dataloader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    seen_steps = 0

    # ==========================================
    # ⚡️ Hyperparameters
    # ==========================================
    DYNAMIC_WEIGHT = 1.0      # 동적 객체 가중치
    COSINE_WEIGHT = 0.5       # 방향 Loss 가중치 (MSE와 스케일 맞춤)
    GRAD_CLIP_NORM = 2.0      # Gradient Clipping 임계값
    # ==========================================

    for batch_idx, batch in enumerate(dataloader):
        curr_in = batch[0].to(DEVICE)
        prev_in = batch[1].to(DEVICE)
        ego_vector = batch[2].to(DEVICE)
        raw_ego_vel = batch[3].to(DEVICE)
        target_vel = batch[4].to(DEVICE)
        
        # 라벨 로드 (없으면 자동 생성)
        if len(batch) > 5:
            labels = batch[5].to(DEVICE).view(-1)
        else:
            target_speed = torch.norm(target_vel, dim=1)
            labels = (target_speed > 0.5).long()

        optimizer.zero_grad()
        
        # Forward
        output = model(curr_in, prev_in, ego_vector, raw_ego_vel)
        if isinstance(output, tuple):
            pred_vel = output[0]
        else:
            pred_vel = output

        # 유효 데이터 마스킹
        mask = ~torch.isnan(target_vel).any(dim=1)
        if mask.sum() == 0: continue
            
        valid_pred = pred_vel[mask]
        valid_target = target_vel[mask]
        valid_labels = labels[mask]

        # -----------------------------------------------------------
        # 🔥 Hybrid Loss: MSE + Cosine Direction
        # -----------------------------------------------------------
        
        # 1. MSE Loss (기본: 크기 + 방향)
        #    reduction='none'으로 샘플별 오차 계산
        mse_per_sample = F.mse_loss(valid_pred, valid_target, reduction='none').mean(dim=1)

        # 2. Cosine Similarity Loss (방향 집중)
        # 
        #    Target 속도가 너무 작으면(정지) 방향 정의 불가 -> 마스킹 필요
        target_norm = torch.norm(valid_target, dim=1)
        #    속도가 0.1 m/s 이상인 경우만 방향 오차 계산
        direction_mask = (target_norm > 0.1)
        
        cosine_loss_per_sample = torch.zeros_like(mse_per_sample)
        if direction_mask.sum() > 0:
            # Cosine Sim은 1(일치) ~ -1(반대).
            # Loss로 쓰려면: 1 - Cosine (0:일치, 2:반대)
            cos_sim = F.cosine_similarity(valid_pred[direction_mask], valid_target[direction_mask], dim=1)
            cosine_loss_per_sample[direction_mask] = 1.0 - cos_sim

        # 3. 가중치 적용 (Total Loss)
        #    Dynamic 객체에 가중치(5배) 적용
        weights = torch.ones_like(mse_per_sample)
        weights[valid_labels == 1] = DYNAMIC_WEIGHT
        
        #    최종 결합: (MSE + 0.5 * Cosine) * Dynamic_Weight
        total_loss_per_sample = mse_per_sample + (COSINE_WEIGHT * cosine_loss_per_sample)
        loss = (total_loss_per_sample * weights).mean()
        # -----------------------------------------------------------

        loss.backward()
        
        # ⚡️ Gradient Clipping (Loss Spike 방지)
        # 
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
        
        optimizer.step()

        running_loss += loss.item()
        seen_steps += 1

        if batch_idx % 100 == 0:
            print(f"[Epoch {epoch}] Batch {batch_idx}: Hybrid Loss={loss.item():.4f}")

    epoch_loss = (running_loss / seen_steps) if seen_steps > 0 else 0.0
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
    print(f"Settings: Batch={args.batch_size}, Workers={args.num_workers}, Device={DEVICE}")
    print("Optimization: Label-Based Weight x3 + LR Scheduler Active")
    
    # 여러 시나리오(하위 디렉토리) 자동 병합 로직
    if os.path.exists(args.data_root):
        subdirs = [os.path.join(args.data_root, d) for d in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, d))]
    else:
        subdirs = []
    
    datasets = []
    if len(subdirs) > 0:
        print(f"Found {len(subdirs)} scenarios. Merging...")
        for d in subdirs:
            try:
                ds = ClusterDataset(root=d, split="train", num_points=64)
                if len(ds) > 0:
                    datasets.append(ds)
                    print(f"  -> Loaded: {d} ({len(ds)} samples)")
            except Exception as e:
                print(f"  -> Skipping {d}: {e}")
                
        if len(datasets) > 0:
            train_dataset = ConcatDataset(datasets)
            print(f"Total Combined Samples: {len(train_dataset)}")
        else:
            print("  -> No valid datasets found in subdirectories. Trying root directly.")
            train_dataset = ClusterDataset(root=args.data_root, split="train", num_points=64)
    else:
        train_dataset = ClusterDataset(root=args.data_root, split="train", num_points=64)

    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=True,
        drop_last=True 
    )
    
    print(f"Initializing Model on {DEVICE}...")
    model = ClusterFlowNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ⭐️ 스케줄러 정의: Loss가 5 epoch 동안 개선 안되면 LR을 절반(0.5)으로 줄임
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    print("Starting Training...")
    for epoch in range(1, args.epochs + 1):
        epoch_loss = train_one_epoch(model, train_loader, optimizer, epoch)
        
        # 스케줄러 업데이트
        scheduler.step(epoch_loss)
        
        # 현재 LR 확인
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch} Finished. Avg Loss: {epoch_loss:.6f} | LR: {current_lr:.2e}")

        if epoch % 5 == 0:
            ckpt_path = f"checkpoint_epoch_{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(), # 스케줄러 상태도 저장
            }, ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

if __name__ == "__main__":
    main()
