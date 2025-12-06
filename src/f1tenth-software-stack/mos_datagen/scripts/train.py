#!/usr/bin/env python3
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from model import ClusterFlowNet
from dataset import ClusterDataset

# ---------------- hyperparams ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
NUM_EPOCHS = 50
LR = 1e-4
NUM_WORKERS = 4
# ---------------------------------------------

def train_one_epoch(model, dataloader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    seen_steps = 0

    for batch_idx, batch in enumerate(dataloader):
        curr_in = batch[0].to(DEVICE)
        prev_in = batch[1].to(DEVICE)
        ego_vector = batch[2].to(DEVICE)
        raw_ego_vel = batch[3].to(DEVICE)
        target_vel = batch[4].to(DEVICE)

        optimizer.zero_grad()
        
        # Forward (Returns only velocity)
        pred_vel = model(curr_in, prev_in, ego_vector, raw_ego_vel)

        mask = ~torch.isnan(target_vel).any(dim=1)
        if mask.sum() == 0: continue
            
        valid_pred = pred_vel[mask]
        valid_target = target_vel[mask]

        # Simple MSE Loss (No Confidence)
        loss = F.mse_loss(valid_pred, valid_target)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        seen_steps += 1

        if batch_idx % 100 == 0:
            print(f"[Epoch {epoch}] Batch {batch_idx}: MSE Loss={loss.item():.4f}")

    epoch_loss = (running_loss / seen_steps) if seen_steps > 0 else 0.0
    return epoch_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="../dataset_vel_label")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    args = parser.parse_args()

    print(f"Loading data from {args.data_root}...")
    train_dataset = ClusterDataset(root=args.data_root, split="train", num_points=64)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    
    print(f"Initializing Model on {DEVICE}...")
    model = ClusterFlowNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print("Starting Training...")
    for epoch in range(1, args.epochs + 1):
        epoch_loss = train_one_epoch(model, train_loader, optimizer, epoch)
        print(f"Epoch {epoch} Finished. Avg Loss: {epoch_loss:.6f}")

        if epoch % 5 == 0:
            ckpt_path = f"checkpoint_epoch_{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

if __name__ == "__main__":
    main()
