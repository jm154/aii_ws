import torch
import torch.nn as nn
import torch.nn.functional as F

class ClusterFlowNet(nn.Module):
    def __init__(self):
        super(ClusterFlowNet, self).__init__()
        
        # Input: (x, y, residual, angle_norm)
        in_channels = 4 
        
        # --- Point Feature Encoder (Shared) ---
        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.bn3 = nn.BatchNorm1d(256)

        # --- Ego-Motion Encoder (Input: vx, vy, omega, dt) ---
        self.ego_mlp = nn.Sequential(
            nn.Linear(4, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        # --- Decoding (Combined size: 256 + 256 + 128 = 640) ---
        self.fc1 = nn.Linear(640, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        
        # 1. Main Head: Predict Absolute Object Velocity (V_obj_pred)
        self.fc3 = nn.Linear(128, 2) 

        # 2. ✅ Auxiliary Head: Predict Ego-Motion (V_x, V_y, Omega)
        # Goal: Make the combined features decouple Ego-motion features
        self.aux_ego_head = nn.Sequential(
            nn.Linear(640, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 3) # Output: [vx, vy, omega]
        )

        # Init Weights (Main Head only)
        nn.init.normal_(self.fc3.weight, 0, 0.01)
        nn.init.constant_(self.fc3.bias, 0)

    def forward_one_branch(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x))) 
        x = torch.max(x, 2, keepdim=False)[0] # PointNet Global Feature Extraction
        return x

    def forward(self, curr_cluster, prev_patch, ego_vector, raw_ego_vel):
        feat_curr = self.forward_one_branch(curr_cluster)
        feat_prev = self.forward_one_branch(prev_patch)
        feat_ego = self.ego_mlp(ego_vector) 
        
        # Feature Concatenation (Shared Feature Space)
        combined = torch.cat([feat_curr, feat_prev, feat_ego], dim=1) 
        
        # Shared Decoding layers
        x = F.relu(self.bn4(self.fc1(combined)))
        x = F.relu(self.bn5(self.fc2(x)))
        
        # --- Head 1: Main Velocity Prediction ---
        v_obj_pred = self.fc3(x) 
        pred_vel_relative = v_obj_pred - raw_ego_vel # V_obj - V_ego_raw

        # --- Head 2: Auxiliary Ego-Motion Prediction ---
        aux_ego_pred = self.aux_ego_head(combined) # Predicts [V_x, V_y, Omega]
        
        # ✅ 3개의 출력을 반환하도록 수정
        return pred_vel_relative, v_obj_pred, aux_ego_pred
