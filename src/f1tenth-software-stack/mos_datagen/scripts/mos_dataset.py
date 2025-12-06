#!/usr/bin/env python3
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import glob
import logging

# 로거 설정
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class MosDataset(Dataset):
    """
    F1TENTH MOS 데이터셋 (Single-Head 버전)
    
    반환값 (총 2개):
    1. seq_x (입력): (sequence_length, 1083)
    2. point_labels (포인트별 정답): (1080,)
    """
    def __init__(self, data_path, sequence_length=5):
        self.sequence_length = sequence_length
        self.data_files = sorted(glob.glob(os.path.join(data_path, "*.npz")))
        
        if not self.data_files:
            raise FileNotFoundError(f"No .npz files found in {data_path}")
            
        logger.info(f"Found {len(self.data_files)} data chunks.")
        
        # 모든 데이터를 메모리로 로드
        self.all_ranges = []
        self.all_labels = []
        self.all_ego_twists = []
        
        for file_path in self.data_files:
            try:
                with np.load(file_path) as data:
                    self.all_ranges.append(data['ranges'])
                    self.all_labels.append(data['labels'])
                    self.all_ego_twists.append(data['ego_twist'])
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
        
        self.all_ranges = np.concatenate(self.all_ranges, axis=0)
        self.all_labels = np.concatenate(self.all_labels, axis=0)
        self.all_ego_twists = np.concatenate(self.all_ego_twists, axis=0)
        
        # 비정상 값(inf, nan) 처리
        self.all_ranges[np.isinf(self.all_ranges)] = 30.0
        self.all_ranges[np.isnan(self.all_ranges)] = 0.0
        
        self.total_frames = self.all_ranges.shape[0]
        logger.info(f"Total frames loaded: {self.total_frames}")

    def __len__(self):
        return self.total_frames - self.sequence_length + 1

    def __getitem__(self, idx):
        
        # --- 1. 입력 (X) 생성 ---
        start_idx = idx
        end_idx = idx + self.sequence_length
        
        seq_ranges = self.all_ranges[start_idx:end_idx]
        seq_ego_twist = self.all_ego_twists[start_idx:end_idx]
        seq_x = np.concatenate([seq_ranges, seq_ego_twist], axis=1)
        
        # --- 2. 정답 (Y) 생성 ---
        # (포인트별 레이블만 반환)
        point_labels = self.all_labels[end_idx - 1] # (1080,)
        
        # (총 2개의 값을 반환)
        return (torch.tensor(seq_x, dtype=torch.float32), 
                torch.tensor(point_labels, dtype=torch.long))

# ... (if __name__ == '__main__': 테스트 코드는 동일) ...
