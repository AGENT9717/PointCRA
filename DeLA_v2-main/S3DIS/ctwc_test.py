import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from utils.cutils import ctwc_b11

print("="*60)
print("测试 ctwc_b11 (通道数=16)")
print("="*60)

# 构造测试数据：B=1, N=2, T=3, C=16
B, N, T, C = 1, 2, 3, 16
C_ = C // 4  # C_ = 4

# 创建测试数据
seq_raw = torch.zeros(B, N, T, C, dtype=torch.float32)

# point0: 递增序列 (每个时间步+1)
for t in range(T):
    seq_raw[0, 0, t] = torch.ones(C) * (t + 1)  # t0=1, t1=2, t2=3

# point1: 常数序列 (全是1)
seq_raw[0, 1] = torch.ones(T, C)

print("原始多通道数据:")
print(f"seq_raw shape: {seq_raw.shape}")
print(f"point0 t0: {seq_raw[0,0,0][:4].numpy()} ...")  # 只显示前4个通道
print(f"point0 t1: {seq_raw[0,0,1][:4].numpy()} ...")
print(f"point0 t2: {seq_raw[0,0,2][:4].numpy()} ...")
print(f"point1 t0: {seq_raw[0,1,0][:4].numpy()} ...")

# 【关键】在4个通道上平均，得到每组的值
# seq_grouped shape: [B, N, T, C_] where C_ = 4
seq_grouped = seq_raw.view(B, N, T, C_, 4).mean(dim=-1)
print(f"\n分组平均后:")
print(f"seq_grouped shape: {seq_grouped.shape}")
print(f"point0 t0 (4个组): {seq_grouped[0,0,0].numpy()}")
print(f"point0 t1 (4个组): {seq_grouped[0,0,1].numpy()}")
print(f"point0 t2 (4个组): {seq_grouped[0,0,2].numpy()}")
print(f"point1 t0 (4个组): {seq_grouped[0,1,0].numpy()}")

# knn: point0的邻居是point1，point1的邻居是point0
knn = torch.tensor([[[1], [0]]], dtype=torch.int64)
scale = torch.tensor(1.0)

print(f"\nknn: {knn}")

# 调用ctwc_b11
c_dist = ctwc_b11(seq_grouped, knn, scale)

print("\n" + "="*60)
print("计算结果:")
print(f"c_dist shape: {c_dist.shape}")  # 应该是 [1, 2, 1, 4]

# c_dist[0,0,0] 是 point0与point1的相似度（4个组）
print(f"\nc_dist[0,0,0]: point0与point1的相似度 (4个组):")
print(f"  {c_dist[0,0,0].cpu().numpy()}")

print(f"\nc_dist[0,1,0]: point1与point0的相似度 (4个组):")
print(f"  {c_dist[0,1,0].cpu().numpy()}")

print(f"\n均值:")
print(f"  point0->point1: {c_dist[0,0,0].mean().item():.6f}")
print(f"  point1->point0: {c_dist[0,1,0].mean().item():.6f}")

# 理论值计算
print("\n理论值分析:")
print("point0每个组平均后变化: [1→2→3] 变化量 [1.0, 1.0]")
print("point1每个组平均后变化: [1→1→1] 变化量 [0.0, 0.0]")
print("角度相似度理论值: 0.853553 (每个组应该相同)")
print("="*60)