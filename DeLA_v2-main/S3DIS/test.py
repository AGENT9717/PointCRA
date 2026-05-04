import torch
from torch import nn
import torch.nn.functional as F
from s3dis import S3DIS, s3dis_test_collate_fn
from torch.utils.data import DataLoader
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).absolute().parent.parent))
import utils.util as util
from delasemseg import DelaSemSeg
from config import s3dis_args, dela_args
from torch.cuda.amp import autocast
import os
import numpy as np
from plyfile import PlyData, PlyElement
import argparse
from datetime import datetime

# 颜色映射（从S3DIS类中获取）
CLASS2COLOR = {'ceiling': [0, 255, 0],
               'floor': [0, 0, 255],
               'wall': [0, 255, 255],
               'beam': [255, 255, 0],
               'column': [255, 0, 255],
               'window': [100, 100, 255],
               'door': [200, 200, 100],
               'table': [170, 120, 200],
               'chair': [255, 0, 0],
               'sofa': [200, 100, 100],
               'bookcase': [10, 200, 100],
               'board': [200, 200, 200],
               'clutter': [50, 50, 50]}


def save_ply(points, colors, filename):
    """
    保存点云为PLY文件

    Args:
        points: (N, 3) 点坐标
        colors: (N, 3) RGB颜色值 (0-255)
        filename: 保存路径
    """
    # 确保points是二维数组
    if points.ndim == 1:
        points = points.reshape(-1, 3)

    # 确保颜色是整数类型
    colors = colors.astype(np.uint8)

    # 确保points和colors数量匹配
    assert len(points) == len(colors), f"Points count {len(points)} != Colors count {len(colors)}"

    # 创建结构化数组
    ply_array = np.zeros(len(points), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    ply_array['x'] = points[:, 0]
    ply_array['y'] = points[:, 1]
    ply_array['z'] = points[:, 2]
    ply_array['red'] = colors[:, 0]
    ply_array['green'] = colors[:, 1]
    ply_array['blue'] = colors[:, 2]

    # 创建PLY元素
    ply_element = PlyElement.describe(ply_array, 'vertex')

    # 保存PLY文件
    PlyData([ply_element], text=True).write(filename)
    print(f"Saved: {filename} with {len(points)} points")


def class_to_color(pred_labels):
    """
    将预测的类别标签转换为RGB颜色

    Args:
        pred_labels: (N,) 预测的类别标签 (0-12)

    Returns:
        colors: (N, 3) RGB颜色值
    """
    # 如果是torch tensor，转换为numpy
    if torch.is_tensor(pred_labels):
        pred_labels = pred_labels.cpu().numpy()

    colors = np.zeros((len(pred_labels), 3), dtype=np.uint8)
    cmap = np.array(list(CLASS2COLOR.values()), dtype=np.uint8)
    colors = cmap[pred_labels]
    return colors


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Test S3DIS segmentation')
    parser.add_argument('--vis', action='store_true',
                        help='Visualize results by saving PLY files')
    return parser.parse_args()


torch.set_float32_matmul_precision("high")

# 解析命令行参数
args = parse_args()
vis_enabled = args.vis

# 新的保存路径
base_output_dir = "/home/e702-5090/pointcloud/dela/DeLA_v2-main/S3DIS/output"

# 只在启用可视化时创建带时间戳的输出目录
if vis_enabled:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Visualization enabled, results will be saved to: {output_dir}")
else:
    output_dir = None
    print("Visualization disabled, only metrics will be computed")

loop = 12

testdlr = DataLoader(S3DIS(s3dis_args, partition="5", loop=loop, train=False, test=True), batch_size=1,
                     collate_fn=s3dis_test_collate_fn, pin_memory=True, num_workers=14)

model = DelaSemSeg(dela_args).cuda()

util.load_state("pretrained/best.pt", model=model)

model.eval()

metric = util.Metric(13)
cum = 0
cnt = 0

# 获取数据集
dataset = testdlr.dataset
scene_idx = 0

with torch.no_grad():
    for xyz, feature, indices, nn, y in testdlr:
        xyz = xyz.cuda(non_blocking=True)
        feature = feature.cuda(non_blocking=True)
        indices = [ii.cuda(non_blocking=True).long() for ii in indices[::-1]]
        nn = nn.cuda(non_blocking=True).long()
        with autocast(dtype=torch.bfloat16):
            p = model(xyz, feature, indices)
        cum = cum + p[nn]
        cnt += 1

        if cnt % loop == 0:
            y = y.cuda(non_blocking=True)

            # 累积预测结果
            cum = cum / loop  # 平均预测概率
            pred_labels = cum.argmax(1)  # (N_full,) 原始点云的预测类别

            # 更新指标
            metric.update(cum, y)

            # 只在启用可视化时保存PLY文件
            if vis_enabled:
                # 获取原始点云坐标
                scene_data = dataset.datas[scene_idx]
                full_xyz = scene_data[0].numpy()  # 原始点云坐标

                # 获取预测标签并转换为颜色
                all_pred_labels = pred_labels.cpu()
                colors = class_to_color(all_pred_labels)

                # 获取房间名
                original_name = Path(dataset.paths[scene_idx]).stem
                if original_name.startswith('[') and ']' in original_name:
                    room_name = original_name.split(']', 1)[1]
                else:
                    room_name = original_name

                # 保存预测结果的PLY文件（只保存pred）
                pred_filename = os.path.join(output_dir, f"{room_name}_pred.ply")
                save_ply(full_xyz, colors, pred_filename)

                print(f"Processed room {scene_idx}: {room_name} with {len(full_xyz)} points")

            # 重置
            cnt = 0
            cum = 0
            scene_idx += 1

metric.print("test: ")