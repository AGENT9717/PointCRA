import random, os
import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from s3dis import S3DIS, s3dis_collate_fn
from torch.utils.data import DataLoader
import sys, math, logging
from pathlib import Path
from datetime import datetime
sys.path.append(str(Path(__file__).absolute().parent.parent))
from utils.timm.scheduler.cosine_lr import CosineLRScheduler
from utils.timm.optim import create_optimizer_v2
import utils.util as util
from delasemseg import DelaSemSeg
from time import time, sleep
from config import s3dis_args, s3dis_warmup_args, dela_args, batch_size, learning_rate as lr, epoch, warmup, \
    label_smoothing as ls
import config

import shutil
from pathlib import Path
from s3dis import s3dis_test_collate_fn
from torch.cuda.amp import autocast


def detailed_gradient_analysis(model, param_patterns, logger, epoch):
    """
    详细的梯度分析，包括权重和梯度的统计信息
    """
    logger.info(f"=== Epoch {epoch} 详细梯度分析 ===")

    for name, param in model.named_parameters():
        # 检查是否匹配任何监控模式
        if any(pattern in name for pattern in param_patterns):
            logger.info(f"参数: {name}")
            logger.info(f"  权重统计:")
            logger.info(f"    - 形状: {param.shape}")
            logger.info(f"    - 范数: {param.norm().item():.6e}")
            logger.info(f"    - 均值: {param.mean().item():.6e}")
            logger.info(f"    - 标准差: {param.std().item():.6e}")
            logger.info(f"    - 范围: [{param.min().item():.6e}, {param.max().item():.6e}]")

            if param.grad is not None:
                grad = param.grad
                logger.info(f"  梯度统计:")
                logger.info(f"    - 范数: {grad.norm().item():.6e}")
                logger.info(f"    - 均值: {grad.mean().item():.6e}")
                logger.info(f"    - 标准差: {grad.std().item():.6e}")
                logger.info(f"    - 范围: [{grad.min().item():.6e}, {grad.max().item():.6e}]")

                # 检查梯度健康度
                nan_count = torch.isnan(grad).sum().item()
                inf_count = torch.isinf(grad).sum().item()
                zero_count = (grad == 0).sum().item()
                total_elements = grad.numel()

                logger.info(f"  梯度健康度:")
                logger.info(f"    - NaN比例: {nan_count}/{total_elements} ({nan_count / total_elements * 100:.2f}%)")
                logger.info(f"    - Inf比例: {inf_count}/{total_elements} ({inf_count / total_elements * 100:.2f}%)")
                logger.info(
                    f"    - 零梯度比例: {zero_count}/{total_elements} ({zero_count / total_elements * 100:.2f}%)")

                # 梯度/权重比率
                if param.norm() > 0:
                    grad_weight_ratio = grad.norm() / param.norm()
                    logger.info(f"    - 梯度/权重比率: {grad_weight_ratio:.6e}")
            else:
                logger.info(f"  梯度: 无梯度")

            logger.info("  " + "=" * 50)



def copy_model_file_to_log_dir(cur_id, model_file_path="delasemseg.py"):
    """
    将模型文件复制到日志输出目录

    Args:
        cur_id: 当前运行的ID，用于确定输出目录
        model_file_path: 模型文件路径，默认为delasemseg.py
    """
    try:
        # 构建目标路径
        target_dir = Path(f"output/log/{cur_id}")
        target_dir.mkdir(parents=True, exist_ok=True)

        # 构建目标文件路径
        target_file = target_dir / "delasemseg.py"

        # 复制文件
        shutil.copy2(model_file_path, target_file)

        return True
    except Exception as e:
        print(f"Error copying model file: {e}")
        return False

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False


if config.seed is not None:
    set_seed(config.seed)

torch.set_float32_matmul_precision("high")


def setup_logging(cur_id):
    """设置同时输出到文件和终端的logging"""
    os.makedirs(f"output/log/{cur_id}", exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 清除现有的handler
    if logger.handlers:
        logger.handlers.clear()

    # 文件handler
    file_handler = logging.FileHandler(f"output/log/{cur_id}/out.log", mode='a')
    file_handler.setLevel(logging.INFO)

    # 控制台handler - 输出到终端
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # 设置格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加handler
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def count_parameters(model):
    """计算模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def format_duration(seconds):
    """将秒数格式化为分钟:秒的形式"""
    minutes = int(seconds // 60)
    seconds_remaining = int(seconds % 60)
    return f"{minutes}:{seconds_remaining:02d}"


def print_config_parameters(logger):
    """打印配置参数 - 保持所有信息完整，但使用同一个时间戳"""
    config_output = []
    config_output.append("=" * 80)
    config_output.append("CONFIG PARAMETERS")
    config_output.append("=" * 80)

    # 训练参数
    config_output.append("Training Parameters:")
    config_output.append(f"  - Batch size: {batch_size}")
    config_output.append(f"  - Learning rate: {lr}")
    config_output.append(f"  - Epochs: {epoch}")
    config_output.append(f"  - Warmup: {warmup}")
    config_output.append(f"  - Label smoothing: {ls}")

    # S3DIS数据集参数
    config_output.append("S3DIS Dataset Parameters:")
    s3dis_args_dict = vars(s3dis_args)
    for key, value in s3dis_args_dict.items():
        config_output.append(f"  - {key}: {value}")

    # S3DIS预热参数
    config_output.append("S3DIS Warmup Parameters:")
    s3dis_warmup_args_dict = vars(s3dis_warmup_args)
    for key, value in s3dis_warmup_args_dict.items():
        config_output.append(f"  - {key}: {value}")

    # Delaunay分割参数
    config_output.append("Delaunay Segmentation Parameters:")
    dela_args_dict = vars(dela_args)
    for key, value in dela_args_dict.items():
        config_output.append(f"  - {key}: {value}")

    config_output.append("=" * 80)

    # 一次性输出所有配置信息
    logger.info("\n".join(config_output))


def print_model_parameters(logger, model):
    """打印模型参数量信息"""
    total_params, trainable_params = count_parameters(model)

    model_output = []
    model_output.append("MODEL PARAMETERS:")
    model_output.append(f"  - Total parameters: {total_params:,}")
    model_output.append(f"  - Trainable parameters: {trainable_params:,}")
    model_output.append(f"  - Non-trainable parameters: {total_params - trainable_params:,}")
    model_output.append(f"  - Model size: {total_params * 4 / (1024 ** 2):.2f} MB (FP32)")
    model_output.append("=" * 80)

    logger.info("\n".join(model_output))


def print_model_structure(logger, model):
    """打印模型结构"""
    logger.info("MODEL STRUCTURE:")
    logger.info("=" * 80)

    # 使用torchsummary来打印模型结构（如果可用）
    try:
        from torchsummary import summary
        # 创建一个虚拟输入来获取模型结构
        dummy_xyz = torch.randn(1, 4096, 3).cuda()
        dummy_feature = torch.randn(1, 4096, 6).cuda()
        dummy_indices = [torch.randint(0, 4096, (4096,)).cuda().long() for _ in range(4)]

        logger.info("Model summary with input shape:")
        # 由于模型可能有复杂的输入，我们使用try-catch来避免错误
        try:
            summary(model, input_data=[dummy_xyz, dummy_feature, dummy_indices], verbose=2)
        except:
            logger.info("Unable to generate detailed summary with torchsummary")
    except ImportError:
        logger.info("torchsummary not installed, using simple model print")

    # 直接打印模型结构
    logger.info("Raw model structure:")
    logger.info(str(model))
    logger.info("=" * 80)

def warmup_fn(model, dataset, logger):
    model.train()
    traindlr = DataLoader(dataset, batch_size=len(dataset), collate_fn=s3dis_collate_fn, pin_memory=True,
                          num_workers=12)
    for xyz, feature, indices, pts, y in traindlr:
        xyz = xyz.cuda(non_blocking=True)
        feature = feature.cuda(non_blocking=True)
        indices = [ii.cuda(non_blocking=True).long() for ii in indices[::-1]]
        pts = pts.tolist()[::-1]
        y = y.cuda(non_blocking=True)
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            p, closs = model(xyz, feature, indices, pts)
            loss = F.cross_entropy(p, y) + closs
        loss.backward()


cur_id = "01"
now = datetime.now()
date_str = now.strftime("%Y%m%d")
time_str = now.strftime("%H%M%S")
cur_id = date_str + "_" + time_str
os.makedirs(f"output/model/{cur_id}", exist_ok=True)

# 设置logging代替原来的重定向
logger = setup_logging(cur_id)

copy_success = copy_model_file_to_log_dir(cur_id)
if copy_success:
    logger.info(f"Successfully copied delasemseg.py to output/log/{cur_id}/")
else:
    logger.warning("Failed to copy delasemseg.py to log directory")

logger.info("base")

# 创建模型
model = DelaSemSeg(dela_args).cuda()

# 打印配置参数
print_config_parameters(logger)
# 打印模型结构
print_model_structure(logger, model)
# 打印模型参数量
print_model_parameters(logger, model)

traindlr = DataLoader(S3DIS(s3dis_args, partition="!5", loop=30), batch_size=batch_size,
                      collate_fn=s3dis_collate_fn, shuffle=True, pin_memory=True,
                      persistent_workers=True, drop_last=True, num_workers=13)
testdlr = DataLoader(S3DIS(s3dis_args, partition="5", loop=1, train=False), batch_size=1,
                     collate_fn=s3dis_collate_fn, pin_memory=True,
                     persistent_workers=True, num_workers=13)

step_per_epoch = len(traindlr)

optimizer = create_optimizer_v2(model, lr=lr, weight_decay=5e-2)
scheduler = CosineLRScheduler(optimizer, t_initial=epoch * step_per_epoch, lr_min=lr / 10000,
                              warmup_t=warmup * step_per_epoch, warmup_lr_init=lr / 20)
# if wish to continue from a checkpoint
resume = False
if resume:
    start_epoch = util.load_state(f"output/model/{cur_id}/last.pt", model=model, optimizer=optimizer)["start_epoch"]
else:
    start_epoch = 0

scheduler_step = start_epoch * step_per_epoch

metric = util.Metric(13)
ttls = util.AverageMeter()
corls = util.AverageMeter()
best = 0
warmup_fn(model, S3DIS(s3dis_warmup_args, partition="!5", loop=batch_size, warmup=True), logger)
for i in range(start_epoch, epoch):
    model.train()
    ttls.reset()
    corls.reset()
    metric.reset()
    now = time()
    for xyz, feature, indices, pts, y in traindlr:
        lam = scheduler_step / (epoch * step_per_epoch)
        lam = 3e-3 ** lam * .25
        scheduler.step(scheduler_step)
        scheduler_step += 1
        xyz = xyz.cuda(non_blocking=True)
        feature = feature.cuda(non_blocking=True)
        indices = [ii.cuda(non_blocking=True).long() for ii in indices[::-1]]
        pts = pts.tolist()[::-1]
        y = y.cuda(non_blocking=True)
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            p, closs = model(xyz, feature, indices, pts)
            loss = F.cross_entropy(p, y, label_smoothing=ls)
        metric.update(p.detach(), y)
        ttls.update(loss.item())
        corls.update(closs.item())
        optimizer.zero_grad(set_to_none=True)
        (loss + closs * lam).backward()
        optimizer.step()


    logger.info(f"epoch:{i}/{epoch} || loss: {round(ttls.avg, 4)} || cls: {round(corls.avg, 4)}")
    # 假设metric.print()内部使用print，我们需要修改它或者保持原样
    # 如果metric.print()使用print，它也会输出到终端
    metric.print("train:")
    train_miou = metric.miou
    model.eval()
    metric.reset()
    with torch.no_grad():
        for xyz, feature, indices, pts, y in testdlr:
            xyz = xyz.cuda(non_blocking=True)
            feature = feature.cuda(non_blocking=True)
            indices = [ii.cuda(non_blocking=True).long() for ii in indices[::-1]]
            y = y.cuda(non_blocking=True)
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                p = model(xyz, feature, indices)
            metric.update(p, y)

    metric.print("val:  ")
    duration_seconds = time() - now
    formatted_duration = format_duration(duration_seconds)
    if best is not None:
        logger.info(f"duration: {formatted_duration} (mm:ss),train miou:{train_miou},val miou:{metric.miou} best miou: {best}")
    else:
        logger.info(f"duration: {formatted_duration} (mm:ss)")
    cur = metric.miou
    if best < cur:
        best = cur
        sym1="="
        logger.info(sym1*30+f"  new best! epoch:{i},train:{train_miou},val:{best}  "+sym1*30)
        util.save_state(f"output/model/{cur_id}/best.pt", model=model)

    util.save_state(f"output/model/{cur_id}/last.pt", model=model, optimizer=optimizer, start_epoch=i + 1)

logger.info("=" * 80)
logger.info("Training completed! Starting automatic testing with best model...")
logger.info("=" * 80)

# 重新加载最佳模型
best_model_path = f"output/model/{cur_id}/best.pt"
save_model_path = "pretrained/best.pt"
if os.path.exists(save_model_path):
    os.remove(save_model_path)
shutil.copy2(best_model_path, save_model_path)

if os.path.exists(best_model_path):
    util.load_state(best_model_path, model=model)
    model.eval()

    # 创建测试数据加载器
    test_loop = 12
    testdlr_final = DataLoader(
        S3DIS(s3dis_args, partition="5", loop=test_loop, train=False, test=True),
        batch_size=1,
        collate_fn=s3dis_test_collate_fn,
        pin_memory=True,
        num_workers=14
    )

    # 进行测试
    metric_test = util.Metric(13)
    cum = 0
    cnt = 0

    with torch.no_grad():
        for xyz, feature, indices, nn, y in testdlr_final:
            xyz = xyz.cuda(non_blocking=True)
            feature = feature.cuda(non_blocking=True)
            indices = [ii.cuda(non_blocking=True).long() for ii in indices[::-1]]
            nn = nn.cuda(non_blocking=True).long()
            with autocast(dtype=torch.bfloat16):
                p = model(xyz, feature, indices)
            cum = cum + p[nn]
            cnt += 1
            if cnt % test_loop == 0:
                y = y.cuda(non_blocking=True)
                metric_test.update(cum, y)
                cnt = cum = 0

    # 输出测试结果
    metric_test.print("Final test with best model: ")

    # 记录测试结果到日志文件
    logger.info("FINAL TEST RESULTS:")
    logger.info(f"Best model test accuracy: {metric_test.acc}")
    logger.info(f"Best model test mAcc: {metric_test.macc}")
    logger.info(f"Best model test mIoU: {metric_test.miou}")

else:
    logger.warning(f"Best model not found at {best_model_path}, skipping automatic test.")

logger.info("=" * 80)
logger.info("Training and testing process completed!")
logger.info("=" * 80)