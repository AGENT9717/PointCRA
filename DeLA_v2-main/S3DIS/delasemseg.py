import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torch.nn.init import trunc_normal_
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).absolute().parent.parent))
from utils.timm.models.layers import DropPath
from utils.cutils import (generate_spse_matrix, prepare_m, knn_spse, knn_spse_4, init_coor, la_spse_a4,
                          ctwc, ctwc_a1, ctwc_a0, ctwc_a2, ctwc_a2_1, ctwc_a3, ctwc_a4, ctwc_a4_1, ctwc_b1, ctwc_b11,
                          ctwc_b2, ctwc_wc, ctwc_wc1)
from mamba_ssm.modules.mamba_simple import Mamba
import einops
import numpy as np

all_dist = [[] for _ in range(10)]
spse_m = None


def checkpoint(function, *args, **kwargs):
    return torch_checkpoint(function, *args, use_reentrant=False, **kwargs)


class LFP(nn.Module):
    def __init__(self, in_dim, out_dim, bn_momentum, init=0.):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.bn = nn.BatchNorm1d(out_dim, momentum=bn_momentum)
        out_dim //= 4
        self.coor = nn.Parameter(init_coor(out_dim).flatten())  # 飞波纳切算法生成球面均匀的点坐标集  #c//4,3 flatten
        self.scale = nn.Parameter(torch.zeros(out_dim) + 0.1)
        nn.init.constant_(self.bn.weight, init)

    def forward(self, x, knn):
        knn, xyz = knn  # knn: b,n,k xyz: b,n,4
        # print(f"knn: {knn.shape}, xyz: {xyz.shape}")
        B, N, C = x.shape
        x = self.proj(x)
        w = torch.cat(
            [self.coor.view(-1, 3).transpose(0, 1), self.scale.square().view(1, -1)])  # cat[(3,c//4),(1,c//4)]
        x = la_spse_a4(x, xyz, knn, w, self.training)  # 相当于分组加权最大池化，每个邻域点的特征值上分组上都加入一个学习得到的映射点坐标，并乘以缩放因子
        # 这里是比较邻域点与中心的特征差值，对于每个邻域点通道都减去一个中心点对应通道的缩放值，缩放因子由偏移距离决定
        # 偏移距离：【xyz-固定偏移量（该层学习到的偏移量+中心点坐标）】*scale（学习参数）
        # 对fj-fi*mul作逐通道最大池化
        x = self.bn(x.view(B * N, -1)).view(B, N, -1)
        return x


class Mlp(nn.Module):
    def __init__(self, in_dim, mlp_ratio, bn_momentum, act, init=0.):
        super().__init__()
        hid_dim = round(in_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            act(),
            nn.Linear(hid_dim, in_dim, bias=False),
            nn.BatchNorm1d(in_dim, momentum=bn_momentum),
        )
        nn.init.constant_(self.mlp[-1].weight, init)

    def forward(self, x):
        B, N, C = x.shape
        x = self.mlp(x.view(B * N, -1)).view(B, N, -1)
        return x


class CTWC_Block(nn.Module):
    def __init__(self, in_dim, act, mlp_ratio, bn_momentum, depth, K, init=0.):
        super().__init__()
        hid_dim = round(in_dim * mlp_ratio)
        self.dim = in_dim
        self.drop = DropPath(0.1)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            act(),
            nn.Linear(hid_dim, in_dim, bias=False),
            nn.BatchNorm1d(in_dim, momentum=bn_momentum),
        )
        weight_dim = in_dim // 4

        self.channel_scale = nn.Parameter(torch.ones(weight_dim))
        self.channel_bias = nn.Parameter(torch.zeros(weight_dim))
        conv_dim = int(weight_dim * 1.5)

        self.a = nn.Parameter(torch.ones(weight_dim))  # 控制斜率
        self.b = nn.Parameter(torch.zeros(weight_dim)+0.5)  # 控制中心点
        self.c = nn.Parameter(torch.ones(weight_dim))  # 控制幅度
        # self.sigmoid_scale = nn.Parameter(torch.ones(weight_dim))
        # self.sigmoid_shift = nn.Parameter(torch.zeros(weight_dim))
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(inplace=True),
            nn.Linear(4, 4),
            nn.ReLU(inplace=True),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )
        self.temperature = nn.Parameter(torch.tensor(1.0))

        self.pool = lambda x: torch.mean(x, dim=-2, keepdim=False)
        self.noise_std = 0.02

    def forward(self, x, seq, knn):
        knn, xyz = knn  #knn bnk   xyz bn3
        B, N, C = x.shape
        _, _, K = knn.shape
        #res=x
        if C > 128:
            scale = 4;
        else:
            scale = 4;
        knn = knn.long()

        batch_idx = torch.arange(B, device=xyz.device).view(B, 1, 1)
        batch_idx = batch_idx.expand(-1, N, K)  # [B, N, K]

        # 使用高级索引
        neighbor_points = xyz[batch_idx, knn]  # [B, N, K, 3]
        # neighbor_points = xyz.gather(
        #     dim=1,
        #     index=knn.unsqueeze(-1).expand(-1, -1, -1, 4)
        # )
        center_points = xyz.unsqueeze(2)
        distances = torch.norm(center_points - neighbor_points, dim=-1)

        # 2. 以每个点的最大距离为基准归一化
        max_dist = distances.max(dim=-1, keepdim=True)[0]  # [B, N, 1]
        re_xyz = distances / (max_dist + 1e-8)

        ################ctwc_b
        # print("knn:{},seq:{}".format(knn.shape, seq.shape))
        scale = torch.tensor(1.0, dtype=torch.float, device=x.device)
        #print("seq",seq.shape)

        # if self.training:
        #     noise = torch.randn_like(seq) * self.noise_std
        #     seq = seq + noise
        # else:
        #     seq= seq
        c_dist = ctwc_b11(seq, knn, scale)  # bnkc_
        #c_dist = c_dist**2
        #c_dist = torch.sigmoid(self.sigmoid_scale * (c_dist - self.sigmoid_shift))
        # print("C:{},c_dist:{}".format(C,c_dist.shape))

        mask = (c_dist == 1.0).all(dim=-1)
        mask_expanded = mask.unsqueeze(-1)  # 形状: [B, N, K, 1]
        mask_expanded = mask_expanded.expand_as(c_dist)  # 形状: [B, N, K, C]
        #c_dist[mask_expanded] = 0.5
        c_dist = torch.where(mask_expanded, torch.tensor(0.5, device=c_dist.device), c_dist)
        #c_dist= torch.pow(5,c_dist-1.0)
        pg = c_dist.mean(dim=-1)  # bnk
        pg_var = torch.var(pg, dim=-1, keepdim=True)
        pg_max = pg.max(dim=2, keepdim=True)[0]  # [B, N, 1]
        pg_min = pg.min(dim=2, keepdim=True)[0]  # [B, N, 1]
        pg_range = pg_max - pg_min
        narrow_range_mask = pg_range < 0.01
        max_pg_var = (pg_range ** 2) / 4.0
        max_pg_var = torch.clamp(max_pg_var, 1e-8)
        pn_raw = 1.0-torch.exp(-pg_var / max_pg_var)
        pn_raw = torch.clamp(pn_raw, min=1e-8, max=1.0)
        pn = torch.where(narrow_range_mask,
                         torch.ones_like(pn_raw),
                         pn_raw)  # bn1


        # ==================== 7. 饱和处理 ====================

        gamma = torch.exp(3 * (0.7-pn))
        pg = torch.pow(pg, gamma)

        a = self.a.view(*([1] * (c_dist.dim() - 1)), -1)
        b = self.b.view(*([1] * (c_dist.dim() - 1)), -1)
        c = torch.sigmoid(self.c).view(*([1] * (c_dist.dim() - 1)), -1)  # c∈[0,1]
        pc = c * (torch.sigmoid(a * (c_dist - b)) - torch.sigmoid(-a * b))
        # # ==================== 测试时统计pg和pn分布 ====================
        # if not self.training:
        #     # 收集统计信息
        #     pg_flat = pg.detach().view(-1)  # 展平所有batch和点的pg值
        #     pn_flat = pn.detach().view(-1)
        #
        #     # 计算统计量
        #     pg_mean = pg_flat.mean().item()
        #     pg_std = pg_flat.std().item()
        #     pg_min_val = pg_flat.min().item()
        #     pg_max_val = pg_flat.max().item()
        #     pg_median = pg_flat.median().item()
        #
        #     pn_mean = pn_flat.mean().item()
        #     pn_std = pn_flat.std().item()
        #     pn_min_val = pn_flat.min().item()
        #     pn_max_val = pn_flat.max().item()
        #     pn_median = pn_flat.median().item()
        #
        #     # 计算pg的分位数
        #     pg_percentiles = torch.quantile(pg_flat, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=pg_flat.device))
        #     pn_percentiles = torch.quantile(pn_flat, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=pn_flat.device))
        #
        #     # 计算窄范围比例
        #     narrow_ratio = narrow_range_mask.float().mean().item()
        #
        #     # 打印当前batch的统计信息
        #     print(f"\n{'=' * 60}")
        #     print(f"[测试统计] 当前Batch")
        #     print(f"{'=' * 60}")
        #     print(f"PG分布:")
        #     print(f"  Mean: {pg_mean:.6f}, Std: {pg_std:.6f}")
        #     print(f"  Min: {pg_min_val:.6f}, Max: {pg_max_val:.6f}, Median: {pg_median:.6f}")
        #     print(f"  分位数 (10%, 25%, 50%, 75%, 90%): {pg_percentiles.cpu().numpy()}")
        #     print(f"\nPN分布:")
        #     print(f"  Mean: {pn_mean:.6f}, Std: {pn_std:.6f}")
        #     print(f"  Min: {pn_min_val:.6f}, Max: {pn_max_val:.6f}, Median: {pn_median:.6f}")
        #     print(f"  分位数 (10%, 25%, 50%, 75%, 90%): {pn_percentiles.cpu().numpy()}")
        #     print(f"\n窄范围比例 (pg_range < 0.01): {narrow_ratio:.4f}")
        #     print(f"{'=' * 60}\n")

        k_xyz = 1 - re_xyz
        #pg_k=torch.stack([pg, k_xyz], dim=-1)   #bnk2
        #pg = self.net(pg_k).squeeze(-1)
        pg = pg * k_xyz
        w = pg.unsqueeze(-1) * pc
        #w = F.softmax(w / self.temperature, dim=2)
        xj = ctwc_wc1(x, knn, w)  # bnkc
        x = self.pool(xj)

        x = self.mlp(x.view(B * N, -1)).view(B, N, -1)
        #x=x+res
        c_loss = None

        if self.training:
            c_loss = 0.0
            # c = torch.sigmoid(c)  # c在[0,1]

            losses = []
            #pn_mean = pn.mean()
            # 目标：pn在0.35-0.45之间
            # pn_loss = F.relu(pn_mean - 0.55) + F.relu(0.25 - pn_mean)
            # 1. b应该小于0（最重要！）
            # 惩罚b>0的情况
            #losses.append(F.relu(b).mean())  # b>0就惩罚
            losses.append(F.relu(0.1 - b).mean())  # 惩罚 b < 0.1
            losses.append(F.relu(b - 0.9).mean())  # 惩罚 b > 0.9
            # 2. a应该在合理范围（大概0.2-2.0）
            # 防止a太大或太小
            losses.append(F.relu(0.2 - a).mean())  # 惩罚a<0.2
            losses.append(F.relu(a - 10).mean())  # 惩罚a>2.0

            # 3. c应该在合理范围（大概0.2-0.8）
            losses.append(F.relu(0.2 - c).mean())  # 惩罚c<0.2
            losses.append(F.relu(c - 0.8).mean())  # 惩罚c>0.8

            # 加权组合：b约束最重要
            weights = [0.5,0.5, 0.1, 0.1, 0.15, 0.15]
            c_loss = sum(w * l for w, l in zip(weights, losses))#+pn_loss*0.2
        return x, c_loss



class Block(nn.Module):
    def __init__(self, dim, depth, drop_path, mlp_ratio, bn_momentum, act, k):
        super().__init__()

        self.depth = depth
        self.lfps = nn.ModuleList([
            LFP(dim, dim, bn_momentum) for _ in range(depth)
        ])
        self.mlp = Mlp(dim, mlp_ratio, bn_momentum, act, 0.2)
        self.mlps = nn.ModuleList([
            Mlp(dim, mlp_ratio, bn_momentum, act) for _ in range(depth // 2)
        ])
        if isinstance(drop_path, list):
            drop_rates = drop_path
            self.dp = [dp > 0. for dp in drop_path]
        else:
            drop_rates = torch.linspace(0., drop_path, self.depth).tolist()
            self.dp = [drop_path > 0.] * depth
        # print(drop_rates)
        drop_rates.append(0.3)
        self.drop_paths = nn.ModuleList([
            DropPath(dpr) for dpr in drop_rates
        ])
        ################ ctwc ###################
        self.seq_dim = self.depth + self.depth // 2
        self.final_ctwc = CTWC_Block(dim, act, mlp_ratio, bn_momentum, depth, k)
        self.final_drop_path = DropPath(0.2)
        self.gate = nn.Parameter(torch.tensor(0.1))

    def drop_path(self, x, i, pts):
        if not self.dp[i] or not self.training:
            return x
        return torch.cat([self.drop_paths[i](xx) for xx in torch.split(x, pts, dim=1)], dim=1)

    def forward(self, x, knn, pts=None):
        x = x + self.drop_path(self.mlp(x), 0, pts)
        seq_list = []
        for i in range(self.depth):
            # seq_xi = torch.mean(x.view(x.shape[0], x.shape[1], -1, 4), dim=-1, keepdim=False)
            # seq_list.append(seq_xi)
            x_i = self.lfps[i](x, knn)
            seq_xi = torch.mean(x_i.view(x.shape[0], x.shape[1], -1, 4), dim=-1, keepdim=False)
            seq_list.append(seq_xi)
            x = x + self.drop_path(x_i, i, pts)
            if i % 2 == 1:
                x_ii = self.mlps[i // 2](x)
                seq_xii = torch.mean(x_ii.view(x.shape[0], x.shape[1], -1, 4), dim=-1, keepdim=False)
                seq_list.append(seq_xii)
                x = x + self.drop_path(x_ii, i, pts)
        #
        seq = torch.stack(seq_list, dim=-1)  # b,n,c,t
        b, n, c, t = seq.shape
        #print("b,n,c,t:", b, n, c, t)
        seq = seq.permute(0, 1, 3, 2).contiguous()  # b,n,t,c

        final_x, c_loss = self.final_ctwc(x, seq, knn)
        x = x + final_x
        #x = x + self.drop_path(final_x, -1, pts)
        #x = x*(1.0-self.gate)+self.gate * final_x
        # x = x  + self.drop_path(final_x, self.depth-1, pts)

        return x, c_loss


class Stage(nn.Module):
    def __init__(self, args, depth=0):
        super().__init__()

        self.depth = depth
        self.up_depth = len(args.depths) - 1

        self.first = first = depth == 0
        self.last = last = depth == self.up_depth

        self.k = args.ks[depth]

        self.cp = cp = args.use_cp
        cp_bn_momentum = args.cp_bn_momentum if cp else args.bn_momentum

        dim = args.dims[depth]
        nbr_out_dim = args.dims[0]
        self.nbr_bn = nn.BatchNorm1d(dim, momentum=args.bn_momentum)
        nn.init.constant_(self.nbr_bn.weight, 0.8 if first else 0.2)
        self.nbr_proj = nn.Sequential(
            nn.BatchNorm1d(nbr_out_dim, momentum=args.bn_momentum),
            nn.Linear(nbr_out_dim, nbr_out_dim * 2),
            args.act(),
            nn.Linear(nbr_out_dim * 2, dim, bias=False)
        )

        self.sp_dim = nbr_out_dim
        if self.depth <= 1:
            self.spse_m = nn.Parameter(
                generate_spse_matrix(nbr_out_dim, args.all_dist if self.depth == 1 else args.all_dist0).flatten())
            # spse_m: torch.cat([coor, scale, axis, angle], dim=1) nbr_dim,10 coor:坐标dim,3 scale:缩放dim,3,axis:坐标轴矫正dim,3 angle:角度校正dim,1
        if first:
            height = torch.linspace(0, 1, nbr_out_dim)
            height = height[torch.randperm(nbr_out_dim)]  # 生成nbr_dim个随机序号
            self.spse_h = nn.Parameter(height)  # nbr_dim 随机排列的等间隔序列
            col = torch.rand(3 * nbr_out_dim)
            self.spse_c = nn.Parameter(col)  # 3*nbr_dim

        if not first:
            in_dim = args.dims[depth - 1]
            self.lfp = LFP(in_dim, dim, args.bn_momentum, 0.3)
            self.skip_proj = nn.Sequential(
                nn.Linear(in_dim, dim, bias=False),
                nn.BatchNorm1d(dim, momentum=args.bn_momentum)
            )
            nn.init.constant_(self.skip_proj[1].weight, 0.3)

        self.blk = Block(dim, args.depths[depth], args.drop_paths[depth], args.mlp_ratio, cp_bn_momentum, args.act,
                         self.k)
        self.drop = DropPath(args.head_drops[depth])
        self.postproj = nn.Sequential(
            nn.BatchNorm1d(dim, momentum=args.bn_momentum),
            nn.Linear(dim, args.head_dim, bias=False),
        )
        nn.init.constant_(self.postproj[0].weight, (args.dims[0] / dim) ** 0.5)

        self.cor_std = 1 / args.cor_std[depth]
        self.cor_head = nn.Sequential(
            nn.Linear(dim, 32, bias=False),
            nn.BatchNorm1d(32, momentum=args.bn_momentum),
            args.act(),
            nn.Linear(32, 3, bias=False),
        )

        if not last:
            self.sub_stage = Stage(args, depth + 1)

    def local_aggregation(self, x, knn, pts):
        x = x.unsqueeze(0)
        x, c_loss = self.blk(x, knn, pts)
        x = x.squeeze(0)
        return x, c_loss

    def forward(self, x, xyz, prev_knn, indices, pts_list):
        """
        x: N x C
        xyz:N,3
        prev_knn:b,n,k,b,n,4 knn索引+坐标补0
        indices:indices = [
    # ===== 下采样阶段 =====
    sub_indices_3,      # 尺度3的下采样索引（最粗尺度）
    knn_indices_3,      # 尺度3的KNN索引 (k=32)

    sub_indices_2,      # 尺度2的下采样索引
    knn_indices_2,      # 尺度2的KNN索引 (k=32)

    sub_indices_1,      # 尺度1的下采样索引
    knn_indices_1,      # 尺度1的KNN索引 (k=32)

    sub_indices_0,      # 尺度0的下采样索引（最细尺度）
    knn_indices_0,      # 尺度0的KNN索引 (k=24)

    # ===== 回传阶段 =====
    back_indices_2,     # 尺度2→尺度1的回传索引
    back_indices_1,     # 尺度1→尺度0的回传索引
    back_indices_0      # 尺度3→尺度2的回传索引（注意顺序！）
]
        """
        # downsampling
        if not self.first:
            ids = indices.pop()  # 提出下采样点的idx
            xyz = xyz[ids]
            x = self.skip_proj(x)[ids] + self.lfp(x.unsqueeze(0), prev_knn).squeeze(0)[
                ids]  # 在这里进行一波邻域聚合，但是是基于通道位置的自适应池化

        knn = indices.pop()  # 提出当前knn的idx

        N, k = knn.shape
        # rel_k = torch.randint(self.k - 1, (N, 1), device=x.device)
        # rel_k = torch.gather(knn[:, 1:], 1, rel_k).squeeze(1)
        # rel_cor = (xyz[rel_k] - xyz)
        # rel_cor = rel_cor.view(-1, 3)[::[523, 131, 31, 17, 7][self.depth]].mul_(self.cor_std)
        # r = rel_cor.square().sum(dim=1).sqrt().flatten()
        # all_dist[self.depth].append(r)
        # if len(all_dist[0]) == 600:
        #     dist = torch.cat(all_dist[self.depth][1:]).flatten().sort()[0]
        #     torch.save(dist.cpu(), f"dist{self.depth}.pt")
        #     if self.last:
        #         exit()
        # print(f"xyz:{xyz.shape},:xyz[0]{xyz[0,0,:]}")

        # spatial encoding
        lxyz = F.pad(xyz, (0, 1)) * self.cor_std  # 一维扩充xyz，在右侧最后一维填充0
        nbr_knn = knn.unsqueeze(0).to(torch.int32)  # 1,n,k

        if self.first:
            ipt = torch.cat([lxyz, x], dim=1).view(1, -1, 8)  # cat(n,4),(n,4)在最后一维，形成1,n,8的输入特征，xyz0 rgbh
            m = prepare_m(self.spse_m).view(self.sp_dim, 12).transpose(0, 1).contiguous()  # 12,nbr_dim
            m = torch.cat([m, self.spse_c.view(3, self.sp_dim), self.spse_h.view(1, self.sp_dim)],
                          dim=0)  # cat 12,3,1 dim 0-8旋转矩阵，9-11平移量，12-14颜色值，15高度值
            nbr = knn_spse_4(ipt, nbr_knn, m, self.training).view(N,
                                                                  -1).sqrt()  # 对输入点特征：xyz,rgb,h进行变换，对邻域中心点XYZ进行三轴旋转与缩放操作，然后计算出三轴偏移量
            # ，同时对邻域点的XYZ也进行相同的三轴旋转与缩放操作，再加上中心点的三轴偏移量
            # 对rgb、h直接进行相加偏移量
        else:
            if self.depth == 1:
                global spse_m
                spse_m = prepare_m(self.spse_m).view(self.sp_dim, 12).transpose(0,
                                                                                1).contiguous()  # 将原有的M变为三轴旋转矩阵+3维平移向量
            ipt = lxyz.view(1, -1, 4)
            nbr = knn_spse(ipt, nbr_knn, spse_m, self.training).view(N,
                                                                     -1).sqrt()  # 这里的ipt输入特征只包含xyz，因此变换也就只剩下三轴平移旋转与缩放
            # 以邻域点的相对坐标进行旋转后且加入偏移量后的最小距离点为该通道的代表点
        nbr = self.nbr_proj(nbr)  # 将经过逐通道变换的邻域点特征（主要是低维度的几何变换）映射至语义特征维度
        nbr = self.nbr_bn(nbr)
        x = nbr if self.first else nbr + x

        # main block
        knn = (nbr_knn, lxyz.view(1, -1, 4))
        pts = pts_list.pop() if pts_list is not None else None
        x, c_loss = checkpoint(self.local_aggregation, x, knn,
                               pts) if self.training and self.cp else self.local_aggregation(x, knn, pts)

        # get subsequent feature maps
        if not self.last:
            sub_x, sub_c = self.sub_stage(x, xyz, knn, indices, pts_list)
        else:
            sub_x = sub_c = None

        # regularization
        if self.training:
            rel_k = torch.randint(self.k, (N, 1), device=x.device)  # 随机挑选一个邻域点
            rel_k = torch.gather(knn[0].long().squeeze(0), 1, rel_k).squeeze(1)  # 挑出邻域点的id
            rel_cor = (xyz[rel_k] - xyz)  # 计算邻域点与中心点的相对坐标
            rel_cor.mul_(self.cor_std)  # 乘标准差
            # print(rel_cor.std(dim=0))
            rel_p = x[rel_k] - x  # 随机邻域点的特征偏差，特征空间的差异ReLU
            rel_p = self.cor_head(rel_p)  # 从特征偏差上预测坐标
            closs = F.mse_loss(rel_p, rel_cor)  # 计算均方误差，这个损失是强制网络在编码时让相近的坐标带有类似的含义，校正特征空间的分布
            # print("closs:", closs.shape,closs)
            sub_c = sub_c + closs + c_loss if sub_c is not None else closs + c_loss

        # upsampling
        x = self.postproj(x)  # 映射到统一的head维度上，因此可以直接叠加
        if not self.first:
            back_nn = indices[self.depth - 1]  # 回到上一层的点坐标
            x = x[back_nn]  #
        x = self.drop(x)
        sub_x = sub_x + x if sub_x is not None else x

        return sub_x, sub_c


class DelaSemSeg(nn.Module):
    def __init__(self, args):
        super().__init__()

        # bn momentum for checkpointed layers
        args.cp_bn_momentum = 1 - (1 - args.bn_momentum) ** 0.5

        self.stage = Stage(args)

        hid_dim = args.head_dim
        out_dim = args.num_classes

        self.head = nn.Sequential(
            nn.BatchNorm1d(hid_dim, momentum=args.bn_momentum),
            args.act(),
            nn.Linear(hid_dim, out_dim)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, xyz, x, indices, pts_list=None):
        indices = indices[:]
        x, closs = self.stage(x, xyz, None, indices, pts_list)
        if self.training:
            return self.head(x), closs
        return self.head(x)

