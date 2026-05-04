"""Official implementation of PointNext
PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies
https://arxiv.org/abs/2206.04670
Guocheng Qian, Yuchen Li, Houwen Peng, Jinjie Mai, Hasan Abed Al Kader Hammoud, Mohamed Elhoseiny, Bernard Ghanem
"""
from typing import List, Type
import logging
import torch
import torch.nn as nn
from Cython.Compiler.Naming import self_cname

from .debug_invvit import drop_path
#from .debug_invvit import alpha
from ..build import MODELS
from ..layers import create_convblock1d, create_convblock2d, create_act, CHANNEL_MAP, \
    create_grouper, furthest_point_sample, random_sample, three_interpolation, get_aggregation_feautres,grouping_operation
from ..lib import ctwc_b11,ctwc_b2,ctwc_wc ,ctwc_wc1
import torch.nn.functional as F
from timm.models.layers import DropPath

class FeatureRecorder(nn.Module):
    """轻量级特征记录器，使用环形缓冲区"""
    def __init__(self, max_depth=12, feature_dim=256):
        super().__init__()
        self.max_depth = max_depth
        self.feature_dim = feature_dim
        self.current_idx = 0
        self.is_full = False

        # 不再预分配固定缓冲区
        self.feature_buffer = None

    def reset(self):
        """重置缓冲区，应在每个batch开始时调用"""
        self.current_idx = 0
        self.is_full = False
        # 不释放 buffer，而是重用
        if hasattr(self, 'feature_buffer') and self.feature_buffer is not None:
            # 重置为0，而不是创建新的
            self.feature_buffer.zero_()

    def ensure_buffer_size(self, batch_size, num_points, device):
        """确保缓冲区大小匹配输入"""
        if self.feature_buffer is None:
            # 第一次初始化缓冲区
            self.feature_buffer = torch.zeros(
                batch_size, self.feature_dim, num_points, self.max_depth,
                device=device, requires_grad=False
            )
        else:
            B_old, C_old, N_old, D_old = self.feature_buffer.shape

            # 检查是否需要调整大小
            need_resize = (B_old != batch_size) or (N_old != num_points)

            if need_resize:
                # 创建新的缓冲区
                new_buffer = torch.zeros(
                    batch_size, self.feature_dim, num_points, self.max_depth,
                    device=device, requires_grad=False
                )

                # 尝试从旧缓冲区复制数据（只复制共有的部分）
                if self.current_idx > 0:
                    # 确定可以复制的范围
                    min_batch = min(B_old, batch_size)
                    min_points = min(N_old, num_points)
                    min_depth = min(self.current_idx, D_old)

                    # 只复制两个缓冲区共有的部分
                    new_buffer[:min_batch, :, :min_points, :min_depth] = \
                        self.feature_buffer[:min_batch, :, :min_points, :min_depth]

                self.feature_buffer = new_buffer

    def record(self, features):
        """记录特征到环形缓冲区"""
        # features: [B, C, N]
        B, C, N = features.shape

        # 确保缓冲区大小正确
        self.ensure_buffer_size(B, N, features.device)

        # 压缩特征维度（只取部分通道，减少内存）
        compressed = features

        # 存入环形缓冲区
        idx = self.current_idx % self.max_depth
        self.feature_buffer[:, :, :, idx].copy_(compressed.detach())

        self.current_idx += 1
        if self.current_idx >= self.max_depth:
            self.is_full = True

    def get_recorded_features(self):
        """获取已记录的特征"""
        if self.feature_buffer is None:
            return None
        if not self.is_full:
            return self.feature_buffer[:, :, :, :self.current_idx]
        return self.feature_buffer

    def __del__(self):
        """析构函数，确保显存释放"""
        try:
            if hasattr(self, 'feature_buffer') and self.feature_buffer is not None:
                # 先删除引用，再尝试清理缓存
                del self.feature_buffer
                # 避免在CUDA上下文可能已破坏时调用empty_cache
                if torch.cuda.is_initialized():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
        except:
            # 如果CUDA上下文已破坏，静默失败
            pass

class CTWC_Block(nn.Module):
    def __init__(self, in_dim,act, mlp_ratio, bn_momentum,depth,K, init=0,seq=None,group_args=None,drop_path_rate=None,
                 norm_args=None, act_args=None,conv_args=None,less_act=False,):
        super().__init__()
        hid_dim = round(in_dim * mlp_ratio)
        self.group_args = group_args
        self.group_args.return_idx = True
        self.grouper = create_grouper(self.group_args)
        self.depth = depth
        self.seq = seq
        self.dim = in_dim
        pwconv = []
        #channels = [in_dim,hid_dim,in_dim]
        channels = [in_dim, in_dim]
        # point wise after depth wise conv (without last layer) #不再使用邻域聚合，而是1d的单通道卷积深化
        # for i in range(len(channels) - 1):
        #     pwconv.append(create_convblock1d(channels[i], channels[i + 1],
        #                                      norm_args=norm_args,
        #                                      act_args=act_args if
        #                                      (i != len(channels) - 2) and not less_act else None,
        #                                      **conv_args)
        #                   )
        # self.pwconv = nn.Sequential(*pwconv)
        self.pwconv = nn.Sequential(nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1,bias=True),
                                    nn.ReLU(inplace=True),
                                    #nn.BatchNorm1d(in_dim),
                                    #nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(in_dim),)
                                    # nn.ReLU(inplace=True),)



        # self.convs = create_convblock1d(in_dim, in_dim,
        #                                 norm_args=norm_args,
        #                                 act_args=None if i == (
        #                                         len(channels) - 2) and not less_act else act_args,
        #                                 **conv_args)
        self.act = create_act(act_args)
        #self.conv = create_convblock1d(in_dim,hid_dim)

        weight_dim = in_dim //4
        self.channel_scale = nn.Parameter(torch.ones(weight_dim))
        self.channel_bias = nn.Parameter(torch.zeros(weight_dim))
        #self.bias = nn.Parameter(torch.tensor(0.1))
        #conv_dim = int(weight_dim*1.5)
        self.dropout = nn.Dropout(0.2)
        self.drop_path_rates = {
            128: 0.025,  # 较小维度，较小drop率
            256: 0.05,  # 中等维度
            512: 0.075,  # 较大维度
            1024: 0.1,  # 最大维度，最大drop率
        }

        # 根据输入维度动态设置drop率
        drop_rate = drop_path_rate if drop_path_rate is not None else self.drop_path_rates.get(in_dim, 0.2)
        self.drop_path = DropPath(drop_rate) if drop_rate > 0 else nn.Identity()
        self.gate = nn.Parameter(torch.tensor(2.0))
        self.pool = lambda x: torch.mean(x, dim=-2, keepdim=False)
            #lambda x: torch.max(x, dim=-1, keepdim=False)[0]

        #pool = lambda x: torch.mean(x, dim=-1, keepdim=False)
        self.a = nn.Parameter(torch.ones(weight_dim))  # 控制斜率
        self.b = nn.Parameter(torch.zeros(weight_dim))  # 控制中心点
        self.c = nn.Parameter(torch.ones(weight_dim))  # 控制幅度

        # if in_dim ==128:
        #     self.scale = 3.0#3.0
        # elif in_dim ==256:
        #     self.scale = 3.0#4.0
        # elif in_dim ==512:
        #     self.scale = 6.0#2.0
        # elif in_dim ==1024:
        #     self.scale = 9.0

        if in_dim ==64:
            self.scale = 4.0#3.0
        elif in_dim ==128:
            self.scale = 4.0#4.0
        elif in_dim ==256:
            self.scale = 6.0#2.0
        elif in_dim ==512:
            self.scale = 8.0

    def forward(self, pf):
        p,x=pf
        seq = self.seq.get_recorded_features() #b,c_,n,t
        b, c_, n, t = seq.shape
        seq = seq.permute(0, 2, 3, 1).contiguous()  # b,n,t,c

        res = x
        B,C,N = x.shape

        _, fj, knn_idx = self.grouper(p, p, x)       #bnk    dp,fj

        ################ctwc_b
        scale = torch.tensor(self.scale, dtype=torch.float, device=x.device)
        c_dist = ctwc_b11(seq,knn_idx,scale)   #bnkc_        seq:bntc

        mask = (c_dist == 1.0).all(dim=-1)
        mask_expanded = mask.unsqueeze(-1)  # 形状: [B, N, K, 1]
        mask_expanded = mask_expanded.expand_as(c_dist)  # 形状: [B, N, K, C]
        c_dist[mask_expanded] = 0.5
        #c_dist = torch.pow(10,c_dist-1.0)
        # base = torch.sigmoid(self.channel_scale) * 4 + 6
        # c_dist = torch.pow(base.view(1,1,1,-1),c_dist-1.0)

        #pc = c_dist*self.channel_scale+self.channel_bias
        pg = c_dist.mean(dim=-1)    #bnk
        pg_var = torch.var(pg,dim=-1,keepdim=True)
        pg_max = pg.max(dim=2, keepdim=True)[0]  # [B, N, 1]
        pg_min = pg.min(dim=2, keepdim=True)[0]  # [B, N, 1]
        pg_range = pg_max - pg_min
        narrow_range_mask = pg_range < 0.01
        max_pg_var = (pg_range**2)/4.0
        max_pg_var = torch.clamp(max_pg_var,1e-3)
        pn_raw = 1.0-torch.exp(-pg_var/(max_pg_var+1e-5))
        pn_raw = torch.clamp(pn_raw,min=1e-3,max=1.0)
        pn = torch.where(narrow_range_mask,
                             torch.ones_like(pn_raw),
                             pn_raw) #bn1

        # ==================== 7. 饱和处理 ====================

        gamma = torch.exp(3 * (0.7-pn))
        #print("gate",self.gate)
        #gamma = torch.exp(torch.clamp(self.gate * ( 0.7-pn), max=20.0))
        pg_ad = torch.pow(pg,gamma)

        #pc = F.softmax(c_dist*self.channel_scale+self.channel_bias,dim=-2)
        #pc = c_dist*self.channel_scale+self.channel_bias

        la = self.a.view(*([1] * (c_dist.dim() - 1)), -1)
        lb = self.b.view(*([1] * (c_dist.dim() - 1)), -1)
        lc = torch.sigmoid(self.c).view(*([1] * (c_dist.dim() - 1)), -1)  # c∈[0,1]
        pc = lc * (torch.sigmoid(la * (c_dist - lb)) - torch.sigmoid(-la * lb))
        w = pg_ad.unsqueeze(-1) *pc
        # # ==================== 测试时统计pg和pn分布 ====================
        # if not self.training:
        #     # print("seq:",seq.shape,seq[0][0][0:-1][0:3])
        #     # print("cdist:",c_dist.shape,c_dist[0][0][0:8][0:3])
        #     # print("center seq (point 0, channel 0, all T):")
        #     # print(seq[0, 0, :, 0])  # 原始 seq 的 [B,N,T,C] 格式
        #     # seq = seq.view(c_dist.shape[0],c_dist.shape[1],-1).permute(0,2,1).contiguous() #bntc->b,tc,n
        #     # seq_knn = grouping_operation(seq,knn_idx)   #b,tc,nk
        #     # seq_knn = seq_knn.view(b,t,c_dist.shape[-1],n,knn_idx.shape[-1])    #b t c n k
        #     # seq_knn = seq_knn.permute(0,3,4,2,1)    #b t c n k -> b n k c t
        #     # # print("seq_knn",seq_knn[0][0][0:8][0:3])
        #     # print(f"knn_idx[0,0,0]: {knn_idx[0, 0, 0]}")  # 第一个点的第一个邻居索引
        #     #
        #     # print(f"\nseq_knn[0][0][0] (neighbor 0, channel 0):")
        #     # print(seq_knn[0][0][0][0])  # [T]
        #
        #     print("pg:",pg.shape,pg[0][0][0:8][0:3])
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

        #del c_dist, pc, pg, pn,alpha
        #x = self.pool(fj*w)
        #x=x.permute(0,2,1).contiguous()


        # w = w.permute(0,3,1,2).contiguous() #bnkc->bcnk
        # w = self.dropout(w)
        # w = w.permute(0,2,3,1).contiguous() #bcnk ->bnkc

        fj = fj #- x.unsqueeze(-1)
        fj = fj.permute(0,2,3,1).contiguous()   #bcnk->bnkc
        b,n,k,c = fj.shape
        fj = fj.view(b,n,k,-1,4)
        fj = fj * w.unsqueeze(-1)
        x = self.pool(fj.view(b,n,k,c)) #bnkc->bnc
        del fj,w,c_dist


# ##########################text#####################################
        x = x.permute(0,2,1)
        #x = res+self.drop_path(x)
        x = self.pwconv(x)#+self.bias
        # print(f"pwconv output mean: {x.mean().item():.6f}")
        # print(f"pwconv output std: {x.std().item():.6f}")
        # print(f"pwconv output min: {x.min().item():.6f}")
        # print(f"pwconv output max: {x.max().item():.6f}")
        # print(f"pwconv.abs().mean(): {x.abs().mean().item():.6f}")
        # print(f"res output mean: {res.mean().item():.6f}")
        # print(f"res output std: {res.std().item():.6f}")
        # print(f"res output min: {res.min().item():.6f}")
        # print(f"res output max: {res.max().item():.6f}")
        f = x + res
        # print("gate",self.gate)
        #gate = torch.sigmoid(self.gate)
        #print(f"gate: {gate.item():.6f}")
        #f = res * (1 - gate) + gate*x #* self.drop_path(x)
        #x = self.drop_path(x) +res

        #print("x",x.shape)
        #x = self.convs(x)
        #f = self.act(f)

        #x = self.mlp(x.view(B*N, -1)).view(B, N, -1)
        #x = res*(1-self.gate) + self.gate*self.drop_path(x.permute(0, 2, 1))
        #x = x.permute(0, 2,1)
        #f = self.act(x.contiguous())


        c_loss = None



        if self.training:
            c_loss =0.0
            # w_flat = w

            losses = []


            # target_magnitude = 0.3
            # magnitude = torch.abs(x).mean()
            # mag_loss = torch.relu(target_magnitude - magnitude)
            # c_loss = c_loss + 0.1 * mag_loss
            #
            # # 同时约束 bias 不要太小
            # bias_loss = torch.relu(0.5 - torch.abs(self.bias))
            # c_loss = c_loss + 0.05 * bias_loss


            # 1. b应该小于0（最重要！）
            # 惩罚b>0的情况
            # losses.append(F.relu(-lb).mean())  # b>0就惩罚
            #
            # # 2. a应该在合理范围（大概0.2-2.0）
            # # 防止a太大或太小
            # losses.append(F.relu(0.2 - la).mean())  # 惩罚a<0.2
            # losses.append(F.relu(la - 2.0).mean())  # 惩罚a>2.0
            #
            # # 3. c应该在合理范围（大概0.2-0.8）
            # losses.append(F.relu(0.2 - lc).mean())  # 惩罚c<0.2
            # losses.append(F.relu(lc - 0.8).mean())  # 惩罚c>0.8
            #
            # # 加权组合：b约束最重要
            # weights = [0.5, 0.1, 0.15, 0.15]
            # c_loss = c_loss + sum(w * l for w, l in zip(weights, losses))
            # 1. b应该小于0（最重要！）
            # 惩罚b>0的情况
            # losses.append(F.relu(lb).mean())  # b>0就惩罚
            #
            # 2. a应该在合理范围（大概0.2-2.0）
            # 防止a太大或太小
            # losses.append(F.relu(0.2 - la).mean())  # 惩罚a<0.2
            # losses.append(F.relu(la - 2.0).mean())  # 惩罚a>2.0
            #
            # # 3. c应该在合理范围（大概0.2-0.8）
            # losses.append(F.relu(0.2 - lc).mean())  # 惩罚c<0.2
            # losses.append(F.relu(lc - 0.8).mean())  # 惩罚c>0.8

            # 加权组合：b约束最重要
            # weights = [0.5, 0.1, 0.1, 0.15, 0.15]
            # c_loss = sum(w * l for w, l in zip(weights, losses))
            # pn_mean = pn.mean()
            # c_loss +=0.75 * torch.relu(torch.abs(pn_mean - 0.4) - 0.1)

            # print("pn_mean", pn_mean)
            # print("c_loss",c_loss)

        return p, f,c_loss
        # else:
        #     print("seq is None")
        #     return pf



def get_reduction_fn(reduction):
    reduction = 'mean' if reduction.lower() == 'avg' else reduction
    assert reduction in ['sum', 'max', 'mean']
    if reduction == 'max':
        pool = lambda x: torch.max(x, dim=-1, keepdim=False)[0]
    elif reduction == 'mean':
        pool = lambda x: torch.mean(x, dim=-1, keepdim=False)
    elif reduction == 'sum':
        pool = lambda x: torch.sum(x, dim=-1, keepdim=False)
    return pool


class LocalAggregation(nn.Module):
    """Local aggregation layer for a set 
    Set abstraction layer abstracts features from a larger set to a smaller set
    Local aggregation layer aggregates features from the same set
    """

    def __init__(self,
                 channels: List[int],
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'},
                 group_args={'NAME': 'ballquery', 'radius': 0.1, 'nsample': 16},
                 conv_args=None,
                 feature_type='dp_fj',
                 reduction='max',
                 last_act=True,
                 **kwargs
                 ):
        super().__init__()
        if kwargs:
            logging.warning(f"kwargs: {kwargs} are not used in {__class__.__name__}")
        channels[0] = CHANNEL_MAP[feature_type](channels[0])
        convs = []
        for i in range(len(channels) - 1):  # #layers in each blocks
            convs.append(create_convblock2d(channels[i], channels[i + 1],
                                            norm_args=norm_args,
                                            act_args=None if i == (
                                                    len(channels) - 2) and not last_act else act_args,
                                            **conv_args)
                         )
        self.convs = nn.Sequential(*convs)
        self.grouper = create_grouper(group_args)
        self.reduction = reduction.lower()
        self.pool = get_reduction_fn(self.reduction)
        self.feature_type = feature_type

    def forward(self, pf) -> torch.Tensor:
        # p: position, f: feature
        p, f = pf
        # neighborhood_features
        #f = self.convs(f)
        dp, fj = self.grouper(p, p, f)  #dp:grouped_xyz [b,3,n,m],fj:grouped_feature [b,c,n,nsample]
        fj = get_aggregation_feautres(p, dp, f, fj, self.feature_type)  #按照指定顺序将p:原xyz dp:聚合后的xyz n,nsample f:原特征 fj:聚合特征 拼接在一起
        # B, C, N = f.shape
        # zeros = torch.zeros(B, 3, N, device=f.device, dtype=f.dtype)
        # f_padded = torch.cat([zeros, f], dim=1)
        f = self.pool(self.convs(fj))
        #f = self.pool(self.convs(fj-f_padded.unsqueeze(-1)))
        #f = self.pool(fj-f.unsqueeze(-1))
        """ DEBUG neighbor numbers. 
        if f.shape[-1] != 1:
            query_xyz, support_xyz = p, p
            radius = self.grouper.radius
            dist = torch.cdist(query_xyz.cpu(), support_xyz.cpu())
            points = len(dist[dist < radius]) / (dist.shape[0] * dist.shape[1])
            logging.info(
                f'query size: {query_xyz.shape}, support size: {support_xyz.shape}, radius: {radius}, num_neighbors: {points}')
        DEBUG end """
        return f    #[b,c,n]


class SetAbstraction(nn.Module):
    """The modified set abstraction module in PointNet++ with residual connection support
    """

    def __init__(self,
                 in_channels, out_channels,
                 layers=1,
                 stride=1,
                 group_args={'NAME': 'ballquery',
                             'radius': 0.1, 'nsample': 16},
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'},
                 conv_args=None,
                 sampler='fps',
                 feature_type='dp_fj',
                 use_res=False,
                 is_head=False,
                 **kwargs, 
                 ):
        super().__init__()
        self.stride = stride
        self.is_head = is_head
        self.all_aggr = not is_head and stride == 1
        self.use_res = use_res and not self.all_aggr and not self.is_head
        self.feature_type = feature_type

        mid_channel = out_channels // 2 if stride > 1 else out_channels
        channels = [in_channels] + [mid_channel] * \
                   (layers - 1) + [out_channels]
        channels[0] = in_channels if is_head else CHANNEL_MAP[feature_type](channels[0])

        if self.use_res:
            self.skipconv = create_convblock1d(
                in_channels, channels[-1], norm_args=None, act_args=None) if in_channels != channels[
                -1] else nn.Identity()
            self.act = create_act(act_args)

        # actually, one can use local aggregation layer to replace the following
        create_conv = create_convblock1d if is_head else create_convblock2d     #conv1d这里是自定义了一个初始化函数，默认kernelsize=1
        convs = []
        for i in range(len(channels) - 1):
            convs.append(create_conv(channels[i], channels[i + 1],
                                     norm_args=norm_args if not is_head else None,
                                     act_args=None if i == len(channels) - 2
                                                      and (self.use_res or is_head) else act_args,
                                     **conv_args)
                         )
        self.convs = nn.Sequential(*convs)
        if not is_head:
            if self.all_aggr:
                group_args.nsample = None
                group_args.radius = None
                group_args.return_idx = False
            group_args.return_idx = False
            self.grouper = create_grouper(group_args)
            self.pool = lambda x: torch.max(x, dim=-1, keepdim=False)[0]
            if sampler.lower() == 'fps':
                self.sample_fn = furthest_point_sample
            elif sampler.lower() == 'random':
                self.sample_fn = random_sample

    def forward(self, pf):
        p, f = pf   #p [b,n,3], f: [b,c,n]
        if self.is_head:
            f = self.convs(f)  # (n, c)
        else:
            if not self.all_aggr:
                idx = self.sample_fn(p, p.shape[1] // self.stride).long()
                new_p = torch.gather(p, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
            else:
                new_p = p
            """ DEBUG neighbor numbers. 
            query_xyz, support_xyz = new_p, p
            radius = self.grouper.radius
            dist = torch.cdist(query_xyz.cpu(), support_xyz.cpu())
            points = len(dist[dist < radius]) / (dist.shape[0] * dist.shape[1])
            logging.info(f'query size: {query_xyz.shape}, support size: {support_xyz.shape}, radius: {radius}, num_neighbors: {points}')
            DEBUG end """
            if self.use_res or 'df' in self.feature_type:
                fi = torch.gather(f, -1, idx.unsqueeze(1).expand(-1, f.shape[1], -1)) #fi:根据fps下采样idx选择得到的特征，用以做差值
                if self.use_res:
                    identity = self.skipconv(fi)
            else:
                fi = None
            dp, fj = self.grouper(new_p, p, f)
            fj = get_aggregation_feautres(new_p, dp, fi, fj, feature_type=self.feature_type)
            f = self.pool(self.convs(fj))
            if self.use_res:
                f = self.act(f + identity)  #残差，直接在经过conv的特征f上加上最初的（经过简单1d处理的）f
            p = new_p
        return p, f


class FeaturePropogation(nn.Module):
    """The Feature Propogation module in PointNet++
    """

    def __init__(self, mlp,
                 upsample=True,
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'}
                 ):
        """
        Args:
            mlp: [current_channels, next_channels, next_channels]
            out_channels:
            norm_args:
            act_args:
        """
        super().__init__()
        if not upsample:
            self.linear2 = nn.Sequential(
                nn.Linear(mlp[0], mlp[1]), nn.ReLU(inplace=True))
            mlp[1] *= 2
            linear1 = []
            for i in range(1, len(mlp) - 1):
                linear1.append(create_convblock1d(mlp[i], mlp[i + 1],
                                                  norm_args=norm_args, act_args=act_args
                                                  ))
            self.linear1 = nn.Sequential(*linear1)
        else:
            convs = []
            for i in range(len(mlp) - 1):
                convs.append(create_convblock1d(mlp[i], mlp[i + 1],
                                                norm_args=norm_args, act_args=act_args
                                                ))
            self.convs = nn.Sequential(*convs)

        self.pool = lambda x: torch.mean(x, dim=-1, keepdim=False)

    def forward(self, pf1, pf2=None):
        # pfb1 is with the same size of upsampled points
        if pf2 is None:
            _, f = pf1  # (B, N, 3), (B, C, N)
            f_global = self.pool(f)
            f = torch.cat(
                (f, self.linear2(f_global).unsqueeze(-1).expand(-1, -1, f.shape[-1])), dim=1)
            f = self.linear1(f)
        else:
            p1, f1 = pf1    #pf1: n1 [b,n1.3],[b,c,n1]
            p2, f2 = pf2    #pf2: n0 [b,n0,3].[b,c,n0]
            if f1 is not None:
                f = self.convs(
                    torch.cat((f1, three_interpolation(p1, p2, f2)), dim=1))
            else:
                f = self.convs(three_interpolation(p1, p2, f2))
        return f


class InvResMLP(nn.Module):
    def __init__(self,
                 in_channels,
                 norm_args=None,
                 act_args=None,
                 aggr_args={'feature_type': 'dp_fj', "reduction": 'max'},
                 group_args={'NAME': 'ballquery'},
                 conv_args=None,
                 expansion=1,
                 use_res=True,
                 num_posconvs=2,
                 less_act=False,
                 seq = None,
                 **kwargs
                 ):
        super().__init__()
        self.use_res = use_res
        mid_channels = int(in_channels * expansion)
        self.convs = LocalAggregation([in_channels, in_channels],
                                      norm_args=norm_args, act_args=act_args if num_posconvs > 0 else None,
                                      group_args=group_args, conv_args=conv_args,
                                      **aggr_args, **kwargs)    #利用knn或者ballquery进行领域查询，并聚合
        if num_posconvs < 1:
            channels = []
        elif num_posconvs == 1:
            channels = [in_channels, in_channels]
        else:
            channels = [in_channels, mid_channels, in_channels]
        pwconv = []
        # point wise after depth wise conv (without last layer) #不再使用邻域聚合，而是1d的单通道卷积深化
        for i in range(len(channels) - 1):
            pwconv.append(create_convblock1d(channels[i], channels[i + 1],
                                             norm_args=norm_args,
                                             act_args=act_args if
                                             (i != len(channels) - 2) and not less_act else None,
                                             **conv_args)
                          )
        self.pwconv = nn.Sequential(*pwconv)
        self.act = create_act(act_args)

        self.drop_path_rates = {
            64: 0.025,  # 较小维度，较小drop率
            128: 0.05,  # 中等维度
            256: 0.075,  # 较大维度
            512: 0.075,  # 最大维度，最大drop率
        }
        drop_path_rate = None
        drop_rate = drop_path_rate if drop_path_rate is not None else self.drop_path_rates.get(in_channels, 0.1)
        self.drop_path = DropPath(drop_rate) if drop_rate > 0 else nn.Identity()
        if seq is None:
            self.seq=[]
        else:
            self.seq = seq

    def forward(self, pf):
        p, f = pf
        identity = f
        f = self.convs([p, f])  #领域聚合，inchannel进，inchannel出
        #f1 = f+identity
        # cf = identity
        # df = f
        # hf = df - cf
        seq_xi = torch.mean(f.view(f.shape[0], -1, 4,f.shape[-1]), dim=-2, keepdim=False)
        self.seq.record(seq_xi)
        #self.seq.append(seq_xi)
        f = self.pwconv(f)
        #f = f1+f2
        # cf = df
        # hf = f -cf
        # seq_xi = torch.mean(f.view(f.shape[0], -1, 4,f.shape[-1]), dim=-2, keepdim=False)
        # self.seq.record(seq_xi)
        #self.seq.append(seq_xi)
        # print("self.seq in InvResMlp",len(self.seq))
        if f.shape[-1] == identity.shape[-1] and self.use_res:
            f = self.drop_path(identity)+f
        f = self.act(f)
        seq_xi = torch.mean(f.view(f.shape[0], -1, 4,f.shape[-1]), dim=-2, keepdim=False)
        self.seq.record(seq_xi)
        return [p, f]


class ResBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 norm_args=None,
                 act_args=None,
                 aggr_args={'feature_type': 'dp_fj', "reduction": 'max'},
                 group_args={'NAME': 'ballquery'},
                 conv_args=None,
                 expansion=1,
                 use_res=True,
                 **kwargs
                 ):
        super().__init__()
        self.use_res = use_res
        mid_channels = in_channels * expansion
        self.convs = LocalAggregation([in_channels, in_channels, mid_channels, in_channels],
                                      norm_args=norm_args, act_args=None,
                                      group_args=group_args, conv_args=conv_args,
                                      **aggr_args, **kwargs)
        self.act = create_act(act_args)

    def forward(self, pf):
        p, f = pf
        identity = f
        f = self.convs([p, f])  #领域聚合
        if f.shape[-1] == identity.shape[-1] and self.use_res:
            f += identity   #残差聚合，直接加原始特征
        f = self.act(f)
        return [p, f]


@MODELS.register_module()
class PointNext_CTWC_Encoder(nn.Module):
    r"""The Encoder for PointNext 
    `"PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies".
    <https://arxiv.org/abs/2206.04670>`_.
    .. note::
        For an example of using :obj:`PointNextEncoder`, see
        `examples/segmentation/main.py <https://github.com/guochengqian/PointNeXt/blob/master/cfgs/s3dis/README.md>`_.
    Args:
        in_channels (int, optional): input channels . Defaults to 4.
        width (int, optional): width of network, the output mlp of the stem MLP. Defaults to 32.
        blocks (List[int], optional): # of blocks per stage (including the SA block). Defaults to [1, 4, 7, 4, 4].
        strides (List[int], optional): the downsampling ratio of each stage. Defaults to [4, 4, 4, 4].
        block (strorType[InvResMLP], optional): the block to use for depth scaling. Defaults to 'InvResMLP'.
        nsample (intorList[int], optional): the number of neighbors to query for each block. Defaults to 32.
        radius (floatorList[float], optional): the initial radius. Defaults to 0.1.
        aggr_args (_type_, optional): the args for local aggregataion. Defaults to {'feature_type': 'dp_fj', "reduction": 'max'}.
        group_args (_type_, optional): the args for grouping. Defaults to {'NAME': 'ballquery'}.
        norm_args (_type_, optional): the args for normalization layer. Defaults to {'norm': 'bn'}.
        act_args (_type_, optional): the args for activation layer. Defaults to {'act': 'relu'}.
        expansion (int, optional): the expansion ratio of the InvResMLP block. Defaults to 4.
        sa_layers (int, optional): the number of MLP layers to use in the SA block. Defaults to 1.
        sa_use_res (bool, optional): wheter to use residual connection in SA block. Set to True only for PointNeXt-S. 
    """

    def __init__(self,
                 in_channels: int = 4,
                 width: int = 32,
                 blocks: List[int] = [1, 4, 7, 4, 4],
                 strides: List[int] = [4, 4, 4, 4],
                 block: str or Type[InvResMLP] = 'InvResMLP',
                 nsample: int or List[int] = 32,
                 radius: float or List[float] = 0.1,
                 aggr_args: dict = {'feature_type': 'dp_fj', "reduction": 'max'},
                 group_args: dict = {'NAME': 'ballquery'},
                 sa_layers: int = 1,
                 sa_use_res: bool = False,
                 **kwargs
                 ):
        super().__init__()
        if isinstance(block, str):
            block = eval(block)
        self.blocks = blocks
        self.strides = strides
        self.in_channels = in_channels
        self.aggr_args = aggr_args
        self.norm_args = kwargs.get('norm_args', {'norm': 'bn'}) 
        self.act_args = kwargs.get('act_args', {'act': 'relu'}) 
        self.conv_args = kwargs.get('conv_args', None)
        self.sampler = kwargs.get('sampler', 'fps')
        self.expansion = kwargs.get('expansion', 4)
        self.sa_layers = sa_layers
        self.sa_use_res = sa_use_res
        self.use_res = kwargs.get('use_res', True)
        radius_scaling = kwargs.get('radius_scaling', 2)
        nsample_scaling = kwargs.get('nsample_scaling', 1)

        self.radii = self._to_full_list(radius, radius_scaling)
        self.nsample = self._to_full_list(nsample, nsample_scaling)
        logging.info(f'radius: {self.radii},\n nsample: {self.nsample}')

        # double width after downsampling.
        channels = []
        for stride in strides:
            if stride != 1:
                width *= 2
            channels.append(width)
        encoder = []
        for i in range(len(blocks)):
            group_args.radius = self.radii[i]
            group_args.nsample = self.nsample[i]
            encoder.append(self._make_enc(
                block, channels[i], blocks[i], stride=strides[i], group_args=group_args,
                is_head=i == 0 and strides[i] == 1
            ))
        self.encoder = nn.Sequential(*encoder)
        self.out_channels = channels[-1]
        self.channel_list = channels

    def _to_full_list(self, param, param_scaling=1):
        # param can be: radius, nsample
        param_list = []
        if isinstance(param, List):
            # make param a full list
            for i, value in enumerate(param):
                value = [value] if not isinstance(value, List) else value
                if len(value) != self.blocks[i]:
                    value += [value[-1]] * (self.blocks[i] - len(value))
                param_list.append(value)
        else:  # radius is a scalar (in this case, only initial raidus is provide), then create a list (radius for each block)
            for i, stride in enumerate(self.strides):
                if stride == 1:
                    param_list.append([param] * self.blocks[i])
                else:
                    param_list.append(
                        [param] + [param * param_scaling] * (self.blocks[i] - 1))
                    param *= param_scaling
        return param_list

    def _make_enc(self, block, channels, blocks, stride, group_args, is_head=False):
        layers = []
        radii = group_args.radius
        nsample = group_args.nsample
        group_args.radius = radii[0]
        group_args.nsample = nsample[0]
        layers.append(SetAbstraction(self.in_channels, channels,
                                     self.sa_layers if not is_head else 1, stride,
                                     group_args=group_args,
                                     sampler=self.sampler,
                                     norm_args=self.norm_args, act_args=self.act_args, conv_args=self.conv_args,
                                     is_head=is_head, use_res=self.sa_use_res, **self.aggr_args 
                                     ))
        self.in_channels = channels
        self.seq=FeatureRecorder(max_depth=(blocks-1)*2,feature_dim=channels//4)
        for i in range(1, blocks):
            group_args.radius = radii[i]
            group_args.nsample = nsample[i]
            layers.append(block(self.in_channels,
                                aggr_args=self.aggr_args,
                                norm_args=self.norm_args, act_args=self.act_args, group_args=group_args,
                                conv_args=self.conv_args, expansion=self.expansion,
                                use_res=self.use_res,
                                seq = self.seq
                                ))
        if blocks >1:
            layers.append(CTWC_Block(in_dim=self.in_channels,act=nn.GELU, mlp_ratio=1.5,
                                     bn_momentum=0.02,depth=(blocks-1)*2,K=nsample[-1], init=0.,
                                     group_args=group_args,seq=self.seq,
                                     norm_args=self.norm_args, act_args=self.act_args,conv_args=self.conv_args
                                     ))
        return nn.Sequential(*layers)

    def forward_cls_feat(self, p0, f0=None):
        if hasattr(p0, 'keys'):
            p0, f0 = p0['pos'], p0.get('x', None)
        if f0 is None:
            f0 = p0.clone().transpose(1, 2).contiguous()
        for i in range(0, len(self.encoder)):
            p0, f0 = self.encoder[i]([p0, f0])
        return f0.squeeze(-1)



    def forward_seg_feat(self, p0, f0=None):
        if hasattr(p0, 'keys'):
            p0, f0 = p0['pos'], p0.get('x', None)
        if f0 is None:
            f0 = p0.clone().transpose(1, 2).contiguous()

        def reset_recorders(module):
            for child in module.children():
                if isinstance(child, FeatureRecorder):
                    child.reset()
                else:
                    reset_recorders(child)

        reset_recorders(self)

        p, f = [p0], [f0]
        c_loss=0.0
        _c_l=0
        for i in range(0, len(self.encoder)):

            if i ==0:
                _p, _f= self.encoder[i]([p[-1], f[-1]])
            else:
                if self.training:
                    _p, _f ,_c = self.encoder[i]([p[-1], f[-1]])
                    _c_l=_c+_c_l
                else:
                    _p, _f ,_c = self.encoder[i]([p[-1], f[-1]])
            p.append(_p)
            f.append(_f)

        if self.training:
            c_loss = c_loss+_c_l
            #print("c_loss", c_loss)
            return p, f,c_loss
        else:
            return p, f

    def forward(self, p0, f0=None):
        return self.forward_seg_feat(p0, f0)


@MODELS.register_module()
class PointNext_CTWC_Decoder(nn.Module):
    def __init__(self,
                 encoder_channel_list: List[int],
                 decoder_layers: int = 2,
                 decoder_stages: int = 4, 
                 **kwargs
                 ):
        super().__init__()
        self.decoder_layers = decoder_layers
        self.in_channels = encoder_channel_list[-1]
        skip_channels = encoder_channel_list[:-1]
        if len(skip_channels) < decoder_stages:
            skip_channels.insert(0, kwargs.get('in_channels', 3))
        # the output channel after interpolation
        fp_channels = encoder_channel_list[:decoder_stages]

        n_decoder_stages = len(fp_channels)
        decoder = [[] for _ in range(n_decoder_stages)]
        for i in range(-1, -n_decoder_stages - 1, -1):
            decoder[i] = self._make_dec(
                skip_channels[i], fp_channels[i])
        self.decoder = nn.Sequential(*decoder)
        self.out_channels = fp_channels[-n_decoder_stages]

    def _make_dec(self, skip_channels, fp_channels):
        layers = []
        mlp = [skip_channels + self.in_channels] + \
              [fp_channels] * self.decoder_layers
        layers.append(FeaturePropogation(mlp))
        self.in_channels = fp_channels
        return nn.Sequential(*layers)

    def forward(self, p, f):
        for i in range(-1, -len(self.decoder) - 1, -1):
            f[i - 1] = self.decoder[i][1:](
                [p[i], self.decoder[i][0]([p[i - 1], f[i - 1]], [p[i], f[i]])])[1]
        return f[-len(self.decoder) - 1]


@MODELS.register_module()
class PointNext_CTWC_PartDecoder(nn.Module):
    def __init__(self,
                 encoder_channel_list: List[int],
                 decoder_layers: int = 2,
                 decoder_blocks: List[int] = [1, 1, 1, 1],
                 decoder_strides: List[int] = [4, 4, 4, 4],
                 act_args: str = 'relu',
                 cls_map='pointnet2',
                 num_classes: int = 16,
                 cls2partembed=None,
                 **kwargs
                 ):
        super().__init__()
        self.decoder_layers = decoder_layers
        self.in_channels = encoder_channel_list[-1]
        skip_channels = encoder_channel_list[:-1]
        fp_channels = encoder_channel_list[:-1]
        
        # the following is for decoder blocks
        self.conv_args = kwargs.get('conv_args', None)
        radius_scaling = kwargs.get('radius_scaling', 2)
        nsample_scaling = kwargs.get('nsample_scaling', 1)
        block = kwargs.get('block', 'InvResMLP')
        if isinstance(block, str):
            block = eval(block)
        self.blocks = decoder_blocks
        self.strides = decoder_strides
        self.norm_args = kwargs.get('norm_args', {'norm': 'bn'}) 
        self.act_args = kwargs.get('act_args', {'act': 'relu'}) 
        self.expansion = kwargs.get('expansion', 4)
        radius = kwargs.get('radius', 0.1)
        nsample = kwargs.get('nsample', 16)
        self.radii = self._to_full_list(radius, radius_scaling)
        self.nsample = self._to_full_list(nsample, nsample_scaling)
        self.cls_map = cls_map
        self.num_classes = num_classes
        self.use_res = kwargs.get('use_res', True)
        group_args = kwargs.get('group_args', {'NAME': 'ballquery'})
        self.aggr_args = kwargs.get('aggr_args', 
                                    {'feature_type': 'dp_fj', "reduction": 'max'}
                                    )  
        if self.cls_map == 'curvenet':
            # global features
            self.global_conv2 = nn.Sequential(
                create_convblock1d(fp_channels[-1] * 2, 128,
                                   norm_args=None,
                                   act_args=act_args))
            self.global_conv1 = nn.Sequential(
                create_convblock1d(fp_channels[-2] * 2, 64,
                                   norm_args=None,
                                   act_args=act_args))
            skip_channels[0] += 64 + 128 + 16  # shape categories labels
        elif self.cls_map == 'pointnet2':
            self.convc = nn.Sequential(create_convblock1d(16, 64,
                                                          norm_args=None,
                                                          act_args=act_args))
            skip_channels[0] += 64  # shape categories labels

        elif self.cls_map == 'pointnext':
            self.global_conv2 = nn.Sequential(
                create_convblock1d(fp_channels[-1] * 2, 128,
                                   norm_args=None,
                                   act_args=act_args))
            self.global_conv1 = nn.Sequential(
                create_convblock1d(fp_channels[-2] * 2, 64,
                                   norm_args=None,
                                   act_args=act_args))
            skip_channels[0] += 64 + 128 + 50  # shape categories labels
            self.cls2partembed = cls2partembed
        elif self.cls_map == 'pointnext1':
            self.convc = nn.Sequential(create_convblock1d(50, 64,
                                                          norm_args=None,
                                                          act_args=act_args))
            skip_channels[0] += 64  # shape categories labels
            self.cls2partembed = cls2partembed

        n_decoder_stages = len(fp_channels)
        decoder = [[] for _ in range(n_decoder_stages)]
        for i in range(-1, -n_decoder_stages - 1, -1):
            group_args.radius = self.radii[i]
            group_args.nsample = self.nsample[i]
            decoder[i] = self._make_dec(
                skip_channels[i], fp_channels[i], group_args=group_args, block=block, blocks=self.blocks[i])

        self.decoder = nn.Sequential(*decoder)
        self.out_channels = fp_channels[-n_decoder_stages]

    def _make_dec(self, skip_channels, fp_channels, group_args=None, block=None, blocks=1):
        layers = []
        radii = group_args.radius
        nsample = group_args.nsample
        mlp = [skip_channels + self.in_channels] + \
              [fp_channels] * self.decoder_layers
        layers.append(FeaturePropogation(mlp, act_args=self.act_args))
        self.in_channels = fp_channels
        for i in range(1, blocks):
            group_args.radius = radii[i]
            group_args.nsample = nsample[i]
            layers.append(block(self.in_channels,
                                aggr_args=self.aggr_args,
                                norm_args=self.norm_args, act_args=self.act_args, group_args=group_args,
                                conv_args=self.conv_args, expansion=self.expansion,
                                use_res=self.use_res
                                ))
        return nn.Sequential(*layers)

    def _to_full_list(self, param, param_scaling=1):
        # param can be: radius, nsample
        param_list = []
        if isinstance(param, List):
            # make param a full list
            for i, value in enumerate(param):
                value = [value] if not isinstance(value, List) else value
                if len(value) != self.blocks[i]:
                    value += [value[-1]] * (self.blocks[i] - len(value))
                param_list.append(value)
        else:  # radius is a scalar (in this case, only initial raidus is provide), then create a list (radius for each block)
            for i, stride in enumerate(self.strides):
                if stride == 1:
                    param_list.append([param] * self.blocks[i])
                else:
                    param_list.append(
                        [param] + [param * param_scaling] * (self.blocks[i] - 1))
                    param *= param_scaling
        return param_list

    def forward(self, p, f, cls_label):
        B, N = p[0].shape[0:2]
        if self.cls_map == 'curvenet':
            emb1 = self.global_conv1(f[-2])
            emb1 = emb1.max(dim=-1, keepdim=True)[0]  # bs, 64, 1
            emb2 = self.global_conv2(f[-1])
            emb2 = emb2.max(dim=-1, keepdim=True)[0]  # bs, 128, 1
            cls_one_hot = torch.zeros((B, self.num_classes), device=p[0].device)
            cls_one_hot = cls_one_hot.scatter_(1, cls_label, 1).unsqueeze(-1)
            cls_one_hot = torch.cat((emb1, emb2, cls_one_hot), dim=1)
            cls_one_hot = cls_one_hot.expand(-1, -1, N)
        elif self.cls_map == 'pointnet2':
            cls_one_hot = torch.zeros((B, self.num_classes), device=p[0].device)
            cls_one_hot = cls_one_hot.scatter_(1, cls_label, 1).unsqueeze(-1).repeat(1, 1, N)
            cls_one_hot = self.convc(cls_one_hot)
        elif self.cls_map == 'pointnext':
            emb1 = self.global_conv1(f[-2])
            emb1 = emb1.max(dim=-1, keepdim=True)[0]  # bs, 64, 1
            emb2 = self.global_conv2(f[-1])
            emb2 = emb2.max(dim=-1, keepdim=True)[0]  # bs, 128, 1
            self.cls2partembed = self.cls2partembed.to(p[0].device)
            cls_one_hot = self.cls2partembed[cls_label.squeeze()].unsqueeze(-1)
            cls_one_hot = torch.cat((emb1, emb2, cls_one_hot), dim=1)
            cls_one_hot = cls_one_hot.expand(-1, -1, N)
        elif self.cls_map == 'pointnext1':
            self.cls2partembed = self.cls2partembed.to(p[0].device)
            cls_one_hot = self.cls2partembed[cls_label.squeeze()].unsqueeze(-1).expand(-1, -1, N)
            cls_one_hot = self.convc(cls_one_hot)

        for i in range(-1, -len(self.decoder), -1):
            f[i - 1] = self.decoder[i][1:](
                [p[i-1], self.decoder[i][0]([p[i - 1], f[i - 1]], [p[i], f[i]])])[1]

        # TODO: study where to add this ? 
        f[-len(self.decoder) - 1] = self.decoder[0][1:](
            [p[1], self.decoder[0][0]([p[1], torch.cat([cls_one_hot, f[1]], 1)], [p[2], f[2]])])[1]

        return f[-len(self.decoder) - 1]
