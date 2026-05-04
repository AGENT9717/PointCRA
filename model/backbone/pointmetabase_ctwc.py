"""
PointMetaBase
"""
from re import I
from typing import List, Type
import logging
import torch
import torch.nn as nn
from ..build import MODELS
from ..layers import create_convblock1d, create_convblock2d, create_act, CHANNEL_MAP, \
    create_grouper, furthest_point_sample, random_sample, three_interpolation
from ..lib import ctwc_b11
from timm.models.layers import DropPath
import torch.nn.functional as F
import copy
class FeatureRecorder(nn.Module):
    """轻量级特征记录器，使用环形缓冲区"""
    def __init__(self, max_depth=20, feature_dim=256):
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
        channels = [in_dim,hid_dim,in_dim]

        self.pwconv = nn.Sequential(nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1,bias=True),
                                    nn.ReLU(inplace=True),
                                    #nn.BatchNorm1d(in_dim),
                                    #nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(in_dim),)

        weight_dim = in_dim //4
        self.drop_path_rates = {
            128: 0.15,  # 较小维度，较小drop率
            256: 0.20,  # 中等维度
            512: 0.25,  # 较大维度
            1024: 0.30,  # 最大维度，最大drop率
        }

        # 根据输入维度动态设置drop率
        drop_rate = drop_path_rate if drop_path_rate is not None else self.drop_path_rates.get(in_dim, 0.2)
        self.gate = nn.Parameter(torch.tensor(0.1))
        self.pool = lambda x: torch.mean(x, dim=-2, keepdim=False)
        self.a = nn.Parameter(torch.ones(weight_dim))  # 控制斜率
        self.b = nn.Parameter(torch.zeros(weight_dim))  # 控制中心点
        self.c = nn.Parameter(torch.ones(weight_dim))  # 控制幅度


    def forward(self, pfpe):
        p,x,pe=pfpe
        seq = self.seq.get_recorded_features() #b,c_,n,t
        b, c_, n, t = seq.shape
        seq = seq.permute(0, 2, 3, 1).contiguous()  # b,n,t,c

        res = x
        B,C,N = x.shape

        _, fj, knn_idx = self.grouper(p, p, x)       #bnk    dp,fj
        fj =fj #+ pe
        ################ctwc_b
        scale = torch.tensor(3.0, dtype=torch.float, device=x.device)
        c_dist = ctwc_b11(seq,knn_idx,scale)   #bnkc_        seq:bntc
        mask = (c_dist == 1.0).all(dim=-1)
        mask_expanded = mask.unsqueeze(-1)  # 形状: [B, N, K, 1]
        mask_expanded = mask_expanded.expand_as(c_dist)  # 形状: [B, N, K, C]
        c_dist[mask_expanded] = 0.5

        pg = c_dist.mean(dim=-1)    #bnk
        pg_var = torch.var(pg,dim=-1,keepdim=True)
        pg_max = pg.max(dim=2, keepdim=True)[0]  # [B, N, 1]
        pg_min = pg.min(dim=2, keepdim=True)[0]  # [B, N, 1]
        pg_range = pg_max - pg_min
        narrow_range_mask = pg_range < 0.01
        max_pg_var = (pg_range**2)/4.0
        max_pg_var = torch.clamp(max_pg_var,1e-8)
        pn_raw = 1.0-torch.exp(-pg_var/max_pg_var)
        pn_raw = torch.clamp(pn_raw,min=1e-8,max=1.0)
        pn = torch.where(narrow_range_mask,
                             torch.ones_like(pn_raw),
                             pn_raw) #bn1



        gamma = torch.exp(3*(0.7-pn))
        pg_ad = torch.pow(pg,gamma)


        la = self.a.view(*([1] * (c_dist.dim() - 1)), -1)
        lb = self.b.view(*([1] * (c_dist.dim() - 1)), -1)
        lc = torch.sigmoid(self.c).view(*([1] * (c_dist.dim() - 1)), -1)  # c∈[0,1]
        pc = lc * (torch.sigmoid(la * (c_dist - lb)) - torch.sigmoid(-la * lb))
        w = pg_ad.unsqueeze(-1) *pc


        fj = fj.permute(0,2,3,1).contiguous()#-x.unsqueeze(-1)

        b,n,k,c = fj.shape
        fj = fj.view(b,n,k,-1,4)
        fj = fj * w.unsqueeze(-1)
        x = self.pool(fj.view(b,n,k,c)) #bnkc->bnc
        del fj,w,c_dist

        x = x.permute(0,2,1)
        x = self.pwconv(x)
        x = x+res

        x = x.contiguous()

        c_loss = None

        if self.training:
            c_loss = 0.0
            # c = torch.sigmoid(c)  # c在[0,1]
            losses = []
            B, N, K, C = w.shape

            w_flat = w.view(B * N * K, C)
            w_norm = F.normalize(w_flat, dim=0)
            cos_sim = torch.mm(w_norm.T, w_norm)
            mask = ~torch.eye(C, dtype=torch.bool, device=w.device)
            off_diag = cos_sim[mask]
            losses.append( off_diag.abs().mean())
            losses.append(F.relu(lb).mean())

            losses.append(F.relu(0.2 - la).mean())
            losses.append(F.relu(la - 10).mean())

            losses.append(F.relu(0.2 - lc).mean())
            losses.append(F.relu(lc - 0.8).mean())

            weights = [0.01,0.5, 0.1, 0.1, 0.15, 0.15]
            c_loss = sum(w * l for w, l in zip(weights, losses))

        return p, x,pe,c_loss


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


def get_aggregation_feautres(p, dp, f, fj, feature_type='dp_fj'):
    if feature_type == 'dp_fj':
        fj = torch.cat([dp, fj], 1)
    elif feature_type == 'dp_fj_df':
        df = fj - f.unsqueeze(-1)
        fj = torch.cat([dp, fj, df], 1)
    elif feature_type == 'pi_dp_fj_df':
        df = fj - f.unsqueeze(-1)
        fj = torch.cat([p.transpose(1, 2).unsqueeze(-1).expand(-1, -1, -1, df.shape[-1]), dp, fj, df], 1)
    elif feature_type == 'dp_df':
        df = fj - f.unsqueeze(-1)
        fj = torch.cat([dp, df], 1)
    return fj


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
        channels1 = channels 
        convs1 = []
        for i in range(len(channels1) - 1):  # #layers in each blocks
            convs1.append(create_convblock1d(channels1[i], channels1[i + 1],
                                             norm_args=norm_args,
                                            act_args=None if i == (
                                                    len(channels1) - 2) and not last_act else act_args,
                                             **conv_args)
                          )
        self.convs1 = nn.Sequential(*convs1)
        self.grouper = create_grouper(group_args)
        self.reduction = reduction.lower()
        self.pool = get_reduction_fn(self.reduction)
        self.feature_type = feature_type

    def forward(self, pf, pe) -> torch.Tensor:
        # p: position, f: feature
        p, f = pf
        # preconv
        f = self.convs1(f)
        # grouping
        dp, fj = self.grouper(p, p, f)
        # pe + fj 
        f = fj+pe#-f.unsqueeze(-1)
        f = self.pool(f)
        """ DEBUG neighbor numbers. 
        if f.shape[-1] != 1:
            query_xyz, support_xyz = p, p
            radius = self.grouper.radius
            dist = torch.cdist(query_xyz.cpu(), support_xyz.cpu())
            points = len(dist[dist < radius]) / (dist.shape[0] * dist.shape[1])
            logging.info(
                f'query size: {query_xyz.shape}, support size: {support_xyz.shape}, radius: {radius}, num_neighbors: {points}')
        DEBUG end """
        return f


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
        channels[0] = in_channels #if is_head else CHANNEL_MAP[feature_type](channels[0])
        channels1 = channels
        # channels2 = copy.copy(channels)
        channels2 = [in_channels] + [32,32] * (min(layers, 2) - 1) + [out_channels] # 16
        channels2[0] = 3
        convs1 = []
        convs2 = []

        if self.use_res:
            self.skipconv = create_convblock1d(
                in_channels, channels[-1], norm_args=None, act_args=None) if in_channels != channels[
                -1] else nn.Identity()
            self.act = create_act(act_args)

        # actually, one can use local aggregation layer to replace the following
        for i in range(len(channels1) - 1):  # #layers in each blocks
            convs1.append(create_convblock1d(channels1[i], channels1[i + 1],
                                             norm_args=norm_args if not is_head else None,
                                             act_args=None if i == len(channels) - 2
                                                            and (self.use_res or is_head) else act_args,
                                             **conv_args)
                          )
        self.convs1 = nn.Sequential(*convs1)

        if not is_head:
            for i in range(len(channels2) - 1):  # #layers in each blocks
                convs2.append(create_convblock2d(channels2[i], channels2[i + 1],
                                                 norm_args=norm_args if not is_head else None,
                                                #  act_args=None if i == len(channels) - 2
                                                #                 and (self.use_res or is_head) else act_args,
                                                 act_args=act_args,
                                                **conv_args)
                            )
            self.convs2 = nn.Sequential(*convs2)
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

    def forward(self, pf_pe):
        p, f, pe = pf_pe
        if self.is_head:
            f = self.convs1(f)  # (n, c)
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
                fi = torch.gather(
                    f, -1, idx.unsqueeze(1).expand(-1, f.shape[1], -1))
                if self.use_res:
                    identity = self.skipconv(fi)
            else:
                fi = None
            # preconv
            f = self.convs1(f)
            # grouping
            #print("f.shape", f.shape)
            dp, fj= self.grouper(new_p, p, f)
            # conv on neighborhood_dp
            pe = self.convs2(dp)
            # pe + fj
            #print("f,fj",f.shape,fj.shape)
            f = pe + fj#-f.unsqueeze(-1)
            f = self.pool(f)
            if self.use_res:
                f = self.act(f + identity)
            p = new_p
        return p, f, pe


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
            p1, f1 = pf1
            p2, f2 = pf2
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
                 num_posconvs=2,#2,
                 less_act=False,
                 seq=None,
                 **kwargs
                 ):
        super().__init__()
        self.use_res = use_res
        mid_channels = int(in_channels * expansion)
        self.convs = LocalAggregation([in_channels, in_channels],
                                      norm_args=norm_args, act_args=act_args ,#if num_posconvs > 0 else None,
                                      group_args=group_args, conv_args=conv_args,
                                      **aggr_args, **kwargs)
        if num_posconvs < 1:
            channels = []
        elif num_posconvs == 1:
            channels = [in_channels, in_channels]
        elif num_posconvs == 4:
            channels = [in_channels, in_channels, in_channels, in_channels, in_channels]
        elif num_posconvs == 3:
            channels = [in_channels, in_channels, in_channels, in_channels]
        else:
            channels = [in_channels, mid_channels, in_channels]
        pwconv = []
        # point wise after depth wise conv (without last layer)
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

    def forward(self, pf_pe):
        p, f, pe = pf_pe
        identity = f
        f = self.convs([p, f], pe)
        seq_xi = torch.mean(f.view(f.shape[0], -1, 4, f.shape[-1]), dim=-2, keepdim=False)
        self.seq.record(seq_xi)
        f = self.pwconv(f)
        seq_xi = torch.mean(f.view(f.shape[0], -1, 4, f.shape[-1]), dim=-2, keepdim=False)
        self.seq.record(seq_xi)
        if f.shape[-1] == identity.shape[-1] and self.use_res:
            f = self.drop_path(identity)+f
        f = self.act(f)
        return [p, f, pe]




@MODELS.register_module()
class PointMetaBase_CTWC_Encoder(nn.Module):
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
        pe_encoder = nn.ModuleList() #[]
        pe_grouper = []
        for i in range(len(blocks)):
            group_args.radius = self.radii[i]
            group_args.nsample = self.nsample[i]
            encoder.append(self._make_enc(
                block, channels[i], blocks[i], stride=strides[i], group_args=group_args,
                is_head=i == 0 and strides[i] == 1
            ))
            if i == 0:
                pe_encoder.append(nn.ModuleList())
                pe_grouper.append([])
            else:
                pe_encoder.append(self._make_pe_enc(
                    block, channels[i], blocks[i], stride=strides[i], group_args=group_args,
                    is_head=i == 0 and strides[i] == 1
                ))
                pe_grouper.append(create_grouper(group_args))
        self.encoder = nn.Sequential(*encoder)
        self.pe_encoder = pe_encoder #nn.Sequential(*pe_encoder)
        self.pe_grouper = pe_grouper
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

    def _make_pe_enc(self, block, channels, blocks, stride, group_args, is_head=False):
        ## for PE of this stage
        channels2 = [3, channels]
        convs2 = []
        if blocks > 1:
            for i in range(len(channels2) - 1):  # #layers in each blocks
                convs2.append(create_convblock2d(channels2[i], channels2[i + 1],
                                                norm_args=self.norm_args,
                                                act_args=self.act_args,
                                                **self.conv_args)
                            )
            convs2 = nn.Sequential(*convs2)
            return convs2
        else:
            return nn.ModuleList()

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
        self.seq = FeatureRecorder(max_depth=(blocks - 1) * 2, feature_dim=channels // 4)
        for i in range(1, blocks):
            group_args.radius = radii[i]
            group_args.nsample = nsample[i]
            layers.append(block(self.in_channels,
                                aggr_args=self.aggr_args,
                                norm_args=self.norm_args, act_args=self.act_args, group_args=group_args,
                                conv_args=self.conv_args, expansion=self.expansion,
                                use_res=self.use_res,
                                seq=self.seq
                                ))
        if blocks > 1:
            layers.append(CTWC_Block(in_dim=self.in_channels, act=nn.GELU, mlp_ratio=1.0,
                                     bn_momentum=0.02, depth=(blocks - 1) * 2, K=nsample[-1], init=0.,
                                     group_args=group_args, seq=self.seq,
                                     norm_args=self.norm_args, act_args=self.act_args, conv_args=self.conv_args
                                     ))
        return nn.Sequential(*layers)

    def forward_cls_feat(self, p0, f0=None):
        if hasattr(p0, 'keys'):
            p0, f0 = p0['pos'], p0.get('x', None)
        if f0 is None:
            f0 = p0.clone().transpose(1, 2).contiguous()
        for i in range(0, len(self.encoder)):
            pe = None
            p0, f0, pe = self.encoder[i]([p0, f0, pe])
        return f0.squeeze(-1)

    def forward_seg_feat(self, p0, f0=None):
        if hasattr(p0, 'keys'):
            p0, f0 = p0['pos'], p0.get('x', None)
        if f0 is None:
            f0 = p0.clone().transpose(1, 2).contiguous()
        p, f = [p0], [f0]
        def reset_recorders(module):
            for child in module.children():
                if isinstance(child, FeatureRecorder):
                    child.reset()
                else:
                    reset_recorders(child)

        reset_recorders(self)

        c_loss=0.0
        _c_l=0
        for i in range(0, len(self.encoder)):

            if i ==0:
                pe = None

                _p, _f, _ = self.encoder[i]([p[-1], f[-1], pe])
            else:
                _p, _f, _= self.encoder[i][0]([p[-1], f[-1], pe])
                if self.blocks[i] > 1:
                    # grouping
                    dp, _,_ = self.pe_grouper[i](_p, _p, None)
                    # conv on neighborhood_dp
                    pe = self.pe_encoder[i](dp)
                    if self.training:
                        _p, _f, _ ,_c= self.encoder[i][1:]([_p, _f, pe])
                        _c_l = _c + _c_l
                    else:
                        _p, _f, _,_c= self.encoder[i][1:]([_p, _f, pe])


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

