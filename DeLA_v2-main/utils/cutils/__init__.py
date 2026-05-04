from pathlib import Path
import torch
from torch.autograd import Function
from torch.utils.cpp_extension import load
from torch.nn import functional as F
from torch.amp import custom_fwd, custom_bwd  # 修改这里
import os

path = Path(__file__).parent
build_dir = path / "build"
build_dir.mkdir(exist_ok=True)
sources = [str(p) for p in path.glob("srcs/*.*") if p.suffix in [".cpp", ".cu"]]

cutils = load("cutils_", sources=sources, extra_cflags=["-O3", "-mavx2", "-funroll-loops"],
              extra_cuda_cflags=["-Xptxas", "-v"],
              verbose=True, build_directory=build_dir)


def next_prime(x) -> int:
    r"""
    Finds the next prime, x included.
    x should be >= 3 for a correct result.
    """
    x = int(x) | 1
    for i in range(x, 2 * x, 2):
        prime = True
        for j in range(3, int(i ** 0.5) + 1, 2):
            if i % j == 0:
                prime = False
                break
        if prime:
            return i


def grid_subsampling(xyz: torch.Tensor, grid_size: float, hash_size: float = 1.) -> torch.Tensor:
    r"""
    xyz: N x 3, float, non-negative coordinates
    grid_size: float, positive
    hash_size: How large the hash table should be relative to the original point cloud size.
                If estimated downsampling ratio is k, i.e., ori_size = k * subsampled_size,
                then recommended value is 2~3 / k.
                Must be greater than 1 / real_k
    return value: M, int64, selected indices
    """
    assert xyz.ndim == 2 and xyz.shape[1] == 3 and xyz.dtype == torch.float
    if xyz.stride(0) != 3:
        xyz = xyz.contiguous()
    size = xyz.shape[0] * hash_size
    size = next_prime(size + 1)
    table = torch.zeros((size,), dtype=torch.int64)
    storage = torch.empty((size * 3,), dtype=torch.int64)
    indices = cutils.grid_subsampling(xyz, grid_size, table, storage)
    return indices


def grid_subsampling_test(xyz: torch.Tensor, grid_size: float, hash_size: float = 1., pick=0) -> torch.Tensor:
    r"""
    xyz: N x 3, float, non-negative coordinates
    grid_size: float, positive
    hash_size: How large the hash table should be relative to the original point cloud size.
                If estimated downsampling ratio is k, i.e., ori_size = k * subsampled_size,
                then recommended value is 2~3 / k.
                Must be greater than 1 / real_k
    pick:  the nth point in the same grid to pick, random picked if actual resident points < pick
    return value: M, int64, selected indices
    """
    assert xyz.ndim == 2 and xyz.shape[1] == 3 and xyz.dtype == torch.float
    if xyz.stride(0) != 3:
        xyz = xyz.contiguous()
    size = xyz.shape[0] * hash_size
    size = next_prime(size + 1)
    table = torch.zeros((size,), dtype=torch.int64)
    storage = torch.empty((size * 4,), dtype=torch.int64)
    indices = cutils.grid_subsampling_test(xyz, grid_size, table, storage, pick)
    return indices


class KDTree():
    r"""
    kdt = KDTree(xyz)
    indices, squared_dists = kdt.knn(query_xyz, k=16, ordered=True)
    indices: int32
    dists: float

    Setting ordered = False (default) can be 1.1-1.2x faster.
    If there are not enough neighbors, the nearest point is used for padding.
    Resources (reference to xyz, built tree) are freed when kdt goes out of life scope.
    """

    def __init__(self, xyz: torch.Tensor, max_leaf_size=20):
        assert xyz.ndim == 2 and xyz.shape[1] == 3 and xyz.dtype == torch.float
        if xyz.stride(0) != 3:
            xyz = xyz.contiguous()
        # reserve xyz for knn search
        self.xyz = xyz
        self.n = xyz.shape[0]
        self.tree, self.pca = cutils.kdtree_build(xyz, max_leaf_size)

    def __del__(self):
        cutils.kdtree_free(self.tree, self.pca)

    def knn(self, query: torch.Tensor, k=1, ordered=False):
        assert query.ndim == 2 and query.shape[1] == 3 and query.dtype == torch.float
        if query.stride(0) != 3:
            query = query.contiguous()
        queries = query.shape[0]
        nbrs = min(self.n, k)
        if self.n < k: ordered = True
        indices = torch.empty((queries, nbrs), dtype=torch.int32)
        dists = torch.empty((queries, nbrs), dtype=torch.float)
        cutils.kdtree_knn(self.tree, query, indices, dists, ordered)
        if self.n < k:
            indices = torch.cat([indices, indices[:, :1].expand(-1, k - self.n)], dim=1)
            dists = torch.cat([dists, dists[:, :1].expand(-1, k - self.n)], dim=1)
        return indices, dists


class KEMP(Function):
    r"""
    f_i = max{f_j | j in knn_i} - f_i
    output = knn_edge_maxpooling(feature, knn, training=True)

    Only cuda version supported.

    feature: BNC, float / half
    knn:     BNk, int64
    output:  BNC, float / half

    While not training and gradient is not required,
    backward indices are not saved. Consumed time and space reduced slightly.
    """

    @staticmethod
    @custom_fwd(device_type='cuda')  # 修改这里
    def forward(ctx, feature: torch.Tensor, knn: torch.Tensor, training: bool = True) -> torch.Tensor:
        assert feature.is_cuda and knn.is_cuda
        assert feature.is_contiguous() and knn.is_contiguous() and feature.shape[:2] == knn.shape[:2]
        assert knn.dtype == torch.int64
        if feature.dtype == torch.half:
            assert feature.shape[-1] % 8 == 0, "KEMP half precision impl only supports multiples of 8 as feature dim"
        elif feature.dtype == torch.float32:
            assert feature.shape[-1] % 4 == 0, "KEMP single precision impl only supports multiples of 4 as feature dim"
        else:
            raise NotImplementedError

        output = torch.empty_like(feature)
        if training or feature.requires_grad:
            indices = torch.empty_like(feature, dtype=torch.int32)
            if feature.dtype == torch.half:
                cutils.half_aligned_knn_edge_maxpooling_forward(output, indices, feature, knn)
            else:
                cutils.aligned_knn_edge_maxpooling_forward(output, indices, feature, knn)
            ctx.save_for_backward(indices)
        else:
            if feature.dtype == torch.half:
                cutils.half_aligned_knn_edge_maxpooling_infer(output, feature, knn)
            else:
                cutils.aligned_knn_edge_maxpooling_infer(output, feature, knn)
        return output

    @staticmethod
    @custom_bwd(device_type='cuda')  # 修改这里
    def backward(ctx, grad: torch.Tensor):
        grad = grad.contiguous()
        output = -grad
        indices, = ctx.saved_tensors
        if grad.dtype == torch.half:
            cutils.half_knn_edge_maxpooling_backward(output, indices, grad)
        else:
            cutils.knn_edge_maxpooling_backward(output, indices, grad)
        return output, None, None


knn_edge_maxpooling = KEMP.apply

import numpy as np


def rand_rot():
    dim = 1
    axis = torch.randn(dim, 3)
    axis = F.normalize(axis, dim=1)
    angle = torch.rand(dim, 1, 1) * torch.pi
    A = torch.zeros(dim, 3, 3)
    A[:, 0, 1] = -axis[:, 2]
    A[:, 0, 2] = axis[:, 1]
    A[:, 1, 0] = axis[:, 2]
    A[:, 1, 2] = -axis[:, 0]
    A[:, 2, 0] = -axis[:, 1]
    A[:, 2, 1] = axis[:, 0]
    M = A * angle.sin() + torch.bmm(A, A) * (1 - angle.cos()) + torch.eye(3)
    return M.reshape(3, 3)


# gpt wrote this
# works
def fibonacci_sphere(samples=1, randomize=True):
    rnd = 1.
    if randomize:
        rnd = np.random.random() * samples

    points = []
    offset = 2. / samples
    increment = np.pi * (3. - np.sqrt(5.))

    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2)
        radius = np.sqrt(1 - pow(y, 2))

        phi = ((i + rnd) % samples) * increment

        x = np.cos(phi) * radius
        z = np.sin(phi) * radius

        points.append([x, y, z])

    return points


# gpt taught me this trick
def init_coor(dim, all_dist_=None):
    if all_dist_ is None:
        all_dist = init_coor.all_dist
    else:
        all_dist = init_coor.all_dist = all_dist_
    s = len(all_dist) // (dim + 2)
    length = all_dist[s::s][:dim]
    length = length[torch.randperm(dim)]
    coor = torch.tensor(fibonacci_sphere(dim), dtype=torch.float) @ rand_rot() * length.view(-1, 1)
    return coor


def generate_spse_matrix(dim, all_dist=None):
    coor = init_coor(dim, all_dist)
    scale = torch.empty(dim, 3).uniform_(0, 1)
    axis = torch.randn(dim, 3)
    axis = F.normalize(axis, dim=1)
    angle = torch.rand(dim, 1) * torch.pi
    M = torch.cat([coor, scale, axis, angle], dim=1)
    return M


def prepare_m(m):
    m = m.view(-1, 10)
    dim = m.shape[0]
    coor = m[:, :3]
    scale = m[:, 3:6]
    axis = m[:, 6:9]
    angle = m[:, 9].view(-1, 1, 1)
    axis = F.normalize(axis, dim=1)
    A = torch.zeros(dim, 3, 3, device=axis.device)
    A[:, 0, 1] = -axis[:, 2]
    A[:, 0, 2] = axis[:, 1]
    A[:, 1, 0] = axis[:, 2]
    A[:, 1, 2] = -axis[:, 0]
    A[:, 2, 0] = -axis[:, 1]
    A[:, 2, 1] = axis[:, 0]
    M = A * angle.sin() + torch.bmm(A, A) * (1 - angle.cos()) + torch.eye(3, device=axis.device)
    M = M @ torch.diag_embed(scale)
    return torch.cat([M.view(-1, 9), coor], dim=1)


class NSPSE4(Function):
    @staticmethod
    @custom_fwd(device_type='cuda', cast_inputs=torch.float32)  # 修改这里
    def forward(ctx, xyz: torch.Tensor, knn: torch.Tensor, weight: torch.Tensor, training: bool = True) -> torch.Tensor:
        B, N, _ = xyz.shape
        C = weight.numel() // 16
        output = torch.empty(B, N, C, device=xyz.device)
        back_idx = torch.empty(B, N, C, dtype=torch.int32, device=xyz.device)
        cutils.knn_spse_4_forward(output, back_idx, xyz, knn, weight)
        if training or weight.requires_grad:
            ctx.save_for_backward(xyz, weight, back_idx)
        return output

    @staticmethod
    @custom_bwd(device_type='cuda')  # 修改这里
    def backward(ctx, grad: torch.Tensor):
        grad = grad.contiguous()
        xyz, weight, back_idx = ctx.saved_tensors
        C = grad.shape[2]
        wgrad = torch.zeros(256, 16, C, device=xyz.device)
        cutils.knn_spse_4_backward(wgrad, grad, back_idx, xyz, weight)
        wgrad = wgrad.sum(dim=0).reshape(*weight.shape)
        return None, None, wgrad, None


knn_spse_4 = NSPSE4.apply


class NSPSE4N(Function):
    @staticmethod
    @custom_fwd(device_type='cuda', cast_inputs=torch.float32)  # 修改这里
    def forward(ctx, xyz: torch.Tensor, knn: torch.Tensor, weight: torch.Tensor, training: bool = True) -> torch.Tensor:
        B, N, _ = xyz.shape
        C = weight.numel() // 19
        output = torch.empty(B, N, C, device=xyz.device)
        back_idx = torch.empty(B, N, C, dtype=torch.int32, device=xyz.device)
        cutils.knn_spse_4n_forward(output, back_idx, xyz, knn, weight)
        if training or weight.requires_grad:
            ctx.save_for_backward(xyz, weight, back_idx)
        return output

    @staticmethod
    @custom_bwd(device_type='cuda')  # 修改这里
    def backward(ctx, grad: torch.Tensor):
        grad = grad.contiguous()
        xyz, weight, back_idx = ctx.saved_tensors
        C = grad.shape[2]
        wgrad = torch.zeros(256, 19, C, device=xyz.device)
        cutils.knn_spse_4n_backward(wgrad, grad, back_idx, xyz, weight)
        wgrad = wgrad.sum(dim=0).reshape(*weight.shape)
        return None, None, wgrad, None


knn_spse_4n = NSPSE4N.apply


class NSPSE(Function):
    @staticmethod
    @custom_fwd(device_type='cuda', cast_inputs=torch.float32)  # 修改这里
    def forward(ctx, xyz: torch.Tensor, knn: torch.Tensor, weight: torch.Tensor, training: bool = True) -> torch.Tensor:
        B, N, _ = xyz.shape
        C = weight.numel() // 12
        output = torch.empty(B, N, C, device=xyz.device)
        back_idx = torch.empty(B, N, C, dtype=torch.int32, device=xyz.device)
        cutils.knn_spse_forward(output, back_idx, xyz, knn, weight)
        if training or weight.requires_grad:
            ctx.save_for_backward(xyz, weight, back_idx)
        return output

    @staticmethod
    @custom_bwd(device_type='cuda')  # 修改这里
    def backward(ctx, grad: torch.Tensor):
        grad = grad.contiguous()
        xyz, weight, back_idx = ctx.saved_tensors
        C = grad.shape[2]
        wgrad = torch.zeros(256, 12, C, device=xyz.device)
        cutils.knn_spse_backward(wgrad, grad, back_idx, xyz, weight)
        wgrad = wgrad.sum(dim=0).reshape(*weight.shape)
        return None, None, wgrad, None


knn_spse = NSPSE.apply


class LASPSEA4(Function):
    @staticmethod
    # note  custom conversion saves memory i.e. only bf16 tensor for backward
    # @custom_fwd(device_type='cuda', cast_inputs=torch.float32)  # 如果需要取消注释，也要修改
    def forward(ctx, feature: torch.Tensor, xyz: torch.Tensor, knn: torch.Tensor, weight: torch.Tensor,
                training: bool = True) -> torch.Tensor:
        output = torch.empty_like(feature, dtype=torch.float)
        back_idx = torch.empty_like(feature, dtype=torch.int32)
        cutils.la_spse_a4_forward(output, back_idx, feature.float(), xyz.float(), knn, weight.float())
        if training or weight.requires_grad or feature.requires_grad:
            ctx.save_for_backward(back_idx, feature, xyz, weight)
        return output

    @staticmethod
    # @custom_bwd(device_type='cuda')  # 如果需要取消注释，也要修改
    def backward(ctx, grad: torch.Tensor):
        grad = grad.float().contiguous()
        back_idx, feature, xyz, weight = ctx.saved_tensors
        weight = weight.view(4, -1, 1).repeat(1, 1, 4).view(4, -1)
        f_grad = torch.zeros_like(grad)
        w_grad = torch.zeros(256, *weight.shape, device=xyz.device)
        cutils.la_spse_backward(f_grad, w_grad, grad, back_idx, feature.float(), xyz.float(), weight.float())
        w_grad = w_grad.sum(dim=0).view(4, -1, 4).sum(dim=2)
        return f_grad, None, None, w_grad, None


la_spse_a4 = LASPSEA4.apply



class CTWC(Function):
    @staticmethod
    def forward(ctx, feature: torch.Tensor, trend_seq: torch.Tensor, nbr_idx: torch.Tensor,
                training: bool = True) -> torch.Tensor:
        output = torch.empty_like(feature, dtype=torch.float)
        back_idx = torch.empty_like(feature, dtype=torch.uint32)
        weight_store = torch.empty_like(feature, dtype=torch.float32)

        # 调用CUDA前向传播
        cutils.ctwc_forward(output, back_idx, feature.float(), trend_seq.float(), nbr_idx,weight_store)

        if training or feature.requires_grad:
            ctx.save_for_backward(back_idx,weight_store)
        return output

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        grad = grad.float().contiguous()
        back_idx, weight_store = ctx.saved_tensors

        f_grad = torch.zeros_like(grad)

        # 调用CUDA反向传播
        cutils.ctwc_backward(f_grad, grad, back_idx,weight_store)

        # trend_seq和nbr_idx不需要在这个算子中求导
        return f_grad, None, None, None


# 应用函数
ctwc = CTWC.apply

class CTWC_A0(Function):
    @staticmethod
    def forward(ctx, feature: torch.Tensor, trend_seq: torch.Tensor,xyz:torch.Tensor, nbr_idx: torch.Tensor,mean,std,
                training: bool = True) -> torch.Tensor:
        B, N, C = feature.shape
        output = torch.empty_like(feature, dtype=torch.float)
        back_idx = torch.empty_like(feature, dtype=torch.uint32)
        weight_store =  torch.empty(B, N, C, 4, dtype=torch.float32, device=feature.device)

        # 调用CUDA前向传播
        cutils.ctwc_a0_forward(output, back_idx, feature.float(), trend_seq.float(), xyz.float(),nbr_idx,weight_store,mean,std)

        if training or feature.requires_grad:
            ctx.save_for_backward(back_idx,weight_store,feature)
        return output

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        grad = grad.float().contiguous()
        back_idx, weight_store,feature = ctx.saved_tensors
        B, N, C = feature.shape
        f_grad = torch.zeros_like(grad)
        #print("grad",grad.shape)
        # 调用CUDA反向传播
        cutils.ctwc_a0_backward( grad,back_idx,feature,weight_store,f_grad)

        # trend_seq和nbr_idx不需要在这个算子中求导
        return f_grad, None, None, None, None, None,None


# 应用函数
ctwc_a0 = CTWC_A0.apply



class CTWC_A1(Function):
    @staticmethod
    def forward(ctx, feature: torch.Tensor, trend_seq: torch.Tensor,xyz:torch.Tensor, nbr_idx: torch.Tensor,weight,mean,std,
                training: bool = True) -> torch.Tensor:
        B, N, C = feature.shape
        output = torch.empty_like(feature, dtype=torch.float)
        back_idx = torch.empty_like(feature, dtype=torch.uint32)
        weight_store =  torch.empty(B, N, C, 4, dtype=torch.float32, device=feature.device)
        #print("weight_store",weight_store)
        # 调用CUDA前向传播
        cutils.ctwc_a1_forward(output, back_idx, feature.float(), trend_seq.float(), xyz.float(),nbr_idx,weight,weight_store,mean,std)

        if training or feature.requires_grad:
            ctx.save_for_backward(back_idx,weight_store,feature)
        return output

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        grad = grad.float().contiguous()
        back_idx, weight_store,feature = ctx.saved_tensors
        B, N, C = feature.shape
        f_grad = torch.zeros_like(grad)
        w_grad =  torch.zeros(C//4, 4, dtype=torch.float32, device=feature.device)
        #print("grad",grad.shape)
        # 调用CUDA反向传播
        cutils.ctwc_a1_backward( grad,back_idx,feature,weight_store,f_grad, w_grad)

        # trend_seq和nbr_idx不需要在这个算子中求导
        return f_grad, None, None, None, w_grad, None, None,None


# 应用函数
ctwc_a1 = CTWC_A1.apply


class CTWC_A2(Function):
    @staticmethod
    def forward(ctx, feature: torch.Tensor, trend_seq: torch.Tensor, nbr_idx: torch.Tensor,weight,mean,std,scale,
                training: bool = True) -> torch.Tensor:
        B, N, C = feature.shape
        output = torch.empty_like(feature, dtype=torch.float)
        back_idx = torch.empty_like(feature, dtype=torch.uint32)
        weight_store =  torch.empty(B, N, C, 4, dtype=torch.float32, device=feature.device)

        # 调用CUDA前向传播
        cutils.ctwc_a2_forward(output, back_idx, feature.float(), trend_seq.float(), nbr_idx,weight,weight_store,mean,std,scale)
        # d1 = torch.mean(weight_store[:,:,:,0])
        # d2 = torch.mean(weight_store[:,:,:,1])
        # d3 = torch.mean(weight_store[:,:,:,2])
        # print(f"d1:{d1},d2:{d2},d3{d3}")
        # print("weight_store:",weight_store[0][0][0:3])
        #print(f"weight_store:{weight_store.shape,weight_store[0][0:5][0:10]}")
        if training or feature.requires_grad:
            ctx.save_for_backward(back_idx,weight_store,feature)
        return output

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        grad = grad.float().contiguous()
        back_idx, weight_store,feature = ctx.saved_tensors
        B, N, C = feature.shape
        f_grad = torch.zeros_like(grad)
        w_grad =  torch.zeros(C//4, 4, dtype=torch.float32, device=feature.device)
        # 调用CUDA反向传播
        cutils.ctwc_a2_backward( grad,back_idx,feature,weight_store,f_grad, w_grad)

        # trend_seq和nbr_idx不需要在这个算子中求导
        return f_grad, None, None, w_grad, None, None,None,None


# 应用函数
ctwc_a2 = CTWC_A2.apply


class CTWC_A2_1(Function):
    @staticmethod
    def forward(ctx, feature: torch.Tensor, trend_seq: torch.Tensor, nbr_idx: torch.Tensor,weight,mean,std,scale,
                training: bool = True) -> torch.Tensor:
        B, N, C = feature.shape
        output = torch.empty_like(feature, dtype=torch.float)
        back_idx = torch.empty_like(feature, dtype=torch.uint32)
        weight_store =  torch.empty(B, N, C, 4, dtype=torch.float32, device=feature.device)

        # 调用CUDA前向传播
        cutils.ctwc_a2_1_forward(output, back_idx, feature.float(), trend_seq.float(), nbr_idx,weight,weight_store,mean,std,scale)
        # batch_indices = [0, 0, 0,0,0]  # 查看3个位置，都在batch 0
        # point_indices = [0, 100, 200,300,400]  # 点索引
        # channel_indices = [[0,1,2,3, 24], [0,1,2,3, 24],[0,1,2,3, 24],[0,1,2,3, 24]] # 通道索引
        #print(f"weight_store:{weight_store.shape},\n,{ weight_store[batch_indices, point_indices, channel_indices]}")
        if training or feature.requires_grad:
            ctx.save_for_backward(back_idx,weight_store,feature)
        return output

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        grad = grad.float().contiguous()
        back_idx, weight_store,feature = ctx.saved_tensors
        B, N, C = feature.shape
        f_grad = torch.zeros_like(grad)

        w_grad =  torch.zeros(C//4, 4, dtype=torch.float32, device=feature.device)
        # 调用CUDA反向传播
        cutils.ctwc_a2_backward( grad,back_idx,feature,weight_store,f_grad, w_grad)

        # trend_seq和nbr_idx不需要在这个算子中求导
        return f_grad, None, None, w_grad, None, None,None,None


# 应用函数
ctwc_a2_1 = CTWC_A2_1.apply

class CTWC_A3(Function):
    @staticmethod
    def forward(ctx, feature: torch.Tensor, trend_seq: torch.Tensor, nbr_idx: torch.Tensor,weight,mean,std,scale,gate,
                training: bool = True) -> torch.Tensor:
        B, N, C = feature.shape
        output = torch.empty_like(feature, dtype=torch.float)
        back_idx = torch.empty_like(feature, dtype=torch.uint32)
        weight_store =  torch.empty(B, N, C, 4, dtype=torch.float32, device=feature.device)

        # 调用CUDA前向传播
        cutils.ctwc_a3_forward(output, back_idx, feature.float(), trend_seq.float(), nbr_idx,weight,weight_store,mean,std,scale,gate)

        if training or feature.requires_grad:
            ctx.save_for_backward(back_idx,weight,gate,weight_store,feature)
        return output

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        grad = grad.float().contiguous()
        back_idx,weight,gate,weight_store,feature = ctx.saved_tensors
        B, N, C = feature.shape
        f_grad = torch.zeros_like(grad)
        w_grad =  torch.zeros(C//4, 4, dtype=torch.float32, device=feature.device)
        gate_grad = torch.zeros_like(gate)
        # 调用CUDA反向传播
        cutils.ctwc_a3_backward( grad,back_idx,feature,weight,weight_store,gate,f_grad, w_grad,gate_grad)

        # trend_seq和nbr_idx不需要在这个算子中求导
        return f_grad, None, None, w_grad, None, None,None,gate_grad,None


# 应用函数
ctwc_a3 = CTWC_A3.apply


class CTWC_A4(Function):
    @staticmethod
    def forward(ctx, feature: torch.Tensor, trend_seq: torch.Tensor,xyz:torch.Tensor, nbr_idx: torch.Tensor,weight,mean,std,
                training: bool = True) -> torch.Tensor:
        B, N, C = feature.shape
        output = torch.empty_like(feature, dtype=torch.float)
        back_idx = torch.empty_like(feature, dtype=torch.uint32)
        weight_store =  torch.empty(B, N, C, 4, dtype=torch.float32, device=feature.device)
        #print("weight_store",weight_store)
        # 调用CUDA前向传播
        cutils.ctwc_a4_forward(output, back_idx, feature.float(), trend_seq.float(), xyz.float(),nbr_idx,weight,weight_store,mean,std)

        if training or feature.requires_grad:
            ctx.save_for_backward(back_idx,weight,weight_store,feature)
        return output

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        grad = grad.float().contiguous()
        back_idx, weight,weight_store,feature = ctx.saved_tensors
        B, N, C = feature.shape
        f_grad = torch.zeros_like(grad)
        w_grad =  torch.zeros(C//4, 4, dtype=torch.float32, device=feature.device)

        cutils.ctwc_a4_backward( grad,back_idx,feature.float(),weight.float(),weight_store,f_grad.float(), w_grad.float())
        #print("w_grad",w_grad.shape,w_grad)
        # trend_seq和nbr_idx不需要在这个算子中求导
        return f_grad, None, None, None, w_grad, None, None,None

ctwc_a4 = CTWC_A4.apply


class CTWC_A4_1(Function):
    @staticmethod
    def forward(ctx, feature: torch.Tensor, trend_seq: torch.Tensor,xyz:torch.Tensor, nbr_idx: torch.Tensor,weight,mean,std,
                training: bool = True) -> torch.Tensor:
        B, N, C = feature.shape
        output = torch.empty_like(feature, dtype=torch.float)
        back_idx = torch.empty_like(feature, dtype=torch.uint32)
        weight_store =  torch.zeros(B, N, C, 4, dtype=torch.float32, device=feature.device)
        #print("weight_store",weight_store)
        # 调用CUDA前向传播
        cutils.ctwc_a4_1_forward(output, back_idx, feature.float(), trend_seq.float(), xyz.float(),nbr_idx,weight,weight_store,mean,std)

        if training or feature.requires_grad:
            ctx.save_for_backward(back_idx,weight,weight_store,feature)
        return output

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        grad = grad.float().contiguous()
        back_idx, weight,weight_store,feature = ctx.saved_tensors
        B, N, C = feature.shape
        f_grad = torch.zeros_like(grad)
        w_grad =  torch.zeros(C//4, 4, dtype=torch.float32, device=feature.device)

        cutils.ctwc_a4_1_backward( grad,back_idx,feature.float(),weight.float(),weight_store,f_grad.float(), w_grad.float())
        #print("w_grad",w_grad.shape,w_grad)
        # trend_seq和nbr_idx不需要在这个算子中求导
        return f_grad, None, None, None, w_grad, None, None,None

ctwc_a4_1 = CTWC_A4_1.apply

class CTWC_B1(Function):
    @staticmethod
    def forward(ctx, trend_seq: torch.Tensor, nbr_idx: torch.Tensor,scale,
                training: bool = True) -> torch.Tensor:
        B, N, K = nbr_idx.shape
        C = trend_seq.shape[-1]
        weight_store = torch.zeros((B, N, K,C // 4), dtype=torch.float32, device=trend_seq.device)
        with torch.no_grad():
            # 调用CUDA前向传播
            cutils.ctwc_b1_forward( trend_seq.float(), nbr_idx,weight_store,scale)
            return weight_store
    @staticmethod
    def backward(ctx, grad_output):
        # 返回与forward输入相同数量的None
        return None, None, None, None

# 应用函数
ctwc_b1 = CTWC_B1.apply

class CTWC_B11(Function):
    @staticmethod
    def forward(ctx, trend_seq: torch.Tensor, nbr_idx: torch.Tensor,scale,
                training: bool = True) -> torch.Tensor:
        B, N, K = nbr_idx.shape
        C_ = trend_seq.shape[-1]
        weight_store = torch.zeros((B, N, K,C_), dtype=torch.float32, device=trend_seq.device)
        with torch.no_grad():
            # 调用CUDA前向传播
            cutils.ctwc_b11_forward( trend_seq.float(), nbr_idx,weight_store,scale)
            return weight_store
    @staticmethod
    def backward(ctx, grad_output):
        # 返回与forward输入相同数量的None
        return None, None, None, None

# 应用函数
ctwc_b11 = CTWC_B11.apply


class CTWC_B2(Function):
    @staticmethod
    def forward(ctx,nbr_idx: torch.Tensor,mean,std,scale,
                training: bool = True) -> torch.Tensor:
        B, N, K = nbr_idx.shape

        weight_store = torch.zeros((B, N, K), dtype=torch.float32, device=nbr_idx.device)
        with torch.no_grad():
            # 调用CUDA前向传播
            cutils.ctwc_b2_forward( nbr_idx,weight_store,mean,std,scale)
            return weight_store
    @staticmethod
    def backward(ctx, grad_output):
        # 返回与forward输入相同数量的None
        return None, None, None, None, None

# 应用函数
ctwc_b2 = CTWC_B2.apply

class CTWC_B3(Function):
    @staticmethod
    def forward(ctx,nbr_idx: torch.Tensor,trend_seq,mean,std,scale,
                training: bool = True) -> torch.Tensor:
        B, N, K = nbr_idx.shape
        B,N,T,C = trend_seq.shape

        weight_store = torch.zeros((B, N, K,C//4), dtype=torch.float32, device=nbr_idx.device)
        with torch.no_grad():
            # 调用CUDA前向传播
            cutils.ctwc_b3_forward( weight_store,trend_seq.float(),nbr_idx,mean,std,scale)
            return weight_store
    @staticmethod
    def backward(ctx, grad_output):
        # 返回与forward输入相同数量的None
        return None, None, None, None, None

# 应用函数
ctwc_b3 = CTWC_B3.apply


class CTWC_wc(Function):
    @staticmethod
    def forward(ctx, feature: torch.Tensor, nbr_idx: torch.Tensor,weight, training: bool = True) -> torch.Tensor:
        B, N, K,C_ = weight.shape
        output = torch.empty_like(feature, dtype=torch.float)
        back_idx = torch.empty_like(feature, dtype=torch.uint32).contiguous()
        weight_back_idx =  torch.zeros((B, N, C_*4), dtype=torch.uint32, device=feature.device).contiguous()

        # 调用CUDA前向传播
        cutils.ctwc_wc_forward(output, back_idx,weight_back_idx, feature.float(), nbr_idx,weight.float())
        if training or feature.requires_grad:
            ctx.save_for_backward(back_idx,weight_back_idx,weight,feature)
        return output

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        grad = grad.float().contiguous()

        back_idx,weight_back_idx,weight,feature = ctx.saved_tensors
        B, N, K,C_ = weight.shape
        f_grad = torch.zeros_like(grad)
        w_grad =  torch.zeros((B,N,K,C_), dtype=torch.float32, device=feature.device)

        cutils.ctwc_wc_backward( grad,back_idx,weight_back_idx,feature.float(),weight.float(),f_grad.float(), w_grad.float())

        return f_grad, None, w_grad

ctwc_wc = CTWC_wc.apply


class CTWC_wc1(Function):
    @staticmethod
    def forward(ctx, feature: torch.Tensor, nbr_idx: torch.Tensor,weight, training: bool = True) -> torch.Tensor:
        B ,N,K , C_= weight.shape
        output = torch.zeros((B, N, K, C_*4), dtype=torch.float32, device=feature.device)

        # 调用CUDA前向传播
        cutils.ctwc_wc1_forward(output, feature.float(), nbr_idx,weight.float())
        if training or feature.requires_grad:
            ctx.save_for_backward(nbr_idx,weight,feature)
        return output

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        grad = grad.float().contiguous()
        #print("grad_output", grad.shape)
        nbr_idx,weight,feature = ctx.saved_tensors
        B,N,K, C = grad.shape
        f_grad = torch.zeros_like(feature, dtype=torch.float32, device=feature.device)
        w_grad =  torch.zeros((B,N,K,C//4), dtype=torch.float32, device=feature.device)

        cutils.ctwc_wc1_backward( grad,f_grad,w_grad,feature.float(),nbr_idx,weight.float())

        return f_grad, None, w_grad

ctwc_wc1 = CTWC_wc1.apply
