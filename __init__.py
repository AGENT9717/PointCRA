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
#sources = [str(p) for p in path.glob("srcs/*.*") if p.suffix in [".cpp", ".cu"]]

sources = []
sources.extend(path.glob("*.cpp"))
sources.extend(path.glob("*.cu"))
sources = [str(p) for p in sources]

if not sources:
    raise RuntimeError("No .cpp or .cu source files found for CUDA extension!")

# 添加头文件路径
extra_include_paths = [
    str(path),  # 当前目录
    str(path.parent / "include"),  # 可能存在的include目录
]

cutils = load(
    "cutils_",
    sources=sources,
    extra_cflags=["-O3", "-mavx2", "-funroll-loops"],
    extra_cuda_cflags=["-Xptxas", "-v"],
    verbose=True,
    build_directory=build_dir,
    # 添加头文件包含路径
    extra_include_paths=extra_include_paths
)



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
