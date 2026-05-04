
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <cstdint>  // 添加这个头文件
#include <torch/extension.h>
#include "cuda_util.h"

constexpr uint64_t group_size = 4;

struct __builtin_align__(group_size*4) floatn
{
    float v[group_size];
};


template <uint64_t block_size>
__global__ void __launch_bounds__(block_size) ctwc_wc1_backward_kernel(
    const float *grad_output,    // B N K C  输出梯度
    float *grad_input,           // B N C    输入梯度（形状与前向input相同）
    float *grad_weight,          // B N K C_ 权重梯度
    const float *input,          // B N C    输入特征
    const uint32_t *nbr_idx,     // B N k    邻居索引
    const float *weight,         // B N K C_ 权重
    const uint64_t N,
    const uint64_t C_,
    const uint64_t k,
    const uint64_t BNC_
) {
    uint64_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= BNC_) return;
    
    const uint64_t c_ = idx % C_,
                   c = c_ * 4,
                   C = C_ * 4,
                   bNn = idx / C_,
                   n = bNn % N,
                   b = bNn / N,
                   input_base = b * N * C,
                   nbr_base = b * N * k + n * k;

    // 读取中心点特征
    const floatn cf = *(floatn*)(input + input_base + n * C + c);

    // 为当前中心点的4个通道累加梯度
    floatn grad_center = {0, 0, 0, 0};
    
    // 遍历所有邻居
    for (uint64_t i = 0; i < k; ++i) {
        const uint64_t nidx = nbr_idx[nbr_base + i];
        if (nidx >= N) {
            continue;
        }
        
        const float l_weight = weight[b*N*k*C_+n*k*C_+i*C_+c_];
        
        // 读取邻居点特征
        const floatn nf = *(floatn*)(input + input_base + nidx * C + c);
        
        // 对当前通道组的4个通道分别处理
        for (uint64_t f_idx = 0; f_idx < group_size; ++f_idx) {
            const uint64_t output_idx = b*N*k*C+n*k*C+i*C+c+f_idx;
            const float grad_out = grad_output[output_idx];
            
            // 权重梯度：dL/dweight = dL/doutput * (input_nbr - input_center)
            const float weight_grad = grad_out * (nf.v[f_idx] + cf.v[f_idx]);
            atomicAdd(&grad_weight[b * N * k * C_  + n * k * C_ + i*C_ + c_], weight_grad);
            
            // 输入特征梯度：
            // 对中心点：dL/dinput_center = dL/doutput * (-weight)
            grad_center.v[f_idx] += grad_out * (l_weight);
            
            // 对邻居点：dL/dinput_nbr = dL/doutput * weight
            atomicAdd(&grad_input[input_base + nidx * C + c + f_idx], grad_out * l_weight);
        }
    }
    
    // 累加中心点的梯度
    for (uint64_t f_idx = 0; f_idx < group_size; ++f_idx) {
        atomicAdd(&grad_input[input_base + n * C + c + f_idx], grad_center.v[f_idx]);
    }
}

void ctwc_wc1_backward(
    torch::Tensor &grad_output,
    torch::Tensor &grad_input,
    torch::Tensor &grad_weight,
    const torch::Tensor &input,
    const torch::Tensor &nbr_idx,
    const torch::Tensor &weight
) {
    const uint64_t B = grad_output.size(0),
                   N = grad_output.size(1),
                   k = grad_output.size(2),
                   C = grad_output.size(3),
                   C_ = C / 4;
    
    constexpr uint64_t block_size = 512;
    
    // 初始化梯度为0
    grad_input.zero_();
    grad_weight.zero_();
    
    ctwc_wc1_backward_kernel<block_size>
        <<<(B * N * C_ + block_size - 1) / block_size, block_size>>>(
        (const float*)grad_output.data_ptr(),
        (float*)grad_input.data_ptr(),
        (float*)grad_weight.data_ptr(),
        (const float*)input.data_ptr(),
        (const uint32_t*)nbr_idx.data_ptr(),
        (const float*)weight.data_ptr(),
        N, C_, k, B * N * C_
    );
    
    checkCudaError();
}