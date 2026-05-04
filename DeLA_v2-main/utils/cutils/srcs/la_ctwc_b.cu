#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <math.h>
#include <assert.h>
#include <torch/extension.h>
#include "cuda_util.h"


constexpr int group_size = 4;

template <int block_size>
__global__ void ctwc_backward_kernel(
    float *grad_input,           // B N C      只需要输入特征的梯度
    const float *grad_output,    // B N C      输出梯度
    const uint32_t *back_idx,    // B N C      前向传播记录的索引
    const float *weight_store,    // B N C     存储的权重系数
    const int N, const int C_, const int BNC_
){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= BNC_) return;

    const int c_ = idx % C_;
    const int c = c_ * 4;
    const int C = C_ * 4;
    const int bNn = idx / C_;
    const int n = bNn % N;
    const int b = bNn / N;
    const int f_base = b * N * C + c;

    // 读取输出梯度
    float4 grad_out = *(float4*)(grad_output + f_base + n * C);
    float grad_values[4] = {grad_out.x, grad_out.y, grad_out.z, grad_out.w};
    float4 stored_weights = *(float4*)(weight_store + f_base + n * C);
    float weight_values[4] = {stored_weights.x, stored_weights.y, stored_weights.z, stored_weights.w};
    // 处理4个通道
    for (int f_idx = 0; f_idx < 4; ++f_idx) {
        // 找到前向传播选中的邻居索引
        const int max_nidx = back_idx[f_base + n * C + f_idx];

        if (max_nidx < N) { // 有效的邻居索引
            // 梯度传播：只有被选中的邻居和中心点需要梯度
            float grad_value = grad_values[f_idx];

            float weighted_grad = grad_value *  weight_values[f_idx];
            atomicAdd((float*)(grad_input + f_base + max_nidx * C + f_idx),
                     weighted_grad);
            atomicAdd((float*)(grad_input + f_base + n * C + f_idx),
                     -weighted_grad);
        }
    }
}

void ctwc_backward(
    torch::Tensor &grad_input,
    const torch::Tensor &grad_output,
    const torch::Tensor &back_idx,
    const torch::Tensor &weight_store
){
    const int B = grad_input.size(0);
    const int N = grad_input.size(1);
    const int C = grad_input.size(2) / 4;

    // 初始化梯度为0
    grad_input.zero_();

    constexpr int block_size = 256;
    const int grid_size = (B * N * C + block_size - 1) / block_size;

    ctwc_backward_kernel<block_size><<<grid_size, block_size>>>(
        grad_input.data_ptr<float>(),
        grad_output.data_ptr<float>(),
        back_idx.data_ptr<uint32_t>(),
        weight_store.data_ptr<float>(),
        N, C, B * N * C
    );

    checkCudaError();
}

