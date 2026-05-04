#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <assert.h>
#include <torch/extension.h>
#include "cuda_util.h"

constexpr uint64_t group_size = 4;

struct __builtin_align__(group_size*4) floatn {
    float v[group_size];
};
struct __builtin_align__(16*4) float16  // 对齐到16个float的大小
{
    float v[16];
};

// 反向传播核函数
template <uint64_t block_size>
__global__ void __launch_bounds__(block_size) ctwc_a4_1_backward_kernel(
    const float *grad_output,        // B N C     输出梯度
    const uint32_t *back_idx,        // B N C     前向传播记录的邻居索引
    const float *input,              // B N C     输入特征
    const float *weight,              // B N C     输入特征
    const float *weight_store,       // B N C 4  前向存储的权重信息
    float *grad_input,               // B N C     输入梯度_存储矩阵
    float *grad_weight,              // C_ 4      权重梯度_存储矩阵
    const uint64_t N,
    const uint64_t C_,               // C = 4 * C_
    const uint64_t BNC_              // B * N * C_
) {
    uint64_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= BNC_) return;

    const uint64_t c_ = idx % C_,           //判断当前通道组
                   c = c_ * 4,
                   C = C_ * 4,
                   bNn = idx / C_,
                   n = bNn % N,
                   b = bNn / N,
                   f_base = b * N * C + c;

    // 读取前向传播存储的权重信息
    const float16 stored_weights = *(float16*)(weight_store + b * N * C * 4 + n * C * 4 + c * 4);
    const float4 l_weight = *(float4*)(weight + c_*4);

    // 读取当前权重值
    //float4 l_weight = *(float4*)(grad_weight + c_ * 4); // 注意：这里读取的是当前权重值，不是梯度

    // 读取输出梯度
    const floatn grad_out = *(floatn*)(grad_output + f_base + n * C);

    // 读取中心点特征
    const floatn center_feat = *(floatn*)(input + f_base + n * C);

    // 初始化输入梯度

    float4 grad_weight_local = {0.0f, 0.0f, 0.0f, 0.0f};

    // 对每个通道处理
    for (uint64_t f_idx = 0; f_idx < group_size; ++f_idx) {
        const uint64_t nidx = back_idx[f_base + n * C + f_idx];
        const float final_weight = stored_weights.v[f_idx*4+0];
        const float xyz_dist = stored_weights.v[f_idx*4+1];
        const float was_dist = stored_weights.v[f_idx*4+2];
        const float cos_dist = stored_weights.v[f_idx*4+3];

        if (nidx < N) { // 有效的邻居索引
            // 读取邻居点特征
            const floatn neighbor_feat = *(floatn*)(input + f_base + nidx * C);

            // 计算特征差异的梯度
            const float diff = neighbor_feat.v[f_idx] - center_feat.v[f_idx];

            // 输入梯度计算
            // 对中心点的梯度: -final_weight * grad_output

            atomicAdd(&grad_input[ f_base + n * C + f_idx],
                     -grad_out.v[f_idx] * final_weight);
            // 对邻居点的梯度: +final_weight * grad_output
            atomicAdd(&grad_input[f_base + nidx * C + f_idx],
                     grad_out.v[f_idx] * final_weight);

            // 权重相关的梯度计算
            const float grad_final_weight = grad_out.v[f_idx] * diff * final_weight;

            float gaussian_xyz = __expf(-xyz_dist * l_weight.x);
            float gaussian_was = __expf(-was_dist * l_weight.y);
            float gaussian_cos = __expf(-(1.0f - cos_dist) * l_weight.z);


            // 计算权重对各个距离的偏导
            const float d_final_weight_d_xyz = -grad_final_weight * xyz_dist*gaussian_cos;
            const float d_final_weight_d_was = -grad_final_weight * was_dist*gaussian_cos;
            const float d_final_weight_d_cos = grad_final_weight * (gaussian_was+gaussian_xyz)*cos_dist;

            // 累积权重梯度
            grad_weight_local.x +=  d_final_weight_d_xyz;
            grad_weight_local.y +=  d_final_weight_d_was;
            grad_weight_local.z +=  d_final_weight_d_cos;

        }
    }


    // 原子操作累积权重梯度
    atomicAdd(&grad_weight[c_ * 4 + 0], grad_weight_local.x);
    atomicAdd(&grad_weight[c_ * 4 + 1], grad_weight_local.y);
    atomicAdd(&grad_weight[c_ * 4 + 2], grad_weight_local.z);
}

// 包装函数
void ctwc_a4_1_backward(
    const torch::Tensor &grad_output,
    const torch::Tensor &back_idx,
    const torch::Tensor &input,
    const torch::Tensor &weight,
    const torch::Tensor &weight_store,
    torch::Tensor &grad_input,
    torch::Tensor &grad_weight
) {
    const uint64_t B = grad_output.size(0),
                   N = grad_output.size(1),
                   C = grad_output.size(2),
                   C_ = C / 4;

    constexpr uint64_t block_size = 512;
    const uint64_t grid_size = (B * N * C_ + block_size - 1) / block_size;

    // 初始化梯度为0
    grad_input.zero_();
    grad_weight.zero_();

    ctwc_a4_1_backward_kernel<block_size><<<grid_size, block_size>>>(
        grad_output.data_ptr<float>(),
        back_idx.data_ptr<uint32_t>(),
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        weight_store.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        grad_weight.data_ptr<float>(),
        N, C_, B * N * C_
    );

    checkCudaError();
}

