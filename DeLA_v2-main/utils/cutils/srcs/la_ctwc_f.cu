#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <assert.h>
#include <torch/extension.h>
#include "cuda_util.h"

/*
    仿照 la_spse_a4_forward_kernel 结构
    实现基于通道趋势相似度的注意力池化
*/

constexpr uint64_t group_size = 4;
constexpr uint64_t MAX_T = 16;



struct __builtin_align__(group_size*4) floatn
{
    float v[group_size];
};

// 通用版本：适用于任意长度的向量（需要指定长度）
__device__ __forceinline__ float calc_cosine_similarity(
    const float* vec1,
    const float* vec2,
    const int length
) {
    float dot_product = 0.0f;
    float norm1 = 0.0f;
    float norm2 = 0.0f;

    for (int i = 0; i < length; ++i) {
        dot_product += vec1[i] * vec2[i];
        norm1 += vec1[i] * vec1[i];
        norm2 += vec2[i] * vec2[i];
    }

    norm1 = sqrtf(norm1);
    norm2 = sqrtf(norm2);

    const float epsilon = 1e-8f;
    return dot_product / (norm1 * norm2 + epsilon);
}

// __device__ __forceinline__ float calc_mean(
//     float* vec1,
//     const int length
// ) {
//     float sum = 0.0f;
//     for (int i = 0; i < length; ++i) {
//         sum += vec1[i];
//     }
//     sum = sum / static_cast<float>(length);
//     for (int i = 0; i < length; ++i) {
//         vec1[i] -= sum;
//     }
// }

template <uint64_t block_size>
__global__ void __launch_bounds__(block_size) ctwc_forward_kernel(
    float *output,            //      B N C      输出特征
    uint32_t *back_idx,       //      B N C      记录最大值的邻居索引
    const float *input,       //      B N C      输入特征
    const float *trend_seq,   //      B N T C    趋势序列 (新增参数)
    const uint32_t *nbr_idx,  //      B N k      邻居索引
    float *weight_store,   //      B N C    存储相关性权重 (为方向传播存储)
    const uint64_t N,
    const uint64_t C_,        //      C = 4 * C_    通道分组总数
    const uint64_t k,
    const uint64_t T,         //     趋势序列长度 (新增参数)
    const uint64_t BNC_       //     特征分组总数
){
    uint64_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= BNC_) return;
    const uint64_t c_ = idx % C_,   // 当前通道组索引
                   c = c_ * 4,      // 当前通道数索引
                   C = C_ * 4,      //真正的特征总数，单个
                   bNn = idx / C_,  //当前总点数
                   n = bNn % N,     //当前批次点数
                   b = bNn / N,     //当前批次
                   f_base = b * N * C + c,  //前批次索引，但是缺少当前批次点的个数计算
                   trend_base_c = b * N * T * C + n * T * C,  // 趋势序列基址 通道连续
                   n_base = bNn * k;


    // 读取中心点的趋势序列（当前通道组）
    // float4 center_trend{-1e8, -1e8, -1e8, -1e8};
    // 为当前通道组分配一个长度为T的趋势向量
    float channel_group_trend_center[MAX_T];

    for (uint64_t t = 0; t < T; ++t) {
        // 读取当前时间步的4个通道值
        const float4 trend_val = *(float4*)(trend_seq + trend_base_c + t * C + c );

        // 对4个通道求平均，得到该时间步的趋势值
        channel_group_trend_center[t] = (trend_val.x + trend_val.y + trend_val.z + trend_val.w) / 4.0f;
    }

    //中心化
    //calc_mean(channel_group_trend_center,T);

    // 读取中心点特征
    const floatn cf = *(floatn*)(input + f_base + n * C);

    // 池化
    floatn max_val{-1e8, -1e8, -1e8, -1e8};
    uint32_t max_idx[group_size];
    float max_weight[group_size] = {0.0f};  // 为每个通道存储权重

    for (uint64_t i = 0; i < k; ++i) {
        const uint64_t nidx = nbr_idx[n_base + i];
        float channel_group_trend_neighbor[MAX_T]; // 初始化为0
        // 读取邻居点的趋势序列
        //loat4 neighbor_trend{-1e8, -1e8, -1e8, -1e8};

        const uint64_t neighbor_trend_base = b * N * C * T + nidx * C * T;
        for (uint64_t t = 0; t < T; ++t) {
            const float4 trend_val = *(float4*)(trend_seq + neighbor_trend_base + c + t * C);
            channel_group_trend_neighbor[t] = (trend_val.x + trend_val.y + trend_val.z + trend_val.w) / 4.0f;
        }
        //中心化
        //calc_mean(channel_group_trend_neighbor,T);
        // 计算通道趋势相似度作为权重
        const float similarity = calc_cosine_similarity(channel_group_trend_center, channel_group_trend_neighbor, T);
        // 将相似度映射到合适的范围（0-1之间）
        const float weight = (similarity + 1.0f) * 0.5f;  // 从[-1,1]映射到[0,1]

        // 读取邻居点特征并应用权重
        const floatn valn = *(floatn*)(input + f_base + nidx * C);
        // 计算加权后的差异并进行最大池化
        for (uint64_t f_idx = 0; f_idx < group_size; ++f_idx) {
            const float diff = (valn.v[f_idx] - cf.v[f_idx])*weight;  // 加权邻域特征 - 中心特征
            if (diff > max_val.v[f_idx]) {
                max_val.v[f_idx] = diff;
                max_idx[f_idx] = nidx;
                max_weight[f_idx] = weight;
            }
        }
    }

    // 写入结果
    for (uint64_t f_idx = 0; f_idx < group_size; ++f_idx) {
        back_idx[f_base + n * C + f_idx] = max_idx[f_idx];
        weight_store[f_base +n*C + f_idx] = max_weight[f_idx];
    }
    *(floatn*)(output + f_base + n * C) = max_val;
}

void ctwc_forward(
    torch::Tensor &output,
    torch::Tensor &back_idx,
    const torch::Tensor &input,
    const torch::Tensor &trend_seq,  // 新增：趋势序列张量
    const torch::Tensor &nbr_idx,
    torch::Tensor &weight_store    // 存储相关性权重
){
    const uint64_t B = output.size(0),
                   N = output.size(1),
                   C = output.size(2) / 4,
                   k = nbr_idx.size(2),
                   T = trend_seq.size(2);
    constexpr uint64_t block_size = 512;
    ctwc_forward_kernel<block_size>
        <<<(B * N * C + block_size - 1) / block_size, block_size>>>(
        (float*)output.data_ptr(),
        (uint32_t*)back_idx.data_ptr(),
        (const float*)input.data_ptr(),
        (const float*)trend_seq.data_ptr(),
        (const uint32_t*)nbr_idx.data_ptr(),
        (float*)weight_store.data_ptr(),
        N, C, k, T, B * N * C
    );

    checkCudaError();
}
