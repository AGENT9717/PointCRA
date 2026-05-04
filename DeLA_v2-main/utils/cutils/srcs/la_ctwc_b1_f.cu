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

struct __builtin_align__(16*4) float16  // 对齐到16个float的大小
{
    float v[16];
};

struct __builtin_align__(group_size*4) floatn
{
    float v[group_size];
};

// 通用版本：适用于任意长度的向量（需要指定长度）
// __device__ __forceinline__ float calc_cosine_similarity(
//     const float* vec1,  // center
//     const float* vec2,  // neighbor
//     const int length
// ) {
//     float dot_product = 0.0f;
//     float norm1 = 0.0f;
//     float norm2 = 0.0f;
//
//     for (int i = 0; i < length; ++i) {
//         dot_product += vec1[i] * vec2[i];
//         norm1 += vec1[i] * vec1[i];
//         norm2 += vec2[i] * vec2[i];
//     }
//
//     norm1 = sqrtf(norm1);
//     norm2 = sqrtf(norm2);
//
//     const float epsilon = 1e-8f;
//     return dot_product / (norm1 * norm2 + epsilon);
//     //return dot_product / (norm1 + epsilon);
// }

__device__ __forceinline__ float calc_spherical_similarity(
    const float* seq1,
    const float* seq2,
    const int length
) {
    if (length < 2) return 1e-8f;

    float angle_similarity = 1e-8f;
    int valid_comparisons = 0;

    for (int i = 0; i < length - 1; ++i) {
        float vec1_x = 1.0f;  // 时间维度
        float vec1_y = seq1[i + 1] - seq1[i];  // 值变化

        float vec2_x = 1.0f;
        float vec2_y = seq2[i + 1] - seq2[i];

        // 计算向量长度
        float len1 = sqrtf(vec1_x * vec1_x + vec1_y * vec1_y);
        float len2 = sqrtf(vec2_x * vec2_x + vec2_y * vec2_y);

        if (len1 > 1e-8f && len2 > 1e-8f) {
            // 归一化
            vec1_x /= len1; vec1_y /= len1;
            vec2_x /= len2; vec2_y /= len2;

            // 计算余弦相似度（向量夹角）
            float dot = vec1_x * vec2_x + vec1_y * vec2_y;
            dot = fmaxf(fminf(dot, 1.0f), -1.0f);  // 数值安全

            // 夹角越小，相似度越高
            float angle_sim = (dot + 1.0f) * 0.5f;  // 映射到[0,1]
            angle_similarity += angle_sim;
            valid_comparisons++;
        }
    }

    if (valid_comparisons > 0) {
        return angle_similarity / valid_comparisons;
    }
    return 1e-8f;
}


template <uint64_t block_size>
__global__ void __launch_bounds__(block_size) ctwc_b1_forward_kernel(
    const float *trend_seq,   //      B N T C    趋势序列 (新增参数)
    const uint32_t *nbr_idx,  //      B N k      邻居索引
    float *weight_store,   //      B N K C_+1    存储相关性权重 (为方向传播存储)
    const float scale,
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
                   trend_base_c = b * N * T * C + n * T * C,  // 趋势序列基址 通道连续
                   n_base = bNn * k;


    // 为当前通道组分配一个长度为T的趋势向量
    float channel_group_trend_center[MAX_T];
    for (uint64_t t = 0; t < T; ++t) {
        // 读取当前时间步的4个通道值
        const float4 trend_val = *(float4*)(trend_seq + trend_base_c + t * C + c );

        // 对4个通道求平均，得到该时间步的趋势值
        channel_group_trend_center[t] = (trend_val.x + trend_val.y + trend_val.z + trend_val.w) / 4.0f;
    }


    // 池化


    for (uint64_t i = 0; i < k; ++i) {
        const uint64_t nidx = nbr_idx[n_base + i];
        //float4 cal_dist={0.0f,0.0f,0.0f,0.0f} ;  // 为每个通道存储权重

        // 计算通道趋势相似度作为权重
        float channel_group_trend_neighbor[MAX_T]; // 初始化为0
        const uint64_t neighbor_trend_base = b * N * C * T + nidx * C * T;
        for (uint64_t t = 0; t < T; ++t) {
            const float4 trend_val = *(float4*)(trend_seq + neighbor_trend_base + c + t * C);
            channel_group_trend_neighbor[t] = (trend_val.x + trend_val.y + trend_val.z + trend_val.w) / 4.0f;
        }

//         const float similarity = calc_cosine_similarity(channel_group_trend_center, channel_group_trend_neighbor, T); //邻域点在中心点上的投影
//         float temp = (1.0f-(similarity + 1.0f)*0.5f);  // 从[-1,1]映射到[2,0]
        const float similarity = calc_spherical_similarity(channel_group_trend_center, channel_group_trend_neighbor, T);
        float temp =  similarity* scale;
        //float temp = (1.0f - similarity) * scale;
        const float cos_dist =temp;

        weight_store[b*N*k*C_ + n*k*C_ + i*C_+c_] = cos_dist;
    }


}

void ctwc_b1_forward(
    const torch::Tensor &trend_seq,  // 新增：趋势序列张量
    const torch::Tensor &nbr_idx,
    torch::Tensor &weight_store,   // 存储相关性权重
    const torch::Tensor &scale
){
    const uint64_t B = trend_seq.size(0),
                   N = trend_seq.size(1),
                   C = trend_seq.size(3),
                   k = nbr_idx.size(2),
                   T = trend_seq.size(2),
                   C_ = C /4;
    constexpr uint64_t block_size = 512;
    float scale_val = scale.item<float>();
    ctwc_b1_forward_kernel<block_size>
        <<<(B * N * C_ + block_size - 1) / block_size, block_size>>>(
        (const float*)trend_seq.data_ptr(),
        (const uint32_t*)nbr_idx.data_ptr(),
        (float*)weight_store.data_ptr(),
        scale_val,
        N, C_, k, T, B * N * C_
    );

    checkCudaError();
}
