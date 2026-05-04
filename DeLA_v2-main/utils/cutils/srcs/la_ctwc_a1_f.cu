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
__device__ __forceinline__ float calc_cosine_similarity(
    const float* vec1,  // center
    const float* vec2,  // neighbor
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
    //return dot_product / (norm1 + epsilon);
}


__device__ static float calc_distance(
    const float4 center_xyz,
    const float4 xyz
){
    const float x = xyz.x - center_xyz.x;
    const float y = xyz.y - center_xyz.y;
    const float z = xyz.z - center_xyz.z;
    return sqrtf(1e-8f + x*x + y*y + z*z);
}

__device__ static float calc_wasserstein_dist(
    const float* seq_mean_c,
    const float* seq_mean_n,
    const float* seq_std_c,
    const float* seq_std_n,
    const int length
){
    float dist = 1e-8f;
        for (int i = 0; i < length; ++i) {
    float mean_diff = seq_mean_c[i] - seq_mean_n[i];
    float std_diff = seq_std_c[i] - seq_std_n[i];
    dist += mean_diff * mean_diff + std_diff * std_diff;

    }
    return sqrtf(dist);
}



template <uint64_t block_size>
__global__ void __launch_bounds__(block_size) ctwc_a1_forward_kernel(
    float *output,            //      B N C      输出特征
    uint32_t *back_idx,       //      B N C      记录最大值的邻居索引
    const float *input,       //      B N C      输入特征
    const float *trend_seq,   //      B N T C    趋势序列 (新增参数)
    const float *xyz,         //      B N 4      坐标
    const uint32_t *nbr_idx,  //      B N k      邻居索引
    const float *weight,         //   C 4     s1,s2,s3,0
    float *weight_store,   //      B N C_ 4    存储相关性权重 (为方向传播存储)
    const float *seq_mean, //    B N T
    const float *seq_std, // B N T
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
    const float4 l_weight = *(float4*)(weight + c_*4);
    for (uint64_t t = 0; t < T; ++t) {
        // 读取当前时间步的4个通道值
        const float4 trend_val = *(float4*)(trend_seq + trend_base_c + t * C + c );

        // 对4个通道求平均，得到该时间步的趋势值
        channel_group_trend_center[t] = (trend_val.x + trend_val.y + trend_val.z + trend_val.w) / 4.0f;
    }

    //中心化


    // 读取中心点特征
    const floatn cf = *(floatn*)(input + f_base + n * C);
    //const floatn  seq_mean_c = *(floatn*)(seq_mean + b * N * T + n * T);
    //const floatn  seq_std_c = *(floatn*)(seq_std + b * N * T + n * T);

    // 池化
    floatn max_val{-1e8, -1e8, -1e8, -1e8};
    uint32_t max_idx[group_size];
    float16 max_weight ;  // 为每个通道存储权重

    for (uint64_t i = 0; i < k; ++i) {
        const uint64_t nidx = nbr_idx[n_base + i];
        //const floatn seq_mean_n = *(floatn*)(seq_mean + b * N * T + nidx * T);
        //const floatn seq_std_n = *(floatn*)(seq_std + b * N * T + nidx * T);
        // 读取邻居点的趋势序列
        //loat4 neighbor_trend{-1e8, -1e8, -1e8, -1e8};
        //欧式距离
        const float4 center_xyz = *(float4*)(xyz + bNn*4);
        const float4 neighbor_xyz = *(float4*)(xyz + b*N*4 + nidx*4);
        const float xyz_dist = calc_distance(center_xyz,neighbor_xyz);
        // wassertein距离
        const float was_dist = calc_wasserstein_dist(seq_mean + b * N * T + n * T,
                                                     seq_mean + b * N * T + nidx * T,
                                                     seq_std + b * N * T + n * T,
                                                     seq_std + b * N * T + nidx * T,
                                                     T);
        // 计算通道趋势相似度作为权重
        float channel_group_trend_neighbor[MAX_T]; // 初始化为0
        const uint64_t neighbor_trend_base = b * N * C * T + nidx * C * T;
        for (uint64_t t = 0; t < T; ++t) {
            const float4 trend_val = *(float4*)(trend_seq + neighbor_trend_base + c + t * C);
            channel_group_trend_neighbor[t] = (trend_val.x + trend_val.y + trend_val.z + trend_val.w) / 4.0f;
        }
        const float similarity = calc_cosine_similarity(channel_group_trend_center, channel_group_trend_neighbor, T); //邻域点在中心点上的投影
        const float cos_dist = 1.0f-(similarity + 1.0f) * 0.5f;  // 从[-1,1]映射到[0,1]


        const float final_weight = __expf(-(xyz_dist * l_weight.x + was_dist*l_weight.y + cos_dist*l_weight.z));
        // 读取邻居点特征并应用权重
        const floatn valn = *(floatn*)(input + f_base + nidx * C);
        // 计算加权后的差异并进行最大池化
        for (uint64_t f_idx = 0; f_idx < group_size; ++f_idx) {
            const float diff = (valn.v[f_idx] - cf.v[f_idx])*final_weight;  // 加权邻域特征 - 中心特征
            if (diff > max_val.v[f_idx]) {
                max_val.v[f_idx] = diff;
                max_idx[f_idx] = nidx;
                max_weight.v[f_idx*4+0]= final_weight;
                max_weight.v[f_idx*4+1]= xyz_dist;
                max_weight.v[f_idx*4+2]= was_dist;
                max_weight.v[f_idx*4+3]= cos_dist;
            }
        }
    }

    // 写入结果
    for (uint64_t f_idx = 0; f_idx < group_size; ++f_idx) {
        back_idx[f_base + n * C + f_idx] = max_idx[f_idx];

    }
    *(float16*)(weight_store+b*N*C*4 + n * C* 4 + c*4) = max_weight;
    *(floatn*)(output + f_base + n * C) = max_val;
}

void ctwc_a1_forward(
    torch::Tensor &output,
    torch::Tensor &back_idx,
    const torch::Tensor &input,
    const torch::Tensor &trend_seq,  // 新增：趋势序列张量
    const torch::Tensor &xyz,
    const torch::Tensor &nbr_idx,
    const torch::Tensor &weight,    // 学习得到的权重
    torch::Tensor &weight_store,   // 存储相关性权重
    const torch::Tensor &seq_mean,
    const torch::Tensor &seq_std

){
    const uint64_t B = output.size(0),
                   N = output.size(1),
                   C = output.size(2),
                   k = nbr_idx.size(2),
                   T = trend_seq.size(2),
                   C_ = C /4;
    constexpr uint64_t block_size = 512;
    ctwc_a1_forward_kernel<block_size>
        <<<(B * N * C_ + block_size - 1) / block_size, block_size>>>(
        (float*)output.data_ptr(),
        (uint32_t*)back_idx.data_ptr(),
        (const float*)input.data_ptr(),
        (const float*)trend_seq.data_ptr(),
        (const float*)xyz.data_ptr(),
        (const uint32_t*)nbr_idx.data_ptr(),
        (const float*)weight.data_ptr(),
        (float*)weight_store.data_ptr(),
        (const float*)seq_mean.data_ptr(),
        (const float*)seq_std.data_ptr(),
        N, C_, k, T, B * N * C_
    );

    checkCudaError();
}
