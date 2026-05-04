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
__global__ void __launch_bounds__(block_size) ctwc_b2_forward_kernel(
    const uint32_t *nbr_idx,  //      B N k      邻居索引
    float *weight_store,   //      B N k    存储相关性权重 (为方向传播存储)
    const float *seq_mean, //    B N T
    const float *seq_std, // B N T
    const float scale,
    const uint64_t N,
    const uint64_t K,
    const uint64_t T,         //     趋势序列长度 (新增参数)
    const uint64_t BNK       //     特征分组总数
){
    uint64_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= BNK) return;
    const uint64_t k_i = idx % K,   // 当前通道组索引
                   bNn = idx / K,  //当前总点数
                   n = bNn % N,     //当前批次点数
                   b = bNn / N,     //当前批次
                   n_base = bNn * K;


    const uint64_t nidx = nbr_idx[n_base + k_i];
     if (nidx >= N) {
        weight_store[b * N * K + n * K + k_i] = 0.0f;
        return;
    }
    const float was_dist = calc_wasserstein_dist(seq_mean + b * N * T + n * T,
                                                 seq_mean + b * N * T + nidx * T,
                                                 seq_std + b * N * T + n * T,
                                                 seq_std + b * N * T + nidx * T,
                                                 T);

    weight_store[b*N*K + n*K + k_i] = was_dist*scale;


}

void ctwc_b2_forward(
    const torch::Tensor &nbr_idx,
    torch::Tensor &weight_store,   // 存储相关性权重
    const torch::Tensor &seq_mean,
    const torch::Tensor &seq_std,
    const torch::Tensor &scale
){
    const uint64_t B = weight_store.size(0),
                   N = weight_store.size(1),
                   K = weight_store.size(2),
                   T = seq_mean.size(2);
    constexpr uint64_t block_size = 512;
    float scale_val = scale.item<float>();
    ctwc_b2_forward_kernel<block_size>
        <<<(B * N * K + block_size - 1) / block_size, block_size>>>(
        (const uint32_t*)nbr_idx.data_ptr(),
        (float*)weight_store.data_ptr(),
        (const float*)seq_mean.data_ptr(),
        (const float*)seq_std.data_ptr(),
        scale_val,
        N, K, T, B * N * K
    );

    checkCudaError();
}
