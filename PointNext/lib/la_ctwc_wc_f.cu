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




template <uint64_t block_size>
__global__ void __launch_bounds__(block_size) ctwc_wc_forward_kernel(
    float *output,            //      B N C      输出特征
    uint32_t *back_idx,       //      B N C      记录最大值的邻居索引
    uint32_t *weight_back_idx,       // B N C      记录最大值的邻居索引
    const float *input,       //      B N C      输入特征
    const uint32_t *nbr_idx,  //      B N k      邻居索引
    const float *weight,         //   B N K C_     s1,s2,s3,0
    const uint64_t N,
    const uint64_t C_,        //      C = 4 * C_    通道分组总数
    const uint64_t k,
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
                   n_base = bNn * k;


    // 读取中心点特征
    const floatn cf = *(floatn*)(input + f_base + n * C);


    // 池化
    floatn max_val{-1e8, -1e8, -1e8, -1e8};
    uint32_t max_idx[group_size];
    uint32_t max_weight_idx[group_size];
    for (uint64_t i = 0; i < k; ++i) {
        const uint64_t nidx = nbr_idx[n_base + i];
         if (nidx >= N) {
                continue;
            }
        // 读取邻居点特征并应用权重
        const floatn valn = *(floatn*)(input + f_base + nidx * C);
        const float l_weight = weight[bNn*k*C_ + i*C_ + c_];
        // 计算加权后的差异并进行最大池化
        for (uint64_t f_idx = 0; f_idx < group_size; ++f_idx) {
            //const float diff = (valn.v[f_idx] - cf.v[f_idx])*l_weight;  // 加权邻域特征 - 中心特征
            const float diff = valn.v[f_idx]*l_weight;
            if (diff > max_val.v[f_idx]) {
                max_val.v[f_idx] = diff;
                max_idx[f_idx] = nidx;
                max_weight_idx[f_idx] = bNn*k*C_ + i*C_ + c_;
            }
        }
    }

    // 写入结果
    for (uint64_t f_idx = 0; f_idx < group_size; ++f_idx) {
        back_idx[f_base + n * C + f_idx] = max_idx[f_idx];
        weight_back_idx[f_base + n * C + f_idx] = max_weight_idx[f_idx];

    }
    *(floatn*)(output + f_base + n * C) = max_val;
}

void ctwc_wc_forward(
    torch::Tensor &output,
    torch::Tensor &back_idx,
    torch::Tensor &weight_back_idx,
    const torch::Tensor &input,
    const torch::Tensor &nbr_idx,
    const torch::Tensor &weight    // 学习得到的权重

){
    const uint64_t B = output.size(0),
                   N = output.size(1),
                   C_ = output.size(2),
                   k = weight.size(2),
                   C = C_ *4;
    constexpr uint64_t block_size = 512;
    ctwc_wc_forward_kernel<block_size>
        <<<(B * N * C_ + block_size - 1) / block_size, block_size>>>(
        (float*)output.data_ptr(),
        (uint32_t*)back_idx.data_ptr(),
        (uint32_t*)weight_back_idx.data_ptr(),
        (const float*)input.data_ptr(),
        (const uint32_t*)nbr_idx.data_ptr(),
        (const float*)weight.data_ptr(),
        N, C_, k,  B * N * C_
    );

    checkCudaError();
}
