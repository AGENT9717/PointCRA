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

void ctwc_b_backward(
    torch::Tensor &grad_input,
    const torch::Tensor &grad_output,
    const torch::Tensor &back_idx,
    const torch::Tensor &weight_store
) {
    // 空实现 - 什么都不做
    // 或者如果需要，设置梯度为0
    if (grad_input.defined()) {
        grad_input.zero_();
    }
}
