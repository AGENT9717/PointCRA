#include <torch/extension.h>



void ctwc_b1_forward(
    const torch::Tensor &trend_seq,
    const torch::Tensor &nbr_idx,
    torch::Tensor &weight_store,
    const torch::Tensor &scale
);

void ctwc_b11_forward(
    const torch::Tensor &trend_seq,
    const torch::Tensor &nbr_idx,
    torch::Tensor &weight_store,
    const torch::Tensor &scale
);

void ctwc_b2_forward(
    const torch::Tensor &nbr_idx,
    torch::Tensor &weight_store,
    const torch::Tensor &seq_mean,
    const torch::Tensor &seq_std,
    const torch::Tensor &scale
);

void ctwc_b_backward(
    torch::Tensor &grad_input,
    const torch::Tensor &grad_output,
    const torch::Tensor &back_idx,
    const torch::Tensor &weight_store
);

void ctwc_wc_forward(
    torch::Tensor &output,
    torch::Tensor &back_idx,
    torch::Tensor &weight_back_idx,
    const torch::Tensor &input,
    const torch::Tensor &nbr_idx,
    const torch::Tensor &weight    // 学习得到的权重

);

void ctwc_wc_backward(
    const torch::Tensor &grad_output,
    const torch::Tensor &back_idx,
    const torch::Tensor &weight_back_idx,
    const torch::Tensor &input,
    const torch::Tensor &weight,
    torch::Tensor &grad_input,
    torch::Tensor &grad_weight
) ;

void ctwc_wc1_forward(
    torch::Tensor &output,
    const torch::Tensor &input,
    const torch::Tensor &nbr_idx,
    const torch::Tensor &weight    // 学习得到的权重

);

void ctwc_wc1_backward(
    torch::Tensor &grad_output,
    torch::Tensor &grad_input,
    torch::Tensor &grad_weight,
    const torch::Tensor &input,
    const torch::Tensor &nbr_idx,
    const torch::Tensor &weight
) ;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    // 原有的函数声明

    // 新增的通道趋势注意力函数
    m.def("ctwc_b1_forward", &ctwc_b1_forward);
    m.def("ctwc_b11_forward", &ctwc_b11_forward);
    m.def("ctwc_b_backward", &ctwc_b_backward);
    m.def("ctwc_b2_forward", &ctwc_b2_forward);
    m.def("ctwc_wc_forward", &ctwc_wc_forward);
    m.def("ctwc_wc_backward", &ctwc_wc_backward);
    m.def("ctwc_wc1_forward", &ctwc_wc1_forward);
    m.def("ctwc_wc1_backward", &ctwc_wc1_backward);
}
