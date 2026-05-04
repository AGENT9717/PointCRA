#include <torch/extension.h>

torch::Tensor grid_subsampling(const torch::Tensor &pc_, const float grid_size_, torch::Tensor &hash_table, torch::Tensor &hash_storage);
torch::Tensor grid_subsampling_test(const torch::Tensor &pc_, const float grid_size_, torch::Tensor &hash_table, torch::Tensor &hash_storage, uint32_t ra);

std::vector<size_t> kdtree_build(const torch::Tensor &pc, const size_t max_leaf_size);
void kdtree_free(size_t kdtree, size_t pca);
void kdtree_knn(size_t kdtree, const torch::Tensor &qpc, torch::Tensor &indices, torch::Tensor &dists, const bool sorted);

void knn_edge_maxpooling_backward(
    torch::Tensor &output,
    const torch::Tensor &indices,
    const torch::Tensor &grad
);

void aligned_knn_edge_maxpooling_forward(
    torch::Tensor &output,
    torch::Tensor &indices,
    const torch::Tensor &feature,
    const torch::Tensor &knn
);

void aligned_knn_edge_maxpooling_infer(
    torch::Tensor &output,
    const torch::Tensor &feature,
    const torch::Tensor &knn
);

void half_aligned_knn_edge_maxpooling_forward(
    torch::Tensor &output,
    torch::Tensor &indices,
    const torch::Tensor &feature,
    const torch::Tensor &knn
);

void half_aligned_knn_edge_maxpooling_infer(
    torch::Tensor &output,
    const torch::Tensor &feature,
    const torch::Tensor &knn
);

void half_knn_edge_maxpooling_backward(
    torch::Tensor &output,
    const torch::Tensor &indices,
    const torch::Tensor &grad
);

void knn_spse_forward(
    torch::Tensor &output,
    torch::Tensor &back_idx,
    const torch::Tensor &xyz,
    const torch::Tensor &nbr_idx,
    const torch::Tensor &weight
);

void knn_spse_backward(
    torch::Tensor &wgrad,
    const torch::Tensor &grad,
    const torch::Tensor &back_idx,
    const torch::Tensor &xyz,
    const torch::Tensor &weight
);

void knn_spse_4_forward(
    torch::Tensor &output,
    torch::Tensor &back_idx,
    const torch::Tensor &xyz,
    const torch::Tensor &nbr_idx,
    const torch::Tensor &weight
);

void knn_spse_4_backward(
    torch::Tensor &wgrad,
    const torch::Tensor &grad,
    const torch::Tensor &back_idx,
    const torch::Tensor &xyz,
    const torch::Tensor &weight
);

void knn_spse_4n_forward(
    torch::Tensor &output,
    torch::Tensor &back_idx,
    const torch::Tensor &xyz,
    const torch::Tensor &nbr_idx,
    const torch::Tensor &weight
);

void knn_spse_4n_backward(
    torch::Tensor &wgrad,
    const torch::Tensor &grad,
    const torch::Tensor &back_idx,
    const torch::Tensor &xyz,
    const torch::Tensor &weight
);

void la_spse_backward(
    torch::Tensor &f_grad,
    torch::Tensor &w_grad,
    const torch::Tensor &grad,
    const torch::Tensor &back_idx,
    const torch::Tensor &input,
    const torch::Tensor &xyz,
    const torch::Tensor &weight
);

void la_spse_a4_forward(
    torch::Tensor &output,
    torch::Tensor &back_idx,
    const torch::Tensor &input,
    const torch::Tensor &xyz,
    const torch::Tensor &nbr_idx,
    const torch::Tensor &weight
);

void ctwc_forward(
    torch::Tensor &output,
    torch::Tensor &back_idx,
    const torch::Tensor &input,
    const torch::Tensor &trend_seq,
    const torch::Tensor &nbr_idx,
    torch::Tensor &weight_store
);

// 反向传播声明
void ctwc_backward(
    torch::Tensor &grad_input,
    const torch::Tensor &grad_output,
    const torch::Tensor &back_idx,
    const torch::Tensor &weight_store
);


void ctwc_a0_forward(
    torch::Tensor &output,
    torch::Tensor &back_idx,
    const torch::Tensor &input,
    const torch::Tensor &trend_seq,
    const torch::Tensor &xyz,
    const torch::Tensor &nbr_idx,
    torch::Tensor &weight_store,
    const torch::Tensor &seq_mean,
    const torch::Tensor &seq_std
);

// 反向传播声明
void ctwc_a0_backward(
    const torch::Tensor &grad_output,
    const torch::Tensor &back_idx,
    const torch::Tensor &input,
    const torch::Tensor &weight_store,
    torch::Tensor &grad_input
);

void ctwc_a1_forward(
    torch::Tensor &output,
    torch::Tensor &back_idx,
    const torch::Tensor &input,
    const torch::Tensor &trend_seq,
    const torch::Tensor &xyz,
    const torch::Tensor &nbr_idx,
    const torch::Tensor &weight,
    torch::Tensor &weight_store,
    const torch::Tensor &seq_mean,
    const torch::Tensor &seq_std
);

// 反向传播声明
void ctwc_a1_backward(
    const torch::Tensor &grad_output,
    const torch::Tensor &back_idx,
    const torch::Tensor &input,
    const torch::Tensor &weight_store,
    torch::Tensor &grad_input,
    torch::Tensor &grad_weight
);

void ctwc_a2_forward(
    torch::Tensor &output,
    torch::Tensor &back_idx,
    const torch::Tensor &input,
    const torch::Tensor &trend_seq,
    const torch::Tensor &nbr_idx,
    const torch::Tensor &weight,
    torch::Tensor &weight_store,
    const torch::Tensor &seq_mean,
    const torch::Tensor &seq_std,
    const torch::Tensor &scale
);
void ctwc_a2_1_forward(
    torch::Tensor &output,
    torch::Tensor &back_idx,
    const torch::Tensor &input,
    const torch::Tensor &trend_seq,
    const torch::Tensor &nbr_idx,
    const torch::Tensor &weight,
    torch::Tensor &weight_store,
    const torch::Tensor &seq_mean,
    const torch::Tensor &seq_std,
    const torch::Tensor &scale
);

// 反向传播声明
void ctwc_a2_backward(
    const torch::Tensor &grad_output,
    const torch::Tensor &back_idx,
    const torch::Tensor &input,
    const torch::Tensor &weight_store,
    torch::Tensor &grad_input,
    torch::Tensor &grad_weight
);

void ctwc_a3_forward(
    torch::Tensor &output,
    torch::Tensor &back_idx,
    const torch::Tensor &input,
    const torch::Tensor &trend_seq,  // 新增：趋势序列张量
    const torch::Tensor &nbr_idx,
    const torch::Tensor &weight,    // 学习得到的权重
    torch::Tensor &weight_store,   // 存储相关性权重
    const torch::Tensor &seq_mean,
    const torch::Tensor &seq_std,
    const torch::Tensor &scale,
    const torch::Tensor &gate
);

// 反向传播声明
void ctwc_a3_backward(
    const torch::Tensor &grad_output,
    const torch::Tensor &back_idx,
    const torch::Tensor &input,
    const torch::Tensor &weight,
    const torch::Tensor &weight_store,
    const torch::Tensor &gate,
    torch::Tensor &grad_input,
    torch::Tensor &grad_weight,
    torch::Tensor &grad_gate
);

void ctwc_a4_forward(
    torch::Tensor &output,
    torch::Tensor &back_idx,
    const torch::Tensor &input,
    const torch::Tensor &trend_seq,
    const torch::Tensor &xyz,
    const torch::Tensor &nbr_idx,
    const torch::Tensor &weight,
    torch::Tensor &weight_store,
    const torch::Tensor &seq_mean,
    const torch::Tensor &seq_std
);

// 反向传播声明
void ctwc_a4_backward(
    const torch::Tensor &grad_output,
    const torch::Tensor &back_idx,
    const torch::Tensor &input,
    const torch::Tensor &weight,
    const torch::Tensor &weight_store,
    torch::Tensor &grad_input,
    torch::Tensor &grad_weight
);

void ctwc_a4_1_forward(
    torch::Tensor &output,
    torch::Tensor &back_idx,
    const torch::Tensor &input,
    const torch::Tensor &trend_seq,
    const torch::Tensor &xyz,
    const torch::Tensor &nbr_idx,
    const torch::Tensor &weight,
    torch::Tensor &weight_store,
    const torch::Tensor &seq_mean,
    const torch::Tensor &seq_std
);

// 反向传播声明
void ctwc_a4_1_backward(
    const torch::Tensor &grad_output,
    const torch::Tensor &back_idx,
    const torch::Tensor &input,
    const torch::Tensor &weight,
    const torch::Tensor &weight_store,
    torch::Tensor &grad_input,
    torch::Tensor &grad_weight
);

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
void ctwc_b3_forward(
    torch::Tensor &output,
    const torch::Tensor &trend_seq,  // 新增：趋势序列张量
    const torch::Tensor &nbr_idx,
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
    m.def("grid_subsampling", &grid_subsampling);
    m.def("grid_subsampling_test", &grid_subsampling_test);
    m.def("kdtree_build", &kdtree_build);
    m.def("kdtree_free", &kdtree_free);
    m.def("kdtree_knn", &kdtree_knn);
    m.def("knn_edge_maxpooling_backward", &knn_edge_maxpooling_backward);
    m.def("aligned_knn_edge_maxpooling_forward", &aligned_knn_edge_maxpooling_forward);
    m.def("aligned_knn_edge_maxpooling_infer", &aligned_knn_edge_maxpooling_infer);
    m.def("half_aligned_knn_edge_maxpooling_forward", &half_aligned_knn_edge_maxpooling_forward);
    m.def("half_aligned_knn_edge_maxpooling_infer", &half_aligned_knn_edge_maxpooling_infer);
    m.def("half_knn_edge_maxpooling_backward", &half_knn_edge_maxpooling_backward);

    m.def("knn_spse_forward", &knn_spse_forward);
    m.def("knn_spse_backward", &knn_spse_backward);
    m.def("knn_spse_4_forward", &knn_spse_4_forward);
    m.def("knn_spse_4_backward", &knn_spse_4_backward);
    m.def("knn_spse_4n_forward", &knn_spse_4n_forward);
    m.def("knn_spse_4n_backward", &knn_spse_4n_backward);

    m.def("la_spse_a4_forward", &la_spse_a4_forward);
    m.def("la_spse_backward", &la_spse_backward);

    // 新增的通道趋势注意力函数
    m.def("ctwc_forward", &ctwc_forward);
    m.def("ctwc_backward", &ctwc_backward);
    m.def("ctwc_a0_forward", &ctwc_a0_forward);
    m.def("ctwc_a0_backward", &ctwc_a0_backward);
    m.def("ctwc_a1_forward", &ctwc_a1_forward);
    m.def("ctwc_a1_backward", &ctwc_a1_backward);
    m.def("ctwc_a2_forward", &ctwc_a2_forward);
    m.def("ctwc_a2_1_forward", &ctwc_a2_1_forward);
    m.def("ctwc_a2_backward", &ctwc_a2_backward);
    m.def("ctwc_a3_forward", &ctwc_a3_forward);
    m.def("ctwc_a3_backward", &ctwc_a3_backward);
    m.def("ctwc_a4_forward", &ctwc_a4_forward);
    m.def("ctwc_a4_backward", &ctwc_a4_backward);
    m.def("ctwc_a4_1_forward", &ctwc_a4_1_forward);
    m.def("ctwc_a4_1_backward", &ctwc_a4_1_backward);
    m.def("ctwc_b1_forward", &ctwc_b1_forward);
    m.def("ctwc_b11_forward", &ctwc_b11_forward);
    m.def("ctwc_b_backward", &ctwc_b_backward);
    m.def("ctwc_b2_forward", &ctwc_b2_forward);
    m.def("ctwc_b3_forward", &ctwc_b3_forward);
    m.def("ctwc_wc_forward", &ctwc_wc_forward);
    m.def("ctwc_wc_backward", &ctwc_wc_backward);
    m.def("ctwc_wc1_forward", &ctwc_wc1_forward);
    m.def("ctwc_wc1_backward", &ctwc_wc1_backward);
}