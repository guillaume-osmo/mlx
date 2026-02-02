// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <optional>
#include <utility>
#include <variant>

#include "mlx/utils.h"

namespace mlx::core::fast {

MLX_API array rms_norm(
    const array& x,
    const std::optional<array>& weight,
    float eps,
    StreamOrDevice s = {});

MLX_API array layer_norm(
    const array& x,
    const std::optional<array>& weight,
    const std::optional<array>& bias,
    float eps,
    StreamOrDevice s = {});

MLX_API array rope(
    const array& x,
    int dims,
    bool traditional,
    std::optional<float> base,
    float scale,
    int offset,
    const std::optional<array>& freqs = std::nullopt,
    StreamOrDevice s = {});

MLX_API array rope(
    const array& x,
    int dims,
    bool traditional,
    std::optional<float> base,
    float scale,
    const array& offset,
    const std::optional<array>& freqs = std::nullopt,
    StreamOrDevice s = {});

/** Computes attention directly from TurboQuant compressed KV cache data.
 *  Fuses MSE score + QJL correction + value dequantization + online softmax
 *  in a single Metal kernel with zero intermediate allocations.
 *  Returns (acc, max_score, sum_exp) for log-sum-exp merge with buffer. **/
MLX_API std::vector<array> turboquant_attention(
    const array& queries,
    const array& k_packed,
    const array& k_signs,
    const array& k_norms,
    const array& k_res_norms,
    const array& centroids,
    const array& v_packed,
    const array& v_scales,
    const array& v_zeros,
    const array& rotation_matrix,
    const array& sketch_matrix,
    const float scale,
    const float qjl_scale,
    const int mse_bits = 2,
    const int v_bits = 2,
    const int group_size = 32,
    StreamOrDevice s = {});

/** Computes: O = softmax(Q @ K.T) @ V **/
MLX_API array scaled_dot_product_attention(
    const array& queries,
    const array& keys,
    const array& values,
    const float scale,
    const std::string& mask_mode = "",
    std::optional<array> mask_arr = {},
    const std::optional<array>& sinks = {},
    StreamOrDevice s = {});

<<<<<<< HEAD
/** Fused TurboQuant decode+QK score path for packed codebook indices. */
MLX_API array turboquant_qk_packed_scores(
    const array& q_rot,
    const array& k_packed,
    const array& k_norms,
    const array& centroids,
    int bits,
    StreamOrDevice s = {});

// Batched fused TurboQuant decode+QK score path.
MLX_API array turboquant_qk_packed_scores_batched(
    const array& q_rot,
    const array& k_packed,
    const array& k_norms,
    const array& centroids,
    int bits,
    int n_repeats,
    StreamOrDevice s = {});

// Batched TurboQuant prod/QJL score path.
// Computes:
//   q_rot @ dequant_mse(k_packed).T + q_model @ dequant_qjl(qjl_packed).T
MLX_API array turboquant_qjl_score_batched(
    const array& q_proj,
    const array& k_norms,
    const array& qjl_gamma,
    const array& qjl_packed,
    StreamOrDevice s = {});

MLX_API array turboquant_qk_prod_scores_batched(
    const array& q_rot,
    const array& q_model,
    const array& k_packed,
    const array& k_norms,
    const array& centroids,
    int bits,
    const array& qjl_packed,
    const array& qjl_gamma,
    const array& qjl_projection,
    int n_repeats,
    StreamOrDevice s = {});

// Batched fused TurboQuant attention output from packed values.
MLX_API array turboquant_av_packed_values_batched(
    const array& probs,
    const array& v_packed,
    const array& v_norms,
    const array& centroids,
    int bits,
    int n_repeats,
    int value_dim,
    StreamOrDevice s = {});

// Batched fused TurboQuant decode attention:
// out = softmax(q_rot @ dequant(k).T) @ dequant(v)
MLX_API array turboquant_decode_attention_packed_batched(
    const array& q_rot,
    const array& k_packed,
    const array& k_norms,
    const array& v_packed,
    const array& v_norms,
    const array& centroids,
    int bits,
    int n_repeats,
    int value_dim,
    StreamOrDevice s = {});

// Batched fused TurboQuant decode attention in model basis:
// out = inverse_rotate(softmax(q_rot @ dequant(k).T) @ dequant(v))
MLX_API array turboquant_decode_attention_packed_model_batched(
    const array& q_rot,
    const array& k_packed,
    const array& k_norms,
    const array& v_packed,
    const array& v_norms,
    const array& centroids,
    int bits,
    int n_repeats,
    int value_dim,
    const array& value_rotation,
    StreamOrDevice s = {});

// Batched TurboQuant prod/QJL decode attention:
// out = softmax(q_rot @ dequant_mse(k).T + q_model @ dequant_qjl(qjl).T)
//       @ dequant(v)
MLX_API array turboquant_decode_attention_prod_batched(
    const array& q_rot,
    const array& q_model,
    const array& k_packed,
    const array& k_norms,
    const array& k_centroids,
    int k_bits,
    const array& qjl_packed,
    const array& qjl_gamma,
    const array& qjl_projection,
    const array& v_packed,
    const array& v_norms,
    const array& v_centroids,
    int v_bits,
    int n_repeats,
    int value_dim,
    StreamOrDevice s = {});

// Batched TurboQuant prod/QJL decode attention in model basis:
// out = inverse_rotate(softmax(q_rot @ dequant_mse(k).T + q_model @ dequant_qjl(qjl).T)
//       @ dequant(v))
MLX_API array turboquant_decode_attention_prod_model_batched(
    const array& q_rot,
    const array& q_model,
    const array& k_packed,
    const array& k_norms,
    const array& k_centroids,
    int k_bits,
    const array& qjl_packed,
    const array& qjl_gamma,
    const array& qjl_projection,
    const array& v_packed,
    const array& v_norms,
    const array& v_centroids,
    int v_bits,
    int n_repeats,
    int value_dim,
    const array& value_rotation,
    StreamOrDevice s = {});

/** Fused GRU cell (Metal RNN). One step: out = (1-z)*n + z*h_prev with r,z,n
 * from gates. */
MLX_API array gru_cell(
    const array& input_proj,
    const array& hidden_proj,
    const array& hidden_prev,
    StreamOrDevice s = {});

/** Same with optional recurrent bias bhn [H] for n-gate; avoids per-step add in
 * Python. */
MLX_API array gru_cell(
    const array& input_proj,
    const array& hidden_proj,
    const array& hidden_prev,
    const std::optional<array>& bhn,
    StreamOrDevice s = {});

/** Fused LSTM cell (Metal RNN). One step: cell_new = f*c_prev + i*g, hidden_new
 * = o*tanh(cell_new). */
MLX_API std::pair<array, array> lstm_cell(
    const array& input_proj,
    const array& hidden_proj,
    const array& cell_prev,
    const array& hidden_prev,
    StreamOrDevice s = {});

using TemplateArg = std::variant<int, bool, Dtype>;
using ScalarArg = std::variant<bool, int, float>;

using CustomKernelFunction = std::function<std::vector<array>(
    const std::vector<array>&,
    const std::vector<Shape>&,
    const std::vector<Dtype>&,
    std::tuple<int, int, int>,
    std::tuple<int, int, int>,
    std::vector<std::pair<std::string, TemplateArg>>,
    std::optional<float>,
    bool,
    StreamOrDevice)>;

MLX_API CustomKernelFunction metal_kernel(
    const std::string& name,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::string& source,
    const std::string& header = "",
    bool ensure_row_contiguous = true,
    bool atomic_outputs = false);

MLX_API CustomKernelFunction cuda_kernel(
    const std::string& name,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::string& source,
    const std::string& header = "",
    bool ensure_row_contiguous = true,
    int shared_memory = 0);

MLX_API std::vector<array> precompiled_cuda_kernel(
    const std::string& name,
    const std::string& compiled_source,
    const std::vector<array>& inputs,
    const std::vector<Shape>& output_shapes,
    const std::vector<Dtype>& output_dtypes,
    const std::vector<ScalarArg>& scalars,
    std::tuple<int, int, int> grid,
    std::tuple<int, int, int> threadgroup,
    int shared_memory = 0,
    std::optional<float> init_value = std::nullopt,
    bool ensure_row_contiguous = false,
    StreamOrDevice s = {});

} // namespace mlx::core::fast
