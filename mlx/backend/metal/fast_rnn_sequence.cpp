// Copyright © 2024 Apple Inc.
// Full-sequence LSTM/GRU.  eval_gpu dispatches a fused persistent kernel
// that uses simdgroup_matrix for the recurrent GEMM — single Metal dispatch
// for all T timesteps.

#include <cstdlib>

#include "mlx/allocator.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/matmul.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"

namespace mlx::core::fast {

namespace {

constexpr uint32_t TG_MEM_LIMIT = 32768; // 32KB Apple Metal threadgroup memory

array alloc_buf(Shape shape, Dtype dtype) {
  size_t nbytes = 1;
  for (auto d : shape)
    nbytes *= d;
  nbytes *= size_of(dtype);
  return array(allocator::malloc(nbytes), std::move(shape), dtype);
}

bool use_bf16_kernel() {
  static const bool enabled = []() {
    const char* e = std::getenv("MLX_LSTM_BF16");
    return (e && (e[0] == '1' || e[0] == 't' || e[0] == 'T'));
  }();
  return enabled;
}

} // namespace

// ================================================================== LSTM

void FastLSTMSequence::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  outputs = fallback_(inputs);
}

void FastLSTMSequence::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  const array& input_proj = inputs[0];
  const array& Wh = inputs[1];
  const array& h_init = inputs[2];
  const array& c_init = inputs[3];

  int B = input_proj.shape(0);
  int T = input_proj.shape(1);
  int H4 = input_proj.shape(2);
  int H = H4 / 4;

  // At this point, fast.cpp already validated: H % 8 == 0, H <= 204, float32.
  // The fused kernel is guaranteed to fit in TG memory.
  bool bf16 = use_bf16_kernel();

  array& out_h = outputs[0];
  array& out_c = outputs[1];
  out_h.set_data(allocator::malloc(out_h.nbytes()));
  out_c.set_data(allocator::malloc(out_c.nbytes()));

  std::vector<array> temps;
  temps.reserve(16);
  auto contig = [&](const array& a) -> array {
    if (a.flags().row_contiguous)
      return a;
    temps.push_back(contiguous_copy_gpu(a, s));
    return temps.back();
  };

  array x = contig(input_proj);
  array hi = contig(h_init);
  array ci = contig(c_init);
  array wh_orig = contig(Wh);

  // Transpose Wh from [4H, H] to [H, 4H]
  array wh_t = alloc_buf({H, H4}, float32);
  {
    array wh_view({H, H4}, float32, nullptr, {});
    Strides trans_strides{1, static_cast<int64_t>(H)};
    auto flags = wh_orig.flags();
    flags.row_contiguous = false;
    flags.col_contiguous = true;
    wh_view.copy_shared_buffer(
        wh_orig, std::move(trans_strides), flags,
        static_cast<size_t>(H) * H4, 0);
    copy_gpu_inplace(wh_view, wh_t, CopyType::General, s);
  }
  temps.push_back(wh_t);

  constexpr uint32_t b_tile = 8;
  constexpr uint32_t b_tile_pad = 8;
  uint32_t num_tgs = (static_cast<uint32_t>(B) + b_tile - 1) / b_tile;
  uint32_t threads_per_tg = 1024;

  uint32_t bs = static_cast<uint32_t>(B);
  uint32_t sl = static_cast<uint32_t>(T);
  uint32_t hs = static_cast<uint32_t>(H);

  if (bf16) {
    // BFloat16 path: cast Wh_t to bfloat16
    array wh_t_bf = alloc_buf({H, H4}, bfloat16);
    copy_gpu_inplace(wh_t, wh_t_bf, CopyType::Vector, s);
    temps.push_back(wh_t_bf);

    auto kernel = d.get_kernel("lstm_fused_smm_bfloat16");
    threads_per_tg = std::min(
        threads_per_tg,
        static_cast<uint32_t>(kernel->maxTotalThreadsPerThreadgroup()));

    uint32_t shared_h_bytes = b_tile_pad * hs * sizeof(uint16_t);
    uint32_t gate_buf_bytes = b_tile_pad * static_cast<uint32_t>(H4) * sizeof(float);

    auto& enc = d.get_command_encoder(s.index);
    enc.set_compute_pipeline_state(kernel);
    enc.set_input_array(x, 0);
    enc.set_input_array(wh_t_bf, 1);
    enc.set_input_array(hi, 2);
    enc.set_input_array(ci, 3);
    enc.set_output_array(out_h, 4);
    enc.set_output_array(out_c, 5);
    enc.set_bytes(bs, 6);
    enc.set_bytes(sl, 7);
    enc.set_bytes(hs, 8);
    enc.set_bytes(b_tile, 9);
    enc.set_bytes(b_tile_pad, 10);
    enc.set_threadgroup_memory_length(shared_h_bytes, 0);
    enc.set_threadgroup_memory_length(gate_buf_bytes, 1);
    enc.dispatch_threadgroups(
        MTL::Size(num_tgs, 1, 1),
        MTL::Size(threads_per_tg, 1, 1));
  } else {
    auto kernel = d.get_kernel("lstm_fused_smm_float");
    threads_per_tg = std::min(
        threads_per_tg,
        static_cast<uint32_t>(kernel->maxTotalThreadsPerThreadgroup()));

    uint32_t shared_h_bytes = b_tile_pad * hs * sizeof(float);
    uint32_t gate_buf_bytes = b_tile_pad * static_cast<uint32_t>(H4) * sizeof(float);

    auto& enc = d.get_command_encoder(s.index);
    enc.set_compute_pipeline_state(kernel);
    enc.set_input_array(x, 0);
    enc.set_input_array(wh_t, 1);
    enc.set_input_array(hi, 2);
    enc.set_input_array(ci, 3);
    enc.set_output_array(out_h, 4);
    enc.set_output_array(out_c, 5);
    enc.set_bytes(bs, 6);
    enc.set_bytes(sl, 7);
    enc.set_bytes(hs, 8);
    enc.set_bytes(b_tile, 9);
    enc.set_bytes(b_tile_pad, 10);
    enc.set_threadgroup_memory_length(shared_h_bytes, 0);
    enc.set_threadgroup_memory_length(gate_buf_bytes, 1);
    enc.dispatch_threadgroups(
        MTL::Size(num_tgs, 1, 1),
        MTL::Size(threads_per_tg, 1, 1));
  }

  d.add_temporaries(std::move(temps), s.index);
}

bool FastLSTMSequence::is_equivalent(const Primitive& other) const {
  return other.name() == name();
}

// ================================================================== GRU

void FastGRUSequence::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  outputs = fallback_(inputs);
}

void FastGRUSequence::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  outputs = fallback_(inputs);
}

bool FastGRUSequence::is_equivalent(const Primitive& other) const {
  return other.name() == name();
}

} // namespace mlx::core::fast
