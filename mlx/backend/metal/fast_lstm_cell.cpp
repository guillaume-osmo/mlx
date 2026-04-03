// Copyright © 2024 Apple Inc.
// Fused LSTM cell Metal backend.

#include <algorithm>
#include <cstdlib>

#include "mlx/allocator.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"
#include "mlx/utils.h"

namespace mlx::core::fast {

namespace {

uint32_t parse_lstm_precise_math_env() {
  const char* precise_env = std::getenv("MLX_FAST_LSTM_PRECISE_MATH");
  return (precise_env &&
          (precise_env[0] == '1' || precise_env[0] == 't' ||
           precise_env[0] == 'T'))
      ? 1u
      : 0u;
}

uint32_t pick_threads_per_group(
    uint32_t hidden_size,
    uint32_t batch_size,
    uint32_t total_threads,
    uint32_t max_threads_per_group) {
  uint32_t target = 256u;
  if (hidden_size >= 512 || batch_size >= 512) {
    target = 1024u;
  } else if (hidden_size >= 256 || batch_size >= 128) {
    target = 512u;
  } else if (hidden_size >= 64 || batch_size >= 32) {
    target = 256u;
  } else {
    target = 128u;
  }
  uint32_t tg = std::min(target, std::max(1u, max_threads_per_group));
  if (tg >= 32u) {
    tg = (tg / 32u) * 32u;
  }
  tg = std::max(1u, std::min(tg, total_threads));
  return tg;
}

} // namespace

void FastLSTMCell::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  outputs = fallback_(inputs);
}

void FastLSTMCell::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  const array& input_proj = inputs[0];
  const array& hidden_proj = inputs[1];
  const array& cell_prev = inputs[2];
  array& out_cell = outputs[0];
  array& out_hidden = outputs[1];

  if (input_proj.dtype() != float32 && input_proj.dtype() != bfloat16) {
    outputs = fallback_(inputs);
    return;
  }
  if (input_proj.dtype() != hidden_proj.dtype() ||
      input_proj.dtype() != cell_prev.dtype()) {
    outputs = fallback_(inputs);
    return;
  }

  // Allocate output buffers on GPU (required before set_output_array)
  out_cell.set_data(allocator::malloc(out_cell.nbytes()));
  out_hidden.set_data(allocator::malloc(out_hidden.nbytes()));

  std::vector<array> copies;
  copies.reserve(3);
  auto copy_if_needed = [&copies, &s](const array& a) -> array {
    if (a.flags().row_contiguous) {
      return a;
    }
    copies.push_back(contiguous_copy_gpu(a, s));
    return copies.back();
  };
  array in_proj = copy_if_needed(input_proj);
  array hid_proj = copy_if_needed(hidden_proj);
  array c_prev = copy_if_needed(cell_prev);

  size_t batch_size = in_proj.shape(0);
  size_t hidden_size = c_prev.shape(1);
  uint32_t h_quads = (static_cast<uint32_t>(hidden_size) + 3) / 4;
  uint32_t total_threads = static_cast<uint32_t>(batch_size) * h_quads;

  std::string kname = (in_proj.dtype() == bfloat16)
      ? "lstm_cell_fused_bfloat16"
      : "lstm_cell_fused_float";

  auto& enc = metal::get_command_encoder(s);
  auto kernel = d.get_kernel(kname);
  enc.set_compute_pipeline_state(kernel);

  enc.set_input_array(in_proj, 0);
  enc.set_input_array(hid_proj, 1);
  enc.set_input_array(c_prev, 2);
  enc.set_output_array(out_cell, 3);
  enc.set_output_array(out_hidden, 4);
  uint32_t bs = static_cast<uint32_t>(batch_size);
  uint32_t hs = static_cast<uint32_t>(hidden_size);
  uint32_t use_precise = parse_lstm_precise_math_env();
  enc.set_bytes(bs, 5);
  enc.set_bytes(hs, 6);
  enc.set_bytes(use_precise, 7);

  uint32_t threads_per_group = pick_threads_per_group(
      hs,
      bs,
      total_threads,
      static_cast<uint32_t>(kernel->maxTotalThreadsPerThreadgroup()));
  uint32_t num_groups =
      (total_threads + threads_per_group - 1) / threads_per_group;
  MTL::Size grid_dims(num_groups, 1, 1);
  MTL::Size group_dims(threads_per_group, 1, 1);
  enc.dispatch_threadgroups(grid_dims, group_dims);

  enc.add_temporaries(std::move(copies));
}

bool FastLSTMCell::is_equivalent(const Primitive& other) const {
  return other.name() == name();
}

void FastLSTMCellVJP::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  outputs = fallback_(inputs);
}

void FastLSTMCellVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  const array& input_proj = inputs[0];
  const array& hidden_proj = inputs[1];
  const array& cell_prev = inputs[2];
  const array& cot_cell = inputs[4];
  const array& cot_hidden = inputs[5];
  array& d_input_proj = outputs[0];
  array& d_hidden_proj = outputs[1];
  array& d_cell_prev = outputs[2];

  if (input_proj.dtype() != float32 || hidden_proj.dtype() != float32 ||
      cell_prev.dtype() != float32 || cot_cell.dtype() != float32 ||
      cot_hidden.dtype() != float32) {
    outputs = fallback_(inputs);
    return;
  }

  d_input_proj.set_data(allocator::malloc(d_input_proj.nbytes()));
  d_hidden_proj.set_data(allocator::malloc(d_hidden_proj.nbytes()));
  d_cell_prev.set_data(allocator::malloc(d_cell_prev.nbytes()));

  std::vector<array> copies;
  copies.reserve(5);
  auto copy_if_needed = [&copies, &s](const array& a) -> array {
    if (a.flags().row_contiguous) {
      return a;
    }
    copies.push_back(contiguous_copy_gpu(a, s));
    return copies.back();
  };
  array in_proj = copy_if_needed(input_proj);
  array hid_proj = copy_if_needed(hidden_proj);
  array c_prev = copy_if_needed(cell_prev);
  array cot_c = copy_if_needed(cot_cell);
  array cot_h = copy_if_needed(cot_hidden);

  size_t batch_size = in_proj.shape(0);
  size_t hidden_size = c_prev.shape(1);
  uint32_t h_quads = (static_cast<uint32_t>(hidden_size) + 3u) / 4u;
  uint32_t total_threads = static_cast<uint32_t>(batch_size) * h_quads;
  uint32_t bs = static_cast<uint32_t>(batch_size);
  uint32_t hs = static_cast<uint32_t>(hidden_size);
  uint32_t use_precise = parse_lstm_precise_math_env();

  auto& enc = metal::get_command_encoder(s);
  auto kernel = d.get_kernel("lstm_cell_fused_float_vjp");
  enc.set_compute_pipeline_state(kernel);

  enc.set_input_array(in_proj, 0);
  enc.set_input_array(hid_proj, 1);
  enc.set_input_array(c_prev, 2);
  enc.set_input_array(cot_c, 3);
  enc.set_input_array(cot_h, 4);
  enc.set_output_array(d_input_proj, 5);
  enc.set_output_array(d_hidden_proj, 6);
  enc.set_output_array(d_cell_prev, 7);
  enc.set_bytes(bs, 8);
  enc.set_bytes(hs, 9);
  enc.set_bytes(use_precise, 10);

  uint32_t threads_per_group = pick_threads_per_group(
      hs,
      bs,
      total_threads,
      static_cast<uint32_t>(kernel->maxTotalThreadsPerThreadgroup()));
  uint32_t num_groups =
      (total_threads + threads_per_group - 1) / threads_per_group;
  MTL::Size grid_dims(num_groups, 1, 1);
  MTL::Size group_dims(threads_per_group, 1, 1);
  enc.dispatch_threadgroups(grid_dims, group_dims);

  enc.add_temporaries(std::move(copies));
}

bool FastLSTMCellVJP::is_equivalent(const Primitive& other) const {
  return other.name() == name();
}

} // namespace mlx::core::fast
