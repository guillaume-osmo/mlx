// Copyright © 2024 Apple Inc.
// Fused GRU cell Metal backend. See Apple Metal docs:
// https://developer.apple.com/documentation/metal

#include <algorithm>
#include <cstdlib>
#include <optional>

#include "mlx/allocator.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"
#include "mlx/utils.h"

namespace mlx::core::fast {

namespace {

uint32_t parse_precise_math_env() {
  const char* precise_env = std::getenv("MLX_FAST_GRU_PRECISE_MATH");
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

void FastGruCell::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  outputs = fallback_(inputs);
}

void FastGruCell::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  const array& input_proj = inputs[0];
  const array& hidden_proj = inputs[1];
  const array& hidden_prev = inputs[2];
  const bool has_bhn = (inputs.size() == 4);
  array& out = outputs[0];

  if (input_proj.dtype() != float32 && input_proj.dtype() != bfloat16) {
    outputs = fallback_(inputs);
    return;
  }
  if (input_proj.dtype() != hidden_proj.dtype() ||
      input_proj.dtype() != hidden_prev.dtype() ||
      (has_bhn && input_proj.dtype() != inputs[3].dtype())) {
    outputs = fallback_(inputs);
    return;
  }

  out.set_data(allocator::malloc(out.nbytes()));

  std::vector<array> copies;
  copies.reserve(has_bhn ? 4 : 3);
  auto copy_if_needed = [&copies, &s](const array& a) -> array {
    if (a.flags().row_contiguous) {
      return a;
    }
    copies.push_back(contiguous_copy_gpu(a, s));
    return copies.back();
  };
  array in_proj = copy_if_needed(input_proj);
  array hid_proj = copy_if_needed(hidden_proj);
  array hid_prev = copy_if_needed(hidden_prev);
  std::optional<array> bhn_contig;
  if (has_bhn) {
    bhn_contig = copy_if_needed(inputs[3]);
  }

  size_t batch_size = in_proj.shape(0);
  size_t hidden_size = hid_prev.shape(1);
  uint32_t h_quads = (static_cast<uint32_t>(hidden_size) + 3) / 4;
  uint32_t total_threads = static_cast<uint32_t>(batch_size) * h_quads;

  std::string kname;
  if (has_bhn) {
    kname = (in_proj.dtype() == bfloat16)
        ? "gru_cell_fused_bfloat16_bias"
        : "gru_cell_fused_float_bias";
  } else {
    kname = (in_proj.dtype() == bfloat16)
        ? "gru_cell_fused_bfloat16"
        : "gru_cell_fused_float";
  }

  auto& enc = d.get_command_encoder(s.index);
  auto kernel = d.get_kernel(kname);
  enc.set_compute_pipeline_state(kernel);

  enc.set_input_array(in_proj, 0);
  enc.set_input_array(hid_proj, 1);
  enc.set_input_array(hid_prev, 2);
  if (has_bhn) {
    enc.set_input_array(*bhn_contig, 3);
    enc.set_output_array(out, 4);
  } else {
    enc.set_output_array(out, 3);
  }
  uint32_t bs = static_cast<uint32_t>(batch_size);
  uint32_t hs = static_cast<uint32_t>(hidden_size);
  uint32_t use_precise = parse_precise_math_env();
  uint32_t bytes_base = has_bhn ? 5u : 4u;
  enc.set_bytes(bs, bytes_base);
  enc.set_bytes(hs, bytes_base + 1);
  enc.set_bytes(use_precise, bytes_base + 2);

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

  d.add_temporaries(std::move(copies), s.index);
}

bool FastGruCell::is_equivalent(const Primitive& other) const {
  return other.name() == name();
}

void FastGruCellVJP::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  outputs = fallback_(inputs);
}

void FastGruCellVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  const array& input_proj = inputs[0];
  const array& hidden_proj = inputs[1];
  const array& hidden_prev = inputs[2];
  const bool has_bhn = (inputs.size() == 5);
  const array* bhn = has_bhn ? &inputs[3] : nullptr;
  const array& cotangent = inputs[has_bhn ? 4 : 3];
  array& d_input_proj = outputs[0];
  array& d_hidden_proj = outputs[1];
  array& d_hidden_prev = outputs[2];

  if (input_proj.dtype() != float32 || hidden_proj.dtype() != float32 ||
      hidden_prev.dtype() != float32 || cotangent.dtype() != float32 ||
      (has_bhn && bhn->dtype() != float32)) {
    outputs = fallback_(inputs);
    return;
  }

  d_input_proj.set_data(allocator::malloc(d_input_proj.nbytes()));
  d_hidden_proj.set_data(allocator::malloc(d_hidden_proj.nbytes()));
  d_hidden_prev.set_data(allocator::malloc(d_hidden_prev.nbytes()));

  std::vector<array> copies;
  copies.reserve(has_bhn ? 5 : 4);
  auto copy_if_needed = [&copies, &s](const array& a) -> array {
    if (a.flags().row_contiguous) {
      return a;
    }
    copies.push_back(contiguous_copy_gpu(a, s));
    return copies.back();
  };
  array in_proj = copy_if_needed(input_proj);
  array hid_proj = copy_if_needed(hidden_proj);
  array hid_prev = copy_if_needed(hidden_prev);
  array cot_out = copy_if_needed(cotangent);
  std::optional<array> bhn_contig;
  if (has_bhn) {
    bhn_contig = copy_if_needed(*bhn);
  }

  size_t batch_size = in_proj.shape(0);
  size_t hidden_size = hid_prev.shape(1);
  uint32_t h_quads = (static_cast<uint32_t>(hidden_size) + 3u) / 4u;
  uint32_t total_threads = static_cast<uint32_t>(batch_size) * h_quads;
  uint32_t bs = static_cast<uint32_t>(batch_size);
  uint32_t hs = static_cast<uint32_t>(hidden_size);
  uint32_t use_precise = parse_precise_math_env();

  std::string kname;
  if (has_bhn) {
    kname = "gru_cell_fused_float_vjp_bias";
  } else {
    kname = "gru_cell_fused_float_vjp";
  }

  auto& enc = d.get_command_encoder(s.index);
  auto kernel = d.get_kernel(kname);
  enc.set_compute_pipeline_state(kernel);

  enc.set_input_array(in_proj, 0);
  enc.set_input_array(hid_proj, 1);
  enc.set_input_array(hid_prev, 2);
  if (has_bhn) {
    enc.set_input_array(*bhn_contig, 3);
    enc.set_input_array(cot_out, 4);
    enc.set_output_array(d_input_proj, 5);
    enc.set_output_array(d_hidden_proj, 6);
    enc.set_output_array(d_hidden_prev, 7);
    enc.set_bytes(bs, 8);
    enc.set_bytes(hs, 9);
    enc.set_bytes(use_precise, 10);
  } else {
    enc.set_input_array(cot_out, 3);
    enc.set_output_array(d_input_proj, 4);
    enc.set_output_array(d_hidden_proj, 5);
    enc.set_output_array(d_hidden_prev, 6);
    enc.set_bytes(bs, 7);
    enc.set_bytes(hs, 8);
    enc.set_bytes(use_precise, 9);
  }

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

  d.add_temporaries(std::move(copies), s.index);
}

bool FastGruCellVJP::is_equivalent(const Primitive& other) const {
  return other.name() == name();
}

} // namespace mlx::core::fast
