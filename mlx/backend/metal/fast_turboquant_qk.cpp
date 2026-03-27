// Copyright © 2026 Apple Inc.
// Fused TurboQuant decode + QK score path on Metal.

#include <cctype>
#include <cstdlib>

#include "mlx/allocator.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"
#include "mlx/utils.h"

namespace mlx::core::fast {

namespace {

int getenv_int(const char* name, int default_value) {
  const char* v = std::getenv(name);
  if (!v || !*v) {
    return default_value;
  }
  char* end = nullptr;
  long x = std::strtol(v, &end, 10);
  if (end == v || *end != '\0') {
    return default_value;
  }
  return static_cast<int>(x);
}

std::string getenv_lower(const char* name, const std::string& default_value) {
  std::string out = default_value;
  if (const char* v = std::getenv(name)) {
    out = v;
  }
  for (auto& ch : out) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return out;
}

} // namespace

void FastTurboQuantQK::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  outputs = fallback_(inputs);
}

void FastTurboQuantQK::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  const array& q_rot = inputs[0];
  const array& k_packed = inputs[1];
  const array& k_norms = inputs[2];
  const array& centroids = inputs[3];
  array& out = outputs[0];

  if (q_rot.dtype() != float32 || k_packed.dtype() != uint32 ||
      k_norms.dtype() != float32 || centroids.dtype() != float32) {
    outputs = fallback_(inputs);
    return;
  }

  out.set_data(allocator::malloc(out.nbytes()));

  std::vector<array> copies;
  auto copy_if_needed = [&copies, &s](const array& a) -> const array& {
    if (a.flags().row_contiguous) {
      return a;
    }
    copies.push_back(contiguous_copy_gpu(a, s));
    return copies.back();
  };

  const array& q = copy_if_needed(q_rot);
  const array& kp = copy_if_needed(k_packed);
  const array& norms = copy_if_needed(k_norms);
  const array& c = copy_if_needed(centroids);

  uint32_t n_queries = static_cast<uint32_t>(q.shape(0));
  uint32_t dim = static_cast<uint32_t>(q.shape(1));
  uint32_t n_keys = static_cast<uint32_t>(kp.shape(0));
  uint32_t words_per_key = static_cast<uint32_t>(kp.shape(1));
  uint32_t vals_per_word = static_cast<uint32_t>(32 / bits_);
  uint32_t mask = static_cast<uint32_t>((1 << bits_) - 1);
  uint32_t expected_words = (dim + vals_per_word - 1) / vals_per_word;
  if (words_per_key != expected_words) {
    outputs = fallback_(inputs);
    return;
  }

  std::string mode = getenv_lower("MLX_TQ_NATIVE_MODE", "auto");

  bool use_simd = false;
  int unroll = 1;
  int auto_min_keys = getenv_int("MLX_TQ_AUTO_MIN_KEYS", 1024);
  int auto_min_dim = getenv_int("MLX_TQ_AUTO_MIN_DIM", 128);
  if (n_queries == 1 && mode != "scalar") {
    auto gen = d.get_architecture_gen();
    if (mode == "simd") {
      use_simd = true;
    } else if (mode == "auto") {
      // Arch-aware defaults for decode-like batched path.
      if (gen >= 14) {
        use_simd = (static_cast<int>(dim) >= 64 && static_cast<int>(n_keys) >= 128);
      } else if (gen >= 13) {
        use_simd = (static_cast<int>(dim) >= 64 && static_cast<int>(n_keys) >= 128);
      } else {
        use_simd = (static_cast<int>(dim) >= auto_min_dim &&
                    static_cast<int>(n_keys) >= auto_min_keys);
      }
    }
    if (use_simd) {
      if (gen >= 14) {
        unroll = (dim >= 256) ? 4 : 2;
      } else if (gen >= 13) {
        unroll = (dim >= 192) ? 2 : 1;
      } else {
        unroll = (dim >= 256) ? 2 : 1;
      }
    }
  }
  int force_unroll = getenv_int("MLX_TQ_SIMD_UNROLL", 0);
  if (use_simd && (force_unroll == 1 || force_unroll == 2 || force_unroll == 4)) {
    unroll = force_unroll;
  }

  std::string kname = "turboquant_qk_decode_scalar";
  if (use_simd) {
    if (unroll >= 4) {
      kname = "turboquant_qk_decode_simd_u4";
    } else if (unroll == 2) {
      kname = "turboquant_qk_decode_simd_u2";
    } else {
      kname = "turboquant_qk_decode_simd_u1";
    }
  }

  auto& enc = d.get_command_encoder(s.index);
  auto kernel = d.get_kernel(kname);
  enc.set_compute_pipeline_state(kernel);

  int cidx = 0;
  enc.set_input_array(q, cidx++);
  enc.set_input_array(kp, cidx++);
  enc.set_input_array(norms, cidx++);
  enc.set_input_array(c, cidx++);
  enc.set_output_array(out, cidx++);
  enc.set_bytes(n_queries, cidx++);
  enc.set_bytes(n_keys, cidx++);
  enc.set_bytes(dim, cidx++);
  enc.set_bytes(words_per_key, cidx++);
  enc.set_bytes(vals_per_word, cidx++);
  enc.set_bytes(static_cast<uint32_t>(bits_), cidx++);
  enc.set_bytes(mask, cidx++);

  if (use_simd) {
    MTL::Size group_dims(32, 1, 1);
    MTL::Size grid_dims(n_keys * 32, n_queries, 1);
    enc.dispatch_threads(grid_dims, group_dims);
  } else {
    int tgx_i = getenv_int("MLX_TQ_SCALAR_TGX", 16);
    int tgy_i = getenv_int(
        "MLX_TQ_SCALAR_TGY", (n_keys >= 2048) ? 8 : 4);
    if (tgx_i < 1) tgx_i = 16;
    if (tgy_i < 1) tgy_i = 4;
    uint32_t tgx = static_cast<uint32_t>(tgx_i);
    uint32_t tgy = static_cast<uint32_t>(tgy_i);
    MTL::Size group_dims(tgx, tgy, 1);
    MTL::Size grid_dims(n_keys, n_queries, 1);
    check_kernel_threadgroup_size(kernel, group_dims, kname);
    enc.dispatch_threads(grid_dims, group_dims);
  }

  d.add_temporaries(std::move(copies), s.index);
}

bool FastTurboQuantQK::is_equivalent(const Primitive& other) const {
  const FastTurboQuantQK& a_other = static_cast<const FastTurboQuantQK&>(other);
  return bits_ == a_other.bits_;
}

void FastTurboQuantQKBatched::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  outputs = fallback_(inputs);
}

void FastTurboQuantQKBatched::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  const array& q_rot = inputs[0];
  const array& k_packed = inputs[1];
  const array& k_norms = inputs[2];
  const array& centroids = inputs[3];
  array& out = outputs[0];

  if (q_rot.dtype() != float32 || k_packed.dtype() != uint32 ||
      k_norms.dtype() != float32 || centroids.dtype() != float32) {
    outputs = fallback_(inputs);
    return;
  }

  out.set_data(allocator::malloc(out.nbytes()));

  std::vector<array> copies;
  auto copy_if_needed = [&copies, &s](const array& a) -> const array& {
    if (a.flags().row_contiguous) {
      return a;
    }
    copies.push_back(contiguous_copy_gpu(a, s));
    return copies.back();
  };

  const array& q = copy_if_needed(q_rot);
  const array& kp = copy_if_needed(k_packed);
  const array& norms = copy_if_needed(k_norms);
  const array& c = copy_if_needed(centroids);

  uint32_t batch = static_cast<uint32_t>(q.shape(0));
  uint32_t n_q_heads = static_cast<uint32_t>(q.shape(1));
  uint32_t n_q_len = static_cast<uint32_t>(q.shape(2));
  uint32_t dim = static_cast<uint32_t>(q.shape(3));
  uint32_t n_kv_heads = static_cast<uint32_t>(kp.shape(1));
  uint32_t n_keys = static_cast<uint32_t>(kp.shape(2));
  uint32_t words_per_key = static_cast<uint32_t>(kp.shape(3));
  uint32_t vals_per_word = static_cast<uint32_t>(32 / bits_);
  uint32_t mask = static_cast<uint32_t>((1 << bits_) - 1);

  uint32_t expected_words = (dim + vals_per_word - 1) / vals_per_word;
  if (words_per_key != expected_words ||
      static_cast<uint32_t>(norms.shape(0)) != batch ||
      static_cast<uint32_t>(norms.shape(1)) != n_kv_heads ||
      static_cast<uint32_t>(norms.shape(2)) != n_keys ||
      static_cast<uint32_t>(c.shape(0)) != (1u << bits_) ||
      n_repeats_ <= 0 ||
      static_cast<uint32_t>(n_repeats_) * n_kv_heads != n_q_heads) {
    outputs = fallback_(inputs);
    return;
  }

  std::string mode = getenv_lower("MLX_TQ_BATCHED_NATIVE_MODE", "auto");
  if (mode.empty() || mode == "default") {
    mode = getenv_lower("MLX_TQ_NATIVE_MODE", "auto");
  }

  bool use_simd = false;
  int unroll = 1;
  int auto_min_keys = getenv_int("MLX_TQ_BATCHED_AUTO_MIN_KEYS", 128);
  int auto_min_dim = getenv_int("MLX_TQ_BATCHED_AUTO_MIN_DIM", 64);
  bool decode_like = (n_q_len == 1);
  if (decode_like && mode != "scalar") {
    auto gen = d.get_architecture_gen();
    if (mode == "simd") {
      use_simd = true;
    } else if (mode == "auto") {
      // Arch-aware defaults for decode-like batched path.
      if (gen >= 14) {
        use_simd = (static_cast<int>(dim) >= 64 && static_cast<int>(n_keys) >= 128);
      } else if (gen >= 13) {
        use_simd = (static_cast<int>(dim) >= 64 && static_cast<int>(n_keys) >= 128);
      } else {
        use_simd = (static_cast<int>(dim) >= auto_min_dim &&
                    static_cast<int>(n_keys) >= auto_min_keys);
      }
    }

    if (use_simd) {
      if (gen >= 14) {
        unroll = (dim >= 256) ? 4 : 2; // M4 family
      } else if (gen >= 13) {
        unroll = (dim >= 192) ? 2 : 1; // M3 family
      } else {
        unroll = (dim >= 256) ? 2 : 1; // M2 family / older
      }
    }
  }

  int force_unroll = getenv_int("MLX_TQ_BATCHED_SIMD_UNROLL", 0);
  if (use_simd && (force_unroll == 1 || force_unroll == 2 || force_unroll == 4)) {
    unroll = force_unroll;
  }

  std::string kname = "turboquant_qk_decode_batched_scalar";
  if (use_simd) {
    if (unroll >= 4) {
      kname = "turboquant_qk_decode_batched_simd_u4";
    } else if (unroll == 2) {
      kname = "turboquant_qk_decode_batched_simd_u2";
    } else {
      kname = "turboquant_qk_decode_batched_simd_u1";
    }
  }

  auto& enc = d.get_command_encoder(s.index);
  auto kernel = d.get_kernel(kname);
  enc.set_compute_pipeline_state(kernel);

  int cidx = 0;
  enc.set_input_array(q, cidx++);
  enc.set_input_array(kp, cidx++);
  enc.set_input_array(norms, cidx++);
  enc.set_input_array(c, cidx++);
  enc.set_output_array(out, cidx++);
  enc.set_bytes(batch, cidx++);
  enc.set_bytes(n_q_heads, cidx++);
  enc.set_bytes(n_kv_heads, cidx++);
  enc.set_bytes(static_cast<uint32_t>(n_repeats_), cidx++);
  enc.set_bytes(n_q_len, cidx++);
  enc.set_bytes(n_keys, cidx++);
  enc.set_bytes(dim, cidx++);
  enc.set_bytes(words_per_key, cidx++);
  enc.set_bytes(vals_per_word, cidx++);
  enc.set_bytes(static_cast<uint32_t>(bits_), cidx++);
  enc.set_bytes(mask, cidx++);

  if (use_simd) {
    MTL::Size group_dims(32, 1, 1);
    MTL::Size grid_dims(n_keys * 32, n_q_heads * n_q_len, batch);
    enc.dispatch_threads(grid_dims, group_dims);
  } else {
    auto gen = d.get_architecture_gen();
    int default_tgx = (gen >= 13) ? 16 : 8;
    int default_tgy = (n_keys >= 2048) ? ((gen >= 13) ? 8 : 4) : 4;
    int tgx_i = getenv_int("MLX_TQ_BATCHED_SCALAR_TGX", default_tgx);
    int tgy_i = getenv_int("MLX_TQ_BATCHED_SCALAR_TGY", default_tgy);
    if (tgx_i < 1) tgx_i = default_tgx;
    if (tgy_i < 1) tgy_i = default_tgy;
    uint32_t tgx = static_cast<uint32_t>(tgx_i);
    uint32_t tgy = static_cast<uint32_t>(tgy_i);
    MTL::Size group_dims(tgx, tgy, 1);
    MTL::Size grid_dims(n_keys, n_q_heads * n_q_len, batch);
    check_kernel_threadgroup_size(kernel, group_dims, kname);
    enc.dispatch_threads(grid_dims, group_dims);
  }

  d.add_temporaries(std::move(copies), s.index);
}

bool FastTurboQuantQKBatched::is_equivalent(const Primitive& other) const {
  const FastTurboQuantQKBatched& a_other =
      static_cast<const FastTurboQuantQKBatched&>(other);
  return bits_ == a_other.bits_ && n_repeats_ == a_other.n_repeats_;
}

} // namespace mlx::core::fast
