// Copyright © 2026 Apple Inc.
// Fused TurboQuant decode + QK score path on Metal.

#include <algorithm>
#include <cctype>
#include <cmath>
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

  auto& enc = metal::get_command_encoder(s);
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

  enc.add_temporaries(std::move(copies));
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

  const char* block_env = std::getenv("MLX_TQ_BLOCK");
  int block_keys = block_env ? getenv_int("MLX_TQ_BLOCK", 1) : 0;
  if (block_keys != 8 && block_keys != 16 && block_keys != 32) {
    block_keys = 1;
  }
  if (use_simd && block_keys == 1 && block_env == nullptr) {
    auto gen = d.get_architecture_gen();
    block_keys = (gen >= 14) ? 16 : 8;
  }

  std::string lut_mode = getenv_lower("MLX_TQ_QK_CENTROID_LUT", "auto");
  bool use_centroid_lut = false;
  if (use_simd && block_keys > 1 && decode_like && dim <= 256 &&
      (bits_ == 3 || bits_ == 4)) {
    if (lut_mode == "1" || lut_mode == "true" || lut_mode == "on" ||
        lut_mode == "yes" || lut_mode == "lut") {
      use_centroid_lut = true;
    } else if (
        lut_mode != "0" && lut_mode != "false" && lut_mode != "off" &&
        lut_mode != "no" && lut_mode != "legacy") {
      use_centroid_lut = true;
    }
  }

  std::string kname = "turboquant_qk_decode_batched_scalar";
  if (use_simd) {
    auto qk_blocked_name = [&](int selected_unroll) {
      std::string prefix = "turboquant_qk_decode_batched_simd_u" +
                           std::to_string(selected_unroll);
      if (block_keys > 1) {
        prefix += "_b" + std::to_string(block_keys);
        if (use_centroid_lut) {
          prefix += (bits_ == 3) ? "_lut3" : "_lut4";
        }
      }
      return prefix;
    };
    if (unroll >= 4) {
      kname = qk_blocked_name(4);
    } else if (unroll == 2) {
      kname = qk_blocked_name(2);
    } else {
      kname = qk_blocked_name(1);
    }
  }

  auto& enc = metal::get_command_encoder(s);
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
    uint32_t simd_block = static_cast<uint32_t>(block_keys > 1 ? block_keys : 1);
    MTL::Size group_dims(32 * simd_block, 1, 1);
    MTL::Size grid_dims(n_keys * 32, n_q_heads * n_q_len, batch);
    check_kernel_threadgroup_size(kernel, group_dims, kname);
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

  enc.add_temporaries(std::move(copies));
}

bool FastTurboQuantQKBatched::is_equivalent(const Primitive& other) const {
  const FastTurboQuantQKBatched& a_other =
      static_cast<const FastTurboQuantQKBatched&>(other);
  return bits_ == a_other.bits_ && n_repeats_ == a_other.n_repeats_;
}

void FastTurboQuantQJLScoreBatched::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  outputs = fallback_(inputs);
}

void FastTurboQuantQJLScoreBatched::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  const array& q_proj = inputs[0];
  const array& k_norms = inputs[1];
  const array& qjl_gamma = inputs[2];
  const array& qjl_packed = inputs[3];
  array& out = outputs[0];

  if (q_proj.dtype() != float32 || k_norms.dtype() != float32 ||
      qjl_gamma.dtype() != float32 || qjl_packed.dtype() != uint32) {
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

  const array& q = copy_if_needed(q_proj);
  const array& norms = copy_if_needed(k_norms);
  const array& gamma = copy_if_needed(qjl_gamma);
  const array& qp = copy_if_needed(qjl_packed);

  uint32_t batch = static_cast<uint32_t>(q.shape(0));
  uint32_t n_kv_heads = static_cast<uint32_t>(q.shape(1));
  uint32_t repeat_count = static_cast<uint32_t>(q.shape(2));
  uint32_t dim = static_cast<uint32_t>(q.shape(3));
  uint32_t token_count = static_cast<uint32_t>(norms.shape(2));
  uint32_t packed_width = static_cast<uint32_t>(qp.shape(3));
  uint32_t expected_packed_width = (dim + 31u) / 32u;
  if (packed_width != expected_packed_width ||
      static_cast<uint32_t>(norms.shape(0)) != batch ||
      static_cast<uint32_t>(norms.shape(1)) != n_kv_heads ||
      static_cast<uint32_t>(gamma.shape(0)) != batch ||
      static_cast<uint32_t>(gamma.shape(1)) != n_kv_heads ||
      static_cast<uint32_t>(gamma.shape(2)) != token_count ||
      static_cast<uint32_t>(qp.shape(0)) != batch ||
      static_cast<uint32_t>(qp.shape(1)) != n_kv_heads ||
      static_cast<uint32_t>(qp.shape(2)) != token_count) {
    outputs = fallback_(inputs);
    return;
  }

  constexpr float kPi = 3.14159265358979323846f;
  float alpha = std::sqrt(kPi / 2.0f) / static_cast<float>(dim);

  auto& enc = metal::get_command_encoder(s);
  bool use_blocked = (dim <= 128u && token_count >= 128u);
  std::string kname =
      use_blocked ? "turboquant_qjl_score_batched_b8"
                  : "turboquant_qjl_score_batched";
  auto kernel = d.get_kernel(kname);
  enc.set_compute_pipeline_state(kernel);

  int cidx = 0;
  enc.set_input_array(q, cidx++);
  enc.set_input_array(norms, cidx++);
  enc.set_input_array(gamma, cidx++);
  enc.set_input_array(qp, cidx++);
  enc.set_output_array(out, cidx++);
  enc.set_bytes(batch, cidx++);
  enc.set_bytes(n_kv_heads, cidx++);
  enc.set_bytes(repeat_count, cidx++);
  enc.set_bytes(token_count, cidx++);
  enc.set_bytes(dim, cidx++);
  enc.set_bytes(packed_width, cidx++);
  enc.set_bytes(alpha, cidx++);

  MTL::Size group_dims(use_blocked ? 256 : 32, 1, 1);
  MTL::Size grid_dims(
      use_blocked
          ? ((token_count + 7u) / 8u) * 256u
          : 32u,
      repeat_count,
      use_blocked ? (batch * n_kv_heads) : (batch * n_kv_heads * token_count));
  check_kernel_threadgroup_size(kernel, group_dims, kname);
  enc.dispatch_threads(grid_dims, group_dims);

  enc.add_temporaries(std::move(copies));
}

bool FastTurboQuantQJLScoreBatched::is_equivalent(const Primitive& other) const {
  const auto& a_other = static_cast<const FastTurboQuantQJLScoreBatched&>(other);
  (void)a_other;
  return true;
}


void FastTurboQuantAVBatched::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  outputs = fallback_(inputs);
}

void FastTurboQuantAVBatched::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  const array& probs = inputs[0];
  const array& v_packed = inputs[1];
  const array& v_norms = inputs[2];
  const array& centroids = inputs[3];
  array& out = outputs[0];

  if (probs.dtype() != float32 || v_packed.dtype() != uint32 ||
      v_norms.dtype() != float32 || centroids.dtype() != float32) {
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

  const array& p = copy_if_needed(probs);
  const array& vp = copy_if_needed(v_packed);
  const array& norms = copy_if_needed(v_norms);
  const array& c = copy_if_needed(centroids);

  uint32_t batch = static_cast<uint32_t>(p.shape(0));
  uint32_t n_q_heads = static_cast<uint32_t>(p.shape(1));
  uint32_t n_q_len = static_cast<uint32_t>(p.shape(2));
  uint32_t n_keys = static_cast<uint32_t>(p.shape(3));

  uint32_t n_kv_heads = static_cast<uint32_t>(vp.shape(1));
  uint32_t words_per_key = static_cast<uint32_t>(vp.shape(3));
  uint32_t vals_per_word = static_cast<uint32_t>(32 / bits_);
  uint32_t dim = static_cast<uint32_t>(value_dim_);
  uint32_t mask = static_cast<uint32_t>((1 << bits_) - 1);
  uint32_t expected_words = (dim + vals_per_word - 1) / vals_per_word;

  if (words_per_key != expected_words ||
      static_cast<uint32_t>(vp.shape(0)) != batch ||
      static_cast<uint32_t>(vp.shape(2)) != n_keys ||
      static_cast<uint32_t>(norms.shape(0)) != batch ||
      static_cast<uint32_t>(norms.shape(1)) != n_kv_heads ||
      static_cast<uint32_t>(norms.shape(2)) != n_keys ||
      static_cast<uint32_t>(c.shape(0)) != (1u << bits_) ||
      n_repeats_ <= 0 ||
      static_cast<uint32_t>(n_repeats_) * n_kv_heads != n_q_heads) {
    outputs = fallback_(inputs);
    return;
  }

  bool use_simd = false;
  int unroll = 1;
  {
    auto mode = getenv_lower("MLX_TQ_BATCHED_NATIVE_MODE", "auto");
    int auto_min_dim = getenv_int("MLX_TQ_BATCHED_AUTO_MIN_DIM", 64);
    int auto_min_keys = getenv_int("MLX_TQ_BATCHED_AUTO_MIN_KEYS", 128);
    auto gen = d.get_architecture_gen();
    if (mode == "simd") {
      use_simd = true;
    } else if (mode == "auto") {
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
        unroll = (n_keys >= 256) ? 4 : 2;
      } else if (gen >= 13) {
        unroll = (n_keys >= 192) ? 2 : 1;
      } else {
        unroll = (n_keys >= 256) ? 2 : 1;
      }
    }
  }

  int force_unroll = getenv_int("MLX_TQ_BATCHED_SIMD_UNROLL", 0);
  if (use_simd && (force_unroll == 1 || force_unroll == 2 || force_unroll == 4)) {
    unroll = force_unroll;
  }

  const char* block_env = std::getenv("MLX_TQ_BLOCK");
  int block_dims = block_env ? getenv_int("MLX_TQ_BLOCK", 1) : 0;
  if (block_dims != 8 && block_dims != 16 && block_dims != 32) {
    block_dims = 1;
  }
  if (use_simd && block_dims == 1 && block_env == nullptr) {
    auto gen = d.get_architecture_gen();
    block_dims = (gen >= 14) ? 8 : 8;
  }

  auto& enc = metal::get_command_encoder(s);
  std::string kname = "turboquant_av_decode_batched_scalar";
  if (use_simd) {
    auto av_blocked_name = [&](int selected_unroll) {
      std::string prefix = "turboquant_av_decode_batched_simd_u" +
                           std::to_string(selected_unroll);
      return (block_dims > 1) ? prefix + "_b" + std::to_string(block_dims) : prefix;
    };
    if (unroll >= 4) {
      kname = av_blocked_name(4);
    } else if (unroll == 2) {
      kname = av_blocked_name(2);
    } else {
      kname = av_blocked_name(1);
    }
  }
  auto kernel = d.get_kernel(kname);
  enc.set_compute_pipeline_state(kernel);

  int cidx = 0;
  enc.set_input_array(p, cidx++);
  enc.set_input_array(vp, cidx++);
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
    uint32_t simd_block = static_cast<uint32_t>(block_dims > 1 ? block_dims : 1);
    MTL::Size group_dims(32 * simd_block, 1, 1);
    MTL::Size grid_dims(dim * 32, n_q_heads * n_q_len, batch);
    check_kernel_threadgroup_size(kernel, group_dims, kname);
    enc.dispatch_threads(grid_dims, group_dims);
  } else {
    MTL::Size group_dims(16, 4, 1);
    MTL::Size grid_dims(dim, n_q_heads * n_q_len, batch);
    check_kernel_threadgroup_size(kernel, group_dims, kname);
    enc.dispatch_threads(grid_dims, group_dims);
  }

  enc.add_temporaries(std::move(copies));
}

bool FastTurboQuantAVBatched::is_equivalent(const Primitive& other) const {
  const FastTurboQuantAVBatched& a_other =
      static_cast<const FastTurboQuantAVBatched&>(other);
  return bits_ == a_other.bits_ && n_repeats_ == a_other.n_repeats_ &&
         value_dim_ == a_other.value_dim_;
}

void FastTurboQuantDecodeAttentionBatched::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  outputs = fallback_(inputs);
}

void FastTurboQuantDecodeAttentionBatched::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  const array& q_rot = inputs[0];
  const array& k_packed = inputs[1];
  const array& k_norms = inputs[2];
  const array& v_packed = inputs[3];
  const array& v_norms = inputs[4];
  const array& centroids = inputs[5];
  array& out = outputs[0];

  if (q_rot.dtype() != float32 || k_packed.dtype() != uint32 ||
      k_norms.dtype() != float32 || v_packed.dtype() != uint32 ||
      v_norms.dtype() != float32 || centroids.dtype() != float32) {
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
  const array& kn = copy_if_needed(k_norms);
  const array& vp = copy_if_needed(v_packed);
  const array& vn = copy_if_needed(v_norms);
  const array& c = copy_if_needed(centroids);

  uint32_t batch = static_cast<uint32_t>(q.shape(0));
  uint32_t n_q_heads = static_cast<uint32_t>(q.shape(1));
  uint32_t n_q_len = static_cast<uint32_t>(q.shape(2));
  uint32_t q_dim = static_cast<uint32_t>(q.shape(3));
  uint32_t n_kv_heads = static_cast<uint32_t>(kp.shape(1));
  uint32_t n_keys = static_cast<uint32_t>(kp.shape(2));
  uint32_t k_words_per_key = static_cast<uint32_t>(kp.shape(3));
  uint32_t v_words_per_key = static_cast<uint32_t>(vp.shape(3));
  uint32_t vals_per_word = static_cast<uint32_t>(32 / bits_);
  uint32_t value_dim = static_cast<uint32_t>(value_dim_);
  uint32_t mask = static_cast<uint32_t>((1 << bits_) - 1);

  uint32_t expected_k_words = (q_dim + vals_per_word - 1) / vals_per_word;
  uint32_t expected_v_words = (value_dim + vals_per_word - 1) / vals_per_word;
  if (k_words_per_key != expected_k_words ||
      v_words_per_key != expected_v_words ||
      static_cast<uint32_t>(kn.shape(0)) != batch ||
      static_cast<uint32_t>(kn.shape(1)) != n_kv_heads ||
      static_cast<uint32_t>(kn.shape(2)) != n_keys ||
      static_cast<uint32_t>(vn.shape(0)) != batch ||
      static_cast<uint32_t>(vn.shape(1)) != n_kv_heads ||
      static_cast<uint32_t>(vn.shape(2)) != n_keys ||
      static_cast<uint32_t>(c.shape(0)) != (1u << bits_) ||
      n_repeats_ <= 0 ||
      static_cast<uint32_t>(n_repeats_) * n_kv_heads != n_q_heads ||
      q_dim > 128 || value_dim > 128) {
    outputs = fallback_(inputs);
    return;
  }

  uint32_t max_dim = std::max(q_dim, value_dim);
  int required_unroll = (max_dim > 64) ? 4 : ((max_dim > 32) ? 2 : 1);
  int unroll = required_unroll;
  auto gen = d.get_architecture_gen();
  if (gen >= 14 && max_dim >= 64) {
    unroll = 4;
  } else if (gen >= 13 && max_dim >= 64) {
    unroll = std::max(unroll, 2);
  }
  int force_unroll = getenv_int("MLX_TQ_FUSED_ATTN_UNROLL", 0);
  if ((force_unroll == 1 || force_unroll == 2 || force_unroll == 4) &&
      force_unroll >= required_unroll) {
    unroll = force_unroll;
  }

  std::string kname = "turboquant_decode_attention_batched_u1";
  if (unroll >= 4) {
    kname = "turboquant_decode_attention_batched_u4";
  } else if (unroll == 2) {
    kname = "turboquant_decode_attention_batched_u2";
  }

  auto& enc = metal::get_command_encoder(s);
  auto kernel = d.get_kernel(kname);
  enc.set_compute_pipeline_state(kernel);

  int cidx = 0;
  enc.set_input_array(q, cidx++);
  enc.set_input_array(kp, cidx++);
  enc.set_input_array(kn, cidx++);
  enc.set_input_array(vp, cidx++);
  enc.set_input_array(vn, cidx++);
  enc.set_input_array(c, cidx++);
  enc.set_output_array(out, cidx++);
  enc.set_bytes(batch, cidx++);
  enc.set_bytes(n_q_heads, cidx++);
  enc.set_bytes(n_kv_heads, cidx++);
  enc.set_bytes(static_cast<uint32_t>(n_repeats_), cidx++);
  enc.set_bytes(n_q_len, cidx++);
  enc.set_bytes(n_keys, cidx++);
  enc.set_bytes(q_dim, cidx++);
  enc.set_bytes(value_dim, cidx++);
  enc.set_bytes(k_words_per_key, cidx++);
  enc.set_bytes(v_words_per_key, cidx++);
  enc.set_bytes(vals_per_word, cidx++);
  enc.set_bytes(static_cast<uint32_t>(bits_), cidx++);
  enc.set_bytes(mask, cidx++);

  MTL::Size group_dims(32, 1, 1);
  MTL::Size grid_dims(32, n_q_heads * n_q_len, batch);
  check_kernel_threadgroup_size(kernel, group_dims, kname);
  enc.dispatch_threads(grid_dims, group_dims);

  enc.add_temporaries(std::move(copies));
}

bool FastTurboQuantDecodeAttentionBatched::is_equivalent(
    const Primitive& other) const {
  const FastTurboQuantDecodeAttentionBatched& a_other =
      static_cast<const FastTurboQuantDecodeAttentionBatched&>(other);
  return bits_ == a_other.bits_ && n_repeats_ == a_other.n_repeats_ &&
         value_dim_ == a_other.value_dim_;
}

} // namespace mlx::core::fast
