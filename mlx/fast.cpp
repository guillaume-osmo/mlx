// Copyright © 2023-2024 Apple Inc.
#include <cassert>
#include <cmath>
#include <numeric>

#include "mlx/fast.h"
#include "mlx/fast_primitives.h"
#include "mlx/ops.h"
#include "mlx/transforms.h"
#include "mlx/transforms_impl.h"

namespace mlx::core::fast {

std::vector<array> Custom::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {
  auto [_, vjps] = mlx::core::vjp(fallback_, primals, cotangents);
  std::vector<array> vjp_outs;
  for (int i = 0, j = 0; i < vjps.size(); ++i) {
    if (j < argnums.size() && i == argnums[j]) {
      vjp_outs.push_back(vjps[i]);
      j++;
    }
  }
  return vjp_outs;
}

std::vector<array> Custom::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  std::vector<array> all_tangents;
  for (int i = 0, j = 0; i < primals.size(); i++) {
    if (j < argnums.size() && i == argnums[j]) {
      all_tangents.emplace_back(tangents[j++]);
    } else {
      all_tangents.emplace_back(zeros_like(primals[i]));
    }
  }
  auto [_, jvps] = mlx::core::jvp(fallback_, primals, all_tangents);
  return jvps;
}

std::pair<std::vector<array>, std::vector<int>> Custom::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto outputs = mlx::core::vmap(fallback_, axes)(inputs);
  auto out_axes = std::vector<int>(outputs.size(), 0);
  return {outputs, out_axes};
}

array rms_norm(
    const array& x,
    const std::optional<array>& weight,
    float eps,
    StreamOrDevice s_ /* = {} */) {
  bool has_weight = weight.has_value();

  if (x.ndim() == 0) {
    std::ostringstream msg;
    msg << "[rms_norm] Input must have at least 1 dimension but got input with "
           "0 dimensions.";
    throw std::invalid_argument(msg.str());
  }
  if (has_weight) {
    if ((*weight).ndim() != 1) {
      std::ostringstream msg;
      msg << "[rms_norm] (*weight) must have 1 dimension but has "
          << (*weight).ndim() << " dimensions.";
      throw std::invalid_argument(msg.str());
    }
    if ((*weight).size() != x.shape(-1)) {
      std::ostringstream msg;
      msg << "[rms_norm] (*weight) must have the same size as the last dimension of"
             " x but has "
          << (*weight).size() << " elements.";
      throw std::invalid_argument(msg.str());
    }
  }

  auto out_type = (weight.has_value()) ? result_type(x, (*weight)) : x.dtype();
  if (!issubdtype(out_type, floating)) {
    std::ostringstream msg;
    msg << "[rms_norm] Received unsupported type " << out_type << ".";
    throw std::invalid_argument(msg.str());
  }

  auto s = to_stream(s_);
  auto fallback =
      [has_weight, eps, out_type, s](const std::vector<array>& inputs) {
        auto x = astype(inputs[0], float32, s);
        x = multiply(
            x,
            rsqrt(
                add(mean(square(x, s), -1, /* keepdims */ true, s),
                    array(eps, float32),
                    s),
                s),
            s);
        x = astype(x, out_type, s);

        if (has_weight) {
          x = multiply(x, inputs[1], s);
        }

        return std::vector<array>{x};
      };

  auto passed_weight =
      (has_weight) ? astype(*weight, out_type, s) : array(1, out_type);

  if (!RMSNorm::use_fallback(s)) {
    return array(
        x.shape(),
        out_type,
        std::make_shared<RMSNorm>(s, fallback, eps),
        {astype(x, out_type, s), passed_weight});
  }
  return fallback({x, passed_weight})[0];
}

std::vector<array> RMSNorm::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {
  assert(primals.size() == 2);
  assert(outputs.size() == 1);
  assert(cotangents.size() == 1);

  auto s = stream();
  auto fallback = [eps = eps_, s](const std::vector<array>& inputs) {
    auto& x = inputs[0];
    auto& w = inputs[1];
    auto& g = inputs[2];

    std::vector<array> vjps;

    auto n = rsqrt(
        add(mean(square(x, s), /* axis= */ -1, /* keepdims= */ true, s),
            array(eps, x.dtype()),
            s),
        s);
    auto n3 = power(n, array(3, x.dtype()), s);

    // df/dx
    auto gw = multiply(g, w, s);
    auto t = mean(multiply(gw, x, s), /* axis= */ -1, /* keepdims= */ true, s);
    t = multiply(multiply(x, t, s), n3, s);
    vjps.push_back(subtract(multiply(gw, n, s), t, s));

    // df/dw
    std::vector<int> axes(g.ndim() - 1);
    std::iota(axes.begin(), axes.end(), 0);
    if (w.ndim() == 0) {
      vjps.push_back(zeros_like(w, s));
    } else {
      vjps.push_back(sum(
          multiply(g, multiply(x, n, s), s), axes, /* keepdims= */ false, s));
    }

    return vjps;
  };

  auto vjps = array::make_arrays(
      {primals[0].shape(), primals[1].shape()},
      {primals[0].dtype(), primals[1].dtype()},
      std::make_shared<RMSNormVJP>(s, fallback, eps_),
      {primals[0], primals[1], cotangents[0]});

  std::vector<array> returned_vjps;
  for (auto& arg : argnums) {
    returned_vjps.push_back(std::move(vjps[arg]));
  }

  return returned_vjps;
}

bool RMSNorm::is_equivalent(const Primitive& other) const {
  const RMSNorm& a_other = static_cast<const RMSNorm&>(other);
  return eps_ == a_other.eps_;
}

bool RMSNormVJP::is_equivalent(const Primitive& other) const {
  const RMSNormVJP& a_other = static_cast<const RMSNormVJP&>(other);
  return eps_ == a_other.eps_;
}

array layer_norm(
    const array& x,
    const std::optional<array>& weight,
    const std::optional<array>& bias,
    float eps,
    StreamOrDevice s_ /* = {} */) {
  bool has_weight = weight.has_value();
  bool has_bias = bias.has_value();

  if (x.ndim() == 0) {
    std::ostringstream msg;
    msg << "[layer_norm] Input must have at least 1 dimension but got input with "
           "0 dimensions.";
    throw std::invalid_argument(msg.str());
  }
  if (has_weight) {
    if ((*weight).ndim() != 1) {
      std::ostringstream msg;
      msg << "[layer_norm] weight must have 1 dimension but has "
          << (*weight).ndim() << " dimensions.";
      throw std::invalid_argument(msg.str());
    }
    if ((*weight).size() != x.shape(-1)) {
      std::ostringstream msg;
      msg << "[layer_norm] weight must have the same size as the last dimension of"
             " x but has "
          << (*weight).size() << " elements.";
      throw std::invalid_argument(msg.str());
    }
  }
  if (has_bias) {
    if ((*bias).ndim() != 1) {
      std::ostringstream msg;
      msg << "[layer_norm] bias must have 1 dimension but has "
          << (*bias).ndim() << " dimensions.";
      throw std::invalid_argument(msg.str());
    }
    if ((*bias).size() != x.shape(-1)) {
      std::ostringstream msg;
      msg << "[layer_norm] bias must have the same size as the last dimension of"
             " x but has "
          << (*bias).size() << " elements.";
      throw std::invalid_argument(msg.str());
    }
  }

  auto out_type = (has_weight)
      ? ((has_bias) ? result_type(x, *weight, *bias) : result_type(x, *weight))
      : x.dtype();
  if (!issubdtype(out_type, floating)) {
    std::ostringstream msg;
    msg << "[layer_norm] Received unsupported type " << out_type << ".";
    throw std::invalid_argument(msg.str());
  }

  auto s = to_stream(s_);
  auto fallback = [has_weight, has_bias, eps, out_type, s](
                      const std::vector<array>& inputs) {
    auto x = astype(inputs[0], float32, s);

    auto mu = mean(x, /* axis= */ -1, /* keepdims= */ true, s);
    auto xc = subtract(x, mu, s);
    auto v = mean(square(xc, s), /* axis= */ -1, /* keepdims= */ true, s);

    x = multiply(xc, rsqrt(add(v, array(eps, float32), s), s));
    x = astype(x, out_type, s);

    // If the LN is affine then transform x according to the weight and bias
    if (has_weight) {
      x = multiply(x, inputs[1], s);
    }
    if (has_bias) {
      x = add(x, inputs[2], s);
    }

    return std::vector<array>{x};
  };

  auto passed_weight =
      (has_weight) ? astype(*weight, out_type, s) : array(1, out_type);
  auto passed_bias =
      (has_bias) ? astype(*bias, out_type, s) : array(0, out_type);

  if (!LayerNorm::use_fallback(s)) {
    return array(
        x.shape(),
        out_type,
        std::make_shared<LayerNorm>(s, fallback, eps),
        {astype(x, out_type, s), passed_weight, passed_bias});
  }
  return fallback({x, passed_weight, passed_bias})[0];
}

std::vector<array> LayerNorm::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {
  assert(primals.size() == 3);
  assert(outputs.size() == 1);
  assert(cotangents.size() == 1);

  auto s = stream();
  auto fallback = [eps = eps_, s](const std::vector<array>& inputs) {
    auto& x = inputs[0];
    auto& w = inputs[1];
    auto& b = inputs[2];
    auto& g = inputs[3];

    std::vector<array> vjps;

    auto norm = number_of_elements(x, {-1}, true, x.dtype(), s);
    auto sumx = sum(x, /* axis= */ -1, /* keepdims= */ true, s);
    auto sumx2 = sum(square(x, s), /* axis= */ -1, /* keepdims= */ true, s);
    auto mu = multiply(sumx, norm, s);
    auto mu2 = multiply(sumx2, norm, s);
    auto var = subtract(mu2, square(mu, s), s);
    auto n = rsqrt(add(var, array(eps, x.dtype()), s));
    auto n3 = power(n, array(3, x.dtype()), s);
    auto x_c = subtract(x, mu, s);

    // df/dx
    auto wg = multiply(w, g, s);
    auto sumwg =
        multiply(sum(wg, /* axis= */ -1, /* keepdims= */ true, s), norm, s);
    auto sumwgxc = multiply(
        sum(multiply(wg, x_c, s), /* axis= */ -1, /* keepdims= */ true, s),
        norm,
        s);
    auto t1 = multiply(multiply(x_c, sumwgxc, s), n3, s);
    auto t2 = multiply(subtract(wg, sumwg, s), n, s);
    vjps.push_back(subtract(t2, t1, s));

    // df/dw
    std::vector<int> axes(g.ndim() - 1);
    std::iota(axes.begin(), axes.end(), 0);
    if (w.ndim() == 0) {
      vjps.push_back(zeros_like(w, s));
    } else {
      vjps.push_back(sum(
          multiply(g, multiply(x_c, n, s), s), axes, /* keepdims= */ false, s));
    }

    // df/db
    if (b.ndim() == 0) {
      vjps.push_back(zeros_like(w, s));
    } else {
      vjps.push_back(sum(g, axes, /* keepdims= */ false, s));
    }

    return vjps;
  };

  auto vjps = array::make_arrays(
      {primals[0].shape(), primals[1].shape(), primals[2].shape()},
      {primals[0].dtype(), primals[1].dtype(), primals[2].dtype()},
      std::make_shared<LayerNormVJP>(s, fallback, eps_),
      {primals[0], primals[1], primals[2], cotangents[0]});

  std::vector<array> returned_vjps;
  for (auto& arg : argnums) {
    returned_vjps.push_back(std::move(vjps[arg]));
  }

  return returned_vjps;
}

bool LayerNorm::is_equivalent(const Primitive& other) const {
  const LayerNorm& a_other = static_cast<const LayerNorm&>(other);
  return eps_ == a_other.eps_;
}

bool LayerNormVJP::is_equivalent(const Primitive& other) const {
  const LayerNormVJP& a_other = static_cast<const LayerNormVJP&>(other);
  return eps_ == a_other.eps_;
}

array rope(
    std::vector<array> inputs,
    int dims,
    bool traditional,
    float base,
    float scale,
    bool forward,
    StreamOrDevice s) {
  auto& x = inputs[0];
  auto& offset = inputs[1];
  if (x.ndim() < 3) {
    std::ostringstream msg;
    msg << "[rope] Input must have at least 3 dimensions but got input with "
        << x.ndim() << " dimensions.";
    throw std::invalid_argument(msg.str());
  }
  if (!issubdtype(x.dtype(), floating)) {
    std::ostringstream msg;
    msg << "[rope] Input must be a floating type but got " << x.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (offset.ndim() > 1) {
    std::ostringstream msg;
    msg << "[rope] offset must have at most one dimension but has shape "
        << offset.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (offset.size() != 1 && offset.size() != x.shape(0)) {
    std::ostringstream msg;
    msg << "[rope] offset must be a scalar or vector with " << x.shape(0)
        << " elements but has shape " << offset.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (!issubdtype(offset.dtype(), integer)) {
    std::ostringstream msg;
    msg << "[rope] offset must be an integer but got type " << offset.dtype()
        << ".";
    throw std::invalid_argument(msg.str());
  }
  if (offset.dtype().size() != 4) {
    inputs[1] = astype(offset, int32, s);
  }
  if (inputs.size() == 3 &&
      (inputs[2].ndim() != 1 || inputs[2].shape(0) != dims / 2)) {
    std::ostringstream msg;
    msg << "[rope] freqs must be one dimensional with size " << dims / 2
        << " but got shape " << inputs[2].shape() << ".";
    throw std::invalid_argument(msg.str());
  }

  auto fallback = [dims, traditional, base, scale, forward, s](
                      std::vector<array> inputs) {
    auto x = inputs[0];
    auto shape = x.shape();
    if (x.ndim() == 3) {
      x = expand_dims(x, 1, s);
    } else if (x.ndim() > 4) {
      x = flatten(x, 1, 1 + (x.ndim() - 4), s);
    }

    auto B = x.shape(0);
    auto N = x.shape(1);
    auto T = x.shape(2);
    auto t = x.dtype();
    // Compute sines and cosines
    auto half_dims = dims / 2;
    auto offset = inputs[1];
    if (offset.size() > 1) {
      offset = expand_dims(offset, {-1, -2}, s);
    }
    auto positions = multiply(
        add(arange(x.shape(2), float32, s), offset, s),
        array(scale, float32),
        s);

    auto default_inv_freqs = [&s, base, half_dims]() {
      return exp(
          multiply(
              arange(0, -half_dims, -1, float32, s),
              array(std::log(base) / half_dims, float32),
              s),
          s);
    };

    auto inv_freqs =
        inputs.size() == 3 ? reciprocal(inputs[2], s) : default_inv_freqs();
    auto theta = multiply(expand_dims(positions, -1, s), inv_freqs, s);
    auto coss = astype(cos(theta, s), t, s);
    auto sins = astype(sin(theta, s), t, s);

    auto apply_rope = [forward, s](
                          const array& x1,
                          const array& x2,
                          const array& coss,
                          const array& sins) {
      std::vector<array> outs;
      if (forward) {
        outs.push_back(
            subtract(multiply(x1, coss, s), multiply(x2, sins, s), s));
        outs.push_back(add(multiply(x1, sins, s), multiply(x2, coss, s), s));
      } else {
        outs.push_back(add(multiply(x2, sins, s), multiply(x1, coss, s), s));
        outs.push_back(
            subtract(multiply(x2, coss, s), multiply(x1, sins, s), s));
      }
      return outs;
    };

    if (traditional) {
      auto x1 = slice(x, {0, 0, 0, 0}, {B, N, T, dims}, {1, 1, 1, 2}, s);
      auto x2 = slice(x, {0, 0, 0, 1}, {B, N, T, dims}, {1, 1, 1, 2}, s);
      auto outs = apply_rope(x1, x2, coss, sins);
      for (auto& o : outs) {
        o = expand_dims(o, -1, s);
      }
      auto out = reshape(concatenate(outs, -1, s), {B, N, T, dims}, s);
      if (dims < x.shape(-1)) {
        out =
            concatenate({out, slice(x, {0, 0, 0, dims}, x.shape(), s)}, -1, s);
      }
      return std::vector<array>{reshape(out, shape, s)};
    } else {
      auto out_s = x.shape();
      out_s.back() = half_dims;
      auto x1 = slice(x, {0, 0, 0, 0}, out_s, s);
      out_s.back() = dims;
      auto x2 = slice(x, {0, 0, 0, half_dims}, out_s, s);

      auto outs = apply_rope(x1, x2, coss, sins);
      if (dims < x.shape(-1)) {
        outs.push_back(slice(x, {0, 0, 0, dims}, x.shape(), s));
      }
      return std::vector<array>{reshape(concatenate(outs, -1, s), shape, s)};
    }
  };
  auto stream = to_stream(s);
  if (!RoPE::use_fallback(stream)) {
    return array(
        x.shape(),
        x.dtype(),
        std::make_shared<RoPE>(
            stream, fallback, dims, traditional, base, scale, forward),
        std::move(inputs));
  }
  return fallback(std::move(inputs))[0];
}

array rope(
    const array& x,
    int dims,
    bool traditional,
    std::optional<float> base,
    float scale,
    const array& offset,
    const std::optional<array>& freqs /* = std::nullopt */,
    StreamOrDevice s /* = {} */) {
  std::vector<array> inputs = {x, offset};
  if (freqs) {
    inputs.push_back(astype(*freqs, float32, s));
    if (base) {
      throw std::invalid_argument(
          "[rope] Only one of base or freqs can have a value.");
    }
  } else if (!base) {
    throw std::invalid_argument("[rope] Neither base nor freqs has a value.");
  }
  return rope(
      std::move(inputs),
      dims,
      traditional,
      base.has_value() ? *base : 1.0,
      scale,
      true,
      s);
}

array rope(
    const array& x,
    int dims,
    bool traditional,
    std::optional<float> base,
    float scale,
    int offset,
    const std::optional<array>& freqs /* = std::nullopt */,
    StreamOrDevice s /* = {} */) {
  return rope(
      x, dims, traditional, base, scale, array(offset, int32), freqs, s);
}

std::vector<array> RoPE::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {
  auto s = stream();
  auto fallback = [dims = dims_,
                   traditional = traditional_,
                   base = base_,
                   scale = scale_,
                   forward = forward_,
                   s](std::vector<array> inputs) {
    return std::vector<array>{
        rope(std::move(inputs), dims, traditional, base, scale, !forward, s)};
  };
  if (argnums.size() > 1 || argnums[0] != 0) {
    throw std::invalid_argument(
        "[RoPE::vjp] vjp for offset or frequencies not supported");
  }
  auto inputs = std::vector<array>{cotangents[0], primals[1]};
  if (primals.size() == 3) {
    inputs.push_back(primals[2]);
  }
  return {array(
      cotangents[0].shape(),
      cotangents[0].dtype(),
      std::make_shared<RoPE>(
          s, fallback, dims_, traditional_, base_, scale_, !forward_),
      std::move(inputs))};
}

bool RoPE::is_equivalent(const Primitive& other) const {
  const RoPE& a_other = static_cast<const RoPE&>(other);
  return (
      dims_ == a_other.dims_ && base_ == a_other.base_ &&
      scale_ == a_other.scale_ && traditional_ == a_other.traditional_ &&
      forward_ == a_other.forward_);
}

/** Computes: O = softmax(Q @ K.T) @ V **/
array scaled_dot_product_attention(
    const array& queries,
    const array& keys,
    const array& values,
    const float scale,
    const std::string& mask_mode /* = "" */,
    std::optional<array> mask_arr /* = {} */,
    const std::optional<array>& sinks /* = {} */,
    StreamOrDevice s /* = {}*/) {
  for (const auto& tensor : {queries, keys, values}) {
    if (tensor.ndim() != 4) {
      std::ostringstream msg;
      msg << "[scaled_dot_product_attention] input with shape "
          << tensor.shape() << " expected to be rank 4";
      throw std::invalid_argument(msg.str());
    }
  }
  // Check valid mask
  if (mask_mode != "" && mask_mode != "causal" && mask_mode != "array") {
    std::ostringstream msg;
    msg << "[scaled_dot_product_attention] Invalid mask_mode " << mask_mode
        << ". mask_mode must be 'causal', 'array' or ''.";
    throw std::invalid_argument(msg.str());
  }

  bool do_causal = false;
  bool has_mask = false;
  bool has_arr_mask = false;
  bool has_bool_mask = false;

  if (mask_mode == "causal") {
    has_mask = true;
    do_causal = true;

    if (mask_arr) {
      std::ostringstream msg;
      msg << "[scaled_dot_product_attention] Invalid mask_arr for mask_mode "
          << "'casusal'. No array mask should be passed.";
      throw std::invalid_argument(msg.str());
    }
  } else if (mask_arr) {
    has_mask = true;
    has_arr_mask = true;
    has_bool_mask = mask_arr->dtype() == bool_;
  }

  if (has_arr_mask && mask_arr->ndim() > 4) {
    std::ostringstream msg;
    msg << "[scaled_dot_product_attention] the mask with shape "
        << mask_arr->shape() << " expected to have at most rank 4.";
    throw std::invalid_argument(msg.str());
  }

  const size_t batch_dim = queries.shape(0);
  for (const auto& tensor : {keys, values}) {
    if (tensor.shape(0) != batch_dim) {
      std::ostringstream msg;
      msg << "[scaled_dot_product_attention] mismatching batch dimension for input with shape "
          << tensor.shape() << ".";
      throw std::invalid_argument(msg.str());
    }
  }

  // Q, K must have matching last dims (d_k aka 'head_dim');
  if (queries.shape(-1) != keys.shape(-1)) {
    std::ostringstream msg;
    msg << "[scaled_dot_product_attention] query, keys expected to have matching last dimension; found query shape "
        << queries.shape() << " for keys shape " << keys.shape() << ".";
    throw std::invalid_argument(msg.str());
  }

  // K, V must have matching number of heads (n_kv_heads);
  auto n_q_heads = queries.shape(-3);
  auto n_kv_heads = keys.shape(-3);

  if (keys.shape(-3) != values.shape(-3)) {
    std::ostringstream msg;
    msg << "[scaled_dot_product_attention] keys, values expected to have matching n_kv_heads; found keys with n_heads "
        << keys.shape(-3) << " for values with n_heads " << values.shape(-3)
        << ".";
    throw std::invalid_argument(msg.str());
  }

  // n_heads % n_kv_heads == 0; n_heads >= 1, n_kv_heads >= 1.
  if (n_q_heads % n_kv_heads != 0) {
    std::ostringstream msg;
    msg << "[scaled_dot_product_attention] n_heads must be a multiple of n_kv_heads, found n_heads "
        << n_q_heads << " for n_kv_heads " << n_kv_heads << ".";
    throw std::invalid_argument(msg.str());
  }

  auto final_type = result_type(queries, keys, values);
  if (!issubdtype(final_type, floating)) {
    std::ostringstream msg;
    msg << "[scaled_dot_product_attention] Received unsupported type "
        << final_type << ".";
    throw std::invalid_argument(msg.str());
  }
  bool has_sinks = sinks.has_value();

  auto q = astype(queries, final_type, s);
  auto k = astype(keys, final_type, s);
  auto v = astype(values, final_type, s);

  auto fallback = [scale,
                   n_q_heads,
                   n_kv_heads,
                   do_causal,
                   has_sinks,
                   has_arr_mask,
                   s](const std::vector<array>& inputs) {
    auto q = multiply(array(scale, inputs[0].dtype()), inputs[0], s);
    int n_repeats = n_q_heads / n_kv_heads;
    auto k = inputs[1];
    auto v = inputs[2];
    if (n_repeats > 1) {
      q = unflatten(q, 1, {n_kv_heads, n_repeats}, s);
      k = expand_dims(k, 2, s);
      v = expand_dims(v, 2, s);
    }
    auto scores = matmul(q, swapaxes(k, -1, -2, s), s);
    if (has_arr_mask || do_causal) {
      // Mask must be broadcast-compatible with [B, n_q_heads, L_q, L_kv]
      auto make_or_fetch_mask = [&]() {
        if (do_causal) {
          int kL = k.shape(-2);
          int qL = q.shape(-2);
          int q_off = (kL - qL) < 0 ? 0 : (kL - qL);
          auto q_idx = arange(q_off, q_off + qL, s);
          auto k_idx = arange(0, kL, s);
          q_idx = expand_dims(q_idx, 1, s);
          k_idx = expand_dims(k_idx, 0, s);
          return greater_equal(q_idx, k_idx, s);
        }
        return inputs[3];
      };
      auto mask = make_or_fetch_mask();

      if (n_repeats > 1 && mask.ndim() >= 3) {
        if (mask.shape(-3) == 1) {
          mask = expand_dims(mask, -3, s);
        } else {
          mask = unflatten(mask, -3, {n_kv_heads, n_repeats}, s);
        }
      }
      if (mask.dtype() == bool_) {
        scores = where(
            mask, scores, array(finfo(scores.dtype()).min, scores.dtype()), s);
      } else {
        scores = add(scores, mask, s);
      }
    }
    if (has_sinks) {
      auto sinks = inputs.back();
      // scores has shape B N_q N_k L_q L_k
      sinks = expand_dims(sinks, {0, 2, 3}, s);
      if (scores.ndim() == 5) {
        sinks = unflatten(sinks, 1, {n_kv_heads, n_repeats}, s);
      }
      auto bsx_shape = scores.shape();
      bsx_shape.back() = 1;
      scores = concatenate({broadcast_to(sinks, bsx_shape, s), scores}, -1, s);
    }
    scores = softmax(scores, std::vector<int>{-1}, true, s);
    if (has_sinks) {
      // Slice off scores
      auto start = Shape(scores.ndim(), 0);
      start.back() = 1;
      auto stop = scores.shape();
      scores = slice(scores, std::move(start), std::move(stop), s);
    }
    auto out = matmul(scores, v, s);
    if (n_repeats > 1) {
      out = flatten(out, 1, 2, s);
    }
    return std::vector<array>{out};
  };

  auto stream = to_stream(s);
  std::vector<array> inputs = {q, k, v};
  if (has_arr_mask) {
    // Check type
    has_bool_mask = mask_arr->dtype() == bool_;
    if (promote_types(mask_arr->dtype(), final_type) != final_type) {
      std::ostringstream msg;
      msg << "[scaled_dot_product_attention] Mask type must promote to output type "
          << final_type << ".";
      throw std::invalid_argument(msg.str());
    } else if (!has_bool_mask) {
      mask_arr = astype(*mask_arr, final_type, stream);
    }
    // Broadcast mask
    auto mask_shape = queries.shape();
    mask_shape.back() = keys.shape(-2);
    inputs.push_back(broadcast_to(*mask_arr, mask_shape, stream));
  }
  if (has_sinks) {
    if (promote_types(sinks->dtype(), final_type) != final_type) {
      std::ostringstream msg;
      msg << "[scaled_dot_product_attention] Type of sinks must promote to output type "
          << final_type << ".";
      throw std::invalid_argument(msg.str());
    }
    if (sinks->ndim() != 1 || sinks->shape(0) != n_q_heads) {
      std::ostringstream msg;
      msg << "[scaled_dot_product_attention] Received invalid shape for sinks "
          << sinks->shape() << ".";
      throw std::invalid_argument(msg.str());
    }
    inputs.push_back(astype(*sinks, final_type, stream));
  }

  bool is_training = detail::in_grad_tracing();
  bool has_fast_vjp = !ScaledDotProductAttentionVJP::use_fallback(q, stream);
  bool output_logsumexp = is_training && has_fast_vjp;
  if (!ScaledDotProductAttention::use_fallback(
          q,
          k,
          v,
          has_mask,
          has_arr_mask,
          do_causal,
          is_training,
          output_logsumexp,
          stream)) {
    if (has_bool_mask && !ScaledDotProductAttention::supports_bool_mask()) {
      // Convert bool mask to additive mask.
      float inf = std::numeric_limits<float>::infinity();
      array& mask = inputs[3];
      mask = where(
          mask,
          full_like(mask, 0, final_type, s),
          full_like(mask, -inf, final_type, s));
    }
    Shape out_shape{q.shape(0), q.shape(1), q.shape(2), v.shape(-1)};
    auto primitive = std::make_shared<ScaledDotProductAttention>(
        stream, fallback, scale, do_causal, has_sinks, output_logsumexp);
    if (output_logsumexp) {
      return array::make_arrays(
          {std::move(out_shape), Shape{q.shape(0), q.shape(1), q.shape(2), 1}},
          {final_type, float32},
          primitive,
          std::move(inputs))[0];
    } else {
      return array(
          std::move(out_shape), final_type, primitive, std::move(inputs));
    }
  }
  return fallback(std::move(inputs))[0];
}

array turboquant_qk_packed_scores(
    const array& q_rot,
    const array& k_packed,
    const array& k_norms,
    const array& centroids,
    int bits,
    StreamOrDevice s_) {
  auto s = to_stream(s_);
  if (bits != 2 && bits != 3 && bits != 4) {
    throw std::invalid_argument(
        "[turboquant_qk_packed_scores] bits must be one of {2, 3, 4}.");
  }
  if (q_rot.ndim() != 2 || k_packed.ndim() != 2 || k_norms.ndim() != 1 ||
      centroids.ndim() != 1) {
    throw std::invalid_argument(
        "[turboquant_qk_packed_scores] shapes must be q_rot [Q,D], k_packed [K,W], k_norms [K], centroids [L].");
  }

  int n_queries = q_rot.shape(0);
  int dim = q_rot.shape(1);
  int n_keys = k_packed.shape(0);
  if (k_norms.shape(0) != n_keys) {
    throw std::invalid_argument(
        "[turboquant_qk_packed_scores] k_norms length must match k_packed first dimension.");
  }
  int vals_per_word = 32 / bits;
  int expected_words = (dim + vals_per_word - 1) / vals_per_word;
  if (k_packed.shape(1) != expected_words) {
    throw std::invalid_argument(
        "[turboquant_qk_packed_scores] k_packed last dim does not match packed width for q_rot.shape(-1).");
  }
  if (centroids.shape(0) != (1 << bits)) {
    throw std::invalid_argument(
        "[turboquant_qk_packed_scores] centroids length must equal 2**bits.");
  }

  auto fallback = [s, bits](const std::vector<array>& inputs) {
    const array& q = inputs[0];
    const array& kp = inputs[1];
    const array& norms = inputs[2];
    const array& c = inputs[3];

    int d = q.shape(1);
    int nk = kp.shape(0);
    int vals_per_word = 32 / bits;
    uint32_t mask = (1u << bits) - 1u;

    auto d_idx = arange(d, uint32, s);
    auto words = floor_divide(d_idx, array(vals_per_word, uint32), s);
    auto shifts =
        multiply(remainder(d_idx, array(vals_per_word, uint32), s),
                 array(bits, uint32),
                 s);
    auto words_2d = broadcast_to(reshape(words, {1, d}, s), {nk, d}, s);
    auto shifts_2d = broadcast_to(reshape(shifts, {1, d}, s), {nk, d}, s);
    auto packed_words = take_along_axis(kp, words_2d, -1, s);
    auto idx =
        bitwise_and(right_shift(packed_words, shifts_2d, s), array(mask, uint32), s);
    auto z = take(c, idx, 0, s);
    auto k_deq = multiply(z, expand_dims(norms, -1, s), s);
    return std::vector<array>{matmul(q, swapaxes(k_deq, -1, -2, s), s)};
  };

  array q = astype(q_rot, float32, s);
  array kp = astype(k_packed, uint32, s);
  array norms = astype(k_norms, float32, s);
  array c = astype(centroids, float32, s);
  auto primitive = std::make_shared<FastTurboQuantQK>(s, std::move(fallback), bits);
  return array(
      {n_queries, n_keys},
      float32,
      primitive,
      std::vector<array>{q, kp, norms, c});
}

array turboquant_qk_packed_scores_batched(
    const array& q_rot,
    const array& k_packed,
    const array& k_norms,
    const array& centroids,
    int bits,
    int n_repeats,
    StreamOrDevice s_) {
  auto s = to_stream(s_);
  if (bits != 2 && bits != 3 && bits != 4) {
    throw std::invalid_argument(
        "[turboquant_qk_packed_scores_batched] bits must be one of {2, 3, 4}.");
  }
  if (n_repeats <= 0) {
    throw std::invalid_argument(
        "[turboquant_qk_packed_scores_batched] n_repeats must be > 0.");
  }
  if (q_rot.ndim() != 4 || k_packed.ndim() != 4 || k_norms.ndim() != 3 ||
      centroids.ndim() != 1) {
    throw std::invalid_argument(
        "[turboquant_qk_packed_scores_batched] shapes must be q_rot [B,Hq,L,D], k_packed [B,Hkv,T,W], k_norms [B,Hkv,T], centroids [C].");
  }

  int B = q_rot.shape(0);
  int Hq = q_rot.shape(1);
  int L = q_rot.shape(2);
  int D = q_rot.shape(3);
  int Bk = k_packed.shape(0);
  int Hkv = k_packed.shape(1);
  int T = k_packed.shape(2);

  if (Bk != B || k_norms.shape(0) != B || k_norms.shape(1) != Hkv ||
      k_norms.shape(2) != T) {
    throw std::invalid_argument(
        "[turboquant_qk_packed_scores_batched] batch/head/time dims must match between packed keys and norms.");
  }
  if (Hq != Hkv * n_repeats) {
    throw std::invalid_argument(
        "[turboquant_qk_packed_scores_batched] q heads must equal kv heads * n_repeats.");
  }

  int vals_per_word = 32 / bits;
  int expected_words = (D + vals_per_word - 1) / vals_per_word;
  if (k_packed.shape(3) != expected_words) {
    throw std::invalid_argument(
        "[turboquant_qk_packed_scores_batched] packed width does not match q_rot last dimension.");
  }
  if (centroids.shape(0) != (1 << bits)) {
    throw std::invalid_argument(
        "[turboquant_qk_packed_scores_batched] centroids length must equal 2**bits.");
  }

  auto fallback = [s, bits, n_repeats](const std::vector<array>& inputs) {
    const array& q = inputs[0];
    const array& kp = inputs[1];
    const array& norms = inputs[2];
    const array& c = inputs[3];

    int B = q.shape(0);
    int Hq = q.shape(1);
    int L = q.shape(2);
    int D = q.shape(3);
    int Hkv = kp.shape(1);
    int T = kp.shape(2);
    int vals_per_word = 32 / bits;
    uint32_t mask = (1u << bits) - 1u;

    auto d_idx = arange(D, uint32, s);
    auto words = floor_divide(d_idx, array(vals_per_word, uint32), s);
    auto shifts = multiply(
        remainder(d_idx, array(vals_per_word, uint32), s), array(bits, uint32), s);

    auto words4 = broadcast_to(
        reshape(words, {1, 1, 1, D}, s), {B, Hkv, T, D}, s);
    auto shifts4 = broadcast_to(
        reshape(shifts, {1, 1, 1, D}, s), {B, Hkv, T, D}, s);
    auto packed_words = take_along_axis(kp, words4, -1, s);
    auto idx =
        bitwise_and(right_shift(packed_words, shifts4, s), array(mask, uint32), s);
    auto z = take(c, idx, 0, s);
    auto k_deq = multiply(z, expand_dims(norms, -1, s), s); // [B,Hkv,T,D]

    auto qg = reshape(q, {B, Hkv, n_repeats, L, D}, s);
    auto kt = expand_dims(swapaxes(k_deq, -1, -2, s), 2, s); // [B,Hkv,1,D,T]
    auto out = matmul(qg, kt, s); // [B,Hkv,n_repeats,L,T]
    return std::vector<array>{reshape(out, {B, Hq, L, T}, s)};
  };

  array q = astype(q_rot, float32, s);
  array kp = astype(k_packed, uint32, s);
  array norms = astype(k_norms, float32, s);
  array c = astype(centroids, float32, s);
  auto primitive = std::make_shared<FastTurboQuantQKBatched>(
      s, std::move(fallback), bits, n_repeats);
  return array(
      {B, Hq, L, T}, float32, primitive, std::vector<array>{q, kp, norms, c});
}

array turboquant_qk_prod_scores_batched(
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
    StreamOrDevice s_) {
  auto s = to_stream(s_);
  if (bits != 2 && bits != 3 && bits != 4) {
    throw std::invalid_argument(
        "[turboquant_qk_prod_scores_batched] bits must be one of {2, 3, 4}.");
  }
  if (n_repeats <= 0) {
    throw std::invalid_argument(
        "[turboquant_qk_prod_scores_batched] n_repeats must be > 0.");
  }
  if (q_rot.ndim() != 4 || q_model.ndim() != 4 || k_packed.ndim() != 4 ||
      k_norms.ndim() != 3 || centroids.ndim() != 1 || qjl_packed.ndim() != 4 ||
      qjl_gamma.ndim() != 3 || qjl_projection.ndim() != 2) {
    throw std::invalid_argument(
        "[turboquant_qk_prod_scores_batched] shapes must be q_rot [B,Hq,L,D], q_model [B,Hq,L,D], k_packed [B,Hkv,T,Wk], k_norms [B,Hkv,T], centroids [C], qjl_packed [B,Hkv,T,W1], qjl_gamma [B,Hkv,T], qjl_projection [D,D].");
  }

  int B = q_rot.shape(0);
  int Hq = q_rot.shape(1);
  int L = q_rot.shape(2);
  int D = q_rot.shape(3);
  int Bk = k_packed.shape(0);
  int Hkv = k_packed.shape(1);
  int T = k_packed.shape(2);

  if (q_model.shape(0) != B || q_model.shape(1) != Hq || q_model.shape(2) != L ||
      q_model.shape(3) != D) {
    throw std::invalid_argument(
        "[turboquant_qk_prod_scores_batched] q_model shape must match q_rot shape.");
  }
  if (Bk != B || k_norms.shape(0) != B || k_norms.shape(1) != Hkv ||
      k_norms.shape(2) != T || qjl_packed.shape(0) != B ||
      qjl_packed.shape(1) != Hkv || qjl_packed.shape(2) != T ||
      qjl_gamma.shape(0) != B || qjl_gamma.shape(1) != Hkv ||
      qjl_gamma.shape(2) != T) {
    throw std::invalid_argument(
        "[turboquant_qk_prod_scores_batched] batch/head/time dims must match between packed keys, qjl state, and norms.");
  }
  if (Hq != Hkv * n_repeats) {
    throw std::invalid_argument(
        "[turboquant_qk_prod_scores_batched] q heads must equal kv heads * n_repeats.");
  }

  int vals_per_word = 32 / bits;
  int expected_words = (D + vals_per_word - 1) / vals_per_word;
  if (k_packed.shape(3) != expected_words) {
    throw std::invalid_argument(
        "[turboquant_qk_prod_scores_batched] packed key width does not match q_rot last dimension.");
  }
  int expected_qjl_words = (D + 31) / 32;
  if (qjl_packed.shape(3) != expected_qjl_words) {
    throw std::invalid_argument(
        "[turboquant_qk_prod_scores_batched] packed qjl width does not match q_rot last dimension.");
  }
  if (centroids.shape(0) != (1 << bits)) {
    throw std::invalid_argument(
        "[turboquant_qk_prod_scores_batched] centroids length must equal 2**bits.");
  }
  if (qjl_projection.shape(0) != D || qjl_projection.shape(1) != D) {
    throw std::invalid_argument(
        "[turboquant_qk_prod_scores_batched] qjl_projection must have shape [D,D].");
  }

  auto mse_scores =
      turboquant_qk_packed_scores_batched(q_rot, k_packed, k_norms, centroids, bits, n_repeats, s);

  array q = astype(q_model, float32, s);
  array qp = astype(qjl_packed, uint32, s);
  array gamma = astype(qjl_gamma, float32, s);
  array norms = astype(k_norms, float32, s);
  array proj = astype(qjl_projection, float32, s);

  auto q_proj = matmul(q, swapaxes(proj, -1, -2, s), s); // [B,Hq,L,D]

  auto d_idx = arange(D, uint32, s);
  auto words = floor_divide(d_idx, array(32, uint32), s);
  auto shifts = remainder(d_idx, array(32, uint32), s);

  auto words4 =
      broadcast_to(reshape(words, {1, 1, 1, D}, s), {B, Hkv, T, D}, s);
  auto shifts4 =
      broadcast_to(reshape(shifts, {1, 1, 1, D}, s), {B, Hkv, T, D}, s);
  auto packed_words = take_along_axis(qp, words4, -1, s);
  auto idx =
      bitwise_and(right_shift(packed_words, shifts4, s), array(uint32_t(1), uint32), s);
  auto signs = subtract(multiply(astype(idx, float32, s), array(2.0f, float32), s),
                        array(1.0f, float32),
                        s); // [B,Hkv,T,D] in {-1,+1}

  auto qg = reshape(q_proj, {B, Hkv, n_repeats, L, D}, s);
  auto st = expand_dims(swapaxes(signs, -1, -2, s), 2, s); // [B,Hkv,1,D,T]
  auto corr_scores = matmul(qg, st, s); // [B,Hkv,n_repeats,L,T]

  auto alpha = array(float(std::sqrt(M_PI / 2.0) / D), float32);
  auto corr_scale =
      multiply(multiply(norms, gamma, s), alpha, s); // [B,Hkv,T]
  corr_scores = multiply(
      corr_scores, expand_dims(expand_dims(corr_scale, 2, s), 2, s), s);

  auto corr_out = reshape(corr_scores, {B, Hq, L, T}, s);
  return add(mse_scores, corr_out, s);
}

array turboquant_av_packed_values_batched(
    const array& probs,
    const array& v_packed,
    const array& v_norms,
    const array& centroids,
    int bits,
    int n_repeats,
    int value_dim,
    StreamOrDevice s_) {
  auto s = to_stream(s_);
  if (bits != 2 && bits != 3 && bits != 4) {
    throw std::invalid_argument(
        "[turboquant_av_packed_values_batched] bits must be one of {2, 3, 4}.");
  }
  if (n_repeats <= 0) {
    throw std::invalid_argument(
        "[turboquant_av_packed_values_batched] n_repeats must be > 0.");
  }
  if (probs.ndim() != 4 || v_packed.ndim() != 4 || v_norms.ndim() != 3 ||
      centroids.ndim() != 1) {
    throw std::invalid_argument(
        "[turboquant_av_packed_values_batched] shapes must be probs [B,Hq,L,T], v_packed [B,Hkv,T,W], v_norms [B,Hkv,T], centroids [C].");
  }

  int B = probs.shape(0);
  int Hq = probs.shape(1);
  int L = probs.shape(2);
  int T = probs.shape(3);
  int Bv = v_packed.shape(0);
  int Hkv = v_packed.shape(1);
  int Tv = v_packed.shape(2);

  if (Bv != B || Tv != T || v_norms.shape(0) != B || v_norms.shape(1) != Hkv ||
      v_norms.shape(2) != T) {
    throw std::invalid_argument(
        "[turboquant_av_packed_values_batched] batch/head/time dims must match between packed values and norms/probs.");
  }
  if (Hq != Hkv * n_repeats) {
    throw std::invalid_argument(
        "[turboquant_av_packed_values_batched] q heads must equal kv heads * n_repeats.");
  }

  int vals_per_word = 32 / bits;
  int W = v_packed.shape(3);
  int expected_words = (value_dim + vals_per_word - 1) / vals_per_word;
  if (W != expected_words) {
    throw std::invalid_argument(
        "[turboquant_av_packed_values_batched] packed width does not match provided value_dim.");
  }
  if (centroids.shape(0) != (1 << bits)) {
    throw std::invalid_argument(
        "[turboquant_av_packed_values_batched] centroids length must equal 2**bits.");
  }

  auto fallback = [s, bits, n_repeats, value_dim](const std::vector<array>& inputs) {
    const array& p = inputs[0];
    const array& vp = inputs[1];
    const array& norms = inputs[2];
    const array& c = inputs[3];

    int B = p.shape(0);
    int Hq = p.shape(1);
    int L = p.shape(2);
    int T = p.shape(3);
    int Hkv = vp.shape(1);
    int vals_per_word = 32 / bits;
    int D = value_dim;
    uint32_t mask = (1u << bits) - 1u;

    auto d_idx = arange(D, uint32, s);
    auto words = floor_divide(d_idx, array(vals_per_word, uint32), s);
    auto shifts = multiply(
        remainder(d_idx, array(vals_per_word, uint32), s), array(bits, uint32), s);

    auto words4 = broadcast_to(
        reshape(words, {1, 1, 1, D}, s), {B, Hkv, T, D}, s);
    auto shifts4 = broadcast_to(
        reshape(shifts, {1, 1, 1, D}, s), {B, Hkv, T, D}, s);
    auto packed_words = take_along_axis(vp, words4, -1, s);
    auto idx =
        bitwise_and(right_shift(packed_words, shifts4, s), array(mask, uint32), s);
    auto z = take(c, idx, 0, s);
    auto v_deq = multiply(z, expand_dims(norms, -1, s), s); // [B,Hkv,T,D]

    auto pg = reshape(p, {B, Hkv, n_repeats, L, T}, s);
    auto vv = expand_dims(v_deq, 2, s); // [B,Hkv,1,T,D]
    auto out = matmul(pg, vv, s); // [B,Hkv,n_repeats,L,D]
    return std::vector<array>{reshape(out, {B, Hq, L, D}, s)};
  };

  array p = astype(probs, float32, s);
  array vp = astype(v_packed, uint32, s);
  array norms = astype(v_norms, float32, s);
  array c = astype(centroids, float32, s);
  auto primitive = std::make_shared<FastTurboQuantAVBatched>(
      s, std::move(fallback), bits, n_repeats, value_dim);
  return array(
      {B, Hq, L, value_dim},
      float32,
      primitive,
      std::vector<array>{p, vp, norms, c});
}

array turboquant_decode_attention_packed_batched(
    const array& q_rot,
    const array& k_packed,
    const array& k_norms,
    const array& v_packed,
    const array& v_norms,
    const array& centroids,
    int bits,
    int n_repeats,
    int value_dim,
    StreamOrDevice s_) {
  auto s = to_stream(s_);
  if (bits != 2 && bits != 3 && bits != 4) {
    throw std::invalid_argument(
        "[turboquant_decode_attention_packed_batched] bits must be one of {2, 3, 4}.");
  }
  if (n_repeats <= 0) {
    throw std::invalid_argument(
        "[turboquant_decode_attention_packed_batched] n_repeats must be > 0.");
  }
  if (q_rot.ndim() != 4 || k_packed.ndim() != 4 || k_norms.ndim() != 3 ||
      v_packed.ndim() != 4 || v_norms.ndim() != 3 || centroids.ndim() != 1) {
    throw std::invalid_argument(
        "[turboquant_decode_attention_packed_batched] shapes must be q_rot [B,Hq,L,D], k_packed [B,Hkv,T,Wk], k_norms [B,Hkv,T], v_packed [B,Hkv,T,Wv], v_norms [B,Hkv,T], centroids [C].");
  }

  int B = q_rot.shape(0);
  int Hq = q_rot.shape(1);
  int L = q_rot.shape(2);
  int Dq = q_rot.shape(3);
  int Bk = k_packed.shape(0);
  int Hkv = k_packed.shape(1);
  int T = k_packed.shape(2);

  if (Bk != B || k_norms.shape(0) != B || k_norms.shape(1) != Hkv ||
      k_norms.shape(2) != T || v_packed.shape(0) != B ||
      v_packed.shape(1) != Hkv || v_packed.shape(2) != T ||
      v_norms.shape(0) != B || v_norms.shape(1) != Hkv ||
      v_norms.shape(2) != T) {
    throw std::invalid_argument(
        "[turboquant_decode_attention_packed_batched] batch/head/time dims must match between packed K/V tensors and norms.");
  }
  if (Hq != Hkv * n_repeats) {
    throw std::invalid_argument(
        "[turboquant_decode_attention_packed_batched] q heads must equal kv heads * n_repeats.");
  }

  int vals_per_word = 32 / bits;
  int expected_k_words = (Dq + vals_per_word - 1) / vals_per_word;
  int expected_v_words = (value_dim + vals_per_word - 1) / vals_per_word;
  if (k_packed.shape(3) != expected_k_words) {
    throw std::invalid_argument(
        "[turboquant_decode_attention_packed_batched] packed key width does not match q_rot last dimension.");
  }
  if (v_packed.shape(3) != expected_v_words) {
    throw std::invalid_argument(
        "[turboquant_decode_attention_packed_batched] packed value width does not match provided value_dim.");
  }
  if (centroids.shape(0) != (1 << bits)) {
    throw std::invalid_argument(
        "[turboquant_decode_attention_packed_batched] centroids length must equal 2**bits.");
  }

  auto fallback =
      [s, bits, n_repeats, value_dim](const std::vector<array>& inputs) {
        const array& q = inputs[0];
        const array& kp = inputs[1];
        const array& kn = inputs[2];
        const array& vp = inputs[3];
        const array& vn = inputs[4];
        const array& c = inputs[5];

        int B = q.shape(0);
        int Hq = q.shape(1);
        int L = q.shape(2);
        int Dq = q.shape(3);
        int Hkv = kp.shape(1);
        int T = kp.shape(2);
        int Dv = value_dim;
        int vals_per_word = 32 / bits;
        uint32_t mask = (1u << bits) - 1u;

        auto q_idx = arange(Dq, uint32, s);
        auto q_words = floor_divide(q_idx, array(vals_per_word, uint32), s);
        auto q_shifts = multiply(
            remainder(q_idx, array(vals_per_word, uint32), s),
            array(bits, uint32),
            s);
        auto q_words4 = broadcast_to(
            reshape(q_words, {1, 1, 1, Dq}, s), {B, Hkv, T, Dq}, s);
        auto q_shifts4 = broadcast_to(
            reshape(q_shifts, {1, 1, 1, Dq}, s), {B, Hkv, T, Dq}, s);
        auto q_packed_words = take_along_axis(kp, q_words4, -1, s);
        auto q_idx_unpack = bitwise_and(
            right_shift(q_packed_words, q_shifts4, s), array(mask, uint32), s);
        auto q_z = take(c, q_idx_unpack, 0, s);
        auto k_deq = multiply(q_z, expand_dims(kn, -1, s), s);

        auto v_idx = arange(Dv, uint32, s);
        auto v_words = floor_divide(v_idx, array(vals_per_word, uint32), s);
        auto v_shifts = multiply(
            remainder(v_idx, array(vals_per_word, uint32), s),
            array(bits, uint32),
            s);
        auto v_words4 = broadcast_to(
            reshape(v_words, {1, 1, 1, Dv}, s), {B, Hkv, T, Dv}, s);
        auto v_shifts4 = broadcast_to(
            reshape(v_shifts, {1, 1, 1, Dv}, s), {B, Hkv, T, Dv}, s);
        auto v_packed_words = take_along_axis(vp, v_words4, -1, s);
        auto v_idx_unpack = bitwise_and(
            right_shift(v_packed_words, v_shifts4, s), array(mask, uint32), s);
        auto v_z = take(c, v_idx_unpack, 0, s);
        auto v_deq = multiply(v_z, expand_dims(vn, -1, s), s);

        auto qg = reshape(q, {B, Hkv, n_repeats, L, Dq}, s);
        auto kt = expand_dims(swapaxes(k_deq, -1, -2, s), 2, s);
        auto scores = matmul(qg, kt, s);
        scores = reshape(scores, {B, Hq, L, T}, s);
        scores = softmax(scores, std::vector<int>{-1}, true, s);

        auto pg = reshape(scores, {B, Hkv, n_repeats, L, T}, s);
        auto vv = expand_dims(v_deq, 2, s);
        auto out = matmul(pg, vv, s);
        return std::vector<array>{reshape(out, {B, Hq, L, Dv}, s)};
      };

  array q = astype(q_rot, float32, s);
  array kp = astype(k_packed, uint32, s);
  array kn = astype(k_norms, float32, s);
  array vp = astype(v_packed, uint32, s);
  array vn = astype(v_norms, float32, s);
  array c = astype(centroids, float32, s);
  auto primitive = std::make_shared<FastTurboQuantDecodeAttentionBatched>(
      s, std::move(fallback), bits, n_repeats, value_dim);
  return array(
      {B, Hq, L, value_dim},
      float32,
      primitive,
      std::vector<array>{q, kp, kn, vp, vn, c});
}

std::vector<array> ScaledDotProductAttention::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {
  assert(primals.size() >= 3);
  assert(cotangents.size() == outputs.size());

  auto s = stream();
  if (ScaledDotProductAttentionVJP::use_fallback(primals[0], s)) {
    assert(outputs.size() == 1);
    return Custom::vjp(primals, cotangents, argnums, outputs);
  }

  auto fallback = [sdpa = fallback_, s](const std::vector<array>& inputs) {
    std::vector<array> primals(inputs.begin(), std::prev(inputs.end()));
    auto [_, vjps] = mlx::core::vjp(sdpa, primals, {inputs.back()});
    return vjps;
  };

  std::vector<Shape> shapes;
  std::vector<Dtype> dtypes;
  for (int i = 0; i < /* outputs size */ 3; ++i) {
    shapes.push_back(primals[i].shape());
    dtypes.push_back(primals[i].dtype());
  }
  auto primitive = std::make_shared<ScaledDotProductAttentionVJP>(
      s, fallback, scale_, do_causal_, has_sinks_);
  std::vector<array> inputs = primals;
  inputs.push_back(outputs[0]);
  inputs.push_back(outputs[1]);
  inputs.push_back(cotangents[0]);
  auto vjps = array::make_arrays(std::move(shapes), dtypes, primitive, inputs);

  std::vector<array> returned_vjps;
  for (int arg : argnums) {
    if (arg >= 3) {
      throw std::invalid_argument(
          "[scale_dot_product_attention] Does not support VJP with respect "
          " to mask or attention sinks.");
    }
    returned_vjps.push_back(std::move(vjps[arg]));
  }
  return returned_vjps;
}

bool ScaledDotProductAttention::is_equivalent(const Primitive& other) const {
  const ScaledDotProductAttention& a_other =
      static_cast<const ScaledDotProductAttention&>(other);
  return scale_ == a_other.scale_ && do_causal_ == a_other.do_causal_ &&
      has_sinks_ == a_other.has_sinks_ &&
      output_logsumexp_ == a_other.output_logsumexp_;
}

bool ScaledDotProductAttentionVJP::is_equivalent(const Primitive& other) const {
  const ScaledDotProductAttentionVJP& a_other =
      static_cast<const ScaledDotProductAttentionVJP&>(other);
  return scale_ == a_other.scale_ && do_causal_ == a_other.do_causal_ &&
      has_sinks_ == a_other.has_sinks_;
}

bool Quantize::is_equivalent(const Primitive& other) const {
  const Quantize& p_other = static_cast<const Quantize&>(other);
  return (
      p_other.group_size_ == group_size_ && p_other.bits_ == bits_ &&
      p_other.mode_ == mode_ && p_other.dequantize_ == dequantize_);
}

std::vector<Shape> Quantize::output_shapes(const std::vector<array>& inputs) {
  auto& w = inputs[0];
  if (dequantize_) {
    auto out_size = w.shape(-1) * 32 / bits_;
    auto out_shape = w.shape();
    out_shape.back() = out_size;
    return {std::move(out_shape)};
  } else {
    auto wq_shape = w.shape();
    wq_shape.back() = w.shape(-1) * bits_ / 32;
    auto sshape = w.shape();
    sshape.back() = w.shape(-1) / group_size_;
    if (inputs.size() == 2) {
      return {std::move(wq_shape), std::move(sshape)};
    } else {
      auto bshape = sshape;
      return {std::move(wq_shape), std::move(sshape), std::move(bshape)};
    }
  }
}

bool ConvertFP8::is_equivalent(const Primitive& other) const {
  const ConvertFP8& a_other = static_cast<const ConvertFP8&>(other);
  return to_fp8_ == a_other.to_fp8_;
}

} // namespace mlx::core::fast
