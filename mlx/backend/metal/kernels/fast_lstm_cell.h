// Copyright © 2024 Apple Inc.
// Fused LSTM cell for RNN on Metal. Same pattern as fast_gru_cell.

#include <metal_stdlib>
#include <metal_math>

#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

inline float fast_sigmoid(float x) {
  float y = 1.0f / (1.0f + metal::fast::exp(-metal::abs(x)));
  return (x < 0.0f) ? 1.0f - y : y;
}

inline float4 fast_sigmoid4(float4 x) {
  return float4(fast_sigmoid(x.x), fast_sigmoid(x.y), fast_sigmoid(x.z), fast_sigmoid(x.w));
}

inline float stable_sigmoid(float x) {
  if (x >= 0.0f) {
    return 1.0f / (1.0f + metal::precise::exp(-x));
  }
  float ex = metal::precise::exp(x);
  return ex / (1.0f + ex);
}

inline float stable_tanh(float x) {
  return metal::precise::tanh(x);
}

inline float lstm_sigmoid(float x, constant uint& use_precise_math) {
  return use_precise_math ? stable_sigmoid(x) : fast_sigmoid(x);
}

inline float lstm_tanh(float x, constant uint& use_precise_math) {
  return use_precise_math ? stable_tanh(x) : metal::fast::tanh(x);
}

inline float4 lstm_sigmoid4(float4 x, constant uint& use_precise_math) {
  return float4(
      lstm_sigmoid(x.x, use_precise_math),
      lstm_sigmoid(x.y, use_precise_math),
      lstm_sigmoid(x.z, use_precise_math),
      lstm_sigmoid(x.w, use_precise_math));
}

inline float4 lstm_tanh4(float4 x, constant uint& use_precise_math) {
  return float4(
      lstm_tanh(x.x, use_precise_math),
      lstm_tanh(x.y, use_precise_math),
      lstm_tanh(x.z, use_precise_math),
      lstm_tanh(x.w, use_precise_math));
}

// LSTM: i,f,g,o = gates; cell_new = f*cell_prev + i*g; hidden_new = o*tanh(cell_new)
[[host_name("lstm_cell_fused_float")]] [[kernel]] void lstm_cell_fused_float(
    const device float* input_proj,   // [B, 4H]
    const device float* hidden_proj, // [B, 4H]
    const device float* cell_prev,   // [B, H]
    device float* output_cell,       // [B, H]
    device float* output_hidden,     // [B, H]
    constant uint& batch_size,
    constant uint& hidden_size,
    constant uint& use_precise_math, // 0 = fast, 1 = precise
    uint idx [[thread_position_in_grid]]) {

  uint stride_4h = 4u * hidden_size;
  uint h_quads = (hidden_size + 3u) / 4u;
  uint total_quads = batch_size * h_quads;
  if (idx >= total_quads) return;

  uint batch_idx = idx / h_quads;
  uint h_base = (idx % h_quads) * 4u;
  uint base = batch_idx * stride_4h + h_base;
  uint prev_base = batch_idx * hidden_size + h_base;

  if (h_base + 4u <= hidden_size) {
    float4 i4 = lstm_sigmoid4(*reinterpret_cast<const device float4*>(input_proj + base) +
                                  *reinterpret_cast<const device float4*>(hidden_proj + base),
                              use_precise_math);
    float4 f4 = lstm_sigmoid4(*reinterpret_cast<const device float4*>(input_proj + base + hidden_size) +
                                  *reinterpret_cast<const device float4*>(hidden_proj + base + hidden_size),
                              use_precise_math);
    float4 g4 = lstm_tanh4(*reinterpret_cast<const device float4*>(input_proj + base + 2u * hidden_size) +
                               *reinterpret_cast<const device float4*>(hidden_proj + base + 2u * hidden_size),
                           use_precise_math);
    float4 o4 = lstm_sigmoid4(*reinterpret_cast<const device float4*>(input_proj + base + 3u * hidden_size) +
                                  *reinterpret_cast<const device float4*>(hidden_proj + base + 3u * hidden_size),
                              use_precise_math);
    float4 c_prev4 = *reinterpret_cast<const device float4*>(cell_prev + prev_base);

    float4 c_new4 = f4 * c_prev4 + i4 * g4;
    *reinterpret_cast<device float4*>(output_cell + prev_base) = c_new4;
    *reinterpret_cast<device float4*>(output_hidden + prev_base) =
        o4 * lstm_tanh4(c_new4, use_precise_math);
    return;
  }

  for (uint i = 0u; i < 4u && (h_base + i) < hidden_size; i++) {
    uint b = base + i;
    uint pb = prev_base + i;
    float i_g = lstm_sigmoid(input_proj[b] + hidden_proj[b], use_precise_math);
    float f_g = lstm_sigmoid(
        input_proj[b + hidden_size] + hidden_proj[b + hidden_size],
        use_precise_math);
    float g_g = lstm_tanh(
        input_proj[b + 2u * hidden_size] + hidden_proj[b + 2u * hidden_size],
        use_precise_math);
    float o_g = lstm_sigmoid(
        input_proj[b + 3u * hidden_size] + hidden_proj[b + 3u * hidden_size],
        use_precise_math);
    float c_prev = cell_prev[pb];
    float c_new = f_g * c_prev + i_g * g_g;
    output_cell[pb] = c_new;
    output_hidden[pb] = o_g * lstm_tanh(c_new, use_precise_math);
  }
}

// BFloat16 path
[[host_name("lstm_cell_fused_bfloat16")]] [[kernel]] void lstm_cell_fused_bfloat16(
    const device bfloat16_t* input_proj,
    const device bfloat16_t* hidden_proj,
    const device bfloat16_t* cell_prev,
    device bfloat16_t* output_cell,
    device bfloat16_t* output_hidden,
    constant uint& batch_size,
    constant uint& hidden_size,
    constant uint& use_precise_math,
    uint idx [[thread_position_in_grid]]) {

  uint stride_4h = 4u * hidden_size;
  uint h_quads = (hidden_size + 3u) / 4u;
  uint total_quads = batch_size * h_quads;
  if (idx >= total_quads) return;

  uint batch_idx = idx / h_quads;
  uint h_base = (idx % h_quads) * 4u;
  uint base = batch_idx * stride_4h + h_base;
  uint prev_base = batch_idx * hidden_size + h_base;

  if (h_base + 4u <= hidden_size) {
    float4 i4 = lstm_sigmoid4(
        float4(input_proj[base], input_proj[base + 1], input_proj[base + 2], input_proj[base + 3]) +
            float4(hidden_proj[base], hidden_proj[base + 1], hidden_proj[base + 2], hidden_proj[base + 3]),
        use_precise_math);
    float4 f4 = lstm_sigmoid4(
        float4(
            input_proj[base + hidden_size],
            input_proj[base + hidden_size + 1],
            input_proj[base + hidden_size + 2],
            input_proj[base + hidden_size + 3]) +
            float4(
                hidden_proj[base + hidden_size],
                hidden_proj[base + hidden_size + 1],
                hidden_proj[base + hidden_size + 2],
                hidden_proj[base + hidden_size + 3]),
        use_precise_math);
    float4 g4 = lstm_tanh4(
        float4(
            input_proj[base + 2u * hidden_size],
            input_proj[base + 2u * hidden_size + 1],
            input_proj[base + 2u * hidden_size + 2],
            input_proj[base + 2u * hidden_size + 3]) +
            float4(
                hidden_proj[base + 2u * hidden_size],
                hidden_proj[base + 2u * hidden_size + 1],
                hidden_proj[base + 2u * hidden_size + 2],
                hidden_proj[base + 2u * hidden_size + 3]),
        use_precise_math);
    float4 o4 = lstm_sigmoid4(
        float4(
            input_proj[base + 3u * hidden_size],
            input_proj[base + 3u * hidden_size + 1],
            input_proj[base + 3u * hidden_size + 2],
            input_proj[base + 3u * hidden_size + 3]) +
            float4(
                hidden_proj[base + 3u * hidden_size],
                hidden_proj[base + 3u * hidden_size + 1],
                hidden_proj[base + 3u * hidden_size + 2],
                hidden_proj[base + 3u * hidden_size + 3]),
        use_precise_math);
    float4 c_prev4 = float4(cell_prev[prev_base], cell_prev[prev_base+1], cell_prev[prev_base+2], cell_prev[prev_base+3]);

    float4 c_new4 = f4 * c_prev4 + i4 * g4;
    float4 h_new4 = o4 * lstm_tanh4(c_new4, use_precise_math);
    output_cell[prev_base] = bfloat16_t(c_new4.x); output_cell[prev_base+1] = bfloat16_t(c_new4.y);
    output_cell[prev_base+2] = bfloat16_t(c_new4.z); output_cell[prev_base+3] = bfloat16_t(c_new4.w);
    output_hidden[prev_base] = bfloat16_t(h_new4.x); output_hidden[prev_base+1] = bfloat16_t(h_new4.y);
    output_hidden[prev_base+2] = bfloat16_t(h_new4.z); output_hidden[prev_base+3] = bfloat16_t(h_new4.w);
    return;
  }

  for (uint i = 0u; i < 4u && (h_base + i) < hidden_size; i++) {
    uint b = base + i;
    uint pb = prev_base + i;
    float i_g = lstm_sigmoid(float(input_proj[b]) + float(hidden_proj[b]), use_precise_math);
    float f_g = lstm_sigmoid(
        float(input_proj[b + hidden_size]) + float(hidden_proj[b + hidden_size]),
        use_precise_math);
    float g_g = lstm_tanh(
        float(input_proj[b + 2u * hidden_size]) + float(hidden_proj[b + 2u * hidden_size]),
        use_precise_math);
    float o_g = lstm_sigmoid(
        float(input_proj[b + 3u * hidden_size]) + float(hidden_proj[b + 3u * hidden_size]),
        use_precise_math);
    float c_prev = float(cell_prev[pb]);
    float c_new = f_g * c_prev + i_g * g_g;
    output_cell[pb] = bfloat16_t(c_new);
    output_hidden[pb] = bfloat16_t(o_g * lstm_tanh(c_new, use_precise_math));
  }
}

[[host_name("lstm_cell_fused_float_vjp")]] [[kernel]] void
lstm_cell_fused_float_vjp(
    const device float* input_proj,   // [B, 4H]
    const device float* hidden_proj, // [B, 4H]
    const device float* cell_prev,   // [B, H]
    const device float* cot_cell,    // [B, H]
    const device float* cot_hidden,  // [B, H]
    device float* d_input_proj,      // [B, 4H]
    device float* d_hidden_proj,     // [B, 4H]
    device float* d_cell_prev,       // [B, H]
    constant uint& batch_size,
    constant uint& hidden_size,
    constant uint& use_precise_math,
    uint idx [[thread_position_in_grid]]) {
  uint stride_4h = 4u * hidden_size;
  uint h_quads = (hidden_size + 3u) / 4u;
  uint total_quads = batch_size * h_quads;
  if (idx >= total_quads) return;

  uint batch_idx = idx / h_quads;
  uint h_base = (idx % h_quads) * 4u;
  uint base = batch_idx * stride_4h + h_base;
  uint prev_base = batch_idx * hidden_size + h_base;

  if (h_base + 4u <= hidden_size) {
    float4 gi4 = *reinterpret_cast<const device float4*>(input_proj + base) +
        *reinterpret_cast<const device float4*>(hidden_proj + base);
    float4 gf4 =
        *reinterpret_cast<const device float4*>(input_proj + base + hidden_size) +
        *reinterpret_cast<const device float4*>(
            hidden_proj + base + hidden_size);
    float4 gg4 = *reinterpret_cast<const device float4*>(
                     input_proj + base + 2u * hidden_size) +
        *reinterpret_cast<const device float4*>(
            hidden_proj + base + 2u * hidden_size);
    float4 go4 = *reinterpret_cast<const device float4*>(
                     input_proj + base + 3u * hidden_size) +
        *reinterpret_cast<const device float4*>(
            hidden_proj + base + 3u * hidden_size);

    float4 i4 = lstm_sigmoid4(gi4, use_precise_math);
    float4 f4 = lstm_sigmoid4(gf4, use_precise_math);
    float4 g4 = lstm_tanh4(gg4, use_precise_math);
    float4 o4 = lstm_sigmoid4(go4, use_precise_math);

    float4 c_prev4 = *reinterpret_cast<const device float4*>(cell_prev + prev_base);
    float4 c_new4 = f4 * c_prev4 + i4 * g4;
    float4 tanh_c4 = lstm_tanh4(c_new4, use_precise_math);

    float4 cot_c4 = *reinterpret_cast<const device float4*>(cot_cell + prev_base);
    float4 cot_h4 = *reinterpret_cast<const device float4*>(cot_hidden + prev_base);

    float4 dc4 = cot_c4 + cot_h4 * o4 * (1.0f - tanh_c4 * tanh_c4);
    float4 do4 = cot_h4 * tanh_c4;
    float4 di4 = dc4 * g4;
    float4 df4 = dc4 * c_prev4;
    float4 dg4 = dc4 * i4;

    float4 d_i_gate4 = di4 * i4 * (1.0f - i4);
    float4 d_f_gate4 = df4 * f4 * (1.0f - f4);
    float4 d_g_gate4 = dg4 * (1.0f - g4 * g4);
    float4 d_o_gate4 = do4 * o4 * (1.0f - o4);

    *reinterpret_cast<device float4*>(d_input_proj + base) = d_i_gate4;
    *reinterpret_cast<device float4*>(d_input_proj + base + hidden_size) =
        d_f_gate4;
    *reinterpret_cast<device float4*>(d_input_proj + base + 2u * hidden_size) =
        d_g_gate4;
    *reinterpret_cast<device float4*>(d_input_proj + base + 3u * hidden_size) =
        d_o_gate4;

    *reinterpret_cast<device float4*>(d_hidden_proj + base) = d_i_gate4;
    *reinterpret_cast<device float4*>(d_hidden_proj + base + hidden_size) =
        d_f_gate4;
    *reinterpret_cast<device float4*>(d_hidden_proj + base + 2u * hidden_size) =
        d_g_gate4;
    *reinterpret_cast<device float4*>(d_hidden_proj + base + 3u * hidden_size) =
        d_o_gate4;

    *reinterpret_cast<device float4*>(d_cell_prev + prev_base) = dc4 * f4;
    return;
  }

  for (uint i = 0u; i < 4u && (h_base + i) < hidden_size; i++) {
    uint b = base + i;
    uint pb = prev_base + i;

    float gi = input_proj[b] + hidden_proj[b];
    float gf = input_proj[b + hidden_size] + hidden_proj[b + hidden_size];
    float gg = input_proj[b + 2u * hidden_size] + hidden_proj[b + 2u * hidden_size];
    float go = input_proj[b + 3u * hidden_size] + hidden_proj[b + 3u * hidden_size];

    float i_g = lstm_sigmoid(gi, use_precise_math);
    float f_g = lstm_sigmoid(gf, use_precise_math);
    float g_g = lstm_tanh(gg, use_precise_math);
    float o_g = lstm_sigmoid(go, use_precise_math);

    float c_prev_v = cell_prev[pb];
    float c_new = f_g * c_prev_v + i_g * g_g;
    float tanh_c = lstm_tanh(c_new, use_precise_math);

    float cot_c_v = cot_cell[pb];
    float cot_h_v = cot_hidden[pb];
    float dc = cot_c_v + cot_h_v * o_g * (1.0f - tanh_c * tanh_c);
    float do_v = cot_h_v * tanh_c;
    float di_v = dc * g_g;
    float df_v = dc * c_prev_v;
    float dg_v = dc * i_g;

    float d_i_gate = di_v * i_g * (1.0f - i_g);
    float d_f_gate = df_v * f_g * (1.0f - f_g);
    float d_g_gate = dg_v * (1.0f - g_g * g_g);
    float d_o_gate = do_v * o_g * (1.0f - o_g);

    d_input_proj[b] = d_i_gate;
    d_input_proj[b + hidden_size] = d_f_gate;
    d_input_proj[b + 2u * hidden_size] = d_g_gate;
    d_input_proj[b + 3u * hidden_size] = d_o_gate;

    d_hidden_proj[b] = d_i_gate;
    d_hidden_proj[b + hidden_size] = d_f_gate;
    d_hidden_proj[b + 2u * hidden_size] = d_g_gate;
    d_hidden_proj[b + 3u * hidden_size] = d_o_gate;

    d_cell_prev[pb] = dc * f_g;
  }
}
