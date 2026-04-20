// Copyright © 2026 Apple Inc.

#include <algorithm>
#include <string>

#include "mlx/allocator.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/jit/includes.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"
#include "mlx/utils.h"

namespace mlx::core::fast {

namespace {

const char* relu2_kernel_source() {
  return R"metal(
#include <metal_common>

using namespace metal;

template <typename T>
[[kernel]] void relu2_kernel(
    const device T* x,
    device T* out,
    constant uint& size,
    uint gid [[thread_position_in_grid]]) {
  if (gid >= size) {
    return;
  }
  float v = static_cast<float>(x[gid]);
  float r = v > 0.0f ? v : 0.0f;
  out[gid] = static_cast<T>(r * r);
}

#define instantiate_relu2(name, type) \
  template [[host_name("relu2_" #name)]] [[kernel]] decltype(relu2_kernel<type>) relu2_kernel<type>;

instantiate_relu2(float32, float)
instantiate_relu2(float16, half)
instantiate_relu2(bfloat16, bfloat16_t)
)metal";
}

bool is_supported_relu2_dtype(Dtype t) {
  return t == float32 || t == float16 || t == bfloat16;
}

} // namespace

void ReluSquared::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);
  const array& x = inputs[0];
  array& out = outputs[0];

  if (!x.flags().contiguous || !is_supported_relu2_dtype(x.dtype())) {
    outputs = fallback_(inputs);
    return;
  }

  if (x.is_donatable()) {
    out.copy_shared_buffer(x);
  } else {
    out.set_data(
        allocator::malloc(x.data_size() * x.itemsize()),
        x.data_size(),
        x.strides(),
        x.flags());
  }

  auto lib = d.get_library("relu2_lib", []() {
    std::string kernel_source = metal::utils();
    kernel_source += relu2_kernel_source();
    return kernel_source;
  });
  auto base_name = std::string("relu2_") + type_to_name(out);
  auto kernel = d.get_kernel(base_name, lib);

  auto& enc = d.get_command_encoder(s.index);
  enc.set_compute_pipeline_state(kernel);
  enc.set_input_array(x, 0);
  enc.set_output_array(out, 1);
  uint32_t size = static_cast<uint32_t>(x.data_size());
  enc.set_bytes(size, 2);

  uint32_t threads_per_group = std::min<uint32_t>(
      256u,
      static_cast<uint32_t>(kernel->maxTotalThreadsPerThreadgroup()));
  enc.dispatch_threads(MTL::Size(size, 1, 1), MTL::Size(threads_per_group, 1, 1));
}

} // namespace mlx::core::fast
