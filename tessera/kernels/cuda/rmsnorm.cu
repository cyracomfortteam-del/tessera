// Fused RMSNorm forward in raw CUDA C++.
//
// One block per row (token). Threads stride across the feature dim with vectorized
// float4 loads (16-byte, fully coalesced transactions), accumulate the partial sum of
// squares in registers, then a two-stage reduction (warp shuffle -> shared memory ->
// warp shuffle) produces the row RMS. A second vectorized pass applies the gain.
//
// Build as a torch extension:
//   torch.utils.cpp_extension.load(name="tessera_rmsnorm", sources=["rmsnorm.cu"])
// Profile:
//   ncu --set full --kernel-name rmsnorm_fwd_kernel ./a.out   # check memory throughput
//
#include <torch/extension.h>
#include <cuda_runtime.h>

namespace {

constexpr int kWarp = 32;

__inline__ __device__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffff, v, offset);
  }
  return v;
}

__inline__ __device__ float block_reduce_sum(float v) {
  __shared__ float shared[kWarp];  // one slot per warp
  const int lane = threadIdx.x % kWarp;
  const int wid = threadIdx.x / kWarp;

  v = warp_reduce_sum(v);
  if (lane == 0) shared[wid] = v;
  __syncthreads();

  const int n_warps = (blockDim.x + kWarp - 1) / kWarp;
  v = (threadIdx.x < n_warps) ? shared[lane] : 0.0f;
  if (wid == 0) v = warp_reduce_sum(v);
  return v;
}

// x, y: [n_rows, n_cols] row-major. weight: [n_cols].
__global__ void rmsnorm_fwd_kernel(const float* __restrict__ x,
                                   const float* __restrict__ weight,
                                   float* __restrict__ y,
                                   int n_cols,
                                   float eps) {
  const int row = blockIdx.x;
  const float* x_row = x + static_cast<long>(row) * n_cols;
  float* y_row = y + static_cast<long>(row) * n_cols;

  // Vectorized accumulation of sum of squares (assumes n_cols % 4 == 0; a scalar tail
  // loop would handle the remainder in the general case).
  float local = 0.0f;
  const float4* x4 = reinterpret_cast<const float4*>(x_row);
  const int n4 = n_cols / 4;
  for (int i = threadIdx.x; i < n4; i += blockDim.x) {
    float4 v = x4[i];
    local += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
  }

  float sum_sq = block_reduce_sum(local);
  __shared__ float r_rms;
  if (threadIdx.x == 0) r_rms = rsqrtf(sum_sq / n_cols + eps);
  __syncthreads();
  const float scale = r_rms;

  const float4* w4 = reinterpret_cast<const float4*>(weight);
  float4* y4 = reinterpret_cast<float4*>(y_row);
  for (int i = threadIdx.x; i < n4; i += blockDim.x) {
    float4 v = x4[i];
    float4 w = w4[i];
    float4 o;
    o.x = v.x * scale * w.x;
    o.y = v.y * scale * w.y;
    o.z = v.z * scale * w.z;
    o.w = v.w * scale * w.w;
    y4[i] = o;
  }
}

}  // namespace

torch::Tensor rmsnorm_forward(torch::Tensor x, torch::Tensor weight, double eps) {
  TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32, "this reference kernel is fp32");
  auto x2d = x.reshape({-1, x.size(-1)}).contiguous();
  const int rows = x2d.size(0);
  const int cols = x2d.size(1);
  auto y = torch::empty_like(x2d);

  const int threads = 256;
  rmsnorm_fwd_kernel<<<rows, threads>>>(
      x2d.data_ptr<float>(), weight.data_ptr<float>(), y.data_ptr<float>(), cols,
      static_cast<float>(eps));
  return y.reshape(x.sizes());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rmsnorm_forward", &rmsnorm_forward, "Fused RMSNorm forward (CUDA)");
}
