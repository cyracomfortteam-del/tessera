// Tiled FlashAttention forward in raw CUDA C++ (teaching reference).
//
// Block layout: one CUDA block computes one (batch, head, query-tile) of BLOCK_M queries.
// K/V are streamed in BLOCK_N tiles staged through shared memory; each query row keeps its
// running max (m) and denominator (l) in registers and rescales its accumulator on the fly
// (online softmax). This is the same algorithm as the Triton kernel — kept here to show the
// shared-memory staging, coalesced global loads, and avoidance of the full T x T score
// matrix at the C++ level.
//
// The Triton kernel in ../triton/flash_attention.py is the path actually wired into the
// model; this file documents the lower-level memory choreography and is buildable with:
//   nvcc -O3 -arch=sm_80 flash_attention.cu -c
// Profile occupancy / shared-memory pressure with:
//   ncu --set full --kernel-name flash_attn_fwd_kernel ./a.out
//
#include <cuda_runtime.h>
#include <math.h>

namespace {

template <int HEAD_DIM, int BLOCK_M, int BLOCK_N>
__global__ void flash_attn_fwd_kernel(const float* __restrict__ Q,  // [B,H,T,D]
                                      const float* __restrict__ K,  // [B,Hkv,T,D]
                                      const float* __restrict__ V,  // [B,Hkv,T,D]
                                      float* __restrict__ O,        // [B,H,T,D]
                                      int B, int H, int Hkv, int T,
                                      float scale, bool causal) {
  const int q_tile = blockIdx.x;          // which BLOCK_M of queries
  const int bh = blockIdx.y;              // flattened (batch, head)
  const int b = bh / H;
  const int h = bh % H;
  const int h_kv = h / (H / Hkv);         // grouped-query attention mapping

  const int row = threadIdx.x;            // one thread == one query row in the tile
  const int q_idx = q_tile * BLOCK_M + row;

  const long q_off = (((long)b * H + h) * T + q_idx) * HEAD_DIM;
  const long kv_base = ((long)b * Hkv + h_kv) * T * HEAD_DIM;

  // Load this thread's query row into registers.
  float q[HEAD_DIM];
  if (q_idx < T) {
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) q[d] = Q[q_off + d];
  }

  float acc[HEAD_DIM];
#pragma unroll
  for (int d = 0; d < HEAD_DIM; ++d) acc[d] = 0.0f;
  float m_i = -INFINITY;
  float l_i = 0.0f;

  __shared__ float k_tile[BLOCK_N][HEAD_DIM];
  __shared__ float v_tile[BLOCK_N][HEAD_DIM];

  const int n_end = causal ? min(T, (q_tile + 1) * BLOCK_M) : T;
  for (int start_n = 0; start_n < n_end; start_n += BLOCK_N) {
    // Cooperatively stage K/V tiles into shared memory (coalesced: consecutive threads
    // read consecutive addresses along the feature dim).
    for (int idx = threadIdx.x; idx < BLOCK_N * HEAD_DIM; idx += blockDim.x) {
      const int n = idx / HEAD_DIM;
      const int d = idx % HEAD_DIM;
      const int key = start_n + n;
      const float kv_valid = key < T ? 1.0f : 0.0f;
      k_tile[n][d] = kv_valid ? K[kv_base + (long)key * HEAD_DIM + d] : 0.0f;
      v_tile[n][d] = kv_valid ? V[kv_base + (long)key * HEAD_DIM + d] : 0.0f;
    }
    __syncthreads();

    if (q_idx < T) {
#pragma unroll 1
      for (int n = 0; n < BLOCK_N; ++n) {
        const int key = start_n + n;
        if (key >= T) break;
        if (causal && key > q_idx) break;

        float s = 0.0f;
#pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) s += q[d] * k_tile[n][d];
        s *= scale;

        const float m_new = fmaxf(m_i, s);
        const float alpha = __expf(m_i - m_new);
        const float p = __expf(s - m_new);
        l_i = l_i * alpha + p;
#pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) acc[d] = acc[d] * alpha + p * v_tile[n][d];
        m_i = m_new;
      }
    }
    __syncthreads();
  }

  if (q_idx < T) {
    const float inv_l = 1.0f / l_i;
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) O[q_off + d] = acc[d] * inv_l;
  }
}

}  // namespace

// Explicit instantiation for the common head dim used in the demos.
template __global__ void flash_attn_fwd_kernel<64, 64, 64>(
    const float*, const float*, const float*, float*, int, int, int, int, float, bool);
