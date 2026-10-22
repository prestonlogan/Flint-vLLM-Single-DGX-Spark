// NVFP4 W4A16 batch-1 GEMVs for the Flint attention block on sm_121 (GB10), raw ModelOpt layout (no Marlin repack):
//   W [N, K/2] u8 (low nibble = even k), S [N, K/16] e4m3, gs scalar fp32 (global scale).  y = bf16(acc_fp32 * gs).
//   qkv_gemv: fused (residual add +) RMSNorm of the hidden state, then 5120 x 2048 GEMV; writes the new residual.
//   o_gemv:   2048 x 4096 GEMV on the attention output.
// e2m1 is decoded with integer ops into an fp16 bit pattern (value * 2^-14, exact), 2^14 folded into the block scale.
// PDL: weights are prefetched into registers before griddepcontrol.wait; activations are read after it (pointers laundered
// so the compiler cannot hoist ld.nc above the wait). Kernels never trigger dependents early (their successors are plain launches).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

typedef __nv_bfloat16 bf16;
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
template <class T> __device__ __forceinline__ T* launder(T* p) { asm volatile("" : "+l"(p)::"memory"); return p; }

__device__ __forceinline__ float nib2f(uint32_t w, int j) {
  uint32_t t = w >> (4 * j);
  uint32_t p = ((t & 7u) << 9) | ((t & 8u) << 12);
  return __half2float(__ushort_as_half((unsigned short)p));
}
__device__ __forceinline__ float2 sc2(uint32_t two) {
  __half2_raw h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(two & 0xFFFFu), __NV_E4M3);
  __half2 hh = *reinterpret_cast<__half2*>(&h);
  float2 f = __half22float2(hh);
  return make_float2(f.x * 16384.f, f.y * 16384.f);
}
__device__ __forceinline__ void bf16x8_to_f32(uint4 u, float* f) {
  uint32_t a[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) { f[2 * i] = __uint_as_float(a[i] << 16); f[2 * i + 1] = __uint_as_float(a[i] & 0xFFFF0000u); }
}
__device__ __forceinline__ float dot32(uint4 q, const float* xf, float s0, float s1) {
  uint32_t w[4] = {q.x, q.y, q.z, q.w};
  float r = 0.f;
#pragma unroll
  for (int g = 0; g < 2; ++g) {
    float a = 0.f;
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int j = 0; j < 8; ++j) a = fmaf(nib2f(w[2 * g + i], j), xf[(2 * g + i) * 8 + j], a);
    r = fmaf(a, g == 0 ? s0 : s1, r);
  }
  return r;
}
__device__ __forceinline__ float rsqrt_ftz(float x) { float r; asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x)); return r; }

#define NWARP 8

// y[row] for R rows per warp; x in smem (bf16 [K])
template <int K, int R>
__device__ __forceinline__ void gemv_smem(const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, float gscale, const bf16* xs,
                                          uint4 (&q)[R][K / 1024], uint32_t (&sc)[R][K / 1024], int row0, int lane, bf16* __restrict__ y) {
  constexpr int CH = K / 1024;
  float acc[R];
#pragma unroll
  for (int r = 0; r < R; ++r) acc[r] = 0.f;
#pragma unroll
  for (int c = 0; c < CH; ++c) {
    float xf[32];
    const uint4* xp = (const uint4*)(xs + c * 1024 + lane * 32);
#pragma unroll
    for (int i = 0; i < 4; ++i) bf16x8_to_f32(xp[i], xf + 8 * i);
#pragma unroll
    for (int r = 0; r < R; ++r) {
      float2 a = sc2(sc[r][c]);
      acc[r] += dot32(q[r][c], xf, a.x, a.y);
    }
  }
#pragma unroll
  for (int r = 0; r < R; ++r) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], o);
  }
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < R; ++r) y[row0 + r] = __float2bfloat16_rn(acc[r] * gscale);
  }
}

template <int K, int R>
__device__ __forceinline__ void prefetch_w(const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, int row0, int lane,
                                           uint4 (&q)[R][K / 1024], uint32_t (&sc)[R][K / 1024]) {
  constexpr int CH = K / 1024;
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const uint8_t* wp = W + (size_t)(row0 + r) * (K / 2);
    const uint8_t* sp = S + (size_t)(row0 + r) * (K / 16);
#pragma unroll
    for (int c = 0; c < CH; ++c) {
      q[r][c] = __ldcs((const uint4*)(wp + c * 512 + lane * 16));
      sc[r][c] = *(const uint16_t*)(sp + c * 64 + 2 * lane);
    }
  }
}

// ---- qkv: x/res [2048] bf16, nw [2048] (any float dtype loaded as bf16), N rows ----
template <int R>
__global__ void __launch_bounds__(NWARP * 32)
qkv_gemv_kernel(const bf16* x, const bf16* res, const bf16* __restrict__ nw, float eps, const uint8_t* __restrict__ W,
                const uint8_t* __restrict__ S, const float* __restrict__ gs, bf16* __restrict__ res_out, bf16* __restrict__ y) {
  constexpr int K = 2048, CH = 2;
  __shared__ __align__(16) bf16 xs[K];
  __shared__ float red[NWARP];
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const int row0 = (blockIdx.x * NWARP + warp) * R;
  uint4 q[R][CH];
  uint32_t sc[R][CH];
  prefetch_w<K, R>(W, S, row0, lane, q, sc);
  float gscale = gs[0];
  uint4 wv = *(const uint4*)(nw + t * 8);
  pdl_wait();
  x = launder(x);
  res = launder(res);
  float s[8], xv[8], wf[8];
  bf16x8_to_f32(*(const uint4*)(x + t * 8), s);
  bf16x8_to_f32(wv, wf);
  if (res) {
    bf16x8_to_f32(*(const uint4*)(res + t * 8), xv);
#pragma unroll
    for (int i = 0; i < 8; ++i) s[i] += xv[i];
  }
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) ss = fmaf(s[i], s[i], ss);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  if (lane == 0) red[warp] = ss;
  __syncthreads();
  float tot = 0.f;
#pragma unroll
  for (int i = 0; i < NWARP; ++i) tot += red[i];
  const float rs = rsqrt_ftz(tot / (float)K + eps);
  uint32_t pk[4], rk[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    __nv_bfloat162 n = __floats2bfloat162_rn(s[2 * i] * rs * wf[2 * i], s[2 * i + 1] * rs * wf[2 * i + 1]);
    pk[i] = *reinterpret_cast<uint32_t*>(&n);
    __nv_bfloat162 r2 = __floats2bfloat162_rn(s[2 * i], s[2 * i + 1]);
    rk[i] = *reinterpret_cast<uint32_t*>(&r2);
  }
  *(uint4*)(xs + t * 8) = make_uint4(pk[0], pk[1], pk[2], pk[3]);
  if (blockIdx.x == 0) *(uint4*)(res_out + t * 8) = make_uint4(rk[0], rk[1], rk[2], rk[3]);
  __syncthreads();
  gemv_smem<K, R>(W, S, gscale, xs, q, sc, row0, lane, y);
  __threadfence();
}


// ---- v2: qkv GEMV with the per-head q/k RMSNorm + NeoX RoPE + KV-cache write fused in (the last CTA of each head finishes it) ----
// Rows of a head (128) are spread over 16/R CTAs; each CTA publishes its rows (bf16) to `y`, bumps a per-head arrival counter and the
// last one normalizes/rotates the head and writes q / the KV cache. Counters self-reset (zeroed once at allocation).
// With `trigger` the grid releases its PDL successor (the Flint decode attention, which waits on the full grid) right away.
__device__ __forceinline__ float ldcg_bf16(const bf16* p) {
  unsigned short u;
  asm volatile("ld.global.cg.u16 %0, [%1];" : "=h"(u) : "l"(p) : "memory");
  return __uint_as_float(((uint32_t)u) << 16);
}
__device__ __forceinline__ float bf16r2(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

struct HeadArgs {
  const bf16* q_w; const bf16* k_w; const int64_t* positions; const float* cos_sin; int64_t cs_stride;
  bf16* k_cache; bf16* v_cache; const int64_t* slots; int64_t st_block, st_slot, st_head; int block_size;
  bf16* q_out; unsigned* counters;
};

template <int R>
__global__ void __launch_bounds__(NWARP * 32)
qkv_head_kernel(const bf16* x, const bf16* res, const bf16* __restrict__ nw, float eps, const uint8_t* __restrict__ W,
                const uint8_t* __restrict__ S, const float* __restrict__ gs, bf16* __restrict__ res_out, bf16* y, HeadArgs ha, int trigger) {
  constexpr int K = 2048, CH = 2, CPH = 16 / R;
  __shared__ __align__(16) bf16 xs[K];
  __shared__ float red[NWARP];
  __shared__ int s_last;
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const int row0 = (blockIdx.x * NWARP + warp) * R;
  uint4 q[R][CH];
  uint32_t sc[R][CH];
  prefetch_w<K, R>(W, S, row0, lane, q, sc);
  float gscale = gs[0];
  uint4 wv = *(const uint4*)(nw + t * 8);
  pdl_wait();
  if (trigger) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  x = launder(x);
  res = launder(res);
  float s[8], xv[8], wf[8];
  bf16x8_to_f32(*(const uint4*)(x + t * 8), s);
  bf16x8_to_f32(wv, wf);
  if (res) {
    bf16x8_to_f32(*(const uint4*)(res + t * 8), xv);
#pragma unroll
    for (int i = 0; i < 8; ++i) s[i] += xv[i];
  }
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) ss = fmaf(s[i], s[i], ss);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  if (lane == 0) red[warp] = ss;
  __syncthreads();
  float tot = 0.f;
#pragma unroll
  for (int i = 0; i < NWARP; ++i) tot += red[i];
  const float rs = rsqrt_ftz(tot / (float)K + eps);
  uint32_t pk[4], rk[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    __nv_bfloat162 n = __floats2bfloat162_rn(s[2 * i] * rs * wf[2 * i], s[2 * i + 1] * rs * wf[2 * i + 1]);
    pk[i] = *reinterpret_cast<uint32_t*>(&n);
    __nv_bfloat162 r2 = __floats2bfloat162_rn(s[2 * i], s[2 * i + 1]);
    rk[i] = *reinterpret_cast<uint32_t*>(&r2);
  }
  *(uint4*)(xs + t * 8) = make_uint4(pk[0], pk[1], pk[2], pk[3]);
  if (blockIdx.x == 0) *(uint4*)(res_out + t * 8) = make_uint4(rk[0], rk[1], rk[2], rk[3]);
  __syncthreads();
  gemv_smem<K, R>(W, S, gscale, xs, q, sc, row0, lane, y);

  // ---- arrival: the last CTA of this head finishes it ----
  const int head = (blockIdx.x * NWARP * R) >> 7;
  __threadfence();
  __syncthreads();
  if (t == 0) {
    unsigned prev = atomicAdd(&ha.counters[head], 1u);
    s_last = (prev == CPH - 1);
    if (s_last) ha.counters[head] = 0u;
  }
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  const bool is_q = head < 32, is_v = head >= 36;
  const bf16* row = y + head * 128;
  float x1 = 0.f, x2 = 0.f;
  if (t < 64) { x1 = ldcg_bf16(row + t); x2 = ldcg_bf16(row + 64 + t); }
  float v2 = x1 * x1 + x2 * x2;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v2 += __shfl_xor_sync(0xffffffffu, v2, o);
  if (lane == 0) red[warp] = v2;
  __syncthreads();
  if (t >= 64) return;
  const float var = (red[0] + red[1]) / 128.f;
  const float rs2 = rsqrt_ftz(var + eps);
  const bf16* wq = is_q ? ha.q_w : ha.k_w;
  float n1 = bf16r2(x1 * rs2 * __bfloat162float(wq[t]));
  float n2 = bf16r2(x2 * rs2 * __bfloat162float(wq[64 + t]));
  if (ha.cos_sin) {
    const float* cs = ha.cos_sin + ha.positions[0] * ha.cs_stride;
    const float c = cs[t], sn = cs[64 + t];
    const float r1 = fmaf(n1, c, -(n2 * sn)), r2 = fmaf(n2, c, n1 * sn);
    n1 = r1; n2 = r2;
  }
  const bf16 o1 = __float2bfloat16_rn(is_v ? x1 : n1), o2 = __float2bfloat16_rn(is_v ? x2 : n2);
  if (is_q) { ha.q_out[head * 128 + t] = o1; ha.q_out[head * 128 + 64 + t] = o2; }
  else if (!is_v) { y[head * 128 + t] = o1; y[head * 128 + 64 + t] = o2; }
  const int64_t slot = ha.slots[0];
  if (!is_q && slot >= 0) {
    const int kvh = is_v ? head - 36 : head - 32;
    const int64_t dst = (slot / ha.block_size) * ha.st_block + (slot % ha.block_size) * ha.st_slot + (int64_t)kvh * ha.st_head;
    bf16* c = (is_v ? ha.v_cache : ha.k_cache) + dst;
    c[t] = o1; c[64 + t] = o2;
  }
  __threadfence();
}

// ---- o_proj: a [4096] bf16 ----
template <int R>
__global__ void __launch_bounds__(NWARP * 32)
o_gemv_kernel(const bf16* a, const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, const float* __restrict__ gs, bf16* __restrict__ y, int trigger) {
  constexpr int K = 4096, CH = 4;
  __shared__ __align__(16) bf16 xs[K];
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const int row0 = (blockIdx.x * NWARP + warp) * R;
  uint4 q[R][CH];
  uint32_t sc[R][CH];
  prefetch_w<K, R>(W, S, row0, lane, q, sc);
  float gscale = gs[0];
  if (trigger == 2) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");   // successor = the MoE route_norm kernel (PDL, prefetches the gate rows, waits)
  pdl_wait();
  if (trigger == 1) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  a = launder(a);
  uint4 v0 = *(const uint4*)(a + t * 8), v1 = *(const uint4*)(a + 2048 + t * 8);
  *(uint4*)(xs + t * 8) = v0;
  *(uint4*)(xs + 2048 + t * 8) = v1;
  __syncthreads();
  gemv_smem<K, R>(W, S, gscale, xs, q, sc, row0, lane, y);
  __threadfence();
}

template <typename... A>
static void launch(void (*kern)(A...), dim3 grid, dim3 block, cudaStream_t st, bool pdl, A... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid; cfg.blockDim = block; cfg.dynamicSmemBytes = 0; cfg.stream = st;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at; cfg.numAttrs = pdl ? 1 : 0;
  cudaLaunchKernelEx(&cfg, kern, args...);
}

template <int R>
static void qkv_impl(torch::Tensor x, c10::optional<torch::Tensor> res, torch::Tensor nw, double eps, torch::Tensor W, torch::Tensor S, torch::Tensor gs,
                     torch::Tensor res_out, torch::Tensor y, bool pdl) {
  const int N = W.size(0);
  TORCH_CHECK(N % (NWARP * R) == 0 && W.size(1) == 1024);
  auto st = c10::cuda::getCurrentCUDAStream();
  launch(qkv_gemv_kernel<R>, dim3(N / (NWARP * R)), dim3(NWARP * 32), st, pdl, (const bf16*)x.data_ptr(),
         (const bf16*)(res.has_value() ? res->data_ptr() : nullptr), (const bf16*)nw.data_ptr(), (float)eps, (const uint8_t*)W.data_ptr(),
         (const uint8_t*)S.data_ptr(), (const float*)gs.data_ptr(), (bf16*)res_out.data_ptr(), (bf16*)y.data_ptr());
}
template <int R>
static void o_impl(torch::Tensor a, torch::Tensor W, torch::Tensor S, torch::Tensor gs, torch::Tensor y, bool pdl, int trigger) {
  const int N = W.size(0);
  TORCH_CHECK(N % (NWARP * R) == 0 && W.size(1) == 2048);
  auto st = c10::cuda::getCurrentCUDAStream();
  launch(o_gemv_kernel<R>, dim3(N / (NWARP * R)), dim3(NWARP * 32), st, pdl, (const bf16*)a.data_ptr(), (const uint8_t*)W.data_ptr(),
         (const uint8_t*)S.data_ptr(), (const float*)gs.data_ptr(), (bf16*)y.data_ptr(), (int)trigger);
}
void qkv_gemv(torch::Tensor x, c10::optional<torch::Tensor> res, torch::Tensor nw, double eps, torch::Tensor W, torch::Tensor S, torch::Tensor gs,
              torch::Tensor res_out, torch::Tensor y, bool pdl, int64_t R) {
  if (R == 1) qkv_impl<1>(x, res, nw, eps, W, S, gs, res_out, y, pdl);
  else if (R == 2) qkv_impl<2>(x, res, nw, eps, W, S, gs, res_out, y, pdl);
  else if (R == 4) qkv_impl<4>(x, res, nw, eps, W, S, gs, res_out, y, pdl);
  else TORCH_CHECK(false, "R");
}
void o_gemv(torch::Tensor a, torch::Tensor W, torch::Tensor S, torch::Tensor gs, torch::Tensor y, bool pdl, int64_t R, int64_t trigger) {
  if (R == 1) o_impl<1>(a, W, S, gs, y, pdl, trigger);
  else if (R == 2) o_impl<2>(a, W, S, gs, y, pdl, trigger);
  else if (R == 4) o_impl<4>(a, W, S, gs, y, pdl, trigger);
  else TORCH_CHECK(false, "R");
}

template <int R>
static void qkv_head_impl(torch::Tensor x, c10::optional<torch::Tensor> res, torch::Tensor nw, double eps, torch::Tensor W, torch::Tensor S, torch::Tensor gs,
                          torch::Tensor res_out, torch::Tensor y, torch::Tensor q_w, torch::Tensor k_w, torch::Tensor positions,
                          c10::optional<torch::Tensor> cos_sin, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor slots,
                          torch::Tensor q_out, torch::Tensor counters, bool pdl, bool trigger) {
  const int N = W.size(0);
  TORCH_CHECK(N == 5120 && W.size(1) == 1024);
  HeadArgs ha;
  ha.q_w = (const bf16*)q_w.data_ptr(); ha.k_w = (const bf16*)k_w.data_ptr();
  ha.positions = (const int64_t*)positions.data_ptr();
  ha.cos_sin = cos_sin.has_value() ? (const float*)cos_sin->data_ptr() : nullptr;
  ha.cs_stride = cos_sin.has_value() ? cos_sin->stride(0) : 0;
  ha.k_cache = (bf16*)k_cache.data_ptr(); ha.v_cache = (bf16*)v_cache.data_ptr(); ha.slots = (const int64_t*)slots.data_ptr();
  ha.st_block = k_cache.stride(0); ha.st_slot = k_cache.stride(1); ha.st_head = k_cache.stride(2); ha.block_size = (int)k_cache.size(1);
  ha.q_out = (bf16*)q_out.data_ptr(); ha.counters = (unsigned*)counters.data_ptr();
  auto st = c10::cuda::getCurrentCUDAStream();
  launch(qkv_head_kernel<R>, dim3(N / (NWARP * R)), dim3(NWARP * 32), st, pdl, (const bf16*)x.data_ptr(),
         (const bf16*)(res.has_value() ? res->data_ptr() : nullptr), (const bf16*)nw.data_ptr(), (float)eps, (const uint8_t*)W.data_ptr(),
         (const uint8_t*)S.data_ptr(), (const float*)gs.data_ptr(), (bf16*)res_out.data_ptr(), (bf16*)y.data_ptr(), ha, (int)trigger);
}
void qkv_head(torch::Tensor x, c10::optional<torch::Tensor> res, torch::Tensor nw, double eps, torch::Tensor W, torch::Tensor S, torch::Tensor gs,
              torch::Tensor res_out, torch::Tensor y, torch::Tensor q_w, torch::Tensor k_w, torch::Tensor positions, c10::optional<torch::Tensor> cos_sin,
              torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor slots, torch::Tensor q_out, torch::Tensor counters, bool pdl, bool trigger, int64_t R) {
  if (R == 1) qkv_head_impl<1>(x, res, nw, eps, W, S, gs, res_out, y, q_w, k_w, positions, cos_sin, k_cache, v_cache, slots, q_out, counters, pdl, trigger);
  else if (R == 2) qkv_head_impl<2>(x, res, nw, eps, W, S, gs, res_out, y, q_w, k_w, positions, cos_sin, k_cache, v_cache, slots, q_out, counters, pdl, trigger);
  else if (R == 4) qkv_head_impl<4>(x, res, nw, eps, W, S, gs, res_out, y, q_w, k_w, positions, cos_sin, k_cache, v_cache, slots, q_out, counters, pdl, trigger);
  else TORCH_CHECK(false, "R");
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("qkv_head", &qkv_head);
  m.def("qkv_gemv", &qkv_gemv);
  m.def("o_gemv", &o_gemv);
}
