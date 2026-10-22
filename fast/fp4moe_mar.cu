// NVFP4 W4A16 decode MoE (single token, top-6 + shared expert) for sm_121 reading the *Marlin-repacked* expert / shared-expert tensors directly
// (so no second copy of the weights exists).  Layout (derived from gptq_marlin_repack, tile 16(k) x 64(n), verified by test_mar.py):
//   weight  [kt = K/16][nt = N/64][128 words]  (int32, 8 nibbles each); lane l (= 4*q + c) of a warp reads words 4l..4l+3 (one uint4) of a tile:
//     word g (0..3), nibble j (0..7):  k = 16*kt + 2c + 8*(j&1) + ((j>>2)&1),  n = 64*nt + q + 16*g + 8*((j>>1)&1);  e2m1 sign|e1 e0|m
//   scale   [kt][N] bytes, processed S0E5M3: for tile (kt, nt) lane q reads 8 bytes at 64*nt + 8*q; byte index of subtile s (n = 64nt + 8s + q) is (s&4)+[0,2,1,3][s&3]
//           value h = fp16(byte << 7) (0 flushed);  weight = fp4 * h * g_marlin * 2^-126 (g_marlin = vLLM-processed global scale)
// e2m1 -> fp32 pattern (m<<22 | e<<23 | s<<31) = fp4 * 2^-126 exactly (incl. subnormal); x is pre-multiplied by 2^100 so products stay normal.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

#define K_ 2048
#define N_ 512
#define TOPK_ 6
#define SHN 1024
#define XSCALE 1.2676506002282294e30f   // 2^100
#define GSCALE 7.888609052210118e-31f   // 2^-100

__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
template <class T> __device__ __forceinline__ T* launder(T* p) { asm volatile("" : "+l"(p)::"memory"); return p; }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

template <int J> __device__ __forceinline__ float pat(uint32_t w) {  // nibble J of w -> fp4 * 2^-126
  constexpr int sh = 22 - 4 * J;
  uint32_t mag = (sh >= 0 ? (w << sh) : (w >> (-sh))) & 0x01C00000u;
  uint32_t sg = (w << (28 - 4 * J)) & 0x80000000u;
  return __uint_as_float(mag | sg);
}
__device__ __forceinline__ float hdec(uint32_t b) {  // processed scale byte -> fp16(b<<7) as float (0 stays 0)
  return b ? __uint_as_float((b << 20) + 0x38000000u) : 0.f;
}
// scale byte of subtile s inside the lane's 8 scale bytes
template <int S> struct SIdx { static constexpr int v = (S & 4) + ((S & 3) == 1 ? 2 : (S & 3) == 2 ? 1 : (S & 3)); };

template <int G>
__device__ __forceinline__ void word_fma(uint32_t w, float x0, float x1, float x8, float x9, uint32_t slo, uint32_t shi, float* acc) {
  float p0 = pat<0>(w) * x0;
  p0 = fmaf(pat<1>(w), x8, p0);
  p0 = fmaf(pat<4>(w), x1, p0);
  p0 = fmaf(pat<5>(w), x9, p0);
  float p1 = pat<2>(w) * x0;
  p1 = fmaf(pat<3>(w), x8, p1);
  p1 = fmaf(pat<6>(w), x1, p1);
  p1 = fmaf(pat<7>(w), x9, p1);
  constexpr int s0 = 2 * G, s1 = 2 * G + 1;
  constexpr int b0 = SIdx<s0>::v, b1 = SIdx<s1>::v;
  uint32_t sb0 = ((b0 < 4 ? slo : shi) >> (8 * (b0 & 3))) & 0xFFu;
  uint32_t sb1 = ((b1 < 4 ? slo : shi) >> (8 * (b1 & 3))) & 0xFFu;
  acc[s0] = fmaf(hdec(sb0), p0, acc[s0]);
  acc[s1] = fmaf(hdec(sb1), p1, acc[s1]);
}
__device__ __forceinline__ void tile_fma(uint4 q, uint2 sc, float x0, float x1, float x8, float x9, float* acc) {
  word_fma<0>(q.x, x0, x1, x8, x9, sc.x, sc.y, acc);
  word_fma<1>(q.y, x0, x1, x8, x9, sc.x, sc.y, acc);
  word_fma<2>(q.z, x0, x1, x8, x9, sc.x, sc.y, acc);
  word_fma<3>(q.w, x0, x1, x8, x9, sc.x, sc.y, acc);
}
__device__ __forceinline__ float bf_lo(uint32_t u) { return __uint_as_float(u << 16) * XSCALE; }
__device__ __forceinline__ float bf_hi(uint32_t u) { return __uint_as_float(u & 0xFFFF0000u) * XSCALE; }

// ------------------------------------------------------------------------------------------------------------------------------------------
// up: 48 routed groups (slot, j) + 16 shared groups; group g: gate tile nt_g, up tile nt_g + NTH (NTH = N/64 : 8 routed, 16 shared)
// ------------------------------------------------------------------------------------------------------------------------------------------
template <int NW, int T, int SPLIT>
__global__ void __launch_bounds__(NW * 32)
up_mar(const __nv_bfloat16* __restrict__ x, const int* __restrict__ ids, const uint8_t* __restrict__ w13, const uint8_t* __restrict__ s13,
       const float* __restrict__ g13, __nv_bfloat16* __restrict__ act, const uint8_t* __restrict__ sw13, const uint8_t* __restrict__ ss13,
       const float* __restrict__ sg13, __nv_bfloat16* __restrict__ sact, float* __restrict__ part, unsigned* __restrict__ counter) {
  static_assert(NW * T * SPLIT == 128, "tiles");
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int q = lane >> 2, c = lane & 3;
  const int gb = blockIdx.x / SPLIT, sp = blockIdx.x % SPLIT;
  const bool is_sh = gb >= TOPK_ * 8;
  const uint8_t *wb, *sb;
  size_t rs, ss;
  int ntg, ntu;
  int slot = 0, e = 0;
  if (is_sh) { ntg = gb - TOPK_ * 8; ntu = ntg + 16; wb = sw13; sb = ss13; rs = 4096 * 4; ss = 2048; }
  else { slot = gb >> 3; ntg = gb & 7; ntu = ntg + 8; rs = 2048 * 4; ss = 1024; wb = w13; sb = s13; }
  uint4 qg[T], qu[T];
  uint2 sg[T], su[T];
  auto fetch = [&]() {
    const uint8_t* wbase = wb; const uint8_t* sbase = sb;
    if (!is_sh) { wbase += (size_t)e * 128 * rs; sbase += (size_t)e * 128 * ss; }
#pragma unroll
    for (int i = 0; i < T; ++i) {
      const int kt = sp * (128 / SPLIT) + warp + NW * i;
      qg[i] = __ldcs((const uint4*)(wbase + (size_t)kt * rs + ntg * 512 + lane * 16));
      qu[i] = __ldcs((const uint4*)(wbase + (size_t)kt * rs + ntu * 512 + lane * 16));
      sg[i] = *(const uint2*)(sbase + (size_t)kt * ss + ntg * 64 + 8 * q);
      su[i] = *(const uint2*)(sbase + (size_t)kt * ss + ntu * 64 + 8 * q);
    }
  };
  if (is_sh) fetch();
  pdl_wait();
  x = launder(x); ids = launder(ids);
  pdl_trigger();
  if (!is_sh) { e = min(max(ids[slot], 0), 255); fetch(); }
  float ag[8], au[8];
#pragma unroll
  for (int s = 0; s < 8; ++s) { ag[s] = 0.f; au[s] = 0.f; }
#pragma unroll
  for (int i = 0; i < T; ++i) {
    const int kt = sp * (128 / SPLIT) + warp + NW * i;
    const uint32_t xa = *(const uint32_t*)(x + kt * 16 + 2 * c), xb = *(const uint32_t*)(x + kt * 16 + 8 + 2 * c);
    const float x0 = bf_lo(xa), x1 = bf_hi(xa), x8 = bf_lo(xb), x9 = bf_hi(xb);
    tile_fma(qg[i], sg[i], x0, x1, x8, x9, ag);
    tile_fma(qu[i], su[i], x0, x1, x8, x9, au);
  }
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    ag[s] += __shfl_xor_sync(0xffffffffu, ag[s], 1); ag[s] += __shfl_xor_sync(0xffffffffu, ag[s], 2);
    au[s] += __shfl_xor_sync(0xffffffffu, au[s], 1); au[s] += __shfl_xor_sync(0xffffffffu, au[s], 2);
  }
  __shared__ float red[NW][2][64];
  __shared__ int s_last;
  if (c == 0) {
#pragma unroll
    for (int s = 0; s < 8; ++s) { red[warp][0][s * 8 + q] = ag[s]; red[warp][1][s * 8 + q] = au[s]; }
  }
  __syncthreads();
  const int t = threadIdx.x;
  float tg = 0.f, tu = 0.f;
  if (t < 64) {
#pragma unroll
    for (int w = 0; w < NW; ++w) { tg += red[w][0][t]; tu += red[w][1][t]; }
  }
  if (SPLIT > 1) {
    if (t < 64) { __stcg(part + (size_t)blockIdx.x * 128 + t, tg); __stcg(part + (size_t)blockIdx.x * 128 + 64 + t, tu); }
    __threadfence();
    __syncthreads();
    if (t == 0) { unsigned prev = atomicAdd(counter + gb, 1u); s_last = ((prev & (SPLIT - 1)) == SPLIT - 1); }
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    if (t < 64) {
      tg = 0.f; tu = 0.f;
#pragma unroll
      for (int p = 0; p < SPLIT; ++p) { tg += __ldcg(part + (size_t)(gb * SPLIT + p) * 128 + t); tu += __ldcg(part + (size_t)(gb * SPLIT + p) * 128 + 64 + t); }
    }
  }
  if (t < 64) {
    const float gs = (is_sh ? sg13[0] : g13[e]) * GSCALE;
    float g = bf16r(tg * gs), u = bf16r(tu * gs);
    float sgm = bf16r(g / (1.0f + expf(-g)));
    const __nv_bfloat16 r = __float2bfloat16_rn(sgm * u);
    if (is_sh) sact[ntg * 64 + t] = r; else act[slot * N_ + ntg * 64 + t] = r;
  }
}

// ------------------------------------------------------------------------------------------------------------------------------------------
// down: grid = 32 n-tiles x 8 parts (6 routed experts, 2 halves of the shared expert), 32 k-tiles per part
// ------------------------------------------------------------------------------------------------------------------------------------------
template <int NW>
__global__ void __launch_bounds__(NW * 32)
down_mar(const __nv_bfloat16* __restrict__ act, const __nv_bfloat16* __restrict__ sact, const int* __restrict__ ids, const float* __restrict__ wts,
         const uint8_t* __restrict__ w2, const uint8_t* __restrict__ s2, const float* __restrict__ g2, const uint8_t* __restrict__ sw2,
         const uint8_t* __restrict__ ss2, const float* __restrict__ sg2, __nv_bfloat16* __restrict__ out, float* __restrict__ part,
         unsigned* __restrict__ counter, int trigger) {
  constexpr int T = 32 / NW;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int q = lane >> 2, c = lane & 3;
  const int nt = blockIdx.x >> 3, pt = blockIdx.x & 7;
  const bool is_sh = pt >= TOPK_;
  const size_t rs = 4096 * 4, ss = 2048;
  uint4 qq[T];
  uint2 sc[T];
  int e = 0;
  const uint8_t *wbase, *sbase;
  int kt0;
  if (is_sh) { wbase = sw2; sbase = ss2; kt0 = (pt - TOPK_) * 32; }
  else { ids = launder(ids); e = min(max(ids[pt], 0), 255); wbase = w2 + (size_t)e * 32 * rs; sbase = s2 + (size_t)e * 32 * ss; kt0 = 0; }
#pragma unroll
  for (int i = 0; i < T; ++i) {
    const int kt = kt0 + warp + NW * i;
    const int kl = kt - (is_sh ? 0 : 0);
    qq[i] = __ldcs((const uint4*)(wbase + (size_t)kl * rs + nt * 512 + lane * 16));
    sc[i] = *(const uint2*)(sbase + (size_t)kl * ss + nt * 64 + 8 * q);
  }
  if (trigger == 2) pdl_trigger();   // successor = next layer's qkv_head (PDL, prefetches its weights, waits for this whole grid before reading)
  pdl_wait();
  act = launder(act); sact = launder(sact); wts = launder(wts);
  if (trigger == 1) pdl_trigger();
  const __nv_bfloat16* xs = is_sh ? sact : act + pt * N_;
  float acc[8];
#pragma unroll
  for (int s = 0; s < 8; ++s) acc[s] = 0.f;
#pragma unroll
  for (int i = 0; i < T; ++i) {
    const int kt = kt0 + warp + NW * i;
    const uint32_t xa = *(const uint32_t*)(xs + kt * 16 + 2 * c), xb = *(const uint32_t*)(xs + kt * 16 + 8 + 2 * c);
    tile_fma(qq[i], sc[i], bf_lo(xa), bf_hi(xa), bf_lo(xb), bf_hi(xb), acc);
  }
#pragma unroll
  for (int s = 0; s < 8; ++s) { acc[s] += __shfl_xor_sync(0xffffffffu, acc[s], 1); acc[s] += __shfl_xor_sync(0xffffffffu, acc[s], 2); }
  __shared__ float red[NW][64];
  __shared__ int s_last;
  if (c == 0) {
#pragma unroll
    for (int s = 0; s < 8; ++s) red[warp][s * 8 + q] = acc[s];
  }
  __syncthreads();
  const int t = threadIdx.x;
  if (t < 64) {
    float r = 0.f;
#pragma unroll
    for (int w = 0; w < NW; ++w) r += red[w][t];
    if (!is_sh) {
      const float cw = bf16r(g2[e] * GSCALE * wts[pt]);   // Marlin: bf16(bf16(acc) * bf16(g2*w))
      r = bf16r(bf16r(r) * cw);
    }
    __stcg(part + (size_t)blockIdx.x * 64 + t, r);
  }
  __threadfence();
  __syncthreads();
  if (t == 0) { unsigned prev = atomicAdd(counter + nt, 1u); s_last = ((prev & 7u) == 7u); }
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  if (t < 64) {
    const float* pp = part + (size_t)(nt * 8) * 64 + t;
    float tot = 0.f;
#pragma unroll
    for (int s = 0; s < TOPK_; ++s) tot += __ldcg(pp + s * 64);
    const float fused = bf16r(tot);
    const float rsh = __ldcg(pp + 6 * 64) + __ldcg(pp + 7 * 64);
    const float shared = bf16r(rsh * (sg2[0] * GSCALE));
    out[nt * 64 + t] = __float2bfloat16_rn(shared + fused);
  }
}

// ---- fused router (unchanged from fp4moe v6) ----
__device__ __forceinline__ void bf16x8_to_f32(uint4 u, float* f) {
  uint32_t a[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) { f[2 * i] = __uint_as_float(a[i] << 16); f[2 * i + 1] = __uint_as_float(a[i] & 0xFFFF0000u); }
}
#define RT_BLOCKS 64
#define E_ 256
__global__ void __launch_bounds__(128)
route_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ gw, float* __restrict__ lg,
             unsigned* __restrict__ counter, int* __restrict__ ids, float* __restrict__ tw) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int row = blockIdx.x * 4 + warp;
  uint4 wv[8];
  const uint4* wp = (const uint4*)(gw + (size_t)row * K_);
#pragma unroll
  for (int i = 0; i < 8; ++i) wv[i] = __ldcs(wp + lane + 32 * i);
  pdl_wait();
  x = launder(x);
  pdl_trigger();
  const uint4* xp = (const uint4*)x;
  float acc = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    float xf[8], wf[8];
    bf16x8_to_f32(xp[lane + 32 * i], xf);
    bf16x8_to_f32(wv[i], wf);
#pragma unroll
    for (int j = 0; j < 8; ++j) acc = fmaf(wf[j], xf[j], acc);
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
  if (lane == 0) __stcg(lg + row, bf16r(acc));
  __threadfence();
  __syncthreads();
  __shared__ int s_last;
  if (threadIdx.x == 0) {
    unsigned prev = atomicAdd(counter, 1u);
    s_last = ((prev & (RT_BLOCKS - 1)) == RT_BLOCKS - 1);
  }
  __syncthreads();
  if (!s_last || warp != 0) return;
  __threadfence();
  float sc[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) { float l = __ldcg(lg + lane + 32 * j); sc[j] = 1.0f / (1.0f + expf(-l)); }
  int sel[TOPK_]; float val[TOPK_];
#pragma unroll
  for (int t = 0; t < TOPK_; ++t) {
    float bv = -1.f; int bi = 0x7fffffff;
#pragma unroll
    for (int j = 0; j < 8; ++j) { int idx = lane + 32 * j; if (sc[j] > bv || (sc[j] == bv && idx < bi)) { bv = sc[j]; bi = idx; } }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      float ov = __shfl_xor_sync(0xffffffffu, bv, o); int oi = __shfl_xor_sync(0xffffffffu, bi, o);
      if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    sel[t] = bi; val[t] = bv;
#pragma unroll
    for (int j = 0; j < 8; ++j) if (lane + 32 * j == bi) sc[j] = -2.f;
  }
  float sum = 0.f;
#pragma unroll
  for (int t = 0; t < TOPK_; ++t) sum += val[t];
  if (lane == 0) {
#pragma unroll
    for (int t = 0; t < TOPK_; ++t) { ids[t] = sel[t]; tw[t] = val[t] / sum; }
  }
}


__device__ __forceinline__ float rsqrt_ftz(float v) { float r; asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(v)); return r; }

// ---- router with the post-attention (residual add +) RMSNorm fused in: every CTA recomputes the norm of x + res (2048 elements) into smem,
//      CTA 0 publishes the normalized hidden state (consumed by the up kernel) and the new residual.  Numerics = flint ops/norm.py
//      fused_add_rms_norm: s = x + res (fp32), residual_out = bf16(s), out = bf16(s * rsqrt(mean(s^2) + eps) * w).
__global__ void __launch_bounds__(128)
route_norm_kernel(const __nv_bfloat16* x, const __nv_bfloat16* res, const __nv_bfloat16* __restrict__ nw, float eps, const __nv_bfloat16* __restrict__ gw,
                  float* __restrict__ lg, unsigned* __restrict__ counter, int* __restrict__ ids, float* __restrict__ tw,
                  __nv_bfloat16* __restrict__ xn_out, __nv_bfloat16* __restrict__ res_out, int trigger) {
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const int row = blockIdx.x * 4 + warp;
  uint4 wv[8];
  const uint4* wp = (const uint4*)(gw + (size_t)row * K_);
#pragma unroll
  for (int i = 0; i < 8; ++i) wv[i] = __ldcs(wp + lane + 32 * i);
  uint4 nwv[2];
  nwv[0] = *(const uint4*)(nw + t * 8);
  nwv[1] = *(const uint4*)(nw + (t + 128) * 8);
  if (trigger == 2) pdl_trigger();   // up kernel reads ids/xn only after its own wait; it prefetches the shared-expert weights before
  pdl_wait();
  x = launder(x); res = launder(res);
  if (trigger == 1) pdl_trigger();
  __shared__ __align__(16) __nv_bfloat16 xs[K_];
  __shared__ float red[4];
  float s[16];
  {
    float a[8], b[8];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      bf16x8_to_f32(*(const uint4*)(x + (t + 128 * i) * 8), a);
      bf16x8_to_f32(*(const uint4*)(res + (t + 128 * i) * 8), b);
#pragma unroll
      for (int j = 0; j < 8; ++j) s[8 * i + j] = a[j] + b[j];
    }
  }
  float ss = 0.f;
#pragma unroll
  for (int j = 0; j < 16; ++j) ss = fmaf(s[j], s[j], ss);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  if (lane == 0) red[warp] = ss;
  __syncthreads();
  const float tot = red[0] + red[1] + red[2] + red[3];
  const float rs = rsqrt_ftz(tot / (float)K_ + eps);
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    float wf[8];
    bf16x8_to_f32(nwv[i], wf);
    uint32_t pk[4], rk[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      __nv_bfloat162 n = __floats2bfloat162_rn(s[8 * i + 2 * j] * rs * wf[2 * j], s[8 * i + 2 * j + 1] * rs * wf[2 * j + 1]);
      pk[j] = *reinterpret_cast<uint32_t*>(&n);
      __nv_bfloat162 r2 = __floats2bfloat162_rn(s[8 * i + 2 * j], s[8 * i + 2 * j + 1]);
      rk[j] = *reinterpret_cast<uint32_t*>(&r2);
    }
    *(uint4*)(xs + (t + 128 * i) * 8) = make_uint4(pk[0], pk[1], pk[2], pk[3]);
    if (blockIdx.x == 0) {
      *(uint4*)(xn_out + (t + 128 * i) * 8) = make_uint4(pk[0], pk[1], pk[2], pk[3]);
      *(uint4*)(res_out + (t + 128 * i) * 8) = make_uint4(rk[0], rk[1], rk[2], rk[3]);
    }
  }
  __syncthreads();
  const uint4* xp = (const uint4*)xs;
  float acc = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    float xf[8], wf[8];
    bf16x8_to_f32(xp[lane + 32 * i], xf);
    bf16x8_to_f32(wv[i], wf);
#pragma unroll
    for (int j = 0; j < 8; ++j) acc = fmaf(wf[j], xf[j], acc);
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
  if (lane == 0) __stcg(lg + row, bf16r(acc));
  __threadfence();
  __syncthreads();
  __shared__ int s_last;
  if (threadIdx.x == 0) {
    unsigned prev = atomicAdd(counter, 1u);
    s_last = ((prev & (RT_BLOCKS - 1)) == RT_BLOCKS - 1);
  }
  __syncthreads();
  if (!s_last || warp != 0) return;
  __threadfence();
  float sc[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) { float l = __ldcg(lg + lane + 32 * j); sc[j] = 1.0f / (1.0f + expf(-l)); }
  int sel[TOPK_]; float val[TOPK_];
#pragma unroll
  for (int tt = 0; tt < TOPK_; ++tt) {
    float bv = -1.f; int bi = 0x7fffffff;
#pragma unroll
    for (int j = 0; j < 8; ++j) { int idx = lane + 32 * j; if (sc[j] > bv || (sc[j] == bv && idx < bi)) { bv = sc[j]; bi = idx; } }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      float ov = __shfl_xor_sync(0xffffffffu, bv, o); int oi = __shfl_xor_sync(0xffffffffu, bi, o);
      if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    sel[tt] = bi; val[tt] = bv;
#pragma unroll
    for (int j = 0; j < 8; ++j) if (lane + 32 * j == bi) sc[j] = -2.f;
  }
  float sum = 0.f;
#pragma unroll
  for (int tt = 0; tt < TOPK_; ++tt) sum += val[tt];
  if (lane == 0) {
#pragma unroll
    for (int tt = 0; tt < TOPK_; ++tt) { ids[tt] = sel[tt]; tw[tt] = val[tt] / sum; }
  }
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

#define F(t) ((const float*)(t).data_ptr())
#define U8(t) ((const uint8_t*)(t).data_ptr())
#define BF(t) ((__nv_bfloat16*)(t).data_ptr())
template <int NW, int T, int SPLIT>
static void run_up(torch::Tensor x, torch::Tensor ids, torch::Tensor w13, torch::Tensor s13, torch::Tensor g13, torch::Tensor act, torch::Tensor sw13,
                   torch::Tensor ss13, torch::Tensor sg, torch::Tensor sact, torch::Tensor part, torch::Tensor counter, bool pdl) {
  auto st = c10::cuda::getCurrentCUDAStream();
  launch(up_mar<NW, T, SPLIT>, dim3((TOPK_ * 8 + 16) * SPLIT), dim3(NW * 32), st, pdl, (const __nv_bfloat16*)x.data_ptr(), (const int*)ids.data_ptr(),
         U8(w13), U8(s13), F(g13), BF(act), U8(sw13), U8(ss13), F(sg), BF(sact), (float*)part.data_ptr(), (unsigned*)counter.data_ptr());
}
void moe_up_m(int cfg, torch::Tensor x, torch::Tensor ids, torch::Tensor w13, torch::Tensor s13, torch::Tensor g13, torch::Tensor act, torch::Tensor sw13,
              torch::Tensor ss13, torch::Tensor sg, torch::Tensor sact, torch::Tensor part, torch::Tensor counter, bool pdl) {
  switch (cfg) {
    case 0: run_up<16, 8, 1>(x, ids, w13, s13, g13, act, sw13, ss13, sg, sact, part, counter, pdl); break;
    case 1: run_up<8, 8, 2>(x, ids, w13, s13, g13, act, sw13, ss13, sg, sact, part, counter, pdl); break;
    case 2: run_up<8, 4, 4>(x, ids, w13, s13, g13, act, sw13, ss13, sg, sact, part, counter, pdl); break;
    case 3: run_up<4, 4, 8>(x, ids, w13, s13, g13, act, sw13, ss13, sg, sact, part, counter, pdl); break;
    case 4: run_up<8, 2, 8>(x, ids, w13, s13, g13, act, sw13, ss13, sg, sact, part, counter, pdl); break;
    case 5: run_up<4, 2, 16>(x, ids, w13, s13, g13, act, sw13, ss13, sg, sact, part, counter, pdl); break;
    default: TORCH_CHECK(false, "bad up cfg");
  }
}
template <int NW>
static void run_down(torch::Tensor act, torch::Tensor sact, torch::Tensor ids, torch::Tensor wts, torch::Tensor w2, torch::Tensor s2, torch::Tensor g2,
                     torch::Tensor sw2, torch::Tensor ss2, torch::Tensor sg2, torch::Tensor out, torch::Tensor part, torch::Tensor counter, bool pdl, int64_t trigger) {
  auto st = c10::cuda::getCurrentCUDAStream();
  launch(down_mar<NW>, dim3(32 * 8), dim3(NW * 32), st, pdl, (const __nv_bfloat16*)act.data_ptr(), (const __nv_bfloat16*)sact.data_ptr(),
         (const int*)ids.data_ptr(), F(wts), U8(w2), U8(s2), F(g2), U8(sw2), U8(ss2), F(sg2), BF(out), (float*)part.data_ptr(), (unsigned*)counter.data_ptr(), (int)trigger);
}
void moe_down_m(int cfg, torch::Tensor act, torch::Tensor sact, torch::Tensor ids, torch::Tensor wts, torch::Tensor w2, torch::Tensor s2, torch::Tensor g2,
                torch::Tensor sw2, torch::Tensor ss2, torch::Tensor sg2, torch::Tensor out, torch::Tensor part, torch::Tensor counter, bool pdl, int64_t trigger) {
  switch (cfg) {
    case 0: run_down<4>(act, sact, ids, wts, w2, s2, g2, sw2, ss2, sg2, out, part, counter, pdl, trigger); break;
    case 1: run_down<8>(act, sact, ids, wts, w2, s2, g2, sw2, ss2, sg2, out, part, counter, pdl, trigger); break;
    case 2: run_down<2>(act, sact, ids, wts, w2, s2, g2, sw2, ss2, sg2, out, part, counter, pdl, trigger); break;
    case 3: run_down<16>(act, sact, ids, wts, w2, s2, g2, sw2, ss2, sg2, out, part, counter, pdl, trigger); break;
    default: TORCH_CHECK(false, "bad down cfg");
  }
}
void moe_route(torch::Tensor x, torch::Tensor gw, torch::Tensor lg, torch::Tensor counter, torch::Tensor ids, torch::Tensor tw, bool pdl) {
  auto st = c10::cuda::getCurrentCUDAStream();
  launch(route_kernel, dim3(RT_BLOCKS), dim3(128), st, pdl, (const __nv_bfloat16*)x.data_ptr(), (const __nv_bfloat16*)gw.data_ptr(),
         (float*)lg.data_ptr(), (unsigned*)counter.data_ptr(), (int*)ids.data_ptr(), (float*)tw.data_ptr());
}
void moe_route_norm(torch::Tensor x, torch::Tensor res, torch::Tensor nw, double eps, torch::Tensor gw, torch::Tensor lg, torch::Tensor counter, torch::Tensor ids,
                    torch::Tensor tw, torch::Tensor xn, torch::Tensor res_out, bool pdl, int64_t rtrig) {
  auto st = c10::cuda::getCurrentCUDAStream();
  launch(route_norm_kernel, dim3(RT_BLOCKS), dim3(128), st, pdl, (const __nv_bfloat16*)x.data_ptr(), (const __nv_bfloat16*)res.data_ptr(),
         (const __nv_bfloat16*)nw.data_ptr(), (float)eps, (const __nv_bfloat16*)gw.data_ptr(), (float*)lg.data_ptr(), (unsigned*)counter.data_ptr(),
         (int*)ids.data_ptr(), (float*)tw.data_ptr(), (__nv_bfloat16*)xn.data_ptr(), (__nv_bfloat16*)res_out.data_ptr(), (int)rtrig);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_route_norm", &moe_route_norm);
  m.def("moe_up_m", &moe_up_m);
  m.def("moe_down_m", &moe_down_m);
  m.def("moe_route", &moe_route);
}
