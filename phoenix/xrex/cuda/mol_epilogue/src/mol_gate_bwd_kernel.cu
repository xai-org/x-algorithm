#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "mol_epilogue_kernel.hpp"
#include "mol_gate.hpp"
#include "xrex/cuda/xla_utils/cuda_error_utils.hpp"


namespace {

constexpr int kWarps = 4;
constexpr int kThreads = 32 * kWarps;
constexpr int kItemsPerWarp = 16;
constexpr int kItemsPerBlock = kWarps * kItemsPerWarp;
static_assert(kItemsPerBlock == kMolGateBwdItemsPerBlock);
constexpr unsigned kFullMask = 0xffffffffu;
constexpr int kMinBlocksPerSm = 4;
constexpr float kMasked = -1.0e30f;

struct Params {
  const __nv_bfloat16* cos;
  const float* ubias;
  const float* ibias;
  const float* w1;
  const float* b1;
  const float* w2;
  const float* b2;
  const float* dscore;
  __nv_bfloat16* dcos;
  float* dibias;
  float* dub_partial;
  float* dw_partial;
  int num_users;
  long long num_items;
};

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&v);
}

__device__ __forceinline__ uint32_t transpose8x8(uint32_t x) {
  uint32_t y;
  asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(y) : "r"(x));
  return y;
}

__device__ __forceinline__ void mma16816(
    float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1
) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
  );
}

__device__ __forceinline__ void mma1688(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(b0)
  );
}

__device__ __forceinline__ float quad_max(float v) {
  v = fmaxf(v, __shfl_xor_sync(kFullMask, v, 1));
  return fmaxf(v, __shfl_xor_sync(kFullMask, v, 2));
}

__device__ __forceinline__ float quad_sum(float v) {
  v += __shfl_xor_sync(kFullMask, v, 1);
  return v + __shfl_xor_sync(kFullMask, v, 2);
}

__device__ __forceinline__ float column_sum(float v) {
  v += __shfl_xor_sync(kFullMask, v, 4);
  v += __shfl_xor_sync(kFullMask, v, 8);
  return v + __shfl_xor_sync(kFullMask, v, 16);
}

template <int KL, int H>
__global__ void __launch_bounds__(kThreads, kMinBlocksPerSm) mol_gate_bwd_kernel(Params p) {
  static_assert(KL == 4 || KL == 8, "the logits are one n = 8 mma tile plus a bias lane");
  static_assert(H % 16 == 0, "H must be a multiple of 16");
  constexpr int NT = H / 8;
  constexpr int KS = H / 16;
  constexpr int kWeights = 2 * KL * H + H;

  __shared__ float s_w1p[16][H];
  __shared__ float s_w2[H][8];
  __shared__ float s_red[kWarps][kWeights];
  extern __shared__ float s_dub_all[];

  const int B = p.num_users;
  for (int i = threadIdx.x; i < 16 * H; i += kThreads) {
    const int k = i / H, h = i % H;
    s_w1p[k][h] = k < KL ? p.w1[k * H + h] : (k == KL ? p.b1[h] : 0.0f);
  }
  for (int i = threadIdx.x; i < H * 8; i += kThreads) {
    const int h = i / 8, c = i % 8;
    s_w2[h][c] = c < KL ? p.w2[h * KL + c] : 0.0f;
  }
  for (int i = threadIdx.x; i < kWarps * KL * B; i += kThreads) s_dub_all[i] = 0.0f;
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int gid = lane >> 2;
  const int tig = lane & 3;
  const long long M = p.num_items;
  const long long m0 = static_cast<long long>(blockIdx.x) * kItemsPerBlock + warp * kItemsPerWarp;
  const long long row[2] = {m0 + gid, m0 + gid + 8};
  const bool row_ok[2] = {row[0] < M, row[1] < M};
  const int c0 = 2 * tig;
  const bool col_ok = c0 < KL;

  uint32_t bw1[NT][2], bw2t[NT], bw2[KS][2], bw1t[KS][2];
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    const int h = nt * 8 + gid;
    bw1[nt][0] = pack_bf16(s_w1p[c0][h], s_w1p[c0 + 1][h]);
    bw1[nt][1] = pack_bf16(s_w1p[c0 + 8][h], s_w1p[c0 + 9][h]);
    bw2t[nt] = pack_bf16(s_w2[h][c0], s_w2[h][c0 + 1]);
  }
#pragma unroll
  for (int ks = 0; ks < KS; ++ks) {
    const int h = 16 * ks + c0;
    bw2[ks][0] = pack_bf16(s_w2[h][gid], s_w2[h + 1][gid]);
    bw2[ks][1] = pack_bf16(s_w2[h + 8][gid], s_w2[h + 9][gid]);
    const bool real = gid < KL;
    bw1t[ks][0] = pack_bf16(real ? s_w1p[gid][h] : 0.0f, real ? s_w1p[gid][h + 1] : 0.0f);
    bw1t[ks][1] = pack_bf16(real ? s_w1p[gid][h + 8] : 0.0f, real ? s_w1p[gid][h + 9] : 0.0f);
  }

  float lbias[2][2];
#pragma unroll
  for (int r = 0; r < 2; ++r)
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int c = c0 + j;
      lbias[r][j] = !col_ok     ? kMasked
                    : row_ok[r] ? p.b2[c] + p.ibias[static_cast<long long>(c) * M + row[r]]
                                : p.b2[c];
    }
  const uint32_t one = pack_bf16(1.0f, 0.0f);
  const bool one_in_lo = KL == 4 && tig == 2;
  const bool one_in_hi = KL == 8 && tig == 0;

  float acc_dib[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
  float acc_dw1[NT][4];
  float acc_dw2[KS][4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) acc_dw1[nt][i] = 0.0f;
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) acc_dw2[ks][i] = 0.0f;
  }
  float* s_dub = s_dub_all + warp * KL * B;

  for (int b = 0; b < B; ++b) {
    float cs[2][2], ds[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      ds[r] = row_ok[r] ? p.dscore[static_cast<long long>(b) * M + row[r]] : 0.0f;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const long long idx = (static_cast<long long>(c0 + j) * B + b) * M + row[r];
        cs[r][j] = col_ok && row_ok[r] ? __bfloat162float(p.cos[idx]) : 0.0f;
      }
    }
    const uint32_t a_cos[4] = {
        one_in_lo ? one : pack_bf16(cs[0][0], cs[0][1]),
        one_in_lo ? one : pack_bf16(cs[1][0], cs[1][1]),
        one_in_hi ? one : 0u,
        one_in_hi ? one : 0u,
    };

    float pre[NT][4], th[NT][4];
    uint32_t a_hid[KS][4];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
      for (int i = 0; i < 4; ++i) pre[nt][i] = 0.0f;
      mma16816(pre[nt], a_cos[0], a_cos[1], a_cos[2], a_cos[3], bw1[nt][0], bw1[nt][1]);
    }
    float hid[NT][4];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float half_pre = 0.5f * pre[nt][i];
        th[nt][i] = xrex_mol::tanh_approx(half_pre);
        hid[nt][i] = fmaf(half_pre, th[nt][i], half_pre);
      }
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
      a_hid[ks][0] = pack_bf16(hid[2 * ks][0], hid[2 * ks][1]);
      a_hid[ks][1] = pack_bf16(hid[2 * ks][2], hid[2 * ks][3]);
      a_hid[ks][2] = pack_bf16(hid[2 * ks + 1][0], hid[2 * ks + 1][1]);
      a_hid[ks][3] = pack_bf16(hid[2 * ks + 1][2], hid[2 * ks + 1][3]);
    }
    float lg[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int ks = 0; ks < KS; ++ks)
      mma16816(lg, a_hid[ks][0], a_hid[ks][1], a_hid[ks][2], a_hid[ks][3], bw2[ks][0], bw2[ks][1]);

    const float ub0 = col_ok ? __ldg(&p.ubias[c0 * B + b]) : 0.0f;
    const float ub1 = col_ok ? __ldg(&p.ubias[(c0 + 1) * B + b]) : 0.0f;
    float dl[4], dc[4];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const float l0 = lg[2 * r] + lbias[r][0] + ub0;
      const float l1 = lg[2 * r + 1] + lbias[r][1] + ub1;
      const float mx = quad_max(fmaxf(l0, l1));
      const float e0 = __expf(l0 - mx), e1 = __expf(l1 - mx);
      const float inv = __frcp_rn(quad_sum(e0 + e1));
      const float g0 = e0 * inv, g1 = e1 * inv;
      const float score = quad_sum(fmaf(g0, cs[r][0], g1 * cs[r][1]));
      dc[2 * r] = ds[r] * g0;
      dc[2 * r + 1] = ds[r] * g1;
      dl[2 * r] = dc[2 * r] * (cs[r][0] - score);
      dl[2 * r + 1] = dc[2 * r + 1] * (cs[r][1] - score);
    }
    const uint32_t a_dl[2] = {pack_bf16(dl[0], dl[1]), pack_bf16(dl[2], dl[3])};

    float dpre[NT][4];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
      for (int i = 0; i < 4; ++i) dpre[nt][i] = 0.0f;
      mma1688(dpre[nt], a_dl[0], a_dl[1], bw2t[nt]);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float t = th[nt][i];
        const float half_pre = 0.5f * pre[nt][i];
        dpre[nt][i] *= 0.5f * fmaf(half_pre, fmaf(-t, t, 1.0f), 1.0f + t);
      }
    }

#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
      mma16816(
          dc,
          pack_bf16(dpre[2 * ks][0], dpre[2 * ks][1]),
          pack_bf16(dpre[2 * ks][2], dpre[2 * ks][3]),
          pack_bf16(dpre[2 * ks + 1][0], dpre[2 * ks + 1][1]),
          pack_bf16(dpre[2 * ks + 1][2], dpre[2 * ks + 1][3]),
          bw1t[ks][0],
          bw1t[ks][1]
      );
    }
    if (col_ok) {
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        if (!row_ok[r])
          continue;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const long long idx = (static_cast<long long>(c0 + j) * B + b) * M + row[r];
          p.dcos[idx] = __float2bfloat16(dc[2 * r + j]);
        }
      }
    }

    const uint32_t at0 = transpose8x8(a_cos[0]);
    const uint32_t at2 = transpose8x8(a_cos[1]);
    const uint32_t at1 = KL == 8 ? transpose8x8(a_cos[2]) : 0u;
    const uint32_t at3 = KL == 8 ? transpose8x8(a_cos[3]) : 0u;
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      mma16816(
          acc_dw1[nt],
          at0,
          at1,
          at2,
          at3,
          transpose8x8(pack_bf16(dpre[nt][0], dpre[nt][1])),
          transpose8x8(pack_bf16(dpre[nt][2], dpre[nt][3]))
      );
    }
    const uint32_t bt0 = transpose8x8(a_dl[0]);
    const uint32_t bt1 = transpose8x8(a_dl[1]);
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
      mma16816(
          acc_dw2[ks],
          transpose8x8(a_hid[ks][0]),
          transpose8x8(a_hid[ks][2]),
          transpose8x8(a_hid[ks][1]),
          transpose8x8(a_hid[ks][3]),
          bt0,
          bt1
      );
    }

#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
      for (int j = 0; j < 2; ++j) acc_dib[r][j] += dl[2 * r + j];
    const float su0 = column_sum(dl[0] + dl[2]);
    const float su1 = column_sum(dl[1] + dl[3]);
    if (gid == 0 && col_ok) {
      s_dub[c0 * B + b] = su0;
      s_dub[(c0 + 1) * B + b] = su1;
    }
  }

  if (col_ok) {
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      if (!row_ok[r])
        continue;
#pragma unroll
      for (int j = 0; j < 2; ++j)
        p.dibias[static_cast<long long>(c0 + j) * M + row[r]] = acc_dib[r][j];
    }
  }

  float* red = s_red[warp];
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int k = gid + 8 * half;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int h = nt * 8 + c0 + j;
        const float v = acc_dw1[nt][2 * half + j];
        if (k < KL) {
          red[k * H + h] = v;
        } else if (k == KL) {
          red[KL * H + h] = v;
        }
      }
    }
  }
#pragma unroll
  for (int ks = 0; ks < KS; ++ks) {
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int h = 16 * ks + gid + 8 * half;
      if (col_ok) {
#pragma unroll
        for (int j = 0; j < 2; ++j) red[KL * H + H + h * KL + c0 + j] = acc_dw2[ks][2 * half + j];
      }
    }
  }
  __syncthreads();

  float* dw_out = p.dw_partial + static_cast<long long>(blockIdx.x) * kWeights;
  for (int i = threadIdx.x; i < kWeights; i += kThreads) {
    float v = s_red[0][i];
#pragma unroll
    for (int w = 1; w < kWarps; ++w) v += s_red[w][i];
    dw_out[i] = v;
  }
  float* dub_out = p.dub_partial + static_cast<long long>(blockIdx.x) * KL * B;
  for (int i = threadIdx.x; i < KL * B; i += kThreads) {
    float v = s_dub_all[i];
#pragma unroll
    for (int w = 1; w < kWarps; ++w) v += s_dub_all[w * KL * B + i];
    dub_out[i] = v;
  }
}

template <int KL, int H>
ffi::Error launch(const Params& p, cudaStream_t stream, unsigned blocks) {
  const size_t smem = static_cast<size_t>(kWarps) * KL * p.num_users * sizeof(float);
  if (smem > 48 * 1024) {
    XAI_RETURN_IF_CUDA_ERROR(cudaFuncSetAttribute(
        mol_gate_bwd_kernel<KL, H>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem)
    ));
  }
  mol_gate_bwd_kernel<KL, H><<<blocks, kThreads, smem, stream>>>(p);
  return ffi::Error::Success();
}

}

ffi::Error mol_gate_bwd(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::BF16> cos,
    ffi::Buffer<ffi::DataType::F32> ubias,
    ffi::Buffer<ffi::DataType::F32> ibias,
    ffi::Buffer<ffi::DataType::F32> w1,
    ffi::Buffer<ffi::DataType::F32> b1,
    ffi::Buffer<ffi::DataType::F32> w2,
    ffi::Buffer<ffi::DataType::F32> b2,
    ffi::Buffer<ffi::DataType::F32> dscore,
    ffi::Result<ffi::Buffer<ffi::DataType::BF16>> dcos,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> dibias,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> dub_partial,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> dw_partial,
    int64_t num_users
) {
  const auto cos_dims = cos.dimensions();
  const auto w1_dims = w1.dimensions();
  if (cos_dims.size() != 2 || w1_dims.size() != 2) {
    return ffi::Error::InvalidArgument("cos must be [KL*B, M] and w1 [KL, H]");
  }
  const int64_t kl = w1_dims[0];
  const int64_t hidden = w1_dims[1];
  const int64_t M = cos_dims[1];
  const int64_t blocks = (M + kItemsPerBlock - 1) / kItemsPerBlock;
  if (cos_dims[0] != kl * num_users)
    return ffi::Error::InvalidArgument("cos rows != KL * num_users");
  if (num_users > kMolGateBwdMaxUsers) {
    return ffi::Error::InvalidArgument("too many users for the kernel");
  }
  auto is_2d = [](auto dims, int64_t a, int64_t b) {
    return dims.size() == 2 && dims[0] == a && dims[1] == b;
  };
  if (!is_2d(ubias.dimensions(), kl, num_users)) {
    return ffi::Error::InvalidArgument("ubias must be [KL, B]");
  }
  if (!is_2d(ibias.dimensions(), kl, M))
    return ffi::Error::InvalidArgument("ibias must be [KL, M]");
  if (!is_2d(dscore.dimensions(), num_users, M)) {
    return ffi::Error::InvalidArgument("dscore must be [B, M]");
  }
  if (!is_2d(dw_partial->dimensions(), blocks, 2 * kl * hidden + hidden)) {
    return ffi::Error::InvalidArgument("dw_partial must be [blocks, 2*KL*H + H]");
  }
  const auto dub_dims = dub_partial->dimensions();
  if (dub_dims.size() != 3 || dub_dims[0] != blocks || dub_dims[1] != kl ||
      dub_dims[2] != num_users) {
    return ffi::Error::InvalidArgument("dub_partial must be [blocks, KL, B]");
  }
  Params p{
      reinterpret_cast<const __nv_bfloat16*>(cos.typed_data()),
      ubias.typed_data(),
      ibias.typed_data(),
      w1.typed_data(),
      b1.typed_data(),
      w2.typed_data(),
      b2.typed_data(),
      dscore.typed_data(),
      reinterpret_cast<__nv_bfloat16*>(dcos->typed_data()),
      dibias->typed_data(),
      dub_partial->typed_data(),
      dw_partial->typed_data(),
      static_cast<int>(num_users),
      M
  };
  const auto grid = static_cast<unsigned>(blocks);
  ffi::Error status = ffi::Error::Success();
  if (kl == 4 && hidden == 32) {
    status = launch<4, 32>(p, stream, grid);
  } else if (kl == 8 && hidden == 16) {
    status = launch<8, 16>(p, stream, grid);
  } else if (kl == 8 && hidden == 32) {
    status = launch<8, 32>(p, stream, grid);
  } else {
    return ffi::Error::InvalidArgument("unsupported (KL, H); compiled: (4,32), (8,16), (8,32)");
  }
  if (status.failure())
    return status;
  XAI_RETURN_IF_CUDA_ERROR(cudaGetLastError());
  return ffi::Error::Success();
}
