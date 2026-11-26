#include <cub/cub.cuh>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include "cuda_error_utils.hpp"
#include "row_emb_common.cuh"
#include "row_emb_kernel.hpp"

namespace xai::kernels::row_emb {

namespace {

constexpr int kThreads = 256;
constexpr int kWarpsPerBlock = kThreads / kWarpThreads;
constexpr int kSideGrid = 2048;
constexpr int kMainGrid = 4096;
constexpr int kRowVecs = 4;
constexpr int kChunkCols = kRowVecs * kWarpThreads * 8;

void check_width(int64_t width) {
  XAI_ASSERT(width > 0 && width % 8 == 0, ": row width must be a positive multiple of 8");
}

void check_count(int64_t n) {
  XAI_ASSERT(n >= 0 && n <= std::numeric_limits<int32_t>::max(), ": count out of int32 range");
}

__device__ __forceinline__ void zero_row(bf16* dst, int64_t width, int lane) {
  const bf16x8 zero{uint4{0u, 0u, 0u, 0u}};
  for (int64_t v = lane; v < width / 8; v += kWarpThreads) store8(dst + v * 8, zero);
}

__device__ __forceinline__ void copy_row(bf16* dst, const bf16* src, int64_t width, int lane) {
  for (int64_t v = lane; v < width / 8; v += kWarpThreads) store8(dst + v * 8, load8(src + v * 8));
}

__device__ __forceinline__ void sum_occurrences(
    const bf16* grads,
    int64_t width,
    const int32_t* order,
    int32_t begin,
    int32_t end,
    int64_t col0,
    int lane,
    float (&acc)[kRowVecs][8]
) {
#pragma unroll
  for (int i = 0; i < kRowVecs; ++i)
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[i][j] = 0.f;
  constexpr int kBatchRows = 4;
  int32_t k = begin;
  for (; k + kBatchRows <= end; k += kBatchRows) {
    bf16x8 v[kBatchRows][kRowVecs];
#pragma unroll
    for (int r = 0; r < kBatchRows; ++r) {
      const bf16* g = grads + int64_t(__ldg(order + k + r)) * width;
#pragma unroll
      for (int i = 0; i < kRowVecs; ++i) {
        const int64_t col = col0 + int64_t(lane + i * kWarpThreads) * 8;
        v[r][i] = col < width ? load8(g + col) : bf16x8{uint4{0u, 0u, 0u, 0u}};
      }
    }
#pragma unroll
    for (int r = 0; r < kBatchRows; ++r)
#pragma unroll
      for (int i = 0; i < kRowVecs; ++i) {
        float t[8];
        unpack8(v[r][i], t);
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[i][j] += t[j];
      }
  }
  for (; k < end; ++k) {
    const bf16* g = grads + int64_t(__ldg(order + k)) * width;
#pragma unroll
    for (int i = 0; i < kRowVecs; ++i) {
      const int64_t col = col0 + int64_t(lane + i * kWarpThreads) * 8;
      if (col < width) {
        float t[8];
        unpack8(load8(g + col), t);
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[i][j] += t[j];
      }
    }
  }
}

__global__ void sanitize_ids_kernel(const int32_t* ids, int32_t* out, int64_t n, int64_t rows) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int64_t g = ids[i];
  if (g < 0) g += rows;
  if (g < 0) g = 0;
  if (g >= rows) g = rows - 1;
  out[i] = int32_t(g);
}

__global__ void iota_kernel(int32_t* v, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) v[i] = i;
}

__global__ void flag_kernel(const int32_t* sorted, int32_t* flags, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) flags[i] = (i == 0 || sorted[i] != sorted[i - 1]) ? 1 : 0;
}

__global__ void scatter_unique_kernel(
    const int32_t* sorted,
    const int32_t* order,
    const int32_t* flags,
    const int32_t* pos,
    int n,
    int32_t* unique,
    int32_t* inverse,
    int32_t* offsets,
    int32_t* count
) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int p = pos[i] - 1;
  if (flags[i]) {
    unique[p] = sorted[i];
    offsets[p] = i;
  }
  inverse[order[i]] = p;
  if (i == n - 1) {
    *count = p + 1;
    offsets[p + 1] = n;
  }
}

struct UniqueWs {
  int32_t* iota;
  int32_t* sorted;
  int32_t* pos;
  void* tmp;
  size_t tmp_bytes;
  size_t bytes;
};

size_t sort_temp_bytes(int n) {
  size_t bytes = 0;
  XAI_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
      nullptr,
      bytes,
      static_cast<const int32_t*>(nullptr),
      static_cast<int32_t*>(nullptr),
      static_cast<const int32_t*>(nullptr),
      static_cast<int32_t*>(nullptr),
      n
  ));
  return bytes;
}

size_t scan_temp_bytes(int n) {
  size_t bytes = 0;
  XAI_CUDA_CHECK(cub::DeviceScan::InclusiveSum(
      nullptr, bytes, static_cast<const int32_t*>(nullptr), static_cast<int32_t*>(nullptr), n
  ));
  return bytes;
}

UniqueWs carve_unique(void* base, int n) {
  Carver c(base);
  UniqueWs w;
  w.iota = c.take<int32_t>(n);
  w.sorted = c.take<int32_t>(n);
  w.pos = c.take<int32_t>(n);
  w.tmp_bytes = std::max(sort_temp_bytes(n), scan_temp_bytes(n));
  w.tmp = c.take<char>(w.tmp_bytes);
  w.bytes = c.bytes();
  return w;
}

__global__ void owner_counts_kernel(
    const int32_t* unique, const int32_t* n_unique, Ownership ownership, int32_t* counts
) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= ownership.world) return;
  const int n = *n_unique;
  auto lower_bound = [&](int64_t key) {
    int lo = 0, hi = n;
    while (lo < hi) {
      const int mid = (lo + hi) >> 1;
      if (int64_t(unique[mid]) < key) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    return lo;
  };
  const int begin = lower_bound(int64_t(p) * ownership.rows_per_rank);
  const int end = lower_bound(int64_t(p + 1) * ownership.rows_per_rank);
  counts[p] = end - begin;
}

__global__ void gather_rows_kernel(
    const bf16* table,
    int64_t rows_local,
    int64_t width,
    const int32_t* ids,
    int64_t n,
    Ownership ownership,
    int rank,
    bf16* rows
) {
  const int lane = threadIdx.x % kWarpThreads;
  const int64_t warps = int64_t(gridDim.x) * kWarpsPerBlock;
  for (int64_t i = int64_t(blockIdx.x) * kWarpsPerBlock + threadIdx.x / kWarpThreads; i < n;
       i += warps) {
    const int64_t g = __ldg(ids + i);
    const int64_t local = ownership.local(g);
    if (ownership.owner(g) != rank || local >= rows_local) {
      zero_row(rows + i * width, width, lane);
    } else {
      copy_row(rows + i * width, table + local * width, width, lane);
    }
  }
}

__global__ void scatter_lookup_kernel(
    const bf16* rows, int64_t width, const int32_t* inverse, int64_t n, bf16* out
) {
  const int lane = threadIdx.x % kWarpThreads;
  const int64_t warps = int64_t(gridDim.x) * kWarpsPerBlock;
  for (int64_t i = int64_t(blockIdx.x) * kWarpsPerBlock + threadIdx.x / kWarpThreads; i < n;
       i += warps) {
    copy_row(out + i * width, rows + int64_t(__ldg(inverse + i)) * width, width, lane);
  }
}

constexpr int kLongSegment = 128;
constexpr int kLongThreads = 1024;
constexpr int kLongWarps = kLongThreads / kWarpThreads;

__global__ void segment_reduce_rows_kernel(
    const bf16* grads,
    int64_t width,
    const int32_t* order,
    const int32_t* offsets,
    const int32_t* n_unique,
    bf16* out
) {
  const int lane = threadIdx.x % kWarpThreads;
  const int n = *n_unique;
  const int64_t warps = int64_t(gridDim.x) * kWarpsPerBlock;
  for (int64_t u = int64_t(blockIdx.x) * kWarpsPerBlock + threadIdx.x / kWarpThreads; u < n;
       u += warps) {
    const int32_t begin = __ldg(offsets + u), end = __ldg(offsets + u + 1);
    if (end - begin > kLongSegment) continue;
    bf16* dst = out + u * width;
    for (int64_t col0 = 0; col0 < width; col0 += kChunkCols) {
      float acc[kRowVecs][8];
      sum_occurrences(grads, width, order, begin, end, col0, lane, acc);
#pragma unroll
      for (int i = 0; i < kRowVecs; ++i) {
        const int64_t col = col0 + int64_t(lane + i * kWarpThreads) * 8;
        if (col < width) store8(dst + col, pack8(acc[i]));
      }
    }
  }
}

__global__ void __launch_bounds__(kLongThreads) long_segment_reduce_rows_kernel(
    const bf16* grads,
    int64_t width,
    const int32_t* order,
    const int32_t* offsets,
    const int32_t* n_unique,
    bf16* out
) {
  __shared__ float partial[kLongWarps][kWarpThreads * 8];
  const int lane = threadIdx.x % kWarpThreads;
  const int warp = threadIdx.x / kWarpThreads;
  const int n = *n_unique;
  for (int64_t u = blockIdx.x; u < n; u += gridDim.x) {
    const int32_t begin = __ldg(offsets + u), end = __ldg(offsets + u + 1);
    if (end - begin <= kLongSegment) continue;
    bf16* dst = out + u * width;
    for (int64_t col0 = 0; col0 < width; col0 += kChunkCols) {
      float acc[kRowVecs][8];
#pragma unroll
      for (int i = 0; i < kRowVecs; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[i][j] = 0.f;
      for (int32_t k = begin + warp; k < end; k += kLongWarps) {
        const bf16* g = grads + int64_t(__ldg(order + k)) * width;
#pragma unroll
        for (int i = 0; i < kRowVecs; ++i) {
          const int64_t col = col0 + int64_t(lane + i * kWarpThreads) * 8;
          if (col < width) {
            float t[8];
            unpack8(load8(g + col), t);
#pragma unroll
            for (int j = 0; j < 8; ++j) acc[i][j] += t[j];
          }
        }
      }
#pragma unroll
      for (int i = 0; i < kRowVecs; ++i) {
#pragma unroll
        for (int j = 0; j < 8; ++j) partial[warp][lane * 8 + j] = acc[i][j];
        __syncthreads();
        if (threadIdx.x < kWarpThreads * 8) {
          const int64_t col = col0 + int64_t(i) * kWarpThreads * 8 + threadIdx.x;
          if (col < width) {
            float sum = 0.f;
            for (int w = 0; w < kLongWarps; ++w) sum += partial[w][threadIdx.x];
            dst[col] = __float2bfloat16(sum);
          }
        }
        __syncthreads();
      }
    }
  }
}

__global__ void owner_reduce_rows_kernel(
    bf16* grads,
    int64_t width,
    const int32_t* order,
    const int32_t* offsets,
    const int32_t* n_unique,
    float* row_sq_sums,
    UpdateScalars* scalars
) {
  __shared__ float block_sums[kWarpsPerBlock];
  const int lane = threadIdx.x % kWarpThreads;
  const int warp = threadIdx.x / kWarpThreads;
  const int n = *n_unique;
  const int64_t warps = int64_t(gridDim.x) * kWarpsPerBlock;
  float total = 0.f;
  for (int64_t u = int64_t(blockIdx.x) * kWarpsPerBlock + warp; u < n; u += warps) {
    const int32_t begin = __ldg(offsets + u), end = __ldg(offsets + u + 1);
    bf16* dst = grads + int64_t(__ldg(order + begin)) * width;
    float ss = 0.f;
    for (int64_t col0 = 0; col0 < width; col0 += kChunkCols) {
      float acc[kRowVecs][8];
      sum_occurrences(grads, width, order, begin, end, col0, lane, acc);
#pragma unroll
      for (int i = 0; i < kRowVecs; ++i) {
        const int64_t col = col0 + int64_t(lane + i * kWarpThreads) * 8;
        if (col < width) {
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            acc[i][j] = round_bf16(acc[i][j]);
            ss += acc[i][j] * acc[i][j];
          }
          store8(dst + col, pack8(acc[i]));
        }
      }
    }
    ss = warp_sum(ss);
    if (lane == 0) row_sq_sums[u] = ss;
    total += ss;
  }
  if (lane == 0) block_sums[warp] = total;
  __syncthreads();
  if (threadIdx.x == 0) {
    float block_total = 0.f;
    for (int w = 0; w < kWarpsPerBlock; ++w) block_total += block_sums[w];
    atomicAdd(&scalars->total_sq_sum, block_total);
  }
}

__device__ __forceinline__ uint32_t hash_u32(uint32_t x) {
  x = (x ^ (x >> 16)) * 0x7feb352du;
  x = (x ^ (x >> 15)) * 0x846ca68bu;
  return x ^ (x >> 16);
}

__device__ __forceinline__ float
stochastic_round_bf16(float value, uint32_t step, uint32_t row, uint32_t col) {
  const uint32_t base = hash_u32(hash_u32(step) ^ hash_u32(row));
  const uint32_t rand = (col * 0x9e3779b1u + base) >> 16;
  return __uint_as_float((__float_as_uint(value) + rand) & 0xffff0000u);
}

__global__ void rowwise_adagrad_apply_kernel(
    const bf16* grads,
    const int32_t* order,
    const int32_t* offsets,
    const int32_t* unique,
    const int32_t* n_unique,
    const float* row_sq_sums,
    float* row_state,
    int32_t* last_step,
    bf16* table,
    int64_t rows_local,
    int64_t width,
    Ownership ownership,
    int rank,
    UpdateScalars* scalars,
    AdagradParams adagrad
) {
  const float norm = sqrtf(__ldg(&scalars->total_sq_sum));
  const bool apply = isfinite(norm) && __ldg(&scalars->pending) != 0;
  const bool lazy = last_step != nullptr;
  const int32_t clock = lazy ? __ldg(&scalars->step) : 0;
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    scalars->norm = norm;
    scalars->valid = apply ? 1 : 0;
  }
  if (!apply) return;
  const int lane = threadIdx.x % kWarpThreads;
  const int n = *n_unique;
  const int64_t warps = int64_t(gridDim.x) * kWarpsPerBlock;
  for (int64_t u = int64_t(blockIdx.x) * kWarpsPerBlock + threadIdx.x / kWarpThreads; u < n;
       u += warps) {
    const int64_t g = __ldg(unique + u);
    const int64_t r = ownership.local(g);
    if (ownership.owner(g) != rank || r >= rows_local) continue;
    const float ss = __ldg(row_sq_sums + u);
    float accum_factor = adagrad.decay;
    float wd = ss > 0.f ? adagrad.weight_decay_factor : 1.f;
    if (lazy) {
      const float delta = fmaxf(0.f, float(clock - last_step[r]));
      accum_factor = expf(-adagrad.accum_decay_rate * delta);
      wd = (ss > 0.f && delta > adagrad.weight_decay_exempt_window)
               ? expf(-adagrad.weight_decay_rate * delta)
               : 1.f;
    }
    const float accum = row_state[r] * accum_factor + ss * adagrad.inv_emb_width;
    const float step = (-adagrad.lr) * (1.f / (sqrtf(accum) + adagrad.eps));
    const bf16* g_row = grads + int64_t(__ldg(order + __ldg(offsets + u))) * width;
    bf16* x_row = table + r * width;
    for (int64_t v = lane; v < width / 8; v += kWarpThreads) {
      float grad[8], cell[8];
      unpack8(load8(g_row + v * 8), grad);
      unpack8(load8(x_row + v * 8), cell);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        if (adagrad.stochastic_rounding) {
          const float candidate = __fadd_rn(__fmul_rn(cell[j], wd), __fmul_rn(step, grad[j]));
          cell[j] = stochastic_round_bf16(
              candidate, uint32_t(clock), uint32_t(g), uint32_t(v * 8 + j)
          );
        } else {
          const float delta = round_bf16(step * grad[j]);
          const float decayed = round_bf16(cell[j] * wd);
          cell[j] = decayed + delta;
        }
      }
      store8(x_row + v * 8, pack8(cell));
    }
    if (lane == 0) {
      row_state[r] = accum;
      if (lazy) last_step[r] = clock;
    }
  }
}

}

void launch_sanitize_ids(
    const int32_t* ids, int32_t* out, int64_t n, int64_t rows, cudaStream_t stream
) {
  XAI_ASSERT(rows > 0, ": table has no rows");
  if (n == 0) return;
  sanitize_ids_kernel<<<grid_1d(n, kThreads), kThreads, 0, stream>>>(ids, out, n, rows);
  XAI_CUDA_LAUNCH_CHECK();
}

size_t unique_workspace_bytes(int64_t n) {
  check_count(n);
  return carve_unique(nullptr, int(std::max<int64_t>(n, 1))).bytes;
}

void launch_unique_ids(
    const int32_t* ids,
    int64_t n,
    int32_t* unique,
    int32_t* inverse,
    int32_t* order,
    int32_t* offsets,
    int32_t* count,
    void* ws,
    size_t ws_bytes,
    cudaStream_t stream
) {
  check_count(n);
  if (n == 0) {
    XAI_CUDA_CHECK(cudaMemsetAsync(count, 0, sizeof(int32_t), stream));
    XAI_CUDA_CHECK(cudaMemsetAsync(offsets, 0, sizeof(int32_t), stream));
    return;
  }
  UniqueWs w = carve_unique(ws, int(n));
  XAI_ASSERT(w.bytes <= ws_bytes, ": unique workspace too small");
  const int grid = grid_1d(n, kThreads);
  iota_kernel<<<grid, kThreads, 0, stream>>>(w.iota, int(n));
  XAI_CUDA_LAUNCH_CHECK();
  XAI_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
      w.tmp, w.tmp_bytes, ids, w.sorted, w.iota, order, int(n), 0, 32, stream
  ));
  int32_t* flags = w.iota;
  flag_kernel<<<grid, kThreads, 0, stream>>>(w.sorted, flags, int(n));
  XAI_CUDA_LAUNCH_CHECK();
  XAI_CUDA_CHECK(cub::DeviceScan::InclusiveSum(w.tmp, w.tmp_bytes, flags, w.pos, int(n), stream));
  scatter_unique_kernel<<<grid, kThreads, 0, stream>>>(
      w.sorted, order, flags, w.pos, int(n), unique, inverse, offsets, count
  );
  XAI_CUDA_LAUNCH_CHECK();
}

void launch_owner_counts(
    const int32_t* unique,
    const int32_t* n_unique,
    Ownership ownership,
    int32_t* counts,
    cudaStream_t stream
) {
  XAI_ASSERT(ownership.world > 0 && ownership.rows_per_rank > 0, ": invalid ownership");
  owner_counts_kernel<<<grid_1d(ownership.world, kThreads), kThreads, 0, stream>>>(
      unique, n_unique, ownership, counts
  );
  XAI_CUDA_LAUNCH_CHECK();
}

void launch_gather_rows(
    const bf16* table,
    int64_t rows_local,
    int64_t width,
    const int32_t* ids,
    int64_t n,
    Ownership ownership,
    int rank,
    bf16* rows,
    cudaStream_t stream
) {
  check_width(width);
  if (n == 0) return;
  gather_rows_kernel<<<grid_1d(n, kWarpsPerBlock, kSideGrid), kThreads, 0, stream>>>(
      table, rows_local, width, ids, n, ownership, rank, rows
  );
  XAI_CUDA_LAUNCH_CHECK();
}

void launch_scatter_lookup(
    const bf16* rows,
    int64_t width,
    const int32_t* inverse,
    int64_t n,
    bf16* out,
    cudaStream_t stream
) {
  check_width(width);
  if (n == 0) return;
  scatter_lookup_kernel<<<grid_1d(n, kWarpsPerBlock, kMainGrid), kThreads, 0, stream>>>(
      rows, width, inverse, n, out
  );
  XAI_CUDA_LAUNCH_CHECK();
}

void launch_segment_reduce_rows(
    const bf16* grads,
    int64_t width,
    const int32_t* order,
    const int32_t* offsets,
    const int32_t* n_unique,
    int64_t capacity,
    bf16* out,
    cudaStream_t stream
) {
  check_width(width);
  if (capacity == 0) return;
  segment_reduce_rows_kernel<<<
      grid_1d(capacity, kWarpsPerBlock, kMainGrid),
      kThreads,
      0,
      stream>>>(grads, width, order, offsets, n_unique, out);
  XAI_CUDA_LAUNCH_CHECK();
  long_segment_reduce_rows_kernel<<<grid_1d(capacity, 1, 256), kLongThreads, 0, stream>>>(
      grads, width, order, offsets, n_unique, out
  );
  XAI_CUDA_LAUNCH_CHECK();
}

void launch_owner_reduce_rows(
    bf16* grads,
    int64_t width,
    const int32_t* order,
    const int32_t* offsets,
    const int32_t* n_unique,
    int64_t capacity,
    float* row_sq_sums,
    UpdateScalars* scalars,
    cudaStream_t stream
) {
  check_width(width);
  owner_reduce_rows_kernel<<<
      grid_1d(std::max<int64_t>(capacity, 1), kWarpsPerBlock, kSideGrid),
      kThreads,
      0,
      stream>>>(grads, width, order, offsets, n_unique, row_sq_sums, scalars);
  XAI_CUDA_LAUNCH_CHECK();
}

void launch_rowwise_adagrad_apply(
    const bf16* grads,
    const int32_t* order,
    const int32_t* offsets,
    const int32_t* unique,
    const int32_t* n_unique,
    int64_t capacity,
    const float* row_sq_sums,
    float* row_state,
    int32_t* last_step,
    bf16* table,
    int64_t rows_local,
    int64_t width,
    Ownership ownership,
    int rank,
    UpdateScalars* scalars,
    const AdagradParams& adagrad,
    cudaStream_t stream
) {
  check_width(width);
  rowwise_adagrad_apply_kernel<<<
      grid_1d(std::max<int64_t>(capacity, 1), kWarpsPerBlock, kSideGrid),
      kThreads,
      0,
      stream>>>(
      grads,
      order,
      offsets,
      unique,
      n_unique,
      row_sq_sums,
      row_state,
      last_step,
      table,
      rows_local,
      width,
      ownership,
      rank,
      scalars,
      adagrad
  );
  XAI_CUDA_LAUNCH_CHECK();
}

}
