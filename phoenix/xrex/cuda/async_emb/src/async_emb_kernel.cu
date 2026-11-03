#include <algorithm>
#include <cstdint>

#include "async_emb_kernel.hpp"
#include "cuda_error_utils.hpp"

namespace xai::kernels::async_emb {

namespace {

constexpr int kThreads = 256;
constexpr int kSideGrid = 2048;
constexpr int kMainGrid = 1024;

int grid_for(int64_t work, int max_blocks) {
  int64_t blocks = (work + kThreads - 1) / kThreads;
  if (blocks > max_blocks) {
    blocks = max_blocks;
  }
  return static_cast<int>(blocks < 1 ? 1 : blocks);
}

__device__ __forceinline__ void copy_row(
    __nv_bfloat16* dst, const __nv_bfloat16* src, int64_t width, int lane, int num_lanes
) {
  if (width % 8 == 0) {
    const int4* src_vec = reinterpret_cast<const int4*>(src);
    int4* dst_vec = reinterpret_cast<int4*>(dst);
    for (int64_t unit = lane; unit < width / 8; unit += num_lanes) {
      dst_vec[unit] = src_vec[unit];
    }
  } else if (width % 4 == 0) {
    const int2* src_vec = reinterpret_cast<const int2*>(src);
    int2* dst_vec = reinterpret_cast<int2*>(dst);
    for (int64_t unit = lane; unit < width / 4; unit += num_lanes) {
      dst_vec[unit] = src_vec[unit];
    }
  } else {
    for (int64_t col = lane; col < width; col += num_lanes) {
      dst[col] = src[col];
    }
  }
}

template <typename T, typename Combine>
__device__ __forceinline__ T block_reduce(T value, Combine combine) {
  __shared__ T partials[kThreads];
  partials[threadIdx.x] = value;
  __syncthreads();
  for (int span = blockDim.x / 2; span > 0; span >>= 1) {
    if (threadIdx.x < span) {
      partials[threadIdx.x] = combine(partials[threadIdx.x], partials[threadIdx.x + span]);
    }
    __syncthreads();
  }
  return partials[0];
}

__device__ __forceinline__ int64_t table_row_of(int32_t token, int64_t vocab_rows) {
  const int64_t row = token < 0 ? token + vocab_rows : token;
  return row >= 0 && row < vocab_rows ? row : -1;
}

__device__ __forceinline__ bool segment_in_range(int32_t segment, int64_t num_unique) {
  return segment >= 0 && segment < num_unique;
}

__device__ __forceinline__ uint32_t hash_u32(uint32_t x) {
  x = (x ^ (x >> 16)) * 0x7feb352du;
  x = (x ^ (x >> 15)) * 0x846ca68bu;
  return x ^ (x >> 16);
}

__device__ __forceinline__ __nv_bfloat16
stochastic_round_bf16(float value, uint32_t step, uint32_t row, uint32_t col) {
  const uint32_t base = hash_u32(hash_u32(step) ^ hash_u32(row));
  const uint32_t rand = (col * 0x9e3779b1u + base) >> 16;
  __nv_bfloat16_raw raw;
  raw.x = static_cast<uint16_t>((__float_as_uint(value) + rand) >> 16);
  return raw;
}

}

__global__ void lookup_dispatch(
    const int32_t* token_ids_all,
    const __nv_bfloat16* table_shard,
    __nv_bfloat16* send_slices,
    int64_t tokens_per_rank,
    int64_t shard_width,
    int64_t vocab_rows,
    int nranks,
    int64_t lanes_per_row
) {
  const int64_t total = tokens_per_rank * nranks * lanes_per_row;
  const int64_t grid_stride = gridDim.x * int64_t(blockDim.x);
  for (int64_t item = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; item < total;
       item += grid_stride) {
    const int64_t token = item / lanes_per_row;
    const int lane = static_cast<int>(item - token * lanes_per_row);
    int64_t row = token_ids_all[token];
    if (row < 0) {
      row += vocab_rows;
    }
    if (row < 0) {
      row = 0;
    }
    if (row >= vocab_rows) {
      row = vocab_rows - 1;
    }
    const __nv_bfloat16* src = table_shard + row * shard_width;
    const int dst_rank = static_cast<int>(token / tokens_per_rank);
    const int64_t slot = token - int64_t(dst_rank) * tokens_per_rank;
    __nv_bfloat16* dst =
        send_slices + dst_rank * (tokens_per_rank * shard_width) + slot * shard_width;
    copy_row(dst, src, shard_width, lane, static_cast<int>(lanes_per_row));
  }
}

void launch_lookup_dispatch(
    const int32_t* token_ids_all,
    const __nv_bfloat16* table_shard,
    __nv_bfloat16* send_slices,
    const SliceLayout& slices,
    int64_t vocab_rows,
    cudaStream_t stream
) {
  const int64_t lanes = slices.shard_width <= 16 ? 1 : 32;
  const int64_t work = slices.tokens_per_rank * int64_t(slices.nranks) * lanes;
  lookup_dispatch<<<grid_for(work, kSideGrid), kThreads, 0, stream>>>(
      token_ids_all,
      table_shard,
      send_slices,
      slices.tokens_per_rank,
      slices.shard_width,
      vocab_rows,
      slices.nranks,
      lanes
  );
  XAI_CUDA_LAUNCH_CHECK();
}

__global__ void lookup_combine(
    const __nv_bfloat16* recv_slices,
    __nv_bfloat16* embeddings_out,
    int64_t tokens_per_rank,
    int64_t shard_width,
    int nranks
) {
  const int64_t emb_width = int64_t(nranks) * shard_width;
  const int64_t thread_id = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
  const int64_t grid_stride = gridDim.x * int64_t(blockDim.x);
  if (shard_width % 8 == 0) {
    const int64_t units_per_slice = shard_width / 8;
    const int64_t units_total = tokens_per_rank * emb_width / 8;
    const int4* src = reinterpret_cast<const int4*>(recv_slices);
    int4* dst = reinterpret_cast<int4*>(embeddings_out);
    for (int64_t unit = thread_id; unit < units_total; unit += grid_stride) {
      const int64_t slice_unit = unit % units_per_slice;
      const int64_t token_rank = unit / units_per_slice;
      const int rank = static_cast<int>(token_rank % nranks);
      const int64_t token = token_rank / nranks;
      dst[token * (emb_width / 8) + int64_t(rank) * units_per_slice + slice_unit] =
          src[int64_t(rank) * (tokens_per_rank * units_per_slice) + token * units_per_slice +
              slice_unit];
    }
  } else {
    const int64_t total = tokens_per_rank * emb_width;
    for (int64_t elem = thread_id; elem < total; elem += grid_stride) {
      const int64_t col = elem % emb_width;
      const int64_t token = elem / emb_width;
      const int rank = static_cast<int>(col / shard_width);
      const int64_t slice_col = col - int64_t(rank) * shard_width;
      embeddings_out[elem] = recv_slices
          [int64_t(rank) * (tokens_per_rank * shard_width) + token * shard_width + slice_col];
    }
  }
}

void launch_lookup_combine(
    const __nv_bfloat16* recv_slices,
    __nv_bfloat16* embeddings_out,
    const SliceLayout& slices,
    cudaStream_t stream
) {
  const int64_t work = slices.tokens_per_rank * int64_t(slices.nranks) * slices.shard_width /
                       (slices.shard_width % 8 == 0 ? 8 : 1);
  lookup_combine<<<grid_for(work, kMainGrid), kThreads, 0, stream>>>(
      recv_slices, embeddings_out, slices.tokens_per_rank, slices.shard_width, slices.nranks
  );
  XAI_CUDA_LAUNCH_CHECK();
}

__global__ void grad_dispatch(
    const __nv_bfloat16* grads,
    __nv_bfloat16* send_slices,
    int64_t tokens_per_rank,
    int64_t shard_width,
    int nranks
) {
  const int64_t emb_width = int64_t(nranks) * shard_width;
  const int64_t thread_id = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
  const int64_t grid_stride = gridDim.x * int64_t(blockDim.x);
  if (shard_width % 8 == 0) {
    const int64_t units_per_slice = shard_width / 8;
    const int64_t units_total = tokens_per_rank * emb_width / 8;
    const int4* src = reinterpret_cast<const int4*>(grads);
    for (int64_t unit = thread_id; unit < units_total; unit += grid_stride) {
      const int64_t slice_unit = unit % units_per_slice;
      const int64_t token_rank = unit / units_per_slice;
      const int rank = static_cast<int>(token_rank % nranks);
      const int64_t token = token_rank / nranks;
      int4* dst =
          reinterpret_cast<int4*>(send_slices + int64_t(rank) * (tokens_per_rank * shard_width));
      dst[token * units_per_slice + slice_unit] =
          src[token * (emb_width / 8) + int64_t(rank) * units_per_slice + slice_unit];
    }
  } else {
    const int64_t total = tokens_per_rank * emb_width;
    for (int64_t elem = thread_id; elem < total; elem += grid_stride) {
      const int64_t col = elem % emb_width;
      const int64_t token = elem / emb_width;
      const int rank = static_cast<int>(col / shard_width);
      const int64_t slice_col = col - int64_t(rank) * shard_width;
      __nv_bfloat16* dst = send_slices + int64_t(rank) * (tokens_per_rank * shard_width);
      dst[token * shard_width + slice_col] = grads[elem];
    }
  }
}

void launch_grad_dispatch(
    const __nv_bfloat16* grads,
    __nv_bfloat16* send_slices,
    const SliceLayout& slices,
    cudaStream_t stream
) {
  const int64_t work = slices.tokens_per_rank * int64_t(slices.nranks) * slices.shard_width /
                       (slices.shard_width % 8 == 0 ? 8 : 1);
  grad_dispatch<<<grid_for(work, kMainGrid), kThreads, 0, stream>>>(
      grads, send_slices, slices.tokens_per_rank, slices.shard_width, slices.nranks
  );
  XAI_CUDA_LAUNCH_CHECK();
}

__global__ void update_row_bound(
    const int32_t* unique_tokens,
    const int32_t* segment_ids_all,
    UpdateScalars* scalars,
    int64_t n_tokens_global,
    int64_t num_unique,
    int64_t vocab_rows
) {
  const int64_t thread_id = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
  const int64_t grid_stride = gridDim.x * int64_t(blockDim.x);
  unsigned long long row_bound = 0;
  for (int64_t row = thread_id; row < num_unique; row += grid_stride) {
    if (table_row_of(__ldg(unique_tokens + row), vocab_rows) >= 0) {
      row_bound = row + 1;
    }
  }
  for (int64_t token = thread_id; token < n_tokens_global; token += grid_stride) {
    const int32_t segment = __ldg(segment_ids_all + token);
    if (segment_in_range(segment, num_unique)) {
      row_bound = max(row_bound, segment + 1ULL);
    }
  }
  const unsigned long long block_row_bound =
      block_reduce(row_bound, [](unsigned long long a, unsigned long long b) { return max(a, b); });
  if (threadIdx.x == 0) {
    atomicMax(&scalars->row_bound, block_row_bound);
  }
}

void launch_update_row_bound(
    const int32_t* unique_tokens,
    const int32_t* segment_ids_all,
    UpdateScalars* scalars,
    const SliceLayout& slices,
    int64_t num_unique,
    int64_t vocab_rows,
    cudaStream_t stream
) {
  const int64_t n_tokens_global = slices.tokens_per_rank * int64_t(slices.nranks);
  XAI_CUDA_CHECK(cudaMemsetAsync(&scalars->row_bound, 0, sizeof(scalars->row_bound), stream));
  update_row_bound<<<
      grid_for(std::max(n_tokens_global, num_unique), kSideGrid),
      kThreads,
      0,
      stream>>>(unique_tokens, segment_ids_all, scalars, n_tokens_global, num_unique, vocab_rows);
  XAI_CUDA_LAUNCH_CHECK();
}

__global__ void zero_update_rows(
    float* grad_accum, const UpdateScalars* scalars, int64_t shard_width
) {
  const int64_t thread_id = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
  const int64_t grid_stride = gridDim.x * int64_t(blockDim.x);
  const int64_t total = int64_t(scalars->row_bound) * shard_width;
  if (total % 4 == 0) {
    float4* grad_accum_vec = reinterpret_cast<float4*>(grad_accum);
    for (int64_t unit = thread_id; unit < total / 4; unit += grid_stride) {
      grad_accum_vec[unit] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
  } else {
    for (int64_t elem = thread_id; elem < total; elem += grid_stride) {
      grad_accum[elem] = 0.f;
    }
  }
}

void launch_zero_update_rows(
    float* grad_accum, const UpdateScalars* scalars, int64_t shard_width, cudaStream_t stream
) {
  zero_update_rows<<<kSideGrid, kThreads, 0, stream>>>(grad_accum, scalars, shard_width);
  XAI_CUDA_LAUNCH_CHECK();
}

__global__ void grad_segment_sum(
    const __nv_bfloat16* routed_grads,
    const int32_t* segment_ids_all,
    float* grad_accum,
    float* row_sq_sums,
    int64_t n_tokens_global,
    int64_t shard_width,
    int64_t num_unique
) {
  const int64_t thread_id = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
  const int64_t grid_stride = gridDim.x * int64_t(blockDim.x);
  if (thread_id == 0) {
    row_sq_sums[num_unique] = 0.f;
  }
  const int64_t total = n_tokens_global * shard_width;
  for (int64_t elem = thread_id; elem < total; elem += grid_stride) {
    const int64_t token = elem / shard_width;
    const int64_t col = elem - token * shard_width;
    const int32_t segment = __ldg(segment_ids_all + token);
    if (!segment_in_range(segment, num_unique)) {
      continue;
    }
    atomicAdd(
        grad_accum + int64_t(segment) * shard_width + col,
        __bfloat162float(__ldg(routed_grads + elem))
    );
  }
}

__global__ void grad_row_stats(
    const float* grad_accum,
    float* row_sq_sums,
    const UpdateScalars* scalars,
    int64_t num_unique,
    int64_t shard_width
) {
  const int64_t thread_id = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
  const int64_t grid_stride = gridDim.x * int64_t(blockDim.x);
  const int64_t row_bound = scalars->row_bound;
  float thread_total = 0.f;
  for (int64_t row = thread_id; row < num_unique; row += grid_stride) {
    if (row >= row_bound) {
      row_sq_sums[row] = 0.f;
      continue;
    }
    float sq_sum = 0.f;
    const float* row_vals = grad_accum + row * shard_width;
    for (int64_t col = 0; col < shard_width; ++col) {
      const float grad = __bfloat162float(__float2bfloat16(row_vals[col]));
      sq_sum += grad * grad;
    }
    row_sq_sums[row] = sq_sum;
    thread_total += sq_sum;
  }
  const float block_total = block_reduce(thread_total, [](float a, float b) { return a + b; });
  if (threadIdx.x == 0) {
    atomicAdd(row_sq_sums + num_unique, block_total);
  }
}

void launch_grad_segment_sum(
    const __nv_bfloat16* routed_grads,
    const int32_t* segment_ids_all,
    float* grad_accum,
    float* row_sq_sums,
    const UpdateScalars* scalars,
    const SliceLayout& slices,
    int64_t num_unique,
    cudaStream_t stream
) {
  const int64_t n_tokens_global = slices.tokens_per_rank * int64_t(slices.nranks);
  grad_segment_sum<<<
      grid_for(n_tokens_global * slices.shard_width, kSideGrid),
      kThreads,
      0,
      stream>>>(
      routed_grads,
      segment_ids_all,
      grad_accum,
      row_sq_sums,
      n_tokens_global,
      slices.shard_width,
      num_unique
  );
  XAI_CUDA_LAUNCH_CHECK();
  grad_row_stats<<<grid_for(num_unique, kSideGrid), kThreads, 0, stream>>>(
      grad_accum, row_sq_sums, scalars, num_unique, slices.shard_width
  );
  XAI_CUDA_LAUNCH_CHECK();
}

template <int kShardWidth>
__global__ void rowwise_adagrad_apply(
    const float* grad_accum,
    const float* row_sq_sums,
    const int32_t* unique_tokens,
    float* row_state,
    int32_t* last_step,
    __nv_bfloat16* table_shard,
    UpdateScalars* scalars,
    AdagradParams adagrad,
    int64_t num_unique,
    int64_t vocab_rows
) {
  const float norm = sqrtf(__ldg(row_sq_sums + num_unique));
  const bool apply = isfinite(norm) && __ldg(&scalars->pending) != 0;
  const bool lazy = last_step != nullptr;
  const int32_t clock = lazy ? __ldg(&scalars->step) : 0;
  const int64_t thread_id = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
  if (thread_id == 0) {
    scalars->norm = norm;
    scalars->valid = apply ? 1 : 0;
  }
  if (!apply) {
    return;
  }
  const int64_t row_bound = scalars->row_bound;
  const int64_t grid_stride = gridDim.x * int64_t(blockDim.x);
  for (int64_t row = thread_id; row < row_bound; row += grid_stride) {
    const int64_t token = table_row_of(__ldg(unique_tokens + row), vocab_rows);
    if (token < 0) {
      continue;
    }
    const float row_sq_sum = __ldg(row_sq_sums + row);
    float accum_factor = adagrad.decay;
    float wd;
    if (lazy) {
      const float delta = fmaxf(0.f, float(clock - last_step[token]));
      accum_factor = expf(-adagrad.accum_decay_rate * delta);
      wd = (row_sq_sum > 0.f && delta > adagrad.weight_decay_exempt_window)
               ? expf(-adagrad.weight_decay_rate * delta)
               : 1.f;
      last_step[token] = clock;
    } else {
      wd = row_sq_sum > 0.f ? adagrad.weight_decay_factor : 1.f;
    }
    const float accum =
        __fmaf_rn(row_state[token], accum_factor, __fmul_rn(row_sq_sum, adagrad.inv_emb_width));
    const float step = (-adagrad.lr) * (1.f / (sqrtf(accum) + adagrad.eps));
    __nv_bfloat16* table_row = table_shard + token * kShardWidth;
    const float* grad_row = grad_accum + row * kShardWidth;
    alignas(16) __nv_bfloat16 cells[kShardWidth];
    alignas(16) float grads[kShardWidth];
    copy_row(cells, table_row, kShardWidth, 0, 1);
    if constexpr (kShardWidth % 4 == 0) {
#pragma unroll
      for (int unit = 0; unit < kShardWidth / 4; ++unit) {
        reinterpret_cast<float4*>(grads)[unit] =
            __ldg(reinterpret_cast<const float4*>(grad_row) + unit);
      }
    } else {
#pragma unroll
      for (int col = 0; col < kShardWidth; ++col) {
        grads[col] = __ldg(grad_row + col);
      }
    }
#pragma unroll
    for (int col = 0; col < kShardWidth; ++col) {
      const float grad = __bfloat162float(__float2bfloat16(grads[col]));

      if (adagrad.stochastic_rounding) {
        const float candidate =
            __fadd_rn(__fmul_rn(__bfloat162float(cells[col]), wd), __fmul_rn(step, grad));

        cells[col] = stochastic_round_bf16(
            candidate,
            static_cast<uint32_t>(clock),
            static_cast<uint32_t>(token),
            static_cast<uint32_t>(adagrad.col_offset + col)
        );
      } else {
        const float delta = __bfloat162float(__float2bfloat16(step * grad));
        const float decayed = __bfloat162float(__float2bfloat16(__bfloat162float(cells[col]) * wd));
        cells[col] = __float2bfloat16(decayed + delta);
      }
    }
    copy_row(table_row, cells, kShardWidth, 0, 1);
    row_state[token] = accum;
  }
}

void launch_rowwise_adagrad_apply(
    const float* grad_accum,
    const float* row_sq_sums,
    const int32_t* unique_tokens,
    float* row_state,
    int32_t* last_step,
    __nv_bfloat16* table_shard,
    int64_t vocab_rows,
    UpdateScalars* scalars,
    const AdagradParams& adagrad,
    int64_t num_unique,
    int64_t shard_width,
    cudaStream_t stream
) {
  auto launch = [&](auto kernel) {
    kernel<<<grid_for(num_unique, kSideGrid), kThreads, 0, stream>>>(
        grad_accum,
        row_sq_sums,
        unique_tokens,
        row_state,
        last_step,
        table_shard,
        scalars,
        adagrad,
        num_unique,
        vocab_rows
    );
    XAI_CUDA_LAUNCH_CHECK();
  };
  static_assert(kMaxShardWidth == 32, "add a case for every divisor of kMaxShardWidth");
  switch (shard_width) {
    case 1:
      return launch(rowwise_adagrad_apply<1>);
    case 2:
      return launch(rowwise_adagrad_apply<2>);
    case 4:
      return launch(rowwise_adagrad_apply<4>);
    case 8:
      return launch(rowwise_adagrad_apply<8>);
    case 16:
      return launch(rowwise_adagrad_apply<16>);
    case 32:
      return launch(rowwise_adagrad_apply<32>);
    default:
      XAI_ASSERT(
          false,
          ": no rowwise_adagrad_apply instance for shard_width " + std::to_string(shard_width)
      );
  }
}

}
