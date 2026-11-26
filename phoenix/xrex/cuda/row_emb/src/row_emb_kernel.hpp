#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define ROW_EMB_HD __host__ __device__ __forceinline__
#else
#define ROW_EMB_HD inline
#endif

namespace xai::kernels::row_emb {

using bf16 = __nv_bfloat16;

struct Ownership {
  int world = 0;
  int64_t rows_per_rank = 0;
  ROW_EMB_HD int64_t rows() const { return int64_t(world) * rows_per_rank; }
  ROW_EMB_HD int owner(int64_t g) const { return int(g / rows_per_rank); }
  ROW_EMB_HD int64_t local(int64_t g) const { return g % rows_per_rank; }
};

struct AdagradParams {
  float lr;
  float eps;
  float decay;
  float inv_emb_width;
  float weight_decay_factor;
  float accum_decay_rate;
  float weight_decay_rate;
  float weight_decay_exempt_window;
  int32_t stochastic_rounding;
};

struct UpdateScalars {
  float norm;
  int32_t valid;
  int32_t pending;
  int32_t step;
  float total_sq_sum;
};

void launch_sanitize_ids(
    const int32_t* ids, int32_t* out, int64_t n, int64_t rows, cudaStream_t stream
);

size_t unique_workspace_bytes(int64_t n);
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
);

void launch_owner_counts(
    const int32_t* unique,
    const int32_t* n_unique,
    Ownership ownership,
    int32_t* counts,
    cudaStream_t stream
);

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
);

void launch_scatter_lookup(
    const bf16* rows,
    int64_t width,
    const int32_t* inverse,
    int64_t n,
    bf16* out,
    cudaStream_t stream
);

void launch_segment_reduce_rows(
    const bf16* grads,
    int64_t width,
    const int32_t* order,
    const int32_t* offsets,
    const int32_t* n_unique,
    int64_t capacity,
    bf16* out,
    cudaStream_t stream
);

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
);

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
);

}
