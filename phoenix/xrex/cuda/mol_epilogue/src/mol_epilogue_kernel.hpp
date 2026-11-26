#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

ffi::Error mol_epilogue(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::BF16> dots,
    ffi::Buffer<ffi::DataType::F32> norms,
    ffi::Buffer<ffi::DataType::F32> ubias,
    ffi::Buffer<ffi::DataType::F32> ibias,
    ffi::Buffer<ffi::DataType::F32> w1,
    ffi::Buffer<ffi::DataType::F32> b1,
    ffi::Buffer<ffi::DataType::F32> w2,
    ffi::Buffer<ffi::DataType::F32> b2,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out,
    int64_t num_users,
    int64_t num_components
);

ffi::Error mol_fused_int8(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::S8> posts,
    ffi::Buffer<ffi::DataType::F32> post_scales,
    ffi::Buffer<ffi::DataType::S8> queries,
    ffi::Buffer<ffi::DataType::F32> query_scales,
    ffi::Buffer<ffi::DataType::F32> norms,
    ffi::Buffer<ffi::DataType::F32> ubias,
    ffi::Buffer<ffi::DataType::F32> ibias,
    ffi::Buffer<ffi::DataType::F32> w1,
    ffi::Buffer<ffi::DataType::F32> b1,
    ffi::Buffer<ffi::DataType::F32> w2,
    ffi::Buffer<ffi::DataType::F32> b2,
    ffi::Result<ffi::Buffer<ffi::DataType::BF16>> out,
    int64_t num_users,
    int64_t col_offset
);

constexpr int64_t kMolGateBwdMaxUsers = 512;
constexpr int64_t kMolGateBwdItemsPerBlock = 64;

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
);
