#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "nanobind/nanobind.h"
#include "nanobind/stl/string.h"
#include "async_emb_comm.hpp"
#include "async_emb_kernel.hpp"
#include "comm_utils.h"
#include "cuda_error_utils.hpp"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
namespace nb = nanobind;

namespace xai::kernels::async_emb {

namespace common = xai::kernels::common;

namespace {

std::mutex context_init_mu;

ffi::Error invalid(const std::string& message) {
  return ffi::Error(ffi::ErrorCode::kInvalidArgument, message);
}

template <typename Fn>
ffi::Error ffiGuard(Fn&& fn) noexcept {
  try {
    return fn();
  } catch (const std::exception& error) {
    return ffi::Error::Internal(error.what());
  } catch (...) {
    return ffi::Error::Internal("unknown exception in async_emb FFI handler");
  }
}

std::shared_ptr<AsyncEmbContext> findContext(int64_t context_id) {
  return std::dynamic_pointer_cast<AsyncEmbContext>(
      common::CommContextRegistry::getInstance().getContext(context_id)
  );
}

std::shared_ptr<AsyncEmbContext> readyContext(int64_t context_id) {
  auto ctx = findContext(context_id);
  if (ctx == nullptr) {
    return nullptr;
  }
  ctx->ensureHealthy();
  if (!ctx->isInitialized()) {
    return nullptr;
  }
  return ctx;
}

ffi::Error initializeContext(ffi::Dictionary attrs, ffi::CollectiveParamsPartial params) {
  std::lock_guard<std::mutex> init_lock(context_init_mu);
  auto context_id = attrs.get<int64_t>("context_id");
  auto group_size = attrs.get<int64_t>("group_size");
  auto flatten_replicas = attrs.get<ffi::Span<const int64_t>>("flatten_replicas");
  if (!context_id.has_value() || !group_size.has_value() || !flatten_replicas.has_value()) {
    return invalid("async_emb: missing context attrs");
  }
  auto existing = common::CommContextRegistry::getInstance().getContext(*context_id);
  if (existing != nullptr) {
    if (existing->isInitialized()) {
      return ffi::Error::Success();
    }
    return invalid("async_emb: aborted context requires process restart");
  }

  bool ok = true;
  auto getInt = [&](const char* name) -> int64_t {
    auto value = attrs.get<int64_t>(name);
    ok &= value.has_value();
    return value.has_value() ? *value : 0;
  };
  PipelineSpec spec;
  spec.tokens_per_rank = getInt("tokens_per_rank");
  spec.shard_width = getInt("shard_width");
  spec.num_unique = getInt("num_unique");
  spec.emb_width = getInt("emb_width");
  int64_t num_devices_per_node = getInt("num_devices_per_node");
  if (!ok) {
    return invalid("async_emb: missing pipeline attrs");
  }
  if (*group_size <= 0 || flatten_replicas->size() == 0 ||
      flatten_replicas->size() % size_t(*group_size) != 0) {
    return invalid("async_emb: invalid replica groups");
  }
  if (num_devices_per_node <= 0) {
    return invalid("async_emb: num_devices_per_node must be positive");
  }
  if (spec.tokens_per_rank <= 0 || spec.shard_width <= 0 || spec.num_unique <= 0 ||
      spec.emb_width <= 0 || spec.shard_width > std::numeric_limits<int64_t>::max() / *group_size ||
      spec.emb_width != spec.shard_width * *group_size) {
    return invalid("async_emb: invalid pipeline dimensions");
  }

  int64_t local_gpu_index = params.global_device_id % num_devices_per_node;
  if (local_gpu_index < 0 || local_gpu_index >= num_devices_per_node) {
    return invalid("async_emb: unsupported local gpu index");
  }

  std::vector<int64_t> replicas(flatten_replicas->begin(), flatten_replicas->end());
  common::GroupInfo info = common::findGroupInfo(*group_size, replicas, params.global_device_id);
  if (info.group_id == -1) {
    return invalid("async_emb: device not found in replica groups");
  }
  auto ctx = std::make_shared<AsyncEmbContext>(uint64_t(*context_id), int(*group_size), spec);
  if (auto err = ctx->init(info, local_gpu_index)) {
    return invalid(*err);
  }
  common::CommContextRegistry::getInstance().registerContext(ctx, uint64_t(*context_id));
  return ffi::Error::Success();
}

ffi::Error LookupStartInit(
    cudaStream_t,
    ffi::CollectiveParamsPartial params,
    ffi::AnyBuffer,
    ffi::AnyBuffer,
    ffi::AnyBuffer,
    ffi::Dictionary attrs,
    ffi::Result<ffi::AnyBuffer>,
    ffi::Result<ffi::AnyBuffer>
) {
  return ffiGuard([&] { return initializeContext(attrs, params); });
}

ffi::Error LookupStart(
    cudaStream_t stream,
    ffi::CollectiveParamsPartial,
    ffi::AnyBuffer token_ids,
    ffi::AnyBuffer table,
    ffi::AnyBuffer ,
    ffi::Dictionary attrs,
    ffi::Result<ffi::AnyBuffer> table_out,
    ffi::Result<ffi::AnyBuffer> pin
) {
  return ffiGuard([&]() -> ffi::Error {
    auto context_id = attrs.get<int64_t>("context_id");
    if (!context_id.has_value()) {
      return invalid("lookup_start: missing context_id");
    }
    auto ctx = readyContext(*context_id);
    if (ctx == nullptr) {
      return invalid("lookup_start: context not initialized");
    }
    const auto& spec = ctx->spec();
    if (token_ids.element_type() != ffi::DataType::S32) {
      return invalid("lookup_start: ids must be s32");
    }
    if (table.element_type() != ffi::DataType::BF16 || table.dimensions().size() != 2 ||
        table.dimensions()[1] != spec.shard_width) {
      return invalid("lookup_start: bad table shape/dtype");
    }
    if (token_ids.element_count() != size_t(spec.tokens_per_rank)) {
      return invalid("lookup_start: bad ids count");
    }
    if (table_out->untyped_data() != table.untyped_data()) {
      return invalid("lookup_start: table_out must alias table (check input_output_aliases)");
    }
    XAI_CUDA_CHECK(cudaMemsetAsync(pin->untyped_data(), 0, pin->size_bytes(), stream));
    LookupJob job{
        static_cast<const int32_t*>(token_ids.untyped_data()),
        static_cast<const __nv_bfloat16*>(table.untyped_data()),
        table.dimensions()[0]
    };
    ctx->armLookup(job, stream);
    return ffi::Error::Success();
  });
}

ffi::Error LookupDone(
    cudaStream_t stream,
    ffi::AnyBuffer ,
    ffi::Dictionary attrs,
    ffi::Result<ffi::AnyBuffer> embeddings
) {
  return ffiGuard([&]() -> ffi::Error {
    auto context_id = attrs.get<int64_t>("context_id");
    if (!context_id.has_value()) {
      return invalid("lookup_done: missing context_id");
    }
    auto ctx = readyContext(*context_id);
    if (ctx == nullptr) {
      return invalid("lookup_done: context not initialized");
    }
    const auto& spec = ctx->spec();
    if (embeddings->element_count() !=
        size_t(spec.tokens_per_rank * spec.shard_width * ctx->worldSize())) {
      return invalid("lookup_done: bad output count");
    }
    ctx->streamConsumeDone(stream, AsyncEmbContext::Operation::Lookup);
    launch_lookup_combine(
        reinterpret_cast<const __nv_bfloat16*>(ctx->arena() + ctx->layout().recv),
        static_cast<__nv_bfloat16*>(embeddings->untyped_data()),
        SliceLayout{spec.tokens_per_rank, spec.shard_width, ctx->worldSize()},
        stream
    );
    return ffi::Error::Success();
  });
}

ffi::Error StageUpdate(
    cudaStream_t stream,
    ffi::AnyBuffer grads,
    ffi::AnyBuffer segment_ids,
    ffi::AnyBuffer unique_tokens,
    ffi::AnyBuffer pending,
    ffi::AnyBuffer ,
    ffi::Dictionary attrs,
    ffi::Result<ffi::AnyBuffer> pin
) {
  return ffiGuard([&]() -> ffi::Error {
    auto context_id = attrs.get<int64_t>("context_id");
    if (!context_id.has_value()) {
      return invalid("stage_update: missing context_id");
    }
    auto ctx = readyContext(*context_id);
    if (ctx == nullptr) {
      return invalid("stage_update: context not initialized");
    }
    const auto& spec = ctx->spec();
    if (grads.element_type() != ffi::DataType::BF16 || grads.dimensions().size() != 2 ||
        grads.dimensions()[0] != spec.tokens_per_rank || grads.dimensions()[1] != spec.emb_width) {
      return invalid("stage_update: bad grads shape/dtype");
    }
    if (segment_ids.element_type() != ffi::DataType::S32 ||
        segment_ids.element_count() != size_t(spec.tokens_per_rank)) {
      return invalid("stage_update: bad segment ids");
    }
    if (unique_tokens.element_type() != ffi::DataType::S32 ||
        unique_tokens.element_count() != size_t(spec.num_unique)) {
      return invalid("stage_update: bad unique tokens");
    }
    if (pending.element_type() != ffi::DataType::S32 || pending.element_count() != 1) {
      return invalid("stage_update: pending must be s32[1]");
    }
    XAI_CUDA_CHECK(cudaMemsetAsync(pin->untyped_data(), 0, pin->size_bytes(), stream));
    ctx->stageUpdate(
        static_cast<const __nv_bfloat16*>(grads.untyped_data()),
        static_cast<const int32_t*>(segment_ids.untyped_data()),
        static_cast<const int32_t*>(unique_tokens.untyped_data()),
        static_cast<const int32_t*>(pending.untyped_data()),
        stream
    );
    return ffi::Error::Success();
  });
}

ApplyUpdateRule makeRowwiseAdagradRule(
    AdagradParams params, float* row_state, int32_t* last_step = nullptr
) {
  return [params,
          row_state,
          last_step](const ReducedGradients& reduced, const UpdateJob& job, cudaStream_t stream) {
    launch_rowwise_adagrad_apply(
        reduced.row_grads,
        reduced.row_sq_sums,
        reduced.unique_tokens,
        row_state,
        last_step,
        job.table,
        job.vocab_rows,
        reduced.scalars,
        params,
        reduced.num_unique,
        reduced.shard_width,
        stream
    );
  };
}

std::once_flag rowwise_adagrad_warmup_once;

ffi::Error initializeRowwiseAdagrad(
    ffi::Dictionary attrs, ffi::CollectiveParamsPartial params, const char* operation
) {
  ffi::Error error = initializeContext(attrs, params);
  if (error.failure()) {
    return error;
  }
  auto ctx = readyContext(*attrs.get<int64_t>("context_id"));
  if (ctx == nullptr) {
    return invalid(std::string(operation) + ": context not initialized");
  }
  std::call_once(rowwise_adagrad_warmup_once, [&] {
    ctx->warmupUpdateRule(
        makeRowwiseAdagradRule(AdagradParams{0.f, 1.f, 1.f, 1.f, 1.f, 0.f, 0.f, 0.f, 0, 0}, nullptr)
    );
  });
  return ffi::Error::Success();
}

ffi::Error RowwiseAdagradUpdateStartInit(
    cudaStream_t,
    ffi::CollectiveParamsPartial params,
    ffi::AnyBuffer,
    ffi::AnyBuffer,
    ffi::AnyBuffer,
    ffi::Dictionary attrs,
    ffi::Result<ffi::AnyBuffer>,
    ffi::Result<ffi::AnyBuffer>,
    ffi::Result<ffi::AnyBuffer>
) {
  return ffiGuard([&] {
    return initializeRowwiseAdagrad(attrs, params, "rowwise_adagrad_update_start");
  });
}

ffi::Error validateRowwiseAdagradBuffers(
    const PipelineSpec& spec, const ffi::AnyBuffer& table, const ffi::AnyBuffer& row_state
) {
  if (kMaxShardWidth % spec.shard_width != 0) {
    return invalid(
        "rowwise_adagrad_update_start: shard_width must divide " + std::to_string(kMaxShardWidth)
    );
  }
  if (table.element_type() != ffi::DataType::BF16 || table.dimensions().size() != 2 ||
      table.dimensions()[1] != spec.shard_width) {
    return invalid("rowwise_adagrad_update_start: bad table");
  }
  if (row_state.element_type() != ffi::DataType::F32 ||
      row_state.element_count() != size_t(table.dimensions()[0])) {
    return invalid("rowwise_adagrad_update_start: bad row state");
  }
  return ffi::Error::Success();
}

bool bufferAliased(const ffi::AnyBuffer& in, ffi::Result<ffi::AnyBuffer>& out) {
  return out->untyped_data() == in.untyped_data();
}

ffi::Error RowwiseAdagradUpdateStart(
    cudaStream_t stream,
    ffi::CollectiveParamsPartial,
    ffi::AnyBuffer table,
    ffi::AnyBuffer row_state,
    ffi::AnyBuffer ,
    ffi::Dictionary attrs,
    ffi::Result<ffi::AnyBuffer> table_out,
    ffi::Result<ffi::AnyBuffer> state_out,
    ffi::Result<ffi::AnyBuffer> pin
) {
  return ffiGuard([&]() -> ffi::Error {
    auto context_id = attrs.get<int64_t>("context_id");
    if (!context_id.has_value()) {
      return invalid("rowwise_adagrad_update_start: missing context_id");
    }
    auto ctx = readyContext(*context_id);
    if (ctx == nullptr) {
      return invalid("rowwise_adagrad_update_start: context not initialized");
    }
    auto learning_rate = attrs.get<double>("learning_rate");
    auto eps = attrs.get<double>("eps");
    auto decay_factor = attrs.get<double>("decay_factor");
    auto weight_decay_factor = attrs.get<double>("weight_decay_factor");
    if (!learning_rate.has_value() || !eps.has_value() || !decay_factor.has_value() ||
        !weight_decay_factor.has_value()) {
      return invalid("rowwise_adagrad_update_start: missing optimizer attrs");
    }
    const float lr_f = float(*learning_rate);
    const float eps_f = float(*eps);
    const float decay_f = float(*decay_factor);
    const float wd_f = float(*weight_decay_factor);
    if (!std::isfinite(lr_f) || !std::isfinite(eps_f) || !std::isfinite(decay_f) || eps_f <= 0.f ||
        decay_f < 0.f || decay_f > 1.f || !std::isfinite(wd_f) || wd_f <= 0.f || wd_f > 1.f) {
      return invalid("rowwise_adagrad_update_start: invalid optimizer parameters");
    }
    ffi::Error buffers_error = validateRowwiseAdagradBuffers(ctx->spec(), table, row_state);
    if (buffers_error.failure()) {
      return buffers_error;
    }
    if (!bufferAliased(table, table_out) || !bufferAliased(row_state, state_out)) {
      return invalid(
          "rowwise_adagrad_update_start: outputs must alias inputs (check input_output_aliases)"
      );
    }
    XAI_CUDA_CHECK(cudaMemsetAsync(pin->untyped_data(), 0, pin->size_bytes(), stream));
    const auto& spec = ctx->spec();
    const AdagradParams adagrad{
        lr_f, eps_f, decay_f, 1.f / float(spec.emb_width), wd_f, 0.f, 0.f, 0.f, 0, 0
    };
    UpdateJob job{static_cast<__nv_bfloat16*>(table.untyped_data()), table.dimensions()[0]};
    ctx->armUpdate(
        job, makeRowwiseAdagradRule(adagrad, static_cast<float*>(row_state.untyped_data())), stream
    );
    return ffi::Error::Success();
  });
}

ffi::Error RowwiseAdagradLazyUpdateStartInit(
    cudaStream_t,
    ffi::CollectiveParamsPartial params,
    ffi::AnyBuffer,
    ffi::AnyBuffer,
    ffi::AnyBuffer,
    ffi::AnyBuffer,
    ffi::AnyBuffer,
    ffi::Dictionary attrs,
    ffi::Result<ffi::AnyBuffer>,
    ffi::Result<ffi::AnyBuffer>,
    ffi::Result<ffi::AnyBuffer>,
    ffi::Result<ffi::AnyBuffer>
) {
  return ffiGuard([&] {
    return initializeRowwiseAdagrad(attrs, params, "rowwise_adagrad_lazy_update_start");
  });
}

ffi::Error RowwiseAdagradLazyUpdateStart(
    cudaStream_t stream,
    ffi::CollectiveParamsPartial,
    ffi::AnyBuffer table,
    ffi::AnyBuffer row_state,
    ffi::AnyBuffer last_step,
    ffi::AnyBuffer logical_step,
    ffi::AnyBuffer ,
    ffi::Dictionary attrs,
    ffi::Result<ffi::AnyBuffer> table_out,
    ffi::Result<ffi::AnyBuffer> state_out,
    ffi::Result<ffi::AnyBuffer> last_step_out,
    ffi::Result<ffi::AnyBuffer> pin
) {
  return ffiGuard([&]() -> ffi::Error {
    auto context_id = attrs.get<int64_t>("context_id");
    if (!context_id.has_value()) {
      return invalid("rowwise_adagrad_lazy_update_start: missing context_id");
    }
    auto ctx = readyContext(*context_id);
    if (ctx == nullptr) {
      return invalid("rowwise_adagrad_lazy_update_start: context not initialized");
    }
    auto learning_rate = attrs.get<double>("learning_rate");
    auto eps = attrs.get<double>("eps");
    auto accum_decay_rate = attrs.get<double>("accum_decay_rate");
    auto weight_decay_rate = attrs.get<double>("weight_decay_rate");
    auto weight_decay_exempt_window = attrs.get<double>("weight_decay_exempt_window");
    auto stochastic_rounding = attrs.get<int64_t>("stochastic_rounding");
    if (!learning_rate.has_value() || !eps.has_value() || !accum_decay_rate.has_value() ||
        !weight_decay_rate.has_value() || !weight_decay_exempt_window.has_value() ||
        !stochastic_rounding.has_value()) {
      return invalid("rowwise_adagrad_lazy_update_start: missing optimizer attrs");
    }
    const float lr_f = float(*learning_rate);
    const float eps_f = float(*eps);
    const float accum_rate_f = float(*accum_decay_rate);
    const float wd_rate_f = float(*weight_decay_rate);
    const float wd_window_f = float(*weight_decay_exempt_window);
    if (!std::isfinite(lr_f) || !std::isfinite(eps_f) || eps_f <= 0.f ||
        !std::isfinite(accum_rate_f) || accum_rate_f < 0.f || !std::isfinite(wd_rate_f) ||
        wd_rate_f < 0.f || !std::isfinite(wd_window_f) || wd_window_f < 0.f) {
      return invalid("rowwise_adagrad_lazy_update_start: invalid optimizer parameters");
    }
    ffi::Error buffers_error = validateRowwiseAdagradBuffers(ctx->spec(), table, row_state);
    if (buffers_error.failure()) {
      return buffers_error;
    }
    if (last_step.element_type() != ffi::DataType::S32 ||
        last_step.element_count() != size_t(table.dimensions()[0])) {
      return invalid("rowwise_adagrad_lazy_update_start: bad last_step");
    }
    if (logical_step.element_type() != ffi::DataType::S32 || logical_step.element_count() != 1) {
      return invalid("rowwise_adagrad_lazy_update_start: step must be s32[1]");
    }
    if (!bufferAliased(table, table_out) || !bufferAliased(row_state, state_out) ||
        !bufferAliased(last_step, last_step_out)) {
      return invalid(
          "rowwise_adagrad_lazy_update_start: outputs must alias inputs (check "
          "input_output_aliases)"
      );
    }
    XAI_CUDA_CHECK(cudaMemsetAsync(pin->untyped_data(), 0, pin->size_bytes(), stream));
    const auto& spec = ctx->spec();
    const AdagradParams adagrad{
        lr_f,
        eps_f,
        1.f,
        1.f / float(spec.emb_width),
        1.f,
        accum_rate_f,
        wd_rate_f,
        wd_window_f,
        *stochastic_rounding != 0 ? 1 : 0,
        int32_t(ctx->rank() * spec.shard_width)
    };
    UpdateJob job{static_cast<__nv_bfloat16*>(table.untyped_data()), table.dimensions()[0]};
    ctx->armUpdate(
        job,
        makeRowwiseAdagradRule(
            adagrad,
            static_cast<float*>(row_state.untyped_data()),
            static_cast<int32_t*>(last_step.untyped_data())
        ),
        stream,
        static_cast<const int32_t*>(logical_step.untyped_data())
    );
    return ffi::Error::Success();
  });
}

ffi::Error RowwiseAdagradUpdateDone(
    cudaStream_t stream,
    ffi::AnyBuffer ,
    ffi::Dictionary attrs,
    ffi::Result<ffi::AnyBuffer> norm,
    ffi::Result<ffi::AnyBuffer> valid,
    ffi::Result<ffi::AnyBuffer> pending,
    ffi::Result<ffi::AnyBuffer> done_pin
) {
  return ffiGuard([&]() -> ffi::Error {
    auto context_id = attrs.get<int64_t>("context_id");
    if (!context_id.has_value()) {
      return invalid("rowwise_adagrad_update_done: missing context_id");
    }
    auto ctx = readyContext(*context_id);
    if (ctx == nullptr) {
      return invalid("rowwise_adagrad_update_done: context not initialized");
    }
    ctx->streamConsumeDone(stream, AsyncEmbContext::Operation::Update);
    const int8_t* scalars = ctx->arena() + ctx->layout().scalars;
    auto copy = [&](ffi::Result<ffi::AnyBuffer>& out, size_t offset) {
      XAI_CUDA_CHECK(cudaMemcpyAsync(
          out->untyped_data(), scalars + offset, out->size_bytes(), cudaMemcpyDeviceToDevice, stream
      ));
    };
    copy(norm, offsetof(UpdateScalars, norm));
    copy(valid, offsetof(UpdateScalars, valid));
    copy(pending, offsetof(UpdateScalars, pending));
    XAI_CUDA_CHECK(cudaMemsetAsync(done_pin->untyped_data(), 0, done_pin->size_bytes(), stream));
    return ffi::Error::Success();
  });
}

}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    kLookupStartInit,
    LookupStartInit,
    ffi::Ffi::Bind<ffi::ExecutionStage::kInitialize>()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::CollectiveParamsPartial>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Attrs<ffi::Dictionary>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    kLookupStart,
    LookupStart,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::CollectiveParamsPartial>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Attrs<ffi::Dictionary>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    kLookupDone,
    LookupDone,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Attrs<ffi::Dictionary>()
        .Ret<ffi::AnyBuffer>()
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    kStageUpdate,
    StageUpdate,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Attrs<ffi::Dictionary>()
        .Ret<ffi::AnyBuffer>()
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    kRowwiseAdagradUpdateStartInit,
    RowwiseAdagradUpdateStartInit,
    ffi::Ffi::Bind<ffi::ExecutionStage::kInitialize>()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::CollectiveParamsPartial>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Attrs<ffi::Dictionary>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    kRowwiseAdagradUpdateStart,
    RowwiseAdagradUpdateStart,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::CollectiveParamsPartial>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Attrs<ffi::Dictionary>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    kRowwiseAdagradLazyUpdateStartInit,
    RowwiseAdagradLazyUpdateStartInit,
    ffi::Ffi::Bind<ffi::ExecutionStage::kInitialize>()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::CollectiveParamsPartial>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Attrs<ffi::Dictionary>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    kRowwiseAdagradLazyUpdateStart,
    RowwiseAdagradLazyUpdateStart,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::CollectiveParamsPartial>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Attrs<ffi::Dictionary>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    kRowwiseAdagradUpdateDone,
    RowwiseAdagradUpdateDone,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Attrs<ffi::Dictionary>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
);

namespace {

template <typename T>
nb::capsule encapsulate(T* fn) {
  static_assert(
      std::is_invocable_r_v<XLA_FFI_Error*, T, XLA_FFI_CallFrame*>,
      "Encapsulated function must be an XLA FFI handler"
  );
  return nb::capsule(reinterpret_cast<void*>(fn));
}

uint64_t waitLatest(int64_t context_id, AsyncEmbContext::Operation pipeline, const char* label) {
  auto ctx = findContext(context_id);
  if (ctx == nullptr) {
    return 0;
  }
  ctx->ensureHealthy();
  uint64_t step = ctx->armedStep(pipeline);
  if (step == 0) {
    return 0;
  }
  if (!ctx->hostWaitDone(pipeline, step)) {
    throw std::runtime_error(
        "async_emb: " + std::string(label) + " timed out waiting for step " + std::to_string(step)
    );
  }
  return step;
}

nb::bytes testSnapshot(int64_t context_id, const std::string& region) {
  auto ctx = readyContext(context_id);
  if (ctx == nullptr) {
    throw std::invalid_argument("async_emb context not initialized");
  }
  const auto& spec = ctx->spec();
  const auto& layout = ctx->layout();
  struct Region {
    size_t offset;
    size_t bytes;
  };
  const std::unordered_map<std::string, Region> regions = {
      {"lookup_ids", {layout.token_ids_all, layout.index_bytes}},
      {"lookup_send", {layout.lookup_send, layout.block_bytes}},
      {"recv", {layout.recv, layout.block_bytes}},
      {"update_indices", {layout.segment_ids_all, layout.index_bytes}},
      {"update_stage", {layout.update_stage, layout.block_bytes}},
      {"update_unique", {layout.unique_tokens, size_t(spec.num_unique) * sizeof(int32_t)}},
      {"update_stats", {layout.row_sq_sums, size_t(spec.num_unique) * sizeof(float)}},
      {"update_grad_accum", {layout.grad_accum, layout.accum_bytes}},
  };
  auto found = regions.find(region);
  if (found == regions.end()) {
    throw std::invalid_argument("unknown async_emb snapshot region: " + region);
  }
  waitLatest(context_id, AsyncEmbContext::Operation::Lookup, "test snapshot");
  waitLatest(context_id, AsyncEmbContext::Operation::Update, "test snapshot");
  std::vector<uint8_t> snapshot = ctx->snapshot(found->second.offset, found->second.bytes);
  return nb::bytes(reinterpret_cast<const char*>(snapshot.data()), snapshot.size());
}

}

NB_MODULE(async_emb_api, m) {
  m.def("lookup_start_init", [] { return encapsulate(kLookupStartInit); });
  m.def("lookup_start", [] { return encapsulate(kLookupStart); });
  m.def("lookup_done", [] { return encapsulate(kLookupDone); });
  m.def("stage_update", [] { return encapsulate(kStageUpdate); });
  m.def("rowwise_adagrad_update_start_init", [] {
    return encapsulate(kRowwiseAdagradUpdateStartInit);
  });
  m.def("rowwise_adagrad_update_start", [] { return encapsulate(kRowwiseAdagradUpdateStart); });
  m.def("rowwise_adagrad_lazy_update_start_init", [] {
    return encapsulate(kRowwiseAdagradLazyUpdateStartInit);
  });
  m.def("rowwise_adagrad_lazy_update_start", [] {
    return encapsulate(kRowwiseAdagradLazyUpdateStart);
  });
  m.def("rowwise_adagrad_update_done", [] { return encapsulate(kRowwiseAdagradUpdateDone); });
  m.def(
      "wait_step_ready",
      [](int64_t context_id) {
        return waitLatest(context_id, AsyncEmbContext::Operation::Lookup, "readiness");
      },
      nb::call_guard<nb::gil_scoped_release>()
  );
  m.def(
      "abort",
      [](int64_t context_id) {
        auto ctx = findContext(context_id);
        if (ctx != nullptr) {
          ctx->abort();
        }
      },
      nb::call_guard<nb::gil_scoped_release>()
  );
  m.def("_test_snapshot", &testSnapshot);
  m.def("nccl_version", [] {
    int version = 0;
    ncclResult_t result = ncclGetVersion(&version);
    if (result != ncclSuccess) {
      throw std::runtime_error(ncclGetErrorString(result));
    }
    return version;
  });
  m.def("nccl_library_path", [] {
    Dl_info info{};
    if (dladdr(reinterpret_cast<void*>(&ncclGetVersion), &info) == 0 || info.dli_fname == nullptr) {
      throw std::runtime_error("could not resolve loaded NCCL library path");
    }
    return std::string(info.dli_fname);
  });
}

}
