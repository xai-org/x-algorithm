#include "async_emb_comm.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <thread>
#include <utility>

#include "absl/log/log.h"
#include "async_emb_kernel.hpp"
#include "cuda_error_utils.hpp"

namespace xai::kernels::async_emb {

namespace {

constexpr size_t kAlign = 128;
constexpr int kNcclMinCtas = 1;
constexpr int kNcclMaxCtas = 4;
constexpr std::chrono::minutes kNcclReadyTimeout{5};

size_t checkedProduct(std::initializer_list<size_t> factors) {
  size_t result = 1;
  for (size_t factor : factors) {
    if (factor != 0 && result > std::numeric_limits<size_t>::max() / factor) {
      throw std::overflow_error("async_emb arena size overflow");
    }
    result *= factor;
  }
  return result;
}

size_t alignUp(size_t value) {
  if (value > std::numeric_limits<size_t>::max() - (kAlign - 1)) {
    throw std::overflow_error("async_emb arena alignment overflow");
  }
  return (value + kAlign - 1) / kAlign * kAlign;
}

std::runtime_error ncclError(const char* operation, ncclResult_t result) {
  return std::runtime_error(
      std::string("NCCL ") + operation + " failed: " + ncclGetErrorString(result)
  );
}

void requireEnvironment(const char* name, const char* expected) {
  const char* value = std::getenv(name);
  if (value == nullptr || std::strcmp(value, expected) != 0) {
    throw std::runtime_error(
        "async_emb requires " + std::string(name) + "=" + expected + " before process startup"
    );
  }
}

std::chrono::seconds watchdogTimeout() {
  const char* value = std::getenv("XAI_ASYNC_EMB_TIMEOUT_SECONDS");
  if (value == nullptr) {
    return std::chrono::seconds(1800);
  }
  char* end = nullptr;
  long seconds = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || seconds <= 0) {
    throw std::invalid_argument("XAI_ASYNC_EMB_TIMEOUT_SECONDS must be a positive integer");
  }
  return std::chrono::seconds(seconds);
}

std::once_flag warmup_once;

constexpr int64_t kWarmupScalarsOffset = 256;

void warmupKernels(int8_t* scratch, cudaStream_t stream) {
  auto* bf16 = reinterpret_cast<__nv_bfloat16*>(scratch);
  auto* f32 = reinterpret_cast<float*>(scratch);
  auto* i32 = reinterpret_cast<int32_t*>(scratch);
  auto* scalars = reinterpret_cast<UpdateScalars*>(f32 + kWarmupScalarsOffset);
  const SliceLayout slices{0, 8, 1};
  launch_lookup_dispatch(i32, bf16, bf16, slices, 1, stream);
  launch_grad_dispatch(bf16, bf16, slices, stream);
  launch_update_row_bound(i32, i32, scalars, slices, 0, 1, stream);
  launch_zero_update_rows(f32, scalars, slices.shard_width, stream);
  launch_grad_segment_sum(bf16, i32, f32, f32, scalars, slices, 0, stream);
  launch_lookup_combine(bf16, bf16, slices, stream);
  XAI_CUDA_CHECK(cudaStreamSynchronize(stream));
}

}

ArenaLayout ArenaLayout::build(const PipelineSpec& spec, int world_size) {
  ArenaLayout layout;
  size_t offset = 0;
  auto take = [&offset](size_t bytes) {
    size_t result = offset;
    if (offset > std::numeric_limits<size_t>::max() - bytes) {
      throw std::overflow_error("async_emb arena offset overflow");
    }
    offset = alignUp(offset + bytes);
    return result;
  };

  layout.block_bytes = checkedProduct(
      {size_t(world_size),
       size_t(spec.tokens_per_rank),
       size_t(spec.shard_width),
       sizeof(__nv_bfloat16)}
  );
  layout.index_bytes =
      checkedProduct({size_t(world_size), size_t(spec.tokens_per_rank), sizeof(int32_t)});
  layout.accum_bytes =
      checkedProduct({size_t(spec.num_unique), size_t(spec.shard_width), sizeof(float)});
  layout.token_ids_all = take(layout.index_bytes);
  layout.lookup_send = take(std::max(layout.block_bytes, layout.accum_bytes));
  layout.grad_accum = layout.lookup_send;
  layout.recv = take(layout.block_bytes);
  layout.segment_ids_all = take(layout.index_bytes);
  layout.update_stage = take(layout.block_bytes);
  layout.unique_tokens = take(checkedProduct({size_t(spec.num_unique), sizeof(int32_t)}));
  layout.row_sq_sums = take(checkedProduct({size_t(spec.num_unique) + 1, sizeof(float)}));
  layout.scalars = take(sizeof(UpdateScalars));
  layout.total = offset;
  return layout;
}

CudaEvent::CudaEvent() {
  XAI_CUDA_CHECK(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
}

CudaEvent::~CudaEvent() {
  if (event_ != nullptr) {
    cudaEventDestroy(event_);
  }
}

CudaStream::CudaStream() {
  int least = 0, greatest = 0;
  XAI_CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
  XAI_CUDA_CHECK(cudaStreamCreateWithPriority(&stream_, cudaStreamNonBlocking, greatest));
}

CudaStream::~CudaStream() {
  if (stream_ != nullptr) {
    cudaStreamDestroy(stream_);
  }
}

DeviceArena::DeviceArena(size_t bytes) {
  XAI_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), bytes));
}

DeviceArena::~DeviceArena() {
  if (data_ != nullptr) {
    cudaFree(data_);
  }
}

AsyncEmbContext::AsyncEmbContext(uint64_t context_id, int world_size, PipelineSpec spec)
    : spec_(spec),
      name_("async_emb_" + std::to_string(context_id)),
      layout_(ArenaLayout::build(spec, world_size)),
      world_size_(world_size),
      watchdog_timeout_(watchdogTimeout()),
      arena_(layout_.total) {
  XAI_CUDA_CHECK(cudaGetDevice(&device_));
  XAI_CUDA_CHECK(cudaMemset(arena(), 0, layout_.total));
  std::call_once(warmup_once, [this] {
    DeviceArena scratch(4096);
    XAI_CUDA_CHECK(cudaMemset(scratch.data(), 0, 4096));
    warmupKernels(scratch.data(), side_stream_);
  });
}

AsyncEmbContext::~AsyncEmbContext() {
  if (comm_ == nullptr) {
    return;
  }

  shutdown_ = true;
  if (watchdog_.joinable()) {
    watchdog_.join();
  }

  if (failed_) {
    if (!waitAbortedSideStream(30000)) {
      LOG(ERROR) << "aborted NCCL kernels did not leave the side stream; "
                    "leaking stream and arena until process exit";
      side_stream_.release();
      arena_.release();
    }
    comm_ = nullptr;
    return;
  }

  if (!waitSideStream(30000)) {
    abortCommunicator("timed out draining side stream during teardown");
    comm_ = nullptr;
    return;
  }

  if (arena_registration_ != nullptr) {
    ncclResult_t deregister_result = ncclCommDeregister(comm_, arena_registration_);
    if (deregister_result != ncclSuccess) {
      LOG(ERROR) << "ncclCommDeregister failed: " << ncclGetErrorString(deregister_result);
    }
    arena_registration_ = nullptr;
  }

  ncclResult_t result = ncclCommFinalize(comm_);
  if (result == ncclInProgress && !failed_) {
    try {
      waitNcclReady("communicator finalization");
      result = ncclSuccess;
    } catch (const std::exception& error) {
      LOG(ERROR) << error.what();
      abortCommunicator(error.what());
      comm_ = nullptr;
      return;
    }
  }
  if (result != ncclSuccess) {
    LOG(ERROR) << "ncclCommFinalize failed: " << ncclGetErrorString(result);
    abortCommunicator("ncclCommFinalize failed");
    comm_ = nullptr;
    return;
  }

  result = ncclCommDestroy(comm_);
  if (result != ncclSuccess) {
    LOG(ERROR) << "ncclCommDestroy failed: " << ncclGetErrorString(result);
  }
  comm_ = nullptr;
}

std::vector<std::vector<uint8_t>> AsyncEmbContext::reset(int rank, int world_size) {
  if (world_size <= 0 || rank < 0 || rank >= world_size) {
    throw std::invalid_argument("async_emb received an invalid rank or world size");
  }
  if (world_size != world_size_) {
    throw std::invalid_argument(
        "async_emb world size changed from " + std::to_string(world_size_) + " to " +
        std::to_string(world_size)
    );
  }
  if ((world_size & (world_size - 1)) != 0) {
    throw std::invalid_argument("async_emb currently requires power-of-two EP");
  }

  rank_ = rank;
  initialized_ = false;
  std::vector<std::vector<uint8_t>> bootstrap(world_size);
  if (rank != 0) {
    return bootstrap;
  }

  ncclUniqueId id;
  ncclResult_t result = ncclGetUniqueId(&id);
  if (result != ncclSuccess) {
    throw ncclError("ncclGetUniqueId", result);
  }

  std::vector<uint8_t> bytes(sizeof(id));
  std::memcpy(bytes.data(), &id, sizeof(id));
  std::fill(bootstrap.begin(), bootstrap.end(), bytes);
  return bootstrap;
}

void AsyncEmbContext::handshake(std::vector<std::vector<uint8_t>> data) {
  if (data.size() != size_t(world_size_) || data[0].size() != sizeof(ncclUniqueId)) {
    throw std::invalid_argument("async_emb received an invalid NCCL bootstrap payload");
  }

  requireEnvironment("NCCL_RUNTIME_CONNECT", "0");
  requireEnvironment("NCCL_LAUNCH_ORDER_IMPLICIT", "1");

  ncclUniqueId id;
  std::memcpy(&id, data[0].data(), sizeof(id));
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.blocking = 0;
  config.minCTAs = kNcclMinCtas;
  config.maxCTAs = kNcclMaxCtas;
  ncclResult_t result = ncclCommInitRankConfig(&comm_, world_size_, id, rank_, &config);
  if (result != ncclSuccess && result != ncclInProgress) {
    throw ncclError("ncclCommInitRankConfig", result);
  }
  waitNcclReady("communicator initialization");

  ncclResult_t register_result =
      ncclCommRegister(comm_, arena(), layout_.total, &arena_registration_);
  if (register_result != ncclSuccess && register_result != ncclInProgress) {
    throw ncclError("ncclCommRegister", register_result);
  }
  waitNcclReady("arena registration");

  warmupCollectives();
  XAI_CUDA_CHECK(cudaMemset(arena(), 0, layout_.total));
  initialized_ = true;
  watchdog_ = std::thread([this] { watchdogMain(); });
  LOG(INFO) << "async_emb rank=" << rank_ << " initialized private NCCL communicator"
            << " world_size=" << world_size_ << " arena_bytes=" << layout_.total
            << " min_ctas=" << kNcclMinCtas << " max_ctas=" << kNcclMaxCtas;
}

void AsyncEmbContext::submitNccl(ncclResult_t result, const char* operation) {
  ensureHealthy();
  if (result == ncclSuccess) {
    return;
  }
  if (result == ncclInProgress) {
    waitNcclReady(operation);
    return;
  }
  throw ncclError(operation, result);
}

void AsyncEmbContext::ensureHealthy() const {
  if (!failed_) {
    return;
  }
  std::lock_guard<std::mutex> lock(state_mu_);
  throw std::runtime_error(failure_.empty() ? "async_emb communicator aborted" : failure_);
}

void AsyncEmbContext::abortCommunicator(const std::string& reason) {
  bool expected = false;
  if (!failed_.compare_exchange_strong(expected, true)) {
    return;
  }
  LOG(ERROR) << "async_emb rank=" << rank_ << " aborting communicator: " << reason;
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    failure_ = "async_emb communicator aborted: " + reason;
  }
  initialized_ = false;
  if (comm_ != nullptr) {
    std::lock_guard<std::recursive_mutex> comm_lock(comm_mu_);
    ncclResult_t result = ncclCommAbort(comm_);
    if (result != ncclSuccess) {
      LOG(ERROR) << "ncclCommAbort failed: " << ncclGetErrorString(result);
    }
  }
}

void AsyncEmbContext::abort() { abortCommunicator("requested by host"); }

void AsyncEmbContext::watchdogMain() {
  if (cudaSetDevice(device_) != cudaSuccess) {
    abortCommunicator("watchdog could not select its CUDA device");
    return;
  }
  while (!shutdown_ && !failed_) {
    ncclResult_t async_result = ncclSuccess;
    ncclResult_t query_result = lockedQueryAsyncError(&async_result);
    if (query_result != ncclSuccess) {
      abortCommunicator(
          std::string("ncclCommGetAsyncError failed: ") + ncclGetErrorString(query_result)
      );
      return;
    }
    if (async_result != ncclSuccess && async_result != ncclInProgress) {
      abortCommunicator(
          std::string("asynchronous NCCL failure: ") + ncclGetErrorString(async_result)
      );
      return;
    }

    const auto now = std::chrono::steady_clock::now();
    std::string timeout;
    {
      std::lock_guard<std::mutex> lock(state_mu_);
      for (auto [pipeline, label] :
           {std::pair{&lookup_, "lookup"}, std::pair{&update_, "update"}}) {
        if (!pipeline->in_flight) {
          continue;
        }
        cudaError_t event_result = cudaEventQuery(pipeline->done);
        if (event_result == cudaSuccess) {
          pipeline->in_flight = false;
        } else if (event_result != cudaErrorNotReady) {
          timeout =
              std::string(label) + " completion event failed: " + cudaGetErrorString(event_result);
          break;
        } else if (now - pipeline->submitted_at > watchdog_timeout_) {
          timeout = std::string(label) + " step " + std::to_string(pipeline->armed_step) +
                    " exceeded " + std::to_string(watchdog_timeout_.count()) + " seconds";
          break;
        }
      }
    }
    if (!timeout.empty()) {
      abortCommunicator(timeout);
      return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

ncclResult_t AsyncEmbContext::lockedQueryAsyncError(ncclResult_t* async_result) const {
  std::lock_guard<std::recursive_mutex> comm_lock(comm_mu_);
  return ncclCommGetAsyncError(comm_, async_result);
}

bool AsyncEmbContext::commHealthy() const {
  ncclResult_t async_result = ncclSuccess;
  ncclResult_t query_result = lockedQueryAsyncError(&async_result);
  return query_result == ncclSuccess &&
         (async_result == ncclSuccess || async_result == ncclInProgress);
}

bool AsyncEmbContext::waitSideStream(int timeout_ms) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  while (std::chrono::steady_clock::now() < deadline && !failed_) {
    cudaError_t stream_result = cudaStreamQuery(side_stream_);
    if (stream_result == cudaSuccess) {
      return true;
    }
    if (stream_result != cudaErrorNotReady) {
      abortCommunicator(std::string("side stream failed: ") + cudaGetErrorString(stream_result));
      return false;
    }
    if (!commHealthy()) {
      abortCommunicator("NCCL failed while draining side stream");
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return false;
}

bool AsyncEmbContext::waitAbortedSideStream(int timeout_ms) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  while (std::chrono::steady_clock::now() < deadline) {
    cudaError_t stream_result = cudaStreamQuery(side_stream_);
    if (stream_result == cudaSuccess) {
      return true;
    }
    if (stream_result != cudaErrorNotReady) {
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return false;
}

void AsyncEmbContext::waitNcclReady(const char* operation) {
  const auto deadline = std::chrono::steady_clock::now() + kNcclReadyTimeout;
  while (std::chrono::steady_clock::now() < deadline && !failed_) {
    ncclResult_t async_result = ncclInProgress;
    ncclResult_t query_result = lockedQueryAsyncError(&async_result);
    if (query_result != ncclSuccess) {
      break;
    }
    if (async_result == ncclSuccess) {
      return;
    }
    if (async_result != ncclInProgress) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  abortCommunicator(std::string("NCCL failed or timed out during ") + operation);
  ensureHealthy();
}

void AsyncEmbContext::warmupCollectives() {
  std::lock_guard<std::recursive_mutex> comm_lock(comm_mu_);
  const auto& layout = layout_;
  const auto& spec = spec_;
  int32_t* ids = reinterpret_cast<int32_t*>(arena() + layout.token_ids_all);
  int32_t* local_ids = ids + size_t(rank_) * size_t(spec.tokens_per_rank);
  const size_t block_count = size_t(spec.tokens_per_rank) * size_t(spec.shard_width);

  submitNccl(
      ncclAllGather(local_ids, ids, spec.tokens_per_rank, ncclInt32, comm_, side_stream_),
      "warmup id all-gather"
  );
  submitNccl(
      ncclAlltoAll(
          arena() + layout.lookup_send,
          arena() + layout.recv,
          block_count,
          ncclBfloat16,
          comm_,
          side_stream_
      ),
      "warmup embedding all-to-all"
  );
  submitNccl(
      ncclAlltoAll(
          arena() + layout.update_stage,
          arena() + layout.recv,
          block_count,
          ncclBfloat16,
          comm_,
          side_stream_
      ),
      "warmup gradient all-to-all"
  );
  submitNccl(
      ncclAllReduce(
          arena() + layout.row_sq_sums,
          arena() + layout.row_sq_sums,
          spec.num_unique + 1,
          ncclFloat32,
          ncclSum,
          comm_,
          side_stream_
      ),
      "warmup row-stat all-reduce"
  );
  CudaEvent done;
  XAI_CUDA_CHECK(cudaEventRecord(done, side_stream_));
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(5);
  while (std::chrono::steady_clock::now() < deadline) {
    cudaError_t event_result = cudaEventQuery(done);
    if (event_result == cudaSuccess) {
      return;
    }
    if (event_result != cudaErrorNotReady) {
      break;
    }
    if (!commHealthy()) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  abortCommunicator("collective warmup failed or timed out");
  ensureHealthy();
}

AsyncEmbContext::PipelineState& AsyncEmbContext::pipeline(Operation operation) {
  return operation == Operation::Lookup ? lookup_ : update_;
}

const AsyncEmbContext::PipelineState& AsyncEmbContext::pipeline(Operation operation) const {
  return operation == Operation::Lookup ? lookup_ : update_;
}

void AsyncEmbContext::recordInputReady(PipelineState& pipeline, cudaStream_t main_stream) {
  XAI_CUDA_CHECK(cudaEventRecord(pipeline.input_ready, main_stream));
  XAI_CUDA_CHECK(cudaStreamWaitEvent(side_stream_, pipeline.input_ready, 0));
}

void AsyncEmbContext::recordDone(PipelineState& pipeline) {
  XAI_CUDA_CHECK(cudaEventRecord(pipeline.done, side_stream_));
}

void AsyncEmbContext::bindTable(Operation operation, const void* table) {
  std::lock_guard<std::mutex> lock(state_mu_);
  const TablePhase next = operation == Operation::Lookup ? TablePhase::Lookup : TablePhase::Update;
  if (last_table_phase_ != TablePhase::None && last_table_phase_ != next && last_table_ != table) {
    throw std::invalid_argument(
        "alternating lookup/update table buffers differ; XLA interposed a copy"
    );
  }
  last_table_phase_ = next;
  last_table_ = table;
}

void AsyncEmbContext::enqueueLookup(const LookupJob& job) {
  std::lock_guard<std::recursive_mutex> comm_lock(comm_mu_);
  const auto& layout = layout_;
  const auto& spec = spec_;
  int32_t* ids = reinterpret_cast<int32_t*>(arena() + layout.token_ids_all);
  int32_t* local_ids = ids + size_t(rank_) * size_t(spec.tokens_per_rank);
  submitNccl(
      ncclAllGather(local_ids, ids, spec.tokens_per_rank, ncclInt32, comm_, side_stream_),
      "lookup id all-gather"
  );

  const SliceLayout slices{spec.tokens_per_rank, spec.shard_width, world_size_};
  launch_lookup_dispatch(
      ids,
      job.table,
      reinterpret_cast<__nv_bfloat16*>(arena() + layout.lookup_send),
      slices,
      job.vocab_rows,
      side_stream_
  );
  submitNccl(
      ncclAlltoAll(
          arena() + layout.lookup_send,
          arena() + layout.recv,
          size_t(spec.tokens_per_rank) * size_t(spec.shard_width),
          ncclBfloat16,
          comm_,
          side_stream_
      ),
      "lookup embedding all-to-all"
  );
  recordDone(lookup_);
}

void AsyncEmbContext::enqueueUpdate(const UpdateJob& job, const ApplyUpdateRule& apply) {
  std::lock_guard<std::recursive_mutex> comm_lock(comm_mu_);
  const auto& layout = layout_;
  const auto& spec = spec_;
  const SliceLayout slices{spec.tokens_per_rank, spec.shard_width, world_size_};
  int32_t* segment_ids_all = reinterpret_cast<int32_t*>(arena() + layout.segment_ids_all);
  submitNccl(
      ncclAllGather(
          segment_ids_all + size_t(rank_) * size_t(spec.tokens_per_rank),
          segment_ids_all,
          spec.tokens_per_rank,
          ncclInt32,
          comm_,
          side_stream_
      ),
      "update index all-gather"
  );
  submitNccl(
      ncclAlltoAll(
          arena() + layout.update_stage,
          arena() + layout.recv,
          size_t(spec.tokens_per_rank) * size_t(spec.shard_width),
          ncclBfloat16,
          comm_,
          side_stream_
      ),
      "update gradient all-to-all"
  );

  auto* grad_accum = reinterpret_cast<float*>(arena() + layout.grad_accum);
  const auto* unique_tokens = reinterpret_cast<const int32_t*>(arena() + layout.unique_tokens);
  auto* scalars = reinterpret_cast<UpdateScalars*>(arena() + layout.scalars);
  launch_update_row_bound(
      unique_tokens, segment_ids_all, scalars, slices, spec.num_unique, job.vocab_rows, side_stream_
  );
  launch_zero_update_rows(grad_accum, scalars, spec.shard_width, side_stream_);
  launch_grad_segment_sum(
      reinterpret_cast<const __nv_bfloat16*>(arena() + layout.recv),
      segment_ids_all,
      grad_accum,
      reinterpret_cast<float*>(arena() + layout.row_sq_sums),
      scalars,
      slices,
      spec.num_unique,
      side_stream_
  );
  submitNccl(
      ncclAllReduce(
          arena() + layout.row_sq_sums,
          arena() + layout.row_sq_sums,
          spec.num_unique + 1,
          ncclFloat32,
          ncclSum,
          comm_,
          side_stream_
      ),
      "update row-stat all-reduce"
  );

  const float* row_sq_sums = reinterpret_cast<const float*>(arena() + layout.row_sq_sums);
  const ReducedGradients reduced{
      grad_accum, row_sq_sums, unique_tokens, scalars, spec.num_unique, spec.shard_width
  };
  apply(reduced, job, side_stream_);
  recordDone(update_);
}

void AsyncEmbContext::armLookup(LookupJob job, cudaStream_t main_stream) {
  std::lock_guard<std::mutex> arm_lock(arm_mu_);
  ensureHealthy();
  uint64_t step;
  {
    std::lock_guard<std::mutex> state_lock(state_mu_);
    if (lookup_.armed_step != lookup_.consumed_step) {
      throw std::invalid_argument(
          "async_emb scheduled lookup_start before the pending lookup_done"
      );
    }
    step = lookup_.armed_step + 1;
  }
  bindTable(Operation::Lookup, job.table);
  XAI_CUDA_CHECK(cudaMemcpyAsync(
      arena() + layout_.token_ids_all +
          size_t(rank_) * size_t(spec_.tokens_per_rank) * sizeof(int32_t),
      job.token_ids,
      size_t(spec_.tokens_per_rank) * sizeof(int32_t),
      cudaMemcpyDeviceToDevice,
      main_stream
  ));
  recordInputReady(lookup_, main_stream);
  enqueueLookup(job);
  {
    std::lock_guard<std::mutex> state_lock(state_mu_);
    lookup_.armed_step = step;
    lookup_.in_flight = true;
    lookup_.submitted_at = std::chrono::steady_clock::now();
  }
}

void AsyncEmbContext::stageUpdate(
    const __nv_bfloat16* grads,
    const int32_t* segment_ids,
    const int32_t* unique_tokens,
    const int32_t* pending,
    cudaStream_t main_stream
) {
  std::lock_guard<std::mutex> arm_lock(arm_mu_);
  ensureHealthy();
  {
    std::lock_guard<std::mutex> state_lock(state_mu_);
    if (update_.armed_step != update_.consumed_step) {
      throw std::invalid_argument(
          "async_emb scheduled stage_update before the pending update_done"
      );
    }
    if (staged_) {
      throw std::invalid_argument("async_emb scheduled stage_update twice without an update_start");
    }
  }
  const SliceLayout slices{spec_.tokens_per_rank, spec_.shard_width, world_size_};
  launch_grad_dispatch(
      grads, reinterpret_cast<__nv_bfloat16*>(arena() + layout_.update_stage), slices, main_stream
  );
  auto copy = [&](size_t offset, const int32_t* src, size_t count) {
    XAI_CUDA_CHECK(cudaMemcpyAsync(
        arena() + offset, src, count * sizeof(int32_t), cudaMemcpyDeviceToDevice, main_stream
    ));
  };
  const size_t tokens_per_rank = size_t(spec_.tokens_per_rank);
  copy(
      layout_.segment_ids_all + size_t(rank_) * tokens_per_rank * sizeof(int32_t),
      segment_ids,
      tokens_per_rank
  );
  copy(layout_.unique_tokens, unique_tokens, size_t(spec_.num_unique));
  copy(layout_.scalars + offsetof(UpdateScalars, pending), pending, 1);
  {
    std::lock_guard<std::mutex> state_lock(state_mu_);
    staged_ = true;
  }
}

void AsyncEmbContext::armUpdate(
    UpdateJob job, ApplyUpdateRule apply, cudaStream_t main_stream, const int32_t* logical_step
) {
  std::lock_guard<std::mutex> arm_lock(arm_mu_);
  ensureHealthy();
  uint64_t step;
  {
    std::lock_guard<std::mutex> state_lock(state_mu_);
    if (update_.armed_step != update_.consumed_step) {
      throw std::invalid_argument(
          "async_emb scheduled update_start before the pending update_done"
      );
    }
    if (lookup_.armed_step != lookup_.consumed_step) {
      throw std::invalid_argument(
          "async_emb scheduled update_start before the pending lookup_done"
      );
    }
    if (!staged_ && update_.armed_step != 0) {
      throw std::invalid_argument("async_emb scheduled update_start without a staged step");
    }
    step = update_.armed_step + 1;
  }
  bindTable(Operation::Update, job.table);
  if (logical_step != nullptr) {
    XAI_CUDA_CHECK(cudaMemcpyAsync(
        arena() + layout_.scalars + offsetof(UpdateScalars, step),
        logical_step,
        sizeof(int32_t),
        cudaMemcpyDeviceToDevice,
        main_stream
    ));
  }
  recordInputReady(update_, main_stream);
  enqueueUpdate(job, apply);
  {
    std::lock_guard<std::mutex> state_lock(state_mu_);
    staged_ = false;
    update_.armed_step = step;
    update_.in_flight = true;
    update_.submitted_at = std::chrono::steady_clock::now();
  }
}

void AsyncEmbContext::warmupUpdateRule(const ApplyUpdateRule& apply) {
  std::lock_guard<std::mutex> arm_lock(arm_mu_);
  ensureHealthy();
  DeviceArena scratch(4096);
  XAI_CUDA_CHECK(cudaMemset(scratch.data(), 0, 4096));
  auto* bf16 = reinterpret_cast<__nv_bfloat16*>(scratch.data());
  auto* f32 = reinterpret_cast<float*>(scratch.data());
  auto* i32 = reinterpret_cast<int32_t*>(scratch.data());
  auto* scalars = reinterpret_cast<UpdateScalars*>(f32 + kWarmupScalarsOffset);
  const UpdateJob job{bf16, 1};
  for (int64_t width = 1; width <= kMaxShardWidth; width *= 2) {
    apply(ReducedGradients{f32, f32, i32, scalars, 0, width}, job, side_stream_);
  }
  XAI_CUDA_CHECK(cudaStreamSynchronize(side_stream_));
}

uint64_t AsyncEmbContext::armedStep(Operation operation) const {
  std::lock_guard<std::mutex> lock(state_mu_);
  return pipeline(operation).armed_step;
}

void AsyncEmbContext::streamConsumeDone(cudaStream_t stream, Operation operation) {
  ensureHealthy();
  std::lock_guard<std::mutex> lock(state_mu_);
  PipelineState& state = pipeline(operation);
  if (state.armed_step != state.consumed_step + 1) {
    const std::string name = operation == Operation::Lookup ? "lookup" : "update";
    throw std::invalid_argument(
        "async_emb " + name + "_done has no armed " + name + "_start to consume"
    );
  }
  XAI_CUDA_CHECK(cudaStreamWaitEvent(stream, state.done, 0));
  state.consumed_step = state.armed_step;
}

bool AsyncEmbContext::hostWaitDone(Operation operation, uint64_t step) {
  ensureHealthy();
  const cudaEvent_t done = pipeline(operation).done;
  const auto deadline = std::chrono::steady_clock::now() + watchdog_timeout_;
  while (std::chrono::steady_clock::now() < deadline) {
    cudaError_t event_result = cudaEventQuery(done);
    if (event_result == cudaSuccess) {
      return true;
    }
    if (event_result != cudaErrorNotReady) {
      throw std::runtime_error(
          std::string("cudaEventQuery failed while waiting for embedding communication: ") +
          cudaGetErrorString(event_result)
      );
    }

    if (!commHealthy()) {
      abortCommunicator("NCCL failed during host readiness wait");
      ensureHealthy();
    }
    std::this_thread::sleep_for(std::chrono::microseconds(200));
  }
  abortCommunicator(
      "timed out waiting for " + std::string(operation == Operation::Lookup ? "lookup" : "update") +
      " step " + std::to_string(step)
  );
  return false;
}

std::vector<uint8_t> AsyncEmbContext::snapshot(size_t offset, size_t bytes) const {
  ensureHealthy();
  if (offset > layout_.total || bytes > layout_.total - offset) {
    throw std::out_of_range("async_emb snapshot is outside the arena");
  }
  std::vector<uint8_t> result(bytes);
  XAI_CUDA_CHECK(cudaMemcpy(result.data(), arena() + offset, bytes, cudaMemcpyDeviceToHost));
  return result;
}

}
