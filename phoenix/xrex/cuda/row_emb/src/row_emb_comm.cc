#include "row_emb_comm.hpp"

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
#include "cuda_error_utils.hpp"
#include "row_emb_kernel.hpp"

namespace xai::kernels::row_emb {

namespace {

constexpr size_t kAlign = 256;
constexpr int kNcclMinCtas = 1;
constexpr int kNcclMaxCtas = 4;
constexpr std::chrono::minutes kNcclReadyTimeout{5};

size_t checkedProduct(std::initializer_list<size_t> factors) {
  size_t result = 1;
  for (size_t factor : factors) {
    if (factor != 0 && result > std::numeric_limits<size_t>::max() / factor) {
      throw std::overflow_error("row_emb arena size overflow");
    }
    result *= factor;
  }
  return result;
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
        "row_emb requires " + std::string(name) + "=" + expected + " before process startup"
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

void warmupKernels(int8_t* scratch, cudaStream_t stream) {
  auto* rows = reinterpret_cast<bf16*>(scratch);
  auto* i32 = reinterpret_cast<int32_t*>(scratch + 2048);
  auto* f32 = reinterpret_cast<float*>(scratch + 3072);
  auto* scalars = reinterpret_cast<UpdateScalars*>(scratch + 3584);
  const Ownership ownership{1, 1};
  XAI_CUDA_CHECK(cudaMemsetAsync(scratch, 0, 4096, stream));
  launch_sanitize_ids(i32, i32 + 8, 8, 1, stream);
  launch_owner_counts(i32, i32 + 16, ownership, i32 + 24, stream);
  launch_gather_rows(rows, 1, 8, i32, 1, ownership, 0, rows + 8, stream);
  launch_scatter_lookup(rows, 8, i32, 1, rows + 8, stream);
  launch_segment_reduce_rows(rows, 8, i32, i32 + 8, i32 + 16, 1, rows + 8, stream);
  launch_owner_reduce_rows(rows, 8, i32, i32 + 8, i32 + 16, 1, f32, scalars, stream);
  launch_rowwise_adagrad_apply(
      rows,
      i32,
      i32 + 8,
      i32 + 24,
      i32 + 16,
      1,
      f32,
      f32 + 8,
      nullptr,
      rows + 8,
      1,
      8,
      ownership,
      0,
      scalars,
      AdagradParams{0.f, 1.f, 1.f, 1.f, 1.f, 0.f, 0.f, 0.f, 0},
      stream
  );
  XAI_CUDA_CHECK(cudaStreamSynchronize(stream));
}

}

ArenaLayout ArenaLayout::build(const PipelineSpec& spec, int world_size) {
  ArenaLayout layout;
  size_t offset = 0;
  auto take = [&offset](size_t bytes) {
    size_t result = offset;
    if (offset > std::numeric_limits<size_t>::max() - bytes - kAlign) {
      throw std::overflow_error("row_emb arena offset overflow");
    }
    offset = (offset + bytes + kAlign - 1) / kAlign * kAlign;
    return result;
  };
  const size_t n = size_t(spec.tokens_per_rank);
  const size_t capacity = size_t(spec.recv_capacity);
  const size_t world = size_t(world_size);
  const size_t row_bytes = checkedProduct({size_t(spec.emb_width), sizeof(bf16)});
  for (int s = 0; s < 2; ++s) {
    layout.slot_ids[s] = take(checkedProduct({n, sizeof(int32_t)}));
    layout.slot_unique[s] = take(checkedProduct({n, sizeof(int32_t)}));
    layout.slot_inverse[s] = take(checkedProduct({n, sizeof(int32_t)}));
    layout.slot_order[s] = take(checkedProduct({n, sizeof(int32_t)}));
    layout.slot_offsets[s] = take(checkedProduct({n + 1, sizeof(int32_t)}));
    layout.slot_n_unique[s] = take(sizeof(int32_t));
    layout.slot_send_counts[s] = take(checkedProduct({world, sizeof(int32_t)}));
    layout.slot_recv_counts[s] = take(checkedProduct({world, sizeof(int32_t)}));
    layout.slot_req_ids[s] = take(checkedProduct({capacity, sizeof(int32_t)}));
  }
  layout.unique_ws_bytes = unique_workspace_bytes(int64_t(std::max(n, capacity)));
  layout.unique_ws = take(layout.unique_ws_bytes);
  layout.recv_rows = take(checkedProduct({n, row_bytes}));
  layout.grads_by_owner = take(checkedProduct({n, row_bytes}));
  layout.exchange = take(checkedProduct({capacity, row_bytes}));
  layout.owner_unique = take(checkedProduct({capacity, sizeof(int32_t)}));
  layout.owner_inverse = take(checkedProduct({capacity, sizeof(int32_t)}));
  layout.owner_order = take(checkedProduct({capacity, sizeof(int32_t)}));
  layout.owner_offsets = take(checkedProduct({capacity + 1, sizeof(int32_t)}));
  layout.owner_count = take(sizeof(int32_t));
  layout.owner_row_sq_sums = take(checkedProduct({capacity, sizeof(float)}));
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

RowEmbContext::RowEmbContext(uint64_t context_id, int world_size, PipelineSpec spec)
    : spec_(spec),
      name_("row_emb_" + std::to_string(context_id)),
      world_size_(world_size),
      layout_(ArenaLayout::build(spec, world_size)),
      watchdog_timeout_(watchdogTimeout()),
      arena_(layout_.total) {
  XAI_CUDA_CHECK(cudaGetDevice(&device_));
  XAI_CUDA_CHECK(cudaMemset(arena(), 0, layout_.total));
  for (int s = 0; s < 2; ++s) {
    LookupSlot& slot = slots_[s];
    auto at = [this](size_t offset) { return reinterpret_cast<int32_t*>(arena() + offset); };
    slot.ids = at(layout_.slot_ids[s]);
    slot.unique = at(layout_.slot_unique[s]);
    slot.inverse = at(layout_.slot_inverse[s]);
    slot.order = at(layout_.slot_order[s]);
    slot.offsets = at(layout_.slot_offsets[s]);
    slot.n_unique = at(layout_.slot_n_unique[s]);
    slot.send_counts = at(layout_.slot_send_counts[s]);
    slot.recv_counts = at(layout_.slot_recv_counts[s]);
    slot.req_ids = at(layout_.slot_req_ids[s]);
    XAI_CUDA_CHECK(cudaHostAlloc(
        reinterpret_cast<void**>(&slot.host_counts),
        2 * size_t(world_size) * sizeof(int32_t),
        cudaHostAllocDefault
    ));
    XAI_CUDA_CHECK(cudaEventCreateWithFlags(&slot.counts_ready, cudaEventDisableTiming));
    slot.send_offsets.assign(size_t(world_size) + 1, 0);
    slot.recv_offsets.assign(size_t(world_size) + 1, 0);
  }
  std::call_once(warmup_once, [this] {
    DeviceArena scratch(4096);
    warmupKernels(scratch.data(), side_stream_);
  });
}

RowEmbContext::~RowEmbContext() {
  shutdown_ = true;
  if (watchdog_.joinable()) {
    watchdog_.join();
  }
  for (LookupSlot& slot : slots_) {
    if (slot.counts_ready != nullptr) {
      cudaEventDestroy(slot.counts_ready);
    }
    if (slot.host_counts != nullptr) {
      cudaFreeHost(slot.host_counts);
    }
  }
  if (comm_ == nullptr) {
    return;
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

std::vector<std::vector<uint8_t>> RowEmbContext::reset(int rank, int world_size) {
  if (world_size <= 0 || rank < 0 || rank >= world_size) {
    throw std::invalid_argument("row_emb received an invalid rank or world size");
  }
  if (world_size != world_size_) {
    throw std::invalid_argument(
        "row_emb world size changed from " + std::to_string(world_size_) + " to " +
        std::to_string(world_size)
    );
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

void RowEmbContext::handshake(std::vector<std::vector<uint8_t>> data) {
  if (data.size() != size_t(world_size_) || data[0].size() != sizeof(ncclUniqueId)) {
    throw std::invalid_argument("row_emb received an invalid NCCL bootstrap payload");
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
  LOG(INFO) << "row_emb rank=" << rank_ << " initialized private NCCL communicator"
            << " world_size=" << world_size_ << " rows_per_rank=" << spec_.rows_per_rank
            << " tokens_per_rank=" << spec_.tokens_per_rank
            << " recv_capacity=" << spec_.recv_capacity << " arena_bytes=" << layout_.total;
}

void RowEmbContext::submitNccl(ncclResult_t result, const char* operation) {
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

void RowEmbContext::ensureHealthy() const {
  if (!failed_) {
    return;
  }
  std::lock_guard<std::mutex> lock(state_mu_);
  throw std::runtime_error(failure_.empty() ? "row_emb communicator aborted" : failure_);
}

void RowEmbContext::abortCommunicator(const std::string& reason) {
  bool expected = false;
  if (!failed_.compare_exchange_strong(expected, true)) {
    return;
  }
  LOG(ERROR) << "row_emb rank=" << rank_ << " aborting communicator: " << reason;
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    failure_ = "row_emb communicator aborted: " + reason;
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

void RowEmbContext::abort() { abortCommunicator("requested by host"); }

void RowEmbContext::watchdogMain() {
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
        if (!pipeline->tail_pending) {
          cudaError_t event_result = cudaEventQuery(pipeline->done);
          if (event_result == cudaSuccess) {
            pipeline->in_flight = false;
            continue;
          }
          if (event_result != cudaErrorNotReady) {
            timeout = std::string(label) +
                      " completion event failed: " + cudaGetErrorString(event_result);
            break;
          }
        }
        if (now - pipeline->submitted_at > watchdog_timeout_) {
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

ncclResult_t RowEmbContext::lockedQueryAsyncError(ncclResult_t* async_result) const {
  std::lock_guard<std::recursive_mutex> comm_lock(comm_mu_);
  return ncclCommGetAsyncError(comm_, async_result);
}

bool RowEmbContext::commHealthy() const {
  ncclResult_t async_result = ncclSuccess;
  ncclResult_t query_result = lockedQueryAsyncError(&async_result);
  return query_result == ncclSuccess &&
         (async_result == ncclSuccess || async_result == ncclInProgress);
}

bool RowEmbContext::waitSideStream(int timeout_ms) {
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

bool RowEmbContext::waitAbortedSideStream(int timeout_ms) {
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

void RowEmbContext::waitNcclReady(const char* operation) {
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

void RowEmbContext::warmupCollectives() {
  std::lock_guard<std::recursive_mutex> comm_lock(comm_mu_);
  LookupSlot& slot = slots_[0];
  submitNccl(
      ncclAlltoAll(slot.send_counts, slot.recv_counts, 1, ncclInt32, comm_, side_stream_),
      "warmup count all-to-all"
  );
  std::vector<int64_t> one(size_t(world_size_) + 1);
  for (size_t p = 0; p <= size_t(world_size_); ++p) {
    one[p] = int64_t(p);
  }
  if (layout_.recv_rows + size_t(world_size_) * sizeof(int32_t) <= layout_.grads_by_owner) {
    exchange(
        slot.send_counts,
        one,
        arena() + layout_.recv_rows,
        one,
        sizeof(int32_t),
        ncclInt32,
        "warmup send/recv group"
    );
  }
  auto* scalars = reinterpret_cast<UpdateScalars*>(arena() + layout_.scalars);
  submitNccl(
      ncclAllReduce(
          &scalars->total_sq_sum,
          &scalars->total_sq_sum,
          1,
          ncclFloat32,
          ncclSum,
          comm_,
          side_stream_
      ),
      "warmup norm all-reduce"
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

RowEmbContext::PipelineState& RowEmbContext::pipeline(Operation operation) {
  return operation == Operation::Lookup ? lookup_ : update_;
}

const RowEmbContext::PipelineState& RowEmbContext::pipeline(Operation operation) const {
  return operation == Operation::Lookup ? lookup_ : update_;
}

void RowEmbContext::bindTable(Operation operation, const void* table) {
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

void RowEmbContext::waitCountsReady(LookupSlot& slot) {
  const auto deadline = std::chrono::steady_clock::now() + watchdog_timeout_;
  while (std::chrono::steady_clock::now() < deadline) {
    ensureHealthy();
    if (shutdown_) {
      throw std::runtime_error("shut down while waiting for the count exchange");
    }
    cudaError_t event_result = cudaEventQuery(slot.counts_ready);
    if (event_result == cudaSuccess) {
      return;
    }
    if (event_result != cudaErrorNotReady) {
      throw std::runtime_error(
          std::string("count exchange event failed: ") + cudaGetErrorString(event_result)
      );
    }
    if (!commHealthy()) {
      throw std::runtime_error("NCCL failed during the count exchange");
    }
    std::this_thread::sleep_for(std::chrono::microseconds(20));
  }
  throw std::runtime_error("timed out waiting for the count exchange");
}

bool RowEmbContext::countsReady(LookupSlot& slot) {
  cudaError_t event_result = cudaEventQuery(slot.counts_ready);
  if (event_result == cudaSuccess) {
    return true;
  }
  if (event_result != cudaErrorNotReady) {
    throw std::runtime_error(
        std::string("count exchange event failed: ") + cudaGetErrorString(event_result)
    );
  }
  return false;
}

void RowEmbContext::exchange(
    const void* send,
    const std::vector<int64_t>& send_offsets,
    void* recv,
    const std::vector<int64_t>& recv_offsets,
    size_t elem_bytes,
    ncclDataType_t type,
    const char* operation
) {
  const size_t type_bytes = type == ncclInt32 ? sizeof(int32_t) : sizeof(bf16);
  const size_t elems = elem_bytes / type_bytes;
  const auto* send_bytes = static_cast<const int8_t*>(send);
  auto* recv_bytes = static_cast<int8_t*>(recv);
  const int64_t self_count = send_offsets[rank_ + 1] - send_offsets[rank_];
  if (self_count != recv_offsets[rank_ + 1] - recv_offsets[rank_]) {
    throw std::runtime_error(std::string(operation) + ": self block size mismatch");
  }
  if (self_count > 0) {
    XAI_CUDA_CHECK(cudaMemcpyAsync(
        recv_bytes + size_t(recv_offsets[rank_]) * elem_bytes,
        send_bytes + size_t(send_offsets[rank_]) * elem_bytes,
        size_t(self_count) * elem_bytes,
        cudaMemcpyDeviceToDevice,
        side_stream_
    ));
  }
  std::lock_guard<std::recursive_mutex> comm_lock(comm_mu_);
  submitNccl(ncclGroupStart(), operation);
  for (int p = 0; p < world_size_; ++p) {
    if (p == rank_) {
      continue;
    }
    const int64_t send_count = send_offsets[p + 1] - send_offsets[p];
    if (send_count > 0) {
      submitNccl(
          ncclSend(
              send_bytes + size_t(send_offsets[p]) * elem_bytes,
              size_t(send_count) * elems,
              type,
              p,
              comm_,
              side_stream_
          ),
          operation
      );
    }
    const int64_t recv_count = recv_offsets[p + 1] - recv_offsets[p];
    if (recv_count > 0) {
      submitNccl(
          ncclRecv(
              recv_bytes + size_t(recv_offsets[p]) * elem_bytes,
              size_t(recv_count) * elems,
              type,
              p,
              comm_,
              side_stream_
          ),
          operation
      );
    }
  }
  submitNccl(ncclGroupEnd(), operation);
}

void RowEmbContext::enqueueLookupTail(LookupSlot& slot, const LookupJob& job) {
  const PipelineSpec& spec = spec_;
  slot.send_offsets[0] = 0;
  slot.recv_offsets[0] = 0;
  for (int p = 0; p < world_size_; ++p) {
    slot.send_offsets[p + 1] = slot.send_offsets[p] + slot.host_counts[p];
    slot.recv_offsets[p + 1] = slot.recv_offsets[p] + slot.host_counts[world_size_ + p];
  }
  slot.received = slot.recv_offsets[world_size_];
  if (slot.send_offsets[world_size_] > spec.tokens_per_rank) {
    throw std::runtime_error("row_emb: per-owner counts exceed the token count");
  }
  if (slot.received > spec.recv_capacity) {
    throw std::runtime_error(
        "row_emb: rank " + std::to_string(rank_) + " was asked for " +
        std::to_string(slot.received) + " rows in one lookup but recv_capacity is " +
        std::to_string(spec.recv_capacity) + "; raise row_emb_recv_factor"
    );
  }

  exchange(
      slot.unique,
      slot.send_offsets,
      slot.req_ids,
      slot.recv_offsets,
      sizeof(int32_t),
      ncclInt32,
      "lookup id exchange"
  );
  auto* rows = reinterpret_cast<bf16*>(arena() + layout_.exchange);
  launch_gather_rows(
      job.table,
      job.rows_local,
      spec.emb_width,
      slot.req_ids,
      slot.received,
      ownership(),
      rank_,
      rows,
      side_stream_
  );
  exchange(
      rows,
      slot.recv_offsets,
      arena() + layout_.recv_rows,
      slot.send_offsets,
      size_t(spec.emb_width) * sizeof(bf16),
      ncclBfloat16,
      "lookup row exchange"
  );
  XAI_CUDA_CHECK(cudaEventRecord(lookup_.done, side_stream_));
}

void RowEmbContext::enqueueUpdate(
    int slot_index, const UpdateJob& job, const ApplyUpdateRule& apply
) {
  const PipelineSpec& spec = spec_;
  auto* scalars = reinterpret_cast<UpdateScalars*>(arena() + layout_.scalars);
  XAI_CUDA_CHECK(cudaStreamWaitEvent(side_stream_, update_.input_ready, 0));
  if (slot_index < 0) {
    XAI_CUDA_CHECK(cudaMemsetAsync(scalars, 0, offsetof(UpdateScalars, pending), side_stream_));
    XAI_CUDA_CHECK(cudaEventRecord(update_.done, side_stream_));
    return;
  }
  LookupSlot& slot = slots_[slot_index];
  auto* grads = reinterpret_cast<bf16*>(arena() + layout_.exchange);
  exchange(
      arena() + layout_.grads_by_owner,
      slot.send_offsets,
      grads,
      slot.recv_offsets,
      size_t(spec.emb_width) * sizeof(bf16),
      ncclBfloat16,
      "update gradient exchange"
  );
  XAI_CUDA_CHECK(cudaMemsetAsync(&scalars->total_sq_sum, 0, sizeof(float), side_stream_));
  auto* owner_unique = reinterpret_cast<int32_t*>(arena() + layout_.owner_unique);
  auto* owner_inverse = reinterpret_cast<int32_t*>(arena() + layout_.owner_inverse);
  auto* owner_order = reinterpret_cast<int32_t*>(arena() + layout_.owner_order);
  auto* owner_offsets = reinterpret_cast<int32_t*>(arena() + layout_.owner_offsets);
  auto* owner_count = reinterpret_cast<int32_t*>(arena() + layout_.owner_count);
  auto* row_sq_sums = reinterpret_cast<float*>(arena() + layout_.owner_row_sq_sums);
  launch_unique_ids(
      slot.req_ids,
      slot.received,
      owner_unique,
      owner_inverse,
      owner_order,
      owner_offsets,
      owner_count,
      arena() + layout_.unique_ws,
      layout_.unique_ws_bytes,
      side_stream_
  );
  launch_owner_reduce_rows(
      grads,
      spec.emb_width,
      owner_order,
      owner_offsets,
      owner_count,
      slot.received,
      row_sq_sums,
      scalars,
      side_stream_
  );
  {
    std::lock_guard<std::recursive_mutex> comm_lock(comm_mu_);
    submitNccl(
        ncclAllReduce(
            &scalars->total_sq_sum,
            &scalars->total_sq_sum,
            1,
            ncclFloat32,
            ncclSum,
            comm_,
            side_stream_
        ),
        "update norm all-reduce"
    );
  }
  const ReducedGradients reduced{
      grads,
      owner_order,
      owner_offsets,
      owner_unique,
      owner_count,
      slot.received,
      row_sq_sums,
      scalars,
      ownership(),
      rank_,
      spec.emb_width
  };
  apply(reduced, job, side_stream_);
  XAI_CUDA_CHECK(cudaEventRecord(update_.done, side_stream_));
}

void RowEmbContext::armLookup(LookupJob job, cudaStream_t main_stream) {
  std::lock_guard<std::mutex> arm_lock(arm_mu_);
  ensureHealthy();
  uint64_t step;
  {
    std::lock_guard<std::mutex> state_lock(state_mu_);
    if (lookup_.armed_step != lookup_.consumed_step) {
      throw std::invalid_argument("row_emb scheduled lookup_start before the pending lookup_done");
    }
    step = lookup_.armed_step + 1;
  }
  bindTable(Operation::Lookup, job.table);
  const int slot_index = int(step & 1);
  LookupSlot& slot = slots_[slot_index];
  launch_sanitize_ids(
      job.token_ids, slot.ids, spec_.tokens_per_rank, ownership().rows(), main_stream
  );
  XAI_CUDA_CHECK(cudaEventRecord(lookup_.input_ready, main_stream));
  XAI_CUDA_CHECK(cudaStreamWaitEvent(side_stream_, lookup_.input_ready, 0));
  launch_unique_ids(
      slot.ids,
      spec_.tokens_per_rank,
      slot.unique,
      slot.inverse,
      slot.order,
      slot.offsets,
      slot.n_unique,
      arena() + layout_.unique_ws,
      layout_.unique_ws_bytes,
      side_stream_
  );
  launch_owner_counts(slot.unique, slot.n_unique, ownership(), slot.send_counts, side_stream_);
  {
    std::lock_guard<std::recursive_mutex> comm_lock(comm_mu_);
    submitNccl(
        ncclAlltoAll(slot.send_counts, slot.recv_counts, 1, ncclInt32, comm_, side_stream_),
        "lookup count all-to-all"
    );
  }
  const size_t counts_bytes = size_t(world_size_) * sizeof(int32_t);
  XAI_CUDA_CHECK(cudaMemcpyAsync(
      slot.host_counts, slot.send_counts, counts_bytes, cudaMemcpyDeviceToHost, side_stream_
  ));
  XAI_CUDA_CHECK(cudaMemcpyAsync(
      slot.host_counts + world_size_,
      slot.recv_counts,
      counts_bytes,
      cudaMemcpyDeviceToHost,
      side_stream_
  ));
  XAI_CUDA_CHECK(cudaEventRecord(slot.counts_ready, side_stream_));
  pending_lookup_ = job;
  std::lock_guard<std::mutex> state_lock(state_mu_);
  lookup_.armed_step = step;
  lookup_.in_flight = true;
  lookup_.tail_pending = true;
  lookup_.submitted_at = std::chrono::steady_clock::now();
}

bool RowEmbContext::flushLookupTail(bool block) {
  std::lock_guard<std::mutex> arm_lock(arm_mu_);
  ensureHealthy();
  uint64_t step;
  {
    std::lock_guard<std::mutex> state_lock(state_mu_);
    if (!lookup_.tail_pending) {
      return true;
    }
    step = lookup_.armed_step;
  }
  LookupSlot& slot = slots_[step & 1];
  if (block) {
    waitCountsReady(slot);
  } else if (!countsReady(slot)) {
    return false;
  }
  enqueueLookupTail(slot, pending_lookup_);
  std::lock_guard<std::mutex> state_lock(state_mu_);
  lookup_.tail_pending = false;
  return true;
}

void RowEmbContext::finishLookup(cudaStream_t stream, bf16* embeddings) {
  ensureHealthy();
  uint64_t step;
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    if (lookup_.armed_step != lookup_.consumed_step + 1) {
      throw std::invalid_argument("row_emb lookup_done has no armed lookup_start to consume");
    }
    step = lookup_.armed_step;
  }
  flushLookupTail(true);
  XAI_CUDA_CHECK(cudaStreamWaitEvent(stream, lookup_.done, 0));
  const LookupSlot& slot = slots_[step & 1];
  launch_scatter_lookup(
      reinterpret_cast<const bf16*>(arena() + layout_.recv_rows),
      spec_.emb_width,
      slot.inverse,
      spec_.tokens_per_rank,
      embeddings,
      stream
  );
  std::lock_guard<std::mutex> lock(state_mu_);
  lookup_.consumed_step = step;
}

void RowEmbContext::stageUpdate(const bf16* grads, const int32_t* pending, cudaStream_t main_stream) {
  std::lock_guard<std::mutex> arm_lock(arm_mu_);
  ensureHealthy();
  uint64_t lookup_step;
  {
    std::lock_guard<std::mutex> state_lock(state_mu_);
    if (update_.armed_step != update_.consumed_step) {
      throw std::invalid_argument("row_emb scheduled stage_update before the pending update_done");
    }
    if (staged_) {
      throw std::invalid_argument("row_emb scheduled stage_update twice without an update_start");
    }
    if (lookup_.consumed_step == 0) {
      throw std::invalid_argument("row_emb scheduled stage_update before any lookup_done");
    }
    lookup_step = lookup_.consumed_step;
  }
  const int slot_index = int(lookup_step & 1);
  const LookupSlot& slot = slots_[slot_index];
  launch_segment_reduce_rows(
      grads,
      spec_.emb_width,
      slot.order,
      slot.offsets,
      slot.n_unique,
      spec_.tokens_per_rank,
      reinterpret_cast<bf16*>(arena() + layout_.grads_by_owner),
      main_stream
  );
  XAI_CUDA_CHECK(cudaMemcpyAsync(
      arena() + layout_.scalars + offsetof(UpdateScalars, pending),
      pending,
      sizeof(int32_t),
      cudaMemcpyDeviceToDevice,
      main_stream
  ));
  std::lock_guard<std::mutex> state_lock(state_mu_);
  staged_ = true;
  staged_slot_ = slot_index;
}

void RowEmbContext::armUpdate(
    UpdateJob job, ApplyUpdateRule apply, cudaStream_t main_stream, const int32_t* logical_step
) {
  std::lock_guard<std::mutex> arm_lock(arm_mu_);
  ensureHealthy();
  uint64_t step;
  int slot_index;
  {
    std::lock_guard<std::mutex> state_lock(state_mu_);
    if (update_.armed_step != update_.consumed_step) {
      throw std::invalid_argument("row_emb scheduled update_start before the pending update_done");
    }
    if (lookup_.armed_step != lookup_.consumed_step) {
      throw std::invalid_argument("row_emb scheduled update_start before the pending lookup_done");
    }
    if (!staged_ && update_.armed_step != 0) {
      throw std::invalid_argument("row_emb scheduled update_start without a staged step");
    }
    step = update_.armed_step + 1;
    slot_index = staged_ ? staged_slot_ : -1;
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
  XAI_CUDA_CHECK(cudaEventRecord(update_.input_ready, main_stream));
  enqueueUpdate(slot_index, job, apply);
  std::lock_guard<std::mutex> state_lock(state_mu_);
  staged_ = false;
  update_.armed_step = step;
  update_.in_flight = true;
  update_.submitted_at = std::chrono::steady_clock::now();
}

void RowEmbContext::finishUpdate(cudaStream_t stream) {
  ensureHealthy();
  uint64_t step;
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    if (update_.armed_step != update_.consumed_step + 1) {
      throw std::invalid_argument("row_emb update_done has no armed update_start to consume");
    }
    step = update_.armed_step;
  }
  XAI_CUDA_CHECK(cudaStreamWaitEvent(stream, update_.done, 0));
  std::lock_guard<std::mutex> lock(state_mu_);
  update_.consumed_step = step;
}

void RowEmbContext::warmupUpdateRule(const ApplyUpdateRule& apply) {
  std::lock_guard<std::mutex> arm_lock(arm_mu_);
  ensureHealthy();
  DeviceArena scratch(8192);
  XAI_CUDA_CHECK(cudaMemset(scratch.data(), 0, 8192));
  auto* rows = reinterpret_cast<bf16*>(scratch.data());
  auto* i32 = reinterpret_cast<int32_t*>(scratch.data() + 4096);
  auto* f32 = reinterpret_cast<float*>(scratch.data() + 6144);
  const ReducedGradients reduced{
      rows,
      i32,
      i32 + 8,
      i32 + 16,
      i32 + 24,
      0,
      f32,
      reinterpret_cast<UpdateScalars*>(f32 + 64),
      Ownership{1, 1},
      0,
      8
  };
  const UpdateJob job{rows + 8, 1};
  apply(reduced, job, side_stream_);
  XAI_CUDA_CHECK(cudaStreamSynchronize(side_stream_));
}

uint64_t RowEmbContext::armedStep(Operation operation) const {
  std::lock_guard<std::mutex> lock(state_mu_);
  return pipeline(operation).armed_step;
}

bool RowEmbContext::hostWaitDone(Operation operation, uint64_t step) {
  ensureHealthy();
  if (operation == Operation::Lookup) {
    flushLookupTail(true);
  }
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

std::vector<uint8_t> RowEmbContext::snapshot(size_t offset, size_t bytes) const {
  ensureHealthy();
  if (offset > layout_.total || bytes > layout_.total - offset) {
    throw std::out_of_range("row_emb snapshot is outside the arena");
  }
  std::vector<uint8_t> result(bytes);
  XAI_CUDA_CHECK(cudaMemcpy(result.data(), arena() + offset, bytes, cudaMemcpyDeviceToHost));
  return result;
}

}
