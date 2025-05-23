/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "velox/exec/ExchangeQueue.h"
#include "velox/exec/ExchangeSource.h"

namespace facebook::velox::exec {

// Handle for a set of producers. This may be shared by multiple Exchanges, one
// per consumer thread.
class ExchangeClient : public std::enable_shared_from_this<ExchangeClient> {
 public:
  static constexpr int32_t kDefaultMaxQueuedBytes = 32 << 20; // 32 MB.
  static constexpr std::chrono::milliseconds kRequestDataMaxWait{100};
  static inline const std::string kBackgroundCpuTimeMs = "backgroundCpuTimeMs";

  ExchangeClient(
      std::string taskId,
      int destination,
      int64_t maxQueuedBytes,
      int32_t numberOfConsumers,
      uint64_t minOutputBatchBytes,
      memory::MemoryPool* pool,
      folly::Executor* executor,
      int32_t requestDataSizesMaxWaitSec = 10)
      : taskId_{std::move(taskId)},
        destination_(destination),
        maxQueuedBytes_{maxQueuedBytes},
        kRequestDataSizesMaxWaitSec_{requestDataSizesMaxWaitSec},
        pool_(pool),
        executor_(executor),
        queue_(std::make_shared<ExchangeQueue>(
            numberOfConsumers,
            minOutputBatchBytes)),
        // See comment in 'pickSourcesToRequestLocked' for why this is needed
        // for 'minOutputBatchBytes_'. Note: ExchangeQueue does not need max(1,
        // minOutputBatchBytes) because for 'MergeExchangeSource', we want
        // ExchangeQueue 'minOutputBatchBytes' to be be 0 so that it always
        // unblocks. In short, 0 has a special meaning for ExchangeQueue
        minOutputBatchBytes_(
            std::max(static_cast<uint64_t>(1), minOutputBatchBytes)) {
    VELOX_CHECK_NOT_NULL(pool_);
    VELOX_CHECK_NOT_NULL(executor_);
    // NOTE: the executor is used to run async response callback from the
    // exchange source. The provided executor must not be
    // folly::InlineLikeExecutor, otherwise it might cause potential deadlock as
    // the response callback in exchange client might call back into the
    // exchange source under uncertain execution context. For instance, the
    // exchange client might inline close the exchange source from a background
    // thread of the exchange source, and the close needs to wait for this
    // background thread to complete first.
    VELOX_CHECK_NULL(dynamic_cast<const folly::InlineLikeExecutor*>(executor_));
    VELOX_CHECK_GE(
        destination, 0, "Exchange client destination must not be negative");
  }

  ~ExchangeClient();

  memory::MemoryPool* pool() const {
    return pool_;
  }

  // Creates an exchange source and starts fetching data from the specified
  // upstream task. If 'close' has been called already, creates an exchange
  // source and immediately closes it to notify the upstream task that data is
  // no longer needed. Repeated calls with the same 'taskId' are ignored.
  void addRemoteTaskId(const std::string& remoteTaskId);

  void noMoreRemoteTasks();

  // Closes exchange sources.
  void close();

  // Returns runtime statistics aggregated across all of the exchange sources.
  // ExchangeClient is expected to report background CPU time by including a
  // runtime metric named ExchangeClient::kBackgroundCpuTimeMs.
  folly::F14FastMap<std::string, RuntimeMetric> stats() const;

  const std::shared_ptr<ExchangeQueue>& queue() const {
    return queue_;
  }

  /// Returns up to 'maxBytes' pages of data, but no less than one.
  ///
  /// If no data is available returns empty list and sets 'atEnd' to true if no
  /// more data is expected. If data is still expected, sets 'atEnd' to false
  /// and sets 'future' to a Future that will complete when data arrives.
  ///
  /// The data may be compressed, in which case 'maxBytes' applies to compressed
  /// size.
  std::vector<std::unique_ptr<SerializedPage>>
  next(int consumerId, uint32_t maxBytes, bool* atEnd, ContinueFuture* future);

  std::string toString() const;

  folly::dynamic toJson() const;

  std::chrono::seconds requestDataSizesMaxWaitSec() const {
    return kRequestDataSizesMaxWaitSec_;
  }

  const std::unordered_set<std::string>& getRemoteTaskIdList() const {
    return remoteTaskIds_;
  }

 private:
  struct RequestSpec {
    std::shared_ptr<ExchangeSource> source;

    // How much bytes to request from this source.  0 bytes means request data
    // sizes only.
    int64_t maxBytes;
  };

  struct ProducingSource {
    std::shared_ptr<ExchangeSource> source;
    std::vector<int64_t> remainingBytes;
  };

  std::vector<RequestSpec> pickSourcesToRequestLocked();

  void request(std::vector<RequestSpec>&& requestSpecs);

  // Handy for ad-hoc logging.
  const std::string taskId_;
  const int destination_;
  const int64_t maxQueuedBytes_;
  const std::chrono::seconds kRequestDataSizesMaxWaitSec_;

  memory::MemoryPool* const pool_;
  folly::Executor* const executor_;
  const std::shared_ptr<ExchangeQueue> queue_;

  std::unordered_set<std::string> remoteTaskIds_;
  std::vector<std::shared_ptr<ExchangeSource>> sources_;
  bool closed_{false};

  // The minimum byte size the consumer is expected to consume from
  // the exchange queue.
  const uint64_t minOutputBatchBytes_;

  // Total number of bytes in flight.
  int64_t totalPendingBytes_{0};

  // A queue of sources that have returned non-empty response from the latest
  // request.
  std::queue<ProducingSource> producingSources_;
  // A queue of sources that returned empty response from the latest request.
  std::queue<std::shared_ptr<ExchangeSource>> emptySources_;
};

} // namespace facebook::velox::exec
