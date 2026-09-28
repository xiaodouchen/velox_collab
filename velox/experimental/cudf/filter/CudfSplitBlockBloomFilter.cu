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

#include "velox/experimental/cudf/filter/CudfSplitBlockBloomFilter.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

namespace facebook::velox::cudf_velox {
namespace {

constexpr int kThreadsPerBlock = 256;

// Rejects layouts that the community SplitBlockBloomFilter cannot represent.
void validateBloomFilterLayout(
    const uint32_t* deviceBlocks,
    std::size_t numBlocks,
    int32_t wordsPerBlock) {
  CUDF_EXPECTS(deviceBlocks != nullptr, "Bloom filter blocks cannot be null");
  CUDF_EXPECTS(numBlocks > 0, "Bloom filter must contain at least one block");
  CUDF_EXPECTS(
      wordsPerBlock == 4 || wordsPerBlock == 8,
      "Bloom filter blocks must contain four or eight words");
}

__device__ __constant__ uint32_t kBloomSalts[8] = {
    0x2df1424bU,
    0x44974d91U,
    0x47b6137bU,
    0x5c6bfb31U,
    0x705495c7U,
    0x8824ad5bU,
    0x9efc4947U,
    0xa2b7289dU,
};

__device__ __forceinline__ uint64_t twangMix64(uint64_t value) {
  value = (~value) + (value << 21);
  value ^= value >> 24;
  value = value + (value << 3) + (value << 8);
  value ^= value >> 14;
  value = value + (value << 2) + (value << 4);
  value ^= value >> 28;
  value = value + (value << 31);
  return value;
}

__device__ __forceinline__ std::size_t bloomBlockIndex(
    uint64_t hash,
    std::size_t numBlocks) {
  return ((hash >> 32) * numBlocks) >> 32;
}

__device__ __forceinline__ uint32_t bloomSalt(int word, int32_t wordsPerBlock) {
  return kBloomSalts[wordsPerBlock == 8 ? word : word * 2];
}

template <typename T>
__global__ void insertBloomFilter(
    const T* values,
    cudf::size_type size,
    const cudf::bitmask_type* nullMask,
    cudf::size_type nullMaskOffset,
    uint32_t* blocks,
    std::size_t numBlocks,
    int32_t wordsPerBlock) {
  const auto row = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= static_cast<uint64_t>(size) ||
      (nullMask &&
       !cudf::bit_is_set(
           nullMask, nullMaskOffset + static_cast<cudf::size_type>(row)))) {
    return;
  }

  const auto hash =
      twangMix64(static_cast<uint64_t>(static_cast<int64_t>(values[row])));
  auto* words = blocks + bloomBlockIndex(hash, numBlocks) * wordsPerBlock;
  const auto maskHash = static_cast<uint32_t>(hash);
  for (int word = 0; word < wordsPerBlock; ++word) {
    atomicOr(
        words + word,
        1U << ((bloomSalt(word, wordsPerBlock) * maskHash) >> 27));
  }
}

template <typename T>
__global__ void testBloomFilter(
    const T* values,
    cudf::size_type size,
    const cudf::bitmask_type* nullMask,
    cudf::size_type nullMaskOffset,
    const uint32_t* blocks,
    std::size_t numBlocks,
    int32_t wordsPerBlock,
    bool nullAllowed,
    bool* output) {
  const auto row = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= static_cast<uint64_t>(size)) {
    return;
  }
  if (nullMask &&
      !cudf::bit_is_set(
          nullMask, nullMaskOffset + static_cast<cudf::size_type>(row))) {
    output[row] = nullAllowed;
    return;
  }

  const auto hash =
      twangMix64(static_cast<uint64_t>(static_cast<int64_t>(values[row])));
  const auto* words = blocks + bloomBlockIndex(hash, numBlocks) * wordsPerBlock;
  const auto maskHash = static_cast<uint32_t>(hash);
  for (int word = 0; word < wordsPerBlock; ++word) {
    const auto bit = 1U << ((bloomSalt(word, wordsPerBlock) * maskHash) >> 27);
    if ((words[word] & bit) == 0) {
      output[row] = false;
      return;
    }
  }
  output[row] = true;
}

struct InsertDispatcher {
  cudf::column_view input;
  uint32_t* blocks;
  std::size_t numBlocks;
  int32_t wordsPerBlock;
  cuda::stream_ref stream;

  template <typename T>
    requires(std::is_integral_v<T>)
  void operator()() const {
    const auto numThreadBlocks = static_cast<unsigned int>(
        (static_cast<uint64_t>(input.size()) + kThreadsPerBlock - 1) /
        kThreadsPerBlock);
    insertBloomFilter<T>
        <<<numThreadBlocks, kThreadsPerBlock, 0, stream.get()>>>(
            input.data<T>(),
            input.size(),
            input.null_mask(),
            input.offset(),
            blocks,
            numBlocks,
            wordsPerBlock);
    CUDF_CUDA_TRY(cudaPeekAtLastError());
  }

  template <typename T>
    requires(!std::is_integral_v<T>)
  void operator()() const {
    CUDF_FAIL("Split-block Bloom filters require an integral column");
  }
};

struct TestDispatcher {
  cudf::column_view input;
  const uint32_t* blocks;
  std::size_t numBlocks;
  int32_t wordsPerBlock;
  bool nullAllowed;
  cudf::mutable_column_view output;
  cuda::stream_ref stream;

  template <typename T>
    requires(std::is_integral_v<T>)
  void operator()() const {
    const auto numThreadBlocks = static_cast<unsigned int>(
        (static_cast<uint64_t>(input.size()) + kThreadsPerBlock - 1) /
        kThreadsPerBlock);
    testBloomFilter<T><<<numThreadBlocks, kThreadsPerBlock, 0, stream.get()>>>(
        input.data<T>(),
        input.size(),
        input.null_mask(),
        input.offset(),
        blocks,
        numBlocks,
        wordsPerBlock,
        nullAllowed,
        output.data<bool>());
    CUDF_CUDA_TRY(cudaPeekAtLastError());
  }

  template <typename T>
    requires(!std::is_integral_v<T>)
  void operator()() const {
    CUDF_FAIL("Split-block Bloom filters require an integral column");
  }
};

} // namespace

void insertSplitBlockBloomFilter(
    cudf::column_view input,
    uint32_t* deviceBlocks,
    std::size_t numBlocks,
    int32_t wordsPerBlock,
    cuda::stream_ref stream) {
  validateBloomFilterLayout(deviceBlocks, numBlocks, wordsPerBlock);
  if (input.is_empty()) {
    return;
  }
  cudf::type_dispatcher(
      input.type(),
      InsertDispatcher{input, deviceBlocks, numBlocks, wordsPerBlock, stream});
}

std::unique_ptr<cudf::column> makeSplitBlockBloomFilterMask(
    cudf::column_view input,
    const uint32_t* deviceBlocks,
    std::size_t numBlocks,
    int32_t wordsPerBlock,
    bool nullAllowed,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref memoryResource) {
  validateBloomFilterLayout(deviceBlocks, numBlocks, wordsPerBlock);
  auto output = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::BOOL8},
      input.size(),
      cudf::mask_state::UNALLOCATED,
      stream,
      memoryResource);
  if (!input.is_empty()) {
    cudf::type_dispatcher(
        input.type(),
        TestDispatcher{
            input,
            deviceBlocks,
            numBlocks,
            wordsPerBlock,
            nullAllowed,
            output->mutable_view(),
            stream});
  }
  return output;
}

} // namespace facebook::velox::cudf_velox
