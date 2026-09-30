/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "velox/experimental/cudf/connectors/hive/CudfIntegerMembership.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <thrust/transform.h>

#include <type_traits>

namespace facebook::velox::cudf_velox::connector::hive {
namespace {
enum class FilterKind { kRange, kBitmap, kValues };

template <typename T, FilterKind kKind>
struct IntegerMembership {
  const T* values;
  const cudf::bitmask_type* nulls;
  cudf::size_type offset;
  const void* filter;
  cudf::size_type filterSize;
  int64_t minimum;
  int64_t maximum;
  bool nullAllowed;
  bool* mask;
  bool initialize;

  __device__ bool operator()(cudf::size_type row) const {
    if (!initialize && !mask[row]) {
      return false;
    }
    if (nulls && !cudf::bit_is_set(nulls, row + offset)) {
      return nullAllowed;
    }
    if constexpr (kKind == FilterKind::kRange) {
      return values[row] >= minimum && values[row] <= maximum;
    } else if constexpr (kKind == FilterKind::kBitmap) {
      const auto index =
          static_cast<uint64_t>(values[row]) - static_cast<uint64_t>(minimum);
      const auto* bitmap = static_cast<const uint32_t*>(filter);
      return index < static_cast<uint64_t>(filterSize) * 32 &&
          (bitmap[index / 32] & (uint32_t{1} << (index % 32))) != 0;
    } else {
      const auto* keys = static_cast<const T*>(filter);
      cudf::size_type lower = 0;
      auto upper = filterSize;
      while (lower < upper) {
        const auto middle = lower + (upper - lower) / 2;
        if (keys[middle] < values[row]) {
          lower = middle + 1;
        } else {
          upper = middle;
        }
      }
      return lower < filterSize && keys[lower] == values[row];
    }
  }
};
template <FilterKind kKind>
void applyIntegerFilter(
    const cudf::column_view& input,
    const cudf::column_view& filter,
    int64_t minimum,
    int64_t maximum,
    bool nullAllowed,
    std::unique_ptr<cudf::column>& mask,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  const bool initialize = !mask;
  if (initialize) {
    mask = cudf::make_fixed_width_column(
        cudf::data_type{cudf::type_id::BOOL8},
        input.size(),
        cudf::mask_state::UNALLOCATED,
        stream,
        mr);
  }
  auto output = mask->mutable_view();
  cudf::type_dispatcher(input.type(), [&]<typename T>() {
    if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
      const void* filterData = nullptr;
      if constexpr (kKind == FilterKind::kBitmap) {
        filterData = filter.data<uint32_t>();
      } else if constexpr (kKind == FilterKind::kValues) {
        filterData = filter.data<T>();
      }
      auto rows = cuda::counting_iterator<cudf::size_type>{0};
      thrust::transform(
          rmm::exec_policy_nosync(stream, mr),
          rows,
          rows + input.size(),
          output.data<bool>(),
          IntegerMembership<T, kKind>{
              input.data<T>(),
              input.null_mask(),
              input.offset(),
              filterData,
              filter.size(),
              minimum,
              maximum,
              nullAllowed,
              output.data<bool>(),
              initialize});
    } else {
      CUDF_FAIL("Integer membership requires a signed integer column");
    }
  });
}
} // namespace

void applyIntegerMembership(
    const cudf::column_view& input,
    const cudf::column_view& filter,
    int64_t minimum,
    bool nullAllowed,
    std::unique_ptr<cudf::column>& mask,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  if (filter.type().id() == cudf::type_id::UINT32) {
    applyIntegerFilter<FilterKind::kBitmap>(
        input, filter, minimum, 0, nullAllowed, mask, stream, mr);
  } else {
    applyIntegerFilter<FilterKind::kValues>(
        input, filter, 0, 0, nullAllowed, mask, stream, mr);
  }
}

void applyIntegerRangeToMask(
    const cudf::column_view& input,
    int64_t lower,
    int64_t upper,
    bool nullAllowed,
    std::unique_ptr<cudf::column>& mask,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  applyIntegerFilter<FilterKind::kRange>(
      input, {}, lower, upper, nullAllowed, mask, stream, mr);
}
} // namespace facebook::velox::cudf_velox::connector::hive
