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

#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConfig.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConnectorSplit.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveDataSource.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveTableHandle.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/expression/SubfieldFiltersToAst.h"
#include "velox/experimental/cudf/filter/CudfSplitBlockBloomFilter.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/common/Casts.h"
#include "velox/common/io/IoStatisticsRuntimeStats.h"
#include "velox/common/time/Timer.h"
#include "velox/connectors/hive/FileHandle.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/connectors/hive/HiveConnectorUtil.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/core/QueryCtx.h"
#include "velox/expression/ExprOptimizer.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/replace.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/search.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/transform.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/error.hpp>

#include <algorithm>

namespace facebook::velox::cudf_velox::connector::hive {

using namespace facebook::velox::connector;
using namespace facebook::velox::connector::hive;

CudfHiveDataSource::CudfHiveDataSource(
    const RowTypePtr& outputType,
    const ConnectorTableHandlePtr& tableHandle,
    const ColumnHandleMap& columnHandles,
    facebook::velox::FileHandleFactory* fileHandleFactory,
    folly::Executor* executor,
    const ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<CudfHiveConfig>& cudfHiveConfig)
    : NvtxHelper(
          nvtx3::rgb{80, 171, 241}, // CudfHive blue,
          std::nullopt,
          fmt::format("[{}]", tableHandle->name())),
      cudfHiveConfig_(cudfHiveConfig),
      fileHandleFactory_(fileHandleFactory),
      executor_(executor),
      connectorQueryCtx_(connectorQueryCtx),
      outputType_(outputType),
      pool_(connectorQueryCtx->memoryPool()),
      expressionEvaluator_(connectorQueryCtx->expressionEvaluator()) {
  // Set up column projection if needed
  auto readColumnTypes = outputType_->children();
  for (const auto& outputName : outputType_->names()) {
    auto it = columnHandles.find(outputName);
    VELOX_CHECK(
        it != columnHandles.end(),
        "ColumnHandle is missing for output column: {}",
        outputName);

    auto* handle = static_cast<const hive::HiveColumnHandle*>(it->second.get());
    readColumnSet_.emplace(handle->name());
    readColumnNames_.emplace_back(handle->name());
  }

  tableHandle_ =
      std::dynamic_pointer_cast<const hive::HiveTableHandle>(tableHandle);
  VELOX_CHECK_NOT_NULL(
      tableHandle_, "TableHandle must be an instance of HiveTableHandle");

  // Copy subfield filters.
  for (const auto& [k, v] : tableHandle_->subfieldFilters()) {
    subfieldFilters_.emplace(k.clone(), v->clone());
  }

  // Extract additional simple filters from remainingFilter (same as CPU path).
  // This extracts single-column filters like "col = 'X'" or "col <> 'Y'" from
  // complex expressions and adds them to subfieldFilters_ for pushdown.
  double sampleRate = tableHandle_->sampleRate();
  auto remainingFilter =
      facebook::velox::connector::hive::extractFiltersFromRemainingFilter(
          tableHandle_->remainingFilter(),
          expressionEvaluator_,
          subfieldFilters_,
          sampleRate);

  // Add fields in the filter to the columns to read if not there
  for (const auto& [field, _] : subfieldFilters_) {
    if (readColumnSet_.count(field.toString()) == 0) {
      readColumnSet_.emplace(field.toString());
      readColumnNames_.emplace_back(field.toString());
    }
  }
  // Optimize (rewrites + constant folding) the remaining filter before
  // evaluator selection so CudfFunctions never see scalar-only operand sets.
  // TODO: ConnectorQueryCtx does not expose the session QueryCtx, only an
  // ExpressionEvaluator, so constant folding here runs against a transient
  // QueryCtx with default query config rather than the session's. Passing the
  // real session QueryCtx (e.g. by exposing it on ConnectorQueryCtx) should be
  // figured out later. A local QueryCtx is required because
  // expression::optimize constant-folds through exec::ExprSet, whose
  // constructor dereferences the QueryCtx unconditionally; a null QueryCtx
  // would crash.
  auto optimizeQueryCtx = core::QueryCtx::create();
  optimizedRemainingFilter_ = remainingFilter
      ? expression::optimize(remainingFilter, optimizeQueryCtx.get(), pool_)
      : nullptr;
  if (optimizedRemainingFilter_) {
    // Add fields referenced by the filter to the columns to read. Collect from
    // the optimized expression since folding may drop branches and the columns
    // they reference. Read-column order does not affect results: the data
    // source projects its output to the requested output type.
    for (const auto& name : referencedInputFields(optimizedRemainingFilter_)) {
      if (readColumnSet_.count(name) == 0) {
        readColumnSet_.emplace(name);
        readColumnNames_.emplace_back(name);
      }
    }

    // TODO: Prune struct columns to the subfields referenced by the remaining
    // filter; currently the whole column is read even if only one field is
    // used.

    // The filter is already optimized and constant folded above, so compile it
    // directly.
    auto const remainingFilterType = getTableRowType();
    cudfRemainingFilterExpression_ = createCudfExpression(
        optimizedRemainingFilter_, remainingFilterType, pool_);
  }

  // Build a combined AST for all subfield filters once. This is query-constant
  // and doesn't depend on split-specific state.
  if (!subfieldFilters_.empty()) {
    auto const readerFilterType = getTableRowType();
    subfieldFilterAst_ = &createAstFromSubfieldFilters(
        subfieldFilters_, subfieldTree_, subfieldScalars_, readerFilterType);
  }

  VELOX_CHECK_NOT_NULL(fileHandleFactory_, "No FileHandleFactory present");

  // Create empty IOStats and FsStats for later use
  ioStatistics_ = std::make_shared<io::IoStatistics>();
  ioStats_ = std::make_shared<facebook::velox::IoStats>();
}

std::unique_ptr<CudfSplitReader> CudfHiveDataSource::createCudfSplitReader() {
  return std::make_unique<CudfSplitReader>(
      split_,
      tableHandle_,
      outputType_,
      readColumnNames_,
      fileHandleFactory_,
      executor_,
      connectorQueryCtx_,
      cudfHiveConfig_,
      ioStatistics_,
      ioStats_,
      subfieldFilterAst_);
}

void CudfHiveDataSource::convertSplit(std::shared_ptr<ConnectorSplit> split) {
  // Dynamic cast split to `CudfHiveConnectorSplit`
  if (std::dynamic_pointer_cast<CudfHiveConnectorSplit>(split)) {
    split_ = std::dynamic_pointer_cast<CudfHiveConnectorSplit>(split);
    return;
  }

  // Convert `HiveConnectorSplit` to `CudfHiveConnectorSplit`
  auto hiveSplit = checkedPointerCast<hive::HiveConnectorSplit>(split);

  VELOX_CHECK_EQ(
      hiveSplit->fileFormat,
      dwio::common::FileFormat::PARQUET,
      "Unsupported file format for conversion from HiveConnectorSplit to CudfHiveConnectorSplit");

  // Remove "file:" prefix from the file path if present
  std::string cleanedPath = hiveSplit->filePath;
  constexpr std::string_view kFilePrefix = "file:";
  constexpr std::string_view kS3APrefix = "s3a:";
  if (cleanedPath.compare(0, kFilePrefix.size(), kFilePrefix) == 0) {
    cleanedPath = cleanedPath.substr(kFilePrefix.size());
  } else if (cleanedPath.compare(0, kS3APrefix.size(), kS3APrefix) == 0) {
    // KvikIO does not support "s3a:" prefix. We need to translate it to "s3:".
    cleanedPath.erase(kS3APrefix.size() - 2, 1);
  }

  auto cudfHiveSplitBuilder = CudfHiveConnectorSplitBuilder(cleanedPath)
                                  .start(hiveSplit->start)
                                  .length(hiveSplit->length)
                                  .connectorId(hiveSplit->connectorId)
                                  .splitWeight(hiveSplit->splitWeight);
  for (auto const& infoColumn : hiveSplit->infoColumns) {
    cudfHiveSplitBuilder.infoColumn(infoColumn.first, infoColumn.second);
  }
  split_ = cudfHiveSplitBuilder.build();

  VLOG(1) << "Adding split " << split_->toString();
}

void CudfHiveDataSource::addSplit(std::shared_ptr<ConnectorSplit> split) {
  // Virtual method for class-specific conversion of the split
  convertSplit(split);

  cudfSplitReader_ = createCudfSplitReader();
  cudfSplitReader_->prepareSplit(runtimeStats_);

  // Check if preloaded splits should start pre-fetching the first pass of
  // column chunks.
  const bool isPreloadedSplit = split->dataSource != nullptr;
  if (isPreloadedSplit &&
      cudfHiveConfig_->preloadColumnChunksSession(
          connectorQueryCtx_->sessionProperties())) {
    cudfSplitReader_->startColumnChunkFetch();
  }

  // TODO: `completedBytes_` should be updated in `next()` as we read more and
  // more table bytes
  try {
    const auto fileHandleKey = FileHandleKey{
        .filename = split_->filePath,
        .tokenProvider = connectorQueryCtx_->fsTokenProvider()};
    auto fileProperties = FileProperties{};
    auto const fileHandleCachePtr = fileHandleFactory_->generate(
        fileHandleKey, &fileProperties, ioStats_ ? ioStats_.get() : nullptr);
    if (fileHandleCachePtr.get() and fileHandleCachePtr.get()->file) {
      completedBytes_ += fileHandleCachePtr->file->size();
    }
  } catch (const std::exception& e) {
    // Unable to get the file size, log a warning and continue
    LOG(WARNING) << "Failed to get file size for " << split_->filePath << ": "
                 << e.what();
  }
}

void CudfHiveDataSource::setFromDataSource(std::unique_ptr<DataSource> source) {
  auto* preparedSource = checkedPointerCast<CudfHiveDataSource>(source.get());

  split_ = std::move(preparedSource->split_);
  runtimeStats_.skippedSplits += preparedSource->runtimeStats_.skippedSplits;
  runtimeStats_.processedSplits +=
      preparedSource->runtimeStats_.processedSplits;
  runtimeStats_.skippedSplitBytes +=
      preparedSource->runtimeStats_.skippedSplitBytes;
  completedBytes_ += preparedSource->completedBytes_;
  completedRows_ += preparedSource->completedRows_;

  // Drop the reader of the previous split before replacing the AST storage it
  // references below.
  cudfSplitReader_.reset();

  // The reader's Parquet filter references the AST expressions and literal
  // scalars owned by 'source', which is freed right after this call. Adopt that
  // storage so the reader keeps pointing at live expressions; the nodes are
  // heap-allocated, so moving the owners does not move the expressions.
  subfieldScalars_ = std::move(preparedSource->subfieldScalars_);
  subfieldTree_ = std::move(preparedSource->subfieldTree_);
  subfieldFilterAst_ = preparedSource->subfieldFilterAst_;

  cudfSplitReader_ = std::move(preparedSource->cudfSplitReader_);
  VELOX_CHECK_NOT_NULL(cudfSplitReader_);

  // 'source' owns the query context the reader was prepared with and is
  // freed right after this call.
  cudfSplitReader_->setConnectorQueryCtx(connectorQueryCtx_);

  // Start column chunk fetch if it is not already started
  cudfSplitReader_->startColumnChunkFetch();

  // The adopted reader keeps writing I/O statistics to the objects of
  // 'source', so carry the balance accumulated here over to those.
  preparedSource->ioStatistics_->merge(*ioStatistics_);
  ioStatistics_ = std::move(preparedSource->ioStatistics_);
  preparedSource->ioStats_->merge(*ioStats_);
  ioStats_ = std::move(preparedSource->ioStats_);
}

void CudfHiveDataSource::addDynamicFilter(
    column_index_t outputChannel,
    const std::shared_ptr<common::Filter>& filter) {
  VELOX_CHECK_NOT_NULL(filter);
  VELOX_CHECK_LT(outputChannel, outputType_->size());

  const auto& columnType = outputType_->childAt(outputChannel);
  const bool isSupportedInteger = columnType == TINYINT() ||
      columnType == SMALLINT() || columnType == INTEGER() ||
      columnType == BIGINT();
  const auto kind = filter->kind();
  const bool isIntegerValues =
      kind == common::FilterKind::kBigintValuesUsingHashTable ||
      kind == common::FilterKind::kBigintValuesUsingBitmask;
  const bool isBloom =
      kind == common::FilterKind::kBigintValuesUsingBloomFilter;
  auto field = common::Subfield::create(readColumnNames_[outputChannel]);

  if (!isSupportedInteger) {
    VELOX_UNSUPPORTED(
        "cuDF Hive scan cannot apply a dynamic filter to column: {}",
        readColumnNames_[outputChannel]);
  }
  if (isIntegerValues) {
    std::vector<int64_t> values;
    if (kind == common::FilterKind::kBigintValuesUsingHashTable) {
      values =
          static_cast<const common::BigintValuesUsingHashTable*>(filter.get())
              ->values();
    } else {
      values =
          static_cast<const common::BigintValuesUsingBitmask*>(filter.get())
              ->values();
    }
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    dynamicIntegerFilters_.insert_or_assign(
        outputChannel,
        DynamicIntegerFilter{std::move(values), filter->testNull(), nullptr});
    dynamicBloomFilters_.erase(outputChannel);
    dynamicFilters_.erase(*field);
  } else if (isBloom) {
    auto bloom =
        std::dynamic_pointer_cast<common::BigintValuesUsingBloomFilter>(filter);
    VELOX_CHECK_NOT_NULL(bloom);
    dynamicIntegerFilters_.erase(outputChannel);
    dynamicBloomFilters_.insert_or_assign(
        outputChannel, DynamicBloomFilter{std::move(bloom), nullptr});
    dynamicFilters_.erase(*field);
  } else if (kind == common::FilterKind::kBigintRange) {
    dynamicIntegerFilters_.erase(outputChannel);
    dynamicBloomFilters_.erase(outputChannel);
    dynamicFilters_.insert_or_assign(field->clone(), filter->clone());
  } else {
    VELOX_UNSUPPORTED(
        "cuDF Hive scan cannot apply dynamic filter kind {} to column: {}",
        static_cast<int>(kind),
        readColumnNames_[outputChannel]);
  }

  dynamicFilterChannels_.insert(outputChannel);
  dynamicFilterExpr_ = nullptr;
  dynamicFilterTree_.reset();
  dynamicFilterScalars_.clear();
  if (!dynamicFilters_.empty()) {
    dynamicFilterTree_ = std::make_unique<cudf::ast::tree>();
    dynamicFilterExpr_ = &createAstFromSubfieldFilters(
        dynamicFilters_,
        *dynamicFilterTree_,
        dynamicFilterScalars_,
        getTableRowType());
  }
}

std::optional<RowVectorPtr> CudfHiveDataSource::next(
    uint64_t size,
    velox::ContinueFuture& /* future */) {
  VELOX_CHECK_NOT_NULL(split_, "No split present. Call addSplit() first.");
  VELOX_CHECK_NOT_NULL(cudfSplitReader_, "No split to process.");
  auto chunkOpt = cudfSplitReader_->next(size);
  if (!chunkOpt.has_value()) {
    cudfSplitReader_->resetSplit();
    return nullptr;
  }
  auto cudfTable = std::move(chunkOpt.value());
  auto stream = cudfSplitReader_->stream();
  if (cudfTable->num_rows() > 0) {
    for (auto channel : dynamicFilterChannels_) {
      VELOX_CHECK_LT(channel, cudfTable->num_columns());
      if (cudfTable->view().column(channel).type() !=
          veloxToCudfDataType(outputType_->childAt(channel))) {
        VELOX_UNSUPPORTED(
            "cuDF Hive scan column type does not match dynamic filter type: {}",
            readColumnNames_[channel]);
      }
    }
    if (dynamicFilterExpr_) {
      auto mask = cudf::compute_column(
          cudfTable->view(), *dynamicFilterExpr_, stream, get_temp_mr());
      cudfTable = cudf::apply_retention_mask(
          *cudfTable, mask->view(), stream, get_output_mr());
    }

    for (auto& [channel, dynamicFilter] : dynamicIntegerFilters_) {
      if (cudfTable->num_rows() == 0) {
        break;
      }
      auto inputColumn = cudfTable->view().column(channel);
      if (!dynamicFilter.deviceValues) {
        dynamicFilter.deviceValues = cudf::make_fixed_width_column(
            cudf::data_type{cudf::type_id::INT64},
            dynamicFilter.values.size(),
            cudf::mask_state::UNALLOCATED,
            stream,
            get_temp_mr());
        CUDF_CUDA_TRY(cudaMemcpyAsync(
            dynamicFilter.deviceValues->mutable_view().data<int64_t>(),
            dynamicFilter.values.data(),
            dynamicFilter.values.size() * sizeof(int64_t),
            cudaMemcpyHostToDevice,
            stream.get()));
      }

      std::unique_ptr<cudf::column> int64Input;
      if (inputColumn.type().id() != cudf::type_id::INT64) {
        int64Input = cudf::cast(
            inputColumn,
            cudf::data_type{cudf::type_id::INT64},
            stream,
            get_temp_mr());
        inputColumn = int64Input->view();
      }
      auto mask = cudf::contains(
          dynamicFilter.deviceValues->view(),
          inputColumn,
          stream,
          get_temp_mr());
      if (dynamicFilter.nullAllowed && mask->view().has_nulls()) {
        const auto trueScalar =
            cudf::numeric_scalar<bool>(true, true, stream, get_temp_mr());
        mask = cudf::replace_nulls(
            mask->view(), trueScalar, stream, get_temp_mr());
      }
      cudfTable = cudf::apply_retention_mask(
          *cudfTable, mask->view(), stream, get_output_mr());
    }

    for (auto& [channel, dynamicFilter] : dynamicBloomFilters_) {
      if (cudfTable->num_rows() == 0) {
        break;
      }
      auto inputColumn = cudfTable->view().column(channel);
      const auto blocks = dynamicFilter.filter->blocks();
      if (!dynamicFilter.deviceBlocks) {
        dynamicFilter.deviceBlocks = std::make_unique<rmm::device_buffer>(
            dynamicFilter.filter->blocksByteSize(), stream, get_temp_mr());
        CUDF_CUDA_TRY(cudaMemcpyAsync(
            dynamicFilter.deviceBlocks->data(),
            blocks.data(),
            dynamicFilter.filter->blocksByteSize(),
            cudaMemcpyHostToDevice,
            stream.get()));
      }
      auto mask = makeSplitBlockBloomFilterMask(
          inputColumn,
          static_cast<const uint32_t*>(dynamicFilter.deviceBlocks->data()),
          blocks.size(),
          sizeof(SplitBlockBloomFilter::Block) / sizeof(uint32_t),
          dynamicFilter.filter->testNull(),
          stream,
          get_temp_mr());
      cudfTable = cudf::apply_retention_mask(
          *cudfTable, mask->view(), stream, get_output_mr());
    }
  }

  uint64_t filterTimeUs{0};
  if (optimizedRemainingFilter_) {
    MicrosecondWallTimer filterTimer(&filterTimeUs);
    auto cudfTableColumns = cudfTable->release();
    std::vector<cudf::column_view> inputViews;
    inputViews.reserve(cudfTableColumns.size());
    for (auto& col : cudfTableColumns) {
      inputViews.push_back(col->view());
    }
    auto filterResult =
        cudfRemainingFilterExpression_->eval(inputViews, stream, get_temp_mr());
    auto originalTable =
        std::make_unique<cudf::table>(std::move(cudfTableColumns));
    cudfTable = cudf::apply_retention_mask(
        *originalTable, asView(filterResult), stream, get_output_mr());
  }
  totalRemainingFilterTime_.fetch_add(
      filterTimeUs * 1000, std::memory_order_relaxed);

  const auto nRows = cudfTable->num_rows();

  if (outputType_->size() < cudfTable->num_columns()) {
    auto cudfTableColumns = cudfTable->release();
    std::vector<std::unique_ptr<cudf::column>> outputColumns;
    outputColumns.reserve(outputType_->size());
    std::move(
        cudfTableColumns.begin(),
        cudfTableColumns.begin() + outputType_->size(),
        std::back_inserter(outputColumns));
    cudfTable = std::make_unique<cudf::table>(std::move(outputColumns));
  }

  // TODO (dm): Should we only enable table scan if cudf is registered?
  // Earlier we could enable cudf table scans without using other cudf operators
  // We still can, but I'm wondering if this is the right thing to do
  auto output = cudfIsRegistered()
      ? std::make_shared<CudfVector>(
            pool_, outputType_, nRows, std::move(cudfTable), stream)
      : with_arrow::toVeloxColumn(
            cudfTable->view(), pool_, outputType_, stream, get_temp_mr());
  stream.sync();

  VELOX_CHECK_NOT_NULL(output, "Cudf to Velox conversion yielded a nullptr");

  completedRows_ += output->size();

  // TODO: Update `completedBytes_` here instead of in `addSplit()`

  return output;
}

std::unordered_map<std::string, RuntimeMetric>
CudfHiveDataSource::getRuntimeStats() {
  auto result = runtimeStats_.toRuntimeMetricMap();
  io::addIoStatsToRuntimeStats(*ioStatistics_, "", result);
  if (const auto it = result.find(std::string(io::kStorageReadBytes));
      it != result.end()) {
    // Preserve the DWIO value before a ReadFile-layer value overrides it.
    // Overread bytes are defined relative to this counter.
    result.emplace(kDwioStorageReadBytes, it->second);
  }
  // Preserve a zero-valued totalScanTime before scan timing is recorded.
  result.insert({
      {std::string(io::kTotalScanTime),
       RuntimeMetric(
           ioStatistics_->totalScanTimeNs(), RuntimeCounter::Unit::kNanos)},
      {std::string(Connector::kTotalRemainingFilterTime),
       RuntimeMetric(
           totalRemainingFilterTime_.load(std::memory_order_relaxed),
           RuntimeCounter::Unit::kNanos)},
  });
  const auto& ioStats = ioStats_->stats();
  for (const auto& [key, value] : ioStats) {
    // Keep the ReadFile-layer value under the established key.
    if (key == io::kStorageReadBytes) {
      result[std::string(key)] = value;
    } else {
      result.emplace(key, value);
    }
  }
  return result;
}

const RowTypePtr CudfHiveDataSource::getTableRowType() {
  if (cachedTableRowType_) {
    return cachedTableRowType_;
  }
  if (tableHandle_->dataColumns()) {
    std::vector<std::string> names;
    std::vector<TypePtr> types;
    for (const auto& name : readColumnNames_) {
      auto parsedType = tableHandle_->dataColumns()->findChild(name);
      names.emplace_back(std::move(name));
      types.push_back(parsedType);
    }
    cachedTableRowType_ = ROW(std::move(names), std::move(types));
    return cachedTableRowType_;
  }
  cachedTableRowType_ = outputType_;
  return cachedTableRowType_;
}

} // namespace facebook::velox::cudf_velox::connector::hive
