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

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConfig.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConnector.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConnectorSplit.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveDataSource.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveTableHandle.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/expression/SubfieldFiltersToAst.h"
#include "velox/experimental/cudf/filter/CudfSplitBlockBloomFilter.h"
#include "velox/experimental/cudf/tests/utils/CudfHiveConnectorTestBase.h"

#include "velox/common/base/Fs.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/file/tests/FaultyFile.h"
#include "velox/common/file/tests/FaultyFileSystem.h"
#include "velox/common/io/IoStatisticsRuntimeStats.h"
#include "velox/common/memory/MemoryArbitrator.h"
#include "velox/common/testutil/TempDirectoryPath.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/connectors/ConnectorRegistry.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/dwio/common/FileSink.h"
#include "velox/dwio/common/tests/utils/DataFiles.h"
#include "velox/dwio/parquet/RegisterParquetReader.h"
#include "velox/dwio/parquet/writer/Writer.h"
#include "velox/exec/Exchange.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/TableScan.h"
#include "velox/exec/VectorHasher.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/LocalExchangeSource.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/expression/ExprToSubfieldFilter.h"
#include "velox/type/Type.h"
#include "velox/type/tests/SubfieldFiltersBuilder.h"

#include <cudf/copying.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <fmt/ranges.h>
#include <folly/ScopeGuard.h>
#include <folly/synchronization/Baton.h>
#include <folly/synchronization/Latch.h>

#include <atomic>
#include <cstring>
#include <functional>
#include <limits>

using namespace facebook::velox;
using namespace facebook::velox::common::testutil;
using namespace facebook::velox::connector;
using namespace facebook::velox::core;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::common::test;
using namespace facebook::velox::tests::utils;
using namespace facebook::velox::cudf_velox;
using namespace facebook::velox::cudf_velox::exec;
using namespace facebook::velox::cudf_velox::exec::test;

namespace {
struct StatsFilterMetrics {
  cudf::size_type inputRowGroups{0};
  std::optional<cudf::size_type> rowGroupsAfterStats;
  cudf::size_type outputRows{0};
};

StatsFilterMetrics readParquetWithStatsFilter(
    const std::string& filePath,
    const RowTypePtr& rowType,
    const common::SubfieldFilters& filters,
    bool useJitFilter) {
  cudf::ast::tree tree;
  std::vector<std::unique_ptr<cudf::scalar>> scalars;
  auto const& expr =
      createAstFromSubfieldFilters(filters, tree, scalars, rowType);

  auto options =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info(filePath))
          .use_jit_filter(useJitFilter)
          .build();
  options.set_filter(expr);

  auto result = cudf::io::read_parquet(options);
  return {
      result.metadata.num_input_row_groups,
      result.metadata.num_row_groups_after_stats_filter,
      result.tbl->num_rows()};
}
} // namespace

class TableScanTest : public virtual CudfHiveConnectorTestBase {
 protected:
  void SetUp() override {
    CudfHiveConnectorTestBase::SetUp();
    ExchangeSource::factories().clear();
    ExchangeSource::registerFactory(createLocalExchangeSource);
  }

  static void SetUpTestCase() {
    CudfHiveConnectorTestBase::SetUpTestCase();
  }

  std::vector<RowVectorPtr> makeVectors(
      int32_t count,
      int32_t rowsPerVector,
      const RowTypePtr& rowType = nullptr) {
    auto inputs = rowType ? rowType : rowType_;
    return CudfHiveConnectorTestBase::makeVectors(inputs, count, rowsPerVector);
  }

  Split makeCudfHiveSplit(std::string path, int64_t splitWeight = 0) {
    return Split(makeCudfHiveConnectorSplit(std::move(path), splitWeight));
  }

  std::shared_ptr<Task> assertQuery(
      const PlanNodePtr& plan,
      const std::shared_ptr<facebook::velox::connector::ConnectorSplit>&
          parquetSplit,
      const std::string& duckDbSql) {
    return OperatorTestBase::assertQuery(plan, {parquetSplit}, duckDbSql);
  }

  std::shared_ptr<Task> assertQuery(
      const PlanNodePtr& plan,
      const Split&& split,
      const std::string& duckDbSql) {
    return OperatorTestBase::assertQuery(plan, {split}, duckDbSql);
  }

  std::shared_ptr<Task> assertQuery(
      const PlanNodePtr& plan,
      const std::vector<std::shared_ptr<TempFilePath>>& filePaths,
      const std::string& duckDbSql) {
    return CudfHiveConnectorTestBase::assertQuery(plan, filePaths, duckDbSql);
  }

  // Run query with spill enabled.
  std::shared_ptr<Task> assertQuery(
      const PlanNodePtr& plan,
      const std::vector<std::shared_ptr<TempFilePath>>& filePaths,
      const std::string& spillDirectory,
      const std::string& duckDbSql) {
    return AssertQueryBuilder(plan, duckDbQueryRunner_)
        .spillDirectory(spillDirectory)
        .config(core::QueryConfig::kSpillEnabled, false)
        .config(core::QueryConfig::kAggregationSpillEnabled, false)
        .splits(makeCudfHiveConnectorSplits(filePaths))
        .assertResults(duckDbSql);
  }

  core::PlanNodePtr tableScanNode() {
    return tableScanNode(rowType_);
  }

  core::PlanNodePtr tableScanNode(const RowTypePtr& outputType) {
    auto tableHandle = makeTableHandle();
    return PlanBuilder(pool_.get())
        .startTableScan()
        .outputType(outputType)
        .tableHandle(tableHandle)
        .endTableScan()
        .planNode();
  }

  static PlanNodeStats getTableScanStats(const std::shared_ptr<Task>& task) {
    auto planStats = toPlanStats(task->taskStats());
    return std::move(planStats.at("0"));
  }

  static std::unordered_map<std::string, RuntimeMetric>
  getTableScanRuntimeStats(const std::shared_ptr<Task>& task) {
    return task->taskStats().pipelineStats[0].operatorStats[0].runtimeStats;
  }

  // Verifies I/O is bounded by one footer and one data read of the unique file.
  static void assertStorageReadStats(
      const std::unordered_map<std::string, RuntimeMetric>& runtimeStats,
      int64_t fileSize) {
    for (const auto key : {
             io::kStorageReadBytes,
             cudf_velox::connector::hive::CudfHiveDataSource::
                 kDwioStorageReadBytes,
         }) {
      const auto& metric = runtimeStats.at(std::string(key));
      EXPECT_GT(metric.sum, 0);
      EXPECT_LE(metric.sum, 2 * fileSize);
    }
  }

  static int64_t getSkippedStridesStat(const std::shared_ptr<Task>& task) {
    VELOX_NYI(
        "RuntimeStats not yet implemented for the cudf CudfHiveConnector");
    // return getTableScanRuntimeStats(task)["skippedStrides"].sum;
  }

  static int64_t getSkippedSplitsStat(const std::shared_ptr<Task>& task) {
    VELOX_NYI(
        "RuntimeStats not yet implemented for the cudf CudfHiveConnector");
    // return getTableScanRuntimeStats(task)["skippedSplits"].sum;
  }

  static void waitForFinishedDrivers(
      const std::shared_ptr<Task>& task,
      uint32_t n) {
    // Limit wait to 10 seconds.
    size_t iteration{0};
    while (task->numFinishedDrivers() < n and iteration < 100) {
      /* sleep override */
      usleep(100'000); // 0.1 second.
      ++iteration;
    }
    ASSERT_EQ(n, task->numFinishedDrivers());
  }

  void assertDecimalScanRoundTrip(
      const RowVectorPtr& vector,
      const RowTypePtr& rowType) {
    auto filePath = TempFilePath::create();
    auto fs = filesystems::getFileSystem(filePath->getPath(), {});
    auto writeFile = fs->openFileForWrite(
        filePath->getPath(),
        {.shouldCreateParentDirectories = true,
         .shouldThrowOnFileAlreadyExists = false});
    auto sink = std::make_unique<dwio::common::WriteFileSink>(
        std::move(writeFile), filePath->getPath());
    auto writerPool =
        rootPool_->addAggregateChild("TableScanTest.ParquetWriter");
    dwio::common::WriterOptions options;
    options.memoryPool = writerPool.get();
    auto parquetOptions = std::make_shared<parquet::ParquetWriterOptions>();
    parquetOptions->enableStoreDecimalAsInteger = true;
    options.formatSpecificOptions = std::move(parquetOptions);
    parquet::Writer writer(std::move(sink), options, writerPool, rowType);
    writer.write(vector);
    writer.close();
    createDuckDbTable({vector});

    auto assignments =
        facebook::velox::exec::test::HiveConnectorTestBase::allRegularColumns(
            rowType);
    auto plan = PlanBuilder(pool_.get())
                    .startTableScan()
                    .connectorId(kCudfHiveConnectorId)
                    .outputType(rowType)
                    .dataColumns(rowType)
                    .assignments(assignments)
                    .endTableScan()
                    .planNode();

    AssertQueryBuilder(plan, duckDbQueryRunner_)
        .splits(makeCudfHiveConnectorSplits({filePath}))
        .assertResults("SELECT * FROM tmp");
  }

  RowTypePtr rowType_{
      ROW({"c0", "c1", "c2", "c3", "c4", "c5", "c6"},
          {INTEGER(),
           VARCHAR(),
           TINYINT(),
           DOUBLE(),
           BIGINT(),
           VARCHAR(),
           REAL()})};
};

class TableScanTestParameterized : public TableScanTest,
                                   public testing::WithParamInterface<bool> {};

TEST_P(TableScanTestParameterized, allColumns) {
  auto vectors = makeVectors(10, 1'000);
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);

  createDuckDbTable(vectors);
  auto plan = tableScanNode();

  const std::string duckDbSql = "SELECT * FROM tmp";

  // Helper to test scan all columns for the given splits
  auto testScanAllColumns =
      [&](const std::vector<std::shared_ptr<
              facebook::velox::connector::ConnectorSplit>>& splits) {
        auto task = AssertQueryBuilder(duckDbQueryRunner_)
                        .plan(plan)
                        .splits(splits)
                        .assertResults(duckDbSql);

        // A quick sanity check for memory usage reporting. Check that peak
        // total memory usage for the project node is > 0.
        auto planStats = toPlanStats(task->taskStats());
        auto scanNodeId = plan->id();
        auto it = planStats.find(scanNodeId);
        ASSERT_TRUE(it != planStats.end());
        // TODO (dm): enable this test once we start to track gpu memory
        // ASSERT_TRUE(it->second.peakMemoryBytes > 0);

        //  Verifies there is no dynamic filter stats.
        ASSERT_TRUE(it->second.dynamicFilterStats.empty());

        // TODO: We are not writing any customStats yet so disable this check
        // ASSERT_LT(0, it->second.customStats.at("ioWaitWallNanos").sum);
      };

  const bool useBufferedInput = GetParam();
  auto config = std::unordered_map<std::string, std::string>{
      {facebook::velox::cudf_velox::connector::hive::CudfHiveConfig::
           kUseBufferedInput,
       useBufferedInput ? "true" : "false"}};
  resetCudfHiveConnector(
      std::make_shared<config::ConfigBase>(std::move(config)));

  // Test scan all columns with CudfHiveConnectorSplits
  {
    auto splits = makeCudfHiveConnectorSplits({filePath});
    testScanAllColumns(splits);
  }

  // Test scan all columns with HiveConnectorSplits
  {
    std::vector<std::shared_ptr<facebook::velox::connector::ConnectorSplit>>
        splits;
    splits.push_back(
        facebook::velox::connector::hive::HiveConnectorSplitBuilder(
            filePath->getPath())
            .connectorId(kCudfHiveConnectorId)
            .fileFormat(dwio::common::FileFormat::PARQUET)
            .build());
    testScanAllColumns(splits);
  }
}

// Reads several splits of a multi-row-group file with chunk and pass read
// limits small enough that each split is read as multiple row group passes,
// each yielding multiple table chunks.
TEST_P(TableScanTestParameterized, allColumnsWithRowGroupPasses) {
  auto vectors = makeVectors(10, 1'000);
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);

  createDuckDbTable(vectors);
  const std::string duckDbSql =
      "SELECT * FROM tmp UNION ALL "
      "SELECT * FROM tmp UNION ALL "
      "SELECT * FROM tmp UNION ALL "
      "SELECT * FROM tmp UNION ALL "
      "SELECT * FROM tmp";

  auto splits = makeCudfHiveConnectorSplits(
      {filePath, filePath, filePath, filePath, filePath});

  const bool useBufferedInput = GetParam();
  auto config = std::unordered_map<std::string, std::string>{
      {facebook::velox::cudf_velox::connector::hive::CudfHiveConfig::
           kMaxChunkReadLimit,
       "8192"},
      {facebook::velox::cudf_velox::connector::hive::CudfHiveConfig::
           kMaxPassReadLimit,
       "32768"},
      {facebook::velox::cudf_velox::connector::hive::CudfHiveConfig::
           kUseBufferedInput,
       useBufferedInput ? "true" : "false"}};
  resetCudfHiveConnector(
      std::make_shared<config::ConfigBase>(std::move(config)));

  auto plan = tableScanNode();
  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .plan(plan)
                  .splits(splits)
                  .assertResults(duckDbSql);

  auto planStats = toPlanStats(task->taskStats());
  auto it = planStats.find(plan->id());
  ASSERT_TRUE(it != planStats.end());

  // Reading in chunks must not change the number of rows returned.
  const auto& scanStats = it->second.operatorStatsFor("TableScan");
  EXPECT_EQ(scanStats.outputRows, 5 * 10 * 1'000);

  // Splitting the read into chunks must produce more than one output vector
  // per split.
  EXPECT_GT(scanStats.outputVectors, splits.size());

  //  Verifies there is no dynamic filter stats.
  ASSERT_TRUE(it->second.dynamicFilterStats.empty());
}

// Splits prepared in the background by the preloader must produce the same
// results as splits prepared on the driver thread.
TEST_F(TableScanTest, preloadSplits) {
  auto filePaths = makeFilePaths(10);
  auto vectors = makeVectors(10, 1'000);
  for (auto i = 0; i < vectors.size(); ++i) {
    writeToFile(filePaths[i]->getPath(), vectors[i]);
  }
  createDuckDbTable(vectors);

  auto plan = tableScanNode();
  auto task = AssertQueryBuilder(plan, duckDbQueryRunner_)
                  .config(core::QueryConfig::kMaxSplitPreloadPerDriver, "10")
                  .splits(makeCudfHiveConnectorSplits(filePaths))
                  .assertResults("SELECT * FROM tmp");

  auto planStats = toPlanStats(task->taskStats());
  const auto& customStats = planStats.at(plan->id()).customStats;
  ASSERT_EQ(customStats.count(std::string(TableScan::kPreloadedSplits)), 1);
  EXPECT_EQ(
      customStats.at(std::string(TableScan::kPreloadedSplits)).sum,
      filePaths.size());
}

// A busy IO thread pool never runs the queued preload tasks, so every split is
// prepared inline by the driver that comes to read it.
TEST_F(TableScanTest, preloadingSplitClose) {
  auto filePaths = makeFilePaths(20);
  auto vectors = makeVectors(20, 100);
  for (auto i = 0; i < vectors.size(); ++i) {
    writeToFile(filePaths[i]->getPath(), vectors[i]);
  }
  createDuckDbTable(vectors);

  auto* ioExecutor = ioExecutor_.get();
  folly::Latch latch(ioExecutor->numThreads());
  std::vector<folly::Baton<>> batons(ioExecutor->numThreads());
  // Simulate a busy IO thread pool by blocking all its threads.
  for (auto& baton : batons) {
    ioExecutor->add([&]() {
      baton.wait();
      latch.count_down();
    });
  }

  ASSERT_EQ(Task::numRunningTasks(), 0);
  auto plan = tableScanNode();
  auto task = AssertQueryBuilder(plan, duckDbQueryRunner_)
                  .config(core::QueryConfig::kMaxSplitPreloadPerDriver, "4")
                  .splits(makeCudfHiveConnectorSplits(filePaths))
                  .assertResults("SELECT * FROM tmp");

  auto planStats = toPlanStats(task->taskStats());
  EXPECT_GT(
      planStats.at(plan->id())
          .customStats.at(std::string(TableScan::kPreloadedSplits))
          .sum,
      1);

  task.reset();
  // Once all task references are cleared, all the tasks should be destroyed.
  ASSERT_EQ(Task::numRunningTasks(), 0);
  // Unblock the IO thread pool.
  for (auto& baton : batons) {
    baton.post();
  }
  latch.wait();
}

// A query that stops early leaves the splits the preloader has already prepared
// unread, so their readers are destroyed with a payload fetch outstanding.
TEST_F(TableScanTest, abandonPreloadedSplits) {
  auto filePaths = makeFilePaths(10);
  auto vectors = makeVectors(10, 1'000);
  for (auto i = 0; i < vectors.size(); ++i) {
    writeToFile(filePaths[i]->getPath(), vectors[i]);
  }
  createDuckDbTable(vectors);

  // Consume all but one IO threads so the preloader can post column chunk fetch
  // tasks queuing behind the blockers and stay pending. Driver reads complete
  // inline via work-stealing.
  auto* ioExecutor = ioExecutor_.get();
  const auto numBlocked = ioExecutor->numThreads() - 1;
  ASSERT_GE(numBlocked, 1);
  folly::Latch latch(numBlocked);
  std::vector<folly::Baton<>> batons(numBlocked);
  for (auto& baton : batons) {
    ioExecutor->add([&]() {
      baton.wait();
      latch.count_down();
    });
  }

  // The limit is reached partway into the second split, so the splits the
  // preloader prepared behind it are never read. Which rows are returned is
  // unspecified, so only the row count can be asserted.
  constexpr int32_t kLimit = 1'500;
  core::PlanNodeId scanNodeId;
  auto plan = PlanBuilder(pool_.get())
                  .startTableScan()
                  .outputType(rowType_)
                  .tableHandle(makeTableHandle())
                  .endTableScan()
                  .capturePlanNodeId(scanNodeId)
                  .limit(0, kLimit, false)
                  .planNode();

  std::shared_ptr<Task> task;
  auto result = AssertQueryBuilder(plan)
                    .config(core::QueryConfig::kMaxSplitPreloadPerDriver, "8")
                    // Enable column chunk fetch during preload
                    .connectorSessionProperty(
                        kCudfHiveConnectorId,
                        cudf_velox::connector::hive::CudfHiveConfig::
                            kPreloadColumnChunksSession,
                        "true")
                    .splits(makeCudfHiveConnectorSplits(filePaths))
                    .copyResults(pool_.get(), task);
  EXPECT_EQ(result->size(), kLimit);

  // The first split is read before the preloader runs, so only the splits
  // after it are preloaded. The stat counts the preloaded splits that were
  // read, so it confirms preloading was on but cannot measure how many were
  // abandoned.
  auto planStats = toPlanStats(task->taskStats());
  const auto& customStats = planStats.at(scanNodeId).customStats;
  ASSERT_EQ(customStats.count(std::string(TableScan::kPreloadedSplits)), 1);
  EXPECT_GE(customStats.at(std::string(TableScan::kPreloadedSplits)).sum, 1);

  // Tear down while abandoned splits' payload fetches are still queued.
  task.reset();
  ASSERT_EQ(Task::numRunningTasks(), 0);

  // Unblock the IO thread pool.
  for (auto& baton : batons) {
    baton.post();
  }
  latch.wait();
}

// A filter that no row group can satisfy prunes every row group of the split,
// leaving no row group passes to read.
TEST_F(TableScanTest, filterPrunesAllRowGroups) {
  auto rowType = ROW({"c0"}, {BIGINT()});
  // One row group per vector, all holding values well below the filter bound.
  std::vector<RowVectorPtr> vectors = {
      makeRowVector({"c0"}, {makeFlatVector<int64_t>({1, 2, 3})}),
      makeRowVector({"c0"}, {makeFlatVector<int64_t>({4, 5, 6})}),
      makeRowVector({"c0"}, {makeFlatVector<int64_t>({7, 8, 9})}),
  };
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);
  createDuckDbTable(vectors);

  constexpr int64_t kUnmatchedValue = 1'000;
  common::SubfieldFilters subfieldFilters =
      common::test::SubfieldFiltersBuilder()
          .add(
              "c0",
              std::make_unique<common::BigintRange>(
                  kUnmatchedValue, kUnmatchedValue, false))
          .build();

  auto tableHandle = makeTableHandle(
      "parquet_table", rowType, std::move(subfieldFilters), nullptr);

  auto plan = PlanBuilder()
                  .startTableScan()
                  .outputType(rowType)
                  .tableHandle(tableHandle)
                  .assignments(
                      facebook::velox::exec::test::HiveConnectorTestBase::
                          allRegularColumns(rowType))
                  .endTableScan()
                  .planNode();

  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .plan(plan)
          .splits(makeCudfHiveConnectorSplits({filePath}))
          .assertResults(
              fmt::format("SELECT c0 FROM tmp WHERE c0 = {}", kUnmatchedValue));

  auto planStats = toPlanStats(task->taskStats());
  EXPECT_EQ(planStats.at(plan->id()).outputRows, 0);
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TableScanTestParameterized,
    testing::Bool(),
    [](const testing::TestParamInfo<bool>& info) {
      return info.param ? "BufferedInput" : "FileDataSource";
    });

TEST_F(TableScanTest, directBufferInputRawInputBytes) {
  constexpr int kSize = 10;
  auto vector = makeRowVector({
      makeFlatVector<int64_t>(kSize, folly::identity),
      makeFlatVector<int64_t>(kSize, folly::identity),
      makeFlatVector<int64_t>(kSize, folly::identity),
  });
  auto filePath = TempFilePath::create();
  createDuckDbTable({vector});
  writeToFile(filePath->getPath(), {vector});

  auto tableHandle = makeTableHandle();
  auto plan = PlanBuilder(pool_.get())
                  .startTableScan()
                  .tableHandle(tableHandle)
                  .outputType(ROW({"c0", "c2"}, {BIGINT(), BIGINT()}))
                  .endTableScan()
                  .planNode();

  std::unordered_map<std::string, std::string> config;
  std::unordered_map<std::string, std::shared_ptr<config::ConfigBase>>
      connectorConfigs = {};
  auto queryCtx = core::QueryCtx::create(
      executor_.get(),
      core::QueryConfig(std::move(config)),
      connectorConfigs,
      nullptr);

  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .plan(plan)
                  .splits(makeCudfHiveConnectorSplits({filePath}))
                  .queryCtx(queryCtx)
                  .assertResults("SELECT c0, c2 FROM tmp");

  // A quick sanity check for memory usage reporting. Check that peak total
  // memory usage for the project node is > 0.
  auto planStats = toPlanStats(task->taskStats());
  auto scanNodeId = plan->id();
  auto it = planStats.find(scanNodeId);
  ASSERT_TRUE(it != planStats.end());
  auto rawInputBytes = it->second.rawInputBytes;
  // Reduced from 500 to 400 as cudf CudfHive writer seems to be writing smaller
  // files.
  ASSERT_GE(rawInputBytes, 400);

  const auto runtimeStats = getTableScanRuntimeStats(task);
  assertStorageReadStats(runtimeStats, filePath->fileSize());
  ASSERT_GT(runtimeStats.at("totalScanTime").sum, 0);
  ASSERT_GT(runtimeStats.at("ioWaitWallNanos").sum, 0);
}

TEST_F(TableScanTest, scanOnNonDefaultGpu) {
  int deviceCount = 0;
  ASSERT_EQ(cudaSuccess, cudaGetDeviceCount(&deviceCount));
  if (deviceCount < 2) {
    GTEST_SKIP() << "Requires two GPUs";
  }

  int originalDevice = -1;
  ASSERT_EQ(cudaSuccess, cudaGetDevice(&originalDevice));
  unregisterCudf();
  SCOPE_EXIT {
    unregisterCudf();
    cudaSetDevice(originalDevice);
    registerCudf();
  };
  ASSERT_EQ(cudaSuccess, cudaSetDevice(1));
  registerCudf();

  auto vectors = makeVectors(1, 1'000);
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);
  createDuckDbTable(vectors);
  assertQuery(tableScanNode(), {filePath}, "SELECT * FROM tmp");
}

TEST_F(TableScanTest, columnAliases) {
  auto vectors = makeVectors(1, 1'000);
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);
  createDuckDbTable(vectors);

  std::string tableName = "t";
  std::unordered_map<std::string, std::string> aliases = {{"a", "c0"}};
  auto outputType = ROW({"a"}, {INTEGER()});
  auto tableHandle = makeTableHandle();
  auto op = PlanBuilder(pool_.get())
                .startTableScan()
                .tableHandle(tableHandle)
                .tableName(tableName)
                .outputType(outputType)
                .columnAliases(aliases)
                .endTableScan()
                .planNode();
  assertQuery(op, {filePath}, "SELECT c0 FROM tmp");
}

TEST_F(TableScanTest, dynamicFilterUsesPhysicalNamesWithoutDataColumns) {
  auto probe = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int64_t>({1}), makeFlatVector<int64_t>({9})});
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), probe);

  auto build = makeRowVector(
      {"u_key"}, {makeFlatVector<int64_t>({1})});
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto buildSide = PlanBuilder(planNodeIdGenerator, pool_.get())
                       .values({build})
                       .planNode();
  core::PlanNodeId scanId;
  core::PlanNodeId joinId;
  auto outputType = ROW({"c1", "c0"}, {BIGINT(), BIGINT()});
  auto plan = PlanBuilder(planNodeIdGenerator, pool_.get())
                  .startTableScan()
                  .connectorId(kCudfHiveConnectorId)
                  .tableHandle(makeTableHandle())
                  .outputType(outputType)
                  .columnAliases({{"c1", "c0"}, {"c0", "c1"}})
                  .endTableScan()
                  .capturePlanNodeId(scanId)
                  .hashJoin(
                      {"c1"},
                      {"u_key"},
                      buildSide,
                      "",
                      {"c1", "c0"},
                      core::JoinType::kInner)
                  .capturePlanNodeId(joinId)
                  .planNode();
  auto expected = makeRowVector(
      {"c1", "c0"},
      {makeFlatVector<int64_t>({1}), makeFlatVector<int64_t>({9})});
  auto task = AssertQueryBuilder(plan)
                  .config(CudfConfig::kCudfEnabled, "true")
                  .maxDrivers(1)
                  .splits(scanId, makeCudfHiveConnectorSplits({filePath}))
                  .assertResults(expected);
  const auto stats = toPlanStats(task->taskStats());
  EXPECT_EQ(stats.at(joinId).customStats.at("dynamicFiltersProduced").sum, 1);
  EXPECT_EQ(stats.at(scanId).customStats.at("dynamicFiltersAccepted").sum, 1);
}

TEST_F(TableScanTest, filterPushdown) {
  auto rowType =
      ROW({"c0", "c1", "c2", "c3"}, {TINYINT(), BIGINT(), DOUBLE(), BOOLEAN()});
  auto filePaths = makeFilePaths(10);
  auto vectors = makeVectors(10, 1'000, rowType);
  for (int32_t i = 0; i < vectors.size(); i++) {
    writeToFile(filePaths[i]->getPath(), vectors[i]);
  }
  createDuckDbTable(vectors);

  // c1 >= 0 or null and c3 is true
  common::SubfieldFilters subfieldFilters =
      common::test::SubfieldFiltersBuilder()
          .add(
              "c1",
              std::make_unique<common::BigintRange>(
                  int64_t(0), std::numeric_limits<int64_t>::max(), true))
          .add("c3", std::make_unique<common::BoolValue>(true, false))
          .build();

  auto tableHandle = makeTableHandle(
      "parquet_table", rowType, std::move(subfieldFilters), nullptr);

  auto assignments =
      facebook::velox::exec::test::HiveConnectorTestBase::allRegularColumns(
          rowType);

  auto task = assertQuery(
      PlanBuilder()
          .startTableScan()
          .outputType(ROW({"c1", "c3", "c0"}, {BIGINT(), BOOLEAN(), TINYINT()}))
          .tableHandle(tableHandle)
          .assignments(assignments)
          .endTableScan()
          .planNode(),
      filePaths,
      "SELECT c1, c3, c0 FROM tmp WHERE (c1 >= 0 OR c1 IS NULL) AND c3");

  auto tableScanStats = getTableScanStats(task);
  // EXPECT_EQ(tableScanStats.rawInputRows, 10'000);
  // EXPECT_LT(tableScanStats.inputRows, tableScanStats.rawInputRows);
  EXPECT_EQ(tableScanStats.inputRows, tableScanStats.outputRows);

#if 0
  // Repeat the same but do not project out the filtered columns.
  assignments.clear();
  assignments["c0"] =
      facebook::velox::exec::test::HiveConnectorTestBase::regularColumn(
          "c0", TINYINT());
  assertQuery(
      PlanBuilder()
          .startTableScan()
          .outputType(ROW({"c0"}, {TINYINT()}))
          .tableHandle(tableHandle)
          .assignments(assignments)
          .endTableScan()
          .planNode(),
      filePaths,
      "SELECT c0 FROM tmp WHERE (c1 >= 0 ) AND c3");

  // TODO: zero column non-empty table is not possible in cudf, need to implement.
  // Do the same for count, no columns projected out.
  assignments.clear();
  assertQuery(
      PlanBuilder()
          .startTableScan()
          .outputType(ROW({}, {}))
          .tableHandle(tableHandle)
          .assignments(assignments)
          .endTableScan()
          .singleAggregation({}, {"sum(1)"})
          .planNode(),
      filePaths,
      "SELECT count(*) FROM tmp WHERE (c1 >= 0 ) AND c3");

  // Do the same for count, no filter, no projections.
  assignments.clear();
  // subfieldFilters.clear(); // Explicitly clear this.
  tableHandle = makeTableHandle(
      "parquet_table",
      rowType,
      false,
      nullptr,
      nullptr);
  assertQuery(
      PlanBuilder()
          .startTableScan()
          .outputType(ROW({}, {}))
          .tableHandle(tableHandle)
          .assignments(assignments)
          .endTableScan()
          .singleAggregation({}, {"sum(1)"})
          .planNode(),
      filePaths,
      "SELECT count(*) FROM tmp");
#endif
}

// Disable this test and the one below for now, pending a CUDF fix.
// simoneves 2/25/26
// @TODO simoneves/mattgara re-enable once fixed.

TEST_F(TableScanTest, DISABLED_decimalFilterPushdown) {
  auto rowType = ROW({"c0", "c1"}, {DECIMAL(12, 2), DECIMAL(20, 2)});

  auto vector = makeRowVector(
      {"c0", "c1"},
      {
          makeFlatVector<int64_t>(
              {123, 500, -250, 300, 400, 200}, DECIMAL(12, 2)),
          makeFlatVector<int128_t>(
              {int128_t{200},
               int128_t{200},
               int128_t{700},
               int128_t{700},
               int128_t{900},
               int128_t{-100}},
              DECIMAL(20, 2)),
      });

  std::vector<RowVectorPtr> vectors = {vector};
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);
  createDuckDbTable(vectors);

  // c0 between 1.00 and 4.00 and c1 in (2.00, 7.00)
  common::SubfieldFilters subfieldFilters =
      common::test::SubfieldFiltersBuilder()
          .add(
              "c0",
              std::make_unique<common::BigintRange>(
                  int64_t{100}, int64_t{400}, /*nullAllowed*/ false))
          .add(
              "c1",
              common::createHugeintValues(
                  {int128_t{200}, int128_t{700}}, /*nullAllowed*/ false))
          .build();

  auto tableHandle = makeTableHandle(
      "parquet_table", rowType, std::move(subfieldFilters), nullptr);

  auto assignments =
      facebook::velox::exec::test::HiveConnectorTestBase::allRegularColumns(
          rowType);

  auto plan = PlanBuilder()
                  .startTableScan()
                  .outputType(rowType)
                  .tableHandle(tableHandle)
                  .assignments(assignments)
                  .endTableScan()
                  .planNode();

  assertQuery(
      plan,
      {filePath},
      "SELECT c0, c1 FROM tmp "
      "WHERE c0 BETWEEN CAST('1.00' AS DECIMAL(12, 2)) "
      "AND CAST('4.00' AS DECIMAL(12, 2)) "
      "AND c1 IN (CAST('2.00' AS DECIMAL(20, 2)), "
      "CAST('7.00' AS DECIMAL(20, 2)))");
}

TEST_F(TableScanTest, DISABLED_decimalStatsFilterIoPruning) {
  auto rowType = ROW({"c0", "c1"}, {DECIMAL(12, 2), DECIMAL(20, 2)});
  auto vec0 = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int64_t>({100, 200}, DECIMAL(12, 2)),
       makeFlatVector<int128_t>(
           {int128_t{1000}, int128_t{2000}}, DECIMAL(20, 2))});
  auto vec1 = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int64_t>({300, 400}, DECIMAL(12, 2)),
       makeFlatVector<int128_t>(
           {int128_t{3000}, int128_t{4000}}, DECIMAL(20, 2))});
  auto vec2 = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int64_t>({500, 600}, DECIMAL(12, 2)),
       makeFlatVector<int128_t>(
           {int128_t{5000}, int128_t{6000}}, DECIMAL(20, 2))});

  std::vector<RowVectorPtr> vectors = {vec0, vec1, vec2};
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);

  common::SubfieldFilters filters =
      common::test::SubfieldFiltersBuilder()
          .add(
              "c0",
              std::make_unique<common::BigintRange>(
                  int64_t{300}, int64_t{400}, /*nullAllowed*/ false))
          .add(
              "c1",
              std::make_unique<common::HugeintRange>(
                  int128_t{3000}, int128_t{4000}, /*nullAllowed*/ false))
          .build();

  auto metrics = readParquetWithStatsFilter(
      filePath->getPath(), rowType, filters, /*useJitFilter*/ true);
  EXPECT_EQ(metrics.inputRowGroups, 3);
  ASSERT_TRUE(metrics.rowGroupsAfterStats.has_value());
  EXPECT_EQ(metrics.rowGroupsAfterStats.value(), 1);
  EXPECT_EQ(metrics.outputRows, 2);
}

TEST_F(TableScanTest, doubleStatsFilterIoPruning) {
  auto rowType = ROW({"c0", "c1"}, {DOUBLE(), DOUBLE()});
  auto vec0 = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<double>({1.0, 2.0}),
       makeFlatVector<double>({10.0, 20.0})});
  auto vec1 = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<double>({3.0, 4.0}),
       makeFlatVector<double>({30.0, 40.0})});
  auto vec2 = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<double>({5.0, 6.0}),
       makeFlatVector<double>({50.0, 60.0})});

  std::vector<RowVectorPtr> vectors = {vec0, vec1, vec2};
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);

  common::SubfieldFilters filters =
      common::test::SubfieldFiltersBuilder()
          .add(
              "c0",
              std::make_unique<common::DoubleRange>(
                  3.0,
                  /*lowerUnbounded*/ false,
                  /*lowerExclusive*/ false,
                  4.0,
                  /*upperUnbounded*/ false,
                  /*upperExclusive*/ false,
                  /*nullAllowed*/ false))
          .add(
              "c1",
              std::make_unique<common::DoubleRange>(
                  30.0,
                  /*lowerUnbounded*/ false,
                  /*lowerExclusive*/ false,
                  40.0,
                  /*upperUnbounded*/ false,
                  /*upperExclusive*/ false,
                  /*nullAllowed*/ false))
          .build();

  auto metrics = readParquetWithStatsFilter(
      filePath->getPath(), rowType, filters, /*useJitFilter*/ true);
  EXPECT_EQ(metrics.inputRowGroups, 3);
  ASSERT_TRUE(metrics.rowGroupsAfterStats.has_value());
  EXPECT_EQ(metrics.rowGroupsAfterStats.value(), 1);
  EXPECT_EQ(metrics.outputRows, 2);
}

TEST_F(TableScanTest, splitOffsetAndLength) {
  auto vectors = makeVectors(10, 1'000);
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);
  createDuckDbTable(vectors);

  // Note that the number of row groups selected within `halfFileSize` may
  // change in the future and this test may start failing. In such a case,
  // just adjust the duckdb sql string accordingly.
  const auto halfFileSize = fs::file_size(filePath->getPath()) / 2;

  // First half of file - OFFSET 0 LIMIT 6000
  assertQuery(
      tableScanNode(),
      makeCudfHiveConnectorSplit(filePath->getPath(), 0, halfFileSize),
      "SELECT * FROM tmp OFFSET 0 LIMIT 6000");

  // Second half of file - OFFSET 6000 LIMIT 4000
  assertQuery(
      tableScanNode(),
      makeCudfHiveConnectorSplit(filePath->getPath(), halfFileSize),
      "SELECT * FROM tmp OFFSET 6000 LIMIT 4000");

  const auto fileSize = fs::file_size(filePath->getPath());

  // All row groups
  assertQuery(
      tableScanNode(),
      makeCudfHiveConnectorSplit(filePath->getPath(), 0, fileSize),
      "SELECT * FROM tmp");

  // No row groups
  assertQuery(
      tableScanNode(),
      makeCudfHiveConnectorSplit(filePath->getPath(), fileSize),
      "SELECT * FROM tmp LIMIT 0");
}

// Verify that extractFiltersFromRemainingFilter extracts simple single-column
// filters from the remaining filter into subfield filters for pushdown.
// When a filter like "c0 = 1" is fully extracted,
// cudfRemainingFilterExpression_ is null and totalRemainingFilterWallNanos is
// 0. Without extraction, the filter runs post-read on the GPU and the stat is
// > 0.
TEST_F(TableScanTest, remainingFilterExtraction) {
  auto rowType = ROW({"c0", "c1", "c2"}, {BIGINT(), BIGINT(), DOUBLE()});
  auto vectors = makeVectors(5, 1'000, rowType);
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);
  createDuckDbTable(vectors);

  auto assignments =
      facebook::velox::exec::test::HiveConnectorTestBase::allRegularColumns(
          rowType);

  // "c0 = 1" is a single-column equality that should be fully extracted into
  // a subfield filter, leaving no remaining filter to evaluate post-read.
  auto plan = PlanBuilder(pool_.get())
                  .startTableScan()
                  .connectorId(kCudfHiveConnectorId)
                  .outputType(rowType)
                  .dataColumns(rowType)
                  .assignments(assignments)
                  .remainingFilter("c0 = 1")
                  .endTableScan()
                  .planNode();

  auto task = assertQuery(plan, {filePath}, "SELECT * FROM tmp WHERE c0 = 1");

  // Verify the filter was fully extracted: no post-read remaining filter ran.
  auto planStats = toPlanStats(task->taskStats());
  const auto& scanStats = planStats.at(plan->id());
  auto it = scanStats.customStats.find("totalRemainingFilterWallNanos");
  ASSERT_NE(it, scanStats.customStats.end());
  EXPECT_EQ(it->second.sum, 0)
      << "Expected no remaining filter time when filter is fully extracted";
}

TEST_F(TableScanTest, dynamicFilterPrunesReaderRowGroups) {
  auto rowType = ROW({"c0", "c1"}, BIGINT());
  std::vector<RowVectorPtr> vectors;
  for (int group = 0; group < 3; ++group) {
    vectors.push_back(makeRowVector(
        {"c0", "c1"},
        {makeFlatVector<int64_t>(
             1'000, [group](auto row) { return group * 1'000 + row; }),
         makeFlatVector<int64_t>(
             1'000, [group](auto row) { return group * 10'000 + row; })}));
  }
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);
  createDuckDbTable("t", vectors);
  common::SubfieldFilters staticFilters =
      common::test::SubfieldFiltersBuilder()
          .add(
              "c0",
              std::make_unique<common::BigintRange>(
                  int64_t{1500}, int64_t{1500}, false))
          .build();
  auto staticMetrics = readParquetWithStatsFilter(
      filePath->getPath(), rowType, staticFilters, true);
  EXPECT_EQ(staticMetrics.inputRowGroups, 3);
  ASSERT_TRUE(staticMetrics.rowGroupsAfterStats.has_value());
  EXPECT_EQ(*staticMetrics.rowGroupsAfterStats, 1);

  auto build = makeRowVector(
      {"c0"}, {makeFlatVector<int64_t>({1500})});
  createDuckDbTable("u", {build});
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto buildSide = PlanBuilder(planNodeIdGenerator, pool_.get())
                       .values({build})
                       .project({"c0 AS u_key"})
                       .planNode();
  core::PlanNodeId scanId;
  auto scanType = ROW({"c1", "c0"}, BIGINT());
  auto plan = PlanBuilder(planNodeIdGenerator, pool_.get())
                  .startTableScan()
                  .connectorId(kCudfHiveConnectorId)
                  .outputType(scanType)
                  .dataColumns(rowType)
                  .assignments(
                      HiveConnectorTestBase::allRegularColumns(rowType))
                  .endTableScan()
                  .capturePlanNodeId(scanId)
                  .hashJoin(
                      {"c0"},
                      {"u_key"},
                      buildSide,
                      "",
                      {"c0", "c1"},
                      core::JoinType::kInner)
                  .planNode();
  std::atomic<int32_t> selected{-1};
  const bool testValuesWereEnabled = common::testutil::TestValue::enabled();
  common::testutil::TestValue::enable();
  SCOPE_EXIT {
    if (!testValuesWereEnabled) {
      common::testutil::TestValue::disable();
    }
  };
  common::testutil::ScopedTestValue selectedCounter(
      "facebook::velox::cudf_velox::connector::hive::CudfSplitReader::selectedRowGroups",
      std::function<void(void*)>([&](void* value) {
        selected = *static_cast<size_t*>(value);
      }));
  AssertQueryBuilder(plan, duckDbQueryRunner_)
      .maxDrivers(1)
      .splits(scanId, makeCudfHiveConnectorSplits({filePath}))
      .assertResults("SELECT t.c0, t.c1 FROM t JOIN u ON t.c0 = u.c0");
#ifndef NDEBUG
  EXPECT_EQ(selected, 1);
#endif
}

TEST_F(TableScanTest, integerDynamicFilterFromHashJoin) {
  constexpr vector_size_t kNumBuildRows{
      ::facebook::velox::exec::VectorHasher::kMaxDistinct + 1};
  auto probeType = ROW({{"c0", INTEGER()}, {"c1", BIGINT()}});
  auto probe = makeRowVector(
      {makeFlatVector<int32_t>(1'000, folly::identity),
       makeFlatVector<int64_t>(
           1'000, [](auto row) { return static_cast<int64_t>(row) * 10; })});
  auto sparseBuildKey = makeFlatVector<int64_t>(kNumBuildRows, [](auto row) {
    return static_cast<int64_t>(100 + row) * 10;
  });
  for (vector_size_t row = 100; row < kNumBuildRows; ++row) {
    sparseBuildKey->setNull(row, true);
  }
  auto build = makeRowVector(
      {makeFlatVector<int32_t>(
           kNumBuildRows, [](auto row) { return 100 + row; }),
       sparseBuildKey});
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), probe);
  createDuckDbTable("t", {probe});
  createDuckDbTable("u", {build});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto buildSide = PlanBuilder(planNodeIdGenerator, pool_.get())
                       .values({build})
                       .project({"c0 AS u_key0", "c1 AS u_key1"})
                       .planNode();
  core::PlanNodeId scanId;
  core::PlanNodeId joinId;
  auto plan = PlanBuilder(planNodeIdGenerator, pool_.get())
                  .startTableScan()
                  .connectorId(kCudfHiveConnectorId)
                  .outputType(probeType)
                  .dataColumns(probeType)
                  .assignments(
                      facebook::velox::exec::test::HiveConnectorTestBase::
                          allRegularColumns(probeType))
                  .endTableScan()
                  .capturePlanNodeId(scanId)
                  .project({"c1 AS key1", "c0 AS key0"})
                  .hashJoin(
                      {"key0", "key1"},
                      {"u_key0", "u_key1"},
                      buildSide,
                      "",
                      {"key0", "key1"},
                      core::JoinType::kInner)
                  .capturePlanNodeId(joinId)
                  .planNode();

  std::atomic<int32_t> numFiltersBuilt{0};
  const bool testValuesWereEnabled = common::testutil::TestValue::enabled();
  common::testutil::TestValue::enable();
  SCOPE_EXIT {
    if (!testValuesWereEnabled) {
      common::testutil::TestValue::disable();
    }
  };
  common::testutil::ScopedTestValue filterBuildCounter(
      "facebook::velox::cudf_velox::CudfHashJoinProbe::makeIntegerDynamicFilter",
      std::function<void(void*)>([&](void*) { ++numFiltersBuilt; }));
  auto task = AssertQueryBuilder(plan, duckDbQueryRunner_)
                  .maxDrivers(4)
                  .splits(scanId, makeCudfHiveConnectorSplits({filePath}))
                  .assertResults(
                      "SELECT t.c0, t.c1 FROM t JOIN u "
                      "ON t.c0 = u.c0 AND t.c1 = u.c1");
  const auto stats = toPlanStats(task->taskStats());
  EXPECT_GE(stats.at(joinId).customStats.at("dynamicFiltersProduced").sum, 2);
  EXPECT_GE(stats.at(scanId).customStats.at("dynamicFiltersAccepted").sum, 2);
  EXPECT_EQ(stats.at(scanId).outputRows, 100);
  EXPECT_EQ(
      stats.at(scanId).dynamicFilterStats.producerNodeIds,
      std::unordered_set<core::PlanNodeId>{joinId});
#ifndef NDEBUG
  EXPECT_EQ(numFiltersBuilt, 2);
#endif
}

TEST_F(TableScanTest, rejectsDynamicFiltersItCannotApply) {
  auto physicalProbe = makeRowVector(
      {"c0"}, {makeFlatVector<int64_t>({100, 200}, DECIMAL(18, 2))});
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), physicalProbe);

  auto checkJoin = [&](const TypePtr& scanType,
                       bool useGpuJoin,
                       const char* expectedError) {
    auto rowType = ROW("c0", scanType);
    auto build = makeRowVector(
        {"c0"}, {makeFlatVector<int64_t>({100}, scanType)});
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    auto buildSide = PlanBuilder(planNodeIdGenerator, pool_.get())
                         .values({build})
                         .project({"c0 AS u_key"})
                         .planNode();
    core::PlanNodeId scanId;
    core::PlanNodeId joinId;
    auto plan = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .startTableScan()
                    .connectorId(kCudfHiveConnectorId)
                    .outputType(rowType)
                    .dataColumns(rowType)
                    .assignments(
                        HiveConnectorTestBase::allRegularColumns(rowType))
                    .endTableScan()
                    .capturePlanNodeId(scanId)
                    .hashJoin(
                        {"c0"},
                        {"u_key"},
                        buildSide,
                        "",
                        {"c0"},
                        useGpuJoin ? core::JoinType::kInner
                                   : core::JoinType::kCountingLeftSemiFilter,
                        /*nullAware=*/false,
                        /*nullAsValue=*/false)
                    .capturePlanNodeId(joinId)
                    .planNode();

    AssertQueryBuilder query(plan);
    query.config(CudfConfig::kCudfEnabled, "true")
        .splits(scanId, makeCudfHiveConnectorSplits({filePath}));
    if (expectedError) {
      VELOX_ASSERT_THROW(query.copyResults(pool_.get()), expectedError);
    } else {
      auto expected = makeRowVector(
          {"c0"}, {makeFlatVector<int64_t>({100}, scanType)});
      auto task = query.assertResults(expected);
      const auto stats = toPlanStats(task->taskStats());
      EXPECT_TRUE(stats.at(joinId).operatorStats.count("HashProbe"));
      EXPECT_TRUE(stats.at(scanId).dynamicFilterStats.producerNodeIds.empty());
    }
  };

  {
    auto& config = CudfConfig::getInstance();
    const auto previousFallback = config.allowCpuFallback;
    unregisterCudf();
    config.allowCpuFallback = true;
    registerCudf();
    SCOPE_EXIT {
      unregisterCudf();
      config.allowCpuFallback = previousFallback;
      registerCudf();
    };
    checkJoin(DECIMAL(18, 2), false, nullptr);
  }
  checkJoin(BIGINT(), true, "column type does not match dynamic filter type");
}

TEST_F(TableScanTest, rightAntiDynamicFilterKeepsBuildMatches) {
  auto& config = CudfConfig::getInstance();
  const auto previousFallback = config.allowCpuFallback;
  unregisterCudf();
  config.allowCpuFallback = true;
  registerCudf();
  SCOPE_EXIT {
    unregisterCudf();
    config.allowCpuFallback = previousFallback;
    registerCudf();
  };

  auto probe = makeRowVector({"c0"}, {makeFlatVector<int64_t>({1})});
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), probe);
  auto rowType = ROW("c0", BIGINT());
  for (const auto& [buildValues, expectedCount] :
       std::vector<std::pair<std::vector<int64_t>, int64_t>>{
           {{1}, 0}, {{1, 2}, 1}}) {
    auto build = makeRowVector(
        {"u_key"}, {makeFlatVector<int64_t>(buildValues)});
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    auto buildSide = PlanBuilder(planNodeIdGenerator, pool_.get())
                         .values({build})
                         .planNode();
    core::PlanNodeId scanId;
    core::PlanNodeId joinId;
    auto plan = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .startTableScan()
                    .connectorId(kCudfHiveConnectorId)
                    .outputType(rowType)
                    .dataColumns(rowType)
                    .assignments(
                        HiveConnectorTestBase::allRegularColumns(rowType))
                    .endTableScan()
                    .capturePlanNodeId(scanId)
                    .hashJoin(
                        {"c0"},
                        {"u_key"},
                        buildSide,
                        "",
                        {},
                        core::JoinType::kRightAnti,
                        /*nullAware=*/false,
                        /*nullAsValue=*/false)
                    .capturePlanNodeId(joinId)
                    .singleAggregation({}, {"count(1)"})
                    .planNode();
    auto expected = makeRowVector(
        {"a0"}, {makeFlatVector<int64_t>({expectedCount})});
    auto task = AssertQueryBuilder(plan)
                    .config(CudfConfig::kCudfEnabled, "true")
                    .maxDrivers(1)
                    .splits(scanId, makeCudfHiveConnectorSplits({filePath}))
                    .assertResults(expected);
    const auto stats = toPlanStats(task->taskStats());
    EXPECT_TRUE(stats.at(joinId).operatorStats.count("HashProbe"));
    EXPECT_EQ(stats.at(scanId).customStats.at("dynamicFiltersAccepted").sum, 1);
  }
}

TEST_F(TableScanTest, countingAntiPayloadFilterDoesNotChangeSelectedRow) {
  auto& config = CudfConfig::getInstance();
  const auto previousFallback = config.allowCpuFallback;
  unregisterCudf();
  config.allowCpuFallback = true;
  registerCudf();
  SCOPE_EXIT {
    unregisterCudf();
    config.allowCpuFallback = previousFallback;
    registerCudf();
  };

  auto probe = makeRowVector(
      {"k", "p"},
      {makeFlatVector<int64_t>({1, 1}),
       makeFlatVector<int64_t>({10, 20})});
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), probe);
  auto rowType = ROW({"k", "p"}, {BIGINT(), BIGINT()});
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto antiBuild = PlanBuilder(planNodeIdGenerator, pool_.get())
                       .values({makeRowVector(
                           {"u_k"}, {makeFlatVector<int64_t>({1})})})
                       .planNode();
  auto semiBuild = PlanBuilder(planNodeIdGenerator, pool_.get())
                       .values({makeRowVector(
                           {"u_p"}, {makeFlatVector<int64_t>({20})})})
                       .planNode();
  core::PlanNodeId scanId;
  core::PlanNodeId antiId;
  core::PlanNodeId semiId;
  auto plan = PlanBuilder(planNodeIdGenerator, pool_.get())
                  .startTableScan()
                  .connectorId(kCudfHiveConnectorId)
                  .outputType(rowType)
                  .dataColumns(rowType)
                  .assignments(
                      HiveConnectorTestBase::allRegularColumns(rowType))
                  .endTableScan()
                  .capturePlanNodeId(scanId)
                  .hashJoin(
                      {"k"},
                      {"u_k"},
                      antiBuild,
                      "",
                      {"k", "p"},
                      core::JoinType::kCountingAnti,
                      /*nullAware=*/false,
                      /*nullAsValue=*/false)
                  .capturePlanNodeId(antiId)
                  .hashJoin(
                      {"p"},
                      {"u_p"},
                      semiBuild,
                      "",
                      {"k", "p"},
                      core::JoinType::kCountingLeftSemiFilter,
                      /*nullAware=*/false,
                      /*nullAsValue=*/false)
                  .capturePlanNodeId(semiId)
                  .planNode();
  auto expected = makeRowVector(
      {"k", "p"},
      {makeFlatVector<int64_t>({1}), makeFlatVector<int64_t>({20})});
  for (bool pushdown : {false, true}) {
    auto task = AssertQueryBuilder(plan)
                    .config(CudfConfig::kCudfEnabled, "true")
                    .config(
                        core::QueryConfig::kHashProbeDynamicFilterPushdownEnabled,
                        pushdown ? "true" : "false")
                    .maxDrivers(1)
                    .splits(scanId, makeCudfHiveConnectorSplits({filePath}))
                    .assertResults(expected);
    const auto stats = toPlanStats(task->taskStats());
    EXPECT_TRUE(stats.at(antiId).operatorStats.count("HashProbe"));
    EXPECT_TRUE(stats.at(semiId).operatorStats.count("HashProbe"));
  }
}

TEST_F(TableScanTest, countingSemiDynamicFilterKeepsCountsAndNulls) {
  auto& config = CudfConfig::getInstance();
  const auto previousFallback = config.allowCpuFallback;
  unregisterCudf();
  config.allowCpuFallback = true;
  registerCudf();
  SCOPE_EXIT {
    unregisterCudf();
    config.allowCpuFallback = previousFallback;
    registerCudf();
  };

  auto runCase = [&](const RowVectorPtr& probe,
                     const RowVectorPtr& build,
                     const RowVectorPtr& expected,
                     bool nullAsValue,
                     bool expectDynamicFilter) {
    auto filePath = TempFilePath::create();
    writeToFile(filePath->getPath(), probe);
    auto rowType = ROW("c0", BIGINT());
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    auto buildSide = PlanBuilder(planNodeIdGenerator, pool_.get())
                         .values({build})
                         .planNode();
    core::PlanNodeId scanId;
    core::PlanNodeId joinId;
    auto plan = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .startTableScan()
                    .connectorId(kCudfHiveConnectorId)
                    .outputType(rowType)
                    .dataColumns(rowType)
                    .assignments(
                        HiveConnectorTestBase::allRegularColumns(rowType))
                    .endTableScan()
                    .capturePlanNodeId(scanId)
                    .hashJoin(
                        {"c0"},
                        {"u_key"},
                        buildSide,
                        "",
                        {"c0"},
                        core::JoinType::kCountingLeftSemiFilter,
                        /*nullAware=*/false,
                        /*nullAsValue=*/nullAsValue)
                    .capturePlanNodeId(joinId)
                    .planNode();
    auto task = AssertQueryBuilder(plan)
                    .config(CudfConfig::kCudfEnabled, "true")
                    .maxDrivers(1)
                    .splits(scanId, makeCudfHiveConnectorSplits({filePath}))
                    .assertResults(expected);
    const auto stats = toPlanStats(task->taskStats());
    EXPECT_TRUE(stats.at(joinId).operatorStats.count("HashProbe"));
    if (expectDynamicFilter) {
      EXPECT_EQ(
          stats.at(scanId).customStats.at("dynamicFiltersAccepted").sum, 1);
    } else {
      EXPECT_TRUE(stats.at(scanId).dynamicFilterStats.producerNodeIds.empty());
    }
  };

  runCase(
      makeRowVector(
          {"c0"}, {makeNullableFlatVector<int64_t>({1, 1})}),
      makeRowVector(
          {"u_key"}, {makeNullableFlatVector<int64_t>({1})}),
      makeRowVector(
          {"c0"}, {makeNullableFlatVector<int64_t>({1})}),
      false,
      true);
  runCase(
      makeRowVector(
          {"c0"},
          {makeNullableFlatVector<int64_t>({std::nullopt, 1})}),
      makeRowVector(
          {"u_key"},
          {makeNullableFlatVector<int64_t>({std::nullopt, 1})}),
      makeRowVector(
          {"c0"},
          {makeNullableFlatVector<int64_t>({std::nullopt, 1})}),
      true,
      false);
}

TEST_F(TableScanTest, disjointDynamicFiltersProduceNoRows) {
  auto rowType = ROW("c0", BIGINT());
  auto probe = makeRowVector(
      {"c0"}, {makeFlatVector<int64_t>({1, 2, 3})});
  auto firstBuild = makeRowVector(
      {"c0"}, {makeFlatVector<int64_t>({1})});
  auto secondBuild = makeRowVector(
      {"c0"}, {makeFlatVector<int64_t>({2})});
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), probe);
  createDuckDbTable("t", {probe});
  createDuckDbTable("u", {firstBuild});
  createDuckDbTable("v", {secondBuild});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto firstBuildSide = PlanBuilder(planNodeIdGenerator, pool_.get())
                            .values({firstBuild})
                            .project({"c0 AS u_key"})
                            .planNode();
  auto secondBuildSide = PlanBuilder(planNodeIdGenerator, pool_.get())
                             .values({secondBuild})
                             .project({"c0 AS v_key"})
                             .planNode();
  core::PlanNodeId scanId;
  auto plan = PlanBuilder(planNodeIdGenerator, pool_.get())
                  .startTableScan()
                  .connectorId(kCudfHiveConnectorId)
                  .outputType(rowType)
                  .dataColumns(rowType)
                  .assignments(
                      HiveConnectorTestBase::allRegularColumns(rowType))
                  .endTableScan()
                  .capturePlanNodeId(scanId)
                  .hashJoin(
                      {"c0"},
                      {"u_key"},
                      firstBuildSide,
                      "",
                      {"c0"},
                      core::JoinType::kInner)
                  .hashJoin(
                      {"c0"},
                      {"v_key"},
                      secondBuildSide,
                      "",
                      {"c0"},
                      core::JoinType::kInner)
                  .planNode();
  auto task = AssertQueryBuilder(plan, duckDbQueryRunner_)
                  .splits(scanId, makeCudfHiveConnectorSplits({filePath}))
                  .assertResults(
                      "SELECT t.c0 FROM t JOIN u ON t.c0 = u.c0 "
                      "JOIN v ON t.c0 = v.c0");
  const auto stats = toPlanStats(task->taskStats());
  EXPECT_GE(stats.at(scanId).customStats.at("dynamicFiltersAccepted").sum, 2);
  EXPECT_EQ(stats.at(scanId).outputRows, 0);
}

TEST_F(TableScanTest, dynamicFilterAcrossCpuScanAndGpuJoin) {
  const std::string cpuConnectorId{"cpu-hive-dynamic-filter"};
  facebook::velox::connector::hive::HiveConnectorFactory factory;
  auto cpuConnector = factory.newConnector(
      cpuConnectorId,
      std::make_shared<config::ConfigBase>(
          std::unordered_map<std::string, std::string>{}),
      ioExecutor_.get());
  ConnectorRegistry::global().insert(cpuConnectorId, cpuConnector);
  parquet::registerParquetReaderFactory();
  SCOPE_EXIT {
    parquet::unregisterParquetReaderFactory();
    ConnectorRegistry::global().erase(cpuConnectorId);
  };

  auto rowType = ROW({"c0", "c1"}, {BIGINT(), BIGINT()});
  auto probe = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int64_t>({101, 102, 103, 104}),
       makeFlatVector<int64_t>({1, 2, 3, 4})});
  auto build = makeRowVector({"c0"}, {makeFlatVector<int64_t>({2, 4})});
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), probe);
  createDuckDbTable("t", {probe});
  createDuckDbTable("u", {build});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto buildSide = PlanBuilder(planNodeIdGenerator, pool_.get())
                       .values({build})
                       .project({"c0 AS u_key"})
                       .planNode();
  core::PlanNodeId scanId;
  core::PlanNodeId joinId;
  auto plan = PlanBuilder(planNodeIdGenerator, pool_.get())
                  .startTableScan()
                  .connectorId(cpuConnectorId)
                  .outputType(rowType)
                  .dataColumns(rowType)
                  .assignments(
                      HiveConnectorTestBase::allRegularColumns(rowType))
                  .endTableScan()
                  .capturePlanNodeId(scanId)
                  .hashJoin(
                      {"c1"}, {"u_key"}, buildSide, "", {"c0"}, core::JoinType::kInner)
                  .capturePlanNodeId(joinId)
                  .planNode();
  std::vector<std::shared_ptr<ConnectorSplit>> splits{
      facebook::velox::connector::hive::HiveConnectorSplitBuilder(
          filePath->getPath())
          .connectorId(cpuConnectorId)
          .fileFormat(dwio::common::FileFormat::PARQUET)
          .build()};

  auto task = AssertQueryBuilder(plan, duckDbQueryRunner_)
                  .maxDrivers(2)
                  .splits(scanId, splits)
                  .assertResults("SELECT t.c0 FROM t JOIN u ON t.c1 = u.c0");
  const auto stats = toPlanStats(task->taskStats());
  EXPECT_TRUE(stats.at(joinId).operatorStats.count("CudfFromVelox"));
  EXPECT_GE(stats.at(joinId).customStats.at("dynamicFiltersProduced").sum, 1);
  EXPECT_GE(stats.at(scanId).customStats.at("dynamicFiltersAccepted").sum, 1);
}

TEST_F(TableScanTest, splitBlockBloomFilterUsesVeloxFormat) {
  auto stream = cudf::get_default_stream();
  auto memoryResource = cudf::get_current_device_resource_ref();
  auto values = makeFlatVector<int64_t>(259, [](auto row) {
    return (static_cast<int64_t>(row) - 129) * 4'294'967'297LL;
  });
  for (vector_size_t row = 0; row < values->size(); row += 17) {
    values->setNull(row, true);
  }
  values->set(1, std::numeric_limits<int64_t>::min());
  values->setNull(1, false);
  values->set(257, std::numeric_limits<int64_t>::max());
  values->setNull(257, false);
  auto input = makeRowVector({values});
  auto table =
      with_arrow::toCudfTable(input, pool_.get(), stream, memoryResource);
  const auto slices = cudf::slice(table->view().column(0), {1, 258}, stream);
  ASSERT_EQ(slices.size(), 1);
  const auto slice = slices.front();

  common::BigintValuesUsingBloomFilter cpuFilter(slice.size(), false);
  for (vector_size_t row = 1; row < 258; row += 2) {
    if (!values->isNullAt(row)) {
      cpuFilter.insert(values->valueAt(row));
    }
  }
  const auto cpuBlocks = cpuFilter.blocks();
  rmm::device_buffer deviceBlocks(
      cpuFilter.blocksByteSize(), stream, memoryResource);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      deviceBlocks.data(),
      cpuBlocks.data(),
      cpuFilter.blocksByteSize(),
      cudaMemcpyHostToDevice,
      stream.get()));
  auto checkMask = [&](bool nullAllowed) {
    auto mask = makeSplitBlockBloomFilterMask(
        slice,
        static_cast<const uint32_t*>(deviceBlocks.data()),
        cpuBlocks.size(),
        sizeof(SplitBlockBloomFilter::Block) / sizeof(uint32_t),
        nullAllowed,
        stream,
        memoryResource);
    std::vector<uint8_t> hostMask(mask->size());
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        hostMask.data(),
        mask->view().data<bool>(),
        hostMask.size(),
        cudaMemcpyDeviceToHost,
        stream.get()));
    stream.sync();
    for (vector_size_t row = 0; row < slice.size(); ++row) {
      const auto inputRow = row + 1;
      const bool expected = values->isNullAt(inputRow)
          ? nullAllowed
          : cpuFilter.testInt64(values->valueAt(inputRow));
      EXPECT_EQ(hostMask[row] != 0, expected) << "row " << row;
    }
  };
  checkMask(false);
  checkMask(true);

  common::BigintValuesUsingBloomFilter expectedGpuFilter(slice.size(), false);
  for (vector_size_t row = 1; row < 258; ++row) {
    if (!values->isNullAt(row)) {
      expectedGpuFilter.insert(values->valueAt(row));
    }
  }
  const auto expectedGpuBlocks = expectedGpuFilter.blocks();
  ASSERT_EQ(expectedGpuFilter.blocksByteSize(), cpuFilter.blocksByteSize());
  CUDF_CUDA_TRY(cudaMemsetAsync(
      deviceBlocks.data(),
      0,
      expectedGpuFilter.blocksByteSize(),
      stream.get()));
  insertSplitBlockBloomFilter(
      slice,
      static_cast<uint32_t*>(deviceBlocks.data()),
      expectedGpuBlocks.size(),
      sizeof(SplitBlockBloomFilter::Block) / sizeof(uint32_t),
      stream);
  std::vector<SplitBlockBloomFilter::Block> gpuBlocks(expectedGpuBlocks.size());
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      gpuBlocks.data(),
      deviceBlocks.data(),
      expectedGpuFilter.blocksByteSize(),
      cudaMemcpyDeviceToHost,
      stream.get()));
  stream.sync();
  EXPECT_EQ(
      std::memcmp(
          gpuBlocks.data(),
          expectedGpuBlocks.data(),
          expectedGpuFilter.blocksByteSize()),
      0);
  auto gpuFilter = common::BigintValuesUsingBloomFilter::createFromBlocks(
      std::move(gpuBlocks), false);
  for (vector_size_t row = 1; row < 258; ++row) {
    if (!values->isNullAt(row)) {
      EXPECT_TRUE(gpuFilter->testInt64(values->valueAt(row))) << "row " << row;
    }
  }
}

TEST_F(TableScanTest, bloomDynamicFilterFromHashJoin) {
  constexpr vector_size_t kNumBuildRows{100'001};
  constexpr vector_size_t kNumProbeRows{kNumBuildRows * 2};
  auto rowType = ROW("c0", BIGINT());
  auto probe = makeRowVector({makeFlatVector<int64_t>(
      kNumProbeRows, [](auto row) { return static_cast<int64_t>(row); })});
  auto build = makeRowVector({makeFlatVector<int64_t>(
      kNumBuildRows, [](auto row) { return static_cast<int64_t>(row) * 2; })});
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), probe);
  createDuckDbTable("t", {probe});
  createDuckDbTable("u", {build});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto buildSide = PlanBuilder(planNodeIdGenerator, pool_.get())
                       .values({build})
                       .project({"c0 AS u_key"})
                       .planNode();
  core::PlanNodeId scanId;
  core::PlanNodeId joinId;
  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .startTableScan()
          .connectorId(kCudfHiveConnectorId)
          .outputType(rowType)
          .dataColumns(rowType)
          .assignments(
              facebook::velox::exec::test::HiveConnectorTestBase::
                  allRegularColumns(rowType))
          .endTableScan()
          .capturePlanNodeId(scanId)
          .hashJoin(
              {"c0"}, {"u_key"}, buildSide, "", {"c0"}, core::JoinType::kInner)
          .capturePlanNodeId(joinId)
          .project({"c0"})
          .planNode();

  auto task = AssertQueryBuilder(plan, duckDbQueryRunner_)
                  .config(
                      core::QueryConfig::kHashProbeBloomFilterPushdownMaxSize,
                      std::to_string(2 << 20))
                  .maxDrivers(1)
                  .splits(scanId, makeCudfHiveConnectorSplits({filePath}))
                  .assertResults("SELECT t.c0 FROM t JOIN u ON t.c0 = u.c0");
  const auto stats = toPlanStats(task->taskStats());
  EXPECT_EQ(stats.at(joinId).customStats.at("dynamicFiltersProduced").sum, 1);
  EXPECT_GT(stats.at(joinId).customStats.at("bloomFilterSize").sum, 0);
  EXPECT_EQ(stats.at(scanId).customStats.at("dynamicFiltersAccepted").sum, 1);
  EXPECT_GE(stats.at(scanId).outputRows, kNumBuildRows);
  EXPECT_LT(stats.at(scanId).outputRows, kNumProbeRows);
}

TEST_F(TableScanTest, negatedStaticFilterWithBloomDynamicFilter) {
  constexpr vector_size_t kNumBuildRows{100'001};
  constexpr vector_size_t kNumProbeRows{kNumBuildRows * 2};
  auto rowType = ROW("c0", BIGINT());
  auto probe = makeRowVector({makeFlatVector<int64_t>(
      kNumProbeRows, [](auto row) { return static_cast<int64_t>(row); })});
  auto build = makeRowVector({makeFlatVector<int64_t>(
      kNumBuildRows, [](auto row) { return static_cast<int64_t>(row) * 2; })});
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), probe);
  createDuckDbTable("t", {probe});
  createDuckDbTable("u", {build});

  auto runTest = [&](const std::vector<int64_t>& rejectedValues,
                     common::FilterKind expectedKind) {
    auto filter = common::createNegatedBigintValues(rejectedValues, false);
    ASSERT_EQ(filter->kind(), expectedKind);
    auto subfieldFilters = common::test::SubfieldFiltersBuilder()
                               .add("c0", std::move(filter))
                               .build();
    auto tableHandle = makeTableHandle(
        "parquet_table", rowType, std::move(subfieldFilters), nullptr);
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    auto buildSide = PlanBuilder(planNodeIdGenerator, pool_.get())
                         .values({build})
                         .project({"c0 AS u_key"})
                         .planNode();
    core::PlanNodeId scanId;
    core::PlanNodeId joinId;
    auto plan = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .startTableScan()
                    .connectorId(kCudfHiveConnectorId)
                    .outputType(rowType)
                    .tableHandle(tableHandle)
                    .assignments(
                        HiveConnectorTestBase::allRegularColumns(rowType))
                    .endTableScan()
                    .capturePlanNodeId(scanId)
                    .hashJoin(
                        {"c0"},
                        {"u_key"},
                        buildSide,
                        "",
                        {"c0"},
                        core::JoinType::kInner)
                    .capturePlanNodeId(joinId)
                    .planNode();

    auto task = AssertQueryBuilder(plan, duckDbQueryRunner_)
                    .config(
                        core::QueryConfig::kHashProbeBloomFilterPushdownMaxSize,
                        std::to_string(2 << 20))
                    .maxDrivers(1)
                    .splits(scanId, makeCudfHiveConnectorSplits({filePath}))
                    .assertResults(fmt::format(
                        "SELECT t.c0 FROM t JOIN u ON t.c0 = u.c0 "
                        "WHERE t.c0 NOT IN ({})",
                        fmt::join(rejectedValues, ", ")));
    const auto stats = toPlanStats(task->taskStats());
    EXPECT_EQ(
        stats.at(joinId).customStats.at("dynamicFiltersProduced").sum, 1);
    EXPECT_GT(stats.at(joinId).customStats.at("bloomFilterSize").sum, 0);
    EXPECT_EQ(
        stats.at(scanId).customStats.at("dynamicFiltersAccepted").sum, 1);
  };

  runTest(
      {0, 2}, common::FilterKind::kNegatedBigintValuesUsingBitmask);
  runTest(
      {0, 10'000}, common::FilterKind::kNegatedBigintValuesUsingHashTable);
}

TEST_F(TableScanTest, negatedFilterPrunesReaderRowGroups) {
  constexpr vector_size_t kNumBuildRows{100'001};
  auto rowType = ROW("c0", BIGINT());
  std::vector<RowVectorPtr> probe;
  for (int64_t value = 0; value < 3; ++value) {
    probe.push_back(makeRowVector(
        {"c0"},
        {makeFlatVector<int64_t>(
            1'000, [value](auto) { return value; })}));
  }
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), probe);
  createDuckDbTable("t", probe);

  auto filter = common::createNegatedBigintValues({0, 2}, false);
  ASSERT_EQ(
      filter->kind(),
      common::FilterKind::kNegatedBigintValuesUsingBitmask);
  auto subfieldFilters = common::test::SubfieldFiltersBuilder()
                             .add("c0", std::move(filter))
                             .build();
  auto metrics = readParquetWithStatsFilter(
      filePath->getPath(), rowType, subfieldFilters, true);
  EXPECT_EQ(metrics.inputRowGroups, 3);
  ASSERT_TRUE(metrics.rowGroupsAfterStats.has_value());
  EXPECT_EQ(*metrics.rowGroupsAfterStats, 1);
  auto tableHandle = makeTableHandle(
      "parquet_table", rowType, std::move(subfieldFilters), nullptr);

  auto build = makeRowVector({makeFlatVector<int64_t>(
      kNumBuildRows, [](auto row) { return static_cast<int64_t>(row) * 2 + 1; })});
  createDuckDbTable("u", {build});
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto buildSide = PlanBuilder(planNodeIdGenerator, pool_.get())
                       .values({build})
                       .project({"c0 AS u_key"})
                       .planNode();
  core::PlanNodeId scanId;
  core::PlanNodeId joinId;
  auto plan = PlanBuilder(planNodeIdGenerator, pool_.get())
                  .startTableScan()
                  .connectorId(kCudfHiveConnectorId)
                  .outputType(rowType)
                  .tableHandle(tableHandle)
                  .assignments(
                      HiveConnectorTestBase::allRegularColumns(rowType))
                  .endTableScan()
                  .capturePlanNodeId(scanId)
                  .hashJoin(
                      {"c0"},
                      {"u_key"},
                      buildSide,
                      "",
                      {"c0"},
                      core::JoinType::kInner)
                  .capturePlanNodeId(joinId)
                  .planNode();

  std::atomic<int32_t> selected{-1};
  const bool testValuesWereEnabled = common::testutil::TestValue::enabled();
  common::testutil::TestValue::enable();
  SCOPE_EXIT {
    if (!testValuesWereEnabled) {
      common::testutil::TestValue::disable();
    }
  };
  common::testutil::ScopedTestValue selectedCounter(
      "facebook::velox::cudf_velox::connector::hive::CudfSplitReader::selectedRowGroups",
      std::function<void(void*)>([&](void* value) {
        selected = *static_cast<size_t*>(value);
      }));
  auto task = AssertQueryBuilder(plan, duckDbQueryRunner_)
                  .config(
                      core::QueryConfig::kHashProbeBloomFilterPushdownMaxSize,
                      std::to_string(2 << 20))
                  .maxDrivers(1)
                  .splits(scanId, makeCudfHiveConnectorSplits({filePath}))
                  .assertResults(
                      "SELECT t.c0 FROM t JOIN u ON t.c0 = u.c0 "
                      "WHERE t.c0 NOT IN (0, 2)");
#ifndef NDEBUG
  EXPECT_EQ(selected, 1);
#endif
  const auto stats = toPlanStats(task->taskStats());
  EXPECT_EQ(stats.at(joinId).customStats.at("dynamicFiltersProduced").sum, 1);
  EXPECT_GT(stats.at(joinId).customStats.at("bloomFilterSize").sum, 0);
  EXPECT_EQ(stats.at(scanId).customStats.at("dynamicFiltersAccepted").sum, 1);
}

TEST_F(TableScanTest, decimalSubfieldFilter) {
  auto rowType = ROW({"c0", "c1"}, {DECIMAL(5, 2), BIGINT()});
  auto vector = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int64_t>({100, -500, -700, -500}, DECIMAL(5, 2)),
       makeFlatVector<int64_t>({1, 2, 3, 4})});

  std::vector<RowVectorPtr> vectors = {vector};
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);
  createDuckDbTable(vectors);

  common::SubfieldFilters subfieldFilters =
      common::test::SubfieldFiltersBuilder()
          .add(
              "c0",
              std::make_unique<common::BigintRange>(
                  int64_t{-500}, int64_t{-500}, /*nullAllowed*/ false))
          .build();

  auto tableHandle = makeTableHandle(
      "parquet_table", rowType, std::move(subfieldFilters), nullptr);
  auto assignments =
      facebook::velox::exec::test::HiveConnectorTestBase::allRegularColumns(
          rowType);

  auto plan = PlanBuilder()
                  .startTableScan()
                  .outputType(rowType)
                  .tableHandle(tableHandle)
                  .assignments(assignments)
                  .endTableScan()
                  .planNode();

  assertQuery(
      plan,
      {filePath},
      "SELECT c0, c1 FROM tmp WHERE c0 = CAST('-5.00' AS DECIMAL(5, 2))");
}

TEST_F(TableScanTest, decimalRemainingFilter) {
  auto rowType = ROW({"c0", "c1"}, {DECIMAL(5, 2), BIGINT()});
  auto vector = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int64_t>({100, -500, -700, -500}, DECIMAL(5, 2)),
       makeFlatVector<int64_t>({1, 2, 3, 4})});

  std::vector<RowVectorPtr> vectors = {vector};
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);
  createDuckDbTable(vectors);

  auto assignments =
      facebook::velox::exec::test::HiveConnectorTestBase::allRegularColumns(
          rowType);

  auto plan = PlanBuilder(pool_.get())
                  .startTableScan()
                  .connectorId(kCudfHiveConnectorId)
                  .outputType(rowType)
                  .dataColumns(rowType)
                  .assignments(assignments)
                  .remainingFilter("c0 = CAST('-5.00' AS DECIMAL(5, 2))")
                  .endTableScan()
                  .planNode();

  assertQuery(
      plan,
      {filePath},
      "SELECT c0, c1 FROM tmp WHERE c0 = CAST('-5.00' AS DECIMAL(5, 2))");
}

// Velox's parquet writer stores DECIMAL(7, 2) as INT32 when
// enableStoreDecimalAsInteger is true, and cuDF's reader maps INT32 decimals
// to DECIMAL32. Velox short decimals are always DECIMAL64, so the scan output
// must be cast from DECIMAL32 to DECIMAL64.
TEST_F(TableScanTest, lowPrecisionDecimalScan) {
  auto rowType = ROW({"d"}, {DECIMAL(7, 2)});
  auto vector = makeRowVector(
      {"d"},
      {makeNullableFlatVector<int64_t>(
          {12345, std::nullopt, -2500, 300}, DECIMAL(7, 2))});
  assertDecimalScanRoundTrip(vector, rowType);
}

TEST_F(TableScanTest, lowPrecisionDecimalScanNoCast) {
  auto rowType = ROW({"d"}, {DECIMAL(12, 4)});
  auto vector = makeRowVector(
      {"d"},
      {makeNullableFlatVector<int64_t>(
          {123456789, std::nullopt, -999999}, DECIMAL(12, 4))});
  assertDecimalScanRoundTrip(vector, rowType);
}

TEST_F(TableScanTest, nestedDecimalScan) {
  auto rowType = ROW({"s"}, {ROW({"x", "d"}, {INTEGER(), DECIMAL(7, 2)})});
  auto vector = makeRowVector(
      {"s"},
      {makeRowVector(
          {"x", "d"},
          {makeNullableFlatVector<int32_t>({1, 2, std::nullopt}),
           makeNullableFlatVector<int64_t>(
               {100, std::nullopt, -200}, DECIMAL(7, 2))})});
  assertDecimalScanRoundTrip(vector, rowType);
}

TEST_F(TableScanTest, arrayDecimalScan) {
  auto rowType = ROW({"a"}, {ARRAY(DECIMAL(7, 2))});
  auto elements = makeNullableFlatVector<int64_t>(
      {100, 200, std::nullopt, 300}, DECIMAL(7, 2));
  auto vector = makeRowVector({"a"}, {makeArrayVector({0, 2}, elements)});
  assertDecimalScanRoundTrip(vector, rowType);
}

// Exercises the recursive cast through struct -> struct -> list nesting so a
// decimal buried several levels deep is normalized. Schema:
// struct<int, decimal, struct<int, list<decimal>>>.
TEST_F(TableScanTest, multiLevelNestedDecimalScan) {
  auto rowType =
      ROW({"s"},
          {ROW(
              {"x", "d", "nested"},
              {INTEGER(),
               DECIMAL(7, 2),
               ROW({"y", "a"}, {INTEGER(), ARRAY(DECIMAL(7, 2))})})});
  auto listElements = makeNullableFlatVector<int64_t>(
      {100, 200, std::nullopt, 300, 400}, DECIMAL(7, 2));
  auto vector = makeRowVector(
      {"s"},
      {makeRowVector(
          {"x", "d", "nested"},
          {makeNullableFlatVector<int32_t>({1, 2, std::nullopt}),
           makeNullableFlatVector<int64_t>(
               {100, std::nullopt, -200}, DECIMAL(7, 2)),
           makeRowVector(
               {"y", "a"},
               {makeNullableFlatVector<int32_t>({10, std::nullopt, 30}),
                makeArrayVector({0, 2, 4}, listElements)})})});
  assertDecimalScanRoundTrip(vector, rowType);
}
