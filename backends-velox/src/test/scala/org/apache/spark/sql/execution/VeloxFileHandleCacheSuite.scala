/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.spark.sql.execution

import org.apache.gluten.config.VeloxConfig
import org.apache.gluten.execution.{BasicScanExecTransformer, VeloxWholeStageTransformerSuite}

import org.apache.spark.SparkConf

/**
 * Test suite for Velox file handle cache behavior.
 *
 * Tests correctness, config propagation, and edge cases for the file handle cache which caches open
 * file handles (descriptors) to avoid repeated open/close overhead.
 */
class VeloxFileHandleCacheSuite extends VeloxWholeStageTransformerSuite {
  override protected val resourcePath: String = "/parquet-for-read"
  override protected val fileFormat: String = "parquet"

  override protected def sparkConf: SparkConf = {
    super.sparkConf
      .set(VeloxConfig.COLUMNAR_VELOX_FILE_HANDLE_CACHE_ENABLED.key, "true")
      .set(VeloxConfig.COLUMNAR_VELOX_FILE_HANDLE_EXPIRATION_DURATION_MS.key, "2000")
      .set(VeloxConfig.COLUMNAR_VELOX_NUM_CACHE_FILE_HANDLES.key, "10000")
  }

  testWithSpecifiedSparkVersion(
    "basic scan correctness with file handle cache enabled",
    "3.5",
    "3.5") {
    // Verify that enabling file handle cache produces correct scan results
    withTempPath {
      dir =>
        spark
          .range(10000)
          .selectExpr("id", "cast(id % 7 as int) as category", "id * 1.5 as value")
          .repartition(10)
          .write
          .parquet(dir.getCanonicalPath)

        val df = spark.read.parquet(dir.getCanonicalPath)
        df.createOrReplaceTempView("t")

        runQueryAndCompare("SELECT count(*) FROM t") {
          checkGlutenPlan[BasicScanExecTransformer]
        }
        runQueryAndCompare("SELECT sum(value) FROM t WHERE category = 3") {
          checkGlutenPlan[BasicScanExecTransformer]
        }
        runQueryAndCompare("SELECT category, count(*) FROM t GROUP BY category") {
          checkGlutenPlan[BasicScanExecTransformer]
        }
    }
  }

  testWithSpecifiedSparkVersion(
    "repeated scans produce consistent results (cache hit path)",
    "3.5",
    "3.5") {
    // Repeated scans of the same files must produce identical results regardless
    // of whether handles are served from cache or re-opened after TTL eviction.
    withTempPath {
      dir =>
        spark
          .range(5000)
          .selectExpr("id", "cast(id as string) as name")
          .repartition(50) // 50 files to exercise many cache entries
          .write
          .parquet(dir.getCanonicalPath)

        val path = dir.getCanonicalPath
        val expected = spark.read.parquet(path).count()
        assert(expected == 5000)

        // Verify scans go through Gluten/Velox
        checkGlutenPlan[BasicScanExecTransformer](spark.read.parquet(path))

        // Scan the same files multiple times - results must be consistent
        for (i <- 1 to 5) {
          val count = spark.read.parquet(path).count()
          assert(
            count == expected,
            s"Iteration $i: expected $expected rows but got $count")
        }

        // Verify aggregation consistency across repeated scans
        val firstSum = spark.read.parquet(path).selectExpr("sum(id)").collect()(0).getLong(0)
        for (i <- 1 to 3) {
          val sum = spark.read.parquet(path).selectExpr("sum(id)").collect()(0).getLong(0)
          assert(
            sum == firstSum,
            s"Iteration $i: sum mismatch, expected $firstSum but got $sum")
        }
    }
  }

  testWithSpecifiedSparkVersion(
    "many small files do not cause errors with file handle cache",
    "3.5",
    "3.5") {
    // Verify that scanning many small files with caching enabled does not cause
    // file descriptor exhaustion or other resource-related errors.
    withTempPath {
      dir =>
        // Create 200 small parquet files
        spark
          .range(20000)
          .selectExpr("id", "uuid() as payload")
          .repartition(200)
          .write
          .parquet(dir.getCanonicalPath)

        val fileCount = dir.listFiles().count(_.getName.endsWith(".parquet"))
        assert(fileCount >= 200, s"Expected at least 200 files, got $fileCount")

        // Verify scans go through Gluten/Velox
        checkGlutenPlan[BasicScanExecTransformer](spark.read.parquet(dir.getCanonicalPath))

        // Scan all files - should work without resource errors
        val count = spark.read.parquet(dir.getCanonicalPath).count()
        assert(count == 20000)

        // Scan again (cache hit path) - should also work
        val count2 = spark.read.parquet(dir.getCanonicalPath).count()
        assert(count2 == 20000)
    }
  }

  testWithSpecifiedSparkVersion(
    "filtered scan correctness with file handle cache",
    "3.5",
    "3.5") {
    // Verify that predicate pushdown works correctly with cached file handles.
    // This exercises the row group skipping path through cached handles.
    withTempPath {
      dir =>
        spark
          .range(100000)
          .selectExpr(
            "id",
            "cast(id % 10 as int) as partition_key",
            "cast(id * 0.01 as double) as metric")
          .repartition(20)
          .write
          .parquet(dir.getCanonicalPath)

        val path = dir.getCanonicalPath

        // Verify scans go through Gluten/Velox
        checkGlutenPlan[BasicScanExecTransformer](
          spark.read.parquet(path).where("partition_key = 5"))

        // Filter that matches ~10% of rows
        val filtered = spark.read.parquet(path).where("partition_key = 5").count()
        assert(filtered == 10000, s"Expected 10000 filtered rows, got $filtered")

        // Range filter
        val rangeFiltered = spark.read.parquet(path).where("id >= 50000").count()
        assert(rangeFiltered == 50000, s"Expected 50000 range-filtered rows, got $rangeFiltered")

        // Re-run same filters (cache hit path)
        val filtered2 = spark.read.parquet(path).where("partition_key = 5").count()
        assert(filtered2 == filtered, "Filtered count mismatch on repeated scan")
    }
  }

  testWithSpecifiedSparkVersion(
    "scan after file deletion produces appropriate error or empty result",
    "3.5",
    "3.5") {
    // If a file is deleted between scans, the next scan should either:
    // - Succeed (if the cached FD still works on Linux with unlinked inodes)
    // - Produce an error (not silently return wrong data)
    withTempPath {
      dir =>
        spark
          .range(1000)
          .selectExpr("id")
          .repartition(5)
          .write
          .parquet(dir.getCanonicalPath)

        val path = dir.getCanonicalPath
        // First scan populates the cache
        val count1 = spark.read.parquet(path).count()
        assert(count1 == 1000)

        // Verify scans go through Gluten/Velox
        checkGlutenPlan[BasicScanExecTransformer](spark.read.parquet(path))

        // Delete one parquet file
        val parquetFiles = dir.listFiles().filter(_.getName.endsWith(".parquet"))
        assert(parquetFiles.nonEmpty)
        val deletedFile = parquetFiles.head
        val deletedRows = spark.read.parquet(deletedFile.getCanonicalPath).count()
        assert(deletedFile.delete(), s"Failed to delete ${deletedFile.getCanonicalPath}")

        // On Linux, the cached FD to the deleted file may still work (unlinked inode).
        // Either way, the remaining files should be readable.
        // The scan may also throw if the FS detects the missing file.
        try {
          val count2 = spark.read.parquet(path).count()
          // The count should be either (count1 - deletedRows) or count1
          // depending on whether the OS kept the inode accessible
          assert(
            count2 == count1 || count2 == count1 - deletedRows,
            s"Unexpected count after deletion: $count2 (original: $count1, deleted: $deletedRows)")
        } catch {
          case e: Exception
              if e.getMessage != null &&
                (e.getMessage.contains("FileNotFoundException") ||
                  e.getMessage.contains("No such file") ||
                  e.getMessage.contains("Path does not exist") ||
                  e.getMessage.contains("does not exist")) =>
          // Acceptable: the scan failed because the deleted file is no longer accessible.
          // The important thing is that it does not silently return wrong data.
        }
    }
  }

  testWithSpecifiedSparkVersion(
    "TTL-based eviction: scans succeed after cached handles expire",
    "3.5",
    "3.5") {
    // Verify that scans still produce correct results after the configured TTL
    // (2s, set in sparkConf) has elapsed. This exercises the path where cached
    // handles may have been evicted and must be re-opened transparently.
    withTempPath {
      dir =>
        spark
          .range(5000)
          .selectExpr("id", "id * 2 as doubled")
          .repartition(20)
          .write
          .parquet(dir.getCanonicalPath)

        val path = dir.getCanonicalPath

        // First scan populates the cache
        val count1 = spark.read.parquet(path).count()
        assert(count1 == 5000)

        // Verify scans go through Gluten/Velox
        checkGlutenPlan[BasicScanExecTransformer](spark.read.parquet(path))

        val sum1 = spark.read.parquet(path).selectExpr("sum(id)").collect()(0).getLong(0)

        // Wait for TTL to expire (configured to 2s in sparkConf)
        Thread.sleep(3000)

        // Scan after TTL expiration: verify results remain correct
        // (handles may have been evicted and transparently re-opened)
        val count2 = spark.read.parquet(path).count()
        assert(count2 == 5000, s"Count mismatch after TTL expiration: expected 5000, got $count2")
        val sum2 = spark.read.parquet(path).selectExpr("sum(id)").collect()(0).getLong(0)
        assert(sum2 == sum1, s"Sum mismatch after TTL expiration: expected $sum1, got $sum2")
    }
  }

  testWithSpecifiedSparkVersion(
    "column pruning with cached file handles",
    "3.5",
    "3.5") {
    // Verify that column pruning works correctly when file handles are cached.
    // The cache key includes the file path but not the projected columns, so
    // different projections on the same file must still work correctly.
    withTempPath {
      dir =>
        spark
          .range(5000)
          .selectExpr("id", "id * 2 as doubled", "id * 3 as tripled", "uuid() as text")
          .repartition(10)
          .write
          .parquet(dir.getCanonicalPath)

        val path = dir.getCanonicalPath

        // Verify scans go through Gluten/Velox
        checkGlutenPlan[BasicScanExecTransformer](spark.read.parquet(path))

        // Read all columns
        val allCols = spark.read.parquet(path).select("id", "doubled", "tripled", "text").count()
        assert(allCols == 5000)

        // Read subset of columns (same file handles, different projection)
        val subset1Df = spark.read.parquet(path).select("id")
        assert(subset1Df.schema.fieldNames.sameElements(Array("id")))
        assert(subset1Df.count() == 5000)

        // Different subset
        val subset2 = spark.read.parquet(path).selectExpr("sum(doubled)").collect()
        val expectedSum = (0L until 5000L).map(_ * 2).sum
        assert(subset2(0).getLong(0) == expectedSum)
    }
  }
}
