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
package org.apache.spark.sql.execution.benchmark

import org.apache.gluten.config.VeloxConfig

import org.apache.spark.benchmark.Benchmark

/**
 * Benchmark to measure the effect of Velox file handle caching on repeated scans of many small
 * Parquet files. File handle caching avoids repeated open/close overhead per file, which is
 * especially beneficial for remote filesystems (S3, HDFS, ABFS) where handle creation involves
 * network round-trips (DNS, TCP, auth).
 *
 * Even on local filesystems the overhead is measurable when scanning hundreds of small files
 * multiple times (e.g., repeated queries on a heavily-partitioned table).
 *
 * NOTE: `fileHandleCacheEnabled` is a static config (read at backend init). To compare on vs off,
 * run this benchmark twice with different Spark configurations:
 * {{{
 *   # With file handle cache enabled (default):
 *   bin/spark-submit --class <this class> \
 *     --conf spark.gluten.sql.columnar.backend.velox.fileHandleCacheEnabled=true \
 *     --jars <spark core test jar>,<sql core test jar> \
 *     <application jar>
 *
 *   # With file handle cache disabled:
 *   bin/spark-submit --class <this class> \
 *     --conf spark.gluten.sql.columnar.backend.velox.fileHandleCacheEnabled=false \
 *     --jars <spark core test jar>,<sql core test jar> \
 *     <application jar>
 * }}}
 *
 * Expected result: with caching enabled, repeated scans should show lower wall-clock time due to
 * avoiding per-file open() syscalls (or remote filesystem connection establishment) on subsequent
 * scans of the same files.
 */
object FileHandleCacheBenchmark extends SqlBasedBenchmark {

  // Number of small files to generate (simulates a heavily-partitioned table)
  private val numFiles = 200
  // Rows per file (small to emphasize per-file overhead over per-row compute)
  private val rowsPerFile = 5000
  // Number of repeated scans per benchmark iteration
  private val scanIterations = 10

  override def runBenchmarkSuite(mainArgs: Array[String]): Unit = {
    val totalRows = numFiles.toLong * rowsPerFile

    val cacheEnabled = spark.conf.get(
      VeloxConfig.COLUMNAR_VELOX_FILE_HANDLE_CACHE_ENABLED.key,
      "true")
    val label = s"fileHandleCache=$cacheEnabled"

    withTempPath {
      dir =>
        val path = dir.getCanonicalPath

        // Generate many small Parquet files by repartitioning
        spark
          .range(totalRows)
          .selectExpr(
            "id",
            "cast(id % 100 as int) as category",
            "cast(id * 1.5 as double) as value",
            "uuid() as payload"
          )
          .repartition(numFiles)
          .write
          .parquet(path)

        val fileCount = dir.listFiles().count(_.getName.endsWith(".parquet"))
        // scalastyle:off println
        println(s"Generated $fileCount parquet files with ~$rowsPerFile rows each")
        println(s"Config: $label")
        // scalastyle:on println

        val benchmark = new Benchmark(
          s"Repeated scan of $fileCount small files ($label)",
          totalRows * scanIterations,
          output = output)

        // Warm up: first scan populates any one-time init (Velox backend, JIT, etc.)
        spark.read.parquet(path).selectExpr("sum(value)").collect()

        benchmark.addCase(s"full scan ($scanIterations iterations)", 5) {
          _ =>
            for (_ <- 1 to scanIterations) {
              spark.read.parquet(path).selectExpr("sum(value)", "count(*)").collect()
            }
        }

        benchmark.addCase(s"filtered scan ($scanIterations iterations)", 5) {
          _ =>
            for (_ <- 1 to scanIterations) {
              spark.read.parquet(path)
                .where("category < 10")
                .selectExpr("sum(value)", "count(*)")
                .collect()
            }
        }

        benchmark.addCase(s"column pruning scan ($scanIterations iterations)", 5) {
          _ =>
            for (_ <- 1 to scanIterations) {
              spark.read.parquet(path).selectExpr("sum(id)").collect()
            }
        }

        benchmark.run()
    }
  }
}
