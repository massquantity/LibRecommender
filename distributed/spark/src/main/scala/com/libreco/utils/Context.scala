package com.libreco.utils

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

trait Context {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("com").setLevel(Level.ERROR)

  lazy val sparkConf: SparkConf = new SparkConf()
    .setAppName("Spark Recommender")
    .setMaster("local[*]")
  //  .set("spark.core.max", "4")

  lazy val spark: SparkSession = SparkSession
    .builder()
    .config(sparkConf)
    .getOrCreate()

  def time[T](block: => T, info: String): T = {
    val t0 = System.nanoTime()
    val result = block
    val t1 = System.nanoTime()
    println(f"$info time: ${(t1 - t0) / 1e9d}%.2fs")
    result
  }
}