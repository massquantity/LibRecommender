package com.libreco.example

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{Column, Dataset, Row, SparkSession}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.recommendation.{ALS, ALSModel}
import org.apache.spark.sql.functions.{round, sum, lit}
import com.libreco.utils.Context
import com.libreco.data.DataSplitter
import com.libreco.model.Recommender

import scala.util.Random

object AlsExample extends Context{
  import spark.implicits._

  def main(args: Array[String]): Unit = {
    val dataPath = this.getClass.getResource("/ml-1m/ratings.dat").toString
    val splitter = new DataSplitter()
    var data = spark.read.textFile(dataPath)
      .map(splitter.parseRating("::"))
      .toDF("user", "item", "rating", "timestamp")
      .withColumn("label", lit(1))
    data.show(4, truncate = false)
  //  data.columns.foreach(x => println(s"$x -> ${data.filter(data(x).isNull).count}"))
  //  data.groupBy("rating").count().orderBy($"count".desc)
  //      .withColumn("percent", round($"count" / sum("count").over(), 4)).show()

    data = data.sample(withReplacement = false, 0.1)   // sample 0.1
    val Array(train, test) = splitter.stratified_chrono_split(data, 0.8, "user")
    val model = new Recommender()
    time(model.train(data), "Training")
    val transformedData = model.transform(data)
    transformedData.show(4, truncate = false)
    val rec = model.recommendForUsers(test, 10)
    rec.show(20, truncate = false)

  }
}
