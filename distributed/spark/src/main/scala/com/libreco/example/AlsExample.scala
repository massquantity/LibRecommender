package com.libreco.example

import org.apache.spark.sql.functions.{round, sum, lit}
import com.libreco.utils.{Context, ItemNameConverter}
import com.libreco.data.DataSplitter
import com.libreco.model.Recommender

import scala.util.Random


object AlsExample extends Context {
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

    val model = new Recommender()
    time(model.train(data), "Training")
    val transformedData = model.transform(data)
    transformedData.show(4, truncate = false)

    val movieMap = ItemNameConverter.getId2ItemName()
    val rec = model.recommendForUsers(data, 10, movieMap)
    rec.show(20, truncate = false)

  //  val model = new Recommender()
  //  time(model.train(data, evaluate = true, num = 10), "Evaluating")
  }
}
