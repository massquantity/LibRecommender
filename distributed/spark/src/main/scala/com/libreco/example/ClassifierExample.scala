package com.libreco.example

import com.libreco.model.Classifier
import com.libreco.utils.{Context, ItemNameConverter}
import org.apache.spark.sql.functions.{round, sum, udf}


object ClassifierExample extends Context {
  import spark.implicits._

  def main(args: Array[String]): Unit = {
    val movieNameConverter = ItemNameConverter.getItemName()
    val userPath = this.getClass.getResource("/ml-1m/users.dat").toString
    val moviePath = this.getClass.getResource("/ml-1m/movies.dat").toString
    val ratingPath = this.getClass.getResource("/ml-1m/ratings.dat").toString

    val users = spark.read.textFile(userPath)
      .selectExpr("split(value, '::') as col")
      .selectExpr(
        "cast(col[0] as int) as user",
        "cast(col[1] as string) as sex",
        "cast(col[2] as int) as age",
        "cast(col[3] as int) as occupation")
    val items = spark.read.textFile(moviePath)
      .selectExpr("split(value, '::') as col")
      .selectExpr(
        "cast(col[0] as int) as item",
        "cast(col[1] as string) as movie",
        "cast(col[2] as string) as genre")
      .withColumn("movieName", movieNameConverter($"movie"))
      .drop($"movie")
      .withColumnRenamed("movieName", "movie")
      .select("item", "movie", "genre")
    var ratings = spark.read.textFile(ratingPath)
      .selectExpr("split(value, '::') as col")
      .selectExpr(
        "cast(col[0] as int) as user",
        "cast(col[1] as int) as item",
        "cast(col[2] as int) as rating",
        "cast(col[3] as long) as timestamp")

    ratings = ratings.sample(withReplacement = false, 0.1)
    val temp = ratings.join(users, Seq("user"), "left")
    var data = temp.join(items, Seq("item"), "left")
    data.show(4, truncate = false)

  //  val model = new Classifier(Some("mlp"))
  //  time(model.train(data, evaluate = false, debug = false), "Training")
  //  val transformedData = model.transform(data)
  //  transformedData.show(4, truncate = false)

    val model = new Classifier(Some("rf"))
    time(model.train(data, evaluate = true, debug = true), "Evaluating")

    /*
    val totalCount = data.count()
    println(s"data count: $totalCount")
    data.groupBy("rating").count().orderBy($"count".desc)
      .withColumn("percent", round($"count" / sum("count").over(), 4)).show()

    val udfMapValue = udf(mapValue(_:Int): Int)
    data = data.withColumn("label", udfMapValue($"rating"))
    data.groupBy("label").count()
      .withColumn("percent", round($"count" / sum("count").over(), 4)).show()
    */
  }

  def mapValue(rating: Int): Int = {
    rating match {
      case `rating` if rating == 5 => 2
      case `rating` if rating == 4 => 1
      case _ => 0
    }
  }
}
