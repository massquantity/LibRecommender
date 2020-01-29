package com.libreco.example

import org.apache.spark.ml.linalg.SparseVector
import com.libreco.utils.{Context, ItemNameConverter}
import com.libreco.model.Regressor


object RegressorExample extends Context {
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
    val data = temp.join(items, Seq("item"), "left")
    data.show(4, truncate = false)

  //  val model = new Regressor(Some("glr"))
  //  time(model.train(data, evaluate = false), "Training")
  //  val transformedData = model.transform(data)
  //  transformedData.show(4, truncate = false)

    val model = new Regressor(Some("gbdt"))
    time(model.train(data, evaluate = true, debug = true), "Evaluating")
  }
}






