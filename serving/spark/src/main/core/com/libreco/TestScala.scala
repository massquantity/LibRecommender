package com.libreco

import com.libreco.utils.Context
import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer}
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._

object TestScala extends Context{
  import spark.implicits._

  def main(args: Array[String]): Unit = {
    val ratingSchema = new StructType(Array(
      StructField("user_id", IntegerType, nullable = false),
      StructField("anime_id", IntegerType, nullable = false),
      StructField("rating", IntegerType, nullable = false)
    ))
    val animeSchema = new StructType(Array(
      StructField("anime_id", IntegerType, nullable = false),
      StructField("name", StringType, nullable = true),
      StructField("genre", StringType, nullable = true),
      StructField("type", StringType, nullable = true),
      StructField("episodes", IntegerType, nullable = true),
      StructField("rating", DoubleType, nullable = true),
      StructField("members", IntegerType, nullable = true)
    ))

    val ratingPath = this.getClass.getResource("/anime_rating.csv").toString
    val animePath = this.getClass.getResource("/anime_info.csv").toString
    var rating = spark.read
      .option("inferSchema", "true")
      .option("header", "true")
      .schema(ratingSchema)
      .csv(ratingPath)
    var anime = spark.read
      .option("inferSchema", "true")
      .option("header", "true")
      .schema(animeSchema)
      .csv(animePath)

    rating = rating.sample(withReplacement = false, 0.1)
    anime = anime.withColumnRenamed("rating", "web_rating").drop($"rating")
    var data = rating.join(anime, Seq("anime_id"), "inner")
    data.cache()
    data.show(4, truncate = false)

    val total_count = data.count()
    println(s"data count: $total_count")
    data.groupBy("rating").count().orderBy($"count".desc)
      .withColumn("percent", round($"count" / sum("count").over(), 4)).show()
  //  data = data.withColumn("label", when($"rating" >= 8, 1).otherwise(0))
  //  data.groupBy("label").count().orderBy($"count".desc)
  //    .withColumn("percent", round($"count" / sum("count").over(), 4)).show()

    val udfMapValue = udf(mapValue(_:Int): Int)
    data = data.withColumn("label", udfMapValue($"rating"))
    data.groupBy("label").count()
      .withColumn("percent", round($"count" / sum("count").over(), 4)).show()
    data.unpersist()



    val sentenceDataFrame = spark.createDataFrame(Seq(
      (0, "Kimi no Na wa."),
      (1, "Gintama°"),
      (2, "Gintama Movie: Kanketsu-hen - Yorozuya yo Eien Nare"),
      (3, "Yuri!!! on Ice"),
      (4, "Arslan Senki (TV) OVA"),
      (5, "IS: Infinite Stratos 2 - Hitonatsu no Omoide"),
      (6, "Chi&#039")
    )).toDF("id", "sentence")

    val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
    val regexTokenizer = new RegexTokenizer()
      .setInputCol("sentence")
      .setOutputCol("words")
      .setPattern("\\w+")
      .setGaps(false)
      .setToLowercase(true)

    val countTokens = udf { words: Seq[String] => words.length }

    val tokenized = tokenizer.transform(sentenceDataFrame)
    tokenized.select("sentence", "words")
      .withColumn("tokens", countTokens(col("words"))).show(false)

    val regexTokenized = regexTokenizer.transform(sentenceDataFrame)
    regexTokenized.select("sentence", "words")
      .withColumn("tokens", countTokens(col("words"))).show(false)
  }

  def mapValue(rating: Int): Int = {
    rating match {
      case `rating` if rating >= 9 => 2
      case `rating` if rating >= 6 && rating <= 8 => 1
      case _ => 0
    }
  }
}
