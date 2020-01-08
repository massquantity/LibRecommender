package libreco.example

import org.apache.spark.sql.{SparkSession, DataFrame, Dataset}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.{PipelineModel, Pipeline, PipelineStage}
import org.apache.spark.ml.feature.{VectorAssembler, OneHotEncoder, StringIndexer}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType, DoubleType}
import org.apache.spark.ml.regression.{GBTRegressor, GBTRegressionModel}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.log4j.{Level, Logger}
import libreco.model.GBDTRegression
import libreco.serving.jpmml.{ModelSerializer => ModelSerializerJPmml}
import libreco.serving.mleap.{ModelSerializer => ModelSerializerMLeap}

import scala.collection.mutable.ArrayBuffer

object ModelSerialization {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("com").setLevel(Level.ERROR)

    val conf = new SparkConf()
      .setMaster("local[*]")
      .setAppName("ModelSerialization")
    val sc = new SparkContext(conf)

    val spark = SparkSession
      .builder
      .config(conf)
      .getOrCreate()

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
    anime = anime.withColumnRenamed("rating", "web_rating").drop("rating")
    var data = rating.join(anime, Seq("anime_id"), "inner")

  //  println(s"find and fill NAs for each column...")
  //  data.columns.foreach(x => println(s"$x -> ${data.filter(data(x).isNull).count}"))
    data = data.na.fill("Missing", Seq("genre"))
      .na.fill("Missing", Seq("type"))
      .na.fill(7.6, Seq("web_rating"))

    val model = new GBDTRegression()
    val currentTime = System.currentTimeMillis()
    time(model.train(data), "Training")
    val transformedData = model.transform(data)
    transformedData.show(4)

  //  val jpmmlModelPath = "serving/spark/src/main/resources/jpmml_model/GBDT_model.xml"
  //  val jpmmlModelSerializer = new ModelSerializerJPmml()
  //  jpmmlModelSerializer.serializeModel(model.pipelineModel, jpmmlModelPath, transformedData)

    val mleapModelPath = "jar:file:/home/massquantity/Workspace/LibRecommender/" +
      "serving/spark/src/main/resources/mleap_model/GBDT_model.zip"
    val mleapModelSerializer = new ModelSerializerMLeap()
    mleapModelSerializer.serializeModel(model.pipelineModel, mleapModelPath, transformedData)
  }

  def time[T](block: => T, info: String): T = {
    val t0 = System.nanoTime()
    val result = block
    val t1 = System.nanoTime()
    println(f"$info time: ${(t1 - t0) / 1e9d}%.2fs")
    result
  }
}
