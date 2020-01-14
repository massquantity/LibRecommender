package com.libreco.model

import com.libreco.utils.Context
import org.apache.spark.sql.{Column, DataFrame, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
import org.apache.spark.sql.functions.{col, when}
import org.apache.spark.sql.types.{DoubleType, IntegerType, StringType, StructField, StructType}
import org.apache.spark.ml.regression.{GBTRegressor, GeneralizedLinearRegression}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.param.ParamMap
import org.apache.log4j.{Level, Logger}

import scala.util.Random

class Regressor(evaluate: Boolean = false,
                algo: Option[String] = Some("gbdt"),
                convertImplicit: Boolean = false) extends Serializable with Context {
  import spark.implicits._
  var pipelineModel: PipelineModel = _
  var data: DataFrame = _

  def train(dataset: DataFrame): Unit = {
    val prePipelineStages: Array[PipelineStage] = FeatureEngineering.preProcessPipeline(dataset)
  //  val pipeline = new Pipeline().setStages(prePipelineStages)
  //  pipelineModel = pipeline.fit(dataset)
  //  val transformed = pipelineModel.transform(dataset)
  //  transformed.show(4, truncate = false)

    if (convertImplicit) {
      data = dataset.withColumn("label", when($"rating" >= 8, 1).otherwise(0))
    } else {
      data = dataset
    }
    data.cache()

    if (evaluate) {
      val Array(trainData, testData) = data.randomSplit(Array(0.8, 0.2), 2020L)
      trainData.cache()
      testData.cache()
      algo match {
        case Some("gbdt") => evaluateGBDT(trainData, testData, prePipelineStages)
        case Some("glr") => evaluateGLR(trainData, testData, prePipelineStages)
        case None =>
          println("Model muse be GBDTRegressor or GeneralizedLinearRegression")
          System.exit(1)
        case _ =>
          println("Model muse be GBDTRegressor or GeneralizedLinearRegression")
          System.exit(2)
      }
      trainData.unpersist()
      testData.unpersist()
    }
    else {
      algo match {
        case Some("gbdt") =>
          val model = new GBTRegressor()
            .setFeaturesCol("featureVector")
            .setLabelCol("rating")
            .setPredictionCol("pred")
            .setFeatureSubsetStrategy("auto")
            .setMaxDepth(3)
            .setMaxIter(5)
            .setStepSize(0.1)
            .setSubsamplingRate(0.8)
            .setSeed(2020L)
          val pipelineStages = prePipelineStages ++ Array(model)
          val pipeline = new Pipeline().setStages(pipelineStages)
          pipelineModel = pipeline.fit(data)

        case Some("glr") =>
          val model = new GeneralizedLinearRegression()
            .setFeaturesCol("featureVector")
            .setLabelCol("rating")
            .setPredictionCol("pred")
            .setFamily("gaussian")
            .setLink("identity")
            .setRegParam(0.0)
          val pipelineStages = prePipelineStages ++ Array(model)
          val pipeline = new Pipeline().setStages(pipelineStages)
          pipelineModel = pipeline.fit(data)

        case _ =>
          println("Model muse be GBDTRegressor or GeneralizedLinearRegression")
          System.exit(1)
      }
    }
    data.unpersist()
  }

  def transform(dataset: DataFrame): DataFrame = {
    pipelineModel.transform(dataset)
  }

  def evaluateGBDT(trainData: DataFrame,
                   testData: DataFrame,
                   pipelineStages: Array[PipelineStage]): Unit = {
    val gbr = new GBTRegressor()
      .setSeed(Random.nextLong())
      .setFeaturesCol("featureVector")
      .setLabelCol("rating")
      .setPredictionCol("pred")

    val pipeline = new Pipeline().setStages(pipelineStages ++ Array(gbr))
    val paramGrid = new ParamGridBuilder()
      .addGrid(gbr.featureSubsetStrategy, Seq("onethird", "all", "sqrt", "log2"))
      .addGrid(gbr.maxDepth, Seq(3, 5, 7, 8))
      .addGrid(gbr.stepSize, Seq(0.01, 0.03))
      .addGrid(gbr.subsamplingRate, Seq(0.6, 0.8, 1.0))
      .build()
    evaluate(trainData, testData, pipeline, paramGrid)
  }

  def evaluateGLR(trainData: DataFrame,
                  testData: DataFrame,
                  pipelineStages: Array[PipelineStage]): Unit = {
    val glr = new GeneralizedLinearRegression()
      .setFeaturesCol("featureVector")
      .setLabelCol("rating")
      .setPredictionCol("pred")

    val pipeline = new Pipeline().setStages(pipelineStages ++ Array(glr))
    val paramGrid = new ParamGridBuilder()
      .addGrid(glr.family, Seq("gaussian"))
      .addGrid(glr.link, Seq("identity"))
      .addGrid(glr.maxIter, Seq(20, 50))
      .addGrid(glr.regParam, Seq(0.0, 0.01, 0.1))
      .build()
    evaluate(trainData, testData, pipeline, paramGrid)
  }

  def evaluate(trainData: DataFrame,
               testData: DataFrame,
               pipeline: Pipeline,
               params: Array[ParamMap]): Unit = {
    val regressorEval = new RegressionEvaluator()
      .setLabelCol("rating")
      .setPredictionCol("pred")
      .setMetricName("rmse")

    val validator = new TrainValidationSplit()
      .setSeed(Random.nextLong())
      .setEstimator(pipeline)
      .setEvaluator(regressorEval)
      .setEstimatorParamMaps(params)
      .setTrainRatio(0.8)

    val validatorModel = validator.fit(trainData)
    val paramsAndMetrics = validatorModel.validationMetrics
      .zip(validatorModel.getEstimatorParamMaps).sortBy(-_._1)

    println("scores and params: ")
    paramsAndMetrics.foreach { case (metric, params) =>
      println(s"$params \t => $metric")
      println()
    }
    println()

    val bestModel = validatorModel.bestModel
    println(s"best model params: ${bestModel.asInstanceOf[PipelineModel].stages.last.extractParamMap}" +
      s" => rmse: ${validatorModel.validationMetrics.max}")
    println(s"test rmse: ${regressorEval.evaluate(bestModel.transform(testData))}")
  }
}


object Regressor extends Context{
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("com").setLevel(Level.ERROR)

    val conf = new SparkConf()
      .setMaster("local[*]")
      .setAppName("Regressor")
    val sc = new SparkContext(conf)

    val spark = SparkSession
      .builder
      .config(conf)
      .getOrCreate()
    import spark.implicits._

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

    val allCols: Array[Column] = data.columns.map(data.col)
    val nullFilter: Column = allCols.map(_.isNotNull).reduce(_ && _)
    data = data.select(allCols: _*).filter(nullFilter)
    println(s"data length: ${data.count()}")
    data.show(4, truncate = false)

    val model = new Regressor(evaluate = true, algo = Some("glr"))
    time(model.train(data), "Training")

    /*
    val trainPredAndLabel = trainData.select("rating", "featureVector").join(
      gbrModel.transform(trainData.select("featureVector")), Seq("featureVector"))
    val testPredAndLabel = testData.select("rating", "featureVector").join(
      gbrModel.transform(testData.select("featureVector")), Seq("featureVector"))

    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setPredictionCol("pred")
      .setLabelCol("rating")
    println(s"train rmse: ${evaluator.evaluate(trainPredAndLabel)}")
    println(s"test rmse: ${evaluator.evaluate(testPredAndLabel)}")
    */
  }


  def oneHotPipeline(inputCol: String): (Pipeline, String) = {
    val indexer = new StringIndexer()
      .setInputCol(inputCol)
      .setOutputCol(inputCol + "_indexed")
      .setHandleInvalid("skip")
    val encoder = new OneHotEncoder()
      .setInputCol(inputCol + "_indexed")
      .setOutputCol(inputCol + "_vec")
    val pipeline = new Pipeline().setStages(Array(indexer, encoder))
    (pipeline, encoder.getOutputCol)
  }
}

