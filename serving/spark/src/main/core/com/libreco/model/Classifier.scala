package com.libreco.model

import com.libreco.model.Regressor.time
import com.libreco.utils.Context
import ml.combust.mleap.core.classification.RandomForestClassifierModel
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.functions.{col, udf, when}
import org.apache.spark.sql.{Column, DataFrame, SparkSession}
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.ml.classification.{MultilayerPerceptronClassifier, RandomForestClassifier}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.param.{ParamMap, Param}
import org.apache.spark.sql.types._

import scala.util.Random

class Classifier(evaluate: Boolean = false,
                 algo: Option[String] = Some("mlp"),
                 convertLabel: Boolean = false,
                 debug: Boolean = true) extends Serializable with Context{
  import spark.implicits._
  var pipelineModel: PipelineModel = _
  var data: DataFrame = _

  def train(dataset: DataFrame): Unit = {
    val prePipelineStages: Array[PipelineStage] = FeatureEngineering.preProcessPipeline(dataset)
    if (debug) {
      val pipeline = new Pipeline().setStages(prePipelineStages)
      pipelineModel = pipeline.fit(dataset)
      val transformed = pipelineModel.transform(dataset)
      transformed.show(4, truncate = false)
    }

    if (convertLabel) {
      val udfMapValue = udf(mapValue(_:Int): Double)
      data = dataset.withColumn("label", udfMapValue($"rating"))
    } else {
      data = dataset.withColumn("label", $"rating")
    }

    if (evaluate) {
      val Array(trainData, testData) = data.randomSplit(Array(0.8, 0.2), 2020L)
      trainData.cache()
      testData.cache()
      algo match {
        case Some("mlp") => evaluateMLP(trainData, testData, prePipelineStages)
        case Some("rf") => evaluateRF(trainData, testData, prePipelineStages)
        case None =>
          println("Model muse be MultilayerPerceptronClassifier or RandomForestClassifier")
          System.exit(1)
        case _ =>
          println("Model muse be MultilayerPerceptronClassifier or RandomForestClassifier")
          System.exit(2)
      }
      trainData.unpersist()
      testData.unpersist()
    }
    else {
      data.cache()
      algo match {
        case Some("mlp") =>
          val model = new MultilayerPerceptronClassifier()
            .setFeaturesCol("featureVector")
            .setLabelCol("label")
            .setPredictionCol("pred")
            .setProbabilityCol("prob")
            .setLayers(Array(29, 20, 10, 3))
            .setStepSize(0.02)
          val pipelineStages = prePipelineStages ++ Array(model)
          val pipeline = new Pipeline().setStages(pipelineStages)
          pipelineModel = pipeline.fit(data)

        case Some("rf") =>
          val model = new RandomForestClassifier()
            .setFeaturesCol("featureVector")
            .setLabelCol("label")
            .setPredictionCol("pred")
            .setProbabilityCol("prob")
            .setFeatureSubsetStrategy("auto")
            .setMaxDepth(3)
            .setNumTrees(100)
            .setSubsamplingRate(1.0)
            .setSeed(Random.nextLong())
          val pipelineStages = prePipelineStages ++ Array(model)
          val pipeline = new Pipeline().setStages(pipelineStages)
          pipelineModel = pipeline.fit(data)

        case _ =>
          println("Model muse be MultilayerPerceptronClassifier or RandomForestClassifier")
          System.exit(1)
      }
      data.unpersist()
    }
  }

  def transform(dataset: DataFrame): DataFrame = {
    pipelineModel.transform(dataset)
  }

  def evaluateMLP(trainData: DataFrame,
                  testData: DataFrame,
                  pipelineStages: Array[PipelineStage]): Unit = {
    val mlp = new MultilayerPerceptronClassifier()
      .setSeed(Random.nextLong())
      .setFeaturesCol("featureVector")
      .setLabelCol("label")
      .setPredictionCol("pred")
      .setProbabilityCol("prob")

    val pipeline = new Pipeline().setStages(pipelineStages ++ Array(mlp))
    val paramGrid = new ParamGridBuilder()
      .addGrid(mlp.layers, Seq(Array[Int](29, 20, 10, 3), Array[Int](29, 50, 30, 10, 3)))
      .addGrid(mlp.stepSize, Seq(0.01, 0.03, 0.05))
      .addGrid(mlp.maxIter, Seq(100, 300, 500))
      .build()
    evaluate(trainData, testData, pipeline, paramGrid)
  }

  def evaluateRF(trainData: DataFrame,
                  testData: DataFrame,
                  pipelineStages: Array[PipelineStage]): Unit = {
    val rf = new RandomForestClassifier()
      .setSeed(Random.nextLong())
      .setFeaturesCol("featureVector")
      .setLabelCol("label")
      .setPredictionCol("pred")
      .setProbabilityCol("prob")

    val pipeline = new Pipeline().setStages(pipelineStages ++ Array(rf))
    val paramGrid = new ParamGridBuilder()
      .addGrid(rf.featureSubsetStrategy, Seq("all", "sqrt", "log2"))
      .addGrid(rf.maxDepth, Seq(3, 5, 7, 8))
      .addGrid(rf.numTrees, Seq(20, 50, 100))
      .addGrid(rf.subsamplingRate, Seq(0.8, 1.0))
      .build()
    evaluate(trainData, testData, pipeline, paramGrid)
  }

  def evaluate(trainData: DataFrame,
               testData: DataFrame,
               pipeline: Pipeline,
               params: Array[ParamMap]): Unit = {
    val classifierEval = new MulticlassClassificationEvaluator()
      .setLabelCol("rating")
      .setPredictionCol("pred")
      .setMetricName("accuracy")  // f1, weightedPrecision

    val validator = new TrainValidationSplit()
      .setSeed(Random.nextLong())
      .setEstimator(pipeline)
      .setEvaluator(classifierEval)
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
      s" => accuracy: ${validatorModel.validationMetrics.max}")
    println(s"test accuracy: ${classifierEval.evaluate(bestModel.transform(testData))}")
  }

  def mapValue(rating: Int): Double = {
    rating match {
      case rating if rating >= 9 => 2.0
      case rating if rating >= 6 && rating <= 8 => 1.0
      case _ => 0.0
    }
  }
}


object Classifier extends Context{
  def main(args: Array[String]): Unit = {
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

    val model = new Classifier(evaluate = false, algo = Some("rf"), convertLabel = true)
    time(model.train(data), "Training")

    val transformedData = model.transform(data)
    transformedData.show(4, truncate = false)
  }
}



