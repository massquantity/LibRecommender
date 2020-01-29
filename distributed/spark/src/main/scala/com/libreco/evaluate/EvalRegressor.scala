package com.libreco.evaluate

import com.libreco.data.DataSplitter
import org.apache.spark.sql.{Column, DataFrame}
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.ml.regression.{GBTRegressor, GeneralizedLinearRegression}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.param.ParamMap

import scala.util.Random


class EvalRegressor(algo: Option[String],
                    pipelineStages: Array[PipelineStage]) {

  def eval(data: DataFrame): Unit = {
    val splitter = new DataSplitter()
    val Array(trainData, testData) = splitter.stratified_chrono_split(data, 0.8, "user")
    trainData.cache()
    testData.cache()

    algo match {
      case Some("gbdt") => evaluateGBDT(trainData, testData, pipelineStages)
      case Some("glr") => evaluateGLR(trainData, testData, pipelineStages)
      case None =>
        println("Model muse either be GBDTRegressor or GeneralizedLinearRegression")
        System.exit(1)
      case _ =>
        println("Model muse either be GBDTRegressor or GeneralizedLinearRegression")
        System.exit(2)
    }
    trainData.unpersist()
    testData.unpersist()
  }

  private def evaluateGBDT(trainData: DataFrame,
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
      .addGrid(gbr.maxDepth, Seq(3, 5, 7, 10))
      .addGrid(gbr.stepSize, Seq(0.01, 0.03, 0.05))
      .addGrid(gbr.subsamplingRate, Seq(0.8, 1.0))
      .build()
    showScoreAndParam(trainData, testData, pipeline, paramGrid)
  }

  private def evaluateGLR(trainData: DataFrame,
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
    showScoreAndParam(trainData, testData, pipeline, paramGrid)
  }

  private [evaluate] def showScoreAndParam(trainData: DataFrame,
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
