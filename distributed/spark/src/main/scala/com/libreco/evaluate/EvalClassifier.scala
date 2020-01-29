package com.libreco.evaluate

import com.libreco.data.DataSplitter
import org.apache.spark.sql.{Column, DataFrame}
import org.apache.spark.ml.classification.{MultilayerPerceptronClassifier, RandomForestClassifier}
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.param.ParamMap

import scala.util.Random


class EvalClassifier(algo: Option[String],
                     pipelineStages: Array[PipelineStage]) {

  def eval(data: DataFrame): Unit = {
    val splitter = new DataSplitter()
    val Array(trainData, testData) = splitter.stratified_chrono_split(data, 0.8, "user")
    trainData.cache()
    testData.cache()

    algo match {
      case Some("mlp") => evaluateMLP(trainData, testData, pipelineStages)
      case Some("rf") => evaluateRF(trainData, testData, pipelineStages)
      case None =>
        println("Model muse either be MultilayerPerceptronClassifier or RandomForestClassifier")
        System.exit(1)
      case _ =>
        println("Model muse either be MultilayerPerceptronClassifier or RandomForestClassifier")
        System.exit(2)
    }
    trainData.unpersist()
    testData.unpersist()
  }

  private def evaluateMLP(trainData: DataFrame,
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
      .addGrid(mlp.layers, Seq(Array[Int](62, 20, 10, 3), Array[Int](62, 50, 30, 10, 3)))
      .addGrid(mlp.stepSize, Seq(0.01, 0.03, 0.05))
      .addGrid(mlp.maxIter, Seq(100, 300, 500))
      .build()
    showScoreAndParam(trainData, testData, pipeline, paramGrid)
  }

  private def evaluateRF(trainData: DataFrame,
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
    showScoreAndParam(trainData, testData, pipeline, paramGrid)
  }

  private [evaluate] def showScoreAndParam(trainData: DataFrame,
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
}
