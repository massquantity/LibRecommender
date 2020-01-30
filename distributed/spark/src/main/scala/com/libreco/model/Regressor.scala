package com.libreco.model

import org.apache.spark.sql.DataFrame
import com.libreco.feature.FeatureEngineering
import com.libreco.evaluate.EvalRegressor
import org.apache.spark.ml.regression.{GBTRegressor, GeneralizedLinearRegression}
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}

import scala.util.Random

class Regressor(algo: Option[String] = Some("gbdt")) extends Serializable {
  var pipelineModel: PipelineModel = _

  def train(df: DataFrame, evaluate: Boolean = false, debug: Boolean = false): Unit = {
    val prePipelineStages: Array[PipelineStage] = FeatureEngineering.preProcessPipeline(df)
    if (debug) {
      val pipeline = new Pipeline().setStages(prePipelineStages)
      pipelineModel = pipeline.fit(df)
      val transformed = pipelineModel.transform(df)
      transformed.show(4, truncate = false)
    }
    if (evaluate) {
      val evalModel = new EvalRegressor(algo, prePipelineStages)
      evalModel.eval(df)
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
            .setMaxIter(20)
            .setStepSize(0.01)
            .setSubsamplingRate(0.8)
            .setSeed(Random.nextLong())
          val pipelineStages = prePipelineStages ++ Array(model)
          val pipeline = new Pipeline().setStages(pipelineStages)
          pipelineModel = pipeline.fit(df)

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
          pipelineModel = pipeline.fit(df)

        case _ =>
          println("Model muse either be GBDTRegressor or GeneralizedLinearRegression")
          System.exit(1)
      }
    }
  }

  def transform(dataset: DataFrame): DataFrame = {
    pipelineModel.transform(dataset)
  }
}


