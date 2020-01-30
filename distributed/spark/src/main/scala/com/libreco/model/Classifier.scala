package com.libreco.model

import org.apache.spark.sql.DataFrame
import com.libreco.feature.FeatureEngineering
import com.libreco.evaluate.EvalClassifier
import org.apache.spark.ml.classification.{MultilayerPerceptronClassifier, RandomForestClassifier}
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.sql.functions.{udf, col}

import scala.util.Random


class Classifier(algo: Option[String] = Some("mlp")) extends Serializable {
  var pipelineModel: PipelineModel = _

  def train(df: DataFrame, evaluate: Boolean = false, debug: Boolean = false): Unit = {
    val prePipelineStages: Array[PipelineStage] = FeatureEngineering.preProcessPipeline(df)
    if (debug) {
      val pipeline = new Pipeline().setStages(prePipelineStages)
      pipelineModel = pipeline.fit(df)
      val transformed = pipelineModel.transform(df)
      transformed.show(4, truncate = false)
    }

    val udfMapValue = udf(mapValue(_: Int): Int)
    val data = df.withColumn("label", udfMapValue(col("rating")))
    if (evaluate) {
      val evalModel = new EvalClassifier(algo, prePipelineStages)
      evalModel.eval(data)
    }
    else {
      algo match {
        case Some("mlp") =>
          val model = new MultilayerPerceptronClassifier()
            .setFeaturesCol("featureVector")
            .setLabelCol("label")
            .setPredictionCol("pred")
            .setProbabilityCol("prob")
            .setLayers(Array(62, 40, 10, 3))
            .setStepSize(0.01)
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
          println("Model muse either be MultilayerPerceptronClassifier or RandomForestClassifier")
          System.exit(1)
      }
    }
  }

  def transform(dataset: DataFrame): DataFrame = {
    pipelineModel.transform(dataset)
  }

  private def mapValue(rating: Int): Int = {
    rating match {
      case `rating` if rating == 5 => 2
      case `rating` if rating == 4 => 1
      case _ => 0
    }
  }
}
