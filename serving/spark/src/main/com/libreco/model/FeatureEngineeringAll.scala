package libreco.model

import org.apache.spark.sql.{SparkSession, DataFrame, Dataset}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.{PipelineModel, Pipeline, PipelineStage}
import org.apache.spark.ml.feature.{VectorAssembler, OneHotEncoder, StringIndexer}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType, DoubleType}
import org.apache.spark.ml.regression.{GBTRegressor, GBTRegressionModel}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.log4j.{Level, Logger}

import scala.collection.mutable.ArrayBuffer

object FeatureEngineeringAll {
  def preProcessPipeline(dataset: DataFrame): Array[PipelineStage] = {
    var data = dataset
    val dataColumns = data.columns diff Array("rating", "user_id", "anime_id", "name")
    val otherCols = Seq("episodes", "web_rating", "members")
    val cateCols = Set(dataColumns: _*) -- otherCols -- Seq("genre")

    val pipelineStages = ArrayBuffer[PipelineStage]()
    val vectorAassembleCols = ArrayBuffer[String]()
    cateCols.foreach { col =>
      val (encoder, vecCol) = oneHotPipeline(col)
      vectorAassembleCols += vecCol
      pipelineStages += encoder
    }
/*
    val genreList = data
      .select("genre")
      .rdd
      .map(_.getAs[String]("genre"))
      .flatMap(_.split(", ")).distinct.collect()
    genreList.foreach(g => vectorAassembleCols += g)

    val multiHot = new MultiHotEncoder()
      .setInputCol("genre")
      .setOutputCols(genreList)
    pipelineStages += multiHot
*/
    val assembler = new VectorAssembler()
      .setInputCols(vectorAassembleCols.toArray ++ otherCols)
      .setOutputCol("featureVector")
    //  assembler.getInputCols.foreach(x => print(x + " "))
    //  pipeline.getStages.foreach(println)
    println(s"featureVector length: ${assembler.getInputCols.length}")
    //  data.schema.fields.foreach(println)

  //  val pipeline = new Pipeline().setStages(pipelineStages.toArray ++ Array(assembler))
    pipelineStages.toArray ++ Array(assembler)

  }


  def oneHotPipeline(inputCol: String): (Pipeline, String) = {
    val indexer = new StringIndexer()
      .setInputCol(inputCol)
      .setOutputCol(inputCol + "_indexed")
    //  .setHandleInvalid("skip")
    val encoder = new OneHotEncoder()
      .setInputCol(inputCol + "_indexed")
      .setOutputCol(inputCol + "_vec")
    val pipeline = new Pipeline().setStages(Array(indexer, encoder))
    (pipeline, encoder.getOutputCol)
  }
}
